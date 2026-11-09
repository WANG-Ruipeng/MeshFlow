"""Light CPU checks for observation identity, leakage barriers and diagnostics."""
from __future__ import annotations
import copy
import json
import numpy as np
import pytest
import torch
from ..io_utils import array_hash, file_hash
from ..legacy.networks import DomainVelocity
from ..layout_student.train import discrete_bridge, model_sha256
from .diagnose import (CHECKPOINTS, JOBS, UPDATES, aggregate, endpoint_probe, expected_slots,
                       load_network, load_train_archive, save_raw, velocity_probe)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def protocol():
    train = [f"train{i}" for i in range(12)]
    jobs = {}
    for name in JOBS:
        single = name.startswith("single_")
        active = train[:1] if single else train
        jobs[name] = {"diagnostic_uids": active, "active_training_uids": active,
                      "kind": "single" if single else "joint", "seed": 11 if name.endswith("11") else 10,
                      "start_update": 0 if single else 1000}
    return {"train_uids": train, "internal_check_uids": [f"check{i}" for i in range(4)],
            "jobs": jobs, "training": {"checkpoint_updates": list(UPDATES)}}


def test_fixed_denominator_and_internal_check_leakage_barrier():
    p = protocol()
    slots = expected_slots(p)
    assert len(slots) == 156
    assert {s[1] for s in slots} == set(CHECKPOINTS)
    broken = copy.deepcopy(p)
    broken["jobs"]["single_energy_10"]["diagnostic_uids"] = ["check0"]
    with pytest.raises(ValueError, match="Internal-check/unregistered"):
        expected_slots(broken)
    broken = copy.deepcopy(p)
    broken["jobs"]["joint_10"]["diagnostic_uids"] = p["train_uids"][:-1]
    with pytest.raises(ValueError, match="active training"):
        expected_slots(broken)
    broken = copy.deepcopy(p)
    broken["training"]["checkpoint_updates"].append(19000)
    with pytest.raises(ValueError, match="observation updates"):
        expected_slots(broken)


def test_endpoint_energy_projection_and_unique_edge_roughness():
    base = np.eye(3)
    teacher = np.roll(base, 1, axis=0)
    faces = np.array([[0, 1, 2], [2, 1, 0]], np.int32)
    scale = np.mean((teacher - base) ** 2)
    fixed = endpoint_probe(base, base, teacher, faces, scale)
    assert fixed["endpoint_direction_mse_normalized"] == 1
    assert fixed["endpoint_mse_over_zero_motion"] == 1
    assert fixed["movement_energy_over_teacher"] == 0
    assert fixed["teacher_projection_coefficient"] == 0
    assert fixed["displacement_cosine_to_teacher"] is None
    exact = endpoint_probe(teacher, base, teacher, faces, scale)
    assert exact["endpoint_direction_mse_normalized"] == 0
    assert exact["movement_energy_over_teacher"] == 1
    assert exact["teacher_projection_coefficient"] == 1
    assert exact["displacement_cosine_to_teacher"] == 1
    assert exact["orthogonal_displacement_energy"] == 0
    assert exact["edge_count"] == 3
    assert exact["edge_roughness_over_teacher"] == 1
    doubled = endpoint_probe(base + 2 * (teacher - base), base, teacher, faces, scale)
    assert doubled["teacher_projection_coefficient"] == 2
    assert doubled["movement_energy_over_teacher"] == 4
    assert doubled["edge_roughness_over_teacher"] == 4
    assert doubled["endpoint_mse_over_zero_motion"] == 1


def test_zero_teacher_denominators_are_null_not_success_or_nan():
    base = np.eye(3)
    f = np.array([[0, 1, 2]], np.int32)
    result = endpoint_probe(np.roll(base, 1, axis=0), base, base, f, 1.)
    assert result["zero_motion_has_zero_teacher_error"]
    assert result["endpoint_direction_mse_normalized"] > 0
    for key in ("endpoint_mse_over_zero_motion", "movement_energy_over_teacher",
                "teacher_projection_coefficient", "displacement_cosine_to_teacher", "edge_roughness_over_teacher"):
        assert result[key] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="finite and positive"):
        endpoint_probe(base, base, base, f, 0.)


def test_velocity_probe_all32_b1_exact_discrete_grid_and_stationary_target():
    base = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64)
    teacher = base + torch.tensor([[[0., .1, 0.], [.1, 0., 0.]]], dtype=torch.float64)
    teacher = teacher / teacher.norm(dim=-1, keepdim=True)

    class Oracle(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.target = target
            self.observed = []

        def forward(self, u, t, features):
            assert u.shape == (1, 2, 3) and features.shape == (1, 256, 4)
            k = torch.round(t * 32).long()
            self.observed.extend(k.tolist())
            expected_u, velocity, _, _ = discrete_bridge(base, self.target, k)
            assert torch.equal(u, expected_u.float())
            return velocity.float()

    model = Oracle(teacher)
    result = velocity_probe(model, base[0].numpy(), teacher[0].numpy(), np.zeros((256, 4), np.float32))
    assert model.observed == list(range(32))
    assert result["mse"] == 0 and result["zero_velocity_mse"] > 0
    assert result["mse_over_zero_velocity"] == 0
    stationary = velocity_probe(Oracle(base), base[0].numpy(), base[0].numpy(), np.zeros((256, 4), np.float32))
    assert stationary["mse"] == stationary["zero_velocity_mse"] == 0
    assert stationary["mse_over_zero_velocity"] is None


def test_imported_snapshot_byte_identity_and_new_checkpoint_contract(tmp_path):
    p = protocol()
    state = {"model": DomainVelocity().state_dict(), "update": 1000, "config": {"seed": 10},
             "protocol_sha256": "old-protocol", "is_predeclared_final": True}
    state["model_sha256"] = model_sha256(state["model"])
    initial = tmp_path / "training/joint_10/initial.pt"
    initial.parent.mkdir(parents=True)
    torch.save(state, initial)
    p["jobs"]["joint_10"].update(initial_checkpoint_sha256=file_hash(initial), initial_model_sha256=state["model_sha256"])
    put_json(tmp_path / "RUN_PROTOCOL.json", p)
    model, identity = load_network(tmp_path, "joint_10", "initial", p)
    assert identity["imported_snapshot"] and identity["job_update"] == 1000
    assert not any(x.requires_grad for x in model.parameters())
    changed = dict(state, update=999)
    torch.save(changed, initial)
    with pytest.raises(ValueError, match="Imported snapshot file changed"):
        load_network(tmp_path, "joint_10", "initial", p)
    state.update(update=1667, protocol_sha256=file_hash(tmp_path / "RUN_PROTOCOL.json"), is_predeclared_final=False,
                 config={"job": "joint_10", "kind": "joint", "seed": 10, "arm": "c0",
                         "active_training_uids": p["train_uids"]})
    path = initial.with_name("step_01667.pt")
    torch.save(state, path)
    _, identity = load_network(tmp_path, "joint_10", "1667", p)
    assert identity["job_update"] == 1667 and not identity["imported_snapshot"]
    for key, bad, message in [("update", 1500, "observation update"),
                              ("protocol_sha256", "bad", "another run protocol"),
                              ("is_predeclared_final", True, "Final checkpoint marker")]:
        torch.save(dict(state, **{key: bad}), path)
        with pytest.raises(ValueError, match=message):
            load_network(tmp_path, "joint_10", "1667", p)


def test_train_archive_rejects_check_labels_before_opening_any_object(tmp_path):
    p = protocol()
    p["paths"] = {"training_labels": "labels.npz"}
    labels = tmp_path / "labels.npz"
    bad_uids = p["train_uids"][:-1] + ["check0"]
    np.savez(labels, train_uids=bad_uids, directions0=np.zeros((12, 258, 3)),
             directions1=np.zeros((12, 258, 3)), proxy_points=np.zeros((12, 256, 3)),
             point_features=np.zeros((12, 256, 4)), logr=np.zeros((12, 2050)),
             faces=np.zeros((512, 3), np.int32), quality_floors_deg=np.zeros((12, 2)))
    p["hashes"] = {"training_labels_sha256": file_hash(labels)}
    with pytest.raises(ValueError, match="UID order/count differs"):
        load_train_archive(tmp_path, p)
    assert not (tmp_path / "data").exists()


def test_aggregate_keeps_missing_and_failure_slots_without_best_selection(tmp_path):
    p = protocol()
    put_json(tmp_path / "RUN_PROTOCOL.json", p)
    base = np.eye(3)
    endpoint = endpoint_probe(base, base, np.roll(base, 1, axis=0), np.array([[0, 1, 2]]), 1.)
    row = {"job": "joint_10", "checkpoint": "initial", "uid": "train0", "split": "train", "qualified": False,
           "endpoint": endpoint, "velocity": {"mse": 1., "zero_velocity_mse": 2., "mse_over_zero_velocity": .5}}
    common = {"protocol_sha256": file_hash(tmp_path / "RUN_PROTOCOL.json")}
    dest = tmp_path / "diagnostics/mechanism"
    good = {**common, "job": "joint_10", "checkpoint": "initial", "uid": "train0", "status": "success", "row": row}
    put_json(dest / "joint_10/initial/train0/result.json", good)
    put_json(dest / "joint_10/1000/train0/result.json",
             {**common, "job": "joint_10", "checkpoint": "1000", "uid": "train0", "status": "error", "row": None,
              "error": "Missing checkpoint retained"})
    result = aggregate(tmp_path)
    assert result["attempt_count"] == 2 and result["expected_attempt_count"] == 156
    assert len(result["errors"]) == 1 and len(result["unattempted"]) == 154
    assert len(result["missing_success_slots"]) == 155
    assert result["checkpoint_selection"] is False and not result["pooled_across_checkpoints_or_jobs"]
    first = result["groups"][0]
    assert first["expected_parents"] == 12 and first["successful_parents"] == 1
    assert first["missing_or_failed_parents"] == 11
    put_json(dest / "duplicate/path/uid/result.json", good)
    with pytest.raises(ValueError, match="Duplicate/unregistered"):
        aggregate(tmp_path)


def test_raw_npz_readback_preserves_typed_hash_semantics(tmp_path):
    vertices = np.eye(3)
    faces = np.array([[0, 1, 2]], np.int32)
    mesh = {"vertices": vertices, "faces": faces.astype(np.int64), "u": vertices, "r": np.ones(3),
            "alpha": np.ones(32)}
    path = tmp_path / "raw.npz"
    result = save_raw(path, mesh, faces)
    assert result["mesh_hash"] == array_hash(vertices, faces)
    assert result["stored_array_hash"] == array_hash(vertices, mesh["faces"])
    assert result["mesh_hash"] != result["stored_array_hash"]
    assert result["mesh_npz_sha256"] == file_hash(path)
    with np.load(path) as z:
        assert np.array_equal(z["alpha"], mesh["alpha"])
