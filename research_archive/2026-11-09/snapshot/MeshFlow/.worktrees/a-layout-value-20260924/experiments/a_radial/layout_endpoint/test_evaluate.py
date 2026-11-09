"""Regression tests for fixed-model selection, paired statistics, and failure slots."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash, file_hash
from ..layout_student.train import model_sha256
from .evaluate import (MODEL_NAMES, POLICIES, aggregate, expected_slots, load_network,
                       save_mesh_with_identity)
from .summarize import continuation_gate, group_statistics, paired_values


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(uid, model, rms, *, qualified=True):
    return {"uid": uid, "model": model, "policy": "raw", "status": "success",
            "mainfield_hash": uid, "qualified": qualified,
            "metrics": [{"seed": 392401, "rms_bbox_percent": rms[0]},
                        {"seed": 492401, "rms_bbox_percent": rms[1]}],
            "audit": {"valid": True}, "fallback_to_fixed": False,
            "quality": {"min_triangle_angle_deg": 30., "triangle_min_angle_p05_deg": 35.}}


def test_planned_denominators_and_failed_attempt_rows(tmp_path):
    uids = ["a", "b"]
    put_json(tmp_path / "RUN_PROTOCOL.json", {"uids": uids})
    good = row("a", "fixed", [1., 1.])
    put_json(tmp_path / "attempts/a/fixed/result.json",
             {"uid": "a", "model": "fixed", "status": "success", "rows": [good]})
    put_json(tmp_path / "attempts/a/c0_10/result.json",
             {"uid": "a", "model": "c0_10", "status": "error", "error": "Missing checkpoint", "rows": []})
    result = aggregate(tmp_path)
    assert len(MODEL_NAMES) == 10 and len(expected_slots(range(16))) == 288
    assert result["expected_attempt_count"] == 20 and result["expected_row_count"] == 36
    assert result["attempt_count"] == 2 and len(result["errors"]) == 1
    assert set(map(tuple, result["failed_row_slots"])) == {("a", "c0_10", "raw"), ("a", "c0_10", "quality_fallback")}
    assert len(result["unattempted"]) == 18 and len(result["missing_success_rows"]) == 35


def test_success_attempt_cannot_hide_duplicate_policy(tmp_path):
    put_json(tmp_path / "RUN_PROTOCOL.json", {"uids": ["a"]})
    good = row("a", "fixed", [1., 1.])
    put_json(tmp_path / "attempts/a/fixed/result.json",
             {"uid": "a", "model": "fixed", "status": "success", "rows": [good, good]})
    with pytest.raises(ValueError, match="Duplicate policy"):
        aggregate(tmp_path)


def test_paired_median_and_both_measurements_and_missing_denominator():
    rows = []
    for uid, fixed, method in [("a", 1., .9), ("b", 100., 50.), ("c", 1000., 1000.)]:
        baseline = row(uid, "fixed", [fixed, fixed])
        candidate = row(uid, "c2_10", [method, method])
        candidate.update(paired_values(candidate, baseline))
        rows.append(candidate)
    summary = group_statistics(rows, 16)
    assert summary["expected_objects"] == 16 and summary["missing_or_failed_objects"] == 13
    assert summary["paired_improvement_percent_median_seed1"] == pytest.approx(10.)
    assert summary["qualified_ge2pct_both_count"] == 2
    candidate = row("a", "c2_10", [.97, .99])
    assert not paired_values(candidate, row("a", "fixed", [1., 1.]))["qualified_ge2pct_both"]
    candidate["qualified"] = False
    candidate["metrics"][1]["rms_bbox_percent"] = .97
    assert not paired_values(candidate, row("a", "fixed", [1., 1.]))["qualified_improved_both"]
    candidate["metrics"][1]["seed"] += 1
    with pytest.raises(ValueError, match="different evaluation seeds"):
        paired_values(candidate, row("a", "fixed", [1., 1.]))


def test_gate_needs_both_training_seeds_and_raw_gains():
    groups = []
    for seed in (10, 11):
        for split, count, gains in [("all", 16, 8), ("internal_check", 4, 2)]:
            groups.append({"split": split, "model": f"c2_{seed}", "policy": "raw",
                           "recorded_objects": count, "paired_objects": count,
                           "qualified_ge2pct_both_count": gains})
    assert continuation_gate(groups)["passed"]
    groups[-1]["qualified_ge2pct_both_count"] = 1
    assert not continuation_gate(groups)["passed"]
    groups[-1]["qualified_ge2pct_both_count"] = 2
    groups[0]["recorded_objects"] = 15
    assert not continuation_gate(groups)["passed"]


def test_readback_hashes_distinguish_storage_and_template_dtype(tmp_path):
    vertices = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], np.float64)
    template = np.array([[0, 1, 2]], np.int32)
    mesh = {"vertices": vertices, "faces": template.astype(np.int64),
            "u": vertices, "r": np.ones(3)}
    path = tmp_path / "mesh.npz"
    result = save_mesh_with_identity(path, mesh, template)
    assert result["mesh_hash"] == array_hash(vertices, template)
    assert result["stored_array_hash"] == array_hash(vertices, mesh["faces"])
    assert result["mesh_hash"] != result["stored_array_hash"]
    assert result["mesh_npz_sha256"] == file_hash(path)
    with np.load(path) as z:
        assert z["faces"].dtype == np.int64


def test_checkpoint_requires_fixed_final_seed_arm_protocol_and_tensor_hash(tmp_path):
    protocol = {"training": {"updates": 1000}}
    put_json(tmp_path / "RUN_PROTOCOL.json", protocol)
    model = DomainVelocity()
    state = {"model": model.state_dict(), "model_sha256": model_sha256(model.state_dict()),
             "is_predeclared_final": True, "update": 1000,
             "protocol_sha256": file_hash(tmp_path / "RUN_PROTOCOL.json"),
             "config": {"seed": 10, "arm": "c2"}}
    path = tmp_path / "training/c2/seed_10/final.pt"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    observed, identity = load_network(tmp_path, "c2_10", protocol)
    assert identity["model_sha256"] == state["model_sha256"]
    assert not any(p.requires_grad for p in observed.parameters())
    for key, invalid, expected in [
        ("is_predeclared_final", False, "predeclared final"),
        ("update", 999, "fixed final update"),
        ("protocol_sha256", "bad", "protocols differ"),
        ("model_sha256", "bad", "tensor hash mismatch"),
    ]:
        broken = dict(state)
        broken[key] = invalid
        torch.save(broken, path)
        with pytest.raises(ValueError, match=expected):
            load_network(tmp_path, "c2_10", protocol)
    broken = dict(state)
    broken["config"] = {"seed": 10, "arm": "c1"}
    torch.save(broken, path)
    with pytest.raises(ValueError, match="arm differs"):
        load_network(tmp_path, "c2_10", protocol)



def test_endpoint_mse_uses_frozen_global_scale_and_zero_motion_ratio():
    from .summarize import endpoint_error_metrics
    fixed = np.array([[1., 0., 0.]])
    teacher = np.array([[0., 1., 0.]])
    halfway = np.array([[2. ** -.5, 2. ** -.5, 0.]])
    scale = 2. / 3.
    values = endpoint_error_metrics(halfway, teacher, fixed, scale)
    assert values["fixed_zero_motion_direction_mse_normalized"] == pytest.approx(1.)
    assert values["endpoint_direction_mse_normalized"] == pytest.approx(1 - 2. ** -.5)
    assert values["endpoint_mse_over_zero_motion"] == pytest.approx(1 - 2. ** -.5)
    stationary = endpoint_error_metrics(fixed, fixed, fixed, scale)
    assert stationary["endpoint_direction_mse_normalized"] == 0
    assert stationary["endpoint_mse_over_zero_motion"] is None
    assert stationary["zero_motion_has_zero_teacher_error"]
    moving_row = row("a", "c2_10", [.9, .9])
    moving_row.update(paired_values(moving_row, row("a", "fixed", [1., 1.])))
    moving_row.update(endpoint_error_metrics(teacher, teacher, fixed, scale))
    stationary_row = row("b", "c2_10", [1., 1.])
    stationary_row.update(paired_values(stationary_row, row("b", "fixed", [1., 1.])))
    stationary_row.update(stationary)
    group = group_statistics([moving_row, stationary_row], 2)
    assert group["endpoint_direction_mse_normalized_mean"] == 0
    assert group["fixed_zero_motion_direction_mse_normalized_mean"] == .5
    assert group["mean_endpoint_mse_over_mean_zero_motion_mse"] == 0
    assert group["paired_endpoint_mse_ratio_objects"] == 1
    assert group["zero_motion_zero_error_objects"] == 1
    with pytest.raises(ValueError, match="finite and positive"):
        endpoint_error_metrics(fixed, teacher, fixed, 0.)


def test_correctness_scope_preserves_conditional_and_gap_status(tmp_path):
    from .summarize import correctness_scope
    missing = correctness_scope(tmp_path)
    assert missing["final_deployment_parity"]["status"] == "NOT_YET_RECORDED"
    assert missing["overall_correctness_pass_inferred"] is False
    put_json(tmp_path / "IMPLEMENTATION_GATE.json",
             {"status": "CONDITIONAL_SCOPE", "strict_all_model_all_device_correctness_pass": False,
              "original_A_cuda_numerical_parity": "UNSUPPORTED_FOR_3_OF_16_PRESSURE_CASES"})
    put_json(tmp_path / "TRAINING_GATE.json",
             {"status": "APPROVED_FOR_PREDECLARED_V1_INITIALIZED_STUDY",
              "all_model_all_device_correctness_pass": False,
              "implementation_gate_sha256": file_hash(tmp_path / "IMPLEMENTATION_GATE.json")})
    put_json(tmp_path / "results.json", {"rows": []})
    put_json(tmp_path / "FINAL_DEPLOYMENT_PARITY.json",
             {"status": "NUMERICAL_DEPLOYMENT_GAP", "statistics": {"cuda": {"qualification_changed_cases": 1}},
              "records": [], "results_sha256": file_hash(tmp_path / "results.json"),
              "main_table_uses_original_CPU_only": True, "cuda_bitwise_equal_claim": False})
    scope = correctness_scope(tmp_path)
    assert scope["implementation_gate"]["record"]["strict_all_model_all_device_correctness_pass"] is False
    assert scope["training_gate"]["record"]["all_model_all_device_correctness_pass"] is False
    assert scope["final_deployment_parity"]["status"] == "NUMERICAL_DEPLOYMENT_GAP"
    assert scope["overall_correctness_pass_inferred"] is False
    put_json(tmp_path / "results.json", {"rows": [], "changed": True})
    with pytest.raises(ValueError, match="another evaluation ledger"):
        correctness_scope(tmp_path)

