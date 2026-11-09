"""CPU contracts for immutable schedule40 inputs; no model is constructed."""
import copy
from pathlib import Path

import numpy as np
import pytest

from meshflow_control.data import schedule40
from meshflow_control.data.io import atomic_json, digest, save_npz, read_json


def manifest_fixture():
    parents = ["p%02d" % i for i in range(32)]
    tasks = []
    for i, uid in enumerate(parents[:26]):
        for j in range(8 if i < 25 else 3):
            tasks.append(dict(uid=uid, task_id=f"{uid}_train40_{j}", role="train",
                split="train", status="READY", N=128, K=51, ratio=.4,
                source_face_ids=list(range(51)), free_source_ids=list(range(51, 128))))
    for j in range(250):
        uid = parents[j % 32]
        tasks.append(dict(uid=uid, task_id=f"{uid}_train20_{j}", role="train",
            split="train", status="READY", N=128, K=26, ratio=.2,
            source_face_ids=list(range(26)), free_source_ids=list(range(26, 128))))
    return dict(schema="meshflow_control_training_data_v1",
                splits={"train": parents}, parents=[dict(uid=u, N=128) for u in parents],
                tasks=tasks)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path/"data"
    root.mkdir()
    manifest = manifest_fixture()
    for i, row in enumerate(manifest["parents"]):
        full = np.random.RandomState(i).normal(size=(128, 3, 3)).astype(np.float32)
        path = root/(row["uid"]+".npz")
        save_npz(path, full_target=full.reshape(128, 9), pre_ot_full=full*np.float32(.5),
                 model_vertices=full.reshape(-1, 3),
                 source_face_vertex_ids=np.arange(384, dtype=np.int64).reshape(128, 3))
        row.update(npz=path.name, sha256=digest(path))
    atomic_json(root/"train_manifest.json", manifest)
    return root


def test_full_plan_balances_boundaries_and_is_deterministic():
    manifest = manifest_fixture()
    first = schedule40.make_plan(manifest)
    second = schedule40.make_plan(copy.deepcopy(manifest))
    assert first == second
    assert first["sample_count"] == 48000
    assert first["eligible_task_count"] == 203
    assert len(first["excluded_training_parents"]) == 6
    assert set(r["ratio"] for r in first["records"]) == {.4}
    counts = first["object_counts"]
    assert max(counts.values()) - min(counts.values()) <= 1
    for start in range(0, 48000, 8):
        rows = first["records"][start:start+8]
        assert len({r["uid"] for r in rows}) == 8
        assert sum(r["alpha"] == 1. for r in rows) == 4
    # This catches the duplicate-parent failure of naively concatenating fresh
    # 26-permutations into an eight-item batch.
    for start in range(0, 104*3, 104):
        rows = first["records"][start:start+104]
        for uid in first["eligible_parents"]:
            selected = [r for r in rows if r["uid"] == uid]
            assert len(selected) == 4
            assert sum(r["alpha"] == 1. for r in selected) == 2
    seeds = first["records"][0]
    assert len({seeds[k] for k in ("gaussian_seed", "epsilon_seed",
                                  "permutation_seed", "t_seed")}) == 4


def test_invalid_fixed40_partition_is_rejected():
    manifest = manifest_fixture()
    manifest["tasks"][0]["K"] = 50
    with pytest.raises(ValueError, match="face count"):
        schedule40.make_plan(manifest, 1)


def test_cached_inputs_resume_and_fork_without_new_ot(data_root, tmp_path):
    root = tmp_path/"stream"
    schedule40.prepare_training_plan(data_root, root, total_updates=2)
    stream = schedule40.Schedule40Stream(data_root, root)
    with pytest.raises(RuntimeError, match="must be prepared"):
        stream.samples(1)
    assert not (root/"ot_cache").exists()
    receipt = schedule40.precompute(data_root, root, workers=2)
    assert receipt["actual_OT_returns_this_run"] == 16
    first = schedule40.Schedule40Stream(data_root, root, arm="COMMON")
    batch1 = first.next_effective_batch()
    state = first.state_dict()
    resumed = schedule40.Schedule40Stream(data_root, root, arm="LATE")
    resumed.load_state_dict(state)
    expected = first.next_effective_batch()
    actual = resumed.next_effective_batch()
    assert [s["shared_input_sha256"] for s in expected] == [s["shared_input_sha256"] for s in actual]
    assert first.last_audit == resumed.last_audit
    assert resumed.counts["actual_OT_attempts"] == 0
    for sample in batch1 + actual:
        assert int(sample["known_mask"].sum()) == 51
        assert sample["free_coordinate_count"] == 9*77
        assert np.isfinite(sample["xt"]).all()
    again = schedule40.precompute(data_root, root, workers=1)
    assert again["actual_OT_attempts_this_run"] == 0
    assert again["cache_hits_this_run"] == 16
    bad = dict(state, batch_index=3, sample_index=24)
    with pytest.raises(ValueError, match="cursor"):
        resumed.load_state_dict(bad)
    bad = dict(state, plan_sha256="bad")
    with pytest.raises(ValueError, match="identity"):
        resumed.load_state_dict(bad)
    with pytest.raises(ValueError, match="outside"):
        resumed.next_effective_batch()


def test_frozen_plan_changes_are_detected(data_root, tmp_path):
    root = tmp_path/"stream"
    schedule40.prepare_training_plan(data_root, root, total_updates=2)
    plan = read_json(root/"train_plan.json")
    plan["records"][0]["gaussian_seed"] += 1
    atomic_json(root/"train_plan.json", plan)
    with pytest.raises(ValueError, match="does not reproduce"):
        schedule40.Schedule40Stream(data_root, root)


def test_teacher_mapping_validates_clean_actual_geometry(data_root, tmp_path):
    from meshflow_control.teachers.siglip_render import array_hash
    stream_root = tmp_path/"stream"
    schedule40.prepare_training_plan(data_root, stream_root, total_updates=1)
    stream = schedule40.Schedule40Stream(data_root, stream_root, require_cache=False)
    record = next(r for r in stream.records if r["alpha"] == 1.)
    sample = stream.sample(record)
    case = stream.data.task(sample["task_id"])
    geometry_hash = array_hash(case["full_target"].reshape(-1, 3, 3))
    teacher_root = tmp_path/"teacher"
    target = np.ones(768, np.float32)
    target /= np.linalg.norm(target)
    feature_path = teacher_root/"feature.npz"
    save_npz(feature_path, target=target)
    atomic_json(teacher_root/"teacher_manifest.json", dict(
        schema="meshflow_schedule40_reused_teacher_v1", status="COMPLETE",
        inputs=dict(tasks={sample["task_id"]: dict(geometry_sha256=geometry_hash,
                    uid=sample["object_id"], N=128, K=51)},
                    alpha1_samples={str(sample["sample_index"]): dict(task_id=sample["task_id"])}),
        rows=[dict(path=str(feature_path), sha256=digest(feature_path),
                   target_array_sha256=array_hash(target), geometry_sha256=geometry_hash)]))
    store = schedule40.Schedule40TargetStore(teacher_root)
    np.testing.assert_array_equal(store.for_sample(sample), target)
    wrong = copy.deepcopy(sample)
    wrong["x1"][~wrong["known_mask"]] += np.float32(.1)
    with pytest.raises(ValueError, match="not teacher geometry"):
        store.for_sample(wrong)
    with pytest.raises(ValueError, match="Width-augmented"):
        store.for_sample(dict(sample, alpha=.9))
    with pytest.raises(ValueError, match="registered alpha1"):
        store.for_sample(dict(sample, sample_index=48001))
