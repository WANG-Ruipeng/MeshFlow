"""Small synthetic CPU contracts; no real experiment, model, or evaluation inputs."""
import copy
from collections import Counter
import shutil

import numpy as np
import pytest

from meshflow_control.data import recipe_confirmation as confirm
from meshflow_control.data import schedule40
from meshflow_control.data.dataset import gaussian
from meshflow_control.data.io import atomic_json, digest, read_json, save_npz

NS = confirm.NAMESPACE_PREFIX + "cpu_fixture/repeat1"


def synthetic_manifest():
    parents = ["p%02d" % i for i in range(32)]
    tasks = []
    for i, uid in enumerate(parents[:26]):
        for j in range(8 if i < 25 else 3):
            tasks.append(dict(uid=uid, task_id=f"{uid}_train40_{j}", role="train", split="train", status="READY",
                N=128, K=51, ratio=.4, source_face_ids=list(range(51)), free_source_ids=list(range(51, 128))))
    for j in range(250):
        uid = parents[j % 32]
        tasks.append(dict(uid=uid, task_id=f"{uid}_train20_{j}", role="train", split="train", status="READY",
            N=128, K=26, ratio=.2, source_face_ids=list(range(26)), free_source_ids=list(range(26, 128))))
    return dict(schema="meshflow_control_training_data_v1", splits={"train": parents},
                parents=[dict(uid=uid, N=128) for uid in parents], tasks=tasks)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    manifest = synthetic_manifest()
    for i, row in enumerate(manifest["parents"]):
        tri = np.random.RandomState(i).normal(size=(128, 3, 3)).astype(np.float32)
        path = root / (row["uid"] + ".npz")
        save_npz(path, full_target=tri.reshape(128, 9), pre_ot_full=tri * np.float32(.5),
                 model_vertices=tri.reshape(-1, 3), source_face_vertex_ids=np.arange(384, dtype=np.int64).reshape(128, 3))
        row.update(npz=path.name, sha256=digest(path))
    atomic_json(root / "train_manifest.json", manifest)
    return root


def prepare_fixture(data, out, pair=("R2", "R4"), policies=None, namespace=NS):
    return confirm.prepare(data, out, list(pair), policies, namespace=namespace, total_updates=6)


def test_every_training_rng_domain_is_new_reproducible_and_recipe_independent():
    manifest = synthetic_manifest()
    new = confirm.make_base_plan(manifest, NS, 36)
    assert new == confirm.make_base_plan(copy.deepcopy(manifest), NS, 36)
    assert new != confirm.make_base_plan(manifest, NS + "b", 36)
    old = schedule40.make_plan(manifest, 36)
    for key in ("uid", "task_id", "alpha", "slot_index", "gaussian_seed", "epsilon_seed", "permutation_seed", "t_seed"):
        if key != "slot_index":
            assert [r[key] for r in new["records"]] != [r[key] for r in old["records"]]
    for domain in confirm.DOMAINS:
        previous = "task_schedule" if domain == "task40" else domain
        assert new["base_domain_seeds"][domain] != schedule40.domain_seed(previous)
    a, b = new["records"][0], old["records"][0]
    assert not np.array_equal(gaussian(a["gaussian_seed"], 77), gaussian(b["gaussian_seed"], 77))
    assert not np.array_equal(gaussian(a["epsilon_seed"], 77), gaussian(b["epsilon_seed"], 77))
    # Every full 13-update block keeps four appearances per parent, two clean.
    for start in (0, 104):
        block = new["records"][start:start + 104]
        assert set(Counter(r["uid"] for r in block).values()) == {4}
        assert set(Counter(r["uid"] for r in block if r["alpha"] == 1).values()) == {2}
    p = confirm.make_plan(manifest, ["R1", "R2"], namespace=NS, total_updates=36)
    q = confirm.make_plan(manifest, ["R3", "R4"], namespace=NS, total_updates=36)
    assert p["base_records"] == q["base_records"] and p["mixed_records"] == q["mixed_records"]
    assert p["independent_training_input_stream"] and not p["independent_model_initialization"]
    assert p["Geo_seed"] == 1010 and p["strict_same_candidate_selection_policy"] == "NOT_SATISFIED"


def test_mix_and_courses_keep_exact_multiset_balance_and_suffix():
    plan = confirm.make_plan(synthetic_manifest(), ["C20_40_MIX", "C40_20_MIX"],
        {"C20_40_MIX": "late", "C40_20_MIX": "late"}, namespace=NS, total_updates=36)
    for start in range(0, 288, 8):
        batch = plan["mixed_records"][start:start + 8]
        assert Counter((r["ratio"], r["alpha"] == 1) for r in batch) == {(.2, True): 2, (.2, False): 2, (.4, True): 2, (.4, False): 2}
    preserved = ("uid", "N", "alpha", "gaussian_seed", "epsilon_seed", "permutation_seed", "t_seed")
    for base, mixed in zip(plan["base_records"], plan["mixed_records"]):
        assert all(base[k] == mixed[k] for k in preserved)
        if mixed["ratio"] == .4:
            assert mixed["task_id"] == base["task_id"]
    for course, ratios in (("C20_40_MIX", (.2, .4)), ("C40_20_MIX", (.4, .2))):
        rows = confirm.records_for_recipe(plan, course)
        assert sorted(r["origin_sample_index"] for r in rows) == list(range(288))
        assert [r["origin_sample_index"] for r in rows[192:]] == list(range(192, 288))
        for start in range(0, 288, 8):
            batch = rows[start:start + 8]
            assert len({r["uid"] for r in batch}) == 8 and sum(r["alpha"] == 1 for r in batch) == 4
            if start < 192:
                assert {r["ratio"] for r in batch} == {ratios[start // 96]}
        assert all(confirm.cache_record_identity(r) == confirm.cache_record_identity(plan["mixed_records"][r["origin_sample_index"]]) for r in rows)


def test_pair_schedule_namespace_and_real_update_budget_fail_closed(data, tmp_path):
    manifest = synthetic_manifest()
    with pytest.raises(ValueError, match="namespace"):
        confirm.make_plan(manifest, ["R1", "R2"], namespace=schedule40.NAMESPACE, total_updates=6)
    with pytest.raises(ValueError, match="explicit candidate"):
        confirm.make_plan(manifest, ["R1"], namespace=NS, total_updates=6)
    with pytest.raises(ValueError, match="different"):
        confirm.make_plan(manifest, ["R1", "R1_40_H_ALL"], namespace=NS, total_updates=6)
    with pytest.raises(ValueError, match="Course HYBRID"):
        confirm.make_plan(manifest, ["R1", "C20_40_MIX"], namespace=NS, total_updates=6)
    with pytest.raises(ValueError, match="mismatch"):
        confirm.make_plan(manifest, ["R1", "R3"], {"R3_40_H_LATE": "all"}, namespace=NS, total_updates=6)
    with pytest.raises(ValueError, match="exactly6000"):
        confirm.prepare(data, tmp_path / "forbidden", ["R1", "R2"], total_updates=6)
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("pair,policies,maps", [
    (("R1", "R3"), None, 48),
    (("R2", "R4"), None, 48),
    (("R1", "R2"), None, 72),
    (("C20_40_MIX", "C40_20_MIX"), {"C20_40_MIX": "all", "C40_20_MIX": "all"}, 48),
])
def test_precompute_only_selected_pair_union_actual_hashes_and_resume(data, tmp_path, pair, policies, maps):
    out = tmp_path / "stream"
    preparation = prepare_fixture(data, out, pair, policies)
    assert preparation["registered_unique_OT_inputs"] == maps
    assert not (out / "confirmation_ot_cache").exists()
    receipt = confirm.precompute(data, out, workers=2)
    assert receipt["actual_OT_attempts"] == receipt["actual_OT_returns"] == maps
    assert receipt["historical_OT_reused"] == 0
    assert len(list((out / "confirmation_ot_cache").glob("*.npz"))) == maps
    assert len(list((out / "paired_inputs").glob("*.json"))) == 2
    assert read_json(out / "precompute_progress.json")["status"] == "COMPLETE"
    registered = {}
    for recipe in preparation["selected_pair"]:
        stream = confirm.ConfirmationStream(data, out, recipe)
        paired = read_json(stream.paired_path)
        assert paired["plan_sha256"] == stream.plan_sha256
        assert paired["schema"] == confirm.PAIR_SCHEMA and not paired["historical_paired_inputs_reused"]
        assert stream.paired_path.is_relative_to(out)
        rows = paired["records"]
        assert len(rows) == 48 and [r["sample_index"] for r in rows] == [r["sample_index"] for r in stream.records]
        hashes = {r["sample_index"]: r for r in rows}
        for step in range(1, 7):
            for sample in stream.samples(step):
                for key, value in confirm.recipes.paired_record(sample).items():
                    assert value == hashes[sample["sample_index"]][key]
                assert hashes[sample["sample_index"]]["step"] == sample["optimizer_step"] == step
                assert sample["free_coordinate_count"] == 9 * (int(sample["y"]) - sample["K"])
        registered[recipe] = paired
    before = {p.name: (digest(p), p.stat().st_mtime_ns) for p in (out / "confirmation_ot_cache").iterdir()}
    repeat = confirm.precompute(data, out, workers=1)
    assert repeat["actual_OT_attempts"] == repeat["actual_OT_returns"] == 0 and repeat["OT_cache_hits"] == maps
    assert before == {p.name: (digest(p), p.stat().st_mtime_ns) for p in (out / "confirmation_ot_cache").iterdir()}


def test_pure_ot_and_hybrid_share_targets_maps_but_not_actual_input(data, tmp_path):
    out = tmp_path / "stream"
    prepare_fixture(data, out)
    confirm.precompute(data, out, workers=2)
    h = confirm.ConfirmationStream(data, out, "R2")
    p = confirm.ConfirmationStream(data, out, "R4")
    for hybrid, pure in zip(h.samples(1), p.samples(1)):
        assert hybrid["ot_cache_key"] == pure["ot_cache_key"]
        np.testing.assert_array_equal(hybrid["x1"], pure["x1"])
        np.testing.assert_array_equal(hybrid["t"], pure["t"])
        n, k = int(pure["y"]), pure["K"]
        expected = np.zeros((n, 3, 3), np.float32)
        expected[k:] = pure["free_gaussian_before_OT"]
        expected = expected[pure["face_permutation"]].reshape(n, 9)
        np.testing.assert_array_equal(expected, pure["x0"])
        np.testing.assert_array_equal(pure["xt"][pure["known_mask"]], pure["context"][pure["known_mask"]])
        assert hybrid["lambda_value"] == .25 and pure["lambda_value"] == 0
        assert hybrid["shared_input_sha256"] != pure["shared_input_sha256"]
    h.next_effective_batch()
    state = h.state_dict()
    resumed = confirm.ConfirmationStream(data, out, "R2")
    resumed.load_state_dict(state)
    assert [s["shared_input_sha256"] for s in h.next_effective_batch()] == [s["shared_input_sha256"] for s in resumed.next_effective_batch()]
    assert h.last_audit == resumed.last_audit
    with pytest.raises(ValueError, match="identity"):
        p.load_state_dict(state)
    with pytest.raises(ValueError, match="identity"):
        resumed.load_state_dict(dict(state, namespace=NS + "other"))
    with pytest.raises(ValueError, match="cursor"):
        resumed.load_state_dict(dict(state, batch_index=7, consumed_samples=56))
    assert confirm.recipes.lambda_for_step("R4", 4000) == 0
    assert confirm.recipes.lambda_for_step("R4", 4001) == .25


def test_course_actual_inputs_are_same_multiset_and_same_final_suffix(data, tmp_path):
    out = tmp_path / "stream"
    prepare_fixture(data, out, ("R2", "C20_40_MIX"), {"C20_40_MIX": "all"})
    confirm.precompute(data, out, workers=2)
    mix = confirm.ConfirmationStream(data, out, "R2")
    course = confirm.ConfirmationStream(data, out, "C20_40_MIX")
    a = {s["sample_index"]: s for step in range(1, 7) for s in mix.samples(step)}
    for step in range(1, 7):
        samples = course.samples(step)
        for sample in samples:
            assert sample["shared_input_sha256"] == a[sample["sample_index"]]["shared_input_sha256"]
        if step > 4:
            assert [s["sample_index"] for s in samples] == [s["sample_index"] for s in mix.samples(step)]


def test_foreign_cached_arrays_and_unselected_recipe_are_rejected(data, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    prepare_fixture(data, a)
    prepare_fixture(data, b, namespace=NS + "other")
    confirm.precompute(data, a, workers=2)
    shutil.copytree(a / "confirmation_ot_cache", b / "confirmation_ot_cache")
    stream = confirm.ConfirmationStream(data, b, "R2")
    before = {p.name: digest(p) for p in stream.cache_dir.iterdir()}
    with pytest.raises(RuntimeError, match="historical cache substitution"):
        stream.samples(1)
    assert before == {p.name: digest(p) for p in stream.cache_dir.iterdir()}
    with pytest.raises(ValueError, match="outside the selected"):
        confirm.ConfirmationStream(data, b, "R1")


def test_changed_plan_and_incomplete_ot_attempt_fail_before_execution(data, tmp_path):
    out = tmp_path / "stream"
    prepare_fixture(data, out)
    path = out / "confirmation_plan.json"
    original = read_json(path)
    changed = copy.deepcopy(original)
    row = next(r for r in changed["mixed_records"] if r["ratio"] == .2)
    row["task_id"] = next(t for t in changed["eligible20_tasks"][row["uid"]] if t != row["task_id"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="does not reproduce"):
        confirm.ConfirmationStream(data, out, "R2")
    changed = dict(original, base_stream_root="forbidden_historical_stream")
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="does not reproduce"):
        confirm.ConfirmationStream(data, out, "R2")
    atomic_json(path, original)
    cache = out / "confirmation_ot_cache"
    cache.mkdir()
    atomic_json(cache / "orphan.attempt.json", {"status": "INCOMPLETE"})
    with pytest.raises(RuntimeError, match="no silent retry"):
        confirm.precompute(data, out, workers=1)
    assert [p.name for p in cache.iterdir()] == ["orphan.attempt.json"]
    assert not (out / "paired_inputs").exists()