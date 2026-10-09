"""CPU contract tests; tiny tensors only, no generator forward or GPU calls."""
import copy
import numpy as np
import pytest
import torch
from meshflow_control.training import recipe_factorial as r


class ToyGenerator(torch.nn.Linear):
    readout_mode = "none"

    def set_trainable(self, value):
        for p in self.parameters():
            p.requires_grad_(value)
        return self


def objects():
    model = ToyGenerator(3, 2).eval()
    return model, r.make_optimizer(model)


def moments(optimizer, step):
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p] = dict(step=torch.tensor(float(step)),
                exp_avg=torch.full_like(p, .01), exp_avg_sq=torch.full_like(p, .02))


def mock_generator(monkeypatch):
    monkeypatch.setattr(r, "_check_generator", lambda _: None)
    monkeypatch.setattr(r, "build_model", lambda *a, **k: ToyGenerator(3, 2).eval())


def payload(monkeypatch, step=4000, recipe="R3_40_H_LATE", namespace="stage1"):
    mock_generator(monkeypatch)
    model, optimizer = objects()
    if step:
        moments(optimizer, step)
    return r._payload(model, optimizer, recipe=recipe,
        hybrid_schedule=r.resolved_schedule(recipe), namespace=namespace, step=step,
        stream_state={"batch_index": step, "sample_index": 8*step, "namespace": namespace},
        identities={"plan": "frozen", "authorization": "registered"},
        lineage={"official": True}, receipt={"test": True})


def sample_batch(recipe="R3_40_H_LATE", step=4000, schedule="late"):
    ratios = [0.4]*8 if recipe in ("R1_40_H_ALL", "R3_40_H_LATE") else [.2, .4]*4
    if recipe == "C20_40_MIX" and step <= 4000:
        ratios = [0.2 if step <= 2000 else 0.4]*8
    if recipe == "C40_20_MIX" and step <= 4000:
        ratios = [0.4 if step <= 2000 else 0.2]*8
    return [dict(object_id=str(i), alpha=1. if i < 4 else .95,
        step=step, lambda_value=r.lambda_at_step(recipe, step, schedule),
        y=np.asarray(200), K=round(ratio*200), ratio=ratio,
        free_coordinate_count=9*(200-round(ratio*200))) for i, ratio in enumerate(ratios)]


def test_exact_hybrid_boundaries_and_exposure():
    assert all(r.lambda_at_step("R2_MIX_H_ALL", k) == .25 for k in range(1, 6001))
    for recipe in ("R3_40_H_LATE", "R4_MIX_H_LATE"):
        values = [r.lambda_at_step(recipe, k) for k in range(1, 6001)]
        assert values.count(0.) == 4000 and values.count(.25) == 2000
        assert values[3999] == 0. and values[4000] == .25
    with pytest.raises(ValueError):
        r.lambda_at_step("R3_40_H_LATE", 0)
    with pytest.raises(ValueError):
        r.lambda_at_step("R3_40_H_LATE", True)
    with pytest.raises(ValueError):
        r.resolved_schedule("R3_40_H_LATE", "all")
    with pytest.raises(ValueError):
        r.resolved_schedule("C20_40_MIX")


def test_recipe_switch_preserves_adamw_without_reset():
    model, optimizer = objects()
    moments(optimizer, 4000)
    before = copy.deepcopy(optimizer.state_dict())
    assert r.lambda_at_step("R3_40_H_LATE", 4000) == 0
    assert r.lambda_at_step("R3_40_H_LATE", 4001) == .25
    assert r.tree_hash(before) == r.tree_hash(optimizer.state_dict())
    for p in model.parameters():
        p.grad = torch.full_like(p, .1)
    optimizer.step()
    assert r.validate_optimizer(model, optimizer, 4001)
    assert all(torch.count_nonzero(optimizer.state[p]["exp_avg"]) for p in model.parameters())


def test_fresh_adamw_is_empty_and_no_head():
    model, optimizer = objects()
    assert not optimizer.state and len(optimizer.param_groups) == 1
    assert r.validate_optimizer(model, optimizer, 0)
    with pytest.raises(ValueError):
        r.validate_optimizer(model, optimizer, 1)


def test_sample_schedule_and_original_free_denominator():
    for recipe in ("R1_40_H_ALL", "R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE"):
        schedule = r.resolved_schedule(recipe)
        for step in (1, 4000, 4001, 6000):
            samples = sample_batch(recipe, step, schedule)
            assert r.validate_samples(samples, recipe, step, schedule) == r.lambda_at_step(recipe, step, schedule)
            expected = 8*9*120 if recipe in ("R1_40_H_ALL", "R3_40_H_LATE") else 4*9*(120+160)
            assert sum(s["free_coordinate_count"] for s in samples) == expected
    bad = sample_batch()
    bad[0]["lambda_value"] = .25
    with pytest.raises(ValueError, match="lambda"):
        r.validate_samples(bad, "R3_40_H_LATE", 4000, "late")


def test_optional_curriculum_boundaries():
    for recipe in ("C20_40_MIX", "C40_20_MIX"):
        for step in (1, 2000, 2001, 4000, 4001, 6000):
            for schedule in ("all", "late"):
                r.validate_samples(sample_batch(recipe, step, schedule), recipe, step, schedule)


def test_update_delegates_exact_no_auxiliary_path(monkeypatch):
    seen = {}
    model, optimizer = objects()
    samples = sample_batch()
    ledger = object()
    def inherited(*args):
        seen["args"] = args
        return dict(alignment_active=False, aligned_samples=0, head_adamw_step=0)
    monkeypatch.setattr(r.baseline, "update", inherited)
    result = r.update(model, optimizer, samples, ledger, "R3_40_H_LATE", 4000, "late")
    assert seen["args"] == (model, None, optimizer, samples, None, ledger, "COMMON", 4000)
    assert result["lambda_value"] == 0


def test_atomic_complete_resume_preserves_all_state(tmp_path, monkeypatch):
    value = payload(monkeypatch)
    path = tmp_path/"resume.pt"
    receipt = r._write_checkpoint(path, value, immutable=True)
    assert receipt["sha256"] == r.digest(path)
    model, optimizer, restored = r.load_checkpoint(path, recipe="R3_40_H_LATE",
        hybrid_schedule="late", namespace="stage1", device="cpu")
    assert r.state_hash(model.state_dict()) == value["generator_state_sha256"]
    assert r.tree_hash(optimizer.state_dict()) == value["optimizer_state_sha256"]
    assert r.tree_hash(restored["rng"]) == value["rng_sha256"]
    assert restored["stream_state"] == value["stream_state"]
    assert restored["next_lambda"] == .25
    assert r.validate_optimizer(model, optimizer, 4000)
    with pytest.raises(FileExistsError):
        r._write_checkpoint(path, value, immutable=True)


def test_resume_rejects_other_recipe_namespace_and_corruption(tmp_path, monkeypatch):
    value = payload(monkeypatch)
    path = tmp_path/"resume.pt"
    r._write_checkpoint(path, value)
    with pytest.raises(ValueError, match="recipe"):
        r.load_checkpoint(path, recipe="R4_MIX_H_LATE")
    with pytest.raises(ValueError, match="input_namespace"):
        r.load_checkpoint(path, namespace="independent_confirm")
    broken = copy.deepcopy(value)
    broken["optimizer"]["state"][0]["step"] += 1
    with pytest.raises(ValueError, match="hash"):
        r.validate_payload(broken)
    broken = copy.deepcopy(value)
    broken["stream_state"]["batch_index"] += 1
    with pytest.raises(ValueError, match="cursor"):
        r.validate_payload(broken)


def test_pure_generator_export_final_only(tmp_path, monkeypatch):
    value = payload(monkeypatch, step=6000)
    source, target = tmp_path/"final.pt", tmp_path/"generator.pt"
    r._write_checkpoint(source, value)
    r.export_generator(source, target)
    model, identity = r.load_generator(target, device="cpu")
    saved = torch.load(target, map_location="cpu", weights_only=True)
    assert not set(("optimizer", "rng", "aligner_state", "teacher_state")) & set(saved)
    assert identity["recipe"] == "R3_40_H_LATE"
    assert r.state_hash(model.state_dict()) == value["generator_state_sha256"]
    assert not any(p.requires_grad for p in model.parameters())
    early = tmp_path/"not_final.pt"
    r._write_checkpoint(early, payload(monkeypatch, step=4000))
    with pytest.raises(ValueError, match="final"):
        r.export_generator(early, tmp_path/"forbidden.pt")


def test_authorization_allows_only_frozen_phase_jobs(tmp_path, monkeypatch):
    out = tmp_path/"stage1"
    registration = dict(schema=r.AUTH_SCHEMA, experiment=r.EXPERIMENT, namespace="stage1",
        out=str(out.resolve()), authorized_recipes=["R2_MIX_H_ALL"],
        hybrid_schedules={"R2_MIX_H_ALL": "all"}, updates_per_recipe=6000,
        effective_batch=8, microbatch=1, source_hashes={
            "src/meshflow_control/training/recipe_factorial.py": "source",
            "src/meshflow_control/data/recipe_factorial.py": "source",
            "src/meshflow_control/training/schedule40.py": "source",
            "tools/train_recipe_factorial.py": "source"})
    monkeypatch.setattr(r, "digest", lambda _: "source")
    path = tmp_path/"registration.json"
    r.atomic_json(path, registration)
    assert r.check_authorization(path, "R2_MIX_H_ALL", "all", "stage1", out)
    with pytest.raises(ValueError, match="authorized"):
        r.check_authorization(path, "C20_40_MIX", "all", "stage1", out)
    with pytest.raises(ValueError, match="authorized"):
        r.check_authorization(path, "R1_40_H_ALL", "all", "stage1", out)
    # R1 can be independently repeated only in a separately registered phase.
    registration.update(namespace="confirm", authorized_recipes=["R1_40_H_ALL"],
                        hybrid_schedules={"R1_40_H_ALL": "all"})
    r.atomic_json(path, registration)
    assert r.check_authorization(path, "R1_40_H_ALL", "all", "confirm", out)


def test_authorization_rejects_batch_source_and_output_mismatch(tmp_path, monkeypatch):
    out = tmp_path/"phase"
    registration = dict(schema=r.AUTH_SCHEMA, experiment=r.EXPERIMENT, namespace="stage1",
        out=str(out.resolve()), authorized_recipes=["R2_MIX_H_ALL"],
        hybrid_schedules={"R2_MIX_H_ALL": "all"}, updates_per_recipe=6000,
        effective_batch=4, microbatch=1, source_hashes={})
    path = tmp_path/"registration.json"
    r.atomic_json(path, registration)
    with pytest.raises(ValueError, match="effective_batch"):
        r.check_authorization(path, "R2_MIX_H_ALL", "all", "stage1", out)
    registration["effective_batch"] = 8
    r.atomic_json(path, registration)
    with pytest.raises(ValueError, match="source"):
        r.check_authorization(path, "R2_MIX_H_ALL", "all", "stage1", out)
    with pytest.raises(ValueError, match="out"):
        r.check_authorization(path, "R2_MIX_H_ALL", "all", "stage1", tmp_path/"other")


def test_resume_restores_actual_cpu_rng_sequences(tmp_path, monkeypatch):
    value = payload(monkeypatch)
    r.restore_rng(value["rng"])
    expected_np = np.random.randn(4)
    expected_torch = torch.randn(4)
    path = tmp_path/"rng.pt"
    r._write_checkpoint(path, value)
    np.random.seed(98)
    torch.manual_seed(76)
    _, _, restored = r.load_checkpoint(path)
    r.restore_rng(restored["rng"])
    assert np.array_equal(np.random.randn(4), expected_np)
    assert torch.equal(torch.randn(4), expected_torch)


def test_preupdate_oom_is_durable_and_never_self_retried(tmp_path, monkeypatch):
    from meshflow_control.data import recipe_factorial as data_module
    calls = []
    def fail_stream(*args, **kwargs):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("synthetic CPU-only exception fixture")
    monkeypatch.setattr(data_module, "RecipeFactorialStream", fail_stream)
    out = tmp_path/"run"
    with pytest.raises(torch.cuda.OutOfMemoryError):
        r.train_recipe("R2_MIX_H_ALL", "unused_official.pt", tmp_path/"data.json",
            tmp_path/"stream", out, "initial", tmp_path/"registration.json", device="cuda")
    failure = r.read_json(out/"training/R2_MIX_H_ALL/attempts/initial/final_status.json")
    assert failure["status"] == "OOM"
    assert failure["physical_optimizer_updates_this_attempt"] == 0
    assert failure["successful_calls"] == {}
    assert failure["stream_cursor"] is None
    assert calls == [1]
    assert not (out/"training/R2_MIX_H_ALL/completed.json").exists()
