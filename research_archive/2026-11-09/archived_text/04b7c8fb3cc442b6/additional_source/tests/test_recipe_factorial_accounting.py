"""CPU-only tests for accounting semantics; no numerical/model imports."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "tools/audit_recipe_factorial_training.py"
spec = importlib.util.spec_from_file_location("recipe_accounting", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_lambda_boundary():
    assert m.lam(m.ARMS[0], 4000) == .25
    for arm in m.ARMS[1:]:
        assert m.lam(arm, 4000) == 0.
        assert m.lam(arm, 4001) == .25


def values(index):
    planned = dict(sample_index=index, uid="parent" + str(index % 8),
        task_id="task" + str(index % 8), N=200, K=80, ratio=.4, alpha=1. if index % 2 == 0 else 1.05)
    paired = dict(sample_index=index, origin_sample_index=index,
        task_id=planned["task_id"], free_coordinate_count=1080, lambda_value=0.,
        shared_input_sha256="s" + str(index), input_bytehash="i" + str(index),
        label_bytehash="l" + str(index), ot_cache_key="o" + str(index))
    exposure = dict(recipe=m.ARMS[1], attempt_id="initial", step=index // 8 + 1,
        **planned, origin_sample_index=index, t=.5, lambda_value=0.,
        free_coordinates=1080, ot_cache_status="HIT",
        **{k: paired[k] for k in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key")})
    return planned, paired, exposure


def test_exposure_binds_inputs_and_supervision():
    plan, pair, exposure = values(0)
    assert m.compare_exposure(exposure, plan, pair, m.ARMS[1]) == []
    for field, value in [("K", 40), ("input_bytehash", "changed"), ("alpha", 1.1),
                         ("free_coordinates", 100), ("ot_cache_status", "MISS"),
                         ("lambda_value", .25), ("step", 2)]:
        assert m.compare_exposure(dict(exposure, **{field: value}), plan, pair, m.ARMS[1]), field


def put_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def put_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def call_rows(step, attempt, start=1, terminal="CALL_DONE"):
    rows = []; ident = start
    for micro in range(8):
        index = (step - 1) * 8 + micro
        for kind in ("train_forward", "train_backward"):
            context = dict(attempt_id=attempt, segment=m.ARMS[1], call_id=ident,
                kind=kind, step=step, microbatch=micro, sample_index=index,
                input_bytehash="i" + str(index), label_bytehash="l" + str(index),
                time_unix=step * 100 + ident)
            rows += [dict(context, event="CALL_BEGIN"), dict(context, event=terminal)]
            ident += 1
    context = dict(attempt_id=attempt, segment=m.ARMS[1], call_id=ident,
                   kind="optimizer_update", step=step, time_unix=step * 100 + ident)
    rows += [dict(context, event="CALL_BEGIN"), dict(context, event=terminal)]
    return rows


def test_call_accounting_success_failure_and_inflight(tmp_path):
    pair = {i: values(i)[1] for i in range(8)}
    rows = call_rows(1, "initial")
    p = tmp_path / "calls.jsonl"; put_lines(p, rows)
    audit = m.Audit()
    steps, counts, unfinished = m.scan_calls(p, pair, m.ARMS[1], "initial", audit, True)
    assert steps[1]["optimizer_update:CALL_DONE"] == 1
    assert counts[("train_backward", "CALL_DONE")] == 8
    assert not unfinished
    assert all(c["status"] == "PASS" for c in audit.checks)
    rows[-1]["event"] = "CALL_FAILED"; put_lines(p, rows)
    audit = m.Audit(); steps, counts, _ = m.scan_calls(p, pair, m.ARMS[1], "initial", audit, True)
    assert not steps[1]["optimizer_update:CALL_DONE"]
    assert counts[("optimizer_update", "CALL_FAILED")] == 1
    put_lines(p, rows[:-1])
    audit = m.Audit(); _, _, unfinished = m.scan_calls(p, pair, m.ARMS[1], "initial", audit, False)
    assert len(unfinished) == 1
    assert audit.checks[-1]["status"] == "PENDING"


def test_unpaired_or_wrong_input_call_fails(tmp_path):
    rows = call_rows(1, "initial")
    rows[1]["input_bytehash"] = "not-frozen"
    rows[3]["call_id"] = 999
    p = tmp_path / "calls.jsonl"; put_lines(p, rows)
    a = m.Audit()
    m.scan_calls(p, {i: values(i)[1] for i in range(8)}, m.ARMS[1], "initial", a, True)
    assert any(c["status"] == "FAIL" for c in a.checks)


def fixture(tmp_path, replay=False, complete=True):
    arm = m.ARMS[1]; records = [values(i)[0] for i in range(16)]
    pairs = [values(i)[1] for i in range(16)]
    plan = dict(mixed_records=records, data_manifest_sha256="data")
    base = dict(records=records); auth = dict(updates_per_recipe=2)
    put_json(tmp_path / "streams/recipe_plan.json", plan)
    put_json(tmp_path / "registrations/stage1_training.json", auth)
    paired_path = tmp_path / "streams/paired_inputs" / (arm + ".json")
    put_json(paired_path, dict(records=pairs, plan_sha256=m.file_hash(tmp_path / "streams/recipe_plan.json")))
    groups = []
    for i in range(8):
        groups.append(dict(recipe=arm, uid="parent" + str(i), ratio=".4", condition_family="toy",
            planned_samples="2", planned_free_scalar_supervision="2160",
            planned_clean_samples="2" if i % 2 == 0 else "0",
            planned_augmented_samples="0" if i % 2 == 0 else "2",
            planned_step_1_2000_samples="2", planned_step_2001_4000_samples="0",
            planned_step_4001_6000_samples="0"))
    folder = tmp_path / "training" / arm
    for attempt, updates in ([("initial", [1, 2]), ("oom_serial_retry", [2])] if replay else [("initial", [1, 2])]):
        directory = folder / "attempts" / attempt
        directory.mkdir(parents=True)
        logs=[]; exposures=[]; calls=[]; events=[]; call_id=1
        if attempt != "initial":
            events.append(dict(event="RESUME", step=1, optimizer_reset=False))
        for step in updates:
            ex = [dict(values(i)[2], attempt_id=attempt) for i in range((step - 1) * 8, step * 8)]
            exposures.extend(ex)
            hashes = {k: m.json_hash([r[sample] for r in ex]) for k, sample in m.BATCH_HASH.items()}
            logs.append(dict(recipe=arm, attempt_id=attempt, step=step, lambda_value=0.,
                FM=.5, total_loss=.5, generator_gradient_norm=.1,
                generator_adamw_step=step, ratio20_samples=0, ratio40_samples=8,
                free_coordinate_denominator=8640, free20_coordinates=0, free40_coordinates=8640,
                input_seconds=.1, forward_seconds=.2, backward_seconds=.3,
                optimizer_seconds=.1, update_seconds=.6,
                cuda_peak_allocated_bytes=100, cuda_peak_reserved_bytes=200, **hashes))
            events += [dict(event="INPUT_DONE", step=step, **hashes),
                       dict(event="UPDATE_DONE", step=step, generator_adamw_step=step)]
            batch = call_rows(step, attempt, call_id)
            if attempt != "initial":
                for row in batch: row["time_unix"] += 10000
            calls.extend(batch); call_id += 17
        m.write_csv(directory / "training_log.csv", logs)
        m.write_csv(directory / "exposure.csv", exposures)
        put_lines(directory / "calls.jsonl", calls)
        put_lines(directory / "events.jsonl", events)
        state = dict(status="OOM" if replay and attempt == "initial" else "COMPLETE",
            physical_optimizer_updates_this_attempt=len(updates),
            successful_calls=dict(train_forward=8*len(updates), train_backward=8*len(updates),
                                  optimizer_update=len(updates)), wall_seconds=len(updates))
        if complete: put_json(directory / "final_status.json", state)
    if complete:
        checkpoint=tmp_path / "checkpoints" / (arm + "_step6000.pt")
        export=tmp_path / "inference" / (arm + "_generator_step6000.pt")
        receipts=[]
        for path in (checkpoint, export):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic opaque endpoint; never deserialized")
            receipts.append(dict(path=str(path), sha256=m.file_hash(path), global_step=2, recipe=arm,
                generator_state_sha256="same_generator", optimizer_state_sha256=None, rng_sha256=None))
        put_json(checkpoint.with_suffix(".json"), receipts[0])
        # Export intentionally has no sidecar: this matches the real worker contract.
        put_json(folder / "completed.json", dict(status="COMPLETE", step=2,
            identities=dict(plan_sha256=m.file_hash(tmp_path / "streams/recipe_plan.json"),
                paired_inputs_sha256=m.file_hash(paired_path),
                authorization_sha256=m.file_hash(tmp_path / "registrations/stage1_training.json"),
                data_manifest_sha256="data"),
            checkpoint=receipts[0], inference_export=receipts[1]))
    return arm, plan, base, auth, groups


def do_fixture(tmp_path, replay=False, complete=True):
    arm, plan, base, auth, groups = fixture(tmp_path, replay, complete)
    audit=m.Audit()
    tables={k: [] for k in ("updates", "exposures", "groups", "calls", "attempts")}
    result=m.audit_arm(tmp_path, arm, plan, base, auth, groups, audit, tables,
        {"task" + str(i): "toy" for i in range(8)})
    return result, audit, tables


def test_complete_accounting_and_export_without_sidecar(tmp_path):
    result, audit, tables = do_fixture(tmp_path)
    assert result["completion_state"] == "COMPLETE"
    assert result["unique_successful_updates"] == 2
    assert result["physical_successful_optimizer_updates"] == 2
    assert result["successful_replayed_updates"] == 0
    assert all(c["status"] == "PASS" for c in audit.checks), audit.checks
    assert len(tables["exposures"]) == 16
    assert all(r["final_logical_trajectory"] for r in tables["exposures"])
    assert all(r["actual_unique_free_scalar_supervision"] == 2160 for r in tables["groups"])


def test_replay_counted_physically_but_final_exposure_not_duplicated(tmp_path):
    result, audit, tables = do_fixture(tmp_path, replay=True)
    assert result["unique_successful_updates"] == 2
    assert result["physical_successful_optimizer_updates"] == 3
    assert result["successful_replayed_updates"] == 1
    assert len(tables["exposures"]) == 24
    assert sum(r["final_logical_trajectory"] for r in tables["exposures"]) == 16
    assert sum(r["physical_optimizer_free_scalar_supervision"] for r in tables["groups"]) == 25920
    assert all(c["status"] == "PASS" for c in audit.checks), audit.checks


def test_running_arm_is_pending_even_when_all_test_rows_seen(tmp_path):
    result, audit, tables = do_fixture(tmp_path, complete=False)
    assert result["completion_state"] == "PENDING"
    assert not result["endpoint_files"]
    assert any(c["status"] == "PENDING" for c in audit.checks)
    assert not any(c["status"] == "FAIL" for c in audit.checks)


def test_mutated_endpoint_hash_fails_and_latest_is_forbidden(tmp_path):
    p=tmp_path/"final.pt"; p.write_bytes(b"original")
    expected=m.file_hash(p); p.write_bytes(b"altered")
    a=m.Audit()
    receipt=m.verify_file_receipt(p, dict(sha256=expected), a, m.ARMS[0], "checkpoint")
    assert receipt["status"] == "FAIL"
    try:
        m.verify_file_receipt(tmp_path/"latest.pt", {}, a, m.ARMS[0], "checkpoint")
        assert False
    except ValueError:
        pass


def test_partial_log_line_only_pending_for_live_file(tmp_path):
    p=tmp_path/"events.jsonl";p.write_bytes(b'{"event":"x"}\n{"event":')
    a=m.Audit()
    assert a.snapshot(p, False) == '{"event":"x"}\n'
    assert not a.checks
    a.snapshot(p, True)
    assert a.checks[-1]["status"] == "FAIL"


def test_module_is_stdlib_only():
    tree=__import__("ast").parse(SOURCE.read_text(encoding="utf-8-sig"))
    imports=[]
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Import): imports += [x.name for x in node.names]
        if isinstance(node, __import__("ast").ImportFrom): imports += [node.module]
    assert all(name.split(".")[0] in sys.stdlib_module_names for name in imports)

def test_same_parent_ratio_keeps_two_condition_families_separate():
    legacy = dict(uid="p", ratio=.2, task_id="old")
    new = dict(uid="p", ratio=.2, task_id="new")
    families = {"old": "MVP_LEGACY20", "new": "DOMAIN_NEW20"}
    assert m.group_key(legacy, families) != m.group_key(new, families)
    assert m.condition_family(dict(ratio=.2, old_patch_id="p0")) == "MVP_LEGACY20"
    assert m.condition_family(dict(ratio=.2)) == "DOMAIN_NEW20"
    assert m.condition_family(dict(ratio=.4)) == "DOMAIN_NEW40"
