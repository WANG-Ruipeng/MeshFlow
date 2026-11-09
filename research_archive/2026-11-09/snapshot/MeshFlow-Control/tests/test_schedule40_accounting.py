"""CPU accounting contracts: partial logs, failed calls, and receipt-only forks."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "schedule40_accounting", Path(__file__).parents[1]/"tools/analyze_schedule40.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows, trailing=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows)+trailing, encoding="utf-8")


def test_partial_json_line_is_pending_not_a_completed_call(tmp_path):
    path = tmp_path/"calls.jsonl"
    jsonl(path, [{"event": "CALL_BEGIN", "call_id": 1}], trailing='{"event":"CALL_D')
    warnings, errors = [], []
    rows = list(audit.json_rows(path, warnings, errors))
    assert len(rows) == 1
    assert warnings[0]["reason"] == "TRAILING_PARTIAL_LINE_IGNORED"
    assert not errors


def test_malformed_complete_line_is_a_real_error(tmp_path):
    path = tmp_path/"calls.jsonl"
    path.write_text('{"broken":}\n', encoding="utf-8")
    warnings, errors = [], []
    assert list(audit.json_rows(path, warnings, errors)) == []
    assert errors[0]["reason"] == "MALFORMED_COMPLETE_JSON_LINE"
    assert not warnings


def test_optimizer_return_without_update_log_is_visible(tmp_path):
    folder = tmp_path/"attempts/initial"
    jsonl(folder/"events.jsonl", [
        dict(event="INPUT_BEGIN", step=1, time_unix=1.),
        dict(event="UPDATE_BEGIN", step=1, time_unix=2.)])
    jsonl(folder/"calls.jsonl", [
        dict(event="CALL_BEGIN", call_id=1, kind="optimizer_update", step=1),
        dict(event="CALL_DONE", call_id=1, kind="optimizer_update", step=1, seconds=.1),
        dict(event="CALL_BEGIN", call_id=2, kind="train_forward", step=2),
        dict(event="CALL_FAILED", call_id=2, kind="train_forward", step=2, error="OOM"),
        dict(event="CALL_BEGIN", call_id=3, kind="optimizer_update", step=2),
        dict(event="CALL_FAILED", call_id=3, kind="optimizer_update", step=2, error="sync failed")])
    warnings, errors = [], []
    parsed = audit.parse_attempt(folder, "EARLY", warnings, errors)
    assert parsed["updates"][1]["step_status"] == "OPTIMIZER_RETURNED_UPDATE_AUDIT_INCOMPLETE"
    assert parsed["updates"][1]["optimizer_call_done"] == 1
    assert parsed["updates"][2]["step_status"] == "OPTIMIZER_CALL_FAILED_APPLICATION_UNKNOWN"
    assert len(parsed["failed_calls"]) == 2
    assert not parsed["unfinished_calls"]
    assert not errors


def test_missing_return_is_not_success(tmp_path):
    folder = tmp_path/"attempts/initial"
    jsonl(folder/"calls.jsonl", [
        dict(event="CALL_BEGIN", call_id=1, kind="optimizer_update", step=1)])
    warnings, errors = [], []
    parsed = audit.parse_attempt(folder, "COMMON", warnings, errors)
    assert parsed["updates"][1]["optimizer_call_done"] == 0
    assert len(parsed["unfinished_calls"]) == 1
    assert parsed["updates"][1]["step_status"] == "ATTEMPTED_NO_OPTIMIZER_RETURN"


def test_complete_fork_requires_all_state_receipt_hashes(tmp_path):
    common = dict(global_step=5000, sha256="checkpoint", generator_state_sha256="generator",
                  optimizer_state_sha256="optimizer", rng_sha256="rng")
    save(tmp_path/"checkpoints/COMMON_step5000.json", common)
    save(tmp_path/"training/EARLY/initialization_audit.json", dict(head_initial_sha256="head"))
    for segment in ("FM_ONLY", "LATE"):
        line = dict(fork_step=5000, generator_optimizer_reset=False,
                    common_checkpoint_sha256="checkpoint", common_generator_sha256="generator",
                    common_optimizer_sha256="optimizer", common_rng_sha256="rng",
                    fork_destination=segment, head_initial_sha256="head" if segment=="LATE" else None)
        save(tmp_path/"training"/segment/"completed.json", dict(lineage=line))
    result = audit.fork_audit(tmp_path)
    assert all(row["status"] == "PASS" for row in result["rows"])
    assert result["early_late_head_initial_sha256_equal"] is True
    path = tmp_path/"training/LATE/completed.json"
    changed = json.loads(path.read_text())
    changed["lineage"]["common_rng_sha256"] = "different"
    save(path, changed)
    result = audit.fork_audit(tmp_path)
    assert result["rows"][1]["status"] == "FAIL"
    assert result["rows"][1]["checks"]["RNG_sha256"] is False


def test_parallel_owner_times_are_not_added_as_wall_time():
    assert audit.union_seconds([(0, 5), (2, 7), (9, 10)]) == 8
    assert audit.union_seconds([]) == 0
