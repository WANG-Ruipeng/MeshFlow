"""Temporary synthetic ledgers only; no real output, model or score access."""
from pathlib import Path
import importlib.util
import json
import sys

spec=importlib.util.spec_from_file_location("course_sampling_audit",Path(__file__).with_name("audit_sampling.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def events(attempt="initial"):
    ctx=dict(label="C20_40_MIX",request_id="synthetic",attempt=attempt,phase="course",
             checkpoint_sha256="checkpoint",input_sha256="input")
    rows=[dict(event="CELL_REGISTERED",context=ctx),
          dict(event="RESERVE",kind="rollout_attempt",ordinal=1,context=ctx)]
    for step in range(50):
        rows.extend([dict(event="RESERVE",kind="sampling_forward",ordinal=step+1,
                          context=dict(ctx,euler_step=step)),dict(event="RETURN",kind="sampling_forward")])
    rows.append(dict(event="ROLLOUT_COMPLETE",nfe=50,known_all51_bitwise=True))
    return ctx,rows

def saved(tmp_path,rows,newline=True):
    p=tmp_path/"calls.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows)+("\n" if newline else ""))
    return p

def test_exact50_course_forward_and_zero_retry(tmp_path):
    ctx,rows=events();counts,errors,h=a.ledger(saved(tmp_path,rows),ctx)
    assert not errors and counts["sampling_forward_returned"]==50
    assert counts["rollout_attempt_reserved"]==counts["ROLLOUT_COMPLETE"]==1
    assert h==a.sha(tmp_path/"calls.jsonl")
    assert "torch" not in sys.modules

def test_serial_dispatch_can_start_untouched_cell_without_cell_retry(tmp_path):
    ctx,rows=events("oom_serial_retry");counts,errors,_=a.ledger(saved(tmp_path,rows),ctx)
    assert not errors and counts["rollout_attempt_reserved"]==1

def test_wrong_phase_rejected(tmp_path):
    ctx,rows=events();wrong=dict(ctx,phase="main")
    _,errors,_=a.ledger(saved(tmp_path,rows),wrong)
    assert any("context" in s for s in errors)

def test_missing_forward_terminal_and_truncated_tail_rejected(tmp_path):
    ctx,rows=events();rows.pop(-2)
    counts,errors,_=a.ledger(saved(tmp_path,rows,newline=False),ctx)
    assert errors and counts["forward_unfinished"]==1
    assert any("truncated" in s for s in errors)

def test_duplicate_rollout_and_failed_call_recorded(tmp_path):
    ctx,rows=events()
    rows.insert(2,dict(event="RESERVE",kind="rollout_attempt",ordinal=2,context=ctx))
    rows[4]=dict(event="ERROR",kind="sampling_forward")
    counts,errors,_=a.ledger(saved(tmp_path,rows),ctx)
    assert errors and counts["rollout_attempt_reserved"]==2
    assert counts["sampling_forward_failed"]==1 and counts["ERROR"]==1

def test_readiness_needs_both_labels_complete_and128_receipts(tmp_path):
    phase=tmp_path/"course"
    assert len(a.readiness(phase))==3
    for label in a.COURSES:
        p=phase/"sampling"/label/"completed.json";p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(status="COMPLETE")))
        for i in range(64):
            r=phase/"outputs"/label/str(i)/"receipt.json";r.parent.mkdir(parents=True)
            r.write_text("{}")
    assert a.readiness(phase)==[]
    # Extra receipts must reach the actual orphan detector, not wait indefinitely.
    r=phase/"outputs"/"extra"/"x"/"receipt.json";r.parent.mkdir(parents=True);r.write_text("{}")
    assert a.readiness(phase)==[]
    (phase/"sampling"/a.COURSES[0]/"completed.json").write_text('{"status":"PARTIAL_BLOCKED"}')
    assert a.readiness(phase)

def test_source_is_independent_and_original_parser_is_frozen():
    assert a.sha(a.OLD_AUDITOR)==a.OLD_AUDITOR_SHA
    assert a.ledger is a._prior.ledger
    assert a._prior.DEST==a.RUN/"audit/sampling_accounting"
    assert a.DEST==a.RUN/"audit/curriculum_sampling_accounting"
