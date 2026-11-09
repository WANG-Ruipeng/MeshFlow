"""CPU-only process mocks for the independent R1 dispatch recovery boundary."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("schedule40_dispatch_r1", REPO/"tools/run_schedule40_dispatch_r1.py")
r1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r1)


def mock_dispatch(tmp_path, monkeypatch, scenario, retry_oom=False):
    tick = {"value": 0}
    launched = []
    current = {}
    resource = dict(gpu=dict(name="CPU_MOCK_NO_DEVICE",total_mib=16384.,used_mib=4096.,
                            free_mib=12288.,utilization=10.),cpu_cores=24,
                    ram={"MemAvailable":64*1024**3},loadavg=[0.,0.,0.])
    def snapshot(out, reason, active=()):
        if reason == "dispatch_tick":
            tick["value"] += 1
            assert tick["value"] <= 12, "Unexpected extra retry/dispatch loop"
        return resource
    class Process:
        def __init__(self, command, **kwargs):
            self.name,self.attempt=command
            self.pid=1000+len(launched)
            self.started=tick["value"]
            self.final=None
            launched.append((self.name,self.attempt))
            current[self.name]=self
        def poll(self):
            if self.final is not None: return self.final
            outcome=None
            if self.attempt!="initial":
                if tick["value"]>=self.started+1: outcome="OOM" if retry_oom else "COMPLETE"
            elif scenario=="serial":
                if tick["value"]>=2: outcome="OOM"
            elif scenario=="successful_peer_then_solo_oom":
                if self.name=="A" and tick["value"]>=3: outcome="COMPLETE"
                if self.name=="B" and tick["value"]>=6: outcome="OOM"
            elif scenario=="same_tick_dual_oom":
                if tick["value"]>=3: outcome="OOM"
            elif scenario=="separate_tick_oom":
                if self.name=="A" and tick["value"]>=3: outcome="OOM"
                if self.name=="B" and tick["value"]>=4: outcome="OOM"
            else: raise AssertionError(scenario)
            if outcome is None: return None
            folder=tmp_path/"sampling"/self.name
            folder.mkdir(parents=True,exist_ok=True)
            if outcome=="COMPLETE":
                r1.atomic_json(folder/"completed.json",dict(status="COMPLETE",attempt=self.attempt))
                self.final=0
            else:
                r1.atomic_json(folder/(self.attempt+"_failure.json"),
                               dict(status="OOM",attempt=self.attempt,error="OutOfMemoryError: CPU MOCK"))
                self.final=1
            return self.final
    monkeypatch.setattr(r1,"snapshot",snapshot)
    monkeypatch.setattr(r1.subprocess,"Popen",Process)
    monkeypatch.setattr(r1,"job_command",lambda out,job,attempt:[job["name"],attempt])
    monkeypatch.setattr(r1,"progress",lambda out,job:dict(completed=1,attempt=current[job["name"]].attempt))
    monkeypatch.setattr(r1.time,"sleep",lambda seconds:None)
    names=("A",) if scenario=="serial" else ("A","B")
    jobs=[dict(name=n,phase="sampling",checkpoint="NO_WEIGHTS",dependencies=[]) for n in names]
    result=r1.dispatch(tmp_path,jobs)
    state=json.loads((tmp_path/"audit/dispatch_state.json").read_text())
    events=[json.loads(line) for line in (tmp_path/"audit/controller.jsonl").read_text().splitlines()]
    return result,state,events,launched


def test_previous_parallel_survivor_solo_oom_is_not_retried(tmp_path,monkeypatch):
    result,state,events,launched=mock_dispatch(tmp_path,monkeypatch,"successful_peer_then_solo_oom")
    assert launched==[("A","initial"),("B","initial")]
    assert state["failed"]["sampling:B"]["status"]=="BLOCKED"
    assert state["oom_downgrades"]==0
    assert result["status"]=="PARTIAL_BLOCKED"
    cleared=[r for r in events if r["event"]=="PARALLEL_RECOVERY_ELIGIBILITY_CLEARED"]
    assert len(cleared)==1 and cleared[0]["name"]=="B"
    assert not any(r["event"]=="OOM_DOWNGRADE" for r in events)


@pytest.mark.parametrize("retry_oom",[False,True])
def test_same_tick_dual_oom_has_at_most_one_serial_retry_each(tmp_path,monkeypatch,retry_oom):
    result,state,events,launched=mock_dispatch(tmp_path,monkeypatch,"same_tick_dual_oom",retry_oom)
    assert sorted(launched)==sorted([("A","initial"),("B","initial"),
                                     ("A","oom_serial_retry"),("B","oom_serial_retry")])
    assert state["attempts"]=={"sampling:A":2,"sampling:B":2}
    assert state["max_workers"]==1 and state["oom_downgrades"]==1
    retry_starts=[r for r in events if r["event"]=="WORKER_START" and r["attempt"]=="oom_serial_retry"]
    assert len(retry_starts)==2 and all(r["worker_count"]==1 for r in retry_starts)
    if retry_oom:
        assert result["status"]=="PARTIAL_BLOCKED"
        assert all(r["status"]=="BLOCKED" for r in state["failed"].values())
    else:
        assert result["status"]=="COMPLETE" and not state["failed"]


def test_initial_serial_oom_has_no_retry(tmp_path,monkeypatch):
    result,state,events,launched=mock_dispatch(tmp_path,monkeypatch,"serial")
    assert launched==[("A","initial")]
    assert state["failed"]["sampling:A"]["status"]=="BLOCKED"
    assert state["oom_downgrades"]==0
    assert not any(r["event"]=="OOM_DOWNGRADE" for r in events)


def test_survivor_after_peer_oom_loses_eligibility_before_later_tick(tmp_path,monkeypatch):
    result,state,events,launched=mock_dispatch(tmp_path,monkeypatch,"separate_tick_oom")
    assert launched.count(("A","initial"))==1 and launched.count(("A","oom_serial_retry"))==1
    assert launched.count(("B","initial"))==1 and ("B","oom_serial_retry") not in launched
    assert state["failed"]["sampling:B"]["status"]=="BLOCKED"
    assert state["oom_downgrades"]==1


def test_scientific_and_original_freeze_functions_are_unchanged():
    original=ast.parse((REPO/"tools/run_schedule40.py").read_text(encoding="utf-8-sig"))
    revision=ast.parse((REPO/"tools/run_schedule40_dispatch_r1.py").read_text(encoding="utf-8-sig"))
    before={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef)}
    after={n.name:ast.dump(n,include_attributes=False) for n in revision.body if isinstance(n,ast.FunctionDef)}
    assert set(after)-set(before)=={"freeze_dispatch_r1"}
    for name in set(before)-{"dispatch","main"}:
        assert before[name]==after[name], name+" changed outside recovery boundary"
