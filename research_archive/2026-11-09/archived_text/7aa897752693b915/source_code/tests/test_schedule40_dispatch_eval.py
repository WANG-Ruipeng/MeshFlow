"""Meaningful CPU-only tests for immutable geometry and OOM wave recovery."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py')); obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

def test_expand_preserves_C_and_coordinates_with_source_ties():
    m=module('evaluate_schedule40')
    C=np.array([[[0,0,0],[1,0,0],[0,1,0]]],dtype=np.float32)
    F=np.concatenate([C+[0,0,1],C+[0,0,-1],C+[0,0,3]]).astype(np.float32)
    out,ids=m.expand_condition(C,F,3)
    assert ids.tolist()==[0,1]
    assert out[:1].tobytes()==C.tobytes()
    assert out[1:].tobytes()==F[:2].tobytes()
    with pytest.raises(ValueError): m.expand_condition(C.astype(np.float64),F,3)

def test_slot_formula_reserves_headroom_and_respects_RAM():
    m=module('run_schedule40'); r=dict(gpu=dict(total_mib=16303,free_mib=10000),ram=dict(MemAvailable=32*1024**3),cpu_cores=24)
    assert m.slots(r,6000)==1
    r['gpu']['free_mib']=8500; assert m.slots(r,6000)==0
    r['gpu']['free_mib']=14000; r['ram']['MemAvailable']=8*1024**3; assert m.slots(r,6000)==0

def setup_fake(monkeypatch,tmp_path,dual_failure=True,replay_step=42):
    m=module('run_schedule40'); out=tmp_path/'experiments'/'chair_early_alignment_schedule40_v1'; out.mkdir(parents=True)
    r=dict(gpu=dict(name='fake',total_mib=16303,used_mib=6000,free_mib=10303,utilization=20),ram=dict(MemAvailable=32*1024**3),cpu_cores=24)
    monkeypatch.setattr(m,'snapshot',lambda *a,**k:r); monkeypatch.setattr(m.time,'sleep',lambda x:None)
    launched=[]; current=[]
    def dump(p,d): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d))
    class Fake:
        def __init__(self,cmd,**kwargs):
            self.name=cmd[cmd.index('--segment')+1]; self.attempt=cmd[cmd.index('--attempt-id')+1]; self.pid=100+len(launched); self.done=False
            launched.append((self.name,self.attempt)); current.append(self)
            dump(out/'training'/self.name/'progress.json',dict(attempt_id=self.attempt,physical_optimizer_updates_this_attempt=16,cuda_peak_reserved_bytes=3500*1024**2))
        def poll(self):
            if self.done: return self.ret
            if self.attempt=='initial':
                if dual_failure and len([v for v in launched if v[1]=='initial'])<2: return None
                dump(out/'training'/self.name/'attempts'/self.attempt/'final_status.json',dict(status='OOM',attempt_id=self.attempt,step=replay_step,error='OutOfMemoryError'))
                self.ret=1
            else:
                dump(out/'training'/self.name/'completed.json',dict(status='COMPLETE')); self.ret=0
            self.done=True; return self.ret
    monkeypatch.setattr(m.subprocess,'Popen',Fake)
    jobs=[dict(name='EARLY',end=6000,phase='training',dependencies=[])]
    if dual_failure: jobs.append(dict(name='COMMON',end=5000,phase='training',dependencies=[]))
    return m,out,jobs,launched

def test_two_OOM_workers_have_one_downgrade_two_serial_recoveries(monkeypatch,tmp_path):
    m,out,jobs,launched=setup_fake(monkeypatch,tmp_path)
    result=m.dispatch(out,jobs)
    assert result['status']=='COMPLETE' and result['oom_downgrades']==1
    assert sorted(launched)==[('COMMON','initial'),('COMMON','oom_serial_retry'),('EARLY','initial'),('EARLY','oom_serial_retry')]
    ledger=[json.loads(x) for x in (out/'audit/controller.jsonl').read_text().splitlines()]
    assert [x['worker_count'] for x in ledger if x['event']=='WORKER_START']==[1,2,1,1]
    assert [x['completed_updates_to_reexecute'] for x in ledger if x['event']=='RECOVERY_ACCOUNTING']==[42,42]
    assert not json.loads((out/'audit/dispatch_state.json').read_text())['failed']

def test_serial_OOM_does_not_retry(monkeypatch,tmp_path):
    m,out,jobs,launched=setup_fake(monkeypatch,tmp_path,dual_failure=False)
    result=m.dispatch(out,jobs)
    assert result['status']=='PARTIAL_BLOCKED'
    assert launched==[('EARLY','initial')]
    assert result['oom_downgrades']==0

def test_replay_more_than99_stops_without_retraining(monkeypatch,tmp_path):
    m,out,jobs,launched=setup_fake(monkeypatch,tmp_path,replay_step=100)
    result=m.dispatch(out,jobs)
    assert result['status']=='PARTIAL_BLOCKED'
    assert len(launched)==2
    assert all(attempt=='initial' for _,attempt in launched)