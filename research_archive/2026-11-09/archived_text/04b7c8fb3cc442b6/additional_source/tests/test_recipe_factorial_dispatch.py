"""CPU process mocks for finite recipe scheduling; no real workers/devices."""
import importlib.util,json
from pathlib import Path
import pytest
REPO=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('recipe_dispatch',REPO/'tools/dispatch_recipe_factorial.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def simulate(tmp_path,monkeypatch,scenario,retry_oom=False):
    tick={'v':0};launched=[];current={}
    r=dict(gpu=dict(name='MOCK',total_mib=16384.,used_mib=4096.,free_mib=12288.,utilization=20.),
           cpu_cores=24,ram={'MemAvailable':64*1024**3},loadavg=[0.,0.,0.])
    def snap(out,reason,active=()):
        if reason=='dispatch_tick':
            tick['v']+=1
            assert tick['v']<20,'Unexpected dispatch/retry loop'
        return r
    class Process:
        def __init__(self,cmd,**kw):
            self.name,self.attempt=cmd;self.pid=100+len(launched);self.start=tick['v'];self.result=None
            launched.append(tuple(cmd));current[self.name]=self
        def poll(self):
            if self.result is not None:return self.result
            status=None
            if self.attempt!='initial':
                if tick['v']>self.start:status='OOM' if retry_oom else 'COMPLETE'
            elif scenario=='serial' and tick['v']>=2:status='OOM'
            elif scenario=='solo_after_success':
                if self.name=='A' and tick['v']>=3:status='COMPLETE'
                if self.name=='B' and tick['v']>=6:status='OOM'
            elif scenario=='dual_oom' and tick['v']>=3:status='OOM'
            elif scenario=='later_oom':
                if self.name=='A' and tick['v']>=3:status='OOM'
                if self.name=='B' and tick['v']>=4:status='OOM'
            elif scenario=='code_error' and tick['v']>=3:status='FAILED' if self.name=='A' else 'COMPLETE'
            if status is None:return None
            folder=tmp_path/'sampling'/self.name;folder.mkdir(parents=True,exist_ok=True)
            m.atomic_json(folder/('completed.json' if status=='COMPLETE' else self.attempt+'_failure.json'),dict(status=status,attempt=self.attempt,error=status))
            self.result=0 if status=='COMPLETE' else 1;return self.result
    monkeypatch.setattr(m,'snapshot',snap);monkeypatch.setattr(m.subprocess,'Popen',Process)
    monkeypatch.setattr(m,'command',lambda job,attempt:[job['name'],attempt])
    monkeypatch.setattr(m,'progress',lambda job:dict(completed=1,attempt=current[job['name']].attempt))
    monkeypatch.setattr(m.time,'sleep',lambda n:None)
    names=['A'] if scenario=='serial' else ['A','B']
    if scenario=='code_error':names.append('C')
    jobs=[dict(name=n,phase='sampling',status_root=str(tmp_path/'sampling'/n),completed_path=str(tmp_path/'sampling'/n/'completed.json')) for n in names]
    result=m.dispatch(tmp_path,dict(wave='mock',jobs=jobs))
    state=json.loads((tmp_path/'audit/dispatch_state.json').read_text())
    return result,state,launched

def test_successful_peer_does_not_license_later_solo_oom(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'solo_after_success')
    assert launched==[('A','initial'),('B','initial')]
    assert state['failed']['mock:B']['status']=='BLOCKED' and state['oom_downgrades']==0
    assert result['status']=='PARTIAL_BLOCKED'

def test_same_poll_dual_oom_one_serial_recovery_each(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'dual_oom')
    assert sorted(launched)==sorted([('A','initial'),('B','initial'),('A','oom_serial_retry'),('B','oom_serial_retry')])
    assert state['max_workers']==1 and state['oom_downgrades']==1 and result['status']=='COMPLETE'

def test_failed_serial_recovery_stops_new_dispatch(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'dual_oom',True)
    assert len(launched)==3 and result['status']=='PARTIAL_BLOCKED'
    assert all(n<=2 for n in state['attempts'].values())

def test_initial_serial_oom_never_retried(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'serial')
    assert launched==[('A','initial')] and state['oom_downgrades']==0

def test_survivor_later_oom_blocks_queued_recovery(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'later_oom')
    assert launched==[('A','initial'),('B','initial')]
    assert state['failed']['mock:B']['status']=='BLOCKED' and state['oom_downgrades']==1

def test_code_failure_preserves_running_peer_and_stops_pending(tmp_path,monkeypatch):
    result,state,launched=simulate(tmp_path,monkeypatch,'code_error')
    assert ('C','initial') not in launched and 'B' in result['completed']
    assert result['status']=='PARTIAL_BLOCKED'

def test_admission_respects_memory_cpu_and_gpu_compute():
    r=dict(gpu=dict(total_mib=16384,free_mib=10000,utilization=20),ram={'MemAvailable':32*1024**3},cpu_cores=24,loadavg=[0,0,0])
    assert m.slots(r,5000)==1
    r['gpu']['free_mib']=7000
    assert m.slots(r,5000)==0
    r['gpu'].update(free_mib=10000,utilization=95)
    assert m.slots(r,5000)==0

def test_completed_update_replay_cap_input(tmp_path):
    latest=tmp_path/'latest.json';latest.write_text('{"global_step":100}')
    job=dict(phase='training',latest_receipt=str(latest))
    assert m.replay_count(job,[{'step':199}])==99
    assert m.replay_count(job,[{'step':200}])==100
def test_initial_cannot_resume_existing_checkpoint(tmp_path):
    p=tmp_path/'latest.pt';p.write_bytes(b'CPU fixture, not weights')
    job=dict(command=['python','-B','worker.py'],latest_checkpoint=str(p))
    with pytest.raises(ValueError,match='silently resume'):
        m.command(job,'initial')
    assert m.command(job,'oom_serial_retry')[-2:]==['--resume',str(p)]

def test_unclosed_owner_registry_blocks_new_wave(tmp_path):
    p=tmp_path/'audit';p.mkdir()
    (p/'active_workers.json').write_text('{"workers":{"old":{"pid":99999999}}}')
    with pytest.raises(RuntimeError,match='Unclosed previous owner'):
        m.check_prior_owners(tmp_path)