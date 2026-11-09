"""Isolated durable dispatcher, reusing the audited launcher in its own globals.

No parent module is patched. Job identities and output ownership are local.
There is no automatic scientific replay, retry, or stage dispatch.
"""
import argparse
import json
import types
from . import common as Q
from r2.pilots.bench_natural_n_recipe_v1 import workflow as previous

JOBS = {
    'train-wide': ('training', ('--arm','WIDE_HIGH')),
    'sample-b': ('evaluation', ('sample','--model','B')),
    'sample-w': ('evaluation', ('sample','--model','W')),
    'd0-b': ('evaluation', ('d0','--model','B')),
    'd0-w': ('evaluation', ('d0','--model','W')),
    'prepare-scores': ('evaluation', ('prepare-scores',)),
    'score-raw': ('evaluation', ('raw','--models','B','W','--workers','3')),
    'score-post': ('evaluation', ('post','--models','B','W','--workers','3')),
    'collect': ('evaluation', ('collect',)),
}
TRAIN_ARMS = {'train-wide':'WIDE_HIGH'}
namespace = dict(vars(previous), ROOT=Q.ROOT, OUT=Q.OUT, MODULE=Q.MODULE,
                 JOBS=JOBS, TRAIN_ARMS=TRAIN_ARMS, __file__=__file__,
                 VERSION='bench_high_joint_coverage_manual_workflow_v1')
for name,value in vars(previous).items():
    if isinstance(value,types.FunctionType) and value.__module__==previous.__name__:
        defaults=tuple(Q.OUT if x is previous.OUT else x for x in value.__defaults__) if value.__defaults__ else None
        clone=types.FunctionType(value.__code__,namespace,value.__name__,defaults,value.__closure__)
        clone.__kwdefaults__=value.__kwdefaults__
        namespace[name]=clone
for name in ('status','launch','execute','_run_worker','_preflight','_complete_check','job_status','pid_alive'):
    if name in namespace:globals()[name]=namespace[name]

def cpu_tests():
    """Real supervisor checks, with harmless CPU children and no model imports."""
    import os
    from pathlib import Path
    import sys
    import tempfile
    import psutil
    checks=[]
    assert _preflight.__defaults__==(Q.OUT,)
    assert namespace['OUT']==Q.OUT and namespace['MODULE']==Q.MODULE
    assert set(TRAIN_ARMS)=={'train-wide'}
    checks.append('independent output/module/default ownership; exactly one training arm')
    with tempfile.TemporaryDirectory(prefix='wide_workflow_cpu_') as tmp:
        base=Path(tmp)
        blank=base/'blank'
        assert job_status(blank)['status']=='NOT_STARTED'
        blank.mkdir()
        assert job_status(blank)['status']=='LAUNCH_INCOMPLETE_REQUIRES_INSPECTION'
        active=base/'active';active.mkdir()
        Q.write_new(active/'pid.json',dict(status='LAUNCHED',pid=1,process_created=2))
        assert job_status(active,lambda *_:False)['status']=='STALE_PROCESS_REQUIRES_INSPECTION'
        assert job_status(active,lambda *_:None)['supervisor_alive'] is None
        assert pid_alive(os.getpid(),psutil.Process().create_time()) is True
        assert pid_alive(os.getpid(),psutil.Process().create_time()-2) is False
        checks.append('attempt ownership and stale/reused process identity')
        ok=base/'ok';ok.mkdir()
        got=_run_worker(ok,'CPU_STUB',[sys.executable,'-B','-c','print("CPU_ONLY_STUB")'],lambda _:None)
        assert got['status']=='COMPLETE' and got['exit_code']==0
        try:_run_worker(ok,'CPU_STUB',[sys.executable,'-c','raise Exception()'],lambda _:None)
        except FileExistsError:pass
        else:raise AssertionError('duplicate execution')
        checks.append('successful CPU child completion; duplicate supervisor prohibited')
        bad=base/'bad';bad.mkdir()
        try:_run_worker(bad,'CPU_FAIL_STUB',[sys.executable,'-B','-c','raise SystemExit(7)'],lambda _:None)
        except RuntimeError:pass
        else:raise AssertionError('failure not propagated')
        assert Q.read(bad/'status.json')['exit_code']==7
        checks.append('failed CPU child recorded with no automatic retry')
        reg=base/'reg';reg.mkdir()
        Q.write_new(reg/'training_registration.json',dict(status='FROZEN'))
        assert _preflight('sample-w',reg)==Q.sha(reg/'training_registration.json')
        try:_preflight('train-wide',reg)
        except FileNotFoundError:pass
        else:raise AssertionError('training accepted without root GO')
        Q.write_new(reg/'dispatch/WIDE_HIGH_go.json',dict(authorized=True,arm='WIDE_HIGH',training_registration_sha256=Q.sha(reg/'training_registration.json')))
        assert _preflight('train-wide',reg)==Q.sha(reg/'training_registration.json')
        checks.append('frozen registration plus arm-specific root GO checked')
    assert 'torch' not in sys.modules
    result=dict(status='PASS',checks=checks,tests=len(checks),workflow_sha256=Q.sha(__file__),
                scientific_calls=0,model_imports=0,real_CPU_stub_processes=2,GPU_dispatches=0)
    Q.write_new(Q.OUT/'workflow_cpu_gate.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',choices=JOBS)
    p.add_argument('--execute',choices=JOBS)
    p.add_argument('--token')
    p.add_argument('--cpu-tests',action='store_true')
    a=p.parse_args()
    if a.execute:result=execute(a.execute,a.token)
    elif a.launch:result=launch(a.launch)
    elif a.cpu_tests:result=cpu_tests()
    else:result=status()
    print(json.dumps(result,ensure_ascii=False,indent=2))
