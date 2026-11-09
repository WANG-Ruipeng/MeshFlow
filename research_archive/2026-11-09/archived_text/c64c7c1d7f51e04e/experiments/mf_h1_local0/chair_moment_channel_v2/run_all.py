"""Recoverable finite V2 controller. Completed stages are verified and reused."""
import argparse
import fcntl
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from .common import *
from .registration import prepare,code_snapshot,verify_identity

MODULE='experiments.mf_h1_local0.chair_moment_channel_v2'

class SafeInterruption(RuntimeError):
    pass


def execute(out,key,args,resume=False):
    path=out/'controller'/f'{key}.json'
    if path.exists():
        previous=read_json(path)
        if previous['status']=='DONE':
            if key.startswith('train_'):
                from .worker import training_stage_status
                stage=read_json(out/'stages'/f'{key}.json')
                if stage['status']!='DONE' or training_stage_status(stage['result'])!='DONE':
                    raise RuntimeError('Controller DONE conflicts with incomplete worker training')
            return previous
        if key.startswith('train_') and resume and previous['status'] in ('INTERRUPTED_SAFE','RUNNING'):
            from .worker import verify_safe_resume
            proof=verify_safe_resume(out,key[len('train_'):])
            freeze_json(out/'controller'/'recovery_history'/f"{key}_from_{proof['moment_step']}.json",dict(previous=previous,proof=proof))
            args=[*args,'--resume']
        else:raise RuntimeError('Incomplete stage requires verified --resume or documented engineering recovery, not blind replay: '+key)
    log=out/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    state=dict(stage=key,status='RUNNING',started_unix=time.time(),command=[sys.executable,'-B','-m',*args],log=str(log))
    atomic_json(path,state);env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if key.startswith('score'):env['CUDA_VISIBLE_DEVICES']=''
    with log.open('a',buffering=1) as f:
        child=subprocess.Popen(state['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
        state['pid']=child.pid;atomic_json(path,state)
        def request_stop(signum,frame):
            state['stop_requested_signal']=int(signum)
            if child.poll() is None:child.send_signal(signum)
        handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:rc=child.wait()
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    safe=False
    if key.startswith('train_') and (out/'stages'/f'{key}.json').exists():
        stage_state=read_json(out/'stages'/f'{key}.json')
        safe=rc==75 and stage_state.get('status')=='INTERRUPTED_SAFE'
        if rc==0 and (stage_state.get('status')!='DONE' or stage_state.get('result',{}).get('moment_step')!=1000):
            raise RuntimeError('Successful process exit did not complete 1000 training updates')
    state.update(status='INTERRUPTED_SAFE' if safe else ('DONE' if rc==0 else 'BLOCKED'),returncode=rc,finished_unix=time.time())
    atomic_json(path,state)
    if safe:raise SafeInterruption('Training stopped at a verified checkpoint; resume with --resume: '+key)
    if rc:raise RuntimeError('Stage stopped; inspect '+str(log))
    return state

def cpu_gate(out):
    value=read_json(out/'tests/attention/results.json');assert value['status']=='PASS'
    for rel,h in read_json(out/'tests/attention/source_hashes.json').items():
        p=Path(rel);p=p if p.is_absolute() else (Path(__file__).parent/p if len(p.parts)==1 else ROOT/p)
        assert digest(p)==h,rel
    return value

def freeze_training(out):
    code_snapshot(out,'training_implementation_hashes')
    selection=dict(status='FROZEN',production='compact V1 exact-difference reference',
        purpose='padding invariance repair; retain existing low-dimensional value algebra',
        basis='CPU literal/gradient/equivariance, true zero identity, finite real numerical/cost gates; no quality selection',
        protocol_sha256=digest(out/'protocol.json'),precheck_sha256=digest(out/'precheck/result.json'),
        replay_sha256=digest(out/'initial_replay/result.json'),formal_training_started=False)
    freeze_json(out/'registration/training_selection.json',selection)

def finalize(out):
    from .ledger import Budget
    from .reporting import finalize as report
    endpoints=[]
    for arm in ARMS:
        p=out/'checkpoints'/f'{arm}_moment1000_total6000.json'
        if p.exists():
            r=read_json(p);assert digest(r['path'])==r['checkpoint_sha256'];endpoints.append(r)
    atomic_json(out/'checkpoint_manifest.json',dict(start=read_json(out/'registration/start_identity.json'),endpoints=endpoints))
    counts=Budget(out,'finalize').snapshot()
    cpu=read_json(out/'tests/attention/results.json')
    counts['isolated_CPU_tests']=cpu
    counts['summary']['isolated_module_test_updates']=cpu.get('isolated_module_optimizer_updates',1)
    atomic_json(out/'accounting.json',counts)
    verification=verify_identity(out,code=(out/'registration/training_implementation_hashes.json').exists())
    atomic_json(out/'registration/source_verification_after.json',verification)
    results=report(out)
    source=out/'source_code';source.mkdir(exist_ok=True)
    import shutil
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    # Small actual scoring dependencies accompany the local source, not just wrappers.
    for package,names in [('chair_hybrid_coupling_v1',('scoring.py','reporting.py','sampling.py')),('chair_surface_attention_v1',('scoring.py','reporting.py'))]:
        for name in names:
            p=ROOT/'experiments/mf_h1_local0'/package/name;target=source/package/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    delivered={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='delivery_hashes.json' and p.suffix not in ('.pt',)}
    atomic_json(out/'delivery_hashes.json',delivered)
    complete=counts['summary']['optimizer_updates']==2000 and all((out/'evaluation'/label/'scoring/summary.json').exists() for label in LABELS)
    atomic_json(out/('RUN_COMPLETE.json' if complete else 'RUN_STOPPED.json'),dict(status='DONE' if complete else 'BLOCKED',actual=counts['summary'],stopped_unix=time.time(),default_model_changed=False))
    print('V2_FIXED_TASK_'+('COMPLETE' if complete else 'STOPPED'),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--through',choices=('prepare','precheck','replay','all'),default='all');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(out)
        if a.through=='prepare':return
        cpu_gate(out)
        if not (out/'registration/precheck_implementation_hashes.json').exists():code_snapshot(out,'precheck_implementation_hashes')
        from .cache_start import run as cache
        cache(out)
        if not (out/'evaluation/START/scoring/summary.json').exists():execute(out,'score_START',[MODULE+'.scoring','--out',str(out),'--label','START','--generated',str(out/'evaluation/START/generated_rows.json')])
        execute(out,'precheck',[MODULE+'.worker','--out',str(out),'--stage','precheck'])
        real=read_json(out/'precheck/result.json')
        gate=dict(status=real['status'],formal_training_allowed=real['status']=='PASS',precheck=str(out/'precheck/result.json'),
            CPU=str(out/'tests/attention/results.json'),old_SURFACE_gate_applicable=False,
            protocol_sha256=digest(out/'protocol.json'))
        atomic_json(out/'gate_results.json',gate)
        if a.through=='precheck':return
        if gate['status']!='PASS':finalize(out);return
        execute(out,'initial_replay',[MODULE+'.worker','--out',str(out),'--stage','initial_replay'])
        if a.through=='replay':return
        if not (out/'registration/training_implementation_hashes.json').exists():freeze_training(out)
        for arm in ARMS:
            execute(out,'train_'+arm,[MODULE+'.worker','--out',str(out),'--stage','train','--arm',arm],resume=a.resume)
        for arm in ARMS:
            execute(out,'eval_'+arm,[MODULE+'.worker','--out',str(out),'--stage','eval','--arm',arm])
        # All raw exists before dependent quality scoring.
        for label in LABELS:
            if not (out/'evaluation'/label/'scoring/summary.json').exists():execute(out,'score_'+label,[MODULE+'.scoring','--out',str(out),'--label',label,'--generated',str(out/'evaluation'/label/'generated_rows.json')])
        finalize(out)

if __name__=='__main__':
    try:main()
    except SafeInterruption as exc:
        print(str(exc),flush=True)
        raise SystemExit(75)
