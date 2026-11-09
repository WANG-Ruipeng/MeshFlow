"""Recoverable controller, fixed stage order and a single GPU worker."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import *
from .registration import prepare,verify_identity,code_snapshot

MODULE='experiments.mf_h1_local0.chair_hybrid_coupling_v1'


def execute(out,key,args):
    statepath=out/'controller'/f'{key}.json'
    if statepath.exists():
        old=read_json(statepath)
        if old['status']=='DONE':return old
        if old['status']=='BLOCKED':raise RuntimeError('No automatic failed-stage retry: '+key)
    log=out/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='RUNNING',stage=key,started_unix=time.time(),command=[sys.executable,'-B','-m',*args],log=str(log))
    atomic_json(statepath,state)
    with log.open('a',buffering=1) as f:
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if key.startswith('score_'):environment['CUDA_VISIBLE_DEVICES']=''
        p=subprocess.Popen(state['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=environment)
        state['pid']=p.pid;atomic_json(statepath,state);rc=p.wait()
    state.update(status='DONE' if rc==0 else 'BLOCKED',returncode=rc,finished_unix=time.time())
    worker_state=out/'stages'/f'{key}.json'
    if rc==0 and worker_state.exists() and read_json(worker_state)['status']=='INTERRUPTED_SAFE':
        state['status']='INTERRUPTED_SAFE'
    atomic_json(statepath,state)
    if state['status']=='INTERRUPTED_SAFE':raise RuntimeError('Safely interrupted at a saved update boundary; resume explicitly')
    if rc:raise RuntimeError('Stage failed; no hidden retry: '+key+'; see '+str(log))
    return state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--through',choices=('prepare','precheck','start','train','all'),default='all')
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=prepare(out)
        if a.through=='prepare':return
        if not (out/'cpu_tests/summary.json').exists():
            execute(out,'cpu_tests',[MODULE+'.cpu_tests','--source',str(SOURCE),'--chair',str(CHAIR),'--out',str(out)])
        if not (out/'tests/scoring_cpu_tests.json').exists():
            execute(out,'scoring_cpu_tests',[MODULE+'.scoring','--out',str(out),'--cpu-tests'])
        from .diagnostics import cpu_checks
        freeze_json(out/'cpu_tests/diagnostics.json',cpu_checks())
        from .diagnostic_reporting import cpu_checks as reporting_checks
        freeze_json(out/'cpu_tests/diagnostic_reporting.json',reporting_checks())
        code_snapshot(out,freeze=True)
        execute(out,'precheck',[MODULE+'.worker','--out',str(out),'--stage','precheck'])
        if a.through=='precheck':return
        execute(out,'eval_START',[MODULE+'.worker','--out',str(out),'--stage','eval_START'])
        execute(out,'score_START',[MODULE+'.scoring','--out',str(out),'--label','START','--generated',str(out/'evaluation/START/generated_rows.json')])
        if a.through=='start':return
        for arm in ARMS:execute(out,'train_'+arm,[MODULE+'.worker','--out',str(out),'--stage','train_'+arm])
        if a.through=='train':return
        for arm in ARMS:
            execute(out,'eval_'+arm,[MODULE+'.worker','--out',str(out),'--stage','eval_'+arm])
            execute(out,'score_'+arm,[MODULE+'.scoring','--out',str(out),'--label',arm,'--generated',str(out/'evaluation'/arm/'generated_rows.json')])
        from .reporting import finalize
        endpoints=[read_json(out/'checkpoints'/f'{arm}_pilot1000_total5000.json') for arm in ARMS]
        for endpoint in endpoints:
            assert digest(endpoint['path'])==endpoint['checkpoint_sha256']
        atomic_json(out/'checkpoint_manifest.json',dict(source=read_json(out/'registration/start_identity.json'),
            endpoints=endpoints,identity_note='pilot1000/total5000 is this experiment lineage; source path is a reference, no copied large weight'))
        verification=verify_identity(out);atomic_json(out/'registration/source_verification_after.json',verification)
        from .ledger import Budget
        Budget(out,'finalize').snapshot()
        finalize(out)
        atomic_json(out/'RUN_COMPLETE.json',dict(status='DONE',protocol_sha256=digest(out/'protocol.json'),completed_unix=time.time(),
                                              implemented='DONE',trained='DONE',evaluated='DONE'))


if __name__=='__main__':main()
