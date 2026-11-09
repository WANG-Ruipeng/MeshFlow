"""Resumable bounded total entry. Completed stages verified; no hidden retry."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
import traceback
from .common import *


def child(out,stage,arm=None):
    out=Path(out);name=stage+('_'+arm if arm else '')
    status=out/'workers'/f'{name}.json'
    if status.exists() and read_json(status)['status']=='DONE':return read_json(status)
    command=[sys.executable,'-B','-u','-m',__package__+'.worker',stage,'--out',str(out)]
    if arm:command+=['--arm',arm]
    folder=out/'logs';folder.mkdir(parents=True,exist_ok=True)
    count=len(list(folder.glob(name+'_attempt*.log')))+1
    log=folder/f'{name}_attempt{count}.log'
    append_event(out/'ledger/controller.jsonl',dict(event='DISPATCH',stage=name,command=command,log=str(log)))
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
        print(dict(event='DISPATCH',stage=name,pid=process.pid,log=str(log)),flush=True)
        code=process.wait()
    append_event(out/'ledger/controller.jsonl',dict(event='RETURN',stage=name,returncode=code,log=str(log)))
    if code:raise RuntimeError(f'{name} stopped with code {code}; inspect {log}; no silent replay')
    state=read_json(status)
    if state['status']!='DONE':raise RuntimeError('Worker stopped safely before completion: '+name+' '+state['status'])
    return state


def run(out=OUT,through='all'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from . import protocol
    from .sampling import init_cache_rows
    started=time.perf_counter();status=out/'controller_status.json'
    atomic_json(status,dict(status='RUNNING',through=through,pid=os.getpid()))
    try:
        # CPU registration/tests have independent, immutable receipts. No rerun.
        for path in ('registration/summary.json','tests/method/results.json','metric_tests_v1r1/metric_test_results.json'):
            assert read_json(out/path)['status']=='PASS',path
        init_cache_rows(out);protocol.freeze_precheck(out)
        if through=='prepare':return dict(status='PREPARED')
        child(out,'precheck');child(out,'replay')
        p=protocol.freeze_training(out)
        if through=='precheck':return dict(status='PRECHECK_COMPLETE',arm_status=p['arm_status'])
        for arm in ARMS:
            if p['arm_status'][arm]=='PASS':child(out,'train',arm)
            else:atomic_json(out/'training'/f'{arm}_status.json',dict(status=p['arm_status'][arm],actual_updates=0))
        if through=='train':return dict(status='TRAIN_PHASE_COMPLETE')
        for arm in ARMS:
            pointer=out/'training'/f'{arm}_pointer.json'
            if pointer.exists() and read_json(pointer)['metadata']['attachment_step']==1000:child(out,'sample',arm)
        if through=='sample':return dict(status='SAMPLE_PHASE_COMPLETE')
        from . import evaluation
        for label in LABELS:evaluation.run(out,label=label)
        from . import aggregation
        aggregation.finalize(out)
        if through=='score':return dict(status='SCORE_PHASE_COMPLETE')
        from . import gallery,reporting
        gallery.run(out)
        result=reporting.run(out)
        atomic_json(status,dict(status='DONE',through=through,wall_seconds=time.perf_counter()-started,result=result))
        return result
    except BaseException as exc:
        atomic_json(status,dict(status='BLOCKED',through=through,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started))
        raise
    finally:
        from .ledger import Budget
        Budget(out,'controller_summary').snapshot()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT))
    parser.add_argument('--through',choices=['prepare','precheck','train','sample','score','all'],default='all')
    args=parser.parse_args();print(run(args.out,args.through),flush=True)
