"""Budgeted matrix dispatcher: one GPU subprocess, independent CPU scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();out=args.out.resolve()
    prefix='experiments.mf_h1_local0.jedge5_numeric_reference_A.'
    def run(module,tail,log):
        start=time.perf_counter();log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            result=subprocess.run([sys.executable,'-B','-m',prefix+module,'--out',str(out)]+tail,stdout=stream,stderr=subprocess.STDOUT)
        return dict(returncode=result.returncode,seconds=time.perf_counter()-start,log=str(log))
    records=[]
    for model in ('B','J'):
        path=out/'prechecks'/f'{model}_precheck'/'result.json'
        if not path.exists():
            record=run('worker',['--model',model,'--precheck'],out/'logs'/f'{model}_precheck.log');records.append(record)
        if not path.exists() or json.loads(path.read_text())['status']!='PASS':raise RuntimeError('GPU precheck failed '+model)
    init=run('scoring',['--initialize'],out/'logs'/'score_initialize.log')
    if init['returncode']:raise RuntimeError('Scorer initialization failed')
    executor=ThreadPoolExecutor(max_workers=2);futures=[]
    def score(path,ident):return run('scoring',['--trajectory',str(path)],out/'logs'/('score_'+ident+'.log'))
    # Frozen ordering is independent of scores and worker completion times.
    cases=('P0_patch0_seed10401','P1_patch0_seed10401','P0_patch4_seed10401','P0_patch4_seed10402')
    for solver in ('R1','R2','D1','D2'):
        for case_index,case_id in enumerate(cases):
            for model in ('B','J'):
                ident=f'{model}_{case_id}_{solver}';path=out/'trajectories'/ident/'result.json'
                if path.exists():raise FileExistsError('No silent resume/retry '+str(path))
                record=run('worker',['--model',model,'--case',str(case_index),'--solver',solver],out/'logs'/(ident+'.log'))
                records.append(dict(id=ident,**record))
                if not path.exists():raise RuntimeError('Missing worker record '+ident)
                payload=json.loads(path.read_text())
                print(json.dumps(dict(id=ident,status=payload['status'],nfe=payload.get('nfe'),t=payload.get('last_accepted_time'))),flush=True)
                (out/'dispatch_progress.json').write_text(json.dumps(records,indent=2))
                if payload['status']=='WORKER_FAILURE' or record['returncode']:raise RuntimeError('Worker error; no retry '+ident)
                if payload['complete']:futures.append(executor.submit(score,path,ident))
    # Exactly one preregistered independent process repeat, not selected by quality.
    ident='B_P0_patch0_seed10401_R1_repeat'
    record=run('worker',['--model','B','--case','0','--solver','R1','--repeat'],out/'logs'/(ident+'.log'))
    records.append(dict(id=ident,**record))
    if record['returncode']:raise RuntimeError('Repeat worker failed')
    scores=[f.result() for f in futures];executor.shutdown()
    for row in scores:
        if row['returncode']:raise RuntimeError('CPU score failed '+row['log'])
    (out/'dispatch.json').write_text(json.dumps(dict(status='PASS',GPU_workers_this_window=1,CPU_scoring_workers=2,
        tasks=records,scoring=scores,timing_note='Worker wall includes GPU-lock wait and load. Concurrent CPU scoring wall is not isolated throughput.'),indent=2))
    print('MATRIX_FINISHED',flush=True)

if __name__=='__main__':main()
