"""Recoverable fixed schedule; does not add trials based on evaluation results."""
import argparse,os,subprocess,sys
from pathlib import Path
from .common import *


def worker(out,args,stage):
    path=out/'logs'/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-B','-u','-m','experiments.mf_h1_local0.chair_capped500_gradbudget_v1.worker','--out',str(out)]
    # argparse accepts mode first, optional args afterward.
    command=[sys.executable,'-B','-u','-m','experiments.mf_h1_local0.chair_capped500_gradbudget_v1.worker',*args,'--out',str(out)]
    append_event(out/'ledger/orchestrator.jsonl',dict(event='DISPATCH',stage=stage,command=command))
    print('DISPATCH',stage,flush=True)
    with path.open('a',encoding='utf-8') as handle:
        result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    append_event(out/'ledger/orchestrator.jsonl',dict(event='RETURN',stage=stage,returncode=result.returncode))
    if result.returncode:raise RuntimeError('Stage failed; preserve log and inspect before engineering recovery: '+stage)
    print('DONE',stage,flush=True)


def execute(out):
    protocol=read_json(out/'protocol.json')
    if not (out/'preflight/real_complete.json').exists():worker(out,['preflight'],'preflight')
    registrations=read_json(out/'registration/checkpoints.json')
    # Frozen historical new-noise comparison first, then all primary endpoints.
    for label in ('START','HIST_TEACHER500','HIST_CAPPED500'):
        record=registrations[label]
        worker(out,['sample','--label',label,'--panel','FRESH','--checkpoint',str(START) if label=='START' else record['path']],label+'_FRESH')
    for endpoint in (500,1000):
        for arm in ARMS:
            complete=out/'training'/f'{arm}_{endpoint}_complete.json'
            pointer=out/'training'/f'repeat_{arm}_pointer.json'
            if not complete.exists():
                current=read_json(pointer)['path'] if pointer.exists() else str(START)
                worker(out,['train','--arm',arm,'--until',str(endpoint),'--checkpoint',current],f'{arm}_train_{endpoint}')
            endpoint_path=out/'checkpoints'/f'repeat_{arm}_{endpoint}.pt'
            worker(out,['sample','--label',f'{arm}{endpoint}','--panel','BOTH','--checkpoint',str(endpoint_path)],f'{arm}{endpoint}_samples')
    atomic_json(out/'status/gpu_schedule_complete.json',dict(status='PASS',formal_updates=3000,normal_new_generations=240))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args();execute(Path(a.out))
