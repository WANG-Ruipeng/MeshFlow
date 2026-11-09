"""Run the frozen CPU backend only after each registered single D0 is sealed."""
from __future__ import annotations
import argparse
import os
import time
import json

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from . import prepare as P, evaluate as E
    from .run import full_resources, write
    import psutil
    ap=argparse.ArgumentParser();ap.add_argument('--partition',type=int,choices=(0,1),required=True)
    a=ap.parse_args()
    if psutil.virtual_memory().available<4*(1024**3):raise RuntimeError('RAM headroom insufficient')
    folder=P.OUT/'workflow'/('ready_post_'+str(a.partition))
    if folder.exists():raise RuntimeError('Existing attempt; no automatic retry')
    folder.mkdir(parents=True)
    tasks=[t for i,t in enumerate(P.selected_tasks()) if i%2==a.partition]
    write(folder/'dispatch.json',dict(resources=full_resources(),task_ids=[t['cell_id'] for t in tasks],
        CPU_workers_in_process=1,threads=1,model_calls=0,evaluator_sha256=P.sha(E.__file__)))
    pending=list(tasks);start=time.perf_counter()
    while pending:
        progressed=False
        for task in list(pending):
            if not (P.OUT/'post'/task['cell_id']/'receipt.json').exists():continue
            E.post(task)
            pending.remove(task);progressed=True
            print(json.dumps(dict(cell_id=task['cell_id'],status='COMPLETE',remaining=len(pending))),flush=True)
        if not progressed:
            if time.perf_counter()-start>21600:raise RuntimeError('Waiting deadline reached; no model dispatched')
            time.sleep(2)
    write(folder/'complete.json',dict(status='COMPLETE',outputs=len(tasks),model_calls=0,
        automatic_retries=0,seconds=time.perf_counter()-start,resources=full_resources()))

if __name__=='__main__':main()
