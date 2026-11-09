"""Score only registered, sealed RAWs as they arrive; no scientific retry."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--partition',type=int,choices=(0,1),required=True)
    a=ap.parse_args()
    folder=P.OUT/'workflow'/('ready_score_'+str(a.partition))
    if folder.exists():raise RuntimeError('Existing attempt; no automatic retry')
    folder.mkdir(parents=True)
    tasks=P.load_tasks(resolved=True)
    tasks=[t for i,t in enumerate(tasks) if i%2==a.partition]
    write(folder/'dispatch.json',dict(resources=full_resources(),task_ids=[t['cell_id'] for t in tasks],
        CPU_workers_in_process=1,threads=1,model_calls=0,score_source_sha256=P.sha(E.__file__)))
    pending=list(tasks);start=time.perf_counter();new=0;cached=0
    while pending:
        progressed=False
        for task in list(pending):
            receipt=P.OUT/'raw_scores'/task['cell_id']/'audit_receipt.json'
            if not P.raw_receipt(task).exists():continue
            prior=receipt.exists()
            E.score_raw(task)
            cached+=int(prior);new+=int(not prior)
            pending.remove(task);progressed=True
            print(json.dumps(dict(cell_id=task['cell_id'],status='CACHED' if prior else 'SCORED',remaining=len(pending))),flush=True)
        if not progressed:
            if time.perf_counter()-start>21600:raise RuntimeError('Waiting deadline reached; no model dispatched')
            time.sleep(2)
    write(folder/'complete.json',dict(status='COMPLETE',new_RAW_scores=new,cached_RAW_scores=cached,
          model_calls=0,automatic_retries=0,seconds=time.perf_counter()-start,resources=full_resources()))

if __name__=='__main__':main()
