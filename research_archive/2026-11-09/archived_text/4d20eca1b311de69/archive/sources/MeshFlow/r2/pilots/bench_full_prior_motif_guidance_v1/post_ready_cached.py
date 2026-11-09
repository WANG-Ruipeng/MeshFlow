"""CPU scheduling revision: full RAW audit once, then guarded immutable reuse.

The original evaluator, scorers and geometry functions are not edited. Each
task still validates its registered RAW/D0 identity and complete receipts.
"""
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
    folder=P.OUT/'workflow'/('ready_post_cached_'+str(a.partition))
    if folder.exists():raise RuntimeError('Existing attempt; no automatic retry')
    folder.mkdir(parents=True)
    tasks=[t for i,t in enumerate(P.selected_tasks()) if i%2==a.partition]
    write(folder/'dispatch.json',dict(resources=full_resources(),task_ids=[t['cell_id'] for t in tasks],
        CPU_workers_in_process=1,threads=1,model_calls=0,evaluator_sha256=P.sha(E.__file__),
        orchestration_sha256=P.sha(__file__),scientific_functions_modified=False))
    original=P.verify_raw_barrier
    saved=original()
    protected=saved[1]['files']
    stats={p:(P.resolve(p).stat().st_size,P.resolve(p).stat().st_mtime_ns) for p in protected}
    write(folder/'initial_full_RAW_verification.json',saved[1])
    def guarded_barrier():
        for path,stat in stats.items():
            current=P.resolve(path).stat()
            if (current.st_size,current.st_mtime_ns)!=stat:
                raise RuntimeError('Sealed RAW/barrier identity changed; no cached continuation: '+path)
        return saved
    P.verify_raw_barrier=guarded_barrier
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
    P.verify_raw_barrier=original
    final=original()
    if final!=saved:raise RuntimeError('Final full RAW barrier changed')
    write(folder/'complete.json',dict(status='COMPLETE',outputs=len(tasks),model_calls=0,
        automatic_retries=0,seconds=time.perf_counter()-start,resources=full_resources(),
        initial_and_final_full_hash_verification=True,per_task_raw_receipt_stat_guard=True,
        original_per_task_registered_D0_hash_verification=True,scientific_functions_modified=False))

if __name__=='__main__':main()
