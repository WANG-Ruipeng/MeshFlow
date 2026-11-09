"""Render exactly the registered final pages as sealed CPU outputs arrive."""
from __future__ import annotations
import os
import time
import concurrent.futures
import multiprocessing
import json

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from . import prepare as P, evaluate as E
    from .run import full_resources, write
    folder=P.OUT/'workflow/render_ready_final'
    if folder.exists():raise RuntimeError('Existing attempt; no implicit retry')
    folder.mkdir(parents=True)
    stage='CLEAN_FINAL_POST'
    tasks=P.selected_tasks()+P.read(P.OUT/'baseline.json')
    groups={}
    for t in tasks:groups.setdefault(t['request_id'],[]).append(t)
    pages=[]
    for rid,group in sorted(groups.items()):
        group=sorted(group,key=lambda t:E.ARMS.index(t['arm']))
        reference=next(t['q'] for t in group if t['arm']=='R')
        cells=[dict(display_label=t['arm'],cell_id=t['cell_id'],q=t['q'],path=str(E._spec(t,stage)['mesh'])) for t in group]
        page=dict(page_id=stage+'__'+rid,stage=stage,mode='final',reference_q=reference,cells=cells)
        pages.append(page)
        pages.extend(dict(page,mode='final_single',page_id=page['page_id']+'__'+c['display_label'],cells=[c]) for c in cells)
    write(folder/'dispatch.json',dict(pages=len(pages),CPU_workers=2,threads=1,resources=full_resources(),orchestration_sha256=P.sha(__file__),evaluator_sha256=P.sha(E.__file__)))
    pending=list(pages);active={};done=[];start=time.perf_counter()
    def ready(page):
        for c in page['cells']:
            if c['display_label']!='R' and not (P.OUT/'cpu'/c['cell_id']/'complete.json').exists():return False
        return True
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    future.result();done.append(active.pop(future))
            for page in list(pending):
                if len(active)>=2:break
                if ready(page):
                    active[pool.submit(E._render_page,page)]=page['page_id'];pending.remove(page)
            if time.perf_counter()-start>21600:raise RuntimeError('Rendering waiting deadline')
            if pending or active:time.sleep(1)
    write(folder/'complete.json',dict(status='COMPLETE',pages=len(done),seconds=time.perf_counter()-start,ratings_generated=0,model_calls=0,new_score_calls=0,resources=full_resources()))
    print(json.dumps(dict(status='COMPLETE',pages=len(done))),flush=True)

if __name__=='__main__':main()
