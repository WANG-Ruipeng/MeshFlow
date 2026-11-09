"""Bounded CPU scoring/rendering dispatch after the frozen RAW barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
import argparse, concurrent.futures, multiprocessing, time
from .common import *

def score(cell):
    from .evaluate import score_one
    original=cell['original_q'] if cell['q']['K']!=cell['original_q']['K'] else None
    r=score_one(cell['q'],cell['raw_path'],OUT/'scores'/cell['cell_id'],cell['label'],original_q=original)
    return dict(cell=cell['cell_id'],seconds=r['seconds'],peak_RSS_KiB=r['peak_RSS_KiB'])

def render(row):
    from .evaluate import render_page
    q,cells=row
    items=[dict(label=c['label'],path=c['raw_path'],q=c['q']) for c in cells]
    r=render_page(q,items,OUT/'gallery'/q['request_id'])
    return dict(request_id=q['request_id'],image=r['image_path'],seconds=r['seconds'])

def run(action,indices,workers):
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    reg=read(OUT/'registration.json');verify(reg['inputs'])
    cells=read(OUT/'inputs/cells.json');jobs=read(OUT/'inputs/jobs.json')
    for j in jobs:
        r=read(resolve(j['raw_path']).parent/'complete.json');verify(r['files'])
    if action=='score':
        rows=[c for c in cells if c['origin']=='NEW']
        if indices is not None: rows=[rows[i] for i in indices]
        function=score
    else:
        groups={}
        for c in cells:groups.setdefault(c['q']['request_id'],[]).append(c)
        rows=[(cs[0]['original_q'],cs) for cs in groups.values()];function=render
    stamp=str(time.time_ns());stage=OUT/'cpu_dispatch'/stamp
    write(stage/'before.json',dict(resources=snapshot(),action=action,workers=workers,threads_per_worker=1,
        tasks=len(rows),budget='registered ten new RAW quality rows + eight common-C CF-only scans; six comparison pages',
        decision='CPU geometry/render tasks independent; measured host24 logical cores and >60GiB available; max2 workers initially'))
    tick=time.perf_counter();results=[]
    if workers==1:
        for r in rows:
            value=function(r);results.append(value);print(json.dumps(value),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(function,r) for r in rows]
            for future in concurrent.futures.as_completed(futures):
                value=future.result();results.append(value);print(json.dumps(value),flush=True)
    write(stage/'complete.json',dict(status='COMPLETE',action=action,seconds=time.perf_counter()-tick,
        workers=workers,results=results,resources_after=snapshot()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','render']);p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(a.action,a.indices,a.workers)
if __name__=='__main__':main()
