"""CPU only scoring/render dispatch with a complete-RAW barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,concurrent.futures,multiprocessing,time,json
from .common import *

def score(cell):
    from .evaluate import score_one
    r=score_one(cell['q'],cell['raw_path'],OUT/'scores'/cell['cell_id'],cell['label'],original_q=cell['original_q'])
    return dict(cell=cell['cell_id'],seconds=r['seconds'],peak_RSS_KiB=r['peak_RSS_KiB'])

def render(row):
    from .evaluate import render_page
    q,cs=row
    r=render_page(q,[dict(label=c['label'],path=c['raw_path'],q=c['q']) for c in cs],OUT/'gallery'/q['request_id'])
    return dict(request_id=q['request_id'],image=r['image_path'],seconds=r['seconds'])

def main():
    from .evaluate import require_all_raw
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    p=argparse.ArgumentParser();p.add_argument('action',choices=('score','render'));p.add_argument('--workers',type=int,default=2);p.add_argument('--indices',type=int,nargs='+');a=p.parse_args()
    if a.workers not in (1,2):raise ValueError('At most two measured CPU workers per phase')
    cells=require_all_raw();reg=read(OUT/'registration.json');verify(reg['inputs'])
    if a.action=='score':rows=[c for c in cells if c['origin']=='NEW'];fn=score
    else:
        groups={}
        for c in cells:groups.setdefault(c['q']['request_id'],[]).append(c)
        rows=[(cs[0]['original_q'],cs) for cs in groups.values() if len(cs)>1];fn=render
    if a.indices is not None:rows=[rows[i] for i in a.indices]
    d=OUT/'cpu_dispatch'/str(time.time_ns());write(d/'before.json',dict(action=a.action,workers=a.workers,
        threads_per_worker=1,tasks=len(rows),resources=snapshot(),
        reason='Prior same-generator same-N original scorer measured <=115MiB/process; host24 CPUs and >60GiB free; bound to2 per phase'))
    t=time.perf_counter();results=[]
    if a.workers==1:
        for row in rows:results.append(fn(row));print(json.dumps(results[-1]),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(fn,row) for row in rows]
            for f in concurrent.futures.as_completed(futures):results.append(f.result());print(json.dumps(results[-1]),flush=True)
    write(d/'complete.json',dict(status='COMPLETE',action=a.action,workers=a.workers,seconds=time.perf_counter()-t,results=results,resources_after=snapshot()))

if __name__=='__main__':main()
