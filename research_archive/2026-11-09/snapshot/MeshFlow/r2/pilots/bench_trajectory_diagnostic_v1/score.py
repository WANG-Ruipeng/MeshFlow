import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import argparse,concurrent.futures,multiprocessing,numpy as np

def prepare():
    from r2.pilots.bench_parts_convergence_v1.evaluation import prepare as prep
    for label in LABELS:prep(OUT,label,True)

def one(job):
    label,q,mode=job
    from r2.pilots.bench_user_v2_training_v1 import evaluation as E
    E.OUT=OUT
    result=E.raw_one(label,q,sha(OUT/'registration.json')) if mode=='raw' else E.post_one(label,q,sha(OUT/'registration.json'))
    print('SCORED',mode,label,q['request_id'],flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','raw','post']);p.add_argument('--labels',nargs='+',default=list(LABELS));p.add_argument('--workers',type=int,default=2);p.add_argument('--available-only',action='store_true');a=p.parse_args()
    prepare()
    if a.mode=='prepare':return
    jobs=[]
    for label in a.labels:
        qs=read(OUT/'evaluation/score_registrations'/(label+'.json'))['requests'];jobs +=[(label,q,a.mode) for q in qs]
    if a.available_only:jobs=[job for job in jobs if (OUT/'evaluation'/job[0]/job[1]['request_id']/a.mode/'receipt.json').exists()]
    key=a.mode+'_'+'_'.join(a.labels)+('_available' if a.available_only else '');dest=OUT/'scoring'/key
    if (dest/'complete.json').exists():return
    snap=resources();first=one(jobs[0]);available=snap['available_RAM_bytes'];workers=min(a.workers,max(1,int(available/(max(first['peak_RSS_KiB'],1)*1024*2))))
    save(dest/'dispatch.json',dict(resources=snap,workers=workers,first_peak_RSS_KiB=first['peak_RSS_KiB'],threads_per_worker=1))
    tick=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:results=[first,*pool.map(one,jobs[1:])]
    save(dest/'complete.json',dict(status='COMPLETE',count=len(results),seconds=time.perf_counter()-tick,workers=workers,model_calls=0))
if __name__=='__main__':main()
