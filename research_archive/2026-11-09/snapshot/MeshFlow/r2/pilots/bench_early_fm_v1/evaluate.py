"""Reuse actual scorers and CPU finish in separate immutable experiment paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import sys,argparse,time,concurrent.futures,multiprocessing

def one(job):
    mode,q=job
    # Crossover's reduced manifest omits two reporting-only applicability fields.
    original=next(r for r in read(OLD/'evaluation/score_registrations/USER.json')['requests'] if r['request_id']==q['request_id'])
    q=dict(q,part_family=original['part_family'],closed_C=original['closed_C'])
    from r2.pilots.bench_user_v2_training_v1 import evaluation as E
    E.OUT=OUT
    r=E.raw_one('EARLY',q,sha(OUT/'evaluation/registration.json')) if mode=='raw' else E.post_one('EARLY',q,sha(OUT/'evaluation/registration.json'))
    assert 'torch' not in sys.modules
    print('SCORED',mode,q['request_id'],flush=True);return r

def run(mode,workers=3):
    ereg=read(OUT/'evaluation/registration.json');verify(ereg['source_hashes'])
    done=OUT/'scoring'/mode/'complete.json'
    if done.exists():verify(read(done)['files']);return
    qs=registration()['requests'];jobs=[(mode,q) for q in qs];tick=time.perf_counter()
    first=one(jobs[0]);snap=resources()
    peak=first.get('peak_RSS_KiB',200000)*1024
    workers=min(workers,max(1,int(snap['RAM_available_bytes']/(max(peak,1)*2))),max(1,snap['cpu_logical']//4))
    save(OUT/'scoring'/mode/'dispatch.json',dict(workers=workers,first_peak_RSS_KiB=peak/1024,resources=snap,numerical_threads=1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=[first,*pool.map(one,jobs[1:])]
    paths=[OUT/'evaluation/scores/EARLY'/q['request_id']/'RAW/audit_receipt.json' if mode=='raw' else OUT/'cpu/EARLY'/q['request_id']/'complete.json' for q in qs]
    verify(ereg['source_hashes']);save(done,dict(status='COMPLETE',outputs=len(results),workers=workers,seconds=time.perf_counter()-tick,model_calls=0,new_queries=0,files={str(p):sha(p) for p in paths}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','post']);p.add_argument('--workers',type=int,default=3);a=p.parse_args();run(a.mode,a.workers)
