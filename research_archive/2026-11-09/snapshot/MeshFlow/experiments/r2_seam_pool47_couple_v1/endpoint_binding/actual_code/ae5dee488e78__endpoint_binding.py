"""Prepared final-endpoint CPU binding: exactly 2 x 28 x 1.5%L cells."""
from __future__ import annotations
import argparse
import concurrent.futures
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import time
import types
from r2.pilots.seam_pool47_couple_v1 import endpoint_scoring as S
from r2.pilots.owner_layout_v1 import binding_adapter as A

C,E,Q,PE,B=A.C,A.E,A.Q,A.PE,A.B
ROOT=S.ROOT;OUT=ROOT/'endpoint_binding';ARMS=S.ARMS;DELTAS=(.015,);CELLS=56
VERSION='seam_pool47_couple_endpoint_actual_cluster_015_v2'

def private_core():
    if C.sha(B.__file__)!=A.LEGACY_SHA256:raise ValueError('Legacy source changed')
    ns=dict(vars(B));ns.update(BASE=ROOT,OUT=OUT,ARMS=ARMS,DELTAS=DELTAS,VERSION=VERSION)
    for name,obj in vars(B).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(B):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            fn.__kwdefaults__=obj.__kwdefaults__;ns[name]=fn
    ns['state_merge']=A.state_merge
    return ns

CORE=private_core()

def freeze():
    p=OUT/'registration.json'
    if p.exists():
        r=C.read(p);CORE['verify'](r['protected']);CORE['verify'](r['code']);return r
    ss=C.read(ROOT/'evaluation/summary_completion.json')
    assert ss['rows']==112 and ss['numeric_field_count']==124
    requests=S.requests();protected={str(ROOT/'generation_registration.json'):C.sha(ROOT/'generation_registration.json'),
        str(ROOT/'evaluation/score_completion.json'):C.sha(ROOT/'evaluation/score_completion.json'),
        str(ROOT/'evaluation/summary_completion.json'):C.sha(ROOT/'evaluation/summary_completion.json'),
        str(ROOT/'evaluation/per_output.csv'):C.sha(ROOT/'evaluation/per_output.csv')}
    for q in requests:
        for k in ('inference_npz','metadata_npz'):protected[q[k]]=C.sha(q[k])
        d=q['query_descriptor'];assert C.sha(d['path'])==d['sha256'];protected[d['path']]=d['sha256']
        for arm in ARMS:
            folder=ROOT/'evaluation'/arm/q['request_id']
            for rel in ('post/post.npz','post/pre_geo.npz','post/receipt.json'):
                pp=folder/rel;protected[str(pp)]=C.sha(pp)
            A.validate_arrays(C.arrays(folder/'post/post.npz'),C.arrays(folder/'post/pre_geo.npz')['output'],q)
            for name in ('score.json','pairs.json'):
                pp=ROOT/'evaluation/scores'/arm/q['request_id']/'POST'/name;protected[str(pp)]=C.sha(pp)
    code={str(Path(__file__)):C.sha(__file__),str(Path(A.__file__)):C.sha(A.__file__),str(Path(B.__file__)):C.sha(B.__file__),**E.source_hashes(),**Q.definition_hashes()}
    for name in ('geo_c','winding_solver'):
        pp=Path(PE.cpu_module(name).__file__);code[str(pp)]=C.sha(pp)
    r=dict(status='FROZEN',utc=C.utc(),version=VERSION,arms=list(ARMS),deltas=list(DELTAS),cells=CELLS,
        requests=28,requests_manifest=requests,L=C.L,protected=protected,code=code,
        algorithm='Same actual-cluster TIP+LATERAL queue, corrected current-root guard, C fixed and every member PRE_GEO displacement <=1.5%L',
        comparison='New POST to own bound POST; paired B_LONG_LAYOUT original and same-1.5%-bound baseline reused separately',
        no_expansion='No endpoint 2% branch, no head/source-layout inference, no model call or noise draw',
        model_calls=0,GPU_dispatches=0,new_queries=0)
    C.write(p,r)
    for path,h in code.items():
        dest=OUT/'actual_code'/(h[:12]+'__'+Path(path).name);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(C.resolve(path).read_bytes())
    CORE['event']('REGISTERED',cells=CELLS,registration_sha256=C.sha(p));return r

def worker(job):
    arm,q=job
    if arm not in ARMS:raise ValueError('Unknown arm')
    return CORE['worker'](job)

def run(workers=2):
    reg=freeze();p=OUT/'completion.json'
    if p.exists():
        done=C.read(p);CORE['verify'](done['files']);return done
    C.resources(OUT/('resources_before_'+str(time.time_ns())+'.json'))
    jobs=[(arm,q) for q in reg['requests_manifest'] for arm in ARMS]
    tick=time.perf_counter();first=worker(jobs[0]);snap=C.resources(OUT/('resources_measured_'+str(time.time_ns())+'.json'))
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    peak=max(x['peak_RSS_KiB'] for x in first)*1024
    n=2 if workers==2 and (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*peak) and (snap['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(OUT/('dispatch_'+str(time.time_ns())+'.json'),dict(workers=n,threads_per_worker=1,first_actual_peak_RSS_bytes=peak,resources=snap))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(n):
            job=next(todo,None)
            if job:active[pool.submit(worker,job)]=job
        while active:
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                f.result();active.pop(f);job=next(todo,None)
                if job:active[pool.submit(worker,job)]=job
    result=CORE['aggregate']();CORE['verify'](reg['protected']);CORE['verify'](reg['code'])
    assert len(result['rows'])==CELLS
    done=dict(status='COMPLETE',cells=CELLS,model_calls=0,GPU_calls=0,new_queries=0,numeric_metric_count=result['numeric_metric_count'],
        seconds=time.perf_counter()-tick,workers=n,receipts=result['receipts'],protected_verified=len(reg['protected']),
        files={str(x):C.sha(x) for x in OUT.glob('*.csv')},visual_review='PENDING_ACTUAL_VIEW',source_hash=C.sha(__file__))
    C.resources(OUT/('resources_finished_'+str(time.time_ns())+'.json'));C.write(p,done);CORE['event']('CPU_COMPLETE',cells=CELLS,seconds=done['seconds'])
    return done

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('status','freeze','run'),default='status',nargs='?');p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.stage=='freeze':r=freeze();r=dict(status=r['status'],cells=r['cells'])
    elif a.stage=='run':r=run(a.workers);r={k:v for k,v in r.items() if k not in ('receipts','files')}
    else:r=dict(status='PREPARED_NO_AUTO_DISPATCH',cells=56,deltas=[.015],completed=len(list((OUT/'cells').glob('*/*/complete.json'))))
    print(json.dumps(C.clean(r)),flush=True)

if __name__=='__main__':main()
