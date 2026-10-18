"""Original 124-field CPU scoring for two fixed final generator endpoints."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import time
import types
from r2.pilots.seam_abc_v1 import evaluate as E, eval_common as C
from r2.pilots.seam_quality_four_v1.summarize import flatten
import numpy as np

ROOT=C.ROOT/'experiments/r2_seam_pool47_couple_v1'
ARMS=('PLAIN3000','COUPLE3000')
VERSION='seam_pool47_couple_original_124_field_endpoint_score_v1'


def private_score():
    common=types.SimpleNamespace(**vars(C));common.ARMS=ARMS
    ns=dict(vars(E));ns['C']=common
    for name,obj in vars(E).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(E):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__)
            fn.__kwdefaults__=obj.__kwdefaults__;ns[name]=fn
    return ns


API=private_score()


def requests():
    reg=C.read(ROOT/'generation_registration.json')
    if reg['arms']!=list(ARMS) or len(reg['requests'])!=28:raise ValueError('Unexpected endpoint panel')
    return reg['requests']


def prepare():
    p=ROOT/'evaluation/score_registration.json'
    if p.exists():
        r=C.read(p);E.verify(r['protected']);E.verify(r['source_hashes']);return r
    qq=requests();protected={str(ROOT/'generation_registration.json'):C.sha(ROOT/'generation_registration.json')}
    for q in qq:
        for k in ('inference_npz','metadata_npz','noise_npz'):protected[q[k]]=C.sha(q[k])
        desc=q['query_descriptor'];assert C.sha(desc['path'])==desc['sha256'];protected[desc['path']]=desc['sha256']
        for arm in ARMS:
            for stage in ('RAW','POST'):
                mesh,receipt=E.artifact(ROOT,arm,q['request_id'],stage)
                protected[str(mesh)]=C.sha(mesh);protected[str(receipt)]=C.sha(receipt)
    r=dict(status='FROZEN',version=VERSION,arms=list(ARMS),requests=qq,cells=112,stages=['RAW','POST'],
        protected=protected,source_hashes={**E.source_hashes(),str(Path(__file__)):C.sha(__file__)},
        scorer='Original evaluate.score_one in private namespace; only C.ARMS applicability recognizes the two new labels',
        quality_definition=E.helpers()[0].cpu_math()['control'].frozen_definition(),
        baseline='Saved B_LONG_LAYOUT scores reused by results.py, no baseline regeneration',
        model_calls=0,new_queries=0)
    C.write(p,r);return r


def score_one(job):
    return API['score_one'](job)


def run(workers=2):
    reg=prepare();sp=ROOT/'evaluation/score_registration.json';final=ROOT/'evaluation/score_completion.json'
    if final.exists():
        done=C.read(final)
        for rr in done['receipts']:E.verify({rr['path']:rr['sha256']});E.verify(C.read(rr['path'])['files'])
        return done
    C.resources(ROOT/'evaluation'/('resources_before_'+str(time.time_ns())+'.json'))
    jobs=[(str(ROOT),a,q,s,C.sha(sp)) for q in reg['requests'] for a in ARMS for s in ('RAW','POST')]
    tick=time.perf_counter();first=score_one(jobs[0]);snap=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    count=2 if workers==2 and (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snap['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(ROOT/'evaluation'/('dispatch_'+str(time.time_ns())+'.json'),dict(workers=count,threads_per_worker=1,first_actual_cell=first,resources=snap))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(count):
            job=next(todo,None)
            if job:active[pool.submit(score_one,job)]=job
        while active:
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            results=[f.result() for f in ready]
            for f,r in zip(ready,results):active.pop(f);receipts.append(r)
            print(json.dumps(dict(scored=len(receipts),registered=112)),flush=True)
            for _ in ready:
                job=next(todo,None)
                if job:active[pool.submit(score_one,job)]=job
    E.verify(reg['protected']);E.verify(reg['source_hashes']);receipts.sort(key=lambda x:x['path'])
    done=dict(status='COMPLETE',cells=112,receipts=receipts,workers=count,seconds=time.perf_counter()-tick,
        registration_sha256=C.sha(sp),full_quality_scans=sum(not x['reused'] for x in receipts),model_calls=0,new_queries=0)
    C.write(final,done);return done


def metric_row(score,mesh,pairs):
    row=flatten(score,score['label']);K=int(score['K']);ids=mesh['raw_slot_ids'];seam=mesh['raw_seam_slot_mask'][ids]
    counts=collections.Counter();roles=[]
    for kind,scan in pairs.items():
        for contact in scan['contacts']:
            if contact['classification'] not in E.ILLEGAL:continue
            i,j=map(int,contact['pair'])
            if kind=='FF':
                i+=K;j+=K;role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
            else:j+=K;role='C-seam' if seam[j] else 'C-bulk'
            counts[role]+=1
            roles.append(dict(label=score['label'],split=score['split'],stage=score['stage'],request_id=score['request_id'],
                kind=kind,role=role,classification=contact['classification'],raw_face_i=int(ids[i]),raw_face_j=int(ids[j])))
    for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=counts[role]
    assert sum(counts[r] for r in ('seam-seam','seam-bulk','bulk-bulk'))==row['FF_illegal_pair_count']
    deleted=[]
    if score['stage']=='POST':
        for i in np.flatnonzero(~mesh['source_face_kept']):
            deleted.append(dict(label=score['label'],split=score['split'],request_id=score['request_id'],raw_slot_id=int(i),
                role='SEAM' if mesh['raw_seam_slot_mask'][i] else 'BULK',reason=str(mesh['source_face_reason'][i]),
                duplicate_of_raw_slot=int(mesh['duplicate_of_source_face'][i])))
    return row,roles,deleted


def summarize():
    completion=C.read(ROOT/'evaluation/score_completion.json');assert completion['cells']==112
    rows=[];roles=[];deleted=[]
    for q in requests():
        for arm in ARMS:
            for stage in ('RAW','POST'):
                p=ROOT/'evaluation/scores'/arm/q['request_id']/stage/'score.json';s=C.read(p);E.verify(s['files'])
                mesh=C.arrays(s['identity']['mesh_path']);pairs=C.read(p.parent/'pairs.json')
                row,role,drop=metric_row(s,mesh,pairs);rows.append(row);roles.extend(role);deleted.extend(drop)
    numeric=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in (*E.META,'label','stage') and any(isinstance(r.get(k),(float,int,bool)) for r in rows))
    assert len(numeric)==124
    for name,rr in (('per_output',rows),('illegal_pairs_by_role',roles),('deleted_faces',deleted)):C.csvwrite(ROOT/'evaluation'/(name+'.csv'),rr)
    result=dict(status='COMPLETE',rows=len(rows),numeric_fields=numeric,numeric_field_count=124,
        source_completion_sha256=C.sha(ROOT/'evaluation/score_completion.json'),model_calls=0,new_scans=0,
        files={str(ROOT/'evaluation'/(x+'.csv')):C.sha(ROOT/'evaluation'/(x+'.csv')) for x in ('per_output','illegal_pairs_by_role','deleted_faces')})
    C.write(ROOT/'evaluation/summary_completion.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','score','summarize'));p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    r=prepare() if a.stage=='prepare' else run(a.workers) if a.stage=='score' else summarize()
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','source_hashes','receipts','requests','files','numeric_fields')}),flush=True)

if __name__=='__main__':main()
