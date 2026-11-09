"""CPU unchanged ABC scoring for B_LONG_LAYOUT and cached paired references."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import collections
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from . import layout as L
from r2.pilots.seam_abc_v1 import evaluate as E
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_quality_four_v1.summarize import flatten


def score_one(job):
    # Applicability is the only adapter: original scorer and all definitions stay unchanged.
    C.ARMS=(*C.ARMS,L.LABEL)
    return E.score_one(job)


def prepare():
    reg=L.G.read(L.OUT/'registration.json');out=L.OUT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    protected={}
    for q in reg['requests']:
        for k in ('inference_npz','metadata_npz','noise_npz'):protected[q[k]]=C.sha(q[k])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    value=dict(version='seam_followup_original_ABC_score_adapter_v1',status='FROZEN',
        source_hashes={**E.source_hashes(),str(Path(__file__)):C.sha(__file__)},protected=protected,
        registration_sha256=C.sha(L.OUT/'registration.json'),cells=56,requests=28,stages=['RAW','POST'],
        new_label=L.LABEL,cached_references=['LONG','B_POOL47'],new_model_calls=0,new_queries=0,
        scorer='Exact unchanged ABC evaluate.score_one; only new-label seam graph applicability enabled',
        quality_definition=E.helpers()[0].cpu_math()['control'].frozen_definition(),L=C.L)
    C.write(out/'score_registration.json',value);return value


def run():
    reg=L.G.read(L.OUT/'registration.json');sp=L.OUT/'evaluation/score_registration.json';sr=C.read(sp)
    E.verify(sr['source_hashes']);E.verify(sr['protected'])
    if sr['registration_sha256']!=C.sha(L.OUT/'registration.json'):raise ValueError('Registration changed')
    final=L.OUT/'evaluation/score_completion.json'
    if final.exists():
        x=C.read(final)
        for r in x['receipts']:E.verify({r['path']:r['sha256']});E.verify(C.read(r['path'])['files'])
        return x
    L.G.checked_receipt(L.OUT/'raw_complete.json');L.G.checked_receipt(L.OUT/'post_complete.json')
    jobs=[(str(L.OUT),L.LABEL,q,s,C.sha(sp)) for q in reg['requests'] for s in ('RAW','POST')]
    before=C.resources();C.write(L.OUT/'evaluation/resources_before_score.json',before)
    started=time.perf_counter();first=score_one(jobs[0]);after=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (after.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    workers=2 if (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (after['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(L.OUT/'evaluation/score_dispatch.json',dict(workers=workers,threads=1,first_authorized_cell=first,resources=after))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(workers):
            j=next(todo,None)
            if j:active[pool.submit(score_one,j)]=j
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            results=[f.result() for f in done]
            for f,r in zip(done,results):del active[f];receipts.append(r)
            print(json.dumps(dict(scored=len(receipts),registered=56)),flush=True)
            for _ in done:
                j=next(todo,None)
                if j:active[pool.submit(score_one,j)]=j
    receipts.sort(key=lambda r:r['path'])
    x=dict(status='COMPLETE',cells=56,receipts=receipts,workers=workers,seconds=time.perf_counter()-started,
        full_quality_scans=sum(not r['reused'] for r in receipts),new_queries=0,model_calls=0)
    C.write(final,x);return x


def summarize():
    reg=L.G.read(L.OUT/'registration.json');rows=[];roles=[];deleted=[];refs={}
    for label in ('LONG','B_POOL47',L.LABEL):
        root=L.OUT if label==L.LABEL else L.OLD
        for q in reg['requests']:
            for stage in ('RAW','POST'):
                p=root/'evaluation/scores'/label/q['request_id']/stage/'score.json'
                s=C.read(p);E.verify(s['files']);refs[str(p)]=C.sha(p)
                row=flatten(s,label);row['split']=q['split'];pair=C.read(p.parent/'pairs.json');mesh=C.arrays(s['identity']['mesh_path'])
                k=q['K'];ids=mesh['raw_slot_ids'];seam=mesh['raw_seam_slot_mask'][ids];counts=collections.Counter()
                for kind,scan in pair.items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in E.ILLEGAL:continue
                        i,j=map(int,contact['pair'])
                        if kind=='FF':
                            i+=k;j+=k;role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                        else:j+=k;role='C-seam' if seam[j] else 'C-bulk'
                        counts[role]+=1
                        roles.append(dict(label=label,split=q['split'],stage=stage,request_id=q['request_id'],kind=kind,role=role,
                            classification=contact['classification'],raw_face_i=int(ids[i]),raw_face_j=int(ids[j])))
                for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=counts[role]
                if sum(counts[r] for r in ('seam-seam','seam-bulk','bulk-bulk'))!=row['FF_illegal_pair_count']:raise ValueError('Partition mismatch')
                rows.append(row)
                if stage=='POST':
                    for fi in np.flatnonzero(~mesh['source_face_kept']):
                        deleted.append(dict(label=label,split=q['split'],request_id=q['request_id'],raw_slot_id=int(fi),
                            role='SEAM' if mesh['raw_seam_slot_mask'][fi] else 'BULK',reason=str(mesh['source_face_reason'][fi]),
                            duplicate_of_raw_slot=int(mesh['duplicate_of_source_face'][fi])))
    numeric=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in (*E.META,'label','stage') and any(isinstance(r.get(k),(float,int,bool)) for r in rows))
    if len(numeric)!=124:raise ValueError('Expected identical 124 numeric field definitions')
    summary=[]
    for label in ('LONG','B_POOL47',L.LABEL):
        for split in ('TRAIN','DEV','ALL'):
            for stage in ('RAW','POST'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                registered={'TRAIN':16,'DEV':12,'ALL':28}[split]
                for metric in numeric:
                    vals=[r[metric] for r in rr if isinstance(r.get(metric),(int,float,bool))]
                    summary.append(dict(label=label,split=split,stage=stage,metric=metric,n_registered=registered,n_valid=len(vals),
                        mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                        min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None,
                        sum=float(np.sum(vals)) if vals else None,status='PASS' if len(vals)==registered else 'NA_OR_PARTIAL'))
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for base in ('LONG','B_POOL47'):
        for q in reg['requests']:
            for stage in ('RAW','POST'):
                a,b=lookup[base,stage,q['request_id']],lookup[L.LABEL,stage,q['request_id']]
                for metric in numeric:
                    x,y=a.get(metric),b.get(metric);valid=isinstance(x,(float,int,bool)) and isinstance(y,(float,int,bool))
                    paired.append(dict(baseline=base,candidate=L.LABEL,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                        noise_index=q['noise_index'],stage=stage,metric=metric,baseline_value=x,candidate_value=y,
                        delta=float(y)-float(x) if valid else None,status='PASS' if valid else 'NA'))
    groups=collections.defaultdict(list);fields=('baseline','candidate','parent_id','split','stage','metric')
    for r in paired:groups[tuple(r[f] for f in fields)].append(r)
    parents=[]
    for key,rr in sorted(groups.items()):
        vv=[r for r in rr if r['status']=='PASS']
        parents.append(dict(zip(fields,key),n_registered=4,n_valid=len(vv),
            **{f:float(np.mean([r[f] for r in vv])) if vv else None for f in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(vv)==4 else 'NA_OR_PARTIAL'))
    for name,data in [('per_output',rows),('summary',summary),('paired_per_noise',paired),('paired_per_parent',parents),('illegal_pairs_by_role',roles),('deleted_faces',deleted)]:
        C.csvwrite(L.OUT/'evaluation'/(name+'.csv'),data)
    value=dict(status='COMPLETE_NUMERICAL_PENDING_ACTUAL_VIEW',numeric_fields=124,source_scores=refs,
        new_scans=56,reused_scores=112,model_calls=0,new_queries=0,
        files={str(p):C.sha(p) for p in (L.OUT/'evaluation').glob('*.csv')})
    C.write(L.OUT/'evaluation/summary_completion.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','score','summarize'));a=p.parse_args()
    r={'prepare':prepare,'score':run,'summarize':summarize}[a.stage]()
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_hashes','protected','receipts','files','source_scores')}),flush=True)

if __name__=='__main__':main()
