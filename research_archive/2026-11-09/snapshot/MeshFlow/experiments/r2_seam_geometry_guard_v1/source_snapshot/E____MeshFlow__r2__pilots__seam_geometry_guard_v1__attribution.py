"""Read cached RAW/POST scores; score fixed pre-GEO once, no model import."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, json, sys, time
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C, evaluate as E

OUT=C.ROOT/'experiments/r2_seam_geometry_guard_v1/attribution'
OLD=C.OUT
VERSION='seam_role_pair_transition_cpu_v1'

def illegal_pairs(scan, kind, raw_ids, K):
    rows=[]
    for p in scan['contacts']:
        if p['classification'] not in E.ILLEGAL: continue
        i,j=map(int,p['pair'])
        a=int(raw_ids[i if kind=='CF' else K+i]); b=int(raw_ids[K+j])
        if kind=='FF': a,b=sorted((a,b))
        rows.append((a,b,p['classification']))
    return rows

def role(mask, idx, K):return 'C' if idx<K else ('SEAM' if mask[idx] else 'BULK')

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    done=OUT/'completion.json'
    if done.exists():
        x=C.read(done); E.verify(x['protected']); E.verify(x['files']);return x
    m=C.read(OLD/'eval_manifest.json'); old,*_=E.helpers(); mod=old.cpu_math()
    protected={str(OLD/'eval_manifest.json'):C.sha(OLD/'eval_manifest.json')}
    for q in m['requests']:
        for arm in C.ARMS:
            for stage in ('RAW','POST'):
                for f in ('pairs.json','score.json'):
                    p=OLD/'evaluation/scores'/arm/q['request_id']/stage/f
                    protected[str(p)]=C.sha(p)
                p=OLD/'evaluation'/arm/q['request_id']/stage.lower()/(stage.lower()+'.npz')
                protected[str(p)]=C.sha(p)
        p=OLD/'evaluation/SEAM_SLOT_C200'/q['request_id']/'post/pre_geo.npz'
        protected[str(p)]=C.sha(p)
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    reg=dict(version=VERSION,manifest_sha256=C.sha(OLD/'eval_manifest.json'),protected=protected,
             original_scores=64,new_pre_geo_scores=16,model_calls=0,source_hashes=E.source_hashes(),
             identity='generated RAW slot IDs only; no cross-arm face correspondence',L=C.L)
    C.write(OUT/'registration.json',reg); E.verify(protected)
    tick=time.perf_counter(); summary=[]; role_rows=[]; detail=[]; transitions=[]; change_pairs=[]
    for arm in C.ARMS:
        for q in m['requests']:
            rid=q['request_id']; K=q['K']; base=OLD/'evaluation'/arm/rid
            raw=C.arrays(base/'raw/raw.npz'); mask=np.asarray(raw['raw_seam_slot_mask'],bool)
            stages=('RAW','PRE_GEO','POST') if arm=='SEAM_SLOT_C200' else ('RAW','POST')
            stage_sets={}; stage_ids={}
            for stage in stages:
                if stage=='PRE_GEO':
                    p=OUT/'pre_geo'/rid/'score.json'
                    if p.exists():
                        score=C.read(p); pairs=C.read(p.parent/'pairs.json')
                    else:
                        a=C.arrays(base/'post/pre_geo.npz'); tri=a['output']; cc=tri[:K]; F=tri[K:]
                        metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L)
                        trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
                        im,iv=mod['control'].conflict_metrics(F,dict(C=cc,L0=C.L),old.load_queries(q['query_descriptor']),trusted)
                        metrics.update(im);pairs=dict(CF=cf,FF=ff)
                        for kind,scan in pairs.items():
                            pp=[x for x in scan['contacts'] if x['classification'] in E.ILLEGAL]
                            affected={x['pair'][1] for x in pp} if kind=='CF' else {i for x in pp for i in x['pair']}
                            metrics[kind+'_illegal_pair_count']=len(pp)
                            metrics[kind+'_illegal_affected_free_fraction']=len(affected)/len(F)
                        score=dict(request_id=rid,label=arm,stage=stage,metrics=metrics,free_face_denominator=len(F),
                                   status='PASS',model_calls=0,mesh_sha256=C.sha(base/'post/pre_geo.npz'))
                        C.write(p.parent/'pairs.json',pairs);C.write(p,score)
                    ids=np.arange(q['N']);nfree=q['N']-K
                else:
                    sr=OLD/'evaluation/scores'/arm/rid/stage
                    score=C.read(sr/'score.json');pairs=C.read(sr/'pairs.json')
                    a=C.arrays(base/stage.lower()/(stage.lower()+'.npz'))
                    ids=a['raw_slot_ids'];nfree=len(ids)-K
                stage_ids[stage]=set(map(int,ids))
                met=score['metrics'];summary.append(dict(arm=arm,request_id=rid,parent_id=q['parent_id'],stage=stage,nfree=nfree,
                    CF=met['CF_illegal_affected_free_fraction'],FF=met['FF_illegal_affected_free_fraction'],
                    I=met['ConflictCov_C'],I_cross=met['ConflictCov_C_cross'],I_dup=met['ConflictCov_C_dup'],
                    CF_pairs=met['CF_illegal_pair_count'],FF_pairs=met['FF_illegal_pair_count']))
                stage_sets[stage]={}
                for kind in ('CF','FF'):
                    pp=illegal_pairs(pairs[kind],kind,ids,K);stage_sets[stage][kind]={(i,j):cl for i,j,cl in pp}
                    counts=collections.defaultdict(list)
                    for i,j,cl in pp:
                        rr=(role(mask,i,K),role(mask,j,K))
                        if kind=='FF':rr=tuple(sorted(rr))
                        category='-'.join(rr);counts[category].append((i,j))
                        detail.append(dict(arm=arm,request_id=rid,stage=stage,kind=kind,category=category,raw_slot_i=i,raw_slot_j=j,classification=cl))
                    categories=('C-SEAM','C-BULK') if kind=='CF' else ('SEAM-SEAM','BULK-SEAM','BULK-BULK')
                    for cat in categories:
                        rr=counts.get(cat,[]);aff={j for _,j in rr} if kind=='CF' else {i for pair in rr for i in pair}
                        role_rows.append(dict(arm=arm,request_id=rid,stage=stage,kind=kind,category=cat,pairs=len(rr),affected_free_faces=len(aff),free_denominator=nfree,affected_fraction=len(aff)/nfree,
                                              additive_fraction=False))
            for before,after in zip(stages,stages[1:]):
                for kind in ('CF','FF'):
                    b=stage_sets[before][kind];a=stage_sets[after][kind]
                    bs=set(b);aset=set(a)
                    gone=bs-aset; gained=aset-bs; both=bs&aset
                    deleted={p for p in gone if any(i not in stage_ids[after] for i in p)}
                    transitions.append(dict(arm=arm,request_id=rid,kind=kind,before=before,after=after,
                        retained_pairs=len(both),removed_with_deleted_face=len(deleted),resolved_without_deletion=len(gone-deleted),new_pairs=len(gained),
                        changed_illegal_type=sum(b[p]!=a[p] for p in both)))
                    for state,pp in (('NEW',gained),('DELETED_FACE',deleted),('RESOLVED_RETAINED_FACES',gone-deleted)):
                        for i,j in sorted(pp):
                            rr=(role(mask,i,K),role(mask,j,K));rr=tuple(sorted(rr)) if kind=='FF' else rr
                            change_pairs.append(dict(arm=arm,request_id=rid,kind=kind,before=before,after=after,change=state,raw_slot_i=i,raw_slot_j=j,category='-'.join(rr)))
            print(json.dumps(dict(arm=arm,request_id=rid,status='COMPLETE')),flush=True)
    C.csvwrite(OUT/'stage_metrics.csv',summary);C.csvwrite(OUT/'role_partition.csv',role_rows)
    C.csvwrite(OUT/'illegal_pairs_by_slot.csv',detail);C.csvwrite(OUT/'stage_pair_transitions.csv',transitions)
    C.csvwrite(OUT/'changed_pairs_by_slot.csv',change_pairs)
    agg=[]
    for arm,stage in sorted({(r['arm'],r['stage']) for r in summary}):
        rr=[r for r in summary if (r['arm'],r['stage'])==(arm,stage)]
        agg.append(dict(arm=arm,stage=stage,outputs=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('CF','FF','I','I_cross','I_dup')},**{k:sum(r[k] for r in rr) for k in ('CF_pairs','FF_pairs')}))
    C.write(OUT/'summary.json',dict(rows=agg,model_calls=0,new_pre_geo_scans=16,interpretation='Stage association, not force or causal occupancy attribution; D0 stage includes anchor restoration.'))
    E.verify(protected)
    if 'torch' in sys.modules:raise RuntimeError('Unexpected model framework import')
    files={str(p):C.sha(p) for p in OUT.rglob('*') if p.is_file()}
    result=dict(status='COMPLETE',version=VERSION,seconds=time.perf_counter()-tick,protected=protected,files=files,model_calls=0,new_pre_geo_scans=16)
    C.write(done,result);return result

if __name__=='__main__':
    r=run(); print(json.dumps({k:r[k] for k in ('status','seconds','model_calls','new_pre_geo_scans')}))
