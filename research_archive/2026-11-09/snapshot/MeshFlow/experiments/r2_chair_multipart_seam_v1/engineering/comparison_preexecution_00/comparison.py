"""Read-only, identity-paired old/new metrics. No scorer or neural calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import collections
from pathlib import Path
import sys
import numpy as np
from . import evaluate as E
from .eval_prepare import OUT,LABEL
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
from r2.pilots.seam_hinge_quality_v1 import topology as T

C=E.C
DEST=OUT/'comparison'
OLD_ROOT=C.ROOT/'experiments/r2_unified_rescue_post_v1'
STAGES=('RAW','BASE_POST','GUARD_GEO','UNIFIED_FINAL')
KEYS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
      'free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive',
      'C_F_valid_edge_count','C_F_valid_edge_total_length','legal_seam_bulk_edge_count',
      'free_minimum_angle_lt5_fraction','hinge_diag_isolated_1face_edge_components',
      'hinge_diag_isolated_1face_unfiltered_edge_components','nonmanifold_edges',
      'actual_free_faces','deleted_free_faces','free_total_surface_area')

def number(x):return isinstance(x,(int,float,bool)) and np.isfinite(float(x))

def prepare():
    p=DEST/'registration.json'
    value=dict(version='multipart_fixed_paired_comparison_v1',
        analysis_code_sha256=C.sha(__file__),eval_manifest_sha256=C.sha(OUT/'eval_manifest.json'),
        frozen_score_registration_sha256=C.sha(OUT/'evaluation/score_registration.json'),
        stages=list(STAGES),key_metrics=list(KEYS),all_numeric_metrics_preserved=True,
        aggregation='noise -> condition -> parent equal; SEAT/BACK and original split separate',
        comparisons='Old S2000 versus final MULTIPART at identical processing stage; prior final shown separately',
        inference='Descriptive paired evidence only; no unique GT rank, significance claim, or checkpoint selection',
        original_N_queries_noise_preserved=True,neural_calls=0,scoring_calls=0)
    if p.exists():
        assert C.read(p)==value,'Analysis definition changed'
    else:C.write(p,value)
    return value

def old_raw(q):
    rid=q['request_id']
    if q['part_type']=='BACK_FULL':
        p=C.ROOT/'experiments/r2_chair_back_transfer_v1/evaluation/scores/S2000'/rid/'RAW/row.json'
        return C.read(p),p
    base=C.ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'
    sp=base/'evaluation/scores/PLAIN_S2000'/rid/'RAW'
    mesh=C.arrays(base/'evaluation/PLAIN_S2000'/rid/'raw/raw.npz')
    score=C.read(sp/'score.json');pairs=C.read(sp/'pairs.json')
    row=metric_row(score,mesh,pairs)[0]
    # Use saved detector classifications to express existing topology fields.
    # No new pair detection, sampling, or scorer call.
    _,_,graph,_,_=E.V.E.helpers()
    _,edges,_=graph.actual_valid_graph(mesh['vertices'],mesh['faces'],q['K'],pairs)
    row.update(T.measure(mesh,q['K'],edges,C.L)[0])
    return row,sp/'score.json'

def gather():
    prepare();qq=E.requests();oldreg=C.read(OLD_ROOT/'registration.json')
    oldjobs={(j['panel'],j['q']['request_id']):j for j in oldreg['jobs']}
    rows=[];sources={}
    def append(q,model,stage,row,path):
        row=dict(row);row.update(model=model,stage=stage,part_type=q['part_type'],
            split=q['split'],request_id=q['request_id'],parent_id=q['parent_id'],
            condition_id=q['condition_id'],noise_index=q['noise_index'],N=q['N'],K=q['K'])
        rows.append(row);sources[str(path)]=C.sha(path)
    for q in qq:
        panel='CHAIR_'+('SEAT' if q['part_type']=='SEAT_FULL' else 'BACK')+'_S2000'
        oldjob=oldjobs[panel,q['request_id']];oq=oldjob['q']
        for key in ('condition_id','parent_id','noise_index','N','K'):assert oq[key]==q[key],key
        for key in ('inference_npz','noise_npz','metadata_npz'):assert C.sha(oq[key])==C.sha(q[key]),key
        assert oq['query_descriptor']['sha256']==q['query_descriptor']['sha256']
        oldcell=OLD_ROOT/'cells'/panel/q['request_id']
        E.V.E.verify(C.read(oldcell/'complete.json')['files'])
        cached={r['stage']:r for r in C.read(oldcell/'metric_rows.json')}
        raw,path=old_raw(q);append(q,'S2000','RAW',raw,path)
        for source,stage in (('OLD_BASE','BASE_POST'),('OLD_FINAL','PRIOR_FINAL'),('GUARD_GEO','GUARD_GEO'),('FINAL','UNIFIED_FINAL')):
            append(q,'S2000',stage,cached[source],oldcell/'metric_rows.json')
        for stage in ('RAW','BASE_POST'):
            p=OUT/'evaluation/scores'/LABEL/q['request_id']/stage
            E.V.E.verify(C.read(p/'audit_receipt.json')['files'])
            append(q,LABEL,stage,C.read(p/'row.json'),p/'row.json')
        newcell=OUT/'unified/cells'/q['part_type']/q['request_id']
        E.V.E.verify(C.read(newcell/'complete.json')['files'])
        for source,stage in (('GUARD_GEO','GUARD_GEO'),('FINAL','UNIFIED_FINAL')):
            p=newcell/source/'scores/row.json';append(q,LABEL,stage,C.read(p),p)
    return qq,rows,sources

def run():
    qq,rows,sources=gather()
    excluded=set(E.V.E.META)|{'model','stage','label','part_type','noise_index','N','K','C_boundary_edges'}
    policy=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    fields=sorted(set(policy)|{k for r in rows for k,v in r.items() if k not in excluded and number(v)})
    summary=[];conditions=[];parents=[]
    bygroup=collections.defaultdict(list)
    for r in rows:bygroup[r['model'],r['stage'],r['part_type']].append(r)
    for (model,stage,part),group in bygroup.items():
        for split in ('ALL',*sorted({r['split'] for r in group})):
            selected=[r for r in group if split=='ALL' or r['split']==split]
            for metric in fields:
                cc=collections.defaultdict(list)
                for r in selected:cc[r['parent_id'],r['condition_id']].append(r.get(metric))
                pp=collections.defaultdict(list)
                for (parent,cid),vals in cc.items():
                    vv=[float(v) for v in vals if number(v)];value=float(np.mean(vv)) if vv else None
                    pp[parent].append(value)
                    if split!='ALL':conditions.append(dict(model=model,stage=stage,part_type=part,split=split,parent_id=parent,condition_id=cid,metric=metric,value=value,n_outputs=len(vals),n_valid=len(vv)))
                pvalues=[]
                for parent,vals in pp.items():
                    vv=[v for v in vals if v is not None];value=float(np.mean(vv)) if vv else None
                    if value is not None:pvalues.append(value)
                    if split!='ALL':parents.append(dict(model=model,stage=stage,part_type=part,split=split,parent_id=parent,metric=metric,value=value,n_conditions=len(vals),n_valid=len(vv)))
                vv=[float(r[metric]) for r in selected if number(r.get(metric))]
                summary.append(dict(model=model,stage=stage,part_type=part,split=split,metric=metric,
                    mean_parent_equal=float(np.mean(pvalues)) if pvalues else None,
                    mean_output=float(np.mean(vv)) if vv else None,n_outputs=len(selected),n_valid=len(vv),
                    n_parents=len(pp),n_valid_parents=len(pvalues),status='PASS' if vv else 'NA_NOT_RECORDED_OR_NOT_APPLICABLE'))
    index={(r['model'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for q in qq:
        for stage in STAGES:
            x=index['S2000',stage,q['request_id']];y=index[LABEL,stage,q['request_id']]
            for metric in fields:
                a,b=x.get(metric),y.get(metric);valid=number(a) and number(b)
                paired.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                    part_type=q['part_type'],split=q['split'],stage=stage,metric=metric,old_value=a,new_value=b,
                    delta=float(b)-float(a) if valid else None,status='PASS' if valid else 'NA_NOT_COMMON',direction=policy.get(metric,'DESCRIPTIVE')))
    files={}
    for name,rr in (('per_output',rows),('per_condition',conditions),('per_parent',parents),('summary_parent_equal',summary),('paired_per_output',paired)):
        p=DEST/(name+'.csv');C.csvwrite(p,rr);files[str(p)]=C.sha(p)
    key=[r for r in summary if r['metric'] in KEYS];C.csvwrite(DEST/'key_summary.csv',key)
    if 'torch' in sys.modules:raise RuntimeError('Read-only summary imported torch')
    result=dict(status='COMPLETE',requests=len(qq),per_output_rows=len(rows),metrics=len(fields),metric_names=fields,
        files=files,source_files=sources,model_calls=0,scoring_calls=0,registration_sha256=C.sha(DEST/'registration.json'))
    C.write(DEST/'completion.json',result)
    print(dict(status='COMPLETE',requests=len(qq),rows=len(rows),metrics=len(fields)),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else run()
