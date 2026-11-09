"""Read-only all-model matched-request aggregation, no model/scorer calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
from .eval_prepare import OUT,OLD,LABELS,HISTORICAL,P,requests,verify

MODELS=(*HISTORICAL,*LABELS)
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
KEYS=('FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C',
    'free_minimum_angle_lt5_fraction','deleted_free_faces','actual_free_faces','nonmanifold_edges',
    'free_only_reached_bulk_area_fraction','bulk_reach_ge_0_30_descriptive','C_F_valid_edge_count',
    'legal_seam_bulk_edge_count','hinge_diag_vertex_attached_only_free_area_fraction',
    'hinge_diag_isolated_1face_edge_components')

def numeric(x):return isinstance(x,(int,float,bool)) and np.isfinite(float(x))
def writecsv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def cpu_dest(model,q):
    return OUT/('historical' if model in HISTORICAL else 'cpu')/model/q['request_id']

def dev_official_summary(rows,fields):
    """Keep adaptation holdout separate from the official dataset partition."""
    result=[]
    for model in MODELS:
      for stage in STAGES:
       for official in ('test','train'):
        rr=[r for r in rows if r['model']==model and r['stage']==stage and r['split']=='DEV'
            and r['official_split']==official]
        if not rr:continue
        for metric in fields:
            cc=collections.defaultdict(list)
            for r in rr:cc[r['parent_id'],r['condition_id']].append(r.get(metric))
            pp=collections.defaultdict(list)
            for (parent,cid),values in cc.items():
                vv=[float(v) for v in values if numeric(v)]
                pp[parent].append(float(np.mean(vv)) if vv else None)
            pvalues=[]
            for values in pp.values():
                vv=[v for v in values if v is not None]
                if vv:pvalues.append(float(np.mean(vv)))
            valid=[float(r[metric]) for r in rr if numeric(r.get(metric))]
            result.append(dict(model=model,stage=stage,split='DEV',official_split=official,
                evaluation_stratum='DEV_OFFICIAL_TEST' if official=='test' else 'DEV_OFFICIAL_TRAIN_HELDOUT',
                metric=metric,mean_parent_equal=float(np.mean(pvalues)) if pvalues else None,
                mean_output=float(np.mean(valid)) if valid else None,
                n_outputs=len(rr),n_conditions=len(cc),n_parents=len(pp),n_valid=len(valid),
                n_valid_parents=len(pvalues),status='PASS' if len(valid)==len(rr) else 'NA_OR_PARTIAL'))
    return result

def dev_official_paired_summary(paired):
    groups=collections.defaultdict(list)
    for r in paired:
        if r['split']=='DEV':
            groups[r['baseline'],r['candidate'],r['stage'],r['official_split'],r['metric']].append(r)
    result=[]
    for (baseline,candidate,stage,official,metric),rr in sorted(groups.items()):
        cc=collections.defaultdict(list)
        for r in rr:cc[r['parent_id'],r['condition_id']].append(r['delta'])
        pp=collections.defaultdict(list)
        for (parent,cid),values in cc.items():
            vv=[float(v) for v in values if numeric(v)]
            pp[parent].append(float(np.mean(vv)) if vv else None)
        pv=[]
        for values in pp.values():
            vv=[v for v in values if v is not None]
            if vv:pv.append(float(np.mean(vv)))
        result.append(dict(baseline=baseline,candidate=candidate,stage=stage,split='DEV',official_split=official,
            evaluation_stratum='DEV_OFFICIAL_TEST' if official=='test' else 'DEV_OFFICIAL_TRAIN_HELDOUT',
            metric=metric,delta_parent_equal=float(np.mean(pv)) if pv else None,
            n_outputs=len(rr),n_conditions=len(cc),n_parents=len(pp),
            n_valid_outputs=sum(numeric(r['delta']) for r in rr),n_valid_parents=len(pv)))
    return result

def gather():
    qq=requests();rows=[];protected={}
    topology_path=OUT/'topology_audit/condition_topology_audit.csv'
    with topology_path.open(encoding='utf-8-sig') as f:topology={r['condition_id']:r for r in csv.DictReader(f)}
    protected[str(topology_path)]=P.sha(topology_path)
    extras_dir=OUT/'post_extras/all'
    extra_done=P.read(extras_dir/'completion.json');verify(extra_done['files'])
    protected[str(extras_dir/'completion.json')]=P.sha(extras_dir/'completion.json')
    with (extras_dir/'per_output_fixed_BASE_and_pointjump.csv').open(encoding='utf-8-sig') as f:
        extra_rows=list(csv.DictReader(f))
    def scalar(v):
        if v=='':return None
        if v in ('True','False'):return v=='True'
        try:return float(v)
        except ValueError:return v
    extra_metadata={'label','stage','request_id','condition_id','parent_id','split','part_type','source_edge_components','topology_stratum','slit_edge_count','closed_C'}
    extra_fields=set(extra_rows[0])-extra_metadata
    extras={(r['label'],r['request_id'],r['stage']):{k:scalar(r[k]) for k in extra_fields} for r in extra_rows}
    assert len(extras)==480
    for model in MODELS:
        for q in qq:
            d=cpu_dest(model,q);complete=P.read(d/'complete.json');verify(complete['files'])
            protected[str(d/'complete.json')]=P.sha(d/'complete.json')
            baserows=P.read(d/'metric_rows.json');by={r['stage']:r for r in baserows}
            rp=OUT/'evaluation/scores'/model/q['request_id']/'RAW'
            verify(P.read(rp/'audit_receipt.json')['files']);protected[str(rp/'audit_receipt.json')]=P.sha(rp/'audit_receipt.json')
            by['RAW']=P.read(rp/'row.json')
            for stage in STAGES:
                r=dict(by[stage]);r.update(model=model,label=model,stage=stage,request_id=q['request_id'],
                    condition_id=q['condition_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],
                    N=q['N'],K=q['K'],closed_C=q['closed_C'],part_type=q['part_type'],part_family=q['part_family'],official_split=q['official_split'])
                t=topology[q['condition_id']]
                r.update(source_edge_components=int(t['source_edge_components']),topology_stratum=t['topology_stratum'],
                    slit_candidate_edge_count=int(t['slit_edge_count']))
                r.update(extras[model,q['request_id'],stage] if stage!='RAW' else {k:None for k in extra_fields})
                r['additional_post_metrics_status']='NA_RAW_NOT_POST' if stage=='RAW' else 'DERIVED_FROM_CURRENT_POST_AND_FIXED_BASE'
                rows.append(r)
    assert len(rows)==720
    return qq,rows,protected

def run():
    dest=OUT/'comparison'
    done=dest/'completion.json'
    if done.exists():
        r=P.read(done);verify(r['files']);verify(r['source_receipts']);return r
    qq,rows,sources=gather()
    # Freeze the complete previously maintained179 schema, including all-NA
    # fields. Different representations never silently remove inconvenient fields.
    schema_path=P.ROOT/'experiments/r2_chair_multipart_seam_v1/comparison/completion.json'
    schema=P.read(schema_path);fields=schema['metric_names']
    assert len(fields)==179 and all(k in row for row in rows for k in fields)
    meta={'noise_index','N','K','closed_C','source_edge_components','slit_candidate_edge_count'}
    extra=sorted({k for row in rows for k,v in row.items() if numeric(v) and k not in set(fields)|meta})
    sources[str(schema_path)]=P.sha(schema_path)
    conditions=[];parents=[];summaries=[];topology_summaries=[]
    parts=sorted({q['part_type'] for q in qq})
    for model in MODELS:
      for stage in STAGES:
       for split in ('ALL','TRAIN','DEV'):
        for part in ('ALL',*parts):
         rr=[r for r in rows if r['model']==model and r['stage']==stage and (split=='ALL' or r['split']==split) and (part=='ALL' or r['part_type']==part)]
         if not rr:continue
         for metric in fields:
            byc=collections.defaultdict(list)
            for r in rr:byc[r['parent_id'],r['condition_id']].append(r.get(metric))
            byp=collections.defaultdict(list)
            for (parent,cid),values in byc.items():
                valid=[float(v) for v in values if numeric(v)];value=float(np.mean(valid)) if valid else None
                byp[parent].append(value)
                if split!='ALL' and part=='ALL':conditions.append(dict(model=model,stage=stage,split=split,parent_id=parent,condition_id=cid,
                    metric=metric,value=value,n_outputs=len(values),n_valid=len(valid)))
            pvalues=[]
            for parent,values in byp.items():
                valid=[v for v in values if v is not None];value=float(np.mean(valid)) if valid else None
                if value is not None:pvalues.append(value)
                if split!='ALL' and part=='ALL':parents.append(dict(model=model,stage=stage,split=split,parent_id=parent,metric=metric,
                    value=value,n_conditions=len(values),n_valid_conditions=len(valid)))
            valid=[float(r[metric]) for r in rr if numeric(r.get(metric))]
            summaries.append(dict(model=model,stage=stage,split=split,part_type=part,metric=metric,
                mean_parent_equal=float(np.mean(pvalues)) if pvalues else None,
                mean_output=float(np.mean(valid)) if valid else None,median_output=float(np.median(valid)) if valid else None,
                n_outputs=len(rr),n_valid=len(valid),n_parents=len(byp),n_valid_parents=len(pvalues),
                status='PASS' if len(valid)==len(rr) else 'NA_OR_PARTIAL'))
    index={(r['model'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for model in MODELS:
     for stage in STAGES:
      for split in ('ALL','TRAIN','DEV'):
       for stratum in sorted({r['topology_stratum'] for r in rows}):
        rr=[r for r in rows if r['model']==model and r['stage']==stage and (split=='ALL' or r['split']==split)
            and r['topology_stratum']==stratum and not r['closed_C']]
        if not rr:continue
        for metric in [*fields,*extra]:
            cc=collections.defaultdict(list)
            for r in rr:
                if numeric(r.get(metric)):cc[r['parent_id'],r['condition_id']].append(float(r[metric]))
            pp=collections.defaultdict(list)
            for (p,c),vv in cc.items():pp[p].append(float(np.mean(vv)))
            topology_summaries.append(dict(model=model,stage=stage,split=split,topology_stratum=stratum,metric=metric,
                mean_parent_equal=float(np.mean([np.mean(v) for v in pp.values()])) if pp else None,
                n_outputs=len(rr),n_valid=sum(len(v) for v in cc.values()),n_valid_conditions=len(cc),n_valid_parents=len(pp),
                applicability='OPEN_C_ONLY; closed C remains in complete quality tables'))
    contrasts=[('SEAM_S15000','NEW_HEAD'),('SEAM_S21000','NEW_HEAD'),('NATIVE_S21000','NEW_HEAD'),
        ('OLD_S15000_LAYOUT','NEW_HEAD'),('SEAM_S15000','OLD_S15000_LAYOUT')]
    for baseline,candidate in contrasts:
      for stage in STAGES:
       for q in qq:
        a,b=index[baseline,stage,q['request_id']],index[candidate,stage,q['request_id']]
        for metric in [*fields,*extra]:
            x,y=a[metric],b[metric];ok=numeric(x) and numeric(y)
            paired.append(dict(baseline=baseline,candidate=candidate,stage=stage,request_id=q['request_id'],
                condition_id=q['condition_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],part_type=q['part_type'],
                official_split=q['official_split'],
                metric=metric,baseline_value=x,candidate_value=y,delta=float(y)-float(x) if ok else None,status='PASS' if ok else 'NA_NOT_COMMON'))
    paired_c=[];paired_p=[]
    groups=collections.defaultdict(list)
    for r in paired:groups[r['baseline'],r['candidate'],r['stage'],r['split'],r['metric'],r['parent_id'],r['condition_id']].append(r)
    parentgroups=collections.defaultdict(list)
    for (a,b,s,g,m,p,c),values in groups.items():
        valid=[r['delta'] for r in values if r['status']=='PASS'];v=float(np.mean(valid)) if valid else None
        paired_c.append(dict(baseline=a,candidate=b,stage=s,split=g,metric=m,parent_id=p,condition_id=c,delta=v,n_valid=len(valid),n_registered=len(values)))
        parentgroups[a,b,s,g,m,p].append(v)
    for (a,b,s,g,m,p),values in parentgroups.items():
        valid=[v for v in values if v is not None]
        paired_p.append(dict(baseline=a,candidate=b,stage=s,split=g,metric=m,parent_id=p,delta=float(np.mean(valid)) if valid else None,
            n_valid_conditions=len(valid),n_conditions=len(values)))
    files={}
    extrasummaries=[];extrac=[];extrap=[]
    for model in MODELS:
     for stage in STAGES:
      for split in ('TRAIN','DEV','ALL'):
       rr=[r for r in rows if r['model']==model and r['stage']==stage and (split=='ALL' or r['split']==split)]
       for metric in extra:
        cc=collections.defaultdict(list)
        for r in rr:cc[r['parent_id'],r['condition_id']].append(r.get(metric))
        pp=collections.defaultdict(list)
        for (parent,cid),values in cc.items():
            valid=[float(v) for v in values if numeric(v)];value=float(np.mean(valid)) if valid else None;pp[parent].append(value)
            if split!='ALL':extrac.append(dict(model=model,stage=stage,split=split,parent_id=parent,condition_id=cid,metric=metric,value=value,n_valid=len(valid),n_outputs=len(values)))
        pv=[]
        for parent,values in pp.items():
            valid=[v for v in values if v is not None];value=float(np.mean(valid)) if valid else None
            if value is not None:pv.append(value)
            if split!='ALL':extrap.append(dict(model=model,stage=stage,split=split,parent_id=parent,metric=metric,value=value,n_valid_conditions=len(valid),n_conditions=len(values)))
        extrasummaries.append(dict(model=model,stage=stage,split=split,metric=metric,mean_parent_equal=float(np.mean(pv)) if pv else None,
            n_outputs=len(rr),n_valid=sum(numeric(r.get(metric)) for r in rr),n_valid_parents=len(pv)))
    official=dev_official_summary(rows,[*fields,*extra])
    official_paired=dev_official_paired_summary(paired)
    for name,data in (('per_output',rows),('per_condition',conditions),('per_parent',parents),('summary_parent_equal',summaries),
        ('paired_per_output',paired),('paired_per_condition',paired_c),('paired_per_parent',paired_p),
        ('key_summary',[r for r in summaries if r['metric'] in KEYS]),('topology_strata_summary',topology_summaries),
        ('dev_official_split_summary',official),('dev_official_split_paired_summary',official_paired),
        ('additional_per_condition',extrac),('additional_per_parent',extrap),('additional_summary',extrasummaries)):
        path=dest/(name+'.csv');writecsv(path,data);files[str(path)]=P.sha(path)
    result=dict(status='COMPLETE',rows=len(rows),primary_BASE_FINAL_cells=480,RAW_cells=240,metric_names=fields,metric_count=179,
        additional_numeric_fields=extra,
        models=list(MODELS),stages=list(STAGES),files=files,source_receipts=sources,
        aggregation='noise -> condition -> parent equal; original split/part preserved; source style not ground truth',
        DEV_partition='7 official-test parents/28 requests; 1 official-train adaptation-heldout parent/4 requests. DEV is not synonymous with official test.',
        causality='Historical different budgets; new-head vs cached-layout same generator isolates layout intervention, not all weight-layout interactions',
        model_calls=0,scorer_calls=0)
    P.immutable(done,result);return result

def verify_historical_raw():
    from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
    from r2.pilots.seam_abc_v1 import eval_common as C
    fields=P.read(P.ROOT/'experiments/r2_bench_control_transfer_v1/evaluation/summary_completion.json')['numeric_fields']
    differences=[];count=0
    for model in HISTORICAL:
      for q in requests():
        old=OLD/'evaluation/scores'/model/q['request_id']/'RAW'
        score=P.read(old/'score.json');verify(score['files'])
        a=metric_row(score,C.arrays(score['identity']['mesh_path']),P.read(old/'pairs.json'))[0]
        b=P.read(OUT/'evaluation/scores'/model/q['request_id']/'RAW/row.json')
        for field in fields:
            x,y=a.get(field),b.get(field)
            same=(x==y) if not(numeric(x) and numeric(y)) else abs(float(x)-float(y))<=1e-12
            if not same:differences.append(dict(model=model,request_id=q['request_id'],metric=field,old=x,new=y))
            count+=1
    result=dict(status='PASS' if not differences else 'DIFFERENCES_REVIEW_REQUIRED',
        comparisons=count,old_fields=len(fields),outputs=144,differences=differences,model_calls=0,new_scoring=0)
    P.immutable(OUT/'historical_RAW_metric_equivalence.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-historical-raw',action='store_true');a=parser.parse_args()
    if a.verify_historical_raw:r=verify_historical_raw();print(dict(status=r['status'],comparisons=r['comparisons'],differences=len(r['differences'])),flush=True)
    else:r=run();print(dict(status=r['status'],rows=r['rows'],metrics=r['metric_count']),flush=True)
