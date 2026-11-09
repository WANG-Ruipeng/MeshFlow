"""Complete original124 + separately named topology fields, paired by request."""
from __future__ import annotations
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
from .evaluation import C,OUT,HIST,ARMS,requests,mesh_path,score_path
from . import scoring as S,topology as T

def number(x):return isinstance(x,(float,int,bool)) and np.isfinite(float(x))

def table(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def collect(include_hinge):
    rows=[];roles=[];deletions=[];sources={};labels=('PLAIN_S1000','PLAIN_S2000',*ARMS)
    numeric=C.read(HIST/'evaluation/wave1_summary_completion.json')['numeric_fields'];assert len(numeric)==124
    for q in requests():
        for label in labels:
            for stage in ('RAW','POST','BOUND015'):
                sp=score_path(label,q['request_id'],stage);s=C.read(sp);mesh=C.arrays(mesh_path(label,q['request_id'],stage))
                receipt=s if 'files' in s else C.read(sp.parent/'complete.json');S.E.verify(receipt['files']);sources[str(sp)]=C.sha(sp)
                for key in S.E.META:assert s[key]==q[key],(key,sp)
                row,rr,dd=S.metric_row(s,mesh,C.read(sp.parent/'pairs.json'))
                row.update(label=label,stage=stage,mesh_path=str(mesh_path(label,q['request_id'],stage)))
                rows.append(row)
                for r in rr:r.update(label=label,stage=stage)
                for r in dd:r.update(label=label,stage=stage)
                roles.extend(rr);deletions.extend(dd)
    if include_hinge:
        reg=C.read(OUT/'hinge/registry.json');records=reg if isinstance(reg,list) else reg['records']
        assert len(records)==168
        for r in records:
            p=C.resolve(r['score_path']);s=C.read(p);receipt=C.read(p.parent/'complete.json');S.E.verify(receipt['files'])
            base_label=r['label'].split('_HINGE')[0];stage=r['stage'];s={**s,'label':base_label,'stage':'POST'}
            row,rr,dd=S.metric_row(s,C.arrays(r['mesh_path']),C.read(p.parent/'pairs.json'))
            row.update(label=base_label,stage=stage,mesh_path=r['mesh_path']);rows.append(row);sources[str(p)]=C.sha(p)
            for a in rr:a.update(label=base_label,stage=stage)
            for a in dd:a.update(label=base_label,stage=stage)
            roles.extend(rr);deletions.extend(dd)
    return numeric,rows,roles,deletions,sources

def run(include_hinge=False):
    numeric,rows,roles,drops,sources=collect(include_hinge);labels=('PLAIN_S1000','PLAIN_S2000',*ARMS)
    td=T.run(list(labels),include_hinge);tp=next(p for p in td['files'] if p.endswith('/per_output.csv'))
    trows=table(tp);topo_names=sorted(k for k in trows[0] if k.startswith('hinge_diag_'))
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for r in trows:
        label=r['label'].split('_HINGE')[0];row=lookup[label,r['stage'],r['request_id']]
        for k in topo_names:row[k]=float(r[k]) if r[k] else None
        row['topology_definition_version']=T.VERSION
    fields=numeric+topo_names
    for row in rows:
        if row['stage']=='RAW':
            row.update({k:None for k in topo_names});row['topology_definition_version']='NA_RAW_NOT_FINAL_WELDED_GRAPH'
    summary=[];keyrows=[]
    domains=sorted({(r['label'],r['stage']) for r in rows})
    for label,stage in domains:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
            key=dict(label=label,stage=stage,split=split,n=len(rr),bulk_reach30_count=sum(number(r.get('free_only_reached_bulk_area_fraction')) and r['free_only_reached_bulk_area_fraction']>=.3 for r in rr))
            for metric in fields:
                vv=[r[metric] for r in rr if number(r.get(metric))]
                summary.append(dict(label=label,stage=stage,split=split,metric=metric,n_registered=len(rr),n_valid=len(vv),
                    mean=float(np.mean(vv)) if vv else None,median=float(np.median(vv)) if vv else None,
                    min=float(min(vv)) if vv else None,max=float(max(vv)) if vv else None,
                    status='PASS' if len(vv)==len(rr) else 'NA_RAW_GRAPH' if stage=='RAW' and metric in topo_names else 'NA_OR_PARTIAL'))
                key[metric]=float(np.mean(vv)) if vv else None
            keyrows.append(key)
    compare=[]
    for arm in ARMS:
        for stage in ('RAW','POST','BOUND015'):compare.append(('PLAIN_S2000',stage,arm,stage,'PRIMARY_EQUAL_WINDOW1000'))
    for stage in ('RAW','POST','BOUND015'):compare.append(('PLAIN_S1000',stage,'PLAIN_S2000',stage,'CONTINUATION_REFERENCE'))
    if include_hinge:
        for label in ('PLAIN_S2000',*ARMS):
            for stage in ('POST_HINGE015','POST_HINGE025'):compare.append((label,'POST',label,stage,'FROZEN_HINGE_INCREMENT'))
        for label in ARMS:
            for stage in ('POST_HINGE015','POST_HINGE025'):compare.append(('PLAIN_S2000',stage,label,stage,'SAME_HINGE_EQUAL_TRAINING_WINDOW'))
    paired=[]
    for baseline,bst,candidate,cst,purpose in compare:
        for q in requests():
            a,b=lookup[baseline,bst,q['request_id']],lookup[candidate,cst,q['request_id']]
            for metric in fields:
                x,y=a.get(metric),b.get(metric);ok=number(x) and number(y)
                paired.append(dict(baseline=baseline,baseline_stage=bst,candidate=candidate,candidate_stage=cst,purpose=purpose,
                    **{k:q[k] for k in ('request_id','parent_id','noise_index','split')},metric=metric,
                    baseline_value=x,candidate_value=y,delta=float(y)-float(x) if ok else None,status='PASS' if ok else 'NA'))
    groups=collections.defaultdict(list);groupkeys=('baseline','baseline_stage','candidate','candidate_stage','purpose','parent_id','split','metric')
    for r in paired:groups[tuple(r[k] for k in groupkeys)].append(r)
    parents=[]
    for key,rr in sorted(groups.items()):
        vv=[r for r in rr if r['status']=='PASS'];parents.append(dict(zip(groupkeys,key),n_registered=4,n_valid=len(vv),
            **{k:float(np.mean([r[k] for r in vv])) if vv else None for k in ('baseline_value','candidate_value','delta')},status='PASS' if len(vv)==4 else 'NA_OR_PARTIAL'))
    dest=OUT/'results'/('final' if include_hinge else 'before_hinge');files={}
    for name,rr in [('per_output',rows),('summary',summary),('key_results',keyrows),('paired_per_noise',paired),('paired_per_parent',parents),('illegal_pairs_by_role',roles),('deleted_faces',drops)]:
        p=dest/(name+'.csv');C.csvwrite(p,rr);files[str(p)]=C.sha(p)
    result=dict(status='NUMERICAL_COMPLETE_VISUAL_PENDING',rows=len(rows),original_numeric_fields=numeric,original_numeric_count=124,
        additional_topology_fields=topo_names,topology_completion=td,files=files,source_scores=sources,models=labels,
        paired_unit='Identical C,N,baseGaussian,query identity; transformed prior is explicit arm intervention; no sourceGTstyle ranking',
        semantic_slot_warning='Original RAW slot is only within-output provenance. Cross-trained identical slot ID is not a semantic face correspondence.',new_scans=0,model_calls=0)
    C.write(dest/'completion.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-hinge',action='store_true');a=p.parse_args();r=run(a.include_hinge);print(r['status'],r['rows'],flush=True)
