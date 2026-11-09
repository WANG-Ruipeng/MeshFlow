"""Full endpoint comparisons, source role/deletion ledgers and render manifests.

Consumes completed scores/bindings only. No metric definition, geometry, model,
query, or noise is modified, and no image is claimed reviewed by this module.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from r2.pilots.seam_pool47_couple_v1 import endpoint_scoring as S, endpoint_binding as B, cpu_binding as OLD
C,E=S.C,S.E
OUT=S.ROOT/'results'
BASELINE='B_LONG_LAYOUT'
LABELS=(BASELINE,*S.ARMS)


def collect():
    sr=C.read(S.ROOT/'evaluation/summary_completion.json');assert sr['numeric_field_count']==124
    assert C.read(B.OUT/'completion.json')['cells']==56
    numeric=sr['numeric_fields'];rows=[];roles=[];deleted=[];binding=[];refs={}
    for q in S.requests():
        for label in LABELS:
            root=OLD.BASE if label==BASELINE else S.ROOT
            for stage in ('RAW','POST'):
                p=root/'evaluation/scores'/label/q['request_id']/stage/'score.json'
                score=C.read(p);E.verify(score['files']);refs[str(p)]=C.sha(p)
                mesh=C.arrays(score['identity']['mesh_path']);pair=C.read(p.parent/'pairs.json')
                row,rr,dd=S.metric_row(score,mesh,pair);row['mesh_path']=score['identity']['mesh_path']
                rows.append(row);roles.extend(rr);deleted.extend(dd)
            bout=OLD.OUT if label==BASELINE else B.OUT
            folder=bout/'cells'/(label+'_BIND_015')/q['request_id'];receipt=C.read(folder/'complete.json')
            E.verify(receipt['files']);refs[str(folder/'complete.json')]=C.sha(folder/'complete.json')
            raw=receipt['row'];row={k:raw.get(k) for k in (*E.META,*numeric)}
            row.update(label=label,stage='BOUND015',mesh_path=str(folder/'post.npz'),variant_label=raw['label'])
            rows.append(row)
            ledger=C.read(folder/'paired_ledger.json')
            binding.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],noise_index=q['noise_index'],
                **ledger['summary'],accepted_tip=raw['accepted_tip'],accepted_edge=raw['accepted_edge'],max_PRE_GEO_motion=raw['max_PRE_GEO_motion']))
            mesh=C.arrays(folder/'post.npz');pairs=C.read(folder/'pairs.json');bs=C.read(folder/'score.json')
            bs={**bs,'label':label,'stage':'POST'}
            _,rr,dd=S.metric_row(bs,mesh,pairs)
            for r in rr:r['stage']='BOUND015'
            for r in dd:r['stage']='BOUND015'
            roles.extend(rr);deleted.extend(dd)
    for d in deleted:d.setdefault('stage','POST')
    assert len(rows)==252
    return numeric,rows,roles,deleted,binding,refs


def summarize():
    numeric,rows,roles,deleted,binding,refs=collect();lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    summary=[]
    for label in LABELS:
        for stage in ('RAW','POST','BOUND015'):
            for split in ('TRAIN','DEV','ALL'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                for metric in numeric:
                    vals=[r[metric] for r in rr if isinstance(r.get(metric),(float,int,bool))]
                    summary.append(dict(label=label,stage=stage,split=split,metric=metric,n_registered=len(rr),n_valid=len(vals),
                        mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                        min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None,
                        status='PASS' if len(vals)==len(rr) else 'NA_OR_PARTIAL'))
    comparisons=[(BASELINE,S.ARMS[0]),(BASELINE,S.ARMS[1]),(S.ARMS[0],S.ARMS[1])]
    pairs=[]
    for base,candidate in comparisons:
        for q in S.requests():
            for stage in ('RAW','POST','BOUND015'):
                a,b=lookup[base,stage,q['request_id']],lookup[candidate,stage,q['request_id']]
                for metric in numeric:
                    x,y=a.get(metric),b.get(metric);good=isinstance(x,(float,int,bool)) and isinstance(y,(float,int,bool))
                    pairs.append(dict(baseline=base,candidate=candidate,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                        noise_index=q['noise_index'],stage=stage,metric=metric,baseline_value=x,candidate_value=y,
                        delta=float(y)-float(x) if good else None,status='PASS' if good else 'NA'))
    groups=collections.defaultdict(list);fields=('baseline','candidate','parent_id','split','stage','metric')
    for r in pairs:groups[tuple(r[x] for x in fields)].append(r)
    parents=[]
    for key,rr in sorted(groups.items()):
        vv=[r for r in rr if r['status']=='PASS']
        parents.append(dict(zip(fields,key),n_registered=4,n_valid=len(vv),
            **{name:float(np.mean([r[name] for r in vv])) if vv else None for name in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(vv)==4 else 'NA_OR_PARTIAL'))
    effects=[]
    for label in LABELS:
        for q in S.requests():
            a,b=lookup[label,'POST',q['request_id']],lookup[label,'BOUND015',q['request_id']]
            for metric in numeric:
                x,y=a.get(metric),b.get(metric);good=isinstance(x,(float,int,bool)) and isinstance(y,(float,int,bool))
                effects.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],metric=metric,
                    original_POST=x,bound015=y,delta=float(y)-float(x) if good else None,status='PASS' if good else 'NA'))
    keyrows=[]
    for label in LABELS:
        for stage in ('RAW','POST','BOUND015'):
            for split in ('TRAIN','DEV'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and r['split']==split]
                out=dict(label=label,stage=stage,split=split,n=len(rr),bulk_reach30_count=sum(r['free_only_reached_bulk_area_fraction'] is not None and r['free_only_reached_bulk_area_fraction']>=.3 for r in rr))
                for key in ('free_only_reached_bulk_area_fraction','FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C',
                    'free_minimum_angle_lt5_fraction','actual_free_faces','C_F_valid_edge_count','C_F_valid_edge_total_length',
                    'illegal_pairs_seam-seam','illegal_pairs_seam-bulk','illegal_pairs_bulk-bulk'):
                    vals=[r[key] for r in rr if r.get(key) is not None];out[key]=float(np.mean(vals)) if vals else None
                keyrows.append(out)
    tables={'per_output':rows,'summary':summary,'paired_per_noise':pairs,'paired_per_parent':parents,
        'binding_effect_per_output':effects,'binding_ledger':binding,'illegal_pairs_by_role':roles,'deleted_faces':deleted,'key_results':keyrows}
    for name,rr in tables.items():C.csvwrite(OUT/(name+'.csv'),rr)
    result=dict(status='NUMERICAL_COMPLETE_VISUAL_PENDING',rows=252,numeric_fields=124,compared_stages=['RAW','POST','BOUND015'],
        labels=LABELS,source_scores=refs,model_calls=0,new_scans=0,
        files={str(OUT/(name+'.csv')):C.sha(OUT/(name+'.csv')) for name in tables})
    C.write(OUT/'completion.json',result);return result


def gallery_manifests():
    numeric,rows,_,_,_,refs=collect();lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    # Root owns the complete RAW/POST "training" gallery; this manifest adds
    # only the 28 same-cap three-model comparisons, without duplicate renders.
    for kind,stages in (('endpoint_binding',('BOUND015',)),):
        pages=[]
        for q in S.requests():
            for stage in stages:
                cells=[dict(label=label+(' + BIND 1.5%' if stage=='BOUND015' else ''),path=lookup[label,stage,q['request_id']]['mesh_path']) for label in LABELS]
                pages.append(dict(page_id=q['request_id']+'_'+stage,request_id=q['request_id'],condition_id=q['condition_id'],
                    parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],K=q['K'],stage=stage,kind=kind,cells=cells))
        C.write(S.ROOT/'gallery_manifests'/(kind+'.json'),dict(version='pool47_couple_same_noise_three_model_gallery_v1',pages=pages,
            intended_views='Same two depth views and front/side finite sections as prior gallery; same scale per C, no GT display',
            baseline=BASELINE,review_status='PENDING_ACTUAL_IMAGE_VIEW',source_score_hashes=refs))
    result=dict(status='MANIFEST_READY_NOT_RENDERED',endpoint_binding_pages=28,
        training_gallery='Root-owned RAW/POST 56 pages',actual_review_rows=0)
    C.write(S.ROOT/'gallery_manifests/endpoint_binding_completion.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('summarize','gallery'));a=p.parse_args()
    r=summarize() if a.stage=='summarize' else gallery_manifests()
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_scores','files')}),flush=True)

if __name__=='__main__':main()
