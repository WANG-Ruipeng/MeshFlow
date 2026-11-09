"""Read completed 124-field scores, including three read-only historic arms.

No geometry classification, model call, query or visual judgement occurs here.
Wave1 and final tables have separate immutable destinations.
"""
from __future__ import annotations
import argparse
import collections
import json
import numpy as np
from . import scoring as S, binding as B
from .evaluation import C, ROOT, OUT, WAVE1, WAVE2, WAVES, requests, wave_for

E=S.E
BASELINES=('LONG','B_LONG_LAYOUT','PLAIN3000')
BASE_PATHS={
    'LONG': (ROOT/'experiments/r2_seam_abc_v1', ROOT/'experiments/r2_seam_followup_v1/binding'),
    'B_LONG_LAYOUT': (ROOT/'experiments/r2_seam_followup_v1/layout', ROOT/'experiments/r2_seam_pool47_couple_v1/cpu_binding'),
    'PLAIN3000': (ROOT/'experiments/r2_seam_pool47_couple_v1', ROOT/'experiments/r2_seam_pool47_couple_v1/endpoint_binding')}


def labels(scope):
    if scope=='wave1':
        return ('LONG','B_LONG_LAYOUT',*WAVE1[:2],'PLAIN3000',WAVE1[2])
    if scope=='final':
        return ('LONG','B_LONG_LAYOUT',*WAVE1[:2],'PLAIN3000',WAVE1[2],*WAVE2)
    raise ValueError('Unknown summary scope')


def collect(scope):
    active=labels(scope)
    waves=('wave1',) if scope=='wave1' else ('wave1','wave2')
    numeric=None
    for wave in waves:
        done=C.read(S.files(wave)['summary_completion'])
        bound=C.read(B.folder(wave)/'completion.json')
        if done['numeric_field_count']!=124 or bound['cells']!=28*len(WAVES[wave]):
            raise ValueError('Incomplete wave scoring/binding')
        if numeric is not None and numeric!=done['numeric_fields']:
            raise ValueError('Metric schema differs across waves')
        numeric=done['numeric_fields']
    rows,roles,deleted,ledger,refs=[],[],[],[],{}
    for q in requests():
        for label in active:
            if label in BASE_PATHS:
                evalroot,bindroot=BASE_PATHS[label]
            else:
                evalroot,bindroot=OUT,B.folder(wave_for(label))
            for stage in ('RAW','POST'):
                p=evalroot/'evaluation/scores'/label/q['request_id']/stage/'score.json'
                score=C.read(p);E.verify(score['files']);refs[str(p)]=C.sha(p)
                for key in ('request_id','condition_id','parent_id','split','noise_index','N','K'):
                    if score[key]!=q[key]:
                        raise ValueError('Historic or new request identity mismatch: '+str(p)+' '+key)
                mesh=C.arrays(score['identity']['mesh_path']);pairs=C.read(p.parent/'pairs.json')
                row,rr,dd=S.metric_row(score,mesh,pairs)
                row['mesh_path']=score['identity']['mesh_path'];rows.append(row);roles.extend(rr)
                for d in dd:d['stage']=stage
                deleted.extend(dd)
            dest=bindroot/'cells'/(label+'_BIND_015')/q['request_id']
            receipt=C.read(dest/'complete.json');E.verify(receipt['files']);refs[str(dest/'complete.json')]=C.sha(dest/'complete.json')
            original=receipt['row'];row={k:original.get(k) for k in (*E.META,*numeric)}
            row.update(label=label,stage='BOUND015',mesh_path=str(dest/'post.npz'),variant_label=original['label'])
            if row['request_id']!=q['request_id'] or row['split']!=q['split']:
                raise ValueError('Binding request mismatch')
            rows.append(row)
            paired=C.read(dest/'paired_ledger.json')
            ledger.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                noise_index=q['noise_index'],**paired['summary'],accepted_tip=original['accepted_tip'],
                accepted_edge=original['accepted_edge'],max_PRE_GEO_motion=original['max_PRE_GEO_motion']))
            mesh=C.arrays(dest/'post.npz');pairs=C.read(dest/'pairs.json');score=C.read(dest/'score.json')
            score={**score,'label':label,'stage':'POST'}
            _,rr,dd=S.metric_row(score,mesh,pairs)
            for r in rr:r['stage']='BOUND015'
            for r in dd:r['stage']='BOUND015'
            roles.extend(rr);deleted.extend(dd)
    if len(rows)!=len(active)*28*3:
        raise ValueError('Incomplete read-only combined matrix')
    return numeric,rows,roles,deleted,ledger,refs


def number(x):
    return isinstance(x,(float,int,bool)) and np.isfinite(float(x))


def summarize(scope):
    numeric,rows,roles,deleted,ledger,refs=collect(scope)
    active=labels(scope);lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    summary=[];keyrows=[]
    for label in active:
        for stage in ('RAW','POST','BOUND015'):
            for split in ('TRAIN','DEV','ALL'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                for metric in numeric:
                    vals=[r[metric] for r in rr if number(r.get(metric))]
                    summary.append(dict(label=label,stage=stage,split=split,metric=metric,n_registered=len(rr),n_valid=len(vals),
                        mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                        min=float(min(vals)) if vals else None,max=float(max(vals)) if vals else None,
                        status='PASS' if len(vals)==len(rr) else 'NA_OR_PARTIAL'))
                key=dict(label=label,stage=stage,split=split,n=len(rr),
                    bulk_reach30_count=sum(number(r.get('free_only_reached_bulk_area_fraction')) and
                        r['free_only_reached_bulk_area_fraction']>=.3 for r in rr))
                for metric in numeric:
                    vals=[r[metric] for r in rr if number(r.get(metric))]
                    key[metric]=float(np.mean(vals)) if vals else None
                vv=[r['free_only_reached_bulk_area_fraction'] for r in rr if number(r.get('free_only_reached_bulk_area_fraction'))]
                key['free_only_reached_bulk_area_fraction_median']=float(np.median(vv)) if vv else None
                if stage=='BOUND015':
                    ll=[r for r in ledger if r['label']==label and (split=='ALL' or r['split']==split)]
                    for name in ('accepted_tip','accepted_edge','CF_new_illegal_pairs','FF_new_illegal_pairs',
                                 'FF_illegal_pairs_removed','CF_lost_legal_edges','FF_lost_legal_edges','deleted_source_faces'):
                        key['sum_'+name]=sum(r[name] for r in ll)
                keyrows.append(key)
    comparisons=[]
    for candidate in active:
        if candidate not in ('LONG','B_LONG_LAYOUT'):
            comparisons.append(('B_LONG_LAYOUT',candidate,'B_STEP0_REFERENCE'))
        if candidate!='LONG':
            comparisons.append(('LONG',candidate,'HISTORICAL_LONG_REFERENCE'))
    comparisons.append(('PLAIN3000','WISE_B_PLAIN_050','FIXED_WEIGHT_MIDPOINT_VS_FINAL'))
    if scope=='final':
        comparisons.extend(('PLAIN_S1000',arm,'PRIMARY_EQUAL_1000_UPDATES') for arm in WAVE2)
    pairs=[]
    for baseline,candidate,purpose in comparisons:
        for q in requests():
            for stage in ('RAW','POST','BOUND015'):
                a,b=lookup[baseline,stage,q['request_id']],lookup[candidate,stage,q['request_id']]
                for metric in numeric:
                    x,y=a.get(metric),b.get(metric);good=number(x) and number(y)
                    pairs.append(dict(baseline=baseline,candidate=candidate,purpose=purpose,request_id=q['request_id'],
                        parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],stage=stage,metric=metric,
                        baseline_value=x,candidate_value=y,delta=float(y)-float(x) if good else None,
                        status='PASS' if good else 'NA'))
    fields=('baseline','candidate','purpose','parent_id','split','stage','metric')
    groups=collections.defaultdict(list)
    for r in pairs:groups[tuple(r[k] for k in fields)].append(r)
    parents=[]
    for key,rr in sorted(groups.items()):
        vv=[r for r in rr if r['status']=='PASS']
        parents.append(dict(zip(fields,key),n_registered=4,n_valid=len(vv),
            **{k:float(np.mean([r[k] for r in vv])) if vv else None for k in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(vv)==4 else 'NA_OR_PARTIAL'))
    effects=[]
    for label in active:
        for q in requests():
            a,b=lookup[label,'POST',q['request_id']],lookup[label,'BOUND015',q['request_id']]
            for metric in numeric:
                x,y=a.get(metric),b.get(metric);good=number(x) and number(y)
                effects.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],
                    split=q['split'],noise_index=q['noise_index'],metric=metric,original_POST=x,bound015=y,
                    delta=float(y)-float(x) if good else None,status='PASS' if good else 'NA'))
    tables={'per_output':rows,'summary':summary,'key_results':keyrows,'paired_per_noise':pairs,
        'paired_per_parent':parents,'binding_effect_per_output':effects,'binding_ledger':ledger,
        'illegal_pairs_by_role':roles,'deleted_faces':deleted}
    dest=OUT/'results'/scope
    for name,rr in tables.items():C.csvwrite(dest/(name+'.csv'),rr)
    result=dict(status='NUMERICAL_COMPLETE_VISUAL_PENDING',scope=scope,rows=len(rows),numeric_fields=124,
        numeric_field_names=numeric,labels=active,stages=['RAW','POST','BOUND015'],source_scores=refs,
        model_calls=0,new_scans=0,baseline_regeneration=0,
        primary_wave2_comparator='PLAIN_S1000; PLAIN_S2000 is a trajectory reference, not a 1000-update matched control',
        files={str(dest/(name+'.csv')):C.sha(dest/(name+'.csv')) for name in tables})
    C.write(dest/'completion.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('summarize',))
    parser.add_argument('--scope',choices=('wave1','final'),required=True);args=parser.parse_args()
    result=summarize(args.scope)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_scores','files','numeric_field_names')}),flush=True)


if __name__=='__main__':
    main()
