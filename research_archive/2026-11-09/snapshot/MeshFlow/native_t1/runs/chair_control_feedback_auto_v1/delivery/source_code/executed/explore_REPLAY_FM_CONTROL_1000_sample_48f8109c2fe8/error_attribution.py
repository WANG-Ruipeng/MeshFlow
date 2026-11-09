"""Descriptive DEV parent/noise attribution; never replace registered gates.

Reads completed DEV score JSON only. No CONFIRM access, models, geometry,
scoring, candidate selection, or new success thresholds. Means use all eight
registered parents after averaging the two registered noises within each.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np
from . import assets

OUT=assets.OUT
VERSION='chair_feedback_DEV_error_attribution_v1'
METRICS={
    'SeamCov':1,'ConflictCov_C':-1,'ExposedRetain':1,
    'D_sym2048':-1,'GT_to_F_MSE2048_L0':-1,'F_to_GT_MSE2048_L0':-1,
    'fixed_query_MSE':-1,'fixed_query_p95_L':-1,'coverage_fraction':1,
    'CF_proper_affected_free_fraction':-1,'FF_illegal_affected_free_fraction':-1,
    'free_minimum_angle_lt5_fraction':-1,'free_near_degenerate_fraction':-1,'free_exact_zero_area_fraction':-1,
}
BOUNDS={'SeamCov':('SeamCov','SeamCov_upper'),
        'ConflictCov_C':('ConflictCov_C','ConflictCov_C_upper'),
        'ExposedRetain':('ExposedRetain_global_lower','ExposedRetain_global_upper')}
SHAPE=('D_sym2048','fixed_query_MSE','fixed_query_p95_L','FF_illegal_affected_free_fraction',
       'CF_proper_affected_free_fraction','free_minimum_angle_lt5_fraction')


def number(value):
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):return float(value)
    return None


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def percent(candidate,reference):
    if candidate is None or reference is None or reference==0:return None
    return 100*(candidate-reference)/reference


def delta_bounds(candidate,reference,metric):
    if metric not in BOUNDS:
        c,r=number(candidate.get(metric)),number(reference.get(metric))
        return (c-r,c-r) if c is not None and r is not None else (None,None)
    lower,upper=BOUNDS[metric]
    cl,cu,rl,ru=[number(x.get(k)) for x,k in ((candidate,lower),(candidate,upper),(reference,lower),(reference,upper))]
    if any(v is None for v in (cl,cu,rl,ru)):return None,None
    same=bool(candidate.get('geometry_sha256') and candidate.get('geometry_sha256')==reference.get('geometry_sha256') and
        candidate['condition_id']==reference['condition_id'] and candidate.get('scoring_sources')==reference.get('scoring_sources'))
    return (0.,0.) if same else (cl-ru,cu-rl)


def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def compare(candidate_label,reference_label,role,cohorts,requests):
    candidate={r['request_id']:r for r in cohorts[candidate_label]};reference={r['request_id']:r for r in cohorts[reference_label]}
    prefix=dict(candidate_label=candidate_label,reference_label=reference_label,reference_role=role)
    outputs=[];parents=[];lopo=[];summaries=[];noise=[];tail=[]
    for metric,orientation in METRICS.items():
        current=[]
        for request in requests:
            rid=request['request_id'];c,r=candidate[rid],reference[rid];cv,rv=number(c.get(metric)),number(r.get(metric))
            d=cv-rv if cv is not None and rv is not None else None;lo,hi=delta_bounds(c,r,metric)
            value=dict(prefix,metric=metric,uid=request['uid'],candidate=request['candidate'],request_id=rid,
                candidate_value=cv,reference_value=rv,delta=d,favorable_gain=orientation*d if d is not None else None,
                delta_lower=lo,delta_upper=hi,favorable_lower=orientation*(lo if orientation>0 else hi) if lo is not None else None,
                favorable_upper=orientation*(hi if orientation>0 else lo) if hi is not None else None,
                relative_change_percent=percent(cv,rv),percentage_point_change=100*d if d is not None and metric in BOUNDS else None,
                reference_zero=(rv==0),candidate_metric_status=c.get(metric+'_status'),reference_metric_status=r.get(metric+'_status'))
            outputs.append(value);current.append(value)
        parent=[]
        for uid in sorted({r['uid'] for r in requests}):
            group=[r for r in current if r['uid']==uid]
            assert len(group)==2 and {r['candidate'] for r in group}=={0,1}
            value=dict(prefix,metric=metric,uid=uid,noise_count=2)
            for key in ('candidate_value','reference_value','delta','favorable_gain','delta_lower','delta_upper','favorable_lower','favorable_upper'):
                value[key]=mean(r[key] for r in group)
            value['relative_change_percent']=percent(value['candidate_value'],value['reference_value'])
            value['mean_gain_contribution']=value['favorable_gain']/8 if value['favorable_gain'] is not None else None
            value['mean_delta_contribution']=value['delta']/8 if value['delta'] is not None else None
            parent.append(value);parents.append(value)
            gains=[r['favorable_gain'] for r in sorted(group,key=lambda x:x['candidate'])]
            sign='MISSING' if any(v is None for v in gains) else 'OPPOSITE_SIGNS' if gains[0]*gains[1]<0 else \
                'BOTH_FAVORABLE' if min(gains)>0 else 'BOTH_ADVERSE' if max(gains)<0 else 'ZERO_OR_MIXED_WITH_ZERO'
            noise.append(dict(prefix,metric=metric,uid=uid,noise0_gain=gains[0],noise1_gain=gains[1],
                noise_sign_pattern=sign,parent_mean_gain=value['favorable_gain']))
        gains=[r['favorable_gain'] for r in parent];pointmean=mean(gains)
        lower=mean(r['favorable_lower'] for r in parent);upper=mean(r['favorable_upper'] for r in parent)
        full=all(v is not None for v in gains)
        positive=sum(max(0,v) for v in gains) if full else None
        for omitted in parent:
            kept=[r for r in parent if r['uid']!=omitted['uid']]
            lopo.append(dict(prefix,metric=metric,omitted_uid=omitted['uid'],retained_parents=7,
                full_parent_equal_gain=pointmean,leave_one_parent_out_gain=mean(r['favorable_gain'] for r in kept),
                leave_one_parent_out_gain_lower=mean(r['favorable_lower'] for r in kept),
                leave_one_parent_out_gain_upper=mean(r['favorable_upper'] for r in kept),
                omitted_parent_gain=omitted['favorable_gain'],omitted_parent_contribution=omitted['mean_gain_contribution']))
        for row in parent:
            row['share_of_net_gain_percent']=100*row['mean_gain_contribution']/pointmean if pointmean else None
            row['share_of_positive_parent_gains_percent']=100*max(0,row['favorable_gain'])/positive if positive else None
        best=max(parent,key=lambda r:r['favorable_gain']) if full else None
        worst=min(parent,key=lambda r:r['favorable_gain']) if full else None
        without_best=mean(r['favorable_gain'] for r in parent if best and r['uid']!=best['uid']) if best else None
        cv=mean(r['candidate_value'] for r in parent);rv=mean(r['reference_value'] for r in parent)
        current_lopo=[r for r in lopo if r['metric']==metric]
        summary=dict(prefix,metric=metric,higher_is_better=orientation>0,parent_count=8,noise_count=2,
            candidate_parent_equal_mean=cv,reference_parent_equal_mean=rv,delta=mean(r['delta'] for r in parent),
            favorable_gain=pointmean,favorable_lower=lower,favorable_upper=upper,
            ratio_of_means_change_percent=percent(cv,rv),reference_mean_zero=rv==0,
            improved_parents=sum(v>0 for v in gains) if full else None,
            adverse_parents=sum(v<0 for v in gains) if full else None,
            unchanged_parents=sum(v==0 for v in gains) if full else None,
            favorable_parent_median=statistics.median(gains) if full else None,
            most_favorable_parent=best['uid'] if best else None,most_favorable_parent_gain=best['favorable_gain'] if best else None,
            most_favorable_share_of_net_gain_percent=best['share_of_net_gain_percent'] if best else None,
            most_favorable_share_of_positive_gains_percent=best['share_of_positive_parent_gains_percent'] if best else None,
            gain_without_most_favorable_parent=without_best,
            most_adverse_parent=worst['uid'] if worst else None,most_adverse_parent_gain=worst['favorable_gain'] if worst else None,
            leave_one_out_gain_min=min((r['leave_one_parent_out_gain'] for r in current_lopo),default=None) if full else None,
            leave_one_out_gain_max=max((r['leave_one_parent_out_gain'] for r in current_lopo),default=None) if full else None,
            leave_one_out_positive_count=sum(r['leave_one_parent_out_gain']>0 for r in current_lopo) if full else None,
            opposite_noise_parent_count=sum(r['noise_sign_pattern']=='OPPOSITE_SIGNS' for r in noise if r['metric']==metric))
        for index in (0,1):summary['noise'+str(index)+'_mean_favorable_gain']=mean(r['favorable_gain'] for r in current if r['candidate']==index)
        summaries.append(summary)
        if metric in SHAPE:
            ordered=sorted(current,key=lambda r:float('-inf') if r['delta'] is None else r['delta'],reverse=True)
            for rank,r in enumerate(ordered,1):
                tail.append(dict(r,rank_by_absolute_worsening=rank,
                    definition='Largest candidate minus reference first; no candidate/parent removed from main means'))
    return dict(summary=summaries,per_output=outputs,per_parent=parents,leave_one_out=lopo,noise_discordance=noise,shape_tail=tail)


def run(out=OUT,*,matched_map=None):
    out=assets.resolve(out);requests=assets.read_json(out/'manifests/DEV.json')['rows']
    assert len(requests)==16 and len({r['uid'] for r in requests})==8
    assert all(r['split']=='DEV' for r in requests)
    expected={r['request_id'] for r in requests};cohorts={};sources={};incomplete=[]
    for path in sorted((out/'evaluation').glob('*/DEV/per_output.json')):
        label=path.parents[1].name;rows=assets.read_json(path)
        if {r['request_id'] for r in rows}!=expected or len(rows)!=16 or any(r.get('status')!='PASS' for r in rows):
            incomplete.append(dict(label=label,status='INCOMPLETE',rows=len(rows)));continue
        cohorts[label]=rows;sources[label]=dict(path=str(path),sha256=assets.digest(path))
    if 'START' not in cohorts:raise ValueError('Complete new START DEV cohort required')
    mapping=assets.read_json(assets.resolve(matched_map)) if matched_map else {}
    contrasts=[];unavailable=[]
    for label in sorted(cohorts):
        if label=='START':continue
        contrasts.append((label,'START','START'))
        parsed=re.fullmatch(r'(explore|repeat)_(.+)_(500|1000)',label)
        if not parsed:
            unavailable.append(dict(candidate_label=label,reason='Unknown registered reference role; not guessed'));continue
        phase,arm,step=parsed.groups()
        if arm!='FM_CONTROL':
            baseline=f'{phase}_FM_CONTROL_{step}'
            if baseline in cohorts:contrasts.append((label,baseline,'FM_BASELINE_UNMATCHED_AUX_BUDGET'))
            else:unavailable.append(dict(candidate_label=label,reference_label=baseline,role='FM_BASELINE',status='NA_NOT_COMPLETE'))
        matched=mapping.get(label)
        if matched is None and arm=='TRAJECTORY_REPLAY':matched=f'{phase}_REPLAY_FM_CONTROL_{step}'
        if matched:
            if matched in cohorts:contrasts.append((label,matched,'MATCHED_CONTROL'))
            else:unavailable.append(dict(candidate_label=label,reference_label=matched,role='MATCHED_CONTROL',status='NA_NOT_COMPLETE'))
    tables={k:[] for k in ('summary','per_output','per_parent','leave_one_out','noise_discordance','shape_tail')}
    for candidate,reference,role in contrasts:
        result=compare(candidate,reference,role,cohorts,requests)
        for key in tables:tables[key].extend(result[key])
    identity=dict(version=VERSION,sources=sources,DEV_manifest_sha256=assets.digest(out/'manifests/DEV.json'),
        script_sha256=assets.digest(__file__),matched_map=mapping)
    snapshot=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    folder=out/'derived/attribution/DEV'/snapshot
    facts=[]
    for r in tables['summary']:
        if r['metric'] in BOUNDS:
            facts.append({k:r[k] for k in ('candidate_label','reference_label','reference_role','metric',
                'favorable_gain','favorable_lower','favorable_upper','improved_parents','adverse_parents',
                'favorable_parent_median','most_favorable_parent','most_favorable_share_of_net_gain_percent',
                'most_favorable_share_of_positive_gains_percent','gain_without_most_favorable_parent',
                'leave_one_out_gain_min','leave_one_out_gain_max','opposite_noise_parent_count')})
    files={}
    for key,rows in tables.items():
        p=folder/(key+'.csv');_csv(p,rows);files[key]=dict(path=str(p),sha256=assets.digest(p),rows=len(rows))
    report=dict(status='COMPLETE_FOR_AVAILABLE_DEV',version=VERSION,identity=identity,complete_labels=sorted(cohorts),
        incomplete_labels=incomplete,unavailable_references=unavailable,comparisons=len(contrasts),control_facts=facts,
        tables=files,summary=tables['summary'],new_model_calls=0,new_geometry_scores=0,CONFIRM_accessed=False,
        interpretation_rules=[
            'Descriptive attribution only. Original success gates and thresholds are unchanged.',
            'Equal weight across eight parents, after two-noise mean within each. No oracle candidate selection.',
            'Favorable gain reverses sign for I and shape errors, so positive always means improvement.',
            'Net-gain share can exceed 100% when other parents regress; positive-gain share uses only positive contributions.',
            'Leave-one-parent-out recomputes a seven-parent descriptive mean; it never replaces the registered eight-parent gate.',
            'E retains original point estimate while conservative global intervals remain explicit.',
            'FM baseline has different auxiliary compute from trajectory replay and is not labeled matched.',
            'Relative change is (candidate parent-equal mean/reference parent-equal mean - 1)*100; zero reference yields NA.',
            'Tail table sorts all sixteen output deltas; sorting does not change any cohort or selection rule.',
            'Median, both-noise differences, severe individual costs and concentration must accompany any mean improvement claim.'])
    _json(folder/'attribution.json',report)
    latest=dict(status=report['status'],snapshot=snapshot,path=str(folder/'attribution.json'),
        sha256=assets.digest(folder/'attribution.json'),tables=files,complete_labels=report['complete_labels'],
        unavailable_references=unavailable,script_sha256=identity['script_sha256'])
    _json(out/'derived/attribution/DEV/latest.json',latest)
    print(json.dumps(dict(status=report['status'],labels=report['complete_labels'],comparisons=len(contrasts),
        unavailable_references=unavailable,path=latest['path']),ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--matched-map')
    a=parser.parse_args();run(a.out,matched_map=a.matched_map)
