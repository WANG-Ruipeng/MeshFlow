"""Pre-registered parent-equal comparisons; no geometry/model computations.

A/B gains are versus matched control. Quality and I/E costs are checked versus
both START and matched control. B's improving quantity versus START is reported
separately and is never inferred from a matched-control gain. Bounds are combined
conservatively; only identical geometry under identical condition/reference may
cancel paired uncertainty exactly. NA inputs are not converted to perfect scores.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import statistics
from . import assets
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as quality

VERSION='chair_control_feedback_gates_v1'
GATE_DEFINITION=dict(
    version=VERSION,aggregation='Within each parent, arithmetic mean of its fixed noises; then equal weight over all 8 registered parents',
    uncertainty='Independent conservative intervals except identical same-condition geometry has exactly zero paired delta; straddling thresholds is INCOMPLETE',
    applicability='Input-only fixed reference eligibility; no model-dependent dropping; LOW_SUPPORT retained',
    shape_quality=dict(references=['START','matched_control'],D_sym2048_ratio_max=1.05,
        fixed_query_MSE_ratio_max=1.05,fixed_query_p95_L_ratio_max=1.10,FF_illegal_affected_free_fraction_delta_max=.02,
        hard_contracts='C unchanged in 51 states and lossless NPZ/PLY export, N fixed, finite, same registered noise',
        visual='AI_VISUAL_REVIEW of every DEV output; no new obvious chair collapse; human usability NOT_RUN'),
    path_A=dict(S_mean_gain_vs_control_min=.03,S_median_gain_vs_control_strict_min=0,
        S_improved_parents_vs_control_min=6,S_mean_gain_vs_START_min=0,
        I_mean_cost_vs_both_max=.01,E_mean_drop_vs_both_max=.01),
    path_B=dict(S_mean_gain_vs_both_min=-.005,
        I_mean_reduction_vs_control_min=.02,E_mean_gain_vs_control_min=.03,
        selected_primary_median_gain_strict_min=0,selected_primary_improved_parents_min=6,
        other_quantity_mean_cost_vs_both_max=.01,
        main_gain_vs_START='Report separately; not required to equal matched-control gain'),
    severe_parent=dict(I_cost_or_E_drop_threshold=.10,references=['START','matched_control'],
        handling='Explicitly flag confirmed/possible severe parent cost; never describe as cost-free; no extra undocumented numeric veto'),
    noise='Report each registered candidate/noise separately; no best-candidate selection; flag opposite signs and severe cost',
    no_total_score=True,DEV_role='Repeated development evidence, not independent test',
    CONFIRM_role='Frozen new noises on already observed held-out parents, not unseen-parent generalization')

PRIMARY=('SeamCov','ConflictCov_C','ExposedRetain')
SCALARS=('D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L',
    'coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
    'free_minimum_angle_lt5_fraction','predicted_free_minimum_angle_lt5_fraction','predicted_free_area',
    'free_near_degenerate_fraction','free_exact_zero_area_fraction','free_area_over_GT_free_area',
    'free_total_surface_area','GT_free_total_surface_area','generation_seconds','SeamCov_0.005','SeamCov_0.02',
    'reference_visibility_uncertain_fraction','ExposedRetain_uncertain_fraction')
HARD=('N_preserved','C_output_bitwise','finite','C_all_steps_bitwise','initial_noise_bitwise','NPZ_C_bitwise','PLY_C_bitwise')


def number(value):
    if value=='INF':return math.inf
    if value=='-INF':return -math.inf
    if isinstance(value,(int,float)) and not isinstance(value,bool) and not math.isnan(value):return float(value)
    return None


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values and all(x is not None for x in values) else None


def interval(row,key):
    if not row or row.get('status')!='PASS':return (None,None)
    names={'SeamCov':('SeamCov','SeamCov_upper'),'ConflictCov_C':('ConflictCov_C','ConflictCov_C_upper'),
           'ExposedRetain':('ExposedRetain_global_lower','ExposedRetain_global_upper')}
    return tuple(number(row.get(k)) for k in names[key])


def summarize(rows,requests):
    by={r['request_id']:r for r in rows};parents=[];noise=[]
    for uid in sorted({r['uid'] for r in requests}):
        expected=[r for r in requests if r['uid']==uid];group=[by.get(r['request_id'],{}) for r in expected]
        p=dict(uid=uid,expected_outputs=len(expected),completed=sum(r.get('status')=='PASS' for r in group),
               candidates=[r['candidate'] for r in expected])
        for key in PRIMARY:
            bounds=[interval(r,key) for r in group]
            p[key]=mean(number(r.get(key)) for r in group)
            p[key+'_lower']=mean(v[0] for v in bounds);p[key+'_upper']=mean(v[1] for v in bounds)
            p[key+'_statuses']=[r.get(key+'_status','NOT_RUN') for r in group]
        for key in SCALARS:p[key]=mean(number(r.get(key)) for r in group)
        parents.append(p)
    for candidate in sorted({r['candidate'] for r in requests}):
        subset=[r for r in requests if r['candidate']==candidate];group=[by.get(r['request_id'],{}) for r in subset]
        n=dict(candidate=candidate,expected_parents=len(subset),completed=sum(r.get('status')=='PASS' for r in group))
        for key in PRIMARY+SCALARS:n[key]=mean(number(r.get(key)) for r in group)
        noise.append(n)
    metrics={key:mean(number(p.get(key)) for p in parents) for key in PRIMARY+SCALARS}
    for key in PRIMARY:
        for suffix in ('_lower','_upper'):metrics[key+suffix]=mean(p[key+suffix] for p in parents)
    complete=all(by.get(r['request_id'],{}).get('status')=='PASS' for r in requests)
    return dict(status='COMPLETE' if complete else 'INCOMPLETE',expected_outputs=len(requests),
        completed=sum(by.get(r['request_id'],{}).get('status')=='PASS' for r in requests),
        expected_parents=len(parents),parents=parents,noise=noise,metrics=metrics,
        statuses={status:sum(r.get('status')==status for r in rows) for status in sorted({r.get('status','UNKNOWN') for r in rows})},
        denominator_policy='Registered parents/noises retained. Missing numeric cells produce NA, never a smaller conditional mean.')


def _and(checks):
    states=[x if isinstance(x,str) else x['status'] for x in checks]
    return 'FAIL' if 'FAIL' in states else 'INCOMPLETE' if not states or 'INCOMPLETE' in states else 'PASS'


def _or(checks):
    states=[x if isinstance(x,str) else x['status'] for x in checks]
    return 'PASS' if 'PASS' in states else 'FAIL' if states and all(s=='FAIL' for s in states) else 'INCOMPLETE'


def threshold(low,high,bound,*,direction='ge',strict=False):
    low,high=number(low),number(high)
    if low is None or high is None:return dict(status='INCOMPLETE',lower=low,upper=high,threshold=bound,direction=direction,strict=strict)
    if low>high:raise ValueError('Reversed conservative bounds')
    if direction=='ge':
        good=(low>bound if strict else low>=bound);bad=(high<=bound if strict else high<bound)
    else:
        good=(high<bound if strict else high<=bound);bad=(low>=bound if strict else low>bound)
    return dict(status='PASS' if good else 'FAIL' if bad else 'INCOMPLETE',lower=low,upper=high,
                threshold=bound,direction=direction,strict=strict)


def _hard(rows):
    checks=[]
    for row in rows:
        values=[row.get(k) for k in HARD]
        status='FAIL' if row.get('status') in ('INVALID_OUTPUT','GENERATION_FAILED') or False in values else \
            'PASS' if row.get('status')=='PASS' and all(v is True for v in values) else 'INCOMPLETE'
        checks.append(dict(request_id=row.get('request_id'),status=status,facts={k:row.get(k) for k in HARD}))
    return dict(status=_and(checks),outputs=checks)


def paired(candidate,reference,requests):
    c={r['request_id']:r for r in candidate};r={r['request_id']:r for r in reference};outputs=[];parents=[]
    for req in requests:
        a,b=c.get(req['request_id'],{}),r.get(req['request_id'],{})
        out=dict(request_id=req['request_id'],uid=req['uid'],candidate=req['candidate'])
        exact=bool(a.get('geometry_sha256') and a.get('geometry_sha256')==b.get('geometry_sha256') and
                   a.get('condition_id')==b.get('condition_id') and a.get('scoring_sources')==b.get('scoring_sources'))
        out['identical_geometry_uncertainty_cancels']=exact
        for key in PRIMARY:
            ca,cb=interval(a,key),interval(b,key)
            valid=all(v is not None for v in ca+cb)
            out[key+'_delta']=None if number(a.get(key)) is None or number(b.get(key)) is None else number(a[key])-number(b[key])
            out[key+'_delta_lower']=0. if exact and valid else ca[0]-cb[1] if valid else None
            out[key+'_delta_upper']=0. if exact and valid else ca[1]-cb[0] if valid else None
        for key in SCALARS:
            x,y=number(a.get(key)),number(b.get(key));out[key+'_delta']=x-y if x is not None and y is not None else None
        outputs.append(out)
    for uid in sorted({q['uid'] for q in requests}):
        group=[q for q in outputs if q['uid']==uid];p=dict(uid=uid,noise_count=len(group))
        for key in PRIMARY:
            for suffix in ('_delta','_delta_lower','_delta_upper'):p[key+suffix]=mean(g[key+suffix] for g in group)
        for key in SCALARS:p[key+'_delta']=mean(g[key+'_delta'] for g in group)
        parents.append(p)
    stats={}
    for key in PRIMARY:
        lo=[p[key+'_delta_lower'] for p in parents];hi=[p[key+'_delta_upper'] for p in parents]
        gainlo,gainhi=(lo,hi) if key!='ConflictCov_C' else ([-x if x is not None else None for x in hi],[-x if x is not None else None for x in lo])
        valid=all(v is not None for v in lo+hi)
        stats[key]=dict(delta=mean(p[key+'_delta'] for p in parents),lower=mean(lo),upper=mean(hi),
            favorable_median_lower=statistics.median(gainlo) if valid else None,
            favorable_median_upper=statistics.median(gainhi) if valid else None,
            improved_parents_lower=sum(v>0 for v in gainlo) if valid else None,
            improved_parents_upper=sum(v>0 for v in gainhi) if valid else None)
    noise=[]
    for index in sorted({q['candidate'] for q in outputs}):
        group=[q for q in outputs if q['candidate']==index];n=dict(candidate=index,parents=len(group))
        for key in PRIMARY:
            for suffix in ('_delta','_delta_lower','_delta_upper'):n[key+suffix]=mean(g[key+suffix] for g in group)
        noise.append(n)
    severe=[]
    for p in parents:
        i=threshold(p['ConflictCov_C_delta_lower'],p['ConflictCov_C_delta_upper'],.10,direction='le')
        e=threshold(p['ExposedRetain_delta_lower'],p['ExposedRetain_delta_upper'],-.10)
        if i['status']!='PASS' or e['status']!='PASS':severe.append(dict(uid=p['uid'],I_cost=i,E_retention=e))
    return dict(stats=stats,parents=parents,outputs=outputs,noise=noise,severe_parent_costs=severe)


def _growth_gate(csummary,rsummary):
    checks={}
    for key,ratio in [('D_sym2048',1.05),('fixed_query_MSE',1.05),('fixed_query_p95_L',1.10)]:
        c,r=number(csummary['metrics'].get(key)),number(rsummary['metrics'].get(key))
        growth=None if c is None or r is None else (c/r if r else 1. if c==0 else math.inf)
        checks[key]=dict(threshold(growth,growth,ratio,direction='le'),candidate_mean=c,reference_mean=r)
    key='FF_illegal_affected_free_fraction';c,r=number(csummary['metrics'].get(key)),number(rsummary['metrics'].get(key))
    delta=c-r if c is not None and r is not None else None
    checks[key]=dict(threshold(delta,delta,.02,direction='le'),candidate_mean=c,reference_mean=r)
    return dict(status=_and(checks.values()),checks=checks)


def compare_rows(candidate,matched,start,requests,*,visual_review=None):
    """Pure numeric comparison; no filesystem access or additional scoring."""
    cs,ms,ss=[summarize(rows,requests) for rows in (candidate,matched,start)]
    controls=paired(candidate,matched,requests);baseline=paired(candidate,start,requests)
    quality_checks=dict(vs_matched=_growth_gate(cs,ms),vs_START=_growth_gate(cs,ss),
        candidate_contract=_hard(candidate),matched_contract=_hard(matched),START_contract=_hard(start))
    complete='PASS' if all(x['status']=='COMPLETE' for x in (cs,ms,ss)) and len(cs['parents'])==8 else 'INCOMPLETE'
    shape_status=_and([complete]+[x['status'] for x in quality_checks.values()])
    def delta(pair,key,threshold_value,direction='ge'):
        s=pair['stats'][key];return threshold(s['lower'],s['upper'],threshold_value,direction=direction)
    def support(pair,key):
        s=pair['stats'][key]
        return dict(median=threshold(s['favorable_median_lower'],s['favorable_median_upper'],0,strict=True),
                    parents=threshold(s['improved_parents_lower'],s['improved_parents_upper'],6))
    ac=dict(S_gain=delta(controls,'SeamCov',.03),S_START=delta(baseline,'SeamCov',0),**support(controls,'SeamCov'))
    for name,p in [('control',controls),('START',baseline)]:
        ac['I_cost_'+name]=delta(p,'ConflictCov_C',.01,'le');ac['E_cost_'+name]=delta(p,'ExposedRetain',-.01)
    path_A=dict(status=_and(ac.values()),checks=ac)
    common=dict(S_control=delta(controls,'SeamCov',-.005),S_START=delta(baseline,'SeamCov',-.005))
    bi=dict(common,I_gain=delta(controls,'ConflictCov_C',-.02,'le'),**support(controls,'ConflictCov_C'))
    be=dict(common,E_gain=delta(controls,'ExposedRetain',.03),**support(controls,'ExposedRetain'))
    for name,p in [('control',controls),('START',baseline)]:
        bi['E_cost_'+name]=delta(p,'ExposedRetain',-.01)
        be['I_cost_'+name]=delta(p,'ConflictCov_C',.01,'le')
    branch_I=dict(status=_and(bi.values()),checks=bi);branch_E=dict(status=_and(be.values()),checks=be)
    path_B=dict(status=_or([branch_I,branch_E]),I_improvement=branch_I,E_improvement=branch_E)
    numeric_status=_and([shape_status,_or([path_A,path_B])])
    review=visual_review or dict(status='NOT_RUN',reviewer_type='AI_VISUAL_REVIEW')
    visual='PASS' if review.get('status')=='PASS' and review.get('reviewer_type')=='AI_VISUAL_REVIEW' else \
           'FAIL' if review.get('status')=='FAIL' else 'INCOMPLETE'
    return dict(version=VERSION,status=_and([numeric_status,visual]),numeric_status=numeric_status,
        shape_quality_status=shape_status,shape_quality=quality_checks,path_A=path_A,path_B=path_B,
        visual_review=review,vs_matched=controls,vs_START=baseline,candidate_summary=cs,
        matched_summary=ms,START_summary=ss,definition=GATE_DEFINITION,human_usable=None,user_study='NOT_RUN',
        interpretation='This is development or new-noise held-out-parent evidence only. No automatic model/default replacement. Primary gains versus START are separately available in vs_START.',
        severe_cost_caveat=bool(controls['severe_parent_costs'] or baseline['severe_parent_costs']))


def compare_sets(out,candidate_label,matched_label,*,pool='DEV',start_label='START',visual_review=None):
    out=assets.resolve(out);manifest=assets.read_json(out/'manifests'/(pool+'.json'));requests=manifest['rows']
    def read(label):return assets.read_json(out/'evaluation'/label/pool/'per_output.json')
    if isinstance(visual_review,(str,Path)):visual_review=assets.read_json(assets.resolve(visual_review))
    result=compare_rows(read(candidate_label),read(matched_label),read(start_label),requests,visual_review=visual_review)
    result.update(pool=pool,candidate_label=candidate_label,matched_label=matched_label,start_label=start_label)
    folder=out/'comparisons'/pool/(candidate_label+'__vs__'+matched_label)
    quality.write_json(folder/'gate.json',result)
    for key in ('vs_matched','vs_START'):
        quality.write_csv(folder/(key+'_per_parent.csv'),result[key]['parents'])
        quality.write_csv(folder/(key+'_per_noise.csv'),result[key]['noise'])
        quality.write_csv(folder/(key+'_per_output.csv'),result[key]['outputs'])
    return result


def freeze_gates(out=assets.OUT):
    path=assets.resolve(out)/'registration/evaluation_gate_definition.json'
    assets.freeze_json(path,GATE_DEFINITION)
    return dict(status='PASS',path=str(path),sha256=assets.digest(path),source_sha256=assets.digest(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(assets.OUT));p.add_argument('--freeze',action='store_true')
    p.add_argument('--candidate');p.add_argument('--matched');p.add_argument('--pool',default='DEV');p.add_argument('--visual-review');a=p.parse_args()
    result=freeze_gates(a.out) if a.freeze else compare_sets(a.out,a.candidate,a.matched,pool=a.pool,visual_review=a.visual_review)
    print(json.dumps(quality.clean(result),ensure_ascii=False))
