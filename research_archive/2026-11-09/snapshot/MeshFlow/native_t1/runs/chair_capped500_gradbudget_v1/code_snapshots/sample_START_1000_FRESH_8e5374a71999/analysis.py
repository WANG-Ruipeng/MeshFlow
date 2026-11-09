"""Frozen OLD/FRESH research screens and unchanged historical joint gates.

All comparisons retain eight parents and registered noises. A/B below are this
run's distinct research questions, not the historical control A/B branches.
No new metric, best-noise selection, exclusion, or inferential p-value is added.
"""
from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path
from . import evaluation as ev

gates=ev.gates
VERSION='chair_capped500_gradbudget_analysis_v1'
METRICS=tuple(dict.fromkeys(gates.PRIMARY+gates.SCALARS+('ConflictCov_C_cross','ConflictCov_C_dup',
    'ConflictCov_C_cross_and_dup','free_faces','N','K')))
HIGHER={'SeamCov','SeamCov_0.005','SeamCov_0.02','ExposedRetain','coverage_fraction'}
LOWER={'ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','D_sym2048','GT_to_F_MSE2048_L0',
    'F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L','CF_proper_affected_free_fraction',
    'FF_illegal_affected_free_fraction','free_near_degenerate_fraction','free_exact_zero_area_fraction',
    'free_minimum_angle_lt5_fraction','predicted_free_minimum_angle_lt5_fraction'}
DEFINITION=dict(version=VERSION,aggregation='8 equal parents after their 2 OLD or 2 FRESH noises; ALL has all 4, never substitutes for separate panels',
    question_A=dict(candidate='L500',reference='T500',OLD=dict(S_mean_min=.03,median_strict_min=0,improved_parents_min=6),
        FRESH=dict(S_mean_strict_min=0,median_strict_min=0,improved_parents_min=5),also_report='L500 minus START',
        interpretation='Development replication screen, no significance claim; only OLD passing is OLD_PANEL_ONLY'),
    question_B=dict(candidate='G500',reference='L500',S_mean_min=-.01,GT_MSE512_ratio_max=.95,
        GT_MSE2048_ratio_max=.95,GT_MSE2048_improved_parents_min=5,I_mean_max=.01,E_mean_min=-.01,Dsym_ratio_max=1.05,
        separate_required='Report G500 minus T500 S distribution; quality protection alone cannot establish preserved learned benefit'),
    question_C='For T/L/G,1000-500; DID=(G1000-G500)-(L1000-L500). Describe all signs; if L does not regress, do not claim prevention.',
    uncertainty='Original S/I bounds and E global bounds; NA stays incomplete. Identical same-condition same-formula geometry cancels delta uncertainty.',
    original_joint_gate=gates.GATE_DEFINITION,LOPO='Descriptive leave-one-parent-out only; original gate retains all eight parents',
    human_usable=None,user_study='NOT_RUN',default_model_changed=False)


def finite(value):
    value=gates.number(value)
    return value if value is not None and not math.isnan(value) else None


def avg(values):
    values=list(values)
    if not values or any(v is None for v in values):return None
    result=sum(values)/len(values)
    return None if math.isnan(result) else result


def med(values):
    values=list(values)
    return statistics.median(values) if values and all(v is not None for v in values) else None


def _rows(out,label,panel):
    if panel=='ALL':return _rows(out,label,'OLD')+_rows(out,label,'FRESH')
    path=Path(out)/'evaluation'/label/panel/'per_output.json'
    saved={r['request_id']:r for r in ev.read_json(path)} if path.exists() else {}
    requests=ev.manifest(out,panel)
    return [saved.get(q['request_id'],dict(request_id=q['request_id'],uid=q['uid'],candidate=q['candidate'],condition_id=q['condition_id'],
                label=label,panel=panel,split=panel,status='NOT_RUN')) for q in requests]


def interval(row,key):
    if key in gates.PRIMARY:return gates.interval(row,key)
    x=finite(row.get(key))
    return x,x


def compare(candidate,reference,requests,candidate_label,reference_label,panel):
    c={r['request_id']:r for r in candidate};r={q['request_id']:q for q in reference}
    outputs=[];parents=[];summaries=[];noises=[];lopo=[]
    prefix=dict(panel=panel,candidate_label=candidate_label,reference_label=reference_label)
    for metric in METRICS:
        direction=1 if metric in HIGHER else -1 if metric in LOWER else 0
        one=[]
        for request in requests:
            rid=request['request_id'];a,b=c[rid],r[rid];av,bv=finite(a.get(metric)),finite(b.get(metric))
            delta=av-bv if av is not None and bv is not None and not (math.isinf(av) and math.isinf(bv)) else None
            al,au=interval(a,metric);bl,bu=interval(b,metric);valid=all(v is not None for v in (al,au,bl,bu))
            same=bool(a.get('geometry_sha256') and a.get('geometry_sha256')==b.get('geometry_sha256') and
                a.get('condition_id')==b.get('condition_id') and a.get('scoring_sources')==b.get('scoring_sources'))
            low=0. if same and valid else al-bu if valid else None
            high=0. if same and valid else au-bl if valid else None
            row=dict(prefix,metric=metric,uid=request['uid'],request_id=rid,candidate=request['candidate'],
                candidate_value=av,reference_value=bv,delta=delta,lower=finite(low),upper=finite(high),
                favorable_gain=direction*delta if delta is not None and direction else None,
                candidate_status=a.get('status'),reference_status=b.get('status'),identical_geometry_uncertainty_cancels=same)
            one.append(row);outputs.append(row)
        pp=[]
        for uid in sorted({q['uid'] for q in requests}):
            group=[x for x in one if x['uid']==uid]
            p=dict(prefix,metric=metric,uid=uid,registered_noises=len(group),
                **{k:avg(x[k] for x in group) for k in ('candidate_value','reference_value','delta','lower','upper','favorable_gain')})
            p['noise_deltas']={str(x['candidate']):x['delta'] for x in group}
            valid=[x['delta'] for x in group]
            p['opposite_noise_signs']=any(v is not None and v>0 for v in valid) and any(v is not None and v<0 for v in valid)
            p['all_noises_improve']=all(x['favorable_gain'] is not None and x['favorable_gain']>0 for x in group) if direction else None
            p['contribution_to_mean_delta']=p['delta']/8 if p['delta'] is not None else None
            if metric=='ConflictCov_C':p['original_severe_parent_cost_confirmed']=p['lower']>.10 if p['lower'] is not None else None
            if metric=='ExposedRetain':p['original_severe_parent_cost_confirmed']=p['upper']<-.10 if p['upper'] is not None else None
            pp.append(p);parents.append(p)
        d=avg(p['delta'] for p in pp)
        for p in pp:
            p['net_delta_share_percent']=100*p['delta']/(8*d) if d not in (None,0) and p['delta'] is not None else None
            other=[q for q in pp if q['uid']!=p['uid']]
            lopo.append(dict(prefix,metric=metric,omitted_uid=p['uid'],remaining_parents=7,
                delta=avg(q['delta'] for q in other),lower=avg(q['lower'] for q in other),upper=avg(q['upper'] for q in other),
                original_full_delta=d,gate_recomputed=False))
        cm=avg(p['candidate_value'] for p in pp);rm=avg(p['reference_value'] for p in pp)
        complete=all(p['delta'] is not None for p in pp)
        summaries.append(dict(prefix,metric=metric,parents=8,outputs=len(requests),candidate_mean=cm,reference_mean=rm,
            delta=d,lower=avg(p['lower'] for p in pp),upper=avg(p['upper'] for p in pp),median_delta=med(p['delta'] for p in pp),
            percent_change=100*(cm-rm)/rm if cm is not None and rm not in (None,0) else None,
            improved_parents=sum(p['favorable_gain']>0 for p in pp) if direction and complete else None,
            adverse_parents=sum(p['favorable_gain']<0 for p in pp) if direction and complete else None,
            opposite_noise_parents=sum(p['opposite_noise_signs'] for p in pp),
            all_noises_improve_parents=sum(p['all_noises_improve'] for p in pp) if direction and complete else None,
            status='COMPLETE' if complete else 'INCOMPLETE'))
        for noise in sorted({q['candidate'] for q in requests}):
            group=[x for x in one if x['candidate']==noise]
            noises.append(dict(prefix,metric=metric,candidate=noise,parents=len(group),
                **{k:avg(x[k] for x in group) for k in ('candidate_value','reference_value','delta','lower','upper','favorable_gain')}))
    return dict(per_output=outputs,per_parent=parents,summary=summaries,per_noise=noises,leave_one_parent_out=lopo)


def _replication(candidate,teacher,requests,panel):
    p=gates.paired(candidate,teacher,requests);s=p['stats']['SeamCov']
    old=panel=='OLD'
    checks=dict(mean=gates.threshold(s['lower'],s['upper'],.03 if old else 0.,strict=not old),
        median=gates.threshold(s['favorable_median_lower'],s['favorable_median_upper'],0,strict=True),
        parents=gates.threshold(s['improved_parents_lower'],s['improved_parents_upper'],6 if old else 5))
    return dict(status=gates._and(checks.values()),checks=checks,S=s,per_noise=p['noise'],
        scope='New pre-registered development screen; not historical joint promotion or significance')


def research_screens(out,panel):
    requests=ev.manifest(out,panel)
    start,t,l,g=[_rows(out,label,panel) for label in ('START','T500','L500','G500')]
    A=_replication(l,t,requests,panel);A['L500_vs_START']=gates.paired(l,start,requests)
    pair=gates.paired(g,l,requests);gs,ls=[gates.summarize(x,requests) for x in (g,l)]
    checks={}
    for key,bound,direction in [('SeamCov',-.01,'ge'),('ConflictCov_C',.01,'le'),('ExposedRetain',-.01,'ge')]:
        s=pair['stats'][key];checks[key]=gates.threshold(s['lower'],s['upper'],bound,direction=direction)
    for key,limit in [('fixed_query_MSE',.95),('GT_to_F_MSE2048_L0',.95),('D_sym2048',1.05)]:
        cv,rv=[finite(s['metrics'].get(key)) for s in (gs,ls)]
        ratio=cv/rv if cv is not None and rv not in (None,0) else 1. if cv==rv==0 else math.inf if cv is not None and rv==0 else None
        checks[key]=dict(gates.threshold(ratio,ratio,limit,direction='le'),candidate_mean=cv,reference_mean=rv)
    deltas=[p['GT_to_F_MSE2048_L0_delta'] for p in pair['parents']]
    improved=sum(v<0 for v in deltas) if all(v is not None for v in deltas) else None
    checks['GT2048_improved_parents']=gates.threshold(improved,improved,5)
    g_teacher=gates.paired(g,t,requests);ss=g_teacher['stats']['SeamCov']
    B=dict(status=gates._and(checks.values()),quality_protection_conditions=checks,G500_vs_T500=g_teacher,
        G_vs_T_positive_S_mean=gates.threshold(ss['lower'],ss['upper'],0,strict=True),
        interpretation='B numerical conditions alone do not establish retained learning; report G-versus-T S distribution separately, no automatic promotion')
    return dict(panel=panel,question_A=A,question_B=B)


def _review(out,candidate,matched,panel):
    p=Path(out)/'visuals/reviews'/f'{candidate}_vs_{matched}__{panel}.json'
    if not p.exists():return None
    review=ev.read_json(p)
    if review.get('candidate_label')!=candidate or review.get('matched_label')!=matched or review.get('panel',review.get('pool'))!=panel:
        raise ValueError('Visual review identity mismatch')
    if review.get('scope_status')!='COMPLETE' or set(review.get('request_ids',[]))!={r['request_id'] for r in ev.manifest(out,panel)}:
        return dict(review,status='NOT_RUN',scope_reason='Full registered panel review not established')
    return review


def difference_in_differences(out,panel):
    requests=ev.manifest(out,panel);cohorts={lab:{r['request_id']:r for r in _rows(out,lab,panel)} for lab in ('G500','G1000','L500','L1000')}
    outputs=[];parents=[];summary=[];noises=[]
    for metric in METRICS:
        one=[]
        for q in requests:
            r={label:cohorts[label][q['request_id']] for label in cohorts};values={label:finite(row.get(metric)) for label,row in r.items()}
            complete=all(v is not None for v in values.values())
            delta=(values['G1000']-values['G500'])-(values['L1000']-values['L500']) if complete else None
            lower=upper=0.;valid=True
            for label,weight in [('G1000',1),('G500',-1),('L1000',-1),('L500',1)]:
                low,high=interval(r[label],metric)
                if low is None or high is None:valid=False;break
                lower+=weight*(low if weight>0 else high);upper+=weight*(high if weight>0 else low)
            row=dict(panel=panel,metric=metric,uid=q['uid'],request_id=q['request_id'],candidate=q['candidate'],
                G_drift=values['G1000']-values['G500'] if complete else None,L_drift=values['L1000']-values['L500'] if complete else None,
                DID=finite(delta),lower=finite(lower) if valid else None,upper=finite(upper) if valid else None)
            outputs.append(row);one.append(row)
        for noise in sorted({q['candidate'] for q in requests}):
            group=[r for r in one if r['candidate']==noise]
            noises.append(dict(panel=panel,metric=metric,candidate=noise,parents=len(group),**{k:avg(r[k] for r in group) for k in ('G_drift','L_drift','DID','lower','upper')}))
        pp=[]
        for uid in sorted({q['uid'] for q in requests}):
            group=[r for r in one if r['uid']==uid]
            row=dict(panel=panel,metric=metric,uid=uid,**{k:avg(r[k] for r in group) for k in ('G_drift','L_drift','DID','lower','upper')})
            pp.append(row);parents.append(row)
        summary.append(dict(panel=panel,metric=metric,parents=8,**{k:avg(r[k] for r in pp) for k in ('G_drift','L_drift','DID','lower','upper')}))
    return dict(per_output=outputs,per_parent=parents,summary=summary,per_noise=noises)


def freeze(out=ev.OUT):
    out=ev.assert_out(out)
    ev.freeze_json(out/'registration/evaluation_gate_definition.json',DEFINITION)
    return dict(status='PASS',source_sha256=ev.digest(__file__),definition=DEFINITION)


def run(out=ev.OUT):
    out=ev.assert_out(out);freeze(out)
    contrasts=[('HIST_CAPPED500','HIST_TEACHER500'),('HIST_CAPPED500','START'),('HIST_TEACHER500','START'),
        ('T500','START'),('T1000','START'),('T500','HIST_TEACHER500'),
        ('L500','T500'),('L500','START'),('L500','HIST_CAPPED500'),('G500','L500'),('G500','T500'),('G500','START'),
        ('L1000','T1000'),('L1000','START'),('G1000','L1000'),('G1000','T1000'),('G1000','START'),
        ('T1000','T500'),('L1000','L500'),('G1000','G500')]
    all_tables={key:[] for key in ('per_output','per_parent','summary','per_noise','leave_one_parent_out')}
    did_tables={key:[] for key in ('per_output','per_parent','summary','per_noise')};profiles=[];screens={};joint=[];identities={}
    for panel in (*ev.PANELS,'ALL'):
        requests=ev.manifest(out,panel);cohorts={label:_rows(out,label,panel) for label in ev.LABELS}
        for label,rows in cohorts.items():
            summary=gates.summarize(rows,requests)
            profiles.append(dict(label=label,panel=panel,status=summary['status'],completed=summary['completed'],expected_outputs=len(requests),
                mean_registered_N=avg(q['N'] for q in requests),mean_registered_K=avg(q['K'] for q in requests),
                mean_registered_free_faces=avg(q['N']-q['K'] for q in requests),**summary['metrics']))
            if panel!='ALL':
                path=out/'evaluation'/label/panel/'per_output.json'
                if path.exists():identities[str(path)]=ev.digest(path)
        for candidate,reference in contrasts:
            data=compare(cohorts[candidate],cohorts[reference],requests,candidate,reference,panel)
            for key in all_tables:all_tables[key].extend(data[key])
        did=difference_in_differences(out,panel)
        for key in did_tables:did_tables[key].extend(did[key])
        if panel in ev.PANELS:
            screens[panel]=research_screens(out,panel)
            for candidate,matched in [('HIST_CAPPED500','HIST_TEACHER500'),('L500','T500'),('G500','T500'),('L1000','T1000'),('G1000','T1000')]:
                result=gates.compare_rows(cohorts[candidate],cohorts[matched],cohorts['START'],requests,
                                          visual_review=_review(out,candidate,matched,panel))
                result.update(panel=panel,candidate_label=candidate,matched_label=matched,start_label='START')
                path=out/'analysis/original_joint_gates'/panel/(candidate+'__vs__'+matched+'.json')
                ev.write_json(path,result)
                joint.append(dict(panel=panel,candidate_label=candidate,matched_label=matched,status=result['status'],numeric_status=result['numeric_status'],
                    shape_quality_status=result['shape_quality_status'],control_A_status=result['path_A']['status'],control_B_status=result['path_B']['status'],
                    severe_cost_caveat=result['severe_cost_caveat'],path=str(path)))
    folder=out/'analysis';paths={}
    for key,rows in all_tables.items():
        path=folder/(key+'.csv');ev.write_csv(path,rows);paths[key]=dict(path=str(path),sha256=ev.digest(path),rows=len(rows))
    for key,rows in did_tables.items():
        path=folder/('drift_difference_in_differences_'+key+'.csv');ev.write_csv(path,rows);paths['DID_'+key]=dict(path=str(path),sha256=ev.digest(path),rows=len(rows))
    ev.write_csv(folder/'checkpoint_profiles.csv',profiles);ev.write_csv(folder/'original_joint_gates.csv',joint)
    aold,afresh=[screens[p]['question_A']['status'] for p in ev.PANELS]
    A_status='PASS' if aold==afresh=='PASS' else 'OLD_PANEL_ONLY' if aold=='PASS' and afresh=='FAIL' else 'INCOMPLETE' if 'INCOMPLETE' in (aold,afresh) else 'FAIL'
    complete=all(p['status']=='COMPLETE' for p in profiles)
    result=dict(version=VERSION,status='COMPLETE' if complete else 'INCOMPLETE',question_A_status=A_status,
        question_B_status=gates._and([screens[p]['question_B']['status'] for p in ev.PANELS]),research_screens=screens,
        checkpoint_profiles=profiles,original_joint_gates=joint,tables=paths,source_files=identities,
        definition=DEFINITION,model_calls=0,metrics_recomputed=0,human_usable=None,user_study='NOT_RUN',default_model_changed=False)
    ev.write_json(folder/'results.json',result)
    return result


def self_test():
    """Bounded arithmetic tests only; no geometry or real model computation."""
    assert gates.threshold(.029,.031,.03)['status']=='INCOMPLETE'
    assert gates.threshold(0.,0.,0.,strict=True)['status']=='FAIL'
    assert gates.threshold(None,None,0)['status']=='INCOMPLETE'
    assert avg([1.,None]) is None and avg([1.,3.])==2.
    requests=[dict(request_id=f'p{i}n{n}',uid=f'p{i}',candidate=n) for i in range(8) for n in (0,1)]
    def cohort(gain):
        return [dict(q,status='PASS',SeamCov=.1+gain,SeamCov_upper=.1+gain,ConflictCov_C=.02,ConflictCov_C_upper=.02,
            ExposedRetain=.9,ExposedRetain_global_lower=.9,ExposedRetain_global_upper=.9) for q in requests]
    assert _replication(cohort(.04),cohort(0),requests,'OLD')['status']=='PASS'
    assert _replication(cohort(.01),cohort(0),requests,'OLD')['status']=='FAIL'
    assert _replication(cohort(.01),cohort(0),requests,'FRESH')['status']=='PASS'
    assert _replication(cohort(0),cohort(0),requests,'FRESH')['status']=='FAIL'
    missing=cohort(.04);missing.pop()
    assert _replication(missing,cohort(0),requests,'OLD')['status']=='INCOMPLETE'
    return dict(status='PASS',checks=9,model_calls=0,geometry_calls=0)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ev.OUT));p.add_argument('--freeze',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    value=self_test() if a.self_test else freeze(a.out) if a.freeze else run(a.out)
    print(json.dumps(ev.quality.clean(value),ensure_ascii=False))
