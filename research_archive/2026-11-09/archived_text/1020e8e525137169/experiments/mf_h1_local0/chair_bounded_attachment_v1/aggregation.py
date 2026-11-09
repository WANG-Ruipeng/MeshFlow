"""Preregistered eight-parent/two-noise arithmetic and conservative phase gates."""
from pathlib import Path
import argparse
import json
import math
import statistics
from .common import OUT,read_json,digest,atomic_json,freeze_json

VERSION='bounded_attachment_aggregation_v1'
METRICS=('SeamCov_strict','SeamCov_strict_upper','SeamCov','SeamCov_upper','SeamCov_0.005','SeamCov_0.02',
    'ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_GT','ConflictCov_C_minus_GT',
    'ExposedRetain','ExposedRetain_upper','ExposedRetain_global_lower','ExposedRetain_global_upper','reference_visibility_uncertain_fraction',
    'D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L',
    'FF_illegal_affected_free_fraction','CF_proper_affected_free_fraction','FF_illegal_pair_count','CF_proper_pair_count',
    'free_exact_zero_area_count','free_near_degenerate_count','free_minimum_angle_lt5_fraction','free_area_over_GT_free_area',
    'delta','delta_over_L0','RAW_max_corner_displacement','RAW_RMS_corner_displacement','changed_corner_fraction','changed_face_fraction',
    'accepted_count','maximum_constraint_residual','raw_generation_seconds','project_seconds','deployment_seconds')
BOUND_METRICS=('SeamCov_strict','SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE','fixed_query_p95_L')


def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def mean(values):return statistics.fmean(values)


def gate_definition():
    return dict(version=VERSION,parents=8,candidates_per_parent=[0,1],unit='mean the two fixed candidates per parent, then equal mean across eight parents',
        strict_SeamCov_tau=.001,strict_mean_gain_min=.02,strict_parent_median_gain_gt=0.,strict_improving_parents_min=4,
        old_SeamCov_tau=.01,old_SeamCov_mean_drop_max=.005,ConflictCov_C_mean_increase_max=.01,
        ExposedRetain_mean_drop_max=.01,every_parent_conflict_increase_or_exposure_drop_max=.05,
        D_sym2048_relative_mean_increase_max=.05,fixed_query_MSE_relative_mean_increase_max=.05,fixed_query_p95_relative_mean_increase_max=.10,
        contracts=['C','N','finite','budget','accepted_equations','no_new_degenerate','no_new_illegal_events'],
        endpoint_per_corner_displacement_max='delta=.01*L_C',trajectory_explicit_budget='sum norms Delta_i <=delta; not injected displacement',
        trajectory_final_RAW_drift_max='3*delta; additional phase2 requirement',
        uncertainty='Use worst/best conservative intervals. A straddled requirement is INCOMPLETE; never fill favorable values.',
        phase2_dispatch='Only all-nine phase1 PASS, never per-output selection or partial-cohort promotion',
        no_candidate_or_rejected='Retained in every denominator; no filtering by accepted_count',model_calls=0)


def freeze_definition(out=OUT):
    from . import metrics_adapter
    out=Path(out)
    freeze_json(out/'scoring/definition.json',metrics_adapter.definition())
    freeze_json(out/'phase_gate_definition.json',gate_definition())
    return dict(status='PASS',gate_definition_sha256=digest(out/'phase_gate_definition.json'),
        scoring_definition_sha256=digest(out/'scoring/definition.json'))


def bounds(row,metric):
    low=row.get(metric);high=low
    if metric in ('SeamCov_strict','SeamCov','ConflictCov_C'):high=row.get(metric+'_upper')
    if metric=='ExposedRetain':low,high=row.get('ExposedRetain_global_lower'),row.get('ExposedRetain_global_upper')
    return [float(low),float(high)] if number(low) and number(high) and low<=high else None


def parent_rows(requests,by_id):
    groups={}
    for request in requests:groups.setdefault((request['label'],request['uid']),[]).append(request)
    output=[]
    for (label,uid),jobs in sorted(groups.items()):
        observed=[by_id.get(r['request_id'],{}) for r in jobs]
        cohort_ok=len(jobs)==2 and {int(r['candidate']) for r in jobs}=={0,1}
        row=dict(label=label,uid=uid,registered=len(jobs),scored=sum(r.get('status')=='PASS' for r in observed),
            complete=cohort_ok and all(r.get('status')=='PASS' for r in observed))
        for metric in METRICS:
            vals=[r.get(metric) for r in observed]
            good=row['complete'] and all(number(v) for v in vals)
            row[metric]=mean(vals) if good else None
        for metric in BOUND_METRICS:
            vals=[bounds(r,metric) for r in observed]
            row[metric+'_bounds']=[mean(v[i] for v in vals) for i in range(2)] if row['complete'] and all(v is not None for v in vals) else None
        output.append(row)
    return output


def _interval_check(lower,upper,threshold,*,at_least=False,strict=False):
    if not number(lower) or not number(upper):return 'INCOMPLETE'
    if at_least:
        passed=lower>threshold if strict else lower>=threshold
        failed=upper<=threshold if strict else upper<threshold
    else:
        passed=upper<threshold if strict else upper<=threshold
        failed=lower>=threshold if strict else lower>threshold
    return 'PASS' if passed else 'FAIL' if failed else 'INCOMPLETE'


def phase_gate(requests,rows,parents,phase):
    treatment='ENDPOINT' if phase==1 else 'TRAJECTORY';uids=sorted({r['uid'] for r in requests if r['label']=='RAW'})
    index={(r['label'],r['uid']):r for r in parents};by_id={r['request_id']:r for r in rows}
    jobs=[r for r in requests if r['label'] in ('RAW',treatment)]
    ready=(len(uids)==8 and len(jobs)==32 and all(by_id.get(r['request_id'],{}).get('status')=='PASS' for r in jobs)
        and all(index.get((label,u),{}).get('complete') for label in ('RAW',treatment) for u in uids))
    checks=[]
    def add(identifier,name,lower=None,upper=None,threshold=None,*,at_least=False,strict=False,status=None,**details):
        checks.append(dict(id=identifier,name=name,status=status or _interval_check(lower,upper,threshold,at_least=at_least,strict=strict),
            lower=lower,upper=upper,threshold=threshold,direction='at_least' if at_least else 'at_most',strict=strict,**details))
    def differences(metric,*,loss=False):
        data=[]
        for uid in uids:
            t=index.get((treatment,uid),{}).get(metric+'_bounds');r=index.get(('RAW',uid),{}).get(metric+'_bounds')
            if t is None or r is None:return None
            low,high=(r[0]-t[1],r[1]-t[0]) if loss else (t[0]-r[1],t[1]-r[0])
            data.append(dict(uid=uid,lower=low,upper=high))
        return data
    def aggregate_difference(identifier,name,metric,threshold,*,loss=False,at_least=False):
        data=differences(metric,loss=loss) if ready else None
        low,high=(mean(x['lower'] for x in data),mean(x['upper'] for x in data)) if data else (None,None)
        add(identifier,name,low,high,threshold,at_least=at_least,parents=data or [])
        return data
    strict=aggregate_difference(1,'strict_SeamCov_mean_gain','SeamCov_strict',.02,at_least=True)
    if strict:
        ml,mh=statistics.median(x['lower'] for x in strict),statistics.median(x['upper'] for x in strict)
        nl,nh=sum(x['lower']>0 for x in strict),sum(x['upper']>0 for x in strict)
        median_status=_interval_check(ml,mh,0.,at_least=True,strict=True)
        count_status=_interval_check(nl,nh,4,at_least=True)
        status='INCOMPLETE' if 'INCOMPLETE' in (median_status,count_status) else 'PASS' if median_status==count_status=='PASS' else 'FAIL'
        add(2,'positive_parent_median_and_at_least_four_improve',status=status,median_lower=ml,median_upper=mh,
            improving_parents_lower=nl,improving_parents_upper=nh,median_status=median_status,count_status=count_status)
    else:add(2,'positive_parent_median_and_at_least_four_improve',status='INCOMPLETE')
    aggregate_difference(3,'old_SeamCov_mean_drop','SeamCov',.005,loss=True)
    conflict=aggregate_difference(4,'ConflictCov_C_mean_increase','ConflictCov_C',.01)
    exposure=aggregate_difference(5,'ExposedRetain_mean_drop','ExposedRetain',.01,loss=True)
    if conflict and exposure:
        costs=[dict(metric=m,**p) for m,data in [('ConflictCov_C',conflict),('ExposedRetain',exposure)] for p in data]
        add(6,'every_parent_conflict_or_exposure_cost',max(x['lower'] for x in costs),max(x['upper'] for x in costs),.05,
            parent_costs=costs,severe_certain=[x for x in costs if x['lower']>.05],severe_possible=[x for x in costs if x['upper']>.05])
    else:add(6,'every_parent_conflict_or_exposure_cost',status='INCOMPLETE')
    quality=[]
    for metric,limit in [('D_sym2048',.05),('fixed_query_MSE',.05),('fixed_query_p95_L',.10)]:
        values=[(index.get((treatment,u),{}).get(metric),index.get(('RAW',u),{}).get(metric)) for u in uids]
        if not ready or not all(number(a) and number(b) for a,b in values):
            item=dict(metric=metric,status='INCOMPLETE',relative_change=None,limit=limit)
        else:
            t,r=mean(a for a,b in values),mean(b for a,b in values)
            change=t/r-1 if r>0 else 0. if t==0 else None
            item=dict(metric=metric,status=('PASS' if change<=limit else 'FAIL') if change is not None else 'FAIL',
                relative_change=change,reference_zero_positive_error=r==0 and t>0,treatment_mean=t,RAW_mean=r,limit=limit)
        quality.append(item)
    q7='INCOMPLETE' if any(x['status']=='INCOMPLETE' for x in quality[:2]) else 'PASS' if all(x['status']=='PASS' for x in quality[:2]) else 'FAIL'
    add(7,'D_sym_and_original_MSE_relative_mean_cost',status=q7,metrics=quality[:2])
    add(8,'original_p95_relative_mean_cost',status=quality[2]['status'],metrics=quality[2:])
    outputs=[by_id.get(r['request_id'],{}) for r in jobs if r['label']==treatment]
    raw_outputs=[by_id.get(r['request_id'],{}) for r in jobs if r['label']=='RAW']
    contracts_ready=ready and len(outputs)==16 and all(r.get('geometry_contracts_complete') is True for r in outputs)
    identity=all(r.get('RAW_original_score_identity')=='PASS' for r in raw_outputs) and len(raw_outputs)==16
    contract_pass=all(r.get('evaluation_contract_pass') is True for r in outputs+raw_outputs)
    add(9,'C_N_finite_budget_accepted_equations_no_new_degeneracy',status='INCOMPLETE' if not contracts_ready or not identity else 'PASS' if contract_pass else 'FAIL',
        original_RAW_score_identity=identity,failed_requests=[r.get('request_id') for r in outputs if r.get('evaluation_contract_pass') is not True])
    if phase==2:
        drift_ready=ready and all(isinstance(r.get('trajectory_RAW_drift_pass'),bool) for r in outputs)
        passed=all(r.get('trajectory_RAW_drift_pass') is True for r in outputs)
        add(10,'final_TRAJECTORY_RAW_max_corner_drift_at_most_3delta',status='INCOMPLETE' if not drift_ready else 'PASS' if passed else 'FAIL',
            failed_requests=[r.get('request_id') for r in outputs if r.get('trajectory_RAW_drift_pass') is False],
            maximum_observed_drift_over_delta=max((r['RAW_max_corner_displacement']/r['delta'] for r in outputs if number(r.get('delta')) and r['delta']>0),default=None))
    status='INCOMPLETE' if any(x['status']=='INCOMPLETE' for x in checks) else 'FAIL' if any(x['status']=='FAIL' for x in checks) else 'PASS'
    signal=all(x['status']=='PASS' for x in checks[:2])
    classification='BUDGETED_JOINT_IMPROVEMENT' if status=='PASS' else 'INCOMPLETE' if status=='INCOMPLETE' else 'TRADEOFF' if signal else 'NO_PHASE1_SIGNAL' if phase==1 else 'NO_TRAJECTORY_JOINT_SIGNAL'
    return dict(status=status,phase=phase,treatment=treatment,reference='RAW',classification=classification,
        all_registered_results_present=ready,expected_per_label=16,parents=8,noises_per_parent=2,checks=checks,
        phase2_allowed=phase==1 and status=='PASS',phase2_status='ELIGIBLE_NOT_DISPATCHED' if phase==1 and status=='PASS' else 'NOT_RUN_NO_PHASE1_SIGNAL' if phase==1 else 'EVALUATED',
        scientific_thresholds=gate_definition(),GT_used_only_for_registered_phase_decision=True,per_output_GT_selection=False)


def finalize(out=OUT,phase=1):
    from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv
    out=Path(out);requests=read_json(out/'requests.json');rows=read_json(out/'scoring/per_output.json')
    by_id={r['request_id']:r for r in rows};parents=parent_rows(requests,by_id);folder=out/'aggregation'
    atomic_json(folder/'per_parent_metrics.json',parents);write_csv(folder/'per_parent_metrics.csv',parents)
    label_summary=[]
    for label in ('RAW','ENDPOINT','TRAJECTORY'):
        pp=[r for r in parents if r['label']==label]
        for metric in METRICS:
            vals=[r[metric] for r in pp];complete=len(pp)==8 and all(number(v) for v in vals)
            label_summary.append(dict(label=label,metric=metric,mean=mean(vals) if complete else None,parents=8,complete=complete))
    atomic_json(folder/'per_label_summary.json',label_summary);write_csv(folder/'per_label_summary.csv',label_summary)
    index={(r['label'],r['uid']):r for r in parents};uids=sorted({r['uid'] for r in requests if r['label']=='RAW'})
    paired=[];noise=[];summaries=[];noise_summary=[]
    cases={(r['label'],r['uid'],int(r['candidate'])):by_id.get(r['request_id'],{}) for r in requests}
    for treatment,reference in [('ENDPOINT','RAW'),('TRAJECTORY','RAW'),('TRAJECTORY','ENDPOINT')]:
        name=treatment+'-'+reference
        for metric in METRICS:
            differences=[]
            for uid in uids:
                t=index[treatment,uid].get(metric);r=index[reference,uid].get(metric)
                delta=t-r if number(t) and number(r) else None;differences.append(delta)
                paired.append(dict(contrast=name,uid=uid,metric=metric,treatment=t,reference=r,difference=delta,status='PASS' if delta is not None else 'NA'))
                for candidate in (0,1):
                    tr,rr=cases[treatment,uid,candidate],cases[reference,uid,candidate]
                    tv,rv=tr.get(metric),rr.get(metric);valid=tr.get('status')==rr.get('status')=='PASS' and number(tv) and number(rv)
                    noise.append(dict(contrast=name,uid=uid,candidate=candidate,metric=metric,treatment=tv,reference=rv,
                        difference=tv-rv if valid else None,status='PASS' if valid else 'NA'))
            complete=len(differences)==8 and all(number(v) for v in differences)
            high=metric in ('SeamCov_strict','SeamCov','SeamCov_0.005','SeamCov_0.02','ExposedRetain')
            summaries.append(dict(contrast=name,metric=metric,mean_difference=mean(differences) if complete else None,
                median_difference=statistics.median(differences) if complete else None,
                improving_parents=sum((v>0 if high else v<0) for v in differences) if complete else None,
                minimum_difference=min(differences) if complete else None,maximum_difference=max(differences) if complete else None,
                parents=8,complete=complete,positive_is_better=high))
            for candidate in (0,1):
                vals=[r['difference'] for r in noise if r['contrast']==name and r['metric']==metric and r['candidate']==candidate]
                ok=len(vals)==8 and all(number(v) for v in vals)
                noise_summary.append(dict(contrast=name,metric=metric,candidate=candidate,parents=8,complete=ok,
                    mean_difference=mean(vals) if ok else None,median_difference=statistics.median(vals) if ok else None,
                    improving_parents=sum((v>0 if high else v<0) for v in vals) if ok else None))
    for filename,value in [('paired_parent',paired),('paired_noise',noise),('paired_parent_summary',summaries),('paired_noise_summary',noise_summary)]:
        atomic_json(folder/(filename+'.json'),value);write_csv(folder/(filename+'.csv'),value)
    gate=phase_gate(requests,rows,parents,phase)
    gate.update(aggregation_source_sha256=digest(__file__),per_output_sha256=digest(out/'scoring/per_output.json'),
        requests_sha256=digest(out/'requests.json'),model_calls=0,geometry_score_calls=0)
    atomic_json(out/f'phase{phase}_gate.json',gate)
    summary=dict(status=gate['status'],phase=phase,classification=gate['classification'],gate_path=str(out/f'phase{phase}_gate.json'),
        stage2_allowed=gate['phase2_allowed'],all_registered_requests_retained=True,
        accepted_subset_never_reweights_parents=True,request_counts={label:sum(r.get('label')==label and r.get('status')=='PASS' for r in rows) for label in ('RAW','ENDPOINT','TRAJECTORY')})
    atomic_json(folder/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--phase',type=int,choices=(1,2),default=1)
    args=p.parse_args();print(json.dumps(finalize(args.out,args.phase)))
