"""Fixed-parent CPU aggregation and registered 2x2 interaction audit."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from .common import OUT, LABELS, read_json, atomic_json, digest
from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv

METRICS=('SeamCov','SeamCov_upper','SeamCov_0.005','SeamCov_0.01','SeamCov_0.02',
    'ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_GT','ConflictCov_C_minus_GT',
    'ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup',
    'ExposedRetain','ExposedRetain_upper','ExposedRetain_global_lower','ExposedRetain_global_upper','reference_exposed_fraction','ExposedRetain_uncertain_fraction','reference_visibility_uncertain_fraction',
    'D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE',
    'fixed_query_p95_L','coverage_fraction','FF_illegal_affected_free_fraction',
    'CF_proper_affected_free_fraction','free_minimum_angle_lt5_fraction',
    'free_near_degenerate_fraction','free_exact_zero_area_fraction','free_area_over_GT_free_area',
    'free_out_of_GT_cube_corner_fraction','generation_seconds','peak_cuda_allocated_bytes',
    'peak_cuda_reserved_bytes','common_RMS_L0','common_p95_L0','common_recall_0.01')
CONTRASTS=(('D_BOUNDARY_SEAM_END','A_FM_END','D-A'),('B_BOUNDARY_END','A_FM_END','B-A'),
    ('C_SEAM_END','A_FM_END','C-A'),('D_BOUNDARY_SEAM_END','B_BOUNDARY_END','D-B'),
    ('D_BOUNDARY_SEAM_END','C_SEAM_END','D-C'))
PRIMARY=('SeamCov','ConflictCov_C','ExposedRetain')


def number(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def summarize(values,expected):
    values=[float(x) for x in values if number(x)]
    return dict(mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,
        minimum=float(min(values)) if values else None,maximum=float(max(values)) if values else None,
        n=len(values),expected=expected,complete=len(values)==expected)



def _bounds(observation,metric):
    low=observation.get(metric);high=low
    if metric in PRIMARY:
        high=observation.get(metric+'_upper')
        if metric=='ExposedRetain':
            low=observation.get('ExposedRetain_global_lower')
            high=observation.get('ExposedRetain_global_upper')
    elif metric=='FF_illegal_affected_free_fraction' and number(low):
        unresolved=observation.get('FF_unresolved_or_degenerate_pair_count',0)
        free=observation.get('free_faces')
        if number(unresolved) and unresolved>0:
            high=min(1.,low+2*unresolved/free) if number(free) and free>0 else 1.
    return (float(low),float(high)) if number(low) and number(high) else (None,None)


def _parent_rows(requests,by_id):
    groups={}
    for request in requests:
        groups.setdefault((request['label'],request['panel'],request['uid']),[]).append(request)
    rows=[]
    for (label,panel,uid),reqs in groups.items():
        observed=[by_id.get(r['request_id'],{}) for r in reqs]
        applicable=[(r,o) for r,o in zip(reqs,observed) if r['status']!='NOT_APPLICABLE']
        row=dict(label=label,panel=panel,uid=uid,registered=len(reqs),applicable=len(applicable),
            scored=sum(o.get('status')=='PASS' for r,o in applicable),
            original_NA=sum(r['status']=='NOT_APPLICABLE' for r in reqs))
        for metric in METRICS:
            vals=[o.get(metric) for r,o in applicable if o.get('status')=='PASS']
            stats=summarize(vals,len(applicable));row[metric]=stats['mean']
            row[metric+'_n']=stats['n'];row[metric+'_complete']=stats['complete'] and len(applicable)>0
        for metric in PRIMARY:
            statuses=[o.get(metric+'_status','BLOCKED') for r,o in applicable]
            row[metric+'_statuses']='|'.join(sorted(set(statuses)))
            row[metric+'_fully_resolved']=bool(statuses) and all(s=='PASS' for s in statuses)
        for metric in ('SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE','fixed_query_p95_L','FF_illegal_affected_free_fraction'):
            bounds=[_bounds(o,metric) for r,o in applicable if o.get('status')=='PASS']
            complete=bool(applicable) and len(bounds)==len(applicable) and all(number(a) and number(b) for a,b in bounds)
            row[metric+'_bound_lower']=float(np.mean([a for a,b in bounds])) if complete else None
            row[metric+'_bound_upper']=float(np.mean([b for a,b in bounds])) if complete else None
            row[metric+'_bound_complete']=complete
        row['output_contract_pass']=bool(applicable) and all(o.get('finite') is True and
            o.get('C_preserved') is True and o.get('N_preserved') is True for r,o in applicable)
        rows.append(row)
    return rows


def _aggregate_parents(parents):
    groups={}
    for row in parents:groups.setdefault((row['label'],row['panel']),[]).append(row)
    rows=[]
    for (label,panel),group in groups.items():
        eligible=[g for g in group if g['applicable']>0]
        for metric in METRICS:
            st=summarize([g[metric] for g in eligible if g[metric+'_complete']],len(eligible))
            rows.append(dict(label=label,panel=panel,metric=metric,registered_parents=len(group),
                applicable_parents=len(eligible),original_NA_parents=len(group)-len(eligible),
                unit='equal parent mean; all registered candidates inside parent',**st))
    return rows


def _contrasts(parents):
    index={(x['label'],x['panel'],x['uid']):x for x in parents}
    output=[]
    for treatment,reference,name in CONTRASTS:
        for (label,panel,uid),row in index.items():
            if label!=treatment:continue
            ref=index[(reference,panel,uid)]
            for metric in METRICS:
                valid=row[metric+'_complete'] and ref[metric+'_complete']
                output.append(dict(contrast=name,treatment=treatment,reference=reference,panel=panel,
                    uid=uid,metric=metric,treatment_mean=row[metric],reference_mean=ref[metric],
                    difference=row[metric]-ref[metric] if valid else None,status='PASS' if valid else 'NA',
                    direction='signed treatment minus reference'))
    return output


def _interactions(requests,by_id):
    index={(r['label'],r['base_request_id']):by_id.get(r['request_id'],{}) for r in requests}
    base={r['base_request_id']:r for r in requests}
    output=[]
    for bid,req in base.items():
        arms=[index.get((label,bid),{}) for label in ('A_FM_END','B_BOUNDARY_END','C_SEAM_END','D_BOUNDARY_SEAM_END')]
        for metric in METRICS:
            values=[r.get(metric) for r in arms]
            valid=all(number(v) for v in values) and all(r.get('status')=='PASS' for r in arms)
            output.append(dict(base_request_id=bid,panel=req['panel'],uid=req['uid'],
                condition_id=req['condition_id'],candidate=req['candidate'],metric=metric,
                A=values[0],B=values[1],C=values[2],D=values[3],
                interaction=values[3]-values[1]-values[2]+values[0] if valid else None,
                status='PASS' if valid else 'NA',definition='D-B-C+A'))
    groups={}
    for row in output:groups.setdefault((row['panel'],row['uid'],row['metric']),[]).append(row)
    parent=[]
    for (panel,uid,metric),rows in groups.items():
        st=summarize([x['interaction'] for x in rows],len(rows))
        parent.append(dict(panel=panel,uid=uid,metric=metric,definition='mean all registered condition/noise D-B-C+A',**st))
    return output,parent


def _gate(label,parents):
    index={(x['label'],x['uid']):x for x in parents if x['panel']=='P1'}
    uids=sorted({uid for lab,uid in index})
    rows=[index[(label,u)] for u in uids]
    baseline=[index[('A_FM_END',u)] for u in uids]
    start=[index[('START_HYBRID',u)] for u in uids]
    required=lambda rs,key:len(rs)==8 and all(x.get(key+'_bound_complete',False) for x in rs)
    def values(rs,key):
        return np.asarray([x[key+'_bound_lower'] for x in rs]),np.asarray([x[key+'_bound_upper'] for x in rs])
    seam_ready=all(required(rs,'SeamCov') for rs in (rows,baseline,start))
    if seam_ready:
        tl,tu=values(rows,'SeamCov');al,au=values(baseline,'SeamCov');sl,su=values(start,'SeamCov')
        lower,upper=tl-au,tu-al;start_lower,start_upper=tl-su,tu-sl
        lower_checks=dict(mean_gain_at_least_0_05=float(lower.mean())>=.05,
            median_gain_positive=float(np.median(lower))>0,at_least_6_of_8_parents_improve=int((lower>0).sum())>=6,
            mean_not_below_START=float(start_lower.mean())>=0)
        upper_checks=dict(mean_gain_at_least_0_05=float(upper.mean())>=.05,
            median_gain_positive=float(np.median(upper))>0,at_least_6_of_8_parents_improve=int((upper>0).sum())>=6,
            mean_not_below_START=float(start_upper.mean())>=0)
        status='PASS' if all(lower_checks.values()) else 'FAIL' if not all(upper_checks.values()) else 'INCOMPLETE'
        seam=dict(status=status,worst_case_checks=lower_checks,best_case_checks=upper_checks,
            mean_gain_vs_A_lower=float(lower.mean()),mean_gain_vs_A_upper=float(upper.mean()),
            median_gain_vs_A_lower=float(np.median(lower)),median_gain_vs_A_upper=float(np.median(upper)),
            improving_parents_lower=int((lower>0).sum()),improving_parents_upper=int((upper>0).sum()),
            mean_gain_vs_START_lower=float(start_lower.mean()),mean_gain_vs_START_upper=float(start_upper.mean()),parents=8,
            uncertainty_rule='PASS if worst case passes; FAIL if even best case fails; otherwise INCOMPLETE')
    else:seam=dict(status='INCOMPLETE',reason='Need all 8 parents/all 4 candidates and complete full-Gamma bounds for treatment/A/START')
    costs=[]
    for reference,refrows in (('A_FM_END',baseline),('START_HYBRID',start)):
        for metric,limit,kind in (('ConflictCov_C',.02,'difference'),
            ('ExposedRetain',.02,'decrease'),('D_sym2048',.05,'relative_increase'),
            ('fixed_query_MSE',.05,'relative_increase'),('fixed_query_p95_L',.10,'relative_increase'),
            ('FF_illegal_affected_free_fraction',.02,'difference')):
            ready=required(rows,metric) and required(refrows,metric)
            if not ready:
                costs.append(dict(reference=reference,metric=metric,status='INCOMPLETE',limit=limit));continue
            tl,tu=values(rows,metric);rl,ru=values(refrows,metric)
            if kind=='decrease':lower,upper=rl-tu,ru-tl
            else:lower,upper=tl-ru,tu-rl
            if kind=='relative_increase':
                def ratio(a,b):
                    return float(a.mean()/b.mean()-1) if b.mean()>0 else (0. if a.mean()==0 else None)
                low,high=ratio(tl,ru),ratio(tu,rl)
            else:low,high=float(lower.mean()),float(upper.mean())
            robust_serious=[uids[i] for i,d in enumerate(lower) if d>.10] if metric in ('ConflictCov_C','ExposedRetain') else []
            possible_serious=[uids[i] for i,d in enumerate(upper) if d>.10] if metric in ('ConflictCov_C','ExposedRetain') else []
            if robust_serious or (low is not None and low>limit):status='FAIL'
            elif high is not None and high<=limit and not possible_serious:status='PASS'
            else:status='INCOMPLETE'
            costs.append(dict(reference=reference,metric=metric,status=status,kind=kind,observed_lower=low,
                observed_upper=high,limit=limit,serious_parent_regressions=robust_serious,
                possible_serious_parent_regressions=possible_serious,parent_signed_cost_lower=lower.tolist(),
                parent_signed_cost_upper=upper.tolist()))
        contract_ready=all(r['scored']==r['applicable']==4 for r in rows)
        costs.append(dict(reference=reference,metric='N_C_finite',status=('PASS' if all(r['output_contract_pass'] for r in rows) else 'FAIL') if contract_ready else 'INCOMPLETE'))
    cost_status='FAIL' if any(x['status']=='FAIL' for x in costs) else 'INCOMPLETE' if any(x['status']=='INCOMPLETE' for x in costs) else 'PASS'
    if seam['status']=='INCOMPLETE':classification='COMPLETED_WITH_EVALUATION_GAP'
    elif seam['status']=='FAIL':classification='NO_ATTACHMENT_SIGNAL'
    elif cost_status=='INCOMPLETE':classification='COMPLETED_WITH_EVALUATION_GAP'
    elif cost_status=='FAIL':classification='ATTACHMENT_SIGNAL_WITH_TRADEOFF'
    else:classification='JOINT_ATTACHMENT_PILOT_SIGNAL'
    return dict(label=label,seam_screen=seam,cost_status=cost_status,cost_checks=costs,classification=classification,
        no_default_replacement_authorized=True)


def _overall_classification(gates,complete):
    # Local P1 screens remain in gates, but never become a complete-task verdict
    # while another registered panel/output is missing.
    if not complete:return 'COMPLETED_WITH_EVALUATION_GAP'
    classifications=[g['classification'] for g in gates]
    if 'JOINT_ATTACHMENT_PILOT_SIGNAL' in classifications:return 'JOINT_ATTACHMENT_PILOT_SIGNAL'
    if 'ATTACHMENT_SIGNAL_WITH_TRADEOFF' in classifications:return 'ATTACHMENT_SIGNAL_WITH_TRADEOFF'
    if 'COMPLETED_WITH_EVALUATION_GAP' in classifications:return 'COMPLETED_WITH_EVALUATION_GAP'
    return 'NO_ATTACHMENT_SIGNAL'


def finalize(out=OUT):
    out=Path(out);requests=read_json(out/'requests.json')
    rows=read_json(out/'scoring/per_output.json') if (out/'scoring/per_output.json').exists() else []
    by_id={r['request_id']:r for r in rows};parents=_parent_rows(requests,by_id)
    aggregate=_aggregate_parents(parents);contrasts=_contrasts(parents);interactions,ip=_interactions(requests,by_id)
    directory=out/'aggregation'
    for filename,value in (('parent_means',parents),('panel_means',aggregate),
        ('paired_parent_differences',contrasts),('interaction_per_output',interactions),('interaction_per_parent',ip)):
        atomic_json(directory/(filename+'.json'),value);write_csv(directory/(filename+'.csv'),value)
    # Best-of-four is descriptive only: select by minimum D_sym2048, report actual same candidate.
    best=[]
    for label in LABELS:
        for uid in sorted({r['uid'] for r in requests if r['panel']=='P1'}):
            candidates=[r for r in rows if r.get('label')==label and r.get('panel')=='P1' and r['uid']==uid and r.get('status')=='PASS' and number(r.get('D_sym2048'))]
            if len(candidates)==4:
                chosen=min(candidates,key=lambda r:(r['D_sym2048'],r['candidate']))
                best.append({k:chosen.get(k) for k in ('label','uid','request_id','candidate',*METRICS)})
            else:best.append(dict(label=label,uid=uid,status='INCOMPLETE'))
    write_csv(directory/'P1_best_of_four_descriptive.csv',best)
    gates=[_gate(label,parents) for label in ('B_BOUNDARY_END','C_SEAM_END','D_BOUNDARY_SEAM_END')]
    sensitivity=[]
    for unit in read_json(out/'registration/sensitivity_units.json'):
        row=by_id.get(unit['request_id'],{})
        for metric in PRIMARY:
            a,b=row.get(metric),row.get('dense_'+metric)
            sensitivity.append(dict(**unit,metric=metric,main=a,dense=b,
                dense_minus_main=b-a if number(a) and number(b) else None,
                main_status=row.get(metric+'_status','BLOCKED'),dense_status=row.get('dense_'+metric+'_status','BLOCKED')))
    write_csv(directory/'sensitivity.csv',sensitivity)
    # P5 comparisons include only the original two C1 noises and the registered C2/C3 noises.
    multi=[r for r in rows if r.get('panel')=='P5' or (r.get('panel')=='P1' and r.get('candidate') in (0,1) and r['uid'] in sorted({x['uid'] for x in requests if x['panel']=='P5'}))]
    write_csv(directory/'nested_context_common_support.csv',multi)
    complete=len(rows)==400 and all(r.get('status') in ('PASS','NOT_APPLICABLE') for r in rows)
    classification=_overall_classification(gates,complete)
    summary=dict(status='COMPLETE' if complete else 'INCOMPLETE',classification=classification,
        expected=400,applicable_expected=380,scored=sum(r.get('status')=='PASS' for r in rows),
        original_NA=sum(r.get('status')=='NOT_APPLICABLE' for r in rows),gates=gates,
        primary_unit='Eight P1 parents equally weighted; each mean includes all four registered candidates',
        Best_of_four='Descriptive only, minimum D_sym2048 with candidate id tie-break; never substitutes primary mean',
        interaction='D-B-C+A on same condition/noise, then equal parent; panels never pooled',
        incomplete_metric_policy='Missing/NA or changed denominator is INCOMPLETE. PARTIAL uses conservative lower/upper bounds: robust same classification is allowed, threshold-straddling uncertainty is INCOMPLETE. LOW_SUPPORT remains disclosed.',
        aggregation_source_sha256=digest(__file__),model_forwards=0)
    atomic_json(directory/'summary.json',summary)
    text=['# Registered attachment factorial decision\n',
        'Primary estimates average all registered candidates within parent, then weight the eight P1 parents equally. Best-of-four is supplemental only.\n',
        '**Classification: '+classification+'**\n',
        '| Arm | Seam screen | Joint cost gate | Classification |','|---|---|---|---|']
    for g in gates:text.append('| '+g['label']+' | '+g['seam_screen']['status']+' | '+g['cost_status']+' | '+g['classification']+' |')
    text+=['\nNo default model replacement or further training is initiated by this report.',
        '\nAll five panels remain separate. PARTIAL/NA values and original A40 failures retain their denominators. See parent_means.csv and paired_parent_differences.csv for every parent, interaction_per_output.csv for every noise, and sensitivity.csv for the four preselected density checks.']
    (directory/'decision.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args();print(json.dumps(finalize(a.out)))
