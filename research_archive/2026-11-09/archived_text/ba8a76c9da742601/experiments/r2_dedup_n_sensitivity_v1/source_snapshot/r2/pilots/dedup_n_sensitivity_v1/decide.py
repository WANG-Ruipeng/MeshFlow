"""Compute the pre-registered pilot gate from complete cached scores and reviews."""
import collections,json
import numpy as np
from . import common as C

METRICS={'FF_pct':'FF_illegal_affected_free_fraction','CF_pct':'CF_illegal_affected_free_fraction',
 'I_pct':'ConflictCov_C','small_angle_pct':'free_minimum_angle_lt5_fraction',
 'deleted_pct':'GEO_deleted_fraction_of_requested_free','oriented_reach_pct':'opposite_winding_reached_free_area_fraction'}
LABELS=('ORIGINAL_ORIG_N','ORIGINAL_LOW_N','DEDUP_ORIG_N','DEDUP_LOW_N')

def calculate():
    r=C.load_registration();assert C.sha(r['gate_path'])==r['gate_sha256']
    rows=C.csvread(C.OUT/'evaluation/tables/per_output.csv');assert len(rows)==192
    lookup={(x['label'],x['stage'],x['request_id']):x for x in rows};assert len(lookup)==192
    summaries=[];by={}
    for label in LABELS:
        for stage in ('RAW','POST'):
            for split in ('ALL','TRAIN','DEV'):
                group=[x for x in rows if x['label']==label and x['stage']==stage and (split=='ALL' or x['split']==split)]
                assert len(group)==(24 if split=='ALL' else 12)
                s=dict(label=label,stage=stage,split=split,n=len(group),
                    legal_CF_requests=sum(float(x['C_F_valid_edge_count'])>0 for x in group),
                    reach_ge30_requests=sum(float(x['opposite_winding_reached_free_area_fraction'])>=.3 for x in group))
                for name,col in METRICS.items():s[name]=float(np.mean([100*float(x[col]) for x in group]))
                for col in ('actual_free_faces','deleted_free_faces','C_F_valid_edge_count','C_F_valid_edge_total_length','free_total_surface_area','nonmanifold_edges'):
                    s[col]=float(np.mean([float(x[col]) for x in group]))
                s['requested_free']=float(np.mean([float(x['N'])-float(x['K']) for x in group]))
                summaries.append(s);by[label,stage,split]=s
    C.csvwrite(C.OUT/'analysis/main_summary.csv',summaries,replace=True)
    orig=by['ORIGINAL_LOW_N','POST','ALL'];high=by['DEDUP_ORIG_N','POST','ALL'];low=by['DEDUP_LOW_N','POST','ALL']
    changes=[]
    for q in r['requests']:
        rid=q['request_id'];a=lookup['DEDUP_LOW_N','POST',rid];b=lookup['DEDUP_ORIG_N','POST',rid]
        changes.append(dict(request_id=rid,parent_id=q['parent_id'],split=q['split'],
            CF_delta_pp=100*(float(a[METRICS['CF_pct']])-float(b[METRICS['CF_pct']])),
            I_delta_pp=100*(float(a[METRICS['I_pct']])-float(b[METRICS['I_pct']])),
            FF_delta_pp=100*(float(a[METRICS['FF_pct']])-float(b[METRICS['FF_pct']])),
            old_legal_CF=float(b['C_F_valid_edge_count'])>0,new_legal_CF=float(a['C_F_valid_edge_count'])>0))
    lost=[x['request_id'] for x in changes if x['old_legal_CF'] and not x['new_legal_CF']]
    new=[x['request_id'] for x in changes if not x['old_legal_CF'] and x['new_legal_CF']]
    checks=dict(all_scores_and_C_exact=all(x['C_preserved_exact']=='True' for x in rows),
        dedup_FF_better_at_same_low_N=low['FF_pct']<orig['FF_pct'],
        dedup_low_deleted_fraction_lower=low['deleted_pct']<high['deleted_pct'],
        dedup_low_FF_no_higher=low['FF_pct']<=high['FF_pct'],
        CF_mean_within_0_5pp=low['CF_pct']-high['CF_pct']<=.5,
        I_mean_within_0_5pp=low['I_pct']-high['I_pct']<=.5,
        CF_worst_paired_increase_within_5pp=max(x['CF_delta_pp'] for x in changes)<=5,
        I_worst_paired_increase_within_5pp=max(x['I_delta_pp'] for x in changes)<=5,
        existing_legal_CF_requests_all_retained=not lost,
        large_reachable_area_coverage_not_lower=low['reach_ge30_requests']>=high['reach_ge30_requests'])
    visual_path=C.OUT/'evaluation/gallery/per_request_visual_delta.csv';visual=None
    if visual_path.exists():
        allvis=C.csvread(visual_path);vis=[x for x in allvis if x['source_arm']=='DEDUP_C200' and x['stage']=='POST']
        assert len(vis)==24 and len({x['request_id'] for x in vis})==24
        good=[x for x in vis if x['judgment']=='IMPROVED'];bad=[x for x in vis if x['judgment']=='WORSE']
        visual=dict(counts=dict(collections.Counter(x['judgment'] for x in vis)),improved_parents=len({x['parent_id'] for x in good}),
            new_major_body_loss=[x['request_id'] for x in vis if x['new_major_body_loss']=='YES'],
            C_role_lost=[x['request_id'] for x in vis if x['C_role_retained']=='NO'],
            uncertainty=dict(C_role_unknown=sum(x['C_role_retained']=='UNKNOWN' for x in vis),body_unknown=sum(x['new_major_body_loss']=='UNKNOWN' for x in vis)))
        checks.update(visual_multiple_requests_and_parents=len(good)>=3 and visual['improved_parents']>=2,
            visual_improvements_outnumber_worsening=len(good)>len(bad),
            no_new_clear_major_body_loss=not visual['new_major_body_loss'],C_role_retained=not visual['C_role_lost'])
    status='PASS' if visual is not None and all(checks.values()) else ('FAIL' if not all(checks.values()) else 'WAIT_ACTUAL_VISUAL_REVIEW')
    result=dict(status=status,gate_sha256=r['gate_sha256'],checks=checks,
        failed=[k for k,v in checks.items() if not v],old_legal_requests_lost=lost,new_legal_requests=new,
        DEDUP_low_vs_origN_delta={k:low[k]-high[k] for k in METRICS},
        actual_visual=visual,source_score_table_sha256=C.sha(C.OUT/'evaluation/tables/per_output.csv'),
        new_model_calls=0,B_status='ELIGIBLE_PENDING_FROZEN_FEASIBILITY' if status=='PASS' else 'NOT_RUN_GATE_NOT_MET',
        caution='Pilot decision screen; failed A does not disprove all SEAM+dedup designs or all count choices. Do not tune N after these results.')
    C.csvwrite(C.OUT/'analysis/gate_request_changes.csv',changes,replace=True)
    C.write(C.OUT/'analysis/gate_decision.json',result,replace=True)
    return result

if __name__=='__main__':print(json.dumps(calculate(),indent=2))
