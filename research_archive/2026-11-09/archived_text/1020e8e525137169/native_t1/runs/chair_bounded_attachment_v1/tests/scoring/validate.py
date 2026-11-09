"""CPU arithmetic checks and reuse of projection fixture 1; no new geometry."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.mf_h1_local0.chair_bounded_attachment_v1 import aggregation as a, metrics_adapter as m
from experiments.mf_h1_local0.chair_bounded_attachment_v1.common import OUT,atomic_json,digest

requests=[];baseline=[]
for label in ('RAW','ENDPOINT','TRAJECTORY'):
    for parent in range(8):
        for candidate in range(2):
            r=dict(request_id=f'{label}_{parent}_{candidate}',label=label,uid=str(parent),candidate=candidate)
            requests.append(r);v=dict(r,status='PASS',SeamCov_strict=.1 if label=='RAW' else .13,
                SeamCov=.4,ConflictCov_C=.05,ExposedRetain=.9,ExposedRetain_global_lower=.9,ExposedRetain_global_upper=.9,
                D_sym2048=1.,fixed_query_MSE=1.,fixed_query_p95_L=1.,evaluation_contract_pass=True,
                geometry_contracts_complete=True,RAW_original_score_identity='PASS',trajectory_RAW_drift_pass=True,
                RAW_max_corner_displacement=0.,delta=.01,accepted_count=0)
            for key in ('SeamCov_strict','SeamCov','ConflictCov_C'):v[key+'_upper']=v[key]
            baseline.append(v)
def gate(rows,phase=1):return a.phase_gate(requests,rows,a.parent_rows(requests,{r['request_id']:r for r in rows}),phase)
checks=[]
def check(name,expected,modify=None,phase=1,criterion=None):
    rows=copy.deepcopy(baseline)
    if modify:modify(rows)
    result=gate(rows,phase)
    observed=result['status'] if criterion is None else next(x['status'] for x in result['checks'] if x['id']==criterion)
    assert observed==expected,(name,observed,expected)
    if expected!='PASS' and criterion is None:assert result['phase2_allowed'] is False
    checks.append(dict(name=name,status='PASS',expected=expected,observed=observed))
def set_endpoint(rows,**values):
    for r in rows:
        if r['label']=='ENDPOINT':r.update(values)
check('all_nine_pass_without_accepted_count_filter','PASS')
check('strict_mean_below_002','FAIL',lambda r:set_endpoint(r,SeamCov_strict=.11,SeamCov_strict_upper=.11))
def only_parents(rows,count,gain):
    for r in rows:
        if r['label']=='ENDPOINT':r['SeamCov_strict']=r['SeamCov_strict_upper']=.1+(gain if int(r['uid'])<count else 0)
check('four_positive_and_positive_median','PASS',lambda r:only_parents(r,4,.05))
check('three_positive_even_high_mean_fails','FAIL',lambda r:only_parents(r,3,.1))
check('old_seam_drop_over_tolerance','FAIL',lambda r:set_endpoint(r,SeamCov=.394,SeamCov_upper=.394))
check('conflict_mean_cost','FAIL',lambda r:set_endpoint(r,ConflictCov_C=.061,ConflictCov_C_upper=.061))
check('exposure_mean_cost','FAIL',lambda r:set_endpoint(r,ExposedRetain=.889,ExposedRetain_global_lower=.889,ExposedRetain_global_upper=.889))
def one_parent_conflict(rows):
    for r in rows:
        if r['label']=='ENDPOINT' and r['uid']=='0':r['ConflictCov_C']=r['ConflictCov_C_upper']=.11
check('single_parent_cost_not_hidden_by_mean','FAIL',one_parent_conflict,criterion=6)
check('uncertain_interval_cannot_pass','INCOMPLETE',lambda r:set_endpoint(r,ConflictCov_C_upper=.07))
check('Dsym_five_percent_cost','FAIL',lambda r:set_endpoint(r,D_sym2048=1.06))
check('old_MSE_five_percent_cost','FAIL',lambda r:set_endpoint(r,fixed_query_MSE=1.06))
check('old_p95_ten_percent_cost','FAIL',lambda r:set_endpoint(r,fixed_query_p95_L=1.11))
check('geometry_contract_failure','FAIL',lambda r:set_endpoint(r,evaluation_contract_pass=False))
check('geometry_contract_missing','INCOMPLETE',lambda r:set_endpoint(r,geometry_contracts_complete=False))
check('missing_noise_not_dropped','INCOMPLETE',lambda r:r.__setitem__(16,dict(r[16],status='BLOCKED')))
check('phase2_same_constraints_pass','PASS',phase=2)
def drift(rows):
    rows[-1].update(trajectory_RAW_drift_pass=False,RAW_max_corner_displacement=.031)
check('phase2_final_RAW_drift','FAIL',drift,phase=2)
def unknown_identity(rows):rows[0]['RAW_original_score_identity']='BLOCKED'
check('RAW_original_identity_required','INCOMPLETE',unknown_identity)

path=OUT/'tests/project/attempt1/fixtures.npz'
with np.load(path,allow_pickle=False) as z:
    C=z['1_coplanar_gap_C'].copy();F=z['1_coplanar_gap_F'].copy();endpoint=z['1_coplanar_gap_out'].copy()
condition=dict(C=C,L0=float(np.linalg.norm(np.ptp(C.reshape(-1,3),axis=0))),gamma_edges=C[:,:2],gamma_known_face_indices=np.array([0]),gamma_status='PASS')
gamma=m.control._gamma(condition,64)
ref=dict(audits=dict(gamma={'status':'PASS'},gamma_samples_per_edge=64),arrays=gamma['arrays'])
formula=[m.formula_equivalence_check(value,condition,ref) for value in (F,endpoint)]
result=dict(status='PASS',semantic_checks=checks,formula_equivalence=formula,
    reused_fixture='1_coplanar_gap',fixture_file_sha256=digest(path),new_geometry_fixtures=0,
    model_calls=0,original_globals_unchanged=m.control.TAUS==(.005,.01,.02),
    source_sha256={n:digest(ROOT/'experiments/mf_h1_local0/chair_bounded_attachment_v1'/(n+'.py')) for n in ('aggregation','metrics_adapter','evaluation')})
atomic_json(OUT/'tests/scoring/results.json',result)
print(json.dumps(dict(status='PASS',semantic_checks=len(checks),same_fixture_formula_checks=len(formula))))
