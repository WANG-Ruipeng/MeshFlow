"""CPU report semantics only: no geometry, tensor, or model execution."""
import ast
import hashlib
import json
from pathlib import Path
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import reporting as report
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import aggregation as aggregate

root=Path(__file__).resolve().parents[5]
folder=Path(__file__).resolve().parent
source=root/'experiments/mf_h1_local0/chair_boundary_attachment_v1'
old_source=folder/'originals/experiments/mf_h1_local0/chair_boundary_attachment_v1'
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(dict(name=name,status='PASS'))
def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ('_bounds','_parent_rows','_aggregate_parents','_contrasts','_interactions','_gate'):
    check('scientific_arithmetic_unchanged_'+name,ast.dump(node(source/'aggregation.py',name))==ast.dump(node(old_source/'aggregation.py',name)))
old_node=node(old_source/'reporting.py','_recommend')
old_namespace={}
exec(compile(ast.Module(body=[old_node],type_ignores=[]),'preserved_recommend','exec'),old_namespace)
labels=('B_BOUNDARY_END','C_SEAM_END','D_BOUNDARY_SEAM_END')
gates=[dict(label=k,classification='JOINT_ATTACHMENT_PILOT_SIGNAL') for k in labels]
check('old_incomplete_can_promote_reproduced',old_namespace['_recommend'](gates,{},False)['retain']=='B_BOUNDARY_END')
new=report._recommend(gates,{},False)
check('incomplete_never_promotes',new['retain']=='NO_PROMOTION_PENDING_EVIDENCE')
check('incomplete_P1_candidates_preserved',new['provisional_P1_candidates']==sorted(labels))
for i,label in enumerate(labels):
    check('complete_simple_component_priority_'+label,report._recommend(gates[i:],{},True)['retain']==label)
metrics=['ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE','fixed_query_p95_L','FF_illegal_affected_free_fraction','N_C_finite']
costs=[dict(reference=r,metric=m,status='PASS' if r=='START_HYBRID' else 'INCOMPLETE') for r in ('START_HYBRID','A_FM_END') for m in metrics]
audit=dict(cost_status='INCOMPLETE',cost_checks=costs)
check('A_reference_ignores_self_interval_uncertainty',report._reference_cost_status(audit,'START_HYBRID')=='PASS')
audit['cost_checks'][0]['status']='FAIL'
check('A_actual_START_failure_preserved',report._reference_cost_status(audit,'START_HYBRID')=='FAIL')
audit['cost_checks']=costs[1:]
check('A_missing_START_requirement_is_incomplete',report._reference_cost_status(audit,'START_HYBRID')=='INCOMPLETE')
no_signal=[dict(label=k,classification='NO_ATTACHMENT_SIGNAL') for k in labels]
check('A_positive_descriptive_candidate_retained',report._recommend(no_signal,dict(descriptive_improvement_with_cost_pass=True),True)['retain']=='A_FM_END_DESCRIPTIVE')
check('A_candidate_not_promoted_incomplete',report._recommend(no_signal,dict(descriptive_improvement_with_cost_pass=True),False)['retain']=='NO_PROMOTION_PENDING_EVIDENCE')
check('START_retained_when_no_new_signal',report._recommend(no_signal,{},True)['retain']=='START_HYBRID')
check('aggregate_incomplete_dominates_joint',aggregate._overall_classification(gates,False)=='COMPLETED_WITH_EVALUATION_GAP')
check('aggregate_complete_joint_preserved',aggregate._overall_classification(gates,True)=='JOINT_ATTACHMENT_PILOT_SIGNAL')
check('aggregate_complete_no_signal_preserved',aggregate._overall_classification(no_signal,True)=='NO_ATTACHMENT_SIGNAL')
check('aggregate_complete_tradeoff_preserved',aggregate._overall_classification([dict(classification='ATTACHMENT_SIGNAL_WITH_TRADEOFF')],True)=='ATTACHMENT_SIGNAL_WITH_TRADEOFF')
result=dict(status='PASS',checks=checks,check_count=len(checks),geometry_score_calls=0,model_calls=0,backward=0,optimizer_updates=0,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'reporting.py',source/'aggregation.py')})
(folder/'semantic_regression_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
