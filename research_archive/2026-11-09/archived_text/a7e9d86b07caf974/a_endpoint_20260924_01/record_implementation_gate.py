import json,shutil
from pathlib import Path
from experiments.a_radial.io_utils import file_hash,write_json
run=Path('output/a_endpoint_20260924_01')
evidence=run/'correctness_evidence';evidence.mkdir(exist_ok=False)
reports=[]
for i in (1,2):
    src=Path(f'output/a_endpoint_correctness_20260924_0{i}')/'CORRECTNESS.json'
    dest=evidence/f'CORRECTNESS_0{i}.json';shutil.copy2(src,dest)
    assert file_hash(src)==file_hash(dest)
    reports.append(json.loads(dest.read_text()))
r=reports[-1]
assert all(x['pass'] for x in r['cpu_legacy_cases']) and len(r['cpu_legacy_cases'])==51
assert r['synthetic_guard_control']['pass']
assert all(x['pass'] for x in r['piecewise_radial_angle_finite_differences'])
assert all(x['pass'] for x in r['real_network_final_query_backward'])
cpu=[x for x in r['batch_device_comparisons'] if x['device']=='cpu']
actual=[x for x in r['batch_device_comparisons'] if x['device']=='cuda' and x['model'] in ('student_10','student_11')]
pressure=[x for x in r['batch_device_comparisons'] if x['device']=='cuda' and x['model']=='original_a']
assert len(cpu)==48 and len(actual)==32 and all(x['pass'] for x in cpu+actual)
assert all(not x['angle_budget_gate_changed_vs_cpu_single'] for x in pressure)
for name,h in r['source_hashes'].items():assert file_hash(Path('experiments/a_radial/layout_endpoint')/name)==h
write_json(run/'IMPLEMENTATION_GATE.json',{
    'status':'CONDITIONAL_SCOPE_ELIGIBLE_PENDING_CUDA_RESUME_AND_SINGLE_OBJECT_DIAGNOSTIC',
    'allowed_initializations':['v1_seed_10_final_update_3000','v1_seed_11_final_update_3000'],
    'strict_all_model_all_device_correctness_pass':r['pass'],
    'cpu_legacy_cases_passed':len(r['cpu_legacy_cases']),
    'cpu_single_and_batch4_cases_passed':len(cpu),
    'actual_initialization_cuda_cases_passed':len(actual),
    'original_A_cuda_numerical_parity':'UNSUPPORTED_FOR_3_OF_16_PRESSURE_CASES',
    'original_A_cuda_failed_cases':[x for x in pressure if not x['pass']],
    'not_a_claim_of_full_GPU_equivalence':True,
    'gate_rationale':'All actual training initializations and every original CPU path pass; old original A GPU stress limitations retained and original A is not used to initialize this continuation.',
    'training_batch_size':4,'network_calls':'per_parent_B1_forward_and_context_then_aggregate_four_parent_loss',
    'final_authority':'unchanged original CPU deployment plus exact endpoint audit',
    'final_required_diagnostic':'all six final weights x16 GPU mirror vs original CPU, preserving discrepancies and never choosing checkpoints',
    'source_hashes':r['source_hashes'],
    'evidence':{p.name:file_hash(p) for p in evidence.iterdir()},
    'thresholds_changed':False,'failed_evidence_preserved':True})
print(json.dumps({'implementation_scope_gate':'eligible_for_v1_initializations_only','strict_all_case_pass':r['pass'],'cpu_cases':99,'actual_cuda_cases':32,'original_A_gpu_pressure_failures':sum(not x['pass'] for x in pressure)}))