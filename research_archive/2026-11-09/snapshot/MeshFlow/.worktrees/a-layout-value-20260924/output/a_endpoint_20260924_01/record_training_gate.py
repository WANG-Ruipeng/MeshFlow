"""Finalize pretraining evidence without altering any frozen protocol."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from experiments.a_radial.io_utils import file_hash,write_json
run=Path('output/a_endpoint_20260924_01');module=Path('experiments/a_radial');smoke=Path('output/a_endpoint_smoke_20260924_01')
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
impl=json.loads((run/'IMPLEMENTATION_GATE.json').read_text())
assert impl['status']=='CONDITIONAL_SCOPE_ELIGIBLE_PENDING_CUDA_RESUME_AND_SINGLE_OBJECT_DIAGNOSTIC'
s=json.loads((smoke/'GPU_SMOKE.json').read_text())
assert s['status']=='PASS' and s['all_semantic_training_state_bitwise_equal']
assert s['parent_sequence_and_RNG_progression_equal_across_arms']
assert s['all_gradients_cuda_float32_finite_nonzero'] and s['internal_check_teacher_reads']==0
assert s['actual_total_updates']==48 and s['formal_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
assert s['train_source_sha256']==file_hash(module/'layout_endpoint/train.py')
assert s['rollout_source_sha256']==file_hash(module/'layout_endpoint/rollout.py')
d=json.loads((run/'diagnostics/SINGLE_OBJECT_DIAGNOSTIC.json').read_text())
assert d['updates']==200 and d['uid']==protocol['diagnostic']['uid']
assert d['diagnostic_weights_used_for_formal_training'] is False
assert 0<d['final_over_initial_deployment_endpoint_mse']<1
suite=ET.parse(run/'PYTEST_PRETRAIN.xml').getroot().findall('testsuite')
assert sum(int(t.attrib['tests']) for t in suite)==17
assert all(int(t.attrib[k])==0 for t in suite for k in ['failures','errors','skipped'])
source_files=['layout_endpoint/train.py','layout_endpoint/rollout.py','layout_student/train.py','layout_value/optimizer.py','legacy/networks.py','legacy/chart.py']
source_hashes={p:file_hash(module/p) for p in source_files}
write_json(run/'TRAINING_GATE.json',{'status':'APPROVED_FOR_PREDECLARED_V1_INITIALIZED_STUDY',
    'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'training_source_hashes':source_hashes,
    'pytest_passed':17,'pytest_failed':0,'pytest_skipped':0,'pytest_report_sha256':file_hash(run/'PYTEST_PRETRAIN.xml'),
    'cuda_resume_smoke_status':'PASS','cuda_resume_smoke_sha256':file_hash(smoke/'GPU_SMOKE.json'),
    'diagnostic_final_over_initial_endpoint_mse':d['final_over_initial_deployment_endpoint_mse'],
    'diagnostic_sha256':file_hash(run/'diagnostics/SINGLE_OBJECT_DIAGNOSTIC.json'),
    'actual_initializations_v1_only':True,'all_model_all_device_correctness_pass':False,
    'original_A_GPU_pressure_limit_retained':True,'implementation_gate_sha256':file_hash(run/'IMPLEMENTATION_GATE.json'),
    'final_original_CPU_deployment_authoritative':True,'no_changed_hyperparameters_or_labels':True,
    'prespecified_six_runs':[f'{a}_{seed}' for a in ['c0','c1','c2'] for seed in [10,11]]})
print(json.dumps({'training_gate':'APPROVED_FOR_PREDECLARED_V1_INITIALIZED_STUDY','diagnostic_endpoint_mse_ratio':d['final_over_initial_deployment_endpoint_mse'],'tests':17}))