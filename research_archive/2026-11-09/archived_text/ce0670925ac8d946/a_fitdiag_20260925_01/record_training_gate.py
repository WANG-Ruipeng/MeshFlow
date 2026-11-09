from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as ET
run=Path(__file__).resolve().parent
root=run.parent.parent/'experiments/a_radial'
smoke_run=run.parent/'a_fitdiag_smoke_20260925_01'
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
smoke=read(smoke_run/'GPU_SMOKE.json');p=read(run/'RUN_PROTOCOL.json')
assert smoke['status']=='PASS' and smoke['device']=='cuda'
assert smoke['formal_protocol_sha256']==hashfile(run/'RUN_PROTOCOL.json')
assert smoke['full_semantic_training_state_bitwise_equal'] and smoke['manual_frozen_C0_parent_next_update_bitwise_equal']
assert smoke['internal_check_teacher_reads']==0 and smoke['actual_total_optimizer_updates']==68
assert smoke['train_source_sha256']==hashfile(root/'layout_fitdiag/train.py')
assert smoke['smoke_source_sha256']==hashfile(root/'layout_fitdiag/smoke.py')
suites=ET.parse(run/'PYTEST_PRETRAIN.xml').getroot().findall('testsuite')
tests={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
assert tests['tests']==17 and all(tests[k]==0 for k in ('failures','errors','skipped'))
paths=['layout_fitdiag/train.py','layout_fitdiag/smoke.py','layout_fitdiag/run_all.py','layout_endpoint/train.py','layout_endpoint/rollout.py','layout_student/train.py','layout_value/optimizer.py','legacy/networks.py','legacy/chart.py']
record={'status':'APPROVED_FIT_DIAG_FIXED_BUDGET','protocol_sha256':hashfile(run/'RUN_PROTOCOL.json'),
 'cuda_resume_smoke_sha256':hashfile(smoke_run/'GPU_SMOKE.json'),'smoke_run':str(smoke_run),
 'unit_tests_xml_sha256':hashfile(run/'PYTEST_PRETRAIN.xml'),'unit_tests':tests,
 'training_source_hashes':{rel:hashfile(root/rel) for rel in paths},
 'full_parent_next_update_bitwise_equal':True,'continuous_resume_full_state_bitwise_equal':True,
 'internal_check_teacher_reads':0,'scope':'fit diagnosis only; no scientific success inferred',
 'previous_endpoint_study_untouched_and_scientific_gate_still_failed':True,
 'approved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (run/'TRAINING_GATE.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':record['status'],'tests':tests,'cuda_updates':68,'training_source_files':len(paths)}))