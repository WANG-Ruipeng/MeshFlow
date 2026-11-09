from pathlib import Path
import json, hashlib, datetime, xml.etree.ElementTree as ET
run=Path(__file__).resolve().parent
read=lambda p:json.loads((run/p).read_text(encoding='utf-8-sig'))
hashfile=lambda p:hashlib.sha256((run/p).read_bytes()).hexdigest()
suites=ET.parse(run/'PYTEST_FINAL.xml').getroot().findall('testsuite')
tests={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
assert tests['tests']==25 and all(tests[k]==0 for k in ('failures','errors','skipped'))
training=read('TRAINING_COMPLETE.json')
assert training['status']=='COMPLETE' and len(training['jobs'])==6 and training['optimizer_updates_total']==6000
assert training['protocol_sha256']==hashfile('RUN_PROTOCOL.json')
ledger=read('results.json'); summary=read('SUMMARY.json'); parity=read('FINAL_DEPLOYMENT_PARITY.json')
assert ledger['expected_attempt_count']==160 and ledger['expected_row_count']==288
assert parity['results_sha256']==hashfile('results.json')
assert summary['correctness_scope']['final_deployment_parity']['status']==parity['status']
record={
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'unit_tests':{**tests,'xml_sha256':hashfile('PYTEST_FINAL.xml')},
 'formal_training':{'jobs':6,'updates':6000,'status':'COMPLETE','total_serial_wall_seconds':training['elapsed_wall_seconds'],'sha256':hashfile('TRAINING_COMPLETE.json')},
 'scoped_training_gate':read('TRAINING_GATE.json'),
 'strict_all_models_all_devices_correctness_pass':False,
 'evaluation':{'attempt_count':ledger['attempt_count'],'planned_attempts':160,'saved_rows':len(ledger['rows']),'planned_rows':288,'error_attempts':len(ledger['errors']),'missing_success_rows':len(ledger['missing_success_rows']),'sha256':hashfile('results.json')},
 'final_deployment_parity':{'status':parity['status'],'statistics':parity['statistics'],'sha256':hashfile('FINAL_DEPLOYMENT_PARITY.json')},
 'continuation_gate':summary['continuation_gate'],
 'figures_manifest_sha256':hashfile('figures/FIGURE_MANIFEST.json'),
 'scientific_success_not_inferred_from_software_checks':True,
 'archive_integrity_and_fresh_replay_reported_separately':True,
}
with (run/'CHECKS.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'tests':tests,'evaluation':record['evaluation'],'continuation_gate':record['continuation_gate']},indent=2))