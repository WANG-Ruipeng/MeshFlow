import json
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from experiments.a_radial.time_quality16.prepare import verify
from experiments.a_radial.time_quality16.runner import run_case
from experiments.a_radial.time_quality16.report import summarize_case, build_report
from experiments.a_radial.io_utils import file_hash, write_json

repo=Path.cwd()
gate=repo/'output/a_ttq_test_gate_20260925_01'
run=repo/'output/a_time_quality16_20260925_01'
protocol=run/'RUN_PROTOCOL.json'
p=verify(protocol)
x=ET.parse(gate/'pytest.xml').getroot()
suites=list(x.iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites)==29
assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
case=run_case(protocol,p['uids'][0],'fixed',0,gate/'real_fixed_integration')
assert case['status']=='completed',case['failure']
assert case['completed_full_budget'] and case['timings']['completed_full_audit_calls']==7
assert case['final_return_matches_audited_grid16_bitwise']
summary=summarize_case(case)
assert not summary['technical_issues']
assert len(summary['observations'])==6
partial=build_report(run,partial=True)
assert partial['recorded_cases']==0
verify(protocol)
write_json(run/'GATE.json',{'status':'PASS','protocol_sha256':file_hash(protocol),'source_sha256':p['bound_files_sha256'],'pytest_xml':str(gate/'pytest.xml'),'pytest_sha256':file_hash(gate/'pytest.xml'),'tests_passed':29,'real_fixed_integration':str(gate/'real_fixed_integration/TASK.json'),'real_fixed_integration_sha256':file_hash(gate/'real_fixed_integration/TASK.json'),'real_fixed_case_wall_seconds':case['timings']['total_end_to_end_seconds'],'gate_is_not_a_formal_timing_replicate':True})
print(json.dumps({'status':'PASS','tests':29,'case_status':case['status'],'real_case_seconds':case['timings']['total_end_to_end_seconds']}),flush=True)