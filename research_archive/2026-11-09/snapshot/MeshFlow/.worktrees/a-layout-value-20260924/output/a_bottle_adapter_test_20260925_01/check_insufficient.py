"""Expected refusal of the actual insufficient screen; no model execution."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from experiments.a_radial.time_quality_followup import bottle_ttq as driver


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


repo = Path(__file__).resolve().parents[2]
here = Path(__file__).resolve().parent
screen = repo / 'output/a_bottle_screen_20260925_01'
study = repo / 'output/a_bottle_study_20260925_01/STUDY_PROTOCOL_v2.json'
old_run = repo / 'output/a_time_quality16_20260925_01'
forbidden_output = here / 'FORBIDDEN_INSUFFICIENT_RUN'
assert not forbidden_output.exists()
selection = driver.read(screen / 'SELECTION.json')
complete = driver.read(screen / 'COMPLETE.json')
assert selection['status'] == complete['status'] == 'INSUFFICIENT_ELIGIBLE'
assert selection['eligible_count'] == 14 and selection['selected16'] == []
calls = []
original_load = driver.torch.load


def forbidden_load(*args, **kwargs):
    calls.append('torch.load')
    raise AssertionError('Preparation must refuse before loading any model')


driver.torch.load = forbidden_load
try:
    try:
        driver.prepare(screen, old_run, study, forbidden_output)
    except ValueError as exc:
        rejection = str(exc)
        assert 'sixteen eligible' in rejection
    else:
        raise AssertionError('Insufficient screen was accepted')
finally:
    driver.torch.load = original_load
assert not calls and not forbidden_output.exists() and not driver.torch.cuda.is_initialized()
suites = ET.parse(here / 'pytest.xml').getroot().findall('testsuite')
assert sum(int(s.attrib['tests']) for s in suites) == 29
assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
module = repo / 'experiments/a_radial/time_quality_followup'
paths = [module / n for n in ('bottle_ttq.py', 'bottle_runner.py', 'test_bottle_ttq.py',
                              'conditioned_deployment.py', 'test_conditioned_deployment.py')]
report = {
    'status': 'EXPECTED_INSUFFICIENT_SCREEN_REJECTION_PASS',
    'implementation_status': 'PROTOTYPE_NOT_FULLY_INTEGRATED_OR_FORMALLY_VALIDATED',
    'synthetic_tests_passed': 29, 'synthetic_tests_failed': 0, 'synthetic_tests_skipped': 0,
    'pytest_xml_sha256': sha(here / 'pytest.xml'),
    'actual_screen_status': selection['status'], 'eligible_count': 14, 'required_count': 16,
    'rejection': rejection, 'output_created': forbidden_output.exists(),
    'model_load_calls': len(calls), 'real_model_inferences': 0,
    'new_old16_model_gate_executed': False, 'bottle_model_inferences': 0,
    'optimizer_or_MC_calls': 0, 'formal144_started': False,
    'cuda_initialized': driver.torch.cuda.is_initialized(),
    'source_sha256': {str(p): sha(p) for p in paths},
    'input_sha256': {str(p): sha(p) for p in (study, screen / 'SELECTION.json', screen / 'COMPLETE.json')},
    'script_sha256': sha(__file__),
    'limits': 'Only synthetic routing/report/bookkeeping tests and the real insufficient-screen refusal ran. No successful bottle prepare path, full per-case integration gate, or bottle efficacy/timing experiment was executed.',
}
with (here / 'SUMMARY.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: report[k] for k in ('status', 'implementation_status', 'eligible_count',
    'model_load_calls', 'output_created', 'formal144_started')}))
