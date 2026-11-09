"""Prepare/gate/run/report the frozen bottle condition checked-TTQ study.

Preparation requires the complete 225-parent geometry-only screen. The only
preparation model operation is loading/hashing the already frozen shared weights.
The old-parent gate is explicitly not inference on the selected bottle objects.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, new_dir, save_npz, write_json
from ..layout_student.train import model_sha256
from ..time_quality16 import prepare as old_prepare
from ..time_quality16 import runner as old_runner
from ..time_quality16 import orchestrate as old_driver
from . import conditioned_deployment
from . import bottle_runner as adapter

ARMS = adapter.ARMS
STUDY_SHA = '350fcc259a7760fe0b200eea9dba23fc3a8c4f7aa611eaa49cca99f97d84a6e0'
require, read = old_prepare.require, old_prepare.read


def sources():
    bound = old_prepare.sources()
    radial_root = Path(__file__).resolve().parent.parent
    for dependency in sorted(radial_root.rglob('*.py')):
        if '__pycache__' not in dependency.parts:
            bound[str(dependency)] = file_hash(dependency)
    for name in ('__init__.py', 'conditioned_deployment.py', 'test_conditioned_deployment.py',
                 'bottle_runner.py', 'bottle_ttq.py', 'test_bottle_ttq.py'):
        path = Path(__file__).resolve().parent / name
        require(path.is_file(), 'Missing bottle adapter source: ' + name)
        bound[str(path)] = file_hash(path)
    return bound


def verify(protocol_path):
    p = read(protocol_path)
    adapter.validate_contract(p)
    for path, digest in p['bound_files_sha256'].items():
        require(file_hash(path) == digest, 'Bound source or input changed: ' + path)
    return p


def prepare(selection_run, old_run, study_path, out):
    selection_run, old_run = Path(selection_run).resolve(), Path(old_run).resolve()
    study_path = Path(study_path).resolve()
    require(file_hash(study_path) == STUDY_SHA, 'Registered v3 all14 study changed')
    study = read(study_path)
    require(study['model']['tensor_sha256'] == adapter.MODEL_SHA, 'Frozen model registration differs')
    selection = read(selection_run / 'SELECTION.json')
    require(selection['status'] == 'SELECTED_ALL14', 'Need the separately authorized all14 manifest')
    selection_complete = read(selection_run / 'COMPLETE.json')
    require(selection_complete['status'] == 'READY_ALL14_NO_MODEL_RUN' and
            selection_complete['selected_count'] == 14 and selection_complete['planned_cases'] == 42 and
            selection_complete['selection_sha256'] == file_hash(selection_run / 'SELECTION.json') and
            selection_complete['file_manifest_sha256'] == file_hash(selection_run / 'FILE_MANIFEST.json'),
            'All14 selection completion identity differs')
    screen_run = Path(selection['source_screen_run']).resolve()
    screen_selection = read(screen_run / 'SELECTION.json'); screen_complete = read(screen_run / 'COMPLETE.json')
    require(screen_selection['status'] == screen_complete['status'] == 'INSUFFICIENT_ELIGIBLE' and
            screen_selection['selected16'] == [] and screen_complete['eligible_count'] == 14 and
            screen_complete['screened_count'] == 225 and screen_complete['runtime_errors_preserved'] == 0 and
            screen_complete['model_optimizer_MC_calls'] == 0 and screen_complete['source_and_protocol_unchanged'] is True,
            'Original225 screen must remain the unchanged fourteen-eligible insufficient16 result')
    require(screen_complete['selection_sha256'] == selection['source_screen_selection_sha256'] == file_hash(screen_run / 'SELECTION.json') and
            screen_complete['file_manifest_sha256'] == selection['source_screen_file_manifest_sha256'] == file_hash(screen_run / 'FILE_MANIFEST.json') and
            selection['source_screen_complete_sha256'] == file_hash(screen_run / 'COMPLETE.json') and
            screen_complete['screen_protocol_sha256'] == selection['protocol_sha256'],
            'Original screen provenance chain changed')
    require(selection['root_study_protocol_sha256'] == STUDY_SHA, 'Selection did not use v3 authorization')
    manifests = {}
    for folder in (screen_run, selection_run):
        files = read(folder / 'FILE_MANIFEST.json')['files']; manifests[folder] = files
        require('SELECTION.json' in files, 'Selection not bound in manifest')
        for relative, digest in files.items():
            path = (folder / relative).resolve()
            require(path.is_relative_to(folder) and file_hash(path) == digest, 'Screen/selection evidence changed')
    uids = selection['lex_sorted_uids']
    require(len(uids) == 14 and uids == sorted(set(selection['selected_uids'])) and
            selection['selected14'] == selection['selected_uids'], 'Exactly all14 selected parents required')
    require(selection['selected_uids'] == screen_selection['eligible_uids_hash_order'] and
            set(uids) == set(screen_selection['eligible_uids_hash_order']), 'All14 must be used without subset selection')
    require(len(set(selection['train_uids'])) == 225 and set(uids) <= set(selection['train_uids']),
            'All14 must come from original bottle train225')
    parent_order = sorted(uids, key=lambda u: __import__('hashlib').sha256(('a-condition16-parent-v1:' + u).encode()).hexdigest())
    require(selection['selected_uids'] == parent_order, 'Registered parent hash order changed')
    donor_order = sorted(uids, key=lambda u: __import__('hashlib').sha256(('a-condition16-donor-v1:' + u).encode()).hexdigest())
    donors = {u: donor_order[(i + 1) % 14] for i, u in enumerate(donor_order)}
    require(selection['donor_uids_hash_order'] == donor_order and selection['donor_mapping'] == donors,
            'Registered all14 cyclic condition mapping changed')
    old_path = old_run / 'RUN_PROTOCOL.json'; original = read(old_path)
    model = original['model_bindings']['early_v1_10']
    require(model['model_sha256'] == adapter.MODEL_SHA and model['file_sha256'] == study['model']['file_sha256'] and
            file_hash(model['path']) == model['file_sha256'], 'Source shared3k model identity differs')
    begin = time.perf_counter()
    state = torch.load(model['path'], map_location='cpu', weights_only=True)
    require(set(state) == {'model'} and model_sha256(state['model']) == adapter.MODEL_SHA,
            'Expected only frozen3k model tensors')
    model_read_hash_seconds = time.perf_counter() - begin
    support = RadialSupport(); inputs = {}
    for uid in uids:
        binding = selection['data'][uid]; path = old_runner.path_from(selection_run, binding['path'])
        require(file_hash(path) == binding['sha256'], 'Selected data changed')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {'logr', 'base_u', 'faces', 'ref_vertices', 'ref_faces'}, 'Target archive keys changed')
            d = {k: np.array(archive[k]) for k in archive.files}
        require(d['logr'].shape == (2050,) and d['logr'].dtype == np.float32 and
                d['base_u'].shape == (258, 3) and d['base_u'].dtype == np.float64 and
                d['faces'].shape == (512, 3) and d['faces'].dtype == np.int32 and
                all(np.isfinite(v).all() for v in d.values()), 'Target schema or values changed')
        rv, rf = support.mesh(d['logr'])
        require(np.array_equal(rv, d['ref_vertices']) and np.array_equal(rf, d['ref_faces']) and
                array_hash(d['logr']) == binding['mainfield_hash'], 'Target same-field reference changed')
        inputs[uid] = (path, d)
    require(not torch.cuda.is_initialized(), 'CPU-only preparation')
    bound = sources()
    for folder, files in manifests.items():
        for relative, digest in files.items(): bound[str((folder / relative).resolve())] = digest
        for name in ('COMPLETE.json', 'FILE_MANIFEST.json'): bound[str(folder / name)] = file_hash(folder / name)
    for path in (study_path, old_path, Path(selection['protocol_path'])): bound[str(path)] = file_hash(path)
    require(file_hash(selection['protocol_path']) == selection['protocol_sha256'], 'Original screen protocol changed')
    out = new_dir(out).resolve()
    for name in ('models', 'data', 'conditions'): (out / name).mkdir()
    begin = time.perf_counter(); copied_model = out / 'models/shared_early_v1_10.pt'
    shutil.copyfile(model['path'], copied_model)
    require(file_hash(copied_model) == model['file_sha256'], 'Model bytecopy differs')
    model_copy_seconds = time.perf_counter() - begin
    shared = {'path': str(copied_model), 'file_sha256': model['file_sha256'], 'model_sha256': adapter.MODEL_SHA,
              'parameter_count': 29907, 'training_updates': 3000, 'seed': 10, 'source': model}
    data, conditions = {}, {}
    for uid, (source, d) in inputs.items():
        target = out / 'data' / (uid + '.npz'); shutil.copyfile(source, target)
        require(file_hash(target) == selection['data'][uid]['sha256'], 'Data bytecopy differs')
        data[uid] = {**selection['data'][uid], 'path': str(target), 'sha256': file_hash(target),
            'source_prepared_path': str(source), 'split': 'bottle_train_cross_category_validation',
            'base_hash': array_hash(d['base_u']), 'faces_hash': array_hash(d['faces'])}
        condition = out / 'conditions' / (uid + '.npz'); save_npz(condition, logr=d['logr'])
        conditions[uid] = {'path': str(condition), 'sha256': file_hash(condition),
            'mainfield_hash': array_hash(d['logr']), 'dtype': 'float32', 'keys': ['logr']}
        for path in (target, condition): bound[str(path)] = file_hash(path)
    bound[str(copied_model)] = model['file_sha256']; bound[model['path']] = model['file_sha256']
    p = copy.deepcopy(original)
    for key in ('all16_development_seen', 'no_seed11_cherry_picking_or_cross_seed_claim'): p.pop(key, None)
    p.update(schema=adapter.SCHEMA, created_utc=old_driver.utc(), run_root=str(out),
        scope='ALL14_ELIGIBLE_BOTTLE_CROSS_CATEGORY_CONDITION_DIAGNOSTIC_SINGLE_MEASUREMENT',
        uids=uids, all16_sorted_uids=uids, arms=list(ARMS), data=data, conditions=conditions,
        donor_mapping=donors, case_order=adapter.case_order(uids), repeats=1, expected_cases=42,
        model_bindings={arm: copy.deepcopy(shared) for arm in ARMS[1:]},
        primary_budget_seconds=4, auxiliary_budgets_seconds=[2, 8], failure_denominator=14,
        source_category='03691459', target_category='02876657', study_protocol_path=str(study_path),
        study_protocol_sha256=STUDY_SHA, screen_run=str(screen_run), selection_run=str(selection_run),
        selection_sha256=file_hash(selection_run / 'SELECTION.json'), source_screen_status='INSUFFICIENT_ELIGIBLE',
        user_amendment='use all14 eligible parents and one timing run per arm; 42 total; no other numerical changes',
        legacy_all16_sorted_uids_key_is_only_seed_index_not_current_count=True,
        single_measurement_no_repeat_variability_estimate=True, geometry_must_match_across_repeats=False,
        offline_screening_and_field_preparation_excluded_from_case_time=True,
        original_A_pretraining_sample_manifest_not_independently_reconstructed=True,
        field_reference_is_raw_asset=False, correct_and_mismatched_condition_preparation_identical=True,
        condition_mismatch_changes_only_network_proxy=True,
        target_seed_rule='registered base + lexically sorted target index; never donor index',
        model_conversion_seconds=model_read_hash_seconds + model_copy_seconds,
        source_load_validate_extract_seconds=model_read_hash_seconds, model_serialization_hash_seconds=model_copy_seconds,
        timing_scope='warm-process; target/support/shared model-only load, independent condition NPZ read/proxy32, baseline audit, full optimizer/restoration and earlier six observation audits, final target query',
        bound_files_sha256=bound)
    adapter.validate_contract(p)
    write_json(out / 'RUN_PROTOCOL.json', p)
    write_json(out / 'PREPARED.json', {'status': 'READY_FOR_TEST_GATE_NOT_STARTED',
        'protocol_sha256': file_hash(out / 'RUN_PROTOCOL.json'), 'cuda_initialized': False, 'bottle_model_inferences': 0})
    return p


def make_gate(protocol_path, old_gate_path, tests_xml, out):
    p = verify(protocol_path); evidence = read(old_gate_path)
    require(evidence['status'] == 'PASS' and evidence['correct_condition_exact_parity_count'] == 16 and
            evidence['real_mismatched_target_radius_check']['target_query_exact'] is True and
            evidence['real_mismatched_target_radius_check']['donor_radius_negative_control_rejected'] is True and
            evidence['wrapper_sha256'] == file_hash(conditioned_deployment.__file__) and
            evidence['model_tensor_sha256'] == adapter.MODEL_SHA, 'Existing frozen wrapper implementation gate missing')
    suites = ET.parse(tests_xml).getroot()
    nodes = [suites] if suites.tag == 'testsuite' else list(suites.findall('testsuite'))
    count = sum(int(n.attrib.get('tests', 0)) for n in nodes)
    require(count > 0 and all(int(n.attrib.get('failures', 0)) == int(n.attrib.get('errors', 0)) == 0 and
            int(n.attrib.get('skipped', 0)) == 0 for n in nodes), 'Current CPU tests not fully passed')
    gate = {'status': 'PASS', 'protocol_sha256': file_hash(protocol_path), 'source_sha256': sources(),
        'old_parent_gate': str(Path(old_gate_path).resolve()), 'old_parent_gate_sha256': file_hash(old_gate_path),
        'tests_xml': str(Path(tests_xml).resolve()), 'tests_xml_sha256': file_hash(tests_xml),
        'passed_tests': count, 'new_bottle_inference_count': 0, 'expected_cases': p['expected_cases'],
        'gate_scope': 'new14 single-run bookkeeping/routing tests plus existing unchanged wrapper real16 parity; no new old16 experiment'}
    write_json(Path(out), gate)
    return gate


def run(protocol_path, gate_path):
    protocol_path, gate_path = Path(protocol_path).resolve(), Path(gate_path).resolve()
    root = protocol_path.parent; p = verify(protocol_path); gate = read(gate_path)
    require(not (root / 'STARTED.json').exists(), 'An attempted benchmark cannot restart or overwrite')
    require(gate['status'] == 'PASS' and gate['protocol_sha256'] == file_hash(protocol_path), 'Test gate mismatched')
    for path, digest in gate['source_sha256'].items(): require(file_hash(path) == digest, 'Tested code changed')
    require(file_hash(gate['old_parent_gate']) == gate['old_parent_gate_sha256'] and
            file_hash(gate['tests_xml']) == gate['tests_xml_sha256'], 'Gate evidence changed')
    warm_process_record = old_runner.warm_process()
    write_json(root / 'STARTED.json', {'utc': old_driver.utc(), 'pid': os.getpid(),
        'protocol_sha256': file_hash(protocol_path), 'gate_sha256': file_hash(gate_path),
        'warm_process_record': warm_process_record, 'no_training_or_official_queue_interaction': True})
    before = time.perf_counter(); completed = []
    try:
        old_driver.wait_ready(root, p, quiet_samples=3)
        warm = old_driver.warmup(root, p)  # Synthetic surface only; separately timed once.
        old_driver.wait_ready(root, p, quiet_samples=3)
        for spec in p['case_order']:
            host_before = old_driver.wait_ready(root, p)
            folder = adapter.case_folder(root, spec['uid'], spec['arm'], spec['repeat'])
            task = adapter.run_case(protocol_path, spec['uid'], spec['arm'], spec['repeat'], folder)
            host_after = old_driver.host_status(); old_driver.append_jsonl(root / 'HOST_SAMPLES.jsonl', host_after)
            resources = {'before': host_before, 'after': host_after,
                'boundary_resource_gate_pass': old_driver.host_ready(host_before, p) and old_driver.host_ready(host_after, p),
                'continuous_host_isolation_not_certified': True, 'host_probes_outside_case_wall': True}
            write_json(folder / 'RESOURCES.json', resources)
            completed.append({**spec, 'task_path': str(folder / 'TASK.json'),
                'task_sha256': file_hash(folder / 'TASK.json'), 'status': task['status'],
                'boundary_resource_gate_pass': resources['boundary_resource_gate_pass']})
            progress = {'status': 'BENCHMARK_RUNNING', 'completed_cases': len(completed), 'planned_cases': p['expected_cases'],
                'last_case': completed[-1], 'elapsed_driver_seconds': time.perf_counter() - before, 'utc': old_driver.utc()}
            write_json(root / 'PROGRESS.json', progress, replace=True)
            print(json.dumps(progress), flush=True)
            adapter.build_report(root, partial=True)
        verify(protocol_path); summary = adapter.build_report(root, partial=False)
        final = {'status': 'ALL_PLANNED_ATTEMPTS_COMPLETE', 'cases': completed,
            'protocol_sha256': file_hash(protocol_path), 'total_driver_wall_seconds': time.perf_counter() - before,
            'one_time_warmup_seconds': warm['elapsed_seconds'],
            'all_boundary_resource_gates_pass': all(v['boundary_resource_gate_pass'] for v in completed),
            'cuda_initialized': torch.cuda.is_initialized(), 'source_inputs_unchanged': True,
            'scientific_speedup_claim': False, 'finished_utc': old_driver.utc(), 'summary_status': summary['status']}
        write_json(root / 'COMPLETE.json', final)
        write_json(root / 'PROGRESS.json', {k: v for k, v in final.items() if k != 'cases'}, replace=True)
        return final
    except BaseException as exc:
        failure = {'status': 'DRIVER_INTERRUPTED', 'error': repr(exc), 'traceback': traceback.format_exc(),
            'completed_cases': completed, 'all_original_attempts_retained': True, 'no_automatic_retry': True,
            'utc': old_driver.utc()}
        write_json(root / 'FAILED.json', failure); write_json(root / 'PROGRESS.json', failure, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('selection-run', 'old-run', 'study', 'out'): p.add_argument('--' + name, required=True)
    p = sub.add_parser('gate')
    for name in ('protocol', 'old-gate', 'tests-xml', 'out'): p.add_argument('--' + name, required=True)
    p = sub.add_parser('run'); p.add_argument('--protocol', required=True); p.add_argument('--gate', required=True)
    p = sub.add_parser('report'); p.add_argument('--run', required=True); p.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.selection_run, args.old_run, args.study, args.out)
    elif args.command == 'gate': result = make_gate(args.protocol, args.old_gate, args.tests_xml, args.out)
    elif args.command == 'run': result = run(args.protocol, args.gate)
    else: result = adapter.build_report(args.run, partial=args.partial)
    print(json.dumps({'status': result.get('status', 'PREPARED'), 'new_training_updates': 0}), flush=True)


if __name__ == '__main__':
    main()
