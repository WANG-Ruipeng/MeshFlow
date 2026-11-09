"""Read-only provenance audit and exclusive archive for the fixed 240k joint fit."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime, hashlib, json, os
from pathlib import Path, PurePosixPath
import shutil, subprocess, sys, zipfile
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..verify_repo import verify
from ..layout_student.integrity import mesh_identity
from ..layout_student.package_results import inside, mesh_arrays, verify_manifest, write_obj
from ..layout_student.prepare import split_uids
from ..layout_student.train import model_sha256
from ..layout_endpoint.package_results import exact_equal, compare_prior_mesh
from ..layout_fitdiag.package_results import check_mesh, check_same_geometry
from .prepare import read_json, require
from .train import config_digest, validate_parent
from .diagnose import expected_slots as diagnostic_slots, checkpoint_path, endpoint_probe
from .evaluate import MODEL_NAMES, POLICIES, SEED_BASES, POINTS, expected_slots as final_slots


def final_accounting(protocol, ledger, attempts):
    expected = final_slots(protocol)
    expected_attempts = {(uid, model) for uid, model, _ in expected}
    key = lambda r: (r['uid'], r['model'], r['policy'])
    ids = [(a['uid'], a['model']) for a in attempts]
    require(len(ids) == len(set(ids)) and set(ids) <= expected_attempts, 'Duplicate/unregistered final attempt')
    flattened, errors = [], []
    for attempt in attempts:
        require(attempt['status'] in ('success', 'error'), 'Invalid final attempt status')
        if attempt['status'] == 'success':
            rows = attempt['rows']
            require(len(rows) == len(POLICIES[attempt['model']]) and
                    {r['policy'] for r in rows} == set(POLICIES[attempt['model']]), 'Lost/duplicate final policy')
            require(all(r['uid'] == attempt['uid'] and r['model'] == attempt['model'] and r['status'] == 'success'
                        for r in rows), 'Final attempt/row identity mismatch')
            flattened.extend(rows)
        else:
            require(not attempt.get('rows'), 'Error attempt contains successful rows')
            errors.append(attempt)
    rows = ledger['rows']; keys = [key(r) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate final ledger row')
    require(sorted(rows, key=key) == sorted(flattened, key=key), 'Final aggregate lost/changed rows')
    akey = lambda a: (a['uid'], a['model'])
    require(sorted(ledger['errors'], key=akey) == sorted(errors, key=akey), 'Final aggregate lost failures')
    failed = {(a['uid'], a['model'], p) for a in errors for p in POLICIES[a['model']]}
    missing = expected_attempts - set(ids)
    require(set(keys) | failed | {(u, m, p) for u, m in missing for p in POLICIES[m]} == expected,
            'Final planned slots disappeared')
    require(ledger['attempt_count'] == len(attempts) and ledger['expected_attempt_count'] == len(expected_attempts)
            and ledger['expected_row_count'] == len(expected), 'Final denominator changed')
    require({tuple(x) for x in ledger['failed_row_slots']} == failed, 'Final failed slots changed')
    require({tuple(x) for x in ledger['unattempted']} == missing, 'Final missing attempts lost')
    require({tuple(x) for x in ledger['missing_success_rows']} == expected - set(keys), 'Final missing rows lost')
    return {'planned_attempts': len(expected_attempts), 'planned_rows': len(expected),
        'observed_attempts': len(attempts), 'success_rows': len(rows), 'error_attempts': len(errors),
        'unattempted': sorted(missing), 'missing_success_rows': sorted(expected - set(keys)),
        'all_planned_slots_accounted': True}

def diagnostic_accounting(protocol, ledger, attempts):
    expected = diagnostic_slots(protocol)
    key = lambda r: (r['job'], r['checkpoint'], r['uid'])
    keys = [key(a) for a in attempts]
    require(len(keys) == len(set(keys)) and set(keys) <= expected, 'Duplicate/unregistered diagnostic attempt')
    rows, errors = [], []
    for attempt in attempts:
        if attempt['status'] == 'success':
            require(key(attempt['row']) == key(attempt) and attempt['row']['split'] == 'train', 'Diagnostic identity/split changed')
            rows.append(attempt['row'])
        else:
            require(attempt['status'] == 'error' and not attempt.get('row'), 'Invalid diagnostic failure')
            errors.append(attempt)
    require(sorted(ledger['rows'], key=key) == sorted(rows, key=key), 'Diagnostic aggregate lost/changed rows')
    require(sorted(ledger['errors'], key=key) == sorted(errors, key=key), 'Diagnostic aggregate lost failures')
    require(len({key(r) for r in ledger['rows']}) == len(rows), 'Duplicate diagnostic row')
    require(ledger['expected_attempt_count'] == len(expected) and ledger['attempt_count'] == len(attempts), 'Diagnostic denominator changed')
    require({tuple(x) for x in ledger['unattempted']} == expected - set(keys), 'Diagnostic missing attempts lost')
    missing_success = expected - {key(r) for r in rows}
    require({tuple(x) for x in ledger['missing_success_slots']} == missing_success, 'Diagnostic missing slots lost')
    require(ledger['checkpoint_selection'] is False and ledger['internal_check_objects_opened'] == 0
            and ledger['surface_rms_evaluated'] is False, 'Intermediate diagnostic scope changed')
    return {'planned_slots': len(expected), 'observed_attempts': len(attempts), 'success_rows': len(rows),
        'error_attempts': len(errors), 'unattempted': sorted(expected - set(keys)),
        'missing_success_slots': sorted(missing_success), 'train_only': True, 'all_slots_accounted': True}

def training_audit(run, protocol, manifest):
    root = Path(__file__).resolve().parents[1]; ph = file_hash(run / 'RUN_PROTOCOL.json')
    training, models, observation_models = {}, {}, {}
    for name, job in protocol['jobs'].items():
        folder = run / 'training' / name; path = folder / 'final.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        report = read_json(folder / 'TRAIN_REPORT.json'); config = state['config']
        initial = inside(run, job['initial_checkpoint']); initial_sha = file_hash(initial)
        require(initial_sha == job['initial_checkpoint_sha256'] == manifest[job['initial_checkpoint']], 'Initial source changed')
        source = torch.load(initial, map_location='cpu', weights_only=False)
        joint = job['kind'] == 'joint'; active = job['active_training_uids']; trace = state['trace']
        require(config == read_json(folder / 'TRAIN_CONFIG.json'), 'Final training config differs')
        for key, expected in [('job', name), ('job_id', name), ('kind', job['kind']), ('seed', job['seed']),
                ('arm', 'c0'), ('total_updates', 240000), ('start_update', job['start_update']), ('batch_size', 4),
                ('parameter_count', 29907), ('learning_rate', 1e-4), ('weight_decay', 1e-4), ('rollout_steps', 32)]:
            require(config[key] == expected, 'Final config differs: ' + name + ':' + key)
        require(state['job_id'] == name and state['update'] == 240000 and state['is_predeclared_final'] is True
                and state['diagnostic_only'] is True, 'Not the predeclared diagnostic final')
        require(config['train_uids'] == protocol['train_uids'] and config['active_training_uids'] == active
                and not config['network_extra_inputs'], 'Training input/active parents changed')
        require(config['protocol_sha256'] == state['protocol_sha256'] == ph
                and config['config_sha256'] == protocol['hashes']['config_sha256'], 'Training protocol changed')
        require(state['training_labels_sha256'] == config['training_labels_sha256'] == protocol['hashes']['training_labels_sha256'], 'Training labels changed')
        require(state['initial_checkpoint_sha256'] == config['initial_checkpoint_sha256'] == initial_sha
                and config['initial_model_sha256'] == source['model_sha256'], 'Training initializer changed')
        require(config['initialization_mode'] == job['initialization_mode'], 'Optimizer initialization changed')
        for rel, digest in config['source_sha256'].items():
            require(file_hash(inside(root, rel)) == digest, 'Frozen training source changed: ' + rel)
        require(trace == read_json(folder / 'TRAIN_TRACE.json') and len(trace) == 240000
                and [r['update'] for r in trace] == list(range(1, 240001)), 'Final trace incomplete')
        require(state['timing_trace'] == read_json(folder / 'TRAIN_TIMING.json'), 'Timing trace differs')
        exposures = Counter()
        for row in trace:
            require(len(row['uids']) == 4 and set(row['uids']) <= set(active), 'Check/unknown parent entered training')
            require(len(row['rng_matched_k']) == 4 and all(isinstance(k, int) and 0 <= k < 32 for k in row['rng_matched_k']), 'Invalid bridge time draws')
            require(np.isfinite(row['loss']) and np.isfinite(row['gradient_l2']) and row['lr'] == 1e-4, 'Invalid loss/gradient/rate')
            exposures.update(row['uids'])
        require(sum(exposures.values()) == 960000 and set(exposures) == set(active), 'Exposure count differs')
        require(set(exposures.values()) <= {80000}, 'Exposure balance differs')
        require(state['sampler']['exposures'] == [exposures[u] for u in active]
                and state['sampler']['count'] == len(active), 'Sampler exposure differs')
        require(report['parent_exposures'] == {u: exposures[u] for u in protocol['train_uids']}, 'Reported exposure differs')
        require({'optimizer', 'scheduler', 'rng', 'sampler'} <= state.keys(), 'Incomplete resumable state')
        require({'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= state['rng'].keys() and len(state['rng']['torch_cuda']) > 0, 'Incomplete CUDA RNG')
        require(state['optimizer']['state'] and all(float(v['step']) == 240000 for v in state['optimizer']['state'].values()), 'Adam update count differs')
        adam_ids = [pid for group in state['optimizer']['param_groups'] for pid in group['params']]
        require(len(adam_ids) == len(set(adam_ids)) and set(adam_ids) == set(state['optimizer']['state']), 'Incomplete Adam parameter state')
        adam = list(state['optimizer']['state'].values())
        require(sum(value['exp_avg'].numel() for value in adam) == 29907, 'Adam does not cover every network parameter')
        require(all(value['exp_avg'].shape == value['exp_avg_sq'].shape and
                    torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                    for value in adam), 'Invalid Adam moment state')
        require(state['scheduler']['last_epoch'] == 240000, 'Scheduler cursor differs')
        for group in state['optimizer']['param_groups']:
            require(group['lr'] == 1e-4 and group['weight_decay'] == 1e-4 and group['foreach'] is False and group['fused'] is False, 'Adam implementation changed')
        require(all(torch.isfinite(v).all() for v in state['model'].values()), 'Nonfinite final model')
        import_record = read_json(folder / 'IMPORT_RECORD.json')
        require(import_record['source_checkpoint'] == job['initial_checkpoint'] and import_record['source_checkpoint_sha256'] == initial_sha
                and import_record['source_model_sha256'] == source['model_sha256'], 'Source import identity differs')
        require(import_record['source_update'] == import_record['parent_trace_rows'] == job['start_update'] and import_record['initialization_mode'] == job['initialization_mode'], 'Import cursor/mode differs')
        require(all(import_record[key] is joint for key in ('optimizer_state_inherited', 'rng_state_inherited', 'sampler_state_inherited')), 'Import flags differ')
        require(import_record['initial_is_a_snapshot_not_selected_result'], 'Initial snapshot relabeled as result')
        require(file_hash(folder / 'initial.pt') == initial_sha, 'Initial snapshot is not byte-identical')
        if joint:
            parent_sha = validate_parent(source, job, protocol, run, manifest)
            require(state['parent_protocol_sha256'] == config['parent_protocol_sha256'] == import_record['source_protocol_sha256'] == parent_sha, 'Parent protocol differs')
            require(exact_equal(trace[:20000], source['trace']) and exact_equal(state['timing_trace'][:len(source['timing_trace'])], source['timing_trace']), 'Imported parent trace changed')
            require(file_hash(folder / 'step_20000.pt') == initial_sha, 'Parent20000 observation not byte-identical')
        else:
            require(state['parent_protocol_sha256'] is None and config['parent_protocol_sha256'] is None
                    and import_record['source_protocol_sha256'] is None, 'Fresh single job inherited parent state')
        source_counts = Counter(uid for row in source['trace'] for uid in row['uids'])
        new_counts = {uid: exposures[uid] - source_counts[uid] for uid in active}
        require(report['global_parent_exposures_total'] == 960000 and report['imported_parent_exposures'] == dict(source_counts), 'Reported source/global exposures differ')
        require(report['new_parent_exposures_since_import'] == new_counts and sum(new_counts.values()) == 880000 and set(new_counts.values()) == {73333, 73334}, 'Reported new exposures differ')
        require(report['new_optimizer_updates_since_import'] == 220000 and report['parent_exposure_matches_single20k_at_declared_final'], 'Matched-exposure claim differs')
        sha, mh = file_hash(path), model_sha256(state['model'])
        require(mh == state['model_sha256'] == report['model_sha256'] and sha == report['checkpoint_sha256'], 'Final model/checkpoint identity differs')
        require(report['status'] == 'COMPLETE' and report['update'] == report['final_budget'] == 240000
                and report['imported_parent_updates'] == job['start_update'], 'Training completion report differs')
        require(report['internal_check_teacher_files_opened'] == 0 and report['selection_uses_internal_check'] is False, 'Check label isolation failed')
        models[name] = {'checkpoint_sha256': sha, 'model_sha256': mh}
        training[name] = {**models[name], 'updates': 240000, 'source_import_update': job['start_update'],
            'parent_exposures': dict(exposures), 'full_resumable_state_verified': True,
            'parent_trace_preserved_exactly': joint, 'initial_source_sha256': initial_sha}
        for observation in ('initial', '20000', '50000', '100000', '240000'):
            cp = checkpoint_path(run, name, observation)
            imported = observation == 'initial' or (joint and observation == '20000')
            observed = source if imported else state if observation == '240000' else torch.load(cp, map_location='cpu', weights_only=False)
            cp_hash = file_hash(cp)
            require(observed['model_sha256'] == model_sha256(observed['model']), 'Diagnostic snapshot model changed')
            if imported:
                require(cp_hash == initial_sha, 'Imported observation changed')
            else:
                update = int(observation)
                require(observed['update'] == update and observed['protocol_sha256'] == ph
                        and observed['config'] == config and observed['job_id'] == name, 'Observation checkpoint identity differs')
                require(observed['is_predeclared_final'] is (update == 240000), 'Observation final marker differs')
                require(exact_equal(observed['trace'], trace[:update]), 'Observation trace is not a prefix of final')
            observation_models[(name, observation)] = {'checkpoint_file_sha256': cp_hash,
                'model_sha256': observed['model_sha256'], 'import_record_sha256': file_hash(folder / 'IMPORT_RECORD.json')}
    for name in protocol['evaluation']['source_checkpoint_sha256']:
        path = inside(run, protocol['evaluation']['model_checkpoints'][name])
        source = torch.load(path, map_location='cpu', weights_only=False)
        digest = file_hash(path)
        require(digest == protocol['evaluation']['source_checkpoint_sha256'][name], 'Frozen reference checkpoint changed')
        require(source['update'] == 20000 and source['is_predeclared_final'] and source['model_sha256'] == model_sha256(source['model']), 'Wrong frozen reference model')
        require(source['model_sha256'] == protocol['evaluation']['source_model_sha256'][name], 'Reference tensor identity changed')
        models[name] = {'checkpoint_sha256': digest, 'model_sha256': source['model_sha256']}
    return training, models, observation_models

def successful_test_count(record):
    """Accept the two explicit root test-summary schemas, reject incomplete runs."""
    if 'tests' in record:
        total, failed = record['tests'], record['failures']
    else:
        total = sum(record[k] for k in ('passed', 'failed', 'errors', 'skipped'))
        failed = record['failed']
    require(total > 0 and failed == record['errors'] == record['skipped'] == 0, 'Pretrain unit checks incomplete')
    return total


def gate_and_documents_audit(run, smoke_run, protocol):
    root = Path(__file__).resolve().parents[1]; ph = file_hash(run / 'RUN_PROTOCOL.json')
    gate = read_json(run / 'TRAINING_GATE.json'); smoke = read_json(smoke_run / 'GPU_SMOKE.json')
    require(gate['status'] == 'APPROVED_LONGFIT_MATCHED_EXPOSURE' and gate['protocol_sha256'] == ph, 'Training gate differs')
    require(gate['cuda_resume_smoke_sha256'] == file_hash(smoke_run / 'GPU_SMOKE.json'), 'CUDA smoke binding differs')
    require(gate['full_parent_next_update_bitwise_equal'] and gate['continuous_resume_full_state_bitwise_equal']
            and gate['internal_check_teacher_reads'] == 0, 'Scoped training implementation gate failed')
    for rel, digest in gate['training_source_hashes'].items():
        require(file_hash(inside(root, rel)) == digest, 'Training-gate source changed: ' + rel)
    require(smoke['status'] == 'PASS' and smoke['device'] == 'cuda' and smoke['formal_protocol_sha256'] == ph, 'Smoke protocol/device differs')
    require(smoke['actual_total_optimizer_updates'] == 36 and smoke['continuous_new_updates'] == 8
            and smoke['split_new_updates'] == [3, 5], 'Smoke execution budget differs')
    require(smoke['full_semantic_training_state_bitwise_equal'] and smoke['manual_frozen_C0_parent_next_update_bitwise_equal']
            and smoke['formal_inputs_unchanged'] and smoke['internal_check_teacher_reads'] == 0, 'CUDA state/label smoke failed')
    require(smoke['train_source_sha256'] == file_hash(root / 'layout_longfit/train.py')
            and smoke['smoke_source_sha256'] == file_hash(root / 'layout_longfit/smoke.py') == file_hash(smoke_run / 'SMOKE_SOURCE.py'), 'Smoke source changed')
    verify_manifest(smoke_run, read_json(smoke_run / 'DATA_MANIFEST.json')['files'])
    require(set(smoke['jobs']) == set(protocol['jobs']), 'Smoke omitted a job')
    for name, job in protocol['jobs'].items():
        item = smoke['jobs'][name]
        require(item['status'] == 'PASS', 'Smoke job did not pass')
        require(item['manual_frozen_parent_first_step_equal'] is (True if job['kind'] == 'joint' else None), 'Parent first-step smoke differs')
        for label in ('continuous', 'first', 'resumed'):
            record = item[label]
            checkpoint = Path(record['checkpoint']).resolve()
            require(checkpoint.is_relative_to(smoke_run) and file_hash(checkpoint) == record['checkpoint_sha256'], 'Smoke checkpoint changed')
        a = torch.load(Path(item['continuous']['checkpoint']), map_location='cpu', weights_only=False)
        b = torch.load(Path(item['resumed']['checkpoint']), map_location='cpu', weights_only=False)
        require(all(exact_equal(a[key], b[key]) for key in item['exactly_equal_keys']), 'Recorded smoke semantic equality no longer holds')
    require(file_hash(run / 'PYTEST_PRETRAIN.xml') == gate['unit_tests_xml_sha256'], 'Pretrain test evidence changed')
    pretrain_count = successful_test_count(gate['unit_tests'])
    from xml.etree import ElementTree as ET
    pretrain_xml = ET.parse(run / 'PYTEST_PRETRAIN.xml').getroot()
    pretrain_suites = [pretrain_xml] if pretrain_xml.tag == 'testsuite' else pretrain_xml.findall('testsuite')
    require(sum(int(s.get('tests', 0)) for s in pretrain_suites) == pretrain_count
            and all(int(s.get(k, 0)) == 0 for s in pretrain_suites for k in ('failures', 'errors', 'skipped')), 'Pretrain XML/count differs')
    final_xml = ET.parse(run / 'PYTEST_FINAL.xml').getroot()
    suites = [final_xml] if final_xml.tag == 'testsuite' else final_xml.findall('testsuite')
    require(suites and sum(int(s.get('tests', 0)) for s in suites) > 0, 'Final test report empty')
    require(all(int(s.get(k, 0)) == 0 for s in suites for k in ('failures', 'errors', 'skipped')), 'Final code checks did not all pass')
    for name in ('CHECKS.json', 'SUMMARY.json', 'REPORT_zh.md', 'INTERPRETATION_zh.md'):
        require((run / name).is_file() and (run / name).stat().st_size > 0, 'Final documentation missing: ' + name)
    read_json(run / 'CHECKS.json')
    summary = read_json(run / 'SUMMARY.json')
    for key, relative in [('protocol_sha256', 'RUN_PROTOCOL.json'), ('results_sha256', 'results.json'),
                          ('mechanism_sha256', 'TRAIN_DIAGNOSTICS.json')]:
        require(summary[key] == file_hash(run / relative), 'Summary binds another artifact: ' + key)
    require(summary['heldout_generalization'] is False and summary['source_closed_studies_not_reinterpreted'], 'Summary scope changed')
    figures = read_json(run / 'figures/FIGURE_MANIFEST.json')
    verify_manifest(run / 'figures', figures['files'])
    require(figures['source_sha256'] == file_hash(root / 'layout_longfit/visualize.py')
            and figures['final_results_sha256'] == file_hash(run / 'results.json')
            and figures['diagnostics_sha256'] == file_hash(run / 'TRAIN_DIAGNOSTICS.json'), 'Figure provenance differs')
    return {'training_gate_sha256': file_hash(run / 'TRAINING_GATE.json'),
        'cuda_smoke_sha256': file_hash(smoke_run / 'GPU_SMOKE.json'), 'actual_cuda_smoke_updates': 36,
        'all_parent_and_resume_semantic_checks': True, 'scientific_pass_not_required': True,
        'summary_sha256': file_hash(run / 'SUMMARY.json'), 'figure_manifest_sha256': file_hash(run / 'figures/FIGURE_MANIFEST.json')}


def audit_run(run, smoke_run):
    """All static assertions precede any archival write; may retain missing slots."""
    run = Path(run).resolve(); smoke_run = Path(smoke_run).resolve(); root = Path(__file__).resolve().parents[1]
    p = read_json(run / 'RUN_PROTOCOL.json'); ph = file_hash(run / 'RUN_PROTOCOL.json')
    require(p['experiment'] == 'A-LONGFIT-v1' and config_digest(p) == p['hashes']['config_sha256'], 'Wrong/changed protocol')
    require(set(p['jobs']) == {'joint_10', 'joint_11'}, 'Fixed job set changed')
    require((p['train_uids'], p['internal_check_uids']) == split_uids(p['uids']), 'Frozen split changed')
    require(p['training']['updates'] == 240000 and p['training']['checkpoint_updates'] == [20000, 50000, 100000, 240000], 'Fixed budget/checkpoints changed')
    require(p['research_scope']['heldout_generalization'] is False and p['research_scope']['original_endpoint_gate_not_passed_and_not_revised'], 'Research qualification removed')
    dm_path = run / 'DATA_MANIFEST.json'; dm = read_json(dm_path)['files']
    require(file_hash(dm_path) == p['hashes']['data_manifest_sha256'] and len(dm) == 28, 'Prepared manifest changed')
    verify_manifest(run, dm)
    old = Path(p['source_run']).resolve(); om_path = old / 'MANIFEST_SHA256.json'
    require(file_hash(om_path) == p['source_run_manifest_sha256'] == p['hashes']['source_run_manifest_sha256'], 'Parent manifest changed')
    om = read_json(om_path)['files']; require(len(om) == 1942, 'Parent sealed file count differs')
    verify_manifest(old, om)
    require({x.relative_to(old).as_posix() for x in old.rglob('*') if x.is_file()} == set(om) | {'MANIFEST_SHA256.json'}, 'Parent file set changed')
    archive = old.with_name(old.name + '_results.zip'); archive_report = archive.with_suffix('.zip.json')
    require(file_hash(archive) == p['source_archive_sha256'] and file_hash(archive_report) == p['source_seal_report_sha256'], 'Parent ZIP/report changed')
    provenance = read_json(run / 'SOURCE_PROVENANCE.json')
    require(provenance['verified_source_manifest'] == om and provenance['source_manifest_sha256'] == file_hash(om_path), 'Source provenance changed')
    evidence_path = run / p['paths']['source_evidence_manifest']
    require(file_hash(evidence_path) == p['hashes']['source_evidence_manifest_sha256'], 'Copied mechanism manifest changed')
    evidence_manifest = read_json(evidence_path)['files']
    verify_manifest(run, evidence_manifest)
    for target, source in provenance['source_evidence_relative_paths'].items():
        require(file_hash(inside(old, source)) == file_hash(inside(run, target)) == evidence_manifest[target], 'Prior mechanism bytes differ')
    for target, source in provenance['source_and_destination_relative_paths'].items():
        require(file_hash(inside(old, source)) == file_hash(inside(run, target)) == dm[target], 'Copied source bytes differ')
    require(read_json(old / 'FINAL_AUDIT.json')['parent_strict_all_model_all_device_pass'] is False, 'Parent numerical limitations hidden')
    for rel, digest in p['fixed_source_hashes'].items():
        require(file_hash(inside(root, rel)) == digest, 'Frozen geometry/core source changed: ' + rel)
    upstream = verify(root.parents[1])
    require(upstream['pass'] and upstream['protected_files_checked'] == 11, 'Protected official core changed')
    from ..layout_endpoint.train import load_train_archive
    train, train_data, scale, _ = load_train_archive(run, p)
    require(train == p['train_uids'] and scale == p['endpoint_scale_squared'], 'Training archive changed')
    complete = read_json(run / 'TRAINING_COMPLETE.json')
    require(complete['serial_GPU_training_jobs'] is True and complete['concurrent_CPU_evaluation_possible'] is True
            and complete['training_wall_time_is_not_isolated_benchmark'] is True, 'Training timing scope misrepresented')
    gate = gate_and_documents_audit(run, smoke_run, p)
    training, models, observations = training_audit(run, p, dm)
    final_ledger = read_json(run / 'results.json')
    final_attempts = [read_json(x) for x in sorted((run / 'attempts').glob('*/*/result.json'))]
    final_counts = final_accounting(p, final_ledger, final_attempts)
    require(final_counts['planned_attempts'] == 98 and final_counts['planned_rows'] == 164, 'Final denominator changed')
    diag_ledger = read_json(run / 'TRAIN_DIAGNOSTICS.json')
    diag_attempts = [read_json(x) for x in sorted((run / 'diagnostics/mechanism').glob('*/*/*/result.json'))]
    diag_counts = diagnostic_accounting(p, diag_ledger, diag_attempts)
    require(diag_counts['planned_slots'] == 120 and diag_ledger['protocol_sha256'] == ph, 'Diagnostic denominator/protocol changed')
    for attempt in diag_attempts:
        require(attempt['protocol_sha256'] == ph and attempt['diagnose_source_sha256'] == file_hash(root / 'layout_longfit/diagnose.py'), 'Diagnostic source/protocol changed')
    expected_seeds = (p['evaluation']['seed_base'], p['evaluation']['second_seed_base'])
    require(expected_seeds == SEED_BASES == (792501, 892501) and p['evaluation']['points_each_direction'] == POINTS == 8192, 'New evaluation sampling changed')
    require(p['evaluation']['hot_repeats'] == 3 and p['evaluation']['models'] == list(MODEL_NAMES), 'Final model/timing contract changed')
    require(set(p['evaluation']['source_checkpoint_sha256']) == {'parent_joint_10', 'parent_joint_11', 'single_energy_10', 'single_prior_10'}, 'Frozen reference set differs')
    for name in ('single_energy_10', 'single_prior_10'):
        require(p['evaluation']['model_uids'][name] == p['evaluation']['single_reference_uids'][name]
                == p['source_single_jobs'][name]['active_training_uids'], 'Frozen single reference parent changed')

    for invocation in sorted((run / 'eval_invocations').glob('*.json')):
        item = read_json(invocation)
        require(item['protocol_sha256'] == ph and item['evaluation_seed_bases'] == list(SEED_BASES), 'Final invocation provenance changed')
        for rel in ('layout_longfit/evaluate.py', 'layout_student/deployment.py', 'legacy/deform.py', 'legacy/chart.py', 'legacy/networks.py', 'layout_value/geometry.py'):
            require(item['source_hashes'][rel] == file_hash(root / rel), 'Final evaluation implementation changed: ' + rel)
    data = {uid: mesh_arrays(run / 'data' / (uid + '.npz')) for uid in p['uids']}
    support = RadialSupport(); final_meshes = {}; diag_meshes = {}; parent_comparisons = []
    old_rows = {(r['uid'], r['model'], r['policy']): r for r in read_json(old / 'results.json')['rows']}
    for uid, d in data.items():
        require(d['faces'].dtype == np.int32 and d['faces'].shape == (512, 3), 'Logical connectivity changed')
        if uid in train:
            index = train.index(uid)
            for key, archive_key in [('base_u', 'directions0'), ('teacher_u', 'directions1'), ('logr', 'logr')]:
                require(np.array_equal(d[key], train_data[archive_key][index]), 'Per-parent and train-only archive differ')
    for row in final_ledger['rows']:
        uid, name, policy = row['uid'], row['model'], row['policy']; d = data[uid]
        mesh, (fixed_v, fixed_r) = check_mesh(run, row, d, support)
        require(row['split'] == ('train' if uid in train else 'internal_check')
                and row['data_file_sha256'] == dm['data/' + uid + '.npz'], 'Final data/split changed')
        if name in models:
            require(row['identity']['model_sha256'] == models[name]['model_sha256']
                    and row['identity']['file_sha256'] == models[name]['checkpoint_sha256'], 'Final weight identity differs')
        if name == 'fixed':
            require(np.array_equal(mesh['u'], d['base_u']) and np.array_equal(mesh['vertices'], fixed_v), 'Fixed changed')
        if name == 'teacher_reference':
            require(np.array_equal(mesh['u'], d['teacher_u']) and row['offline_teacher'], 'Offline teacher changed/relabeled')
        if row['fallback_to_fixed']:
            require(policy == 'quality_fallback' and np.array_equal(mesh['u'], d['base_u'])
                    and np.array_equal(mesh['vertices'], fixed_v) and np.array_equal(mesh['r'], fixed_r), 'Fallback differs from same-object Fixed')
        metrics = row['metrics']; seeds = [x + sorted(p['uids']).index(uid) for x in SEED_BASES]
        require(len(metrics) == 2 and [m['seed'] for m in metrics] == seeds
                and all(m['points_per_direction'] == POINTS and np.isfinite(m['rms_bbox_percent']) and m['rms_bbox_percent'] >= 0 for m in metrics), 'Final RMS identity differs')
        require(row['independent_evaluation_not_used_for_deployment'], 'Error-based output selection')
        if policy in ('raw', 'offline_teacher') and name in ('fixed', 'teacher_reference', 'parent_joint_10', 'parent_joint_11', 'single_energy_10', 'single_prior_10'):
            old_name = name.replace('parent_', ''); source_row = old_rows[(uid, old_name, policy)]
            source_mesh_path = inside(old, source_row['mesh_file']); source_mesh = mesh_arrays(source_mesh_path)
            source_id = mesh_identity(source_mesh_path, d['faces'])
            for key, value in source_id.items(): require(source_row[key] == value, 'Parent mesh recorded hash changed')
            parent_comparisons.append({'uid': uid, 'model': name,
                **compare_prior_mesh(source_mesh, mesh, d['faces'], source_row, source_id, row)})
        final_meshes[(uid, name, policy)] = mesh
    row_map = {(r['uid'], r['model'], r['policy']): r for r in final_ledger['rows']}
    for row in final_ledger['rows']:
        if row['policy'] == 'quality_fallback':
            raw = row_map[(row['uid'], row['model'], 'raw')]
            require(row['fallback_to_fixed'] == (not raw['qualified']) and row['fallback_reasons'] == raw['failure_reasons'], 'Fallback rule differs')
            if raw['qualified']:
                check_same_geometry(final_meshes[(row['uid'], row['model'], 'raw')], final_meshes[(row['uid'], row['model'], 'quality_fallback')], 'Qualified raw changed in delivery')
    cross_checks = []
    for row in diag_ledger['rows']:
        name, ckpt, uid = row['job'], row['checkpoint'], row['uid']; d = data[uid]
        require(uid in train and uid in p['jobs'][name]['diagnostic_uids'], 'Check/unknown parent in diagnostics')
        require(row['training_labels_sha256'] == p['hashes']['training_labels_sha256']
                and row['teacher_label_hash'] == array_hash(d['base_u'], d['teacher_u']), 'Diagnostic teacher archive changed')
        identity = observations[(name, ckpt)]
        for key, value in identity.items(): require(row['identity'][key] == value, 'Diagnostic weight identity differs')
        mesh, _ = check_mesh(run, row, d, support)
        require(row['unchanged_quality_gate_would_fallback'] == (not row['qualified']), 'Diagnostic gate decision changed')
        require(row['teacher_or_complete_field_added_to_deployment_network_inputs'] is False, 'Diagnostic prediction used teacher inputs')
        require(row['alpha_values'] == mesh['alpha'].tolist(), 'Diagnostic guard history differs')
        endpoint = endpoint_probe(mesh['u'], d['base_u'], d['teacher_u'], d['faces'], scale)
        require(endpoint == row['endpoint'], 'Offline endpoint quantities differ from saved directions')
        velocity = row['velocity']
        require(velocity['steps'] == list(range(32)) and velocity['times'] == [k / 32 for k in range(32)], 'Bridge probe grid changed')
        require(len(velocity['mse_per_step']) == len(velocity['zero_velocity_mse_per_step']) == 32
                and all(np.isfinite(v) and v >= 0 for v in velocity['mse_per_step'] + velocity['zero_velocity_mse_per_step']), 'Invalid velocity probe')
        require(velocity['mse'] == float(np.mean(velocity['mse_per_step']))
                and velocity['zero_velocity_mse'] == float(np.mean(velocity['zero_velocity_mse_per_step'])), 'Velocity reduction differs')
        diag_meshes[(name, ckpt, uid)] = mesh
        if ckpt == '240000' and (uid, name, 'raw') in final_meshes:
            check_same_geometry(mesh, final_meshes[(uid, name, 'raw')], 'Final CPU mesh differs between diagnostic and final evaluation')
            cross_checks.append((name, uid))
    for uid in p['uids']:
        for name in ('joint_10', 'joint_11'):
            if uid in p['jobs'][name]['diagnostic_uids'] and (uid, name, 'raw') in final_meshes:
                require((name, '240000', uid) in diag_meshes or (name, '240000', uid) in {tuple(x) for x in diag_counts['missing_success_slots']}, 'Final diagnostic disappeared')
    replay_uid = sorted(p['uids'])[0]
    require((replay_uid, 'joint_10', 'raw') in final_meshes and (replay_uid, 'joint_10', 'quality_fallback') in final_meshes,
            'Fixed replay sample unavailable; no substitution permitted')
    prior_files = {rel: digest for rel, digest in om.items() if
        rel in ('RUN_PROTOCOL.json', 'DATA_MANIFEST.json', 'TEACHERS.json', 'results.json', 'SUMMARY.json',
                'FINAL_AUDIT.json', 'IMPLEMENTATION_GATE.json', 'TRAINING_GATE.json', 'FINAL_DEPLOYMENT_PARITY.json', 'SOURCE_PROVENANCE.json')
        or rel.startswith('correctness_evidence/')
        or (rel.startswith('attempts/') and any('/' + name + '/' in rel for name in ('fixed', 'teacher_reference', 'joint_10', 'joint_11', 'single_energy_10', 'single_prior_10')))
        or (rel.startswith('training/') and rel.endswith(('TRAIN_REPORT.json', 'TRAIN_CONFIG.json', 'TRAIN_TRACE.json', 'TRAIN_TIMING.json')))}
    referenced = {r['mesh_file'] for r in final_ledger['rows'] + diag_ledger['rows']}
    partial_meshes = [str(x.relative_to(run)) for parent in (run / 'attempts', run / 'diagnostics/mechanism')
        for x in parent.rglob('*.npz') if str(x.relative_to(run)) not in referenced]
    audit = {'protocol_sha256': ph, 'final_evaluation': final_counts, 'training_diagnostics': diag_counts,
        'data_manifest_files_checked': len(dm), 'parent_fitdiag_files_unchanged': len(om),
        'parent_fitdiag_manifest_sha256': file_hash(om_path), 'parent_archive_sha256': file_hash(archive),
        'parent_archive_report_sha256': file_hash(archive_report), 'parent_geometry_comparisons': parent_comparisons,
        'training_checkpoints': training, 'gates_and_documents': gate, 'upstream': upstream,
        'copied_mechanism_evidence_files_checked': len(evidence_manifest), 'training_timing_scope': {k: complete[k] for k in ('serial_GPU_training_jobs', 'concurrent_CPU_evaluation_possible', 'training_wall_time_is_not_isolated_benchmark')},
        'saved_final_meshes_checked': len(final_meshes), 'saved_diagnostic_meshes_checked': len(diag_meshes),
        'accepted_direction_requeries_exact': True, 'logical_storage_and_file_hashes_checked_separately': True,
        'final_vs_train_diagnostic_raw_geometry_comparisons': cross_checks,
        'final_vs_train_comparison_missing_slots_retained': diag_counts['missing_success_slots'],
        'partial_unreferenced_mesh_files_retained': partial_meshes,
        'predeclared_replay': {'model': 'joint_10', 'uid': replay_uid, 'update': 240000},
        'parent_strict_all_model_all_device_pass': False, 'science_success_required_for_sealing': False,
        'equal_parent_exposure_count_at_final': True, 'equal_adam_steps_or_compute_claim': False, 'omitted_unrelated_modules': ['system_probe'], 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return audit, final_ledger['rows'], final_meshes, diag_ledger['rows'], diag_meshes, old, prior_files


REPLAY_SCRIPT = r'''"""Predeclared original-CPU replay using only extracted longfit artifacts."""
import argparse,json,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--report',required=True)
args=parser.parse_args();run=Path(args.run).resolve();sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import torch
from experiments.a_radial import exporter
from experiments.a_radial.layout_student import deployment
from experiments.a_radial.layout_student.train import model_sha256
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.io_utils import array_hash,file_hash
assert Path(exporter.__file__).resolve().is_relative_to(run/'source_snapshot')
assert Path(deployment.__file__).resolve().is_relative_to(run/'source_snapshot')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());uid=sorted(protocol['uids'])[0]
job='joint_10';checkpoint=run/'training'/job/'final.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['update']==240000 and state['is_predeclared_final'] and state['diagnostic_only']
assert state['job_id']==state['config']['job_id']==job
assert state['sampler']['exposures']==[80000]*12
assert state['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
assert state['model_sha256']==model_sha256(state['model'])
initial=run/protocol['jobs'][job]['initial_checkpoint']
assert file_hash(initial)==state['initial_checkpoint_sha256']==protocol['jobs'][job]['initial_checkpoint_sha256']
model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as data:
    logr=data['logr'];base=data['base_u'];faces=data['faces']
support=exporter.RadialSupport(run/'source_snapshot/experiments/a_radial/assets/support.npz')
result=deployment.deploy(model,logr,support,base,faces)
rows=json.loads((run/'attempts'/uid/job/'result.json').read_text())['rows'];checks={}
for policy,prefix in [('raw','raw'),('quality_fallback','delivered')]:
    row=next(r for r in rows if r['policy']==policy)
    mesh=result[prefix+'_mesh'];audit=result[prefix+'_audit'];quality=result[prefix+'_quality']
    assert array_hash(mesh['vertices'],faces)==row['mesh_hash']
    assert array_hash(mesh['vertices'],mesh['faces'])==row['stored_array_hash']
    assert file_hash(run/row['mesh_file'])==row['mesh_npz_sha256']
    with np.load(run/row['mesh_file'],allow_pickle=False) as saved:
        for key in ('vertices','faces','u','r'):
            assert np.array_equal(mesh[key],saved[key]),(policy,key)
        if 'alpha' in mesh:assert np.array_equal(mesh['alpha'],saved['alpha'])
    radii=support.query(logr,mesh['u'])
    assert np.array_equal(mesh['r'],radii) and np.array_equal(mesh['vertices'],mesh['u']*radii[:,None])
    gate=deployment.qualification(quality,audit,result['floor'])
    assert gate['qualified']==row['qualified'] and gate['failure_reasons']==row['failure_reasons']
    assert audit['exact_requested'] and audit['exact_executed']
    for key in ('valid','selfintersection_count'):assert audit[key]==row['audit'][key]
    for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):
        assert abs(quality[key]-row['quality'][key])<=1e-10
    checks[policy]={'mesh_hash':row['mesh_hash'],'stored_array_hash':row['stored_array_hash'],
        'mesh_npz_sha256':row['mesh_npz_sha256'],'geometry_and_accepted_query_bitwise_equal':True,
        'audit_and_qualification_match':True,'qualified':gate['qualified']}
assert result['delivery']['fallback_to_fixed']==next(r for r in rows if r['policy']=='quality_fallback')['fallback_to_fixed']
report={'status':'PASS','model':job,'uid':uid,'update':240000,'device':'cpu',
    'loaded_from_fresh_extracted_source':True,'teacher_label_read_for_prediction':False,
    'initial_checkpoint_sha256_verified':True,'checkpoint_sha256':file_hash(checkpoint),
    'checks':checks,'fallback_to_fixed':result['delivery']['fallback_to_fixed']}
with Path(args.report).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report),flush=True)
'''

SNAPSHOT_MODULES = {'assets', 'legacy', 'layout_value', 'layout_student', 'layout_endpoint', 'layout_fitdiag', 'layout_longfit'}
SNAPSHOT_ROOT_FILES = set(['.gitignore', 'AGENTS.md', '__init__.py', 'doctor.py', 'evaluate.py', 'export_ours.py', 'exporter.py', 'io_utils.py', 'mesh_ops.py', 'models.py', 'official_contract.py', 'official_sample.py', 'official_train.py', 'prepare_data.py', 'registry.py', 'requirements-extra.txt', 'sample_ours.py', 'train_ours.py', 'verify_repo.py'])


def snapshot_ignore(source):
    source = Path(source).resolve()
    base = shutil.ignore_patterns('__pycache__', '.pytest_cache', 'build', '*.pyc', '*.nbi', '*.nbc')
    def ignored(directory, names):
        omitted = set(base(directory, names))
        if Path(directory).resolve() == source:
            omitted.update(name for name in names if name not in SNAPSHOT_MODULES and name not in SNAPSHOT_ROOT_FILES)
        return omitted
    return ignored


def snapshot_files(source):
    source = Path(source).resolve()
    inventory = {}
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if len(relative.parts) == 1:
            include = path.name in SNAPSHOT_ROOT_FILES
        else:
            include = relative.parts[0] in SNAPSHOT_MODULES
        if not include or any(part in ('__pycache__', '.pytest_cache', 'build') for part in relative.parts) or path.suffix in ('.pyc', '.nbi', '.nbc'):
            continue
        require(not path.is_symlink(), 'Snapshot refuses nonregular source artifact')
        inventory[relative.as_posix()] = file_hash(path)
    return inventory


def package_run(run, smoke_run):
    run = Path(run).resolve(); smoke_run = Path(smoke_run).resolve()
    archive = run.with_name(run.name + '_results.zip'); fresh = run.with_name(run.name + '_fresh_extract')
    archive_report = archive.with_suffix('.zip.json'); fresh_report = archive.with_suffix('.zip.fresh.json')
    destinations = [run / 'FINAL_AUDIT.json', run / 'MANIFEST_SHA256.json', run / 'meshes_obj',
        run / 'diagnostic_meshes_obj', run / 'source_snapshot', run / 'gpu_smoke_evidence',
        run / 'sealed_parent_evidence', run / 'REPLAY_FRESH.py', archive, fresh, archive_report, fresh_report]
    for path in destinations:
        require(not path.exists(), 'Exclusive sealing destination exists: ' + str(path))
    audit, rows, meshes, diagnostic_rows, diagnostic_meshes, old, prior_files = audit_run(run, smoke_run)
    compile(REPLAY_SCRIPT, '<fresh-longfit-replay>', 'exec')
    before = {p.relative_to(run).as_posix(): file_hash(p) for p in sorted(run.rglob('*')) if p.is_file()}
    source = Path(__file__).resolve().parents[1]
    ignored = snapshot_ignore(source)
    source_files = snapshot_files(source)
    print(json.dumps({'event': 'preflight_passed', 'run': str(run)}), flush=True)
    # No output mutation occurs until all static training, data, mesh and ledger checks pass.
    for row in rows:
        mesh = meshes[(row['uid'], row['model'], row['policy'])]
        write_obj(run / 'meshes_obj' / row['uid'] / (row['model'] + '__' + row['policy'] + '.obj'), mesh['vertices'], mesh['faces'])
    for row in diagnostic_rows:
        mesh = diagnostic_meshes[(row['job'], row['checkpoint'], row['uid'])]
        write_obj(run / 'diagnostic_meshes_obj' / row['job'] / row['checkpoint'] / (row['uid'] + '.obj'), mesh['vertices'], mesh['faces'])
    shutil.copytree(source, run / 'source_snapshot/experiments/a_radial', ignore=ignored)
    verify_manifest(run / 'source_snapshot/experiments/a_radial', source_files)
    shutil.copytree(smoke_run, run / 'gpu_smoke_evidence', ignore=ignored)
    evidence = run / 'sealed_parent_evidence'; evidence.mkdir()
    shutil.copy2(old / 'MANIFEST_SHA256.json', evidence / 'MANIFEST_SOURCE_FULL.json')
    seal_report = old.with_name(old.name + '_results.zip.json')
    shutil.copy2(seal_report, evidence / 'SOURCE_ZIP_REPORT.json')
    for relative, digest in prior_files.items():
        destination = inside(evidence, relative); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(old, relative), destination)
        require(file_hash(destination) == digest, 'Parent evidence copy changed')
    write_json(evidence / 'COPIED_FILES.json', {'source_run': str(old),
        'source_manifest_sha256': file_hash(old / 'MANIFEST_SHA256.json'), 'files': prior_files,
        'scope': 'Referenced subset; complete source manifest retained; omitted artifacts are not claimed present.'})
    with (run / 'REPLAY_FRESH.py').open('x') as stream:
        stream.write(REPLAY_SCRIPT)
    verify_manifest(run, before)
    audit['exported_final_obj_count'] = len(rows); audit['exported_diagnostic_obj_count'] = len(diagnostic_rows)
    audit['prior_fitdiag_evidence_files_copied'] = len(prior_files)
    audit['source_snapshot_files'] = len(source_files)
    audit['source_snapshot_modules'] = sorted(SNAPSHOT_MODULES)
    audit['omitted_unrelated_modules'] = sorted(p.name for p in source.iterdir() if p.is_dir() and p.name not in SNAPSHOT_MODULES)
    write_json(run / 'FINAL_AUDIT.json', audit)
    manifest = {p.relative_to(run).as_posix(): file_hash(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run / 'MANIFEST_SHA256.json', {'files': manifest})
    members = {**manifest, 'MANIFEST_SHA256.json': file_hash(run / 'MANIFEST_SHA256.json')}
    print(json.dumps({'event': 'copy_and_manifest_complete', 'files': len(members)}), flush=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for relative in sorted(members):
            zipped.write(run / relative, arcname=(PurePosixPath(run.name) / relative).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, 'Archive CRC failed')
        expected = {(PurePosixPath(run.name) / relative).as_posix() for relative in members}
        require(len(zipped.namelist()) == len(expected) and set(zipped.namelist()) == expected, 'Unexpected ZIP members')
        for relative, digest in members.items():
            require(hashlib.sha256(zipped.read((PurePosixPath(run.name) / relative).as_posix())).hexdigest() == digest,
                    'Archive member changed')
        fresh.mkdir()
        for name in zipped.namelist():
            inside(fresh, name)
        zipped.extractall(fresh)
    extracted = fresh / run.name
    require(read_json(extracted / 'MANIFEST_SHA256.json')['files'] == manifest, 'Fresh manifest differs')
    verify_manifest(extracted, members)
    env = dict(os.environ); env['PYTHONDONTWRITEBYTECODE'] = '1'
    print(json.dumps({'event': 'fresh_extract_verified', 'files': len(members)}), flush=True)
    replay = subprocess.run([sys.executable, '-I', '-B', str(extracted / 'REPLAY_FRESH.py'),
        '--run', str(extracted), '--report', str(fresh_report)], cwd=fresh, env=env, capture_output=True, text=True)
    if replay.returncode:
        failed = archive.with_suffix('.zip.replay_failed.json')
        write_json(failed, {'status': 'FAIL', 'returncode': replay.returncode, 'stdout': replay.stdout,
            'stderr': replay.stderr, 'fresh_manifest_passed': True, 'no_retry_or_sample_substitution': True})
        raise RuntimeError('Fresh replay failed; evidence retained: ' + str(failed))
    require(read_json(fresh_report)['status'] == 'PASS', 'Fresh replay did not pass')
    print(json.dumps({'event': 'fresh_original_cpu_replay_passed'}), flush=True)
    verify_manifest(run, members); verify_manifest(extracted, members)
    for sealed in (run, extracted):
        require({p.relative_to(sealed).as_posix() for p in sealed.rglob('*') if p.is_file()} == set(members), 'Sealed/fresh file set changed')
    verify_manifest(old, read_json(old / 'MANIFEST_SHA256.json')['files'])
    require(file_hash(old / 'MANIFEST_SHA256.json') == audit['parent_fitdiag_manifest_sha256'], 'Old manifest changed during sealing')
    parent_archive = old.with_name(old.name + '_results.zip')
    require(file_hash(parent_archive) == audit['parent_archive_sha256']
            and file_hash(parent_archive.with_suffix('.zip.json')) == audit['parent_archive_report_sha256'], 'Old archive changed during sealing')
    result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': file_hash(archive),
        'crc_verified': True, 'member_hashes_verified_including_manifest': len(members),
        'fresh_extracted_manifest_verified': True, 'fresh_original_cpu_replay': 'PASS',
        'fresh_replay_report': str(fresh_report), 'sealed_run_and_fresh_artifacts_unchanged': True,
        'parent_fitdiag_1942_files_and_archive_unchanged_after_sealing': True,
        'saved_final_meshes_and_objs': len(rows), 'saved_diagnostic_meshes_and_objs': len(diagnostic_rows),
        'final_accounting': audit['final_evaluation'], 'diagnostic_accounting': audit['training_diagnostics'],
        'scientific_pass_not_inferred_from_archive_integrity': True,
        'prior_endpoint_strict_all_model_all_device_pass_remains_false': True}
    write_json(archive_report, result)
    print(json.dumps(result, indent=2), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--smoke-run', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_run(args.run, args.smoke_run)[0], indent=2), flush=True)
    else:
        package_run(args.run, args.smoke_run)

if __name__ == '__main__':
    main()
