"""Offline stricter-teacher feasibility; no learned-method or label-update claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, new_dir, save_npz, write_json
from ..layout_value.optimizer import optimize_layout
from ..layout_value.geometry import audit_mesh
from ..layout_student.deployment import named_quality, qualification
from ..layout_student.integrity import mesh_identity, safe_path
from ..layout_student.evaluate import metric_cache
from ..legacy.chart import dets
from ..system_probe.warm_start import serialize_optimizer, snapshot_records

UIDS = ('7f1be3e7b6adeeae9bb399e812556da2', '2379915c82c2b2d02530f02db7e9157e',
        '8c919e992db7cb081e5b68ee10358ea3', '736820a85063a4371e5b68ee10358ea3')
INITIALIZERS = ('fixed', 'teacher')
ENDPOINTS = ('fixed', 'teacher', 'opt_fixed', 'opt_teacher')
SENSITIVE_ENDPOINTS = ('teacher', 'opt_fixed', 'opt_teacher')
ANGLES = ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg')
MARGIN = 0.1
STEPS = 64
SNAPSHOTS = (0, 16, 64)
MAX_SECONDS = 3600
FRACTIONS = (0., .0001, .001, .01, .1, .5, 1.)
TRIALS = 16
OPT_SEED_BASE = 1493501
MC_SEED_BASES = (1593501, 1693501)
NOISE_SEED_BASE = 1393501
POINTS = 8192


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_files():
    root = Path(__file__).resolve().parents[1]
    names = ['layout_teacher_margin/__init__.py', 'layout_teacher_margin/experiment.py',
        'system_probe/warm_start.py', 'system_probe/teacher_sensitivity.py',
        'layout_value/optimizer.py', 'layout_value/geometry.py',
        'layout_student/deployment.py', 'layout_student/integrity.py', 'layout_student/evaluate.py',
        'layout_student/train.py', 'legacy/chart.py', 'legacy/common.py', 'legacy/exact_accel.py',
        'legacy/geometry_task.py', 'legacy/deform.py', 'legacy/networks.py',
        'mesh_ops.py', 'exporter.py', 'io_utils.py', 'assets/support.npz']
    return {str(root / name): file_hash(root / name) for name in names}


def stricter_floor(fixed_quality):
    require(all(np.isfinite(fixed_quality[k]) and fixed_quality[k] >= 0 for k in ANGLES),
            'Invalid Fixed angle floor')
    return {k: float(fixed_quality[k]) + MARGIN for k in ANGLES}


def optimizer_floor(fixed_quality):
    floor = stricter_floor(fixed_quality)
    return {'min_angle_deg': floor[ANGLES[0]], 'angle_q05_deg': floor[ANGLES[1]]}


def validate_data(data):
    expected = {'logr': (2050,), 'base_u': (258, 3), 'teacher_u': (258, 3),
                'faces': (512, 3), 'ref_vertices': (2050, 3), 'ref_faces': (4096, 3)}
    for key, shape in expected.items():
        require(key in data and data[key].shape == shape and np.isfinite(data[key]).all(),
                'Invalid frozen data array: ' + key)
    require(data['faces'].dtype == np.int32 and data['faces'].min() >= 0 and data['faces'].max() < 258,
            'Expected unchanged int32 F512 connectivity')
    for key in ('base_u', 'teacher_u'):
        require(data[key].dtype == np.float64 and np.allclose(np.linalg.norm(data[key], axis=1), 1,
                rtol=0, atol=1e-9), 'Directions must retain finite float64 unit vectors: ' + key)
    return data


def read_data(path):
    with np.load(path, allow_pickle=False) as z:
        data = {key: np.array(z[key]) for key in ('logr', 'base_u', 'teacher_u', 'faces', 'ref_vertices', 'ref_faces')}
    return validate_data(data)


def assert_contract(p):
    require(p['schema'] == 'A-TEACHER-MARGIN-v1' and p['quality_tolerance_deg'] == 1e-6, 'Protocol schema or numerical tolerance changed')
    require(all(p[key] is True for key in ('no_fixed_fallback', 'no_GT_or_MC_selection', 'no_old_label_updates', 'no_network_training_or_learned_online_advantage_claim', 'no_generalization_claim')), 'Research scope changed')
    require(p['uids'] == list(UIDS) and p['initializers'] == list(INITIALIZERS), 'Fixed task set changed')
    require(p['endpoint_names'] == list(ENDPOINTS) and p['expected_tasks'] == 8 and p['expected_endpoint_rows'] == 16,
            'Endpoint denominator changed')
    require(p['sensitivity']['endpoint_names'] == list(SENSITIVE_ENDPOINTS)
        and p['sensitivity']['fractions'] == list(FRACTIONS)
        and p['sensitivity']['trials'] == TRIALS and p['sensitivity']['planned_slots'] == 1344,
        'Perturbation denominator changed')
    require(p['margin_deg'] == MARGIN and p['max_steps'] == STEPS and p['max_seconds'] == MAX_SECONDS
        and p['snapshot_steps'] == list(SNAPSHOTS), 'Margin or budget changed')
    require(p['optimizer_seed_base'] == OPT_SEED_BASE and p['evaluation_seed_bases'] == list(MC_SEED_BASES)
        and p['sensitivity']['seed_base'] == NOISE_SEED_BASE and p['points_each_direction'] == POINTS,
        'Predeclared random seeds changed')
    require(p['device'] == 'cpu' and p['threads'] == 4 and p['interop_threads'] == 1,
        'CPU execution contract changed')
    require(len(p['all16_sorted_uids']) == 16 and len(set(p['all16_sorted_uids'])) == 16
        and p['all16_sorted_uids'] == sorted(p['all16_sorted_uids']), 'Invalid original 16 UID registry')
    require(len(p['all_train_uids']) == 12 and len(set(p['all_train_uids'])) == 12
        and set(UIDS) <= set(p['all_train_uids']) <= set(p['all16_sorted_uids']), 'Not the fixed training subset')


def prepare(source, out):
    require(not torch.cuda.is_initialized(), 'CPU-only preparation')
    source = Path(source).resolve(); out = Path(out).resolve()
    require(not out.exists(), 'Do not overwrite a prepared study')
    p = read_json(source / 'RUN_PROTOCOL.json')
    require(set(UIDS) <= set(p['train_uids']) and not set(UIDS) & set(p['internal_check_uids']),
            'Four predeclared UIDs must belong to train')
    manifest_path = source / 'MANIFEST_SHA256.json'; manifest = read_json(manifest_path)['files']
    archive = source.with_name(source.name + '_results.zip'); archive_report = archive.with_suffix('.zip.json')
    report = read_json(archive_report)
    require(report['crc_verified'] and report['fresh_extracted_manifest_verified']
        and report['fresh_original_cpu_replay'] == 'PASS' and report['sealed_run_and_fresh_artifacts_unchanged'],
        'Source study must be sealed and independently replayed')
    require(report['sha256'] == file_hash(archive) and report['bytes'] == archive.stat().st_size,
        'Sealed source archive changed')
    selected = ['RUN_PROTOCOL.json', 'DATA_MANIFEST.json', 'train_labels.npz'] + [f'data/{uid}.npz' for uid in UIDS]
    bound = {str(source / rel): file_hash(source / rel) for rel in selected}
    require(all(bound[str(source / rel)] == manifest[rel] for rel in selected), 'Source manifest mismatch')
    require(bound[str(source / 'train_labels.npz')] == p['hashes']['training_labels_sha256']
        and bound[str(source / 'DATA_MANIFEST.json')] == p['hashes']['data_manifest_sha256'], 'Source data binding mismatch')
    bound[str(manifest_path)] = file_hash(manifest_path); bound[str(archive_report)] = file_hash(archive_report)
    data_manifest = read_json(source / 'DATA_MANIFEST.json')['files']
    with np.load(source / 'train_labels.npz', allow_pickle=False) as z:
        labels = {key: np.array(z[key]) for key in ('train_uids', 'directions0', 'directions1', 'logr', 'faces')}
    require(labels['train_uids'].tolist() == p['train_uids'], 'Original train-label ordering differs')
    evidence = {}
    for uid in UIDS:
        data = read_data(source / 'data' / (uid + '.npz')); i = p['train_uids'].index(uid)
        for left, right in [('base_u', 'directions0'), ('teacher_u', 'directions1'), ('logr', 'logr')]:
            require(np.array_equal(data[left], labels[right][i]) and data[left].dtype == labels[right].dtype,
                    'Frozen teacher/data alignment differs: ' + uid + '/' + left)
        require(np.array_equal(data['faces'], labels['faces']), 'Template faces differ from frozen labels')
        require(data_manifest[f'data/{uid}.npz'] == bound[str(source / 'data' / (uid + '.npz'))], 'Source data manifest differs')
        energy = float(np.square(data['teacher_u'] - data['base_u']).mean())
        evidence[uid] = {'training_parent_index': i, 'original_sorted16_index': sorted(p['uids']).index(uid),
            'original_teacher_energy': energy, 'zero_teacher': energy == 0,
            'teacher_hash': array_hash(data['teacher_u']), 'base_hash': array_hash(data['base_u']),
            'mainfield_hash': array_hash(data['logr']), 'faces_hash': array_hash(data['faces'])}
    frozen = source_files(); out = new_dir(out)
    copies = {f'data/{uid}.npz': source / 'data' / (uid + '.npz') for uid in UIDS}
    copies.update({'source_evidence/RUN_PROTOCOL.json': source / 'RUN_PROTOCOL.json',
        'source_evidence/DATA_MANIFEST.json': source / 'DATA_MANIFEST.json',
        'source_evidence/MANIFEST_SHA256.json': manifest_path,
        'source_evidence/ARCHIVE_REPORT.json': archive_report,
        'source_evidence/train_labels.npz': source / 'train_labels.npz'})
    copied = {}
    for relative, original in copies.items():
        dest = out / relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, dest)
        copied[relative] = file_hash(dest); require(copied[relative] == file_hash(original), 'Copied source changed')
    write_json(out / 'DATA_MANIFEST.json', {'files': copied})
    protocol = {'schema': 'A-TEACHER-MARGIN-v1', 'scope': 'four training parents; offline stricter-teacher feasibility',
        'uids': list(UIDS), 'all16_sorted_uids': sorted(p['uids']), 'all_train_uids': p['train_uids'],
        'initializers': list(INITIALIZERS), 'expected_tasks': 8, 'endpoint_names': list(ENDPOINTS), 'expected_endpoint_rows': 16,
        'margin_deg': MARGIN, 'quality_tolerance_deg': 1e-6, 'max_steps': STEPS, 'max_seconds': MAX_SECONDS,
        'snapshot_steps': list(SNAPSHOTS), 'device': 'cpu', 'threads': 4, 'interop_threads': 1,
        'optimizer_seed_base': OPT_SEED_BASE, 'optimizer_seed_rule': 'base + original sorted16 UID index; same per UID for both starts',
        'evaluation_seed_bases': list(MC_SEED_BASES), 'points_each_direction': POINTS,
        'MC_scope': 'same predeclared warm_start MC; repeated train-only mechanism evidence, not new unseen MC confirmation',
        'sensitivity': {'endpoint_names': list(SENSITIVE_ENDPOINTS), 'fractions': list(FRACTIONS), 'trials': TRIALS,
            'planned_slots': 1344, 'seed_base': NOISE_SEED_BASE, 'seed_rule': 'base + 16*original training_parent_index + trial',
            'noise': 'same ambient independent Gaussian per UID/trial, projected at each endpoint and unit component RMS',
            'absolute_scale': 'fraction * sqrt(original teacher-vs-Fixed component MSE); identical scale for all endpoints',
            'zero_teacher': 'zero amplitude retained; no informative robustness conclusion',
            'primary_angular_gate': 'original Fixed min and p05, tolerance1e-6deg',
            'no_full_exact_audit_of_perturbations': True, 'random_noise_is_not_learned_error': True},
        'optimizer': 'unmodified layout_value.optimizer.optimize_layout; 2048 internal MC + 3-point output quadrature',
        'optimizer_reference': 'frozen mainfield support triangles only; GT arrays never passed into optimizer',
        'offline_reference': 'frozen original GT triangles, double8192 MC only after endpoint selection',
        'selection': 'original minimum internal surface loss among accepted feasible iterates; failed restoration retained',
        'no_fixed_fallback': True, 'no_GT_or_MC_selection': True, 'no_old_label_updates': True,
        'no_network_training_or_learned_online_advantage_claim': True, 'no_generalization_claim': True,
        'max_seconds_is_only_safety_cap': True, 'parent_evidence': evidence,
        'source_run': str(source), 'source_archive_sha256': report['sha256'], 'bound_source_sha256': bound,
        'source_sha256': frozen, 'data_manifest_sha256': file_hash(out / 'DATA_MANIFEST.json'),
        'cost_estimate': {'basis': 'existing synthetic CPU4 warm_start smoke02; not real-object timing',
            'optimizer_512_steps_linear_seconds': 1.624081593000028 * 512 / 4,
            'sixteen_endpoint_audit_and_double_MC_linear_seconds': 16 * (0.13922124399960012 + 1.2405194970001503),
            'perturbation_runtime_not_measured': True, 'restoration_or_line_search_can_cost_more': True}}
    assert_contract(protocol); write_json(out / 'RUN_PROTOCOL.json', protocol)
    verify_protocol(out)
    return protocol


def verify_protocol(run):
    run = Path(run).resolve(); p = read_json(run / 'RUN_PROTOCOL.json'); assert_contract(p)
    require(file_hash(run / 'DATA_MANIFEST.json') == p['data_manifest_sha256'], 'Prepared data manifest changed')
    for relative, digest in read_json(run / 'DATA_MANIFEST.json')['files'].items():
        require(file_hash(safe_path(run, relative)) == digest, 'Prepared input changed: ' + relative)
    for path, digest in {**p['bound_source_sha256'], **p['source_sha256']}.items():
        require(file_hash(path) == digest, 'Frozen source or implementation changed: ' + path)
    return p


def perturb(endpoint_u, ambient_noise, original_energy, fraction):
    require(np.isfinite(original_energy) and original_energy >= 0 and fraction in FRACTIONS, 'Invalid fixed perturbation')
    u = np.array(endpoint_u, dtype=np.float64, copy=True)
    if fraction == 0 or original_energy == 0:
        return u
    noise = np.asarray(ambient_noise, np.float64).copy()
    require(noise.shape == u.shape and np.isfinite(noise).all(), 'Invalid ambient noise')
    noise -= np.sum(noise * u, axis=-1, keepdims=True) * u
    rms = float(np.sqrt(np.square(noise).mean())); require(rms > 0, 'Degenerate tangent noise')
    noise /= rms; u += fraction * np.sqrt(original_energy) * noise
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def endpoint_row(run, p, uid, name, u, data, support, fixed_floor, claimed_vertices=None):
    row = {'uid': uid, 'endpoint': name, 'status': 'error', 'offline_teacher_study': True,
        'no_fixed_fallback': True, 'mainfield_hash': array_hash(data['logr']),
        'data_file_sha256': file_hash(run / 'data' / (uid + '.npz'))}
    try:
        u = np.asarray(u, np.float64)
        require(u.shape == (258, 3) and np.isfinite(u).all()
            and np.allclose(np.linalg.norm(u, axis=1), 1, rtol=0, atol=1e-9), 'Invalid endpoint directions')
        r = support.query(data['logr'], u); vertices = u * r[:, None]
        if claimed_vertices is not None:
            require(np.array_equal(vertices, claimed_vertices), 'Stale or inconsistent optimizer final radii')
        mesh_path = run / 'endpoints' / uid / name / 'mesh.npz'
        save_npz(mesh_path, u=u, r=r, vertices=vertices, faces=data['faces'].astype(np.int64))
        row.update(mesh_file=str(mesh_path.relative_to(run)), **mesh_identity(mesh_path, data['faces']))
        before = time.perf_counter(); q = named_quality(vertices, data['faces'])
        audit = audit_mesh(vertices, data['faces'], exact=True)
        row['quality_and_exact_seconds'] = time.perf_counter() - before
        strict = stricter_floor(fixed_floor)
        row.update(quality=q, audit=audit, fixed_floor={k: fixed_floor[k] for k in ANGLES}, strict_floor=strict,
            qualified_original=qualification(q, audit, fixed_floor), qualified_strict=qualification(q, audit, strict),
            actual_margin_vs_fixed_deg={k: q[k] - fixed_floor[k] for k in ANGLES},
            actual_margin_vs_strict_deg={k: q[k] - strict[k] for k in ANGLES},
            final_radius_requery_exact=True, direction_determinants_positive=bool(dets(u, data['faces']).min() > 0))
        index = p['all16_sorted_uids'].index(uid); seeds = [base + index for base in MC_SEED_BASES]
        before = time.perf_counter()
        cached = metric_cache(run, uid, vertices, data['faces'], data['ref_vertices'], data['ref_faces'], seeds, POINTS)
        row['offline_surface_seconds'] = time.perf_counter() - before
        row.update(status='success', metrics=[x['metrics'] for x in cached],
            GT_reference_hash=array_hash(data['ref_vertices'], data['ref_faces']),
            surface_reference='original GT triangles; offline only; not used for optimization or output selection')
    except Exception as exc:
        row.update(error=repr(exc), traceback=traceback.format_exc())
    write_json(run / 'endpoints' / uid / name / 'RESULT.json', row)
    return row


def unavailable_row(run, uid, name, error):
    row = {'uid': uid, 'endpoint': name, 'status': 'unavailable', 'error': error,
        'no_fixed_fallback': True, 'planned_slot_retained': True}
    write_json(run / 'endpoints' / uid / name / 'RESULT.json', row)
    return row


def optimize_task(run, p, uid, initializer, data, support, floor):
    folder = new_dir(run / 'attempts' / uid / initializer)
    info = {'uid': uid, 'initializer': initializer, 'status': 'error', 'max_steps': STEPS,
        'strict_floor': stricter_floor(floor), 'no_fixed_fallback': True,
        'seed': OPT_SEED_BASE + p['all16_sorted_uids'].index(uid)}
    endpoint = None
    try:
        start_u = data['base_u'] if initializer == 'fixed' else data['teacher_u']
        before = time.perf_counter()
        # GT geometry and independent surface-evaluation samples do not enter this call.
        result = optimize_layout(data['logr'], start_u, data['faces'], optimizer_floor(floor),
            device='cpu', max_steps=STEPS, max_seconds=MAX_SECONDS, seed=info['seed'], snapshot_steps=SNAPSHOTS)
        info['optimizer_call_seconds'] = time.perf_counter() - before
        serialize_optimizer(result, folder)
        requested, terminal = snapshot_records(result, SNAPSHOTS)
        info.update(status='optimizer_call_completed', stats=result['stats'],
            requested_snapshots=[{k: v for k, v in item.items() if k != 'snapshot'} for item in requested],
            extra_terminal_snapshots=[item['iteration'] for item in terminal],
            restore_failed=result['stats']['best_feasible_iteration'] is None)
        endpoint = endpoint_row(run, p, uid, 'opt_' + initializer, result['u'], data, support, floor, result['vertices'])
        info['endpoint_status'] = endpoint['status']
    except Exception as exc:
        info.update(error=repr(exc), traceback=traceback.format_exc())
    if endpoint is None:
        endpoint = unavailable_row(run, uid, 'opt_' + initializer, info.get('error', 'No optimizer endpoint'))
    info['endpoint_record'] = endpoint
    write_json(folder / 'TASK.json', info)
    return info, endpoint


def check_endpoint_slots(rows):
    expected = {(uid, name) for uid in UIDS for name in ENDPOINTS}
    keys = [(row['uid'], row['endpoint']) for row in rows]
    require(len(keys) == 16 and len(set(keys)) == 16 and set(keys) == expected,
            'Missing or duplicate predeclared endpoint slot')


def sensitivity(run, p, rows):
    lookup = {(row['uid'], row['endpoint']): row for row in rows}; support = RadialSupport()
    output = []
    for uid in UIDS:
        data = read_data(run / 'data' / (uid + '.npz'))
        energy = float(np.square(data['teacher_u'] - data['base_u']).mean())
        floor = named_quality(data['base_u'] * support.query(data['logr'], data['base_u'])[:, None], data['faces'])
        for name in SENSITIVE_ENDPOINTS:
            record = lookup[(uid, name)]; center = None; unavailable = None
            try:
                require('mesh_file' in record, 'Endpoint mesh unavailable')
                path = safe_path(run, record['mesh_file'])
                require(file_hash(path) == record['mesh_npz_sha256'], 'Endpoint file changed')
                with np.load(path, allow_pickle=False) as z:
                    center = np.array(z['u'])
                    require(np.array_equal(z['faces'], data['faces']), 'Endpoint faces changed')
                    require(np.array_equal(z['r'], support.query(data['logr'], center))
                        and np.array_equal(z['vertices'], center * z['r'][:, None]), 'Endpoint radius mismatch')
            except Exception as exc:
                unavailable = repr(exc); center = None
            for trial in range(TRIALS):
                seed = NOISE_SEED_BASE + TRIALS * p['all_train_uids'].index(uid) + trial
                ambient = np.random.default_rng(seed).normal(size=(258, 3))
                for fraction in FRACTIONS:
                    row = {'uid': uid, 'endpoint': name, 'trial': trial, 'seed': seed, 'fraction': fraction,
                        'status': 'unavailable' if center is None else 'error',
                        'original_teacher_energy': energy, 'nominal_tangent_component_RMS': fraction * np.sqrt(energy),
                        'zero_teacher_noise_degenerate': energy == 0, 'informative_perturbation': energy > 0 and fraction > 0,
                        'angular_pass': False, 'direction_determinants_positive': False}
                    if center is None:
                        row['error'] = unavailable
                    else:
                        try:
                            u = perturb(center, ambient, energy, fraction)
                            q = named_quality(u * support.query(data['logr'], u)[:, None], data['faces'])
                            margins = {k: q[k] - floor[k] for k in ANGLES}
                            row.update(status='success', angular_pass=all(v >= -1e-6 for v in margins.values()),
                                margins_vs_fixed_deg=margins, margins_vs_strict_deg={k: v - MARGIN for k, v in margins.items()},
                                direction_determinants_positive=bool(dets(u, data['faces']).min() > 0),
                                actual_direction_MSE_to_endpoint=float(np.square(u - center).mean()),
                                min_triangle_angle_deg=q[ANGLES[0]], triangle_min_angle_p05_deg=q[ANGLES[1]])
                        except Exception as exc:
                            row.update(error=repr(exc))
                    output.append(row)
    expected = {(u, e, t, f) for u in UIDS for e in SENSITIVE_ENDPOINTS for t in range(TRIALS) for f in FRACTIONS}
    require(len(output) == 1344 and {(r['uid'], r['endpoint'], r['trial'], r['fraction']) for r in output} == expected,
        'Sensitivity slots lost or duplicated')
    per_parent = []
    for uid in UIDS:
        for name in SENSITIVE_ENDPOINTS:
            for fraction in FRACTIONS:
                group = [r for r in output if (r['uid'], r['endpoint'], r['fraction']) == (uid, name, fraction)]
                informative = all(r['informative_perturbation'] for r in group)
                passed = sum(r['angular_pass'] for r in group)
                per_parent.append({'uid': uid, 'endpoint': name, 'fraction': fraction, 'planned': TRIALS,
                    'angular_pass': passed, 'success': sum(r['status'] == 'success' for r in group),
                    'missing_or_errors': sum(r['status'] != 'success' for r in group),
                    'informative_perturbation': informative,
                    'angular_pass_fraction_informative': passed / TRIALS if informative else None,
                    'scope': 'angular conditions only; not exact validity or learned error distribution'})
    result = {'planned_slots': 1344, 'rows': output, 'per_parent': per_parent,
        'no_dropped_missing_or_errors': True, 'zero_teacher_cannot_establish_noise_robustness': True,
        'random_noise_is_not_learned_error': True, 'no_exact_intersection_audit': True,
        'protocol_sha256': file_hash(run / 'RUN_PROTOCOL.json')}
    write_json(run / 'SENSITIVITY.json', result)
    return result


def paired_gain(row, reference):
    if row['status'] != 'success' or reference['status'] != 'success':
        return None
    require([x['seed'] for x in row['metrics']] == [x['seed'] for x in reference['metrics']], 'MC pairing differs')
    require(row['mainfield_hash'] == reference['mainfield_hash'] and row['GT_reference_hash'] == reference['GT_reference_hash'],
        'Surface references differ')
    a = np.array([x['rms_bbox_percent'] for x in row['metrics']])
    b = np.array([x['rms_bbox_percent'] for x in reference['metrics']])
    return None if (b <= 0).any() else (100 * (1 - a / b)).tolist()


def summarize(rows, tasks, sensitivity_result):
    check_endpoint_slots(rows)
    require(len(tasks) == 8 and {(t['uid'], t['initializer']) for t in tasks}
        == {(uid, name) for uid in UIDS for name in INITIALIZERS}, 'Optimizer task slots lost')
    lookup = {(row['uid'], row['endpoint']): row for row in rows}; paired = []
    for task in tasks:
        uid, name = task['uid'], 'opt_' + task['initializer']; row = lookup[(uid, name)]
        paired.append({'uid': uid, 'initializer': task['initializer'], 'endpoint_status': row['status'],
            'restore_failed': task.get('restore_failed'), 'optimizer_status': task.get('status'),
            'actual_steps': task.get('stats', {}).get('iterations'),
            'qualified_original': row.get('qualified_original', {}).get('qualified'),
            'qualified_strict': row.get('qualified_strict', {}).get('qualified'),
            'margins_vs_fixed_deg': row.get('actual_margin_vs_fixed_deg'),
            'gain_vs_fixed_percent': paired_gain(row, lookup[(uid, 'fixed')]),
            'gain_vs_teacher_percent': paired_gain(row, lookup[(uid, 'teacher')]),
            'optimizer_call_seconds': task.get('optimizer_call_seconds')})
    return {'planned_tasks': 8, 'planned_endpoint_rows': 16, 'planned_sensitivity_slots': 1344,
        'completed_optimizer_calls': sum(t['status'] == 'optimizer_call_completed' for t in tasks),
        'successful_endpoint_evaluations': sum(r['status'] == 'success' for r in rows),
        'restore_failed_count': sum(t.get('restore_failed') is True for t in tasks),
        'strict_qualified_optimizer_endpoints': sum(r['qualified_strict'] is True for r in paired),
        'sensitivity_success_slots': sum(r['status'] == 'success' for r in sensitivity_result['rows']),
        'all_planned_slots_retained': True, 'paired_endpoints': paired,
        'zero_teacher_UIDs': [uid for uid in UIDS if any(r['uid'] == uid and r['zero_teacher_noise_degenerate'] for r in sensitivity_result['rows'])],
        'no_GT_selection_no_old_label_updates': True, 'no_global_infeasibility_certificate': True,
        'not_a_learned_method_or_generalization_result': True}


def run(run_dir):
    run_dir = Path(run_dir).resolve(); p = verify_protocol(run_dir)
    require(not torch.cuda.is_initialized(), 'CPU only; GPU context must not exist')
    require(not any((run_dir / name).exists() for name in ('attempts', 'endpoints', 'RESULTS.json', 'SENSITIVITY.json')),
        'No overwrites or retries in an existing attempt directory')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    support = RadialSupport(); rows = []; tasks = []
    for uid in UIDS:
        data = read_data(run_dir / 'data' / (uid + '.npz'))
        fixed_vertices = data['base_u'] * support.query(data['logr'], data['base_u'])[:, None]
        floor = named_quality(fixed_vertices, data['faces'])
        for name, key in [('fixed', 'base_u'), ('teacher', 'teacher_u')]:
            rows.append(endpoint_row(run_dir, p, uid, name, data[key], data, support, floor))
        for initializer in INITIALIZERS:
            task, row = optimize_task(run_dir, p, uid, initializer, data, support, floor)
            tasks.append(task); rows.append(row)
            print(json.dumps({'uid': uid, 'initializer': initializer, 'status': task['status'],
                'actual_steps': task.get('stats', {}).get('iterations'), 'restore_failed': task.get('restore_failed')}), flush=True)
    check_endpoint_slots(rows); verify_protocol(run_dir)
    result = {'protocol_sha256': file_hash(run_dir / 'RUN_PROTOCOL.json'), 'rows': rows, 'tasks': tasks,
        'source_inputs_unchanged': True, 'no_old_labels_written': True, 'no_GPU_context_created': not torch.cuda.is_initialized()}
    write_json(run_dir / 'RESULTS.json', result)
    neighborhood = sensitivity(run_dir, p, rows); summary = summarize(rows, tasks, neighborhood)
    verify_protocol(run_dir)
    summary.update(results_sha256=file_hash(run_dir / 'RESULTS.json'), sensitivity_sha256=file_hash(run_dir / 'SENSITIVITY.json'))
    write_json(run_dir / 'SUMMARY.json', summary)
    lines = ['# 四训练对象的更严格教师可行性诊断', '',
        '固定八次原优化器调用；Fixed 双角度门槛各加 0.1 度，每次最多 64 步。无 Fixed 回退，无 GT 选输出，无旧标签更新。',
        '这不是学习方法的在线优势或泛化结果。恢复失败只表示固定局部策略未找到可行端点；不能证明不存在可行网格。', '',
        '|UID|起点|实际步数|新门槛合格|相对 Fixed 双 MC 改善(%)|相对原 teacher 双 MC 改善(%)|',
        '|---|---|---:|---|---|---|']
    for item in summary['paired_endpoints']:
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(item['uid'], item['initializer'], item['actual_steps'],
            item['qualified_strict'], item['gain_vs_fixed_percent'], item['gain_vs_teacher_percent']))
    lines += ['', '敏感性固定 1344 槽；相同 UID/试验共用原始 Gaussian，幅度统一按原 teacher 位移确定。',
        '零位移 teacher 的噪声幅度恒为零，其通过计数不代表扰动稳健性。随机独立切向扰动不代表网络学习误差。',
        '扰动只核查角条件与方向行列式，不具备完整精确相交认证；所有失败与缺失均保留。']
    (run_dir / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--source-run', required=True); prep.add_argument('--out', required=True)
    execute = sub.add_parser('run'); execute.add_argument('--run', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        p = prepare(args.source_run, args.out)
        print(json.dumps({'prepared': str(Path(args.out).resolve()), 'tasks': p['expected_tasks'],
            'endpoint_rows': p['expected_endpoint_rows'], 'sensitivity_slots': p['sensitivity']['planned_slots'],
            'protocol_sha256': file_hash(Path(args.out) / 'RUN_PROTOCOL.json')}, indent=2), flush=True)
    else:
        print(json.dumps(run(args.run), indent=2), flush=True)


if __name__ == '__main__':
    main()
