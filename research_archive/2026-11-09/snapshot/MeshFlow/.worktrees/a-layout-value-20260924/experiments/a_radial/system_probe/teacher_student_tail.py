"""Fixed 72-point, CPU-only teacher-to-student geometry diagnosis.

No network, optimization, Monte Carlo sampling, repair or output selection.
The interpolants are diagnostic geometry and are never replacement outputs.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import math
import traceback
from pathlib import Path
import numpy as np
import torch

from ..exporter import RadialSupport, ASSETS
from ..io_utils import array_hash, file_hash, save_npz, write_json
from ..legacy.chart import dets
from ..layout_student.deployment import named_quality, QUALITY_EPS_DEG
from ..layout_student.integrity import mesh_identity

UIDS = ('7f1be3e7b6adeeae9bb399e812556da2',
        '2379915c82c2b2d02530f02db7e9157e',
        '8c919e992db7cb081e5b68ee10358ea3',
        '736820a85063a4371e5b68ee10358ea3')
KINDS = ('old', 'margin')
LAMBDAS = tuple(i / 8 for i in range(9))
ANGLE_KEYS = ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def validate_u(u):
    require(u.dtype == np.float64 and u.ndim == 2 and u.shape[1] == 3,
            'Directions must be float64 [N,3]')
    require(np.isfinite(u).all() and np.allclose(np.linalg.norm(u, axis=1), 1,
            rtol=0, atol=1e-9), 'Directions are nonfinite or nonunit')


def interpolate(teacher, student, lam):
    validate_u(teacher); validate_u(student)
    require(teacher.shape == student.shape and 0 <= lam <= 1, 'Invalid bridge')
    # Endpoint copies preserve the saved binary64 directions and radius identity.
    if lam == 0:
        return teacher.copy()
    if lam == 1:
        return student.copy()
    chord = (1 - lam) * teacher + lam * student
    length = np.linalg.norm(chord, axis=1)
    require(np.isfinite(length).all() and np.all(length > 1e-12),
            'Undefined/near-antipodal normalized chord; no repair')
    return chord / length[:, None]


def face_details(vertices, faces):
    require(np.issubdtype(faces.dtype, np.integer), 'Noninteger connectivity')
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tol = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) ** 2, 1e-300) * 1e-14
    columns = []
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        columns.append(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                           np.einsum('ij,ij->i', a, b))))
    minimum = np.stack(columns, axis=1).min(axis=1)
    minimum[area2 <= tol] = 0
    return minimum, .5 * area2, area2 <= tol


def tail_details(minimum):
    require(minimum.ndim == 1 and len(minimum) > 1 and np.isfinite(minimum).all(), 'Invalid angles')
    # Stable ordering gives deterministic face IDs even at tied angles.
    order = np.argsort(minimum, kind='stable')
    position = (len(minimum) - 1) * .05
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    hi_weight = position - lo
    return {'minimum_face': int(order[0]),
            'minimum_tied_faces': np.flatnonzero(minimum == minimum.min()).tolist(),
            'bottom_ceil_5pct_faces': order[:math.ceil(.05 * len(minimum))].tolist(),
            'p05_support_faces': [int(order[lo]), int(order[hi])],
            'p05_support_order_indices_zero_based': [lo, hi],
            'p05_support_weights': [1 - hi_weight, hi_weight],
            'p05_support_angles_deg': minimum[order[[lo, hi]]].tolist(),
            'p05_linear_deg': float(np.quantile(minimum, .05)),
            'order': order}


def direction_errors(teacher, student):
    validate_u(teacher); validate_u(student)
    chord_mse = np.square(student - teacher).mean(axis=1)
    angular = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(teacher, student), axis=1),
                                  np.einsum('ij,ij->i', teacher, student)))
    return chord_mse, angular


def error_subset(chord_mse, angular, vertices):
    ids = np.unique(np.asarray(vertices, np.int64))
    require(ids.size == 0 or (ids.min() >= 0 and ids.max() < len(angular)), 'Invalid subset')
    if not ids.size:
        return {'vertex_count': 0, 'vertex_ids': [], 'direction_coordinate_mse': None,
                'angular_rmse_deg': None, 'angular_max_deg': None,
                'coordinate_mse_over_global': None}
    mse = float(chord_mse[ids].mean()); global_mse = float(chord_mse.mean())
    return {'vertex_count': len(ids), 'vertex_ids': ids.tolist(), 'direction_coordinate_mse': mse,
            'angular_rmse_deg': float(np.sqrt(np.mean(angular[ids] ** 2))),
            'angular_max_deg': float(angular[ids].max()),
            'coordinate_mse_over_global': mse / global_mse if global_mse > 0 else None}


def pair_rows(ledger, uid, kind):
    require(uid in UIDS and kind in KINDS, 'Unregistered pair')
    expected = {(u, e) for u in UIDS for e in
                ('fixed', 'old_teacher', 'margin_teacher', 'old_final', 'margin_final')}
    rows = ledger['all_attempts']
    keys = [(r['uid'], r['endpoint']) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Missing/duplicate/unregistered endpoint')
    table = dict(zip(keys, rows))
    pair = [table[(uid, endpoint)] for endpoint in ('fixed', kind + '_teacher', kind + '_final')]
    require(all(r['status'] == 'success' for r in pair), 'Source endpoint unavailable; no replacement')
    identity = pair[2]['identity']
    require(identity['uid'] == uid and identity['target_kind'] == kind
            and identity['job'] == f'{kind}_{uid[:8]}' and identity['update'] == 20000
            and identity['seed'] == 10, 'Student checkpoint paired with wrong target/object')
    return pair


def source_paths(parent, protocol, ledger):
    names = ['RUN_PROTOCOL.json', 'results.json', 'DATA_MANIFEST.json', 'train_labels.npz',
             'TRAINING_COMPLETE.json', 'EXPLORATORY_JOINT_GATE.json']
    for uid in UIDS:
        names.append(f'data/{uid}.npz')
        for kind in KINDS:
            for row in pair_rows(ledger, uid, kind):
                names.extend([row['mesh_file'], str(Path(row['mesh_file']).parent / 'RESULT.json')])
            final = pair_rows(ledger, uid, kind)[2]
            names.append(final['identity']['checkpoint_file'])
            names.append(f'diagnostics/{kind}_{uid[:8]}/20000/directions.npz')
            names.append(f'diagnostics/{kind}_{uid[:8]}/20000/RESULT.json')
    return {str((parent / name).resolve()): file_hash(parent / name) for name in sorted(set(names))}


def prepare(parent, out):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output already exists; no overwrite')
    p, ledger = read(parent / 'RUN_PROTOCOL.json'), read(parent / 'results.json')
    require(tuple(p['uids']) == UIDS and p['training']['total_updates'] == 20000,
            'Wrong source study')
    require(ledger['recorded_slots'] == ledger['success_count'] == 20 and not ledger['errors'],
            'Source 20 endpoints incomplete')
    require(file_hash(parent / 'train_labels.npz') == p['hashes']['training_labels_sha256']
            and file_hash(parent / 'DATA_MANIFEST.json') == p['hashes']['data_manifest_sha256'],
            'Source labels/data manifest do not match source protocol')
    require(ledger['protocol_sha256'] == file_hash(parent / 'RUN_PROTOCOL.json')
            and all(r['protocol_sha256'] == ledger['protocol_sha256'] for r in ledger['all_attempts']),
            'Source endpoint protocol mismatch')
    files = source_paths(parent, p, ledger)
    source_root = Path(__file__).resolve().parents[1]
    bound = ['system_probe/teacher_student_tail.py', 'system_probe/test_teacher_student_tail.py', 'exporter.py', 'io_utils.py',
             'legacy/chart.py', 'legacy/geometry_task.py', 'layout_value/geometry.py',
             'layout_student/deployment.py', 'layout_student/integrity.py']
    protocol = {'schema': 'A-TEACHER-STUDENT-TAIL-v1', 'parent_run': str(parent),
        'uids': UIDS, 'kinds': KINDS, 'lambdas': LAMBDAS, 'pairs': 8, 'geometry_checks': 72,
        'interpolation': 'normalize((1-lambda)*teacher + lambda*student); saved binary64 copies at lambda=0/1',
        'radii': 'query same frozen 2050-node field independently at every accepted diagnostic direction',
        'quality_floor': 'saved common Fixed minimum angle and linear p05, tolerance 1e-6 degree',
        'quality_eps_deg': QUALITY_EPS_DEG,
        'tail_definition': 'stable bottom ceil(.05*F)=26 faces; separately linear p05 ranks25/26 with weights .45/.55',
        'direction_error': 'per-vertex mean squared xyz chord; angular atan2(cross norm,dot) in degrees',
        'failure_face_definition': 'true minimum-angle violations; p05 bottom27/support contributors only if aggregate p05 fails, not independent per-face validity',
        'geometry_reference': 'same frozen radial mainfield complete2050node4096face surface, not original asset GT',
        'exact_intersection_or_topology_audit_executed': False,
        'quality_and_direction_det_only_not_full_validity': True,
        'no_MC': True, 'no_network': True, 'no_training': True, 'no_optimizer': True,
        'no_repair': True, 'no_output_or_lambda_selection': True,
        'diagnostic_interpolants_are_not_algorithm_outputs': True,
        'source_files': files,
        'source_sha256': {str(source_root / name): file_hash(source_root / name) for name in bound},
        'support_sha256': file_hash(ASSETS / 'support.npz'),
        'source_protocol_sha256': file_hash(parent / 'RUN_PROTOCOL.json')}
    out.mkdir(parents=True)
    write_json(out / 'RUN_PROTOCOL.json', protocol)
    return protocol


def verify_bound(p):
    for path, sha in {**p['source_files'], **p['source_sha256']}.items():
        require(file_hash(path) == sha, 'Bound source changed: ' + path)
    require(file_hash(ASSETS / 'support.npz') == p['support_sha256'], 'Support changed')


def verified_mesh(parent, row, data):
    require(file_hash(parent / row['mesh_file']) == row['mesh_npz_sha256'], 'Source mesh file changed')
    identity = mesh_identity(parent / row['mesh_file'], data['faces'])
    require(all(identity[k] == row[k] for k in identity), 'Source mesh identity changed')
    require(file_hash(parent / 'data' / (row['uid'] + '.npz')) == row['data_file_sha256'], 'Data hash changed')
    require(array_hash(data['logr']) == row['mainfield_hash'], 'Wrong field')
    mesh = arrays(parent / row['mesh_file'])
    validate_u(mesh['u'])
    require(mesh['u'].shape == (258, 3) and np.array_equal(mesh['faces'], data['faces']), 'Connectivity changed')
    require(np.array_equal(mesh['vertices'], mesh['u'] * mesh['r'][:, None]), 'Stale saved radius/vertices')
    return mesh


def endpoint_check(u, radius, vertices, quality, mesh, row):
    require(np.array_equal(u, mesh['u']) and np.array_equal(radius, mesh['r'])
            and np.array_equal(vertices, mesh['vertices']), 'Endpoint direction/radius/vertices replay differs')
    for k in ('face_count', 'degenerate_count', *ANGLE_KEYS, 'area_min', 'area_total'):
        require(quality[k] == row['quality'][k], 'Endpoint quality replay differs: ' + k)
    return {'u_radius_vertices_bitwise_equal': True, 'angle_area_replay_exact': True,
            'original_row_mesh_hash': row['mesh_hash'], 'original_mesh_npz_sha256': row['mesh_npz_sha256']}


def analyze_pair(parent, uid, kind, ledger, labels, support, out):
    fixed_row, teacher_row, student_row = pair_rows(ledger, uid, kind)
    data = arrays(parent / 'data' / (uid + '.npz'))
    faces = data['faces']
    require(faces.shape == (512, 3) and faces.dtype == np.int32, 'Wrong face template')
    rv, rf = support.mesh(data['logr'])
    require(np.array_equal(rv, data['ref_vertices']) and np.array_equal(rf, data['ref_faces']), 'Wrong full-field reference')
    teacher = verified_mesh(parent, teacher_row, data)
    student = verified_mesh(parent, student_row, data)
    fixed = verified_mesh(parent, fixed_row, data)
    i = list(labels['train_uids']).index(uid)
    require(np.array_equal(teacher['u'], labels['directions1_' + kind][i])
            and np.array_equal(fixed['u'], labels['directions0'][i]), 'Teacher label UID/condition mismatch')
    diag = read(parent / f'diagnostics/{kind}_{uid[:8]}/20000/RESULT.json')
    diag_npz = arrays(parent / diag['directions_file'])
    require(diag['identity'] == student_row['identity'] and np.array_equal(diag_npz['u'], student['u']),
            'Saved student/diagnostic identity mismatch')
    floor = fixed_row['fixed_floor']
    require(teacher_row['fixed_floor'] == student_row['fixed_floor'] == floor, 'Floor mismatch')
    chord_mse, angular = direction_errors(teacher['u'], student['u'])
    rows, stored = [], {k: [] for k in ('u', 'r', 'vertices', 'face_min_angle_deg', 'face_area', 'direction_det')}
    baseline_tail = None
    for lam in LAMBDAS:
        u = interpolate(teacher['u'], student['u'], lam)
        radius = support.query(data['logr'], u)
        vertices = u * radius[:, None]
        quality = named_quality(vertices, faces)
        minimum, areas, degenerate = face_details(vertices, faces)
        require(float(minimum.min()) == quality[ANGLE_KEYS[0]] and float(np.quantile(minimum, .05)) == quality[ANGLE_KEYS[1]],
                'Independent per-face angle calculation differs')
        tail = tail_details(minimum)
        order = tail.pop('order')
        current = set(tail['bottom_ceil_5pct_faces'])
        if baseline_tail is None:
            baseline_tail = current
        determinant = dets(u, faces)
        margins = {k: quality[k] - floor[k] for k in ANGLE_KEYS}
        passed = {k: quality[k] + QUALITY_EPS_DEG >= floor[k] for k in ANGLE_KEYS}
        minimum_violators = np.flatnonzero(minimum + QUALITY_EPS_DEG < floor[ANGLE_KEYS[0]])
        p05_contributors = order[:int(math.ceil((len(faces) - 1) * .05)) + 1] if not passed[ANGLE_KEYS[1]] else np.array([], np.int64)
        row = {'uid': uid, 'kind': kind, 'lambda': lam, 'quality': quality,
            'margin_vs_fixed_deg': margins, 'angle_conditions_pass': all(passed.values()),
            'individual_angle_conditions_pass': passed, 'tail': tail,
            'teacher_tail_retained_count': len(current & baseline_tail),
            'teacher_tail_left_faces': sorted(baseline_tail - current),
            'new_tail_faces_since_teacher': sorted(current - baseline_tail),
            'tail_jaccard_vs_teacher': len(current & baseline_tail) / len(current | baseline_tail),
            'minimum_floor_violating_faces': minimum_violators.tolist(),
            'p05_failure_contributor_faces': p05_contributors.tolist(),
            'direction_det_min': float(determinant.min()),
            'direction_det_nonpositive_faces': np.flatnonzero(determinant <= 0).tolist(),
            'direction_dets_positive': bool(np.all(determinant > 0)),
            'degenerate_faces': np.flatnonzero(degenerate).tolist(),
            'full_geometry_validity': 'NOT_TESTED', 'exact_audit_executed': False,
            'endpoint_replay': None}
        if lam in (0, 1):
            row['endpoint_replay'] = endpoint_check(u, radius, vertices, quality,
                    teacher if lam == 0 else student, teacher_row if lam == 0 else student_row)
        rows.append(row)
        for key, value in zip(stored, (u, radius, vertices, minimum, areas, determinant)):
            stored[key].append(value)
    first, last = rows[0], rows[-1]
    teacher_vertices = np.unique(faces[first['tail']['bottom_ceil_5pct_faces']])
    contributors = sorted(set(last['minimum_floor_violating_faces']) | set(last['p05_failure_contributor_faces']))
    subsets = {
        'global': np.arange(len(teacher['u'])),
        'teacher_bottom26_corner_vertices': teacher_vertices,
        'teacher_p05_support_corner_vertices': np.unique(faces[first['tail']['p05_support_faces']]),
        'teacher_minimum_face_vertices': faces[first['tail']['minimum_face']],
        'student_bottom26_corner_vertices': np.unique(faces[last['tail']['bottom_ceil_5pct_faces']]),
        'student_minimum_face_vertices': faces[last['tail']['minimum_face']],
        'student_minimum_floor_violator_vertices': np.unique(faces[last['minimum_floor_violating_faces']]),
        'student_failed_angle_contributor_vertices': np.unique(faces[contributors]),
        'student_new_tail_corner_vertices': np.unique(faces[last['new_tail_faces_since_teacher']]),
    }
    detail = {'uid': uid, 'kind': kind, 'teacher_endpoint': teacher_row['endpoint'],
        'student_endpoint': student_row['endpoint'], 'student_identity': student_row['identity'],
        'fixed_floor': floor, 'rows': rows,
        'error_subsets': {k: error_subset(chord_mse, angular, ids) for k, ids in subsets.items()},
        'student_minimum_face_was_in_teacher_tail': last['tail']['minimum_face'] in baseline_tail,
        'teacher_minimum_face_final_angle_deg': float(stored['face_min_angle_deg'][-1][first['tail']['minimum_face']]),
        'student_minimum_face_teacher_angle_deg': float(stored['face_min_angle_deg'][0][last['tail']['minimum_face']]),
        'minimum_face_switched': first['tail']['minimum_face'] != last['tail']['minimum_face'],
        'student_failure_face_ids_are_diagnostic_contributors_not_proof_of_causality': True,
        'student_original_full_qualification': student_row['qualified_original'],
        'teacher_original_full_qualification': teacher_row['qualified_original']}
    folder = out / 'pairs' / uid / kind
    folder.mkdir(parents=True)
    save_npz(folder / 'geometry.npz', faces=faces, lambdas=np.array(LAMBDAS),
             vertex_chord_coordinate_mse=chord_mse, vertex_angular_error_deg=angular,
             **{k: np.stack(v) for k, v in stored.items()})
    detail['geometry_file'] = str((folder / 'geometry.npz').relative_to(out))
    detail['geometry_file_sha256'] = file_hash(folder / 'geometry.npz')
    write_json(folder / 'RESULT.json', detail)
    return detail


def visualize(out, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, mode in [('angle_margins', 'angles'), ('direction_error_localization', 'errors')]:
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
        for pair, ax in zip(pairs, axes.flat):
            uid, kind = pair['uid'], pair['kind']
            if mode == 'angles':
                for key, color, label in zip(ANGLE_KEYS, ('#245d9d', '#bc4b32'), ('Minimum angle', 'p05 of face minima')):
                    ax.plot(LAMBDAS, [r['margin_vs_fixed_deg'][key] for r in pair['rows']], '-o', color=color, label=label, markersize=4)
                ax.axhline(0, color='black', linewidth=.8)
                ax.set_ylabel('Angle minus Fixed floor (degrees)')
                ax.set_xlabel('Teacher (0) to saved student (1)')
            else:
                g = arrays(out / pair['geometry_file'])
                e = g['vertex_angular_error_deg']; ids = np.arange(len(e))
                tail = pair['error_subsets']['teacher_bottom26_corner_vertices']['vertex_ids']
                bad = pair['error_subsets']['student_failed_angle_contributor_vertices']['vertex_ids']
                ax.scatter(ids, e, s=8, color='#acb4bd', label='All vertices')
                ax.scatter(tail, e[tail], s=15, color='#245d9d', label='Teacher bottom26 corners')
                ax.scatter(bad, e[bad], s=24, marker='x', color='#bc4b32', label='Student failed-angle contributors')
                ax.set_xlabel('Template vertex index'); ax.set_ylabel('Teacher-student direction error (degrees)')
            ax.set_title(uid[:8] + ' / ' + kind)
            ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Fixed saved endpoints: interpolation diagnosis only; no exact intersection audit', fontsize=13)
        fig.savefig(out / (filename + '.png'), dpi=170)
        fig.savefig(out / (filename + '.pdf'))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    matrix = np.array([[r['teacher_tail_retained_count'] for r in p['rows']] for p in pairs])
    im = ax.imshow(matrix, vmin=0, vmax=26, cmap='viridis', aspect='auto')
    ax.set_xticks(range(9), [str(l) for l in LAMBDAS]); ax.set_xlabel('Teacher (0) to saved student (1)')
    ax.set_yticks(range(8), [p['uid'][:8] + '/' + p['kind'] for p in pairs])
    for i in range(8):
        for j in range(9):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] < 14 else 'black')
    fig.colorbar(im, ax=ax, label='Original teacher bottom26 faces retained')
    ax.set_title('Tail membership migration; p05 also uses rank26 (the 27th face)')
    fig.savefig(out / 'tail_membership.png', dpi=170); fig.savefig(out / 'tail_membership.pdf')
    plt.close(fig)


def report_text(pairs):
    lines = ['# Teacher-to-student tail geometry diagnosis', '',
        'Fixed 8 pairs x 9 lambdas = 72 diagnostic geometries. No training, network, optimization, MC, repair or output selection.',
        'Reference: the same frozen radial field. No new exact intersection/topology audit; angle gates and positive direction determinants do not certify full validity.', '',
        '|UID / target|global direction RMSE deg|teacher-tail / global coordinate MSE|student min margin deg|student p05 margin deg|teacher tail retained /26|min face teacher->student|',
        '|---|---:|---:|---:|---:|---:|---|']
    for p in pairs:
        a, b = p['rows'][0], p['rows'][-1]; e = p['error_subsets']
        ratio = e['teacher_bottom26_corner_vertices']['coordinate_mse_over_global']
        lines.append(f"|{p['uid'][:8]} / {p['kind']}|{e['global']['angular_rmse_deg']:.6f}|{ratio:.3f}|"
            f"{b['margin_vs_fixed_deg'][ANGLE_KEYS[0]]:.6f}|{b['margin_vs_fixed_deg'][ANGLE_KEYS[1]]:.6f}|"
            f"{b['teacher_tail_retained_count']}|{a['tail']['minimum_face']} -> {b['tail']['minimum_face']}|")
    lines += ['', 'The global MSE averages all 258 vertex coordinates. Minimum angle depends on one local triangle; p05 depends on two order statistics and can change its supporting faces. Local error, differential corner motion, changing radii and rank migration are not bounded by a small global average.',
        'Tail error enrichment is descriptive and is not proof that a particular vertex caused failure. A ratio below one does not guarantee quality: small errors can exceed a small margin or change the ordering.',
        'Each endpoint is re-queried and checked against the saved directions, radii, vertices and angle/area metrics. Endpoints are not recomputed by a model. Interpolants are not deployable replacements and no lambda is selected.', '']
    return '\n'.join(lines)


def run(out):
    out = Path(out).resolve(); p = read(out / 'RUN_PROTOCOL.json')
    require(not (out / 'START.json').exists(), 'Already attempted; no automatic retry')
    verify_bound(p)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    write_json(out / 'START.json', {'protocol_sha256': file_hash(out / 'RUN_PROTOCOL.json'), 'CPU_threads': 1})
    try:
        parent = Path(p['parent_run']); ledger = read(parent / 'results.json')
        labels = arrays(parent / 'train_labels.npz'); support = RadialSupport(); pairs = []
        for uid in UIDS:
            for kind in KINDS:
                pair = analyze_pair(parent, uid, kind, ledger, labels, support, out)
                pairs.append(pair)
                print(json.dumps({'uid': uid, 'kind': kind, 'geometries': 9, 'endpoint_replay': 'PASS'}), flush=True)
        flat = [r for pair in pairs for r in pair['rows']]
        require(len(flat) == 72, 'Incomplete predetermined denominator')
        summary = {'status': 'COMPLETE', 'pairs': pairs, 'geometry_checks': 72,
            'endpoint_checks': 16, 'all_endpoint_replays_exact': True,
            'direction_det_nonpositive_geometry_count': sum(not r['direction_dets_positive'] for r in flat),
            'degenerate_geometry_count': sum(bool(r['degenerate_faces']) for r in flat),
            'angle_condition_pass_count': sum(r['angle_conditions_pass'] for r in flat),
            'exact_audit_executed': False, 'full_validity_claim': False,
            'output_selection': False, 'cuda_initialized': torch.cuda.is_initialized()}
        write_json(out / 'RESULTS.json', summary)
        (out / 'REPORT.md').write_text(report_text(pairs), encoding='utf-8')
        visualize(out, pairs)
        verify_bound(p)
        require(not torch.cuda.is_initialized(), 'CUDA initialized')
        write_json(out / 'COMPLETE.json', {'status': 'COMPLETE', 'geometry_checks': 72,
            'results_sha256': file_hash(out / 'RESULTS.json'), 'source_files_unchanged': True,
            'endpoint_replays_exact': 16, 'cuda_initialized': False,
            'protocol_sha256': file_hash(out / 'RUN_PROTOCOL.json')})
        print(json.dumps({k: v for k, v in summary.items() if k != 'pairs'}), flush=True)
    except Exception as exc:
        write_json(out / 'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'no_retry': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--source-run')
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.source_run is not None, 'Source run required')
        prepare(args.source_run, args.run)
    else:
        run(args.run)


if __name__ == '__main__':
    main()
