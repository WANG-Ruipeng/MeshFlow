"""Offline, fixed-grid teacher geometry evidence; not a path certificate.

No new surface-distance measurement, teacher optimization, or model selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..legacy.chart import dets
from ..layout_student.deployment import named_quality
from .prepare import read_json, require

LAMBDAS = [0., .125, .25, .375, .5, .625, .75, .875, 1.]
QUALITY_KEYS = ['min_triangle_angle_deg', 'triangle_min_angle_p05_deg']


def normalized_chord(base, teacher, lam):
    if lam == 0.:
        return base.copy()
    if lam == 1.:
        return teacher.copy()
    raw = base + lam * (teacher - base)
    length = np.linalg.norm(raw, axis=-1, keepdims=True)
    require(bool(np.all(length > 1e-12)), 'Degenerate normalized chord')
    return raw / length


def split_summary(records):
    energy = np.asarray([row['teacher_displacement_mse'] for row in records], dtype=np.float64)
    total = float(energy.sum())
    sorted_rows = sorted(records, key=lambda row: (-row['teacher_displacement_mse'], row['uid']))
    moving = [row for row in records if not row['zero_motion_teacher']]
    return {'parent_count': len(records), 'moving_teacher_count': len(moving),
        'zero_motion_uids': [row['uid'] for row in records if row['zero_motion_teacher']],
        'sum_teacher_displacement_mse': total,
        'largest_energy_share': float(max(energy) / total) if total else None,
        'top3_energy_share': float(sum(row['teacher_displacement_mse'] for row in sorted_rows[:3]) / total) if total else None,
        'energy_ranked_uids': [row['uid'] for row in sorted_rows],
        'energy_shares': {row['uid']: row['teacher_displacement_mse'] / total if total else None for row in records},
        'teacher_margin_median_deg': {key: float(np.median([r['teacher_quality_margin_deg'][key] for r in records])) for key in QUALITY_KEYS},
        'moving_teacher_margin_median_deg': {key: float(np.median([r['teacher_quality_margin_deg'][key] for r in moving])) if moving else None for key in QUALITY_KEYS},
        'moving_teachers_with_intermediate_quality_failure': [r['uid'] for r in moving if r['any_sampled_interior_quality_failure']],
        'all_sampled_direction_determinants_positive': all(r['all_sampled_direction_determinants_positive'] for r in records)}


def diagnose_geometry(run):
    run = Path(run).resolve()
    output = run / 'GEOMETRY_DIAGNOSTICS.json'
    require(not output.exists(), 'Geometry evidence must not be overwritten')
    protocol = read_json(run / 'RUN_PROTOCOL.json')
    require(protocol['experiment'] == 'A-FIT-DIAG-v1', 'Wrong experiment')
    require(protocol['geometry_diagnostics']['lambdas'] == LAMBDAS, 'Probe grid differs from protocol')
    manifest_path = run / protocol['paths']['data_manifest']
    require(file_hash(manifest_path) == protocol['hashes']['data_manifest_sha256'], 'Data manifest changed')
    manifest = read_json(manifest_path)['files']
    support = RadialSupport()
    tolerance = protocol['quality']['tolerance_deg']
    require(tolerance == 1e-6, 'Qualification tolerance changed')
    records = []
    for uid in protocol['uids']:
        rel = 'data/' + uid + '.npz'
        require(file_hash(run / rel) == manifest[rel], 'Data changed: ' + uid)
        with np.load(run / rel, allow_pickle=False) as z:
            data = {key: np.array(z[key]) for key in ('base_u', 'teacher_u', 'faces', 'logr')}
        base, teacher, faces, logr = (data[key] for key in ('base_u', 'teacher_u', 'faces', 'logr'))
        require(faces.dtype == np.int32 and faces.shape == (512, 3), 'Connectivity changed')
        qualities = []
        for lam in LAMBDAS:
            directions = normalized_chord(base, teacher, lam)
            radii = support.query(logr, directions)
            vertices = directions * radii[:, None]
            quality = named_quality(vertices, faces)
            determinant_min = float(dets(directions, faces).min())
            qualities.append({'lambda': lam, 'direction_hash': array_hash(directions),
                'requeried_mesh_hash': array_hash(vertices, faces), 'minimum_direction_face_determinant': determinant_min,
                'direction_determinants_positive': determinant_min > 0,
                'quality': quality, 'final_accepted_directions_requeried': True})
        floor = qualities[0]['quality']
        for point in qualities:
            point['quality_margin_deg'] = {key: point['quality'][key] - floor[key] for key in QUALITY_KEYS}
            point['both_angle_floors_pass'] = all(point['quality_margin_deg'][key] >= -tolerance for key in QUALITY_KEYS)
            point['full_endpoint_qualification_not_tested'] = True
        record = {'uid': uid, 'split': 'train' if uid in protocol['train_uids'] else 'internal_check',
            'source_data_sha256': manifest[rel], 'mainfield_hash': array_hash(logr),
            'zero_motion_teacher': bool(np.array_equal(base, teacher)),
            'teacher_displacement_mse': float(np.mean((teacher - base) ** 2)),
            'fixed_quality': floor, 'teacher_quality': qualities[-1]['quality'],
            'teacher_quality_margin_deg': qualities[-1]['quality_margin_deg'],
            'sampled_points': qualities,
            'any_sampled_interior_quality_failure': any(not p['both_angle_floors_pass'] for p in qualities[1:-1]),
            'all_sampled_direction_determinants_positive': all(p['direction_determinants_positive'] for p in qualities),
            'passing_lambda_values': [p['lambda'] for p in qualities if p['both_angle_floors_pass']],
            'minimum_sampled_interior_margin_deg': {key: min(p['quality_margin_deg'][key] for p in qualities[1:-1]) for key in QUALITY_KEYS}}
        require(qualities[0]['both_angle_floors_pass'] and qualities[-1]['both_angle_floors_pass'], 'Frozen teacher endpoint angles failed')
        records.append(record)
    result = {'experiment': protocol['experiment'], 'protocol_sha256': file_hash(run / 'RUN_PROTOCOL.json'),
        'data_manifest_sha256': file_hash(manifest_path), 'source_sha256': file_hash(__file__),
        'teacher_manifest_sha256': file_hash(run / protocol['paths']['teacher_manifest']),
        'source_endpoint_manifest_sha256': protocol['source_run_manifest_sha256'],
        'fixed_lambdas': LAMBDAS, 'quality_tolerance_deg': tolerance,
        'evidence_scope': {'offline_development_geometry_only': True,
            'all16_are_previously_seen_development_objects': True,
            'network_training_access_to_internal_check_teachers': False,
            'records_do_not_select_teacher_or_checkpoint': True,
            'sampled_angle_and_direction_sign_checks_only': True,
            'full_continuous_path_certificate': False, 'intersection_audit_rerun': False,
            'source_teacher_endpoint_audit_reused_from_sealed_teacher_manifest': True,
            'new_surface_RMS_measured': False, 'interpolation_is_not_original_A_rollout': True,
            'endpoints_are_preserved_byte_exactly': True,
            'positive_direction_determinants_do_not_imply_3D_angle_floors': True},
        'records': records,
        'summary': {split: split_summary([r for r in records if r['split'] == split]) for split in ('train', 'internal_check')}}
    write_json(output, result)
    print(json.dumps({'output': str(output), 'summary': result['summary']}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    diagnose_geometry(args.run)


if __name__ == '__main__':
    main()
