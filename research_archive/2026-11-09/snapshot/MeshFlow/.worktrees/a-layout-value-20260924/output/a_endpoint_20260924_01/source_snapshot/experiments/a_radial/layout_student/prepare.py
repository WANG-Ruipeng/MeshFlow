"""Prepare audited layout teachers without exposing check labels to training.

This module reads old results and creates a new immutable run. It never trains,
changes old meshes, or calls the original A network to select a teacher.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from ..exporter import ASSETS, MeshExporter
from ..io_utils import array_hash, file_hash, new_dir, save_npz, write_json
from ..legacy.networks import point_features
from ..layout_value.geometry import mesh_quality

SPLIT_SALT = 'a-taskalign-split-v1:'
MIN_IMPROVEMENT = 0.02
QUALITY_TOLERANCE_DEG = 1e-6
TRAINING = {
    'seeds': [10, 11], 'updates': 3000, 'batch_size': 4,
    'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': 1e-4,
    'checkpoint_selection': 'final_update_3000_no_metric_selection',
    'initialization': 'original_A_10.pt_for_both_training_seeds',
    'schedule': 'constant_learning_rate', 'precision': 'float32',
    'objective': 'discrete_normalized_linear_sphere_bridge',
    'time_sampling': 'random_integer_0_to_31_div32',
    'bridge': 'k~UniformInteger(0,31); t=k/32; u=normalize((1-t)*u0+t*u1); q=normalize((1-(k+1)/32)*u0+((k+1)/32)*u1); velocity=32*(q/(u dot q)-u)',
    'trainable': 'original_29907_parameter_DomainVelocity_only',
    'sampler': 'balanced_shuffled_parent_cycles_without_replacement',
    'labels': 'train_split_only; no internal_check teacher reads',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def split_uids(uids):
    uids = list(uids)
    require(len(uids) == 16 and len(set(uids)) == 16, 'Expected exactly 16 unique parent UIDs')
    require(all(re.fullmatch('[A-Za-z0-9_-]+', uid) for uid in uids), 'Unexpected parent UID syntax')
    ranked = sorted(uids, key=lambda uid: hashlib.sha256((SPLIT_SALT + uid).encode()).hexdigest())
    return ranked[:12], ranked[12:]


def select_teacher(qualified, fixed_rms, optimized_rms, threshold=MIN_IMPROVEMENT):
    baseline = np.asarray(fixed_rms, np.float64)
    optimized = np.asarray(optimized_rms, np.float64)
    require(baseline.shape == optimized.shape == (2,), 'Expected two independent measurements')
    require(np.isfinite(baseline).all() and (baseline > 0).all(), 'Invalid Fixed RMS')
    require(np.isfinite(optimized).all() and (optimized >= 0).all(), 'Invalid optimized RMS')
    ratios = optimized / baseline
    moved = bool(qualified and np.all(ratios <= 1.0 - threshold))
    return moved, ratios.tolist()


def named_quality(q):
    return {'min_triangle_angle_deg': float(q['min_angle_deg']),
            'triangle_min_angle_p05_deg': float(q['angle_q05_deg']),
            'degenerate_count': int(q['degenerate_count']),
            'quantile_definition': 'linear 5th percentile over one minimum 3D angle per triangle'}


def prepare_run(source_run, out):
    source = Path(source_run).resolve()
    destination = Path(out)
    if destination.exists():
        raise FileExistsError(destination)
    manifest_path = source / 'MANIFEST_SHA256.json'
    manifest = json.loads(manifest_path.read_text())['files']
    verified = {}

    def checked_path(relative):
        relative = str(relative)
        path = (source / relative).resolve()
        require(path.is_relative_to(source), 'Source path escapes run')
        require(relative in manifest, 'Missing source manifest entry: ' + relative)
        actual = file_hash(path)
        require(actual == manifest[relative], 'Changed source artifact: ' + relative)
        verified[relative] = actual
        return path

    def checked_json(relative):
        return json.loads(checked_path(relative).read_text())

    old_protocol = checked_json('RUN_PROTOCOL.json')
    final_audit = checked_json('FINAL_AUDIT.json')
    require(old_protocol['experiment'] == 'A-LAYOUT-VALUE-v1', 'Wrong source experiment')
    require(final_audit['attempts'] == 96 and final_audit['saved_mesh_hashes_checked'] == 96,
            'Source mesh provenance audit incomplete')
    require(final_audit['same_connectivity_and_accepted_direction_requeries_checked'] == 64,
            'Source accepted-direction audit incomplete')
    require(final_audit['frozen_A_hash_unchanged'] and final_audit['upstream']['pass'],
            'Source baseline identity failed')
    uids = sorted(old_protocol['uids'])
    train_uids, check_uids = split_uids(uids)
    original_a = ASSETS / 'original_A_10.pt'
    require(file_hash(original_a) == old_protocol['a']['weight_sha256'], 'Original A checkpoint changed')
    module_root = Path(__file__).resolve().parents[1]
    for relative in ('exporter.py', 'legacy/networks.py', 'legacy/common.py', 'legacy/deform.py', 'legacy/chart.py', 'assets/support.npz'):
        old_copy = checked_path('source_snapshot/experiments/a_radial/' + relative)
        require(file_hash(module_root / relative) == file_hash(old_copy), 'Input contract changed: ' + relative)
    exporter = MeshExporter(a_seed=10)
    base, faces = exporter.template(512)
    require(base.shape == (258, 3) and faces.shape == (512, 3), 'Incorrect template shapes')
    old_records = old_protocol['records']
    require(len(old_records) == 16 and {r['uid'] for r in old_records} == set(uids), 'Mismatched source records')
    records_by_uid = {r['uid']: r for r in old_records}
    dataset = {}
    teacher_records = []
    for index, uid in enumerate(uids):
        identity = records_by_uid[uid]
        input_path = checked_path('inputs/' + uid + '.npz')
        with np.load(input_path, allow_pickle=False) as z:
            logr, rv, rf = z['logr'], z['vertices'], z['faces']
            require(str(z['uid']) == uid, 'Input UID mismatch')
        require(logr.shape == (2050,) and rv.shape == (2050, 3) and rf.shape == (4096, 3), 'Input shape mismatch')
        require(array_hash(logr) == identity['mainfield_hash'], 'Input mainfield mismatch')
        require(array_hash(rv, rf) == identity['ref_mesh_hash'], 'Input reference mesh mismatch')
        query_rv, query_rf = exporter.support.mesh(logr)
        require(np.array_equal(rv, query_rv) and np.array_equal(rf, query_rf), 'Different radial support reference')
        rows, meshes, checks = {}, {}, {}
        for arm in ('fixed', 'opt_fixed'):
            row = checked_json('attempts/' + uid + '/' + arm + '/result.json')
            require(row['uid'] == uid and row['arm'] == arm and row['status'] == 'success', 'Teacher attempt missing or failed')
            require(row['mainfield_hash'] == identity['mainfield_hash'], 'Teacher belongs to a different mainfield')
            require(row['actual_faces'] == 512 and row['budget_match'], 'Teacher face budget mismatch')
            with np.load(checked_path(row['mesh_file']), allow_pickle=False) as z:
                vertices, connectivity, directions = z['vertices'], z['faces'], z['u']
            require(array_hash(vertices, connectivity) == row['mesh_hash'], 'Teacher mesh hash mismatch')
            require(np.array_equal(connectivity, faces), 'Teacher connectivity changed')
            require(directions.shape == (258, 3) and np.isfinite(directions).all(), 'Invalid teacher direction array')
            require(np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-12), 'Nonunit teacher direction')
            require(np.allclose(vertices, directions * exporter.support.query(logr, directions)[:, None],
                                rtol=1e-12, atol=1e-12), 'Teacher radii are stale')
            quality = mesh_quality(vertices, connectivity)
            require(all(abs(quality[k] - row['quality'][k]) <= 1e-10
                        for k in ('min_angle_deg', 'angle_q05_deg')), 'Quality audit inconsistent')
            require(quality['degenerate_count'] == 0 and row['audit']['valid'], 'Source endpoint validity audit failed')
            second = checked_json('rechecks/' + uid + '/' + arm + '.json')
            require(row['metrics']['points_per_direction'] == second['points_per_direction'] == 8192,
                    'Teacher independent measurement sample count mismatch')
            require(row['metrics']['seed'] == old_protocol['evaluation']['seed_base'] + index and
                    second['seed'] == old_protocol['evaluation']['recheck_seed_base'] + index,
                    'Teacher measurement seed mismatch')
            rows[arm], meshes[arm], checks[arm] = row, (vertices, directions, quality), second
        require(np.array_equal(meshes['fixed'][1], base), 'Fixed is not the original template')
        fixed_q, opt_q = meshes['fixed'][2], meshes['opt_fixed'][2]
        quality_pass = bool(all(opt_q[k] + QUALITY_TOLERANCE_DEG >= fixed_q[k]
                               for k in ('min_angle_deg', 'angle_q05_deg')))
        require(rows['fixed']['qualified'] and rows['fixed']['quality_pass'], 'Fixed source is unqualified')
        require(rows['opt_fixed']['quality_pass'] == quality_pass, 'Source teacher quality flag inconsistent')
        require(rows['opt_fixed']['qualified'] == quality_pass, 'Source teacher qualification flag inconsistent')
        fixed_rms = [rows['fixed']['metrics']['rms_bbox_percent'], checks['fixed']['rms_bbox_percent']]
        optimized_rms = [rows['opt_fixed']['metrics']['rms_bbox_percent'], checks['opt_fixed']['rms_bbox_percent']]
        moved, ratios = select_teacher(rows['opt_fixed']['qualified'], fixed_rms, optimized_rms)
        chosen_arm = 'opt_fixed' if moved else 'fixed'
        teacher_u = meshes[chosen_arm][1].copy()
        proxy = exporter.proxy(logr)
        features = point_features(proxy)
        require(proxy.shape == (256, 3) and features.shape == (256, 4), 'Original input dimensions changed')
        dataset[uid] = {'proxy_points': proxy, 'point_features': features, 'base_u': base.copy(),
                        'teacher_u': teacher_u, 'faces': faces.copy(), 'logr': logr,
                        'ref_vertices': rv, 'ref_faces': rf, 'uid': np.array(uid)}
        teacher_records.append({
            'uid': uid, 'split': 'train' if uid in train_uids else 'internal_check',
            'chosen_arm': chosen_arm, 'moved_label': moved,
            'reason': 'qualified_and_at_least_2pct_improvement_in_both_old_measurements' if moved else
                      'Fixed_zero_displacement_due_to_insufficient_repeat_improvement_or_quality',
            'fixed_rms_bbox_percent_two_seeds': fixed_rms, 'optimized_rms_bbox_percent_two_seeds': optimized_rms,
            'optimized_to_fixed_rms_ratios_two_seeds': ratios,
            'fixed_quality': named_quality(fixed_q), 'opt_fixed_quality': named_quality(opt_q),
            'opt_fixed_qualified': rows['opt_fixed']['qualified'],
            'mainfield_hash': identity['mainfield_hash'], 'ref_mesh_hash': identity['ref_mesh_hash'],
            'fixed_mesh_hash': rows['fixed']['mesh_hash'], 'opt_fixed_mesh_hash': rows['opt_fixed']['mesh_hash'],
            'teacher_mesh_hash': rows[chosen_arm]['mesh_hash'], 'teacher_direction_hash': array_hash(teacher_u),
            'proxy_hash': array_hash(proxy), 'point_features_hash': array_hash(features),
            'exact_audit': {'source_result_file': 'attempts/' + uid + '/' + chosen_arm + '/result.json',
                            'valid': rows[chosen_arm]['audit']['valid'], 'recomputed': False,
                            'provenance': 'reused immutable manifest-verified source endpoint audit'},
        })
    # Validation is complete before creating the destination. Nothing is overwritten.
    out = new_dir(destination)
    data_files = {}
    for uid in uids:
        path = out / 'data' / (uid + '.npz')
        save_npz(path, **dataset[uid])
        data_files['data/' + uid + '.npz'] = file_hash(path)
    labels = out / 'train_labels.npz'
    save_npz(labels, train_uids=np.asarray(train_uids),
             directions0=np.stack([dataset[u]['base_u'] for u in train_uids]),
             directions1=np.stack([dataset[u]['teacher_u'] for u in train_uids]),
             proxy_points=np.stack([dataset[u]['proxy_points'] for u in train_uids]),
             point_features=np.stack([dataset[u]['point_features'] for u in train_uids]))
    data_files['train_labels.npz'] = file_hash(labels)
    write_json(out / 'DATA_MANIFEST.json', {'files': data_files})
    write_json(out / 'TEACHERS.json', {'rows': teacher_records,
               'rule': 'Only Fixed-to-optimization endpoints; otherwise Fixed zero displacement.',
               'no_optimal_teacher_claim': True, 'check_labels_allowed_for_training': False})
    write_json(out / 'SOURCE_PROVENANCE.json', {'source_run': str(source),
               'source_manifest_sha256': file_hash(manifest_path), 'verified_files': verified,
               'source_final_audit': final_audit, 'old_outputs_written': False,
               'checks': {'unique_uids': 16, 'saved_mesh_hashes': 32, 'radial_requeries': 32,
                          'same_connectivity': 32, 'reference_meshes': 16, 'quality_recalculations': 32,
                          'exact_validity_source_audits_reused': 32}})
    source_hashes = {str(p.relative_to(module_root)): file_hash(p) for p in [
        Path(__file__), module_root / 'exporter.py', module_root / 'legacy/networks.py',
        module_root / 'legacy/common.py', module_root / 'legacy/deform.py',
        module_root / 'layout_value/geometry.py', module_root / 'assets/support.npz']}
    protocol = {
        'experiment': 'A-TASKALIGNED-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization': 'User approved improvement following feedback; train original A direction network only.',
        'scope': '16 previously seen development objects; 12 train and 4 internal check, not an unseen confirmation set.',
        'source_run': str(source), 'uids': uids, 'train_uids': train_uids, 'internal_check_uids': check_uids,
        'split': {'rule': "ascending SHA256('a-taskalign-split-v1:' + uid), first12 train, remaining4 internal_check",
                  'parent_level': True, 'all_objects_already_seen_in_development': True,
                  'unseen_confirmation_set': False},
        'paths': {'training_labels': 'train_labels.npz', 'data_dir': 'data',
                  'original_a_checkpoint': str(original_a.resolve()), 'teacher_manifest': 'TEACHERS.json'},
        'training': TRAINING,
        'teacher': {'source_arm': 'opt_fixed', 'minimum_rms_improvement_fraction_both_old_seeds': MIN_IMPROVEMENT,
                    'fallback_label': 'Fixed_zero_displacement', 'all16_retained': True,
                    'threshold_is_exploratory_not_significance': True,
                    'training_reads_only_train_labels_npz': True, 'teacher_uses_full_field_student_only_proxy256': True},
        'input': {'proxy_points': 256, 'proxy_seed': 4501, 'features': 'unchanged legacy.networks.point_features',
                  'mainfield_nodes': 2050, 'reference_faces': 4096, 'target_faces': 512,
                  'extra_student_information': False},
        'deployment': {'original_guard': True, 'updates': 32, 'device': 'cpu', 'final_accepted_direction_requery': True,
                       'policies': ['raw', 'quality_fallback'],
                       'fallback_rule': 'use Fixed only if invalid geometry, non512 budget, or either angle quality below Fixed',
                       'fallback_uses_surface_error': False, 'fallback_rate_reported_separately': True,
                       'added_quality_audit_cost_is_part_of_method': True},
        'evaluation': {'baseline_arms': ['fixed', 'original_a_seed10'], 'student_seeds': [10, 11],
                       'points_each_direction': 8192, 'seed_base': 192401, 'second_seed_base': 292401,
                       'seed_rule': 'base + index in sorted UID list',
                       'fresh_seeds_not_used_for_teacher_or_checkpoint_selection': True,
                       'all_attempts_retained': True, 'paired_improvement_not_ratio_of_medians': True},
        'quality': {'min_triangle_angle_deg': 'minimum of all actual 3D triangle angles',
                    'triangle_min_angle_p05_deg': 'linear 5th percentile over one minimum 3D angle per triangle',
                    'floors': 'per-object Fixed values', 'tolerance_deg': QUALITY_TOLERANCE_DEG,
                    'requires_exact512_and_source_equivalent_endpoint_validity': True},
        'frozen': ['mainfields', 'ShapeAE', 'latent_FM', 'data_conversion', 'official_MeshFlow', 'original_A_checkpoint'],
        'no_optimizer_search': True, 'fixed_source_hashes': source_hashes,
        'hashes': {'config_sha256': config_hash(TRAINING), 'training_labels_sha256': file_hash(labels),
                   'data_manifest_sha256': file_hash(out / 'DATA_MANIFEST.json'),
                   'original_a_sha256': file_hash(original_a), 'source_run_manifest_sha256': file_hash(manifest_path)},
    }
    write_json(out / 'RUN_PROTOCOL.json', protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    protocol = prepare_run(args.source_run, args.run)
    print(json.dumps({'run': args.run, 'train': len(protocol['train_uids']),
                      'internal_check': len(protocol['internal_check_uids']), 'protocol_locked': True}))


if __name__ == '__main__':
    main()


