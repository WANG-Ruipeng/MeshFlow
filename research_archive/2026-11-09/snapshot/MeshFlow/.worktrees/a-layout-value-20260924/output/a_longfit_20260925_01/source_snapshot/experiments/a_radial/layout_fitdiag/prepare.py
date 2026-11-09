"""Create a new fit diagnostic from the byte-verified sealed endpoint run.

No training, teacher optimization, or old output mutation occurs here.
"""
from __future__ import annotations
import argparse
import datetime
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import torch
from ..exporter import ASSETS
from ..io_utils import file_hash, write_json
from ..legacy.networks import point_features
from ..layout_student.prepare import config_hash, split_uids
from ..layout_student.train import model_sha256
from ..verify_repo import verify

PRIOR_UID = '2379915c82c2b2d02530f02db7e9157e'
CHECKPOINT_UPDATES = [1000, 1667, 5000, 10000, 20000]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Path escapes source: ' + str(relative))
    return path


def teacher_energy_uids(labels, train_uids):
    """Select only from the frozen training archive, never check labels."""
    energy = np.mean((labels['directions1'] - labels['directions0']) ** 2, axis=(1, 2))
    require(energy.shape == (len(train_uids),) and np.isfinite(energy).all(), 'Invalid teacher energy')
    ranked = sorted(zip(train_uids, energy.tolist()), key=lambda pair: (-pair[1], pair[0]))
    require(ranked[0][1] > 0, 'No moving training teacher')
    return ranked[0][0], dict(zip(train_uids, energy.tolist()))


def validate_checkpoint(state, seed, joint, source_protocol_sha256, label_sha256):
    require(model_sha256(state['model']) == state['model_sha256'], 'Source model hash mismatch')
    require(sum(value.numel() for value in state['model'].values()) == 29907, 'Source architecture changed')
    if not joint:
        require(state['update'] == 3000 and state['is_predeclared_final'], 'Not the frozen v1 final')
        return
    require(state['update'] == 1000 and state['is_predeclared_final'], 'Not the frozen endpoint C0 final')
    require(state['config']['arm'] == 'c0' and state['config']['seed'] == seed, 'Wrong C0 arm/seed')
    require(state['protocol_sha256'] == source_protocol_sha256, 'C0 source protocol changed')
    require(state['training_labels_sha256'] == label_sha256, 'C0 source training archive changed')
    require(all(k in state for k in ('optimizer', 'scheduler', 'rng', 'sampler', 'trace', 'timing_trace')), 'Incomplete C0 state')
    require(bool(state['optimizer']['state']), 'Missing inherited Adam state')
    require(len(state['trace']) == 1000 and state['trace'][-1]['update'] == 1000, 'C0 trace is incomplete')
    require(state['sampler']['count'] == 12 and sum(state['sampler']['exposures']) == 4000, 'C0 sampler mismatch')
    require(bool(state['rng']['torch_cuda']), 'C0 CUDA RNG is missing')
    for group in state['optimizer']['param_groups']:
        require(group['lr'] == 1e-4 and group['weight_decay'] == 1e-4, 'C0 optimizer hyperparameters changed')


def prepare(source_run, run):
    source = Path(source_run).resolve()
    run = Path(run).resolve()
    require(not run.exists(), 'New run directory must not exist')
    require(not run.is_relative_to(source), 'New run cannot be inside the sealed source')
    seal_report_path = source.with_name(source.name + '_results.zip.json')
    seal = read_json(seal_report_path)
    require(seal.get('crc_verified') and seal.get('fresh_extracted_manifest_verified') and
            seal.get('fresh_original_cpu_replay') == 'PASS' and
            seal.get('sealed_run_and_fresh_artifacts_unchanged'), 'Endpoint sealing is not complete')
    archive = source.with_name(source.name + '_results.zip')
    require(archive.is_file() and file_hash(archive) == seal['sha256'], 'Sealed archive hash mismatch')
    source_manifest_path = source / 'MANIFEST_SHA256.json'
    source_manifest = read_json(source_manifest_path)['files']
    for rel, expected in source_manifest.items():
        path = inside(source, rel)
        require(path.is_file() and file_hash(path) == expected, 'Source manifest mismatch: ' + rel)
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    require(actual == set(source_manifest) | {'MANIFEST_SHA256.json'}, 'Sealed source file set changed')
    old = read_json(source / 'RUN_PROTOCOL.json')
    require(old['experiment'] == 'A-ENDPOINT-v1', 'Wrong source experiment')
    old_data = read_json(source / 'DATA_MANIFEST.json')['files']
    require(len(old_data) == 20, 'Expected the original 20 data and initialization files')
    require(file_hash(source / 'DATA_MANIFEST.json') == old['hashes']['data_manifest_sha256'], 'Source data manifest mismatch')
    for rel, digest in old_data.items():
        require(source_manifest.get(rel) == digest, 'Source data not bound by sealed manifest: ' + rel)
    uids = sorted(old['uids'])
    train, check = split_uids(uids)
    require(train == old['train_uids'] and check == old['internal_check_uids'], 'Parent split changed')
    with np.load(source / 'train_labels.npz', allow_pickle=False) as z:
        labels = {key: np.array(z[key]) for key in z.files}
    require(labels['train_uids'].tolist() == train, 'Training UID order changed')
    require(np.array_equal(point_features(labels['proxy_points']), labels['point_features']), 'Proxy features changed')
    scale = float(np.mean((labels['directions1'] - labels['directions0']) ** 2))
    require(scale == old['endpoint_scale_squared'] and scale > 0, 'Global training scale changed')
    for uid in uids:
        with np.load(source / 'data' / (uid + '.npz'), allow_pickle=False) as z:
            require(z['base_u'].shape == z['teacher_u'].shape == (258, 3), 'Wrong direction shape')
            require(z['faces'].shape == (512, 3) and z['faces'].dtype == np.int32, 'Original connectivity changed')
            require(z['logr'].shape == (2050,), 'Mainfield changed')
            if uid in train:
                i = train.index(uid)
                for archive_key, parent_key in [('directions0', 'base_u'), ('directions1', 'teacher_u'),
                        ('proxy_points', 'proxy_points'), ('point_features', 'point_features'), ('logr', 'logr')]:
                    require(np.array_equal(labels[archive_key][i], z[parent_key]), 'Training/per-parent data disagree')
                require(np.array_equal(labels['faces'], z['faces']), 'Training connectivity differs')
    energy_uid, energies = teacher_energy_uids(labels, train)
    require(PRIOR_UID in train and energies[PRIOR_UID] > 0, 'Prior diagnostic UID is not a moving training object')
    root = Path(__file__).resolve().parents[1]
    identity = verify(root.parents[1])
    require(identity['pass'], 'Protected official source mismatch')
    fixed_hashes = dict(old['fixed_source_hashes'])
    for rel, digest in fixed_hashes.items():
        require(file_hash(root / rel) == digest, 'Frozen geometry source changed: ' + rel)
    protocol_sha = file_hash(source / 'RUN_PROTOCOL.json')
    copy_map = {rel: rel for rel in old_data}
    copy_map['initial/endpoint_RUN_PROTOCOL.json'] = 'RUN_PROTOCOL.json'
    states = {}
    for seed in (10, 11):
        for joint in (False, True):
            src_rel = f'training/c0/seed_{seed}/final.pt' if joint else f'initial/seed_{seed}.pt'
            target_rel = f'initial/c0_seed_{seed}_final.pt' if joint else src_rel
            state = torch.load(source / src_rel, map_location='cpu', weights_only=False)
            validate_checkpoint(state, seed, joint, protocol_sha, old_data['train_labels.npz'])
            if joint:
                require(state['config']['train_uids'] == state['config']['active_training_uids'] == train, 'C0 active training set changed')
                require(state['initial_checkpoint_sha256'] == old_data[f'initial/seed_{seed}.pt'], 'C0 initialization changed')
                for rel, digest in state['config']['source_sha256'].items():
                    require(file_hash(root / rel) == digest, 'Frozen C0 dependency changed: ' + rel)
                    fixed_hashes[rel] = digest
            copy_map[target_rel] = src_rel
            states[target_rel] = state
    # Everything needed to validate the source is checked before creating output.
    training = {'updates': 20000, 'batch_size': 4, 'optimizer': 'AdamW', 'learning_rate': 1e-4,
        'weight_decay': 1e-4, 'schedule': 'constant_learning_rate', 'checkpoint_updates': CHECKPOINT_UPDATES,
        'checkpoint_selection': 'global_update_20000_only_no_metric_selection', 'initial_snapshot': True,
        'precision': 'network_float32_geometry_float64', 'tf32': False, 'deterministic_algorithms': True,
        'trainable': 'unchanged_29907_parameter_DomainVelocity_only', 'rollout_steps': 32,
        'objective': 'discrete_normalized_linear_sphere_bridge', 'time_sampling': 'random_integer_0_to_31_div32',
        'sampler': 'balanced_shuffled_active_parent_cycles_without_replacement',
        'labels': 'train_archive_only_no_internal_check_teacher_reads',
        'equal_updates_not_equal_compute_or_parent_exposure': True}
    loss = {'c0': old['loss']['c0'], 'endpoint_scale_squared': scale,
        'endpoint_scale_definition': old['loss']['endpoint_scale_definition'],
        'endpoint_scale_used_only_for_diagnostic_normalization': True,
        'velocity_objective_is_original_unnormalized_MSE': True,
        'quality_scale_deg': .1, 'quality_penalty_in_training': False,
        'zero_motion_labels_retained': True, 'scale_uses_only_training_labels': True}
    copied_hashes = {dst: file_hash(source / src) for dst, src in copy_map.items()}
    jobs = {}
    for name, seed, active, kind in [('joint_10', 10, train, 'joint'), ('joint_11', 11, train, 'joint'),
            ('single_energy_10', 10, [energy_uid], 'single'), ('single_prior_10', 10, [PRIOR_UID], 'single')]:
        initial = f'initial/c0_seed_{seed}_final.pt' if kind == 'joint' else f'initial/seed_{seed}.pt'
        jobs[name] = {'kind': kind, 'seed': seed, 'active_training_uids': list(active), 'diagnostic_uids': list(active),
            'initial_checkpoint': initial, 'initial_checkpoint_sha256': copied_hashes[initial],
            'initial_model_sha256': states[initial]['model_sha256'],
            'v1_initial_checkpoint': f'initial/seed_{seed}.pt',
            'initialization_mode': 'resume_endpoint_c0_full_state' if kind == 'joint' else 'v1_model_fresh_optimizer',
            'start_update': 1000 if kind == 'joint' else 0, 'total_updates': 20000,
            'source_protocol_sha256': protocol_sha,
            'optimizer_scheduler_rng_sampler_inherited': kind == 'joint',
            'source_update_1000_is_diagnostic_snapshot': kind == 'joint',
            'source_trace_inherited': kind == 'joint',
            'global_parent_exposures_at_final': 80000,
            'per_active_parent_exposures_at_final': [6666, 6667] if kind == 'joint' else [80000],
            'new_optimizer_updates': 19000 if kind == 'joint' else 20000}
    evaluation_models = ['fixed', 'teacher_reference', 'parent_c0_10', 'parent_c0_11',
                         'joint_10', 'joint_11', 'single_energy_10', 'single_prior_10']
    evaluation = {'models': evaluation_models,
        'model_uids': {name: (list(uids) if not name.startswith('single_') else jobs[name]['active_training_uids'])
                       for name in evaluation_models},
        'seed_base': 592501, 'second_seed_base': 692501, 'points_each_direction': 8192, 'hot_repeats': 3,
        'seed_rule': 'base + index in original sorted 16 UID list', 'planned_attempts': 98, 'planned_rows': 164,
        'model_checkpoints': {'parent_c0_10': 'initial/c0_seed_10_final.pt',
                              'parent_c0_11': 'initial/c0_seed_11_final.pt',
                              **{name: f'training/{name}/final.pt' for name in jobs}},
        'model_seeds': {'parent_c0_10': 10, 'parent_c0_11': 11,
                        **{name: job['seed'] for name, job in jobs.items()}},
        'parent_checkpoint_sha256': {f'parent_c0_{seed}': copied_hashes[f'initial/c0_seed_{seed}_final.pt']
                                    for seed in (10, 11)},
        'final_cpu_original_deployment_only': True, 'raw_and_quality_fallback_separate': True,
        'all_attempts_retained': True, 'single_object_models_are_capacity_diagnostics_only': True,
        'intermediate_checkpoints_train_objects_only': True, 'intermediate_checkpoints_select_no_model': True,
        'formal_final_update': 20000}
    paths = {'training_labels': 'train_labels.npz', 'data_manifest': 'DATA_MANIFEST.json',
        'data_dir': 'data', 'teacher_manifest': 'TEACHERS.json', 'source_protocol': 'initial/endpoint_RUN_PROTOCOL.json',
        'initial_checkpoints': {str(seed): f'initial/seed_{seed}.pt' for seed in (10, 11)},
        'joint_source_checkpoints': {str(seed): f'initial/c0_seed_{seed}_final.pt' for seed in (10, 11)}}
    protocol = {'experiment': 'A-FIT-DIAG-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization': 'User authorized approximately eight hours of autonomous diagnosis and independent new experiments; continue diagnosis when the first experiment fails.',
        'source_run': str(source), 'source_run_manifest_sha256': file_hash(source_manifest_path),
        'source_archive_sha256': seal['sha256'], 'source_seal_report_sha256': file_hash(seal_report_path),
        'uids': uids, 'train_uids': train, 'internal_check_uids': check, 'split': old['split'],
        'paths': paths, 'jobs': jobs, 'training': training, 'loss': loss,
        'endpoint_scale_squared': scale, 'evaluation': evaluation,
        'selection': {'single_energy_uid': energy_uid, 'single_prior_uid': PRIOR_UID,
            'energy_uid_rule': 'argmax mean squared coordinate teacher displacement over training archive only; UID ascending tie break',
            'prior_uid_rule': 'existing predeclared endpoint single-object diagnostic UID, retained before this experiment',
            'training_teacher_displacement_energy': energies, 'uses_internal_check': False},
        'geometry_diagnostics': {'lambdas': [0., .125, .25, .375, .5, .625, .75, .875, 1.],
            'endpoint_directions_preserved_exactly': True, 'chart': 'positive oriented direction determinants',
            'surface_rms_measured': False, 'full_path_or_intersection_certificate': False,
            'offline_all16_development_geometry_not_model_selection': True},
        'quality': old['quality'], 'deployment': old['deployment'], 'input': old['input'],
        'frozen': old['frozen'] + ['endpoint_sealed_results', 'original_20_data_manifest_bytes'],
        'fixed_source_hashes': fixed_hashes,
        'research_scope': {'independent_development_exploration': True, 'heldout_generalization': False,
            'architecture_data_mainfield_teacher_unchanged': True, 'zero_teachers_kept': True,
            'original_endpoint_gate_not_passed_and_not_revised': True,
            'no_claim_of_equal_single_vs_joint_parent_exposure': True,
            'single_object_success_is_not_joint_fitting_or_generalization': True,
            'full_MeshFlow_comparison': 'independent pending track'},
        'hashes': {'training_labels_sha256': copied_hashes['train_labels.npz'],
            'source_protocol_sha256': protocol_sha, 'source_run_manifest_sha256': file_hash(source_manifest_path),
            'initial_checkpoint_sha256': {str(seed): copied_hashes[f'initial/seed_{seed}.pt'] for seed in (10, 11)},
            'initial_model_sha256': {str(seed): states[f'initial/seed_{seed}.pt']['model_sha256'] for seed in (10, 11)},
            'joint_source_checkpoint_sha256': {str(seed): copied_hashes[f'initial/c0_seed_{seed}_final.pt'] for seed in (10, 11)},
            'joint_source_model_sha256': {str(seed): states[f'initial/c0_seed_{seed}_final.pt']['model_sha256'] for seed in (10, 11)},
            'config_sha256': config_hash({'training': training, 'loss': loss, 'jobs': jobs})}}
    run.mkdir(parents=True, exist_ok=False)
    for target_rel, source_rel in copy_map.items():
        target = run / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / source_rel, target)
        require(file_hash(target) == copied_hashes[target_rel], 'Copied bytes changed: ' + target_rel)
    write_json(run / 'DATA_MANIFEST.json', {'files': copied_hashes})
    protocol['hashes']['data_manifest_sha256'] = file_hash(run / 'DATA_MANIFEST.json')
    write_json(run / 'RUN_PROTOCOL.json', protocol)
    write_json(run / 'SOURCE_PROVENANCE.json', {
        'source_run': str(source), 'source_manifest_sha256': file_hash(source_manifest_path),
        'all_source_manifest_files_verified': len(source_manifest), 'verified_source_manifest': source_manifest,
        'source_seal_report_sha256': file_hash(seal_report_path), 'source_archive_sha256': seal['sha256'],
        'source_seal_report': seal, 'source_protocol_sha256': protocol_sha,
        'original_20_files_byte_identical': True, 'source_and_destination_relative_paths': copy_map,
        'new_data_manifest_file_count': len(copied_hashes), 'old_outputs_written': False,
        'preparation_reads_all16_for_integrity_only': True, 'training_archive_contains_only_train12': True,
        'joint_sources_contain_full_optimizer_scheduler_sampler_and_rng': True,
        'prepare_source_sha256': file_hash(__file__), 'geometry_diagnostics_source_sha256': file_hash(Path(__file__).with_name('geometry_diagnostics.py'))})
    write_json(run / 'REPO_IDENTITY.json', identity)
    write_json(run / 'ENVIRONMENT.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'torch_cuda_runtime': torch.version.cuda, 'training_device': 'cuda', 'deployment_device': 'cpu',
        'free_output_disk_bytes': shutil.disk_usage(run).free, 'prepare_used_cuda_compute': False})
    # Revalidate source identity after creation without adding anything to the old run.
    require(file_hash(source_manifest_path) == protocol['source_run_manifest_sha256'], 'Source manifest changed during prepare')
    for rel, digest in copy_map.items():
        require(file_hash(source / digest) == copied_hashes[rel], 'Source artifact changed during prepare')
    result = {'run': str(run), 'protocol_sha256': file_hash(run / 'RUN_PROTOCOL.json'),
        'copied_files': len(copied_hashes), 'verified_source_files': len(source_manifest),
        'single_energy_uid': energy_uid, 'single_prior_uid': PRIOR_UID, 'jobs': list(jobs)}
    print(json.dumps(result, indent=2), flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    prepare(args.source_run, args.run)


if __name__ == '__main__':
    main()
