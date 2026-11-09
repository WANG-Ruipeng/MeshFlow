"""Freeze a separate long joint fit only after the parent fitdiag is sealed."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import datetime
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import torch
from ..io_utils import file_hash, write_json
from ..legacy.networks import point_features
from ..layout_student.package_results import inside, verify_manifest
from ..layout_student.prepare import config_hash, split_uids
from ..layout_student.train import model_sha256
from ..layout_fitdiag.prepare import read_json, require
from ..verify_repo import verify

SOURCE_JOBS = ('joint_10', 'joint_11', 'single_energy_10', 'single_prior_10')
CHECKPOINT_UPDATES = [20000, 50000, 100000, 240000]


def validate_source_final(state, name, source_protocol, source_protocol_sha256):
    job = source_protocol['jobs'][name]; config = state['config']
    require(state['update'] == 20000 and state['is_predeclared_final'] and state['diagnostic_only'], 'Not a frozen fitdiag20000 final')
    require(state['job_id'] == config['job_id'] == config['job'] == name and config['seed'] == job['seed'], 'Source final job/seed changed')
    require(state['protocol_sha256'] == config['protocol_sha256'] == source_protocol_sha256, 'Source final protocol changed')
    require(state['training_labels_sha256'] == source_protocol['hashes']['training_labels_sha256'], 'Source final data changed')
    require(model_sha256(state['model']) == state['model_sha256'] and sum(x.numel() for x in state['model'].values()) == 29907, 'Source model identity changed')
    require(config['arm'] == 'c0' and config['batch_size'] == 4 and config['learning_rate'] == 1e-4
            and config['weight_decay'] == 1e-4 and config['rollout_steps'] == 32, 'Source C0 training contract changed')
    require(config['active_training_uids'] == job['active_training_uids'] and config['train_uids'] == source_protocol['train_uids'], 'Source training split changed')
    require(all(key in state for key in ('optimizer', 'scheduler', 'rng', 'sampler', 'trace', 'timing_trace')), 'Incomplete resumable source')
    require(len(state['trace']) == 20000 and [r['update'] for r in state['trace']] == list(range(1, 20001)), 'Source trace incomplete')
    exposures = Counter(uid for row in state['trace'] for uid in row['uids'])
    require(sum(exposures.values()) == 80000 and set(exposures) == set(job['active_training_uids']), 'Source exposure count differs')
    require(state['sampler']['exposures'] == [exposures[u] for u in job['active_training_uids']], 'Source sampler/trace differ')
    require(set(exposures.values()) <= ({6666, 6667} if job['kind'] == 'joint' else {80000}), 'Source exposure balance differs')
    require(state['optimizer']['state'] and all(float(v['step']) == 20000 for v in state['optimizer']['state'].values()), 'Source Adam update differs')
    require(state['scheduler']['last_epoch'] == 20000 and len(state['rng']['torch_cuda']) > 0, 'Source scheduler/CUDA RNG missing')
    for group in state['optimizer']['param_groups']:
        require(group['lr'] == 1e-4 and group['weight_decay'] == 1e-4 and group['foreach'] is False and group['fused'] is False, 'Source Adam settings changed')
    return dict(exposures)


def prepare(source_run, run):
    source = Path(source_run).resolve(); run = Path(run).resolve()
    require(not run.exists() and not run.is_relative_to(source), 'Use an exclusive new run outside the sealed source')
    source_manifest_path = source / 'MANIFEST_SHA256.json'
    seal_path = source.with_name(source.name + '_results.zip.json')
    seal = read_json(seal_path)
    require(seal['crc_verified'] and seal['fresh_extracted_manifest_verified'] and seal['fresh_original_cpu_replay'] == 'PASS'
            and seal['sealed_run_and_fresh_artifacts_unchanged'], 'Parent fitdiag sealing has not passed')
    archive = source.with_name(source.name + '_results.zip')
    require(file_hash(archive) == seal['sha256'], 'Parent archive changed')
    source_manifest = read_json(source_manifest_path)['files']
    verify_manifest(source, source_manifest)
    require({x.relative_to(source).as_posix() for x in source.rglob('*') if x.is_file()} == set(source_manifest) | {'MANIFEST_SHA256.json'}, 'Parent file set changed')
    old = read_json(source / 'RUN_PROTOCOL.json'); old_ph = file_hash(source / 'RUN_PROTOCOL.json')
    require(old['experiment'] == 'A-FIT-DIAG-v1', 'Wrong parent experiment')
    train, check = split_uids(old['uids'])
    require(train == old['train_uids'] and check == old['internal_check_uids'], 'Split changed')
    source_data = read_json(source / 'DATA_MANIFEST.json')['files']
    require(len(source_data) == 23 and file_hash(source / 'DATA_MANIFEST.json') == old['hashes']['data_manifest_sha256'], 'Parent data manifest changed')
    for relative, digest in source_data.items():
        require(source_manifest.get(relative) == digest, 'Data missing from sealed source manifest')
    with np.load(source / old['paths']['training_labels'], allow_pickle=False) as z:
        labels = {key: np.array(z[key]) for key in z.files}
    require(labels['train_uids'].tolist() == train and np.array_equal(point_features(labels['proxy_points']), labels['point_features']), 'Frozen proxy/train archive changed')
    scale = float(np.mean((labels['directions1'] - labels['directions0']) ** 2))
    require(scale == old['endpoint_scale_squared'] and scale > 0, 'Global train-only scale changed')
    for uid in old['uids']:
        with np.load(source / 'data' / (uid + '.npz'), allow_pickle=False) as z:
            require(z['base_u'].shape == z['teacher_u'].shape == (258, 3)
                    and z['faces'].shape == (512, 3) and z['faces'].dtype == np.int32, 'Data topology changed')
            if uid in train:
                i = train.index(uid)
                for source_key, label_key in [('base_u', 'directions0'), ('teacher_u', 'directions1'), ('logr', 'logr'),
                                              ('proxy_points', 'proxy_points'), ('point_features', 'point_features')]:
                    require(np.array_equal(z[source_key], labels[label_key][i]), 'Parent data/train archive disagree')
    root = Path(__file__).resolve().parents[1]
    identity = verify(root.parents[1])
    require(identity['pass'] and identity['protected_files_checked'] == 11, 'Protected official source changed')
    fixed_hashes = dict(old['fixed_source_hashes'])
    states, source_exposures = {}, {}
    copy_map = {relative: relative for relative in source_data}
    copy_map['initial/fitdiag_RUN_PROTOCOL.json'] = 'RUN_PROTOCOL.json'
    for name in SOURCE_JOBS:
        relative = f'training/{name}/final.pt'
        state = torch.load(source / relative, map_location='cpu', weights_only=False)
        source_exposures[name] = validate_source_final(state, name, old, old_ph)
        require(file_hash(source / relative) == source_manifest[relative], 'Frozen source checkpoint bytes differ')
        for dependency, digest in state['config']['source_sha256'].items():
            require(file_hash(inside(root, dependency)) == digest, 'Frozen training dependency changed: ' + dependency)
            fixed_hashes[dependency] = digest
        target = f'initial/fitdiag_{name}_final.pt'
        copy_map[target] = relative; states[name] = state
    for relative, digest in fixed_hashes.items():
        require(file_hash(inside(root, relative)) == digest, 'Frozen deployment asset/source changed: ' + relative)
    mechanism = read_json(source / 'TRAIN_DIAGNOSTICS.json')
    require(mechanism['protocol_sha256'] == old_ph and mechanism['expected_attempt_count'] == 156
            and mechanism['checkpoint_selection'] is False and mechanism['internal_check_objects_opened'] == 0,
            'Source mechanism scope changed')
    keys = [(r['job'], r['checkpoint'], r['uid']) for r in mechanism['rows']]
    require(len(keys) == len(set(keys)) and all(uid in train for _, _, uid in keys), 'Source diagnostic identity changed')
    require(len(mechanism['rows']) + len(mechanism['missing_success_slots']) == 156, 'Source diagnostic denominator changed')
    evidence_sources = {'TRAIN_DIAGNOSTICS.json', 'SUMMARY.json', 'GEOMETRY_DIAGNOSTICS.json', 'TRAINING_GATE.json',
                        'FINAL_AUDIT.json', 'SOURCE_PROVENANCE.json'}
    evidence_sources.update(r for r in source_manifest if r.startswith('diagnostics/mechanism/') or r.startswith('diagnostics/invocations/'))
    require(all(r in source_manifest for r in evidence_sources), 'Required source evidence missing from sealed manifest')
    copied_hashes = {dst: file_hash(source / src) for dst, src in copy_map.items()}
    evidence_map = {'prior_fitdiag_evidence/' + rel: rel for rel in sorted(evidence_sources)}
    evidence_hashes = {dst: source_manifest[src] for dst, src in evidence_map.items()}
    training = copy.deepcopy(old['training'])
    training.update({'updates': 240000, 'checkpoint_updates': CHECKPOINT_UPDATES,
        'checkpoint_selection': 'global_update_240000_only_no_metric_selection',
        'equal_updates_not_equal_compute_or_parent_exposure': False,
        'parent_exposure_matched_to_frozen_single_final': True,
        'equal_adam_updates_compute_or_seed_claim': False})
    loss = copy.deepcopy(old['loss'])
    jobs = {}
    for seed in (10, 11):
        name = f'joint_{seed}'; checkpoint = f'initial/fitdiag_{name}_final.pt'
        jobs[name] = {'kind': 'joint', 'seed': seed, 'active_training_uids': list(train), 'diagnostic_uids': list(train),
            'initialization_mode': 'resume_fitdiag_joint_full_state', 'start_update': 20000, 'total_updates': 240000,
            'initial_checkpoint': checkpoint, 'initial_checkpoint_sha256': copied_hashes[checkpoint],
            'initial_model_sha256': states[name]['model_sha256'], 'source_protocol_sha256': old_ph,
            'v1_initial_checkpoint': old['paths']['initial_checkpoints'][str(seed)],
            'optimizer_scheduler_rng_sampler_inherited': True, 'source_trace_inherited': True,
            'source_initial_update_is_diagnostic_snapshot': True,
            'global_parent_exposures_at_final': 960000, 'per_active_parent_exposures_at_final': [80000],
            'new_optimizer_updates': 220000, 'new_total_parent_exposures': 880000,
            'new_per_parent_exposure_range': [73333, 73334],
            'source_parent_exposures': source_exposures[name]}
    source_single_jobs = {name: copy.deepcopy(old['jobs'][name]) for name in ('single_energy_10', 'single_prior_10')}
    source_models = {'parent_joint_10': 'joint_10', 'parent_joint_11': 'joint_11',
                     'single_energy_10': 'single_energy_10', 'single_prior_10': 'single_prior_10'}
    model_names = ['fixed', 'teacher_reference', 'parent_joint_10', 'parent_joint_11',
                   'joint_10', 'joint_11', 'single_energy_10', 'single_prior_10']
    model_checkpoints = {name: f'initial/fitdiag_{job}_final.pt' for name, job in source_models.items()}
    model_checkpoints.update({name: f'training/{name}/final.pt' for name in jobs})
    uids = sorted(old['uids'])
    evaluation = {'models': model_names,
        'model_uids': {name: list(source_single_jobs[name]['active_training_uids']) if name.startswith('single_') else list(uids) for name in model_names},
        'single_reference_uids': {name: list(job['active_training_uids']) for name, job in source_single_jobs.items()},
        'model_checkpoints': model_checkpoints,
        'model_seeds': {**{name: old['jobs'][job]['seed'] for name, job in source_models.items()}, 'joint_10': 10, 'joint_11': 11},
        'source_checkpoint_sha256': {name: copied_hashes[model_checkpoints[name]] for name in source_models},
        'source_model_sha256': {name: states[job]['model_sha256'] for name, job in source_models.items()},
        'source_model_jobs': source_models, 'source_model_updates': 20000, 'formal_final_update': 240000,
        'seed_base': 792501, 'second_seed_base': 892501, 'points_each_direction': 8192, 'hot_repeats': 3,
        'seed_rule': 'base + index in original sorted 16 UID list', 'planned_attempts': 98, 'planned_rows': 164,
        'final_cpu_original_deployment_only': True, 'raw_and_quality_fallback_separate': True,
        'all_attempts_retained': True, 'single_object_models_are_frozen_capacity_references_only': True,
        'intermediate_checkpoints_train_objects_only': True, 'intermediate_checkpoints_select_no_model': True}
    paths = copy.deepcopy(old['paths'])
    paths.update({'source_protocol': 'initial/fitdiag_RUN_PROTOCOL.json',
        'source_evidence_manifest': 'SOURCE_EVIDENCE_MANIFEST.json',
        'source_mechanism': 'prior_fitdiag_evidence/TRAIN_DIAGNOSTICS.json',
        'source_model_checkpoints': {name: model_checkpoints[name] for name in source_models},
        'joint_source_checkpoints': {str(seed): f'initial/fitdiag_joint_{seed}_final.pt' for seed in (10, 11)}})
    protocol = {'experiment': 'A-LONGFIT-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization': 'User authorized approximately eight hours of autonomous diagnosis; the predeclared long-fit extension tests shared fitting after the shorter budget proved insufficient.',
        'source_run': str(source), 'source_run_manifest_sha256': file_hash(source_manifest_path),
        'source_archive_sha256': seal['sha256'], 'source_seal_report_sha256': file_hash(seal_path),
        'uids': uids, 'train_uids': train, 'internal_check_uids': check, 'split': old['split'],
        'paths': paths, 'training': training, 'loss': loss, 'jobs': jobs,
        'source_single_jobs': source_single_jobs, 'endpoint_scale_squared': scale,
        'evaluation': evaluation, 'diagnostics': {'planned_slots': 120, 'checkpoints': ['initial'] + list(map(str, CHECKPOINT_UPDATES)),
            'intermediate_access': 'only original train12; source initial and source20000 are the same frozen weights',
            'checkpoint_selection': False, 'macro_and_per_parent_error_required': True},
        'quality': old['quality'], 'deployment': old['deployment'], 'input': old['input'],
        'selection': old['selection'], 'frozen': old['frozen'] + ['fitdiag_sealed_results', 'fitdiag_four_final20000_checkpoints'],
        'fixed_source_hashes': fixed_hashes,
        'research_scope': {'independent_development_exploration': True, 'heldout_generalization': False,
            'architecture_data_mainfield_teacher_unchanged': True, 'zero_teachers_kept': True,
            'original_endpoint_gate_not_passed_and_not_revised': True, 'prior_fitdiag_result_not_rewritten': True,
            'matching': 'Each joint training parent receives 80000 total exposures after its v1 initialization, matching the frozen single20000 parent exposure.',
            'joint_optimizer_updates_240000_vs_single_20000': True, 'equal_compute_or_adam_step_claim': False,
            'single_references_seed10_only_joint_seeds10_and11': True,
            'new_per_parent_exposure_73333_or_73334_because_source_has_6666_or_6667': True,
            'single_success_is_not_shared_fitting_or_generalization': True,
            'full_MeshFlow_comparison': 'independent pending track'},
        'hashes': {'training_labels_sha256': copied_hashes['train_labels.npz'], 'source_protocol_sha256': old_ph,
            'source_run_manifest_sha256': file_hash(source_manifest_path),
            'initial_checkpoint_sha256': copy.deepcopy(old['hashes']['initial_checkpoint_sha256']),
            'initial_model_sha256': copy.deepcopy(old['hashes']['initial_model_sha256']),
            'joint_source_checkpoint_sha256': {str(seed): jobs[f'joint_{seed}']['initial_checkpoint_sha256'] for seed in (10, 11)},
            'joint_source_model_sha256': {str(seed): jobs[f'joint_{seed}']['initial_model_sha256'] for seed in (10, 11)},
            'config_sha256': config_hash({'training': training, 'loss': loss, 'jobs': jobs})}}
    # All source preconditions are validated before creating the new run.
    run.mkdir(parents=True, exist_ok=False)
    for mapping, hashes in ((copy_map, copied_hashes), (evidence_map, evidence_hashes)):
        for target_rel, source_rel in mapping.items():
            target = inside(run, target_rel); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inside(source, source_rel), target)
            require(file_hash(target) == hashes[target_rel], 'Copied source bytes changed')
    write_json(run / 'DATA_MANIFEST.json', {'files': copied_hashes})
    write_json(run / 'SOURCE_EVIDENCE_MANIFEST.json', {'files': evidence_hashes})
    protocol['hashes']['data_manifest_sha256'] = file_hash(run / 'DATA_MANIFEST.json')
    protocol['hashes']['source_evidence_manifest_sha256'] = file_hash(run / 'SOURCE_EVIDENCE_MANIFEST.json')
    write_json(run / 'RUN_PROTOCOL.json', protocol)
    write_json(run / 'SOURCE_PROVENANCE.json', {'source_run': str(source),
        'source_manifest_sha256': file_hash(source_manifest_path), 'verified_source_manifest': source_manifest,
        'all_source_manifest_files_verified': len(source_manifest), 'source_archive_sha256': seal['sha256'],
        'source_seal_report_sha256': file_hash(seal_path), 'source_seal_report': seal,
        'source_and_destination_relative_paths': copy_map, 'source_evidence_relative_paths': evidence_map,
        'original_23_files_byte_identical': True, 'data_manifest_file_count': len(copied_hashes),
        'mechanism_evidence_files_copied': len(evidence_hashes), 'source_mechanism_expected_slots': 156,
        'source_mechanism_errors_retained': len(mechanism['errors']),
        'source_mechanism_missing_success_slots_retained': mechanism['missing_success_slots'],
        'source_parent_exposures': source_exposures,
        'old_outputs_written': False, 'prepare_source_sha256': file_hash(__file__)})
    write_json(run / 'REPO_IDENTITY.json', identity)
    write_json(run / 'ENVIRONMENT.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'torch_cuda_runtime': torch.version.cuda, 'prepare_used_cuda_compute': False,
        'training_device': 'cuda', 'deployment_device': 'cpu', 'free_output_disk_bytes': shutil.disk_usage(run).free})
    require(file_hash(source_manifest_path) == protocol['source_run_manifest_sha256'], 'Source manifest changed during preparation')
    for mapping, hashes in ((copy_map, copied_hashes), (evidence_map, evidence_hashes)):
        for target, origin in mapping.items():
            require(file_hash(source / origin) == hashes[target], 'Source artifact changed during preparation')
    print(json.dumps({'run': str(run), 'protocol_sha256': file_hash(run / 'RUN_PROTOCOL.json'),
        'copied_data_files': len(copied_hashes), 'copied_evidence_files': len(evidence_hashes),
        'source_files_verified': len(source_manifest), 'jobs': list(jobs), 'planned_diagnostic_slots': 120}, indent=2), flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True); parser.add_argument('--run', required=True)
    args = parser.parse_args(); prepare(args.source_run, args.run)


if __name__ == '__main__':
    main()
