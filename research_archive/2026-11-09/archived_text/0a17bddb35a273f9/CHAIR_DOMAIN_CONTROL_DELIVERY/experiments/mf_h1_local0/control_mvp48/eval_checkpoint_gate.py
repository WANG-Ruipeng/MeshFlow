"""Read-only Session C dependency gate. No torch import, model or CUDA work.

The gate only hashes files and compares JSON identities. Actual tensor validation
belongs to Session B's official load_mvp_checkpoint, called by the worker only
after this gate succeeds for all three endpoints.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .eval_common import ROOT, RUN, read_json, sha256

MODELS = ('T1', 'Geo', 'Full')
PHASE_UPDATES = {'T1': 1000, 'Geo_shared': 500, 'Geo': 500, 'Full': 500}
LOADER_MODULE = 'experiments.mf_h1_local0.control_mvp48.model_mvp'
LOADER_RELATIVE = 'experiments/mf_h1_local0/control_mvp48/model_mvp.py'
OFFICIAL_FILE_SHA256 = 'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
OFFICIAL_BACKBONE_SHA256 = 'b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
OFFICIAL_CONFIG_SHA256 = 'fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f'
CONFIG_RELATIVE = 'configs/snet/base-120m-ot-v-chair.yaml'


class WaitingReady(RuntimeError):
    """Only an absent TRAIN_READY sentinel permits waiting and trying again."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
             'Malformed SHA256 identity')
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _contained(root, value, *, absolute=False):
    _require(isinstance(value, str) and value and '\\' not in value,
             'Expected canonical WSL path')
    given = Path(value)
    _require(not absolute or given.is_absolute(), 'Endpoint path must be absolute')
    root = Path(root).resolve()
    path = (root / given).resolve()
    _require(path.is_relative_to(root) and path != root, 'Path escapes its bound root')
    return path


def _checked(root, record, *, absolute=False):
    path = _contained(root, record['path'], absolute=absolute)
    _require(path.is_file(), 'Published dependency is missing: ' + str(path))
    expected = _digest(record['sha256'])
    _require(sha256(path) == expected, 'File hash changed: ' + str(path))
    if 'bytes' in record:
        _require(type(record['bytes']) is int and path.stat().st_size == record['bytes'],
                 'File size changed: ' + str(path))
    return path


def _protocol_checks(protocol, ready, root, data_ready):
    expected = {
        'schema': 'control_mvp48_training_protocol_v1', 'training_seed': 10,
        'model_initialization_seed': 1010, 'total_actual_optimizer_updates': 2500,
        'phases': [['T1', 0, 1000], ['Geo_shared', 0, 500], ['Geo', 500, 1000], ['Full', 500, 1000]],
        'effective_batch': 8, 'microbatch': 1, 'N_range': [128, 256],
        'optimizer': {'name': 'AdamW', 'lr': 1e-5, 'betas': [.9, .95], 'weight_decay': 0., 'clip': 1.},
        'scheduler': None, 'ema': None, 'precision': 'BF16_forward_FP32_parameters_coordinates_geometry',
        'determinism': 'strict_TF32_off_SDPA_math', 'full_ramp_index': 'logical_step_minus_500',
        'shared_optimizer_copied': True,
        'FM_denominator': 'all_valid_free_scalar_coordinates_in_effective_batch',
        'feature_statistics_fitted': False, 'gpu_workers': 1,
        'gpu_lock': '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',
        'data_ready_sha256': sha256(root/'DATA_READY.json'),
        'binding_sha256': sha256(root/'registration/train_data_binding.json'),
        'manifest_sha256': data_ready['manifest']['sha256'],
        'plan_sha256': data_ready['task_plans']['train']['sha256'],
        'data_contract_sha256': data_ready['contract']['sha256'],
        'data_rng': 'Frozen per-sample Session A Gaussian/time/permutation seeds; no worker-derived seeds',
    }
    for key, value in expected.items():
        _require(key in protocol and protocol[key] == value, 'Training protocol mismatch: ' + key)
    loss = protocol['loss']
    _require(loss == {
        'name': 'edge5', 'version': 1, 'FM': 'masked_free_coordinate_MSE',
        'time_gate': 'original_FP32_t>=0.5', 'ramp': 'min(additional_step/50,1)',
        'lambda_edge': 4.071385484299878, 'lambda_surface': 0.3266104383520167,
        'auxiliary_denominator': 'all_8_samples_including_inactive', 'geometry_dtype': 'float32',
        'source_ids_are_model_inputs': False,
        'geometry': 'finite_corresponding_interface_edges_and_area_unsigned_normal_MMD'},
        'Training loss recipe differs from the fixed protocol')
    digest = _json_hash(protocol)
    _require(ready['protocol_sha256'] == digest, 'TRAIN_READY training protocol hash differs')
    frozen = read_json(root/'registration/train_protocol.json')
    _require(frozen == {'protocol': protocol, 'protocol_sha256': digest},
             'Published endpoint protocol differs from training registration')
    return digest


def _metadata(meta, variant, step, protocol_sha, fork_sha):
    expected_phases = dict.fromkeys(PHASE_UPDATES, 0)
    if variant == 'T1':
        expected_phases['T1'] = step
    else:
        expected_phases.update(T1=1000, Geo_shared=500)
        if variant in ('Geo', 'Full'):
            expected_phases['Geo'] = 500
        if variant == 'Full':
            expected_phases['Full'] = 500
    expected = dict(variant=variant, logical_step=step,
        context_encoder=None if variant == 'T1' else 'geo',
        official_file_sha256=OFFICIAL_FILE_SHA256,
        official_backbone_sha256=OFFICIAL_BACKBONE_SHA256,
        config_sha256=OFFICIAL_CONFIG_SHA256, protocol_sha256=protocol_sha,
        actual_updates_at_save=sum(expected_phases.values()), executed_phase_updates=expected_phases,
        fork_checkpoint_sha256=fork_sha if variant in ('Geo', 'Full') else None,
        official_pretraining_steps_in_logical_count=False,
        optimizer_reset_at_phase_start=variant in ('T1', 'Geo_shared'),
        loss_recipe='edge5' if variant == 'Full' else 'fm')
    for key, value in expected.items():
        _require(key in meta and meta[key] == value, f'{variant} metadata mismatch: {key}')
    _require(type(meta['logical_step']) is int and type(meta['actual_updates_at_save']) is int,
             'Step accounting must use integers')
    for key in ('model_state_sha256', 'optimizer_state_sha256', 'rng_sha256', 'stream_state_sha256'):
        _digest(meta[key])


def _data_checks(root, evaluation):
    data_path = root/'DATA_READY.json'
    _require(data_path.is_file(), 'TRAIN_READY cannot precede DATA_READY')
    data = read_json(data_path)
    _require(data.get('schema') == 'control_mvp48_data_ready_v1' and data.get('status') == 'READY',
             'DATA_READY schema/status differs')
    _require(data.get('data_revision') == 'repair_v2', 'Only bound repair_v2 is accepted')
    _require(data['splits'] == {'train': 32, 'validation': 8, 'test': 8}, 'Split counts changed')
    _require(data['three_model_generations'] == 96 and data['teacher_checks'] == 24,
             'Data request budget differs')
    identity = evaluation['data_identity']
    digest = sha256(data_path)
    _require(identity['data_ready_sha256'] == digest and identity['ready'] == data,
             'Evaluation frozen DATA_READY identity differs from current data')
    _require(len(data['files']) == 81, 'Expected exactly 81 bound repaired artifacts')
    files = {name: record['sha256'] for name, record in data['files'].items()}
    _require(identity['files'] == files, 'Evaluation data file identity differs')
    for name, record in data['files'].items():
        _checked(root, dict(record, path=name))
    for record in [data[k] for k in ('manifest', 'split', 'contract', 'independent_audit')] + list(data['task_plans'].values()):
        _require(files.get(record['path']) == record['sha256'], 'Data reference is not frozen')
    _require(data['contract']['path'] == 'registration/data_contract_repair_v2.json',
             'Wrong repaired data contract')
    _require(read_json(root/data['independent_audit']['path'])['status'] == 'PASS',
             'Independent data audit did not pass')
    splits = read_json(root/data['split']['path'])
    groups = [set(splits[k]) for k in ('train', 'validation', 'test')]
    _require([len(group) for group in groups] == [32, 8, 8] and
             all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))),
             'Independent parent split overlap/count error')
    binding = read_json(root/'registration/train_data_binding.json')
    _require(binding['status'] == 'FROZEN' and binding['data_ready_sha256'] == digest,
             'Session B data binding differs')
    _require(binding['accepted_contract_sha256'] == data['contract']['sha256'] and
             binding['manifest'] == data['manifest'] and binding['plan'] == data['task_plans']['train'] and
             binding['files'] == files, 'Session B data identities differ')
    _require(set(binding['train_object_ids']) == groups[0] and
             set(binding['evaluation_object_ids']) == groups[1] | groups[2],
             'Session B bound parent split differs')
    _require(_contained(root, binding['loader_path'], absolute=True) == (root/'data/load_sample.py').resolve()
             and _contained(root, binding['data_root'], absolute=True) == (root/'data').resolve(),
             'Session B loader/data root differs')
    ack = read_json(root/'registration/eval_data_ack.json')
    _require(ack['data_ready_sha256'] == digest and ack['contract_sha256'] == data['contract']['sha256']
             and ack['protocol_sha256'] == sha256(root/'registration/eval_protocol.json')
             and ack['requests_sha256'] == sha256(root/'registration/eval_requests.json'),
             'Session C frozen binding receipt differs')
    return data, digest


def inspect_ready(root=RUN, repo=ROOT):
    """Return JSON-compatible verified identities; never deserialize checkpoints.

    Missing TRAIN_READY is the sole WaitingReady case. Any published identity
    inconsistency is a hard failure and must not trigger automatic retries.
    Optional roots exist solely for CPU synthetic fixtures.
    """
    root, repo = Path(root).resolve(), Path(repo).resolve()
    ready_path = root/'TRAIN_READY.json'
    if not ready_path.is_file():
        raise WaitingReady('Session B TRAIN_READY.json is absent; no model load allowed')
    ready = read_json(ready_path)
    ready_hash = sha256(ready_path)
    _require(ready.get('schema') == 'control_mvp48_train_ready_v1' and ready.get('status') == 'READY'
             and ready.get('owner') == 'Session B', 'TRAIN_READY schema/status/owner differs')
    _require(ready['actual_optimizer_updates'] == 2500 and ready['phase_updates'] == PHASE_UPDATES,
             'All four training phases must finish within the fixed 2500 update tree')
    _require(ready.get('default_model_changed') is False, 'Default model unexpectedly changed')
    _require(ready['loader_module'] == LOADER_MODULE and ready['loader_function'] == 'load_mvp_checkpoint'
             and ready['loader_returns'] == ['model', 'actual_metadata', 'hashes'],
             'Unified Session B loader contract differs')
    _require(set(ready['final_checkpoints']) == set(MODELS), 'Exactly three final endpoints are required')

    evaluation_path = root/'registration/eval_protocol.json'
    evaluation = read_json(evaluation_path)
    _require(evaluation.get('schema') == 'control_mvp48_evaluation_v1' and evaluation.get('status') == 'FROZEN'
             and evaluation.get('owner') == 'Session C', 'Evaluation is not frozen')
    _require(evaluation['models'] == list(MODELS) and evaluation['endpoint_step'] == 1000
             and evaluation['formal_requests'] == 96 and evaluation['formal_steps'] == 50
             and evaluation['formal_NFE'] == 4800 and evaluation['teacher_forwards'] == 24,
             'Evaluation matrix/budget changed')
    data_ready, data_hash = _data_checks(root, evaluation)
    _require(ready['data_ready_sha256'] == data_hash, 'TRAIN_READY is bound to stale data')
    protocol = ready['final_checkpoints']['T1']['metadata']['protocol']
    protocol_hash = _protocol_checks(protocol, ready, root, data_ready)

    sources = ready['source_hashes']
    _require(isinstance(sources, dict) and sources == read_json(root/'registration/train_code_hashes.json'),
             'Training source snapshot differs')
    required_sources = {LOADER_RELATIVE, CONFIG_RELATIVE, 'native_t1/portable_checkpoint.py',
        'native_t1/runtime.py', 'native_t1/model.py', 'native_t1/context_geometry.py', 'models/equidit.py',
        'experiments/mf_h1_local0/control_mvp48/train_contract.py',
        'experiments/mf_h1_local0/control_mvp48/train_io.py',
        'experiments/mf_h1_local0/control_mvp48/train_run.py'}
    _require(required_sources <= set(sources), 'Incomplete source identity in TRAIN_READY')
    for name, digest in sources.items():
        _checked(repo, {'path': name, 'sha256': digest})
    _require(ready['loader_sha256'] == sources[LOADER_RELATIVE], 'Unified loader hash differs')
    config_text = (repo/CONFIG_RELATIVE).read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    _require(hashlib.sha256(config_text.encode()).hexdigest() == OFFICIAL_CONFIG_SHA256,
             'Official architecture configuration changed')
    for name, digest in evaluation['scorer_identity'].items():
        _checked(repo, {'path': name, 'sha256': digest})

    _require(ready['checkpoints_json']['path'] == 'checkpoints/checkpoints.json', 'Unexpected checkpoint index')
    index_path = _checked(root, ready['checkpoints_json'])
    index = read_json(index_path)['checkpoints']
    expected_index = {('T1', 500), ('T1', 1000), ('Geo_shared', 500), ('Geo', 1000), ('Full', 1000)}
    _require(len(index) == 5 and {(r['metadata']['variant'], r['metadata']['logical_step']) for r in index} == expected_index,
             'Checkpoint index must contain exactly the five registered save points')
    indexed = {(r['metadata']['variant'], r['metadata']['logical_step']): r for r in index}
    fork = indexed[('Geo_shared', 500)]
    fork_hash = _digest(fork['checkpoint_sha256'])
    verified_index = {}
    for (variant, step), record in indexed.items():
        _require(record.get('strict_reload') == 'PASS', 'Checkpoint did not pass Session B strict reload')
        _metadata(record['metadata'], variant, step, protocol_hash, fork_hash)
        path = _checked(root, dict(path=record['path'], sha256=record['checkpoint_sha256']), absolute=True)
        _require(path == (root/f'checkpoints/{variant}_step{step}.pt').resolve(), 'Unexpected checkpoint location')
        sidecar_path = path.with_suffix('.json')
        sidecar = read_json(sidecar_path)
        _require(sidecar == {k: record[k] for k in ('path', 'checkpoint_sha256', 'metadata')},
                 'Checkpoint sidecar differs from index')
        verified_index[(variant, step)] = dict(path=str(path), checkpoint_sha256=record['checkpoint_sha256'],
            metadata=record['metadata'], sidecar_sha256=sha256(sidecar_path))

    endpoints = {}
    for label in MODELS:
        item = ready['final_checkpoints'][label]
        saved = verified_index[(label, 1000)]
        _require(item['strict_reload'] == 'PASS', label + ' final strict reload did not pass')
        _require(_contained(root, item['path'], absolute=True) == Path(saved['path']), 'Final endpoint path differs')
        expected_hashes = dict(checkpoint_sha256=saved['checkpoint_sha256'],
            model_state_sha256=saved['metadata']['model_state_sha256'], protocol_sha256=protocol_hash)
        actual_meta = dict(saved['metadata'], strict_load=True, checkpoint=saved['path'],
                           protocol=protocol, inference_trainable=False)
        _require(item['hashes'] == expected_hashes and item['metadata'] == actual_meta,
                 label + ' final loader metadata/hashes differ')
        endpoints[label] = dict(path=saved['path'], hashes=expected_hashes, metadata=actual_meta,
                                sidecar_sha256=saved['sidecar_sha256'])
    for name, expected_path in (('report', 'training/TRAIN_REPORT.md'),
                                ('training_log', 'training/training_log.csv'), ('code_diff', 'training/code.diff')):
        _require(ready[name]['path'] == expected_path, 'Unexpected training artifact: ' + name)
        _checked(root, ready[name])
    status = read_json(root/'training/status.json')
    _require(status['status'] == 'COMPLETE' and status['actual_optimizer_updates'] == 2500
             and status['phase_updates'] == PHASE_UPDATES and status['optimizer_step_attempts'] == 2500,
             'Training status does not certify all completed phases without extra attempts')
    _require(sha256(ready_path) == ready_hash and sha256(root/'DATA_READY.json') == data_hash,
             'Readiness changed during identity inspection')
    return dict(schema='control_mvp48_eval_endpoint_gate_v1', status='READY', run=str(root),
        endpoints=endpoints, fork=verified_index[('Geo_shared', 500)],
        train_ready_sha256=ready_hash, data_ready_sha256=data_hash,
        eval_protocol_sha256=sha256(evaluation_path), train_protocol_sha256=protocol_hash,
        source_hashes=sources, loader_module=LOADER_MODULE, loader_function='load_mvp_checkpoint',
        verified_data_files=81, verified_checkpoint_files=5,
        checkpoint_tensor_deserializations=0, model_instantiations=0, model_forwards=0,
        note='Readiness hashes only; worker must strict-load all three final checkpoints on CPU before GPU dispatch.')


def validate_loaded(label, actual_meta, hashes, gate):
    """Match the official loader's actual CPU/GPU return to the gated endpoint."""
    _require(gate.get('status') == 'READY' and label in MODELS, 'Unknown or ungated endpoint')
    endpoint = gate['endpoints'][label]
    _require(actual_meta == endpoint['metadata'], label + ' actual loaded metadata differs from gate')
    _require(hashes == endpoint['hashes'], label + ' actual loaded tensor/file hashes differ from gate')
    _require(actual_meta.get('strict_load') is True and actual_meta.get('inference_trainable') is False,
             'Strict eval-only Session B loading is required')
    _require(Path(actual_meta['checkpoint']).is_absolute() and
             Path(actual_meta['checkpoint']).resolve() == Path(endpoint['path']),
             'Actual loaded checkpoint path differs')
    return dict(status='PASS', model=label, checkpoint=endpoint['path'], hashes=hashes)


if __name__ == '__main__':
    try:
        result = inspect_ready()
    except WaitingReady as exc:
        result = dict(status='WAITING_TRAIN_READY', reason=str(exc), model_instantiations=0, model_forwards=0)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
