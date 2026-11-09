"""CPU-only JSON/byte fixtures for unattended endpoint gate; no tensor loads."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

from . import eval_checkpoint_gate as gate
from .eval_common import RUN, ROOT, read_json, sha256, write_json


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def record(root, path):
    absolute = root/path
    return dict(path=path, sha256=sha256(absolute))


def build(root):
    data = read_json(RUN/'DATA_READY.json')
    # Materialize tiny synthetic bytes only. No live data/checkpoint copy, load,
    # OT, model, numpy, torch, CUDA or GPU activity belongs to this fixture.
    splits = {key: [key+str(i) for i in range(count)]
              for key, count in [('train', 32), ('validation', 8), ('test', 8)]}
    for name in data['files']:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('SYNTHETIC_GATE_FIXTURE '+name+'\n').encode())
    dump(root/data['split']['path'], splits)
    dump(root/data['independent_audit']['path'], dict(status='PASS'))
    data['files'] = {name: dict(sha256=sha256(root/name), bytes=(root/name).stat().st_size)
                     for name in data['files']}
    for item in [data[k] for k in ('manifest', 'split', 'contract', 'independent_audit')] + list(data['task_plans'].values()):
        item['sha256'] = data['files'][item['path']]['sha256']
    dump(root/'DATA_READY.json', data)
    data_hash = sha256(root/'DATA_READY.json')
    binding = dict(schema='control_mvp48_train_binding_v1', status='FROZEN',
        data_ready_sha256=data_hash, accepted_contract_sha256=data['contract']['sha256'],
        train_object_ids=splits['train'], evaluation_object_ids=splits['validation']+splits['test'],
        files={name: item['sha256'] for name, item in data['files'].items()},
        loader_path=str(root/'data/load_sample.py'), data_root=str(root/'data'),
        manifest=data['manifest'], plan=data['task_plans']['train'])
    dump(root/'registration/train_data_binding.json', binding)
    protocol = read_json(RUN/'registration/train_protocol.json')['protocol']
    protocol.update(data_ready_sha256=data_hash,
        binding_sha256=sha256(root/'registration/train_data_binding.json'),
        manifest_sha256=data['manifest']['sha256'], plan_sha256=data['task_plans']['train']['sha256'],
        data_contract_sha256=data['contract']['sha256'])
    protocol_hash = gate._json_hash(protocol)
    dump(root/'registration/train_protocol.json', dict(protocol=protocol, protocol_sha256=protocol_hash))
    evaluation = read_json(RUN/'registration/eval_protocol.json')
    evaluation['data_identity'].update(data_ready_sha256=data_hash, ready=data,
        files={name: item['sha256'] for name, item in data['files'].items()})
    dump(root/'registration/eval_protocol.json', evaluation)
    dump(root/'registration/eval_requests.json', ['SYNTHETIC_NO_MODEL_REQUESTS'])
    dump(root/'registration/eval_data_ack.json', dict(data_ready_sha256=data_hash,
        contract_sha256=data['contract']['sha256'],
        protocol_sha256=sha256(root/'registration/eval_protocol.json'),
        requests_sha256=sha256(root/'registration/eval_requests.json')))
    sources = read_json(RUN/'registration/train_code_hashes.json')
    dump(root/'registration/train_code_hashes.json', sources)
    checkpoint_dir = root/'checkpoints'
    checkpoint_dir.mkdir()
    saves = [('T1', 500), ('T1', 1000), ('Geo_shared', 500), ('Geo', 1000), ('Full', 1000)]
    for label, step in saves:
        (checkpoint_dir/f'{label}_step{step}.pt').write_bytes(f'SYNTHETIC_NON_TENSOR_{label}_{step}'.encode())
    fork_hash = sha256(checkpoint_dir/'Geo_shared_step500.pt')
    index, final = [], {}
    for label, step in saves:
        phases = dict.fromkeys(gate.PHASE_UPDATES, 0)
        if label == 'T1':
            phases['T1'] = step
        else:
            phases.update(T1=1000, Geo_shared=500)
            if label in ('Geo', 'Full'):
                phases['Geo'] = 500
            if label == 'Full':
                phases['Full'] = 500
        path = checkpoint_dir/f'{label}_step{step}.pt'
        meta = dict(variant=label, logical_step=step, context_encoder=None if label == 'T1' else 'geo',
            model_state_sha256=gate._json_hash([label, step, 'model']),
            optimizer_state_sha256=gate._json_hash([label, step, 'optimizer']),
            rng_sha256=gate._json_hash([label, step, 'rng']),
            stream_state_sha256=gate._json_hash([label, step, 'stream']),
            protocol_sha256=protocol_hash, config_sha256=gate.OFFICIAL_CONFIG_SHA256,
            official_file_sha256=gate.OFFICIAL_FILE_SHA256,
            official_backbone_sha256=gate.OFFICIAL_BACKBONE_SHA256,
            actual_updates_at_save=sum(phases.values()), executed_phase_updates=phases,
            fork_checkpoint_sha256=fork_hash if label in ('Geo', 'Full') else None,
            official_pretraining_steps_in_logical_count=False,
            optimizer_reset_at_phase_start=label in ('T1', 'Geo_shared'),
            loss_recipe='edge5' if label == 'Full' else 'fm')
        item = dict(path=str(path), checkpoint_sha256=sha256(path), metadata=meta)
        dump(path.with_suffix('.json'), item)
        index.append(dict(item, strict_reload='PASS', save_and_verify_seconds=0.))
        if label in gate.MODELS and step == 1000:
            final[label] = dict(path=str(path), strict_reload='PASS',
                metadata=dict(meta, strict_load=True, checkpoint=str(path), protocol=protocol, inference_trainable=False),
                hashes=dict(checkpoint_sha256=sha256(path), model_state_sha256=meta['model_state_sha256'],
                            protocol_sha256=protocol_hash))
    dump(checkpoint_dir/'checkpoints.json', dict(checkpoints=index))
    dump(root/'training/status.json', dict(status='COMPLETE', actual_optimizer_updates=2500,
        phase_updates=gate.PHASE_UPDATES, optimizer_step_attempts=2500))
    for relative in ('training/TRAIN_REPORT.md', 'training/training_log.csv', 'training/code.diff'):
        (root/relative).write_text('SYNTHETIC_GATE_FIXTURE_ONLY\n')
    ready = dict(schema='control_mvp48_train_ready_v1', status='READY', owner='Session B',
        actual_optimizer_updates=2500, phase_updates=gate.PHASE_UPDATES,
        protocol_sha256=protocol_hash, data_ready_sha256=data_hash, final_checkpoints=final,
        loader_module=gate.LOADER_MODULE, loader_function='load_mvp_checkpoint',
        loader_returns=['model', 'actual_metadata', 'hashes'], loader_sha256=sources[gate.LOADER_RELATIVE],
        checkpoints_json=record(root, 'checkpoints/checkpoints.json'),
        report=record(root, 'training/TRAIN_REPORT.md'), training_log=record(root, 'training/training_log.csv'),
        code_diff=record(root, 'training/code.diff'), source_hashes=sources, default_model_changed=False)
    dump(root/'TRAIN_READY.json', ready)


def main():
    started = time.perf_counter()
    fixture_root = RUN/'evaluation/eval_gate_fixture'
    fixture_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='synthetic_', dir=fixture_root)).resolve()
    build(root)
    checks = []
    baseline = gate.inspect_ready(root)
    checks.append(dict(name='synthetic_complete_three_endpoints', status='PASS'))
    for label in gate.MODELS:
        endpoint = baseline['endpoints'][label]
        gate.validate_loaded(label, endpoint['metadata'], endpoint['hashes'], baseline)
    checks.append(dict(name='official_loader_return_contract', status='PASS'))

    def reject(name, relative, mutate, expected=ValueError):
        path = root/relative
        before = path.read_bytes()
        try:
            current = read_json(path)
            changed = mutate(current)
            dump(path, changed if changed is not None else current)
            try:
                gate.inspect_ready(root)
            except expected as exc:
                checks.append(dict(name=name, status='PASS', rejection=str(exc)))
            else:
                raise AssertionError('Invalid fixture accepted: '+name)
        finally:
            path.write_bytes(before)

    ready_name = 'TRAIN_READY.json'
    reject('partial_final_set', ready_name, lambda x: x['final_checkpoints'].pop('Full') and x)
    reject('step500_as_final', ready_name, lambda x: x['final_checkpoints']['T1']['metadata'].update(logical_step=500))
    reject('not_all_training_done', ready_name, lambda x: x.update(actual_optimizer_updates=2000))
    reject('wrong_phase_accounting', ready_name, lambda x: x['phase_updates'].update(Full=499))
    reject('stale_data_binding', ready_name, lambda x: x.update(data_ready_sha256='0'*64))
    reject('different_loader', ready_name, lambda x: x.update(loader_module='old.double_object'))
    reject('changed_loader_hash', ready_name, lambda x: x.update(loader_sha256='0'*64))
    reject('endpoint_escape', ready_name, lambda x: x['final_checkpoints']['Geo'].update(path='/tmp/Geo_step1000.pt'))
    reject('relative_endpoint_path', ready_name, lambda x: x['final_checkpoints']['Geo'].update(path='checkpoints/Geo_step1000.pt'))
    reject('unverified_endpoint', ready_name, lambda x: x['final_checkpoints']['Full'].update(strict_reload='NOT_RUN'))
    reject('raw_EMA_mix', ready_name, lambda x: x['final_checkpoints']['T1']['metadata']['protocol'].update(ema=.999))
    reject('foreign_official_initialization', ready_name, lambda x: x['final_checkpoints']['T1']['metadata'].update(official_file_sha256='0'*64))
    reject('different_shared_fork', ready_name, lambda x: x['final_checkpoints']['Full']['metadata'].update(fork_checkpoint_sha256='0'*64))
    reject('sidecar_tamper', 'checkpoints/Geo_step1000.json', lambda x: x['metadata'].update(logical_step=500))
    reject('index_tamper', 'checkpoints/checkpoints.json', lambda x: x['checkpoints'].pop())
    reject('data_revision_tamper', 'DATA_READY.json', lambda x: x.update(data_revision='strict_v1'))
    reject('status_extra_attempt', 'training/status.json', lambda x: x.update(optimizer_step_attempts=2501))
    reject('protocol_unfreeze', 'registration/eval_protocol.json', lambda x: x.update(status='DRAFT'))
    reject('source_identity_missing', ready_name, lambda x: x['source_hashes'].pop(gate.LOADER_RELATIVE) and x)
    path = root/'checkpoints/Full_step1000.pt'
    original = path.read_bytes()
    try:
        path.write_bytes(b'BROKEN_SYNTHETIC_BYTES')
        try:
            gate.inspect_ready(root)
        except ValueError as exc:
            checks.append(dict(name='checkpoint_byte_tamper', status='PASS', rejection=str(exc)))
        else:
            raise AssertionError('Changed checkpoint bytes accepted')
    finally:
        path.write_bytes(original)
    ready_path = root/'TRAIN_READY.json'
    pending_path = root/'eval_unpublished_ready.json'
    ready_path.rename(pending_path)
    try:
        try:
            gate.inspect_ready(root)
        except gate.WaitingReady as exc:
            checks.append(dict(name='absent_ready_waits_without_loading', status='PASS', rejection=str(exc)))
        else:
            raise AssertionError('Missing sentinel accepted')
    finally:
        pending_path.rename(ready_path)
    wrong = copy.deepcopy(baseline['endpoints']['T1']['metadata'])
    wrong['strict_load'] = False
    try:
        gate.validate_loaded('T1', wrong, baseline['endpoints']['T1']['hashes'], baseline)
    except ValueError:
        checks.append(dict(name='actual_non_strict_load_rejected', status='PASS'))
    else:
        raise AssertionError('Non-strict actual load accepted')
    assert 'torch' not in sys.modules
    gate.inspect_ready(root)
    result = dict(schema='control_mvp48_eval_checkpoint_gate_fixture_v1', status='PASS',
        checks=checks, checks_passed=len(checks), synthetic_root=str(root),
        fixture_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        actual_checkpoints_loaded=0, tensor_deserializations=0, torch_imported=False,
        model_instantiations=0, model_forwards=0, optimizer_updates=0, OT_calls=0,
        source_hashes={name: sha256(ROOT/'experiments/mf_h1_local0/control_mvp48'/name)
                       for name in ('eval_checkpoint_gate.py', 'eval_checkpoint_gate_fixture.py')},
        seconds=time.perf_counter()-started)
    write_json(RUN/'registration/eval_checkpoint_gate_fixture.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'checks_passed', 'fixture_bytes', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
