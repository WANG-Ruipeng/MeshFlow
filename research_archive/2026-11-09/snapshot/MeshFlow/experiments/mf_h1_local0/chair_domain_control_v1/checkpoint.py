"""Strict, isolated continuation checkpoints; the old loader remains read-only."""
from native_t1 import runtime as _runtime
import gc
import hashlib
import os
from pathlib import Path
import time
import torch
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.artifacts import state_sha256
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.context_geometry import ContextGeometryEncoder
from native_t1.portable_checkpoint import (_read_config, config_sha256,
    OFFICIAL_FILE_SHA256, OFFICIAL_BACKBONE_SHA256)
from experiments.mf_h1_local0.control_mvp48.model_mvp import MVPModel
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu, tree_hash, capture_rng
from .common import atomic_json, read_json, digest, json_hash

SCHEMA = 'chair_domain_control_checkpoint_v1'
ARMS = ('T1','Geo','Full')


def tensor_state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_payload(payload):
    if payload.get('schema') != SCHEMA or payload.get('update_in_progress') is not False:
        raise ValueError('Not a completed budget checkpoint')
    m = payload['metadata']; arm = m['variant']; step = m['phase_step']
    if arm not in ARMS or type(step) is not int:
        raise ValueError('Unknown continuation arm/step')
    low, high = 0, 3000
    if m['conditional_total_step'] != 1000+step or m['logical_step'] != 1000+step:
        raise ValueError('Phase and conditional cumulative step differ')
    if m['recovery_mode'] != 'RESTORE_ALL' or m['optimizer_reset_at_1000'] is not False:
        raise ValueError('Source AdamW recovery contract differs')
    if not low <= step <= high:
        raise ValueError('Checkpoint outside fixed budget')
    if m['context_encoder'] != ('geo' if arm in ('Geo','Full') else None):
        raise ValueError('Arm/network mismatch')
    if m['loss_recipe'] != ('edge5' if arm == 'Full' else 'fm'):
        raise ValueError('Arm/loss mismatch')
    if m['protocol_sha256'] != json_hash(payload['protocol']):
        raise ValueError('Frozen protocol identity mismatch')
    if m['official_file_sha256'] != OFFICIAL_FILE_SHA256 or m['official_backbone_sha256'] != OFFICIAL_BACKBONE_SHA256:
        raise ValueError('Official lineage mismatch')
    if m['config_sha256'] != config_sha256(DEFAULT_CONFIG):
        raise ValueError('Architecture changed')
    if m['stream_state_sha256'] != json_hash(payload['stream_state']) or payload['stream_state']['batch_index'] != step:
        raise ValueError('Stream cursor/hash mismatch')
    if m['rng_sha256'] != tree_hash(payload['rng']):
        raise ValueError('RNG hash mismatch')
    if m['actual_new_updates'] != step-low:
        raise ValueError('New update accounting mismatch')
    if m.get('counts',{}).get('optimizer_updates') != step:
        raise ValueError('Actual optimizer update counter differs from phase cursor')
    if m.get('counts',{}).get('forward_returns') != 8*step or m.get('counts',{}).get('backward_returns') != 8*step:
        raise ValueError('Actual training microbatch counters differ')
    if m['full_recovery_state'] != ('optimizer' in payload):
        raise ValueError('Full recovery metadata disagrees with actual optimizer payload')
    if m['source_checkpoint_sha256'] != payload['protocol']['source_checkpoints'][arm]['checkpoint_sha256']:
        raise ValueError('Restored source identity changed')
    for key in ('forward_attempts','forward_returns','backward_attempts','backward_returns'):
        if m['counts'].get(key) != 8*step:raise ValueError('Training attempt counter differs: '+key)
    if m['counts'].get('optimizer_step_attempts') != step or not m['counts'].get('update_validated'):
        raise ValueError('Checkpoint was not saved after a validated optimizer boundary')
    if 'optimizer' in payload:
        opt = payload['optimizer']
        if tree_hash(opt) != m['optimizer_state_sha256']:
            raise ValueError('AdamW content changed')
        expected_step = 1000+step
        if not opt['state'] or any(int(s['step']) != expected_step for s in opt['state'].values()):
            raise ValueError('AdamW step mismatch')
        for g in opt['param_groups']:
            if (g['lr'],tuple(g['betas']),g['weight_decay']) != (1e-5,(.9,.95),0.):
                raise ValueError('AdamW settings changed')
    return m


def load_checkpoint(path, device='cpu', trainable=False):
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    m = validate_payload(payload)
    config = _read_config(Path(DEFAULT_CONFIG))
    _runtime.configure_stable_runtime()
    with torch.device('meta'):
        model = MVPModel(DiT(**config), trainable=False)
    model.backbone.x_embedder.embed_fn, _ = get_embedder(config['pe_freq'], input_dims=3)
    if m['context_encoder'] == 'geo':
        model.context_encoder = ContextGeometryEncoder('geo', seed=1010)
    state = payload['model']
    if any(x.dtype != torch.float32 or x.device.type != 'cpu' or not bool(torch.isfinite(x).all()) for x in state.values()):
        raise ValueError('Model must contain finite CPU FP32 tensors')
    model.load_state_dict(state, strict=True, assign=True)
    if state_sha256(model) != m['model_state_sha256']:
        raise ValueError('Model hash changed')
    sha = digest(path)
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        record = read_json(sidecar)
        if record['checkpoint_sha256'] != sha or record['metadata'] != m:
            raise ValueError('Checkpoint sidecar differs')
    model.set_trainable(trainable).eval().to(device)
    return model, dict(m, checkpoint=str(path.resolve()), strict_load=True), dict(
        checkpoint_sha256=sha,model_state_sha256=m['model_state_sha256'],protocol_sha256=m['protocol_sha256'])


def component_changes(state, source_path):
    old = torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)['model']
    result = {g:dict(squared_delta=0.,max_abs_delta=0.,numel=0) for g in ('role','Geo','backbone')}
    for key, value in state.items():
        group = 'role' if key == 'role_embedding' else ('Geo' if key.startswith('context_encoder.') else 'backbone')
        delta = value-old[key]
        r=result[group]; r['squared_delta'] += float(delta.double().square().sum())
        r['max_abs_delta'] = max(r['max_abs_delta'],float(delta.abs().max()))
        r['numel'] += value.numel()
    for r in result.values():
        r['l2_delta'] = r.pop('squared_delta')**.5
        r['changed'] = bool(r['max_abs_delta'] > 0)
    return result


def save_checkpoint(path, model, optimizer, stream, protocol, *, arm, step, source_path,
                    source_hash, full_state=True, changes=False, counts=None, replace=False):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError('Refusing checkpoint overwrite: '+str(path))
    started = time.perf_counter()
    state = tree_cpu(model.state_dict()); rng = capture_rng(); ss = stream.state_dict()
    opt = tree_cpu(optimizer.state_dict()) if full_state else None
    m = dict(variant=arm,phase_step=step,conditional_total_step=1000+step,logical_step=1000+step,context_encoder='geo' if arm in ('Geo','Full') else None,
        model_state_sha256=tensor_state_hash(state),protocol_sha256=json_hash(protocol),
        source_checkpoint_sha256=source_hash,source_checkpoint=str(source_path),
        official_file_sha256=OFFICIAL_FILE_SHA256,official_backbone_sha256=OFFICIAL_BACKBONE_SHA256,
        config_sha256=config_sha256(DEFAULT_CONFIG),rng_sha256=tree_hash(rng),
        stream_state_sha256=json_hash(ss),loss_recipe='edge5' if arm=='Full' else 'fm',
        optimizer_state_sha256=tree_hash(opt) if opt is not None else None,
        actual_new_updates=step,
        recovery_mode='RESTORE_ALL',
        full_recovery_state=full_state,optimizer_reset_at_1000=False,
        counts=dict(counts or {}))
    if changes: m['component_changes_from_source1000'] = component_changes(state,source_path)
    payload=dict(schema=SCHEMA,metadata=m,protocol=protocol,model=state,rng=rng,
                 stream_state=ss,update_in_progress=False)
    if opt is not None: payload['optimizer']=opt
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload,temporary)
    with temporary.open('rb') as f: os.fsync(f.fileno())
    os.replace(temporary,path)
    record=dict(path=str(path.resolve()),checkpoint_sha256=digest(path),metadata=m)
    atomic_json(path.with_suffix('.json'),record)
    # Verify persisted tensors and recovery identity, with no model forward.
    loaded=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    validate_payload(loaded)
    if tensor_state_hash(loaded['model'])!=m['model_state_sha256']:
        raise ValueError('Persisted checkpoint model differs')
    record.update(save_and_verify_seconds=time.perf_counter()-started,strict_reload='PASS')
    del loaded,payload,opt,state;gc.collect()
    return record
