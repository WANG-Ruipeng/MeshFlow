"""Single Control MVP48 train/inference model; no historical endpoint fallback."""
from native_t1 import runtime as _runtime  # CUBLAS policy before torch
from pathlib import Path
import hashlib
import json

import torch
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.model import (NativeInpaintingModel, assert_native_runtime,
    native_block_forward, native_final_forward, native_execution, _corners)
from native_t1.context_geometry import attach_context_encoder, inject_context
from native_t1.portable_checkpoint import (_read_config, load_official_initialization,
    config_sha256, OFFICIAL_FILE_SHA256, OFFICIAL_BACKBONE_SHA256)
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.artifacts import file_sha256, state_sha256

SCHEMA = 'control_mvp48_checkpoint_v1'


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class MVPModel(NativeInpaintingModel):
    """Original Native arithmetic, total N128..256 and independent valid mask."""
    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError('Expected FP32 x[B,padded_N,9]')
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError('Expected free t[B] and integer total N[B]')
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError('Nonfinite coordinates/time')
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError('Free time outside [0,1]')
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError('Colocated separate bool[B,padded_N] masks required')
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError('Known faces must be valid')
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError('y must count real faces only, N128..256')
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError('Each sample requires known and free faces')
        _, hidden = self.backbone.x_embedder(x)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(hidden.dtype)
        hidden = hidden + _corners(role)
        if self.context_encoder is not None:
            delta = self.context_encoder(x, known_mask, valid_mask, observer=self.context_observer)
            hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site='embedding', hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        # Raw known velocity is intentionally unconstrained. Integration owns C.
        return output.view(batch, faces, 9)


def initialize_mvp(official_path, *, geo, device='cpu'):
    """Verified official EMA, zero role, independently seeded zero-exit Geo."""
    original, identity = load_official_initialization(official_path, device='cpu')
    model = MVPModel(original.backbone, trainable=True)
    del original
    if geo:
        attach_context_encoder(model, 'geo', seed=1010)
    model.to(device).eval()
    identity.update(context_encoder='geo' if geo else None, feature_statistics_fitted=False,
                    initialization_state_sha256=state_sha256(model), device=str(device),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, identity


def validate_metadata(payload):
    if payload.get('schema') != SCHEMA:
        raise ValueError('Expected Control MVP48 checkpoint schema')
    meta = payload['metadata']
    variant = meta['variant']
    if variant not in ('T1', 'Geo_shared', 'Geo', 'Full'):
        raise ValueError('Invalid variant')
    step = meta['logical_step']
    allowed_steps = {'T1': (500, 1000), 'Geo_shared': (500,), 'Geo': (1000,), 'Full': (1000,)}
    if type(step) is not int or step not in allowed_steps[variant]:
        raise ValueError('Invalid logical endpoint')
    if meta['context_encoder'] != (None if variant == 'T1' else 'geo'):
        raise ValueError('Variant/network identity mismatch')
    if meta['official_file_sha256'] != OFFICIAL_FILE_SHA256 or meta['official_backbone_sha256'] != OFFICIAL_BACKBONE_SHA256:
        raise ValueError('Unapproved initialization lineage')
    if meta['protocol_sha256'] != json_hash(payload['protocol']):
        raise ValueError('Protocol hash mismatch')
    from .train_contract import validate_protocol
    validate_protocol(payload['protocol'])
    if meta['config_sha256'] != config_sha256(DEFAULT_CONFIG):
        raise ValueError('Architecture config mismatch')
    if payload['stream_state']['batch_index'] != step:
        raise ValueError('Data cursor mismatch')
    if payload.get('update_in_progress') is not False:
        raise ValueError('Not a completed update boundary')
    if not all(key in payload for key in ('optimizer', 'rng', 'model')):
        raise ValueError('Incomplete training checkpoint')
    if variant in ('Geo', 'Full') and not meta.get('fork_checkpoint_sha256'):
        raise ValueError('Branch missing exact shared checkpoint identity')
    from .train_io import tree_hash
    if tree_hash(payload['optimizer']) != meta['optimizer_state_sha256']:
        raise ValueError('Optimizer state hash mismatch')
    if tree_hash(payload['rng']) != meta['rng_sha256'] or json_hash(payload['stream_state']) != meta['stream_state_sha256']:
        raise ValueError('Model/data RNG or cursor hash mismatch')
    phases = {'T1': 0, 'Geo_shared': 0, 'Geo': 0, 'Full': 0}
    if variant == 'T1':
        phases['T1'] = step
    else:
        phases.update(T1=1000, Geo_shared=500)
        if variant in ('Geo', 'Full'): phases['Geo'] = 500
        if variant == 'Full': phases['Full'] = 500
    if meta['executed_phase_updates'] != phases or meta['actual_updates_at_save'] != sum(phases.values()):
        raise ValueError('Actual optimizer update accounting differs')
    if meta['loss_recipe'] != ('edge5' if variant == 'Full' else 'fm'):
        raise ValueError('Checkpoint objective identity differs')
    if meta['optimizer_reset_at_phase_start'] != (variant in ('T1', 'Geo_shared')):
        raise ValueError('AdamW reset/restore history differs')
    if not payload['optimizer']['state'] or not payload['optimizer']['param_groups']:
        raise ValueError('Empty trained AdamW state')
    for group in payload['optimizer']['param_groups']:
        if (group['lr'], tuple(group['betas']), group['weight_decay']) != (1e-5, (.9, .95), 0.):
            raise ValueError('Saved optimizer settings differ')
    if any(int(s['step']) != step for s in payload['optimizer']['state'].values()):
        raise ValueError('AdamW update cursor differs from logical step')
    return meta


def load_mvp_checkpoint(path, device):
    """Return (model, actual_metadata, hashes); the only inference model loader.

    Callers must hold the shared GPU lock before requesting a CUDA device.
    Loading on CPU never initializes CUDA. No optimizer or RNG is restored here.
    """
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = validate_metadata(payload)
    config = _read_config(Path(DEFAULT_CONFIG))
    _runtime.configure_stable_runtime()
    with torch.device('meta'):
        model = MVPModel(DiT(**config), trainable=False)
    model.backbone.x_embedder.embed_fn, _ = get_embedder(config['pe_freq'], input_dims=3)
    if meta['context_encoder'] == 'geo':
        # Explicit CPU construction; assign subsequently replaces every tensor.
        from native_t1.context_geometry import ContextGeometryEncoder
        model.context_encoder = ContextGeometryEncoder('geo', seed=1010)
    state = payload['model']
    if any(v.dtype != torch.float32 or v.device.type != 'cpu' or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Checkpoint must contain finite CPU FP32 model tensors')
    model.load_state_dict(state, strict=True, assign=True)
    actual = state_sha256(model)
    if actual != meta['model_state_sha256']:
        raise ValueError('Model tensor hash mismatch')
    digest = file_sha256(path)
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text())
        if recorded['checkpoint_sha256'] != digest or recorded['metadata'] != meta:
            raise ValueError('Checkpoint sidecar identity mismatch')
    actual_meta = dict(meta, strict_load=True, checkpoint=str(path.resolve()),
                       protocol=payload['protocol'], inference_trainable=False)
    del payload
    model.set_trainable(False).eval().to(device)
    return model, actual_meta, dict(checkpoint_sha256=digest, model_state_sha256=actual,
                                   protocol_sha256=meta['protocol_sha256'])


def velocity_mvp(model, x, t, N, valid_mask, known_mask):
    """Unified BF16 production forward; grad mode belongs to caller."""
    with native_execution(model, coordinates=(x, t), autocast=x.device.type == 'cuda'):
        return model(x, t, N, valid_mask, known_mask).float()


def integrate_mvp_step(model, x, t, dt, N, valid_mask, known_mask, context):
    """FP32 Euler step with exact C preservation; no GT labels or repairs."""
    if x.dtype != torch.float32 or context.dtype != torch.float32:
        raise ValueError('FP32 integration and C required')
    x = torch.where(known_mask[..., None], context, x)
    velocity = velocity_mvp(model, x, t, N, valid_mask, known_mask)
    free = valid_mask & ~known_mask
    return torch.where(free[..., None], x + float(dt) * velocity, x)
