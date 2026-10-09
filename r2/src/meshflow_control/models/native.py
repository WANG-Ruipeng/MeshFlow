"""Portable Native T1 + Geo, with optional direct context readout.

Original block/final arithmetic is preserved. A/B/C use this same class.
Source provenance and local additions are in docs/model_source_map.json.
"""
from __future__ import annotations

# Establish CUBLAS policy before importing torch in a new entrypoint.
from .. import runtime as _stable
from ..precision import current_precision, readout_execution_options

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


_DEPTH = ContextVar('native_inpainting_execution_depth', default=0)
_BF16 = ContextVar('native_inpainting_bf16', default=True)


def assert_native_runtime(model, coordinates=None, *, require_context=True):
    """Allow experiment gradients while enforcing eval and stable arithmetic."""
    facts = _stable.assert_stable_runtime(
        model, require_context=False, coordinates=coordinates)
    if any(module.training for module in model.modules()):
        raise RuntimeError('Every native experiment module must remain eval, even during training')
    for name, module in model.named_modules():
        if hasattr(module, 'dropout') and isinstance(module.dropout, (int, float)):
            if float(module.dropout) != 0.0 and name.endswith('attn'):
                raise RuntimeError(f'Nonzero attention dropout is not authorized: {name}')
        if isinstance(module, nn.Dropout) and module.training:
            raise RuntimeError('Active dropout is forbidden')
    parameters = list(model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    expected = sum(parameter.numel() for parameter in parameters) if model.experiment_trainable else 0
    if trainable != expected:
        raise RuntimeError('Experiment must train every backbone and role parameter, or be fully read-only')
    if any(layer.gradient_checkpointing for layer in model.backbone.layers):
        raise RuntimeError('Gradient checkpointing must be disabled on the experimental copy')
    if require_context:
        if _DEPTH.get() < 1:
            raise RuntimeError('Native forward must execute inside native_execution')
        if not (torch.backends.cuda.math_sdp_enabled()
                and not torch.backends.cuda.flash_sdp_enabled()
                and not torch.backends.cuda.mem_efficient_sdp_enabled()
                and not torch.backends.cuda.cudnn_sdp_enabled()):
            raise RuntimeError('Native execution requires SDPA math only')
        if bool(torch.is_autocast_enabled('cuda')) != _BF16.get():
            raise RuntimeError('Actual autocast state differs from the declared native precision')
        if _BF16.get() and torch.get_autocast_dtype('cuda') != torch.bfloat16:
            raise RuntimeError('Native production autocast must be BF16')
    facts.update(
        native_active_context_depth=_DEPTH.get(),
        native_precision=current_precision() or ('bf16' if _BF16.get() else 'fp32_gate_only'),
        all_modules_eval=True, gradient_checkpointing=False,
        trainable_parameters=trainable, all_backbone_parameters_trainable=model.experiment_trainable,
        role_parameters=model.role_embedding.numel(),
        frozen_backbone_guard_used=False,
    )
    return facts


@contextmanager
def native_execution(model, *, coordinates=None, autocast=True):
    """Shared train/teacher/sample policy; preserve the caller's grad mode.

    ``autocast=False`` supports the predefined FP32 gate and the explicitly
    selected FP32_REFERENCE fixed-weight precision context.
    The context does not use no_grad, alter requires_grad, or catch unsupported
    deterministic operations. Each training microbatch should use a fresh
    context; never keep an autocast weight cache across optimizer updates.
    """
    _stable.configure_stable_runtime()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast(
        'cuda', dtype=torch.bfloat16, enabled=bool(autocast)
    ):
        depth_token = _DEPTH.set(_DEPTH.get() + 1)
        precision_token = _BF16.set(bool(autocast))
        try:
            yield assert_native_runtime(model, coordinates=coordinates)
            assert_native_runtime(model, coordinates=coordinates)
        finally:
            _BF16.reset(precision_token)
            _DEPTH.reset(depth_token)


def _face_select(free, known, known_mask):
    return torch.where(known_mask[..., None], known[:, None, :], free[:, None, :])


def _corners(value):
    batch, faces, dim = value.shape
    return value.unsqueeze(2).repeat(1, 1, 3, 1).reshape(batch, 3 * faces, dim)


def native_block_forward(block, hidden, c_free, c_known, known_mask,
                         valid_mask, *, observer=None, layer_index=None,
                         face_time=None):
    """Official v3 operations, with two time modulations selected per face."""
    batch, corners, dim = hidden.shape
    free_mods = block.adaLN_modulation(c_free).chunk(6, dim=1)
    known_mods = block.adaLN_modulation(c_known).chunk(6, dim=1)
    mods = tuple(_face_select(f, k, known_mask) for f, k in zip(free_mods, known_mods))
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mods
    if observer is not None:
        observer(dict(site='block', layer=layer_index, face_time=face_time,
                      known_mask=known_mask, valid_mask=valid_mask,
                      free_mods=free_mods, known_mods=known_mods,
                      selected_mods=mods, hidden=hidden))
    face_hidden = hidden.reshape(batch, corners // 3, 3, dim).mean(dim=2)
    face_hidden = gate_a * block.attn(
        block.norm1(face_hidden) * (1 + scale_a) + shift_a,
        mask=valid_mask)
    hidden = hidden + _corners(face_hidden)
    hidden = hidden + _corners(gate_m) * block.mlp(
        block.norm2(hidden) * (1 + _corners(scale_m)) + _corners(shift_m),
        mask=valid_mask)
    return hidden


def native_final_forward(final_layer, hidden, c_free, c_known,
                         known_mask, *, observer=None, face_time=None):
    free_mods = final_layer.adaLN_modulation(c_free).chunk(2, dim=1)
    known_mods = final_layer.adaLN_modulation(c_known).chunk(2, dim=1)
    mods = tuple(_face_select(f, k, known_mask) for f, k in zip(free_mods, known_mods))
    shift, scale = mods
    if observer is not None:
        observer(dict(site='final', layer='final', face_time=face_time,
                      known_mask=known_mask, free_mods=free_mods,
                      known_mods=known_mods, selected_mods=mods, hidden=hidden))
    hidden = final_layer.norm_final(hidden) * (1 + _corners(scale)) + _corners(shift)
    return final_layer.linear(hidden)


from .geometry import ContextGeometryEncoder, inject_context
from .readout import (READOUT_LAYERS, READOUT_DIM, READOUT_HEADS,
    READOUT_PARAMETERS, make_readouts, gather_known_corners, apply_readout)
from .backbone.equidit import DiT


class NativeModel(nn.Module):
    """Variable N128..256 velocity model; exact C clamping belongs to sampling.

    CLEAN gathers H0 once after coordinate + role + Geo and before time/blocks.
    The copies preserve autograd and are never mutated or cached across calls.
    """
    def __init__(self, backbone, *, mode="none", readout_mode=None,
                 readout_seed=20261007, trainable=True):
        super().__init__()
        if readout_mode is not None:
            if mode != "none" and mode != readout_mode:
                raise ValueError("Conflicting readout mode arguments")
            mode = readout_mode
        if mode not in ("none", "mixed", "clean"):
            raise ValueError("Readout mode must be none, mixed, or clean")
        if backbone.version != 3 or backbone.hidden_size != 768 or len(backbone.layers) != 12:
            raise ValueError("Registered model requires v3, D768, depth12")
        if backbone.use_dit_like_pe or not backbone.face_cond:
            raise ValueError("Unexpected position/face conditioning")
        if any(m._forward_hooks or m._forward_pre_hooks for m in backbone.modules()):
            raise ValueError("Build a clean backbone without hidden injection hooks")
        self.backbone = backbone
        device = next(backbone.parameters()).device
        self.role_embedding = nn.Parameter(torch.zeros((2, 768), dtype=torch.float32, device=device))
        self.context_encoder = ContextGeometryEncoder("geo", seed=1010)
        self.context_observer = None
        self.readout_mode = mode
        self.readout_seed = readout_seed
        self.readout_observer = None
        self.readouts = nn.ModuleDict() if mode == "none" else make_readouts(readout_seed)
        for layer in self.backbone.layers:
            layer.gradient_checkpointing = False
        self.experiment_trainable = bool(trainable)
        self.requires_grad_(self.experiment_trainable)
        self.eval()

    def train(self, mode=True):
        # Match START: eval disables dropout; all intended gradients remain on.
        return super().train(False)

    def set_trainable(self, value):
        self.experiment_trainable = bool(value)
        self.requires_grad_(self.experiment_trainable)
        return self

    def readout_metadata(self):
        return dict(mode=self.readout_mode, layers=list(READOUT_LAYERS),
            hidden_dimension=768, readout_dimension=READOUT_DIM, heads=READOUT_HEADS,
            head_dimension=48, branch_dtype="float32", output_initialization="zeros",
            qkv_initialization="xavier_uniform", initialization_seed=self.readout_seed,
            bias=False, layernorm_affine=False, layernorm_eps=1e-5,
            extra_parameters=sum(p.numel() for p in self.readouts.parameters()),
            expected_extra_parameters=0 if self.readout_mode == "none" else READOUT_PARAMETERS,
            extra_parameter_names=[n for n,_ in self.named_parameters() if n.startswith("readouts.")],
            clean_memory_detached=False, cross_forward_cache=False,
            coordinate_sorting_in_readout=False, corner_instances_deduplicated=False)

    def _embed(self, x, known_mask, valid_mask):
        batch, faces, _ = x.shape
        _, hidden = self.backbone.x_embedder(x)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(hidden.dtype)
        hidden = hidden + _corners(role)
        if self.context_encoder is not None:
            delta = self.context_encoder(x, known_mask, valid_mask, observer=self.context_observer)
            hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)
        return hidden, role

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected free t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Free time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated separate bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError("y must count real faces only, N128..256")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        hidden, role = self._embed(x, known_mask, valid_mask)
        memory = gather_known_corners(hidden, known_mask, valid_mask) if self.readout_mode == "clean" else None
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
            if memory is not None:
                observer(dict(site="clean_memory", memory=memory))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
            if str(index) in self.readouts:
                hidden = apply_readout(hidden, known_mask, valid_mask, self.readouts[str(index)],
                    memory=memory, observer=self.readout_observer, layer=index,
                    source_mode=self.readout_mode, **readout_execution_options())
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)


def build_model(model_config, readout_mode="none", readout_seed=20261007, trainable=True):
    """Construct only; weights must be loaded strictly by the checkpoint owner.

    Safe under torch.device('meta'); Fourier constants and new branch weights
    explicitly live on CPU until complete state assignment and model.to(device).
    No CUDA initialization, download, old-workspace import, or model forward.
    """
    config = dict(model_config)
    kind = config.pop("model_type", "equidit")
    required = dict(version=3, hidden_dim=768, num_layers=12, num_heads=12,
        max_length=800, face_bin=20, pe_freq=20, use_qknorm=True,
        use_rmsnorm=True, use_coord_encoding=True, use_dit_like_pe=False,
        face_cond=True, gradient_checkpointing=False, mixed_precision="bf16")
    if kind != "equidit" or any(config.get(k) != v for k,v in required.items()):
        raise ValueError("Model config differs from the retained START architecture")
    with torch.random.fork_rng(devices=[]):
        model = NativeModel(DiT(**config), mode=readout_mode,
            readout_seed=readout_seed, trainable=trainable)
    return model
