"""Native face-context experiment using untouched official v3 submodules.

This is a full-model fine-tuning wrapper, not a frozen-backbone adapter. A caller
must supply an independently loaded experimental model, never the read-only
Base instance. No original forward method, source, or weight file is patched.
"""
from __future__ import annotations

# Establish CUBLAS policy before importing torch in a new entrypoint.
from . import runtime as _stable

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
        native_precision='bf16' if _BF16.get() else 'fp32_gate_only',
        all_modules_eval=True, gradient_checkpointing=False,
        trainable_parameters=trainable, all_backbone_parameters_trainable=model.experiment_trainable,
        role_parameters=model.role_embedding.numel(),
        frozen_backbone_guard_used=False,
    )
    return facts


@contextmanager
def native_execution(model, *, coordinates=None, autocast=True):
    """Shared train/teacher/sample policy; preserve the caller's grad mode.

    ``autocast=False`` is reserved for the predefined FP32 implementation gate.
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


class NativeInpaintingModel(nn.Module):
    """Independent EMA copy plus zero 2x768 role table; all weights fine-tune.

    Role 0 is free and role 1 is known. Known coordinates are supplied by the
    caller in x and preserved by the sampler, not by suppressing predicted
    velocities. This model returns velocities for all 112 faces.
    """
    def __init__(self, backbone, trainable=True):
        super().__init__()
        if backbone.version != 3 or backbone.hidden_size != 768:
            raise ValueError('This registered experiment requires the official v3, width-768 chair model')
        if backbone.use_dit_like_pe or not backbone.face_cond:
            raise ValueError('Unexpected official position/face-conditioning configuration')
        if getattr(backbone, '_mf_pre_injection_owner', None) is not None:
            raise RuntimeError('An old Pre wrapper is still attached')
        for name, module in backbone.named_modules():
            if name.startswith('layers') and (module._forward_hooks or module._forward_pre_hooks):
                raise RuntimeError('Remove old submodule hooks before constructing native experiment')
        self.backbone = backbone
        device = next(backbone.parameters()).device
        self.role_embedding = nn.Parameter(torch.zeros((2, 768), dtype=torch.float32, device=device))
        for layer in self.backbone.layers:
            layer.gradient_checkpointing = False
        self.experiment_trainable = bool(trainable)
        self.requires_grad_(self.experiment_trainable)
        self.eval()

    def train(self, mode=True):
        # Eval suppresses label dropout and checkpointing, but does not disable
        # autograd or parameter updates. This is intentional for full tuning.
        return super().train(False)

    def set_trainable(self, value):
        self.experiment_trainable = bool(value)
        self.requires_grad_(self.experiment_trainable)
        return self

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[1:] != (112, 9):
            raise ValueError('Native task requires x[B,112,9]')
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,):
            raise ValueError('Global free time and total face counts must have shape [B]')
        if not bool((y == 112).all()):
            raise ValueError('The face condition is total N=112, never N_free')
        if valid_mask is None or known_mask is None:
            raise ValueError('Separate valid_mask and known_mask are required')
        if (valid_mask.shape != (batch, faces) or known_mask.shape != (batch, faces)
                or valid_mask.dtype != torch.bool or known_mask.dtype != torch.bool):
            raise ValueError('Both masks must be bool[B,112]')
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError('Known faces must be valid attention context')
        if not bool(valid_mask.all()):
            raise ValueError('All 112 faces are valid in this registered experiment')
        _, hidden = self.backbone.x_embedder(x)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(dtype=hidden.dtype)
        hidden = hidden + _corners(role)
        # Two original embedder calls preserve official GEMM dimensions.
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site='embedding', hidden=hidden, role=role,
                          known_mask=known_mask, valid_mask=valid_mask,
                          face_time=face_time, c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(
                block, hidden, c_free, c_known, known_mask, valid_mask,
                observer=observer, layer_index=index, face_time=face_time)
        result = native_final_forward(
            self.backbone.final_layer, hidden, c_free, c_known, known_mask,
            observer=observer, face_time=face_time)
        return result.view(batch, faces, 9)
