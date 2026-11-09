"""Native face-context experiment using untouched official v3 submodules.

This is a full-model fine-tuning wrapper, not a frozen-backbone adapter. A caller
must supply an independently loaded experimental model, never the read-only
Base instance. No original forward method, source, or weight file is patched.
"""
from __future__ import annotations

# Establish CUBLAS policy before importing torch in a new entrypoint.
from . import stable_runtime as _stable

from contextlib import contextmanager
from contextvars import ContextVar
import traceback
from typing import Callable

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


NUMERICAL_POLICY = {
    'all_free_bf16_base_atol': 0.0, 'all_free_bf16_base_rtol': 0.0,
    'same_input_repeat_atol': 0.0, 'same_input_repeat_rtol': 0.0,
    'fp32_formula_atol': 1e-5, 'fp32_formula_rtol': 1e-5,
    'bf16_permutation_atol': .015625, 'bf16_permutation_rtol': .03125,
    'source': 'inference_precheck.py numerical_policy; thresholds fixed before execution',
}
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


def _difference(actual, expected, *, atol, rtol):
    a, b = actual.detach().float(), expected.detach().float()
    error = (a - b).abs()
    return dict(status='PASS' if bool(torch.allclose(a, b, atol=atol, rtol=rtol)) else 'FAIL',
                bitwise_equal=torch.equal(actual, expected), atol=atol, rtol=rtol,
                max_abs=float(error.max()), rms=float(error.square().mean().sqrt()),
                actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype))


def _grad_facts(value):
    if value is None:
        return dict(present=False, finite=False, rms=None, max_abs=None)
    v = value.detach().float()
    return dict(present=True, finite=bool(torch.isfinite(v).all()),
                rms=float(v.square().mean().sqrt()), max_abs=float(v.abs().max()))


def run_model_gates(model, x, t, y, valid_mask, known_mask, counters=None):
    """Bounded root-invoked gates on one disposable full-model copy.

    Root-owned hooks count all native and reference model forwards. This helper
    additionally reports local call counts and adds only backward attempts and
    completions to the provided central counters. No optimizer is constructed.
    The caller must discard this gate model and freshly reload EMA for training.
    """
    from .official_wrapper import state_sha256
    report = dict(status='RUNNING', tests=[], numerical_policy=dict(NUMERICAL_POLICY),
                  actual_counts=dict(native_forward_attempts=0, native_forward_returns=0,
                                     reference_forward_attempts=0, reference_forward_returns=0,
                                     backward_attempts=0, backward_completed=0,
                                     optimizer_updates=0, complete_rollouts=0),
                  model_scope='Disposable independent EMA copy; reload before formal training',
                  nonzero_role_scope='Temporary role values on disposable copy, restored before exit')
    initial_hash = state_sha256(model.backbone)
    original_role = model.role_embedding.detach().clone()
    report['initial_backbone_sha256'] = initial_hash
    counts = report['actual_counts']
    handles = []

    def add(name, value):
        report['tests'].append(dict(name=name, **value))
        if value['status'] != 'PASS':
            raise RuntimeError('Native implementation gate failed: ' + name)

    def call(value, mask, *, observer=None, global_time=t, is_reference=False, precision=True):
        prefix = 'reference' if is_reference else 'native'
        counts[prefix + '_forward_attempts'] += 1
        with native_execution(model, coordinates=(value, global_time), autocast=precision):
            if is_reference:
                result = model.backbone(value, global_time, y, mask=valid_mask)
            else:
                result = model(value, global_time, y, valid_mask, mask, observer=observer)
        counts[prefix + '_forward_returns'] += 1
        return result

    try:
        if not model.experiment_trainable or not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Gate model must permit full-model backward')
        if not torch.equal(original_role, torch.zeros_like(original_role)):
            raise RuntimeError('Formal initialization must have exactly zero role embeddings')
        if sorted(known_mask.sum(1).tolist()) != [1, 2]:
            raise RuntimeError('Gates require a real batch of two: one K=1 and one K=2')
        if not bool(((t > 0) & (t < 1)).all()):
            raise RuntimeError('Mixed-time gate requires free times strictly inside (0,1)')
        all_free = torch.zeros_like(known_mask)
        with torch.no_grad():
            native_fp32 = call(x, all_free, precision=False)
            base_fp32 = call(x, all_free, is_reference=True, precision=False)
            add('all_free_zero_role_fp32_matches_original',
                _difference(native_fp32, base_fp32, atol=1e-5, rtol=1e-5))
            del native_fp32, base_fp32
            native_bf16 = call(x, all_free)
            base_bf16 = call(x, all_free, is_reference=True)
            add('all_free_zero_role_bf16_matches_original_without_shortcut',
                _difference(native_bf16, base_bf16, atol=0, rtol=0))
            del native_bf16, base_bf16
            observed = []
            expected_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])

            def observe(event):
                if event['site'] == 'embedding':
                    return
                checks = []
                for chosen, free, known in zip(event['selected_mods'], event['free_mods'], event['known_mods']):
                    free_exp = free[:, None].expand_as(chosen)
                    known_exp = known[:, None].expand_as(chosen)
                    checks.append(torch.equal(chosen[~known_mask], free_exp[~known_mask])
                                  and torch.equal(chosen[known_mask], known_exp[known_mask]))
                observed.append(dict(site=event['site'], layer=event['layer'],
                                     time_exact=torch.equal(event['face_time'], expected_time),
                                     selected_modulations_exact=all(checks),
                                     known_context_valid=bool(valid_mask[known_mask].all()),
                                     free_and_known_modulation_differ=any(
                                         not torch.equal(f, k) for f, k in zip(event['free_mods'], event['known_mods']))))

            mixed = call(x, known_mask, observer=observe)
            repeated = call(x, known_mask)
            add('mixed_time_visits_every_block_and_final_head', dict(
                status='PASS' if len(observed) == len(model.backbone.layers) + 1
                and all(q['time_exact'] and q['selected_modulations_exact']
                        and q['known_context_valid'] and q['free_and_known_modulation_differ'] for q in observed)
                else 'FAIL', sites=observed,
                known_counts=known_mask.sum(1).tolist(),
                valid_counts=valid_mask.sum(1).tolist(),
                free_counts=(valid_mask & ~known_mask).sum(1).tolist()))
            add('same_input_repeated_native_forward_bitwise',
                _difference(mixed, repeated, atol=0, rtol=0))
            order = torch.arange(x.shape[1] - 1, -1, -1, device=x.device)
            face_perm = call(x[:, order], known_mask[:, order])
            add('bf16_face_role_time_permutation',
                _difference(face_perm, mixed[:, order], atol=.015625, rtol=.03125))
            corner_order = torch.tensor([2, 0, 1], device=x.device)
            corner_x = x.reshape(x.shape[0], 112, 3, 3)[:, :, corner_order].reshape_as(x)
            corner_pred = call(corner_x, known_mask)
            corner_expected = mixed.reshape(x.shape[0], 112, 3, 3)[:, :, corner_order].reshape_as(mixed)
            add('bf16_corner_permutation',
                _difference(corner_pred, corner_expected, atol=.015625, rtol=.03125))
            changed_context = x.clone()
            changed_context[known_mask] += .125
            changed_prediction = call(changed_context, known_mask)
            context_change = _difference(changed_prediction[~known_mask], mixed[~known_mask], atol=0, rtol=0)
            add('free_velocity_reads_known_context', dict(
                status='PASS' if not context_change['bitwise_equal'] else 'FAIL', difference=context_change))
            # This is the disposable full-model gate copy, not either formal arm.
            model.role_embedding[0].fill_(.0078125)
            model.role_embedding[1].fill_(-.0078125)
            nonzero_role_prediction = call(x, known_mask)
            role_change = _difference(nonzero_role_prediction, mixed, atol=0, rtol=0)
            add('nonzero_roles_on_disposable_copy_change_output', dict(
                status='PASS' if not role_change['bitwise_equal'] else 'FAIL', difference=role_change))
            model.role_embedding.copy_(original_role)
            del mixed, repeated, face_perm, corner_pred, changed_prediction, nonzero_role_prediction

        model.zero_grad(set_to_none=True)
        grad_input = x.detach().clone().requires_grad_(True)
        retained = {}

        def retain_context(event):
            if event['site'] == 'embedding':
                event['hidden'].retain_grad()
                retained['embedded_corners'] = event['hidden']

        with torch.enable_grad(), native_execution(model, coordinates=(grad_input, t)):
            counts['native_forward_attempts'] += 1
            pred = model(grad_input, t, y, valid_mask, known_mask, observer=retain_context)
            counts['native_forward_returns'] += 1
            pred.retain_grad()
            loss_mask = valid_mask & ~known_mask
            # Zero velocity target is a structural gate, not training data.
            denominator = int(loss_mask.sum()) * 9
            loss = pred.float().square()[loss_mask].sum() / denominator
            counts['backward_attempts'] += 1
            if counters is not None:
                counters['backward_attempts'] = counters.get('backward_attempts', 0) + 1
            loss.backward()
            torch.cuda.synchronize(x.device)
            counts['backward_completed'] += 1
            if counters is not None:
                counters['backward_completed'] = counters.get('backward_completed', 0) + 1
        selected_parameters = {
            'coordinate_embedding': model.backbone.x_embedder.mlp[0].weight,
            'first_attention_qkv': model.backbone.layers[0].attn.qkv_proj.weight,
            'first_attention_output': model.backbone.layers[0].attn.out_proj.weight,
            'time_embedding': model.backbone.t_embedder.mlp[0].weight,
            'final_velocity_head': model.backbone.final_layer.linear.weight,
            'role_embedding': model.role_embedding,
        }
        parameter_grads = {name: _grad_facts(parameter.grad) for name, parameter in selected_parameters.items()}
        embedded_grad = retained['embedded_corners'].grad.reshape(x.shape[0], 112, 3, -1)
        context_grad = _grad_facts(embedded_grad[known_mask])
        coordinate_grad = _grad_facts(grad_input.grad[known_mask])
        prediction_known_grad = pred.grad[known_mask]
        has_signal = lambda q: q['present'] and q['finite'] and q['max_abs'] > 0
        add('free_only_loss_backpropagates_through_known_context_and_backbone', dict(
            status='PASS' if all(has_signal(q) for q in parameter_grads.values())
            and has_signal(context_grad) and has_signal(coordinate_grad)
            and torch.equal(prediction_known_grad, torch.zeros_like(prediction_known_grad)) else 'FAIL',
            free_coordinate_denominator=denominator, free_counts=loss_mask.sum(1).tolist(),
            known_predicted_velocity_gradient_exact_zero=bool((prediction_known_grad == 0).all()),
            known_embedded_context_gradient=context_grad, known_coordinate_gradient=coordinate_grad,
            parameter_gradients=parameter_grads, loss=float(loss.detach())))
        model.zero_grad(set_to_none=True)
        if state_sha256(model.backbone) != initial_hash:
            raise RuntimeError('A zero-update gate changed pretrained backbone tensors')
        if not torch.equal(model.role_embedding, original_role):
            raise RuntimeError('Disposable role test failed to restore zero initialization')
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['backbone_parameter_count'] = sum(p.numel() for p in model.backbone.parameters())
        report['role_parameter_count'] = model.role_embedding.numel()
        report['backbone_unchanged'] = True
        report['role_restored_bitwise'] = True
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            model.role_embedding.copy_(original_role)
        model.zero_grad(set_to_none=True)
        report['final_backbone_sha256'] = state_sha256(model.backbone)
        report['final_role_restored'] = torch.equal(model.role_embedding, original_role)
    return report


def run_cpu_formula_tests():
    """Tiny algebra fixtures only; no official model/CUDA/optimizer execution."""
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU formula checks must precede CUDA initialization')
    saved_rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(2026092831)
        dim, batch, faces = 8, 2, 4

        class AttentionFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, h, mask=None):
                # Deterministic dense context, preserving face equivariance.
                mean = (h * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
                return self.linear(h + mean)

        class MLPFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(dim, 16), nn.SiLU(), nn.Linear(16, dim))

            def forward(self, h, mask=None):
                return self.net(h)

        block = nn.Module()
        block.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        block.norm1, block.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        block.attn, block.mlp = AttentionFixture(), MLPFixture()
        final = nn.Module()
        final.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        final.norm_final, final.linear = nn.LayerNorm(dim), nn.Linear(dim, 3)
        h = torch.randn(batch, faces * 3, dim)
        cf, ck = torch.randn(batch, dim), torch.randn(batch, dim)
        known = torch.tensor([[True, False, False, False], [False, True, True, False]])
        valid = torch.ones_like(known)
        actual = native_block_forward(block, h, cf, ck, known, valid)
        # Independent explicit face/corner loops for the same formula.
        face_conditions = torch.stack([
            torch.stack([ck[b] if known[b, n] else cf[b] for n in range(faces)])
            for b in range(batch)])
        mods = block.adaLN_modulation(face_conditions).chunk(6, -1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mods
        pooled = h.reshape(batch, faces, 3, dim).mean(2)
        attn_input = torch.empty_like(pooled)
        normalized = block.norm1(pooled)
        for b in range(batch):
            for n in range(faces):
                attn_input[b, n] = normalized[b, n] * (1 + scale_a[b, n]) + shift_a[b, n]
        attn_output = block.attn(attn_input, mask=valid)
        after_attn = h.clone()
        for b in range(batch):
            for n in range(faces):
                for corner in range(3):
                    after_attn[b, 3 * n + corner] += gate_a[b, n] * attn_output[b, n]
        normalized = block.norm2(after_attn)
        mlp_input = torch.empty_like(normalized)
        for b in range(batch):
            for n in range(faces):
                for corner in range(3):
                    mlp_input[b, 3 * n + corner] = normalized[b, 3 * n + corner] * (1 + scale_m[b, n]) + shift_m[b, n]
        mlp_output = block.mlp(mlp_input, mask=valid)
        expected = after_attn.clone()
        for b in range(batch):
            for n in range(faces):
                for corner in range(3):
                    expected[b, 3 * n + corner] += gate_m[b, n] * mlp_output[b, 3 * n + corner]
        block_difference = _difference(actual, expected, atol=1e-5, rtol=1e-5)
        assert block_difference['status'] == 'PASS'
        actual_final = native_final_forward(final, actual, cf, ck, known)
        shift, scale = final.adaLN_modulation(face_conditions).chunk(2, -1)
        normalized = final.norm_final(actual)
        head_input = torch.empty_like(normalized)
        for b in range(batch):
            for n in range(faces):
                for corner in range(3):
                    head_input[b, 3 * n + corner] = normalized[b, 3 * n + corner] * (1 + scale[b, n]) + shift[b, n]
        expected_final = final.linear(head_input)
        final_difference = _difference(actual_final, expected_final, atol=1e-5, rtol=1e-5)
        assert final_difference['status'] == 'PASS'
        return dict(status='PASS', block_formula=block_difference, final_formula=final_difference,
                    fixture_block_evaluations=2, fixture_head_evaluations=2,
                    official_model_forwards=0, native_full_model_forwards=0,
                    backward_calls=0, optimizer_updates=0, gpu_calls=0,
                    numerical_policy=dict(NUMERICAL_POLICY),
                    cuda_initialized=torch.cuda.is_initialized())
    finally:
        torch.set_rng_state(saved_rng)
