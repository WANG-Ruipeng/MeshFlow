"""Window-local Gaussian policy; original FULL base and arithmetic stay frozen.

Euler integration is FP32. Gaussian log densities and KL use FP64 full-coordinate
sums, with one result per trajectory. Neither coordinates nor window steps are
averaged. Saved actions are observations, never reparameterized current actions.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from pathlib import Path

from meshflow_control import runtime as _stable  # Establish CUBLAS before torch.
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash
from meshflow_control.models.native import build_model, native_execution, assert_native_runtime
from r2.pilots.terminal_reward_nft_v1.model import (
    LoRAPolicy, AdapterLinear, nft_execution, file_sha256,
    load_policy as _load_lora, EXPECTED_TRAINABLE, RANK, ALPHA,
)

VERSION = 'window_policy_rl_v1'
ADAPTER_SCHEMA = 'r2_window_policy_rl_adapter_v1'
STEPS = 50
DT = 1.0 / STEPS
WINDOW = tuple(range(10, 30))
SIGMAS = (0.0, 0.005, 0.01, 0.02)


def load_policy(generator_path, expected_sha256=None, device='cpu', init_seed=2026101501):
    """Fresh zero-output rank4 LoRA, never an old NFT/SELF adapter."""
    return _load_lora(generator_path, expected_sha256, device, init_seed)


def load_reference(generator_path, expected_sha256=None, device='cpu', init_seed=2026101501):
    """Calibration base only: no LoRA is constructed or initialized."""
    path = Path(generator_path)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('FULL_S6000 file hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload.get('schema') != 'r2_full_seat_learning_curve_generator_v1'
            or payload.get('local_step') != 6000 or payload.get('global_step') != 12000
            or payload.get('EMA') is not False or payload.get('model_config') != MODEL_CONFIG):
        raise ValueError('Expected live FULL_S6000 generator export')
    state = payload['generator_state']
    digest_state = state_hash(state)
    if digest_state != payload['generator_state_sha256']:
        raise ValueError('FULL internal state hash differs')
    if len(state) != 194 or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('FULL must contain original finite FP32 state')
    with torch.device('meta'):
        model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode='none', trainable=False)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    model.base_file_sha256, model.base_state_sha256 = digest, digest_state
    _stable.configure_stable_runtime()
    assert_native_runtime(model, require_context=False)
    if state_hash(model.state_dict()) != digest_state:
        raise ValueError('Reference construction changed FULL state')
    return model


def load_adapter_checkpoint(path, model, *, expected_sha256=None, registration_sha256=None):
    """Load this experiment's completed factors only; never optimizer or RNG."""
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError('Adapter file hash mismatch')
    p = torch.load(path, map_location='cpu', weights_only=True)
    if p.get('schema') != ADAPTER_SCHEMA or p.get('update_in_progress') is not False:
        raise ValueError('Expected completed window-policy adapter')
    if p.get('model_metadata') != model.metadata():
        raise ValueError('Adapter base/runtime identity mismatch')
    if p.get('base_file_sha256') != model.base_file_sha256:
        raise ValueError('Adapter source base differs')
    if registration_sha256 is not None and p.get('registration_sha256') != registration_sha256:
        raise ValueError('Adapter registration differs')
    if state_hash(p['adapters']) != p['adapters_sha256']:
        raise ValueError('Adapter tensor hash mismatch')
    model.load_adapter_state_dict(p['adapters'])
    return p


@contextmanager
def execution(model, *, coordinates=None, autocast=True):
    context = nft_execution if isinstance(model, LoRAPolicy) else native_execution
    with context(model, coordinates=coordinates, autocast=autocast) as facts:
        yield facts


@contextmanager
def activate(model, branch):
    if branch not in ('reference', 'current', 'old'):
        raise ValueError('Unknown branch')
    if isinstance(model, LoRAPolicy):
        with model.activate(branch):
            yield model
    else:
        if branch != 'reference' or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Native calibration model must be entirely frozen reference')
        yield model


def free_faces(valid_mask, known_mask, state):
    if state.ndim != 3 or state.shape[-1] != 9 or state.dtype != torch.float32:
        raise ValueError('Expected FP32 state[B,N,9]')
    for mask in (valid_mask, known_mask):
        if mask.dtype != torch.bool or mask.shape != state.shape[:2] or mask.device != state.device:
            raise ValueError('Colocated bool[B,N] masks required')
    if bool((known_mask & ~valid_mask).any()):
        raise ValueError('Known faces must be valid')
    free = valid_mask & ~known_mask
    if bool((free.sum(1) == 0).any()):
        raise ValueError('Every trajectory requires at least one valid free face')
    return free


def euler_mean(state, velocity, valid_mask, known_mask):
    """The original FP32 Euler statement, preserving C and any padding exactly."""
    if state.requires_grad:
        raise ValueError('Policy states must be detached rollout observations')
    free = free_faces(valid_mask, known_mask, state)
    if velocity.shape != state.shape or velocity.device != state.device:
        raise ValueError('Velocity shape/device differs')
    if not bool(torch.isfinite(state).all() and torch.isfinite(velocity).all()):
        raise ValueError('Nonfinite state/velocity')
    return torch.where(free[..., None], state + velocity.float() * DT, state)


def transition_mean(model, state, step, y, valid_mask, known_mask, *,
                    model_mode='current', forward_call=None, autocast=True):
    """One owned forward; outside W the adapter arithmetic is physically bypassed.

    forward_call, if supplied, receives a zero-argument callable and must return
    that one model result. It is the caller's call/attempt ledger hook.
    """
    if int(step) != step or not 0 <= int(step) < STEPS:
        raise ValueError('Euler step outside [0,49]')
    if model_mode not in ('reference', 'current'):
        raise ValueError('Unknown policy mode')
    if state.requires_grad:
        raise ValueError('Rollout buffer state must be detached')
    t = torch.full((state.shape[0],), int(step) / STEPS, device=state.device, dtype=torch.float32)
    branch = model_mode if int(step) in WINDOW else 'reference'
    with activate(model, branch), execution(model, coordinates=(state, t), autocast=autocast):
        call = lambda: model(state, t, y, valid_mask, known_mask)
        velocity = call() if forward_call is None else forward_call(call)
        if autocast and velocity.dtype != torch.bfloat16:
            raise RuntimeError('Production model forward must return BF16 velocity')
        return euler_mean(state, velocity, valid_mask, known_mask)


def apply_noise(mean, epsilon, sigma, free_mask):
    """FP32 independent state noise on every valid free coordinate; no RNG draw."""
    sigma = float(sigma)
    if sigma not in SIGMAS:
        raise ValueError('Unregistered state noise sigma')
    if mean.dtype != torch.float32 or epsilon.dtype != torch.float32 or mean.shape != epsilon.shape:
        raise ValueError('Mean and frozen epsilon must be same-shape FP32')
    _density_inputs(mean, mean, free_mask)
    if epsilon.device != mean.device or not bool(torch.isfinite(epsilon).all()):
        raise ValueError('Epsilon device/finite contract')
    if bool((epsilon[~free_mask] != 0).any()):
        raise ValueError('Frozen noise outside free coordinates must be zero')
    if sigma == 0:
        return mean.clone()
    return torch.where(free_mask[..., None], mean + epsilon * sigma, mean)


def _density_inputs(a, b, free_mask):
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 9:
        raise ValueError('Density coordinates must share [B,N,9] shape')
    if a.device != b.device or a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError('Density observations and means must be colocated FP32')
    if free_mask.shape != a.shape[:2] or free_mask.dtype != torch.bool or free_mask.device != a.device:
        raise ValueError('Density requires colocated bool free-face mask')
    if bool((free_mask.sum(1) == 0).any()) or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Density empty free set or nonfinite coordinates')


def _sigma_tensor(sigma, reference):
    value = torch.as_tensor(sigma, dtype=torch.float64, device=reference.device)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],) or not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError('Gaussian density requires finite positive sigma per trajectory')
    if value.requires_grad:
        raise ValueError('State covariance is fixed, not learned')
    return value


def joint_log_prob(action, mean, sigma, free_mask):
    """FP64[B] full joint Gaussian logpdf. Saved action MUST be detached."""
    if action.requires_grad:
        raise ValueError('Action must be the detached actual rollout transition')
    _density_inputs(action, mean, free_mask)
    s = _sigma_tensor(sigma, mean)
    residual = action.double() - mean.double()
    squared = torch.where(free_mask[..., None], residual.square(), 0.0).sum((1, 2))
    d = free_mask.sum(1).double() * 9
    return -squared / (2 * s.square()) - d * s.log() - 0.5 * d * math.log(2 * math.pi)


def analytic_kl(old_mean, new_mean, sigma, free_mask):
    """KL(old||new) for fixed equal diagonal covariance, full-coordinate sum."""
    if old_mean.requires_grad:
        raise ValueError('Old mean must be frozen rollout data')
    _density_inputs(old_mean, new_mean, free_mask)
    s = _sigma_tensor(sigma, new_mean)
    delta = new_mean.double() - old_mean.double()
    squared = torch.where(free_mask[..., None], delta.square(), 0.0).sum((1, 2))
    return squared / (2 * s.square())
