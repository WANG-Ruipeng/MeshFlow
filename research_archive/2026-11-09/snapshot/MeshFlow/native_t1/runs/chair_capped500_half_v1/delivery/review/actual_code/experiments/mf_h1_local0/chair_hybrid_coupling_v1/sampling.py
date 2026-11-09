"""Frozen-input Euler50, owned by the pilot worker's lock and durable budget.

The update expressions are copied without reordering from the original
control_mvp48.eval_runtime.sample_euler50. No target is passed to a model call.
This module has no model loader, GPU lock, or command-line execution path.
"""
from __future__ import annotations

import time
import numpy as np


def array_digest(value):
    from .common import array_hash
    return array_hash(value)


def validate_bank(bank):
    """Read only inference inputs; validate the original contiguous C layout."""
    z = np.asarray(bank['z'])
    C = np.asarray(bank['C']).reshape(-1, 9)
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] != 9:
        raise ValueError('Original Gaussian must be [1,N,9]')
    N, K = z.shape[1], len(C)
    if z.dtype != np.float32 or C.dtype != np.float32 or not 0 < K < N:
        raise ValueError('Original finite FP32 Gaussian and 0<K<N C required')
    if not np.isfinite(z).all() or not np.isfinite(C).all():
        raise ValueError('Nonfinite inference input')
    known = np.arange(N) < K
    if 'known_mask' in bank:
        original = np.asarray(bank['known_mask'])
        if original.dtype != np.bool_ or not np.array_equal(original.reshape(-1), known):
            raise ValueError('Frozen known mask does not match original C-first slots')
    if 'valid_mask' in bank and not np.asarray(bank['valid_mask']).all():
        raise ValueError('Formal natural-N requests must have no padding')
    if 'N' in bank and int(np.asarray(bank['N'])) != N:
        raise ValueError('Frozen N differs from Gaussian')
    return z, C, known


def sample_case(model, bank, save_trajectory, budget, context=None):
    """Return raw, optional 51 states/50 actual velocities, and audit metadata.

    ``budget.call('sampling_forward', model, ..., context=...)`` reserves each
    actual attempted forward before it executes. The caller owns request-level
    accounting, frozen bank verification, persistence, and the GPU lock.
    """
    from native_t1 import runtime
    import torch
    from native_t1.model import native_execution, assert_native_runtime
    from experiments.mf_h1_local0.control_mvp48.eval_runtime import _validate_model

    runtime.configure_stable_runtime()
    _validate_model(model)
    z_np, C_np, known_np = validate_bank(bank)
    source_hash, condition_hash = array_digest(z_np), array_digest(C_np)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Real sampling requires a CUDA model under the worker lock')
    z = torch.from_numpy(z_np.copy()).to(device)
    C = torch.from_numpy(C_np.copy()).to(device)
    N, K = int(z.shape[1]), len(C)
    state = z.clone(); state[:, :K] = C[None]
    known = torch.from_numpy(known_np.copy())[None].to(device)
    valid = torch.ones_like(known)
    y = torch.tensor([N], device=device, dtype=torch.long)
    states = [state[0].cpu().numpy().copy()] if save_trajectory else []
    velocities = []
    known_checks = [True]
    raw_dtypes = set()
    context = dict(context or {})
    forward_count = 0
    backbone_count = 0
    geometry_count = 0
    tick = time.perf_counter()

    def observed_backbone(*unused):
        nonlocal backbone_count
        backbone_count += 1

    def observed_geometry(module, args):
        nonlocal geometry_count
        geometry_count += 1
        if not torch.equal(args[0][0, :K], C):
            raise RuntimeError('Geo did not receive the exact current condition')

    if getattr(model, 'context_encoder', None) is None:
        raise RuntimeError('The registered Native+Geo model must have Geo')
    handles = [model.backbone.x_embedder.register_forward_pre_hook(observed_backbone),
               model.context_encoder.register_forward_pre_hook(observed_geometry)]
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), native_execution(model, coordinates=state) as facts:
            runtime_before = facts
            for step in range(50):
                if not torch.equal(state[0, :K], C):
                    raise RuntimeError('C changed before forward')
                t = torch.full((1,), step / 50, dtype=torch.float32, device=device)
                returned = budget.call('sampling_forward', model, state, t, y, valid, known,
                                       context=dict(context, euler_step=step))
                forward_count += 1
                raw_dtypes.add(str(returned.dtype))
                velocity = returned.float()
                if velocity.shape != state.shape or not bool(torch.isfinite(velocity).all()):
                    raise RuntimeError('Nonfinite or malformed model velocity')
                if save_trajectory:
                    velocities.append(velocity[0].cpu().numpy().copy())
                # Preserve the original arithmetic, including multiplication order.
                next_state = state.clone()
                next_state[:, K:] = state[:, K:] + velocity[:, K:] * (1 / 50)
                next_state[:, :K] = C[None]
                state = next_state
                known_checks.append(torch.equal(state[0, :K], C))
                if not known_checks[-1] or state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError('Invalid clamped Euler state')
                if save_trajectory:
                    states.append(state[0].cpu().numpy().copy())
            runtime_after = assert_native_runtime(model, coordinates=state)
        torch.cuda.synchronize()
        if forward_count != 50 or backbone_count != 50 or geometry_count != 50:
            raise RuntimeError('Expected exactly 50 model, backbone and Geo calls')
        if raw_dtypes != {'torch.bfloat16'}:
            raise RuntimeError('Formal production output was not actual BF16')
        if array_digest(z_np) != source_hash or array_digest(C_np) != condition_hash:
            raise RuntimeError('Sampling mutated its frozen numpy input')
        if not torch.equal(z, torch.from_numpy(z_np).to(device)) or not torch.equal(C, torch.from_numpy(C_np).to(device)):
            raise RuntimeError('Sampling mutated its GPU input')
        path = np.stack(states) if save_trajectory else None
        velocity_path = np.stack(velocities) if save_trajectory else None
        if path is not None and any(s[:K].tobytes() != C_np.tobytes() for s in path):
            raise RuntimeError('C was not bitwise preserved in all 51 stored states')
        raw = state[0].cpu().numpy().copy()
        metadata = dict(status='PASS', N=N, K=K, solver='original_Euler50', nfe=50,
                        accepted_states=51, integration_dtype='float32', forward_dtype='bfloat16',
                        observed_raw_output_dtypes=sorted(raw_dtypes),
                        known_all51_bitwise=all(known_checks), known_state_checks=51,
                        actual_backbone_calls=backbone_count, actual_Geo_recomputations=geometry_count,
                        known_raw_velocity_forced_zero=False, GT_model_inputs=False,
                        initial_gaussian_scaled_by_alpha=False, geometry_repairs=0,
                        free_updated=not np.array_equal(z_np[0, K:], raw[K:]),
                        noise_sha256=source_hash, condition_sha256=condition_hash,
                        runtime_before=runtime_before, runtime_after=runtime_after,
                        seconds=time.perf_counter()-tick,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope='One rollout after reset, includes resident model and autocast cache')
        return dict(raw=raw, states=path, velocities=velocity_path, metadata=metadata)
    except BaseException as exc:
        exc.partial_sampling = dict(states=np.stack(states) if states else None,
                                    velocities=np.stack(velocities) if velocities else None,
                                    returned_forwards=forward_count, known_checks=known_checks)
        raise
    finally:
        for handle in handles:
            handle.remove()
