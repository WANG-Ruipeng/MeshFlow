"""Session C variable-N original Euler50; no model construction or GT access.

This module is provisional until eval_protocol.json is frozen after DATA_READY.
The caller must acquire the shared GPU lock before invoking any CUDA operation.
"""
from native_t1 import runtime as _runtime  # CUBLAS before CUDA initialization.

import time
import numpy as np
import torch
from native_t1.artifacts import array_hash
from native_t1.model import native_execution, assert_native_runtime


def new_counts():
    return dict(model_forward_attempts=0, model_forward_returns=0,
                backbone_forward_attempts=0, backbone_forward_returns=0,
                euler_updates=0, complete_rollouts=0, rollout_attempts=0,
                teacher_attempts=0, teacher_returns=0,
                model_backward_calls=0, optimizer_updates=0,
                geometry_gradient_calls=0, postprocessing_repairs=0)


class SamplingFailure(RuntimeError):
    def __init__(self, reason, states, audit):
        super().__init__(reason)
        self.partial_states = np.stack(states)
        self.audit = audit


def _validate_model(model):
    if model.experiment_trainable or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Evaluation requires frozen parameters and no stored gradients')
    if any(m.training for m in model.modules()):
        raise RuntimeError('Evaluation requires all modules in eval mode')


@torch.no_grad()
def sample_euler50(model, z, C, counts):
    """Exactly 50 original Euler updates, N actual faces, C fixed at 51 states.

    Inputs are exclusively a model, independent Gaussian, C and call ledger.
    No target, source face identifiers, OT, guidance or repair enters this API.
    """
    _runtime.configure_stable_runtime()
    _validate_model(model)
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[2] != 9:
        raise ValueError('Gaussian must be [1,N,9]')
    N = int(z.shape[1]); C = C.reshape(-1, 9); K = len(C)
    if not 0 < K < N:
        raise ValueError('Require 0<K<N')
    if z.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError('FP32 Gaussian, C and integration required')
    if z.device.type != 'cuda' or z.device != C.device or z.device != next(model.parameters()).device:
        raise ValueError('Inputs and model must share one CUDA device')
    if not bool(torch.isfinite(z).all() and torch.isfinite(C).all()):
        raise ValueError('Finite inputs required')
    initial_counts = dict(counts)
    counts['rollout_attempts'] += 1
    source, context_hash = array_hash(z), array_hash(C)
    state = z.clone(); state[:, :K] = C[None]
    known = torch.arange(N, device=z.device)[None] < K
    valid = torch.ones_like(known)
    y = torch.tensor([N], device=z.device, dtype=torch.long)
    states = [state[0].cpu().numpy().copy()]
    accepted_times = [0.0]; known_checks = [True]
    audit = dict(N=N, K=K, solver='original_Euler50', accepted_times=accepted_times,
                 known_checks=known_checks, noise_sha256=source, C_sha256=context_hash,
                 integration_dtype='float32', forward_dtype='bfloat16',
                 known_raw_velocity_forced_zero=False, geometry_repairs=0,
                 GT_model_inputs=False, initial_state_sha256=array_hash(states[0]))
    def before(module, args):
        if counts['model_forward_attempts'] - initial_counts['model_forward_attempts'] >= 50:
            raise RuntimeError('50 NFE budget exceeded')
        counts['model_forward_attempts'] += 1
    def after(module, args, output):
        counts['model_forward_returns'] += 1
        if output.requires_grad:
            raise RuntimeError('Unexpected autograd graph')
    def backbone_start(module, args):
        counts['backbone_forward_attempts'] += 1
    def backbone_end(module, args, output):
        counts['backbone_forward_returns'] += 1
    handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after),
               model.backbone.x_embedder.register_forward_pre_hook(backbone_start),
               model.backbone.final_layer.linear.register_forward_hook(backbone_end)]
    tick = time.perf_counter()
    try:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with native_execution(model, coordinates=state) as facts:
            audit['runtime_before'] = facts
            for step in range(50):
                if not torch.equal(state[0, :K], C):
                    raise RuntimeError('C changed before forward')
                t = torch.full((1,), step / 50, dtype=torch.float32, device=z.device)
                velocity = model(state, t, y, valid, known).float()
                if velocity.shape != state.shape or not bool(torch.isfinite(velocity).all()):
                    raise RuntimeError('Nonfinite or malformed model velocity')
                next_state = state.clone()
                next_state[:, K:] = state[:, K:] + velocity[:, K:] * (1 / 50)
                next_state[:, :K] = C[None]
                state = next_state; counts['euler_updates'] += 1
                states.append(state[0].cpu().numpy().copy())
                accepted_times.append((step + 1) / 50)
                known_checks.append(torch.equal(state[0, :K], C))
                if not known_checks[-1] or state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError('Invalid clamped Euler state')
            audit['runtime_after'] = assert_native_runtime(model, coordinates=state)
        if array_hash(z) != source or array_hash(C) != context_hash:
            raise RuntimeError('Sampling input mutated')
        path = np.stack(states)
        if any(s[:K].tobytes() != C.cpu().numpy().tobytes() for s in path):
            raise RuntimeError('C bitwise preservation failed')
        for key in ('model_forward_attempts', 'model_forward_returns',
                    'backbone_forward_attempts', 'backbone_forward_returns'):
            if counts[key] - initial_counts[key] != 50:
                raise RuntimeError(f'Expected exactly 50 calls: {key}')
        counts['complete_rollouts'] += 1
        audit.update(status='PASS', known_all51_bitwise=True, known_state_checks=51,
                     free_updated=not np.array_equal(path[0, K:], path[-1, K:]))
        return path[-1].copy(), path, audit
    except BaseException as exc:
        audit.update(status='FAIL', reason=repr(exc))
        raise SamplingFailure(str(exc), states, audit) from exc
    finally:
        for handle in handles:
            handle.remove()
        audit.update(seconds=time.perf_counter()-tick,
                     nfe=counts['backbone_forward_attempts']-initial_counts['backbone_forward_attempts'],
                     counts_delta={k: counts[k]-initial_counts[k] for k in counts},
                     peak_scope='One rollout after reset, includes resident model and autocast cache')
        try:
            torch.cuda.synchronize()
            audit.update(seconds=time.perf_counter()-tick,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as exc:
            audit['final_memory_observation_error'] = repr(exc)
