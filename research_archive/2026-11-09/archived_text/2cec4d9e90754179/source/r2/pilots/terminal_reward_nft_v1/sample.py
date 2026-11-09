"""Single-request deterministic Euler50 for an externally owned NFT policy.

No model construction, hidden noise draw, POST, retry, or worker dispatch.
The integration statements match the maintained sampler. A distinct runtime
context permits the pilot's LoRA trainability contract while using no_grad.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from pathlib import Path
import time
import traceback
import numpy as np

from . import data as D

VERSION = 'terminal_reward_nft_euler50_v1'


def verify_receipt(q, receipt, *, model_identity=None):
    """CPU artifact checks; never executes a model or repairs geometry."""
    if isinstance(receipt, (str, Path)):
        receipt = D.read(receipt)
    if receipt.get('status') != 'COMPLETE' or receipt['input_sha256'] != q['input_sha256']:
        raise ValueError('Incomplete or mismatched RAW receipt')
    if model_identity is not None and receipt['model_identity'] != D.clean(model_identity):
        raise ValueError('RAW policy identity mismatch')
    for key in ('raw', 'ply', 'result'):
        if D.sha(receipt[key+'_path']) != receipt[key+'_sha256']:
            raise ValueError('RAW artifact changed: ' + key)
    b = D.load_request(q)
    a = D.arrays(receipt['raw_path'])
    n, k = q['N'], q['K']
    raw = a['raw'].reshape(n, 3, 3)
    path = a['trajectory'].reshape(51, n, 3, 3)
    if raw.dtype != np.float32 or path.dtype != np.float32 or not np.isfinite(path).all():
        raise ValueError('RAW dtype/finite contract')
    facts = dict(C_output_bitwise=D.same(raw[:k], b['C']),
                 C_all51_bitwise=all(D.same(x[:k], b['C']) for x in path),
                 actual_Gaussian_bitwise=D.same(a['z'], b['z']),
                 NPZ_C_bitwise=D.same(a['C'], b['C']),
                 trajectory_final_is_RAW=D.same(path[-1], raw),
                 initial_free_is_Gaussian=D.same(path[0, k:].reshape(1, n-k, 9), b['z'][:, k:]),
                 free_array_matches_raw=D.same(a['free'], raw[k:]),
                 N_preserved=len(raw) == n)
    # This IO helper imports only NumPy; it does not import a model.
    from meshflow_control.mesh_io import read_ply
    ply, _ = read_ply(D.resolve(receipt['ply_path']))
    facts['PLY_full_bitwise'] = D.same(ply, raw)
    runtime = D.read(receipt['result_path'])
    facts['runtime_Euler50'] = runtime['nfe'] == 50 and runtime['known_state_checks'] == 51
    if not all(facts.values()):
        raise ValueError('RAW contract failed: ' + str(facts))
    return facts


def sample_request(model, q, budget, output_dir, *, label, model_identity, runtime_context=None):
    """Run or verify one frozen request, preserving all 51 FP32 states.

    ``budget`` owns reserve(kind, context), call(kind, fn, context), and
    event(name, **fields). The caller owns policy activation and resource checks.
    A completed matching receipt is read-only reuse. Any incomplete directory
    blocks replay: recovery authorization belongs to the experiment controller.
    ``runtime_context`` has signature (model, coordinates=..., autocast=True).
    """
    dest = D.checked(output_dir)
    receipt_path = dest/'receipt.json'
    if receipt_path.exists():
        r = D.read(receipt_path)
        if r['label'] != label or r['request_id'] != q['request_id']:
            raise ValueError('Completed cell identity mismatch')
        verify_receipt(q, r, model_identity=model_identity)
        return r
    if dest.exists():
        raise RuntimeError('Unresolved rollout directory; no silent replay: ' + str(dest))
    b = D.load_request(q)
    n, k = q['N'], q['K']
    import torch
    if runtime_context is None:
        from .model import nft_execution
        runtime_context = nft_execution
    from meshflow_control.mesh_io import write_raw
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Actual Euler50 requires explicitly owned CUDA')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Clear parameter gradients before sampling')
    ctx = dict(label=label, request_id=q['request_id'], input_sha256=q['input_sha256'],
               noise_seed=q['noise_seed'], model_identity=D.clean(model_identity))
    budget.reserve('rollout_attempt', context=ctx)
    dest.mkdir(parents=True, exist_ok=False)
    D.write(dest/'attempt.json', dict(status='STARTED', version=VERSION, **ctx))
    try:
        state = torch.from_numpy(b['z'].copy()).to(device)
        known = torch.zeros((1, n), dtype=torch.bool, device=device)
        known[:, :k] = True
        valid = torch.ones_like(known)
        y = torch.tensor([n], device=device, dtype=torch.long)
        cond = torch.from_numpy(b['C'].reshape(k, 9).copy()).to(device)
        state[:, :k] = cond[None]
        path = [state[0].cpu().numpy().copy()]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        with torch.no_grad(), runtime_context(model, coordinates=state, autocast=True):
            for step in range(50):
                t = torch.full((1,), step/50, dtype=torch.float32, device=device)
                velocity = budget.call('sampling_forward', lambda: model(state, t, y, valid, known),
                                       context={**ctx, 'euler_step': step})
                if velocity.dtype != torch.bfloat16 or velocity.shape != state.shape or not torch.isfinite(velocity).all():
                    raise RuntimeError('Invalid BF16 velocity')
                next_state = state.clone()
                next_state[:, k:] = state[:, k:] + velocity.float()[:, k:] * (1/50)
                next_state[:, :k] = cond[None]
                state = next_state
                if not torch.isfinite(state).all() or not torch.equal(state[0, :k], cond):
                    raise RuntimeError('Euler/C contract failed')
                path.append(state[0].cpu().numpy().copy())
        torch.cuda.synchronize()
        raw = state[0].cpu().numpy()
        trajectory = np.stack(path)
        if any(x[:k].tobytes() != b['C'].tobytes() for x in trajectory):
            raise RuntimeError('C bit pattern not preserved')
        audit = dict(version=VERSION, nfe=50, known_all51_bitwise=True,
                     known_coordinates_bitwise=True, known_state_checks=51,
                     N=n, K=k, seed=q['noise_seed'], seconds=time.perf_counter()-tick,
                     readout_mode='none', model_inputs=['C', 'N', 'Gaussian', 'time'],
                     dtype='BF16 backbone/FP32 parameters+integration; separately registered LoRA runtime',
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     peak_scope='One sampling rollout, resident policy weights included; not full process VRAM',
                     human_usability='NOT_RUN', request=ctx, sampling_source_sha256=D.sha(__file__))
        budget.event('ROLLOUT_COMPLETE', context=ctx, nfe=50, seconds=audit['seconds'], known_all51_bitwise=True)
        write_raw(dest, raw, b['C'], b['z'], trajectory, audit)
        r = dict(q)
        r.update(status='COMPLETE', label=label, model_identity=D.clean(model_identity),
                 new_rollouts=1, new_sampling_F=50, generation_status='NEW_WORLD_FRAME_EULER50',
                 sampling=audit, sample_code_sha256=D.sha(__file__))
        for key, name in (('raw', 'raw.npz'), ('ply', 'raw.ply'), ('result', 'run.json')):
            r[key+'_path'] = str(dest/name)
            r[key+'_sha256'] = D.sha(dest/name)
        r['contracts'] = verify_receipt(q, r, model_identity=model_identity)
        D.write(receipt_path, r)
        return r
    except BaseException as exc:
        D.write(dest/'failure.json', dict(status='FAILED_NO_AUTO_REPLAY', error=repr(exc),
                                        traceback=traceback.format_exc(), **ctx))
        raise
