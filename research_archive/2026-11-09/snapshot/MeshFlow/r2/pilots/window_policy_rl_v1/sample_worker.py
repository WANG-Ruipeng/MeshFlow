"""One explicitly dispatched job of Euler50/window-noise rollouts.

No dispatch, resampling, hidden validation forward, POST, or automatic retry.
Initial noise and all 20 transition arrays are frozen before model loading.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from .bootstrap import bind, ROOT, OUT
D, E = bind()
sys.path.insert(0, str(ROOT / 'r2/src'))

VERSION = 'window_policy_rl_sampler_v1'
WINDOW = tuple(range(10, 30))
SIGMAS = (0.0, .005, .01, .02)


def load_request(q):
    b = D.load_request(q)
    if D.sha(q['transition_noise_path']) != q['transition_noise_sha256']:
        raise ValueError('Transition noise file changed')
    a = D.arrays(q['transition_noise_path'])
    if set(a) != {'epsilon', 'k'}:
        raise ValueError('Transition file keys differ')
    eps, steps = a['epsilon'], a['k']
    if eps.dtype != np.float32 or eps.shape != (20, 1, q['N'], 9):
        raise ValueError('Transition noise must be FP32[20,1,N,9]')
    if steps.dtype != np.int64 or not np.array_equal(steps, np.asarray(WINDOW, np.int64)):
        raise ValueError('Transition step indices differ')
    if not np.isfinite(eps).all() or np.any(eps[:, :, :q['K']]):
        raise ValueError('Transition noise finite/C contract')
    if D.array_sha(eps) != q['transition_Gaussian_sha256']:
        raise ValueError('Transition noise array identity differs')
    return dict(**b, epsilon=eps, k=steps)


def verify_window_receipt(q, receipt, *, model_identity=None):
    """Read-only NumPy/mesh IO checks; no model or GPU calls."""
    from r2.pilots.terminal_reward_nft_v1.sample import verify_receipt
    if isinstance(receipt, (str, Path)):
        receipt = D.read(receipt)
    facts = verify_receipt(q, receipt, model_identity=model_identity)
    b = load_request(q)
    if D.sha(receipt['window_buffer_path']) != receipt['window_buffer_sha256']:
        raise ValueError('Window buffer hash mismatch')
    buffer = D.arrays(receipt['window_buffer_path'])
    n, k, sigma = q['N'], q['K'], float(receipt['sigma'])
    path = D.arrays(receipt['raw_path'])['trajectory'].reshape(51, 1, n, 9)
    expected_shape = (20, 1, n, 9)
    for key in ('state_before', 'state_after', 'mean_old', 'epsilon'):
        value = buffer[key]
        if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Window FP32 shape/finite contract: ' + key)
    facts.update(window_steps=D.same(buffer['k'], np.asarray(WINDOW, np.int64)),
        window_before_matches_trajectory=D.same(buffer['state_before'], path[10:30]),
        window_after_matches_trajectory=D.same(buffer['state_after'], path[11:31]),
        frozen_transition_noise_bitwise=D.same(buffer['epsilon'], b['epsilon']),
        full_joint_dimension=int(buffer['free_dimension']) == 9 * (n-k),
        fixed_sigma=D.same(buffer['sigma'], np.full(20, sigma, np.float64)),
        window_means_C_bitwise=all(D.same(x[0, :k].reshape(k, 3, 3), b['C']) for x in buffer['mean_old']),
        valid_all=buffer['valid_mask'].dtype == np.bool_ and buffer['valid_mask'].shape == (1, n)
                  and bool(buffer['valid_mask'].all()),
        known_mask=np.array_equal(buffer['known_mask'], np.arange(n)[None] < k),
        stored_counts=int(buffer['N']) == n and int(buffer['K']) == k,
        stored_y=D.same(buffer['y'], np.asarray([n], np.int64)),
        stored_time=D.same(buffer['t'], np.asarray(WINDOW, np.float32) / np.float32(50)))
    expected = buffer['mean_old'].copy()
    if sigma:
        expected[:, :, k:] += buffer['epsilon'][:, :, k:] * np.float32(sigma)
    facts['actual_action_equals_registered_transition'] = D.same(expected, buffer['state_after'])
    logp = buffer['logp_old']
    if sigma:
        residual = (buffer['state_after'][:, :, k:].astype(np.float64)
                    - buffer['mean_old'][:, :, k:].astype(np.float64))
        independent = -(residual**2).sum((1, 2, 3)) / (2*sigma*sigma)
        independent -= 9*(n-k) * (np.log(sigma) + .5*np.log(2*np.pi))
        facts['old_logp_full_joint'] = (logp.dtype == np.float64 and logp.shape == (20,)
            and np.isfinite(logp).all() and np.allclose(logp, independent, rtol=0, atol=1e-8)
            and receipt['density_status'] == 'FULL_JOINT_GAUSSIAN')
    else:
        facts['deterministic_has_no_fake_logpdf'] = (logp.dtype == np.float64 and logp.shape == (0,)
            and receipt['density_status'] == 'DETERMINISTIC_NOT_DEFINED')
    if not all(facts.values()):
        raise ValueError('Window artifact contract failed: ' + str(facts))
    return facts


def sample_request(model, q, budget, output_dir, *, label, model_identity,
                   model_mode, sigma, registration_sha256):
    """Exactly 50 forward attempts. No random-number generator is consumed."""
    from . import policy as P
    import torch
    from meshflow_control.mesh_io import write_raw
    dest = D.checked(output_dir)
    receipt_path = dest / 'receipt.json'
    sigma = float(sigma)
    if sigma not in SIGMAS or model_mode not in ('reference', 'current'):
        raise ValueError('Unregistered sigma or model mode')
    if receipt_path.exists():
        r = D.read(receipt_path)
        if (r['label'] != label or r['request_id'] != q['request_id'] or r['sigma'] != sigma
                or r['registration_sha256'] != registration_sha256 or r['model_mode'] != model_mode):
            raise ValueError('Completed request identity mismatch')
        verify_window_receipt(q, r, model_identity=model_identity)
        return r
    if dest.exists():
        raise RuntimeError('Unresolved rollout directory; no automatic replay: ' + str(dest))
    b = load_request(q)
    n, k = q['N'], q['K']
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Actual rollout requires explicitly dispatched CUDA worker')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Sampling requires cleared parameter gradients')
    context = dict(label=label, request_id=q['request_id'], input_sha256=q['input_sha256'],
        transition_noise_sha256=q['transition_noise_sha256'], noise_seed=q['noise_seed'],
        model_identity=D.clean(model_identity), model_mode=model_mode, sigma=sigma,
        registration_sha256=registration_sha256)
    budget.reserve('rollout_attempt', context=context)
    dest.mkdir(parents=True, exist_ok=False)
    D.write(dest / 'attempt.json', dict(status='STARTED', version=VERSION, **context))
    try:
        state = torch.from_numpy(b['z'].copy()).to(device)
        condition = torch.from_numpy(b['C'].reshape(1, k, 9).copy()).to(device)
        state[:, :k] = condition
        known = torch.zeros((1, n), dtype=torch.bool, device=device)
        known[:, :k] = True
        valid = torch.ones_like(known)
        free = valid & ~known
        y = torch.tensor([n], dtype=torch.long, device=device)
        epsilon = torch.from_numpy(b['epsilon'].copy()).to(device)
        states = [state[0].cpu().numpy().copy()]
        means, before, after, logps = [], [], [], []
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        with torch.no_grad():
            for step in range(50):
                hook = lambda call: budget.call('sampling_forward', call,
                    context={**context, 'euler_step': step})
                mean = P.transition_mean(model, state, step, y, valid, known,
                    model_mode=model_mode, forward_call=hook, autocast=True)
                if step in WINDOW:
                    action = P.apply_noise(mean, epsilon[step-10], sigma, free)
                    before.append(state.cpu().numpy().copy())
                    after.append(action.cpu().numpy().copy())
                    means.append(mean.cpu().numpy().copy())
                    if sigma:
                        logps.append(float(P.joint_log_prob(action, mean, sigma, free)[0].item()))
                else:
                    action = mean
                # The base sampler hard-clamps C every step; assert byte identity below.
                action[:, :k] = condition
                if action.dtype != torch.float32 or not bool(torch.isfinite(action).all()):
                    raise FloatingPointError('Invalid FP32 transition')
                state = action
                states.append(state[0].cpu().numpy().copy())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        rng_unchanged = (torch.equal(cpu_rng_before, torch.random.get_rng_state())
                         and torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device)))
        if not rng_unchanged:
            raise RuntimeError('Rollout unexpectedly consumed runtime RNG')
        trajectory = np.stack(states)
        if any(x[:k].tobytes() != b['C'].tobytes() for x in trajectory):
            raise RuntimeError('C bit pattern not preserved for all 51 states')
        density_status = 'FULL_JOINT_GAUSSIAN' if sigma else 'DETERMINISTIC_NOT_DEFINED'
        buffer_path = D.save_npz(dest / 'window_buffer.npz',
            k=np.asarray(WINDOW, np.int64), state_before=np.stack(before), state_after=np.stack(after),
            mean_old=np.stack(means), epsilon=b['epsilon'], sigma=np.full(20, sigma, np.float64),
            logp_old=np.asarray(logps, np.float64), free_dimension=np.asarray(9*(n-k), np.int64),
            known_mask=known.cpu().numpy(), valid_mask=valid.cpu().numpy(),
            N=np.asarray(n, np.int64), K=np.asarray(k, np.int64), y=np.asarray([n], np.int64),
            t=np.asarray(WINDOW, np.float32) / np.float32(50))
        audit = dict(version=VERSION, nfe=50, known_state_checks=51, known_all51_bitwise=True,
            known_coordinates_bitwise=True, N=n, K=k, seed=q['noise_seed'], seconds=elapsed,
            readout_mode='none', model_inputs=['C', 'N', 'initial_Gaussian', 'time', 'window_transition_Gaussian'],
            dtype='BF16 model forward; FP32 parameters/integration; FP64 full joint logpdf/KL',
            window=list(WINDOW), dt=P.DT, sigma=sigma, density_status=density_status,
            noise_interpretation='constant state sigma per transition; no sqrt(dt) scaling',
            window_adapter=model_mode, outside_adapter='reference_physical_bypass',
            runtime_RNG_unchanged=rng_unchanged, new_random_draws=0,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            peak_scope='One rollout with resident weights; process/context bytes separately monitored',
            human_usability='NOT_RUN', request=context, sampling_source_sha256=D.sha(__file__),
            policy_source_sha256=D.sha(P.__file__))
        write_raw(dest, trajectory[-1], b['C'], b['z'], trajectory, audit)
        r = dict(q)
        r.update(status='COMPLETE', label=label, model_identity=D.clean(model_identity),
            model_mode=model_mode, sigma=sigma, registration_sha256=registration_sha256,
            new_rollouts=1, new_sampling_F=50, generation_status='NEW_WINDOW_POLICY_EULER50',
            density_status=density_status, window=list(WINDOW), sampling=audit,
            sample_code_sha256=D.sha(__file__), policy_code_sha256=D.sha(P.__file__),
            window_buffer_path=str(buffer_path), window_buffer_sha256=D.sha(buffer_path))
        for key, name in (('raw', 'raw.npz'), ('ply', 'raw.ply'), ('result', 'run.json')):
            r[key+'_path'], r[key+'_sha256'] = str(dest / name), D.sha(dest / name)
        r['contracts'] = verify_window_receipt(q, r, model_identity=model_identity)
        D.write(receipt_path, r)
        budget.event('ROLLOUT_COMPLETE', context=context, nfe=50, seconds=elapsed,
            known_all51_bitwise=True, window_buffer_sha256=r['window_buffer_sha256'])
        return r
    except BaseException as exc:
        D.write(dest / 'failure.json', dict(status='FAILED_NO_AUTO_REPLAY', error=repr(exc),
            traceback=traceback.format_exc(), **context))
        raise


def validate_job(job_path):
    """CPU-only preparation and identity checks, before any model load."""
    job_path = D.resolve(job_path)
    job = D.read(job_path)
    manifest_path = D.resolve(job.get('manifest_path', OUT / 'inputs/manifest.json'))
    registration_path = D.resolve(job.get('registration_path', OUT / 'registration.json'))
    if D.sha(registration_path) != job['registration_sha256']:
        raise ValueError('Frozen registration changed')
    manifest, registration = D.read(manifest_path), D.read(registration_path)
    if registration.get('manifest_sha256') is not None and D.sha(manifest_path) != registration['manifest_sha256']:
        raise ValueError('Manifest differs from registration')
    if job['source_binding'] != manifest['source_binding']:
        raise ValueError('Source binding differs from frozen manifest')
    source = job['source_binding']
    if D.sha(source['path']) != source['sha256']:
        raise ValueError('Original FULL source changed')
    if job['model_mode'] not in ('reference', 'current') or float(job['sigma']) not in SIGMAS:
        raise ValueError('Unregistered policy mode/sigma')
    if job['model_mode'] == 'reference' and job.get('adapter_checkpoint') is not None:
        raise ValueError('Reference calibration cannot load adapters')
    candidates = manifest.get('calibration_requests', []) + manifest.get('evaluation_requests', [])
    training_path = D.resolve(job.get('training_manifest_path', OUT / 'inputs/training_manifest.json'))
    if training_path.exists():
        train = D.read(training_path)
        if train['evaluation_manifest_sha256'] != D.sha(manifest_path):
            raise ValueError('Training stream belongs to another evaluation registration')
        candidates += train['training_requests']
    registered = {q['request_id']: q for q in candidates}
    requests = job['requests']
    if not requests or len({q['request_id'] for q in requests}) != len(requests):
        raise ValueError('Empty/duplicate job requests')
    for q in requests:
        if q != registered.get(q['request_id']):
            raise ValueError('Worker request differs from frozen manifest')
        load_request(q)
    stage_dir = D.checked(job['stage_dir'])
    ledger_dir = D.checked(job.get('ledger_dir', stage_dir / 'workers' / job_path.stem))
    adapter = job.get('adapter_checkpoint')
    if adapter is not None:
        adapter = D.checked(adapter)
        if not adapter.is_file():
            raise ValueError('Current policy adapter must be an existing new-experiment checkpoint')
    return job, stage_dir, ledger_dir, adapter


def run(job_path):
    job, stage_dir, ledger_dir, adapter = validate_job(job_path)
    from r2.pilots.terminal_reward_nft_v1 import common as W
    from meshflow_control.checkpoints import state_hash
    from . import policy as P
    import torch
    threads = int(job.get('cpu_threads', 2))
    if threads not in (1, 2):
        raise ValueError('This worker allows one or two CPU threads')
    torch.set_num_threads(threads)
    label = job.get('label', stage_dir.name)
    identity = dict(version=VERSION, job_path=str(D.resolve(job_path)), job_sha256=D.sha(job_path),
        model_mode=job['model_mode'], sigma=float(job['sigma']), label=label,
        requests=[q['request_id'] for q in job['requests']], registration_sha256=job['registration_sha256'],
        source_binding=job['source_binding'], sample_source_sha256=D.sha(__file__),
        policy_source_sha256=D.sha(P.__file__))
    with W.Phase(ledger_dir, identity):
        source = job['source_binding']
        loader = P.load_reference if job['model_mode'] == 'reference' else P.load_policy
        model = loader(D.resolve(source['path']), expected_sha256=source['sha256'], device='cuda',
                       init_seed=int(job.get('adapter_init_seed', 2026101501)))
        model_identity = dict(version=P.VERSION, base_file_sha256=model.base_file_sha256,
            base_state_sha256=model.base_state_sha256, model_mode=job['model_mode'])
        if adapter is not None:
            payload = P.load_adapter_checkpoint(adapter, model,
                expected_sha256=job.get('adapter_checkpoint_sha256'),
                registration_sha256=job['registration_sha256'])
            model_identity.update(adapter_checkpoint=str(adapter), adapter_checkpoint_path=str(adapter), adapter_checkpoint_sha256=D.sha(adapter),
                                  adapters_sha256=payload['adapters_sha256'])
        elif isinstance(model, P.LoRAPolicy):
            model_identity.update(adapters_sha256=state_hash(model.adapter_state_dict()),
                                  adapter_initialization='fresh_zero_output', fresh_zero_lora=True, init_seed=model.init_seed)
        else:
            model_identity['adapter_modules'] = 0
        if isinstance(model, P.LoRAPolicy):
            model_identity['current_adapters_sha256'] = state_hash({
                k: v for k, v in model.adapter_state_dict().items()
                if k.endswith(('.current_a', '.current_b'))})
            model_identity['init_seed'] = model.init_seed
        budget = W.Budget(ledger_dir / 'calls.jsonl', max_rollouts=len(job['requests']))
        try:
            receipts = [sample_request(model, q, budget, stage_dir / 'raw' / q['request_id'],
                label=label, model_identity=model_identity, model_mode=job['model_mode'],
                sigma=job['sigma'], registration_sha256=job['registration_sha256']) for q in job['requests']]
            result = dict(status='COMPLETE', identity=identity, model_identity=model_identity,
                receipts=receipts, actual_rollout_attempts=budget.rollouts,
                actual_sampling_forwards=budget.forwards, hidden_validation_forwards=0,
                requested_trajectories=len(job['requests']))
            D.write(ledger_dir / 'raw_index.json', result)
            return result
        finally:
            budget.handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        job, _, _, _ = validate_job(args.job)
        print('PASS CPU-only job validation:', len(job['requests']), 'requests; model calls=0')
    else:
        result = run(args.job)
        print('COMPLETE', result['actual_rollout_attempts'], 'RAW;', result['actual_sampling_forwards'], 'forwards')
