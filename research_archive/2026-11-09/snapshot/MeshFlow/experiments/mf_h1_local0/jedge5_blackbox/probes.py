"""Registered, no-training local directional-response probes.

The two Gaussian offsets have coordinate std 1e-4 and are NOT normalized.
Only saved Euler50 states are queried; FP32 is a same-state comparison, not
another trajectory. These measurements are neither a condition number nor a
Jacobian spectral norm. No autograd, OT, integration, or geometry repair occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

SIGMA = 1e-4
DIRECTION_SEEDS = (61401, 61402)
PATCHES = (0, 4)
NOISE_SEEDS = (10401, 10402)
PRECISIONS = ('bf16', 'fp32')
MODEL_FORWARD_CAP = 120
TOTAL_FORWARD_CAP = 240
CHECKPOINT_SHA256 = {
    'B': '0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660',
    'J': 'cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9',
}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=_json_default, allow_nan=False,
                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_array(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).reshape(-1)))


def _rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


def registered_directions():
    """Local RNG only; identical offsets across model/case/precision."""
    return {
        seed: (np.random.RandomState(seed).randn(100, 9) * SIGMA).astype(np.float32)
        for seed in DIRECTION_SEEDS
    }


def _finite_shape(a, shape, name, *, fp32=False):
    a = np.asarray(a)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite {shape}, got {a.shape}')
    if fp32 and a.dtype != np.float32:
        raise ValueError(f'{name}: FP32 storage required')
    return a


def select_probe_steps(states50, states100, velocities50):
    """Select 3 unique pre-update indices, deterministically, without models."""
    a = _finite_shape(states50, (51, 112, 9), 'states50', fp32=True)
    b = _finite_shape(states100, (101, 112, 9), 'states100', fp32=True)
    v = _finite_shape(velocities50, (50, 112, 9), 'velocities50', fp32=True)
    delta = a[:, 12:].astype(np.float64) - b[::2, 12:].astype(np.float64)
    distance = np.sqrt(np.sum(delta * delta, axis=(1, 2)))
    increments = np.diff(distance)
    vv = v[1:, 12:].astype(np.float64) - v[:-1, 12:].astype(np.float64)
    speed_jump = np.sqrt(np.sum(vv * vv, axis=(1, 2)))
    ranked = sorted(range(50), key=lambda k: (-float(increments[k]), k))
    jump_k = min(range(1, 50), key=lambda k: (-float(speed_jump[k - 1]), k))
    proposals = [(ranked[0], 'largest_signed_matched_D_increment_start'),
                 (jump_k, 'largest_free_velocity_jump'), (25, 'fixed_t_0.5')]
    chosen, reasons = [], {}
    for k, reason in proposals:
        reasons.setdefault(k, []).append(reason)
        if k not in chosen:
            chosen.append(k)
    for k in ranked:
        if len(chosen) == 3:
            break
        if k not in chosen:
            chosen.append(k)
            reasons[k] = ['next_largest_signed_matched_D_increment_fill']
    if len(chosen) != 3 or len(set(chosen)) != 3:
        raise RuntimeError('Probe selection must have exactly three unique indices')
    return dict(
        steps=[dict(k=k, registered_t=k / 50,
                    actual_t_FP32=float(np.float32(k / 50)), reasons=reasons[k],
                    matched_D_increment=float(increments[k]),
                    free_velocity_jump=None if k == 0 else float(speed_jump[k - 1]))
               for k in chosen],
        D_free_L2=distance.tolist(), D_increment=increments.tolist(),
        velocity_jump_free_L2=[None] + speed_jump.tolist(),
        D_definition='L2 of free X50[k]-X100[2k], FP64 read-only arithmetic',
        ordering='largest signed D increment, largest free velocity jump, k25; '
                 'deduplicate then fill by signed D increment descending, ties lower k',
        outcomes_used='trajectory discrepancies only; no geometry score or model response selection',
    )


def perturbed_states(state, offset):
    state = _finite_shape(state, (112, 9), 'state', fp32=True)
    offset = _finite_shape(offset, (100, 9), 'offset', fp32=True)
    plus, minus = state.copy(), state.copy()
    plus[12:] = state[12:] + offset
    minus[12:] = state[12:] - offset
    if not _bits_equal(plus[:12], state[:12]) or not _bits_equal(minus[:12], state[:12]):
        raise RuntimeError('Perturbation changed known-face coordinates')
    dp = plus[12:].astype(np.float64) - state[12:].astype(np.float64)
    dm = minus[12:].astype(np.float64) - state[12:].astype(np.float64)
    chord = plus[12:].astype(np.float64) - minus[12:].astype(np.float64)
    if not _norm(dp) or not _norm(dm) or not _norm(chord):
        raise RuntimeError('Requested perturbation vanished in FP32 input coordinates')
    return plus, minus, dict(
        requested_sigma=SIGMA, requested_offset_RMS=_rms(offset),
        requested_offset_L2=_norm(offset), plus_actual_delta_RMS=_rms(dp),
        minus_actual_delta_RMS=_rms(dm), plus_actual_delta_L2=_norm(dp),
        minus_actual_delta_L2=_norm(dm), central_actual_input_chord_L2=_norm(chord),
        FP32_delta_antisymmetry_RMS=_rms(dp + dm),
        plus_changed_coordinate_fraction=float(np.mean(dp != 0)),
        minus_changed_coordinate_fraction=float(np.mean(dm != 0)),
        known_plus_bitwise_equal=True, known_minus_bitwise_equal=True,
    )


def directional_statistics(state, plus, minus, base_v, plus_v, minus_v):
    """Pure array summary; saved forward outputs, no new derivative calls."""
    for name, a in [('state', state), ('plus', plus), ('minus', minus),
                    ('base_v', base_v), ('plus_v', plus_v), ('minus_v', minus_v)]:
        _finite_shape(a, (112, 9), name, fp32=True)
    dplus = plus[12:].astype(np.float64) - state[12:].astype(np.float64)
    dminus = minus[12:].astype(np.float64) - state[12:].astype(np.float64)
    rp = plus_v[12:].astype(np.float64) - base_v[12:].astype(np.float64)
    rm = minus_v[12:].astype(np.float64) - base_v[12:].astype(np.float64)
    chord = plus_v[12:].astype(np.float64) - minus_v[12:].astype(np.float64)
    xchord = plus[12:].astype(np.float64) - minus[12:].astype(np.float64)
    denominator = _norm(xchord)
    if not denominator or not _norm(dplus) or not _norm(dminus):
        raise ValueError('Zero actual FP32 input perturbation')
    half_chord_rms = _rms(chord) / 2
    even_response = (rp + rm) / 2
    return dict(
        free_central_output_chord_RMS=_rms(chord),
        free_central_output_chord_L2=_norm(chord),
        free_central_directional_L2_gain=_norm(chord) / denominator,
        free_central_derivative_RMS_per_requested_sigma=_rms(chord / (2 * SIGMA)),
        free_plus_response_RMS=_rms(rp), free_minus_response_RMS=_rms(rm),
        free_plus_directional_L2_gain=_norm(rp) / _norm(dplus),
        free_minus_directional_L2_gain=_norm(rm) / _norm(dminus),
        free_even_response_RMS=_rms(even_response),
        free_even_to_half_chord_ratio=None if half_chord_rms == 0 else _rms(even_response) / half_chord_rms,
        free_plus_velocity_unchanged_fraction=float(np.mean(plus_v[12:] == base_v[12:])),
        free_minus_velocity_unchanged_fraction=float(np.mean(minus_v[12:] == base_v[12:])),
        known_predicted_velocity_plus_response_RMS=_rms(plus_v[:12].astype(np.float64) - base_v[:12]),
        known_predicted_velocity_minus_response_RMS=_rms(minus_v[:12].astype(np.float64) - base_v[:12]),
        known_coordinate_change='NONE; predicted known velocities are never integrated here',
    )


def _trajectory_path(out, model, patch, seed, nfe):
    return Path(out) / 'trajectories' / model / f'{model}_P0_patch{patch}_seed{seed}_bf16_Euler{nfe}.npz'


def load_trajectory(path, nfe):
    with np.load(path, allow_pickle=False) as z:
        required = {'raw', 'states', 'velocities', 'times', 'state_indices', 'C'}
        if not required <= set(z.files):
            raise ValueError('Missing trajectory fields: ' + str(sorted(required - set(z.files))))
        arrays = {k: z[k].copy() for k in required}
    _finite_shape(arrays['states'], (nfe + 1, 112, 9), 'states', fp32=True)
    _finite_shape(arrays['velocities'], (nfe, 112, 9), 'velocities', fp32=True)
    _finite_shape(arrays['times'], (nfe,), 'times', fp32=True)
    c = arrays['C'].reshape(12, 9)
    _finite_shape(c, (12, 9), 'C', fp32=True)
    if not _bits_equal(arrays['raw'], arrays['states'][-1]):
        raise ValueError('Saved raw differs from final trajectory state')
    if not np.array_equal(arrays['state_indices'], np.arange(nfe + 1)):
        raise ValueError('Incomplete or reordered trajectory states')
    expected_times = np.asarray([k / nfe for k in range(nfe)], dtype=np.float32)
    if not _bits_equal(arrays['times'], expected_times):
        raise ValueError('Trajectory uses an unexpected time grid')
    for state in arrays['states']:
        if not _bits_equal(state[:12], c):
            raise ValueError('Known-face trajectory drift')
    return arrays, dict(path=str(Path(path).resolve()), sha256=_sha_file(path),
                       nfe=nfe, arrays={k: _sha_array(a) for k, a in arrays.items()})


def _save_npz(path, arrays):
    tmp = Path(path).with_suffix('.partial.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _numpy_rng_same(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def run_probes(out, model_label):
    """One model per process; root owns GPU dispatch. Exactly 120F on success."""
    if model_label not in CHECKPOINT_SHA256:
        raise ValueError('Only frozen B/J are authorized')
    # Configure environment before torch is imported or CUDA initialized.
    from native_t1 import runtime as stable_runtime
    import torch
    from native_t1.artifacts import state_sha256
    from native_t1.model import native_execution
    from experiments.mf_h1_local0.jedge5_diagnosis.common import ASSETS, load_model
    from experiments.mf_h1_local0.jedge5_diagnosis.sampling import DTypeRecorder

    out = Path(out).resolve()
    worker = out / 'probes' / model_label
    worker.mkdir(parents=True, exist_ok=False)
    (worker / 'arrays').mkdir()
    counts = dict(model_forward_attempts=0, model_forward_returns=0,
                  model_load_attempts=0, model_load_returns=0, probe_states_completed=0,
                  model_backward_calls=0, optimizer_updates=0, OT_calls=0,
                  autograd_calls=0, full_rollouts=0, euler_updates=0)
    r = dict(schema='jedge5_blackbox_directional_probes_v1', status='PREPARING',
             model=model_label, sigma=SIGMA, direction_seeds=list(DIRECTION_SEEDS),
             forward_cap=MODEL_FORWARD_CAP, global_forward_cap=TOTAL_FORWARD_CAP,
             counts=counts, selections=[], states=[], forward_log=[],
             limitations=[
                 'Two Gaussian directions are not a condition number or a Jacobian spectral norm.',
                 'Sigma is a coordinate standard deviation in original model coordinates, not bbox normalized.',
                 'BF16 responses at this small offset may contain quantization jumps or unchanged outputs.',
                 'FP32 probes query the same BF16 trajectory states; they are not FP32 trajectories.',
                 'Selection is descriptive and tied to this trajectory discrepancy, not an independent sample.',
             ])
    tick = time.perf_counter()
    model = None
    def flush():
        r['wall_seconds'] = time.perf_counter() - tick
        _write_json(worker / 'results.json', r)
    flush()
    np_rng = np.random.get_state()
    try:
        prepared = []
        for patch in PATCHES:
            for seed in NOISE_SEEDS:
                p50, p100 = (_trajectory_path(out, model_label, patch, seed, n) for n in (50, 100))
                a, ainfo = load_trajectory(p50, 50)
                b, binfo = load_trajectory(p100, 100)
                if not _bits_equal(a['states'][0], b['states'][0]) or not _bits_equal(a['C'], b['C']):
                    raise ValueError('Euler50/100 did not start from identical input and C')
                selection = select_probe_steps(a['states'], b['states'], a['velocities'])
                row = dict(parent=0, patch=patch, seed=seed, trajectories=[ainfo, binfo], **selection)
                r['selections'].append(row)
                prepared.append((patch, seed, a, row))
        directions = registered_directions()
        r['directions'] = {str(seed): dict(array_sha256=_sha_array(a), RMS=_rms(a), L2=_norm(a),
                                         distribution='independent RandomState(seed).randn(100,9)*sigma, then FP32; no normalization')
                           for seed, a in directions.items()}
        _write_json(worker / 'selection.json', dict(selections=r['selections'], directions=r['directions']))
        r['selection_sha256'] = _sha_file(worker / 'selection.json')
        checkpoint = ASSETS[model_label]['path']
        if _sha_file(checkpoint) != CHECKPOINT_SHA256[model_label]:
            raise ValueError('Frozen checkpoint file SHA mismatch')
        stable_runtime.configure_stable_runtime()
        counts['model_load_attempts'] += 1
        model, identity = load_model(model_label)
        counts['model_load_returns'] += 1
        model.set_trainable(False)
        model.eval()
        r['model_identity'] = identity
        r['state_before'] = state_sha256(model)
        if r['state_before'] != ASSETS[model_label]['state']:
            raise ValueError('Unexpected generator state after strict loading')
        device = next(model.parameters()).device
        if device.type != 'cuda' or any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('Production probe requires a frozen FP32 CUDA generator')
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        known = torch.zeros((1, 112), dtype=torch.bool, device=device)
        known[:, :12] = True
        valid = torch.ones_like(known)
        y = torch.tensor([112], device=device)
        r['status'] = 'RUNNING'
        flush()

        for patch, seed, trajectory, selection in prepared:
            for selected in selection['steps']:
                k = selected['k']
                state = trajectory['states'][k].copy()
                c = trajectory['C'].reshape(12, 9)
                key = f'P0_patch{patch}_seed{seed}_k{k:02d}'
                artifact = worker / 'arrays' / (key + '.npz')
                arrays = dict(state=state, C=c.copy(), recorded_bf16_velocity=trajectory['velocities'][k].copy(),
                              sigma=np.asarray(SIGMA, dtype=np.float64),
                              time=np.asarray(trajectory['times'][k], dtype=np.float32))
                row = dict(parent=0, patch=patch, seed=seed, k=k, t=float(trajectory['times'][k]),
                           reasons=selected['reasons'], status='RUNNING', precision_results={},
                           path=str(artifact), state_sha256=_sha_array(state))
                r['states'].append(row)
                _save_npz(artifact, arrays)
                for precision in PRECISIONS:
                    t = torch.full((1,), k / 50, device=device, dtype=torch.float32)
                    state_gpu = torch.from_numpy(state.copy()).unsqueeze(0).to(device)
                    response = dict(status='RUNNING', directions=[])
                    row['precision_results'][precision] = response
                    recorder = DTypeRecorder(model)
                    try:
                        with torch.no_grad(), native_execution(model, coordinates=(state_gpu, t),
                                                              autocast=precision == 'bf16') as runtime:
                            response['runtime'] = runtime

                            def forward(value, tag):
                                if counts['model_forward_attempts'] >= MODEL_FORWARD_CAP:
                                    raise RuntimeError('Registered per-model 120F limit exceeded')
                                if not _bits_equal(value[:12], c):
                                    raise RuntimeError('Known coordinates changed before probe forward')
                                x = torch.from_numpy(value.copy()).unsqueeze(0).to(device)
                                event = dict(case=key, precision=precision, tag=tag, status='ATTEMPTED')
                                r['forward_log'].append(event)
                                counts['model_forward_attempts'] += 1
                                flush()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                v = model(x, t, y, valid, known)
                                counts['model_forward_returns'] += 1
                                end.record()
                                if v.requires_grad or v.shape != x.shape or not bool(torch.isfinite(v).all()):
                                    raise RuntimeError('Non-finite, wrong-shape, or gradient-enabled probe velocity')
                                if tag == 'base':
                                    response['actual_dtypes'] = recorder.finish(v, precision)
                                result = v.float()[0].cpu().numpy().copy()
                                end.synchronize()
                                event.update(status='RETURNED', cuda_event_seconds=start.elapsed_time(end) / 1000,
                                             input_sha256=_sha_array(value), output_sha256=_sha_array(result),
                                             actual_velocity_dtype=str(v.dtype), C_bitwise_equal=True)
                                arrays[precision + '_' + tag] = result
                                _save_npz(artifact, arrays)
                                row['sha256'] = _sha_file(artifact)
                                flush()
                                return result

                            base = forward(state, 'base')
                            if precision == 'bf16':
                                response['recorded_velocity_bitwise_equal'] = _bits_equal(base, trajectory['velocities'][k])
                                if not response['recorded_velocity_bitwise_equal']:
                                    raise RuntimeError('BF16 probe baseline is not bitwise equal to recorded trajectory velocity')
                            else:
                                bf16 = arrays['bf16_base']
                                response['same_state_FP32_minus_BF16_free_RMS'] = _rms(base[12:].astype(np.float64) - bf16[12:])
                            for direction_seed, offset in directions.items():
                                plus, minus, input_audit = perturbed_states(state, offset)
                                prefix = 'direction_' + str(direction_seed)
                                arrays[prefix + '_offset'] = offset
                                arrays[prefix + '_plus_input'] = plus
                                arrays[prefix + '_minus_input'] = minus
                                # This is an input rounding observation, not a first-layer causal claim.
                                b0 = torch.from_numpy(state[12:].copy()).to(torch.bfloat16).float().numpy()
                                bp = torch.from_numpy(plus[12:].copy()).to(torch.bfloat16).float().numpy()
                                bm = torch.from_numpy(minus[12:].copy()).to(torch.bfloat16).float().numpy()
                                input_audit['if_direct_BF16_input_cast_plus_changed_fraction'] = float(np.mean(bp != b0))
                                input_audit['if_direct_BF16_input_cast_minus_changed_fraction'] = float(np.mean(bm != b0))
                                vp = forward(plus, prefix + '_plus')
                                vm = forward(minus, prefix + '_minus')
                                arrays[precision + '_' + prefix + '_central_dv_dsigma'] = (
                                    vp.astype(np.float64) - vm.astype(np.float64)) / (2 * SIGMA)
                                response['directions'].append(dict(seed=direction_seed, input=input_audit,
                                    response=directional_statistics(state, plus, minus, base, vp, vm)))
                                _save_npz(artifact, arrays)
                                row['sha256'] = _sha_file(artifact)
                                flush()
                        response['status'] = 'PASS'
                    finally:
                        recorder.close()
                row['status'] = 'PASS'
                counts['probe_states_completed'] += 1
                flush()
        torch.cuda.synchronize(device)
        r['peak_cuda_allocated'] = torch.cuda.max_memory_allocated(device)
        r['peak_cuda_reserved'] = torch.cuda.max_memory_reserved(device)
        r['memory_scope'] = 'One loaded model; all 120 probe calls. Allocated is a subset of reserved.'
        r['state_after'] = state_sha256(model)
        r['CPU_torch_RNG_unchanged'] = bool(torch.equal(cpu_rng, torch.random.get_rng_state()))
        r['CUDA_torch_RNG_unchanged'] = bool(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        r['NumPy_global_RNG_unchanged'] = _numpy_rng_same(np_rng, np.random.get_state())
        if r['state_after'] != r['state_before'] or not all(r[k] for k in (
                'CPU_torch_RNG_unchanged', 'CUDA_torch_RNG_unchanged', 'NumPy_global_RNG_unchanged')):
            raise RuntimeError('Probe changed frozen state or global RNG')
        if counts['model_forward_attempts'] != MODEL_FORWARD_CAP or counts['model_forward_returns'] != MODEL_FORWARD_CAP or counts['probe_states_completed'] != 12:
            raise RuntimeError('Probe completed with an unexpected forward/state count')
        r['status'] = 'PASS'
        flush()
        summarize_probes(worker)
        return r
    except BaseException as exc:
        r['status'] = 'STOPPED'
        r['error'] = dict(type=type(exc).__name__, message=str(exc))
        flush()
        raise


def summarize_probes(worker):
    """Read existing scalar responses only; no model or geometric scoring."""
    import csv
    worker = Path(worker)
    r = json.loads((worker / 'results.json').read_text(encoding='utf-8'))
    if r['status'] != 'PASS':
        raise ValueError('Complete probe result required for summary')
    rows = []
    for state in r['states']:
        for precision, p in state['precision_results'].items():
            for d in p['directions']:
                rows.append(dict(model=r['model'], parent=state['parent'], patch=state['patch'],
                    seed=state['seed'], k=state['k'], t=state['t'], precision=precision,
                    direction_seed=d['seed'], **d['input'], **d['response']))
    groups = {}
    metric = 'free_central_directional_L2_gain'
    for precision in PRECISIONS:
        for patch in PATCHES:
            chosen = [x for x in rows if x['precision'] == precision and x['patch'] == patch]
            values = np.asarray([x[metric] for x in chosen], dtype=np.float64)
            groups[f'{precision}/patch{patch}'] = dict(count=len(chosen), metric=metric,
                mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()))
    summary = dict(status='PASS', schema='jedge5_blackbox_probe_summary_v1', model=r['model'],
                   source_sha256=_sha_file(worker / 'results.json'), rows=rows, groups=groups,
                   limitations=r['limitations'], new_model_calls=0, new_geometry_calls=0,
                   note='Selected times may differ across cases/models; do not treat group means as common-state B/J comparisons.')
    _write_json(worker / 'summary.json', summary)
    with (worker / 'directional_responses.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', choices=tuple(CHECKPOINT_SHA256), required=True)
    args = parser.parse_args(argv)
    run_probes(args.out, args.model)


if __name__ == '__main__':
    main()
