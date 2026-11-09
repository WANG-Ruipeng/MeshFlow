"""CPU-only analytic gates for the isolated reference integrators."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ..solvers import classic_rk4, dependency_info, dopri5, solve


def run():
    assert not torch.cuda.is_initialized()
    counts = {}
    checks = []
    dependency = dependency_info()
    x0 = torch.tensor([.25, -.5, 2.0], dtype=torch.float64)
    speed = torch.tensor([2.0, -3.0, .125], dtype=torch.float64)
    expected = x0 + speed
    for kind in ('R1', 'R2', 'D1', 'D2'):
        out = solve(kind, lambda t, x: speed, x0, counts=counts, cpu_fixture=True)
        assert out['status'] == 'COMPLETE', out.get('failure')
        assert out['last_accepted_time'] == 1.0
        assert out['actual_t1_function_calls'] >= 1
        assert out['function_time_min'] >= 0 and out['function_time_max'] <= 1
        err = float((out['endpoint'] - expected).abs().max())
        assert err < 2e-12, (kind, err)
        if kind.startswith('R'):
            assert out['nfe'] == (400 if kind == 'R1' else 800)
        else:
            assert out['nfe'] == 1 + 6 * len(out['step_log'])
            assert out['step_log'][0]['h'] == .01
            assert out['step_log'][-1]['t1'] == 1.0
        checks.append({'test': 'constant_velocity', 'solver': kind, 'max_error': err, 'nfe': out['nfe'],
                       't1_calls': out['actual_t1_function_calls'], 'state_kinds': out['state_kinds']})

    errors = []
    for steps in (10, 20):
        out = classic_rk4(lambda t, x: x, torch.ones(1, dtype=torch.float64),
                          steps=steps, counts=counts, cpu_fixture=True)
        error = abs(float(out['endpoint']) - math.e)
        errors.append(error)
    ratio = errors[0] / errors[1]
    assert 14.0 < ratio < 18.0, ratio
    checks.append({'test': 'rk4_linear_order', 'steps': [10, 20], 'errors': errors, 'error_ratio': ratio})

    for name, fn, expected in (
            ('linear', lambda t, x: 4.0 * x, math.exp(4.0)),
            ('smooth_time', lambda t, x: torch.ones_like(x) * (torch.cos(7 * t) + t * t), math.sin(7) / 7 + 1 / 3),
    ):
        init = torch.ones(1, dtype=torch.float64) if name == 'linear' else torch.zeros(1, dtype=torch.float64)
        loose = dopri5(fn, init, rtol=1e-4, atol=1e-6, counts=counts, cpu_fixture=True)
        tight = dopri5(fn, init, rtol=1e-5, atol=1e-7, counts=counts, cpu_fixture=True)
        assert loose['complete'] and tight['complete']
        le = abs(float(loose['endpoint']) - expected)
        te = abs(float(tight['endpoint']) - expected)
        assert te < le, (name, le, te)
        assert tight['nfe'] >= loose['nfe']
        checks.append({'test': 'dopri_tolerance_trend', 'ode': name,
                       'loose_error': le, 'tight_error': te,
                       'loose_nfe': loose['nfe'], 'tight_nfe': tight['nfe'],
                       'loose_rejected': loose['rejected_steps'], 'tight_rejected': tight['rejected_steps']})

    # Preserve arbitrary noncontiguous known/free positions; the state and norm
    # contain ONLY free coordinates. This toy embedding has no neural calls.
    full = torch.arange(45, dtype=torch.float64).reshape(5, 9)
    known = torch.tensor([True, False, True, False, False])
    C = full[known].clone()
    free = full[~known].clone()
    seen_shapes = []
    def clamped(t, state):
        reconstructed = torch.empty_like(full)
        reconstructed[known] = C
        reconstructed[~known] = state
        assert torch.equal(reconstructed[known].view(torch.int64), C.view(torch.int64))
        seen_shapes.append(tuple(state.shape))
        return torch.ones_like(state)
    out = solve('D1', clamped, free, counts=counts, cpu_fixture=True)
    assert set(seen_shapes) == {(3, 9)}
    for state in out['states'].values():
        reconstructed = full.clone()
        reconstructed[~known] = state
        assert torch.equal(reconstructed[known].view(torch.int64), C.view(torch.int64))
    checks.append({'test': 'C_clamping_noncontiguous_free_only_state', 'free_shape': [3, 9], 'known_shape': [2, 9], 'nfe': out['nfe']})

    # Hard cap in the middle of a Dopri attempt: count all 10 real function
    # invocations and preserve the preceding accepted (not unfinished) state.
    capped = dopri5(lambda t, x: x, torch.ones(1, dtype=torch.float64),
                    rtol=1e-4, atol=1e-6, max_nfe=10, counts=counts, cpu_fixture=True)
    assert capped['status'] == 'NFE_BUDGET_EXHAUSTED'
    assert capped['nfe'] == 10 and not capped['complete']
    assert capped['last_accepted_time'] == .01
    assert capped['unfinished_attempts'] == 1
    checks.append({'test': 'hard_NFE_cap', 'nfe': capped['nfe'], 'last_accepted_time': capped['last_accepted_time'],
                   'unfinished_attempts': capped['unfinished_attempts']})

    # Function time 1 is literal 1.0, with no hidden torchdiffeq nextafter.
    visited = []
    def endpoint_exact(t, x):
        visited.append(float(t))
        return torch.zeros_like(x)
    out = solve('D1', endpoint_exact, torch.zeros(2, dtype=torch.float32), counts=counts, cpu_fixture=True)
    assert out['complete'] and 1.0 in visited and max(visited) == 1.0
    assert out['endpoint'].dtype == torch.float32
    assert all(r['velocity_dtype'] == 'torch.float32' for r in out['function_log'])
    checks.append({'test': 'FP32_updates_and_true_endpoint', 'nfe': out['nfe'], 'max_time': max(visited),
                   'true_endpoint_calls': visited.count(1.0)})

    # In the autonomous case the only deliberate adapter distinction (time
    # perturbation) has no effect; compare to public existing odeint arithmetic.
    from torchdiffeq import odeint
    fn = lambda t, x: 2 * x
    initial = torch.tensor([.2, -.7], dtype=torch.float64)
    wrapped = dopri5(fn, initial, rtol=1e-4, atol=1e-6, counts=counts, cpu_fixture=True)
    public_calls = {'count': 0}
    def public_fn(t, x):
        public_calls['count'] += 1
        return fn(t, x)
    public = odeint(public_fn, initial, torch.tensor([0., 1.], dtype=torch.float64),
                    method='dopri5', rtol=1e-4, atol=1e-6,
                    options={'first_step': .01, 'step_t': torch.tensor([1.], dtype=torch.float64)})[-1]
    counts['cpu_fixture_function_evaluations'] += public_calls['count']
    gap = float((wrapped['endpoint'] - public).abs().max())
    assert gap < 1e-12, gap
    checks.append({'test': 'existing_library_autonomous_parity', 'max_error': gap, 'public_function_calls': public_calls['count']})
    assert not torch.cuda.is_initialized()
    return {'status': 'PASS', 'dependency': dependency, 'checks': checks,
            'counts': counts, 'actual_neural_model_forwards': 0, 'backwards': 0,
            'optimizer_updates': 0, 'OT_calls': 0, 'cuda_initialized': torch.cuda.is_initialized()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = run()
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
