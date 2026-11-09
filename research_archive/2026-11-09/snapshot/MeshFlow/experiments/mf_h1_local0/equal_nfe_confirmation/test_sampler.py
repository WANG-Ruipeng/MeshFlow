"""Bounded CPU-only formula, layout, precision and durable-budget fixtures."""
from native_t1.runtime import configure_stable_runtime
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from .sampler import METHODS, sample, new_counts, _validate_velocity
from .worker import ForwardLedger, LIMITS, TOTAL_LIMIT, compare_archive


class AnalyticModel(torch.nn.Module):
    def __init__(self, kind, C, known_slots=None, fail_on=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.experiment_trainable = False
        self.kind, self.C = kind, C.reshape(12, 9).clone()
        self.known_slots = list(range(12) if known_slots is None else known_slots)
        self.fail_on, self.calls, self.times = fail_on, 0, []
        self.eval()

    def forward(self, x, t, y, valid, known):
        self.calls += 1
        self.times.append(float(t[0]))
        assert x.dtype == t.dtype == torch.float32
        assert y.tolist() == [112] and valid.all()
        assert torch.where(known[0])[0].tolist() == sorted(self.known_slots)
        assert torch.equal(x[0, self.known_slots].contiguous().view(torch.int32), self.C.view(torch.int32))
        if self.calls == self.fail_on:
            raise RuntimeError('fixture injected model failure')
        v = (torch.ones_like(x) * .75 if self.kind == 'constant' else
             x.clone() if self.kind == 'linear' else torch.ones_like(x) * t[0])
        v[:, self.known_slots] = 123.
        return v


def run_cpu_tests():
    torch.set_num_threads(1)
    tick = time.perf_counter()
    C = torch.arange(108, dtype=torch.float32).reshape(12, 3, 3) / 100
    C[0, 0, 0] = -0.
    noise = torch.ones((1, 112, 9), dtype=torch.float32)
    rows = []
    counts = dict(real_model_forwards=0, model_backwards=0, optimizer_updates=0, OT_calls=0,
                  CPU_mock_forward_attempts=0, CPU_mock_forward_returns=0,
                  CPU_mock_complete_rollouts=0, expected_rejections=0)
    def accumulate(ledger):
        counts['CPU_mock_forward_attempts'] += ledger['model_forward_attempts']
        counts['CPU_mock_forward_returns'] += ledger['model_forward_returns']
        counts['CPU_mock_complete_rollouts'] += ledger['complete_rollouts']
    for kind in ('constant', 'linear', 'time'):
        for method, spec in METHODS.items():
            model, ledger = AnalyticModel(kind, C), new_counts()
            old_noise, old_C = noise.clone(), C.clone()
            raw, states, audit = sample(model, noise, C, method, ledger, cpu_mock=True)
            arrays = audit['_arrays']
            accumulate(ledger)
            assert model.calls == ledger['model_forward_attempts'] == ledger['model_forward_returns'] == 50
            assert states.shape == (spec['intervals'] + 1, 112, 9)
            assert arrays['stage_states'].shape == arrays['stage_velocities'].shape == (50, 112, 9)
            assert audit['known_accepted_all_bitwise'] and audit['known_stages_all_bitwise']
            assert np.all(arrays['stage_velocities'][:, :12] == 123.)
            assert torch.equal(noise.view(torch.int32), old_noise.view(torch.int32))
            assert torch.equal(C.view(torch.int32), old_C.view(torch.int32))
            assert audit['actual_endpoint_t1_forward_calls'] == 0
            assert min(model.times) == 0. and max(model.times) < 1.
            assert arrays['accepted_times_FP32'][-1] == 1.
            assert model.times[-1] == float(np.float32(.9604 if method == 'Q' else .98))
            h = arrays['step_sizes_control_FP64']
            value = float(raw[12, 0])
            if kind == 'constant':
                expected = 1.75
            elif kind == 'time':
                expected = 1.5 if method == 'M' else 1.5 - .5 * float(np.sum(h * h))
            else:
                expected = float(np.prod(1 + h + .5 * h * h)) if method == 'M' else float(np.prod(1 + h))
            assert abs(value - expected) < 3e-6
            # CPU arithmetic replay explicitly catches accidental midpoint-as-two-half-Euler.
            independent = np.float32(1.)
            times = arrays['accepted_times_control_FP64']
            for k, dt in enumerate(h):
                v1 = np.float32(.75 if kind == 'constant' else independent if kind == 'linear' else times[k])
                if method == 'M':
                    mid = np.float32(independent + v1 * np.float32(dt * .5))
                    vmid = np.float32(.75 if kind == 'constant' else mid if kind == 'linear' else (times[k] + times[k + 1]) * .5)
                    independent = np.float32(independent + vmid * np.float32(dt))
                else:
                    independent = np.float32(independent + v1 * np.float32(dt))
            assert np.float32(value).view(np.uint32) == independent.view(np.uint32)
            if method == 'E':
                assert np.all(arrays['step_sizes_FP32'] == np.float32(1 / 50))
            if method == 'Q':
                assert arrays['step_sizes_FP32'][0] == np.float32(.0004)
                assert arrays['step_sizes_FP32'][-1] == np.float32(.0396)
            rows.append(dict(field=kind, solver=method, calls=model.calls, endpoint=value,
                             discrete_formula=expected, final_query=model.times[-1],
                             float32_arithmetic_bitwise=True))
    # Explicit non-prefix and non-sorted condition slots; C order is preserved.
    slots = [111, 1, 103, 3, 95, 5, 87, 7, 79, 9, 71, 11]
    ledger = new_counts()
    raw, states, audit = sample(AnalyticModel('constant', C, slots), noise, C, 'M', ledger,
                                cpu_mock=True, known_slots=slots)
    accumulate(ledger)
    assert np.array_equal(raw[slots].view(np.uint32), C.numpy().reshape(12, 9).view(np.uint32))
    assert np.allclose(raw[[i for i in range(112) if i not in slots]], 1.75, atol=3e-6, rtol=0.)
    # One returned first stage and a reserved failed second stage: no accepted update.
    ledger = new_counts()
    model = AnalyticModel('constant', C, fail_on=2)
    try:
        sample(model, noise, C, 'M', ledger, cpu_mock=True)
    except RuntimeError as exc:
        assert 'fixture injected' in str(exc)
        assert len(exc.partial_states) == 1
        assert len(exc.partial_stage_states) == 2 and len(exc.partial_stage_velocities) == 1
    else:
        raise AssertionError('Expected injected model failure')
    assert ledger['model_forward_attempts'] == 2 and ledger['model_forward_returns'] == 1
    assert ledger['sampling_failures'] == 1 and ledger['complete_rollouts'] == 0
    accumulate(ledger)
    counts['expected_rejections'] += 1
    with tempfile.TemporaryDirectory(prefix='equal_nfe_budget_') as directory:
        # Historical diagnosis retains only five indexed accepted states for J/E.
        archive_path = Path(directory) / 'sparse_history.npz'
        indices = np.asarray([0, 5, 10, 20, 25], np.int64)
        np.savez(archive_path, raw=raw, states=states[indices], state_indices=indices)
        historical = compare_archive(archive_path, dict(raw=raw, states=states, **audit['_arrays']))
        assert historical['status'] == 'PASS'
        assert historical['comparisons'][1]['archived_state_indices'] == indices.tolist()
        path = Path(directory) / 'forward_ledger.json'
        durable = ForwardLedger(path)
        task = dict(task_id='CPU_fixture', kind='repeat')
        ledger = new_counts()
        model = AnalyticModel('constant', C, fail_on=2)
        try:
            sample(model, noise, C, 'M', ledger, cpu_mock=True,
                   reserve_forward=lambda: durable.reserve(task), forward_returned=lambda: durable.returned(task))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Expected durable failure fixture')
        accumulate(ledger)
        restored = ForwardLedger(path)
        assert restored.state['total_attempts'] == 2 and restored.state['total_returns'] == 1
        assert restored.state['tasks']['CPU_fixture'] == dict(kind='repeat', attempts=2, returns=1)
        # Rejection happens before the mocked model call and does not create an attempt.
        for budget_kind in LIMITS:
            restored.state['attempts_by_kind'][budget_kind] = LIMITS[budget_kind]
            model, ledger = AnalyticModel('constant', C), new_counts()
            try:
                sample(model, noise, C, 'E', ledger, cpu_mock=True,
                       reserve_forward=lambda kind=budget_kind: restored.reserve(dict(task_id='blocked_' + kind, kind=kind)))
            except RuntimeError as exc:
                assert 'budget exhausted' in str(exc)
            else:
                raise AssertionError('Category budget not enforced')
            assert model.calls == ledger['model_forward_attempts'] == 0
            assert restored.state['total_attempts'] == 2
            counts['expected_rejections'] += 1
        restored.state['total_attempts'] = TOTAL_LIMIT
        restored.state['attempts_by_kind']['formal'] = 0
        try:
            restored.reserve(dict(task_id='total_block', kind='formal'))
        except RuntimeError:
            counts['expected_rejections'] += 1
        else:
            raise AssertionError('Global 6320 cap not enforced')
    for action in (
        lambda: sample(AnalyticModel('constant', C), noise, C, 'E', cpu_mock=False),
        lambda: sample(AnalyticModel('constant', C), noise, C, 'Heun25', cpu_mock=True),
        lambda: _validate_velocity(torch.zeros_like(noise), noise, cpu_mock=False),
        lambda: sample(AnalyticModel('constant', C), noise, C, 'M', cpu_mock=True, known_slots=[0] * 12),
    ):
        try:
            action()
        except (ValueError, RuntimeError):
            counts['expected_rejections'] += 1
        else:
            raise AssertionError('Expected fail-closed rejection')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS', rows=rows, counts=counts, CUDA_initialized=False,
                seconds=time.perf_counter() - tick,
                fixture_groups=['three fields x three solvers, formula + exact FP32 recurrence',
                    'non-prefix unsorted mask and signed-zero C bit preservation',
                    'failed-forward attempt/return and partial trajectory accounting',
                    'durable journal replay, four category caps and global 6320 cap',
                    'no CPU/precision fallback; unregistered solver and invalid slot rejection'],
                note='10 successful analytic CPU mock trajectories, 2 intentionally failed mock trajectories. No real model created or GPU initialized. Formula fixtures do not establish neural-field convergence order.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    path = Path(args.record)
    if path.exists():
        raise FileExistsError(path)
    result = run_cpu_tests()
    result['sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), Path(__file__).with_name('sampler.py'), Path(__file__).with_name('worker.py'))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
