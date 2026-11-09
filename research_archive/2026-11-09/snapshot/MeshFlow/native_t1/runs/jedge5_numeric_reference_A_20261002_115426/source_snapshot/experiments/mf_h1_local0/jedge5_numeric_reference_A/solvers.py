"""Isolated numerical-reference solvers; never imports or edits production samplers.

Classic RK4 is written explicitly.  Adaptive stepping delegates to the installed
TorchDiffEq Dopri5Solver (not a reimplementation).  Its public odeint wrapper
perturbs endpoint times with nextafter, so this module deliberately supplies an
unperturbed adapter to the existing solver and records both controller and
actual function times.  No training, gradients, CUDA initialization or RNG here.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from pathlib import Path
from typing import Callable

import torch

FIXED_TIMES = (0.0, 0.2, 0.5, 0.8, 1.0)
SETTINGS = {
    'R1': {'method': 'classic_RK4', 'steps': 100, 'nfe_limit': 400},
    'R2': {'method': 'classic_RK4', 'steps': 200, 'nfe_limit': 800},
    'D1': {'method': 'Dopri5', 'rtol': 1e-4, 'atol': 1e-6, 'first_step': .01, 'nfe_limit': 1200},
    'D2': {'method': 'Dopri5', 'rtol': 1e-5, 'atol': 1e-7, 'first_step': .01, 'nfe_limit': 1200},
}
DOCUMENTATION_URL = 'https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md'


class SolverStop(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class NFEBudgetExceeded(SolverStop):
    def __init__(self, limit):
        super().__init__('NFE_BUDGET_EXHAUSTED', f'Function call cap {limit} reached; no retry.')


def dependency_info():
    """Fingerprint the actually installed implementation without importing CUDA."""
    try:
        import torchdiffeq
        from torchdiffeq._impl.dopri5 import Dopri5Solver
        from torchdiffeq._impl.rk_common import RKAdaptiveStepsizeODESolver, _runge_kutta_step
        from torchdiffeq._impl.interp import _interp_evaluate
        from torchdiffeq._impl.misc import _compute_error_ratio, _PerturbFunc
        signature = inspect.signature(RKAdaptiveStepsizeODESolver.__init__)
        required = {'first_step', 'step_t', 'dtype', 'min_step', 'max_step'}
        if not required.issubset(signature.parameters):
            raise RuntimeError(f'Unsupported installed adaptive API: {signature}')
        objects = [Dopri5Solver, RKAdaptiveStepsizeODESolver, _runge_kutta_step,
                   _interp_evaluate, _compute_error_ratio, _PerturbFunc]
        sources = {}
        for obj in objects:
            path = Path(inspect.getfile(obj))
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            'status': 'AVAILABLE_VERIFIED_INTERFACE', 'torchdiffeq_version': torchdiffeq.__version__,
            'torch_version': torch.__version__, 'sources_sha256': sources,
            'documentation': DOCUMENTATION_URL, 'constructor_signature': str(signature),
            'implementation': 'installed torchdiffeq._impl.dopri5.Dopri5Solver',
            'endpoint_policy': 'Direct adapter ignores PREV perturb request; exact stage endpoint is evaluated. No nextafter or clipping.',
            'state_update_dtype': 'same as initial (FP32 in experiment, FP64 in CPU analytic fixtures)',
            'controller_dtype': 'float64', 'initial_step': .01, 'forced_step_t': [1.0],
            'error_norm': 'sqrt(mean((embedded_error / (atol + rtol * max(abs(y0), abs(y_candidate))))**2)) over free coordinates only',
            'FSAL': 'library Dormand-Prince cached endpoint derivative; not a geometry or model-feature cache',
        }
    except (ImportError, AttributeError, RuntimeError) as exc:
        return {'status': 'BLOCKED_DEPENDENCY', 'reason': repr(exc), 'documentation': DOCUMENTATION_URL}


def _float(t):
    return float(t.detach().cpu()) if isinstance(t, torch.Tensor) else float(t)


def _validate_state(initial, cpu_fixture):
    if not isinstance(initial, torch.Tensor) or not initial.is_floating_point() or initial.numel() == 0:
        raise TypeError('Initial state must be a nonempty floating-point free-coordinate tensor.')
    if cpu_fixture:
        if initial.device.type != 'cpu' or initial.dtype not in (torch.float32, torch.float64):
            raise ValueError('CPU fixtures require CPU FP32/FP64 state.')
    elif initial.dtype != torch.float32:
        raise ValueError('Real-model numerical references require FP32 state.')
    if initial.requires_grad:
        raise ValueError('Numerical-reference integration is no_grad only.')
    if not bool(torch.isfinite(initial).all()):
        raise ValueError('Initial state is not finite.')


class _CountedFunction:
    """No perturbation, clipping, sorting, RNG, context state or target geometry."""
    def __init__(self, func, initial, limit, counts=None, cpu_fixture=False):
        self.func = func
        self.shape = initial.shape
        self.dtype = initial.dtype
        self.device = initial.device
        self.limit = int(limit)
        self.nfe = 0
        self.rows = []
        self.counts = counts
        self.cpu_fixture = cpu_fixture
        self.step_index = -1
        self.stage_index = 0
        self.controller_t0 = None
        self.controller_dt = None
        self.controller_alpha = None
        self.explicit_controller_time = None
        self.progress_observations = []
        self.last_error_ratio = None

    def __call__(self, t, x, *, perturb=None):
        if self.nfe >= self.limit:
            raise NFEBudgetExceeded(self.limit)
        if x.shape != self.shape or x.dtype != self.dtype or x.device != self.device:
            raise SolverStop('STATE_CONTRACT_FAILURE', 'Free shape/dtype/device changed.')
        actual = _float(t)
        if not math.isfinite(actual) or not 0.0 <= actual <= 1.0:
            raise SolverStop('TIME_DOMAIN_FAILURE', f'Function time outside [0,1]: {actual}')
        if self.explicit_controller_time is not None:
            controller = float(self.explicit_controller_time)
        elif self.controller_t0 is not None:
            a = self.controller_alpha[self.stage_index]
            controller = self.controller_t0 + a * self.controller_dt
        else:
            controller = actual
        if not 0.0 <= controller <= 1.0:
            raise SolverStop('TIME_DOMAIN_FAILURE', f'Controller stage outside [0,1]: {controller}')
        # The library already casts its stage times to state dtype. Explicit RK4
        # supplies float64 controller times; network receives the original API's FP32.
        passed_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        passed = _float(passed_t)
        self.nfe += 1
        if self.counts is not None:
            key = 'cpu_fixture_function_evaluations' if self.cpu_fixture else 'solver_function_evaluations'
            self.counts[key] = int(self.counts.get(key, 0)) + 1
        row = {
            'nfe': self.nfe, 'step_index': self.step_index, 'stage_index': self.stage_index,
            'controller_time': controller, 'library_function_time': actual,
            'actual_function_time': passed,
            'controller_to_function_time_error': passed - controller,
            'library_perturb_request_ignored': str(perturb) if perturb is not None else None,
            'function_time_dtype': str(passed_t.dtype), 'state_dtype': str(x.dtype),
        }
        self.rows.append(row)
        self.stage_index += 1
        velocity = self.func(passed_t, x)
        if velocity.shape != self.shape or velocity.dtype != self.dtype or velocity.device != self.device:
            raise SolverStop('VELOCITY_CONTRACT_FAILURE', 'Velocity shape/dtype/device differs from free state.')
        if not bool(torch.isfinite(velocity).all()):
            raise SolverStop('NONFINITE_VELOCITY', 'Function returned NaN/Inf.')
        row['velocity_dtype'] = str(velocity.dtype)
        return velocity

    def callback_step(self, t0, y0, dt):
        # TorchDiffEq calls this BEFORE its step_t clipping. The outer driver
        # records the actual forced terminal interval, not this proposal.
        pass

    def callback_accept_step(self, t0, y0, dt):
        pass

    def callback_reject_step(self, t0, y0, dt):
        pass

    def norm(self, scaled_error):
        value = scaled_error.square().mean().sqrt()
        self.last_error_ratio = _float(value)
        return value


def _snap(x):
    return x.detach().cpu().clone()


def _finish(initial, x, current_time, status, message, states, state_kinds, steps, counted, start, config):
    accepted = [r for r in steps if r.get('accepted') is True]
    rejected = [r for r in steps if r.get('accepted') is False]
    widths = [r['h'] for r in steps]
    complete = status == 'COMPLETE' and current_time == 1.0
    return {
        'status': status, 'complete': complete, 'failure': message,
        'endpoint': _snap(x), 'last_accepted_time': current_time,
        'states': states, 'state_kinds': state_kinds,
        'step_log': steps, 'function_log': counted.rows, 'nfe': counted.nfe,
        'accepted_steps': len(accepted), 'rejected_steps': len(rejected),
        'unfinished_attempts': sum(r.get('accepted') is None for r in steps),
        'min_step': min(widths) if widths else None, 'max_step': max(widths) if widths else None,
        'min_accepted_step': min([r['h'] for r in accepted], default=None),
        'max_accepted_step': max([r['h'] for r in accepted], default=None),
        'wall_seconds': time.perf_counter() - start,
        'configuration': config, 'state_dtype': str(initial.dtype),
        'state_shape': list(initial.shape), 'controller_dtype': 'torch.float64',
        'finite_last_accepted_state': bool(torch.isfinite(x).all()),
        'function_time_min': min([r['actual_function_time'] for r in counted.rows], default=None),
        'function_time_max': max([r['actual_function_time'] for r in counted.rows], default=None),
        'actual_t1_function_calls': sum(r['actual_function_time'] == 1.0 for r in counted.rows),
        'gradient_calls': 0, 'OT_calls': 0,
    }


@torch.no_grad()
def classic_rk4(func: Callable, initial: torch.Tensor, *, steps: int,
                fixed_times=FIXED_TIMES, counts=None, cpu_fixture=False, max_nfe=None):
    """Classical (not 3/8) RK4. Every step makes exactly four function calls."""
    _validate_state(initial, cpu_fixture)
    if steps <= 0:
        raise ValueError('Positive step count required.')
    fixed_times = tuple(sorted(set(float(v) for v in fixed_times)))
    index_times = {}
    for value in fixed_times:
        idx = int(round(value * steps))
        if not 0 <= value <= 1 or abs(idx / steps - value) > 1e-13:
            raise ValueError('RK4 fixed outputs must be actual registered step nodes.')
        index_times[idx] = value
    start = time.perf_counter()
    x = initial.detach().clone()
    limit = 4 * steps if max_nfe is None else min(int(max_nfe), 4 * steps)
    counted = _CountedFunction(func, x, limit, counts, cpu_fixture)
    snapshots, kinds, rows = {}, {}, []
    if 0 in index_times:
        snapshots[index_times[0]] = _snap(x)
        kinds[index_times[0]] = 'initial_state'
    current_time = 0.0
    status, message = 'COMPLETE', None
    h = torch.tensor(1.0 / steps, dtype=x.dtype, device=x.device)
    for i in range(steps):
        row = {'step_index': i, 't0': i / steps, 't1': (i + 1) / steps,
               'h': 1.0 / steps, 'nfe_before': counted.nfe, 'accepted': None,
               'error_ratio': None, 'error_estimate': 'not embedded; RK4 has no per-step error estimate'}
        rows.append(row)
        counted.step_index, counted.stage_index = i, 0
        try:
            def call_at(value, state):
                counted.explicit_controller_time = value
                return counted(torch.tensor(value, dtype=torch.float64, device=x.device), state)
            k1 = call_at(i / steps, x)
            k2 = call_at((i + .5) / steps, x + h * k1 / 2)
            k3 = call_at((i + .5) / steps, x + h * k2 / 2)
            k4 = call_at((i + 1) / steps, x + h * k3)
            candidate = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not bool(torch.isfinite(candidate).all()):
                raise SolverStop('NONFINITE_STATE', 'RK4 weighted state update is not finite.')
            x = candidate
            current_time = (i + 1) / steps
            row['accepted'] = True
            if i + 1 in index_times:
                output_t = index_times[i + 1]
                snapshots[output_t] = _snap(x)
                kinds[output_t] = 'accepted_step_node'
        except SolverStop as exc:
            status, message = exc.status, str(exc)
            break
        finally:
            row['nfe_after'] = counted.nfe
            row['nfe_in_attempt'] = counted.nfe - row['nfe_before']
    return _finish(initial, x, current_time, status, message, snapshots, kinds, rows, counted, start,
                   {'method': 'classic_RK4', 'steps': steps, 'h': 1 / steps,
                    'tableau': 'a21=1/2,a32=1/2,a43=1;b=(1,2,2,1)/6', 'nfe_limit': limit})


@torch.no_grad()
def dopri5(func: Callable, initial: torch.Tensor, *, rtol: float, atol: float,
           fixed_times=FIXED_TIMES, counts=None, cpu_fixture=False, max_nfe=1200):
    """Drive installed Dopri5 one attempt at a time, preserving its arithmetic.

    State is free coordinates only. Controller uses library FP64 time/step size.
    Every actual stage call, including rejected/incomplete attempts, counts.
    On budget exhaustion only the last ACCEPTED state is returned as incomplete.
    """
    _validate_state(initial, cpu_fixture)
    dep = dependency_info()
    if dep['status'] != 'AVAILABLE_VERIFIED_INTERFACE':
        return {'status': 'BLOCKED_DEPENDENCY', 'complete': False, 'nfe': 0, 'dependency': dep}
    from torchdiffeq._impl.dopri5 import Dopri5Solver
    from torchdiffeq._impl.interp import _interp_evaluate
    times = tuple(sorted(set(float(v) for v in fixed_times)))
    if any(not 0 <= v <= 1 for v in times):
        raise ValueError('Fixed output times must lie in [0,1].')
    start = time.perf_counter()
    x = initial.detach().clone()
    counted = _CountedFunction(func, x, min(int(max_nfe), 1200), counts, cpu_fixture)
    # This is the existing library solver; no copied/reimplemented controller.
    solver = Dopri5Solver(func=counted, y0=x, rtol=rtol, atol=atol,
                         first_step=.01, step_t=torch.tensor([1.], dtype=torch.float64, device=x.device),
                         dtype=torch.float64, norm=counted.norm)
    snapshots, kinds, rows = {}, {}, []
    if 0.0 in times:
        snapshots[0.0], kinds[0.0] = _snap(x), 'initial_state'
    status, message, current_time = 'COMPLETE', None, 0.0
    alpha = [float(a) for a in solver.tableau.alpha.detach().cpu()]
    try:
        solver._before_integrate(torch.tensor([0., 1.], dtype=torch.float64, device=x.device))
        while _float(solver.rk_state.t1) < 1.0:
            old = solver.rk_state
            t0 = _float(old.t1)
            proposal = _float(old.dt)
            # Exactly the installed step_t=[1] restriction. It is applied by
            # the library too; here it only prepares audit times and checks.
            forced_endpoint = t0 < 1.0 < t0 + proposal
            end = 1.0 if forced_endpoint else t0 + proposal
            width = 1.0 - t0 if forced_endpoint else proposal
            row = {'step_index': len(rows), 't0': t0, 't1': end, 'h': width,
                   'proposed_h': proposal, 'accepted': None, 'nfe_before': counted.nfe,
                   'nfe_after': counted.nfe, 'nfe_in_attempt': 0, 'forced_endpoint': forced_endpoint,
                   'error_ratio': None, 'error_estimate': 'free-coordinate normalized embedded 5(4) RMS',
                   'time_fp32_endpoint_equal': float(torch.tensor(t0, dtype=torch.float32)) == float(torch.tensor(end, dtype=torch.float32))}
            rows.append(row)
            if not math.isfinite(width) or not width > 0 or not end > t0:
                raise SolverStop('STEP_UNDERFLOW', f'No positive controller time advance: {t0}, {width}')
            # A step whose start/end collapse under model FP32 time cannot
            # provide a trustworthy tighter-time solve. Stop, never epsilon-hop.
            if not cpu_fixture and row['time_fp32_endpoint_equal']:
                raise SolverStop('TIME_FP32_ROUNDING_LIMIT', f'Step endpoints both round to model time at {t0}')
            counted.step_index, counted.stage_index = len(rows) - 1, 0
            counted.controller_t0, counted.controller_dt = t0, width
            counted.controller_alpha = alpha
            counted.last_error_ratio = None
            try:
                new = solver._adaptive_step(old)
                solver.rk_state = new
                ratio = counted.last_error_ratio
                row['error_ratio'] = ratio
                accepted = _float(new.t1) > t0
                row['accepted'] = accepted
                row['next_h'] = _float(new.dt)
                if ratio is None or not math.isfinite(ratio):
                    raise SolverStop('NONFINITE_ERROR_ESTIMATE', f'Embedded ratio is {ratio}')
                if _float(new.t1) > 1.0:
                    raise SolverStop('TIME_DOMAIN_FAILURE', 'Library accepted a step beyond 1.')
                if accepted:
                    x = new.y1
                    current_time = _float(new.t1)
                    if not bool(torch.isfinite(x).all()):
                        raise SolverStop('NONFINITE_STATE', 'Dopri weighted state update is not finite.')
                    for output_t in times:
                        if output_t in snapshots or not t0 <= output_t <= current_time:
                            continue
                        if output_t == current_time:
                            snapshots[output_t] = _snap(x)
                            kinds[output_t] = 'accepted_step_node'
                        else:
                            ti = torch.tensor(output_t, dtype=torch.float64, device=x.device)
                            snapshots[output_t] = _snap(_interp_evaluate(new.interp_coeff, new.t0, new.t1, ti))
                            kinds[output_t] = 'library_dense_interpolated_output'
            finally:
                row['nfe_after'] = counted.nfe
                row['nfe_in_attempt'] = counted.nfe - row['nfe_before']
    except SolverStop as exc:
        status, message = exc.status, str(exc)
    except AssertionError as exc:
        # Installed assertions describe time underflow / nonfinite state; do
        # not silently switch algorithm, relax tolerance or retry.
        status, message = 'LIBRARY_NUMERICAL_ASSERTION', str(exc)
    config = {'method': 'installed_Dopri5_5(4)', 'rtol': rtol, 'atol': atol,
              'first_step': .01, 'step_t': [1.0], 'nfe_limit': counted.limit,
              'error_norm': dep['error_norm'], 'dependency': dep,
              'endpoint_policy': dep['endpoint_policy'],
              'intermediate_alpha_source': 'installed tableau in state dtype; audit reconstructs controller-stage time using these actual coefficients'}
    return _finish(initial, x, current_time, status, message, snapshots, kinds, rows, counted, start, config)


def solve(kind, func, initial, *, counts=None, cpu_fixture=False, fixed_times=FIXED_TIMES, max_nfe=None):
    settings = SETTINGS[kind]
    if kind.startswith('R'):
        return classic_rk4(func, initial, steps=settings['steps'], counts=counts,
                          cpu_fixture=cpu_fixture, fixed_times=fixed_times, max_nfe=max_nfe)
    return dopri5(func, initial, rtol=settings['rtol'], atol=settings['atol'], counts=counts,
                  cpu_fixture=cpu_fixture, fixed_times=fixed_times,
                  max_nfe=settings['nfe_limit'] if max_nfe is None else max_nfe)

