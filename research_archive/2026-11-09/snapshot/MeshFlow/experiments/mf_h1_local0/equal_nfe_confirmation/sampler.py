"""Exactly three registered 50-NFE paths; device-buffered raw trajectories.

Euler is the production constant 1/50 expression. M/Q preserve the pilot's
Python FP64 grid, independent time/dt conversion, and operation order.
"""
from native_t1.runtime import configure_stable_runtime
from native_t1.model import native_execution
from native_t1.artifacts import array_hash
from contextlib import nullcontext
import time
import numpy as np
import torch

METHODS = {
    'E': dict(name='Euler50', solver='Euler', intervals=50, nfe=50),
    'M': dict(name='Midpoint25', solver='Midpoint', intervals=25, nfe=50),
    'Q': dict(name='E_quad50', solver='Euler', intervals=50, nfe=50),
}
ALIASES = {v['name']: k for k, v in METHODS.items()}
TIME_POLICY = ('E: production t=k/50 and constant Python 1/50; '
               'M/Q: pilot FP64 grid, independently cast t and dt; '
               'M stage time=(t0+t1)*.5 and half-step=dt*.5; no accumulation.')


def new_counts():
    return dict(model_forward_attempts=0, model_forward_returns=0,
                accepted_steps=0, full_rollout_attempts=0, complete_rollouts=0,
                sampling_failures=0, model_backward_calls=0, optimizer_updates=0,
                geometry_gradient_calls=0, OT_calls=0, postprocessing_repairs=0)


def _bits_equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def _validate(model, noise, C, method, cpu_mock, known_slots):
    if method not in METHODS:
        raise ValueError('Only preregistered E/M/Q are supported')
    if not isinstance(cpu_mock, bool):
        raise ValueError('cpu_mock must be explicit boolean')
    if noise.dtype != torch.float32 or noise.shape != (1, 112, 9):
        raise ValueError('Original FP32 [1,112,9] Gaussian required')
    if C.dtype != torch.float32 or C.shape not in ((12, 3, 3), (12, 9)):
        raise ValueError('FP32 K12 condition required')
    if len(known_slots) != 12 or len(set(known_slots)) != 12 or any(k not in range(112) for k in known_slots):
        raise ValueError('Twelve distinct explicit known slots required')
    parameters = list(model.parameters())
    expected_device = 'cpu' if cpu_mock else 'cuda'
    if noise.device.type != expected_device or C.device != noise.device or any(p.device != noise.device for p in parameters):
        raise ValueError('Explicit CPU mock or real CUDA required; no device fallback')
    if not parameters or any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in parameters):
        raise RuntimeError('Frozen FP32 parameters without stored gradients required')
    if getattr(model, 'experiment_trainable', True) or any(m.training for m in model.modules()):
        raise RuntimeError('Frozen eval model required')
    if not bool(torch.isfinite(noise).all() and torch.isfinite(C).all()):
        raise FloatingPointError('Nonfinite input')


def _validate_velocity(velocity, state, cpu_mock):
    if not isinstance(velocity, torch.Tensor) or velocity.shape != state.shape or velocity.device != state.device or velocity.requires_grad:
        raise RuntimeError('Invalid velocity shape/device/gradient graph')
    if velocity.dtype != (torch.float32 if cpu_mock else torch.bfloat16):
        raise RuntimeError('Actual velocity dtype violates frozen precision path')
    if not bool(torch.isfinite(velocity).all()):
        raise FloatingPointError('Nonfinite model velocity')


@torch.no_grad()
def sample(model, noise, C, method, counts=None, cpu_mock=False, *,
           known_slots=None, reserve_forward=None, forward_returned=None):
    """Return raw[112,9], accepted states and audit (NPZ arrays in _arrays).

    reserve_forward is called and must persist the attempt BEFORE every model
    call. No retries are performed. Full GPU stage/state arrays are copied only
    after sampling; finite/C checks still synchronize and are disclosed.
    """
    method = ALIASES.get(method, method)
    known_slots = tuple(range(12) if known_slots is None else known_slots)
    _validate(model, noise, C, method, cpu_mock, known_slots)
    if not cpu_mock:
        configure_stable_runtime()
    spec = METHODS[method]
    intervals = spec['intervals']
    counts = new_counts() if counts is None else counts
    for k in new_counts():
        counts.setdefault(k, 0)
    counts['full_rollout_attempts'] += 1
    initial_attempts, initial_returns = counts['model_forward_attempts'], counts['model_forward_returns']
    noise_hash, condition_hash = array_hash(noise), array_hash(C)
    condition = C.reshape(12, 9)
    free_slots = tuple(k for k in range(112) if k not in known_slots)
    # Preserve production/pilot slice arithmetic exactly for the archived layout.
    prefix = known_slots == tuple(range(12))
    ki = slice(0, 12) if prefix else list(known_slots)
    fi = slice(12, 112) if prefix else list(free_slots)
    state = noise.clone()
    state[:, ki] = condition[None]
    known = torch.zeros((1, 112), dtype=torch.bool, device=noise.device)
    known[:, ki] = True
    valid = torch.ones_like(known)
    y = torch.tensor([112], device=noise.device)
    ideal = [(k / intervals)**2 if method == 'Q' else k / intervals for k in range(intervals + 1)]
    steps = [1 / 50 for _ in range(50)] if method == 'E' else [ideal[k + 1] - ideal[k] for k in range(intervals)]
    accepted, stage_states, velocities = [], [], []
    stage_times, stage_intervals, stage_kinds, stage_dt = [], [], [], []
    events, dtype_records = [], []
    check_count = 0
    host_seconds = check_seconds = 0.
    copy_seconds = 0.
    sync_seconds = 0.

    def sync():
        nonlocal sync_seconds
        if not cpu_mock:
            tick = time.perf_counter()
            torch.cuda.synchronize(noise.device)
            sync_seconds += time.perf_counter() - tick

    def assert_state(value):
        nonlocal check_count, check_seconds
        tick = time.perf_counter()
        check_count += 1
        if value.dtype != torch.float32 or not _bits_equal(value[0, ki], condition):
            raise RuntimeError('Stage/accepted C bits changed')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite stage/accepted state')
        check_seconds += time.perf_counter() - tick

    def advance(base, velocity, dt):
        updated = base.clone()
        updated[:, fi] = base[:, fi] + velocity[:, fi] * dt
        updated[:, ki] = condition[None]
        assert_state(updated)
        return updated

    def forward(value, t_control, interval, kind, dt_control):
        nonlocal host_seconds, check_seconds
        if counts['model_forward_attempts'] - initial_attempts >= 50:
            raise RuntimeError('50-NFE per-rollout budget exceeded')
        assert_state(value)
        t = torch.full((1,), t_control, dtype=torch.float32, device=noise.device)
        if not cpu_mock and (not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda') != torch.bfloat16):
            raise RuntimeError('BF16 autocast context lost')
        if reserve_forward is not None:
            reserve_forward()
        counts['model_forward_attempts'] += 1
        stage_states.append(value[0].detach().clone())
        stage_times.append(t_control)
        stage_intervals.append(interval)
        stage_kinds.append(kind)
        stage_dt.append(dt_control)
        if not cpu_mock:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        tick = time.perf_counter()
        velocity = model(value, t, y, valid, known)
        host_seconds += time.perf_counter() - tick
        counts['model_forward_returns'] += 1
        if not cpu_mock:
            end.record()
            events.append((begin, end))
        if forward_returned is not None:
            forward_returned()
        tick = time.perf_counter()
        if isinstance(velocity, torch.Tensor) and velocity.shape == value.shape:
            # Keep returned nonfinite velocities too if numerical validation fails.
            velocities.append(velocity.float()[0].detach().clone())
        _validate_velocity(velocity, value, cpu_mock)
        check_seconds += time.perf_counter() - tick
        dtype_records.append(str(velocity.dtype))
        v = velocity.float()
        return v

    def exported(values):
        nonlocal copy_seconds
        tick = time.perf_counter()
        result = (torch.stack(values).cpu().numpy().copy() if values else np.empty((0, 112, 9), np.float32))
        copy_seconds += time.perf_counter() - tick
        return result

    sync()
    if not cpu_mock:
        torch.cuda.reset_peak_memory_stats(noise.device)
        sample_begin, sample_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sample_begin.record()
    tick = time.perf_counter()
    context = (nullcontext(dict(cpu_mock=True, native_precision='FP32 analytic only')) if cpu_mock else
               native_execution(model, coordinates=(state, condition), autocast=True))
    try:
        assert_state(state)
        accepted.append(state[0].detach().clone())
        with context as runtime:
            for k, dt in enumerate(steps):
                t0, t1 = ideal[k], ideal[k + 1]
                v1 = forward(state, t0, k, 'start', dt * .5 if method == 'M' else dt)
                if method == 'M':
                    midpoint = advance(state, v1, dt * .5)
                    v2 = forward(midpoint, (t0 + t1) * .5, k, 'midpoint', dt)
                    state = advance(state, v2, dt)
                else:
                    state = advance(state, v1, dt)
                counts['accepted_steps'] += 1
                accepted.append(state[0].detach().clone())
        if not cpu_mock:
            sample_end.record()
        sync()
        sampling_wall = time.perf_counter() - tick
        if counts['model_forward_attempts'] - initial_attempts != 50 or counts['model_forward_returns'] - initial_returns != 50:
            raise RuntimeError('Actual NFE differs from registered contract')
        if array_hash(noise) != noise_hash or array_hash(C) != condition_hash:
            raise RuntimeError('Input mutation')
        states = exported(accepted)
        coords = exported(stage_states)
        vels = exported(velocities)
        C_cpu = condition.cpu().numpy()
        for values in (states, coords):
            actual = np.ascontiguousarray(values[:, known_slots, :])
            expect = np.broadcast_to(C_cpu, actual.shape).copy()
            if not np.array_equal(actual.view(np.uint32), expect.view(np.uint32)):
                raise RuntimeError('Exported C bits changed')
        counts['complete_rollouts'] += 1
        arrays = dict(stage_states=coords, stage_velocities=vels,
            stage_times_FP32=np.asarray(stage_times, np.float32),
            stage_times_control_FP64=np.asarray(stage_times, np.float64),
            accepted_times_FP32=np.asarray(ideal, np.float32),
            accepted_times_control_FP64=np.asarray(ideal, np.float64),
            step_sizes_FP32=np.asarray(steps, np.float32), step_sizes_control_FP64=np.asarray(steps, np.float64),
            stage_step_sizes_FP32=np.asarray(stage_dt, np.float32), stage_step_sizes_control_FP64=np.asarray(stage_dt, np.float64),
            stage_interval=np.asarray(stage_intervals, np.int64), stage_kind=np.asarray(stage_kinds, dtype='U16'),
            known_mask=known[0].cpu().numpy(), free_mask=(~known)[0].cpu().numpy(), valid_mask=valid[0].cpu().numpy(),
            known_slots=np.asarray(known_slots, np.int64), free_slots=np.asarray(free_slots, np.int64))
        return states[-1].copy(), states, dict(method=method, name=spec['name'], nfe=50, intervals=intervals,
            model_forward_attempts=50, model_forward_returns=50, accepted_states=intervals + 1, recorded_stage_states=50,
            time_policy=TIME_POLICY, ideal_accepted_times=ideal,
            actual_stage_times_FP32=arrays['stage_times_FP32'].tolist(), actual_step_sizes_FP32=arrays['step_sizes_FP32'].tolist(),
            actual_endpoint_t1_forward_calls=sum(t == 1. for t in stage_times), runtime=runtime,
            actual_velocity_dtypes=sorted(set(dtype_records)), integration_dtype='torch.float32', parameter_dtype='torch.float32',
            known_accepted_all_bitwise=True, known_stages_all_bitwise=True, known_check_count=check_count,
            known_predicted_velocity_may_be_nonzero=True, known_velocity_applied=False, inputs_unchanged=True,
            noise_sha256=noise_hash, condition_sha256=condition_hash, first_stage_velocity_sha256=array_hash(vels[0]),
            total_faces=112, known_faces=12, free_faces=100, seconds=sampling_wall,
            sampling_wall_seconds=sampling_wall, model_forward_host_seconds=host_seconds,
            sampling_cuda_event_seconds=None if cpu_mock else sample_begin.elapsed_time(sample_end) / 1000,
            model_forward_cuda_event_seconds=None if cpu_mock else sum(a.elapsed_time(b) for a, b in events) / 1000,
            explicit_cuda_synchronize_seconds=sync_seconds, finite_and_C_check_seconds=check_seconds,
            stage_and_state_CPU_copy_seconds=copy_seconds,
            peak_cuda_allocated=0 if cpu_mock else torch.cuda.max_memory_allocated(noise.device),
            peak_cuda_reserved=0 if cpu_mock else torch.cuda.max_memory_reserved(noise.device),
            peak_scope='Reset immediately before this rollout; includes resident model, device trajectories and caches.',
            timing_note='Nested timings are not additive. Full arrays transfer after sampling; finite/C checks synchronize each stage; durable budget logging is within wall/event time.',
            model_backwards=0, optimizer_updates=0, OT_calls=0, geometry_gradient_calls=0, postprocessing=None,
            cpu_mock=cpu_mock, _arrays=arrays)
    except BaseException as exc:
        counts['sampling_failures'] += 1
        exc.partial_states = exported(accepted)
        exc.partial_stage_states = exported(stage_states)
        exc.partial_stage_velocities = exported(velocities)
        exc.partial_stage_times_FP32 = np.asarray(stage_times, np.float32)
        exc.partial_stage_times_control_FP64 = np.asarray(stage_times, np.float64)
        exc.partial_counts = dict(counts)
        raise
