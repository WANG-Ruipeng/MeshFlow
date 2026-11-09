"""Independent Euler diagnostics; no production sampler/model arithmetic edits.

Full trajectories and absolute-time suffixes share exactly one integration loop.
Only the registered forward precision and Euler NFE may vary. CPU mock support is
explicit and rejects CUDA; it is never selected by the experiment runners.
"""
from native_t1.runtime import configure_stable_runtime
from contextlib import nullcontext
import time
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.artifacts import array_hash

PRECISIONS = ('bf16', 'fp32')
NFES = (50, 100, 200)


def new_counts():
    return dict(model_forward_attempts=0, model_forward_returns=0, euler_updates=0,
                full_rollout_attempts=0, complete_rollouts=0, suffix_attempts=0,
                suffix_returns=0, sampling_failures=0, model_backward_calls=0,
                optimizer_updates=0, geometry_gradient_calls=0, postprocessing_repairs=0)


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def _dtype_tree(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape), device=value.device.type)
    if isinstance(value, (tuple, list)):
        return [_dtype_tree(v) for v in value]
    return str(type(value).__name__)


def _floating_dtypes(value):
    if isinstance(value, dict):
        return [value['dtype']] if 'dtype' in value else sum((_floating_dtypes(v) for v in value.values()), [])
    if isinstance(value, list):
        return sum((_floating_dtypes(v) for v in value), [])
    return []


class DTypeRecorder:
    """Observe the first actual forward only; hooks never replace tensors."""
    def __init__(self, model):
        self.records = {}
        self.handles = []
        block = model.backbone.layers[0]
        modules = [('coordinate_embedding', model.backbone.x_embedder),
                   ('QKV', block.attn.qkv_proj), ('attention_projected_output', block.attn),
                   ('MLP', block.mlp)]
        for name, module in modules:
            def hook(_, args, output, name=name):
                self.records[name] = dict(input=_dtype_tree(args), output=_dtype_tree(output))
            self.handles.append(module.register_forward_hook(hook))
        def attention_input(_, args):
            self.records['attention_unprojected_output'] = _dtype_tree(args)
        self.handles.append(block.attn.out_proj.register_forward_pre_hook(attention_input))

    def finish(self, velocity, precision):
        self.records['velocity'] = _dtype_tree(velocity)
        self.close()
        required = {'coordinate_embedding', 'QKV', 'attention_projected_output',
                    'attention_unprojected_output', 'MLP', 'velocity'}
        if set(self.records) != required:
            raise RuntimeError('Missing actual dtype observation sites')
        if precision == 'fp32' and any(d != 'torch.float32' for d in _floating_dtypes(self.records)):
            raise RuntimeError('FP32 diagnostic observed a non-FP32 floating tensor')
        return self.records

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _validate(model, noise, C, nfe, precision, initial_state, start_step, cpu_mock):
    if precision not in PRECISIONS or type(nfe) is not int or nfe not in NFES:
        raise ValueError('Only registered bf16/fp32 and Euler50/100/200')
    if type(start_step) is not int or not 0 <= start_step < nfe:
        raise ValueError('Absolute start_step must be in [0,nfe)')
    if (initial_state is None) != (start_step == 0):
        raise ValueError('Suffix requires a saved state and positive absolute start_step')
    if noise.shape != (1,112,9) or noise.dtype != torch.float32:
        raise ValueError('Original FP32 [1,112,9] Gaussian required')
    if C.shape not in ((12,3,3), (12,9)) or C.dtype != torch.float32:
        raise ValueError('Registered C must be FP32 K12')
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in parameters):
        raise RuntimeError('Frozen FP32 model without stored gradients required')
    if getattr(model, 'experiment_trainable', True) or any(m.training for m in model.modules()):
        raise RuntimeError('Read-only eval model required')
    expected_device = 'cpu' if cpu_mock else 'cuda'
    if noise.device.type != expected_device or C.device != noise.device or any(p.device != noise.device for p in parameters):
        raise ValueError('All model/input tensors must be on the explicitly selected device')
    if not bool(torch.isfinite(noise).all() and torch.isfinite(C).all()):
        raise ValueError('Finite Gaussian/C required')
    if initial_state is not None:
        if initial_state.device != noise.device or initial_state.dtype != torch.float32 or initial_state.shape != noise.shape:
            raise ValueError('Saved suffix state must preserve device/dtype/slot shape')
        if not bool(torch.isfinite(initial_state).all()) or not _bits_equal(initial_state[0,:12], C.reshape(12,9)):
            raise ValueError('Saved suffix state must be finite and preserve C bits')


@torch.no_grad()
def sample(model, noise, C, *, nfe, precision, counts=None, initial_state=None,
           start_step=0, cpu_mock=False, observe_dtypes=True):
    """Return raw[112,9], path[nfe-start_step+1,112,9], audit.

    ``noise`` always denotes the registered Gaussian, including for suffixes;
    suffixes start exclusively from an independent clone of ``initial_state``.
    No Gaussian draw, OT, context cache or geometry callback occurs here.
    """
    if not cpu_mock:
        configure_stable_runtime()
    _validate(model, noise, C, nfe, precision, initial_state, start_step, cpu_mock)
    if counts is None:
        counts = new_counts()
    for key in new_counts():
        counts.setdefault(key, 0)
    is_suffix = initial_state is not None
    counts['suffix_attempts' if is_suffix else 'full_rollout_attempts'] += 1
    before_forward = counts['model_forward_attempts']
    total_start = time.perf_counter()
    C = C.reshape(12,9)
    source_hash, condition_hash = array_hash(noise), array_hash(C)
    initial_hash = array_hash(initial_state) if is_suffix else None
    state = initial_state.clone() if is_suffix else noise.clone()
    if not is_suffix:
        state[:,:12] = C[None]
    known = torch.zeros((1,112), device=noise.device, dtype=torch.bool);known[:,:12] = True
    valid = torch.ones_like(known);y = torch.tensor([112], device=noise.device)
    checks, times, path, events = [], [], [], []
    sync_seconds = copy_seconds = forward_host_seconds = 0.0
    recorder = None
    def synchronize():
        nonlocal sync_seconds
        if not cpu_mock:
            tick = time.perf_counter();torch.cuda.synchronize(noise.device);sync_seconds += time.perf_counter()-tick
    def save_state(value):
        nonlocal copy_seconds
        tick = time.perf_counter();path.append(value[0].cpu().numpy().copy());copy_seconds += time.perf_counter()-tick
    synchronize()
    if not cpu_mock:
        torch.cuda.reset_peak_memory_stats(noise.device)
    save_state(state)
    runtime_factory = nullcontext(dict(cpu_mock=True, native_precision=precision)) if cpu_mock else native_execution(
        model, coordinates=state, autocast=precision == 'bf16')
    dtype_observations = {}
    try:
        if observe_dtypes:
            recorder = DTypeRecorder(model)
        with runtime_factory as runtime:
            for k in range(start_step, nfe):
                if counts['model_forward_attempts']-before_forward >= nfe-start_step:
                    raise RuntimeError('Registered suffix/full forward budget exceeded')
                if not _bits_equal(state[0,:12], C):
                    raise RuntimeError('C bits changed before model forward')
                t = torch.full((1,), k/nfe, device=noise.device, dtype=torch.float32)
                # Absolute times are independently constructed, never accumulated or remapped.
                times.append(float(np.float32(k/nfe)))
                if not cpu_mock:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                tick = time.perf_counter();counts['model_forward_attempts'] += 1
                velocity = model(state, t, y, valid, known)
                counts['model_forward_returns'] += 1;forward_host_seconds += time.perf_counter()-tick
                if not cpu_mock:
                    end.record();events.append((begin,end))
                if velocity.requires_grad or velocity.shape != state.shape:
                    raise RuntimeError('Invalid sampling velocity graph/shape')
                if recorder is not None:
                    dtype_observations = recorder.finish(velocity, precision);recorder = None
                v = velocity.float()
                next_state = state.clone()
                next_state[:,12:] = state[:,12:] + v[:,12:] * (1/nfe)
                next_state[:,:12] = C[None]
                state = next_state;counts['euler_updates'] += 1
                ok = _bits_equal(state[0,:12], C);checks.append(ok)
                if not ok or state.dtype != torch.float32 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError('Invalid clamped FP32 Euler state')
                save_state(state)
        synchronize()
        forward_gpu_ms = None if cpu_mock else sum(begin.elapsed_time(end) for begin,end in events)
        expected = nfe-start_step
        if counts['model_forward_attempts']-before_forward != expected:
            raise RuntimeError('Actual forward count differs from registered interval')
        if array_hash(noise)!=source_hash or array_hash(C)!=condition_hash or (is_suffix and array_hash(initial_state)!=initial_hash):
            raise RuntimeError('Sampling mutated registered input')
        stacked = np.stack(path)
        expected_C = np.broadcast_to(C.cpu().numpy(), (len(path),12,9))
        if not np.array_equal(stacked[:,:12].view(np.uint32), expected_C.copy().view(np.uint32)):
            raise RuntimeError('Saved state C-bit preservation failed')
        counts['suffix_returns' if is_suffix else 'complete_rollouts'] += 1
        audit = dict(nfe=nfe, precision=precision, start_step=start_step, end_step=nfe,
                     integration_steps=expected, full_from_noise=not is_suffix,
                     state_indices=list(range(start_step,nfe+1)), absolute_time_values_FP32=times,
                     dt=1/nfe, runtime=runtime, actual_dtypes=dtype_observations,
                     known_all_states_bitwise=True, known_state_checks=len(path),
                     known_slots=list(range(12)), free_slots=list(range(12,112)),
                     total_faces=112, valid_faces=112, integration_dtype='torch.float32',
                     noise_sha256=source_hash, condition_sha256=condition_hash,
                     initial_state_sha256=initial_hash, free_updated=not np.array_equal(stacked[0,12:],stacked[-1,12:]),
                     seconds=time.perf_counter()-total_start,
                     model_forward_host_wall_seconds=forward_host_seconds,
                     model_forward_cuda_event_seconds=None if forward_gpu_ms is None else forward_gpu_ms/1000,
                     explicit_cuda_synchronize_seconds=sync_seconds, state_CPU_copy_seconds=copy_seconds,
                     timing_note='Host forward wall overlaps CUDA execution/synchronization; do not add timing columns. Model internal synchronization is included in host forward wall.',
                     peak_cuda_allocated=0 if cpu_mock else torch.cuda.max_memory_allocated(noise.device),
                     peak_cuda_reserved=0 if cpu_mock else torch.cuda.max_memory_reserved(noise.device),
                     memory_scope='Per-trajectory peak, including already resident model; context/process VRAM is measured by orchestrator',
                     model_backward_calls=0, optimizer_updates=0, geometry_gradient_calls=0,
                     OT_calls=0, postprocessing=None, cpu_mock=cpu_mock)
        return stacked[-1].copy(), stacked, audit
    except BaseException as exc:
        counts['sampling_failures'] += 1
        exc.partial_path = np.stack(path)
        exc.partial_state_indices = list(range(start_step,start_step+len(path)))
        raise
    finally:
        if recorder is not None:
            recorder.close()
