"""Session C fixed validation teacher: CPU pairing, one audited forward.

The caller freezes eight A validation cases after DATA_READY and owns persistence,
GPU locking and the 24-forward budget. No RNG, training loss, optimizer, gradient
or complete generation is invoked here. Pairing is never used for generation.
"""
from native_t1 import runtime as _runtime  # Configure CUBLAS before torch/CUDA.

import hashlib
import json
import time
import numpy as np
import torch

from native_t1.artifacts import array_hash
from native_t1.model import native_execution, assert_native_runtime
from .eval_runtime import _validate_model

MODEL_ARRAYS = ('xt', 't', 'y', 'valid_mask', 'known_mask')


def _hash_arrays(arrays, keys):
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode('utf-8'))
        digest.update(array_hash(arrays[key]).encode('ascii'))
    return digest.hexdigest()


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class TeacherPreparationFailure(RuntimeError):
    def __init__(self, reason, identity, arrays):
        super().__init__(reason)
        self.identity = identity
        self.arrays = arrays


class TeacherFailure(RuntimeError):
    def __init__(self, reason, row, arrays):
        super().__init__(reason)
        self.row = row
        self.arrays = arrays


def prepare_teacher_sample(sample, gaussian64):
    """Return (arrays, identity); one CPU original free-only OT, no random draw.

    `sample` is A's supervision-only load_sample result. `gaussian64` must be
    the registered validation Gaussian already drawn by A's canonical helper.
    Small N is accepted for CPU fixtures; the real MVP model enforces N128..256.
    Known faces remain first, with no teacher-specific face/corner permutation.
    """
    N, K = int(sample['N']), int(sample['K'])
    if not 0 < K < N:
        raise ValueError('Require 0<K<N')
    C = np.asarray(sample['constraints'])
    target = np.asarray(sample['free_target'])
    pre = np.asarray(sample['pre_ot_free'])
    noise = np.asarray(gaussian64)
    L = float(sample['bbox_diagonal'])
    if C.dtype != np.float32 or C.shape != (K, 3, 3):
        raise ValueError('A constraints must be FP32[K,3,3]')
    if target.dtype != np.float32 or target.shape != (N-K, 9):
        raise ValueError('A free_target must be FP32[N-K,9]')
    if pre.dtype not in (np.float32, np.float64) or pre.shape != (N-K, 3, 3):
        raise ValueError('Require original saved pre_ot_free, never target/2')
    if noise.dtype != np.float64 or noise.shape != (N-K, 3, 3):
        raise ValueError('Registered teacher Gaussian must be FP64[N-K,3,3]')
    if not all(np.isfinite(a).all() for a in (C, target, pre, noise)) or not np.isfinite(L) or L <= 0:
        raise ValueError('Finite teacher inputs and positive bbox required')
    if not _same_bits((pre*2).astype(np.float32).reshape(N-K, 9), target):
        raise ValueError('Saved pre-OT x2 identity differs from A target')
    arrays = dict(C=C.copy(), free_target=target.copy(), pre_ot_free=pre.copy(),
                  raw_gaussian=noise.copy())
    identity = dict(schema='control_mvp48_teacher_pair_v1',
        uid=str(sample.get('uid', sample.get('object_id', ''))),
        patch_id=str(sample.get('patch_id', '')), N=N, K=K, bbox_L=L,
        teacher_t=.5, state_order='known first, A corner order, ordered free complement',
        actual_OT_attempts=0, actual_OT_returns=0, random_draws=0,
        pairing='original optimal_sum_numpy(pre_ot_free, gaussian64, optimal=True); reorder Gaussian only',
        model_inputs=list(MODEL_ARRAYS), direct_GT_labels_are_model_inputs=False,
        teacher_state_contains_GT=True, generation_pairing=False,
        model_forwards=0, training_loss_calls=0, geometry_gradient_calls=0)
    try:
        from utils.ot_utils import optimal_sum_numpy
        identity['actual_OT_attempts'] += 1
        paired = optimal_sum_numpy(arrays['pre_ot_free'], arrays['raw_gaussian'], optimal=True)
        identity['actual_OT_returns'] += 1
        if paired.shape != noise.shape or paired.dtype != np.float64 or not np.isfinite(paired).all():
            raise RuntimeError('Original OT returned malformed or nonfinite pairing')
        if not _same_bits(arrays['raw_gaussian'], noise) or not _same_bits(arrays['pre_ot_free'], pre):
            raise RuntimeError('OT mutated its source arrays')
        arrays['paired_noise'] = paired.copy()
        x0 = np.zeros((N, 9), dtype=np.float32)
        x1 = np.zeros_like(x0)
        context = np.zeros_like(x0)
        x0[K:] = paired.astype(np.float32).reshape(N-K, 9)
        x1[K:] = target
        context[:K] = C.reshape(K, 9)
        t = np.asarray(.5, dtype=np.float32)
        xt = (np.float32(1)-t)*x0 + t*x1
        xt[:K] = context[:K]
        arrays.update(x0=x0, x1=x1, u=x1-x0, xt=xt, t=t,
                      y=np.asarray(N, dtype=np.int64), context=context,
                      known_mask=np.arange(N) < K, valid_mask=np.ones(N, dtype=np.bool_))
        identity.update(status='PASS', array_sha256={k: array_hash(v) for k, v in arrays.items()},
                        model_input_sha256=_hash_arrays(arrays, MODEL_ARRAYS),
                        pairing_sha256=_hash_arrays(arrays, ('pre_ot_free', 'raw_gaussian', 'paired_noise', 'x1', 'u')))
        json.dumps(identity, allow_nan=False)
        return arrays, identity
    except BaseException as exc:
        identity.update(status='FAIL', reason=repr(exc))
        raise TeacherPreparationFailure(str(exc), identity, arrays) from exc


def _unpack_bank(bank):
    if isinstance(bank, (tuple, list)) and len(bank) == 2:
        arrays, identity = bank
    else:
        arrays, identity = bank['arrays'], bank['identity']
    if identity.get('status') != 'PASS' or identity.get('schema') != 'control_mvp48_teacher_pair_v1':
        raise ValueError('Require successfully prepared fixed teacher pair')
    if set(arrays) != set(identity['array_sha256']):
        raise ValueError('Teacher array keys changed')
    for key, value in arrays.items():
        if array_hash(value) != identity['array_sha256'][key]:
            raise ValueError('Teacher array identity changed: '+key)
    if _hash_arrays(arrays, MODEL_ARRAYS) != identity['model_input_sha256']:
        raise ValueError('Teacher model input identity changed')
    if _hash_arrays(arrays, ('pre_ot_free', 'raw_gaussian', 'paired_noise', 'x1', 'u')) != identity['pairing_sha256']:
        raise ValueError('Teacher pairing identity changed')
    N, K = identity['N'], identity['K']
    if (arrays['xt'].shape != (N, 9) or arrays['xt'].dtype != np.float32
            or int(arrays['y']) != N or float(arrays['t']) != .5
            or not np.array_equal(arrays['known_mask'], np.arange(N) < K)
            or not bool(arrays['valid_mask'].all())
            or not _same_bits(arrays['xt'][:K], arrays['C'].reshape(K, 9))):
        raise ValueError('Teacher N/time/mask/C contract changed')
    return arrays, identity


def original_teacher_geometry(triangles, bbox):
    """Original teacher._geometry formula on CPU FP32, without loss imports."""
    x = torch.as_tensor(np.array(triangles, copy=True)).detach().float().reshape(-1, 3, 3)
    finite = torch.isfinite(x)
    if not bool(finite.all()):
        return dict(finite=False, NaN=int(torch.isnan(x).sum()), Inf=int(torch.isinf(x).sum()))
    y = x/float(bbox)
    area = torch.linalg.vector_norm(torch.linalg.cross(y[:, 1]-y[:, 0], y[:, 2]-y[:, 0], dim=-1), dim=-1)*.5
    return dict(finite=True, NaN=0, Inf=0, faces=len(x), total_area=float(area.sum())*float(bbox)**2,
        normalized_total_area=float(area.sum()), minimum_normalized_area=float(area.min()),
        zero_area_faces=int((area == 0).sum()), near_degenerate_faces=int((area <= 1e-12).sum()),
        near_degenerate_normalized_area_threshold=1e-12)


def offline_teacher_diagnostics(arrays, identity, velocity):
    """Read-only CPU FP32 FM error and endpoint geometry; no objective call."""
    v = torch.from_numpy(np.array(velocity, copy=True))
    if v.dtype != torch.float32 or tuple(v.shape) != (identity['N'], 9):
        raise ValueError('One FP32[N,9] teacher velocity required')
    if not bool(torch.isfinite(v).all()):
        raise FloatingPointError('Nonfinite teacher velocity')
    xt = torch.from_numpy(np.array(arrays['xt'], copy=True))
    u = torch.from_numpy(np.array(arrays['u'], copy=True))
    known = torch.from_numpy(np.array(arrays['known_mask'], copy=True))
    valid = torch.from_numpy(np.array(arrays['valid_mask'], copy=True))
    context = torch.from_numpy(np.array(arrays['context'], copy=True))
    free = valid & ~known
    denominator = int(free.sum())*9
    with torch.no_grad():
        error = (v-u).square()[free].sum()/denominator
        endpoint = xt + (torch.tensor(1., dtype=torch.float32)-torch.from_numpy(np.array(arrays['t'], copy=True)))*v
        endpoint[known] = context[known]
    if not bool(torch.isfinite(error)) or not bool(torch.isfinite(endpoint).all()):
        raise FloatingPointError('Nonfinite offline FM error or teacher endpoint')
    result = dict(FM_error=float(error), FM_coordinate_denominator=denominator,
        FM_definition='CPU FP32 sum squared free velocity error / (9*valid free faces)',
        training_loss_calls=0, geometry_gradient_calls=0,
        geometry=dict(Yhat_free=original_teacher_geometry(endpoint[free].numpy(), identity['bbox_L']),
                      GT_free=original_teacher_geometry(np.asarray(arrays['x1'])[free.numpy()], identity['bbox_L'])),
        known_preserved=_same_bits(endpoint[known].numpy(), np.asarray(arrays['C']).reshape(identity['K'], 9)))
    return result, endpoint.numpy().copy()


@torch.no_grad()
def evaluate_teacher_once(model, bank, counts):
    """Return (row, output_arrays); exactly one forward, never retry on failure.

    TeacherFailure carries partial .row/.arrays; the caller must persist these
    and the shared counters. All model arguments use the same input allowlist as
    generation; teacher GT/OT enters the precomputed teacher state only.
    """
    arrays, identity = _unpack_bank(bank)
    _runtime.configure_stable_runtime()
    _validate_model(model)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Formal teacher requires the CUDA BF16 runtime')
    initial_counts = dict(counts)
    row = dict(status='RUNNING', uid=identity['uid'], patch_id=identity['patch_id'],
        N=identity['N'], K=identity['K'], teacher_t=.5, bbox_L=identity['bbox_L'],
        model_input_sha256=identity['model_input_sha256'], pairing_sha256=identity['pairing_sha256'],
        forward_dtype='bfloat16', coordinate_dtype='float32', no_grad=True,
        GT_labels_model_inputs=False, teacher_state_contains_GT=True,
        training_loss_calls=0, OT_calls=0, optimizer_updates=0, model_backward_calls=0,
        geometry_gradient_calls=0, geometry_repairs=0, known_raw_velocity_forced_zero=False)
    output_arrays = {}
    handles = []
    tick = time.perf_counter()
    def before(module, args):
        if counts['teacher_attempts'] - initial_counts['teacher_attempts'] >= 1:
            raise RuntimeError('One-forward teacher budget exceeded')
        counts['teacher_attempts'] += 1
        counts['model_forward_attempts'] += 1
    def after(module, args, output):
        counts['teacher_returns'] += 1
        counts['model_forward_returns'] += 1
        if output.requires_grad:
            raise RuntimeError('Unexpected teacher autograd graph')
    def backbone_start(module, args):
        if counts['backbone_forward_attempts'] - initial_counts['backbone_forward_attempts'] >= 1:
            raise RuntimeError('One-backbone teacher budget exceeded')
        counts['backbone_forward_attempts'] += 1
    def backbone_end(module, args, output):
        counts['backbone_forward_returns'] += 1
    try:
        handles.append(model.register_forward_pre_hook(before))
        handles.append(model.register_forward_hook(after))
        handles.append(model.backbone.x_embedder.register_forward_pre_hook(backbone_start))
        handles.append(model.backbone.final_layer.linear.register_forward_hook(backbone_end))
        inputs = {k: torch.from_numpy(np.array(arrays[k], copy=True)).unsqueeze(0).to(device) for k in MODEL_ARRAYS}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        forward_tick = time.perf_counter()
        with native_execution(model, coordinates=(inputs['xt'], inputs['t'])) as facts:
            row['runtime_before'] = facts
            velocity = model(*(inputs[k] for k in MODEL_ARRAYS))
            row['actual_velocity_output_dtype'] = str(velocity.dtype)
            if velocity.shape != inputs['xt'].shape:
                raise RuntimeError('Malformed teacher velocity')
            output_arrays['velocity'] = velocity.float()[0].cpu().numpy().copy()
            row['runtime_after'] = assert_native_runtime(model, coordinates=(inputs['xt'], inputs['t']))
        torch.cuda.synchronize(device)
        row['network_seconds'] = time.perf_counter()-forward_tick
        for key in MODEL_ARRAYS:
            actual = inputs[key][0].cpu().numpy().copy()
            if array_hash(actual) != identity['array_sha256'][key]:
                raise RuntimeError('Teacher model mutated input: '+key)
        output_arrays['actual_Xt'] = inputs['xt'][0].cpu().numpy().copy()
        raw_velocity = output_arrays['velocity']
        row.update(finite_velocity=bool(np.isfinite(raw_velocity).all()),
                   NaN_velocity_count=int(np.isnan(raw_velocity).sum()),
                   Inf_velocity_count=int(np.isinf(raw_velocity).sum()))
        offline_tick = time.perf_counter()
        diagnostics, endpoint = offline_teacher_diagnostics(arrays, identity, raw_velocity)
        output_arrays['Yhat'] = endpoint
        row.update(diagnostics, offline_diagnostic_seconds=time.perf_counter()-offline_tick)
        for key in ('teacher_attempts', 'teacher_returns', 'model_forward_attempts',
                    'model_forward_returns', 'backbone_forward_attempts', 'backbone_forward_returns'):
            if counts[key] - initial_counts[key] != 1:
                raise RuntimeError('Expected exactly one call: '+key)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Teacher created stored parameter gradients')
        _unpack_bank((arrays, identity))
        row.update(status='PASS', output_array_sha256={k: array_hash(v) for k, v in output_arrays.items()})
        return row, output_arrays
    except BaseException as exc:
        row.update(status='FAIL', reason=repr(exc),
                   output_array_sha256={k: array_hash(v) for k, v in output_arrays.items()})
        raise TeacherFailure(str(exc), row, output_arrays) from exc
    finally:
        for handle in handles:
            handle.remove()
        row.update(seconds=time.perf_counter()-tick,
                   nfe=counts['backbone_forward_attempts']-initial_counts['backbone_forward_attempts'],
                   counts_delta={k: counts[k]-initial_counts[k] for k in counts},
                   peak_scope='One teacher forward after reset; resident model and input included')
        try:
            torch.cuda.synchronize(device)
            row.update(seconds=time.perf_counter()-tick,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        except BaseException as exc:
            row['final_memory_observation_error'] = repr(exc)
