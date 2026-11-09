"""Fixed teacher states and read-only velocity/one-layer gradient diagnostics.

No model is constructed here. All scientific calls require an explicit caller;
the caller owns A -> B -> C -> D ordering and GPU dispatch. The only OT occurs
in build_teacher_bank, once per independent pair, never in evaluation/reload.
"""
from native_t1 import runtime as _runtime  # Before torch/CUDA initialization.

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from native_t1.artifacts import atomic_json, file_sha256
from native_t1.data import (NativeTrainingStream, load_archive, native_collate,
                            model_inputs, shared_sample_hash, array_hash)
from native_t1.training import masked_fm_loss
from native_t1.losses.edge import prepare_edge_target, edge_loss
from native_t1.losses.surface import prepare_distribution_target, distribution_loss, work_counters
from native_t1.losses import LAMBDA_EDGE, LAMBDA_L5
from native_t1.model import native_execution

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / 'output/t1_publish_20260930/chair_n112.npz'
INPUT_SHA = '7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
TIMES = (.1, .3, .5, .75, .9)
TEACHER_SEED = 20261002
PARAMETER_SHAPES = {'backbone.layers.8.mlp.net.2.weight': (768, 3072),
                    'backbone.layers.8.mlp.net.2.bias': (768,)}
PAIR_ARRAYS = ('x0', 'x1', 'u', 'y', 'known_mask', 'valid_mask', 'context',
    'context_by_parent', 'known_target_ids', 'source_face_ids', 'corner_permutations',
    'face_permutation', 'free_gaussian_before_OT', 'free_noise_after_OT', 'free_target_before_OT')
SPEC = dict(schema='jedge5_fixed_teacher_v1', teacher_seed=TEACHER_SEED,
    parents=[0, 1], patches=list(range(5)), noises_per_context=2, times=list(TIMES),
    fixed_pairs=20, fixed_states=100, original_free_only_OT_calls=20,
    state_order='parent,patch,noise,time; one private original NativeTrainingStream',
    unused_original_time_draws=20, original_OT_reorders='Z only',
    single_sample_FM_denominator='valid free faces times 9; no effective-batch divisor',
    geometry_denominator='original per-sample Edge/L5, no extra division by eight',
    lambda_edge=LAMBDA_EDGE, lambda_surface=LAMBDA_L5, gate='original FP32 t>=0.5',
    diagnostic_ramp=1., raw_geometry_gradients_computed_also_below_gate=True,
    parameter_precision='bf16', parameter_samples='P0/P1 x patch0/4 x first teacher noise x t=.75',
    parameter_names_and_shapes={k: list(v) for k, v in PARAMETER_SHAPES.items()},
    parameter_source='models/equidit.py FeedForward(silu), DiTLayer mlp_ratio=4, DiT.layers; ninth layer index8',
    parameter_gradient_scope='Three VJPs for the two tensors of one linear layer; not an all-network gradient scan',
    generator_inputs=['xt', 't', 'y', 'valid_mask', 'known_mask'], GT_labels_are_model_inputs=False)


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _plain(value):
    if torch.is_tensor(value):
        return _plain(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NONFINITE:' + str(value)
    return value


def _increment(counts, ledger, key):
    counts[key] = counts.get(key, 0) + 1
    if ledger is not None and ledger is not counts:
        ledger[key] = ledger.get(key, 0) + 1


def _state_at_time(base, value):
    """Do not draw noise, permute Y, or call OT for a new time."""
    sample = dict(base)
    sample['t'] = np.asarray(value, dtype=np.float32)
    sample['xt'] = (np.float32(1)-sample['t'])*sample['x0'] + sample['t']*sample['x1']
    sample['xt'][sample['known_mask']] = sample['context'][sample['known_mask']]
    sample['shared_input_sha256'] = shared_sample_hash(sample)
    sample['context_sha256'] = array_hash(sample['context'])
    sample['teacher_t'] = float(value)
    sample['state_id'] = sample['pair_id'] + f'_t{value:g}'
    return sample


def _cases(cases):
    if cases is None:
        from .common import load_cases
        cases = load_cases()
    if set(cases) != {(p, j) for p in (0, 1) for j in range(5)}:
        raise ValueError('All ten frozen contexts are required')
    return cases


def build_teacher_bank(out, *, cases=None, teacher_seed=TEACHER_SEED):
    """Explicit CPU-only 20-OT bank construction; never advance a training stream."""
    if teacher_seed != TEACHER_SEED:
        raise ValueError('Registered teacher RNG seed changed')
    folder = Path(out) / 'stage_c'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'teacher_bank.npz'
    manifest = folder / 'teacher_bank.json'
    if destination.exists() or manifest.exists():
        raise FileExistsError('Teacher bank already exists; reload without repeating OT')
    cases = _cases(cases)
    archive = load_archive(INPUT, INPUT_SHA)
    stream = NativeTrainingStream(archive, seed=teacher_seed)
    start_state = stream.state_dict()
    counts = {}
    report = dict(status='RUNNING', spec=SPEC, pairs=[], states=[], arrays={}, counts=counts,
        input=dict(path=str(INPUT), sha256=INPUT_SHA), formal_training_stream_accessed=False,
        independent_stream_start_sha256=_seal(start_state), model_forwards=0, model_backwards=0,
        optimizer_updates=0, CUDA_initialized=torch.cuda.is_initialized())
    arrays, samples = {}, []
    start = time.perf_counter()
    try:
        from native_t1.accounting import count_ot_calls
        with count_ot_calls(counts):
            for parent in (0, 1):
                for patch in range(5):
                    # Both possible donors use this same patch index. Actual
                    # teacher contexts are always correctly matched to parent.
                    for donor in (0, 1):
                        stream.cases[donor, 12] = cases[donor, patch]['case']
                    for noise in range(2):
                        base = stream._sample(parent, 12)
                        pair_index = len(report['pairs'])
                        pair_id = f'P{parent}_patch{patch}_n{noise}'
                        base.update(pair_id=pair_id, pair_index=pair_index, patch_index=patch,
                                    noise_index=noise, teacher_seed=teacher_seed)
                        pair = dict(pair_id=pair_id, pair_index=pair_index, parent_index=parent,
                            patch_index=patch, noise_index=noise, teacher_seed=teacher_seed,
                            original_random_time_draw_unused=float(base['t']), arrays={})
                        for key in PAIR_ARRAYS:
                            name = f'pair{pair_index:02d}_{key}'
                            arrays[name] = np.array(base[key], copy=True)
                            pair['arrays'][key] = name
                        report['pairs'].append(pair)
                        for value in TIMES:
                            state = _state_at_time(base, value)
                            index = len(samples)
                            for key in ('xt', 't'):
                                arrays[f'state{index:03d}_{key}'] = np.array(state[key], copy=True)
                            report['states'].append(dict(state_id=state['state_id'], index=index,
                                pair_index=pair_index, teacher_t=value,
                                actual_FP32_t=float(state['t']), shared_input_sha256=state['shared_input_sha256'],
                                context_sha256=state['context_sha256']))
                            samples.append(state)
        if counts != dict(OT_calls=20, OT_returns=20) or len(samples) != 100:
            raise AssertionError('Teacher bank OT/state budget changed')
        for key, value in arrays.items():
            report['arrays'][key] = dict(sha256=array_hash(value), shape=list(value.shape), dtype=str(value.dtype))
        np.savez_compressed(destination, **arrays)
        report.update(status='PASS', artifact=dict(path=str(destination.resolve()), sha256=file_sha256(destination),
            bytes=destination.stat().st_size), independent_stream_end_sha256=_seal(stream.state_dict()),
            independent_stream_samples=stream.sample_index)
    except BaseException as error:
        report.update(status='STOPPED', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter()-start
        atomic_json(manifest, _plain(report))
    return samples, report


def load_teacher_bank(out):
    folder = Path(out) / 'stage_c'
    report = json.loads((folder / 'teacher_bank.json').read_text(encoding='utf-8'))
    path = folder / 'teacher_bank.npz'
    if report.get('status') != 'PASS' or report.get('spec') != SPEC or file_sha256(path) != report['artifact']['sha256']:
        raise ValueError('Fixed teacher bank identity changed')
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(report['arrays']):
            raise ValueError('Teacher bank keys changed')
        arrays = {key: stored[key].copy() for key in stored.files}
    for key, value in arrays.items():
        ref = report['arrays'][key]
        if array_hash(value) != ref['sha256'] or list(value.shape) != ref['shape'] or str(value.dtype) != ref['dtype']:
            raise ValueError('Teacher bank array changed: ' + key)
    samples = []
    for row in report['states']:
        pair = report['pairs'][row['pair_index']]
        base = {key: arrays[name].copy() for key, name in pair['arrays'].items()}
        base.update(parent_index=pair['parent_index'], donor_index=pair['parent_index'], K=12,
            pair_id=pair['pair_id'], pair_index=pair['pair_index'], patch_index=pair['patch_index'],
            noise_index=pair['noise_index'], teacher_seed=TEACHER_SEED)
        sample = _state_at_time(base, row['teacher_t'])
        for key in ('xt', 't'):
            saved = arrays[f"state{row['index']:03d}_{key}"]
            if saved.dtype != sample[key].dtype or saved.tobytes() != sample[key].tobytes():
                raise ValueError('Teacher time reconstruction differs')
        if sample['state_id'] != row['state_id'] or sample['shared_input_sha256'] != row['shared_input_sha256']:
            raise ValueError('Teacher state identity differs')
        samples.append(sample)
    _validate_samples(samples, count=100)
    return samples, report


def _validate_samples(samples, *, count):
    if len(samples) != count or len({s['state_id'] for s in samples}) != count:
        raise ValueError('Unexpected fixed-state count or duplicate state')
    expected_patches = range(5) if count == 100 else (0, 4)
    expected_noises = range(2) if count == 100 else (0,)
    expected_times = TIMES if count in (100, 20) else (.75,)
    expected = {(p, j, n, t) for p in (0, 1) for j in expected_patches
                for n in expected_noises for t in expected_times}
    actual = {(s['parent_index'], s['patch_index'], s['noise_index'], s['teacher_t']) for s in samples}
    if actual != expected:
        raise ValueError('Teacher context/noise/time grid differs')


def curve_subset(samples):
    answer = [s for s in samples if s['patch_index'] in (0, 4) and s['noise_index'] == 0]
    _validate_samples(answer, count=20)
    return answer


def _label(sample, case):
    known, valid = np.asarray(sample['known_mask']), np.asarray(sample['valid_mask'])
    ids, corners = np.asarray(sample['source_face_ids']), np.asarray(sample['corner_permutations'])
    F = np.asarray(case['source_face_vertex_ids'])
    full = np.asarray(case['full']).reshape(112, 3, 3)
    if (F.shape != (112, 3) or F.dtype.kind not in 'iu' or sorted(ids.tolist()) != list(range(112))
            or not valid.all() or known.sum() != 12 or sample['donor_index'] != sample['parent_index']):
        raise ValueError('Trusted teacher source mapping differs')
    if corners.shape != (112, 3) or not np.array_equal(np.sort(corners, axis=1), np.broadcast_to(np.arange(3), (112, 3))):
        raise ValueError('Invalid actual corner permutations')
    vids = np.take_along_axis(F[ids], corners, axis=1)
    target = full[ids][np.arange(112)[:, None], corners]
    Y = sample['x1'].reshape(112, 3, 3)
    C = sample['context'].reshape(112, 3, 3)
    if target[~known].tobytes() != Y[~known].tobytes() or target[known].tobytes() != C[known].tobytes():
        raise ValueError('Source labels are not aligned with fixed target/condition')
    return dict(free_gt=Y[~known].copy(), known=C[known].copy(), free_ids=vids[~known].copy(),
        known_ids=vids[known].copy(), bbox_L=float(case['bbox']),
        mapping_sha256=_seal({k: array_hash(v) for k, v in dict(ids=vids, Y=Y[~known], C=C[known], mask=known).items()}))


def make_target_bank(samples, cases=None):
    """Cache fixed GT targets only, per pair; never cache network conditions."""
    cases = _cases(cases)
    result = {}
    for sample in samples:
        key = sample['pair_id']
        if key in result:
            continue
        label = _label(sample, cases[sample['parent_index'], sample['patch_index']])
        result[key] = dict(edge=prepare_edge_target(label),
            surface=prepare_distribution_target(torch.from_numpy(label['free_gt']), label['bbox_L']),
            bbox_L=label['bbox_L'], mapping_sha256=label['mapping_sha256'])
    return result


def loss_terms(velocity, batch, target):
    """Unchanged per-sample formulas. No model call or derivative is implicit."""
    if velocity.dtype not in (torch.float32, torch.float64) or velocity.ndim != 3 or velocity.shape[0] != 1:
        raise ValueError('Exactly one velocity sample required')
    free = batch['valid_mask'] & ~batch['known_mask']
    if not bool(free.any()):
        raise ValueError('No valid free coordinates')
    with torch.autocast(device_type=velocity.device.type, enabled=False):
        t = batch['t'].detach()
        endpoint = torch.where(batch['known_mask'][..., None], batch['context'].detach(),
            batch['xt'].detach() + (1-t[:, None, None])*velocity)
        prediction = endpoint[free].reshape(-1, 3, 3)
        if not bool(torch.isfinite(endpoint).all()):
            raise FloatingPointError('Nonfinite predicted endpoint')
        fm = masked_fm_loss(velocity, batch['u'].detach(), batch['valid_mask'], batch['known_mask'])
        edge = edge_loss(prediction, target['edge'])
        surface = distribution_loss('L5', prediction, target['surface'])
        active = bool(t[0] >= .5)
        joint = LAMBDA_EDGE*edge['loss'] + LAMBDA_L5*surface['loss']
        if not all(bool(torch.isfinite(v)) for v in (fm, edge['loss'], surface['loss'], joint)):
            raise FloatingPointError('Nonfinite teacher loss')
    return dict(FM=fm, Edge=edge['loss'], Surface=surface['loss'], joint_raw=joint,
        joint_actual=joint*int(active), active=active, endpoint=endpoint,
        edge_record={k: v for k, v in edge.items() if k != 'loss'},
        surface_record={k: v for k, v in surface.items() if k != 'loss'}, free=free,
        FM_coordinate_denominator=int(free.sum())*9)


def _norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()))


def _cosine(a, b):
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return dict(value=None, status='ZERO_NORM', angle_degrees=None)
    value = float((a.detach().double().reshape(-1)*b.detach().double().reshape(-1)).sum())/(na*nb)
    return dict(value=value, status='AVAILABLE', angle_degrees=math.degrees(math.acos(max(-1., min(1., value)))))


def gradient_summary(fm, edge, surface, *, active):
    if any(not bool(torch.isfinite(v).all()) for v in (fm, edge, surface)):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    joint = LAMBDA_EDGE*edge + LAMBDA_L5*surface
    norms = {name: _norm(value) for name, value in dict(FM=fm, Edge=edge, Surface=surface, joint_raw=joint).items()}
    norms['joint_actual'] = norms['joint_raw'] if active else 0.
    return dict(norms=norms, gradient_elements=fm.numel(),
        joint_raw_over_FM=norms['joint_raw']/norms['FM'] if norms['FM'] else None,
        joint_actual_over_FM=norms['joint_actual']/norms['FM'] if norms['FM'] else None,
        ratio_status='AVAILABLE' if norms['FM'] else 'ZERO_FM_NORM',
        cosines=dict(FM_Edge=_cosine(fm, edge), FM_Surface=_cosine(fm, surface),
                     Edge_Surface=_cosine(edge, surface), FM_joint_raw=_cosine(fm, joint),
                     FM_joint_actual=_cosine(fm, joint if active else joint*0)),
        active=bool(active), joint_built_by_linear_combination=True, no_extra_joint_autograd=True)


def _geometry(triangles, bbox):
    x = triangles.detach().float()
    finite = torch.isfinite(x)
    if not bool(finite.all()):
        return dict(finite=False, NaN=int(torch.isnan(x).sum()), Inf=int(torch.isinf(x).sum()))
    y = x/float(bbox)
    area = torch.linalg.vector_norm(torch.linalg.cross(y[:, 1]-y[:, 0], y[:, 2]-y[:, 0], dim=-1), dim=-1)*.5
    return dict(finite=True, NaN=0, Inf=0, faces=len(x), total_area=float(area.sum())*float(bbox)**2,
        normalized_total_area=float(area.sum()), minimum_normalized_area=float(area.min()),
        zero_area_faces=int((area == 0).sum()), near_degenerate_faces=int((area <= 1e-12).sum()),
        near_degenerate_normalized_area_threshold=1e-12)


def _row(sample, terms, batch, target):
    free = terms['free']
    return dict(state_id=sample['state_id'], pair_id=sample['pair_id'], parent=sample['parent_index'],
        patch=sample['patch_index'], noise_index=sample['noise_index'], teacher_t=sample['teacher_t'],
        actual_FP32_t=float(batch['t'][0]), context_group='training_C' if sample['patch_index'] == 0 else 'other_observed_C',
        shared_input_sha256=sample['shared_input_sha256'], target_mapping_sha256=target['mapping_sha256'],
        losses={name: float(terms[name].detach()) for name in ('FM', 'Edge', 'Surface', 'joint_raw', 'joint_actual')},
        weighted_Edge_actual=int(terms['active'])*LAMBDA_EDGE*float(terms['Edge'].detach()),
        weighted_Surface_actual=int(terms['active'])*LAMBDA_L5*float(terms['Surface'].detach()),
        geometry_active=terms['active'], diagnostic_ramp=1.,
        FM_coordinate_denominator=terms['FM_coordinate_denominator'], geometry_extra_batch_divisor=1,
        geometry=dict(Yhat_free=_geometry(terms['endpoint'][free].reshape(-1, 3, 3), target['bbox_L']),
                      GT_free=_geometry(batch['x1'][free].reshape(-1, 3, 3), target['bbox_L'])),
        geometry_loss_records=dict(Edge=_plain(terms['edge_record']), Surface=_plain(terms['surface_record'])),
        known_preserved=torch.equal(terms['endpoint'][batch['known_mask']], batch['context'][batch['known_mask']]),
        B_geometry_interpretation='Same hypothetical J objective evaluated on B; B was trained with FM only')


@contextmanager
def _dtype_hooks(model, trace):
    targets = {'embedding': model.backbone.x_embedder,
        'QKV': model.backbone.layers[0].attn.qkv_proj,
        'attention_output': model.backbone.layers[0].attn,
        'MLP_output': model.backbone.layers[0].mlp.net[2]}
    handles = []
    def make_hook(name):
        def hook(module, args, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            trace[name] = [dict(dtype=str(x.dtype), shape=list(x.shape)) for x in values if torch.is_tensor(x)]
        return hook
    try:
        for name, module in targets.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _start_record(out, phase, label, kind):
    if not re.fullmatch(r'[A-Za-z0-9_]+', label) or phase not in ('C', 'D'):
        raise ValueError('Explicit safe label and phase C/D required')
    folder = Path(out) / ('stage_c' if phase == 'C' else 'stage_d') / 'teacher'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{label}_{kind}.json'
    arrays_path = folder / f'{label}_{kind}.npz'
    if path.exists() or arrays_path.exists():
        raise FileExistsError('No implicit diagnostic retry or duplicate model budget')
    record = dict(status='RUNNING', phase=phase, label=label, spec=SPEC, counts={}, rows=[],
                  optimizer_updates=0, records_are_readonly=True, timing={})
    atomic_json(path, record)
    return path, arrays_path, record


def _validate_dtype_trace(trace, precision):
    required = ('embedding', 'QKV', 'attention_output', 'MLP_output')
    if any(not trace.get(key) for key in required) or 'velocity_output' not in trace:
        raise AssertionError('Actual network dtype observation incomplete')
    entries = [item for key in required for item in trace[key]] + [trace['velocity_output']]
    if precision == 'fp32' and any(item['dtype'] != 'torch.float32' for item in entries):
        raise AssertionError('FP32 diagnostic retained a reduced-precision network operation')
    if precision == 'bf16' and not any(item['dtype'] == 'torch.bfloat16' for item in trace['QKV']):
        raise AssertionError('Registered BF16 path did not reach BF16 QKV')


def _finish_record(record, path, arrays_path, arrays, start, surface_before):
    if arrays:
        np.savez_compressed(arrays_path, **arrays)
        record['arrays'] = dict(path=str(arrays_path.resolve()), sha256=file_sha256(arrays_path),
                                bytes=arrays_path.stat().st_size, keys=len(arrays))
    now = work_counters()
    record['surface_kernel_work'] = {key: now[key]-surface_before[key] for key in now}
    record['timing']['wall_seconds'] = time.perf_counter()-start
    atomic_json(path, _plain(record))


def evaluate_teacher(model, samples, *, model_name, precision, out, ledger=None,
                     targets=None, cases=None, velocity_gradients=True, phase='C'):
    """100 C states per precision; D uses its predetermined 20-state subset."""
    _validate_samples(samples, count=100 if phase == 'C' else 20)
    if precision not in ('bf16', 'fp32') or (phase == 'D' and (precision != 'bf16' or velocity_gradients)):
        raise ValueError('Unregistered diagnostic precision or D gradient request')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Teacher inference requires an independently frozen model')
    path, arrays_path, record = _start_record(out, phase, model_name, precision)
    record.update(model=model_name, precision=precision, velocity_gradients=bool(velocity_gradients),
        expected_forwards=len(samples), model_parameter_gradients=False)
    arrays = {}
    counts = record['counts']
    start = time.perf_counter()
    surface_before = work_counters()
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    try:
        targets = make_target_bank(samples, cases) if targets is None else targets
        device = next(model.parameters()).device
        for index, sample in enumerate(samples):
            record['state_in_progress'] = sample['state_id']
            batch = native_collate([sample], device)
            trace = {}
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.no_grad(), native_execution(model, coordinates=batch['xt'], autocast=precision == 'bf16') as runtime:
                with _dtype_hooks(model, trace):
                    _increment(counts, ledger, 'teacher_forward_attempts')
                    velocity = model(*model_inputs(batch))
                    _increment(counts, ledger, 'teacher_forward_returns')
                    trace['velocity_output'] = dict(dtype=str(velocity.dtype), shape=list(velocity.shape))
                    velocity = velocity.float()
            _validate_dtype_trace(trace, precision)
            torch.cuda.synchronize(device)
            record['timing']['network_seconds'] = record['timing'].get('network_seconds', 0.) + time.perf_counter()-tick
            arrays[f'state{index:03d}_velocity'] = velocity.detach().cpu().numpy().copy()
            arrays[f'state{index:03d}_actual_Xt'] = batch['xt'].detach().cpu().numpy().copy()
            tick = time.perf_counter()
            target = targets[sample['pair_id']]
            with torch.enable_grad():
                leaf = velocity.detach().requires_grad_(velocity_gradients)
                _increment(counts, ledger, 'teacher_loss_attempts')
                terms = loss_terms(leaf, batch, target)
                _increment(counts, ledger, 'teacher_loss_returns')
                row = _row(sample, terms, batch, target)
                if velocity_gradients:
                    gradients = {}
                    for pos, name in enumerate(('FM', 'Edge', 'Surface')):
                        _increment(counts, ledger, 'velocity_leaf_autograd_attempts')
                        value = torch.autograd.grad(terms[name], leaf, retain_graph=pos < 2)[0]
                        _increment(counts, ledger, 'velocity_leaf_autograd_returns')
                        if bool((value[~terms['free']] != 0).any()):
                            raise AssertionError('Known or padding velocity received a geometry/FM gradient')
                        gradients[name] = value
                        arrays[f'state{index:03d}_g_{name}'] = value.detach().cpu().numpy().copy()
                    row['velocity_gradients'] = gradient_summary(*(gradients[name][terms['free']] for name in ('FM', 'Edge', 'Surface')),
                                                                active=terms['active'])
                else:
                    row['velocity_gradients'] = 'NOT_RUN_FIXED_LOSS_ONLY_CURVE'
            torch.cuda.synchronize(device)
            record['timing']['geometry_and_leaf_gradient_seconds'] = record['timing'].get('geometry_and_leaf_gradient_seconds', 0.) + time.perf_counter()-tick
            row.update(runtime=runtime, observed_dtypes=trace, velocity_array=f'state{index:03d}_velocity',
                       actual_Xt_array=f'state{index:03d}_actual_Xt')
            record['rows'].append(row)
            atomic_json(path, _plain(record))
        if counts.get('teacher_forward_returns') != len(samples):
            raise AssertionError('Teacher forward budget differs')
        if velocity_gradients and counts.get('velocity_leaf_autograd_returns') != 3*len(samples):
            raise AssertionError('Velocity gradient budget differs')
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Frozen teacher acquired parameter gradients')
        record.update(status='PASS', completed_states=len(samples), state_in_progress=None)
    except BaseException as error:
        record.update(status='STOPPED', error=repr(error))
        raise
    finally:
        record['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated(device))
        record['peak_cuda_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device))
        _finish_record(record, path, arrays_path, arrays, start, surface_before)
    return record


def parameter_probe(model, samples, *, model_name, out, ledger=None, targets=None, cases=None):
    """Exactly four BF16 forwards and twelve two-tensor parameter VJPs/model."""
    subset = [s for s in curve_subset(samples) if s['teacher_t'] == .75]
    _validate_samples(subset, count=4)
    if model.experiment_trainable or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Parameter probe requires a clean independently frozen model')
    named = dict(model.named_parameters())
    for name, shape in PARAMETER_SHAPES.items():
        if name not in named or tuple(named[name].shape) != shape or named[name].dtype != torch.float32:
            raise ValueError('Actual ninth-block terminal FFN differs: ' + name)
    parameters = tuple(named[name] for name in PARAMETER_SHAPES)
    path, arrays_path, record = _start_record(out, 'C', model_name, 'parameter_probe')
    record.update(model=model_name, precision='bf16', registered_parameter_shapes=SPEC['parameter_names_and_shapes'],
        parameter_elements=sum(x.numel() for x in parameters), expected_forwards=4,
        expected_parameter_VJPs=12, only_selected_parameter_gradients_returned=True)
    arrays = {}
    counts = record['counts']
    start = time.perf_counter()
    surface_before = work_counters()
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    try:
        targets = make_target_bank(subset, cases) if targets is None else targets
        device = next(model.parameters()).device
        # Native's existing runtime requires either all parameters enabled or
        # all frozen. autograd.grad returns only the registered two tensors and
        # never accumulates .grad on any parameter; no optimizer exists here.
        model.set_trainable(True).eval()
        for index, sample in enumerate(subset):
            batch = native_collate([sample], device)
            record['state_in_progress'] = sample['state_id']
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.enable_grad(), native_execution(model, coordinates=batch['xt']) as runtime:
                _increment(counts, ledger, 'parameter_forward_attempts')
                velocity = model(*model_inputs(batch)).float()
                _increment(counts, ledger, 'parameter_forward_returns')
                arrays[f'state{index:03d}_velocity'] = velocity.detach().cpu().numpy().copy()
                _increment(counts, ledger, 'parameter_loss_attempts')
                terms = loss_terms(velocity, batch, targets[sample['pair_id']])
                _increment(counts, ledger, 'parameter_loss_returns')
                gradients = {}
                for pos, name in enumerate(('FM', 'Edge', 'Surface')):
                    _increment(counts, ledger, 'parameter_autograd_attempts')
                    values = torch.autograd.grad(terms[name], parameters, retain_graph=pos < 2)
                    _increment(counts, ledger, 'parameter_autograd_returns')
                    gradients[name] = torch.cat([value.detach().reshape(-1) for value in values])
                row = _row(sample, terms, batch, targets[sample['pair_id']])
                row['parameter_gradients'] = gradient_summary(*(gradients[k] for k in ('FM', 'Edge', 'Surface')), active=terms['active'])
                row.update(runtime=runtime, velocity_array=f'state{index:03d}_velocity')
            torch.cuda.synchronize(device)
            row['network_losses_and_parameter_VJP_seconds'] = time.perf_counter()-tick
            record['rows'].append(row)
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError('autograd.grad unexpectedly accumulated parameter .grad')
            atomic_json(path, _plain(record))
        if counts.get('parameter_forward_returns') != 4 or counts.get('parameter_autograd_returns') != 12:
            raise AssertionError('One-layer sample budget differs')
        record.update(status='PASS', completed_states=4, state_in_progress=None)
    except BaseException as error:
        record.update(status='STOPPED', error=repr(error))
        raise
    finally:
        model.zero_grad(set_to_none=True)
        model.set_trainable(False).eval()
        record['restored_frozen_and_grad_cleared'] = all(not p.requires_grad and p.grad is None for p in model.parameters())
        record['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated(device))
        record['peak_cuda_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device))
        _finish_record(record, path, arrays_path, arrays, start, surface_before)
    return record


def teacher_curve(model, samples, *, model_name, out, ledger=None, targets=None, cases=None):
    """D: twenty fixed states, one BF16 forward/state, losses only; no new OT."""
    return evaluate_teacher(model, curve_subset(samples), model_name=model_name, precision='bf16', out=out,
                            ledger=ledger, targets=targets, cases=cases, velocity_gradients=False, phase='D')


def curve_rows_from_C(record):
    """Reuse raw3000 C/BF16 rows without another forward or derivative."""
    if record.get('status') != 'PASS' or record.get('phase') != 'C' or record.get('precision') != 'bf16':
        raise ValueError('Only completed BF16 C records can supply raw endpoint curves')
    rows = [row for row in record['rows'] if row['patch'] in (0, 4) and row['noise_index'] == 0]
    if len(rows) != 20 or len({row['state_id'] for row in rows}) != 20:
        raise ValueError('Curve subset is incomplete')
    return dict(status='REUSED_C_RAW3000', new_model_forwards=0, new_autograd_calls=0,
                source_label=record['label'], rows=rows)


def main(argv=None):
    """Caller dispatches only after A/B gates; this entrypoint never runs A/B."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('bank', 'model'), required=True)
    parser.add_argument('--model', choices=('B', 'J'))
    args = parser.parse_args(argv)
    if args.stage == 'bank':
        if args.model is not None:
            parser.error('--model is only valid with --stage model')
        _, audit = build_teacher_bank(args.out)
        print(json.dumps(dict(status=audit['status'], counts=audit['counts'], artifact=audit['artifact'])))
        return
    if args.model is None:
        parser.error('--stage model requires --model B or J')
    from .common import load_cases, load_model
    from native_t1.artifacts import state_sha256
    destination = args.out / 'stage_c' / (args.model+'_process.json')
    if destination.exists():
        raise FileExistsError('No implicit replay of an existing model process')
    record = dict(status='RUNNING', model=args.model, counts={}, stages=[], phase='C',
        source_sha256=file_sha256(Path(__file__)), A_B_ordering_enforced_by='parent orchestration')
    atomic_json(destination, record)
    start = time.perf_counter()
    try:
        samples, bank = load_teacher_bank(args.out)
        record['bank_artifact'] = bank['artifact']
        cases = load_cases()
        targets = make_target_bank(samples, cases)
        model, identity = load_model(args.model)
        record['initial_identity'] = identity
        atomic_json(destination, _plain(record))
        for precision in ('bf16', 'fp32'):
            result = evaluate_teacher(model, samples, model_name=args.model, precision=precision,
                out=args.out, ledger=record['counts'], targets=targets, cases=cases)
            record['stages'].append(dict(kind='teacher', precision=precision, status=result['status'],
                path=str((args.out/'stage_c'/'teacher'/f'{args.model}_{precision}.json').resolve()),
                counts=result['counts'], peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
                peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes']))
            atomic_json(destination, _plain(record))
        result = parameter_probe(model, samples, model_name=args.model, out=args.out,
                                 ledger=record['counts'], targets=targets, cases=cases)
        record['stages'].append(dict(kind='parameter_probe', precision='bf16', status=result['status'],
            path=str((args.out/'stage_c'/'teacher'/f'{args.model}_parameter_probe.json').resolve()),
            counts=result['counts'], peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
            peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes']))
        final_hash = state_sha256(model)
        if final_hash != identity['state_sha256']:
            raise AssertionError('Read-only diagnostics changed model tensors')
        record.update(status='PASS', final_state_sha256=final_hash, frozen_tensor_identity=True)
    except BaseException as error:
        record.update(status='STOPPED', error=repr(error))
        raise
    finally:
        record['wall_seconds'] = time.perf_counter()-start
        atomic_json(destination, _plain(record))
    print(json.dumps(dict(status=record['status'], counts=record['counts'], output=str(destination))))


if __name__ == '__main__':
    main()
