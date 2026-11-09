"""One frozen loss-metric intervention; the direct FM path and velocity are unchanged.

Formal training resumes the common 3000-update checkpoint and performs exactly
6000 additional updates. Every coupled batch must match the archived B0 trace.
The --self-test path uses synthetic CPU tensors only and never trains a target.
"""
import argparse
import copy
import datetime
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from fm_model import total_velocity, validate_chart
from interaction_model import InteractionVelocity
from train_fm_pilot import BATCH, coupled_batch, diagnostic_summary, json_safe, sha
from train_interaction import capture_rng, initialize_branch

ARM = 'M0'
MODEL_ARM = 'B0'
BASE_UPDATES = 3000
NEW_UPDATES = 6000
FAULT_WALL_LIMIT_SECONDS = 3600.
OLD_WEIGHTS = {'x': 3 / 22, 'n': 3 / 22, 'F': 16 / 22}
BLOCKS = {'x': slice(0, 3), 'n': slice(3, 6), 'F': slice(6, 22)}
MODEL_SNAPSHOT_STEPS = (5000, 7000)


def validate_weights(weights):
    """Use the frozen positive block-mean weights exactly, without rescaling."""
    if not isinstance(weights, dict) or set(weights) != set(BLOCKS):
        raise ValueError('Weights must have exactly x, n, F keys')
    out = {}
    for name in BLOCKS:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Weight must be a real scalar: ' + name)
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Weight must be finite and strictly positive: ' + name)
        out[name] = value
    return out


def weighted_block_loss(predicted, truth, weights):
    """Return new loss, the unchanged old22 diagnostic loss, and block MSEs.

Weights multiply block means only. They never modify the probability path,
noise, chart, network output, or a sampled velocity.
"""
    squared = (predicted - truth).square()
    blocks = {name: squared[..., sl].mean() for name, sl in BLOCKS.items()}
    loss = weights['x'] * blocks['x'] + weights['n'] * blocks['n'] + weights['F'] * blocks['F']
    return loss, squared.mean(), blocks


def load_control_trace(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows) != NEW_UPDATES:
        raise ValueError('Archived B0 trace must contain all 6000 updates')
    for index, row in enumerate(rows, 1):
        if row.get('new_update') != index or row.get('step') != BASE_UPDATES + index:
            raise ValueError('Control trace is not a contiguous 3001..9000 sequence')
        digest = row.get('batch_sha256')
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('Control trace has a missing/invalid batch hash')
        bytes.fromhex(digest)
    return rows, hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(args):
    started = time.perf_counter()
    if args.steps != NEW_UPDATES:
        raise ValueError('Frozen SINGLE_STATE_METRIC0 requires exactly 6000 new updates')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    weights_path = Path(args.weights).resolve()
    baseline_path = Path(args.baseline_trace).resolve()
    weights_bytes = weights_path.read_bytes()
    weight_record = json.loads(weights_bytes.decode('utf-8'))
    if weight_record.get('passed') is not True or weight_record.get('status') != 'PASS':
        raise ValueError('Require a successful frozen calibration: passed=true and status=PASS')
    weights = validate_weights(weight_record['weights'])
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
    control_rows, control_sha256 = load_control_trace(baseline_path)
    parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    with np.load(data_path, allow_pickle=False) as data:
        states = np.array(data['states'], dtype=np.float64, copy=True)
        mean_np = np.array(data['mean'], dtype=np.float64, copy=True)
        scale_np = np.array(data['scale'], dtype=np.float64, copy=True)
    if states.shape != (1, 256, 22) or not np.isfinite(states).all():
        raise ValueError('Require one finite physical (256,22) target')
    validate_chart(torch.from_numpy(mean_np), torch.from_numpy(scale_np))
    if sha(data_path) != parent['protocol']['data_sha256']:
        raise ValueError('Data bytes differ from the common checkpoint training data')
    if not torch.equal(torch.tensor(mean_np, dtype=torch.float32), parent['mean'].cpu()) or not torch.equal(torch.tensor(scale_np, dtype=torch.float32), parent['scale'].cpu()):
        raise ValueError('Chart differs from common checkpoint')
    model, optimizer, schedule, generator, mapping = initialize_branch(parent, MODEL_ARM, device)
    if mapping['parameter_counts'] != {'total': 88214, 'original': 88214, 'added': 0}:
        raise ValueError('Metric arm must use the unmodified 88214-parameter B0 model')
    if schedule.count != len(states):
        raise ValueError('Restored schedule target count mismatch')
    mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    scale = torch.tensor(scale_np, dtype=torch.float32, device=device)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_names = ['train_metric.py', 'train_interaction.py', 'interaction_model.py', 'fm_model.py', 'train_fm_pilot.py']
    source_files = [Path(__file__).with_name(name) for name in source_names]
    protocol = dict(
        version='SINGLE_STATE_METRIC0_TRAIN', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arm=ARM, model_arm=MODEL_ARM, parent_checkpoint=str(checkpoint_path), parent_checkpoint_sha256=sha(checkpoint_path),
        base_updates=BASE_UPDATES, requested_new_updates=args.steps, requested_total_updates=BASE_UPDATES + args.steps,
        batch_size=BATCH, parameter_counts=mapping['parameter_counts'],
        optimizer=dict(name='AdamW', lr=3e-4, weight_decay=1e-4, betas=[.9, .999], eps=1e-8, gradient_clip_norm=1., scheduler=None),
        optimizer_restoration='initialize_branch(parent,B0): all original states, weights, RNG and schedule restored exactly',
        data_path=str(data_path), data_sha256=sha(data_path), states_shape=list(states.shape), mean=mean_np.tolist(), scale=scale_np.tolist(),
        weights=weights, old_weights=OLD_WEIGHTS, weights_path=str(weights_path), weights_sha256=weights_sha256,
        weights_record=weight_record, weight_policy='Read once; finite positive block-mean coefficients used verbatim for every update; no dynamic adjustment',
        baseline_trace_path=str(baseline_path), baseline_trace_sha256=control_sha256,
        batch_match_policy='Compare every coupled batch hash against the same archived B0 new_update before forward/backward; mismatch stops with failed full-state checkpoint',
        probability_path='Unchanged coupled_batch: physical-xyz Hungarian, whole22 target permutation, linear standardized-chart path',
        loss='weights[x]*mean(error_x^2)+weights[n]*mean(error_n^2)+weights[F]*mean(error_F^2); old mean(error_all22^2) diagnostic only',
        sampling='Unchanged raw direct velocity; never apply loss weights to sampled velocity',
        transport='Not added. Original total_velocity(...,direct) computes unchanged L diagnostics only.',
        model_snapshots=list(MODEL_SNAPSHOT_STEPS), snapshot_policy='Model-only fixed diagnostic snapshots at cumulative5000/7000; final9000 has full optimizer/RNG state',
        selection='Fixed final9000 only; no early stopping, checkpoint selection, extra baseline training, or weight scan',
        fault_wall_limit_seconds=FAULT_WALL_LIMIT_SECONDS, budget_policy='Full6000 new updates, one-hour fault boundary; no performance-based extension',
        source_files=[dict(path=str(path.resolve()), sha256=sha(path)) for path in source_files],
        device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
        cudnn_benchmark=False, cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, amp=False)
    write_json(out / 'PROTOCOL.json', protocol)
    write_json(out / 'OPTIMIZER_MAPPING.json', mapping)
    exposure = np.array(parent['exposure_counts'], dtype=np.int64)
    new_exposure = np.zeros_like(exposure)
    completed = 0
    attempted = 0
    verified = 0
    last_metrics = None
    error = None
    status = 'INITIALIZED'
    stream_digest = bytes.fromhex(parent['attempted_stream_sha256'])
    new_stream = bytes(32)
    saved_model_snapshots = []

    def checkpoint():
        return dict(
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            optimizer_param_names=list(dict(model.named_parameters())), rng=capture_rng(schedule, generator),
            arm=ARM, model_arm=MODEL_ARM, protocol=protocol, protocol_sha256=sha(out / 'PROTOCOL.json'),
            mean=mean.detach().cpu(), scale=scale.detach().cpu(), weights=weights, weights_sha256=weights_sha256,
            completed_steps=BASE_UPDATES + completed, new_updates=completed,
            attempted_batches=parent['attempted_batches'] + attempted, new_attempted_batches=attempted,
            verified_batch_count=verified, baseline_trace_sha256=control_sha256,
            exposure_counts=exposure.tolist(), new_exposure_counts=new_exposure.tolist(), status=status, error=error, last_metrics=last_metrics,
            attempted_stream_sha256=stream_digest.hex(), new_attempted_stream_sha256=new_stream.hex(),
            parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'])

    torch.save(model.state_dict(), out / 'initial_model.pt')
    torch.save(checkpoint(), out / 'initial.pt')
    status = 'RUNNING'
    training_started = time.perf_counter()
    try:
        with (out / 'TRAIN_TRACE.jsonl').open('x', buffering=1, encoding='utf-8') as log:
            while completed < args.steps:
                if time.perf_counter() - started >= FAULT_WALL_LIMIT_SECONDS:
                    status = 'FAULT_WALL_LIMIT_EXHAUSTED'
                    break
                attempted += 1
                step_start = time.perf_counter()
                noise, times, target, ids = coupled_batch(states, mean_np, scale_np, schedule, generator)
                batch_hash = hashlib.sha256(noise.numpy().tobytes() + times.numpy().tobytes() + target.numpy().tobytes() + np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
                stream_digest = hashlib.sha256(stream_digest + bytes.fromhex(batch_hash)).digest()
                new_stream = hashlib.sha256(new_stream + bytes.fromhex(batch_hash)).digest()
                control = control_rows[completed]
                if batch_hash != control['batch_sha256']:
                    mismatch = dict(status='BATCH_MISMATCH', step=BASE_UPDATES + completed + 1,
                        new_update=completed + 1, attempted_new_batches=attempted, batch_sha256=batch_hash,
                        expected_batch_sha256=control['batch_sha256'], completed_updates_before_failure=completed)
                    log.write(json.dumps(mismatch, allow_nan=False) + '\n')
                    raise RuntimeError('Coupled batch differs from archived B0 at new_update=' + str(completed + 1))
                verified += 1
                noise = noise.to(device)
                times = times.to(device)
                target = target.to(device)
                z = (1 - times) * noise + times * target
                truth = target - noise
                optimizer.zero_grad(set_to_none=True)
                predicted, diag = total_velocity(model, z, times, mean, scale, 'direct')
                loss, old_loss, blocks = weighted_block_loss(predicted, truth, weights)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(old_loss)):
                    raise FloatingPointError('Nonfinite fixed-weight direct total-velocity loss')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if time.perf_counter() - started >= FAULT_WALL_LIMIT_SECONDS:
                    status = 'FAULT_WALL_LIMIT_EXHAUSTED'
                    break
                optimizer.step()
                completed += 1
                for index in ids:
                    exposure[index] += 1
                    new_exposure[index] += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                last_metrics = dict(step=BASE_UPDATES + completed, new_update=completed,
                    attempted_new_batches=attempted, batch_sha256=batch_hash, expected_batch_sha256=control['batch_sha256'], batch_match=True,
                    elapsed_seconds=time.perf_counter() - started, step_seconds=time.perf_counter() - step_start,
                    loss=float(loss.detach()), new_weighted_loss=float(loss.detach()), old22_loss=float(old_loss.detach()),
                    **{name + '_mse': float(value.detach()) for name, value in blocks.items()},
                    **{name + '_weighted_contribution': float((weights[name] * value).detach()) for name, value in blocks.items()},
                    gradient_norm_before_clip=float(gradient_norm), **diagnostic_summary(diag))
                log.write(json.dumps(json_safe(last_metrics), allow_nan=False) + '\n')
                if BASE_UPDATES + completed in MODEL_SNAPSHOT_STEPS:
                    snapshot = out / ('step_' + str(BASE_UPDATES + completed) + '.pt')
                    torch.save(dict(model_state_dict=model.state_dict(), completed_steps=BASE_UPDATES + completed,
                        new_updates=completed, mean=mean.detach().cpu(), scale=scale.detach().cpu(), arm=ARM, model_arm=MODEL_ARM,
                        weights=weights, weights_sha256=weights_sha256, weights_path=str(weights_path),
                        protocol_sha256=sha(out / 'PROTOCOL.json'), parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],
                        status='DIAGNOSTIC_MODEL_SNAPSHOT', resumable=False), snapshot)
                    saved_model_snapshots.append(dict(path=str(snapshot), sha256=sha(snapshot), completed_steps=BASE_UPDATES + completed))
                if completed == 1 or completed % 100 == 0:
                    print(json.dumps(json_safe(last_metrics), allow_nan=False), flush=True)
            if sha(weights_path) != weights_sha256 or sha(baseline_path) != control_sha256:
                raise RuntimeError('Frozen weights or archived control trace changed during training')
            if completed == args.steps:
                if verified != NEW_UPDATES:
                    raise AssertionError('Incomplete per-batch control verification')
                status = 'COMPLETED'
    except Exception as exc:
        status = 'FAILED'
        error = dict(exception=repr(exc), traceback=traceback.format_exc())
    training_elapsed = time.perf_counter() - training_started
    torch.save(checkpoint(), out / 'final.pt')
    result = dict(status=status, arm=ARM, model_arm=MODEL_ARM, completed_steps=BASE_UPDATES + completed,
        new_updates=completed, requested_new_updates=args.steps, requested_steps=BASE_UPDATES + args.steps,
        finished_requested_budget=status == 'COMPLETED' and completed == args.steps,
        training_elapsed_seconds=training_elapsed, total_wall_seconds_including_initialization_and_save=time.perf_counter() - started,
        fault_wall_limit_seconds=FAULT_WALL_LIMIT_SECONDS, exposure_counts=exposure.tolist(), new_exposure_counts=new_exposure.tolist(),
        attempted_batches=parent['attempted_batches'] + attempted, new_attempted_batches=attempted,
        verified_batch_count=verified, baseline_trace_sha256=control_sha256,
        attempted_stream_sha256=stream_digest.hex(), new_attempted_stream_sha256=new_stream.hex(), last_metrics=last_metrics, error=error,
        weights=weights, weights_sha256=weights_sha256, saved_model_snapshots=saved_model_snapshots,
        parameter_counts=mapping['parameter_counts'], final_checkpoint_sha256=sha(out / 'final.pt'),
        initial_checkpoint_sha256=sha(out / 'initial.pt'), initial_model_sha256=sha(out / 'initial_model.pt'),
        optimizer_mapping_sha256=sha(out / 'OPTIMIZER_MAPPING.json'), protocol_sha256=sha(out / 'PROTOCOL.json'))
    write_json(out / 'RESULT.json', result)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False), flush=True)
    return result


def self_test():
    """Synthetic CPU parity checks; declared tolerances precede measurements."""
    tolerance = dict(loss_atol=2e-6, gradient_atol=2e-7, parameter_update_atol=2e-7,
        optimizer_moment_atol=2e-8, float64_algebra_atol=2e-14)
    torch.set_num_threads(1)
    torch.manual_seed(260927)
    results = []
    for bad in ({'x': 0., 'n': 1., 'F': 1.}, {'x': -1., 'n': 1., 'F': 1.},
                {'x': float('nan'), 'n': 1., 'F': 1.}, {'x': 1., 'n': 1.}, {'x': True, 'n': 1., 'F': 1.}):
        try:
            validate_weights(bad)
        except ValueError:
            continue
        raise AssertionError('Invalid fixed weight accepted')
    results.append(dict(name='reject_nonpositive_nonfinite_missing_or_boolean_weights', passed=True))
    a = torch.randn(2, 256, 22, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(a)
    loss, old, blocks = weighted_block_loss(a, target, OLD_WEIGHTS)
    delta = abs(float((loss - old).detach()))
    assert delta <= tolerance['float64_algebra_atol']
    results.append(dict(name='old22_block_loss_algebra_float64', passed=True, absolute_error=delta))
    unequal = validate_weights({'x': .71, 'n': .29, 'F': .09})
    unequal_loss, same_old, _ = weighted_block_loss(a, target, unequal)
    grad = torch.autograd.grad(unequal_loss, a, retain_graph=True)[0]
    expected = torch.empty_like(a)
    for name, sl in BLOCKS.items():
        block = a[..., sl] - target[..., sl]
        expected[..., sl] = 2 * unequal[name] * block / block.numel()
    assert torch.equal(old, same_old) and not torch.equal(loss, unequal_loss)
    gradient_error = float((grad - expected).abs().max())
    assert gradient_error <= tolerance['float64_algebra_atol']
    results.append(dict(name='unequal_weights_change_only_loss_derivative', passed=True, gradient_max_error=gradient_error))
    model = InteractionVelocity(arm='B0', width=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    z = torch.randn(2, 256, 22)
    times = torch.rand(2, 1, 1)
    truth = torch.randn_like(z)
    mean = torch.zeros(22)
    scale = torch.ones(22)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        raw, _ = total_velocity(model, z, times, mean, scale, 'direct')
        (raw - truth).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    old_model = copy.deepcopy(model)
    metric_model = copy.deepcopy(model)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=3e-4, weight_decay=1e-4)
    metric_optimizer = torch.optim.AdamW(metric_model.parameters(), lr=3e-4, weight_decay=1e-4)
    old_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    metric_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert all(float(state['step']) == 2. and bool(torch.count_nonzero(state['exp_avg'])) for state in old_optimizer.state.values())
    old_optimizer.zero_grad(set_to_none=True)
    metric_optimizer.zero_grad(set_to_none=True)
    old_pred, old_diag = total_velocity(old_model, z, times, mean, scale, 'direct')
    metric_pred, metric_diag = total_velocity(metric_model, z, times, mean, scale, 'direct')
    assert torch.equal(old_pred, metric_pred)
    original_loss = (old_pred - truth).square().mean()
    metric_loss, _, _ = weighted_block_loss(metric_pred, truth, OLD_WEIGHTS)
    loss_error = abs(float((original_loss - metric_loss).detach()))
    original_loss.backward()
    metric_loss.backward()
    gradient_error = max(float((p.grad - q.grad).abs().max()) for p, q in zip(old_model.parameters(), metric_model.parameters()))
    assert loss_error <= tolerance['loss_atol'] and gradient_error <= tolerance['gradient_atol']
    old_norm = torch.nn.utils.clip_grad_norm_(old_model.parameters(), 1., error_if_nonfinite=True)
    metric_norm = torch.nn.utils.clip_grad_norm_(metric_model.parameters(), 1., error_if_nonfinite=True)
    old_optimizer.step()
    metric_optimizer.step()
    parameter_error = max(float((p - q).detach().abs().max()) for p, q in zip(old_model.parameters(), metric_model.parameters()))
    moment_error = max(float((old_optimizer.state[p][key] - metric_optimizer.state[q][key]).abs().max())
        for p, q in zip(old_model.parameters(), metric_model.parameters()) for key in ('exp_avg', 'exp_avg_sq'))
    assert parameter_error <= tolerance['parameter_update_atol'] and moment_error <= tolerance['optimizer_moment_atol']
    results.append(dict(name='old22_restored_adamw_one_update_parity_float32', passed=True,
        old_loss=float(original_loss.detach()), loss_absolute_error=loss_error, gradient_max_error=gradient_error,
        parameter_update_max_error=parameter_error, optimizer_moment_max_error=moment_error,
        old_clip_norm=float(old_norm), metric_clip_norm=float(metric_norm), nonzero_moments_restored=True))
    # Changing weights does not enter either model or total_velocity invocation.
    raw_before, diag_before = total_velocity(model, z, times, mean, scale, 'direct')
    original, _, _ = weighted_block_loss(raw_before, truth, OLD_WEIGHTS)
    changed, _, _ = weighted_block_loss(raw_before, truth, unequal)
    raw_after, diag_after = total_velocity(model, z, times, mean, scale, 'direct')
    assert torch.equal(raw_before, raw_after) and not torch.equal(original, changed)
    assert diagnostic_summary(diag_before) == diagnostic_summary(diag_after)
    results.append(dict(name='weights_leave_direct_velocity_and_L_diagnostics_unchanged', passed=True))
    return dict(passed=True, tests_run=len(results), scope='Synthetic CPU mathematics/model/AdamW only; no real target training or extraction',
        tolerances=tolerance, checks=results, torch_version=torch.__version__, source_files=[dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__)))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['data', 'checkpoint', 'weights', 'baseline-trace', 'out']:
        parser.add_argument('--' + name)
    parser.add_argument('--steps', type=int, default=NEW_UPDATES)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-out')
    args = parser.parse_args()
    if args.self_test:
        answer = self_test()
        if args.self_test_out:
            test_out = Path(args.self_test_out).resolve()
            test_out.mkdir(parents=True, exist_ok=False)
            write_json(test_out / 'VALIDATION.json', answer)
        print(json.dumps(answer, indent=2, allow_nan=False), flush=True)
    else:
        for name in ['data', 'checkpoint', 'weights', 'baseline_trace', 'out']:
            if not getattr(args, name):
                parser.error('--' + name.replace('_', '-') + ' is required for training')
        answer = main(args)
        raise SystemExit(0 if answer['finished_requested_budget'] else 1)
