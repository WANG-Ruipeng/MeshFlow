"""One frozen B=1,G=8 window-policy batch. No sampling, dispatch or retry.

CLI --job JSON requires: batch_index, samples (q/window_buffer_path/sha256,
receipt_path/sha256), reward_path/sha256, prior_checkpoint (path/sha256 or null),
source_binding, sigma, output_dir, registration_path/sha256,
training_manifest_path/sha256, math_gate_path/sha256, decode_gate_path/sha256.
Reward records carry request_id, reward, post_score_path, post_score_sha256.
Only --self-test executes without all real gates; it uses small CPU tensors.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import uuid

from .bootstrap import bind, ROOT, OUT
D, E = bind()
sys.path.insert(0, str(ROOT / 'r2/src'))
from r2.pilots.terminal_reward_nft_v1 import common as W
import numpy as np
import torch
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from . import policy as P

VERSION = 'r2_window_policy_rl_train_worker_v1'
INIT_SEED = 2026101501
GROUP = 8
TRAINING_FORWARDS = GROUP * len(P.WINDOW)
DIAGNOSTIC_FORWARDS = TRAINING_FORWARDS
OPTIMIZER_CONTRACT = dict(lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=0.0, amsgrad=False, foreach=False, fused=False)


def sha(path):
    return P.file_sha256(path)


def scoped(path):
    value = D.resolve(path).resolve()
    if not value.is_relative_to(OUT.resolve()):
        raise ValueError('New artifact must remain in this experiment: ' + str(value))
    return value


def read_bound(path, expected):
    path = D.resolve(path)
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError('Bound input hash differs: ' + str(path))
    return D.read(path)


def current_hash(model):
    return state_hash({n: v for n, v in model.adapter_state_dict().items()
                       if n.endswith(('.current_a', '.current_b'))})


def old_hash(model):
    return state_hash({n: v for n, v in model.adapter_state_dict().items()
                       if n.endswith(('.old_a', '.old_b'))})


def rloo(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    if r.shape != (GROUP,) or not np.isfinite(r).all() or not np.isin(r, [0., 1.]).all():
        raise ValueError('Exactly eight complete binary rewards required')
    return r - (r.sum() - r) / (GROUP - 1)


def make_optimizer(model):
    return torch.optim.AdamW(model.current_parameters(), **OPTIMIZER_CONTRACT)


def validate_optimizer(model, optimizer, steps):
    if len(optimizer.param_groups) != 1:
        raise ValueError('One adapter-only optimizer group required')
    group = optimizer.param_groups[0]
    params = model.current_parameters()
    if [id(x) for x in group['params']] != [id(x) for x in params]:
        raise ValueError('Optimizer ownership/order differs')
    for name, expected in OPTIMIZER_CONTRACT.items():
        actual = tuple(group[name]) if name == 'betas' else group[name]
        if actual != expected:
            raise ValueError('Optimizer setting changed: ' + name)
    for p in params:
        state = optimizer.state.get(p, {})
        if not state:
            if steps != 0:
                raise ValueError('Missing AdamW history')
            continue
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != steps:
            raise ValueError('AdamW clock/schema mismatch')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = state[name]
            if value.dtype != torch.float32 or value.shape != p.shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid AdamW moment')


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def prepare_job(path):
    """Validate all small inputs before loading a base or initializing CUDA."""
    job = D.read(path)
    batch = job['batch_index']
    if type(batch) is not int or batch < 0:
        raise ValueError('Nonnegative integer batch_index required')
    destination = scoped(job['output_dir'])
    if any((destination / name).exists() for name in ('attempt.json', 'completion.json', 'checkpoint.pt', 'failure.json')):
        raise RuntimeError('Attempt already exists: no automatic retry or overwrite')
    registration = read_bound(job['registration_path'], job['registration_sha256'])
    training = read_bound(job['training_manifest_path'], job['training_manifest_sha256'])
    if training.get('status') != 'FROZEN' or (training.get('B'), training.get('G')) != (1, GROUP):
        raise ValueError('Frozen B1G8 training manifest required')
    if batch >= training['batches'] or training['batches'] not in (64, 128):
        raise ValueError('Batch exceeds finite registered opportunities')
    for name in ('math_gate', 'decode_gate'):
        gate = read_bound(job[name + '_path'], job[name + '_sha256'])
        if gate.get('status') != 'PASS':
            raise RuntimeError('Required gate is not PASS: ' + name)
        if name == 'decode_gate' and float(gate['selected_sigma']) != float(job['sigma']):
            raise ValueError('Training sigma differs from completed calibration gate')
    if float(job['sigma']) not in P.SIGMAS[1:]:
        raise ValueError('Positive registered transition sigma required')
    base_manifest = read_bound(OUT / 'inputs/manifest.json', registration['manifest_sha256'])
    if job['source_binding'] != base_manifest['source_binding']:
        raise ValueError('Source binding differs from frozen FULL_S6000')
    if training.get('evaluation_manifest_sha256') != registration['manifest_sha256']:
        raise ValueError('Training/evaluation manifest identity differs')
    group = next(g for g in training['groups'] if g['batch_index'] == batch)
    requests = {q['request_id']: q for q in training['training_requests'] if q['batch_index'] == batch}
    samples = job['samples']
    if len(samples) != GROUP or len(requests) != GROUP or len(set(group['requests'])) != GROUP:
        raise ValueError('Incomplete B1G8 group')
    ids = [s['q']['request_id'] for s in samples]
    if ids != group['requests'] or set(ids) != set(requests):
        raise ValueError('Sample order/identities differ from frozen batch')
    reward_payload = read_bound(job['reward_path'], job['reward_sha256'])
    reward_rows = reward_payload if isinstance(reward_payload, list) else reward_payload['records']
    if len(reward_rows) != GROUP or len({r['request_id'] for r in reward_rows}) != GROUP:
        raise ValueError('Exactly eight unique score-derived rewards required')
    reward_map = {r['request_id']: r for r in reward_rows}
    if set(reward_map) != set(ids):
        raise ValueError('Reward request identities differ')
    cache, rewards = [], []
    for sample in samples:
        q = sample['q']; rid = q['request_id']
        if q != requests[rid] or q['condition_id'] != group['condition_id']:
            raise ValueError('Request changed or mixed conditions')
        reward = reward_map[rid]
        score = read_bound(reward['post_score_path'], reward['post_score_sha256'])
        if (score.get('status') != 'PASS' or score.get('C_preserved_exact') is not True
                or score.get('stage') != 'POST' or score.get('request_id') != rid
                or score.get('original_definition_unchanged') is not True
                or score.get('identity', {}).get('registration_sha256') != job['registration_sha256']):
            raise ValueError('Missing/invalid original POST score or C preservation')
        value = int(int(score['topology']['C_F_valid_edge_count']) > 0)
        if type(reward['reward']) not in (int, float) or reward['reward'] != value:
            raise ValueError('Reward is not original binary legal-CF outcome')
        rewards.append(value)
        receipt = read_bound(sample['receipt_path'], sample['receipt_sha256'])
        if (receipt.get('status') != 'COMPLETE' or receipt.get('request_id') != rid
                or receipt.get('model_mode') != 'current'
                or receipt.get('registration_sha256') != job['registration_sha256']
                or float(receipt['sigma']) != float(job['sigma'])):
            raise ValueError('Rollout receipt invalid or wrong request')
        from .sample_worker import verify_window_receipt
        verify_window_receipt(q, receipt)
        if (D.resolve(receipt['window_buffer_path']).resolve() != D.resolve(sample['window_buffer_path']).resolve()
                or receipt['window_buffer_sha256'] != sample['window_buffer_sha256']):
            raise ValueError('Job buffer differs from completed RAW receipt')
        score_input_hashes = {str(D.resolve(p).resolve()): h
                             for p, h in score['identity']['input_hashes'].items()}
        if score_input_hashes.get(str(D.resolve(receipt['raw_path']).resolve())) != receipt['raw_sha256']:
            raise ValueError('Reward was scored on a different RAW trajectory')
        buffer_path = D.resolve(sample['window_buffer_path'])
        if sha(buffer_path) != sample['window_buffer_sha256']:
            raise ValueError('Window buffer hash differs')
        with np.load(buffer_path, allow_pickle=False) as f:
            buf = {k: f[k].copy() for k in f.files}
        n, k = q['N'], q['K']
        shape = (len(P.WINDOW), 1, n, 9)
        for name in ('state_before', 'state_after', 'mean_old', 'epsilon'):
            if buf[name].shape != shape or buf[name].dtype != np.float32 or not np.isfinite(buf[name]).all():
                raise ValueError('Invalid FP32 rollout buffer: ' + name)
        if not np.array_equal(buf['k'], np.asarray(P.WINDOW, dtype=np.int64)):
            raise ValueError('Window index differs')
        if buf['sigma'].shape != (len(P.WINDOW),) or not np.all(buf['sigma'] == float(job['sigma'])):
            raise ValueError('Actual transition sigma differs')
        if buf['logp_old'].shape != (len(P.WINDOW),) or not np.isfinite(buf['logp_old']).all():
            raise ValueError('Old full joint logprob missing/nonfinite')
        known = np.zeros((1, n), np.bool_); known[:, :k] = True
        if not D.same(buf['known_mask'], known) or not D.same(buf['valid_mask'], np.ones_like(known)):
            raise ValueError('Known/valid mask changed')
        cond = D.load_request(q)['C'].reshape(k, 9)
        for name in ('state_before', 'state_after', 'mean_old'):
            if any(x[0, :k].tobytes() != cond.tobytes() for x in buf[name]):
                raise ValueError('Buffered C not bitwise preserved')
        if not np.array_equal(buf['state_after'][:-1], buf['state_before'][1:]):
            raise ValueError('Adjacent actual window states are discontinuous')
        if not np.array_equal(buf['epsilon'][:, :, :k], np.zeros_like(buf['epsilon'][:, :, :k])):
            raise ValueError('Transition noise includes C')
        eps = D.arrays(q['transition_noise_path'])['epsilon']
        if sha(D.resolve(q['transition_noise_path'])) != q['transition_noise_sha256'] or not D.same(eps, buf['epsilon']):
            raise ValueError('Actual noise differs from pre-registered independent stream')
        expected_after = buf['mean_old'].copy()
        expected_after[:, :, k:] += buf['epsilon'][:, :, k:] * np.float32(job['sigma'])
        if not D.same(expected_after, buf['state_after']):
            raise ValueError('Stored action does not realize the registered Gaussian transition')
        cache.append(dict(q=q, arrays=buf, receipt=receipt, sample=sample))
    prior = job.get('prior_checkpoint')
    if (batch == 0) != (prior is None):
        raise ValueError('Only first batch may start from fresh adapters')
    if prior is not None:
        prior_path = scoped(prior['path'])
        if sha(prior_path) != prior['sha256']:
            raise ValueError('Prior completed checkpoint hash differs')
        sidecar = D.read(prior_path.with_suffix('.json'))
        completion = D.read(prior_path.parent / 'completion.json')
        if (sidecar.get('status') != 'COMPLETE' or sidecar.get('sha256') != prior['sha256']
                or sidecar.get('batch_index') != batch - 1 or completion.get('status') != 'COMPLETE'):
            raise ValueError('Prior checkpoint lacks completed batch/Phase proof')
    return job, cache, rewards, destination


class Ledger:
    def __init__(self, path, batch_index, validation_forwards=0):
        self.handle = Path(path).open('x', encoding='utf-8', buffering=1)
        self.batch_index = batch_index
        self.maximum = dict(training=TRAINING_FORWARDS, postupdate_kl=DIAGNOSTIC_FORWARDS,
                            zero_lora_validation=validation_forwards)
        self.counts = {k: 0 for k in self.maximum}
        self.returns = {k: 0 for k in self.maximum}
        self.backwards = 0

    def event(self, event, **fields):
        row = dict(event=event, utc=W.utc(), pid=os.getpid(), batch_index=self.batch_index)
        for key, value in fields.items():
            if key in row and row[key] != value:
                raise ValueError('Ledger identity conflict: ' + key)
            row[key] = value
        self.handle.write(json.dumps(D.clean(row), allow_nan=False) + '\n')
        self.handle.flush(); os.fsync(self.handle.fileno())

    def call(self, kind, fn, **context):
        if kind not in self.maximum or self.counts[kind] >= self.maximum[kind]:
            raise RuntimeError('Forward budget exceeded: ' + kind)
        self.counts[kind] += 1
        self.event('FORWARD_START', kind=kind, count=self.counts[kind], **context)
        result = fn()
        self.returns[kind] += 1
        self.event('FORWARD_RETURN', kind=kind, count=self.counts[kind], **context)
        return result

    def backward(self, loss, **context):
        if self.backwards >= TRAINING_FORWARDS:
            raise RuntimeError('Backward budget exceeded')
        self.backwards += 1
        self.event('BACKWARD_START', count=self.backwards, **context)
        loss.backward()
        self.event('BACKWARD_RETURN', count=self.backwards, **context)

    def close(self):
        self.handle.close()


def materialize(cached, device):
    b = cached['arrays']
    return dict(q=cached['q'], **{name: torch.from_numpy(b[name]).to(device)
        for name in ('state_before', 'state_after', 'mean_old', 'known_mask', 'valid_mask')},
        sigma=b['sigma'], logp_old=b['logp_old'],
        y=torch.tensor([cached['q']['N']], dtype=torch.int64, device=device))


def update_once(model, optimizer, buffers, rewards, *, completed_steps, ledger, mean_fn=P.transition_mean):
    """Exactly 160 fixed-action backwards, <=1 step, then 160 diagnostic F."""
    if len(buffers) != GROUP:
        raise ValueError('Exactly one complete G8 group required')
    advantages = rloo(rewards)
    active = bool(np.any(advantages != 0))
    validate_optimizer(model, optimizer, completed_steps)
    before = current_hash(model); old_before = old_hash(model)
    before_values = [p.detach().clone() for p in model.current_parameters()]
    optimizer.zero_grad(set_to_none=True)
    rows, total_loss, max_logp_error = [], 0., 0.
    ledger.event('BATCH_GRADIENT_START', active=active, rewards=list(rewards), advantages=advantages.tolist(),
                 current_adapters_sha256=before, expected_training_forwards=TRAINING_FORWARDS)
    for g, b in enumerate(buffers):
        free = b['valid_mask'] & ~b['known_mask']
        for index, step in enumerate(P.WINDOW):
            context = dict(request_id=b['q']['request_id'], member=g, euler_step=step)
            mu = mean_fn(model, b['state_before'][index], step, b['y'], b['valid_mask'], b['known_mask'],
                         model_mode='current', forward_call=lambda fn, c=context: ledger.call('training', fn, **c))
            if not torch.equal(mu.detach(), b['mean_old'][index]):
                raise RuntimeError('Current preupdate mean differs from actual rollout old mean; off-policy or runtime mismatch')
            lp = P.joint_log_prob(b['state_after'][index], mu, b['sigma'][index], free)
            error = abs(float(lp.detach().item()) - float(b['logp_old'][index]))
            max_logp_error = max(max_logp_error, error)
            if error > 1e-8:
                raise RuntimeError('Stored full joint logprob failed preupdate recomputation')
            loss = -(float(advantages[g]) / GROUP) * lp.sum()
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise FloatingPointError('Nonfinite or disconnected RLOO loss')
            scalar = float(loss.detach().item()); total_loss += scalar
            ledger.backward(loss, **context)
            rows.append(dict(**context, advantage=float(advantages[g]), logp=float(lp.detach().item()),
                             loss_contribution=scalar, old_logp_abs_error=error))
            del loss, lp, mu
    params = model.current_parameters()
    if current_hash(model) != before or old_hash(model) != old_before:
        raise RuntimeError('Policy changed before the sole optimizer step')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError('Missing/nonfinite LoRA gradient')
    trainable = {id(p) for p in params}
    if any(p.grad is not None for p in model.parameters() if id(p) not in trainable):
        raise RuntimeError('Frozen base received a gradient')
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True, foreach=False))
    clipped_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
    if active:
        ledger.event('OPTIMIZER_STEP_START', optimizer_step=completed_steps + 1)
        optimizer.step()
        ledger.event('OPTIMIZER_STEP_RETURN', optimizer_step=completed_steps + 1)
    else:
        if grad_norm != 0.0:
            raise RuntimeError('Zero-advantage whole batch produced nonzero gradient')
        ledger.event('OPTIMIZER_STEP_SKIPPED', reason='ALL_ADVANTAGES_ZERO_NO_REDRAW', optimizer_step=completed_steps)
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError('Nonfinite adapter after update')
    optimizer.zero_grad(set_to_none=True)
    completed_steps += int(active)
    validate_optimizer(model, optimizer, completed_steps)
    delta = math.sqrt(sum(float((p.detach().double() - old.double()).square().sum())
                          for p, old in zip(params, before_values)))
    if not active and (delta != 0. or current_hash(model) != before):
        raise RuntimeError('Skipped batch moved current parameters')
    if old_hash(model) != old_before:
        raise RuntimeError('Old adapter snapshot changed during update')
    trajectory_kl = []
    with torch.no_grad():
        for g, b in enumerate(buffers):
            free = b['valid_mask'] & ~b['known_mask']; total = 0.
            for index, step in enumerate(P.WINDOW):
                context = dict(request_id=b['q']['request_id'], member=g, euler_step=step)
                mu = mean_fn(model, b['state_before'][index], step, b['y'], b['valid_mask'], b['known_mask'],
                             model_mode='current', forward_call=lambda fn, c=context: ledger.call('postupdate_kl', fn, **c))
                value = float(P.analytic_kl(b['mean_old'][index], mu, b['sigma'][index], free).item())
                if not math.isfinite(value) or value < 0:
                    raise FloatingPointError('Invalid postupdate conditional KL')
                total += value
                rows[g * len(P.WINDOW) + index]['postupdate_conditional_kl'] = value
            trajectory_kl.append(total)
    if not active and any(x != 0. for x in trajectory_kl):
        raise RuntimeError('Skipped batch has nonzero exact conditional KL')
    ledger.event('BATCH_UPDATE_COMPLETE', optimizer_step=completed_steps, actual_optimizer_updates=int(active),
                 gradient_norm_before_clip=grad_norm, gradient_norm_after_clip=clipped_norm,
                 adapter_delta_l2=delta, postupdate_trajectory_KL_mean=float(np.mean(trajectory_kl)))
    return dict(rewards=list(rewards), advantages=advantages.tolist(), active_group_fraction=float(active),
        actual_optimizer_updates=int(active), optimizer_steps=completed_steps, loss=total_loss,
        gradient_norm_before_clip=grad_norm, gradient_norm_after_clip=clipped_norm,
        adapter_delta_l2=delta, preupdate_old_logp_max_abs_error=max_logp_error,
        current_hash_before=before, current_hash_after=current_hash(model), old_hash=old_before,
        postupdate_conditional_KL_sum_by_trajectory=trajectory_kl,
        postupdate_trajectory_KL_mean=float(np.mean(trajectory_kl)),
        KL_scope='Exact conditional Gaussian KL, summed on sampled old states; empirical trajectory mean, not terminal geometry KL',
        rows=rows)


def save_checkpoint(path, model, optimizer, job, metrics):
    path = scoped(path)
    if path.exists():
        raise FileExistsError('Checkpoint already exists; no overwrite')
    if any(p.grad is not None for p in model.current_parameters()):
        raise RuntimeError('Checkpoint requires cleared completed gradients')
    validate_optimizer(model, optimizer, metrics['optimizer_steps'])
    adapters = model.adapter_state_dict()
    payload = dict(schema=P.ADAPTER_SCHEMA, version=VERSION, adapters=adapters,
        adapters_sha256=state_hash(adapters), current_adapters_sha256=current_hash(model),
        model_metadata=model.metadata(), base_file_sha256=model.base_file_sha256,
        registration_sha256=job['registration_sha256'], training_manifest_sha256=job['training_manifest_sha256'],
        update_in_progress=False, batch_index=job['batch_index'], opportunities_completed=job['batch_index'] + 1,
        optimizer_steps=metrics['optimizer_steps'], optimizer=cpu_copy(optimizer.state_dict()),
        optimizer_names=model.current_parameter_names(), rng=cpu_copy(rng_state()),
        init_seed=INIT_SEED, window=list(P.WINDOW), sigma=float(job['sigma']),
        loss_contract='RLOO_full_joint_Gaussian_sum_window_one_onpolicy_update_no_other_loss',
        metrics=cpu_copy(metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    if path.exists():
        raise FileExistsError('Checkpoint appeared while writing; preserve temporary for review')
    os.replace(temporary, path)
    return dict(status='COMPLETE', path=str(path), sha256=sha(path),
        adapters_sha256=payload['adapters_sha256'], current_adapters_sha256=payload['current_adapters_sha256'],
        batch_index=job['batch_index'], opportunities_completed=payload['opportunities_completed'],
        optimizer_steps=payload['optimizer_steps'], registration_sha256=job['registration_sha256'])


def load_training_checkpoint(prior, model, optimizer, job):
    payload = P.load_adapter_checkpoint(D.resolve(prior['path']), model,
                expected_sha256=prior['sha256'], registration_sha256=job['registration_sha256'])
    if (payload['batch_index'], payload['opportunities_completed']) != (job['batch_index'] - 1, job['batch_index']):
        raise ValueError('Prior checkpoint opportunity clock differs')
    if (payload['init_seed'] != INIT_SEED or payload['sigma'] != float(job['sigma'])
            or payload['window'] != list(P.WINDOW)
            or payload['training_manifest_sha256'] != job['training_manifest_sha256']
            or payload['optimizer_names'] != model.current_parameter_names()):
        raise ValueError('Prior checkpoint policy/optimizer contract differs')
    optimizer.load_state_dict(payload['optimizer'])
    validate_optimizer(model, optimizer, payload['optimizer_steps'])
    restore_rng(payload['rng'])
    return payload['optimizer_steps']


def run(job_path):
    job, cache, rewards, destination = prepare_job(job_path)
    identity = dict(stage='window_rl_training', version=VERSION, batch_index=job['batch_index'],
                    job_path=str(D.resolve(job_path)), job_sha256=sha(D.resolve(job_path)),
                    registration_sha256=job['registration_sha256'])
    with W.Phase(destination, identity):
        ledger = Ledger(destination / 'events.jsonl', job['batch_index'], validation_forwards=2 if job['batch_index'] == 0 else 0)
        try:
            torch.set_num_threads(2)
            random.seed(INIT_SEED); np.random.seed(INIT_SEED); torch.manual_seed(INIT_SEED)
            binding = job['source_binding']
            model = P.load_policy(D.resolve(binding['path']), expected_sha256=binding['sha256'],
                                  device='cuda', init_seed=INIT_SEED)
            optimizer = make_optimizer(model)
            steps = 0
            if job['prior_checkpoint'] is not None:
                steps = load_training_checkpoint(job['prior_checkpoint'], model, optimizer, job)
            if len(model.adapter_modules()) != 48 or sum(p.numel() for p in model.current_parameters()) != 589824:
                raise ValueError('Registered rank4 adapter coverage changed')
            current = current_hash(model)
            for item in cache:
                ident = item['receipt'].get('model_identity', {})
                if ident.get('current_adapters_sha256') != current or item['receipt'].get('registration_sha256') != job['registration_sha256']:
                    raise ValueError('Rollout does not bind exact current policy/registration')
                if job['prior_checkpoint'] is None:
                    if ident.get('fresh_zero_lora') is not True or ident.get('init_seed') != INIT_SEED:
                        raise ValueError('First rollout was not registered fresh zero-LoRA policy')
                elif ident.get('adapter_checkpoint_sha256') != job['prior_checkpoint']['sha256']:
                    raise ValueError('Rollout checkpoint identity differs')
            model.update_old(0.0)
            if current_hash(model) != current:
                raise RuntimeError('Old snapshot altered current policy')
            buffers = [materialize(item, 'cuda') for item in cache]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            if job['batch_index'] == 0:
                b = buffers[0]
                with torch.no_grad():
                    actual = P.transition_mean(model, b['state_before'][0], P.WINDOW[0], b['y'], b['valid_mask'], b['known_mask'],
                        model_mode='current', forward_call=lambda fn: ledger.call('zero_lora_validation', fn, branch='current'))
                    reference = P.transition_mean(model, b['state_before'][0], P.WINDOW[0], b['y'], b['valid_mask'], b['known_mask'],
                        model_mode='reference', forward_call=lambda fn: ledger.call('zero_lora_validation', fn, branch='reference'))
                equal = torch.equal(actual, reference)
                ledger.event('ZERO_LORA_CHECK', bitwise_equal=equal, maxabs=float((actual - reference).abs().max()))
                if not equal:
                    raise RuntimeError('Fresh zero-LoRA changed same-state base mean')
            metrics = update_once(model, optimizer, buffers, rewards, completed_steps=steps, ledger=ledger)
            torch.cuda.synchronize()
            if ledger.counts != ledger.maximum or ledger.returns != ledger.maximum or ledger.backwards != TRAINING_FORWARDS:
                raise RuntimeError('Completed batch forward/backward ledger differs from fixed budget')
            metrics.update(status='COMPLETE', version=VERSION, batch_index=job['batch_index'],
                seconds=time.perf_counter() - tick, actual_model_forwards=sum(ledger.counts.values()),
                forward_counts=ledger.counts.copy(), backward_calls=ledger.backwards,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='Reset after model/optimizer/buffer load; includes resident memory and whole update/backward/AdamW/post-KL; process polls separate',
                actual_model_forward_attempts=ledger.counts.copy(), retry_allowed=False)
            receipt = save_checkpoint(destination / 'checkpoint.pt', model, optimizer, job, metrics)
            D.write(destination / 'training_metrics.json', metrics)
            D.write(destination / 'checkpoint.json', receipt)
            ledger.event('CHECKPOINT_COMPLETE', **receipt)
            return dict(status='COMPLETE', checkpoint=receipt, metrics_path=str(destination / 'training_metrics.json'))
        except BaseException as exc:
            ledger.event('TRAINING_FAILURE', error=repr(exc), consumed_forward_attempts=ledger.counts.copy(),
                         returned_forwards=ledger.returns.copy(), backward_attempts=ledger.backwards, retry_allowed=False)
            raise
        finally:
            ledger.close()


def self_test():
    """CPU-only algebra/gradient/LoRA checks; no base load or CUDA initialization."""
    if torch.cuda.is_initialized():
        raise RuntimeError('Self-test must start without CUDA context')
    torch.set_num_threads(2)
    torch.manual_seed(12019)
    checks = {}
    for m in range(GROUP + 1):
        advantages = rloo([1] * m + [0] * (GROUP - m))
        assert abs(advantages.sum()) < 1e-14
        assert np.allclose(advantages[:m], (GROUP - m) / (GROUP - 1))
        assert np.allclose(advantages[m:], -m / (GROUP - 1))
    checks['rloo_all_binary_success_counts'] = True
    free = torch.tensor([[False, True, True, False]])
    action = torch.randn(1, 4, 9)
    mean = torch.randn(1, 4, 9, requires_grad=True)
    sigma = 0.01
    lp = P.joint_log_prob(action, mean, sigma, free)
    expected = torch.distributions.Normal(mean.double()[free], sigma).log_prob(action.double()[free]).sum()
    assert torch.allclose(lp.sum(), expected, atol=1e-8, rtol=1e-13)
    lp.sum().backward()
    derivative = torch.where(free[..., None], (action.double() - mean.detach().double()) / sigma**2, 0.)
    assert torch.allclose(mean.grad.double(), derivative, rtol=2e-7, atol=1e-4)
    checks['joint_density_and_fixed_action_gradient'] = True
    altered = action.clone(); altered[:, [0, 3]] += 100.
    assert torch.equal(P.joint_log_prob(action, mean, sigma, free), P.joint_log_prob(altered, mean, sigma, free))
    checks['known_padding_excluded'] = True
    old = mean.detach().clone(); changed = old + .001
    kl = P.analytic_kl(old, changed, sigma, free)
    expected_kl = ((changed.double() - old.double())[free].square().sum() / (2 * sigma**2))
    assert torch.equal(P.analytic_kl(old, old, sigma, free), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(kl.sum(), expected_kl, atol=1e-12, rtol=1e-12)
    checks['exact_conditional_kl'] = True
    layer = P.AdapterLinear(torch.nn.Linear(9, 9))
    x = torch.randn(1, 4, 9)
    layer.active_adapter = 'reference'; base = layer(x).detach().clone()
    with torch.no_grad(): layer.current_b.fill_(.25)
    assert torch.equal(layer(x), base)
    layer.active_adapter = 'current'; assert not torch.equal(layer(x), base)
    layer(x).sum().backward()
    assert layer.base.weight.grad is None and layer.current_a.grad is not None and layer.current_b.grad is not None
    checks['nonzero_lora_reference_bypass_and_base_frozen'] = True
    param = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([param], **OPTIMIZER_CONTRACT)
    param.sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    before = param.detach().clone(); clock = float(opt.state[param]['step'])
    # The chosen skip does not call step; the counterexample does, with zero grad.
    assert torch.equal(param, before) and float(opt.state[param]['step']) == clock
    param.grad = torch.zeros_like(param); opt.step()
    assert not torch.equal(param, before)
    checks['adamw_zero_gradient_step_is_not_skip'] = True
    for m in (0, 3, 8):
        advantages = rloo([1] * m + [0] * (GROUP - m))
        theta_full = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        theta_micro = theta_full.detach().clone().requires_grad_()
        fixed_actions = torch.linspace(-1., 1., GROUP, dtype=torch.float64)
        losses = [-(float(a) / GROUP) * (-(fixed_actions[g] - .02 * theta_full).square() / (2 * sigma**2))
                  for g, a in enumerate(advantages)]
        sum(losses).backward()
        for g, a in enumerate(advantages):
            (-(float(a) / GROUP) * (-(fixed_actions[g] - .02 * theta_micro).square() / (2 * sigma**2))).backward()
        assert torch.allclose(theta_full.grad, theta_micro.grad, rtol=1e-13, atol=1e-12)
    checks['microbatch_gradient_equals_full_batch'] = True
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = P.AdapterLinear(torch.nn.Linear(9, 9))
        def current_parameters(self):
            return [self.layer.current_a, self.layer.current_b]
        def adapter_state_dict(self):
            return {'layer.' + k: getattr(self.layer, k).detach().clone()
                    for k in ('current_a', 'current_b', 'old_a', 'old_b')}
    class MemoryLedger:
        def __init__(self):
            self.counts = dict(training=0, postupdate_kl=0)
            self.backwards = 0
            self.events = []
        def event(self, name, **fields):
            self.events.append((name, fields))
        def call(self, kind, fn, **context):
            self.counts[kind] += 1
            return fn()
        def backward(self, loss, **context):
            self.backwards += 1
            loss.backward()
    tiny = TinyPolicy()
    tiny_optimizer = make_optimizer(tiny)
    tiny_free = torch.tensor([[False, True, True]])
    def tiny_mean(model, state, step, y, valid, known, *, model_mode, forward_call):
        velocity = forward_call(lambda: model.layer(state))
        return P.euler_mean(state, velocity, valid, known)
    def make_tiny_buffers():
        buffers = []
        with torch.no_grad():
            for g in range(GROUP):
                states, actions, means, logps = [], [], [], []
                state = torch.randn(1, 3, 9)
                for step in P.WINDOW:
                    mu = P.euler_mean(state, tiny.layer(state), torch.ones_like(tiny_free), ~tiny_free)
                    action = torch.where(tiny_free[..., None], mu + .01 * torch.randn_like(mu), mu)
                    states.append(state.clone()); actions.append(action.clone()); means.append(mu.clone())
                    logps.append(float(P.joint_log_prob(action, mu, .01, tiny_free)))
                    state = action
                buffers.append(dict(q={'request_id': 'CPU_TOY_' + str(g)},
                    state_before=torch.stack(states), state_after=torch.stack(actions), mean_old=torch.stack(means),
                    known_mask=~tiny_free, valid_mask=torch.ones_like(tiny_free), y=torch.tensor([3]),
                    sigma=np.full(20, .01), logp_old=np.asarray(logps)))
        return buffers
    ledger = MemoryLedger()
    stats = update_once(tiny, tiny_optimizer, make_tiny_buffers(), [1, 0, 1, 0, 1, 0, 1, 0],
                        completed_steps=0, ledger=ledger, mean_fn=tiny_mean)
    assert stats['optimizer_steps'] == 1 and stats['actual_optimizer_updates'] == 1
    assert ledger.counts == dict(training=160, postupdate_kl=160) and ledger.backwards == 160
    assert sum(n == 'OPTIMIZER_STEP_RETURN' for n, f in ledger.events) == 1
    assert stats['adapter_delta_l2'] > 0 and stats['postupdate_trajectory_KL_mean'] >= 0
    # Now retain real AdamW momentum while a fresh, complete all-zero group skips.
    ledger2 = MemoryLedger()
    old_optimizer = cpu_copy(tiny_optimizer.state_dict())
    skipped = update_once(tiny, tiny_optimizer, make_tiny_buffers(), [0] * GROUP,
                          completed_steps=1, ledger=ledger2, mean_fn=tiny_mean)
    assert skipped['optimizer_steps'] == 1 and skipped['actual_optimizer_updates'] == 0
    assert skipped['adapter_delta_l2'] == 0 and skipped['postupdate_trajectory_KL_mean'] == 0
    assert ledger2.counts == dict(training=160, postupdate_kl=160) and ledger2.backwards == 160
    for key, value in old_optimizer['state'].items():
        for name, before in value.items():
            assert torch.equal(before, tiny_optimizer.state_dict()['state'][key][name])
    checks['real_update_kernel_CPU_tiny_160_backward_160_KL_one_step'] = True
    checks['real_update_kernel_CPU_tiny_inactive_preserves_momentum_and_parameters'] = True
    assert not torch.cuda.is_initialized()
    return dict(status='PASS', version=VERSION, checks=checks, model_loads=0, gpu_calls=0,
                production_forward_backward_status='NOT_RUN', cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--job')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else run(args.job), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
