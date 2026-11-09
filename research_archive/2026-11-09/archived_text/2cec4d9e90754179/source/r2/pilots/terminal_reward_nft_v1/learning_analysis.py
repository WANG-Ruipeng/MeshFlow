"""Read-only CPU analysis of the bounded pilot's original learning evidence.

Writes immutable, input-addressed snapshots under learning_analysis/snapshots.
No rollout, model construction/forward, new scoring, or semantic judgement.
Only small, receipted adapter/optimizer checkpoints may be deserialized on CPU.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from . import data as D

VERSION = 'terminal_reward_nft_learning_analysis_v1'
ARMS = ('NFT', 'SELF_FM')
LABELS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
ROUNDS, UPDATES = 16, 16


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


class Inputs:
    def __init__(self):
        self.sources = {}
        self.partial_tails = []

    def blob(self, path):
        p = D.resolve(path)
        b = p.read_bytes()
        key = str(p.relative_to(D.ROOT))
        self.sources[key] = dict(sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
        return b

    def read(self, path):
        return json.loads(self.blob(path).decode('utf-8-sig'))

    def events(self, path):
        b = self.blob(path)
        if b and not b.endswith(b'\n'):
            head, _, tail = b.rpartition(b'\n')
            self.partial_tails.append(dict(path=str(path), ignored_trailing_bytes=len(tail)))
            b = head
        return [json.loads(s) for s in b.splitlines() if s.strip()]


def digest_tensors(state, torch):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        v = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str((tuple(v.shape), v.dtype)).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def adapter_summary(inp, path, receipt, arm, ri, previous):
    # The large generator cannot enter this loader, even through a bad receipt.
    path = path.resolve()
    if not path.is_relative_to((D.OUT / 'checkpoints' / arm).resolve()):
        raise ValueError('Adapter path outside the expected arm directory')
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Refusing checkpoint larger than 64 MiB; base loading prohibited')
    payload = inp.blob(path)
    if hashlib.sha256(payload).hexdigest() != receipt['sha256']:
        raise ValueError('Checkpoint receipt hash mismatch')
    import torch
    torch.set_num_threads(1)
    value = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (value['schema'] != 'terminal_reward_nft_training_v1' or value['arm'] != arm
            or value['round_index'] != ri or value['step'] != (ri + 1) * UPDATES
            or value['update_in_progress']):
        raise ValueError('Checkpoint clock/arm/schema differs')
    state = value['adapters']
    if digest_tensors(state, torch) != value['adapters_sha256'] or value['adapters_sha256'] != receipt['adapters_sha256']:
        raise ValueError('Adapter state hash differs')
    names = value['model_metadata']['target_names']
    if len(names) != 48 or len(state) != 48 * 4:
        raise ValueError('Unexpected adapter inventory')
    scale = value['model_metadata']['alpha'] / value['model_metadata']['rank']
    aa = bb = delta = residual = effective_delta = 0.0
    prior_matches = True if previous is not None else None
    for name in names:
        tensors = [state[name + '.' + suffix] for suffix in ('current_a', 'current_b', 'old_a', 'old_b')]
        if any(t.device.type != 'cpu' or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError('Adapter tensors must be finite CPU values')
        a, b, a0, b0 = [t.double() for t in tensors]
        aa += float(a.square().sum()); bb += float(b.square().sum())
        delta += float((a - a0).square().sum() + (b - b0).square().sum())
        # Rank-sized Gram matrices avoid constructing full weight matrices.
        current_sq = float(((b.T @ b) * (a @ a.T)).sum())
        old_sq = float(((b0.T @ b0) * (a0 @ a0.T)).sum())
        cross = float(((b.T @ b0) * (a @ a0.T)).sum())
        residual += scale * scale * max(0.0, current_sq)
        effective_delta += scale * scale * max(0.0, current_sq + old_sq - 2 * cross)
        if previous is not None:
            for suffix in ('a', 'b'):
                prior_matches &= torch.equal(state[name + '.old_' + suffix], previous[name + '.current_' + suffix])
    if prior_matches is False:
        raise ValueError('Old adapter is not previous checkpoint current adapter')
    if torch.cuda.is_initialized():
        raise ValueError('Analysis unexpectedly initialized CUDA')
    row = dict(arm=arm, round_index=ri, update=(ri + 1) * UPDATES,
        checkpoint_path=str(path), checkpoint_sha256=receipt['sha256'],
        adapters_sha256=receipt['adapters_sha256'], modules=len(names),
        current_a_l2=math.sqrt(aa), current_b_l2=math.sqrt(bb),
        factor_change_from_round_start_l2=math.sqrt(delta),
        effective_adapter_weight_frobenius=math.sqrt(residual),
        effective_weight_change_from_round_start_frobenius=math.sqrt(effective_delta),
        old_equals_previous_current=prior_matches, cuda_initialized=False)
    return row, state


def update_summary(event, worker):
    row = {k: event[k] for k in ('arm', 'round_index', 'update', 'microbatches', 'examples',
        'gradient_norm_before_clip', 'clip_norm', 'actual_model_forwards', 'seconds',
        'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')}
    row['worker'] = str(worker)
    row['gradient_was_clipped'] = row['gradient_norm_before_clip'] > row['clip_norm']
    for key in ('loss', 'reference_velocity_mse', 'preference_loss', 'self_reflow_loss',
            'positive_endpoint_loss', 'negative_endpoint_loss', 'old_velocity_mse',
            'free_velocity_target_mse', 'positive_weight', 'negative_weight', 'endpoint_weight'):
        row[key + '_mean'] = mean(v.get(key) for v in event['rows'])
    row['reward_known'] = sum(v.get('reward_known', 0) for v in event['rows'])
    row['reward_unknown'] = sum(v.get('reward_unknown', 0) for v in event['rows'])
    row['reference_is_exact_kl'] = False
    return row


def read_training(inp):
    physical, curves, adapters, completed, previous = [], [], [], {a: [] for a in ARMS}, {}
    histories = {}
    for ri in range(ROUNDS):
        stage = D.OUT / f'training/round_{ri:02d}'
        for worker in sorted(stage.glob('train_*')):
            event_path = worker / 'events.jsonl'
            if not event_path.exists():
                continue
            events = inp.events(event_path)
            attempts = [v for v in events if v['event'] == 'UPDATE_START']
            updates = [v for v in events if v['event'] == 'UPDATE_COMPLETE']
            if not events:
                continue
            arm = events[0]['arm']
            if arm not in ARMS or any(v.get('arm') != arm or v.get('round_index') != ri for v in events):
                raise ValueError('Worker event identity mismatch')
            counts = Counter(v['event'] for v in events)
            row = dict(arm=arm, round_index=ri, worker=str(worker),
                physical_update_attempts=counts['UPDATE_START'], physical_completed_updates=counts['UPDATE_COMPLETE'],
                physical_forward_attempts=counts['FORWARD_START'], physical_forward_returns=counts['FORWARD_RETURN'],
                incomplete_update_attempts=counts['UPDATE_START'] - counts['UPDATE_COMPLETE'],
                numeric_replay_checks=counts['RECOVERY_NUMERIC_REPLAY_CHECK'],
                numeric_replay_checks_equal=sum(v.get('exact_logged_loss_and_grad_equal') is True for v in events if v['event'] == 'RECOVERY_NUMERIC_REPLAY_CHECK'),
                failure_events=counts['TRAINING_FAILURE'], logical_committed_updates=0,
                disposition='UNCOMMITTED_OR_ACTIVE', completed_update_seconds=sum(v['seconds'] for v in updates))
            if len({v['update'] for v in updates}) != len(updates):
                raise ValueError('Duplicate completed update in one worker')
            if len(updates) > len(attempts):
                raise ValueError('More completed than attempted updates')
            histories[(ri, arm, worker.name)] = (events, updates, row)
            completion = next((p for p in (worker / 'completion.json', worker / 'completion_recovery.json') if p.exists()), None)
            if completion:
                done = inp.read(completion)
                if done.get('status') == 'COMPLETE':
                    metrics = inp.read(worker / 'training_metrics.json')
                    expected = list(range(ri * UPDATES + 1, (ri + 1) * UPDATES + 1))
                    if ([v['update'] for v in updates] != expected or metrics['completed_updates'] != UPDATES
                            or metrics['status'] != 'COMPLETE' or metrics['actual_model_forwards'] != row['physical_forward_returns']
                            or row['physical_forward_returns'] != row['physical_forward_attempts']):
                        raise ValueError('Committed worker clock/forward metrics mismatch')
                    if ri in completed[arm]:
                        raise ValueError('Two committed workers for same arm/round')
                    if completed[arm] != list(range(ri)):
                        raise ValueError('Non-contiguous committed checkpoint chain')
                    receipt = inp.read(D.OUT / f'checkpoints/{arm}/round{ri:02d}.json')
                    ar, state = adapter_summary(inp, D.OUT / f'checkpoints/{arm}/round{ri:02d}.pt', receipt, arm, ri, previous.get(arm))
                    previous[arm] = state
                    adapters.append(ar)
                    completed[arm].append(ri)
                    row['logical_committed_updates'] = UPDATES
                    row['disposition'] = 'COMMITTED'
                    curves.extend(update_summary(v, worker) for v in updates)
            physical.append(row)
    recovery = dict(status='NOT_PRESENT', discarded_completed_updates=0, discarded_attempts=0,
        discarded_forward_attempts=0, discarded_forward_returns=0, replayed_completed_updates=0,
        NFT_replayed_updates=0)
    auth_path = D.OUT / 'engineering_recovery_round00/authorization.json'
    if auth_path.exists():
        auth = inp.read(auth_path)
        old = histories.get((0, 'SELF_FM', 'train_SELF_FM'))
        new = histories.get((0, 'SELF_FM', 'train_SELF_FM_recovery01'))
        if old is None:
            raise ValueError('Recovery has no original event ledger')
        original_path = D.OUT / 'training/round_00/train_SELF_FM/events.jsonl'
        if inp.sources[str(original_path.relative_to(D.ROOT))]['sha256'] != auth['old_events_sha256']:
            raise ValueError('Discarded event ledger changed')
        fields = [('physical_completed_updates', 'discarded_complete_updates'),
            ('physical_update_attempts', 'discarded_update_attempts'),
            ('physical_forward_attempts', 'discarded_training_forward_attempts'),
            ('physical_forward_returns', 'discarded_training_forward_returns')]
        if any(old[2][k] != auth[a] for k, a in fields):
            raise ValueError('Recovery authorization accounting differs')
        old[2]['disposition'] = 'DISCARDED_ENGINEERING_INTERRUPTION'
        replay_count = min(len(old[1]), len(new[1])) if new else 0
        for i in range(replay_count):
            a, b = old[1][i], new[1][i]
            if a['rows'] != b['rows'] or a['gradient_norm_before_clip'] != b['gradient_norm_before_clip']:
                raise ValueError('Replayed recorded loss/gradient mismatch')
        recovery = dict(status='COMPLETE' if new and new[2]['logical_committed_updates'] == UPDATES else 'PARTIAL',
            authorization=str(auth_path), discarded_completed_updates=old[2]['physical_completed_updates'],
            discarded_attempts=old[2]['physical_update_attempts'], discarded_forward_attempts=old[2]['physical_forward_attempts'],
            discarded_forward_returns=old[2]['physical_forward_returns'], replayed_completed_updates=replay_count,
            NFT_replayed_updates=0, original_parameter_state_not_saved=True,
            replay_evidence='Exact original recorded microbatch loss rows and preclip gradient norms; not bytewise original full-state proof',
            NFT_recovery='Durable 16-update checkpoint closed by CPU receipt validation; zero replay')
    return physical, curves, adapters, completed, recovery


def read_rewards(inp, manifest):
    per_output, per_condition, search = [], [], []
    for ri in range(ROUNDS):
        stage = D.OUT / f'training/round_{ri:02d}'
        path = stage / 'rewards.json'
        if not path.exists():
            continue
        rewards = inp.read(path)
        expected = {q['request_id']: q for q in manifest['training_requests'] if q['round_index'] == ri}
        rows = rewards['rows']
        if rewards['status'] != 'COMPLETE' or len(rows) != 32 or {r['request_id'] for r in rows} != set(expected):
            raise ValueError('Reward set incomplete or not the registered 32 requests')
        if (stage / 'learning_input_seal.json').exists():
            seal = inp.read(stage / 'learning_input_seal.json')
            for filename in ('rewards.json', 'reviews_actual.json', 'training_stream.json'):
                p = stage / filename
                inp.blob(p)
                matching = [s for s, h in seal['hashes'].items() if D.resolve(s).resolve() == p.resolve()
                    and h == inp.sources[str(p.relative_to(D.ROOT))]['sha256']]
                if len(matching) != 1:
                    raise ValueError('Learning evidence no longer matches its seal')
        groups = defaultdict(list)
        for r in rows:
            q = expected[r['request_id']]
            if r['condition_id'] != q['condition_id'] or r['parent_id'] != q['parent_id']:
                raise ValueError('Reward condition mapping differs')
            flat = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            for section in ('review_labels', 'components', 'quality_factors'):
                flat.update({section + '__' + k: v for k, v in r.get(section, {}).items()})
            flat.update(round_index=ri, rollout_policy='FROZEN_FULL' if ri == 0 else 'ONLINE_NFT',
                rollout_update=ri * UPDATES, noise_seed=q['noise_seed'])
            per_output.append(flat); groups[r['condition_id']].append(r)
        if len(groups) != 4 or any(len(v) != 8 for v in groups.values()):
            raise ValueError('Expected four conditions with eight samples each')
        for cid, group in sorted(groups.items()):
            vals = [r['reward'] for r in group if r['reward'] is not None]
            known = [r for r in group if r['reward'] is not None]
            pc = dict(round_index=ri, condition_id=cid, parent_id=group[0]['parent_id'], count=8,
                reward_known=len(vals), reward_unknown=8-len(vals), raw_reward_mean=mean(vals),
                raw_reward_std_population=statistics.pstdev(vals) if vals else None,
                raw_reward_min=min(vals) if vals else None, raw_reward_max=max(vals) if vals else None,
                preference_known=sum(r['preference_known'] for r in group), r_NFT_mean=mean(r['r_NFT'] for r in group),
                review_gate_pass_count=sum(r['review_gate'] == 1 for r in group),
                positive_reward_count=sum(r['reward'] is not None and r['reward'] > 0 for r in group),
                legal_CF_pass_count=sum(r['legal_CF_edges'] is not None and r['legal_CF_edges'] > 0 for r in group),
                strict_joint_pass_count=sum(r['review_gate'] == 1 and r['legal_CF_edges'] is not None and r['legal_CF_edges'] > 0 for r in group),
                rollout_policy='FROZEN_FULL' if ri == 0 else 'ONLINE_NFT', rollout_update=ri * UPDATES)
            for label in LABELS:
                counts = Counter(r['review_labels'][label] for r in group)
                for value in ('YES', 'NO', 'UNKNOWN', 'UNCERTAIN', 'NA'):
                    pc[label + '__' + value] = counts[value]
            per_condition.append(pc)
            best = sorted(known, key=lambda r: (-r['reward'], r['request_id']))[0] if known else None
            complete = len(known) == 8
            search.append(dict(pc, search_scope='ROUND00_FROZEN_FULL_FOUR_TRAIN_CONDITIONS' if ri == 0 else 'ONLINE_CURRENT_NFT_CONDITIONS',
                search_status='COMPLETE_EIGHT' if complete else 'PARTIAL_KNOWN_REWARDS',
                best_request_id=best['request_id'] if best else None, best_reward=best['reward'] if best else None,
                uniform_draw_from_registered_eight_expected_reward=mean(vals) if complete else None,
                best_of_eight_gain_over_uniform_draw=best['reward']-mean(vals) if complete else None,
                uniform_draw_review_gate_confirmed_yes_fraction=pc['review_gate_pass_count']/8,
                uniform_draw_strict_joint_confirmed_yes_fraction=pc['strict_joint_pass_count']/8,
                review_gate_unknown_count=sum(r['review_gate'] is None for r in group),
                best_review_gate=best['review_gate'] if best else None,
                best_has_legal_CF=(best['legal_CF_edges'] > 0) if best and best['legal_CF_edges'] is not None else None))
    return per_output, per_condition, search


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader(); writer.writerows(rows)
    payload = stream.getvalue().encode('utf-8-sig')
    path = D.checked(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Existing analysis CSV differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(payload)


def plots(out, curves, adapters):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return dict(status='UNAVAILABLE', reason='matplotlib absent; full curve CSV/JSON provided')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for arm in ARMS:
        rows = [r for r in curves if r['arm'] == arm]
        for ax, key, title in zip(axes.flat[:3], ('loss_mean', 'gradient_norm_before_clip', 'reference_velocity_mse_mean'),
                ('Own objective (different scales across arms)', 'Gradient norm before clipping', 'Reference velocity MSE (not KL)')):
            ax.plot([r['update'] for r in rows], [r[key] for r in rows], label=arm)
            ax.set_title(title); ax.set_xlabel('Committed logical update'); ax.grid(alpha=.2)
        rows = [r for r in adapters if r['arm'] == arm]
        axes.flat[3].plot([r['update'] for r in rows], [r['effective_adapter_weight_frobenius'] for r in rows], marker='o', label=arm)
    axes.flat[3].set_title('Effective adapter weight Frobenius norm')
    axes.flat[3].set_xlabel('Committed logical update')
    for ax in axes.flat:
        ax.legend()
    path = out / 'learning_curves.png'
    if not path.exists():
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(status='WRITTEN', path=str(path), sha256=D.sha(path))


def run():
    inp = Inputs()
    inp.blob(Path(__file__))
    manifest = inp.read(D.OUT / 'inputs/manifest.json')
    physical, curves, adapters, completed, recovery = read_training(inp)
    outputs, conditions, search = read_rewards(inp, manifest)
    reward_rounds = sorted({r['round_index'] for r in conditions})
    total_keys = ('physical_update_attempts', 'physical_completed_updates', 'physical_forward_attempts',
        'physical_forward_returns', 'incomplete_update_attempts', 'logical_committed_updates')
    totals = {k: sum(r[k] for r in physical) for k in total_keys}
    totals['logical_committed_forward_returns'] = sum(r['actual_model_forwards'] for r in curves)
    totals['per_arm'] = {arm: {k: sum(r[k] for r in physical if r['arm'] == arm) for k in total_keys} for arm in ARMS}
    frozen = [r for r in search if r['round_index'] == 0]
    online = [r for r in search if r['round_index'] != 0]
    status = 'COMPLETE' if all(completed[a] == list(range(ROUNDS)) for a in ARMS) and reward_rounds == list(range(ROUNDS)) else 'PARTIAL'
    digest = hashlib.sha256(json.dumps(inp.sources, sort_keys=True).encode()).hexdigest()[:20]
    out = D.OUT / 'learning_analysis/snapshots' / digest
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('training_updates', curves), ('physical_workers', physical), ('adapter_changes', adapters),
            ('reward_outputs', outputs), ('reward_conditions', conditions), ('bestof8_frozen_FULL_round00', frozen), ('bestof8_online_NFT', online)):
        D.write(out / (name + '.json'), rows)
        write_csv(out / (name + '.csv'), rows)
    summary = dict(version=VERSION, status=status, scope='TRAINING_EVIDENCE_ONLY_NOT_FINAL_EVALUATION',
        completed_rounds=completed, reward_rounds=reward_rounds, planned_rounds=ROUNDS,
        planned_logical_updates_per_arm=ROUNDS * UPDATES, totals=totals, engineering_recovery=recovery,
        frozen_FULL_round00_search=dict(conditions=len(frozen), outputs=sum(r['count'] for r in frozen),
            status='COMPLETE_EIGHT_FOR_FOUR' if len(frozen) == 4 and all(r['search_status'] == 'COMPLETE_EIGHT' for r in frozen) else 'PARTIAL',
            mean_condition_random_eight_reward=mean(r['uniform_draw_from_registered_eight_expected_reward'] for r in frozen),
            mean_condition_best_reward=mean(r['best_reward'] for r in frozen),
            mean_bestof8_gain=mean(r['best_of_eight_gain_over_uniform_draw'] for r in frozen),
            review_gate_pass_count=sum(r['review_gate_pass_count'] for r in frozen),
            strict_joint_pass_count=sum(r['strict_joint_pass_count'] for r in frozen),
            scope='Only initial four TRAIN conditions; excludes other four TRAIN conditions and all DEV'),
        definitions=dict(review_gate='All three original review labels YES with scorable geometry',
            strict_joint='Review gate passes and POST legal CF edge count > 0',
            pass_fractions='Confirmed YES / registered eight; UNKNOWN retained separately, never imputed as NO',
            gradient_curve='Preclip norm from original optimizer update logs',
            adapter_norm='Across 48 modules, sqrt(sum ||(alpha/rank) B A||_F^2)',
            adapter_round_change='Across modules, Frobenius norm of current effective BA minus old round-start BA'),
        interpretation=[
            'Online rewards use changing policies, alternating condition groups and fresh registered noises; trends are not fixed-test improvement.',
            'r_NFT is within-condition normalized and cannot establish increasing absolute quality.',
            'Best-of-eight is descriptive selection on the same eight observed rewards, not an independent learned-policy gain.',
            'Uniform baseline is the mean of the same eight candidates, without drawing or generating another sample.',
            'Original AI_PROXY/human labels are retained; this analysis performs no semantic review.',
            'SELF_FM reuses NFT endpoint pools off policy and has no reward term; objective magnitudes across arms are not directly comparable.',
            'Reference velocity MSE is not an exact distribution KL; weight norms are not output-quality measures.',
            'Completed rounds require durable checkpoint and completion receipt. Active physical attempts are listed separately.',
            'Concurrent worker seconds are observations of that dispatch, not isolated throughput benchmarks.'],
        partial_event_tails=inp.partial_tails, model_forwards_performed_by_analysis=0,
        plots=plots(out, curves, adapters), input_sources=inp.sources)
    D.write(out / 'summary.json', summary)
    print(json.dumps(dict(status=status, output=str(out), completed_rounds=completed, totals=totals), ensure_ascii=False))
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
