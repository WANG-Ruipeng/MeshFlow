"""Independent read-only CPU accounting of the terminal-reward NFT pilot.

Only audit/snapshots and audit/latest.json are written. Existing RAW, D0,
review, score and transaction records are checked; no model is constructed,
no score or visual label is recomputed, and no worker is launched. Small
adapter/AdamW checkpoints are deserialized on CPU; base weights are hash-only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

from . import data as D

VERSION = 'terminal_reward_nft_execution_audit_v1'
ARMS = ('NFT', 'SELF_FM')
LABELS = ('BASE', 'NFT', 'SELF_FM')
REVIEW_FIELDS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
REVIEW_VALUES = {'YES', 'NO', 'UNKNOWN', 'UNCERTAIN', 'NA'}
D0_SHA = '6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'


class Audit:
    def __init__(self):
        self.sources, self.values, self.stats = {}, {}, {}
        self.violations, self.partial, self.changes = [], [], []
        self.definitions = None
        self.code_sha = D.sha(__file__)

    def fail(self, scope, message):
        self.violations.append(dict(scope=str(scope), message=str(message)))

    def check(self, condition, scope, message):
        if not condition:
            self.fail(scope, message)
        return bool(condition)

    def pending(self, scope, reason):
        self.partial.append(dict(scope=str(scope), reason=reason))

    def guard(self, scope, fn, fallback=None):
        try:
            return fn()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as exc:
            self.fail(scope, repr(exc))
            return fallback

    def key(self, path):
        return os.path.abspath(D.resolve(path))

    def stamp(self, path):
        s = D.resolve(path).stat()
        return (s.st_size, s.st_mtime_ns, s.st_ino)

    def _record(self, path, digest, size, before, after):
        key = self.key(path)
        self.sources[key] = dict(sha256=digest, bytes=size)
        self.stats[key] = after
        if before != after:
            self.changes.append(dict(path=key, reason='CHANGED_WHILE_READING'))
        return digest

    def blob(self, path):
        before = self.stamp(path)
        payload = D.resolve(path).read_bytes()
        self._record(path, hashlib.sha256(payload).hexdigest(), len(payload), before, self.stamp(path))
        return payload

    def hash(self, path):
        key = self.key(path)
        if key in self.sources:
            return self.sources[key]['sha256']
        before = self.stamp(path)
        h = hashlib.sha256()
        with D.resolve(path).open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                h.update(block)
        return self._record(path, h.hexdigest(), before[0], before, self.stamp(path))

    def read(self, path):
        key = self.key(path)
        if key not in self.values:
            self.values[key] = json.loads(self.blob(path).decode('utf-8-sig'))
        return self.values[key]

    def events(self, path):
        payload = self.blob(path)
        if payload and not payload.endswith(b'\n'):
            payload, _, tail = payload.rpartition(b'\n')
            self.pending(path, 'PARTIAL_EVENT_TAIL_' + str(len(tail)) + '_BYTES')
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    def verify(self, mapping, scope):
        for path, expected in mapping.items():
            self.guard(scope, lambda p=path, h=expected: self.check(self.hash(p) == h, scope, 'HASH_CHANGED: ' + str(p)))

    def finish_reads(self):
        for path, stamp in self.stats.items():
            if self.stamp(path) != stamp:
                self.changes.append(dict(path=path, reason='CHANGED_AFTER_SNAPSHOT_READ'))
        if self.changes:
            self.pending('snapshot', 'Inputs were still changing; rerun when transactions finish')


def event_pairs(a, events, scope, *, completed=False):
    starts = [e.get('count') for e in events if e['event'] == 'FORWARD_START']
    returns = [e.get('count') for e in events if e['event'] == 'FORWARD_RETURN']
    a.check(starts == list(range(1, len(starts) + 1)), scope, 'Forward-start sequence is not contiguous and unique')
    a.check(returns == starts[:len(returns)] and len(returns) <= len(starts), scope, 'Forward-return sequence does not match starts')
    if completed:
        a.check(starts == returns, scope, 'Completed transaction has unmatched forwards')
    return Counter(e['event'] for e in events)


def score_record(a, path, q, label, stage):
    score = a.read(path)
    identity = score['identity']
    for key, expected in dict(request_id=q['request_id'], condition_id=q['condition_id'],
                              label=label, stage=stage, N=q['N'], K=q['K']).items():
        a.check(identity.get(key) == expected, path, 'Score identity differs: ' + key)
    a.check(score.get('status') == 'PASS' and score.get('C_preserved_exact') is True,
            path, 'Score status/hard C preservation does not pass')
    a.check(score.get('model_calls') == 0 and score.get('new_queries') == 0,
            path, 'Stored scorer records unexpected model/query calls')
    a.verify(score['files'], path)
    a.verify(identity['input_hashes'], path)
    definitions = identity['definition_hashes']
    a.verify(definitions, 'original_scoring_definitions')
    if a.definitions is None:
        a.definitions = definitions
    else:
        a.check(definitions == a.definitions, path, 'Scoring definition versions differ across outputs')
    return dict(path=str(path), sha256=a.hash(path), status=score['status'])


def sampling_stage(a, manifest, stage, requests, label, ri=None):
    expected = {q['request_id']: q for q in requests}
    for q in requests:
        a.verify({q['input_path']: q['input_sha256'], q['condition_source']: q['condition_source_sha256']}, stage)
    result = dict(stage=str(stage), label=label, round_index=ri, expected_requests=len(requests),
                  RAW_receipts=0, D0_attempts=0, D0_returns=0, raw_complete=False, D0_complete=False)
    all_events = []
    index_path = stage / 'raw_index.json'
    index = a.read(index_path) if index_path.exists() else None
    for path in sorted(stage.glob('sampling_events*.jsonl')):
        events = a.events(path)
        event_pairs(a, events, path, completed=index is not None)
        all_events.extend(events)
    counts = Counter(e['event'] for e in all_events)
    result.update(rollout_attempts=counts['ROLLOUT_ATTEMPT'], rollout_returns=counts['ROLLOUT_COMPLETE'],
                  sampling_forward_attempts=counts['FORWARD_START'], sampling_forward_returns=counts['FORWARD_RETURN'],
                  validation_forward_attempts=counts['VALIDATION_FORWARD_START'], validation_forward_returns=counts['VALIDATION_FORWARD_RETURN'])
    for e in all_events:
        if e['event'] == 'ZERO_LORA_CHECK':
            a.check(e.get('bitwise_equal') is True and e.get('maxabs') == 0, stage, 'Initial zero-LoRA equivalence failed')
    starts = [e['context']['request_id'] for e in all_events if e['event'] == 'ROLLOUT_ATTEMPT']
    a.check(len(starts) == len(set(starts)) and set(starts) <= set(expected), stage, 'Duplicate or unregistered rollout attempt')
    by_request = defaultdict(list)
    for e in all_events:
        if e['event'] == 'FORWARD_START':
            by_request[e['request_id']].append(e['euler_step'])
    for rid, steps in by_request.items():
        a.check(rid in expected and steps == list(range(len(steps))) and len(steps) <= 50,
                stage, 'Euler step sequence differs: ' + rid)
    sys.path.insert(0, str(D.ROOT / 'r2/src')) if str(D.ROOT / 'r2/src') not in sys.path else None
    from .sample import verify_receipt
    receipts = {}
    for path in sorted((stage / 'raw').glob('*/receipt.json')):
        r = a.read(path)
        rid = r['request_id']
        if rid not in expected:
            a.fail(path, 'Unregistered RAW output'); continue
        q = expected[rid]
        a.check(rid not in receipts and r['label'] == label, path, 'Duplicate RAW or wrong label')
        receipts[rid] = r
        a.verify({q['input_path']: q['input_sha256'], q['condition_source']: q['condition_source_sha256']}, path)
        a.verify({r[k + '_path']: r[k + '_sha256'] for k in ('raw', 'ply', 'result')}, path)
        facts = verify_receipt(q, r)
        a.check(all(facts.values()) and r.get('new_rollouts') == 1 and r.get('new_sampling_F') == 50,
                path, 'RAW preservation/count contract differs')
        identity = r['model_identity']
        a.check(identity['base_sha256'] == manifest['source_binding']['sha256'], path, 'Wrong frozen base')
        cp = (D.OUT / f'checkpoints/NFT/round{ri - 1:02d}.pt' if ri is not None and ri > 0
              else D.OUT / f'checkpoints/{label}/round15.pt' if ri is None and label != 'BASE' else None)
        if cp is None:
            a.check(identity.get('checkpoint_path') is None, path, 'Unexpected non-base sampling checkpoint')
        else:
            cr = a.read(cp.with_suffix('.json'))
            a.check(D.resolve(identity['checkpoint_path']) == cp and identity['checkpoint_sha256'] == a.hash(cp)
                    and identity['adapter_sha256'] == cr['adapters_sha256'], path, 'Wrong sampling adapter checkpoint')
        result['RAW_receipts'] += 1
    if index is not None:
        a.check(index.get('status') == 'COMPLETE' and index.get('label') == label, index_path, 'RAW index is not complete')
        rows = index['receipts']
        a.check(len(rows) == len(expected) and {r['request_id'] for r in rows} == set(expected), index_path, 'RAW index request coverage differs')
        a.check(all(r == receipts.get(r['request_id']) for r in rows), index_path, 'Index and per-output receipt differ')
        n = len(expected)
        a.check(index['actual_new_rollouts'] == n == result['RAW_receipts'] == len(starts)
                and index['actual_sampling_forwards'] == 50 * n == counts['FORWARD_RETURN'] == counts['FORWARD_START']
                and counts['ROLLOUT_COMPLETE'] == n, index_path, 'Completed sampling count mismatch')
        phases = list(stage.glob('sampling_worker*/completion.json'))
        a.pending(stage, 'RAW index saved; sampling Phase completion not yet present') if not phases else None
        for phase in phases:
            a.check(a.read(phase)['status'] == 'COMPLETE', phase, 'Sampling Phase not complete')
        result['raw_complete'] = True
    else:
        a.pending(stage, 'RAW index not complete')
    for path in sorted((stage / 'proposal').glob('*/started.json')):
        v = a.read(path); rid = v['request_id']
        a.check(rid in expected and v.get('event') == 'D0_FORWARD_STARTED' and v.get('checkpoint_sha256') == D0_SHA,
                path, 'D0 attempt identity differs')
        result['D0_attempts'] += 1
    for path in sorted((stage / 'proposal').glob('*/receipt.json')):
        v = a.read(path); rid = v['request_id']
        a.check(rid in expected and rid in receipts and v.get('status') == 'PASS' and v.get('model_forwards') == 1
                and v.get('checkpoint_sha256') == D0_SHA, path, 'D0 return identity differs')
        if rid in receipts:
            a.check(v['raw_sha256'] == receipts[rid]['raw_sha256'], path, 'D0 raw hash differs')
        a.verify({path.parent / 'proposal.npz': v['output_sha256']}, path)
        result['D0_returns'] += v['model_forwards']
    d0_index = stage / 'proposal_index.json'
    if d0_index.exists():
        v = a.read(d0_index)
        a.check(v.get('status') == 'COMPLETE' and v['jobs'] == len(expected) == result['D0_returns'] == result['D0_attempts']
                and v['new_D0_forwards'] == len(expected), d0_index, 'Complete D0 budget differs')
        a.check({r['request_id'] for r in v['receipts']} == set(expected) and len(v['receipts']) == len(expected), d0_index, 'D0 index coverage differs')
        for r in v['receipts']:
            a.verify({r['proposal_path']: r['proposal_sha256']}, d0_index)
        if ri is not None:
            phase = stage / 'denoise_worker/completion.json'
            if not phase.exists():
                a.pending(stage, 'D0 index saved; Phase completion not yet present')
            else:
                a.check(a.read(phase).get('status') == 'COMPLETE', phase, 'D0 Phase not complete')
        result['D0_complete'] = True
    else:
        a.pending(stage, 'D0 index not complete')
    a.check(result['D0_attempts'] <= len(expected) and result['D0_returns'] <= result['D0_attempts'], stage, 'D0 attempts exceed authorized request set')
    return result


def actual_reviews(a, path, gallery_path, requests, *, final=False):
    review = a.read(path); rows = review['rows'] if isinstance(review, dict) else review
    gallery = a.read(gallery_path)
    images = {v['request_id']: v for v in gallery['images']}
    expected = {(q['request_id'], label) for q in requests for label in (('A', 'B', 'C') if final else ('NFT',))}
    seen = set()
    a.check(gallery.get('status') == 'COMPLETE', gallery_path, 'Reviewed gallery not complete')
    if isinstance(review, dict) and 'gallery_manifest_sha256' in review:
        a.check(review['gallery_manifest_sha256'] == a.hash(gallery_path), path, 'Review gallery hash differs')
    for row in rows:
        key = row['request_id'], row['label']
        a.check(key not in seen and key in expected, path, 'Review duplicated or unexpected')
        seen.add(key)
        a.check(row.get('review_status') == 'REVIEWED' and row.get('reviewer_kind') in ('AI_PROXY', 'HUMAN')
                and bool(row.get('reviewer')), path, 'Actual reviewer record missing')
        a.check(all(row.get(k) in REVIEW_VALUES for k in REVIEW_FIELDS), path, 'Unknown review label vocabulary')
        if final:
            a.check(row.get('blinded') is True, path, 'Final actual review is not marked blinded')
        im = images[row['request_id']]
        a.check(row['image_sha256'] == im['image_sha256'], path, 'Review refers to a different image')
        a.verify({im['image_path']: im['image_sha256']}, path)
        a.verify(im.get('source_files', {}), gallery_path)
        if row.get('viewed_contact_sheet_path'):
            a.verify({row['viewed_contact_sheet_path']: row['viewed_contact_sheet_sha256']}, path)
    a.check(seen == expected and len(rows) == len(expected), path, 'Actual review coverage incomplete')
    return dict(rows=len(rows), by_reviewer=dict(Counter(r['reviewer_kind'] for r in rows)),
                unknown_labels=sum(r[k] not in ('YES', 'NO') for r in rows for k in REVIEW_FIELDS),
                path=str(path), sha256=a.hash(path), rows_by_id={r['request_id']: r for r in rows} if not final else None)


def review_round(a, manifest, ri, requests):
    stage = D.OUT / f'training/round_{ri:02d}'
    path = stage / 'reviews_actual.json'
    out = dict(round_index=ri, review_rows=0, seal_complete=False, training_stream_rows=0)
    if not path.exists():
        a.pending(stage, 'Actual training review unavailable'); return out
    reviews = actual_reviews(a, path, stage / 'gallery/manifest.json', requests)
    out.update(review_rows=reviews['rows'], unknown_labels=reviews['unknown_labels'])
    seal_path = stage / 'learning_input_seal.json'
    if not seal_path.exists():
        a.pending(stage, 'Learning input seal unavailable'); return out
    seal = a.read(seal_path)
    a.check(seal['status'] == 'PASS', seal_path, 'Learning input seal did not pass')
    a.verify(seal['hashes'], seal_path)
    required = [stage / n for n in ('reviews_actual.json', 'rewards.json', 'reward_gate.json', 'training_stream.json', 'gallery/manifest.json')]
    required += [D.OUT / 'inputs/manifest.json', D.OUT / 'inputs/training_stream_schedule.json']
    required += [stage / 'scores' / q['request_id'] / 'POST/score.json' for q in requests]
    a.check({a.key(p) for p in required} <= {a.key(p) for p in seal['hashes']}, seal_path, 'Seal omits learning evidence')
    rewards = a.read(stage / 'rewards.json')
    rows = rewards['rows']; by_id = {r['request_id']: r for r in rows}
    a.check(rewards['status'] == 'COMPLETE' and len(rows) == 32 and set(by_id) == {q['request_id'] for q in requests}, stage, 'Reward coverage differs')
    for q in requests:
        rid = q['request_id']; r = by_id[rid]; rev = reviews['rows_by_id'][rid]
        a.check(r['condition_id'] == q['condition_id'] and r['parent_id'] == q['parent_id'], stage, 'Reward condition differs')
        a.check(r['review_labels'] == {k: rev[k] for k in REVIEW_FIELDS}, stage, 'Reward labels differ from actual review')
        for score_stage in ('RAW', 'POST'):
            score_record(a, stage / 'scores' / rid / score_stage / 'score.json', q, 'NFT', score_stage)
    stream = a.read(stage / 'training_stream.json')
    schedule = a.read(D.OUT / 'inputs/training_stream_schedule.json')[ri]['items']
    a.check(len(stream) == 128 == len(schedule), stage, 'Training stream must contain exactly four passes')
    for row, locked in zip(stream, schedule):
        a.check(all(row[k] == v for k, v in locked.items()), stage, 'Renoising/time/shuffle schedule changed')
        r = by_id[row['request_id']]
        raw = a.read(stage / 'raw' / row['request_id'] / 'receipt.json')
        a.check(row['r'] == r['r_NFT'] and row['known_reward'] == r['preference_known']
                and row['raw_sha256'] == raw['raw_sha256'] and D.resolve(row['raw_path']) == D.resolve(raw['raw_path']), stage, 'Training endpoint or reward stream differs')
    a.check(a.read(stage / 'reward_gate.json')['status'] == 'PASS', stage, 'Learning started without passed reward gate')
    out.update(seal_complete=True, sealed_files=len(seal['hashes']), training_stream_rows=len(stream))
    return out


def tensor_hash(state, torch):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint(a, path, arm, ri, manifest):
    receipt = a.read(path.with_suffix('.json'))
    digest = a.hash(path)
    if not a.check(digest == receipt['sha256'], path, 'Checkpoint receipt hash differs'):
        return None
    if not a.check(path.stat().st_size <= 64 * 1024 * 1024, path, 'Refusing non-adapter checkpoint larger than 64 MiB'):
        return None
    a.check((receipt['step'], receipt['round_index']) == ((ri + 1) * 16, ri), path, 'Checkpoint sidecar clock differs')
    cache = D.OUT / 'audit/checkpoint_cache' / a.code_sha / (digest + '.json')
    if cache.exists():
        row = a.read(cache)
        valid = a.check((row['checkpoint_sha256'], row['arm'], row['round_index'], row.get('auditor_sha256'), row.get('validation_status')) == (digest, arm, ri, a.code_sha, 'PASS'), cache, 'CPU checkpoint cache identity differs')
        valid = a.check(receipt['adapters_sha256'] == row['adapters_sha256'], cache, 'Cached adapter/sidecar hash differs') and valid
        return row if valid else None
    before_errors = len(a.violations)
    import torch
    torch.set_num_threads(1)
    value = torch.load(io.BytesIO(a.blob(path)), map_location='cpu', weights_only=True)
    step = (ri + 1) * 16
    a.check((value['schema'], value['arm'], value['round_index'], value['step'], value['update_in_progress'])
            == ('terminal_reward_nft_training_v1', arm, ri, step, False), path, 'Checkpoint transaction clock differs')
    a.check((receipt['step'], receipt['round_index']) == (step, ri), path, 'Checkpoint sidecar clock differs')
    state, meta = value['adapters'], value['model_metadata']
    a.check(meta['base_file_sha256'] == manifest['source_binding']['sha256'] and meta['rank'] == 4
            and meta['alpha'] == 4 and meta['dropout'] == 0 and meta['trainable_parameters'] == 589824, path, 'Base/LoRA metadata differs')
    a.check(tensor_hash(state, torch) == value['adapters_sha256'] == receipt['adapters_sha256'], path, 'Adapter tensor hash differs')
    names = [name + '.' + suffix for name in meta['target_names'] for suffix in ('current_a', 'current_b')]
    a.check(len(meta['target_names']) == 48 and len(state) == 192 and value['optimizer_names'] == names, path, 'Adapter/optimizer inventory differs')
    a.check(all(v.device.type == 'cpu' and v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()), path, 'Adapter state not finite FP32 CPU')
    opt = value['optimizer']; groups = opt['param_groups']
    a.check(len(groups) == 1, path, 'Unexpected AdamW groups')
    group = groups[0]
    a.check((group['lr'], tuple(group['betas']), group['eps'], group['weight_decay'], group['amsgrad'], group['foreach'], group['fused'])
            == (1e-5, (.9, .999), 1e-8, 0, False, False, False), path, 'AdamW hyperparameters differ')
    a.check(len(group['params']) == len(names) == len(opt['state']) == 96, path, 'AdamW parameter inventory differs')
    steps = []
    for pid, name in zip(group['params'], names):
        s = opt['state'][pid]
        a.check(set(s) == {'step', 'exp_avg', 'exp_avg_sq'}, path, 'Unexpected AdamW state fields')
        steps.append(float(s['step']))
        for k in ('exp_avg', 'exp_avg_sq'):
            v = s[k]
            a.check(v.device.type == 'cpu' and v.dtype == torch.float32 and v.shape == state[name].shape and bool(torch.isfinite(v).all()), path, 'AdamW moment invalid: ' + name)
    a.check(all(v == step for v in steps), path, 'AdamW steps differ from completed logical update')
    if ri == 0:
        a.check(all(bool((v == 0).all()) for k, v in state.items() if k.endswith('.old_b')), path, 'Initial old adapter not zero residual')
    a.check(not torch.cuda.is_initialized(), path, 'Audit initialized CUDA')
    if len(a.violations) != before_errors:
        return None
    row = dict(validation_status='PASS', arm=arm, round_index=ri, step=value['step'], checkpoint_path=str(path), checkpoint_sha256=digest,
               adapters_sha256=value['adapters_sha256'], optimizer_states=len(steps), optimizer_step_min=min(steps), optimizer_step_max=max(steps),
               current_sha256=tensor_hash({k.replace('.current_', '.'): v for k, v in state.items() if '.current_' in k}, torch),
               old_sha256=tensor_hash({k.replace('.old_', '.'): v for k, v in state.items() if '.old_' in k}, torch),
               auditor_sha256=a.code_sha, CUDA_initialized=False)
    return row


def training(a, manifest):
    workers, checkpoints, histories = [], [], {}
    committed = {arm: [] for arm in ARMS}
    previous = {}
    for ri in range(16):
        stage = D.OUT / f'training/round_{ri:02d}'
        stream = a.read(stage / 'training_stream.json') if (stage / 'training_stream.json').exists() else None
        for worker in sorted(stage.glob('train_*')):
            path = worker / 'events.jsonl'
            if not path.exists(): continue
            events = a.events(path)
            if not events: continue
            arm = events[0]['arm']
            a.check(arm in ARMS and all(e.get('arm') == arm and e.get('round_index') == ri for e in events), worker, 'Worker event identity differs')
            completion = next((p for p in (worker / 'completion.json', worker / 'completion_recovery.json') if p.exists()), None)
            counts = event_pairs(a, events, worker, completed=completion is not None)
            branch_order = ('old', 'reference', 'current') if arm == 'NFT' else ('reference', 'current')
            forward_starts = [e for e in events if e['event'] == 'FORWARD_START']
            a.check(len(forward_starts) <= 16 * 8 * len(branch_order), worker, 'Worker forward budget exceeded')
            for i, e in enumerate(forward_starts):
                c = e['context']; width = len(branch_order)
                a.check(c.get('branch') == branch_order[i % width] and c.get('microbatch') == (i // width) % 8
                        and c.get('update') == ri * 16 + i // (8 * width) + 1, worker, 'Training branch/microbatch/update call order differs')
            updates = [e for e in events if e['event'] == 'UPDATE_COMPLETE']
            starts = [e for e in events if e['event'] == 'UPDATE_START']
            a.check(len(starts) <= 16 and len(updates) <= len(starts), worker, 'Update attempts exceed worker budget')
            a.check([e['update'] for e in starts] == list(range(ri * 16 + 1, ri * 16 + 1 + len(starts))), worker, 'Update attempt order differs')
            a.check([e['update'] for e in updates] == [e['update'] for e in starts[:len(updates)]], worker, 'Update completion order differs')
            for e in starts:
                if stream is not None:
                    i = e['stream_row_start']
                    a.check(e['request_ids'] == [r['request_id'] for r in stream[i:i + 8]], worker, 'Arm did not consume frozen cloned stream')
            for e in updates:
                a.check(e['microbatches'] == 8 and e['actual_model_forwards'] == (24 if arm == 'NFT' else 16)
                        and len(e['rows']) == 8, worker, 'Completed update microbatch/forward budget differs')
            row = dict(worker=str(worker), arm=arm, round_index=ri, update_attempts=counts['UPDATE_START'],
                       completed_updates=counts['UPDATE_COMPLETE'], forward_attempts=counts['FORWARD_START'], forward_returns=counts['FORWARD_RETURN'],
                       logical_committed_updates=0, disposition='UNCOMMITTED_OR_ACTIVE',
                       failure_events=counts['TRAINING_FAILURE'], replay_checks=counts['RECOVERY_NUMERIC_REPLAY_CHECK'],
                       replay_checks_equal=sum(e.get('exact_logged_loss_and_grad_equal') is True for e in events if e['event'] == 'RECOVERY_NUMERIC_REPLAY_CHECK'))
            histories[ri, arm, worker.name] = (row, updates)
            if completion:
                done = a.read(completion); metrics = a.read(worker / 'training_metrics.json')
                a.check(done['status'] == 'COMPLETE' and len(updates) == 16 and metrics['completed_updates'] == 16
                        and metrics['stop_step'] == (ri + 1) * 16 and metrics['actual_model_forwards'] == row['forward_returns'], completion, 'Completed training transaction mismatch')
                a.check(metrics['manifest_sha256'] == a.hash(D.OUT / 'inputs/manifest.json')
                        and metrics['training_stream_sha256'] == a.hash(stage / 'training_stream.json'), completion, 'Training inputs differ from sealed stream')
                cp = D.OUT / f'checkpoints/{arm}/round{ri:02d}.pt'
                cr = a.guard(cp, lambda: checkpoint(a, cp, arm, ri, manifest))
                if cr is not None:
                    if arm in previous:
                        a.check(cr['old_sha256'] == previous[arm]['current_sha256'], cp, 'Old adapter is not preceding round current snapshot')
                    a.check(committed[arm] == list(range(ri)), cp, 'Committed round chain is duplicated or noncontiguous')
                    previous[arm] = cr; checkpoints.append(cr); committed[arm].append(ri)
                    if completion.name == 'completion_recovery.json':
                        a.check(done['checkpoint_sha256'] == cr['checkpoint_sha256'] and done['zero_replayed_updates'] is True
                                and done['step'] == cr['step'], completion, 'NFT completion recovery binding differs')
                        a.verify({done['original_failure_path']: done['original_failure_sha256'], worker / 'training_metrics.json': done['metrics_sha256']}, completion)
                    row.update(logical_committed_updates=16, disposition='COMMITTED')
            else:
                if not (ri == 0 and worker.name == 'train_SELF_FM' and (D.OUT / 'engineering_recovery_round00/authorization.json').exists()):
                    a.pending(worker, 'Training transaction has not completed')
            workers.append(row)
    auth_path = D.OUT / 'engineering_recovery_round00/authorization.json'
    recovery = dict(status='ABSENT', discarded_complete_updates=0, discarded_update_attempts=0, discarded_forward_attempts=0, discarded_forward_returns=0, replayed_complete_updates=0)
    if auth_path.exists():
        auth = a.read(auth_path)
        old = histories.get((0, 'SELF_FM', 'train_SELF_FM'))
        new = histories.get((0, 'SELF_FM', 'train_SELF_FM_recovery01'))
        if old is None:
            a.fail(auth_path, 'Original interrupted ledger missing')
        else:
            old_row, old_updates = old
            a.verify({D.OUT / 'training/round_00/train_SELF_FM/events.jsonl': auth['old_events_sha256'],
                      D.OUT / 'training/round_00/training_stream.json': auth['stream_sha256']}, auth_path)
            pairs = [('completed_updates', 'discarded_complete_updates'), ('update_attempts', 'discarded_update_attempts'),
                     ('forward_attempts', 'discarded_training_forward_attempts'), ('forward_returns', 'discarded_training_forward_returns')]
            a.check(all(old_row[k] == auth[v] for k, v in pairs), auth_path, 'Physical discarded costs differ from authorization')
            a.check((old_row['completed_updates'], old_row['update_attempts'], old_row['forward_returns']) == (14, 15, 228), auth_path, 'Recovery exceeds the one recorded allowance')
            old_row['disposition'] = 'DISCARDED_ENGINEERING_INTERRUPTION'
            replayed = min(len(old_updates), len(new[1])) if new else 0
            for i in range(replayed):
                original, replay = old_updates[i], new[1][i]
                a.check(original['rows'] == replay['rows'] and original['gradient_norm_before_clip'] == replay['gradient_norm_before_clip'], auth_path, 'Numeric replay log does not match original')
            if new and new[0]['logical_committed_updates'] == 16:
                a.check(new[0]['replay_checks'] == new[0]['replay_checks_equal'] == 14, auth_path, 'Recovery numeric checks not complete or failed')
            recovery = dict(status='COMPLETE' if new and new[0]['logical_committed_updates'] == 16 else 'PARTIAL',
                discarded_complete_updates=old_row['completed_updates'], discarded_update_attempts=old_row['update_attempts'],
                discarded_forward_attempts=old_row['forward_attempts'], discarded_forward_returns=old_row['forward_returns'],
                replayed_complete_updates=replayed, original_full_parameter_state_unavailable=True,
                evidence='Exact recorded microbatch losses and preclip gradients; not bytewise proof of unsaved original optimizer/model state')
    for arm in ARMS:
        if len(committed[arm]) < 16:
            a.pending(arm, 'Not all sixteen training checkpoints committed')
    return dict(workers=workers, checkpoints=checkpoints, completed_rounds=committed, recovery=recovery)


def final_evaluation(a, manifest):
    queries = manifest['evaluation_requests']; ev = D.OUT / 'evaluation'
    stages = []
    for label in LABELS:
        selected = [q for q in queries if label != 'BASE' or q['evaluation_noise_partition'] == 'FRESH']
        row = a.guard('final_' + label, lambda: sampling_stage(a, manifest, ev / label, selected, label), {})
        stages.append(row)
    cached_path = D.FULL / 'evaluation/indices/FULL_S6000/post_index.json'
    cached = a.read(cached_path); posts = {r['request_id']: r for r in cached['rows']}
    cache_count = 0
    for q in queries:
        if q['evaluation_noise_partition'] != 'OLD': continue
        r = manifest['base_raw_cache'][q['request_id']]; p = posts[q['request_id']]
        a.check(r['checkpoint_sha256'] == manifest['source_binding']['sha256'], cached_path, 'Cached BASE source differs')
        a.check(all(r[k] == q[k] == p[k] for k in ('input_sha256', 'C_sha256', 'Gaussian_sha256', 'N', 'K')), cached_path, 'Cached BASE condition/noise differs')
        a.verify({r['raw_path']: r['raw_sha256'], p['post_path']: p['post_sha256']}, cached_path)
        cache_count += 1
    a.check(cache_count == 24, cached_path, 'Expected precisely 24 cached BASE RAW/POST')
    scores = []
    for label in LABELS:
        for q in queries:
            for stage in ('RAW', 'POST'):
                path = ev / label / 'scores' / q['request_id'] / stage / 'score.json'
                if path.exists():
                    row = a.guard(path, lambda: score_record(a, path, q, label, stage))
                    if row is not None: scores.append(row)
    if len(scores) != 216: a.pending('final_scores', 'Not all 216 final RAW/POST scores exist')
    review_path = ev / 'blind_gallery/reviews_actual.json'; review = None
    if review_path.exists():
        review = a.guard(review_path, lambda: actual_reviews(a, review_path, ev / 'blind_gallery/manifest.json', queries, final=True))
        mapping_path = ev / 'private/label_mapping.json'
        mapping = a.read(mapping_path)
        a.check(mapping['manifest_sha256'] == a.hash(D.OUT / 'inputs/manifest.json') and mapping['status'] == 'FROZEN', mapping_path, 'Blind mapping not tied to frozen manifest')
        maps = mapping['rows']
        a.check(len(maps) == 36 and {r['request_id'] for r in maps} == {q['request_id'] for q in queries}
                and all(set(r['aliases']) == {'A', 'B', 'C'} and set(r['aliases'].values()) == set(LABELS) for r in maps), mapping_path, 'Blind alias assignment differs')
    else:
        a.pending('final_reviews', 'Actual anonymous final reviews unavailable')
    cpu_path = ev / 'cpu/post_score_render_complete.json'
    if cpu_path.exists():
        cpu = a.read(cpu_path)
        a.check((cpu['status'], cpu['requests'], cpu['required_new_POST'], cpu['model_calls']) == ('COMPLETE', 108, 84, 0), cpu_path, 'Final CPU accounting differs')
        a.verify(cpu['files'], cpu_path)
    else: a.pending('final_CPU', 'Final CPU completion unavailable')
    all_d0 = ev / 'final_proposal_index.json'
    if all_d0.exists():
        value = a.read(all_d0)
        a.check((value['status'], value['jobs'], value['new_D0_forwards'], value['resident_models_loaded']) == ('COMPLETE', 84, 84, 1), all_d0, 'Final resident D0 batch differs')
        phase = ev / 'denoise_all_worker/completion.json'
        a.check(phase.exists() and a.read(phase)['status'] == 'COMPLETE', all_d0, 'Final resident D0 Phase incomplete')
    else: a.pending('final_D0', 'Final resident batch receipt unavailable')
    return dict(stages=stages, BASE_cached_RAW_POST=cache_count, scores=scores, score_count=len(scores),
                review_rows=review['rows'] if review else 0, actual_review=review)


def run():
    a = Audit(); a.hash(__file__)
    manifest_path = D.OUT / 'inputs/manifest.json'; manifest = a.read(manifest_path)
    registration = a.read(D.OUT / 'registration.json')
    a.check(manifest['status'] == 'FROZEN' and registration['manifest_sha256'] == a.hash(manifest_path), manifest_path, 'Manifest/registration identity differs')
    registered_stages = {str(D.OUT / f'training/round_{ri:02d}') for ri in range(16)} | {str(D.OUT / 'evaluation' / label) for label in LABELS}
    for pattern in ('**/raw/*/receipt.json', '**/proposal/*/started.json'):
        for path in D.OUT.glob(pattern):
            if str(path.parents[2]) not in registered_stages:
                a.fail(path, 'Output or D0 attempt outside registered stages')
    a.verify(manifest['protected_hashes'], 'manifest.protected_hashes')
    protected_errors = len(a.violations)
    lock = a.read(D.OUT / 'execution_source_lock.json'); a.verify(lock['hashes'], 'execution_source_lock')
    query_sidecar = a.read(D.OUT / 'inputs/training_queries.json')
    a.verify(query_sidecar['original_source_hashes'], 'original_query_sources')
    rounds, reviews = [], []
    for ri in range(16):
        queries = [q for q in manifest['training_requests'] if q['round_index'] == ri]
        a.check(len(queries) == 32 and len({q['request_id'] for q in queries}) == 32, ri, 'Expected 32 unique registered training requests')
        row = a.guard('round_' + str(ri), lambda: sampling_stage(a, manifest, D.OUT / f'training/round_{ri:02d}', queries, 'NFT', ri), {})
        rounds.append(row)
        row = a.guard('reviews_' + str(ri), lambda: review_round(a, manifest, ri, queries), {})
        reviews.append(row)
    final = a.guard('final_evaluation', lambda: final_evaluation(a, manifest), dict(stages=[], score_count=0, review_rows=0, BASE_cached_RAW_POST=0))
    learned = a.guard('training', lambda: training(a, manifest), dict(workers=[], checkpoints=[], completed_rounds={arm: [] for arm in ARMS}, recovery={}))
    stages = rounds + final['stages']; workers = learned['workers']
    actual = {key: sum(row.get(key, 0) for row in stages) for key in ('RAW_receipts', 'D0_attempts', 'D0_returns', 'rollout_attempts', 'rollout_returns',
        'sampling_forward_attempts', 'sampling_forward_returns', 'validation_forward_attempts', 'validation_forward_returns')}
    actual.update(training_forward_attempts=sum(r['forward_attempts'] for r in workers), training_forward_returns=sum(r['forward_returns'] for r in workers),
        physical_update_attempts=sum(r['update_attempts'] for r in workers), physical_completed_updates=sum(r['completed_updates'] for r in workers),
        logical_updates_per_arm={arm: sum(r['logical_committed_updates'] for r in workers if r['arm'] == arm) for arm in ARMS},
        final_new_RAW=sum(r.get('RAW_receipts', 0) for r in final['stages']), BASE_cached_RAW_POST=final['BASE_cached_RAW_POST'],
        final_scores=final['score_count'], training_review_rows=sum(r.get('review_rows', 0) for r in reviews), final_review_rows=final['review_rows'])
    b = registration['budget']; recovery = learned['recovery']
    expected = dict(RAW_receipts=b['total_new_rollouts'], D0_attempts=b['denoiser_forwards'], D0_returns=b['denoiser_forwards'],
        rollout_attempts=b['total_new_rollouts'], rollout_returns=b['total_new_rollouts'], sampling_forward_attempts=b['sampling_forwards'], sampling_forward_returns=b['sampling_forwards'],
        validation_forward_attempts=b['zero_adapter_validation_forwards'], validation_forward_returns=b['zero_adapter_validation_forwards'],
        training_forward_attempts=b['updates_per_arm'] * b['effective_batch'] * (3 + 2) + 228,
        training_forward_returns=b['updates_per_arm'] * b['effective_batch'] * (3 + 2) + 228,
        physical_update_attempts=2 * b['updates_per_arm'] + 15, physical_completed_updates=2 * b['updates_per_arm'] + 14,
        logical_updates_per_arm={arm: b['updates_per_arm'] for arm in ARMS}, final_new_RAW=b['final_new_rollouts'], BASE_cached_RAW_POST=b['BASE_old_cache'],
        final_scores=3 * b['evaluation_per_model'] * 2, training_review_rows=b['training_rollouts'], final_review_rows=3 * b['evaluation_per_model'])
    checks = {}
    for key, want in expected.items():
        got = actual[key]
        checks[key] = 'PASS' if got == want else 'PARTIAL'
        excess = any(got[k] > want[k] for k in want) if isinstance(want, dict) else got > want
        if excess:
            checks[key] = 'FAIL'; a.fail('budget.' + key, 'Measured count exceeds registered/explicit recovery allowance')
    a.finish_reads()
    if any(v == 'PARTIAL' for v in checks.values()): a.pending('budgets', 'Registered execution not yet complete')
    status = 'FAIL' if a.violations else 'PARTIAL' if a.partial else 'COMPLETE'
    budget_status = 'FAIL' if any(v == 'FAIL' for v in checks.values()) else 'PASS' if all(v == 'PASS' for v in checks.values()) else 'PARTIAL'
    cuda = bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
    if cuda: a.fail('runtime', 'CUDA initialized in CPU audit'); status = 'FAIL'
    digest = hashlib.sha256(json.dumps(dict(version=VERSION, sources=a.sources), sort_keys=True).encode()).hexdigest()[:20]
    out = D.OUT / 'audit/snapshots' / digest
    records = dict(rounds=rounds, reviews=reviews, training=learned, final=final)
    for name, value in records.items(): D.write(out / (name + '.json'), value)
    D.write(out / 'input_sources.json', a.sources)
    summary = dict(version=VERSION, status=status, code_sha256=a.code_sha, input_digest=digest, output_dir=str(out),
        budget_checks=dict(status=budget_status, expected=expected, actual=actual, checks=checks),
        violations=a.violations, partial_reasons=a.partial, changing_sources=a.changes,
        protected_inputs=dict(status='PASS' if protected_errors == 0 else 'FAIL', manifest_protected_files=len(manifest['protected_hashes']),
            source_scoring_definition_files=len(a.definitions or {}), hash_only_base_read=True),
        completed_rounds=learned['completed_rounds'], engineering_recovery=recovery,
        review_evidence_scope='Checks preserved actual-review records and viewed-image identities; does not independently reconstruct image-tool observation or infer visual labels',
        audit_scope='Read existing evidence; no generation, training, scoring, rendering, worker dispatch or network',
        model_forwards_performed_by_audit=0, CUDA_initialized=cuda,
        files={str(out / (name + '.json')): D.sha(out / (name + '.json')) for name in (*records, 'input_sources')})
    D.write(out / 'summary.json', summary)
    for row in learned['checkpoints']:
        D.write(D.OUT / 'audit/checkpoint_cache' / a.code_sha / (row['checkpoint_sha256'] + '.json'), row)
    latest = D.checked(D.OUT / 'audit/latest.json'); latest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=VERSION, status=status, output_dir=str(out), snapshot_path=str(out), summary_path=str(out / 'summary.json'), summary_sha256=D.sha(out / 'summary.json'), input_digest=digest, code_sha256=a.code_sha)
    temp = latest.with_suffix('.json.tmp'); temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n'); temp.replace(latest)
    print(json.dumps(dict(status=status, output=str(out), budget_status=budget_status, actual=actual, violations=len(a.violations), partial_reasons=len(a.partial)), sort_keys=True))
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
