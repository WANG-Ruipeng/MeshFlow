"""Independent read-only CPU audit of actual bounded-repeat evidence.

Live snapshots allow in-flight work and skip large checkpoint hashing. Final
mode checks registered source bytes and only this run's actual dependencies.
No model/torch import, scoring, matching, Gaussian draw, or mesh modification.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.assets import read_binary_ply

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
ARMS = ('T', 'L', 'G')
LABELS = ('START', 'HIST_TEACHER500', 'HIST_CAPPED500', 'T500', 'T1000', 'L500', 'L1000', 'G500', 'G1000')
LIMITS = dict(training_forward=27000, training_backward=27000, optimizer_update=3000,
              sampling=290, sampling_forward=14500, precheck_forward=24,
              precheck_backward=12, precheck_optimizer_update=1)
FORWARDS = ('training_forward', 'sampling_forward', 'precheck_forward')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k].copy() for k in value.files}


class Audit:
    def __init__(self, out, live):
        self.out, self.live = Path(out).resolve(), bool(live)
        self.cutoff = time.time()
        self.issues, self.pending, self.notes = [], [], []
        self.hashes = {}
        self.source_bytes = 0

    def check(self, condition, message, **context):
        if not condition:
            self.issues.append(dict(message=message, **context))
        return bool(condition)

    def need(self, condition, message, **context):
        if not condition:
            (self.pending if self.live else self.issues).append(dict(message=message, **context))
        return bool(condition)

    def hash(self, path, expected=None):
        path = Path(path); key = str(path)
        if key not in self.hashes:
            self.hashes[key] = sha(path); self.source_bytes += path.stat().st_size
        if expected is not None:
            self.check(self.hashes[key] == expected, 'File SHA256 mismatch', path=key,
                       expected=expected, actual=self.hashes[key])
        return self.hashes[key]

    def lines(self, path):
        path = Path(path)
        if not path.exists():
            return []
        data = path.read_bytes(); output = []
        for index, line in enumerate(data.splitlines(keepends=True)):
            if not line.strip():
                continue
            if self.live and not line.endswith(b'\n'):
                self.notes.append(dict(path=str(path), line=index + 1, note='Ignored incomplete live last line'))
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                self.check(False, 'Invalid journal JSON', path=str(path), line=index+1, error=repr(exc)); continue
            if float(row.get('time_unix', 0)) <= self.cutoff:
                output.append(row)
        return output

    def guarded(self, name, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.check(False, 'Audit section could not complete', section=name, error=repr(exc))
            return dict(status='BLOCKED', exception=repr(exc))


def audit_calls(a):
    rows = []
    for path in sorted((a.out / 'ledger').glob('calls_*.jsonl')):
        rows.extend(a.lines(path))
    rows.sort(key=lambda r: (r.get('time_unix', 0), r.get('call_id', ''), r.get('event', '')))
    begins, ends = {}, {}
    attempts, returned, failed, seconds = Counter(), Counter(), Counter(), Counter()
    groups = defaultdict(lambda: defaultdict(list))
    sampling, active_samples = {}, {}
    for row in rows:
        event, cid, kind = row.get('event'), row.get('call_id'), row.get('kind')
        if not a.check(event in ('CALL_BEGIN', 'CALL_RETURN', 'CALL_FAILED') and
                       isinstance(cid, str) and kind in LIMITS, 'Unknown CALL journal format', row=row):
            continue
        context = row.get('context', {})
        a.check(context.get('split') not in ('TRAIN_FEEDBACK', 'CONFIRM'), 'Forbidden cohort model call', call_id=cid)
        if event == 'CALL_BEGIN':
            if not a.check(cid not in begins, 'Duplicate CALL_BEGIN', call_id=cid):
                continue
            begins[cid] = row; attempts[kind] += 1
            a.check(row.get('batch_size') == 1, 'Unexpected model-call batch size', call_id=cid)
            if kind in ('training_forward', 'training_backward', 'optimizer_update'):
                key = (context.get('arm'), context.get('pilot_step'))
                groups[key][kind].append(row)
            if kind == 'sampling':
                key = (context.get('label'), context.get('request_id'))
                a.check(key not in active_samples, 'Overlapping sampling attempts for same output', key=list(key))
                active_samples[key] = cid
                sampling[cid] = dict(label=key[0], request_id=key[1], split=context.get('split'),
                                     start=row['time_unix'], forwards=[], status='PENDING')
            elif kind == 'sampling_forward':
                key = (context.get('label'), context.get('request_id'))
                parent = active_samples.get(key)
                if a.check(parent is not None, 'Sampling forward has no outer attempt', call_id=cid):
                    sampling[parent]['forwards'].append(cid)
        else:
            if not a.check(cid in begins and cid not in ends, 'Unmatched or duplicate CALL terminal', call_id=cid):
                continue
            begin = begins[cid]; ends[cid] = row
            a.check(begin['kind'] == kind and begin['context'] == context, 'CALL identity changed at return', call_id=cid)
            a.check(row['time_unix'] >= begin['time_unix'], 'CALL terminal precedes begin', call_id=cid)
            a.check(math.isfinite(row.get('wall_seconds', -1)) and row.get('wall_seconds', -1) >= 0,
                    'Invalid actual call duration', call_id=cid)
            (returned if event == 'CALL_RETURN' else failed)[kind] += 1
            seconds[kind] += row.get('wall_seconds', 0)
            if kind == 'sampling':
                sampling[cid]['status'] = event
                sampling[cid]['end'] = row['time_unix']
                active_samples.pop((context['label'], context['request_id']), None)
    pending = sorted(set(begins) - set(ends))
    a.need(not pending, 'In-flight or unresolved model calls', call_ids=pending)
    for kind, limit in LIMITS.items():
        a.check(attempts[kind] <= limit, 'Actual call hard limit exceeded', kind=kind, attempts=attempts[kind], limit=limit)
    forward_count = sum(attempts[k] for k in FORWARDS)
    a.check(forward_count <= 41524, 'Top-level forward hard limit exceeded', attempts=forward_count)
    for cid, sample in sampling.items():
        child = sample['forwards']
        steps = [begins[x]['context'].get('euler_step') for x in child]
        a.check(len(child) <= 50 and len(set(steps)) == len(steps), 'Invalid repeated Euler steps', call_id=cid, steps=steps)
        if sample['status'] == 'CALL_RETURN':
            a.check(steps == list(range(50)) and all(ends.get(x, {}).get('event') == 'CALL_RETURN' for x in child),
                    'Completed sampling lacks 50 successful ordered forwards', call_id=cid)
    ap = a.out / 'accounting.json'
    account = read(ap) if ap.exists() else {}
    exact = all(Counter(account.get(k, {})) == actual for k, actual in
                (('attempts', attempts), ('returns', returned), ('failed', failed)))
    if a.live:
        if not exact:
            a.notes.append(dict(note='Accounting snapshot may lag active live journal', accounting_exact=False))
    else:
        a.check(exact and sorted(account.get('pending', [])) == pending, 'Final accounting does not equal CALL journal')
        a.check(account.get('top_level_forward_attempts') == forward_count, 'Final forward accounting mismatch')
    pre = a.out / 'preflight/real_complete.json'
    if a.need(pre.exists(), 'Preflight completion missing'):
        p = read(pre)
        for kind, expected in (('precheck_forward', 12), ('precheck_backward', 12), ('precheck_optimizer_update', 1)):
            a.check(attempts[kind] == returned[kind] == expected and failed[kind] == 0,
                    'Actual preflight call count differs', kind=kind, actual=attempts[kind], expected=expected)
        a.check(p['status'] == 'PASS' and p['original_model_restored'] and p['original_AdamW_restored'] and
                p['retained_optimizer_updates'] == 0 and p['isolated_discarded_optimizer_updates'] == 1,
                'Preflight retained state contract differs')
    summary = dict(attempts=dict(attempts), returns=dict(returned), failed=dict(failed), pending=pending,
        top_level_forward_attempts=forward_count, top_level_forward_limit=41524,
        limits=LIMITS, call_begin_count=len(begins), terminal_count=len(ends),
        phase_call_seconds=dict(seconds), accounting_exact_at_read=exact,
        duration_scope='CALL wall durations include nested sampling forwards; do not add sampling and child times.')
    return summary, begins, ends, groups, sampling


def audit_occupancy(a, begins, ends):
    rows = a.lines(a.out / 'ledger/gpu_occupancy.jsonl')
    active = None; intervals = []; used = 0.
    for row in rows:
        if row.get('event') == 'GPU_ACQUIRE':
            a.check(active is None, 'Multiple simultaneous GPU lock owners', row=row)
            active = row
        elif row.get('event') == 'GPU_RELEASE':
            if not a.check(active is not None, 'GPU release without acquisition', row=row):
                continue
            a.check((row['stage'], row['pid']) == (active['stage'], active['pid']), 'GPU release owner mismatch', row=row)
            elapsed = row['time_unix'] - active['time_unix']
            duration = row.get('seconds', -1)
            a.check(duration >= 0 and abs(duration - elapsed) < .5, 'GPU duration differs from journal timestamps', stage=row['stage'])
            intervals.append(dict(stage=row['stage'], pid=row['pid'], start=active['time_unix'], end=row['time_unix'],
                                  seconds=duration, status=row.get('status')))
            used += duration; active = None
        else:
            a.check(False, 'Unknown occupancy journal format', row=row)
    if active is not None:
        intervals.append(dict(stage=active['stage'], pid=active['pid'], start=active['time_unix'], end=a.cutoff,
                              seconds=max(0., a.cutoff-active['time_unix']), status='ACTIVE'))
    total = sum(r['seconds'] for r in intervals)
    a.check(total <= 14400, 'Four-hour GPU occupancy ceiling exceeded', seconds=total)
    a.need(active is None, 'GPU occupation remains open', active=active)
    for cid, begin in begins.items():
        owners = [r for r in intervals if r['pid'] == begin['pid'] and r['start'] <= begin['time_unix'] <= r['end']]
        if a.check(len(owners) == 1, 'Model call not within exactly one GPU lock interval', call_id=cid):
            if cid in ends:
                a.check(ends[cid]['time_unix'] <= owners[0]['end'] + .01, 'Model call ends outside its lock', call_id=cid)
    failed = []
    for r in intervals:
        if r['status'] == 'FAILED':
            scoped = [x for x in begins.values() if x['pid'] == r['pid'] and r['start'] <= x['time_unix'] <= r['end']]
            failed.append(dict(**r, actual_call_begins=len(scoped)))
    return dict(closed_seconds=used, active_seconds=total-used, total_seconds_through_snapshot=total,
        limit_seconds=14400, maximum_simultaneous_workers=1 if intervals else 0,
        closed_intervals=sum(x['status'] != 'ACTIVE' for x in intervals),
        failed_intervals=failed, intervals=intervals,
        scope='Actual lock ownership including loading, saving, and in-worker CPU wait; not pure GPU kernel time.')


def audit_training(a, ends, groups):
    plan = read(a.out / 'streams/repeat/train_plan.json')['records']
    auxplan = read(a.out / 'registration/auxiliary_queue_plan.json')['records']
    protocol = read(a.out / 'protocol.json')
    all_metrics, summaries = {}, {}
    for arm in ARMS:
        events = a.lines(a.out / 'training' / (arm + '_events.jsonl'))
        done = [r['pilot_step'] for r in events if r.get('event') == 'UPDATE_DONE']
        a.check(done == list(range(1, len(done)+1)), 'Training completed updates not contiguous', arm=arm)
        a.need(len(done) == 1000, 'Formal arm is not at1000 completed updates', arm=arm, completed=len(done))
        metrics = a.lines(a.out / 'training' / (arm + '_metrics.jsonl'))
        mmap = {r['pilot_step']: r for r in metrics}
        a.check(len(mmap) == len(metrics), 'Duplicate training metric step', arm=arm)
        exposures = Counter(); snapshots = []; g_excess = []; norms = []; cap_count = 0
        for step in done:
            if not a.check(step in mmap, 'Completed update has no metric row', arm=arm, step=step):
                continue
            r = mmap[step]; callgroup = groups.get((arm, step), {})
            for kind, expected in (('training_forward', 9), ('training_backward', 9), ('optimizer_update', 1)):
                calls = callgroup.get(kind, [])
                a.check(len(calls) == expected and all(ends.get(c['call_id'], {}).get('event') == 'CALL_RETURN' for c in calls),
                        'Completed update lacks exact successful9F9B1step', arm=arm, step=step, kind=kind, count=len(calls))
                if kind != 'optimizer_update':
                    a.check(Counter(str(c['context']['microbatch']) for c in calls) == Counter(map(str, list(range(8)) + ['aux'])),
                            'Main/auxiliary call layout differs', arm=arm, step=step, kind=kind)
            sample = r['sample_diagnostics']; expected_plan = plan[8*(step-1):8*step]
            a.check(len(sample) == 8 and len({x['uid'] for x in sample}) == 8 and sum(x['alpha'] == 1. for x in sample) == 4,
                    'Training parent/alpha batch contract differs', arm=arm, step=step)
            for actual, expected in zip(sample, expected_plan):
                a.check(all(actual[k] == expected[k] for k in ('sample_index', 'uid', 'task_id', 'N', 'K', 'alpha')),
                        'Actual exposure differs from frozen plan', arm=arm, step=step, sample_index=actual['sample_index'])
                exposures[actual['uid']] += 1
            den = 9 * sum(x['N'] - x['K'] for x in sample)
            a.check(r['free_coordinate_denominator'] == r['effective_free_coordinates'] == den,
                    'Global free-scalar FM denominator mismatch', arm=arm, step=step)
            a.check(math.isclose(sum(x['FM_contribution'] for x in sample), r['FM'], rel_tol=0., abs_tol=1e-12),
                    'FM microbatch sum mismatch', arm=arm, step=step)
            au, expected = r['aux_audit'], auxplan[step-1]
            a.check(all(au[k] == expected[k] for k in ('step', 'queue', 'uid', 'request_id', 'candidate', 'state_index', 'payload_sha256')),
                    'Auxiliary queue changed', arm=arm, step=step)
            a.check(au['training_seed'] == 20261008 and math.isclose(au['t'], expected['t'], abs_tol=3e-8),
                    'Auxiliary frozen seed/time changed', arm=arm, step=step)
            target = expected['teacher_target_sha256'] if arm == 'T' else expected['trajectory_target_sha256']
            a.check(au['target_sha256'] == target and au['denominator'] == r['aux_free_denominator'],
                    'Auxiliary target/denominator identity differs', arm=arm, step=step)
            lf, la, nf, na = [r[k] for k in ('FM', 'aux_unweighted', 'gF_norm', 'gA_norm')]
            ll = min(.1, lf/max(la, 1e-30)); lg = nf/(na+1e-12)
            weight = min(ll, lg) if arm == 'G' else ll
            if arm == 'G' and nf == 0: weight = 0.
            a.check(all(math.isclose(r[k], v, rel_tol=1e-12, abs_tol=1e-15) for k,v in
                        (('lambda_loss', ll), ('lambda_grad', lg), ('actual_lambda', weight))),
                    'Detached coefficient formula mismatch', arm=arm, step=step)
            a.check(all(math.isfinite(r[k]) for k in ('FM', 'aux_unweighted', 'gF_norm', 'gA_norm',
                'weighted_aux_norm', 'grad_norm_before_clip', 'actual_lambda')) and r['finite'],
                'Nonfinite actual training diagnostics', arm=arm, step=step)
            a.check(r['parameter_count'] == 194 and r['parameter_elements'] == 130401155 and
                r['buffer_dtype'] == 'torch.float32' and r['primary_buffer_released'],
                'Full original parameter carrier contract differs', arm=arm, step=step)
            a.check(r['primary_none_count'] == len(r['primary_none_names']) and
                    r['auxiliary_none_count'] == len(r['auxiliary_none_names']) and
                    r['both_none_count'] == len(r['both_none_names']), 'None-gradient accounting differs', arm=arm, step=step)
            cap_count += bool(r['gradient_cap_active'])
            if arm == 'G':
                a.check(weight*na <= nf + 1e-12, 'Theoretical full-gradient budget violated', step=step)
                # Store actual FP32 residual rather than inventing a new gate.
                g_excess.append(r['actual_FP32_budget_excess'])
            if step in protocol['training']['true_parameter_update_steps']:
                u = r.get('actual_parameter_update', {})
                a.check(u.get('status') == 'PASS' and u.get('no_gradient_proxy') is True and
                    u.get('elements') == 130401155 and len(u.get('per_parameter', [])) == 194 and
                    u.get('actual_update_norm', 0) > 0, 'Missing actual parameter-update measurement', arm=arm, step=step)
                snapshots.append(step)
                if u.get('actual_update_norm') is not None: norms.append(u['actual_update_norm'])
        if len(done) == 1000:
            a.check(len(exposures) == 32 and set(exposures.values()) == {250}, 'Final parent exposure differs from32x250', arm=arm)
        all_metrics[arm] = {s:mmap[s] for s in done if s in mmap}
        summaries[arm] = dict(completed_updates=len(done), parent_count=len(exposures), exposures=dict(exposures),
            gradient_cap_active=cap_count, actual_parameter_update_steps=snapshots,
            actual_update_norm_min=min(norms) if norms else None, actual_update_norm_max=max(norms) if norms else None,
            maximum_G_actual_FP32_budget_excess=max(g_excess) if g_excess else None)
    paired = 0
    common = sorted(set.intersection(*(set(all_metrics[k]) for k in ARMS)))
    for step in common:
        refs = [all_metrics[k][step] for k in ARMS]
        for key in ('shared_batch_sha256', 'coupled_batch_sha256', 'effective_free_coordinates'):
            a.check(len({r[key] for r in refs}) == 1, 'Cross-arm main input mismatch', step=step, field=key)
        for slot in range(8):
            a.check(len({r['sample_diagnostics'][slot]['shared_input_sha256'] for r in refs}) == 1,
                    'Cross-arm per-sample actual hash mismatch', step=step, slot=slot)
        a.check(refs[1]['aux_audit']['xt_sha256'] == refs[2]['aux_audit']['xt_sha256'] and
                refs[1]['aux_audit']['u_sha256'] == refs[2]['aux_audit']['u_sha256'],
                'L/G auxiliary actual state or target mismatch', step=step)
        paired += 8
    return dict(arms=summaries, common_completed_steps=len(common), actual_main_inputs_paired=paired,
                formal_updates=sum(v['completed_updates'] for v in summaries.values()))


def audit_sources(a):
    sources = read(a.out / 'registration/source_files.json')
    protocol = read(a.out / 'protocol.json')
    expected = {p:r['sha256'] for p,r in sources.items()}
    expected.update(protocol['scientific_sources']); expected.update(protocol['original_scorers'])
    # Follow only the32 actual training parents plus their pinned loader/C bank.
    # The immutable binding already contains before-run hashes; never inspect
    # unrelated assets or CONFIRM outputs.
    data_root = ROOT / 'native_t1/runs/control_mvp48_v1'
    binding = read(data_root / 'registration/train_data_binding.json')
    training_parents = set(read(a.out / 'manifests/TRAINING.json')['parents'])
    dependencies = ['data/load_sample.py', 'data/manifest.json', 'data/patches.json']
    dependencies += ['data/meshes/' + uid + '.npz' for uid in sorted(training_parents)]
    for relative in dependencies:
        expected[str(data_root / relative)] = binding['files'][relative]
    for name, value in protocol['manifest_sha256'].items():
        expected[str(a.out / 'manifests' / (name + '.json'))] = value
    skipped, verified = [], []
    for p, value in expected.items():
        path = Path(p)
        if not a.check(path.exists(), 'Registered source missing', path=p):
            continue
        if a.live and path.suffix == '.pt':
            a.check(path.stat().st_size == sources[p]['bytes'], 'Registered weight size changed', path=p)
            skipped.append(p); continue
        a.hash(path, value); verified.append(p)
    return dict(source_files=len(expected), files_hashed=len(verified), explicit_training_assets=len(dependencies), large_weights_skipped_live=skipped,
        whole_workspace_scan=False, scope='Registration source_files, protocol sources, registered manifests and32 training-parent assets pinned by original binding',
        before_after_hashes={p:dict(expected=expected[p], actual=a.hashes[p]) for p in verified})


def audit_checkpoints(a):
    registered = read(a.out / 'registration/checkpoints.json')
    rows = []
    for label, entry in registered.items():
        rows.append(dict(label=label, role=entry['role'], path=entry['path'], bytes=entry['bytes'],
            checkpoint_sha256=entry['sha256'], actual_file_sha256=a.hashes.get(entry['path']),
            hash_recomputed=entry['path'] in a.hashes, feedback_step=entry['feedback_step'],
            conditional_total_step=entry['conditional_total_step'], payload_loaded=False))
    original_pool_sha = sha(a.out / 'feedback/pool_v1/manifest.json')
    plan_sha = sha(a.out / 'streams/repeat/train_plan.json')
    protocol = read(a.out / 'protocol.json')
    phash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    # The runtime's JSON identity convention is tested against its explicit
    # serialized source; no model/checkpoint module is imported.
    from experiments.mf_h1_local0.chair_control_feedback_auto_v1.common import json_hash
    phash = json_hash(protocol)
    for arm in ARMS:
        for step in (500, 1000):
            a.need((a.out / 'checkpoints' / f'repeat_{arm}_{step}.json').exists(),
                   'Named checkpoint endpoint missing', arm=arm, step=step)
    for side in sorted((a.out / 'checkpoints').glob('*.json')):
        if a.live and side.stat().st_mtime > a.cutoff:
            continue
        entry = read(side)
        if not a.check('metadata' in entry and 'path' in entry and 'checkpoint_sha256' in entry,
                       'Unknown checkpoint sidecar schema', path=str(side)):
            continue
        path = Path(entry['path']); m = entry['metadata']; arm, step = m['arm'], m['feedback_step']
        a.check(arm in ARMS and 0 <= step <= 1000 and m['conditional_total_step'] == 5000+step and
                m['phase'] == 'repeat', 'Checkpoint step/arm identity differs', path=str(path))
        a.check(m['feedback_pool_sha256'] == original_pool_sha and m['auxiliary_seed'] == 20261008 and
                m['auxiliary_next_step'] == step+1 and m['protocol_sha256'] == phash and
                m['source_checkpoint_sha256'] == protocol['start_checkpoint_sha256'] and
                m['full_recovery_state'] and not m['optimizer_reset'], 'Checkpoint recovery metadata differs', path=str(path))
        if not a.need(path.exists(), 'Checkpoint sidecar has no saved weight', path=str(path)):
            continue
        actual = None if a.live else a.hash(path, entry['checkpoint_sha256'])
        rows.append(dict(label=arm+str(step), role='NAMED_ENDPOINT' if step in (500,1000) else 'ROLLING_RECOVERY',
            path=str(path), bytes=path.stat().st_size, checkpoint_sha256=entry['checkpoint_sha256'],
            actual_file_sha256=actual, hash_recomputed=not a.live, feedback_step=step,
            conditional_total_step=m['conditional_total_step'], payload_loaded=False,
            model_state_sha256=m['model_state_sha256'], optimizer_state_sha256=m['optimizer_state_sha256'],
            stream_state_sha256=m['stream_state_sha256'], rng_sha256=m['rng_sha256'],
            sidecar_path=str(side), sidecar_sha256=a.hash(side)))
    loaded = []
    for p in sorted((a.out / 'audit').glob('*_load.json')):
        d = read(p); m, h = d['metadata'], d['hashes']
        loaded.append(dict(path=str(p), sha256=a.hash(p), checkpoint_sha256=h['checkpoint_sha256'],
            optimizer_restored=d['optimizer_restored'], conditional_total_step=m['conditional_total_step'],
            feedback_step=m.get('feedback_step'), arm=m['arm']))
    for row in rows:
        if row['role'] == 'NAMED_ENDPOINT':
            a.need(any(x['checkpoint_sha256'] == row['checkpoint_sha256'] for x in loaded),
                   'Named endpoint has no actual strict load receipt yet', label=row['label'])
    return dict(checkpoints=rows, load_receipts=loaded, train_plan_sha256=plan_sha,
        checkpoint_tensors_loaded_by_audit=False,
        tensor_validation_scope='Runtime checkpoint.save and subsequent real sampling load validate tensor/AdamW hashes and actual step; this independent audit binds their metadata and full-file SHA without deserializing weights.')


def audit_outputs(a, sampling, checkpoints):
    requests = {panel:read(a.out / 'manifests' / (panel+'.json'))['rows'] for panel in ('OLD','FRESH')}
    expected_checkpoints = {r['label']:r['checkpoint_sha256'] for r in checkpoints.get('checkpoints', [])}
    banks, conditions, completed = {}, {}, []
    missing = []; cache_count = generated_count = 0
    successful = Counter((r['label'], r['request_id']) for r in sampling.values() if r['status'] == 'CALL_RETURN')
    for panel, panel_rows in requests.items():
        for req in panel_rows:
            rid = req['request_id']; n, k = req['N'], req['K']
            if req['bank_npz'] not in banks:
                a.hash(req['bank_npz'], req['bank_sha256']); banks[req['bank_npz']] = npz(req['bank_npz'])
            bank = banks[req['bank_npz']]
            if req['condition_npz'] not in conditions:
                a.hash(req['condition_npz'], req['condition_sha256']); conditions[req['condition_npz']] = npz(req['condition_npz'])
            condition = conditions[req['condition_npz']]
            for label in LABELS:
                cache = req.get('historical_cache', {}).get(label) if panel == 'OLD' else None
                if cache:
                    rp, rawp = Path(cache['result_path']), Path(cache['raw_path'])
                    a.hash(rp, cache['result_sha256']); a.hash(rawp, cache['raw_sha256'])
                    a.hash(cache['joined_path'], cache['joined_sha256'])
                    joined = read(cache['joined_path'])
                    a.check(joined['identity']['bank_sha256'] == req['bank_sha256'] and
                            joined['identity']['condition_sha256'] == req['condition_sha256'] and
                            joined['identity']['raw_sha256'] == cache['raw_sha256'],
                            'Historical cache bank/condition/raw identity differs', label=label, request_id=rid)
                else:
                    rp = a.out / 'runtime' / label / rid / 'result.json'; rawp = rp.with_name('raw.npz')
                if not rp.exists() or (a.live and not cache and rp.stat().st_mtime > a.cutoff):
                    missing.append(dict(label=label, panel=panel, request_id=rid)); continue
                result = read(rp)
                if not a.check(result.get('status') == 'PASS', 'Saved output is not PASS', path=str(rp)):
                    continue
                a.hash(rawp, result['raw_sha256']); raw = npz(rawp)
                path = raw['path'].reshape(51,n,9); output = raw['output'].reshape(n,3,3)
                C = bank['C']; known = np.asarray(bank['known_mask'], bool).reshape(n)
                a.check(raw['output'].dtype == raw['path'].dtype == C.dtype == np.float32 and np.isfinite(path).all(),
                        'Saved geometry dtype/finite contract differs', label=label, request_id=rid)
                a.check(same(output.reshape(n,9), path[-1]) and same(raw['C'],C) and same(raw['z'],bank['z']) and
                        same(path[0,k:],bank['z'][0,k:]), 'Raw Gaussian/end-state identity differs', label=label, request_id=rid)
                cpreserved = all(same(state[known], C.reshape(k,9)) for state in path)
                a.check(cpreserved, 'Known C differs in51 states', label=label, request_id=rid)
                expected_sha = expected_checkpoints.get(label)
                a.need(expected_sha is not None, 'Output checkpoint identity not yet available', label=label)
                if expected_sha:
                    a.check(result['checkpoint_sha256'] == expected_sha, 'Output checkpoint hash differs', label=label, request_id=rid)
                a.check(result['metadata']['nfe'] == 50 and result['metadata']['accepted_states'] == 51,
                        'Saved output is not full Euler50', label=label, request_id=rid)
                ex = result['export']; a.hash(ex['npz'],ex['npz_sha256']); a.hash(ex['ply'],ex['ply_sha256'])
                exported = npz(ex['npz']); ply, faces = read_binary_ply(ex['ply'])
                ids = np.flatnonzero(known)
                a.check(same(exported['output'].reshape(n,3,3),output) and same(ply,output),
                        'NPZ/PLY output coordinates changed', label=label, request_id=rid)
                a.check(same(exported['C'],C) and same(ply[known],C) and np.array_equal(exported['known_to_output_face'],ids) and
                        np.array_equal(faces['known'].astype(bool),known) and
                        np.array_equal(faces['known_slot'][known],np.arange(k)) and
                        np.array_equal(faces['source_face'][known],condition['source_face_ids']),
                        'NPZ/PLY known order/winding/multiplicity/source mapping changed', label=label, request_id=rid)
                a.check(len(ply)==n and int(ex['vertex_count'])==3*n and not ex['geometry_modified'],
                        'Export geometry size/modification differs', label=label, request_id=rid)
                if cache:
                    cache_count += 1
                else:
                    generated_count += 1
                    a.check(successful[(label,rid)] >= 1, 'Saved output has no actual completed sampling call', label=label, request_id=rid)
                    ap = rp.with_name('attempt.json'); attempt = read(ap)
                    a.check(attempt['bank_sha256']==req['bank_sha256'] and attempt['checkpoint_sha256']==result['checkpoint_sha256'],
                            'Generation attempt identity differs', label=label, request_id=rid)
                completed.append(dict(label=label,panel=panel,request_id=rid,uid=req['uid'],candidate=req['candidate'],N=n,K=k,
                    status='PASS',origin='VERIFIED_HISTORICAL_CACHE' if cache else 'ACTUAL_NEW_EULER50',
                    raw_path=str(rawp),raw_sha256=result['raw_sha256'],result_path=str(rp),result_sha256=a.hash(rp),
                    checkpoint_sha256=result['checkpoint_sha256'],C_all51_bitwise=cpreserved,
                    C_NPZ_PLY_mapping_and_oriented_multiset=True,npz_sha256=ex['npz_sha256'],ply_sha256=ex['ply_sha256']))
    a.need(not missing, 'Registered evaluation outputs incomplete', missing_count=len(missing))
    a.check(cache_count==48, 'Historical cache count differs', actual=cache_count)
    if not a.live:
        a.check(generated_count==240 and len(completed)==288, 'Final output count differs from240new+48cache', actual=len(completed))
    return dict(completed=len(completed),cached=cache_count,new_generated=generated_count,missing=missing,rows=completed,
        scope='Direct CPU re-read of original FP32 output/path51/C, saved NPZ and binary PLY. No metrics/geometry predicates recomputed.')


def run(out=OUT, live=False):
    a = Audit(out, live); started = time.perf_counter()
    if a.out != OUT.resolve():
        raise ValueError('Only the isolated registered experiment may be audited')
    calls = a.guarded('calls', audit_calls, a)
    if isinstance(calls, tuple):
        call_summary, begins, ends, groups, sampling = calls
    else:
        call_summary = calls; begins,ends,groups,sampling = {},{},{},{}
    occupancy = a.guarded('GPU_occupancy', audit_occupancy, a, begins, ends)
    training = a.guarded('training', audit_training, a, ends, groups)
    sources = a.guarded('protected_sources', audit_sources, a)
    checkpoints = a.guarded('checkpoints', audit_checkpoints, a)
    outputs = a.guarded('outputs', audit_outputs, a, sampling, checkpoints)
    if not live:
        a.check(call_summary.get('returns',{}).get('optimizer_update') == 3000, 'Final formal update total differs')
        a.check(call_summary.get('returns',{}).get('training_forward') == 27000 and
                call_summary.get('returns',{}).get('training_backward') == 27000, 'Final formal F/B totals differ')
        a.need((a.out/'status/gpu_schedule_complete.json').exists(), 'GPU schedule completion marker missing')
    status = 'BLOCKED' if a.issues else 'PENDING' if a.pending else 'PASS'
    prefix = 'runtime_live' if live else 'runtime_final'
    folder = a.out/'audit';folder.mkdir(parents=True,exist_ok=True)
    output_rows = outputs.pop('rows', [])
    if output_rows:
        with (folder/(prefix+'_outputs.csv')).open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(output_rows[0]));writer.writeheader();writer.writerows(output_rows)
    ckname='checkpoint_manifest_live' if live else 'checkpoint_manifest'
    write(folder/(ckname+'.json'),checkpoints)
    if checkpoints.get('checkpoints'):
        ckrows=checkpoints['checkpoints'];fields=sorted({k for r in ckrows for k in r})
        with (folder/(ckname+'.csv')).open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(ckrows)
    result=dict(status=status,mode='LIVE' if live else 'FINAL',snapshot_unix=a.cutoff,
        snapshot_utc=datetime.datetime.fromtimestamp(a.cutoff,datetime.timezone.utc).isoformat(),
        seconds=time.perf_counter()-started,source_path=str(Path(__file__)),source_sha256=sha(__file__),
        issues=a.issues,pending=a.pending,notes=a.notes,calls=call_summary,GPU_occupancy=occupancy,
        training=training,protected_sources=sources,checkpoint_manifest=str(folder/(ckname+'.json')),
        outputs=outputs,unique_files_hashed=len(a.hashes),unique_bytes_hashed=a.source_bytes,
        model_loads=0,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,
        new_Gaussian_draws=0,new_scores=0,CONFIRM_results_read=False)
    write(folder/(prefix+'.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--live',action='store_true')
    args=parser.parse_args();result=run(args.out,args.live)
    print(json.dumps({k:result[k] for k in ('status','mode','seconds','issues','pending','unique_files_hashed','unique_bytes_hashed')},indent=2))
    raise SystemExit(1 if result['status']=='BLOCKED' else 0)
