"""CPU-only endpoint accounting. Refuses to summarize incomplete A/B training.

No torch/model import, neural forward, geometry scoring, or scientific retry.
Large artifacts are streamed once for file hashes; existing JSON sidecars carry
the training serializer's state hashes. This is not a new tensor-state rehash.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .common import ROOT, OUT, ARMS, read, sha, resolve, write_new

DEST = OUT / 'training_closeout'
EXPECTED = dict(train_forward=12000, train_backward=12000, optimizer_update=1500)
CAUTION = ('Different group mixtures change the distribution of training FM. '
           'Do not rank generated quality from mixed FM or group training errors; '
           'use the fixed paired generation evaluation separately.')


def require(test, message):
    if not test:
        raise ValueError(message)


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def table(path, rows):
    require(bool(rows), 'Empty table: ' + str(path))
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summary(rows):
    require(bool(rows), 'Missing group observations')
    means = [r['all_free_velocity_sse'] / r['all_free_scalar_count'] for r in rows]
    endpoints = [r['all_free_endpoint_sse'] / r['all_free_scalar_count'] for r in rows]
    parents = defaultdict(list)
    for r, value in zip(rows, means):
        parents[r['parent_id']].append(value)
    den = sum(r['all_free_scalar_count'] for r in rows)
    return dict(instances=len(rows), parents_observed=len(parents), free_scalar_count=den,
                sample_macro_velocity_mse=statistics.mean(means),
                parent_macro_velocity_mse=statistics.mean(statistics.mean(v) for v in parents.values()),
                coordinate_micro_velocity_mse=sum(r['all_free_velocity_sse'] for r in rows) / den,
                sample_macro_endpoint_mse=statistics.mean(endpoints),
                coordinate_micro_endpoint_mse=sum(r['all_free_endpoint_sse'] for r in rows) / den)


def check_ledger(path, arm, exposures):
    rows = jsonl(path)
    pending, starts, done = {}, [], []
    successful, attempted = Counter(), Counter()
    for row in rows:
        require(row['arm'] == arm, 'Ledger arm changed')
        require(row['event'] in ('CALL_BEGIN', 'CALL_DONE'), 'Failed/unknown ledger event')
        require(row['kind'] in EXPECTED, 'Unexpected head/probe/scientific call')
        call_id = row['call_id']
        if row['event'] == 'CALL_BEGIN':
            require(call_id == len(starts) + 1 and not pending,
                    'Reused/skipped call ID or overlapping pending call')
            pending[call_id] = row
            starts.append(row); attempted[row['kind']] += 1
        else:
            require(call_id in pending, 'Completed call lacks begin')
            begin = pending.pop(call_id)
            require(all(row[k] == begin[k] for k in ('kind', 'context', 'attempt_id')),
                    'Call completion identity changed')
            require(math.isfinite(row['seconds']) and row['seconds'] >= 0, 'Invalid call duration')
            done.append(row); successful[row['kind']] += 1
    require(not pending and dict(attempted) == EXPECTED and dict(successful) == EXPECTED,
            'Incomplete/excess training budget')
    expected_order = []
    for step in range(1, 1501):
        for micro in range(8):
            for kind in ('train_forward', 'train_backward'):
                expected_order.append((step, micro, kind))
        expected_order.append((step, None, 'optimizer_update'))
    for row, (step, micro, kind) in zip(starts, expected_order):
        c = row['context']
        require((c['stage_step'], c.get('microbatch'), row['kind']) == (step, micro, kind),
                'Ledger step/microbatch order changed')
        require(c['global_step'] == 19500 + step and c['native_total_step'] == 7500 + step
                and c['local_step'] == step and c['seam_total_step'] == 0,
                'Ledger cumulative cursor changed')
        if micro is not None:
            require(c['condition_id'] == exposures[(step - 1) * 8 + micro]['condition_id'],
                    'Ledger/exposure condition differs')
    return dict(attempted=dict(attempted), successful=dict(successful), failed=0,
                pending=0, head_calls=0, probe_calls=0, ledger_events=len(rows),
                exact_order_checked=True,
                scientific_call_seconds={k:sum(r['seconds'] for r in done if r['kind'] == k)
                                         for k in EXPECTED})


def artifact(path, expected, arm, schema):
    path = Path(path)
    side = read(path.with_suffix('.json'))
    require(side == expected, 'Status and sidecar differ: ' + str(path))
    require(side['arm'] == arm and side['schema'] == schema and side['stage'] == 'NATIVE'
            and side['stage_step'] == 1500 and side['local_step'] == 1500
            and side['global_step'] == 21000 and side['curriculum_step'] == 3000,
            'Final artifact cursor/schema differs')
    require(path.stat().st_size == side['bytes'] and sha(path) == side['sha256'],
            'Final artifact bytes/hash changed')
    return side


def audit_arm(reg, arm):
    folder = OUT / 'training' / arm
    state = read(folder / 'status.json')
    require(state['status'] == 'COMPLETE' and state['stage_step'] == state['local_step'] == 1500
            and state['global_step'] == 21000 and state['native_total_step'] == 9000
            and state['seam_total_step'] == state['head_calls'] == state['probe_calls'] == 0,
            'Training endpoint incomplete or budget differs')
    require(state['registration_sha256'] == sha(OUT / 'training_registration.json'), 'Registration changed')
    spec = reg['schedules'][arm]
    require(sha(spec['path']) == spec['sha256'], 'Frozen schedule changed')
    schedule = read(spec['path'])
    updates = jsonl(folder / 'training_log.jsonl')
    exposures = jsonl(folder / 'exposure.jsonl')
    require(len(updates) == 1500 and len(exposures) == 12000, 'Log/exposure count differs')
    identity_rows, by_parent = [], defaultdict(list)
    expected_counts = Counter()
    for step, entry in enumerate(schedule['steps'], 1):
        require(entry['step'] == entry['stage_step'] == step, 'Frozen schedule cursor differs')
        u = updates[step - 1]
        require(u['stage_step'] == u['local_step'] == step and u['global_step'] == 19500 + step
                and u['native_total_step'] == 7500 + step and u['seam_total_step'] == 0
                and u['head_calls'] == u['head_step'] == 0 and u['arm'] == arm,
                'Update log cursor/head count differs')
        batch = exposures[(step - 1) * 8:step * 8]
        for micro, (planned, actual) in enumerate(zip(entry['samples'], batch)):
            identity = {k:actual[k] for k in planned}
            require(identity == planned, 'Actual exposure differs from frozen stream')
            require(actual['arm'] == arm and actual['stage_step'] == actual['local_step'] == step
                    and actual['microbatch'] == micro and actual['global_step'] == 19500 + step,
                    'Exposure cursor differs')
            require(actual['head_forward_called'] is False and actual['mechanism']['active_seam_slots'] == 0
                    and actual['mechanism']['policy'] == 'NATIVE_ALL_FREE_OT', 'Extra mechanism detected')
            require(actual['all_free_scalar_count'] == 9 * (actual['N'] - actual['K']) > 0,
                    'Free-coordinate denominator mismatch')
            for k in ('all_free_velocity_sse', 'all_free_endpoint_sse', 'loss_contribution', 't'):
                require(math.isfinite(actual[k]), 'Nonfinite actual exposure statistic')
            expected_counts[planned['condition_id']] += 1
            by_parent[actual['parent_id']].append(actual)
            identity_rows.append(dict(arm=arm, local_step=step, microbatch=micro,
                condition_id=actual['condition_id'], parent_id=actual['parent_id'],
                expected_identity_sha256=object_hash(planned), actual_identity_sha256=object_hash(identity)))
        den = sum(r['all_free_scalar_count'] for r in batch)
        sse = sum(r['all_free_velocity_sse'] for r in batch)
        require(den == u['free_coordinate_denominator'] == u['all_free_scalar_count'], 'Batch denominator changed')
        require(math.isclose(sse, u['all_free_velocity_sse'], rel_tol=1e-10, abs_tol=1e-8), 'Batch SSE differs')
        require(math.isclose(sse / den, u['FM'], rel_tol=5e-6, abs_tol=2e-7)
                and math.isclose(sum(r['loss_contribution'] for r in batch), u['FM'], rel_tol=1e-9, abs_tol=1e-9),
                'FM no longer equals original free-coordinate weighted objective')
    require(dict(expected_counts) == schedule['condition_exposure'], 'Condition exposure total differs')
    ledger = check_ledger(folder / 'calls.jsonl', arm, exposures)
    require(not state['calls']['failed'] and not state['calls']['pending_call_ids'], 'Status records failed/pending work')
    for key in ('attempted', 'successful'):
        require(dict(state['calls'][key]) == EXPECTED, 'Status/ledger counts differ')
    init = read(folder / 'branch_initialization.json')
    for key in ('source_checkpoint_sha256', 'source_generator_sha256', 'source_optimizer_sha256', 'source_rng_sha256'):
        require(init[key] == reg[key], 'Branch source identity mismatch')
    require(init['inherited_optimizer'] and init['inherited_rng'] and not init['fresh_AdamW']
            and not init['head_used'] and init['new_branch_arm'] == arm, 'Initialization policy mismatch')
    cp = artifact(folder / 'checkpoint_step1500.pt', state['checkpoint'], arm,
                  'bench_exposure_allocation_training_v1')
    export = artifact(folder / 'generator_step1500.pt', state['inference_export'], arm,
                      'bench_exposure_allocation_generator_v1')
    require(cp['generator_state_sha256'] == export['generator_state_sha256']
            and cp['head_state_sha256'] == export['head_state_sha256'], 'Checkpoint/export state hash mismatch')
    require(bool(cp.get('optimizer_state_sha256')) and bool(cp.get('rng_sha256')), 'Missing recovery state hashes')
    parent_rows = []
    for parent, rows in sorted(by_parent.items()):
        require(len({r['condition_id'] for r in rows}) == 1, 'Multiple C per parent')
        parent_rows.append(dict(arm=arm, parent_id=parent, condition_id=rows[0]['condition_id'],
            group='OLD' if rows[0]['replay_group'] == 'OLD' else 'FOCUS',
            clean_instances=sum(r['augmentation'] == 'CLEAN' for r in rows),
            xwidth_instances=sum(r['augmentation'] == 'X_WIDTH' for r in rows), **summary(rows)))
    group_counts = {g:sum(r['instances'] for r in parent_rows if r['group'] == g) for g in ('OLD', 'FOCUS')}
    require(group_counts == ({'OLD':6000,'FOCUS':6000} if arm == ARMS[0] else {'OLD':9000,'FOCUS':3000}),
            'Actual group exposure allocation mismatch')
    bins = []
    for first in range(1, 1501, 50):
        last = first + 49; subset = exposures[(first - 1) * 8:last * 8]
        for group in ('ALL_MIXTURE', 'OLD', 'FOCUS'):
            rs = [r for r in subset if group == 'ALL_MIXTURE' or
                  ('OLD' if r['replay_group'] == 'OLD' else 'FOCUS') == group]
            bins.append(dict(arm=arm, first_local_step=first, last_local_step=last, group=group,
                actual_update_FM_mean=statistics.mean(r['FM'] for r in updates[first-1:last]) if group == 'ALL_MIXTURE' else '',
                **summary(rs)))
    attempts = []
    require(not list((folder / 'attempts').glob('*/failure.json')), 'Recorded failed attempt exists')
    for p in sorted((folder / 'attempts').iterdir()):
        final = read(p / 'final_status.json'); resource = read(p / 'resources.json')
        require(final['status'] in ('COMPLETE', 'INTERRUPTED_SAFE'), 'Unclosed or failed attempt')
        attempts.append(dict(attempt=p.name, status=final['status'], final_local_step=final['local_step'],
            active_wall_seconds=final['elapsed_seconds'], resources_summary=resource['summary'],
            resources_path=str(p / 'resources.json'), resources_sha256=sha(p / 'resources.json')))
    timing = dict(arm=arm, attempts=attempts, attempt_count=len(attempts),
        active_attempt_wall_seconds=sum(r['active_wall_seconds'] for r in attempts),
        sum_update_seconds=sum(r['update_seconds'] for r in updates),
        sum_input_seconds=sum(r['input_seconds'] for r in updates),
        first_logged_update_unix=updates[0]['time_unix'], last_logged_update_unix=updates[-1]['time_unix'],
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in updates),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in updates),
        final_resource_summary=state.get('resources'),
        timing_scope='Attempts include load/checkpoint IO; update sums exclude input and most checkpoint IO. Concurrent worker wall times are not additive elapsed time or isolated benchmarks.')
    result = dict(arm=arm, status='PASS', local_step=1500, global_step=21000, native_total_step=9000,
        ledger=ledger, exposure_rows_checked=12000, unique_parents=len(by_parent), group_exposures=group_counts,
        identity_sequence_sha256=object_hash(identity_rows), final_checkpoint=cp, inference_export=export,
        metadata_scope='Global/curriculum from hashed artifact sidecars; Native9000 from linked completed training status and exact update ledger. No second tensor load/state rehash.',
        evidence_hashes={str(folder/n):sha(folder/n) for n in ('status.json','calls.jsonl','training_log.jsonl','exposure.jsonl','branch_initialization.json')})
    return result, identity_rows, parent_rows, bins, timing


def plot_curves(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, group in zip(axes, ('ALL_MIXTURE', 'OLD', 'FOCUS')):
        for arm in ARMS:
            rs = [r for r in rows if r['arm'] == arm and r['group'] == group]
            key = 'actual_update_FM_mean' if group == 'ALL_MIXTURE' else 'parent_macro_velocity_mse'
            ax.plot([r['last_local_step'] for r in rs], [r[key] for r in rs], label=arm)
        ax.set_title(group + (' / actual mixed FM' if group == 'ALL_MIXTURE' else ' / parent-equal MSE'))
        ax.set_xlabel('New optimizer updates'); ax.set_ylabel('Velocity MSE'); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('50-update bins; exposure mixtures differ. Training loss is not generated quality.', fontsize=11)
    fig.savefig(path, dpi=160); plt.close(fig)


def run():
    # No artifact hashing or write is performed while either arm is unfinished.
    for arm in ARMS:
        p = OUT / 'training' / arm / 'status.json'
        require(p.exists() and read(p).get('status') == 'COMPLETE', 'Both endpoints must be COMPLETE before closeout')
    if (DEST / 'complete.json').exists():
        done = read(DEST / 'complete.json')
        for p, h in done['outputs'].items():
            require(sha(p) == h, 'Existing closeout output changed')
        return done
    require(not DEST.exists(), 'Partial closeout exists; inspect it before retrying this CPU audit')
    started = time.time(); reg = read(OUT / 'training_registration.json')
    require(reg['status'] == 'FROZEN' and reg['arms'] == list(ARMS), 'Registration differs')
    for p, h in reg['training_code_hashes'].items():
        require(sha(p) == h, 'Frozen source changed: ' + p)
    for key in ('manifest', 'eval_manifest', 'source_checkpoint'):
        require(sha(reg[key]) == reg[key + '_sha256'], 'Protected source/input changed: ' + key)
    source_sidecar = read(resolve(reg['source_checkpoint']).with_suffix('.json'))
    require(source_sidecar['sha256'] == reg['source_checkpoint_sha256']
            and source_sidecar['generator_state_sha256'] == reg['source_generator_sha256']
            and source_sidecar['optimizer_state_sha256'] == reg['source_optimizer_sha256']
            and source_sidecar['rng_sha256'] == reg['source_rng_sha256'], 'Source sidecar states differ')
    results, identities, parents, bins, times = [], [], [], [], []
    for arm in ARMS:
        result, identity_rows, parent_rows, bin_rows, timing = audit_arm(reg, arm)
        results.append(result); identities.extend(identity_rows); parents.extend(parent_rows); bins.extend(bin_rows); times.append(timing)
    DEST.mkdir()
    table(DEST / 'actual_exposure_per_parent.csv', parents)
    table(DEST / 'exposure_identity_checks.csv', identities)
    table(DEST / 'FM_50step_bins.csv', bins)
    write_new(DEST / 'resource_and_attempt_wallclock.json', dict(arms=times, caution=CAUTION,
        dispatch_observations={str(p):sha(p) for p in sorted((OUT/'dispatch').glob('*observations.json'))}))
    plot_curves(bins, DEST / 'learning_curves.png')
    write_new(DEST / 'training_audit.json', dict(status='PASS', arms=results, caution=CAUTION,
        source_Coverage_unchanged=True, source_checkpoint_sha256=reg['source_checkpoint_sha256'],
        model_calls=0, tensor_loads=0, new_scientific_work=0))
    code = Path(__file__)
    diff = ''.join(difflib.unified_diff([], code.read_text(encoding='utf-8').splitlines(True),
                                       fromfile='/dev/null', tofile=str(code.relative_to(ROOT))))
    (DEST / 'code.diff').write_text(diff, encoding='utf-8')
    done = dict(schema='bench_exposure_allocation_training_closeout_v1', status='PASS',
        arms=list(ARMS), total_updates=3000, total_forwards=24000, total_backwards=24000,
        total_head_calls=0, failed_calls=0, pending_calls=0, exposure_identity_rows=24000,
        source_Coverage_unchanged=True, training_registration_sha256=sha(OUT/'training_registration.json'),
        audit_code_sha256=sha(code), outputs={str(p):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()},
        seconds=time.time()-started, model_calls=0, tensor_loads=0, caution=CAUTION)
    write_new(DEST/'complete.json', done)
    return done


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run()
    print(json.dumps({k:v for k,v in result.items() if k != 'outputs'}, indent=2))
