"""Read saved training records only; no torch, model, checkpoint tensor or scorer.

--snapshot writes nothing and checks the completed-log prefix while reporting
unfinished appends/calls separately. --final requires both exact registered
endpoints and writes results/loss_curves.csv/png plus detailed audit tables.
Loss ratios pool saved SSE and scalar denominators; they are training diagnostics,
not open-ended completion quality or held-out generalization measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

from .common import ROOT, OUT, resolve, read, sha

SCHEMA = 'BENCH_FULL_SEAT_CURRICULUM_TRAINING_SUMMARY_V1'
STAGES = ('NATIVE', 'SEAM')
ROLES = ('all_free', 'seam_tip', 'bulk')
FIELDS = tuple(role + suffix for role in ROLES for suffix in ('_velocity_sse', '_endpoint_sse', '_scalar_count'))
KINDS = ('train_forward', 'train_backward', 'optimizer_update', 'head_forward', 'head_backward', 'head_optimizer_update', 'probe_forward')
T_BUCKETS = ((0., .25), (.25, .5), (.5, .75), (.75, 1.))


def require(value, message):
    if not value:
        raise ValueError(message)


def near(a, b, message, rel=1e-7):
    require(math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-8), message)


def stream_jsonl(path, meta, final=False):
    """Bound reads to the byte extent observed on open; preserve partial-tail NA."""
    path = resolve(path)
    meta.update(exists=path.exists(), rows=0, unfinished_tail=False, observed_bytes=0)
    if not path.exists():
        return
    with path.open('rb') as f:
        left = os.fstat(f.fileno()).st_size
        meta['observed_bytes'] = left
        while left:
            line = f.readline(left); left -= len(line)
            if not line.endswith(b'\n'):
                meta['unfinished_tail'] = True
                require(not final, 'Final record has an unfinished JSONL append: ' + str(path))
                break
            if line.strip():
                meta['rows'] += 1
                yield json.loads(line)


def cursor(row, stage, step, reg):
    offset = 0 if stage == 'NATIVE' else reg['stage_updates']['NATIVE']
    expected = dict(stage=stage, stage_step=step, local_step=step,
        curriculum_step=offset + step, global_step=reg['source_global_step'] + offset + step,
        native_total_step=step if stage == 'NATIVE' else reg['stage_updates']['NATIVE'],
        seam_total_step=step if stage == 'SEAM' else 0)
    require(all(row.get(k) == v for k, v in expected.items()), 'Stage/global cursor differs: ' + repr(expected))


def add(acc, row):
    acc['records'] += 1
    for key in FIELDS:
        value = row[key]
        require(math.isfinite(float(value)) and value >= 0, 'Invalid saved SSE/count: ' + key)
        acc[key] += value


def metrics(acc):
    out = {k: acc.get(k, 0) for k in FIELDS}
    out['records'] = acc.get('records', 0)
    for role in ROLES:
        count = out[role + '_scalar_count']
        out[role + '_status'] = 'OK' if count else 'NA_NO_FREE_SCALARS_IN_STRATUM'
        for kind in ('velocity', 'endpoint'):
            out[role + '_' + kind + '_MSE'] = out[role + '_' + kind + '_sse'] / count if count else None
    return out


def pooled(rows):
    acc = Counter()
    for row in rows:
        add(acc, row)
    return metrics(acc)


def bucket(t):
    require(math.isfinite(t) and 0 <= t <= 1, 'Time outside [0,1]')
    return min(int(t * 4), 3)


def calls_summary(path, stage, reg, schedule, cutoff, meta, final):
    budget = reg['stage_updates'][stage]
    limits = {k: budget * (8 if k in ('train_forward', 'train_backward') else 1) if k in KINDS[:3] else 0 for k in KINDS}
    offset = 0 if stage == 'NATIVE' else reg['stage_updates']['NATIVE']
    pending, attempted, successful, failed, seconds = {}, Counter(), Counter(), Counter(), Counter()
    prefix_done, attempts = Counter(), set()
    last_id = 0
    for row in stream_jsonl(path, meta, final):
        kind, ident, context = row['kind'], row['call_id'], row['context']
        require(row['arm'] == stage and kind in limits, 'Call owner/kind differs')
        attempts.add(row['attempt_id'])
        cursor(context, stage, int(context['stage_step']), reg)
        if row['event'] == 'CALL_BEGIN':
            require(ident == last_id + 1 and not pending, 'Ledger duplicate/nonsequential/pending reservation')
            last_id = ident; pending[ident] = row; attempted[kind] += 1
            require(attempted[kind] <= limits[kind], 'Call budget exceeded: ' + kind)
            idx = attempted[kind] - 1
            expected_step = idx // 8 + 1 if kind != 'optimizer_update' else idx + 1
            require(context['stage_step'] == expected_step, 'Call stage-step sequence differs')
            if kind in ('train_forward', 'train_backward'):
                mi = idx % 8
                expected = schedule[offset + expected_step - 1]['samples'][mi]
                require(context['microbatch'] == mi and context['condition_id'] == expected['condition_id'], 'Call/sample sequence differs')
        elif row['event'] in ('CALL_DONE', 'CALL_FAILED'):
            begun = pending.pop(ident, None)
            require(begun is not None and begun['kind'] == kind and begun['context'] == context, 'Call end lacks matching begin')
            if row['event'] == 'CALL_DONE':
                successful[kind] += 1; seconds[kind] += row['seconds']
                if context['stage_step'] <= cutoff:
                    prefix_done[kind] += 1
            else:
                failed[kind] += 1
        else:
            raise ValueError('Unknown call event')
    for kind in KINDS:
        wanted = cutoff * (8 if kind in ('train_forward', 'train_backward') else 1) if kind in KINDS[:3] else 0
        require(prefix_done[kind] == wanted, 'Completed-log prefix calls differ: ' + kind)
    if final:
        require(not pending and not failed and all(attempted[k] == successful[k] == limits[k] for k in KINDS), 'Final ledger incomplete/failed')
    return dict(limits=limits, attempted={k: attempted[k] for k in KINDS}, successful={k: successful[k] for k in KINDS},
        failed={k: failed[k] for k in KINDS}, pending_call_ids=sorted(pending), attempt_ids=sorted(attempts),
        completed_prefix_calls=dict(prefix_done), successful_call_seconds=dict(seconds), no_head_or_probe_activity=all(attempted[k] == 0 for k in KINDS[3:]))


def stage_summary(stage, reg, manifest, schedule, final, reg_sha):
    folder = OUT / 'training' / stage
    status_path = folder / 'status.json'
    status = read(status_path) if status_path.exists() else dict(status='NOT_STARTED', stage_step=0)
    budget = reg['stage_updates'][stage]
    if status_path.exists():
        require(status['registration_sha256'] == reg_sha, 'Status registration hash differs')
        cursor(status, stage, int(status['stage_step']), reg)
        require(status.get('head_step', 0) == status.get('head_calls', 0) == status.get('probe_calls', 0) == 0, 'Head/probe activity in status')
    require(not final or status['status'] == 'COMPLETE' and status['stage_step'] == budget, '--final requires exact COMPLETE endpoint: ' + stage)
    file_meta = {}
    logs = list(stream_jsonl(folder / 'training_log.jsonl', file_meta.setdefault('training_log', {}), final))
    require(len(logs) <= budget, 'Too many completed training logs')
    for step, row in enumerate(logs, 1):
        cursor(row, stage, step, reg)
    # Status is saved separately after the log. Use the earlier committed cursor;
    # later complete log/exposure/call rows are disclosed rather than reclassified.
    cutoff = min(len(logs), int(status['stage_step']))
    all_log_rows = len(logs); logs = logs[:cutoff]
    require(not final or cutoff == budget and all_log_rows == budget, 'Final log cursor differs')
    offset = 0 if stage == 'NATIVE' else reg['stage_updates']['NATIVE']
    planned = schedule[offset:offset + budget]
    by_id = {r['condition_id']: r for r in manifest['train']}
    grouped, parents = defaultdict(Counter), defaultdict(Counter)
    expected_parents = Counter(r['parent_id'] for entry in planned for r in entry['samples'])
    expected_prefix = Counter(r['parent_id'] for entry in planned[:cutoff] for r in entry['samples'])
    exposed, batch_acc, batch_loss = 0, Counter(), 0.
    actual_rows = 0
    for row in stream_jsonl(folder / 'exposure.jsonl', file_meta.setdefault('exposure', {}), final):
        actual_rows += 1
        step, mi = (actual_rows - 1) // 8 + 1, (actual_rows - 1) % 8
        cursor(row, stage, step, reg)
        require(row['microbatch'] == mi and step <= budget, 'Exposure order/budget differs')
        expected = planned[step - 1]['samples'][mi]
        require(all(row.get(k) == v for k, v in expected.items()), 'Exposure differs from the frozen stream')
        require(row.get('head_forward_called') is False, 'Head was used during training')
        original = by_id[row['condition_id']]
        require((row['N'], row['K']) == (original['N'], original['K']), 'Exposure N/K differs')
        mechanism = row['mechanism']; owners = int(original['source_owner_count'])
        active = owners if stage == 'SEAM' else 0
        free = 9 * (row['N'] - row['K']) - 6 * active
        require(mechanism['source_owner_count'] == owners and mechanism['active_seam_slots'] == active and mechanism['free_coordinate_count'] == free, 'Native/SEAM source-owner mechanism differs')
        require(row['all_free_scalar_count'] == free and row['seam_tip_scalar_count'] == 3 * active and row['bulk_scalar_count'] == 9 * (row['N'] - row['K'] - active), 'Saved free-scalar accounting differs')
        if step > cutoff:
            continue
        exposed += 1
        add(batch_acc, row); batch_loss += row['loss_contribution']
        keys = [('part', row['part_type']), ('t_bucket', str(bucket(float(row['t'])))),
                ('part_t_bucket', row['part_type'] + '|' + str(bucket(float(row['t']))))]
        scopes = ['ALL']
        if step <= min(100, cutoff):
            scopes.append('FIRST100')
        if step > max(0, cutoff - 100):
            scopes.append('LAST100')
        for scope in scopes:
            for key in keys:
                add(grouped[(scope, *key)], row)
        acc = parents[row['parent_id']]; add(acc, row)
        acc['identity_instances' if row['augmentation'] == 'CLEAN' else 'x_width_instances'] += 1
        if mi == 7:
            log = logs[step - 1]
            for key in FIELDS:
                near(log[key], batch_acc[key], 'Exposure/log sum differs: ' + key)
            near(log['FM'], batch_loss, 'FM contribution sum differs', 4e-6)
            near(log['FM'], log['all_free_velocity_sse'] / log['all_free_scalar_count'], 'FM batch denominator differs', 4e-6)
            require(log['free_coordinate_denominator'] == log['all_free_scalar_count'], 'Batch denominator count differs')
            require(log['head_step'] == log['head_calls'] == 0, 'Head activity in training log')
            for kind in ('velocity', 'endpoint'):
                near(log['all_free_' + kind + '_sse'], log['bulk_' + kind + '_sse'] + log['seam_tip_' + kind + '_sse'], 'Role SSE partition differs', 3e-7)
            batch_acc, batch_loss = Counter(), 0.
    require(exposed == cutoff * 8, 'Completed prefix lacks eight exposures per update')
    require(not final or actual_rows == budget * 8, 'Final exposure count differs')
    ledger = calls_summary(folder / 'calls.jsonl', stage, reg, schedule, cutoff,
                           file_meta.setdefault('calls', {}), final)
    parent_rows = []
    for parent in sorted(expected_parents):
        acc = parents[parent]
        require(acc['records'] == expected_prefix[parent], 'Parent exposure differs from registered prefix')
        parent_rows.append(dict(stage=stage, parent_id=parent, actual_instances=acc['records'],
            planned_prefix_instances=expected_prefix[parent], planned_final_instances=expected_parents[parent],
            identity_instances=acc['identity_instances'], x_width_instances=acc['x_width_instances'], **metrics(acc)))
    strata = [dict(stage=stage, window=scope, stratum_type=unit, stratum=key, **metrics(acc))
              for (scope, unit, key), acc in sorted(grouped.items())]
    windows = [dict(stage=stage, start_stage_step=start + 1, end_stage_step=min(start + 100, cutoff),
        **pooled(logs[start:start+100])) for start in range(0, cutoff, 100)]
    curve = []
    for row in logs:
        curve.append({**{k: row[k] for k in ('stage', 'stage_step', 'curriculum_step', 'global_step', 'FM',
            'free_coordinate_denominator', 'generator_gradient_norm', 'input_seconds', 'update_seconds',
            'peak_allocated_bytes', 'peak_reserved_bytes')}, **metrics(dict(row, records=1))})
    attempt_evidence = []
    for path in sorted((folder / 'attempts').glob('*/initialization.json')):
        audit = read(path)
        require(audit['head_frozen'] is True and audit['head_calls'] == 0 and audit['head_state_sha256'] == reg['head_state_sha256'], 'Initialization head identity differs')
        require(audit['inherited_optimizer'] is True and audit['fresh_AdamW'] is False and audit['inherited_rng'] is True, 'COMMON moments/RNG inheritance differs')
        attempt_evidence.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), initialization=audit))
    resources = [dict(path=str(p.relative_to(ROOT)), **read(p).get('summary', {})) for p in sorted((folder / 'attempts').glob('*/resources.json'))]
    input_paths = [folder / name for name in ('status.json', 'training_log.jsonl', 'exposure.jsonl', 'calls.jsonl')]
    endpoints = None
    if final:
        require(attempt_evidence and resources, 'Final attempt provenance/resources missing')
        require('calls' in status and (folder / 'attempts' / status['attempt'] / 'final_status.json').exists(),
                'FINALIZATION_PENDING: COMPLETE is visible but the worker final ledger is not sealed yet')
        require(status['calls']['attempted'] == {k: v for k, v in ledger['attempted'].items() if v}, 'Final status attempted ledger differs')
        cp_path = folder / f'checkpoint_step{budget}.json'
        export_path = folder / f'generator_step{budget}.json'
        cp, exported = read(cp_path), read(export_path)
        require(status['checkpoint'] == cp and status['inference_export'] == exported, 'Final status/endpoint JSON receipts differ')
        require(exported['schema'] == 'bench_full_seat_curriculum_generator_v1', 'Export schema differs')
        for key in ('generator_state_sha256', 'head_state_sha256', 'stage', 'stage_step', 'local_step', 'global_step', 'curriculum_step'):
            require(exported[key] == cp[key], 'Checkpoint/export saved identity differs: ' + key)
        endpoints = dict(checkpoint_receipt=cp, inference_export_receipt=exported,
            comparison='Matching saved JSON receipt identities; no checkpoint tensor load or weight-storage rehash.',
            training_checkpoint_reference_in_export_json='NOT_RECORDED_IN_HISTORICAL_RECEIPT_SCHEMA')
        input_paths.extend((cp_path, export_path))
    info = dict(stage=stage, status=status['status'], registered_updates=budget, observed_status_step=status['stage_step'],
        completed_prefix_updates=cutoff, completed_prefix_instances=exposed,
        rows_after_prefix=dict(training_log=all_log_rows-cutoff, exposure=actual_rows-exposed),
        file_observation=file_meta, calls=ledger, losses=dict(all=pooled(logs),
            first100=pooled(logs[:100]), last100=pooled(logs[-100:]),
            first100_updates=min(100, cutoff), last100_updates=min(100, cutoff),
            overlap_when_fewer_than200=cutoff < 200),
        timing=dict(input_seconds=sum(r['input_seconds'] for r in logs), update_seconds=sum(r['update_seconds'] for r in logs),
                    observed_attempt_elapsed_seconds=status.get('elapsed_seconds')),
        resources=resources, initialization=attempt_evidence, endpoint_saved_identity=endpoints,
        input_hashes={str(p.relative_to(ROOT)): sha(p) for p in input_paths if p.exists()} if final else {})
    return info, curve, windows, strata, parent_rows


def transition_check(reg, final):
    def receipt(stage, step):
        path = OUT / 'training' / stage / f'checkpoint_step{step}.json'
        return read(path) if path.exists() else None
    native0 = receipt('NATIVE', 0)
    if native0:
        for key, expected in (('generator_state_sha256', 'source_generator_sha256'), ('optimizer_state_sha256', 'source_optimizer_sha256'), ('rng_sha256', 'source_rng_sha256'), ('head_state_sha256', 'head_state_sha256')):
            require(native0[key] == reg[expected], 'COMMON -> NATIVE saved receipt identity differs: ' + key)
    end = receipt('NATIVE', reg['stage_updates']['NATIVE']); seam0 = receipt('SEAM', 0)
    require(not final or native0 and end and seam0, 'Final boundary receipts missing')
    if end and seam0:
        for key in ('generator_state_sha256', 'optimizer_state_sha256', 'rng_sha256', 'head_state_sha256'):
            require(end[key] == seam0[key], 'Native -> SEAM continuity differs: ' + key)
    milestones = []
    for stage in STAGES:
        offset = 0 if stage == 'NATIVE' else reg['stage_updates']['NATIVE']
        for step in reg['stage_full_steps'][stage]:
            saved = receipt(stage, step)
            require(not final or saved is not None, 'Final immutable milestone receipt missing')
            if saved is not None:
                expected = dict(stage=stage, arm=stage, stage_step=step, local_step=step,
                    curriculum_step=offset+step, global_step=reg['source_global_step']+offset+step,
                    head_state_sha256=reg['head_state_sha256'])
                require(all(saved.get(k) == v for k,v in expected.items()), 'Milestone cursor/head identity differs')
                milestones.append(dict(**expected, checkpoint_sha256=saved['sha256'],
                    generator_state_sha256=saved['generator_state_sha256'],
                    optimizer_state_sha256=saved['optimizer_state_sha256'], rng_sha256=saved['rng_sha256']))
    return dict(common_to_native='PASS_SAVED_JSON_RECEIPTS' if native0 else 'NOT_YET_AVAILABLE',
        native_to_seam='PASS_SAVED_JSON_RECEIPTS' if end and seam0 else 'NOT_YET_AVAILABLE',
        milestone_receipts=milestones,
        weight_tensors_loaded=False, checkpoint_storage_hash_recomputed=False)


def summarize(final=False):
    reg_path = OUT / 'training_registration.json'; reg = read(reg_path)
    for name in ('manifest', 'input_schedule'):
        require(sha(reg[name]) == reg[name + '_sha256'], 'Frozen ' + name + ' changed')
    manifest, schedule = read(reg['manifest']), read(reg['input_schedule'])['steps']
    require(len(schedule) == sum(reg['stage_updates'].values()), 'Registered total schedule differs')
    info, curves, windows, strata, parents = [], [], [], [], []
    for stage in STAGES:
        a, b, c, d, e = stage_summary(stage, reg, manifest, schedule, final, sha(reg_path))
        info.append(a); curves.extend(b); windows.extend(c); strata.extend(d); parents.extend(e)
    summary = dict(schema=SCHEMA, status='COMPLETE' if final else 'SNAPSHOT',
        aggregation='Pooled sum(saved SSE)/sum(saved free scalar count), never mean of per-sample ratios.',
        loss_interpretation='Source-supervised training diagnostics, with different free-coordinate roles in Native and SEAM; no held-out or open-ended generation-quality claim.',
        t_buckets=[dict(index=i, low=a, high=b, low_inclusive=True, high_inclusive=i==3) for i, (a,b) in enumerate(T_BUCKETS)],
        global_steps=dict(source=reg['source_global_step'], native_end=reg['source_global_step']+reg['stage_updates']['NATIVE'],
                          seam_end=reg['final_global_step']),
        registration_sha256=sha(reg_path), code_sha256=sha(__file__), stages=info,
        stage_transition=transition_check(reg, final),
        own_activity=dict(model_forwards=0, head_forwards=0, optimizer_updates=0, scoring_calls=0, OT_calls=0, checkpoint_tensors_loaded=0),
        snapshot_writes=0, source_files_modified=False)
    return summary, curves, windows, strata, parents


def _csv(rows):
    stream = io.StringIO(newline='')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def _new_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, 'Final summary output already exists with other contents: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(content)


def _plot(rows, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), dpi=130)
    colors = {'NATIVE': '#2868a6', 'SEAM': '#d07826'}
    for ri, role in enumerate(ROLES):
        for ci, kind in enumerate(('velocity', 'endpoint')):
            ax = axes[ri,ci]
            for stage in STAGES:
                records = [r for r in rows if r['stage'] == stage]
                xs, ys = [], []
                for at, row in enumerate(records):
                    block = records[max(0, at-99):at+1]
                    denominator = sum(r[role+'_scalar_count'] for r in block)
                    if denominator:
                        xs.append(row['global_step']); ys.append(sum(r[role+'_'+kind+'_sse'] for r in block)/denominator)
                if xs:
                    ax.plot(xs, ys, label=stage, color=colors[stage], linewidth=1.3)
            ax.set_title(role + ' / ' + kind + ' MSE'); ax.set_xlabel('Generator global update'); ax.set_ylabel('Pooled SSE / free scalars')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
            else:
                ax.text(.5,.5,'NA: no free scalars in this role',ha='center',transform=ax.transAxes)
    fig.suptitle('Saved training diagnostics: trailing100 updates pooled within each stage\nNo extra forward; Native seam-tip is NA; stage roles differ', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    buffer=io.BytesIO();fig.savefig(buffer,format='png');plt.close(fig)
    _new_bytes(target,buffer.getvalue())


def finalize():
    summary, curves, windows, strata, parents = summarize(final=True)
    result = OUT / 'results'
    paths = {'loss_curves.csv': _csv(curves), 'training_windows100.csv': _csv(windows),
             'training_strata.csv': _csv(strata), 'training_parent_exposure.csv': _csv(parents)}
    for name, content in paths.items():
        _new_bytes(result/name, content)
    _plot(curves,result/'loss_curves.png')
    summary['artifacts'] = {name: sha(result/name) for name in (*paths, 'loss_curves.png')}
    _new_bytes(result/'training_summary.json', (json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    return summary


def archive_source():
    """Copy hash-verified registered source text only; never copy model weights."""
    reg = read(OUT/'training_registration.json')
    destination = OUT/'results/source_archive'
    sources = dict(reg['training_code_hashes'])
    sources[str(Path(__file__))] = sha(__file__)
    records = []
    for path, expected in sorted(sources.items()):
        source = resolve(path)
        require(source.suffix == '.py' and source.is_relative_to(ROOT), 'Only workspace Python sources can be archived')
        require(sha(source) == expected, 'Registered source changed before archive: '+str(source))
        relative=source.relative_to(ROOT); target=destination/relative
        _new_bytes(target,source.read_bytes())
        records.append(dict(source=str(relative),archive=str(target.relative_to(ROOT)),sha256=expected))
    receipt=dict(schema=SCHEMA+'_SOURCE_ARCHIVE',status='COMPLETE',registration_sha256=sha(OUT/'training_registration.json'),files=records,model_files_copied=0)
    _new_bytes(destination/'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--snapshot',action='store_true'); mode.add_argument('--final',action='store_true')
    mode.add_argument('--archive-source',action='store_true')
    args=p.parse_args()
    value=archive_source() if args.archive_source else finalize() if args.final else summarize()[0]
    print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
