"""Read-only CPU replay comparison for the third bounded engineering recovery."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
AUDIT = ROOT / 'audits/worker_write_race_20261019'
LIVE = ROOT / 'training/COMMON'
FIRST, LAST, NUM = 8751, 8886, 136

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def snapshot(path):
    raw = Path(path).read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            return None, None
    return rows, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                      bytes=len(raw), snapshot_unix=time.time())

def immutable(path, data):
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    final = AUDIT / 'replay_verification.json'
    if final.exists():
        print(json.dumps(read(final)))
        return
    metrics, mh = snapshot(LIVE / 'training_log.jsonl')
    if metrics is None:
        print(json.dumps(dict(status='WAIT_FOR_COMPLETE_JSONL_LINE')))
        return
    maxstep = max(r['step'] for r in metrics)
    if maxstep < LAST:
        print(json.dumps(dict(status='WAIT_FOR_REPLAY', last_logged_update=maxstep,
                              required_update=LAST)))
        return
    exposure, eh = snapshot(LIVE / 'exposure.jsonl')
    ledger, lh = snapshot(LIVE / 'ledger.jsonl')
    if exposure is None or ledger is None:
        print(json.dumps(dict(status='WAIT_FOR_COMPLETE_JSONL_LINE')))
        return
    recovery = read(AUDIT / 'authorized_recovery_receipt.json')
    expected = {}
    original_snapshots = []
    for name in ('training_log.jsonl', 'exposure.jsonl', 'ledger.jsonl'):
        rows, identity = snapshot(AUDIT / ('uncommitted.' + name))
        assert identity['sha256'] == recovery['log_files'][name]['tail_sha256']
        assert rows is not None
        assert all(FIRST <= r.get('step', 0) <= LAST for r in rows)
        expected[name] = rows
        original_snapshots.append(identity)
    assert len(metrics) == len({r['step'] for r in metrics})
    assert len(exposure) == len({(r['step'], r['micro']) for r in exposure})
    new_metrics = {r['step']: r for r in metrics if FIRST <= r['step'] <= LAST}
    new_exposure = {(r['step'], r['micro']): r for r in exposure if FIRST <= r['step'] <= LAST}
    assert len(new_metrics) == len(expected['training_log.jsonl']) == NUM
    assert len(new_exposure) == len(expected['exposure.jsonl']) == NUM * 8
    timing = {'unix', 'input_seconds', 'update_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes'}
    differences = []
    stats = collections.defaultdict(lambda: dict(comparisons=0, mismatches=0, max_abs_delta=0.))

    def compare(kind, key, old, new, excluded):
        if old.keys() != new.keys():
            differences.append(dict(kind=kind, key=str(key), field='SCHEMA',
                                    original=sorted(old), replayed=sorted(new)))
        for field in sorted((old.keys() & new.keys()) - excluded):
            a, b = old[field], new[field]
            s = stats[kind + '.' + field]
            s['comparisons'] += 1
            if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
                delta = abs(float(a) - float(b))
                assert math.isfinite(delta)
                s['max_abs_delta'] = max(s['max_abs_delta'], delta)
            if type(a) != type(b) or a != b:
                s['mismatches'] += 1
                differences.append(dict(kind=kind, key=str(key), field=field, original=a, replayed=b))

    for r in expected['training_log.jsonl']:
        compare('update', r['step'], r, new_metrics[r['step']], timing)
    for r in expected['exposure.jsonl']:
        compare('exposure', (r['step'], r['micro']), r, new_exposure[r['step'], r['micro']], {'unix'})
    counts = collections.Counter(r['event'] for r in ledger if FIRST <= r.get('step', 0) <= LAST)
    old_counts = collections.Counter(r['event'] for r in expected['ledger.jsonl'])
    assert counts == old_counts
    assert counts['FORWARD_COMPLETE'] == counts['BACKWARD_COMPLETE'] == NUM * 8
    assert counts['OPTIMIZER_COMPLETE'] == counts['UPDATE_COMPLETE'] == NUM
    assert counts['PROBE_FORWARD_COMPLETE'] == 0
    skip = [r for r in ledger if r.get('step') == FIRST - 1 and r['event'] == 'PROBE_NOT_SCHEDULED']
    assert len(skip) == 1
    assert skip[0]['status'] == 'NOT_SCHEDULED'
    assert skip[0]['reason'] == 'EXACT_OFF_GRID_RESUME_CURSOR'
    assert skip[0]['generator_forwards'] == skip[0]['generator_forward_budget'] == skip[0]['head_forwards'] == 0
    assert skip[0]['checkpoint_sha256'] == recovery['checkpoint_sha256']
    assert not (LIVE / 'fixed_loss/step8750.json').exists()
    assert not any(r.get('step') == 8750 and r['event'] in (
        'PROBE_STARTED', 'PROBE_FORWARD_STARTED', 'PROBE_FORWARD_COMPLETE', 'PROBE_COMPLETE') for r in ledger)
    with (AUDIT / 'replay_field_comparison.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field', 'comparisons', 'mismatches', 'max_abs_delta'])
        writer.writeheader()
        for field, value in sorted(stats.items()):
            writer.writerow(dict(field=field, **value))
    immutable(AUDIT / 'replay_differences.json', differences)
    result = dict(status='PASS_EXACT_SAVED_SCALARS_AND_EXPOSURE_REPLAY' if not differences else 'FAIL_REPLAY_NUMERIC_MISMATCH',
        replay_incident=3, range=[FIRST, LAST], updates_compared=NUM, exposure_rows_compared=NUM * 8,
        all_compared_values_exactly_equal=not differences, mismatch_count=len(differences),
        comparison='Exact JSON scalar equality; no tolerance or rounding; timing and allocator fields excluded',
        excluded_update_fields=sorted(timing), excluded_exposure_fields=['unix'],
        replay_event_counts=dict(counts), original_event_counts=dict(old_counts),
        offgrid_resume_probe=dict(status='PASS_NO_EXTRA_FORWARD', record=skip[0], artifact_exists=False),
        explicit_limit='Original step8886 parameters were not saved. Exact saved trajectory diagnostics do not directly prove parameter-by-parameter8886 identity.',
        live_snapshots=[mh, eh, lh], original_tail_snapshots=original_snapshots,
        last_live_logged_update=maxstep, model_forwards=0, GPU_calls=0, production_files_modified=False,
        verified_unix=time.time())
    immutable(final, result)
    previous = ROOT / 'recovery2_cpu_audit_20261019_1792345993997925922/recovery_compute_accounting.json'
    before = read(previous)
    assert before['status'] == 'BOTH_REPLAYS_VERIFIED'
    incidents = before['incidents'] + [dict(incident=3, optimizer_updates=136,
        forward_returns=1088, backward_returns=1088, backward_result_unknown=0,
        extra_probe_forwards=0, source=str(AUDIT / 'checkpoint_audit.json'))]
    accounting = dict(schema='bench_parts_three_recoveries_compute_accounting_v1',
        status='ALL_THREE_REPLAYS_VERIFIED' if not differences else 'THIRD_REPLAY_MISMATCH_REQUIRES_REVIEW',
        scientific_registration_changed=False, unique_endpoint_extension=0, incidents=incidents,
        total_discarded_attempt_extra=dict(optimizer_updates=325, training_forward_returns=2604,
            training_backward_returns=2603, training_backward_starts=2604, backward_result_unknown=1,
            head_optimizer_updates=0, extra_fixed_probe_forwards=0),
        final_common_if_unique12000=dict(unique_optimizer_updates=12000, physical_optimizer_updates=12325,
            training_forward_returns=98604, training_backward_returns=98603, training_backward_starts=98604,
            excludes_registered_fixed_probe_forwards=True),
        all_arms_formula='unique optimizer updates=12000+2*(paired_final_global_step-12000); physical optimizer updates=unique+325',
        log_policy='Canonical logs retain one committed row per global step. Archived discarded-attempt counts add physical compute, not new scientific updates or duplicated learning-curve points.',
        replay_verification=str(final), previous_incident_accounting=str(previous), created_unix=time.time())
    immutable(AUDIT / 'recovery_compute_accounting.json', accounting)
    print(json.dumps(dict(status=result['status'], mismatches=len(differences), updates=NUM,
                          exposures=NUM * 8, skip8750='PASS_0_FORWARD', receipt=str(final))))

if __name__ == '__main__':
    main()
