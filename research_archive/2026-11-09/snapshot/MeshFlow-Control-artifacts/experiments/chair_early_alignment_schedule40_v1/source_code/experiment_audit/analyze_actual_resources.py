#!/usr/bin/env python3
"""Read-only CPU accounting of observed schedule40 resources and worker ownership.

Independent of the frozen scientific scorers and original accounting analyzer.
Never imports torch, queries hardware, starts workers, or infers phase from names.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

DEFAULT = '/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1'
SCHEMA = 'schedule40_actual_resources_v1'


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def read_jsonl_snapshot(path):
    """Read only the initial byte extent; tolerate a concurrent incomplete last line."""
    path = Path(path)
    if not path.exists():
        return [], dict(path=str(path), status='MISSING', bytes_read=0), []
    with path.open('rb') as handle:
        extent = os.fstat(handle.fileno()).st_size
        data = handle.read(extent)
    rows, warnings = [], []
    lines = data.splitlines(keepends=True)
    for index, raw in enumerate(lines, 1):
        if index == len(lines) and not raw.endswith((b'\n', b'\r')):
            warnings.append(dict(kind='INCOMPLETE_TRAILING_LINE_IGNORED', path=str(path), line=index))
            continue
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError('row is not an object')
            row['_line'] = index
            rows.append(row)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            warnings.append(dict(kind='MALFORMED_LINE', path=str(path), line=index, error=str(error)))
    return rows, dict(path=str(path), status='READ', bytes_read=len(data), initial_extent_bytes=extent,
                     read_prefix_sha256=hashlib.sha256(data).hexdigest(), complete_rows=len(rows)), warnings


def intervals_from_events(events, prefix='WORKER'):
    intervals, opened, issues = [], {}, []
    for sequence, event in enumerate(events):
        kind = event.get('event')
        if kind not in (prefix + '_START', prefix + '_END'):
            continue
        stamp = number(event.get('time_unix'))
        if stamp is None:
            issues.append(dict(kind='EVENT_TIME_MISSING', sequence=sequence, event=kind))
            continue
        phase = event.get('phase') if prefix == 'WORKER' else 'cpu'
        name, attempt = event.get('name'), event.get('attempt', 'unspecified')
        key = (phase, name, attempt)
        if kind.endswith('_START'):
            item = dict(identity=f'{prefix}:{phase}:{name}:{attempt}:{sequence}', phase=phase or 'UNKNOWN',
                        name=name, attempt=attempt, start_unix=stamp, end_unix=None,
                        start_sequence=sequence, end_sequence=None, owner_wall_seconds=None,
                        returncode=None, closed=False)
            if key in opened:
                issues.append(dict(kind='OVERLAPPING_SAME_IDENTITY_START', identity=item['identity']))
            opened.setdefault(key, []).append(item)
            intervals.append(item)
        else:
            candidates = opened.get(key, [])
            if len(candidates) != 1:
                issues.append(dict(kind='UNMATCHED_OR_AMBIGUOUS_END', phase=phase, name=name,
                                   attempt=attempt, time_unix=stamp, candidate_count=len(candidates)))
                continue
            item = candidates.pop()
            opened.pop(key)
            item.update(end_unix=stamp, end_sequence=sequence, owner_wall_seconds=number(event.get('wall_seconds')),
                        returncode=event.get('returncode'), closed=True)
            if stamp < item['start_unix']:
                issues.append(dict(kind='NEGATIVE_INTERVAL', identity=item['identity']))
    return intervals, issues


def overlaps(item, stamp, sequence=None):
    if stamp is None or stamp < item['start_unix']:
        return False
    if item['end_unix'] is not None and stamp > item['end_unix']:
        return False
    if sequence is not None:
        if sequence < item['start_sequence']:
            return False
        if item['end_sequence'] is not None and sequence >= item['end_sequence']:
            return False
    elif item['end_unix'] is not None and stamp >= item['end_unix']:
        return False
    return True


def gpu_values(value):
    if not isinstance(value, dict):
        return None
    if 'used_mib' in value:
        return {key: value.get(key) if key == 'name' else number(value.get(key))
                for key in ('name', 'total_mib', 'used_mib', 'free_mib', 'utilization')}
    if value.get('returncode', value.get('code')) != 0:
        return None
    try:
        rows = list(csv.reader(io.StringIO(value.get('stdout', ''))))
        if len(rows) != 1 or len(rows[0]) != 5:
            return None
        fields = [part.strip() for part in rows[0]]
        return dict(name=fields[0], total_mib=number(fields[1]), used_mib=number(fields[2]),
                    free_mib=number(fields[3]), utilization=number(fields[4]))
    except (csv.Error, ValueError):
        return None


def argument(argv, key):
    try:
        return argv[argv.index(key) + 1]
    except (ValueError, IndexError):
        return None


def guard_accounting(events, cutoff):
    windows, issues, signal_events = [], [], []
    pending = None
    for event in events:
        kind = event.get('event', '')
        if kind.startswith('SIG'):
            signal_events.append({key: value for key, value in event.items() if key != '_line'})
        if kind == 'SIGSTOP_CONTROLLER_ONLY':
            controller, late = event.get('controller', {}), event.get('late', {})
            verified = (argument(controller.get('argv', []), '--stage') == 'train'
                        and argument(late.get('argv', []), '--segment') == 'LATE'
                        and late.get('ppid') == controller.get('pid')
                        and controller.get('pid') is not None)
            pending = dict(start_unix=number(event.get('time_unix')), end_unix=None,
                           controller_pid=controller.get('pid'), late_pid=late.get('pid'),
                           verified_training_LATE_identity=verified, closed=False,
                           pause_target='controller only', gpu_worker_pause_seconds=0)
            windows.append(pending)
            if not verified:
                issues.append(dict(kind='GUARD_STOP_IDENTITY_NOT_VERIFIED'))
        elif kind in ('SIGCONT_CONTROLLER_AFTER_LATE_COMPLETE', 'SIGKILL_VERIFIED_CONTROLLER_ONLY'):
            if pending is None:
                issues.append(dict(kind='GUARD_END_WITHOUT_STOP', event=kind))
            else:
                pending.update(end_unix=number(event.get('time_unix')), end_event=kind, closed=True)
                pending = None
        if 'ALERT' in kind or kind == 'STOP_NOT_VERIFIED':
            issues.append(dict(kind='GUARD_ALERT', event=kind, time_unix=event.get('time_unix')))
        if number(event.get('gpu_worker_signals')) not in (None, 0):
            issues.append(dict(kind='GUARD_REPORTED_GPU_WORKER_SIGNAL', event=kind))
    for window in windows:
        end = window['end_unix'] if window['closed'] else cutoff
        window['observed_controller_stopped_seconds'] = max(0, end - window['start_unix']) if end is not None and window['start_unix'] is not None else None
        window['censored'] = not window['closed']
    return dict(windows=windows, signal_events=signal_events, issues=issues,
                controller_stop_is_not_gpu_worker_pause=True,
                gpu_worker_wall_time_subtracted_seconds=0,
                note='SIGSTOP/SIGCONT target the verified controller. The GPU worker continues; owner wall time includes this interval.')


def assign_snapshot(snapshot, workers, cpus, source, sequence=None, expected_names=None, guard=None):
    stamp = number(snapshot.get('time_unix'))
    candidates = [item for item in workers if overlaps(item, stamp, sequence)]
    issues = []
    if expected_names is not None:
        selected = []
        for name in expected_names:
            matches = [item for item in candidates if item['name'] == name]
            if len(matches) == 1:
                selected.extend(matches)
            else:
                issues.append(f'active name {name!r} matches {len(matches)} timed worker intervals')
        candidates = selected
    phases = sorted({item['phase'] for item in candidates})
    active_cpu = [item for item in cpus if overlaps(item, stamp, sequence)]
    if issues:
        phase = 'UNKNOWN'
    elif len(phases) == 1:
        phase = phases[0]
    elif len(phases) > 1:
        phase = 'MIXED'
    elif active_cpu:
        phase = 'cpu'
    else:
        phase = 'idle_or_unattributed'
    guard_window = None
    if source == 'serial_tail_guard':
        for index, window in enumerate((guard or {}).get('windows', [])):
            if window['start_unix'] is not None and stamp is not None and stamp >= window['start_unix'] and (window['end_unix'] is None or stamp < window['end_unix']):
                guard_window = index
                if not (window['verified_training_LATE_identity'] and phase == 'training'
                        and len(candidates) == 1 and candidates[0]['name'] == 'LATE'):
                    issues.append('guard window cannot be matched uniquely to the training LATE worker interval')
                    phase = 'UNKNOWN'
                break
    ram = snapshot.get('ram') or {}
    gpu = gpu_values(snapshot.get('gpu'))
    process_query = snapshot.get('gpu_processes') or {}
    query_text = process_query.get('stdout', '')
    return dict(time_unix=stamp, source=source, source_sequence=sequence, phase=phase,
                worker_identities=[item['identity'] for item in candidates],
                worker_names=[item['name'] for item in candidates], observed_active_owner_count=len(candidates),
                registered_active_names=expected_names, cpu_names=[item['name'] for item in active_cpu],
                guard_window=guard_window, guard_state=snapshot.get('phase') if source == 'serial_tail_guard' else None,
                gpu=gpu, cpu_cores=snapshot.get('cpu_cores'),
                mem_total_bytes=number(ram.get('MemTotal')), mem_available_bytes=number(ram.get('MemAvailable')),
                loadavg=snapshot.get('loadavg'), gpu_process_memory_unavailable='[N/A]' in query_text,
                issues=issues)


def interval_totals(items, cutoff):
    spans, closed_spans, owner_values, open_count = [], [], [], 0
    for item in items:
        end = item['end_unix']
        if end is None:
            open_count += 1
            end = cutoff
        if end is not None and end >= item['start_unix']:
            spans.append((item['start_unix'], end))
            if item['closed']:
                closed_spans.append((item['start_unix'], end))
        if item['closed'] and item['owner_wall_seconds'] is not None:
            owner_values.append(item['owner_wall_seconds'])
    def union(parts):
        total = 0.0
        edge = None
        for start, end in sorted(parts):
            if edge is None or start > edge:
                total += end - start
                edge = end
            elif end > edge:
                total += end - edge
                edge = end
        return total
    changes = []
    for start, end in spans:
        if end > start:
            changes.extend([(start, 1), (end, -1)])
    level = peak = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        level += delta
        peak = max(peak, level)
    return dict(worker_attempt_count=len(items), closed_attempt_count=sum(item['closed'] for item in items),
                open_attempt_count=open_count, failed_closed_attempt_count=sum(item['closed'] and item['returncode'] != 0 for item in items),
                recorded_owner_wall_sum_seconds=sum(owner_values), owner_wall_valid_count=len(owner_values),
                closed_worker_interval_union_seconds=union(closed_spans),
                observed_worker_interval_union_seconds=union(spans),
                observed_owner_interval_sum_seconds=sum(end-start for start, end in spans),
                max_registered_owner_concurrency=peak,
                includes_censored_open_intervals=open_count > 0)


def summary(rows):
    def values(key, gpu=False):
        return [value for row in rows if (value := number((row.get('gpu') or {}).get(key) if gpu else row.get(key))) is not None]
    def extreme(key, fn, gpu=False):
        entries = values(key, gpu)
        return fn(entries) if entries else None
    used, free, util = values('used_mib', True), values('free_mib', True), values('utilization', True)
    return dict(snapshot_count=len(rows), snapshot_count_by_source=dict(Counter(row['source'] for row in rows)),
                gpu_valid_snapshot_count=sum(row['gpu'] is not None for row in rows),
                gpu_device_names=sorted({row['gpu']['name'] for row in rows if row['gpu'] and row['gpu'].get('name')}),
                gpu_total_mib_observed=sorted(set(values('total_mib', True))),
                device_used_mib_max_observed=max(used) if used else None,
                device_free_mib_min_observed=min(free) if free else None,
                device_utilization_percent_max_observed=max(util) if util else None,
                device_utilization_percent_sample_mean=sum(util)/len(util) if util else None,
                gpu_process_memory_unavailable_snapshot_count=sum(row['gpu_process_memory_unavailable'] for row in rows),
                cpu_core_counts_observed=sorted(set(values('cpu_cores'))),
                ram_available_bytes_min_observed=extreme('mem_available_bytes', min),
                ram_total_bytes_observed=sorted(set(values('mem_total_bytes'))),
                ram_used_bytes_max_observed=max([row['mem_total_bytes']-row['mem_available_bytes'] for row in rows if row.get('mem_total_bytes') is not None and row.get('mem_available_bytes') is not None] or [None]),
                loadavg_1m_max_observed=max([row['loadavg'][0] for row in rows if row.get('loadavg')] or [None]),
                max_snapshot_registered_owner_concurrency=max([row['observed_active_owner_count'] for row in rows] or [0]),
                snapshot_attribution_issue_count=sum(bool(row['issues']) for row in rows))


def analyze(events, guard_events, guard_resources):
    workers, worker_issues = intervals_from_events(events)
    cpus, cpu_issues = intervals_from_events(events, 'CPU')
    times = [number(item.get('time_unix')) for item in events + guard_events + guard_resources]
    cutoff = max([stamp for stamp in times if stamp is not None] or [0])
    guard = guard_accounting(guard_events, cutoff)
    snapshots = []
    for sequence, event in enumerate(events):
        if event.get('event') == 'RESOURCES':
            snap = dict(event.get('snapshot') or {})
            snap.setdefault('time_unix', event.get('time_unix'))
            row = assign_snapshot(snap, workers, cpus, 'controller', sequence, event.get('active'), guard)
            row.update(reason=event.get('reason'), event_time_unix=event.get('time_unix'))
            snapshots.append(row)
    for resource in guard_resources:
        snapshots.append(assign_snapshot(resource, workers, cpus, 'serial_tail_guard', guard=guard))
    snapshots.sort(key=lambda item: (item['time_unix'] or 0, item['source']))
    phases = sorted(set(['training', 'sampling', 'cpu', 'idle_or_unattributed', 'UNKNOWN', 'MIXED']) | {row['phase'] for row in snapshots})
    by_phase = {}
    for phase in phases:
        rows = [row for row in snapshots if row['phase'] == phase]
        items = [item for item in (cpus if phase == 'cpu' else workers) if item['phase'] == phase]
        by_phase[phase] = dict(resources=summary(rows), timing=interval_totals(items, cutoff))
    complete_phases = [event.get('phase') for event in events if event.get('event') == 'CONTROLLER_PHASE_COMPLETE']
    return dict(schema=SCHEMA, status='INCREMENTAL' if any(not item['closed'] for item in workers+cpus) else 'CLOSED_OBSERVED_INTERVALS',
                scope='Resource and observed owner timing audit only; does not certify scientific experiment completion.',
                observation_cutoff_unix=cutoff, controller_completed_phases=complete_phases,
                worker_attempts=workers, cpu_attempts=cpus, phases=by_phase,
                all_resource_snapshots=summary(snapshots), all_worker_timing=interval_totals(workers, cutoff),
                guard=guard, issues=worker_issues+cpu_issues+guard['issues'], snapshots=snapshots,
                limitations=[
                    'Phase uses timed WORKER_START/WORKER_END identities (phase, name, attempt); name alone never determines phase.',
                    'GPU figures are whole-device observations at logged snapshots, including desktop/unrelated processes; maxima are sampled lower bounds on instantaneous peaks, not per-process VRAM peaks.',
                    'Controller and guard sampling cadences differ. Utilization sample means are not time-weighted GPU utilization.',
                    'Registered owner concurrency may include a finished child not yet reaped; it is not simultaneous CUDA-kernel concurrency.',
                    'Worker owner wall includes loading, CPU work, checkpointing, contention and polling lag. Parallel wall times are not isolated throughput benchmarks.',
                    'Owner wall sum counts all observed attempts, including failed attempts. Interval union measures elapsed ownership coverage; the two are different quantities.',
                    'The serial-tail guard stops/resumes only the controller; GPU worker time is never paused or subtracted.',
                    'Open intervals are right-censored at the latest captured log timestamp. CLOSED_OBSERVED_INTERVALS does not assert all planned workers have run.',
                    'Only RESOURCES events and guard resources rows are counted; embedded WORKER_START resource payloads are not duplicated.',
                    'Resource queries occur sequentially within a snapshot; recorded capture times are observation anchors, not simultaneous hardware samples.',
                    'Each input log is read to its initial byte extent and hashed separately; an incremental read is not an atomic snapshot across files. Re-run after dispatch completion for final observed intervals.'
                ])


def self_test():
    def event(kind, stamp, name='EARLY', phase='training', **extra):
        return dict(event=kind, time_unix=stamp, name=name, phase=phase, attempt='initial', **extra)
    def resource(stamp, active):
        return dict(event='RESOURCES', time_unix=stamp, active=active,
                    snapshot=dict(time_unix=stamp, gpu=dict(name='fake', total_mib=16, used_mib=4, free_mib=12, utilization=50)))
    events = [event('WORKER_START',0), event('WORKER_START',5,'COMMON'), resource(6,['EARLY','COMMON']),
              event('WORKER_END',10,wall_seconds=10,returncode=0), event('WORKER_END',15,'COMMON',wall_seconds=10,returncode=0),
              event('WORKER_START',20,phase='sampling'), resource(25,['EARLY']), event('WORKER_END',30,phase='sampling',wall_seconds=10,returncode=0),
              event('WORKER_START',40,'LATE'), event('WORKER_END',51,'LATE',wall_seconds=11,returncode=0),
              event('WORKER_START',60,'LATE','sampling'), resource(65,['LATE']), event('WORKER_END',70,'LATE','sampling',wall_seconds=10,returncode=0)]
    guards = [dict(event='SIGSTOP_CONTROLLER_ONLY',time_unix=42,
                   controller=dict(pid=1,argv=['controller','--stage','train']),
                   late=dict(pid=2,ppid=1,argv=['worker','--segment','LATE'])),
              dict(event='SIGCONT_CONTROLLER_AFTER_LATE_COMPLETE',time_unix=50,gpu_worker_signals=0)]
    guard_rows = [dict(time_unix=45,phase='CONTROLLER_STOPPED_LATE_RUNNING',gpu=dict(returncode=0,stdout='fake, 16, 5, 11, 70\n'))]
    report = analyze(events, guards, guard_rows)
    assert [row['phase'] for row in report['snapshots']] == ['training','sampling','training','sampling']
    assert report['phases']['training']['timing']['closed_worker_interval_union_seconds'] == 26
    assert report['phases']['training']['timing']['recorded_owner_wall_sum_seconds'] == 31
    assert report['phases']['training']['timing']['max_registered_owner_concurrency'] == 2
    assert report['guard']['windows'][0]['observed_controller_stopped_seconds'] == 8
    assert report['guard']['gpu_worker_wall_time_subtracted_seconds'] == 0
    assert report['phases']['training']['resources']['snapshot_count_by_source'] == {'controller':1,'serial_tail_guard':1}
    assert report['phases']['sampling']['resources']['snapshot_count'] == 2
    assert report['phases']['training']['resources']['device_used_mib_max_observed'] == 5
    open_report = analyze([event('WORKER_START',40,'LATE')], guards[:1], guard_rows)
    assert open_report['status'] == 'INCREMENTAL'
    assert open_report['all_worker_timing']['observed_worker_interval_union_seconds'] == 5
    assert open_report['all_worker_timing']['owner_wall_valid_count'] == 0
    assert open_report['guard']['windows'][0]['censored']
    bad_report = analyze([], guards, guard_rows)
    assert bad_report['snapshots'][0]['phase'] == 'UNKNOWN'
    assert bad_report['snapshots'][0]['issues']
    print(json.dumps(dict(status='PASS', checks=['same_name_training_sampling','overlap_union_sum_concurrency','guard_training_attribution',
        'controller_pause_no_worker_deduction','guard_gpu_parser','open_interval_censoring','unmatched_guard_unknown']),sort_keys=True))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(DEFAULT))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    audit = args.out / 'audit'
    paths = [audit/'controller.jsonl', audit/'serial_tail_guard/events.jsonl', audit/'serial_tail_guard/resources.jsonl']
    loaded = [read_jsonl_snapshot(path) for path in paths]
    report = analyze(*(item[0] for item in loaded))
    report['input_snapshots'] = [item[1] for item in loaded]
    report['read_warnings'] = [warning for item in loaded for warning in item[2]]
    report['generated_unix'] = time.time()
    report['source'] = dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = audit / 'resource_timing_audit.json'
    atomic_json(output, report)
    print(json.dumps(dict(status=report['status'], output=str(output),
                         worker_attempts=len(report['worker_attempts']), snapshots=report['all_resource_snapshots']['snapshot_count'],
                         issue_count=len(report['issues']), read_warning_count=len(report['read_warnings'])),sort_keys=True))


if __name__ == '__main__':
    main()