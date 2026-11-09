"""Summarize recorded pilot resources without executing or measuring new work.

Reads Phase polling, dispatch records, existing CUDA counters and CPU receipts.
Never imports torch, opens checkpoint weights, hashes large protected assets,
queries current hardware, launches workers, or infers WDDM process memory.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

from . import data as D

VERSION = 'terminal_reward_nft_recorded_resources_v1'


def stamp(value):
    if not value: return None
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def numeric(value):
    try: return float(value)
    except (TypeError, ValueError): return None


def maximum(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def minimum(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


class Inputs:
    def __init__(self):
        self.sources, self.values, self.stamps = {}, {}, {}
        self.notes, self.errors = [], []

    def read_bytes(self, path):
        path = D.resolve(path)
        if path.suffix not in ('.json', '.jsonl', '.py'):
            raise ValueError('Resource summary only reads small text records: ' + str(path))
        before = path.stat()
        if before.st_size > 64 * 1024 * 1024:
            raise ValueError('Refusing unexpectedly large resource record')
        payload = path.read_bytes(); after = path.stat()
        key = os.path.abspath(path)
        self.sources[key] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        self.stamps[key] = (after.st_size, after.st_mtime_ns)
        if (before.st_size, before.st_mtime_ns) != self.stamps[key]:
            self.notes.append(dict(path=key, reason='CHANGED_DURING_READ'))
        return payload

    def read(self, path):
        key = os.path.abspath(D.resolve(path))
        if key not in self.values:
            self.values[key] = json.loads(self.read_bytes(path).decode('utf-8-sig'))
        return self.values[key]

    def events(self, path):
        payload = self.read_bytes(path)
        if payload and not payload.endswith(b'\n'):
            payload, _, tail = payload.rpartition(b'\n')
            self.notes.append(dict(path=str(path), reason='PARTIAL_JSONL_TAIL', ignored_bytes=len(tail)))
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    def safe(self, scope, fn, fallback=None):
        try: return fn()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.errors.append(dict(scope=str(scope), error=repr(exc)))
            return fallback

    def finish(self):
        for path, before in self.stamps.items():
            after = Path(path).stat()
            if before != (after.st_size, after.st_mtime_ns):
                self.notes.append(dict(path=path, reason='CHANGED_AFTER_READ'))


def cpu_processes(command):
    if not command or command.get('returncode') != 0: return []
    lines = command.get('stdout', '').splitlines()
    if not lines: return []
    header = lines[0].split(); rows = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != len(header): continue
        value = dict(zip(header, fields))
        rows.append(dict(pid=int(value['PID']) if value.get('PID', '').isdigit() else None,
            ppid=int(value['PPID']) if value.get('PPID', '').isdigit() else None,
            command=value.get('COMMAND', value.get('CMD')), CPU_percent=numeric(value.get('%CPU')),
            RAM_percent=numeric(value.get('%MEM')), RSS_MiB=(numeric(value.get('RSS')) / 1024 if numeric(value.get('RSS')) is not None else None),
            threads=numeric(value.get('NLWP'))))
    return rows


def point(snapshot, source, index, kind):
    devices = []
    gpu = snapshot.get('gpu', {})
    if gpu.get('returncode') == 0:
        for row in csv.reader(io.StringIO(gpu.get('stdout', ''))):
            if len(row) == 5:
                devices.append(dict(name=row[0].strip(), total_MiB=numeric(row[1]), used_MiB=numeric(row[2]),
                                    free_MiB=numeric(row[3]), utilization_percent=numeric(row[4])))
    processes = []
    proc = snapshot.get('processes', {})
    if proc.get('returncode') == 0:
        for row in csv.reader(io.StringIO(proc.get('stdout', ''))):
            if len(row) >= 3:
                processes.append(dict(pid=int(row[0]) if row[0].strip().isdigit() else None,
                    name=row[1].strip(), used_MiB=numeric(row[-1]), reported_memory=row[-1].strip()))
    memory = {}
    for line in (snapshot.get('meminfo') or '').splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].rstrip(':') in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[fields[0].rstrip(':') + '_MiB'] = float(fields[1]) / 1024
    return dict(utc=snapshot.get('utc'), timestamp=stamp(snapshot.get('utc')), snapshot_pid=snapshot.get('pid'),
        source_path=str(source), index=index, source_kind=kind, devices=devices, GPU_processes=processes,
        GPU_query_returncode=gpu.get('returncode'), GPU_process_query_returncode=proc.get('returncode'),
        CPU_cores=snapshot.get('available_CPU_cores', snapshot.get('cpu_count')),
        CPU_load_average=snapshot.get('load_average'), CPU_processes=cpu_processes(snapshot.get('CPU_process_load')),
        **memory)


def profile(points, owned_pid=None):
    points = sorted(points, key=lambda p: p['timestamp'] if p['timestamp'] is not None else float('inf'))
    devices = [d for p in points for d in p['devices']]
    own_gpu = [r['used_MiB'] for p in points for r in p['GPU_processes'] if owned_pid is not None and r['pid'] == owned_pid]
    own_cpu = [r for p in points for r in p['CPU_processes'] if owned_pid is not None and r['pid'] == owned_pid]
    stamps = sorted({p['timestamp'] for p in points if p['timestamp'] is not None})
    total = maximum(d['total_MiB'] for d in devices)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return dict(recorded_snapshots=len(points), recorded_midphase_snapshots=sum(p['source_kind'] == 'PHASE_POLL' for p in points),
        GPU_names=sorted({d['name'] for d in devices}), GPU_total_MiB=total,
        device_used_max_MiB=maximum(d['used_MiB'] for d in devices), device_free_min_MiB=minimum(d['free_MiB'] for d in devices),
        device_free_max_MiB=maximum(d['free_MiB'] for d in devices), device_utilization_max_percent=maximum(d['utilization_percent'] for d in devices),
        device_utilization_min_percent=minimum(d['utilization_percent'] for d in devices),
        device_used_initial_MiB=points[0]['devices'][0]['used_MiB'] if points and points[0]['devices'] else None,
        reported_owned_GPU_process_peak_MiB=maximum(own_gpu),
        owned_GPU_process_memory_status='REPORTED_NUMERIC' if maximum(own_gpu) is not None else 'NA_NO_MATCHED_NUMERIC_PROCESS_MEMORY',
        GPU_process_memory_NA_records=sum(r['used_MiB'] is None for p in points for r in p['GPU_processes']),
        global_GPU_process_count_max=maximum(len(p['GPU_processes']) for p in points if p['GPU_process_query_returncode'] == 0),
        GPU_query_failure_records=sum(p['GPU_query_returncode'] not in (0, None) for p in points),
        samples_at_least_85pct_GPU=sum(d['utilization_percent'] is not None and d['utilization_percent'] >= 85 for d in devices),
        CPU_cores_min=minimum(p['CPU_cores'] for p in points), CPU_cores_max=maximum(p['CPU_cores'] for p in points),
        CPU_load1_max=maximum(p['CPU_load_average'][0] for p in points if p['CPU_load_average']),
        recorded_CPU_load_samples=sum(bool(p['CPU_load_average']) for p in points),
        host_RAM_total_MiB=maximum(p.get('MemTotal_MiB') for p in points),
        host_RAM_available_min_MiB=minimum(p.get('MemAvailable_MiB') for p in points),
        host_RAM_available_max_MiB=maximum(p.get('MemAvailable_MiB') for p in points),
        reported_owned_ps_CPU_percent_max=maximum(r['CPU_percent'] for r in own_cpu),
        reported_owned_RSS_peak_MiB=maximum(r['RSS_MiB'] for r in own_cpu),
        snapshot_gap_max_seconds=maximum(gaps), snapshot_gap_min_seconds=minimum(gaps),
        samples_below_2GiB_or_10pct_free_headroom=sum(d['free_MiB'] is not None and d['total_MiB'] is not None
            and d['free_MiB'] < max(2048, .1 * d['total_MiB']) for d in devices),
        scope='Sampled whole-device extrema include unrelated processes; polling can miss transient peaks. No process VRAM is inferred from device deltas or PyTorch counters. ps CPU percentages are reported lifetime averages, not instantaneous CPU utilization.')


def classify(path, identity):
    path = Path(path)
    final = 'evaluation' in path.parts
    stage = str(identity.get('stage', ''))
    name = path.name
    if name.startswith('sampling_worker'): kind = 'sampling'
    elif name.startswith('train_'): kind = 'training'
    elif 'denoise' in name or 'D0' in stage: kind = 'D0'
    elif name == 'cpu': kind = 'CPU_post_score_render'
    else: kind = stage or name
    ri = identity.get('round_index', identity.get('round'))
    if ri is None:
        for part in path.parts:
            if part.startswith('round_') and part[6:].isdigit(): ri = int(part[6:])
    label = identity.get('arm', identity.get('label'))
    return dict(kind=kind, scope='FINAL' if final else 'TRAINING', round_index=ri, label=label,
        group=('final/' + kind) if final else f'round_{ri:02d}/{kind}' if ri is not None else 'other/' + kind,
        GPU_worker=kind in ('sampling', 'training', 'D0'))


def cuda_record(value, source, scope=None):
    return dict(path=str(source), allocated_bytes=value.get('cuda_peak_allocated_bytes'), reserved_bytes=value.get('cuda_peak_reserved_bytes'),
        scope=scope or value.get('peak_scope', 'Scope not recorded in this row; see phase training_metrics.json'),
        measured_seconds=value.get('seconds'), update=value.get('update'), request_id=value.get('request_id'))


def worker_work(inp, phase, identity, kind):
    records, times, events = [], [], []
    work = dict(completed_rollouts=0, sampling_forward_returns=0, validation_forward_returns=0,
                completed_D0_calls=0, training_update_attempts=0, completed_training_updates=0, training_forward_returns=0)
    metrics = {}
    if kind == 'training':
        event_path = phase / 'events.jsonl'
        events = inp.events(event_path) if event_path.exists() else []
        counts = Counter(e['event'] for e in events)
        work.update(training_update_attempts=counts['UPDATE_START'], completed_training_updates=counts['UPDATE_COMPLETE'], training_forward_returns=counts['FORWARD_RETURN'])
        metric_path = phase / 'training_metrics.json'
        metrics = inp.read(metric_path) if metric_path.exists() else {}
        scope = metrics.get('peak_scope', 'Training update CUDA counters; reset before this round loop according to execution source. Full process memory is separate.')
        for e in events:
            if e['event'] == 'UPDATE_COMPLETE':
                records.append(cuda_record(e, event_path, scope)); times.append(e['seconds'])
    elif kind == 'sampling':
        stage = phase.parent
        suffix = phase.name[len('sampling_worker'):]
        event_path = stage / ('sampling_events' + suffix + '.jsonl')
        events = inp.events(event_path) if event_path.exists() else []
        counts = Counter(e['event'] for e in events)
        work.update(completed_rollouts=counts['ROLLOUT_COMPLETE'], sampling_forward_returns=counts['FORWARD_RETURN'], validation_forward_returns=counts['VALIDATION_FORWARD_RETURN'])
        for e in events:
            if e['event'] != 'ROLLOUT_COMPLETE': continue
            path = stage / 'raw' / e['context']['request_id'] / 'receipt.json'
            if path.exists():
                value = inp.read(path)['sampling']; records.append(cuda_record(dict(value, request_id=e['context']['request_id']), path)); times.append(value['seconds'])
    elif kind == 'D0':
        completion = phase / 'batch/completion.json'
        if completion.exists():
            metrics = inp.read(completion)
            paths = [D.resolve(r['proposal_receipt']) for r in metrics['receipts']]
        elif phase.name == 'denoise_all_worker':
            paths = sorted((D.OUT / 'evaluation').glob('*/proposal/*/receipt.json'))
        else:
            paths = sorted((phase.parent / 'proposal').glob('*/receipt.json'))
        for path in paths:
            value = inp.read(path); records.append(cuda_record(value, path)); times.append(value['seconds'])
            work['completed_D0_calls'] += value['model_forwards']
    return dict(counts=work, operation_seconds_sum=sum(times), operation_seconds_min=minimum(times), operation_seconds_max=maximum(times),
        CUDA_allocated_peak_bytes=maximum(r['allocated_bytes'] for r in records), CUDA_reserved_peak_bytes=maximum(r['reserved_bytes'] for r in records),
        CUDA_counter_records=len(records), CUDA_counter_scopes=sorted({r['scope'] for r in records}), CUDA_records=records,
        training_loop_seconds=metrics.get('training_seconds'), D0_batch_seconds=metrics.get('seconds') if kind == 'D0' else None,
        D0_model_load_seconds=metrics.get('load_seconds') if kind == 'D0' else None,
        last_event_utc=events[-1].get('utc') if events else None)


def phase_row(inp, attempt_path):
    phase = attempt_path.parent; attempt = inp.read(attempt_path)
    identity = attempt['identity']; row = dict(worker_id=str(phase.relative_to(D.OUT)), phase_path=str(phase), identity=identity, **classify(phase, identity))
    points = [point(attempt['resources'], attempt_path, 0, 'PHASE_ATTEMPT')]
    resources_path = phase / 'resources.json'
    if resources_path.exists():
        resources = inp.read(resources_path)
        points = [point(resources['initial'], resources_path, 0, 'PHASE_INITIAL')]
        points += [point(v, resources_path, i + 1, 'PHASE_POLL') for i, v in enumerate(resources['samples']) if 'gpu' in v]
        points.append(point(resources['final'], resources_path, len(resources['samples']) + 1, 'PHASE_FINAL'))
        row['monitor_scope'] = resources.get('scope')
        row['monitor_errors'] = [v for v in resources['samples'] if 'monitor_error' in v]
    else:
        row.update(monitor_scope='NO_SAVED_PHASE_SERIES', monitor_errors=[])
    terminal = next((p for p in (phase / 'completion.json', phase / 'failure.json') if p.exists()), None)
    done = inp.read(terminal) if terminal else {}
    recovery_path = phase / 'completion_recovery.json'
    recovered = inp.read(recovery_path) if recovery_path.exists() else None
    work = worker_work(inp, phase, identity, row['kind'])
    start = attempt['resources'].get('utc')
    end = done.get('utc') or work['last_event_utc'] or points[-1]['utc']
    row.update(phase_status=done.get('status', 'OPEN_OR_INTERRUPTED'), completion_recovered=bool(recovered),
        start_utc=start, end_utc=end, start=stamp(start), end=stamp(end), phase_wall_seconds=done.get('seconds'),
        interval_scope='PHASE_ENTRY_TO_EXIT' if terminal else 'OBSERVED_PREFIX_ONLY_END_IS_LAST_RECORDED_ACTIVITY',
        owned_pid=attempt['resources'].get('pid'), source_paths=[str(attempt_path)] + ([str(resources_path)] if resources_path.exists() else []) + ([str(terminal)] if terminal else []),
        resource_profile=profile(points, attempt['resources'].get('pid')), work=work, dispatch_ids=[])
    if phase.name == 'train_SELF_FM' and identity.get('round_index') == 0 and (D.OUT / 'engineering_recovery_round00/authorization.json').exists():
        row['disposition'] = 'DISCARDED_ROUND00_SELF_INTERRUPTION; original monitor series unavailable'
    else:
        row['disposition'] = 'DURABLE_COMPLETION_RECOVERED_WITHOUT_REPLAY' if recovered else 'NORMAL_OR_ACTIVE'
    row['coverage'] = ('MIDPHASE_POLLS_PRESENT' if row['resource_profile']['recorded_midphase_snapshots'] else 'NO_MIDPHASE_POLLS_SAVED')
    return row, points


def arg(args, key, default=None):
    return args[args.index(key) + 1] if key in args else default


def dispatch_target(command):
    module, args = command[0], command[1:]
    ri = arg(args, '--round'); label = arg(args, '--evaluation')
    stage = D.OUT / (f'training/round_{int(ri):02d}' if ri is not None else 'evaluation/' + str(label))
    if module.endswith('sample_worker'):
        if '--join' in args: return stage / 'sampling_join'
        workers, worker = int(arg(args, '--workers', '1')), int(arg(args, '--worker', '0'))
        return stage / ('sampling_worker' if workers == 1 else f'sampling_worker_{worker:02d}_of_{workers:02d}')
    if module.endswith('train_worker'):
        return stage / ('train_' + arg(args, '--arm') + ('_recovery01' if '--recovery-round0-self' in args else ''))
    if module.endswith('denoise_worker'): return stage / 'denoise_worker'
    if module.endswith('final_run') and arg(args, '--stage') == 'denoise-worker': return D.OUT / 'evaluation/denoise_all_worker'
    if module.endswith('final_cpu'): return D.OUT / 'evaluation/cpu'
    if module.endswith('cpu_worker'): return stage / 'cpu'
    if module.endswith('prepare_rewards'): return stage / 'rewards_CPU'
    return None


def dispatch_rows(inp, rows):
    result, observations = [], []
    by_path = {r['phase_path']: r for r in rows}
    for path in sorted((D.OUT / 'dispatch').glob('*/admission.json')):
        admitted = inp.read(path); target = dispatch_target(admitted['command']); exit_path = path.parent / 'exit.json'
        ended = inp.read(exit_path) if exit_path.exists() else {}
        points = [point(admitted['resources'], path, 0, 'DISPATCH_ADMISSION')]
        if ended.get('resources'): points.append(point(ended['resources'], exit_path, 1, 'DISPATCH_COLLECTION'))
        observations.extend(points)
        row = dict(dispatch_id=path.parent.name, command=admitted['command'], target=str(target) if target else None,
            phase_worker_id=by_path[str(target)]['worker_id'] if str(target) in by_path else None,
            status='COLLECTED' if ended else 'NO_EXIT_RECORD', returncode=ended.get('returncode'),
            launch_observation_utc=points[0]['utc'], collection_observation_utc=points[-1]['utc'] if ended else None,
            launch_to_parent_collection_seconds=ended.get('seconds'),
            wall_scope='Includes possible parent wait after child finished; use Phase wall or CPU completion wall when available',
            admission_decision=admitted.get('decision', admitted.get('reason')), resource_profile=profile(points),
            source_paths=[str(path)] + ([str(exit_path)] if ended else []))
        if str(target) in by_path:
            by_path[str(target)]['dispatch_ids'].append(row['dispatch_id'])
        result.append(row)
    return result, observations


def cpu_rows(inp, dispatch):
    rows = []
    targets = {r['target']: r for r in dispatch if r['target']}
    paths = list((D.OUT / 'training').glob('round_*/cpu/post_score_render_complete.json'))
    final = D.OUT / 'evaluation/cpu/post_score_render_complete.json'
    if final.exists(): paths.append(final)
    for path in sorted(paths):
        value = inp.read(path); phase = path.parent; owner = targets.get(str(phase))
        row = dict(worker_id=str(phase.relative_to(D.OUT)), phase_path=str(phase), identity=dict(stage='CPU_post_score_render'),
            **classify(phase, {}), phase_status=value['status'], disposition='CPU_COMPLETION_RECEIPT',
            phase_wall_seconds=value['seconds'], start_utc=owner['launch_observation_utc'] if owner else None,
            end_utc=owner['collection_observation_utc'] if owner else None,
            start=stamp(owner['launch_observation_utc']) if owner else None,
            end=stamp(owner['collection_observation_utc']) if owner else None,
            interval_scope='DISPATCH_BOUND_CONTAINS_PARENT_COLLECTION_DELAY' if owner else 'NO_RECORDED_UTC_INTERVAL',
            CPU_pool_workers_recorded=value['workers'], CPU_threads_per_pool_worker=value['CPU_threads_per_worker'],
            actual_CPU_pool_process_peak=None, actual_CPU_pool_process_peak_status='NA_NO_CONTINUOUS_OWNED_POOL_PID_SERIES',
            requests=value['requests'], requests_per_phase_wall_second=value['requests'] / value['seconds'],
            model_calls=value['model_calls'], source_paths=[str(path)], resource_profile=owner['resource_profile'] if owner else profile([]),
            coverage='ADMISSION_AND_COLLECTION_ONLY' if owner else 'WALL_AND_POOL_SIZE_RECEIPT_ONLY',
            dispatch_ids=[owner['dispatch_id']] if owner else [])
        rows.append(row)
    return rows


def overlap(rows):
    events = []
    for row in rows:
        if row.get('start') is None or row.get('end') is None or row['end'] <= row['start']: continue
        events.extend([(row['start'], 1, row['worker_id']), (row['end'], -1, row['worker_id'])])
    events.sort(key=lambda e: (e[0], e[1]))
    active, segments, last = set(), [], None
    for when, delta, worker in events:
        if last is not None and when > last and active:
            segments.append(dict(start=last, end=when, seconds=when-last, concurrent_processes=len(active), worker_ids=sorted(active)))
        if delta < 0: active.discard(worker)
        else: active.add(worker)
        last = when
    return dict(peak_concurrent_processes=max((r['concurrent_processes'] for r in segments), default=0),
        observed_active_union_seconds=sum(r['seconds'] for r in segments),
        observed_overlap_seconds=sum(r['seconds'] for r in segments if r['concurrent_processes'] > 1), segments=segments,
        scope='Overlapping recorded worker lifetime intervals, not proof of simultaneous GPU kernels; incomplete workers contribute only their observed prefix')


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row['group']].append(row)
    result = []
    for group, workers in sorted(grouped.items()):
        concurrency = overlap(workers)
        counts = Counter()
        for worker in workers: counts.update(worker.get('work', {}).get('counts', {}))
        walls = [r['phase_wall_seconds'] for r in workers if r.get('phase_wall_seconds') is not None]
        result.append(dict(group=group, worker_records=len(workers), worker_ids=[r['worker_id'] for r in workers],
            recorded_phase_wall_sum_seconds=sum(walls), completed_or_failed_phase_clocks=len(walls),
            peak_observed_concurrent_workers=concurrency['peak_concurrent_processes'], observed_active_union_seconds=concurrency['observed_active_union_seconds'],
            observed_overlap_seconds=concurrency['observed_overlap_seconds'], counts=dict(counts),
            aggregate_counts_per_observed_union_second={k: v/concurrency['observed_active_union_seconds'] for k, v in counts.items()} if concurrency['observed_active_union_seconds'] else {},
            throughput_scope='Measured recorded concurrency; never an isolated-worker benchmark. CPU pool workers are separate declared receipt fields.'))
    return result


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for key, v in row.items()})
    payload = stream.getvalue().encode('utf-8-sig'); path = D.checked(path)
    if path.exists():
        if path.read_bytes() != payload: raise ValueError('Immutable CSV changed')
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)


def run():
    inp = Inputs(); inp.read_bytes(__file__)
    rows, points = [], []
    for path in sorted(D.OUT.glob('**/attempt.json')):
        value = inp.read(path)
        if 'identity' not in value or 'resources' not in value: continue
        result = inp.safe(path, lambda p=path: phase_row(inp, p))
        if result:
            row, observations = result; rows.append(row); points.extend(observations)
    dispatch, admission_points = dispatch_rows(inp, rows)
    points.extend(admission_points)
    rows.extend(cpu_rows(inp, dispatch))
    plans = []
    for path in sorted((D.OUT / 'training').glob('round_*/*_dispatch.json')):
        v = inp.read(path); plans.append(dict(path=str(path), **{k: x for k, x in v.items() if k != 'resources'}))
        if v.get('resources'): points.append(point(v['resources'], path, 0, 'INCREMENTAL_ADMISSION'))
    final_events = D.OUT / 'evaluation/final_controller/events.jsonl'
    if final_events.exists():
        for i, event in enumerate(inp.events(final_events)):
            if event.get('resources'): points.append(point(event['resources'], final_events, i, 'FINAL_CONTROLLER_POLL'))
    gpu_rows = [r for r in rows if r['GPU_worker']]
    concurrency = overlap(gpu_rows)
    for row in gpu_rows:
        segments = [s for s in concurrency['segments'] if row['worker_id'] in s['worker_ids']]
        row['peak_concurrent_owned_GPU_workers_during_interval'] = max((s['concurrent_processes'] for s in segments), default=0)
        row['seconds_overlapping_another_owned_GPU_worker'] = sum(s['seconds'] for s in segments if s['concurrent_processes'] > 1)
        row['overlapping_owned_workers'] = sorted({w for s in segments for w in s['worker_ids'] if w != row['worker_id']})
        wall = row['phase_wall_seconds']
        row['counts_per_phase_wall_second'] = {k: v/wall for k, v in row['work']['counts'].items()} if wall else None
        row['throughput_scope'] = 'Phase clock includes load/save; overlapping process intervals are disclosed; not isolated throughput'
    complete = {}
    for arm in ('NFT', 'SELF_FM'):
        path = D.OUT / f'checkpoints/{arm}/round15.json'
        complete[arm] = bool(path.exists() and inp.read(path).get('step') == 256 and inp.read(path).get('round_index') == 15)
    cpu_done = D.OUT / 'evaluation/cpu/post_score_render_complete.json'
    d0_done = D.OUT / 'evaluation/denoise_all_worker/completion.json'
    complete['final_CPU'] = bool(cpu_done.exists() and inp.read(cpu_done).get('status') == 'COMPLETE')
    complete['final_D0'] = bool(d0_done.exists() and inp.read(d0_done).get('status') == 'COMPLETE')
    inp.finish()
    status = 'FAIL' if inp.errors else 'COMPLETE' if all(complete.values()) and not inp.notes else 'PARTIAL'
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(dict(version=VERSION, sources=inp.sources), sort_keys=True).encode()).hexdigest()[:20]
    out = D.OUT / 'resource_summary/snapshots' / digest
    groups = group_rows(rows)
    by_group = {g['group']: g for g in groups}
    for plan in plans:
        path = Path(plan['path'])
        group = path.parent.name + '/' + path.name.replace('_dispatch.json', '')
        plan['observed_group'] = group
        plan['peak_observed_concurrent_workers'] = by_group.get(group, {}).get('peak_observed_concurrent_workers')
        plan['worker_records'] = by_group.get(group, {}).get('worker_records')
    gaps = [dict(worker_id=r['worker_id'], coverage=r['coverage'], interval_scope=r['interval_scope'],
                 note='Unavailable observations are retained as NA; no imputation') for r in rows
            if r['coverage'] != 'MIDPHASE_POLLS_PRESENT' or r['interval_scope'] != 'PHASE_ENTRY_TO_EXIT']
    records = dict(workers=rows, groups=groups, dispatch=dispatch, incremental_admission_plans=plans,
                   observations=points, GPU_concurrency=concurrency, input_sources=inp.sources)
    for name, value in records.items(): D.write(out / (name + '.json'), value)
    write_csv(out / 'workers.csv', [{k: v for k, v in row.items() if k != 'work'} | {k: v for k, v in row.get('work', {}).items() if k != 'CUDA_records'} for row in rows])
    write_csv(out / 'groups.csv', groups)
    summary = dict(version=VERSION, status=status, code_sha256=code_sha, input_digest=digest, output_dir=str(out),
        completion_markers=complete, worker_records=len(rows), GPU_worker_records=len(gpu_rows), CPU_stage_records=len(rows)-len(gpu_rows),
        dispatch_records=len(dispatch), recorded_resource_snapshots=len(points),
        global_recorded_resources=profile(points), peak_observed_concurrent_owned_GPU_workers=concurrency['peak_concurrent_processes'],
        owned_GPU_active_union_seconds=concurrency['observed_active_union_seconds'], owned_GPU_overlap_seconds=concurrency['observed_overlap_seconds'],
        coverage_status='DOCUMENTED_GAPS_PRESENT' if gaps else 'SAVED_PHASE_MONITORING_PRESENT', coverage_gaps=gaps,
        source_read_notes=inp.notes, errors=inp.errors,
        limitations=[
            'Whole-device memory/utilization includes unrelated processes. Sampled extrema may miss short spikes.',
            'WDDM per-process N/A stays null. No value is inferred from total device deltas, weight size or CUDA allocation counters.',
            'CUDA allocated/reserved counters are reported with their actual reset scope and never relabelled full-process memory.',
            'Concurrent process lifetime intervals do not establish concurrent GPU kernels; aggregate throughput is specific to measured contention.',
            'Dispatch collection time may follow child completion; precise Phase wall clocks or CPU completion clocks take precedence.',
            'CPU pool worker count is the completed-stage receipt setting, not an observed pool-process peak.',
            'Original interrupted SELF round0 may have no saved monitor series; its work/CUDA counters and subsequent recovery are separate workers.',
            'Early records may lack CPU load fields; unavailable historical observations cannot be reconstructed.'],
        new_hardware_queries=0, weights_opened=0, model_forwards=0, torch_imported='torch' in sys.modules,
        files={str(out / name): D.sha(out / name) for name in [*(name + '.json' for name in records), 'workers.csv', 'groups.csv']})
    D.write(out / 'summary.json', summary)
    latest = D.checked(D.OUT / 'resource_summary/latest.json'); latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(status=status, summary_path=str(out / 'summary.json'), summary_sha256=D.sha(out / 'summary.json'),
        output_dir=str(out), snapshot_path=str(out), code_sha256=code_sha), indent=2, sort_keys=True) + '\n'); tmp.replace(latest)
    print(json.dumps(dict(status=status, output=str(out), workers=len(rows), GPU_peak_concurrency=concurrency['peak_concurrent_processes'],
        resource_snapshots=len(points), coverage_gaps=len(gaps), errors=len(inp.errors)), sort_keys=True))
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
