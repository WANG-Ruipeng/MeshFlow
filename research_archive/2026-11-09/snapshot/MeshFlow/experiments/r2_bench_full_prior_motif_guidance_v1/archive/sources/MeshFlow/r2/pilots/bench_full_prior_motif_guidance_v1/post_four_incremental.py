"""Two workers per safely released old parity; at most four CPU workers.

This wrapper never signals a process, retries a partial cascade, or changes a
scientific function. Planning/launching are explicit independent commands.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from . import post_handoff_four as H
from . import prepare as P, evaluate as E
from .run import full_resources, observe_child, write

VERSION = 'incremental_safe_parity_two_plus_two_CPU_v1'
MODULE = 'r2.pilots.bench_full_prior_motif_guidance_v1.post_four_incremental'
DEST = P.OUT/'workflow/post_four_incremental'
HANDOFF = H.DEST
EXPECTED_DELEGATE_SHA = '3e038eec473f8bdf33485b918f67ac16d1c7ba9307fbb9479afd8e833341a9f5'


def group(old_partition):
    if old_partition not in (0, 1):
        raise ValueError('Old partition must be zero or one')
    return DEST/('even' if old_partition == 0 else 'odd')


def selected():
    tasks = P.selected_tasks()
    if len(tasks) != 184 or len({t['cell_id'] for t in tasks}) != 184:
        raise ValueError('Selected184 identity changed')
    if any(sum(t['arm'] == arm for t in tasks) != 46 for arm in ('G0', 'G1', 'G2', 'G3')):
        raise ValueError('Selected arm counts changed')
    return tasks


def old_safe(old_partition):
    if P.sha(H.__file__) != EXPECTED_DELEGATE_SHA:
        raise ValueError('Frozen delegated orchestration changed')
    registration = P.read(HANDOFF/'handoff_registration.json')
    if registration['source_sha256'] != EXPECTED_DELEGATE_SHA:
        raise ValueError('Original handoff registration source differs')
    expected = registration['old_processes'][str(old_partition)]
    receipt = P.read(HANDOFF/f'handoff_{old_partition}.json')
    if (receipt.get('status') not in ('SAFE_BOUNDARY_HANDOFF', 'ALREADY_EXITED_SAFE') or
            receipt.get('partition') != old_partition or receipt.get('process') != expected or
            receipt.get('incomplete')):
        raise ValueError('This old parity lacks a safe handoff receipt')
    if H.process_identity(expected) is not None:
        raise RuntimeError('This parity old worker is still active; no launch')
    return registration, receipt


def checked_plan(old_partition):
    folder = group(old_partition)
    p = P.read(folder/'plan.json')
    if (p['source_sha256'] != EXPECTED_DELEGATE_SHA or
            p['wrapper_sha256'] != P.sha(__file__) or
            p['tasks_sha256'] != P.sha(P.TASKS) or
            p['old_partition'] != old_partition):
        raise ValueError('Plan/source identity changed')
    H.verify(p['protected'])
    tasks = selected()
    owned = [t['cell_id'] for i, t in enumerate(tasks) if i % 2 == old_partition]
    if owned != p['owned_task_ids'] or len(owned) != 92:
        raise ValueError('Old parity ownership changed')
    expected_workers = (0, 1) if old_partition == 0 else (2, 3)
    if set(p['partitions']) != {'0', '1', '2', '3'}:
        raise ValueError('Four-key worker map required')
    ids = []
    for part in range(4):
        jobs = p['partitions'][str(part)]
        if part not in expected_workers and jobs:
            raise ValueError('Cross-parity worker assignment')
        if P.read(folder/f'jobs_{part}.json')['task_ids'] != jobs:
            raise ValueError('Worker joblist changed')
        ids.extend(jobs)
    done = [r['cell_id'] for r in p['already_completed']]
    if len(ids + done) != 92 or len(set(ids + done)) != 92 or set(ids + done) != set(owned):
        raise ValueError('Ownership overlap or missing owned task')
    return p


def plan(old_partition):
    old_safe(old_partition)
    folder = group(old_partition)
    if (folder/'plan.json').exists():
        return checked_plan(old_partition)
    if folder.exists():
        raise RuntimeError('Partial group plan retained; no implicit retry')
    if not (P.OUT/'evaluation_code.json').is_file():
        raise RuntimeError('Existing frozen evaluation code is required')
    E.freeze_evaluation_code()
    saved, stats = H.barrier_cache()
    tasks = selected()
    owned = [t for i, t in enumerate(tasks) if i % 2 == old_partition]
    if len(owned) != 92:
        raise ValueError('Expected exactly92 owned requests')
    done, pending = [], []
    proxy = E._post_proxy()
    for task in owned:
        result = H.completed(task)
        if result:
            done.append(result)
        else:
            # No new worker may fail merely because D0 has not reached this task.
            proxy._checked_post(task, P.TASKS)
            pending.append(task)
    pending.sort(key=lambda t: (-int(t['q']['N']), t['cell_id']))
    workers = (0, 1) if old_partition == 0 else (2, 3)
    partitions = {str(i): [] for i in range(4)}
    for offset, part in enumerate(workers):
        partitions[str(part)] = [t['cell_id'] for t in pending[offset::2]]
    protected_paths = [HANDOFF/'handoff_registration.json', HANDOFF/f'handoff_{old_partition}.json',
                       P.TASKS, P.OUT/'lock.json', P.OUT/'evaluation_code.json', H.SOURCE_BARRIER]
    protected = {str(p): P.sha(p) for p in protected_paths}
    result = dict(status='FROZEN', version=VERSION, utc=H.utc(), old_partition=old_partition,
        source_sha256=EXPECTED_DELEGATE_SHA, wrapper_sha256=P.sha(__file__),
        tasks_sha256=P.sha(P.TASKS), owned_task_ids=[t['cell_id'] for t in owned],
        owned_outputs=92, selected_tasks=184, already_completed=done, pending_tasks=len(pending),
        worker_count=2, global_worker_limit=4, partitions=partitions,
        partition_policy='Frozen original parity; descending N then cell ID; round robin two workers',
        protected=protected, barrier_source=str(H.SOURCE_BARRIER),
        initial_403_full_hashes_verified=True, barrier_stats=stats, resources=full_resources(),
        evaluator_sha256=P.sha(E.__file__), prepare_sha256=P.sha(P.__file__),
        model_calls=0, scoring_replays=0, scientific_functions_modified=False)
    for part in range(4):
        write(folder/f'jobs_{part}.json', dict(partition=part, task_ids=partitions[str(part)]))
    write(folder/'plan.json', result)
    return result


def live_identity(receipt):
    import psutil
    try:
        proc = psutil.Process(receipt['pid'])
    except psutil.NoSuchProcess:
        return None
    if proc.status() == psutil.STATUS_ZOMBIE:
        return None
    if proc.create_time() != receipt['create_time'] or proc.cmdline() != receipt['command']:
        raise ValueError('Recorded process identity changed')
    return proc


def active_workers():
    registration = P.read(HANDOFF/'handoff_registration.json')
    active = []
    for part, expected in registration['old_processes'].items():
        proc = H.process_identity(expected)
        if proc is not None:
            active.append(dict(kind='old', partition=int(part), pid=proc.pid,
                               rss_bytes=proc.memory_info().rss))
    for part in range(4):
        folder = group(part // 2)
        entry = folder/f'worker_entry_{part}.json'
        launched = folder/f'launch_{part}.json'
        worker_proc = live_identity(P.read(entry)) if entry.exists() else None
        observer_proc = live_identity(P.read(launched)['observer']) if launched.exists() else None
        if worker_proc is not None or observer_proc is not None:
            active.append(dict(kind='new', partition=part,
                pid=worker_proc.pid if worker_proc is not None else None,
                observer_pid=observer_proc.pid if observer_proc is not None else None,
                rss_bytes=worker_proc.memory_info().rss if worker_proc is not None else None))
    return active


def launch(partition):
    import psutil
    old_partition = partition // 2
    old_safe(old_partition)
    p = checked_plan(old_partition)
    folder = group(old_partition)
    if (folder/f'launch_{partition}.json').exists() or (folder/f'launch_claim_{partition}.json').exists():
        raise RuntimeError('Existing launch attempt; no implicit retry')
    if not p['partitions'][str(partition)]:
        raise RuntimeError('Empty partition: no process is needed')
    previous_rss = None
    if partition % 2:
        previous = folder/f'worker_{partition-1}'/'started.json'
        if not previous.is_file():
            raise RuntimeError('Previous same-group worker must complete actual initialization first')
        entry = P.read(folder/f'worker_entry_{partition-1}.json')
        previous_proc = live_identity(entry)
        if previous_proc is not None:
            previous_rss = previous_proc.memory_info().rss
        elif not (folder/f'worker_{partition-1}'/'complete.json').is_file():
            raise RuntimeError('Previous worker exited incomplete; halt further dispatch')
        else:
            previous_rss = P.read(folder/f'worker_{partition-1}'/'complete.json')['peak_RSS_KiB'] * 1024
    active = active_workers()
    if len(active) >= 4:
        raise RuntimeError('Four active experiment CPU workers already present')
    snapshot = full_resources()
    ram = psutil.virtual_memory().available
    cpu = psutil.cpu_percent(interval=1)
    measured = [a['rss_bytes'] for a in active if a.get('rss_bytes')]
    if previous_rss:
        measured.append(previous_rss)
    observed = max(measured, default=256 << 20)
    minimum_ram = max(8 * 1024**3, int(4 * 1024**3 + 1.2 * observed))
    if ram < minimum_ram or (psutil.cpu_count() or 0) < 8 or cpu > 85:
        raise RuntimeError('Measured CPU/RAM headroom insufficient; no launch')
    # One exclusive claim prevents a duplicate launch; failures require explicit recovery.
    write(folder/f'launch_claim_{partition}.json', dict(utc=H.utc(), partition=partition,
          wrapper_sha256=P.sha(__file__), automatic_retries=0))
    command = [sys.executable, '-B', '-u', '-m', MODULE, 'observe', '--partition', str(partition)]
    with (folder/f'observer_{partition}.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=P.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    observed_proc = psutil.Process(proc.pid)
    result = dict(status='LAUNCHED', utc=H.utc(), partition=partition, old_partition=old_partition,
        wrapper_sha256=P.sha(__file__), delegated_source_sha256=EXPECTED_DELEGATE_SHA,
        observer=dict(pid=proc.pid, create_time=observed_proc.create_time(), command=observed_proc.cmdline()),
        resources=snapshot, available_RAM_bytes=ram, minimum_RAM_bytes=minimum_ram,
        host_CPU_percent=cpu, previous_worker_RSS_bytes=previous_rss,
        observed_comparable_RSS_bytes=observed, active_workers_before=active,
        resulting_maximum_active_workers=len(active)+1, threads_per_worker=1,
        incremental=True, model_calls=0, scoring_replays=0)
    write(folder/f'launch_{partition}.json', result)
    return result


def observe(partition):
    checked_plan(partition // 2)
    old_safe(partition // 2)
    return observe_child([sys.executable, '-B', '-u', '-m', MODULE, 'worker', '--partition', str(partition)],
        'CPU_POST_INCREMENTAL', f'partition_{partition}', group(partition // 2)/'plan.json')


def worker(partition):
    import psutil
    checked_plan(partition // 2)
    old_safe(partition // 2)
    folder = group(partition // 2)
    proc = psutil.Process(os.getpid())
    write(folder/f'worker_entry_{partition}.json', dict(pid=proc.pid, create_time=proc.create_time(),
        command=proc.cmdline(), utc=H.utc(), wrapper_sha256=P.sha(__file__),
        delegated_source_sha256=EXPECTED_DELEGATE_SHA, partition=partition))
    # Only this process sees the redirected evidence root. H.worker and E.post stay verbatim.
    H.DEST = folder
    return H.worker(partition)


def finish():
    path = DEST/'complete.json'
    if path.exists():
        saved = P.read(path)
        if saved['wrapper_sha256'] != P.sha(__file__):
            raise ValueError('Completed wrapper changed')
        H.verify(saved['files'])
        return saved
    plans = [checked_plan(part) for part in (0, 1)]
    for part in (0, 1):
        old_safe(part)
    rows, launches = [], []
    for part in range(4):
        folder = group(part // 2)
        ids = plans[part // 2]['partitions'][str(part)]
        if not ids:
            continue
        receipt = P.read(folder/f'worker_{part}'/'complete.json')
        if (receipt.get('status') != 'COMPLETE' or receipt['outputs'] != len(ids) or
                [r['cell_id'] for r in receipt['results']] != ids):
            raise ValueError('Worker has not completed its exact task list')
        rows.extend(receipt['results'])
        launches.append(P.read(folder/f'launch_{part}.json'))
    completed = [H.completed(t) for t in selected()]
    if any(r is None for r in completed):
        raise ValueError('Missing completed CPU output')
    if len(rows) + sum(len(p['already_completed']) for p in plans) != 184:
        raise ValueError('Final completed denominator differs')
    audit_path = DEST/'final_global_full_RAW_verification.json'
    if audit_path.exists():
        audit = P.read(audit_path)
        if audit['wrapper_sha256'] != P.sha(__file__):
            raise ValueError('Final verification source changed')
        H.guard(audit['stats'])
    else:
        actual = P.verify_raw_barrier()
        if actual[1] != P.read(H.SOURCE_BARRIER):
            raise ValueError('Final full RAW verification differs')
        audit = dict(status='PASS', utc=H.utc(), wrapper_sha256=P.sha(__file__),
                     barrier=actual[1], stats=H.statbook(actual[1]['files']))
        write(audit_path, audit)
    files = {r['path']: r['sha256'] for r in completed}
    for part in (0, 1):
        files[str(group(part)/'plan.json')] = P.sha(group(part)/'plan.json')
    files[str(audit_path)] = P.sha(audit_path)
    start = min((r['utc'] for r in launches), default=H.utc())
    wall = (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(start)).total_seconds()
    result = dict(status='COMPLETE', utc=H.utc(), version=VERSION, wrapper_sha256=P.sha(__file__),
        delegated_source_sha256=EXPECTED_DELEGATE_SHA, outputs=184, previous_completed=184-len(rows),
        newly_completed=len(rows), launched_workers=len(launches), maximum_allowed_workers=4,
        elapsed_since_first_launch_seconds=wall, summed_per_task_seconds=sum(r['seconds'] for r in rows),
        summed_CPU_seconds=sum(r['cpu_seconds'] for r in rows),
        final_global_full_RAW_verification=True, model_calls=0, scoring_replays=0,
        resources=full_resources(), files=files,
        timing_note='Incremental concurrent task-mix throughput; not an isolated compute-speed benchmark.')
    write(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'launch', 'observe', 'worker', 'finish'))
    parser.add_argument('--old-partition', type=int, choices=(0, 1))
    parser.add_argument('--partition', type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == 'plan':
        if args.old_partition is None:
            parser.error('plan requires --old-partition')
        result = plan(args.old_partition)
    elif args.action == 'finish':
        result = finish()
    else:
        if args.partition is None:
            parser.error('launch/observe/worker requires --partition')
        result = globals()[args.action](args.partition)
    if result is not None:
        print(json.dumps({k: v for k, v in result.items() if k not in
            ('files', 'resources', 'partitions', 'already_completed', 'barrier_stats', 'protected')},
            ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
