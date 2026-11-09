"""Direct CPU-only tests of the production metadata reader and failure drain."""
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from r2.pilots.bench_parts_convergence_v1 import controller as C
from r2.pilots.bench_parts_convergence_v1 import run_all as R


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    os.replace(temporary, path)


def child(folder, mode):
    folder = Path(folder)
    stopped = {'count': 0}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(count=stopped['count']+1))
    save(folder/'ready.json', {'pid': os.getpid(), 'mode': mode})
    if mode == 'training':
        while not stopped['count']: time.sleep(.005)
        time.sleep(.18)
        cp = folder/'rolling.pt'
        cp.write_bytes(b'CPU mock checkpoint; no model')
        sha = hashlib.sha256(cp.read_bytes()).hexdigest()
        save(folder/'rolling.json', {'step': 1113, 'sha256': sha})
        save(folder/'status.json', {'status': 'INTERRUPTED_SAFE', 'arm': 'COMMON',
              'step': 1113, 'pid': os.getpid(), 'checkpoint': str(cp), 'signals': stopped['count']})
        time.sleep(.15)  # Must wait for real exit, not merely observe SAFE status.
    else:
        time.sleep(.55)
        save(folder/'natural_complete.json', {'signals': stopped['count']})


def run():
    out = Path(__file__).resolve().parent/('production_'+str(time.time_ns()))
    out.mkdir()
    root = out/'fixture';folder = root/'training/COMMON';folder.mkdir(parents=True)
    source_before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (C.__file__, R.__file__)}
    checks = [];durations = {}
    status = folder/'status.json'
    timer = threading.Timer(.06, lambda: save(status, {'arm': 'COMMON', 'status': 'TRAINING', 'step': 10}))
    timer.start();payload = C.read(status, timeout=.5);timer.join()
    assert payload['step'] == 10
    checks.append('production_read_recovers_FileNotFound')
    status.write_text('{', encoding='utf-8')
    timer = threading.Timer(.06, lambda: save(status, {'arm': 'COMMON', 'status': 'TRAINING', 'step': 10}))
    timer.start();payload = R.read(status);timer.join()
    assert payload['step'] == 10
    checks.append('run_all_delegated_read_recovers_JSONDecode')
    with patch.object(Path, 'read_text', side_effect=[PermissionError(errno.EACCES, 'simulated transient'), '{}']) as reader:
        assert C.read(status, timeout=.2) == {} and reader.call_count == 2
    checks.append('production_read_recovers_allowed_PermissionError')
    with patch.object(Path, 'read_text', side_effect=OSError(errno.EIO, 'not transient')) as reader:
        try: C.read(status, timeout=.2)
        except OSError as exc: assert exc.errno == errno.EIO and reader.call_count == 1
        else: raise AssertionError('EIO hidden')
    checks.append('production_read_does_not_retry_EIO')
    status.write_text('{', encoding='utf-8')
    start = time.monotonic()
    try: C.read(status, timeout=.04)
    except json.JSONDecodeError: pass
    else: raise AssertionError('Malformed persistent JSON hidden')
    durations['persistent_json_seconds'] = time.monotonic()-start
    assert .035 <= durations['persistent_json_seconds'] < .3
    checks.append('production_persistent_JSONDecode_fails_bounded')
    status.unlink();(folder/'training_log.jsonl').write_text('{"step":1112}\n')
    timer = threading.Timer(.06, lambda: save(status, {'arm': 'COMMON', 'status': 'TRAINING', 'step': 1112}))
    timer.start();precheck = C.recovery_precheck(root, 'COMMON');timer.join()
    assert precheck['state'] == 'TRAINING' and not precheck['safe_to_start']
    checks.append('production_recovery_waits_out_visibility_gap')
    status.unlink();start = time.monotonic();precheck = C.recovery_precheck(root, 'COMMON')
    durations['persistent_missing_seconds'] = time.monotonic()-start
    assert precheck['state'] == 'AMBIGUOUS_ORPHAN_ARTIFACTS' and not precheck['safe_to_start']
    assert 1.9 <= durations['persistent_missing_seconds'] < 2.5
    checks.append('production_persistent_missing_after_full_budget_is_unsafe')
    for payload in ({'arm': 'SEAM', 'status': 'TRAINING', 'step': 1112},
                    {'arm': 'COMMON', 'status': 'TRAINING', 'step': -1},
                    {'arm': 'COMMON', 'status': 'TRAINING'}):
        save(status, payload);start = time.monotonic()
        try: C.recovery_precheck(root, 'COMMON')
        except ValueError: assert time.monotonic()-start < .5
        else: raise AssertionError('Malformed schema accepted')
    checks.append('production_integrity_schema_errors_not_retried_or_treated_absent')

    driver = object.__new__(R.Driver)
    driver.root = root;driver.dest = root/'dispatch';driver.dest.mkdir()
    driver.events = driver.dest/'events.jsonl';driver.status_path = driver.dest/'status.json'
    driver.phase = 'CPU_FIXTURE';driver.started = time.time();driver.poll = .02
    driver.jobs = [];driver.measurements = {}
    status.unlink()
    timer = threading.Timer(.06, lambda: save(status, {'arm': 'COMMON', 'status': 'TRAINING', 'step': 10}))
    timer.start()
    job = {'kind': 'training', 'arm': 'COMMON', 'start_step': 0, 'peak_system_delta_mib': 0.,
           'peak_reserved_mib': 0., 'baseline': {'gpu_used_mib': 0.}}
    assert driver.progress_peak(job, {'gpu_used_mib': 0.}) is None
    timer.join();checks.append('production_progress_peak_survives_status_visibility_gap')
    status.unlink()
    for kind in ('training', 'sampling'):
        child_folder = folder if kind == 'training' else root/'sampling';child_folder.mkdir(exist_ok=True)
        job_folder = driver.dest/'jobs'/kind;job_folder.mkdir(parents=True)
        log = (job_folder/'stdout.log').open('wb')
        process = subprocess.Popen([sys.executable, '-B', __file__, '--child', str(child_folder), '--mode', kind],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        driver.jobs.append({'key': 'fixture_'+kind, 'kind': kind, 'arm': 'COMMON',
            'process': process, 'log': log, 'folder': job_folder, 'started': time.time(), 'finished': False,
            'peak_system_delta_mib': 0., 'peak_reserved_mib': 0.})
        C.read(child_folder/'ready.json', timeout=3)
    driver.launch = lambda *_: (_ for _ in ()).throw(AssertionError('No dispatch allowed while draining'))
    start = time.monotonic()
    failure = driver.drain_after_failure(RuntimeError('intentional CPU-fixture coordinator failure'))
    durations['production_drain_seconds'] = time.monotonic()-start
    assert all(j['process'].poll() == 0 and j['finished'] for j in driver.jobs)
    assert failure['drain_status'] == 'ALL_OWNED_CHILDREN_EXITED'
    train_state = C.read(folder/'status.json')
    natural_state = C.read(root/'sampling/natural_complete.json')
    assert train_state['status'] == 'INTERRUPTED_SAFE' and train_state['signals'] == 1
    assert natural_state['signals'] == 0
    assert all((j['folder']/'completion.json').exists() for j in driver.jobs)
    recovery = failure['child_outcomes'][0]['recovery_precheck']
    assert recovery['safe_to_start'] and recovery['step'] == 1113
    assert C.read(driver.status_path)['status'] == 'BLOCKED'
    checks.extend(['production_drain_sends_training_SIGTERM_once',
        'production_drain_waits_for_checkpoint_and_actual_child_exit',
        'production_drain_preserves_natural_sampling_completion',
        'production_drain_records_exit_receipts_and_safe_cursor',
        'production_drain_does_not_retry_or_dispatch_any_job'])
    # Repeat with production record writes/updates failing and an interrupted
    # wait. Child checkpoint writes run in independent stdlib processes.
    fault_root = out/'injected_fixture';fault_folder = fault_root/'training/COMMON'
    fault_folder.mkdir(parents=True)
    fault_driver = object.__new__(R.Driver)
    fault_driver.root = fault_root;fault_driver.dest = fault_root/'dispatch';fault_driver.dest.mkdir()
    fault_driver.events = fault_driver.dest/'events.jsonl';fault_driver.status_path = fault_driver.dest/'status.json'
    fault_driver.phase = 'FAULT_INJECTED_CPU_FIXTURE';fault_driver.started = time.time();fault_driver.poll = .02
    fault_driver.jobs = [];fault_driver.measurements = {}
    fault_driver.launch = lambda *_: (_ for _ in ()).throw(AssertionError('No dispatch while draining'))
    for kind in ('training', 'sampling'):
        child_folder = fault_folder if kind == 'training' else fault_root/'sampling';child_folder.mkdir(exist_ok=True)
        job_folder = fault_driver.dest/'jobs'/kind;job_folder.mkdir(parents=True)
        log = (job_folder/'stdout.log').open('wb')
        process = subprocess.Popen([sys.executable, '-B', __file__, '--child', str(child_folder), '--mode', kind],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        fault_driver.jobs.append({'key': 'fixture_'+kind, 'kind': kind, 'arm': 'COMMON',
            'process': process, 'log': log, 'folder': job_folder, 'started': time.time(), 'finished': False,
            'peak_system_delta_mib': 0., 'peak_reserved_mib': 0.})
        C.read(child_folder/'ready.json', timeout=3)
    true_sleep = time.sleep;sleep_calls = {'n': 0}
    def interrupted_sleep(seconds):
        sleep_calls['n'] += 1
        if sleep_calls['n'] == 1: raise KeyboardInterrupt('fixture interruption during drain')
        true_sleep(seconds)
    start = time.monotonic()
    with patch.object(R, 'write', side_effect=PermissionError(errno.EACCES, 'injected dispatcher write denial')), \
         patch.object(fault_driver, 'update', side_effect=OSError(errno.EBUSY, 'injected status replacement conflict')), \
         patch.object(R.time, 'sleep', side_effect=interrupted_sleep):
        fault_failure = fault_driver.drain_after_failure(RuntimeError('injected metadata failure'))
    durations['fault_injected_drain_seconds'] = time.monotonic()-start
    assert all(j['process'].poll() == 0 for j in fault_driver.jobs)
    assert fault_failure['drain_status'] == 'ALL_OWNED_CHILDREN_EXITED'
    stages = {e['stage'] for e in fault_failure['shutdown_record_errors']}
    assert {'INITIAL_FAILURE_RECORD', 'INITIAL_STOPPING_STATUS', 'CHILD_COMPLETION_RECORD',
            'FINAL_FAILURE_RECORD', 'FINAL_BLOCKED_STATUS', 'WAIT_INTERRUPTED'} <= stages
    assert C.read(fault_folder/'status.json')['signals'] == 1
    assert C.read(fault_root/'sampling/natural_complete.json')['signals'] == 0
    assert fault_failure['child_outcomes'][0]['recovery_precheck']['safe_to_start']
    checks.extend(['production_drain_survives_initial_and_final_metadata_write_failure',
        'production_drain_survives_status_update_and_completion_receipt_failure',
        'production_drain_survives_wait_KeyboardInterrupt_without_resignal',
        'production_fault_drain_still_waits_for_both_real_child_exits'])
    source_after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_before}
    result = {'status': 'PASS', 'checks': checks, 'durations': durations,
              'production_hashes_before': source_before, 'production_hashes_after': source_after,
              'source_unchanged_during_test': source_after == source_before,
              'torch_imported': 'torch' in sys.modules, 'model_calls': 0, 'GPU_calls': 0,
              'scientific_jobs_retried': 0, 'mock_children': 4, 'failure_record': failure,
              'injected_failure_record': fault_failure}
    assert not result['torch_imported']
    save(out/'results.json', result)
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'result': str(out/'results.json'),
                      'durations': durations, 'source_unchanged': source_after == source_before}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--child');parser.add_argument('--mode')
    args = parser.parse_args()
    if args.child: child(args.child, args.mode)
    else: run()
