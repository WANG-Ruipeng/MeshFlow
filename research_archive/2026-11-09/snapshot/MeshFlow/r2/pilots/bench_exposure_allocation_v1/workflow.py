"""Manual durable stage launcher; no automatic dispatch, replay, or concurrency policy.

Run under the registered WSL Python. A claimed job directory is immutable evidence
of an attempt, even if launch failed before the scientific worker started.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import psutil

from .common import ROOT, OUT, MODULE, read, sha, write_new, atomic_write, resources

JOBS = {
    'train-a': ('training', ('--arm', 'A_50_50')),
    'train-b': ('training', ('--arm', 'B_75_25')),
    'sample-a': ('evaluation', ('sample', '--model', 'A')),
    'sample-b': ('evaluation', ('sample', '--model', 'B')),
    'd0-a': ('evaluation', ('d0', '--model', 'A')),
    'd0-b': ('evaluation', ('d0', '--model', 'B')),
    'prepare-scores': ('evaluation', ('prepare-scores',)),
    'score-raw': ('evaluation', ('raw', '--models', 'A', 'B', '--workers', '3')),
    'score-post': ('evaluation', ('post', '--models', 'A', 'B', '--workers', '3')),
    'collect': ('evaluation', ('collect',)),
}
TRAIN_ARMS = {'train-a': 'A_50_50', 'train-b': 'B_75_25'}
VERSION = 'bench_exposure_allocation_manual_workflow_v1'


def pid_alive(pid, created=None):
    """Read identity, never signal or take ownership from a bare PID."""
    try:
        p = psutil.Process(int(pid))
        return (p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                and created is not None and abs(p.create_time() - float(created)) < .01)
    except (TypeError, ValueError, psutil.NoSuchProcess):
        return False
    except psutil.AccessDenied:
        return None


def job_status(folder, alive=pid_alive):
    folder = Path(folder)
    if (folder / 'status.json').exists():
        state = read(folder / 'status.json')
    elif (folder / 'pid.json').exists():
        state = read(folder / 'pid.json')
    elif folder.exists():
        return dict(status='LAUNCH_INCOMPLETE_REQUIRES_INSPECTION',
                    reason='Attempt directory already claimed; no automatic relaunch',
                    automatic_retry=False)
    else:
        return dict(status='NOT_STARTED')
    if state['status'] in ('RUNNING', 'LAUNCHED'):
        live = alive(state.get('pid'), state.get('process_created'))
        state = dict(state, supervisor_alive=live)
        if 'child_pid' in state:
            state['child_process_alive'] = alive(
                state['child_pid'], state.get('child_process_created'))
        if live is not True:
            state.update(recorded_status=state['status'],
                         status='STALE_PROCESS_REQUIRES_INSPECTION',
                         reason='Supervisor exited, PID reused, or visibility unavailable',
                         automatic_retry=False)
        # A live supervisor owns wait/exit handling. A just-exited child is not
        # itself evidence of failure while that supervisor writes completion.
    return state


def status():
    return {name: job_status(OUT / 'workflow' / name) for name in JOBS}


def _preflight(name, out=OUT):
    registration_path = out / 'training_registration.json'
    registration = read(registration_path)
    if registration.get('status') != 'FROZEN':
        raise ValueError('Training registration is not frozen')
    registration_sha = sha(registration_path)
    if name in TRAIN_ARMS:
        arm = TRAIN_ARMS[name]
        go = read(out / 'dispatch' / (arm + '_go.json'))
        if (go.get('authorized') is not True or go.get('arm') != arm
                or go.get('training_registration_sha256') != registration_sha):
            raise ValueError('Missing or mismatched independently authorized training GO')
    return registration_sha


def launch(name):
    """Claim exactly one named attempt; return prior status rather than rerun it."""
    if os.name != 'posix':
        raise RuntimeError('Launch with the registered WSL Python, not Windows Python')
    if name not in JOBS:
        raise ValueError(name)
    folder = OUT / 'workflow' / name
    previous = job_status(folder)
    if previous['status'] != 'NOT_STARTED':
        return previous
    registration_sha = _preflight(name)
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()  # Atomic attempt ownership; concurrent launch cannot pass.
    except FileExistsError:
        return job_status(folder)
    token = uuid.uuid4().hex  # Workflow identity only; not a scientific RNG seed.
    command = [sys.executable, '-B', '-m', MODULE + 'workflow',
               '--execute', name, '--token', token]
    try:
        write_new(folder / 'resources_before.json', resources())
        write_new(folder / 'launch.json', dict(
            schema=VERSION, job=name, token=token, command=command,
            registration_sha256=registration_sha, workflow_sha256=sha(__file__),
            time=time.time(), automatic_retry=False, automatic_followup=False))
        with (folder / 'supervisor.log').open('xb') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            created = psutil.Process(p.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        launched = dict(schema=VERSION, status='LAUNCHED', job=name, pid=p.pid,
                        process_created=created, time=time.time(), token=token,
                        automatic_retry=False)
        write_new(folder / 'pid.json', launched)
        return job_status(folder)
    except BaseException as exc:
        # Do not overwrite a status produced by a supervisor already launched.
        if not (folder / 'status.json').exists():
            write_new(folder / 'status.json', dict(
                status='STOPPED_REQUIRES_INSPECTION', job=name, error=repr(exc),
                traceback=traceback.format_exc(), time=time.time(),
                automatic_retry=False))
        raise


def _complete_check(name):
    if name in TRAIN_ARMS:
        actual = read(OUT / 'training' / TRAIN_ARMS[name] / 'status.json')
        if (actual.get('status') != 'COMPLETE' or actual.get('local_step') != 1500
                or actual.get('global_step') != 21000):
            raise RuntimeError('Training exited without its frozen 1500/21000 endpoint')
    # Evaluation entry points validate their own receipt counts and barriers.


def _run_worker(folder, name, command, complete_check):
    """Supervisor core; testable using a harmless CPU child, never auto-retried."""
    folder = Path(folder)
    # Exclusive marker prevents duplicate --execute processes racing on status.
    write_new(folder / 'supervisor_claim.json', dict(
        pid=os.getpid(), process_created=psutil.Process().create_time(), time=time.time()))
    dest = folder / 'status.json'
    if dest.exists():
        raise RuntimeError('Recorded stage exists; explicit inspection is required')
    state = dict(schema=VERSION, status='RUNNING', job=name, pid=os.getpid(),
                 process_created=psutil.Process().create_time(), started=time.time(),
                 automatic_retry=False, automatic_followup=False, command=command)
    atomic_write(dest, state)
    try:
        with (folder / 'output.log').open('xb') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = p.pid
            try:
                state['child_process_created'] = psutil.Process(p.pid).create_time()
            except psutil.NoSuchProcess:
                state['child_process_created'] = None
            atomic_write(dest, state)
            state['exit_code'] = p.wait()
        if state['exit_code']:
            raise RuntimeError('Worker failed; preserve attempt, no automatic retry')
        complete_check(name)
        state['status'] = 'COMPLETE'
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION', error=repr(exc),
                     traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished=time.time(), seconds=time.time() - state['started'])
        atomic_write(dest, state)
    return state


def execute(name, token):
    folder = OUT / 'workflow' / name
    launch_record = read(folder / 'launch.json')
    if (not token or launch_record.get('token') != token
            or launch_record.get('job') != name
            or launch_record.get('workflow_sha256') != sha(__file__)
            or launch_record.get('registration_sha256') != _preflight(name)):
        raise RuntimeError('Supervisor launch identity changed; no worker dispatched')
    module, args = JOBS[name]
    return _run_worker(folder, name,
                       [sys.executable, '-B', '-m', MODULE + module, *args],
                       _complete_check)


def cpu_tests():
    """No model imports, GPU calls, geometry, or experiment stage invocation."""
    import tempfile
    checks = []
    with tempfile.TemporaryDirectory(prefix='exposure_workflow_cpu_') as tmp:
        base = Path(tmp)
        blank = base / 'blank'
        assert job_status(blank)['status'] == 'NOT_STARTED'
        blank.mkdir()
        assert job_status(blank)['status'] == 'LAUNCH_INCOMPLETE_REQUIRES_INSPECTION'
        checks.append('empty claimed directory cannot relaunch')
        active = base / 'active'; active.mkdir()
        write_new(active / 'pid.json', dict(status='LAUNCHED', pid=1, process_created=2))
        assert job_status(active, lambda *_: False)['status'] == 'STALE_PROCESS_REQUIRES_INSPECTION'
        assert job_status(active, lambda *_: None)['supervisor_alive'] is None
        assert job_status(active, lambda *_: True)['status'] == 'LAUNCHED'
        checks.append('dead, unknown, and live process identities')
        assert pid_alive(os.getpid(), psutil.Process().create_time()) is True
        assert pid_alive(os.getpid(), psutil.Process().create_time() - 2) is False
        checks.append('PID reuse detected by creation timestamp')
        ok = base / 'ok'; ok.mkdir()
        result = _run_worker(ok, 'CPU_STUB', [sys.executable, '-B', '-c', 'print("CPU_ONLY_STUB")'], lambda _: None)
        assert result['status'] == 'COMPLETE' and result['exit_code'] == 0
        assert 'CPU_ONLY_STUB' in (ok / 'output.log').read_text()
        try:
            _run_worker(ok, 'CPU_STUB', [sys.executable, '-c', 'raise Exception()'], lambda _: None)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Duplicate execution was accepted')
        checks.append('CPU child wait/log/complete and exclusive supervisor claim')
        bad = base / 'bad'; bad.mkdir()
        try:
            _run_worker(bad, 'CPU_FAIL_STUB', [sys.executable, '-B', '-c', 'raise SystemExit(7)'], lambda _: None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Worker failure not propagated')
        assert read(bad / 'status.json')['exit_code'] == 7
        assert job_status(bad)['status'] == 'STOPPED_REQUIRES_INSPECTION'
        checks.append('nonzero child exit remains inspect-only evidence')
        reg = base / 'reg'; reg.mkdir()
        write_new(reg / 'training_registration.json', dict(status='FROZEN'))
        assert _preflight('sample-a', reg) == sha(reg / 'training_registration.json')
        try:
            _preflight('train-a', reg)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing root GO was accepted')
        checks.append('registration and independently supplied training GO required')
    result = dict(schema=VERSION, status='PASS', checks=checks, tests=len(checks),
                  workflow_sha256=sha(__file__), scientific_calls=0, model_imports=0,
                  real_CPU_stub_processes=2, GPU_dispatches=0)
    write_new(OUT / 'workflow_cpu_gate.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--launch', choices=JOBS)
    group.add_argument('--execute', choices=JOBS, help=argparse.SUPPRESS)
    group.add_argument('--cpu-tests', action='store_true')
    p.add_argument('--token', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.execute:
        result = execute(a.execute, a.token)
    elif a.launch:
        result = launch(a.launch)
    elif a.cpu_tests:
        result = cpu_tests()
    else:
        result = status()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
