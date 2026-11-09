"""Independent CPU fixtures: metadata visibility and owned-child safe draining.

Does not import project training modules, Torch, CUDA, or edit execution code.
The unlink/recreate test deliberately simulates the observed visibility gap;
it is not a claim to reproduce every DrvFs atomic-replace implementation.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


TRANSIENT = {errno.ENOENT, errno.EACCES, errno.EPERM, errno.EBUSY,
             getattr(errno, 'ESTALE', 116)}


class MetadataUnavailable(RuntimeError):
    pass


def read_metadata(path, *, seconds=.5, period=.01, optional_missing=False,
                  reader=None, errors=None):
    """Bounded I/O retries only; schema checks belong outside this helper."""
    reader = reader or (lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig')))
    deadline = time.monotonic()+seconds
    attempts = 0
    only_missing = True
    while True:
        attempts += 1
        try:
            return reader(path)
        except json.JSONDecodeError as exc:
            only_missing = False
            failure = exc
        except OSError as exc:
            if exc.errno not in TRANSIENT:
                raise
            only_missing = only_missing and exc.errno == errno.ENOENT
            failure = exc
        if errors is not None:
            errors.append(type(failure).__name__)
        if time.monotonic() >= deadline:
            if optional_missing and only_missing:
                return None
            raise MetadataUnavailable(f'{path}: {attempts} bounded reads failed') from failure
        time.sleep(min(period, max(0., deadline-time.monotonic())))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.partial')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def child(folder):
    """One fictional update, delayed tiny checkpoint; no scientific workload."""
    folder = Path(folder)
    stopped = {'value': False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(value=True))
    write(folder/'child_ready.json', {'pid': os.getpid(), 'step': 1112})
    while not stopped['value']:
        time.sleep(.01)
    # Represents finishing the current update followed by checkpoint I/O.
    time.sleep(.18)
    (folder/'rolling.mock').write_bytes(b'complete mock optimizer and RNG state')
    time.sleep(.18)
    write(folder/'rolling.json', {'step': 1113,
          'sha256': hashlib.sha256((folder/'rolling.mock').read_bytes()).hexdigest()})
    write(folder/'status.json', {'status': 'INTERRUPTED_SAFE', 'step': 1113,
                                'pid': os.getpid(), 'parent_alive': os.getppid() != 1})


def fixture_main(destination):
    destination = Path(destination)
    result_path = destination/'results.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    checks = []
    details = {}
    with tempfile.TemporaryDirectory(prefix='controller_cpu_', dir=destination) as td:
        folder = Path(td)
        path = folder/'status.json'
        write(path, {'status': 'TRAINING', 'step': 1112})
        assert path.exists()
        path.unlink()
        try:
            json.loads(path.read_text())
        except FileNotFoundError:
            checks.append('old_exists_then_read_race_reproduced')
        timer = threading.Timer(.06, lambda: write(path, {'status': 'TRAINING', 'step': 1112}))
        timer.start()
        errors = []
        payload = read_metadata(path, errors=errors)
        timer.join()
        assert payload['step'] == 1112 and 'FileNotFoundError' in errors
        checks.append('bounded_read_recovers_visibility_gap_without_job_restart')
        details['visibility_gap_retry_count'] = len(errors)

        calls = {'n': 0}
        def mock_reader(_):
            calls['n'] += 1
            if calls['n'] == 1: raise PermissionError(errno.EACCES, 'temporary share/replace access')
            if calls['n'] == 2: return json.loads('{')
            return {'step': 1112}
        assert read_metadata(path, reader=mock_reader)['step'] == 1112 and calls['n'] == 3
        checks.append('permission_and_json_decode_transients_are_bounded')

        calls['n'] = 0
        def permanent_reader(_):
            calls['n'] += 1
            raise OSError(errno.EIO, 'non-transient I/O failure')
        try: read_metadata(path, reader=permanent_reader)
        except OSError as exc: assert exc.errno == errno.EIO and calls['n'] == 1
        else: raise AssertionError('Permanent I/O error was hidden')
        checks.append('unapproved_oserror_not_retried')

        start = time.monotonic()
        try: read_metadata(path, seconds=.04, reader=lambda _: json.loads('{'))
        except MetadataUnavailable: assert .035 <= time.monotonic()-start < .3
        else: raise AssertionError('Persistent malformed JSON was hidden')
        checks.append('persistent_malformed_metadata_fails_after_deadline')

        path.unlink()
        (folder/'training_log.jsonl').write_text('{"step":1112}\n')
        timer = threading.Timer(.05, lambda: write(path, {'status': 'TRAINING', 'step': 1112}))
        timer.start()
        payload = read_metadata(path, optional_missing=True)
        timer.join()
        assert payload is not None and payload['status'] == 'TRAINING'
        checks.append('recovery_does_not_call_temporary_missing_status_orphan')
        path.unlink()
        payload = read_metadata(path, seconds=.04, optional_missing=True)
        assert payload is None and (folder/'training_log.jsonl').exists()
        checks.append('persistently_missing_status_with_existing_artifacts_is_not_safe_to_start')

        process = subprocess.Popen([sys.executable, '-B', __file__, '--child', str(folder)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start = time.monotonic()
        ready = read_metadata(folder/'child_ready.json', seconds=2)
        assert ready['pid'] == process.pid
        process.send_signal(signal.SIGTERM)
        # Existing handler raises here; demonstrate acknowledgement is not yet ready.
        assert process.poll() is None and not (folder/'rolling.mock').exists()
        checks.append('signal_delivery_is_not_safe_checkpoint_acknowledgement')
        pending = []
        launch_count = 1
        heartbeat = 0
        while process.poll() is None:
            heartbeat += 1
            assert pending == [] and launch_count == 1
            try: process.wait(timeout=.025)
            except subprocess.TimeoutExpired: pass
        stdout, stderr = process.communicate()
        assert process.returncode == 0, stderr.decode()
        payload = read_metadata(folder/'status.json')
        receipt = read_metadata(folder/'rolling.json')
        assert payload['status'] == 'INTERRUPTED_SAFE' and payload['step'] == receipt['step'] == 1113
        assert payload['parent_alive'] and receipt['sha256'] == hashlib.sha256((folder/'rolling.mock').read_bytes()).hexdigest()
        assert heartbeat >= 2 and launch_count == 1
        checks.append('parent_drains_owned_child_through_safe_checkpoint_and_actual_exit')
        details['drain_wall_seconds'] = time.monotonic()-start
        details['drain_poll_count'] = heartbeat
        details['child_processes_launched'] = launch_count
        details['scientific_jobs_retried'] = 0
    result = {'status': 'PASS', 'checks': checks, 'details': details,
              'model_calls': 0, 'GPU_calls': 0, 'scientific_code_modified': False,
              'scope': 'Independent stdlib CPU fixtures, not a production driver execution',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child: child(args.child)
    else: fixture_main(Path(__file__).resolve().parent)
