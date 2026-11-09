"""One bounded preregistered task group under the shared lifetime GPU flock.

No torch import (and no CUDA/model creation) precedes the lock. The append-only
forward journal is fsynced before each call, so failed attempts retain budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

LIMITS = dict(formal=6000, regression=200, repeat=100, warmup=20)
TOTAL_LIMIT = 6320
LOCK_PATH = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
FILE_HASHES = {
    'B': '0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660',
    'J': 'cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ForwardLedger:
    """Caller must own the GPU flock; journal is the durable source of truth."""
    def __init__(self, path):
        self.path = Path(path)
        self.journal = self.path.with_suffix('.jsonl')
        self.state = dict(total_attempts=0, total_returns=0,
                          attempts_by_kind={k: 0 for k in LIMITS},
                          returns_by_kind={k: 0 for k in LIMITS}, tasks={},
                          limits=LIMITS, total_limit=TOTAL_LIMIT,
                          durability='Every attempt fsynced before model call; returns fsynced after model return.')
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                self._apply(json.loads(line))
        elif self.path.exists():
            raise RuntimeError('Ledger summary exists without its durable journal')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.path, self.state)

    def _apply(self, event):
        key, kind, action = event['task_id'], event['kind'], event['action']
        if kind not in LIMITS or action not in ('attempt', 'return'):
            raise ValueError('Invalid durable ledger event')
        task = self.state['tasks'].setdefault(key, dict(kind=kind, attempts=0, returns=0))
        if task['kind'] != kind:
            raise RuntimeError('Task ledger kind changed')
        field = 'attempts' if action == 'attempt' else 'returns'
        task[field] += 1
        self.state['total_' + field] += 1
        self.state[field + '_by_kind'][kind] += 1
        if (self.state['total_attempts'] > TOTAL_LIMIT or
                self.state['attempts_by_kind'][kind] > LIMITS[kind] or
                task['returns'] > task['attempts']):
            raise RuntimeError('Corrupt or over-budget forward ledger')

    def _append(self, task, action):
        event = dict(task_id=task['task_id'], kind=task['kind'], action=action,
                     timestamp=time.time(), pid=os.getpid())
        with self.journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(event)

    def reserve(self, task):
        kind, key = task['kind'], task['task_id']
        if kind not in LIMITS:
            raise ValueError('Unregistered budget category')
        existing = self.state['tasks'].get(key, {})
        if (self.state['total_attempts'] >= TOTAL_LIMIT or
                self.state['attempts_by_kind'][kind] >= LIMITS[kind] or
                existing.get('attempts', 0) >= (1 if kind == 'warmup' else 50)):
            raise RuntimeError('Durable forward budget exhausted before model call')
        self._append(task, 'attempt')

    def returned(self, task):
        previous = self.state['tasks'].get(task['task_id'])
        if previous is None or previous['returns'] >= previous['attempts']:
            raise RuntimeError('Forward return without a reserved attempt')
        self._append(task, 'return')

    def flush(self):
        save_json(self.path, self.state)


def snapshot():
    def cmd(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            return dict(returncode=-1, stdout='', stderr=repr(exc))
    memory_path = Path('/proc/meminfo')
    memory = ({s.split(':')[0]: s.split(':', 1)[1].strip() for s in memory_path.read_text().splitlines()
               if s.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:'))} if memory_path.exists() else {})
    result = dict(timestamp=time.time(),
        gpu=cmd(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
        processes=cmd(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits']),
        cpu_cores=os.cpu_count(), available_cpu_cores=len(os.sched_getaffinity(0)),
        load=list(os.getloadavg()), memory=memory,
        cpu_processes=cmd(['ps', '-eo', 'pid,comm,pcpu,pmem,nlwp', '--sort=-pcpu']))
    return result


def peak_resources(snapshots):
    used, process_used = [], []
    pid = str(os.getpid())
    for item in snapshots:
        try:
            used.append(float(item['gpu']['stdout'].splitlines()[0].split(',')[2]))
        except (ValueError, IndexError):
            pass
        for line in item['processes']['stdout'].splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3 and parts[0] == pid:
                try:
                    process_used.append(float(parts[-1]))
                except ValueError:
                    pass
    return dict(nvidia_smi_process_peak_MiB=max(process_used) if process_used else None,
                nvidia_smi_total_used_peak_MiB=max(used) if used else None,
                process_measurement_note='Sampled every 1 s, not a guaranteed continuous peak; WSL may report process VRAM N/A. Total used includes unrelated processes.')


def compare_archive(path, current, *, require_stages=False):
    import numpy as np
    path = Path(path)
    if not path.exists():
        if require_stages:
            raise FileNotFoundError(path)
        return dict(status='ARCHIVE_REGRESSION_UNAVAILABLE', path=str(path))
    aliases = dict(raw=('raw', 'output'), states=('states', 'path'),
                   stage_states=('stage_states',), stage_velocities=('stage_velocities', 'velocities'),
                   stage_times_FP32=('stage_times_FP32', 'times_FP32'),
                   accepted_times_FP32=('accepted_times_FP32',), step_sizes_FP32=('step_sizes_FP32',))
    rows = []
    with np.load(path, allow_pickle=False) as archive:
        for target, names in aliases.items():
            name = next((n for n in names if n in archive.files), None)
            if name is None:
                if require_stages:
                    raise RuntimeError('Repeat archive missing ' + target)
                continue
            a, b = np.asarray(current[target]), archive[name]
            selected_indices = None
            if target == 'states' and a.shape != b.shape and 'state_indices' in archive.files:
                indices = archive['state_indices']
                if (indices.dtype.kind not in 'iu' or indices.ndim != 1 or
                        len(indices) != len(b) or np.any(indices < 0) or np.any(indices >= len(a))):
                    raise RuntimeError('Invalid archived accepted-state indices')
                a = a[indices]
                selected_indices = indices.tolist()
            if target == 'raw' and b.shape != a.shape and b.size == a.size:
                b = b.reshape(a.shape)
            equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
                np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
            rows.append(dict(current_key=target, archived_key=name, bitwise_equal=bool(equal),
                             current_shape=list(a.shape), archived_shape=list(b.shape), archived_state_indices=selected_indices))
    if not rows:
        raise RuntimeError('Archive contains no comparable raw/states arrays')
    status = 'PASS' if all(row['bitwise_equal'] for row in rows) else 'FAIL'
    return dict(status=status, path=str(path), sha256=sha256(path), comparisons=rows,
                unarchived_fields=[k for k in aliases if k not in {r['current_key'] for r in rows}])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = Path(args.out).resolve()
    protocol_path = out / 'registration/protocol.json'
    protocol = read_json(protocol_path)
    if protocol.get('status') != 'FROZEN':
        raise RuntimeError('Protocol must be frozen before worker dispatch')
    matching = [g for g in protocol['groups'] if str(g['group_id']) == args.group]
    if len(matching) != 1:
        raise ValueError('Exactly one registered group must match')
    tasks = matching[0]['tasks']
    if any(t['kind'] not in ('regression', 'formal', 'repeat') for t in tasks):
        raise ValueError('Worker only executes registered full trajectories')
    dest = out / 'workers' / (args.group + '.json')
    if dest.exists():
        raise FileExistsError('Worker group already attempted; automatic retry forbidden: ' + str(dest))
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    info = dict(group_id=args.group, status='WAITING_LOCK', rows=[], counts={},
                lock_path=LOCK_PATH, actual_gpu_workers=1, requested_tasks=len(tasks),
                protocol_sha256=sha256(protocol_path), resource_snapshots=[snapshot()], load_events=[])
    save_json(dest, info)
    import fcntl
    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    waiting = time.perf_counter()
    print('WAIT_LOCK', args.group, flush=True)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    info.update(lock_inode=os.fstat(lock_fd).st_ino, lock_owner_pid=os.getpid(),
                lock_wait_seconds=time.perf_counter() - waiting, lock_acquired_timestamp=time.time())
    info['resource_snapshots'].append(snapshot())
    parts = info['resource_snapshots'][-1]['gpu']['stdout'].splitlines()[0].split(',')
    total, free = float(parts[1]), float(parts[3])
    headroom = max(2048., total * .1)
    info['dispatch'] = dict(headroom_MiB=headroom, free_MiB=free,
                            concurrency_reason='Experiment requires one lifetime GPU flock owner.',
                            minimum_additional_capacity_MiB=1536)
    if free < headroom + 1536:
        info['status'] = 'RESOURCE_BLOCKED'
        save_json(dest, info)
        raise RuntimeError('Insufficient available memory with required headroom')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from native_t1.runtime import configure_stable_runtime
    import numpy as np
    import torch
    from native_t1.artifacts import array_hash, state_sha256
    from experiments.mf_h1_local0.jedge5_diagnosis.common import ASSETS, load_model, load_cases, noise_bank
    from .sampler import sample, new_counts
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before locked worker runtime')
    info['CUDA_initialized_before_locked_runtime'] = False
    configure_stable_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ledger = ForwardLedger(out / 'forward_ledger.json')
    for task in tasks:
        if task['task_id'] in ledger.state['tasks'] or (out / 'raw' / (task['task_id'] + '.npz')).exists():
            raise RuntimeError('Task already attempted; retry forbidden: ' + task['task_id'])
    counts = new_counts()
    info.update(status='RUNNING', counts=counts)
    save_json(dest, info)
    stop = threading.Event()
    def monitor():
        while not stop.wait(1.):
            info['resource_snapshots'].append(snapshot())
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    cache, identities = {}, {}
    active_label = None
    current_row = None
    task_tick = None
    try:
        for source, expected in protocol.get('source_sha256', {}).items():
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = root / source_path
            if sha256(source_path) != expected:
                raise RuntimeError('Frozen source changed: ' + str(source_path))
        cases = load_cases()
        new_noise = noise_bank(out)
        old_run = protocol.get('historical_run', root / 'native_t1/runs/jedge5_diagnosis_20261002_102215')
        historical_noise = noise_bank(old_run) if any(t['noise_source'] == 'historical' for t in tasks) else {}
        for task in tasks:
            task_tick = time.perf_counter()
            current_row = dict(task, status='RUNNING', sampling_status='RUNNING', counts_before=dict(counts))
            info['rows'].append(current_row)
            save_json(dest, info)
            label = task['model']
            if label not in FILE_HASHES:
                raise ValueError('Only frozen B/J models authorized')
            loading = time.perf_counter()
            if active_label != label:
                if active_label is not None:
                    cache[active_label].cpu()
                    if state_sha256(cache[active_label]) != identities[active_label]['state_sha256']:
                        raise RuntimeError('Frozen model state mutated')
                    torch.cuda.empty_cache()
                if label not in cache:
                    if sha256(ASSETS[label]['path']) != FILE_HASHES[label]:
                        raise RuntimeError('Checkpoint file identity changed')
                    cache[label], identities[label] = load_model(label, device='cpu')
                cache[label].cuda()
                torch.cuda.synchronize()
                active_label = label
                info['load_events'].append(dict(model=label, seconds=time.perf_counter() - loading,
                    identity=identities[label], cuda_allocated=torch.cuda.memory_allocated(),
                    cuda_reserved=torch.cuda.memory_reserved()))
            current_row['load_seconds'] = time.perf_counter() - loading
            case = cases[int(task['parent']), int(task['patch'])]
            bank = historical_noise if task['noise_source'] == 'historical' else new_noise
            noise = bank[int(task['seed'])].clone().cuda()
            C = torch.from_numpy(case['C'].copy()).cuda()
            # Two representative linear hooks on the first real call only.
            dtype_facts, handles = {}, []
            linears = [(n, m) for n, m in cache[label].named_modules() if isinstance(m, torch.nn.Linear)]
            for name, module in (linears[:1] + linears[-1:]):
                def dtype_hook(module, inputs, output, name=name):
                    if name not in dtype_facts:
                        dtype_facts[name] = dict(parameter=str(module.weight.dtype),
                            input=str(inputs[0].dtype), output=str(output.dtype))
                handles.append(module.register_forward_hook(dtype_hook))
            try:
                raw, states, audit = sample(cache[label], noise, C, task['solver'], counts=counts,
                    known_slots=protocol.get('known_slots', list(range(12))),
                    reserve_forward=lambda task=task: ledger.reserve(task),
                    forward_returned=lambda task=task: ledger.returned(task))
            finally:
                for handle in handles:
                    handle.remove()
            arrays = audit.pop('_arrays')
            audit['actual_accepted_times_FP32'] = arrays['accepted_times_FP32'].tolist()
            audit['representative_layer_dtypes'] = dtype_facts
            payload = dict(raw=raw, states=states, C=case['C'], **arrays)
            writing = time.perf_counter()
            path = out / 'raw' / (task['task_id'] + '.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream, **payload)
            current_row.update(path=str(path.relative_to(out)), sha256=sha256(path), audit=audit,
                nfe=50, model_file_sha256=FILE_HASHES[label], model_state_sha256=identities[label]['state_sha256'],
                input_sha256=array_hash(states[0]), C_sha256=array_hash(case['C']), noise_sha256=array_hash(bank[int(task['seed'])]),
                noise_free_sha256=array_hash(bank[int(task['seed'])][:, 12:]),
                trajectory_write_seconds=time.perf_counter() - writing)
            reference = task.get('reference_npz')
            if task['kind'] == 'repeat':
                reference = out / 'raw' / (task['repeat_of'] + '.npz')
            if reference:
                comparison = compare_archive(reference, payload, require_stages=task['kind'] == 'repeat')
                current_row['regression'] = comparison
                if comparison['status'] == 'FAIL':
                    raise RuntimeError('Bitwise archived trajectory comparison failed')
            elif task['kind'] == 'regression':
                current_row['regression'] = dict(status='ARCHIVE_REGRESSION_UNAVAILABLE')
            current_row.update(status='PASS', sampling_status='PASS', counts_after=dict(counts),
                               wall_seconds=time.perf_counter() - task_tick)
            ledger.flush()
            save_json(dest, info)
            print('COMPLETE', task['task_id'], 'NFE', 50, 'seconds', round(current_row['wall_seconds'], 3), flush=True)
            current_row = None
        info['final_model_state_sha256'] = {label: state_sha256(model) for label, model in cache.items()}
        if any(info['final_model_state_sha256'][label] != identities[label]['state_sha256'] for label in cache):
            raise RuntimeError('Final frozen model identity changed')
        info['status'] = 'PASS'
    except BaseException as exc:
        info['status'] = 'STOPPED'
        info['error'] = dict(message=repr(exc), traceback=traceback.format_exc())
        if current_row is not None:
            current_row.update(status='FAIL', sampling_status='FAIL', error=repr(exc), counts_after=dict(counts),
                               wall_seconds=time.perf_counter() - task_tick)
            if hasattr(exc, 'partial_states'):
                partial_path = out / 'raw' / (current_row['task_id'] + '.failed_partial.npz')
                with partial_path.open('xb') as stream:
                    np.savez_compressed(stream, states=exc.partial_states,
                        stage_states=exc.partial_stage_states, stage_velocities=exc.partial_stage_velocities,
                        stage_times_FP32=exc.partial_stage_times_FP32,
                        stage_times_control_FP64=exc.partial_stage_times_control_FP64)
                current_row['partial_path'] = str(partial_path.relative_to(out))
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=10)
        info['seconds'] = time.perf_counter() - started
        info['resource_snapshots'].append(snapshot())
        info.update(peak_resources(info['resource_snapshots']))
        if torch.cuda.is_initialized():
            info['last_rollout_peak_cuda_allocated'] = torch.cuda.max_memory_allocated()
            info['last_rollout_peak_cuda_reserved'] = torch.cuda.max_memory_reserved()
        info['lock_held_until_process_exit'] = True
        info['ledger'] = ledger.state
        ledger.flush()
        save_json(dest, info)
        cache.clear()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        # Keep lock_fd open: process exit releases after the CUDA context dies.


if __name__ == '__main__':
    main()
