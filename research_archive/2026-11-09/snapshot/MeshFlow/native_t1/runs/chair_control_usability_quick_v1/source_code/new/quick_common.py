"""Small isolated IO and durable inference budget; no model side effects."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_control_usability_quick_v1'
CODE = Path(__file__).resolve().parent
LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'


def resolve(path):
    value = str(path).replace('\\', '/')
    if value.lower().startswith('e:/meshflow/'):
        return ROOT / value[len('e:/meshflow/'):]
    if value.startswith('/mnt/e/MeshFlow/'):
        return ROOT / value[len('/mnt/e/MeshFlow/'):]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    p = resolve(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def event(path, row):
    p = resolve(path); p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def resource_snapshot(out, phase):
    commands = {
        'gpu': ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'],
        'cpu_processes': ['ps', '-eo', 'pid,comm,pcpu,pmem,args', '--sort=-pcpu'],
    }
    data = dict(phase=phase, pid=os.getpid(), cpu_cores=os.cpu_count(), gpu_workers=1,
                cpu_threads_per_worker=2, parallel_cpu_work='registration/scoring/rendering; each bounded independently')
    for key, cmd in commands.items():
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data[key] = dict(returncode=p.returncode, stdout='\n'.join(p.stdout.splitlines()[:16]), stderr=p.stderr)
    mem = Path('/proc/meminfo')
    data['memory'] = mem.read_text() if mem.exists() else 'NA'
    try:
        parts = data['gpu']['stdout'].splitlines()[0].split(',')
        data['free_mib'] = int(parts[-2].strip()); data['total_mib'] = int(parts[-4].strip())
    except (ValueError, IndexError):
        data['free_mib'] = None; data['total_mib'] = None
    event(Path(out) / 'resources/snapshots.jsonl', data)
    return data


class Budget:
    limits = {'sampling': 82, 'sampling_forward': 4100, 'precheck_forward': 12}

    def __init__(self, out=OUT):
        self.out = Path(out); self.path = self.out / 'ledger/calls.jsonl'
        self.attempts = Counter(); self.returns = Counter(); self.failed = Counter(); self.pending = {}
        self.per_request_forwards = Counter(); self.seconds = Counter()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line); k = row['kind']; cid = row['call_id']
                if row['event'] == 'BEGIN':
                    self.attempts[k] += 1; self.pending[cid] = row
                    if k == 'sampling_forward': self.per_request_forwards[row['context']['request_id']] += 1
                else:
                    self.pending.pop(cid, None)
                    (self.returns if row['event'] == 'RETURN' else self.failed)[k] += 1
                    self.seconds[k] += row['seconds']
        self.serial = sum(self.attempts.values())
        if self.pending: raise RuntimeError('Unresolved calls require inspection; no automatic retry: ' + str(self.pending))

    def begin(self, kind, context=None):
        if kind not in self.limits or self.attempts[kind] >= self.limits[kind]:
            raise RuntimeError('Forbidden or exhausted call: ' + kind)
        if kind.endswith('forward') and self.attempts['sampling_forward'] + self.attempts['precheck_forward'] >= 4112:
            raise RuntimeError('Top-level forward hard limit')
        context = dict(context or {})
        if kind == 'sampling_forward':
            rid = context['request_id']
            if self.per_request_forwards[rid] >= 50: raise RuntimeError('Per-output Euler50 limit')
            self.per_request_forwards[rid] += 1
        self.serial += 1; cid = str(self.serial)
        row = dict(event='BEGIN', kind=kind, call_id=cid, context=context)
        event(self.path, row); self.attempts[kind] += 1; self.pending[cid] = row
        return cid, time.perf_counter()

    def end(self, token, error=None):
        cid, tick = token; row = self.pending.pop(cid); k = row['kind']; sec = time.perf_counter() - tick
        event(self.path, dict(event='RETURN' if error is None else 'FAILED', kind=k, call_id=cid,
                             context=row['context'], seconds=sec, error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[k] += 1; self.seconds[k] += sec

    def call(self, kind, fn, *args, context=None, **kwargs):
        token = self.begin(kind, context)
        try: value = fn(*args, **kwargs)
        except BaseException as exc:
            self.end(token, exc); raise
        self.end(token); return value

    def snapshot(self):
        rows = [read(p) for p in (self.out / 'outputs').glob('*/result.json')]
        value = dict(attempts=dict(self.attempts), returns=dict(self.returns), failed_calls=dict(self.failed),
                     pending=list(self.pending), hard_limits=self.limits, top_level_forward_limit=4112,
                     total_forward_attempts=self.attempts['sampling_forward'] + self.attempts['precheck_forward'],
                     batch_forward_calls=self.attempts['sampling_forward'] + self.attempts['precheck_forward'],
                     backward_attempts=0, optimizer_updates=0, training_updates=0,
                     generation_requests=len(rows), cached=sum(r['status']=='CACHED' for r in rows),
                     newly_generated=sum(r['status']=='SAMPLED' for r in rows),
                     failed_outputs=sum(r['status']=='FAILED' for r in rows),
                     unavailable=sum(r['status'] in ('BLOCKED','NOT_APPLICABLE') for r in rows),
                     complete_sampling_attempts=self.attempts['sampling'],
                     complete_sampling_returns=self.returns['sampling'],
                     identity_replays=len(list((self.out / 'replays').glob('*/result.json'))),
                     wall_seconds_by_kind=dict(self.seconds), gpu_workers=1,
                     timing_note='Rollout and nested forward timers overlap; cache reads are not generation throughput.')
        write(self.out / 'sampling_accounting.json', value); return value
