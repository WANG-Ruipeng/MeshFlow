"""Small CPU-only durable state helpers for the single managed experiment."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
GPU_LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
LIMITS = dict(main_updates=9000, optimizer_updates=9000, generations=600, repeat_generations=2, all_generations=602, teacher_forwards=288, wiring_forwards=32, training_microbatch_forwards=72000, model_forwards=102420)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def append_event(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(time_unix=time.time(), pid=os.getpid(), **obj)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())

def freeze_json(path, obj):
    path = Path(path)
    if path.exists():
        if read_json(path) != obj:
            raise ValueError('Frozen artifact identity changed: '+str(path))
    else:
        atomic_json(path, obj)
    return digest(path)

@contextlib.contextmanager
def controller_lock(out):
    import fcntl
    path = Path(out)/'controller.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An identical experiment controller is already running')
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
