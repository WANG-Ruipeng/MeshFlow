"""Local identities and durable CPU helpers; historical assets remain read-only."""
from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'native_t1/runs/control_mvp48_v1'
CHAIR = ROOT/'native_t1/runs/chair_domain_control_v1'
OUT = ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
START = CHAIR/'checkpoints/Geo_phase3000_total4000.pt'
START_SHA = '190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf'
ARMS = ('OT_BASE', 'OT_HYBRID')
LABELS = ('START',) + ARMS
GPU_LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def freeze_json(path, value):
    p = Path(path)
    if p.exists():
        if read_json(p) != value:
            raise ValueError('Frozen artifact changed: '+str(p))
    else:
        atomic_json(p, value)
    return digest(p)


def append_event(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(time_unix=time.time(), pid=os.getpid(), **value)
    with p.open('a', encoding='utf8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def ledger_rows(path):
    p = Path(path)
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


def save_npz(path, **arrays):
    import numpy as np
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return digest(p)
