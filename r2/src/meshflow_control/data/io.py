"""Small durable I/O helpers shared by the data and training boundary."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.dtype, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp." + str(os.getpid()))
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def freeze_json(path, value):
    p = Path(path)
    if p.exists():
        if read_json(p) != value:
            raise ValueError("Frozen identity differs: " + str(p))
    else:
        atomic_json(p, value)
    return digest(p)

def save_npz(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp." + str(os.getpid()))
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    return digest(p)

def append_event(path, value):
    import time
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(time_unix=time.time(), pid=os.getpid(), **value)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []

def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
