from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_multilod_data_v1'
SOURCE = ROOT / 'experiments/r2_bench_high_joint_coverage_v1/data/manifest.json'
L = float(np.sqrt(3.) * 2 / .3762)
VERSION = 'BENCH_FIXED_C_MULTILOD_V1'


def resolve(value):
    s = str(value).replace('\\', '/')
    for prefix in ('/mnt/e/MeshFlow/', 'E:/MeshFlow/'):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def safe_json(value):
    if isinstance(value, dict): return {str(k): safe_json(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray): return safe_json(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def write(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = safe_json(value)
    if path.exists():
        if read(path) != value: raise RuntimeError('Immutable artifact differs: '+str(path))
        return
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = arrays(path)
        if set(old) != set(value) or any(not np.array_equal(old[k], value[k]) for k in value):
            raise RuntimeError('Immutable NPZ differs: '+str(path))
        return
    with path.open('xb') as f: np.savez_compressed(f, **value)


def resources():
    import psutil
    def query(args):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=15)
            return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
        except Exception as e: return dict(status='UNAVAILABLE', reason=str(e))
    processes = []
    for p in psutil.process_iter(['pid','name','memory_info','cpu_times']):
        try:
            if any(s in p.info['name'].lower() for s in ('python','blender')):
                processes.append(dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss,
                                      cpu_seconds=sum(p.info['cpu_times'][:2])))
        except (psutil.AccessDenied,psutil.NoSuchProcess): pass
    return dict(unix=time.time(), cpu_cores=os.cpu_count(),
        available_ram_bytes=psutil.virtual_memory().available,
        load=list(os.getloadavg()), cpu_percent=psutil.cpu_percent(.2),
        disk_free_bytes=psutil.disk_usage(ROOT).free, processes=processes,
        gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        gpu_model_calls=0, numeric_threads_per_worker=1)


def event(kind, **payload):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(safe_json(dict(unix=time.time(),kind=kind,**payload)),ensure_ascii=False,allow_nan=False)+'\n')
