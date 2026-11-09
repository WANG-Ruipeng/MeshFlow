"""CPU-safe experiment identities and append-only provenance helpers."""
from pathlib import Path
import json, hashlib, os, time
from r2.pilots.bench_c_tessellation_v1.common import resolve, read, sha, arrays, npz, verify, csvwrite

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_three_mechanisms_v1'
EXPERIMENT = 'R2_BENCH_THREE_MECHANISMS_V1'
OLD = ROOT / 'experiments/r2_bench_four_routes_v1/exposure'
PRIOR = ROOT / 'experiments/r2_bench_multilod_training_v1'
SOURCE = OLD / 'training/M_PLUS3000/checkpoint_step3000.pt'
SOURCE_INFERENCE = OLD / 'training/M_PLUS3000/generator_step3000.pt'
BASE_STEP = 25500
NATIVE_BASE = 13500
UPDATES = {'PLAIN':267, 'PATCH':200, 'LOOP':200}
LABELS = ('EXPOSURE','PLAIN200','PLAIN267','PATCH200','LOOP200','RL_ZERO','RL16')
NEW_LABELS = LABELS[1:]
PANEL_PREFIXES = ('a8e987','27eb79','bcb4f5','459f90','f495bd','df6fd3')
RL_PREFIXES = ('a8e987','27eb79','35948e','bcb4f5')
REGISTRATION = OUT / 'registration.json'
L = 9.208138264587333

def write(path, value, replace=False):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if path.exists() and not replace:
        if path.read_bytes() != data:
            raise RuntimeError('Immutable file differs: '+str(path))
        return
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def event(path, value):
    path=resolve(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'wall_time':time.time(),**value},ensure_ascii=False,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(EXPERIMENT,)+parts)).encode()).digest()[:4],'little')

def resources():
    import psutil, subprocess
    query=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    v=[s.strip() for s in query.split(',')]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    return dict(wall_time=time.time(),gpu_name=v[0],total_mib=float(v[1]),used_mib=float(v[2]),free_mib=float(v[3]),gpu_util_percent=float(v[4]),
        compute_processes=processes,cpu_cores=psutil.cpu_count(),cpu_percent=psutil.cpu_percent(interval=.2),
        available_ram_bytes=psutil.virtual_memory().available,total_ram_bytes=psutil.virtual_memory().total,
        cpu_processes=[dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss if p.info['memory_info'] else None)
            for p in psutil.process_iter(['name','memory_info']) if p.info['name'] and ('python' in p.info['name'] or 'audit' in p.info['name'])])

def register_resource(phase):
    r=resources(); event(OUT/'resources/snapshots.jsonl',dict(phase=phase,**r)); return r
