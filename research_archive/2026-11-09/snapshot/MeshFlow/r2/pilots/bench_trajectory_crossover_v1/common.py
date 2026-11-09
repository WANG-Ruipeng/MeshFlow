import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, json, csv, io, time, subprocess
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
OUT = ROOT / 'experiments/r2_bench_trajectory_crossover_v1'
OLD = ROOT / 'experiments/r2_bench_trajectory_diagnostic_v1'
from r2.pilots.bench_parts_convergence_v1.data import resolve, read, sha, arrays, write_new as save
from r2.pilots.bench_parts_convergence_v1.panel import array_file
COMBOS = ('CC', 'CU', 'UC', 'UU')
MODELS = {'C': 'COMMON', 'U': 'USER'}

def event(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time=time.time(), **values), ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    b = io.StringIO(newline=''); w = csv.DictWriter(b, fieldnames=keys)
    w.writeheader(); w.writerows(rows)
    text = b.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding='utf-8-sig') == text.replace('\r\n', '\n'), path
    else:
        path.write_text('\ufeff' + text, encoding='utf-8', newline='')

def verify(files):
    for path, digest in files.items():
        assert sha(path) == digest, path

def resources():
    import psutil
    def query(args):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=15)
            return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
        except subprocess.TimeoutExpired:
            return dict(returncode=-1, stdout='', stderr='TIMEOUT; no dispatch allowed')
    active = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'python' in (p.info['name'] or '').lower():
                active.append(dict(pid=p.pid, cmdline=cmd, rss_bytes=p.info['memory_info'].rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return dict(time=time.time(), pid=os.getpid(), cpu_logical=psutil.cpu_count(),
        cpu_utilization_percent=psutil.cpu_percent(.25), loadavg=os.getloadavg(),
        RAM_total_bytes=psutil.virtual_memory().total, RAM_available_bytes=psutil.virtual_memory().available,
        python_processes=active, numerical_threads_per_worker=1,
        gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']))

def registration(): return read(OUT / 'registration.json')

def raw_path(combo, rid):
    if combo in ('CC', 'UU'):
        return OLD / 'evaluation' / MODELS[combo[0]] / rid / 'raw/raw.npz'
    return OUT / 'evaluation' / combo / rid / 'raw/raw.npz'

def score_path(combo, rid):
    if combo in ('CC', 'UU'):
        return OLD / 'evaluation/scores' / MODELS[combo[0]] / rid / 'RAW'
    return OUT / 'scores' / combo / rid / 'RAW'
