"""Isolated, fixed-budget Bench trajectory diagnosis; no implicit model calls."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import sys, json, hashlib, csv, io, time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
OUT=ROOT/'experiments/r2_bench_trajectory_diagnostic_v1'
V2=ROOT/'experiments/r2_bench_user_v2_training_v1'
LABELS=('COMMON','USER')
from r2.pilots.bench_parts_convergence_v1.data import resolve,read,sha,arrays,write_new,resources as original_resources
from r2.pilots.bench_parts_convergence_v1.panel import array_file
def save(p,x):write_new(p,x)
def event(p,**x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),**x),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r));b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=keys);w.writeheader();w.writerows(rows)
    text=b.getvalue()
    if p.exists():assert p.read_text(encoding='utf-8-sig')==text.replace('\r\n','\n')
    else:p.write_text('\ufeff'+text,encoding='utf-8',newline='')
def manifest():return read(OUT/'registration.json')
def verify(d):
    for p,h in d.items():assert sha(p)==h,p

def resources():
    if (OUT/'gpu_dispatch_blocked.json').exists():
        import psutil
        return dict(unix=time.time(),pid=os.getpid(),cpu_logical=psutil.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,
            cpu_utilization_percent=psutil.cpu_percent(.1),gpu=dict(returncode=1,stdout='',stderr='BLOCKED: prior NVML query uninterruptible; GPU dispatch forbidden'),
            gpu_processes=dict(returncode=1,stdout='',stderr='UNKNOWN_DRIVER_UNRESPONSIVE'),workers=1,numeric_threads=1)
    return original_resources()
