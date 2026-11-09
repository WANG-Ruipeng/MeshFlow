from __future__ import annotations
import csv, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_relative_deformation_v1'
OLD=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
PREV=ROOT/'experiments/r2_bench_backend_shape_preservation_v1'
SCALE=.95*.3762
L=float(np.sqrt(3)*2/.3762)

def resolve(p):
 s=str(p).replace('\\','/')
 if os.name=='nt' and s.startswith('/mnt/e/'):s='E:/'+s[7:]
 elif os.name!='nt' and s.startswith('E:/'):s='/mnt/e/'+s[3:]
 p=Path(s);return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 return x
def write(p,v):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False)
def arrays(p):
 with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def npz(p,a):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.savez_compressed(f,**a)
def csvwrite(p,rows):
 p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def verify(book):
 for p,h in book.items():
  if sha(p)!=h:raise RuntimeError('Protected hash differs: '+str(p))
def resource_snapshot(phase):
 import psutil
 def cmd(a):
  r=subprocess.run(a,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 r=dict(phase=phase,time=time.time(),gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_count=psutil.cpu_count(),ram=psutil.virtual_memory()._asdict(),cpu_percent=psutil.cpu_percent(1),disk=psutil.disk_usage(str(ROOT))._asdict(),processes=[])
 for p in psutil.process_iter(['pid','name','cmdline','memory_info']):
  try:
   if 'python' in (p.info['name'] or '').lower():
    a=p.info;a['memory_info']=a['memory_info']._asdict();r['processes'].append(a)
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 write(OUT/'resources'/f'{phase}_{time.time_ns()}.json',r);return r
