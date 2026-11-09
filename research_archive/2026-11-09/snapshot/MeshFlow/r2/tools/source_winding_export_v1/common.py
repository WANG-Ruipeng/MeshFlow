"""Independent CPU-only continuation; historical sources and results are read-only."""
from pathlib import Path
import csv, hashlib, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_source_winding_export_v1'
OLD=ROOT/'experiments/r2_c_restore_bounded_v1'
VERSION='R2_SOURCE_WINDING_EXPORT_V1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def checked(p):
 p=Path(p).absolute()
 if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('Symlink output')
 p=p.resolve()
 if not p.is_relative_to(OUT.resolve()) or p==OUT.resolve():raise ValueError('Write outside experiment: '+str(p))
 return p
def atomic(p,fn,binary=False):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
  fn(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def clean(v):
 import numpy as np
 if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
 if isinstance(v,(tuple,list)):return [clean(a) for a in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def write(p,obj):atomic(p,lambda f:json.dump(clean(obj),f,indent=2,ensure_ascii=False,allow_nan=False))
def textfile(p,s):atomic(p,lambda f:f.write(s))
def save_npz(p,**arrays):
 import numpy as np
 atomic(p,lambda f:np.savez_compressed(f,**arrays),True)
def csvwrite(p,data):
 data=clean(data);fields=list(dict.fromkeys(k for r in data for k in r)) or ['status']
 def dump(f):
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)
 atomic(p,dump)
def payload(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def event(kind,**kw):
 p=checked(OUT/'ledger.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:
  f.write(json.dumps(clean(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw)),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def resources():
 def call(args):
  r=subprocess.run(args,text=True,capture_output=True,timeout=20);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
 return dict(gpu=call(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),gpu_processes=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_cores=os.cpu_count(),loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text(),cpu_processes=call(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']))
def verify(mapping):
 for p,h in mapping.items():
  if sha(p)!=h:raise ValueError('Protected input/source changed: '+p)
 return len(mapping)
