"""Isolated output protection, atomic artifacts, and fixed pilot accounting."""
from pathlib import Path
import csv, hashlib, io, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_anchor_aware_denoiser_v1'
OLD=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'
CODE=Path(__file__).parent
ASSET=OLD/'assets/chair_denoiser_0030000.pt'
EXPECTED='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
SCALE=.95*.3762
SIGMA=.02
ARMS=('D_CONT_NO_ROLE','D_ANCHOR')
VERSION='R2_ANCHOR_AWARE_DENOISER_PILOT_V1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def checked(p):
 lexical=Path(p).absolute()
 for part in (lexical,*lexical.parents):
  if part.is_symlink():raise ValueError('SYMLINK_OUTPUT: '+str(part))
 p=lexical.resolve();root=OUT.resolve()
 if p==root or not p.is_relative_to(root):raise ValueError('WRITE_OUTSIDE_PILOT: '+str(p))
 # Reject all symlink ancestors, even if an internal link resolves back inside OUT.
 q=Path(p)
 while q!=root:
  if q.is_symlink():raise ValueError('SYMLINK_OUTPUT: '+str(q))
  q=q.parent
 return p
def atomic(p,writer,binary=False):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=checked(p.with_name(p.name+'.tmp'))
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
  writer(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False))
def save_npz(p,**arrays):
 import numpy as np
 atomic(p,lambda f:np.savez_compressed(f,**arrays),True)
def textfile(p,s):atomic(p,lambda f:f.write(s))
def csvwrite(p,rows):
 if not rows:raise ValueError('Empty CSV has no registered schema')
 fields=list(dict.fromkeys(k for r in rows for k in r))
 def dump(f):
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 atomic(p,dump)
def event(kind,ledger='ledger.jsonl',**kw):
 p=checked(OUT/ledger);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:
  f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw),allow_nan=False)+'\n');f.flush()
def seed(domain,*args):return int.from_bytes(hashlib.sha256(json.dumps([VERSION,domain,*args]).encode()).digest()[:4],'little')
def resources():
 def run(args):
  p=subprocess.run(args,text=True,capture_output=True,timeout=20);return dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
 return dict(gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
  processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_cores=os.cpu_count(),
  loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text(),disk=run(['df','-B1',str(OUT)]))
def safety_checks():
 old=read(OLD/'registration.json');paths=[ASSET,Path(old['records'][0]['raw_path']),Path(old['records'][0]['condition_source']),OLD/'REPORT.md']
 for p in paths:
  try:checked(p)
  except ValueError:pass
  else:raise AssertionError('Source write not rejected before open')
 write(OUT/'write_guard_check.json',dict(status='PASS',rejected_before_open=list(map(str,paths)),source_target_distinct=True))
def verify(protected):
 for p,h in protected.items():
  if sha(p)!=h:raise ValueError('PROTECTED_INPUT_CHANGED: '+p)
 return len(protected)
