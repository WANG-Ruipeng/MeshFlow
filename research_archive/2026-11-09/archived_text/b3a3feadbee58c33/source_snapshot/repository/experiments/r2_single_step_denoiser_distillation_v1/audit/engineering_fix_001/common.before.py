"""Isolated output protection, atomic artifacts, and fixed pilot accounting."""
from pathlib import Path
import csv, hashlib, io, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_single_step_denoiser_distillation_v1'
OLD=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'
CODE=Path(__file__).parent
ASSET=OLD/'assets/chair_denoiser_0030000.pt'
EXPECTED='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
SCALE=.95*.3762
SIGMA=.02
ARMS=('S8_W512_KD',)
VERSION='R2_SINGLE_STEP_DENOISER_DISTILLATION_V1'

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

ANCHOR=ROOT/'experiments/r2_anchor_aware_denoiser_v1'
GENERATOR_EXPECTED='8c1287b13541b59fa7b1e70974a70b06fbe0d67160449c225e5bdc7fdd0897cd'
GENERATOR_STATE_EXPECTED='1ee83ae12c370ddaf21c7aa83aabd5cf44692d09eba345464b3c6b5bb943cd89'
STUDENT='S8_W512_KD'
GPU_SECONDS_MAX=14400
LIMITS={'generator_F':12450,'teacher_target_F':1211,'teacher_preflight_F':8,'teacher_benchmark_F':228,
        'student_train_F':32000,'student_train_B':32000,'student_updates':4000,
        'student_diagnostic_F':165,'student_eval_F':66,'student_benchmark_F':228,
        'student_preflight_F':32,'student_preflight_B':16,'student_discarded_updates':2}
def runtime(threads=4):
 import torch
 torch.set_num_threads(threads)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
def memory():
 import torch
 return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
 allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
def state_hash(state):
 h=hashlib.sha256()
 for k,t in state.items():
  h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng_hash():
 import torch
 d={'cpu':torch.get_rng_state()}
 if torch.cuda.is_initialized():d['cuda']=torch.cuda.get_rng_state()
 return {k:ah(t.cpu().numpy()) for k,t in d.items()}
class Phase:
 """One owner per GPU phase; all scientific attempts counted before execution."""
 def __init__(self,name,estimated_mib=2500):
  self.name=name;self.path=OUT/'gpu_phases'/f'{name}.json';self.start=time.monotonic();self.counts={};self.pending=None
  prior=[read(p) for p in (OUT/'gpu_phases').glob('*.json')]
  if self.path.exists():raise RuntimeError('Previously attempted phase: verify completion/clean boundary; no automatic replay')
  if any(p['status']=='RUNNING' for p in prior):raise RuntimeError('Unclosed GPU phase')
  self.prior_seconds=sum(p.get('gpu_occupancy_seconds',0) for p in prior)
  self.prior_counts={k:sum(p.get('counts',{}).get(k,0) for p in prior) for k in LIMITS}
  if self.remaining()<30:raise RuntimeError('GPU budget exhausted')
  r=resources();write(OUT/'resources'/f'{name}_before.json',r)
  line=r['gpu']['stdout'].strip().splitlines()[0].split(',')
  total,free=float(line[1]),float(line[3]);headroom=max(2048.,.1*total)
  if free-headroom<estimated_mib:raise RuntimeError('RESOURCE_BLOCKED: insufficient measured memory headroom')
  self.dispatch=dict(workers=1,headroom_mib=headroom,estimated_stage_mib=estimated_mib,reason='User fixed one GPU worker; models loaded sequentially')
  write(self.path,self.state('RUNNING'))
 def remaining(self):return GPU_SECONDS_MAX-self.prior_seconds-(time.monotonic()-self.start)
 def state(self,status,error=None):
  return dict(name=self.name,status=status,counts=self.counts,pending=self.pending,
   gpu_occupancy_seconds=time.monotonic()-self.start,prior_gpu_seconds=self.prior_seconds,error=error,dispatch=getattr(self,'dispatch',{}))
 def begin(self,kind,identity=None):
  if self.pending is not None:raise RuntimeError('Previous model operation not closed')
  if kind not in LIMITS:raise ValueError('Unregistered operation '+kind)
  if self.counts.get(kind,0)+self.prior_counts[kind]>=LIMITS[kind]:raise RuntimeError('Count limit '+kind)
  if self.remaining()<10:raise RuntimeError('GPU time limit')
  self.counts[kind]=self.counts.get(kind,0)+1
  self.pending=dict(kind=kind,call=self.counts[kind],identity=identity)
  event('OP_STARTED',ledger='gpu_calls.jsonl',phase=self.name,**self.pending)
 def end(self):
  if self.pending is None:raise RuntimeError('Operation not started')
  event('OP_COMPLETE',ledger='gpu_calls.jsonl',phase=self.name,**self.pending);self.pending=None
 def call(self,kind,fn,identity=None):
  self.begin(kind,identity)
  value=fn()
  self.end()
  return value
 def finish(self,status='COMPLETE',error=None):
  import torch
  if torch.cuda.is_initialized():torch.cuda.synchronize()
  if status=='COMPLETE' and self.pending is not None:raise RuntimeError('Cannot complete unclosed model call')
  write(self.path,self.state(status,error));write(OUT/'resources'/f'{self.name}_after.json',resources())
