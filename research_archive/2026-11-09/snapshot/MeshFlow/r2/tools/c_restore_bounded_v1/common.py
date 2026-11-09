"""Independent CPU-only continuation; historical sources and results are read-only."""
from pathlib import Path
import csv, hashlib, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_c_restore_bounded_v1'
OLD=ROOT/'experiments/r2_c_restore_coupling_v1'
VERSION='R2_C_RESTORE_BOUNDED_AND_ORIENTATION_V1'

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
def registration():
 reg=read(OUT/'registration.json');verify(reg['runner_hashes']);return reg

def prepare():
 path=OUT/'registration.json'
 if path.exists():
  reg=registration();verify(reg['protected']);return reg
 from . import blend, orientation
 from r2.pilots.anchor_aware_denoiser_v1 import evaluation as E
 old=read(OLD/'registration.json');sr=read(OLD/'score_registration.json')
 verify(old['protected']);verify(old['runner_hashes']);verify(sr['protected'])
 OUT.mkdir(parents=True,exist_ok=True)
 protected=dict(old['protected']);protected.update(sr['protected'])
 for base in (OLD,ROOT/'r2/tools/c_restore_coupling_v1'):
  for p in base.rglob('*'):
   if p.is_file() and '__pycache__' not in str(p):protected[str(p)]=sha(p)
 for p in (ROOT/'AGENTS.md',ROOT/'native_t1/docs/control_metrics.md'):protected[str(p)]=sha(p)
 baselines=[E.canonical_identity(r) for r in rows(OLD/'post_index.csv') if r['method'] in ('D0_BASE','RBF')]
 assert len(baselines)==24
 for r in baselines:
  d=read(r['diagnostic_path']);assert d['status']=='PASS'
  assert d['identity']['post_sha256']==sha(r['post_path'])
  for p in [r['post_path'],r['denoised_path'],r['diagnostic_path'],d['quality']['pair_audit_path'],*d['quality']['files']]:protected[str(p)]=sha(p)
 runners=['common.py','run.py','evaluation.py','orientation.py','orientation_checks.py','blend.py','blend_checks.py']
 reg=dict(version=VERSION,status='REGISTERED',records=old['records'],baselines=baselines,
  fixed_L=old['fixed_L'],settings=old['settings'],postprocess_config=old['postprocess_config'],
  parent_registration_sha256=sha(OLD/'registration.json'),protected=protected,
  runner_hashes={str(CODE/n):sha(CODE/n) for n in runners},methods=['D0_BASE','RBF','BLEND'],
  blend_definition=blend.definition(),orientation_definition=orientation.definition(),
  budget=dict(new_model_forwards=0,training_updates=0,new_noise=0,new_generator_outputs=0,
   new_blend_outputs=12,new_GEO_C=12,new_POST_quality=12,reused_quality=24,
   orientation_POST_outputs=36,full_PRE_quality=0,comparison_figures=24),
  protocol=dict(selection='Same six USER_MIX parents x two frozen noises; no selection by new results',
   order='First orientation D0/RBF, then12BLEND, GEO_C, quality, BLEND orientation; all36 receive same solver',
   raw_and_old_outputs_readonly=True,C_bitwise_and_winding=True,geometry_no_new_parameters=True,
   quality_after_orientation='Reused orientation-independent scalar/geometry evidence; exact triangle vertex sets checked; no full rescoring of flip-only outputs',
   graph_after_orientation='Existing legal pair classifications reused; only winding and constrained feasibility newly calculated',
   boundary='All original C source-topology open vertices; descriptive, no must-join interface registered',
   orientation_not_outward_or_manifold_certification=True,full_PRE_quality='NA_NOT_BUDGETED',
   workers='CPU measured first job; max2 workers and2threads each; GPU0',
   replay='Completed receipts reused after hashes; incomplete attempted scientific stage blocks, no automatic retry',
   stop='Fixed panel, full per-output evidence, visual review, report, small ZIP; no defaults/commit/push/scan'))
 write(OUT/'resources_before_experiment.json',resources())
 write(path,reg);event('REGISTERED',registration_sha256=sha(path),budget=reg['budget'])
 textfile(OUT/'README_RESUME.md','Run in WSL from /mnt/e/MeshFlow:\n\n`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.c_restore_bounded_v1.run --stage all`\n\nCPU only. Successful stages are hash-verified and reused. An incomplete attempted scientific stage blocks rather than silently replaying. Originals remain read-only. Final reporting/audit/packaging never dispatch models.\n')
 return reg
