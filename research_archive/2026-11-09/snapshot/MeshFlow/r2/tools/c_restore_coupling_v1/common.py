"""Isolated, fixed-budget C-restoration coupling pilot. Old assets are read-only."""
from pathlib import Path
import csv, hashlib, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_c_restore_coupling_v1'
OLD=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'
MULTI=ROOT/'experiments/r2_multistep_denoise_v1'
ASSET=OLD/'assets/chair_denoiser_0030000.pt'
EXPECTED='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
SCALE=.95*.3762
VERSION='R2_C_RESTORE_COUPLING_V1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(p):
 p=Path(p).absolute()
 if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('Symlink output')
 p=p.resolve()
 if not p.is_relative_to(OUT.resolve()) or p==OUT.resolve():raise ValueError('Write outside new experiment: '+str(p))
 return p
def atomic(p,fn,binary=False):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
  fn(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False))
def textfile(p,s):atomic(p,lambda f:f.write(s))
def save_npz(p,**arrays):
 import numpy as np
 atomic(p,lambda f:np.savez_compressed(f,**arrays),True)
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 def dump(f):
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 atomic(p,dump)
def event(kind,**fields):
 p=checked(OUT/'ledger.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:
  f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**fields),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def resources():
 def run(args):
  r=subprocess.run(args,text=True,capture_output=True,timeout=20);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
 return dict(gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_cores=os.cpu_count(),loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text(),cpu_processes=run(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']),disk=run(['df','-B1',str(OUT)]))
def verify(reg,weights=True):
 for p,h in {**reg['protected'],**reg['runner_hashes']}.items():
  if not weights and p==str(ASSET):continue
  if sha(p)!=h:raise ValueError('Frozen source/input changed: '+p)
 return len(reg['protected'])
def prepare():
 path=OUT/'registration.json'
 if path.exists():
  reg=read(path);verify(reg,weights=False);return reg
 OUT.mkdir(parents=True,exist_ok=True)
 import numpy as np
 old=read(MULTI/'registration.json');records=[];protected={}
 def protect(p):
  p=Path(p)
  if not p.is_file():raise FileNotFoundError(p)
  protected[str(p)]=sha(p)
 for p in (ASSET,OLD/'registration.json',OLD/'denoise_registration.json',OLD/'strict_load.json',MULTI/'registration.json',OLD/'asset_audit/official_denoiser_source_manifest.json'):protect(p)
 if protected[str(ASSET)]!=EXPECTED:raise ValueError('Wrong D0')
 for orig in old['records']:
  r=dict(orig);rid=r['request_id'];K,N=int(r['K']),int(r['N'])
  r.update(D0_denoised_path=r['D1_denoised_path'],D0_post_path=r['D1_post_path'],D3_denoised_path=str(MULTI/'denoised/D3'/rid/'denoised.npz'),D3_post_path=str(MULTI/'outputs/D3'/rid/'post.npz'))
  for key in ('raw_path','condition_source','input_path','D1_denoised_path','D1_receipt_path','D1_post_path','D1_score_path','D3_denoised_path','D3_post_path'):protect(r[key])
  for key in ('raw','condition_source','input','D1_denoised'):
   pk={'raw':'raw_path','condition_source':'condition_source','input':'input_path','D1_denoised':'D1_denoised_path'}[key]
   if protected[r[pk]]!=r[key+'_sha256']:raise ValueError('Original identity mismatch')
  with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(N,3,3);c=a['C']
  with np.load(r['condition_source'],allow_pickle=False) as a:
   if c.tobytes()!=a['C'].tobytes() or a['C_source_vertex_ids'].shape!=(K,3):raise ValueError('C source identity mismatch')
  with np.load(r['D0_denoised_path'],allow_pickle=False) as a:
   if a['output'][:K].tobytes()!=c.tobytes() or a['official_unconstrained_proposal'].shape!=(N,3,3):raise ValueError('D0 identity mismatch')
  for base,tag in ((OLD,'R2_USER_MIX'),(MULTI,'D3')):
   for sub in ('outputs','scores'):
    for p in (base/sub/tag/rid).rglob('*'):
     if p.is_file():protect(p)
  records.append(r)
 if len(records)!=12 or len({r['parent_id'] for r in records})!=6:raise ValueError('Wrong panel')
 for base in (OLD,MULTI):
  for p in base.glob('*.csv'):protect(p)
  protect(base/'REPORT.md')
 for base in (ROOT/'r2/tools/postprocess_avmr_pmr_v1',ROOT/'r2/tools/avmr_pmr_v1',ROOT/'r2/pilots/anchor_aware_denoiser_v1'):
  for p in base.glob('*.py'):protect(p)
 for p in (ROOT/'native_t1/postprocessing.py',ROOT/'native_t1/postprocess_cli.py'):protect(p)
 for p in (OLD/'source_snapshot/official_denoiser_db86bc3').rglob('*.py'):protect(p)
 L=next(s['threshold_model']/s['tau'] for s in old['settings'] if s['scale']=='fixed_L_fraction')
 runners=[CODE/n for n in ('common.py','run.py','denoise.py','geometry.py','geometry_checks.py')]
 reg=dict(version=VERSION,status='REGISTERED',records=records,settings=old['settings'],postprocess_config=old['postprocess_config'],fixed_L=L,scale=SCALE,checkpoint_path=str(ASSET),checkpoint_sha256=EXPECTED,protected=protected,runner_hashes={str(p):sha(p) for p in runners},budget=dict(new_D0_real_forwards=24,new_D0_synthetic_forwards=0,new_generator_forwards=0,training_updates=0,new_noise=0,new_GEO_C=24,new_POST_quality=24,real_requests=12,CPU_geometry_synthetic_cases=4,CPU_analytic_feedback_cases=3),methods=dict(RBF=dict(kernel='Wendland C2 compact',radius_L=.10,ridge_lambda=1e-6,weights='1/source-ID corner multiplicity; all proposal copies retained',system='(K+lambda*diag(1/w))*a=b',max_free_corner_shift_L=.02),FEEDBACK=dict(updates=2,eta=.5,max_working_C_corner_delta_L=.02,delta_arithmetic='FP64 accumulation/projection; FP32 working input',free_input='Original RAW every query',driving_error='Original C minus full D0 C proposal BEFORE restoration',working_C_corners='Independent corner deltas; internal query may split source copies',endpoint='Always query delta_2 and use that query free output; no best-iteration selection')),protocol=dict(main_reference='D0_BASE_POST',auxiliary='Existing D3, three total D0 calls',original_C_readonly=True,export_C_bitwise=True,internal_C_may_move=True,no_intermediate_GEO=True,deterministic=True,BF16_forward=True,FP32_parameters_residual_scale=True,no_auto_retry=True,selection='All12 fixed USER_MIX; no result-based filtering',PRE_metrics='Full existing threshold matching, distances, exact-coordinate topology; full PRE quality NA_NOT_BUDGETED',POST_metrics='Existing full quality and matching, valid edges/winding/reachable area; GEO fates',required_join_boundary='NA_NOT_REGISTERED; all C source open edges descriptive',synthetic='Four CPU geometry fixtures and three analytical feedback fixtures; no claim of real D0 synthetic recovery',stopping='Complete fixed panel, audit, report and small review ZIP; no default change/commit/scan'))
 write(OUT/'resources_before_experiment.json',resources())
 write(OUT/'asset_audit/official_denoiser_source_manifest.json',read(OLD/'asset_audit/official_denoiser_source_manifest.json'))
 for p in (ASSET,Path(records[0]['raw_path']),OLD/'REPORT.md'):
  try:checked(p)
  except ValueError:pass
  else:raise AssertionError('Write guard failed')
 write(OUT/'write_guard_check.json',dict(status='PASS',protected_outputs_rejected=3))
 write(path,reg);event('REGISTERED',registration_sha256=sha(path),budget=reg['budget'])
 return reg
