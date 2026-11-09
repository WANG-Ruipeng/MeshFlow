"""Isolated fixed-budget student S1/S3/S5 repetition experiment."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_student_multistep_denoise_v1'
DIST=ROOT/'experiments/r2_single_step_denoiser_distillation_v1'
OLD=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'
MULTI=ROOT/'experiments/r2_multistep_denoise_v1'
STUDENT=DIST/'checkpoints/student_inference.pt'
EXPECTED='2b7bd5536ec9c4b32ca36d3d8c32c7d0510b74eddcfd73e23287fdafce38c34d'
SCALE=.95*.3762
VERSION='R2_STUDENT_MULTISTEP_DENOISE_V1'
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
 if p==OUT.resolve() or not p.is_relative_to(OUT.resolve()):raise ValueError('Write outside new experiment: '+str(p))
 return p
def atomic(p,fn,binary=False):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
  fn(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False))
def textfile(p,s):atomic(p,lambda f:f.write(s))
def save_npz(p,**a):
 import numpy as np
 atomic(p,lambda f:np.savez_compressed(f,**a),True)
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 def dump(f):
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 atomic(p,dump)
def event(kind,**fields):
 p=checked(OUT/'denoise_ledger.jsonl');p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:
  f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**fields),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def resources():
 def run(a):
  r=subprocess.run(a,text=True,capture_output=True,timeout=20);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
 return dict(gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_cores=os.cpu_count(),loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text(),cpu_processes=run(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu']),disk=run(['df','-B1',str(OUT)]))
def verify(reg):
 for p,h in {**reg['protected'],**reg['runner_hashes']}.items():
  if sha(p)!=h:raise ValueError('Frozen source/input changed: '+p)
 return len(reg['protected'])
def prepare():
 path=OUT/'registration.json'
 if path.exists():
  reg=read(path);verify(reg);return reg
 OUT.mkdir(parents=True,exist_ok=True)
 old=read(OLD/'registration.json');records=[];protected={}
 def protect(p):
  p=Path(p)
  if not p.is_file():raise FileNotFoundError(p)
  protected[str(p)]=sha(p)
 for base in (OLD,DIST,MULTI):
  for f in base.glob('*.json'):protect(f)
  if (base/'REPORT.md').exists():protect(base/'REPORT.md')
 for f in (DIST/'evaluation').glob('*.csv'):protect(f)
 for f in MULTI.glob('*.csv'):protect(f)
 for f in (OLD/'quality_per_output.csv',OLD/'post_index.csv'):protect(f)
 protect(STUDENT)
 if protected[str(STUDENT)]!=EXPECTED:raise ValueError('Wrong student checkpoint')
 for orig in old['records']:
  if orig['label']!='R2_USER_MIX':continue
  r=dict(orig);rid=r['request_id']
  base=DIST/'inference/S8_W512_KD/R2_USER_MIX'/rid
  s1=base/'denoised.npz';receipt=base/'denoised.json'
  if not receipt.exists():
   js=list(base.glob('*.json'))
   if len(js)!=1:raise ValueError('S1 receipt ambiguous')
   receipt=js[0]
  sr=read(receipt)
  if sr['output_sha256']!=sha(s1) or sr['raw_sha256']!=r['raw_sha256'] or sr['student_checkpoint_sha256']!=EXPECTED:raise ValueError('S1 identity mismatch')
  r.update(S1_denoised_path=str(s1),S1_denoised_sha256=sha(s1),S1_receipt_path=str(receipt),S1_post_path=str(DIST/'evaluation/outputs/STUDENT_GEO/R2_USER_MIX'/rid/'post.npz'),S1_score_path=str(DIST/'evaluation/scores/STUDENT_GEO/R2_USER_MIX'/rid/'score.json'),identity_post_path=str(DIST/'evaluation/outputs/IDENTITY_GEO/R2_USER_MIX'/rid/'post.npz'),D1_post_path=str(OLD/'outputs/R2_USER_MIX'/rid/'post.npz'))
  for key in ('raw_path','condition_source','input_path','S1_denoised_path','S1_receipt_path','S1_post_path','S1_score_path','identity_post_path','D1_post_path'):protect(r[key])
  for key in ('raw','condition_source','input'):
   pk={'raw':'raw_path','condition_source':'condition_source','input':'input_path'}[key]
   if protected[r[pk]]!=r[key+'_sha256']:raise ValueError('Original identity mismatch')
  for stage in ('STUDENT_GEO','IDENTITY_GEO','D0_GEO'):
   for sub in ('outputs','scores','diagnostics'):
    folder=DIST/'evaluation'/sub/stage/'R2_USER_MIX'/rid
    for f in folder.rglob('*'):
     if f.is_file() and f.suffix in ('.json','.csv','.npz'):protect(f)
  records.append(r)
 if len(records)!=12 or len({r['parent_id'] for r in records})!=6:raise ValueError('Wrong panel')
 runners=[CODE/x for x in ('common.py','denoise.py','run.py')]
 runners+=list((ROOT/'r2/pilots/single_step_denoiser_distillation_v1').glob('*.py'))
 runners+=list((OLD/'source_snapshot/official_denoiser_db86bc3').rglob('*.py'))
 reg=dict(version=VERSION,status='REGISTERED',records=records,settings=old['settings'],postprocess_config=old['postprocess_config'],student_checkpoint_path=str(STUDENT),student_checkpoint_sha256=EXPECTED,scale=SCALE,protected=protected,runner_hashes={str(p):sha(p) for p in runners},budget=dict(student_new_forwards=48,teacher_forwards=0,generator_forwards=0,training_updates=0,new_noise=0,new_GEO_C=24,new_quality_scores=24,reused_S1=12),protocol=dict(rounds_new=[2,3,4,5],endpoints=[1,3,5],C_original_restored_every_round=True,intermediate_GEO_C=0,final_GEO_C_per_endpoint=1,full_N_every_round=True,FP32_residual_and_scale=True,BF16_forward=True,deterministic=True,no_auto_retry=True,main_reference='S1',selection='All fixed 12 USER_MIX, no result-based filtering'))
 rejected=[]
 for f in (STUDENT,Path(records[0]['raw_path']),DIST/'REPORT.md'):
  try:checked(f)
  except ValueError:rejected.append(str(f))
  else:raise AssertionError('Output guard failed')
 write(OUT/'write_guard_check.json',dict(status='PASS',rejected_before_open=rejected))
 write(OUT/'resources_before_experiment.json',resources());write(path,reg)
 return reg
