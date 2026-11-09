"""Immutable fixed-panel registration and output-only file utilities."""
from pathlib import Path
import hashlib,json,time,shutil
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_multistep_denoise_v1'
PREV=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'
CODE=Path(__file__).parent
ASSET=PREV/'assets/chair_denoiser_0030000.pt'
EXPECTED='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
SCALE=.95*.3762
VERSION='r2_user_mix_repeated_frozen_denoiser_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(p):
 p=Path(p).resolve()
 if not p.is_relative_to(OUT.resolve()):raise ValueError('Output outside new experiment: '+str(p))
 return p
def write(p,value):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def event(name,**kw):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'denoise_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=name,**kw))+'\n')
def verify(reg):
 for path,expected in reg['protected'].items():
  if sha(path)!=expected:raise ValueError('Protected file changed: '+path)
 return len(reg['protected'])
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'registration.json'
 if path.exists():
  reg=read(path);verify(reg)
  for source,h in reg['runner_hashes'].items():
   if sha(source)!=h:raise ValueError('Registered runner changed')
  return reg
 old=read(PREV/'registration.json');records=[r for r in old['records'] if r['label']=='R2_USER_MIX']
 if len(records)!=12 or len({r['request_id'] for r in records})!=12:raise ValueError('Require exactly12 registered MIX outputs')
 protected=dict(old['protected'])
 # Verify inherited old-input expectations BEFORE extending new protection map.
 verify(dict(protected=protected))
 for file in PREV.rglob('*'):
  if file.is_file() and not file.is_relative_to(PREV/'assets'):protected[str(file)]=sha(file)
 protected[str(ROOT/'r2/tools/postprocess_avmr_pmr_v1/denoise_run.py')]=sha(ROOT/'r2/tools/postprocess_avmr_pmr_v1/denoise_run.py')
 for file in (CODE/'score.py',CODE/'gallery.py',ROOT/'r2/tools/postprocess_avmr_pmr_v1/run.py',ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py'):
  protected[str(file)]=sha(file)
 for r in records:
  rid=r['request_id'];base=PREV/'denoised/R2_USER_MIX'/rid
  proof=read(base/'receipt.json')
  for k in ['request_id','label','raw_sha256']:
   if proof[k]!=r[k]:raise ValueError('D1 identity mismatch')
  if sha(base/'denoised.npz')!=proof['output_sha256'] or proof['checkpoint_sha256']!=EXPECTED:raise ValueError('D1 asset mismatch')
  r['D1_denoised_path']=str(base/'denoised.npz');r['D1_denoised_sha256']=proof['output_sha256']
  r['D1_receipt_path']=str(base/'receipt.json');r['D1_post_path']=str(PREV/'outputs/R2_USER_MIX'/rid/'post.npz')
  r['D1_score_path']=str(PREV/'scores/R2_USER_MIX'/rid/'score.json')
 reg=dict(version=VERSION,status='REGISTERED',label='R2_USER_MIX',records=records,settings=old['settings'],protected=protected,
  runner_hashes={str(CODE/f):sha(CODE/f) for f in ('common.py','denoise.py','run.py')},
  stages=['RAW','D1','D3','D5'],new_rounds=[2,3,4,5],GEO_terminal_rounds=[3,5],
  budget=dict(existing_RAW=12,reused_D1=12,new_denoiser_forwards=48,new_generator_forwards=0,training_updates=0,new_noise=0,new_GEO_outputs=24,new_quality_outputs=24,render_PNG=24),
  coordinate_scale=SCALE,checkpoint_path=str(ASSET),checkpoint_sha256=EXPECTED,
  recurrence='X0=existing RAW; Xr=project_original_C(official_denoise(Xr-1)); reuse frozen X1; no GEO between rounds; one independent fixed GEO_C at each evaluated endpoint',
  precision=read(PREV/'denoise_registration.json')['precision'],postprocess_config=old['postprocess_config'],
  metrics='Same eight strict F-only A-VMR/PMR settings, frozen C2048 quality queries, actual POST free denominators and indexed topology',
  scope='Current USER_MIX only; six previously inspected DEV parents and two fixed noises; no other generator checkpoints',
  failure_policy='Stop on nonfinite/OOM; completed per-round receipts reusable; no silent replay of a started forward lacking receipt; no >48 attempts',
  stopping='Complete fixed D3/D5 plus baseline comparison and stop; no training/distillation/default switch/commit/push')
 write(path,reg)
 dst=OUT/'asset_audit/official_denoiser_source_manifest.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PREV/'asset_audit/official_denoiser_source_manifest.json',dst)
 event('REGISTERED',budget=reg['budget'],protected_files=len(protected),registration_sha256=sha(path))
 # Negative safety check is read-only: the forbidden path must reject before opening.
 try:checked(records[0]['raw_path'])
 except ValueError:write(OUT/'output_guard_check.json',dict(status='PASS',source_write_rejected=True))
 else:raise AssertionError('Unsafe output guard')
 return reg