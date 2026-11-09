"""Recoverable orchestration; each completed GPU phase is checked and skipped."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import common as C
def prepare():
 C.OUT.mkdir(parents=True,exist_ok=True);C.safety_checks()
 path=C.OUT/'registration.json'
 if path.exists():
  reg=C.read(path);C.verify(reg['protected'])
  for p,h in reg['training_source_hashes'].items():
   if C.sha(p)!=h:raise ValueError('Frozen training code changed: '+p)
  return reg
 data=C.read(C.OUT/'data_audit/training_whitelist.json')
 if len(data['parent_ids'])<8:raise RuntimeError('DATA_CONTRACT_BLOCKED: fewer than eight parents')
 if not data['rows'] or not any(r['source_supported_edge_count']>0 for r in data['rows']):raise RuntimeError('DATA_CONTRACT_BLOCKED: no supported interface')
 manifest=C.read(C.OUT/'stream/manifest.json');synthetic=C.read(C.OUT/'stream/synthetic/manifest.json')
 old=C.read(C.OLD/'registration.json');protected=dict(old['protected'])
 protected.update(C.read(C.OUT/'data_audit/protected_inputs.json')['before'])
 for p in [C.ASSET,C.OLD/'registration.json',C.OLD/'strict_load.json',C.OLD/'asset_audit/denoiser_asset_registration.json',C.OLD/'denoise_registration.json',C.OLD/'post_index.csv',C.OLD/'REPORT.md']:
  protected[str(p)]=C.sha(p)
 for row in old['records']:
  for k in ('raw_path','input_path','condition_source'):
   protected[row[k]]=C.sha(row[k])
  for sub in ('denoised','outputs','scores'):
   base=C.OLD/sub/row['label']/row['request_id']
   for p in base.rglob('*'):
    if p.is_file():protected[str(p)]=C.sha(p)
 for base in (C.OLD/'source_snapshot/official_denoiser_db86bc3',C.ROOT/'r2/tools/postprocess_avmr_pmr_v1'):
  for p in base.rglob('*'):
   if p.is_file() and '__pycache__' not in str(p):protected[str(p)]=C.sha(p)
 if protected[str(C.ASSET)]!=C.EXPECTED:raise ValueError('Official weight differs')
 training_code=['common.py','gpu_run.py','model.py','data_stream.py']
 reg=dict(version=C.VERSION,status='REGISTERED',arms=list(C.ARMS),baseline='D0_OFFICIAL',records=old['records'],settings=old['settings'],postprocess_config=old['postprocess_config'],
  protected=protected,training_source_hashes={str(C.CODE/name):C.sha(C.CODE/name) for name in training_code},whitelist_sha256=C.sha(C.OUT/'data_audit/training_whitelist.json'),stream_sha256=C.sha(C.OUT/'stream/manifest.json'),synthetic_manifest_sha256=C.sha(C.OUT/'stream/synthetic/manifest.json'),
  parameters=dict(updates_per_arm=1000,batch=8,microbatch=1,different_parents=8,clean=4,width_augmented=4,width=[.9,1.1],width_transform='multiply x coordinate FP32 about origin, C/Y jointly',lr=1e-5,betas=[.9,.95],weight_decay=0,gradient_clip=1,optimizer='fresh AdamW foreach=False',new_EMA=False,scheduler=False,train_mode=True,face_count_dropout=.1,activation_checkpointing='official non-reentrant True; preserve_rng_state=True',precision='BF16 forward, FP32 parameters/sum/loss; strict deterministic; TF32 off; SDPA math; unpadded mask=None'),
  noise=dict(sigma=.02,units='denoiser coordinates',provenance='strict published Chair checkpoint config.model.fixed_std',free_corner_copies_independent=True,known_clean=True,shared_between_arms=True),
  objective='sum free coordinate squared errors / (9 * sum batch free face counts); no other terms',role='1024D zero e_known added in activation dtype after post_quant_linear, trainable anchor only; no-role zero buffer and zero marker',
  budget=dict(formal_updates=2000,formal_exposures=16000,formal_logical_forward=16000,formal_logical_backward=16000,preflight_forward_max=32,preflight_backward_max=16,discarded_optimizer_updates_max=2,real_forward=96,synthetic_forward=54,generator_forward=0,generator_backward=0,Euler=0,OT=0,GPU_occupancy_seconds_max=10800,GPU_workers=1),
  checkpoint_policy='500 recovery only,1000 unique endpoint; no best selection',failure_policy='No silent replay of unresolved model/optimizer calls. Stop and preserve failures on OOM/nonfinite/data/asset problems. Three-hour cumulative GPU process cap.',
  current_primary_generator='R2_USER_MIX',limitations='Six repeatedly inspected USER DEV conditions, no independent test. Source-supported interface is descriptive reference, not required user interface.',
  output_guard='All runtime outputs resolve inside current OUT and reject lexical symlink ancestors; atomic writes; old inputs hashes before/after')
 C.verify(protected);C.write(path,reg);C.event('REGISTERED',registration_sha256=C.sha(path),protected_files=len(protected),training_parents=len(data['parent_ids']),training_conditions=len(data['rows']),budget=reg['budget'])
 return reg
def gpu(stage,arm=None):
 cmd=[sys.executable,'-B',str(C.CODE/'gpu_run.py'),stage]
 if arm:cmd+=['--arm',arm]
 result=subprocess.run(cmd)
 if result.returncode:raise RuntimeError('GPU child failed, no automatic retry: '+str(result.returncode))
def invoke(args):
 result=subprocess.run([sys.executable,'-B',*args],cwd=C.ROOT)
 if result.returncode:raise RuntimeError('Stage failed without automatic retry: '+str(args))
def inference():
 jobs=[('synthetic','D0_OFFICIAL'),('real','D_CONT_NO_ROLE'),('synthetic','D_CONT_NO_ROLE'),('real','D_ANCHOR'),('synthetic','D_ANCHOR')]
 for stage,arm in jobs:
  invoke(['-m','r2.pilots.anchor_aware_denoiser_v1.inference',stage,'--arm',arm])
  if C.read(C.OUT/'eval_completion'/f'{stage}_{arm}.json')['status']!='COMPLETE':raise RuntimeError('Evaluation partial; retain actual completion and stop')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','preflight','train','inference','score','gallery','finalize','all'],default='all');a=p.parse_args()
 prepare()
 if a.stage=='all' and (C.OUT/'finalization_completion.json').exists():
  invoke([str(C.CODE/'finalize.py')]);return
 if a.stage=='prepare':return
 if a.stage in ('preflight','all'):gpu('preflight')
 if a.stage in ('train','all'):
  for arm in C.ARMS:
   gpu('train',arm)
   if C.read(C.OUT/'training'/arm/'completion.json')['status']!='COMPLETE':raise RuntimeError('Partial arm; preserve progress and do not start next')
 if a.stage in ('inference','all'):inference()
 if a.stage in ('score','all'):invoke([str(C.CODE/'evaluation.py'),'--stage','all'])
 if a.stage in ('gallery','all'):invoke(['-m','r2.pilots.anchor_aware_denoiser_v1.gallery'])
 if a.stage in ('finalize','all'):
  script=C.CODE/'finalize.py'
  if not script.exists():raise RuntimeError('Numerical stages prepared; final report writer not yet available')
  invoke([str(script)])
if __name__=='__main__':main()
