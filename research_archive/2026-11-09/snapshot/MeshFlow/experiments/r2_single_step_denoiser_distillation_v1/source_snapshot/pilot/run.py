"""Freeze data identities and dispatch the bounded single-student experiment."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from . import common as C
def register():
 p=C.OUT/'registration.json'
 if p.exists():
  r=C.read(p);C.verify(r['protected'])
  if C.sha(C.OUT/'manifest/data.json')!=r['data_manifest_sha256']:raise ValueError('Frozen data manifest changed')
  return r
 d=C.read(C.OUT/'manifest/data.json')
 if (len(d['train_inputs']),len(d['internal_val']),len(d['dev_real']),len(d['dev_synthetic']),len(d['generation_plan']))!=(1120,25,48,18,249):raise ValueError('Data budget mismatch')
 C.safety_checks()
 protected=dict(C.read(C.OLD/'registration.json')['protected'])
 protected.update(C.read(C.OUT/'manifest/protected_sources.json'))
 protected[str(C.ASSET)]=C.EXPECTED
 generator=C.ROOT/'experiments/r2_user_mix_from_official_v1/inference/R2_USER_MIX_generator_step6000.pt'
 protected[str(generator)]=C.GENERATOR_EXPECTED
 for p0 in (C.ANCHOR/'data_audit/training_whitelist.json',C.ANCHOR/'stream/synthetic/manifest.json',C.OUT/'USER_TASK.md'):
  protected[str(p0)]=C.sha(p0)
 for r0 in d['train_inputs']+d['internal_val']+d['dev_real']+d['dev_synthetic']:
  for key in ('source_npz','raw_path'):
   path=r0.get(key)
   if path and Path(path).exists() and not Path(path).resolve().is_relative_to(C.OUT.resolve()):protected[path]=C.sha(path)
 for root in (C.ROOT/'r2/src',C.ROOT/'r2/pilots/user_mix_from_official_v1',C.ROOT/'r2/pilots/anchor_aware_denoiser_v1'):
  for src in root.rglob('*.py'):protected[str(src)]=C.sha(src)
 C.verify(protected)
 r=dict(task=C.VERSION,status='REGISTERED',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  protected=protected,data_manifest_sha256=C.sha(C.OUT/'manifest/data.json'),train_stream_sha256=C.sha(d['train_stream_path']),
  task_sha256=C.sha(C.OUT/'USER_TASK.md'),student=C.STUDENT,architecture=dict(layers=8,width=512,heads=8,max_parameters=50000000,role=False,readout=False,Geo_branch=False),
  student_initialization_seed=C.seed('student_initialization'),student_initialization='random official rules, final residual layer zero; no teacher tensor slicing',
  teacher=dict(path=str(C.ASSET),sha256=C.EXPECTED,EMA=True,parameters=455592963,state_keys=296,source_commit='db86bc32423c316c00bcc2f645a26243e8010f20',scale=C.SCALE),
  generator=dict(path=str(generator),sha256=C.GENERATOR_EXPECTED,state_sha256=C.GENERATOR_STATE_EXPECTED,Euler=50,precision='BF16 / FP32 integration',train=False),
  optimizer=dict(updates=4000,batch=8,microbatch=1,parents_per_batch=8,real=4,synthetic=4,lr_start=1e-6,lr_peak=1e-4,warmup_updates=100,cosine_end=1e-5,betas=[.9,.95],weight_decay=0,clip=1,new_EMA=False),
  lr_convention='Update1 uses1e-6, update100 uses1e-4 (99 intervals); update101 begins cosine, update4000 uses1e-5.',
  KD='sum free FP32 effective residual squared error / total batch free scalars; no Y target, no auxiliary loss',
  runtime=dict(forward='BF16',parameters='FP32',residual_add_sub='FP32',TF32=False,SDPA='math',strict_deterministic=True,student_dropout=0,student_activation_checkpointing=False),
  monitoring=dict(steps=[0,500,1000,2000,4000],internal_val=25,fixed_training=8,total_forwards=165),
  budget=dict(counters=C.LIMITS,GPU_seconds=C.GPU_SECONDS_MAX,GPU_workers=1),
  evaluation=dict(real_DEV=48,synthetic_DEV=18,POST_methods=['IDENTITY_GEO','D0_GEO','STUDENT_GEO'],POST=144,RAW_reference=48,GEO_radius=.015/C.SCALE,metric_definitions='unchanged original scorers and8thresholds, full per-output',unique_GT=False),
  limitations='Repeated6-parent DEV, internal validation same25training parents/new noise; small pilot not independent generalization.',
  stop='Fixed4000endpoint or real numerical/resource/asset failure; no architecture/seed/loss scan, no automatic default/commit/push')
 C.write(p,r);C.write(C.OUT/'resources/registration.json',C.resources());C.event('REGISTERED',sha256=C.sha(p),budget=r['budget'])
 return r
def call(stage,module,*args):
 C.event('STAGE_STARTED',stage=stage)
 tick=time.monotonic()
 p=subprocess.run([sys.executable,'-B','-m','r2.pilots.single_step_denoiser_distillation_v1.'+module,*args],cwd=C.ROOT)
 C.event('STAGE_FINISHED',stage=stage,returncode=p.returncode,seconds=time.monotonic()-tick)
 if p.returncode:raise SystemExit(p.returncode)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','register','generate','teacher-preflight','cache-teacher','student-preflight','train','infer','benchmark','score','gallery','finalize','all']);a=p.parse_args()
 if a.stage=='prepare':
  result=subprocess.run([sys.executable,'-B',str(C.CODE/'prepare_data.py')],cwd=C.ROOT)
  if result.returncode:raise SystemExit(result.returncode)
  return
 register()
 if a.stage=='register':return
 stages=[('generate','generate',[]),('teacher-preflight','preflight',['--stage','teacher']),('cache-teacher','teacher_cache',[]),('student-preflight','preflight',['--stage','student']),('train','train',[]),('infer','inference',[]),('benchmark','benchmark',[]),('score','evaluation',['--stage','all']),('gallery','gallery',[]),('finalize','finalize',[])]
 if a.stage=='all' and (C.OUT/'completion.json').exists():call('verify-completion','finalize','--verify');return
 for name,module,args in stages:
  if a.stage in ('all',name):call(name,module,*args)
if __name__=='__main__':main()
