"""Frozen R2_USER_MIX Euler50 for exactly 224 training and 25 validation RAWs."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import gc,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from . import common as C

class SamplingBudget:
 def __init__(self,phase,row):self.phase=phase;self.row=row;self.f=0;self.reserved=False
 def reserve(self,kind,context=None):
  if kind!='rollout_attempt' or self.reserved:raise ValueError('One rollout per input')
  self.reserved=True;C.event('RAW_STARTED',ledger='generation/ledger.jsonl',sample_id=self.row['sample_id'])
 def call(self,kind,fn,context=None):
  if kind!='sampling_forward' or not self.reserved or self.f>=50:raise ValueError('Euler50 budget')
  self.f+=1
  return self.phase.call('generator_F',fn,dict(sample_id=self.row['sample_id'],euler_step=self.f-1))
 def event(self,kind,**kw):
  C.event(kind,ledger='generation/ledger.jsonl',sample_id=self.row['sample_id'],**kw)

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 C.runtime()
 manifest=C.read(C.OUT/'manifest/data.json');rows=manifest['generation_plan']
 if len(rows)!=249 or len({r['sample_id'] for r in rows})!=249:raise ValueError('Registered RAW count')
 done=C.OUT/'generation/completion.json'
 if done.exists():
  d=C.read(done)
  for r in d['rows']:
   for pkey,hkey in (('raw_path','raw_sha256'),('input_path','input_sha256')):
    if C.sha(r[pkey])!=r[hkey]:raise ValueError('Completed generation changed')
  print('SKIP_VERIFIED_COMPLETE generator');return
 phase_name='generate_training_raw'
 if (C.OUT/'generation/ledger.jsonl').exists():
  repair=C.read(C.OUT/'audit/engineering_fix_001/receipt.json')
  if repair['actual_forward']!=0:raise RuntimeError('Prior model invocation; no replay')
  phase_name=repair['new_phase']
 phase=C.Phase(phase_name,estimated_mib=3500);records=[]
 try:
  path=C.ROOT/'experiments/r2_user_mix_from_official_v1/inference/R2_USER_MIX_generator_step6000.pt'
  if C.sha(path)!=C.GENERATOR_EXPECTED:raise ValueError('Required R2_USER_MIX checkpoint missing/different')
  spec=importlib.util.spec_from_file_location('_distill_readonly_generator',C.ROOT/'r2/pilots/user_mix_from_official_v1/trainer.py')
  trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
  from meshflow_control.sampling import sample
  model,identity=trainer.load_generator(path,device='cuda',expected_sha256=C.GENERATOR_EXPECTED)
  if identity['generator_state_sha256']!=C.GENERATOR_STATE_EXPECTED or identity['global_step']!=6000:raise ValueError('Generator tensor identity')
  model.eval().requires_grad_(False);C.write(C.OUT/'generation/generator_identity.json',identity)
  tick=time.monotonic()
  for index,row in enumerate(rows):
   if phase.remaining()<60:raise RuntimeError('GPU_TIME_LIMIT before next complete rollout')
   if C.sha(row['generation_input_path'])!=row['generation_input_sha256']:raise ValueError('Frozen generation input changed')
   b=arrays(row['generation_input_path'])
   if set(b)-{'C','z','N','K','known_mask','valid_mask'}:raise ValueError('Generation input contains unauthorized target fields')
   n,k=int(b['N']),int(b['K'])
   if (n,k)!=(row['N'],row['K']) or b['C'].dtype!=np.float32 or b['z'].dtype!=np.float32:raise ValueError('Input contract')
   budget=SamplingBudget(phase,row);before=C.rng_hash();start=time.monotonic()
   raw,trajectory,audit=sample(model,b['C'],n,seed=row.get('noise_seed'),z=b['z'],budget=budget,out=None,context={'sample_id':row['sample_id']})
   raw=raw.reshape(n,3,3);trajectory=trajectory.reshape(51,n,3,3)
   if budget.f!=50 or C.rng_hash()!=before:raise AssertionError('Unexpected sampling count or RNG')
   if any(t[:k].tobytes()!=b['C'].tobytes() for t in trajectory):raise AssertionError('C not preserved')
   rawpath=C.checked(row['generated_raw_path']);inputpath=C.checked(row['input_path'])
   if rawpath.exists() or inputpath.exists():raise RuntimeError('Output already exists without closed receipt')
   C.save_npz(rawpath,raw=raw,C=b['C'],z=b['z'],trajectory=trajectory,N=np.array(n),K=np.array(k))
   xd=(torch.from_numpy(raw).cuda()*C.SCALE).cpu().numpy()
   known=np.arange(n)<k
   C.save_npz(inputpath,input_model=raw,input_d=xd,C=b['C'],known_mask=known,N=np.array(n),K=np.array(k))
   rec=dict(sample_id=row['sample_id'],domain=row['domain'],N=n,K=k,
    raw_path=str(rawpath),raw_sha256=C.sha(rawpath),input_path=str(inputpath),input_sha256=C.sha(inputpath),
    generator_file_sha256=C.GENERATOR_EXPECTED,generator_state_sha256=C.GENERATOR_STATE_EXPECTED,
    generation_input_path=row['generation_input_path'],generation_input_sha256=C.sha(row['generation_input_path']),
    actual_Gaussian_sha256=C.ah(b['z']),C_byte_exact=True,all_51_C_byte_exact=True,
    forwards=budget.f,no_rng_consumed=True,seconds=time.monotonic()-start,audit=audit)
   C.write(C.OUT/'generation/receipts'/f"{row['sample_id']}.json",rec);records.append(rec)
   C.event('RAW_COMPLETE',ledger='generation/ledger.jsonl',sample_id=row['sample_id'],input_sha256=rec['input_sha256'])
   if index==0:
    C.write(C.OUT/'resources/generation_first_complete_rollout.json',dict(resource=C.resources(),memory=C.memory(),audit=audit))
   if index==0 or (index+1)%10==0:
    C.write(C.OUT/'generation/progress.json',dict(completed=len(records),total=249,seconds=time.monotonic()-tick,remaining_GPU_seconds=phase.remaining(),memory=C.memory()))
    print(json.dumps(dict(stage='generate',done=len(records),total=249,seconds=time.monotonic()-tick)),flush=True)
  C.write(done,dict(status='COMPLETE',rows=records,rollouts=249,forwards=phase.counts.get('generator_F'),generator_training=0,seconds=time.monotonic()-tick))
  phase.finish()
 except BaseException as exc:
  C.write(C.OUT/'generation/failure.json',dict(error=repr(exc),completed=len(records),no_automatic_retry=True))
  phase.finish('FAILED',repr(exc));raise
if __name__=='__main__':main()
