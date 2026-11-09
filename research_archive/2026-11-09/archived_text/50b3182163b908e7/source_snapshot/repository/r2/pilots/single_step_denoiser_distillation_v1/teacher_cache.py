"""One eval-only official teacher query per registered training/validation/real DEV input."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from . import common as C
from .model import load_teacher,effective_proposal,tensor_state_sha256

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(row,b,proposal_d,residual_d,proposal_model,*,reused=False,source=None):
 n,k=int(row['N']),int(row['K']);known=b['known_mask'];original=b['C']
 output=proposal_model.copy();output[known]=original
 if output[known].tobytes()!=original.tobytes():raise AssertionError('Teacher C contract')
 target=C.checked(row['teacher_cache_path'])
 if target.exists():raise RuntimeError('Teacher target exists without verified completion')
 C.save_npz(target,input_d=b['input_d'],proposal_T_d=proposal_d,r_T_d=residual_d,
  proposal_T_model=proposal_model,output=output,C=original,known_mask=known,N=np.array(n),K=np.array(k))
 rec=dict(status='PASS',sample_id=row['sample_id'],domain=row['domain'],input_path=row['input_path'],
  input_sha256=C.sha(row['input_path']),path=str(target),sha256=C.sha(target),N=n,K=k,
  teacher_sha256=C.EXPECTED,reused=reused,source=source,FP32_effective_residual=True,
  known_C_byte_exact=True,teacher_forwards=0 if reused else 1,no_rng_consumed=True)
 C.write(target.with_suffix('.json'),rec);return rec
def reuse_synthetic(rows):
 records=[]
 for row in rows:
  b=arrays(row['input_path']);p=Path(row['teacher_reuse_path'])
  if C.sha(p)!=row['teacher_reuse_sha256']:raise ValueError('Exact synthetic teacher changed')
  prior=arrays(p);proposal=prior[row.get('teacher_proposal_key','proposal_denoiser')].astype(np.float32).reshape(row['N'],3,3)
  xd=b['input_d']
  r=(proposal-xd).astype(np.float32)
  # Exact old d-domain proposal is the KD target; model-space proposal retained
  # from its own output if available. CPU inverse divide uses the same FP32 scalar.
  pm=prior['proposal'].astype(np.float32)
  target=Path(row['teacher_cache_path'])
  if target.with_suffix('.json').exists():
   rec=C.read(target.with_suffix('.json'))
   if C.sha(target)!=rec['sha256'] or rec['input_sha256']!=C.sha(row['input_path']):raise ValueError('Reused teacher identity drift')
  else:rec=save(row,b,proposal,r,pm,reused=True,source=dict(path=str(p),sha256=C.sha(p)))
  records.append(rec)
 return records
def main():
 C.runtime();d=C.read(C.OUT/'manifest/data.json')
 done=C.OUT/'teacher_cache/completion.json'
 if done.exists():
  rec=C.read(done)
  for row in rec['rows']:
   if C.sha(row['path'])!=row['sha256'] or C.sha(row['input_path'])!=row['input_sha256']:raise ValueError('Teacher cache changed')
  print('SKIP_VERIFIED_COMPLETE teacher targets');return
 if C.read(C.OUT/'preflight/teacher_completion.json')['status']!='PASS':raise RuntimeError('Teacher preflight required')
 rows=d['train_inputs']+d['internal_val']+d['dev_real']
 if len(rows)!=1193:raise ValueError('Teacher1193 target count')
 if any(Path(r['teacher_cache_path']).exists() for r in rows):raise RuntimeError('Prior target calls detected; no silent replay')
 records=reuse_synthetic(d['dev_synthetic'])
 phase=C.Phase('teacher_targets',estimated_mib=2500)
 try:
  tick=time.monotonic();teacher,spec=load_teacher()
  parameter_before=tensor_state_sha256(teacher.base.state_dict())
  teacher.cuda().eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
  C.write(C.OUT/'teacher_cache/load.json',dict(spec=spec,parameter_state_sha256=parameter_before,load_seconds=time.monotonic()-tick))
  for index,row in enumerate(rows):
   if phase.remaining()<15:raise RuntimeError('GPU_TIME_LIMIT before teacher target')
   b=arrays(row['input_path']);n,k=int(row['N']),int(row['K'])
   if b['input_d'].shape!=(n,3,3) or b['input_d'].dtype!=np.float32 or int(b['known_mask'].sum())!=k:raise ValueError('Teacher input contract')
   # RAW model-to-d scaling must match the frozen actual GPU input.
   xd=torch.from_numpy(b['input_d'].reshape(1,n,9)).cuda()
   if row['domain']=='real':
    scaled=torch.from_numpy(b['input_model'].reshape(1,n,9)).cuda()*C.SCALE
    if not torch.equal(scaled,xd):raise AssertionError('Frozen real input_d differs from actual FP32 scale')
   before=C.rng_hash()
   def forward():
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     return effective_proposal(teacher,xd)
   proposal,residual=phase.call('teacher_target_F',forward,row['sample_id'])
   if not torch.isfinite(proposal).all() or C.rng_hash()!=before:raise AssertionError('Teacher numerical/RNG contract')
   pm=(proposal/C.SCALE).reshape(n,3,3).cpu().numpy()
   if 'historical_teacher_model_space_cache' in row:
    old=arrays(row['historical_teacher_model_space_cache'])
    expected=old['official_unconstrained_proposal'].reshape(n,3,3)
    if pm.tobytes()!=expected.tobytes():raise AssertionError('D0 historical proposal does not exactly replay')
   rec=save(row,b,proposal.reshape(n,3,3).cpu().numpy(),residual.reshape(n,3,3).cpu().numpy(),pm)
   records.append(rec)
   if index==0:C.write(C.OUT/'resources/teacher_target_first.json',dict(memory=C.memory(),resources=C.resources()))
   if index==0 or (index+1)%100==0:
    print(json.dumps(dict(stage='teacher',done=index+1,total=1193,seconds=time.monotonic()-tick)),flush=True)
  after=tensor_state_sha256(teacher.base.state_dict())
  if after!=parameter_before or any(p.grad is not None for p in teacher.parameters()):raise AssertionError('Teacher trained or changed')
  C.write(done,dict(status='COMPLETE',rows=records,new_teacher_forwards=1193,exact_synthetic_reuses=18,
   teacher_parameter_before=parameter_before,teacher_parameter_after=after,teacher_gradients=0,
   peak=C.memory(),seconds=time.monotonic()-tick,source_sha256=C.sha(__file__)))
  phase.finish()
 except BaseException as exc:
  C.write(C.OUT/'teacher_cache/failure.json',dict(error=repr(exc),saved=len(records),no_automatic_retry=True))
  phase.finish('FAILED',repr(exc));raise
if __name__=='__main__':main()
