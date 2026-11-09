"""48 authorized student forwards; recurrence is restored pre-GEO model space."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import sys,time,json
from pathlib import Path
import numpy as np
import common as C
sys.path.insert(0,str(C.ROOT))
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 reg=C.prepare();rh=C.sha(C.OUT/'registration.json');expected={(r['request_id'],j) for r in reg['records'] for j in range(2,6)}
 ledger=C.OUT/'denoise_ledger.jsonl'
 events=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
 starts=[(e['request_id'],e['round']) for e in events if e['event']=='STUDENT_STARTED']
 ends=[(e['request_id'],e['round']) for e in events if e['event']=='STUDENT_COMPLETE']
 if len(starts)!=len(set(starts)) or len(ends)!=len(set(ends)) or set(starts)-expected or set(starts)!=set(ends):raise RuntimeError('Unclosed/duplicate attempt; no automatic replay')
 pending=[]
 for r in reg['records']:
  inp=Path(r['S1_denoised_path'])
  for j in range(2,6):
   out=C.OUT/'denoised'/f'S{j}'/r['request_id']/'denoised.npz';rp=out.with_name('receipt.json')
   if rp.exists():
    rec=C.read(rp)
    identity=dict(status='PASS',request_id=r['request_id'],round=j,raw_sha256=r['raw_sha256'],input_path=str(inp),input_sha256=C.sha(inp),registration_sha256=rh,checkpoint_sha256=C.EXPECTED)
    if any(rec.get(k)!=v for k,v in identity.items()) or C.sha(out)!=rec['output_sha256'] or (r['request_id'],j) not in ends:raise ValueError('Invalid completed round')
   else:
    if out.exists() or (r['request_id'],j) in starts:raise RuntimeError('Unclosed artifact, no automatic replay')
    pending.append((r,j))
   inp=out
 if not pending:
  C.verify(reg);print('All 48 rounds verified; zero new forwards');return
 before=C.resources();C.write(C.OUT/'resources_before_gpu.json',before)
 line=before['gpu']['stdout'].strip().splitlines()[0].split(',');total,free=float(line[1]),float(line[3]);headroom=max(2048.,.1*total)
 if free-headroom<1024:raise RuntimeError('RESOURCE_BLOCKED: no dispatch')
 import torch
 from r2.pilots.single_step_denoiser_distillation_v1.common import runtime,memory
 from r2.pilots.single_step_denoiser_distillation_v1.model import load_student,tensor_state_sha256
 from r2.pilots.single_step_denoiser_distillation_v1.inference import single_student_proposal
 runtime(4);tick=time.perf_counter()
 student,spec,payload=load_student(C.STUDENT,require_step=4000);student=student.to('cuda').eval().requires_grad_(False)
 state_before=tensor_state_sha256(student.state_dict());peak_alloc=peak_res=0;model_seconds=0.;new=0
 dispatch=dict(GPU_workers=1,headroom_mib=headroom,prior_comparable_peak_reserved_bytes=218103808,prior_process_VRAM_mib=None,reason='48 very short dependent calls across 12 chains; model duplication/loading and GPU contention unlikely to improve throughput. One authorized first call measured; no probe jobs.')
 C.write(C.OUT/'dispatch.json',dispatch)
 for r in reg['records']:
  rid=r['request_id'];K=int(r['K']);N=int(r['N'])
  with np.load(r['raw_path'],allow_pickle=False) as z:raw=z['raw'].reshape(N,3,3).copy();originalC=z['C'].copy()
  if not same(originalC,raw[:K]):raise ValueError('Original C differs from RAW')
  with np.load(r['S1_denoised_path'],allow_pickle=False) as z:s1=z['output'].copy()
  previous=s1;inp=Path(r['S1_denoised_path'])
  for j in range(2,6):
   out=C.OUT/'denoised'/f'S{j}'/rid/'denoised.npz';rp=out.with_name('receipt.json')
   if rp.exists():
    with np.load(out,allow_pickle=False) as z:previous=z['output'].copy()
    inp=out;continue
   if len(starts)>=48:raise RuntimeError('Forward limit exhausted')
   if previous.shape!=raw.shape or previous.dtype!=np.float32 or not same(previous[:K],originalC):raise ValueError('Recurrence shape/C changed')
   identity=dict(request_id=rid,round=j,raw_sha256=r['raw_sha256'],input_path=str(inp),input_sha256=C.sha(inp),registration_sha256=rh,checkpoint_sha256=C.EXPECTED)
   torch.cuda.reset_peak_memory_stats();C.event('STUDENT_STARTED',**identity);starts.append((rid,j));t=time.perf_counter()
   try:
    arrays,info=single_student_proposal(student,C_original=originalC,input_model=previous,input_d=None,device='cuda',bf16=True,phase=None)
    output=arrays['output']
    if not same(output[:K],originalC) or output.shape!=raw.shape or not np.isfinite(output).all() or not info['no_rng_consumed']:raise ValueError('Output finite/C/N/RNG invariant failed')
    # The reused function calls its relative displacement RAW; correct that field for this recurrence.
    arrays['input_model']=previous.copy();arrays['displacement_from_previous']=output-previous;arrays['displacement_from_RAW']=output-raw;arrays['displacement_from_S1']=output-s1
    C.save_npz(out,**arrays)
    mem=memory();peak_alloc=max(peak_alloc,mem['peak_allocated_bytes']);peak_res=max(peak_res,mem['peak_reserved_bytes']);model_seconds+=info['model_forward_wall_seconds']
    rec=dict(status='PASS',**identity,output_sha256=C.sha(out),N=N,K=K,**{k:v for k,v in info.items() if k not in ('N','K','displacement_reference')},seconds=time.perf_counter()-t,memory=mem,no_intermediate_GEO=True,displacement_reference='previous input for step; original RAW and S1 separately saved',free_step_mean=float(np.linalg.norm((output-previous)[K:].astype(np.float64),axis=-1).mean()),free_step_max=float(np.linalg.norm((output-previous)[K:].astype(np.float64),axis=-1).max()),free_total_mean=float(np.linalg.norm((output-raw)[K:].astype(np.float64),axis=-1).mean()))
    C.write(rp,rec);C.event('STUDENT_COMPLETE',**identity,output_sha256=rec['output_sha256'],seconds=rec['seconds']);new+=1
    if new==1:
     rs=C.resources();rs['first_authorized_call']=rec;rs['dispatch']=dispatch;C.write(C.OUT/'resources_after_first_forward.json',rs)
    print(f'{rid} S{j} PASS step={rec["free_step_mean"]:.6g}',flush=True)
   except Exception as exc:
    C.event('STUDENT_FAILED',**identity,error=repr(exc));raise
   previous=output;inp=out
 state_after=tensor_state_sha256(student.state_dict())
 if state_before!=state_after:raise ValueError('Student weights changed')
 C.verify(reg)
 C.write(C.OUT/'denoise_completion.json',dict(status='PASS',new_forwards_this_run=new,total_student_forwards=48,teacher_forwards=0,generator_forwards=0,training_updates=0,no_new_noise=True,C_bitwise_all=True,N_fixed=True,intermediate_GEO_C=0,student_state_before=state_before,student_state_after=state_after,student_checkpoint_sha256=C.EXPECTED,peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_res,model_forward_seconds=model_seconds,wall_seconds=time.perf_counter()-tick,GPU_workers=1))
 C.write(C.OUT/'resources_after_gpu.json',C.resources())
if __name__=='__main__':main()
