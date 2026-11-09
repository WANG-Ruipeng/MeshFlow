"""Budgeted one-GPU sequential preflight and full-parameter adaptation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,gc,hashlib,json,time,traceback,signal
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.anchor_aware_denoiser_v1 import common as C
from r2.pilots.anchor_aware_denoiser_v1.model import load_official_wrapper
from r2.pilots.anchor_aware_denoiser_v1.data_stream import load_batch
_BATCH_MANIFEST=None
_STOP=False
def stop_at_boundary(signum,frame):
 global _STOP
 _STOP=True
 print("Stop requested; saving after current complete optimizer update",flush=True)

def runtime():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
def rng_hash():
 return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()).items()}
def memory():
 return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
class Phase:
 def __init__(self,name):
  self.name=name;self.start=time.monotonic();self.f=self.b=self.u=0
  self.path=C.OUT/'gpu_phases'/f'{name}.json'
  prior=[C.read(p) for p in (C.OUT/'gpu_phases').glob('*.json')] if (C.OUT/'gpu_phases').exists() else []
  if any(r['status']=='RUNNING' for r in prior):raise RuntimeError('Unresolved GPU phase; no silent retry')
  self.prior_seconds=sum(r.get('gpu_occupancy_seconds',0) for r in prior)
  prefix=name.split('__resume')[0]
  earlier=[r for r in prior if name.startswith('train') and r['name'].split('__resume')[0]==prefix]
  self.offset_f=sum(r['forward_attempts'] for r in earlier);self.offset_b=sum(r['backward_attempts'] for r in earlier);self.offset_u=sum(r['optimizer_updates'] for r in earlier)
  if self.path.exists():raise RuntimeError('Phase already attempted; validate receipt or explicit boundary resume')
  if self.prior_seconds>=10800:raise RuntimeError('GPU 3h budget exhausted')
  resource=C.resources();C.write(C.OUT/'resources'/f'{name}_before.json',resource)
  vals=resource['gpu']['stdout'].strip().splitlines()[0].split(',')
  total,free=float(vals[1]),float(vals[3]);self.headroom=max(2048,.1*total)
  if free<self.headroom+8000 and name.startswith(('train','preflight')):raise RuntimeError('RESOURCE_BLOCKED: training estimate8GiB plus headroom unavailable')
  C.write(self.path,self.state('RUNNING'))
 def state(self,status,error=None):
  return dict(name=self.name,status=status,forward_attempts=self.f,backward_attempts=self.b,optimizer_updates=self.u,prior_gpu_seconds=self.prior_seconds,gpu_occupancy_seconds=time.monotonic()-self.start,error=error)
 def remaining(self):return 10800-self.prior_seconds-(time.monotonic()-self.start)
 def call(self,model,x,known,official=False,capture=False):
  limit=32 if self.name=='preflight' else 8000 if self.name.startswith('train') else 48 if self.name.startswith('real') else 18
  if self.f+self.offset_f>=limit or self.remaining()<15:raise RuntimeError('Forward/time budget reached')
  self.f+=1;C.event('FORWARD_STARTED',ledger='gpu_calls.jsonl',phase=self.name,call=self.f,official_reference=official)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   r=model.official_residual(x) if official else model(x,known,capture_role_stats=capture)
  if not bool(torch.isfinite(r).all()):raise FloatingPointError('Nonfinite residual')
  return r
 def backward(self,loss):
  if self.b+self.offset_b>=(16 if self.name=='preflight' else 8000):raise RuntimeError('Backward cap')
  self.b+=1;C.event('BACKWARD_STARTED',ledger='gpu_calls.jsonl',phase=self.name,call=self.b);loss.backward()
 def updated(self):
  if self.u+self.offset_u>=(2 if self.name=='preflight' else 1000):raise RuntimeError('Optimizer cap')
  self.u+=1
 def finish(self,status,error=None):
  if torch.cuda.is_initialized():torch.cuda.synchronize()
  C.write(self.path,self.state(status,error));C.write(C.OUT/'resources'/f'{self.name}_after.json',C.resources())
def batch(update):
 global _BATCH_MANIFEST
 if _BATCH_MANIFEST is None:
  _BATCH_MANIFEST=C.read(C.OUT/'stream/manifest.json')
  if C.sha(C.OUT/'stream/manifest.json')!=C.read(C.OUT/'registration.json')['stream_sha256']:raise ValueError('Stream manifest changed')
 return load_batch(update,_BATCH_MANIFEST)[0]
def tensors(a,j):
 lo,hi=map(int,a['offsets'][j:j+2])
 return (torch.from_numpy(a['x_d'][lo:hi].reshape(1,-1,9)).cuda(),torch.from_numpy(a['target_d'][lo:hi].reshape(1,-1,9)).cuda(),torch.from_numpy(a['known'][lo:hi].reshape(1,-1)).cuda())
def cpu_state_hash(model):
 h=hashlib.sha256()
 for name,t in model.base.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make_model(arm):
 m,spec=load_official_wrapper(anchor=arm=='D_ANCHOR',activation_checkpointing=True,face_count_dropout=.1)
 spec['all_base_state_hash']=cpu_state_hash(m)
 C.write(C.OUT/'load_checks'/f'{arm}.json',spec);return m,spec
def optimizer_for(model):
 return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95),weight_decay=0,foreach=False)
def grad_report(model):
 available=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
 if not available or any(not bool(torch.isfinite(p.grad).all()) for _,p in available):raise FloatingPointError('Gradient absent/nonfinite')
 selected={n:float(p.grad.float().norm()) for n,p in available if n in ('e_known','base.post_quant_linear.weight') or n.endswith('layers.0.attn.qkv.weight')}
 return dict(parameters_with_grad=len(available),parameters_without_grad=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None],selected_gradient_l2=selected)
def train_update(model,optimizer,a,phase,capture=False,known_grad=False):
 optimizer.zero_grad(set_to_none=True);denominator=9*int((~a['known']).sum());parts=[];known_norm=[]
 if denominator<=0:raise ValueError('No free coordinates')
 for j in range(8):
  x,y,m=tensors(a,j);holder=[]
  def hook(_module,_inputs,output):
   output.retain_grad();holder.append(output)
  handle=model.base.post_quant_linear.register_forward_hook(hook) if known_grad and j==0 else None
  residual=phase.call(model,x,m,capture=capture and j==7);prediction=x.float()+residual.float()
  sse=((prediction-y.float())[~m]**2).sum();loss=sse/denominator
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
  phase.backward(loss);parts.append(float(sse.detach()))
  if handle is not None:
   handle.remove();g=holder[0].grad;m3=m.repeat_interleave(3,dim=1)
   if g is None or not bool(torch.isfinite(g).all()):raise AssertionError('Known feature gradient missing/nonfinite')
   known_norm.append(float(g[m3].float().norm()))
   if known_norm[-1]<=0:raise AssertionError('Free loss cannot reach known features')
  del x,y,m,residual,prediction,sse,loss,holder
 report=grad_report(model) if capture or known_grad else None
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
 if phase.u+phase.offset_u >= (2 if phase.name=='preflight' else 1000):raise RuntimeError('Optimizer cap before update')
 optimizer.step();phase.updated()
 return dict(loss=sum(parts)/denominator,free_denominator=denominator,gradient_norm_before_clip=float(norm),role_l2=float(model.e_known.detach().norm()),role=model.last_role_stats if capture else None,gradient=report,known_hidden_gradient_l2=known_norm)
def preflight():
 receipt=C.OUT/'preflight/completion.json'
 if receipt.exists():
  if C.read(receipt)['status']!='PASS':raise RuntimeError('Previously failed preflight')
  return
 phase=Phase('preflight');result=[]
 try:
  rows=C.read(C.OUT/'stream/plan.json')['rows'];selected=[min(rows,key=lambda r:r['K']),max(rows,key=lambda r:r['N'])]
  for region in ('seat','back'):selected.append(next(r for r in rows if r['region_type']==region))
  for arm in C.ARMS:
   model,spec=make_model(arm);torch.cuda.reset_peak_memory_stats();model.cuda().eval()
   torch.manual_seed(C.seed('preflight_torch'));torch.cuda.manual_seed_all(C.seed('preflight_torch'))
   for case,row in enumerate(selected):
    a=batch(int(row['step']));x,y,m=tensors(a,int(row['slot']));before=rng_hash()
    with torch.no_grad():ref=phase.call(model,x,m,official=True);ours=phase.call(model,x,m)
    if not torch.equal(ref,ours):raise AssertionError('Zero role differs from official BF16 residual')
    if rng_hash()!=before:raise AssertionError('Eval consumed RNG')
    lo,hi=map(int,a['offsets'][int(row['slot']):int(row['slot'])+2])
    original=a['target_model'][lo:hi]
    proposal=((x.float()+ours.float())/C.SCALE).reshape(-1,3,3).cpu().numpy()
    restored=proposal.copy();known=m.cpu().numpy().ravel();restored[known]=original[known]
    if restored[known].tobytes()!=original[known].tobytes():raise AssertionError('C projection changed bytes')
    path=C.OUT/'preflight'/f'{arm}_case{case}_export.npz'
    C.save_npz(path,output=restored,C=original[known],known=known,proposal=proposal)
    with np.load(path,allow_pickle=False) as check:
     if check['output'][known].tobytes()!=original[known].tobytes():raise AssertionError('C export roundtrip failed')
    result.append(dict(arm=arm,condition_id=row['condition_id'],N=row['N'],K=row['K'],zero_role_bitwise=True,C_projection_export_byte_exact=True,export_sha256=C.sha(path)))
    del x,y,m,ref,ours
   model.train();optimizer=optimizer_for(model);a=batch(1);before=rng_hash();tick=time.monotonic()
   stats=train_update(model,optimizer,a,phase,capture=True,known_grad=True)
   if arm=='D_ANCHOR' and stats['gradient']['selected_gradient_l2'].get('e_known',0)<=0:raise AssertionError('Role gradient missing')
   if arm=='D_CONT_NO_ROLE' and stats['role_l2']!=0:raise AssertionError('No-role moved')
   stats.update(arm=arm,seconds=time.monotonic()-tick,**memory(),rng_before=before,rng_after=rng_hash(),discarded_update=True)
   C.write(C.OUT/'preflight'/f'{arm}.json',stats);C.write(C.OUT/'resources'/f'preflight_{arm}_after_full_AdamW.json',C.resources())
   del optimizer,model;gc.collect();torch.cuda.empty_cache()
  if (phase.f,phase.b,phase.u)!=(32,16,2):raise AssertionError('Preflight accounting')
  C.write(receipt,dict(status='PASS',zero_equivalence=result,forwards=phase.f,backwards=phase.b,discarded_updates=phase.u,fresh_initialization_for_formal=True))
  phase.finish('COMPLETE')
 except BaseException as e:
  C.event('PREFLIGHT_FAILED',error=repr(e));phase.finish('FAILED',repr(e));raise
def save_checkpoint(path,model,optimizer,step,phase,identity):
 payload=dict(schema='anchor_denoiser_full_v1',arm=identity['arm'],step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),identity=identity,phase=phase.state('CHECKPOINT'),model_training=model.training)
 C.atomic(path,lambda f:torch.save(payload,f),True)
 receipt=dict(status='PASS',path=str(path),sha256=C.sha(path),step=step,identity=identity)
 C.write(path.with_suffix('.json'),receipt);return receipt
def train(arm):
 final=C.OUT/'checkpoints'/arm/'step1000.pt';done=final.with_suffix('.json')
 if done.exists():
  if C.sha(final)!=C.read(done)['sha256']:raise ValueError('Final changed')
  export=C.OUT/'checkpoints'/arm/'inference.pt'
  completion=C.OUT/'training'/arm/'completion.json'
  if not export.exists() or not export.with_suffix('.json').exists() or not completion.exists():
   state=torch.load(final,map_location='cpu',weights_only=True,mmap=True)
   if state['step']!=1000 or state['identity']['registration_sha256']!=C.sha(C.OUT/'registration.json'):raise ValueError('Final checkpoint recovery identity differs')
   if not export.exists():C.atomic(export,lambda f:torch.save(dict(schema='anchor_denoiser_inference_v1',arm=arm,step=1000,model=state['model'],identity=state['identity']),f),True)
   C.write(export.with_suffix('.json'),dict(status='PASS',path=str(export),sha256=C.sha(export),step=1000,identity=state['identity']))
   C.write(completion,dict(status='COMPLETE',completed_updates=1000,exposures=8000,forwards=8000,backwards=8000,optimizer_updates=1000,checkpoint=C.read(done),inference_path=str(export),recovered_export_on_CPU=True))
   del state
  if C.sha(export)!=C.read(export.with_suffix('.json'))['sha256'] or C.read(completion)['status']!='COMPLETE':raise ValueError('Final inference/completion identity invalid')
  print(json.dumps(dict(arm=arm,status='SKIP_VERIFIED_COMPLETE')),flush=True);return
 if C.read(C.OUT/'preflight/completion.json')['status']!='PASS':raise RuntimeError('Preflight not passed')
 resume=None;phase_name='train_'+arm
 previous=C.OUT/'gpu_phases'/f'{phase_name}.json'
 if previous.exists():
  boundary=C.read(C.OUT/'training'/arm/'completion.json')
  if boundary['status'] not in ('PAUSED_BOUNDARY','GPU_TIME_LIMIT'):raise RuntimeError('No clean resumable boundary; never replay completed updates')
  resume=boundary['checkpoint']
  if C.sha(resume['path'])!=resume['sha256']:raise ValueError('Resume checkpoint changed')
  phase_name+='__resume'+str(resume['step'])
 phase=Phase(phase_name);last=0;model=optimizer=None
 try:
  model,spec=make_model(arm);model.cuda().train();optimizer=optimizer_for(model)
  torch.manual_seed(C.seed('formal_torch'));torch.cuda.manual_seed_all(C.seed('formal_torch'))
  identity=dict(arm=arm,official_sha256=C.EXPECTED,base_initial_state_hash=spec['all_base_state_hash'],stream_sha256=C.sha(C.OUT/'stream/manifest.json'),registration_sha256=C.sha(C.OUT/'registration.json'),runner_sha256=C.sha(__file__),wrapper_sha256=C.sha(C.CODE/'model.py'),fresh_AdamW=True,lr=1e-5,betas=[.9,.95],weight_decay=0,effective_batch=8,microbatch=1,clip=1.,dropout=.1,activation_checkpointing=True,new_EMA=False)
  if resume:
   state=torch.load(resume['path'],map_location='cpu',weights_only=True,mmap=True)
   if state['identity']!=identity or int(state['step'])!=phase.offset_u or phase.offset_f!=8*phase.offset_u or phase.offset_b!=8*phase.offset_u:raise ValueError('Resume identity or accounting differs')
   model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
   torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);last=int(state['step']);del state
   C.event('EXACT_BOUNDARY_RESUME',arm=arm,step=last,checkpoint_sha256=resume['sha256'])
  else:C.write(C.OUT/'training'/arm/'initialization.json',dict(identity=identity,rng=rng_hash(),optimizer_state_entries=len(optimizer.state)))
  torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  for update in range(last+1,1001):
   if phase.remaining()<90 or _STOP:
    cp=save_checkpoint(C.OUT/'checkpoints'/arm/f'budget_boundary_step{last:04d}.pt',model,optimizer,last,phase,identity)
    status='PAUSED_BOUNDARY' if _STOP else 'GPU_TIME_LIMIT'
    C.write(C.OUT/'training'/arm/'completion.json',dict(status=status,completed_updates=last,checkpoint=cp));phase.finish(status);return
   a=batch(update);before=rng_hash();tick=time.monotonic()
   C.event('UPDATE_STARTED',ledger=f'training/{arm}/ledger.jsonl',update=update,phase_forwards=phase.f,phase_backwards=phase.b)
   stats=train_update(model,optimizer,a,phase,capture=update in (1,1000));last=update
   stats.update(update=update,seconds=time.monotonic()-tick,rng_before=before,rng_after=rng_hash(),batch_sha256=C.sha(C.OUT/'stream/batches'/f'{update:06d}.npz'))
   C.event('UPDATE_COMPLETE',ledger=f'training/{arm}/ledger.jsonl',**stats)
   if update==1:C.write(C.OUT/'resources'/f'{arm}_first_full_update.json',dict(resource=C.resources(),torch=memory(),update=stats))
   if update%50==0 or update==1:
    progress=dict(arm=arm,step=update,loss=stats['loss'],seconds=time.monotonic()-start,seconds_per_update=(time.monotonic()-start)/update,remaining_GPU_seconds=phase.remaining(),**memory())
    C.write(C.OUT/'training'/arm/'progress.json',progress);print(json.dumps(progress),flush=True)
   if update==500:save_checkpoint(C.OUT/'checkpoints'/arm/'step0500.pt',model,optimizer,update,phase,identity)
  cp=save_checkpoint(final,model,optimizer,1000,phase,identity)
  export=C.OUT/'checkpoints'/arm/'inference.pt'
  C.atomic(export,lambda f:torch.save(dict(schema='anchor_denoiser_inference_v1',arm=arm,step=1000,model=model.state_dict(),identity=identity),f),True)
  C.write(export.with_suffix('.json'),dict(status='PASS',path=str(export),sha256=C.sha(export),step=1000,identity=identity))
  C.write(C.OUT/'training'/arm/'completion.json',dict(status='COMPLETE',completed_updates=1000,exposures=8000,forwards=phase.f+phase.offset_f,backwards=phase.b+phase.offset_b,optimizer_updates=phase.u+phase.offset_u,checkpoint=cp,inference_path=str(export),wall_seconds=time.monotonic()-start,**memory()))
  phase.finish('COMPLETE')
 except BaseException as e:
  C.event('TRAINING_FAILED',arm=arm,completed_updates=last,error=repr(e))
  C.write(C.OUT/'training'/arm/'failure.json',dict(status='FAILED',completed_updates=last,error=repr(e),traceback=traceback.format_exc(),checkpoint500_available=(C.OUT/'checkpoints'/arm/'step0500.pt').exists()))
  phase.finish('FAILED',repr(e));raise
def main():
 signal.signal(signal.SIGINT,stop_at_boundary);signal.signal(signal.SIGTERM,stop_at_boundary)
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train']);p.add_argument('--arm',choices=C.ARMS);a=p.parse_args();runtime();C.safety_checks()
 if a.stage=='preflight':preflight()
 elif not a.arm:raise ValueError('Arm required')
 else:train(a.arm)
if __name__=='__main__':main()
