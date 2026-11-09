"""Exactly one fresh-AdamW 4000-update free-residual distillation stream."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='4'
import gc,json,math,signal,time
from pathlib import Path
import numpy as np
import torch
from . import common as C
from . import distill_metrics as DM
from .model import load_student,effective_proposal,export_student,tensor_state_sha256

def lr_at(step):
 if not 1<=step<=4000:raise ValueError('Update outside4000')
 if step<=100:return 1e-6+(1e-4-1e-6)*(step-1)/99
 return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(step-100)/3900))
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def checkpoint(model,opt,step,monitor_steps,phase):
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 optimizer=opt.state_dict()
 optimizer['state']={k:{n:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for n,v in values.items()} for k,values in optimizer['state'].items()}
 payload=dict(schema='S8_W512_KD_FULL_V1',student=C.STUDENT,step=step,model=state,optimizer=optimizer,
  state_sha256=tensor_state_sha256(state),stream_sha256=C.sha(C.OUT/'manifest/train_stream.json'),
  monitor_steps=monitor_steps,torch_CPU_RNG=torch.get_rng_state(),torch_CUDA_RNG=torch.cuda.get_rng_state(),
  source_sha256=C.sha(__file__),initial_seed=C.seed('student_initialization'),phase_counts=dict(phase.counts))
 path=C.OUT/'checkpoints/latest.pt';C.atomic(path,lambda f:torch.save(payload,f),True)
 receipt=dict(status='CLOSED_UPDATE',step=step,path=str(path),sha256=C.sha(path),state_sha256=payload['state_sha256'],
  optimizer_steps=sorted({int(v['step']) for v in optimizer['state'].values()}),monitor_steps=monitor_steps,phase_counts=dict(phase.counts))
 if receipt['optimizer_steps']!=[step] and step:raise AssertionError('AdamW real update count')
 C.write(C.OUT/'checkpoints/latest.json',receipt)
 C.write(C.OUT/'checkpoints'/f'closed_step{step:04d}.json',receipt)
 if step in (500,1000):export_student(C.OUT/'checkpoints'/f'student_step{step}.pt',model,step)
 if step==4000:
  final=C.OUT/'checkpoints/student_final4000.pt'
  C.atomic(final,lambda f:torch.save(payload,f),True)
  C.write(final.with_suffix('.json'),dict(receipt,path=str(final),sha256=C.sha(final)))
 return receipt

def monitor(model,rows,cache,phase,step):
 model.eval();results=[];before=C.rng_hash();start=time.monotonic()
 for row in rows:
  sid=row['sample_id'];b=cache[sid];n,k=int(b['N']),int(b['K']);x=torch.from_numpy(b['input_d'].reshape(1,n,9)).cuda()
  def forward():
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(model,x)
  p,r=phase.call('student_diagnostic_F',forward,dict(step=step,sample_id=sid))
  pd=p.reshape(n,3,3).cpu().numpy().copy();pm=(p/C.SCALE).reshape(n,3,3).cpu().numpy().copy()
  if not np.isfinite(pd).all():raise FloatingPointError('Monitor nonfinite')
  metrics=DM.residual_metrics(b['input_d'],b['proposal_T_d'],pd,k,teacher_model=b['proposal_T_model'],student_model=pm)
  metrics.update(step=step,sample_id=sid,domain=row['domain'],split=('fixed_training' if row['split']=='train' else 'internal_val'),uid=row['uid'],condition_id=row['condition_id'])
  results.append(metrics)
  C.save_npz(C.OUT/'monitor'/f'step{step:04d}'/(sid+'.npz'),proposal_S_d=pd,proposal_S_model=pm)
 if C.rng_hash()!=before:raise AssertionError('Monitoring consumed RNG')
 C.csvwrite(C.OUT/'monitor'/f'step{step:04d}.csv',results)
 summary=DM.summarize(results,('step','split','domain'));C.write(C.OUT/'monitor'/f'step{step:04d}.json',dict(rows=results,summary=summary,seconds=time.monotonic()-start))
 model.train();return summary

def main():
 C.runtime();done=C.OUT/'training/completion.json'
 if done.exists():
  d=C.read(done)
  for key in ('final_full','inference'):
   r=d[key]
   if C.sha(r['path'])!=r['sha256']:raise ValueError('Completed student changed')
  print('SKIP_VERIFIED_COMPLETE train');return
 if C.read(C.OUT/'preflight/student_completion.json')['status']!='PASS':raise RuntimeError('Student preflight required')
 manifest=C.read(C.OUT/'manifest/data.json');stream=C.read(manifest['train_stream_path']);registration=C.read(C.OUT/'registration.json')
 if C.sha(manifest['train_stream_path'])!=registration['train_stream_sha256']:raise ValueError('Frozen stream changed')
 byid={r['sample_id']:r for r in manifest['train_inputs']+manifest['internal_val']};cache={}
 # Load all1120train+25same-parent validation labels on CPU; no online teacher.
 for sid,row in byid.items():
  path=Path(row['teacher_cache_path']);receipt=C.read(path.with_suffix('.json'))
  if C.sha(path)!=receipt['sha256'] or C.sha(row['input_path'])!=receipt['input_sha256'] or receipt['teacher_sha256']!=C.EXPECTED:raise ValueError('Frozen teacher/input changed')
  b=arrays(path);n,k=row['N'],row['K']
  if b['input_d'].shape!=(n,3,3) or b['proposal_T_d'].dtype!=np.float32 or b['known_mask'].tolist()!=(np.arange(n)<k).tolist():raise ValueError('KD cache contract')
  if not np.array_equal(b['r_T_d'],np.subtract(b['proposal_T_d'],b['input_d'],dtype=np.float32)):raise ValueError('Teacher effective residual changed')
  cache[sid]=b
 monitoring=manifest['internal_val']+[byid[sid] for sid in manifest['monitor_ids']]
 if len(monitoring)!=33 or len(stream['batches'])!=4000:raise ValueError('Frozen counts')
 initial=C.OUT/'preflight/student_initial_fp32.pt';initial_receipt=C.read(initial.with_suffix('.json'))
 if C.sha(initial)!=initial_receipt['sha256']:raise ValueError('Initial export changed')
 resume=None;phase_name='student_training'
 prior=list((C.OUT/'gpu_phases').glob('student_training*.json'))
 if prior:
  states=[C.read(p) for p in prior];last=max(states,key=lambda d:d.get('prior_gpu_seconds',0))
  if last['status']!='PAUSED_AT_CLOSED_CHECKPOINT' or last['pending'] is not None:raise RuntimeError('No closed paused training boundary; no automatic replay')
  cr=C.read(C.OUT/'checkpoints/latest.json')
  if C.sha(cr['path'])!=cr['sha256']:raise ValueError('Latest checkpoint hash')
  resume=torch.load(cr['path'],map_location='cpu',weights_only=True)
  if resume['source_sha256']!=C.sha(__file__) or resume['stream_sha256']!=registration['train_stream_sha256']:raise ValueError('Resume code/stream changed')
  if tensor_state_sha256(resume['model'])!=resume['state_sha256']:raise ValueError('Resume model changed')
  if last['counts']!=resume['phase_counts']:raise ValueError('Operations beyond closed checkpoint')
  phase_name='student_training_resume_'+str(resume['step'])
 phase=C.Phase(phase_name,estimated_mib=3500);step=0;monitor_steps=[];curves=[];summaries=[];stop_requested=[False]
 def safe_stop(signum,frame):stop_requested[0]=True
 signal.signal(signal.SIGTERM,safe_stop)
 try:
  model,spec,payload=load_student(initial,require_step=0);del payload
  model.cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.95),weight_decay=0,foreach=False)
  if len(opt.state):raise AssertionError('Fresh AdamW must have empty state')
  if resume is None:C.write(C.OUT/'training/initialization.json',dict(status='PASS',initial_file_sha256=C.sha(initial),initial_state_sha256=tensor_state_sha256(model.state_dict()),spec=spec,fresh_optimizer=True,source_sha256=C.sha(__file__)))
  if resume is not None:
   model.load_state_dict(resume['model'],strict=True);opt.load_state_dict(resume['optimizer'])
   torch.set_rng_state(resume['torch_CPU_RNG']);torch.cuda.set_rng_state(resume['torch_CUDA_RNG'])
   step=int(resume['step']);monitor_steps=resume['monitor_steps']
   curves=[{k:v for k,v in json.loads(line).items() if k not in ('utc','event')} for line in (C.OUT/'training/updates.jsonl').read_text().splitlines()]
   if len(curves)!=step or curves[-1]['step']!=step:raise ValueError('Resume cursor differs from closed update ledger')
   for m in monitor_steps:summaries+=C.read(C.OUT/'monitor'/f'step{m:04d}.json')['summary']
   C.event('TRAIN_RESUME_CLOSED',step=step,checkpoint_sha256=cr['sha256']);del resume
  start_step=step
  torch.cuda.reset_peak_memory_stats();before_rng=C.rng_hash();tick=time.monotonic()
  if not monitor_steps:summaries+=monitor(model,monitoring,cache,phase,0);monitor_steps.append(0)
  for batch in stream['batches'][start_step:]:
   stream_index=int(batch['step']);step=stream_index+1;ids=batch['sample_ids'];rows=[byid[sid] for sid in ids];den=int(batch['free_scalar_denominator']);start=time.monotonic()
   if len({r['uid'] for r in rows})!=8 or sum(r['domain']=='real' for r in rows)!=4 or den!=sum(9*(r['N']-r['K']) for r in rows):raise ValueError('Batch8 balanced parent contract')
   if phase.remaining()<30:
    checkpoint(model,opt,step-1,monitor_steps,phase)
    if curves:C.csvwrite(C.OUT/'training/learning_curve.csv',curves)
    C.csvwrite(C.OUT/'monitor/summary.csv',summaries)
    C.write(C.OUT/'training/stopped_budget.json',dict(status='STOPPED_GPU_BUDGET',completed_updates=step-1,remaining_GPU_seconds=phase.remaining()))
    phase.finish('STOPPED_GPU_BUDGET');return
   lr=lr_at(step)
   for group in opt.param_groups:group['lr']=lr
   opt.zero_grad(set_to_none=True);total=0.;domains={k:dict(SSE=0.,energy=0.,scalars=0) for k in ('real','synthetic')}
   for micro,(sid,row) in enumerate(zip(ids,rows)):
    b=cache[sid];n,k=row['N'],row['K'];x=torch.from_numpy(b['input_d'].reshape(1,n,9)).cuda();target=torch.from_numpy(b['r_T_d'].reshape(1,n,9)).cuda()
    def forward():
     with torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(model,x)
    p,r=phase.call('student_train_F',forward,dict(step=step,micro=micro,sample_id=sid))
    with torch.autocast('cuda',enabled=False):
     sse=((r[:,k:]-target[:,k:])**2).sum();loss=sse/den
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite KD loss')
    phase.call('student_train_B',lambda:loss.backward(),dict(step=step,micro=micro))
    value=float(sse.detach());total+=value
    d=domains[row['domain']];d['SSE']+=value;d['energy']+=float((target[:,k:].double()**2).sum());d['scalars']+=9*(n-k)
    del p,r,loss,sse,x,target
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
   if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
   phase.call('student_updates',opt.step,dict(step=step,lr=lr))
   if C.rng_hash()!=before_rng:raise AssertionError('Dropout0 frozen stream must not consume RNG')
   curve=dict(step=step,lr=lr,loss=total/den,free_scalars=den,gradient_norm=float(norm),seconds=time.monotonic()-start)
   for name,d in domains.items():curve.update({name+'_E':d['SSE']/d['scalars'],name+'_B':d['energy']/d['scalars'],name+'_R':d['SSE']/d['energy'] if d['energy']/d['scalars']>1e-14 else None,name+'_scalars':d['scalars']})
   curves.append(curve);C.event('UPDATE_CLOSED',ledger='training/updates.jsonl',**curve)
   if step==1:C.write(C.OUT/'resources/student_training_first_full_update.json',dict(resource=C.resources(),memory=C.memory(),optimizer_state_entries=len(opt.state)))
   if step in (500,1000,2000,4000):summaries+=monitor(model,monitoring,cache,phase,step);monitor_steps.append(step)
   if step%100==0 or stop_requested[0]:
    checkpoint(model,opt,step,monitor_steps,phase)
    C.csvwrite(C.OUT/'training/learning_curve.csv',curves);C.csvwrite(C.OUT/'monitor/summary.csv',summaries)
    C.write(C.OUT/'training/progress.json',dict(step=step,total=4000,seconds=time.monotonic()-tick,memory=C.memory(),remaining_GPU_seconds=phase.remaining()))
    print(json.dumps(dict(stage='train',step=step,loss=curve['loss'],real_R=curve['real_R'],synthetic_R=curve['synthetic_R'],seconds=time.monotonic()-tick)),flush=True)
   if stop_requested[0]:
    phase.finish('PAUSED_AT_CLOSED_CHECKPOINT');print('Safe signal stop at closed checkpoint; no automatic replay.');return
  inference=export_student(C.OUT/'checkpoints/student_inference.pt',model,4000,dict(training_stream_sha256=registration['train_stream_sha256'],teacher_sha256=C.EXPECTED))
  full=C.read(C.OUT/'checkpoints/student_final4000.json')
  completion=dict(status='COMPLETE',updates=4000,train_forwards=32000,train_backwards=32000,diagnostic_forwards=165,counts=dict(phase.counts),monitor_steps=monitor_steps,
   final_full=full,inference=inference,seconds=time.monotonic()-tick,memory=C.memory(),teacher_loaded=False,no_rng_consumed=True)
  expected=dict(student_diagnostic_F=165,student_train_F=32000,student_train_B=32000,student_updates=4000)
  actual={k:phase.counts.get(k,0)+phase.prior_counts.get(k,0) for k in expected}
  assert actual==expected
  completion['counts']=actual
  C.write(done,completion);phase.finish()
 except BaseException as exc:
  if curves:C.csvwrite(C.OUT/'training/learning_curve.csv',curves)
  C.write(C.OUT/'training/failure.json',dict(step=step,error=repr(exc),no_automatic_replay=True))
  phase.finish('FAILED',repr(exc));raise
if __name__=='__main__':main()