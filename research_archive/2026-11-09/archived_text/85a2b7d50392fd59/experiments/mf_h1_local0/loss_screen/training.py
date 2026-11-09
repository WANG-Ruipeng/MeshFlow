"""Seven identical-stream, fixed 500-update branches; no generation in training."""
from .common import *
from .calibration import strict_initial,inc,endpoint,velocity_grad
from . import checkpoint
from native_t1.training import make_optimizer,masked_fm_loss
from native_t1.data import native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.accounting import count_ot_calls
from .distribution_losses import work_counters
import traceback

def source_guard(r):
 for item in r['source_snapshot']['files']:
  if file_sha256(item['original_path'])!=item['sha256']:raise RuntimeError('Frozen computation source changed: '+item['original_path'])

def grouped(model,before=None):
 predicates={'Geo':lambda n:n.startswith('context_encoder.'),'role':lambda n:n=='role_embedding',
  'velocity_head':lambda n:n.startswith('backbone.final_layer.'),'first_layer':lambda n:n.startswith('backbone.layers.0.')}
 ans={}
 for label,predicate in predicates.items():
  entries=[(n,p) for n,p in model.named_parameters() if predicate(n)]
  vals=[p.grad for n,p in entries] if before is None else [p.detach()-before[n] for n,p in entries]
  ans[label]=magnitude(vals)
 return ans

def tracked_state(model):
 return {n:p.detach().clone() for n,p in model.named_parameters()
  if n.startswith(('context_encoder.','backbone.final_layer.','backbone.layers.0.')) or n=='role_embedding'}

def diagnostic_leaf(arm,v,b,target,denom,weight,ramp,active,r,stage):
 c=r['counts']['training'][arm]
 leaf=v.detach().requires_grad_(True)
 fm=masked_fm_loss(leaf,b['u'],b['valid_mask'],b['known_mask'],denominator=denom)
 gf=velocity_grad(fm,leaf,c);nf=float(torch.linalg.vector_norm(gf))
 aux=evaluate_loss(arm,endpoint(leaf,b),target,r,stage)
 ga=velocity_grad(aux['loss']/8,leaf,c);na=float(torch.linalg.vector_norm(ga))
 return dict(FM_microbatch_gradient_L2=nf,geometry_microbatch_gradient_L2=na,
  raw_ratio=na/nf if nf else None,weighted_ratio=weight*ramp*float(active)*na/nf if nf else None,
  applied_gate=bool(active),lambda_geometry=weight,ramp=ramp,t=float(b['t'][0]),
  actual_velocity_sha256=array_hash(v.detach()),microbatch_index=0,
  normalization='FM original effective coordinate denominator; auxiliary per-sample mean /8')

def train_one(r,out,arm,bank):
 entry=r['arms'][arm]
 if entry['status']!='READY':return
 source_guard(r)
 c=r['counts']['training'].setdefault(arm,{})
 model=None;optimizer=None
 try:
  model,identity=strict_initial();model.set_trainable(True).eval()
  optimizer=make_optimizer(model)
  if optimizer.state:raise RuntimeError('Fresh optimizer unexpectedly contains moments')
  if any(p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters()):raise RuntimeError('Full FP32 generator not trainable')
  if model.training:raise RuntimeError('Dropout must stay off')
  stream=stream_for(r);entry.update(status='TRAINING',initial_identity=identity,fresh_optimizer=True,
   initial_stream_sha256=seal(stream.state_dict()),logs=[],stream_audit=[],timing=dict(CPU_OT_and_stream_seconds=0.,
   model_update_seconds=0.,supervision_prepare_seconds=0.,checkpoint_seconds=0.))
  weight=0. if arm=='L0' else r['calibration']['candidates'][arm]['lambda']
  rootdir=Path(out)/arm;rootdir.mkdir(exist_ok=False)
  entry['lambda']=weight;write(r,out);tick=time.perf_counter()
  for step in range(1,501):
   if sum(x['completed_updates'] for x in r['arms'].values())>=3500:raise RuntimeError('Global update budget exhausted')
   ot_tick=time.perf_counter()
   with count_ot_calls(c):samples=stream.next_effective_batch()
   entry['timing']['CPU_OT_and_stream_seconds']+=time.perf_counter()-ot_tick
   prep_tick=time.perf_counter();labels=[bank.sample(s) for s in samples]
   audit=audit_batch(stream,samples,step,labels)
   if arm!='L0' and audit!=r['arms']['L0']['stream_audit'][step-1]:raise RuntimeError('Paired formal stream mismatch before update')
   entry['stream_audit'].append(audit)
   targets=[prepare_target(arm,lab) for lab in labels] if arm!='L0' else None
   entry['timing']['supervision_prepare_seconds']+=time.perf_counter()-prep_tick
   denom=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
   if denom not in (7524,7452,7488):raise RuntimeError('Original FM denominator differs')
   log_now=step in LOG_STEPS;before=tracked_state(model) if log_now else None
   fm_total=0.;geom_total=0.;active_count=0;applicable_count=0;ratio=None
   probes={name:dict(gated_mean=0.,raw_active_mean=0.,active_samples=0,applicable_active_samples=0) for name in CANDIDATES} if arm=='L0' and step==500 else None
   numerical=[];ramp=min(step/50.,1.);optimizer.zero_grad(set_to_none=True)
   entry['batch_in_progress']=step;write(r,out) if step==1 else None
   update_tick=time.perf_counter()
   for micro,(s,lab) in enumerate(zip(samples,labels)):
    b=native_collate([s],'cuda');active=bool(b['t'][0]>=.5);active_count+=int(active)
    with native_execution(model,coordinates=b['xt']) as runtime:
     inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
     fm=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=denom)
     total=fm
     if arm!='L0' and active:
      result=evaluate_loss(arm,endpoint(v,b),targets[micro],r,'train_'+arm)
      applicable_count+=int(result['applicable']);geom_total+=float(result['loss'].detach())/8
      total=total+weight*ramp*result['loss']/8
      if log_now:numerical.append(dict(microbatch=micro,**plain(result.get('diagnostics',{})),counts=plain(result['counts'])))
     if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite total objective')
     if arm!='L0' and step in (100,500) and micro==0:
      ratio=diagnostic_leaf(arm,v,b,targets[micro],denom,weight,ramp,active,r,'velocity_ratio_'+arm)
     if probes is not None and active:
      with torch.no_grad():
       yhat=endpoint(v.detach(),b)
       for name in CANDIDATES:
        if r['arms'][name]['status']!='READY':
         probes[name]['status']='NOT_RUN_AFTER_CANDIDATE_FAILURE';continue
        try:
         pr=evaluate_loss(name,yhat,prepare_target(name,lab),r,'L0_objective_probe')
         probes[name]['gated_mean']+=float(pr['loss'])/8
         probes[name]['raw_active_mean']+=float(pr['loss'])
         probes[name]['active_samples']+=1;probes[name]['applicable_active_samples']+=int(pr['applicable'])
        except Exception as exc:
         probes[name]['status']='FAILED';probes[name]['error']=repr(exc)
         r['arms'][name].update(status='OBJECTIVE_PROBE_FAILED',error=repr(exc))
     inc(c,'backward_attempts');total.backward();inc(c,'backward_returns')
    fm_total+=float(fm.detach())
   gradient_groups=grouped(model) if log_now else None
   gradnorm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.,error_if_nonfinite=True))
   entry['optimizer_step_in_progress']=True
   inc(c,'optimizer_step_attempts');optimizer.step();inc(c,'optimizer_updates')
   entry['completed_updates']=step;entry['optimizer_step_in_progress']=False;entry['batch_in_progress']=False
   torch.cuda.synchronize()
   if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):raise FloatingPointError('Nonfinite updated generator')
   entry['timing']['model_update_seconds']+=time.perf_counter()-update_tick
   if log_now:
    row=dict(step=step,cumulative_updates=2500+step,FM=fm_total,gated_geometry_mean=geom_total,
     geometry_normalization='Unweighted by lambda/ramp; gate applied; sum per-sample means divided by all8',
     raw_active_geometry_mean=geom_total*8/active_count if active_count and arm!='L0' else None,
     weighted_geometry=weight*ramp*geom_total,total=fm_total+weight*ramp*geom_total,
     active_samples=active_count,applicable_active_samples=applicable_count,effective_batch=8,microbatch=1,
     free_coordinate_denominator=denom,lambda_geometry=weight,ramp=ramp,grad_norm_before_clip=gradnorm,
     gradients=gradient_groups,parameter_updates=grouped(model,before),velocity_gradient_ratio=ratio,runtime=runtime,
     numerical_diagnostics=numerical,loss_probes=probes,paired_stream_sha256=seal(audit))
    entry['logs'].append(row);del before
    event('training_log',arm=arm,step=step,FM=fm_total,weighted_geometry=row['weighted_geometry'],grad_norm=gradnorm)
   if step in (250,500):
    cp_tick=time.perf_counter();streamhash=seal(stream.state_dict())
    entry['checkpoints'][str(step)]=checkpoint.save(rootdir/f'additional_step{step}_cumulative_step{2500+step}.pt',
      model,optimizer,stream,arm=arm,step=step,r=r)
    if seal(stream.state_dict())!=streamhash:raise RuntimeError('Saving advanced training RNG')
    entry['timing']['checkpoint_seconds']+=time.perf_counter()-cp_tick
   if step%25==0 or log_now:
    entry['elapsed_seconds']=time.perf_counter()-tick;write(r,out)
    if not log_now:event('training_progress',arm=arm,updates=step,seconds=round(entry['elapsed_seconds'],2))
  entry.update(status='PASS',final_stream_state=stream.state_dict(),paired_stream_sha256=seal(entry['stream_audit']))
  entry['timing']['wall_seconds']=time.perf_counter()-tick
  if entry['completed_updates']!=500 or c.get('optimizer_updates')!=500 or c.get('forward_returns')!=4000 or c.get('OT_returns')!=4000:
   raise RuntimeError('Completed training accounting differs')
  source_guard(r);event('arm_complete',arm=arm,seconds=entry['timing']['wall_seconds'],endpoint=entry['checkpoints']['500'])
 except Exception as exc:
  entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
  event('arm_failed',arm=arm,updates=entry['completed_updates'],error=repr(exc))
 finally:
  entry['distribution_solver_process_counts_at_exit']=work_counters()
  if optimizer is not None:del optimizer
  if model is not None:del model
  release();write(r,out)

def train_all(r,out):
 bank=TargetBank()
 for arm in ARMS:
  if arm!='L0' and r['arms']['L0']['status']!='PASS':
   if r['arms'][arm]['status']=='READY':r['arms'][arm].update(status='BLOCKED_BASELINE_FAILURE')
   continue
  train_one(r,out,arm,bank)
 r['training_total_updates']=sum(x['completed_updates'] for x in r['arms'].values())
 if r['training_total_updates']>3500:raise RuntimeError('Update budget exceeded')
 write(r,out)
