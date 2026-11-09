"""Only two new fixed500-update arms replay the previous500-batch input stream."""
from .common import *
from . import checkpoint
from ..loss_screen.training import grouped,tracked_state,source_guard
from ..loss_screen.distribution_losses import work_counters
from native_t1.training import make_optimizer,masked_fm_loss
from native_t1.data import native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.accounting import count_ot_calls
import traceback

def train_one(r,out,arm,bank,old_audits):
 entry=r['arms'][arm]
 if entry['status']!='READY':return
 source_guard(r);c=r['counts']['training'].setdefault(arm,{});model=None;optimizer=None
 try:
  model,identity=strict_initial();model.set_trainable(True).eval();optimizer=make_optimizer(model)
  if optimizer.state or any(p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters()):raise RuntimeError('Fresh full-FP32 training required')
  stream=stream_for(r)
  if seal(stream.state_dict())!=START_STREAM_SHA:raise RuntimeError('Wrong input RNG start')
  entry.update(status='TRAINING',initial_identity=identity,fresh_optimizer=True,initial_stream_sha256=seal(stream.state_dict()),
   logs=[],stream_audit=[],timing=dict(CPU_OT_and_stream_seconds=0.,supervision_prepare_seconds=0.,model_update_seconds=0.,checkpoint_seconds=0.))
  first='L3' if arm=='J35' else 'D_edge';weight=LAMBDA3 if arm=='J35' else r['calibration']['lambda_edge']
  if weight is None:raise RuntimeError('No valid interface coefficient; no fallback')
  root=Path(out)/arm;root.mkdir(exist_ok=False);tick=time.perf_counter();write(r,out)
  for step in range(1,501):
   if sum(x['completed_updates'] for x in r['arms'].values())>=1000:raise RuntimeError('Global1000-update cap')
   ot_tick=time.perf_counter()
   with count_ot_calls(c):samples=stream.next_effective_batch()
   entry['timing']['CPU_OT_and_stream_seconds']+=time.perf_counter()-ot_tick
   prep_tick=time.perf_counter();labels=[bank.sample(s) for s in samples];audit=audit_batch(stream,samples,step,labels)
   if audit!=old_audits[step-1]:raise RuntimeError('Input stream differs from historical L0/L3/L5 before update')
   if arm=='JEdge5' and len(r['arms']['J35'].get('stream_audit',[]))>=step and audit!=r['arms']['J35']['stream_audit'][step-1]:raise RuntimeError('New arms input mismatch')
   entry['stream_audit'].append(audit);prepared=[targets(lab) for lab in labels]
   entry['timing']['supervision_prepare_seconds']+=time.perf_counter()-prep_tick
   denom=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
   if denom not in (7524,7452,7488):raise RuntimeError('FM denominator changed')
   log_now=step in LOG_STEPS;before=tracked_state(model) if log_now else None
   totals={'FM':0.,first:0.,'L5':0.};active_count=0;ratio=None;numerical=[];ramp=min(step/50.,1.)
   optimizer.zero_grad(set_to_none=True);entry['batch_in_progress']=step;update_tick=time.perf_counter()
   for micro,s in enumerate(samples):
    b=native_collate([s],'cuda');active=bool(b['t'][0]>=.5);active_count+=int(active)
    with native_execution(model,coordinates=b['xt']) as runtime:
     inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
     fm=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=denom);total=fm
     if active:
      pred=endpoint(v,b);a=component(first,pred,prepared[micro][first],r,'train_'+arm)
      d=component('L5',pred,prepared[micro]['L5'],r,'train_'+arm)
      total=total+ramp*(weight*a['loss']+LAMBDA5*d['loss'])/8
      totals[first]+=float(a['loss'].detach())/8;totals['L5']+=float(d['loss'].detach())/8
      if log_now:numerical.append(dict(microbatch=micro,interface=plain(a.get('diagnostics',{})),interface_counts=plain(a.get('counts',{})),L5=plain(d.get('diagnostics',{}))))
     if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite total loss')
     if step in (100,500) and active and ratio is None:
      grads,vals,diag=velocity_terms(v,b,prepared[micro],r,'training_gradient_'+arm,c,fm_denominator=denom,aux_denominator=8,allow_edge_failure=arm=='J35',skip_edge=r['calibration']['lambda_edge'] is None)
      ratio=dict(microbatch_index=micro,t=float(b['t'][0]),values=vals,gradient=gradient_record(grads,r['calibration']['lambda_edge'],ramp=ramp),
       normalization='Actual effective-batch FM denominator; each auxiliary divided by8',velocity_sha256=array_hash(v))
     inc(c,'backward_attempts');total.backward();inc(c,'backward_returns')
    totals['FM']+=float(fm.detach())
   gradient_groups=grouped(model) if log_now else None
   gradnorm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.,error_if_nonfinite=True))
   entry['optimizer_step_in_progress']=True;inc(c,'optimizer_step_attempts');optimizer.step();inc(c,'optimizer_updates')
   entry['completed_updates']=step;entry['optimizer_step_in_progress']=False;entry['batch_in_progress']=False
   torch.cuda.synchronize()
   if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):raise FloatingPointError('Nonfinite updated model')
   entry['timing']['model_update_seconds']+=time.perf_counter()-update_tick
   if log_now:
    weighted=ramp*(weight*totals[first]+LAMBDA5*totals['L5'])
    row=dict(step=step,cumulative_updates=2500+step,FM=totals['FM'],geometry_gated_means={first:totals[first],'L5':totals['L5']},
      weighted_geometry=weighted,total=totals['FM']+weighted,active_samples=active_count,free_coordinate_denominator=denom,
      auxiliary_denominator=8,interface=first,lambda_interface=weight,lambda5=LAMBDA5,ramp=ramp,
      grad_norm_before_clip=gradnorm,gradients=gradient_groups,parameter_updates=grouped(model,before),
      velocity_gradient=ratio if ratio is not None else {'status':'NONE','reason':'No active sample in this batch' if step in (100,500) else 'NOT_SCHEDULED'},
      numerical_diagnostics=numerical,runtime=runtime,paired_stream_sha256=seal(audit))
    entry['logs'].append(row);del before
    event('pilot_training_log',arm=arm,step=step,FM=totals['FM'],weighted_geometry=weighted,active=active_count)
   if step in (250,500):
    cp_tick=time.perf_counter();before_stream=seal(stream.state_dict())
    cp=checkpoint.save(root/f'additional_step{step}_cumulative_step{2500+step}.pt',model,optimizer,stream,arm=arm,step=step,r=r)
    entry['checkpoints'][str(step)]=cp
    if step==500:r['assets']['endpoints'][arm]=cp
    if seal(stream.state_dict())!=before_stream:raise RuntimeError('Checkpoint save advanced input RNG')
    entry['timing']['checkpoint_seconds']+=time.perf_counter()-cp_tick
   if step%25==0 or log_now:
    entry['elapsed_seconds']=time.perf_counter()-tick;write(r,out)
    if not log_now:event('pilot_training_progress',arm=arm,updates=step,seconds=round(entry['elapsed_seconds'],2))
  entry.update(status='PASS',final_stream_state=stream.state_dict(),paired_stream_sha256=seal(entry['stream_audit']))
  entry['timing']['wall_seconds']=time.perf_counter()-tick
  if entry['paired_stream_sha256']!=PAIRED_STREAM_SHA:raise RuntimeError('Final500-batch stream hash differs')
  if c.get('optimizer_updates')!=500 or c.get('forward_returns')!=4000 or c.get('OT_returns')!=4000:raise RuntimeError('Training accounting differs')
  source_guard(r);event('pilot_arm_complete',arm=arm,endpoint=entry['checkpoints']['500'])
 except Exception as exc:
  entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());event('pilot_arm_failed',arm=arm,updates=entry['completed_updates'],error=repr(exc))
 finally:
  entry['distribution_work_at_exit']=work_counters()
  if optimizer is not None:del optimizer
  if model is not None:del model
  release();write(r,out)

def train_all(r,out):
 history=json.loads((OLD/'results.json').read_text());old=history['arms']['L0']['stream_audit']
 if seal(old)!=PAIRED_STREAM_SHA:raise RuntimeError('Old stream reference changed')
 for name in ('L3','L5'):
  if history['arms'][name]['stream_audit']!=old:raise RuntimeError('Old factorial arms did not share stream')
 bank=TargetBank()
 for arm in ARMS:train_one(r,out,arm,bank,old)
 r['training_total_updates']=sum(a['completed_updates'] for a in r['arms'].values())
 if r['training_total_updates']>1000:raise RuntimeError('Training cap exceeded')
 write(r,out)
