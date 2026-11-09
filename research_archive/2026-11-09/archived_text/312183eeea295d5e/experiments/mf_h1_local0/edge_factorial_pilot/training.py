"""Only Edge-only, exactly 500 matched updates; no L5 path in backward."""
from .common import *
from . import checkpoint
from native_t1.losses.edge import prepare_edge_target
from native_t1.training import make_optimizer,masked_fm_loss
from native_t1.model import native_execution
from native_t1.accounting import count_ot_calls
import traceback

def velocity_probe(v,b,target,denom,ramp,c,r):
 leaf=v.detach().requires_grad_(True)
 fm=masked_fm_loss(leaf,b['u'],b['valid_mask'],b['known_mask'],denominator=denom)
 inc(c,'velocity_autograd_attempts');gf=torch.autograd.grad(fm,leaf)[0];inc(c,'velocity_autograd_returns')
 edge=component(endpoint(leaf,b),target,r,'training_velocity_probe')['loss']/8
 inc(c,'velocity_autograd_attempts');ge=torch.autograd.grad(edge,leaf)[0];inc(c,'velocity_autograd_returns')
 nf=float(torch.linalg.vector_norm(gf));ne=float(torch.linalg.vector_norm(ge))
 if not bool(torch.isfinite(gf).all() and torch.isfinite(ge).all()):raise FloatingPointError('Velocity probe nonfinite')
 cosine=float((gf*ge).sum())/(nf*ne) if nf*ne else None
 return dict(t=float(b['t'][0]),FM_L2=nf,edge_L2=ne,weighted_edge_L2=ramp*LAMBDA_EDGE*ne,
  weighted_ratio=ramp*LAMBDA_EDGE*ne/nf if nf else None,cosine=cosine,
  angle_degrees=float(np.degrees(np.arccos(np.clip(cosine,-1,1)))) if cosine is not None else None,
  lambda_edge=LAMBDA_EDGE,ramp=ramp,FM_denominator=denom,edge_denominator=8,
  velocity_sha256=array_hash(v),L5_executed=False)

def train(r,out=OUT):
 if r.get('training',{}).get('status')!='READY' or r.get('wiring',{}).get('status')!='PASS':
  raise RuntimeError('Frozen registration and passing wiring required; no automatic resume/retry')
 source_guard(r)
 history=json.loads((OLD/'results.json').read_text());old=history['arms']['L0']['stream_audit']
 pilot=json.loads((PILOT/'results.json').read_text())
 if seal(old)!=PAIRED_STREAM_SHA or history['arms']['L5']['stream_audit']!=old or pilot['arms']['JEdge5']['stream_audit']!=old:
  raise RuntimeError('Old arms input streams differ')
 entry=r['training'];c=r['counts']['training'];model=None;optimizer=None;tick=time.perf_counter()
 try:
  model,entry['initial_identity']=strict_initial();model.set_trainable(True).eval();optimizer=make_optimizer(model)
  if optimizer.state or any(p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters()):raise RuntimeError('Fresh/full FP32 contract')
  stream=stream_for(r);bank=TrainingGeometryTargets(archive())
  entry.update(status='RUNNING',fresh_optimizer=True,logs=[],stream_audit=[],checkpoints={},
   timing=dict(CPU_OT_and_stream_seconds=0.,supervision_prepare_seconds=0.,model_update_seconds=0.,checkpoint_seconds=0.))
  root=Path(out)/'Edge_only';root.mkdir(exist_ok=False);write(r,out)
  for step in range(1,501):
   if c.get('optimizer_step_attempts',0)>=500:raise RuntimeError('Hard 500-update cap')
   start=time.perf_counter()
   with count_ot_calls(c):samples=stream.next_effective_batch()
   entry['timing']['CPU_OT_and_stream_seconds']+=time.perf_counter()-start
   start=time.perf_counter();labels=[bank.sample(s) for s in samples];audit=audit_batch(stream,samples,step,labels)
   if audit!=old[step-1]:raise RuntimeError('Training inputs differ from all three old arms before update')
   entry['stream_audit'].append(audit);prepared=[prepare_edge_target(label) for label in labels]
   entry['timing']['supervision_prepare_seconds']+=time.perf_counter()-start
   denom=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
   if denom not in (7524,7452,7488):raise RuntimeError('FM denominator differs')
   log_now=step in (1,50,100,250,500);before=tracked_state(model) if log_now else None
   fm_total=0.;edge_total=0.;active_count=0;probe=None;ramp=min(step/50.,1.)
   optimizer.zero_grad(set_to_none=True);entry['batch_in_progress']=step;start=time.perf_counter()
   for micro,sample in enumerate(samples):
    b=native_collate([sample],'cuda');active=bool(b['t'][0]>=.5);active_count+=int(active)
    with native_execution(model,coordinates=b['xt']) as runtime:
     inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
     fm=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'],denominator=denom);total=fm
     if active:
      edge=component(endpoint(v,b),prepared[micro],r,'training')['loss']
      total=total+ramp*(LAMBDA_EDGE*edge)/8;edge_total+=float(edge.detach())/8
     if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite total')
     if step in (100,500) and active and probe is None:
      probe=dict(microbatch_index=micro,**velocity_probe(v,b,prepared[micro],denom,ramp,c,r))
     inc(c,'backward_attempts');total.backward();inc(c,'backward_returns')
    fm_total+=float(fm.detach())
   grads=grouped(model) if log_now else None
   norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.,error_if_nonfinite=True))
   entry['optimizer_step_in_progress']=True;inc(c,'optimizer_step_attempts');optimizer.step();inc(c,'optimizer_updates')
   entry['completed_updates']=step;entry['optimizer_step_in_progress']=False;entry['batch_in_progress']=False
   torch.cuda.synchronize()
   if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):raise FloatingPointError('Nonfinite updated model')
   entry['timing']['model_update_seconds']+=time.perf_counter()-start
   if log_now:
    row=dict(step=step,cumulative_updates=2500+step,FM=fm_total,edge_gated_mean=edge_total,
     weighted_geometry=ramp*LAMBDA_EDGE*edge_total,total=fm_total+ramp*LAMBDA_EDGE*edge_total,
     active_samples=active_count,free_coordinate_denominator=denom,auxiliary_denominator=8,ramp=ramp,lambda_edge=LAMBDA_EDGE,
     grad_norm_before_clip=norm,gradients_before_clip=grads,parameter_updates=grouped(model,before),runtime=runtime,
     velocity_gradient=probe if probe is not None else dict(status='NONE',reason='NO_ACTIVE' if step in (100,500) else 'NOT_SCHEDULED'),
     paired_stream_sha256=seal(audit),L5_forward_calls=0,L5_backward_calls=0)
    entry['logs'].append(row);del before;event('edge_training_log',step=step,FM=fm_total,weighted_edge=row['weighted_geometry'])
   if step in (250,500):
    start=time.perf_counter();stream_hash=seal(stream.state_dict())
    cp=checkpoint.save(root/f'additional_step{step}_cumulative_step{2500+step}.pt',model,optimizer,stream,step,r)
    entry['checkpoints'][str(step)]=cp
    if step==500:r['assets']['endpoints']['Edge_only']=cp
    if seal(stream.state_dict())!=stream_hash:raise RuntimeError('Checkpoint advanced stream')
    entry['timing']['checkpoint_seconds']+=time.perf_counter()-start
   if step%25==0 or log_now:
    entry['elapsed_seconds']=time.perf_counter()-tick;write(r,out)
    if not log_now:event('edge_training_progress',updates=step,seconds=entry['elapsed_seconds'])
  entry['paired_stream_sha256']=seal(entry['stream_audit']);entry['stream_end']=stream.state_dict()
  if entry['paired_stream_sha256']!=PAIRED_STREAM_SHA:raise RuntimeError('Final stream hash differs')
  if (c['optimizer_updates'],c['forward_returns'],c['backward_returns'],c['OT_returns'])!=(500,4000,4000,4000):raise RuntimeError('Update accounting differs')
  # One fixed velocity before save/reload, with no update and no new OT.
  model.zero_grad(set_to_none=True);model.set_trainable(False).eval()
  b=native_collate([load_samples()[0]],'cuda',t_override=.5)
  with torch.no_grad():v,_=forward(model,b,r['counts']['wiring'])
  reference=v.detach().cpu().numpy().copy();np.save(root/'reload_velocity_reference.npy',reference)
  del v,b,optimizer;optimizer=None;del model;model=None;release()
  model,a=checkpoint.load(entry['checkpoints']['500']['path'],protocol_sha=r['protocol_sha256'])
  b=native_collate([load_samples()[0]],'cuda',t_override=.5)
  with torch.no_grad():v,_=forward(model,b,r['counts']['wiring'])
  if reference.tobytes()!=v.cpu().numpy().tobytes():raise RuntimeError('Strict reload velocity differs')
  entry['reload_check']=dict(status='PASS',strict_load=a,velocity_sha256=array_hash(reference),bitwise=True,additional_forwards=2)
  entry['status']='PASS';source_guard(r)
 except BaseException as exc:
  entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False);raise
 finally:
  entry.setdefault('timing',{})['wall_seconds']=time.perf_counter()-tick
  optimizer=None;model=None;release();write(r,out)
