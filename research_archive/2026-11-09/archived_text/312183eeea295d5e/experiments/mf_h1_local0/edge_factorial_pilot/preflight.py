"""Bounded identity, archived-value parity and three-forward zero-update wiring."""
from .common import *
from native_t1.losses.edge import prepare_edge_target,edge_loss
from . import legacy_edge_loss
from native_t1.training import masked_fm_loss
from native_t1.model import native_execution

def preflight(r,out=OUT):
 if 'wiring' in r:raise RuntimeError('No silent preflight repeat')
 source_guard(r);samples=load_samples();bank=TrainingGeometryTargets(archive());old=json.loads((PILOT/'results.json').read_text())
 prepared=[prepare_edge_target(bank.sample(s)) for s in samples]
 old_prepared=[legacy_edge_loss.prepare_edge_target(bank.sample(s)) for s in samples]
 rows=[];c=r['counts']['CPU_parity'];p=Path(old['calibration']['reused_velocity_archive']['path'])
 if file_sha256(p)!=old['calibration']['reused_velocity_archive']['sha256']:raise RuntimeError('Archived velocity changed')
 with np.load(p,allow_pickle=False) as saved:
  for i,sample in enumerate(samples):
   lab=bank.sample(sample)
   for t in (.5,.75):
    b=native_collate([sample],'cpu',t_override=t);idx=2*i+int(t==.75);expected=old['calibration']['states'][idx]
    if array_hash(b['xt'].numpy())!=expected['actual_xt_sha256'] or lab['mapping_sha256']!=expected['source_mapping_sha256']:raise RuntimeError('Teacher state/source mapping differs')
    velocity=saved[f's{i}_t'+('0p5' if t==.5 else '0p75')+'_velocity'].copy()
    values=[];grads=[]
    for fn,target in ((edge_loss,prepared[i]),(legacy_edge_loss.edge_loss,old_prepared[i])):
     leaf=torch.from_numpy(velocity.copy()).requires_grad_(True)
     inc(c,'edge_loss_calls');value=fn(endpoint(leaf,b),target)['loss']
     inc(c,'velocity_autograd_attempts');gradient=torch.autograd.grad(value,leaf)[0];inc(c,'velocity_autograd_returns')
     values.append(value.detach());grads.append(gradient)
     if bool(gradient[b['known_mask']].any()):raise RuntimeError('Known slots received loss gradient')
    if not torch.equal(values[0],values[1]) or not torch.equal(grads[0],grads[1]):raise RuntimeError('Production/archive Edge parity failed')
    rows.append(dict(state_index=idx,t=t,value=float(values[0]),value_bitwise=True,velocity_gradient_bitwise=True,
     actual_xt_sha256=expected['actual_xt_sha256'],source_mapping_sha256=lab['mapping_sha256']))
  reference=saved['s0_t0p5_velocity'].copy()
 # Oracle and actual source/corner permutation correspondence require no model.
 oracle=[]
 for i,sample in enumerate(samples):
  b=native_collate([sample],'cpu',t_override=.5)
  inc(c,'edge_loss_calls');result=edge_loss(endpoint(b['u'],b),prepared[i])
  if float(result['loss'])>1e-12:raise RuntimeError('Oracle target mismatch')
  oracle.append(float(result['loss']))
 for target in prepared:
  if any(torch.is_tensor(v) and v.requires_grad for v in target.values()):raise RuntimeError('GT target has gradient')
 r['CPU_parity']=dict(status='PASS',rows=rows,oracle_losses=oracle,GT_requires_grad=False,
  exact_archived_formula=True,model_input_fields=['xt','t','N','valid_mask','known_mask'],new_model_inputs=False,
  old_BF16_permutation_gate=dict(status='INHERITED_FAIL_NOT_RERUN',failed_coordinates=10,total_coordinates=1008,atol=.015625,rtol=.03125,max_abs=.064453125,RMS=.0088499709,source='native_t1/docs/tgeo_pilot.md',tolerance_relaxed=False))
 write(r,out)
 c=r['counts']['wiring'];model=None;entry=dict(status='RUNNING',optimizer_updates=0);r['wiring']=entry;write(r,out)
 try:
  model,entry['load']=strict_initial();b=native_collate([samples[0]],'cuda',t_override=.5)
  with torch.no_grad():v0,rt=forward(model,b,c);v1,_=forward(model,b,c)
  if not torch.equal(v0,v1) or reference.tobytes()!=v0.cpu().numpy().tobytes():raise RuntimeError('Original G0/off or repeated velocity differs')
  model.set_trainable(True).eval();model.zero_grad(set_to_none=True)
  with native_execution(model,coordinates=b['xt']) as rt:
   inc(c,'forward_attempts');v2=model(*model_inputs(b)).float();inc(c,'forward_returns')
   if not torch.equal(v0,v2):raise RuntimeError('Trainable state changes forward')
   loss=component(endpoint(v2,b),prepared[0],r,'wiring')['loss']*LAMBDA_EDGE/8
   inc(c,'backward_attempts');loss.backward();inc(c,'backward_returns')
  gradients=grouped(model)
  if any(not v['finite'] or v['L2']<=0 for v in gradients.values()):raise RuntimeError('Auxiliary model gradient absent/nonfinite')
  if b['xt'].requires_grad or b['context'].requires_grad:
   raise RuntimeError('Input geometry acquired gradient')
  model.zero_grad(set_to_none=True)
  if state_sha256(model)!=INITIAL_STATE:raise RuntimeError('Zero-update wiring changed parameters')
  entry.update(status='PASS',off_matches_archived_G0_bitwise=True,repeated_bitwise=True,trainable_forward_bitwise=True,
   gradients_from_edge_only=gradients,edge_weighted_loss=float(loss.detach()),runtime=rt,
   actual_parameter_updates=0,GT_C_requires_grad=False,model_forwards=3,model_backwards=1)
 finally:
  model=None;release();write(r,out)

