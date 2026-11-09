"""Reuse recorded velocities; calibrate only D_edge and observe joint gradients."""
from .common import *
from native_t1.accounting import count_ot_calls
from native_t1.data import native_collate,model_inputs,shared_sample_hash
from native_t1.training import masked_fm_loss
from native_t1.model import native_execution
import traceback

def preflight(r,out):
 if 'calibration' in r:raise RuntimeError('One calibration only')
 configure_stable_runtime();tick=time.perf_counter();c=r['counts']['calibration']
 history=json.loads((OLD/'results.json').read_text());stream=stream_for(r);bank=TargetBank()
 with count_ot_calls(c):samples=stream.next_effective_batch()
 labels=[bank.sample(s) for s in samples]
 audit=audit_batch(stream,samples,1,labels)
 if audit!=history['arms']['L0']['stream_audit'][0]:raise RuntimeError('Calibration clone differs from old stream')
 r['assets']['teacher_samples']=save_samples(Path(out)/'teacher_samples.npz',samples)
 restored=load_samples(r)
 for a,b in zip(samples,restored):
  if shared_sample_hash(a)!=shared_sample_hash(b):raise RuntimeError('Teacher sample serialization changed stream arrays')
 velocity_info=history['calibration']['velocity_archive']
 if file_sha256(velocity_info['path'])!=velocity_info['sha256']:raise RuntimeError('Archived velocity changed')
 r['calibration']=dict(status='RUNNING',states=[],rows=[],formal_stream_advanced=False,clone_start_sha256=seal(r['stream_start']),
  clone_end_sha256=seal(stream.state_dict()),reused_velocity_archive=velocity_info,new_model_forwards=0,lambda3=LAMBDA3,lambda5=LAMBDA5)
 target_cache=[targets(label) for label in labels];gradsets=[];edge_failed=False
 with np.load(velocity_info['path'],allow_pickle=False) as archive:
  for i,sample in enumerate(samples):
   for t in (.5,.75):
    index=2*i+int(t==.75);old=history['calibration']['states'][index]
    b=native_collate([sample],'cuda',t_override=t)
    v=torch.from_numpy(archive[f's{i}_t{str(t).replace(".","p")}_velocity'].copy()).cuda()
    if array_hash(b['xt'].detach().cpu().numpy())!=old['actual_xt_sha256'] or array_hash(v.detach().cpu().numpy())!=old['velocity_sha256'] or labels[i]['mapping_sha256']!=old['source_mapping_sha256'] or sample['shared_input_sha256']!=old['original_shared_hash']:
     raise RuntimeError('Fixed teacher state/velocity/label does not reproduce archive')
    grads,values,diag=velocity_terms(v,b,target_cache[i],r,'calibration',c,allow_edge_failure=True,skip_edge=edge_failed)
    if grads['D_edge'] is None:edge_failed=True
    checks={}
    for name in ('L3','L5'):
     prev=history['calibration']['candidates'][name]['rows'][index]
     checks[name]=dict(loss_exact=values[name]==prev['loss'],norm_exact=float(torch.linalg.vector_norm(grads[name]))==prev['gradient_L2'])
     if not all(checks[name].values()):raise RuntimeError('Directly reused L3/L5 differs from old saved numerical record')
    if index==0:
     eq={}
     for name in ('L3','L5'):
      leaf=v.detach().requires_grad_(True)
      result=old_loss(name,endpoint(leaf,b),target_cache[i][name],r,'direct_reuse_equivalence')
      g=velocity_grad(result['loss'],leaf,c)
      eq[name]=dict(loss_bitwise=float(result['loss'])==values[name],full_velocity_gradient_bitwise=torch.equal(g,grads[name]))
      if not all(eq[name].values()):raise RuntimeError('Reuse wrapper changed loss/gradient')
     r['calibration']['direct_equivalence']=eq
    if not all(bool(torch.isfinite(g).all()) for g in grads.values() if g is not None):raise FloatingPointError('Nonfinite calibration gradient')
    row=dict(state_index=index,sample_index=i,parent=int(sample['parent_index']),K=int(sample['K']),t=t,
      source_mapping_sha256=labels[i]['mapping_sha256'],actual_xt_sha256=array_hash(b['xt'].detach().cpu().numpy()),velocity_sha256=array_hash(v.detach().cpu().numpy()),
      values=values,old_record_checks=checks,diagnostics=diag)
    r['calibration']['states'].append({k:row[k] for k in ('state_index','sample_index','parent','K','t','actual_xt_sha256','velocity_sha256','source_mapping_sha256')})
    r['calibration']['rows'].append(row);gradsets.append(grads)
 valid=[i for i,row in enumerate(r['calibration']['rows']) if row['diagnostics']['L3']['applicable'] and row['diagnostics']['D_edge']['applicable']]
 norms3=[float(torch.linalg.vector_norm(gradsets[i]['L3'])) for i in valid]
 normsE=[float(torch.linalg.vector_norm(gradsets[i]['D_edge'])) for i in valid]
 med3=float(np.median(norms3)) if valid else None;medE=float(np.median(normsE)) if valid else None
 weight=LAMBDA3*med3/medE if not edge_failed and medE is not None and np.isfinite(medE) and medE>0 else None
 if weight is not None and (not np.isfinite(weight) or weight<=0):weight=None
 if weight is None:
  r['arms']['JEdge5'].update(status='CALIBRATION_FAILED',reason='No positive finite median edge gradient on the common applicable states; no fallback')
 r['calibration']['common_valid_state_indices']=valid
 for row,grads in zip(r['calibration']['rows'],gradsets):row['gradient']=gradient_record(grads,weight)
 r['calibration'].update(lambda_edge=weight,median_g3=med3,median_gE=medE,effective_states=len(valid),
  zero_gE_count=sum(n==0 for n in normsE),definition='lambda3*median(||g3||)/median(||gE||) on same16 finite applicable states')
 r['calibration']['coefficients_sha256']=seal(dict(L3=LAMBDA3,L5=LAMBDA5,D_edge=weight))
 # Eight real oracle geometries, no model call or new OT.
 oracle=[]
 for i,sample in enumerate(samples) if weight is not None else []:
  b=native_collate([sample],'cuda',t_override=.75)
  result=component('D_edge',endpoint(b['u'],b),target_cache[i]['D_edge'],r,'oracle_real')
  oracle.append(float(result['loss']))
 if oracle and max(map(abs,oracle))>2e-6:raise RuntimeError('Real oracle edge loss nonzero')
 r['preflight_oracle']=dict(status='PASS' if oracle else 'NOT_RUN_AFTER_EDGE_CALIBRATION_FAILURE',D_edge=oracle)
 # One total-objective forward/backward for each new branch, zero updates.
 model,load_audit=strict_initial();r['calibration']['strict_initial']=load_audit
 model.set_trainable(True).eval();wc=r['counts']['wiring'];r['wiring']={}
 try:
  for arm in ARMS:
   if arm=='JEdge5' and weight is None:continue
   b=native_collate([samples[0]],'cuda',t_override=.5);model.zero_grad(set_to_none=True)
   with native_execution(model,coordinates=b['xt']) as runtime:
    inc(wc,'forward_attempts');v=model(*model_inputs(b)).float();inc(wc,'forward_returns')
    if array_hash(v.detach().cpu().numpy())!=r['calibration']['states'][0]['velocity_sha256']:raise RuntimeError('Initial zero-update forward differs from archived velocity')
    fm=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'])
    first='L3' if arm=='J35' else 'D_edge';coeff=LAMBDA3 if arm=='J35' else weight
    pred=endpoint(v,b);a=component(first,pred,target_cache[0][first],r,'wiring_'+arm)
    d=component('L5',pred,target_cache[0]['L5'],r,'wiring_'+arm)
    loss=fm+coeff*a['loss']+LAMBDA5*d['loss']
    inc(wc,'backward_attempts');loss.backward();inc(wc,'backward_returns')
   groups={group:magnitude([p.grad for n,p in model.named_parameters() if predicate(n)]) for group,predicate in
    [('backbone',lambda n:n.startswith('backbone.')),('Geo',lambda n:n.startswith('context_encoder.')),('role',lambda n:n=='role_embedding')]}
   if not all(z['finite'] and z['L2']>0 for z in groups.values()):raise RuntimeError('Wiring gradient absent/nonfinite')
   r['wiring'][arm]=dict(status='PASS',total=float(loss.detach()),gradients=groups,runtime=runtime,optimizer_updates=0)
   r['arms'][arm]['status']='READY'
  if state_sha256(model)!=INITIAL_STATE:raise RuntimeError('Zero-update preflight modified model')
  r['wiring_state_sha256_after']=INITIAL_STATE
 finally:
  del model;release()
 r['calibration'].update(status='PASS' if weight is not None else 'PARTIAL_EDGE_FAILED',seconds=time.perf_counter()-tick)
 if c.get('forward_attempts',0)!=0:raise RuntimeError('Saved-velocity calibration unexpectedly called model')
 write(r,out);event('pilot_preflight_pass',lambda_edge=weight,reused_velocities=16,model_forwards=0,wiring_forwards=wc.get('forward_returns',0))
