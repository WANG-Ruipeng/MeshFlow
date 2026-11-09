"""One actual-velocity calibration and bounded zero-update CUDA wiring."""
from .common import *
from native_t1.geometry_checkpoint import load_geometry_checkpoint
from native_t1.accounting import count_ot_calls
from native_t1.data import native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.training import masked_fm_loss
import traceback

def inc(c,k):c[k]=c.get(k,0)+1
def velocity_grad(loss,v,c,retain_graph=False):
 inc(c,'velocity_autograd_attempts')
 grad=torch.autograd.grad(loss,v,retain_graph=retain_graph)[0]
 inc(c,'velocity_autograd_returns')
 if not bool(torch.isfinite(grad).all()):raise FloatingPointError('Nonfinite velocity gradient')
 return grad
def endpoint(v,b):
 mask=b['loss_mask']
 return (b['xt']+(1-b['t'][:,None,None])*v)[mask].reshape(-1,3,3)
def forward(model,b,c):
 with native_execution(model,coordinates=b['xt']):
  inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
 return v
def strict_initial(device='cuda'):
 m,a=load_geometry_checkpoint(INITIAL,device=device)
 if a['sha256']!=INITIAL_SHA or a['state_sha256']!=INITIAL_STATE or a['cumulative_updates']!=2500 or a['parameters']!=PARAMETERS:
  raise RuntimeError('Common pure-FM G0 identity differs')
 return m,a
def preflight(r,out):
 if 'calibration' in r:raise RuntimeError('Calibration may run once only')
 tick=time.perf_counter();configure_stable_runtime()
 c=r['counts']['calibration'];bank=TargetBank();stream=stream_for(r)
 with count_ot_calls(c):samples=stream.next_effective_batch()
 labels=[bank.sample(s) for s in samples]
 r['calibration']=dict(status='RUNNING',candidates={},states=[],clone_start_sha256=seal(r['stream_start']),
  clone_end_sha256=seal(stream.state_dict()),formal_stream_advanced=False,model_forward_budget=16)
 # Canonical GT relation coverage without model/noise calls.
 r['applicability']={}
 for p in (0,1):
  for K in (4,8,12):
   lab=bank.case(p,K);target=prepare_target('L1',lab)
   r['applicability'][f'P{p}_K{K}']=plain(target['counts'])
 candidates=r['calibration']['candidates']
 for arm in CANDIDATES:candidates[arm]=dict(status='PENDING',rows=[])
 write(r,out)
 model,load_audit=strict_initial();r['calibration']['load']=load_audit
 target_cache={}
 for arm in CANDIDATES:
  try:
   for i,lab in enumerate(labels):target_cache[i,arm]=prepare_target(arm,lab)
  except Exception as exc:
   candidates[arm].update(status='CALIBRATION_FAILED',reason='Target preparation failed',error=repr(exc))
 values={};actual_batches=[]
 try:
  for i,sample in enumerate(samples):
   for t in (.5,.75):
    b=native_collate([sample],'cuda',t_override=t)
    with torch.no_grad():v=forward(model,b,c)
    values[f's{i}_t{str(t).replace(".","p")}_velocity']=v.cpu().numpy()
    row=dict(index=len(actual_batches),sample_index=i,t=t,parent=int(sample['parent_index']),K=int(sample['K']),
      source_mapping_sha256=labels[i]['mapping_sha256'],original_shared_hash=sample['shared_input_sha256'],
      actual_xt_sha256=array_hash(b['xt'].cpu().numpy()),velocity_sha256=array_hash(v.cpu().numpy()))
    r['calibration']['states'].append(row);actual_batches.append((i,t))
    leaf=v.detach().requires_grad_(True)
    fm=masked_fm_loss(leaf,b['u'],b['valid_mask'],b['known_mask'])
    fmgrad=velocity_grad(fm,leaf,c);fmnorm=float(torch.linalg.vector_norm(fmgrad))
    for arm in CANDIDATES:
     rec=dict(state_index=row['index'],sample_index=i,t=t,FM_gradient_L2=fmnorm,FM=float(fm.detach()))
     if candidates[arm]['status']=='CALIBRATION_FAILED':
      rec.update(finite=False,status='NOT_RUN_AFTER_CANDIDATE_FAILURE')
      candidates[arm]['rows'].append(rec);continue
     try:
      aux=evaluate_loss(arm,endpoint(leaf,b),target_cache[i,arm],r,'calibration')
      grad=velocity_grad(aux['loss'],leaf,c)
      rec.update(applicable=bool(aux['applicable']),loss=float(aux['loss'].detach()),gradient_L2=float(torch.linalg.vector_norm(grad)),
        finite=True,diagnostics=plain(aux.get('diagnostics',{})),counts=plain(aux['counts']))
     except Exception as e:
      rec.update(finite=False,error=repr(e));candidates[arm]['status']='CALIBRATION_FAILED'
     candidates[arm]['rows'].append(rec)
    event('calibration_state',state=row['index']+1,total=16)
    write(r,out)
  np.savez_compressed(Path(out)/'calibration_velocities.npz',**values)
  r['calibration']['velocity_archive']=record(Path(out)/'calibration_velocities.npz')
  for arm,entry in candidates.items():
   rows=entry['rows'];app=[x for x in rows if x.get('applicable') and x.get('finite')]
   nonzero=[x['gradient_L2'] for x in app if x['gradient_L2']>0]
   entry.update(applicable_states=len(app),total_states=16,applicable_fraction=len(app)/16,
    zero_gradient_fraction=sum(x['gradient_L2']==0 for x in app)/len(app) if app else None)
   if entry['status']=='CALIBRATION_FAILED':continue
   if not app:entry['status']='NOT_APPLICABLE';continue
   if not nonzero:entry['status']='CALIBRATION_FAILED';entry['reason']='No finite nonzero velocity gradient';continue
   numerator=float(np.median([x['FM_gradient_L2'] for x in app]));denominator=float(np.median(nonzero))
   entry.update(status='PASS',lambda_geometry=.1*numerator/denominator,
     median_FM_gradient=numerator,median_nonzero_geometry_gradient=denominator)
   entry['lambda']=entry.pop('lambda_geometry')
   if not np.isfinite(entry['lambda']) or entry['lambda']<=0:
    entry.update(status='CALIBRATION_FAILED',reason='Nonfinite or nonpositive calibration weight');entry.pop('lambda',None)
  if c['forward_attempts']!=16 or c['forward_returns']!=16:raise RuntimeError('Calibration forward budget differs')
  # Oracle checks use the same real labels and FP32 path, no model call.
  oracle={}
  for arm in CANDIDATES:
   if candidates[arm]['status']!='PASS':
    oracle[arm]=dict(status='NOT_RUN_AFTER_CALIBRATION_FAILURE');continue
   errs=[]
   try:
    for i in range(8):
     b=native_collate([samples[i]],'cuda',t_override=.75)
     result=evaluate_loss(arm,endpoint(b['u'],b),target_cache[i,arm],r,'oracle_real')
     errs.append(float(result['loss']))
    oracle[arm]=dict(status='PASS',max_abs=max(abs(x) for x in errs),values=errs)
    if max(abs(x) for x in errs)>2e-6:raise RuntimeError('Actual FP32 oracle did not reach numerical zero')
   except Exception as exc:
    oracle[arm]=dict(status='FAILED',values=errs,error=repr(exc))
    candidates[arm].update(status='ORACLE_FAILED',reason=repr(exc))
  r['preflight_oracle']=oracle
  # One additional native forward/backward per applicable loss proves its
  # gradient reaches the actual generator under strict deterministic BF16.
  model.set_trainable(True).eval();wc=r['counts']['wiring'];r['wiring']={}
  for arm in ARMS:
   if arm!='L0' and candidates[arm]['status']!='PASS':
    r['arms'][arm]['status']=candidates[arm]['status'];continue
   i,t=actual_batches[0] if arm=='L0' else actual_batches[next(x['state_index'] for x in candidates[arm]['rows'] if x.get('finite') and x.get('applicable') and x['gradient_L2']>0)]
   b=native_collate([samples[i]],'cuda',t_override=t);model.zero_grad(set_to_none=True)
   try:
    with native_execution(model,coordinates=b['xt']) as runtime:
     inc(wc,'forward_attempts');v=model(*model_inputs(b)).float();inc(wc,'forward_returns')
     if arm=='L0':loss=masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask'])
     else:loss=evaluate_loss(arm,endpoint(v,b),target_cache[i,arm],r,'wiring')['loss']
     inc(wc,'backward_attempts');loss.backward();inc(wc,'backward_returns')
    groups={group:magnitude([p.grad for name,p in model.named_parameters() if predicate(name)]) for group,predicate in
       [('backbone',lambda n:n.startswith('backbone.')),('Geo',lambda n:n.startswith('context_encoder.')),('role',lambda n:n=='role_embedding')]}
    if not all(x['finite'] for x in groups.values()) or groups['backbone']['L2']<=0:raise RuntimeError('Auxiliary backward failed to reach finite backbone gradients')
    r['wiring'][arm]=dict(status='PASS',gradients=groups,runtime=runtime,optimizer_updates=0)
    r['arms'][arm]['status']='READY'
   except Exception as e:
    r['wiring'][arm]=dict(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    r['arms'][arm]['status']='WIRING_FAILED'
   finally:
    model.zero_grad(set_to_none=True);write(r,out)
  after=state_sha256(model);r['wiring_state_sha256_after']=after
  if after!=INITIAL_STATE:raise RuntimeError('Zero-update preflight changed generator parameters')
  r['calibration']['status']='PASS';r['calibration']['seconds']=time.perf_counter()-tick
  write(r,out)
 finally:
  del model,target_cache;release()
 event('preflight_complete',weights={a:e.get('lambda') for a,e in candidates.items()},status={a:e['status'] for a,e in r['arms'].items()})
