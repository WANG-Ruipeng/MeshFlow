"""Exactly sixteen fixed old states per endpoint; no training or OT."""
from .common import *
from .checkpoint import load_endpoint
from ..loss_screen.training import source_guard
from native_t1.data import native_collate,model_inputs
from native_t1.training import masked_fm_loss
from native_t1.model import native_execution
import traceback

def evaluate_teacher(r,out):
 if 'teacher' in r:raise RuntimeError('Fixed teacher diagnostic may execute only once')
 if any(r['arms'][a]['status']!='PASS' or r['arms'][a]['completed_updates']!=500 or '500' not in r['arms'][a]['checkpoints'] for a in ARMS):raise RuntimeError('Both fixed500 endpoints required before any teacher call')
 if r.get('calibration',{}).get('status')!='PASS' or len(r['calibration']['states'])!=16 or any(r.get('wiring',{}).get(a,{}).get('status')!='PASS' for a in ARMS):raise RuntimeError('Preflight gates incomplete')
 if set(r['assets']['endpoints'])!=set(MODELS):raise RuntimeError('All five frozen endpoints required')
 source_guard(r);samples=load_samples(r);bank=TargetBank();labels=[bank.sample(s) for s in samples]
 prepared=[targets(lab) for lab in labels];c=r['counts']['teacher'];tick=time.perf_counter()
 r['teacher']=dict(status='RUNNING',states=16,models={},sample_archive=r['assets']['teacher_samples'],
  normalization='FM per-sample valid free coordinate MSE; geometry original per-sample formula, no gate/ramp/lambda',
  actual_saved_endpoint_updates=3000,model_forward_cap=80,noise_OT_calls=0,geometry_autograd_calls=0)
 try:
  for name in MODELS:
   entry=dict(status='STARTING',rows=[]);r['teacher']['models'][name]=entry;model=None;mt=time.perf_counter()
   try:
    model,entry['load']=load_endpoint(r,name)
    with torch.no_grad():
     for i,sample in enumerate(samples):
      for t in (.5,.75):
       idx=2*i+int(t==.75);b=native_collate([sample],'cuda',t_override=t)
       expected=r['calibration']['states'][idx]
       if array_hash(b['xt'].cpu().numpy())!=expected['actual_xt_sha256'] or labels[i]['mapping_sha256']!=expected['source_mapping_sha256']:raise RuntimeError('Teacher state changed')
       if c.get('forward_attempts',0)>=80:raise RuntimeError('Teacher80 forward cap')
       with native_execution(model,coordinates=b['xt']) as runtime:
        inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
       scores={'FM':float(masked_fm_loss(v,b['u'],b['valid_mask'],b['known_mask']))};diag={}
       pred=endpoint(v,b)
       for term in ('L3','L5','D_edge'):
        loss=component(term,pred,prepared[i][term],r,'teacher_'+name);scores[term]=float(loss['loss'])
        diag[term]=dict(applicable=bool(loss['applicable']),counts=plain(loss['counts']),diagnostics=plain(loss.get('diagnostics',{})))
       if not all(np.isfinite(x) for x in scores.values()):raise FloatingPointError('Nonfinite teacher metric')
       entry['rows'].append(dict(state_index=idx,sample_index=i,parent=int(sample['parent_index']),K=int(sample['K']),t=t,
        **scores,diagnostics=diag,actual_xt_sha256=expected['actual_xt_sha256'],runtime=runtime))
    entry['state_after']=state_sha256(model)
    if entry['state_after']!=entry['load']['state_sha256']:raise RuntimeError('Teacher modified frozen model')
    entry.update(status='PASS',means={term:float(np.mean([x[term] for x in entry['rows']])) for term in ('FM','L3','L5','D_edge')})
    event('pilot_teacher_complete',model=name,states=16,means=entry['means'])
   except Exception as exc:
    entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
   finally:
    entry['seconds']=time.perf_counter()-mt;model=None;release();write(r,out)
  r['teacher']['status']='PASS'
 finally:
  r['teacher']['seconds']=time.perf_counter()-tick;write(r,out)
