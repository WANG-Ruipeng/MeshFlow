"""Pilot definitions; old L3/L5 are reused without extraction or modification."""
from ..loss_screen.common import *
from ..loss_screen.common import prepare_target as old_prepare, evaluate_loss as old_loss
from ..loss_screen.calibration import inc, endpoint, velocity_grad, strict_initial, forward

PACKAGE=Path(__file__).resolve().parent
OLD=REPO/'experiments/mf_h1_local0/outputs/loss_screen_20260930_232200'
ARMS=('J35','JEdge5')
MODELS=('L0','L3','L5','J35','JEdge5')
LOG_STEPS=(1,50,100,250,500)
SEEDS=(10101,10102,10103,10104)
LAMBDA3=3.6819396517801244
LAMBDA5=0.3266104383520167
START_STREAM_SHA='ca35c12af30f1b3ec9d6d5e9d191edf0a1dc6c7b61584b6976c0ac6c004c93a7'
PAIRED_STREAM_SHA='609071dc47dd8805a43f79e4259ed649a39bee5808cb37978c2d939e5981e11c'
OLD_STATES={'L0':'f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b',
 'L3':'663c2ed010b8a5363e34984eefb42eb1243391292741423937d4377e41e57026',
 'L5':'3c596db36fd7d5836427906fedecd7485b7be3e62cc6f605dbcb25b62c0c6641'}

def targets(label):
 from .edge_loss import prepare_edge_target
 return {'L3':old_prepare('L3',label),'L5':old_prepare('L5',label),'D_edge':prepare_edge_target(label)}

def component(name,pred,target,r,stage):
 if name in ('L3','L5'):return old_loss(name,pred,target,r,stage)
 if name!='D_edge':raise ValueError(name)
 from .edge_loss import edge_loss
 ledger=r['loss_accounting'].setdefault(stage,{}).setdefault(name,dict(calls_attempted=0,calls_returned=0,counters={}))
 ledger['calls_attempted']+=1
 with torch.autocast(device_type=pred.device.type,enabled=False):result=edge_loss(pred,target)
 if result['loss'].ndim or not bool(torch.isfinite(result['loss'])):raise FloatingPointError('Nonfinite edge loss')
 ledger['calls_returned']+=1
 for k,v in plain(result.get('counts',{})).items():
  if isinstance(v,(int,float)) and not isinstance(v,bool):ledger['counters'][k]=ledger['counters'].get(k,0)+v
 return result

def cos(a,b):
 denom=float(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))
 return float((a.flatten()*b.flatten()).sum())/denom if denom>0 else None

def gradient_record(grads,lambda_edge,*,ramp=1.,gate=True):
 norms={k:float(torch.linalg.vector_norm(v)) if v is not None else None for k,v in grads.items()}
 g35=LAMBDA3*grads['L3']+LAMBDA5*grads['L5']
 ge5=lambda_edge*grads['D_edge']+LAMBDA5*grads['L5'] if lambda_edge is not None and grads['D_edge'] is not None else None
 out=dict(norms=norms,cos_L3_L5=cos(grads['L3'],grads['L5']),cos_edge_L5=cos(grads['D_edge'],grads['L5']) if grads['D_edge'] is not None else None,joint={})
 for name,g in [('J35',g35),('JEdge5',ge5)]:
  if g is None:
   out['joint'][name]=dict(status='NOT_RUN_INVALID_EDGE_CALIBRATION');continue
  n=float(torch.linalg.vector_norm(g));out['joint'][name]=dict(L2=n,ratio_to_FM=n/norms['FM'] if norms['FM'] else None,
   cos_to_FM=cos(g,grads['FM']),applied_L2=ramp*float(gate)*n,
   applied_ratio_to_FM=ramp*float(gate)*n/norms['FM'] if norms['FM'] else None)
 out.update(ramp=ramp,gate=bool(gate),lambda3=LAMBDA3,lambda5=LAMBDA5,lambda_edge=lambda_edge)
 return out

def velocity_terms(v,b,target,r,stage,c,*,fm_denominator=None,aux_denominator=1,allow_edge_failure=False,skip_edge=False):
 from native_t1.training import masked_fm_loss
 leaf=v.detach().requires_grad_(True)
 fm=masked_fm_loss(leaf,b['u'],b['valid_mask'],b['known_mask'],denominator=fm_denominator)
 grads={'FM':velocity_grad(fm,leaf,c)};values={'FM':float(fm.detach())};diagnostics={}
 for name in ('L3','L5','D_edge'):
  if name=='D_edge' and skip_edge:
   grads[name]=None;values[name]=None;diagnostics[name]=dict(status='NOT_RUN_AFTER_EDGE_FAILURE',applicable=False);continue
  try:
   result=component(name,endpoint(leaf,b),target[name],r,stage)
   value=float(result['loss'].detach());g=velocity_grad(result['loss']/aux_denominator,leaf,c)
   if not np.isfinite(float(torch.linalg.vector_norm(g))):raise FloatingPointError('Nonfinite gradient norm')
   values[name]=value;diagnostics[name]=dict(plain(result.get('diagnostics',{})),applicable=bool(result['applicable']))
   grads[name]=g
  except Exception as exc:
   if name!='D_edge' or not allow_edge_failure:raise
   grads[name]=None;values[name]=None;diagnostics[name]=dict(status='FAILED',applicable=False,error=repr(exc))
 return grads,values,diagnostics

def save_samples(path,samples):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 values={f's{i}_{k}':np.asarray(v) for i,s in enumerate(samples) for k,v in s.items()}
 if any(v.dtype.hasobject for v in values.values()):raise TypeError('No pickle sample archive')
 with path.open('xb') as f:np.savez_compressed(f,**values)
 return record(path)

def load_samples(r):
 a=r['assets']['teacher_samples'];p=Path(a['path'])
 if file_sha256(p)!=a['sha256']:raise RuntimeError('Fixed teacher samples changed')
 with np.load(p,allow_pickle=False) as f:
  samples=[]
  for i in range(8):
   prefix=f's{i}_';samples.append({k[len(prefix):]:f[k].item() if f[k].ndim==0 and f[k].dtype.kind in 'US' else f[k].copy() for k in f.files if k.startswith(prefix)})
 return samples
