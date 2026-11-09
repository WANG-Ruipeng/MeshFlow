"""Frozen one-arm protocol. Importing does not run models or experiments."""
from native_t1.runtime import configure_stable_runtime
import copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.losses.targets import TrainingGeometryTargets

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'native_t1/runs/edge_factorial_20261001_123000'
OLD=REPO/'experiments/mf_h1_local0/outputs/loss_screen_20260930_232200'
PILOT=REPO/'native_t1/runs/l3_l5_interface_pilot_20261001_104600'
INITIAL=REPO/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt'
INITIAL_SHA='3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e'
INITIAL_STATE='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392'
INPUTS=REPO/'output/t1_publish_20260930/chair_n112.npz'
INPUT_SHA='7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
TEACHER=PILOT/'teacher_samples.npz'
TEACHER_SHA='f7bb3e8d91c2d8be8b62a08e29ae09a355864252673cbee0d9ea7e2a6fe88836'
START_STREAM_SHA='ca35c12af30f1b3ec9d6d5e9d191edf0a1dc6c7b61584b6976c0ac6c004c93a7'
PAIRED_STREAM_SHA='609071dc47dd8805a43f79e4259ed649a39bee5808cb37978c2d939e5981e11c'
LAMBDA_EDGE=4.071385484299878
LAMBDA5=0.3266104383520167
PARAMETERS=130401155
MODELS=('L0','Edge_only','L5','JEdge5')
STATES={'L0':'f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b',
 'L5':'3c596db36fd7d5836427906fedecd7485b7be3e62cc6f605dbcb25b62c0c6641',
 'JEdge5':'b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3'}

def seal(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def record(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha256(p))
def write(r,out=OUT):atomic_json(Path(out)/'results.json',r)
def read(out=OUT):return json.loads((Path(out)/'results.json').read_text())
def inc(c,k):c[k]=c.get(k,0)+1
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():
 gc.collect()
 if torch.cuda.is_initialized():torch.cuda.empty_cache()
def plain(v):
 if torch.is_tensor(v):return v.detach().item()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [plain(x) for x in v]
 return v
def source_guard(r):
 for row in r['source_snapshot']['files']:
  if file_sha256(row['path'])!=row['sha256']:raise RuntimeError('Frozen training source changed: '+row['path'])
def archive():return load_archive(INPUTS,INPUT_SHA)
def stream_for(r):
 s=NativeTrainingStream(archive(),seed=1010);s.load_state_dict(copy.deepcopy(r['stream_start']))
 if seal(s.state_dict())!=START_STREAM_SHA:raise RuntimeError('Wrong G0 input stream')
 return s
def audit_batch(stream,samples,step,labels):
 return dict(step=step,batch_index=stream.batch_index,sample_index=stream.sample_index,
  shared=[s['shared_input_sha256'] for s in samples],C=[s['context_sha256'] for s in samples],
  target_labels=[v['mapping_sha256'] for v in labels],parents=[s['parent_index'] for s in samples],
  K=[s['K'] for s in samples],t=[float(s['t']) for s in samples],stream_sha256=seal(stream.state_dict()))
def load_samples():
 if file_sha256(TEACHER)!=TEACHER_SHA:raise RuntimeError('Teacher samples changed')
 with np.load(TEACHER,allow_pickle=False) as f:
  return [{k[len(f's{i}_'):]:f[k].item() if f[k].ndim==0 and f[k].dtype.kind in 'US' else f[k].copy()
           for k in f.files if k.startswith(f's{i}_')} for i in range(8)]
def endpoint(v,b):return (b['xt']+(1-b['t'][:,None,None])*v)[b['loss_mask']].reshape(-1,3,3)
def component(pred,target,r,stage):
 from native_t1.losses.edge import edge_loss
 ledger=r['loss_accounting'].setdefault(stage,dict(edge_attempts=0,edge_returns=0,L5_calls=0))
 inc(ledger,'edge_attempts')
 with torch.autocast(device_type=pred.device.type,enabled=False):value=edge_loss(pred,target)
 if not bool(torch.isfinite(value['loss'])):raise FloatingPointError('Nonfinite fixed edge loss')
 inc(ledger,'edge_returns');return value
def forward(model,b,c):
 from native_t1.model import native_execution
 with native_execution(model,coordinates=b['xt']) as runtime:
  inc(c,'forward_attempts');v=model(*model_inputs(b)).float();inc(c,'forward_returns')
 return v,runtime
def strict_initial(device='cuda'):
 from native_t1.geometry_checkpoint import load_geometry_checkpoint
 model,a=load_geometry_checkpoint(INITIAL,device=device,expected_recipe='fm')
 if (a['sha256'],a['state_sha256'],a['cumulative_updates'],a['parameters'])!=(INITIAL_SHA,INITIAL_STATE,2500,PARAMETERS):
  raise RuntimeError('Initial G0 identity differs')
 return model,a
def magnitude(values):
 values=[x.detach().float() for x in values if x is not None]
 n=sum(x.numel() for x in values);s=sum(float(x.square().sum()) for x in values)
 return dict(RMS=(s/n)**.5 if n else 0.,L2=s**.5,numel=n,
  nonzero=sum(int(torch.count_nonzero(x)) for x in values),finite=all(bool(torch.isfinite(x).all()) for x in values))
def tracked_state(model):return {n:p.detach().clone() for n,p in model.named_parameters()}
def grouped(model,before=None):
 predicates={'Geo':lambda n:n.startswith('context_encoder.'),'role':lambda n:n=='role_embedding',
  'backbone':lambda n:n.startswith('backbone.')}
 return {name:magnitude([p.grad if before is None else p.detach()-before[n]
  for n,p in model.named_parameters() if choose(n)]) for name,choose in predicates.items()}
