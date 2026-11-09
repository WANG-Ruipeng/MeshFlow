"""Shared assets, supervision identity mapping and accounting for the fixed loss screen."""
from native_t1.runtime import configure_stable_runtime
import copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.data import load_archive,NativeTrainingStream
REPO=Path(__file__).resolve().parents[3]
PACKAGE=Path(__file__).resolve().parent
INITIAL=REPO/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt'
INITIAL_SHA='3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e'
INITIAL_STATE='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392'
ORIGINAL_SOURCE_SHA='1b51c87a8f99ece10c4040ab20bd3b5e6d0d5bebf7cd3d5981311e6202f8360b'
INPUTS=REPO/'output/t1_publish_20260930/chair_n112.npz'
INPUT_SHA='7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
BANK=REPO/'native_t1/runs/geometry_context_revised_20260930_153000/registration/evaluation_inputs.npz'
BANK_SHA='2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e'
HISTORY=REPO/'native_t1/runs/geometry_rep_alignment_20260930_205100/results.json'
HISTORY_SHA='935ce942c9d0244757179c43d28d06e68090b8aa454c7e39d20fd62f8ca589de'
HEAD='5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c'
ARMS=tuple('L'+str(i) for i in range(7))
CANDIDATES=ARMS[1:]
DEV_SEEDS=tuple(range(9901,9909))
CONF_SEEDS=tuple(range(10001,10009))
LOG_STEPS=(1,10,50,100,250,500)
PARAMETERS=130401155
def record(path):
 p=Path(path).resolve()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha256(p))
def seal(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(r,out): atomic_json(Path(out)/'results.json',r)
def event(name,**kw): print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():
 gc.collect()
 if torch.cuda.is_initialized():torch.cuda.empty_cache()
def plain(value):
 if torch.is_tensor(value):
  if value.numel()!=1:raise TypeError('Only scalar diagnostics may be serialized')
  return value.detach().item()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [plain(v) for v in value]
 return value
def magnitude(values):
 values=[v.detach().float() for v in values if v is not None]
 if not values:return dict(RMS=0.,L2=0.,numel=0,nonzero=0,finite=True)
 n=sum(v.numel() for v in values);s=sum(float(v.square().sum()) for v in values)
 return dict(RMS=(s/n)**.5,L2=s**.5,numel=n,nonzero=sum(int(torch.count_nonzero(v)) for v in values),
             finite=all(bool(torch.isfinite(v).all()) for v in values))
def audit_batch(stream,samples,step,labels):
 return dict(step=step,batch_index=stream.batch_index,sample_index=stream.sample_index,
  shared=[s['shared_input_sha256'] for s in samples],C=[s['context_sha256'] for s in samples],
  target_labels=[v['mapping_sha256'] for v in labels],parents=[s['parent_index'] for s in samples],
  K=[s['K'] for s in samples],t=[float(s['t']) for s in samples],stream_sha256=seal(stream.state_dict()))
def stream_for(r):
 stream=NativeTrainingStream(load_archive(INPUTS,INPUT_SHA),seed=r['stream_start']['seed'])
 stream.load_state_dict(copy.deepcopy(r['stream_start']))
 if seal(stream.state_dict())!=r['protocol']['training']['initial_stream_sha256']:raise RuntimeError('Initial input RNG differs')
 return stream

class TargetBank:
 """Labels only: trusted indexed GT is never returned by model_inputs."""
 def __init__(self):
  self.archive=load_archive(INPUTS,INPUT_SHA);self.parents={}
  with np.load(INPUTS,allow_pickle=False) as f:
   for p in (0,1):
    V=f[f'parent{p}_source_vertices'].copy();F=f[f'parent{p}_source_face_vertex_ids'].copy()
    Y=f[f'parent{p}_target'].reshape(112,3,3).copy()
    if V.dtype!=np.float32 or F.dtype!=np.int64 or not np.array_equal(V[F].view(np.uint32),Y.view(np.uint32)):
     raise RuntimeError('Trusted raw-indexed archive does not reconstruct actual target')
    self.parents[p]=dict(V=V,F=F,Y=Y)
 def sample(self,sample):
  parent=int(sample['parent_index']);K=int(sample['K']);entry=self.parents[parent]
  if int(sample['donor_index'])!=parent:raise ValueError('Only correct-context training')
  ids=np.asarray(sample['source_face_ids']);corners=np.asarray(sample['corner_permutations'])
  known=sample['known_mask'];valid=sample['valid_mask']
  if sorted(ids.tolist())!=list(range(112)) or not valid.all() or int(known.sum())!=K:raise ValueError('N/mask/source map changed')
  if not np.array_equal(np.sort(corners,axis=1),np.broadcast_to(np.arange(3),(112,3))):raise ValueError('Invalid corner map')
  vids=np.take_along_axis(entry['F'][ids],corners,axis=1)
  coordinates=entry['V'][vids];Y=sample['x1'].reshape(112,3,3);C=sample['context'].reshape(112,3,3)
  if coordinates[~known].tobytes()!=Y[~known].tobytes() or coordinates[known].tobytes()!=C[known].tobytes():
   raise RuntimeError('Source labels not aligned with actual target slots (OT must reorder Z only)')
  L=self.archive.cases[parent,K].bbox_diagonal
  return dict(free_gt=Y[~known].copy(),known=C[known].copy(),free_ids=vids[~known].copy(),known_ids=vids[known].copy(),
   bbox_L=L,parent=parent,K=K,mapping_sha256=seal(dict(ids=array_hash(vids),Y=array_hash(Y[~known]),C=array_hash(C[known]),mask=array_hash(known))))
 def case(self,parent,K):
  case=self.archive.cases[parent,K];entry=self.parents[parent]
  return dict(free_gt=case.free_target.reshape(-1,3,3).copy(),known=case.constraints.copy(),
   free_ids=entry['F'][case.free_source_ids].copy(),known_ids=entry['F'][case.source_face_ids].copy(),
   bbox_L=case.bbox_diagonal,parent=parent,K=K)
def prepare_target(name,label):
 if name in ('L1','L2','L3','L6'):
  from .relation_losses import prepare_relations
  return prepare_relations(label['free_gt'],label['known'],label['free_ids'],label['known_ids'],label['bbox_L'])
 if name in ('L4','L5'):
  from .distribution_losses import prepare_distribution_target
  return prepare_distribution_target(torch.from_numpy(label['free_gt']),label['bbox_L'])
 raise ValueError(name)
def evaluate_loss(name,pred,target,r=None,stage=None):
 if name in ('L1','L2','L3','L6'):
  from .relation_losses import relation_loss
  fn=relation_loss
 else:
  from .distribution_losses import distribution_loss
  fn=distribution_loss
 if r is not None:
  ledger=r['loss_accounting'].setdefault(stage,{}).setdefault(name,dict(calls_attempted=0,calls_returned=0,counters={}))
  ledger['calls_attempted']+=1
 with torch.autocast(device_type=pred.device.type,enabled=False):
  output=fn(name,pred,target)
 if output['loss'].ndim or not bool(torch.isfinite(output['loss'])):raise FloatingPointError(name+' nonfinite/nonscalar geometry loss')
 if r is not None:
  ledger['calls_returned']+=1
  for key,value in plain(output.get('counts',{})).items():
   if isinstance(value,(int,float)) and not isinstance(value,bool):ledger['counters'][key]=ledger['counters'].get(key,0)+value
 return output
