"""Explicit Edge-only identity. No modification to the public default loader."""
from .common import *
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256,_read_config,NATIVE_META
from native_t1.context_geometry import ContextGeometryEncoder,ENCODER_METADATA
from native_t1.model import NativeInpaintingModel
from models.equidit import DiT
from models.utils import get_embedder
SCHEMA='native_t1_geo_edge_only_factorial_v1'
RECIPE=dict(name='Edge_only',lambda_edge=LAMBDA_EDGE,lambda5=0.,gate='original_FP32_t>=0.5',
 ramp='min(additional_step/50,1)',auxiliary_denominator=8,additional_parameters=0,
 edge_formula='unchanged_centered_finite_segment_GT_correspondence_five_point_trapezoid')

def cpu(v):
 if torch.is_tensor(v):return v.detach().cpu()
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return type(v)(cpu(x) for x in v)
 return v
def validate(p,protocol_sha=None):
 if p.get('schema')!=SCHEMA or p.get('arm')!='Edge_only' or p.get('recipe')!=RECIPE:raise ValueError('Not fixed Edge-only')
 if p.get('initial_checkpoint_sha256')!=INITIAL_SHA or p.get('initial_state_sha256')!=INITIAL_STATE:raise ValueError('Wrong G0')
 step=p.get('additional_updates')
 if step not in (250,500) or p.get('base_cumulative_updates')!=2500 or p.get('cumulative_updates')!=2500+step:raise ValueError('Wrong budget')
 if p.get('native_meta')!=NATIVE_META or p.get('config_sha256')!=config_sha256(DEFAULT_CONFIG):raise ValueError('Wrong architecture')
 context=p.get('context',{})
 if any(context.get(k)!=v for k,v in ENCODER_METADATA.items()) or context.get('context_mode')!='geo' or context.get('additional_parameters')!=149888:raise ValueError('Geo differs')
 if p.get('optimizer_reset_at_start') is not True or any(p.get(k) is not False for k in ('batch_in_progress','optimizer_step_in_progress')):raise ValueError('Incomplete update')
 if protocol_sha is not None and p.get('protocol_sha256')!=protocol_sha:raise ValueError('Protocol changed')
 if p.get('input_sha256')!=INPUT_SHA or seal(p['stream_start'])!=START_STREAM_SHA:raise ValueError('Input identity')
 end=p['stream_state']
 if (end['seed'],end['arm'],end['batch_index'],end['sample_index'])!=(1010,'Native_correct',1000+step,8000+8*step):raise ValueError('Stream progress')
 if any(v.dtype!=torch.float32 for v in p['model'].values()):raise ValueError('Expected FP32')
 return p
def save(path,model,optimizer,stream,step,r):
 path=Path(path);tmp=path.with_suffix('.pt.partial')
 if path.exists() or tmp.exists():raise FileExistsError(path)
 p=dict(schema=SCHEMA,arm='Edge_only',recipe=dict(RECIPE),native_meta=dict(NATIVE_META),
  model=cpu(model.state_dict()),model_state_sha256=state_sha256(model),context=model.context_encoder.metadata(),
  optimizer=cpu(optimizer.state_dict()),optimizer_reset_at_start=True,batch_in_progress=False,optimizer_step_in_progress=False,
  initial_checkpoint_sha256=INITIAL_SHA,initial_state_sha256=INITIAL_STATE,base_cumulative_updates=2500,
  additional_updates=step,cumulative_updates=2500+step,config_sha256=config_sha256(DEFAULT_CONFIG),input_sha256=INPUT_SHA,
  stream_start=r['stream_start'],stream_state=stream.state_dict(),protocol_sha256=r['protocol_sha256'])
 validate(p,r['protocol_sha256']);torch.save(p,tmp);tmp.replace(path)
 return dict(**record(path),state_sha256=p['model_state_sha256'],schema=SCHEMA,cumulative_updates=2500+step)
def load(path,*,protocol_sha=None,device='cuda'):
 p=torch.load(path,map_location='cpu',mmap=True,weights_only=True);validate(p,protocol_sha)
 configure_stable_runtime();config=_read_config(Path(DEFAULT_CONFIG))
 with torch.device('meta'):model=NativeInpaintingModel(DiT(**config),trainable=False)
 model.context_encoder=ContextGeometryEncoder('geo',seed=p['context']['initialization_seed'])
 model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
 keys=model.load_state_dict(p['model'],strict=True,assign=True);actual=state_sha256(model)
 if actual!=p['model_state_sha256'] or sum(v.numel() for v in model.parameters())!=PARAMETERS:raise ValueError('State/count differs')
 if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise FloatingPointError('Nonfinite model')
 model.set_trainable(False).eval().to(device)
 return model,dict(**record(path),status='PASS',state_sha256=actual,strict=True,missing_keys=list(keys.missing_keys),
  unexpected_keys=list(keys.unexpected_keys),parameters=PARAMETERS,cumulative_updates=p['cumulative_updates'],
  recipe=p['recipe'],loss_modules_executed=0,optimizer_restored=False)
def load_endpoint(r,name,device='cuda'):
 item=r['assets']['endpoints'][name]
 if file_sha256(item['path'])!=item['sha256']:raise ValueError('Endpoint file changed')
 if name=='Edge_only':model,a=load(item['path'],protocol_sha=r['protocol_sha256'],device=device)
 else:
  from native_t1.geometry_checkpoint import load_geometry_checkpoint
  model,a=load_geometry_checkpoint(item['path'],device=device,expected_recipe={'L0':'fm','L5':'surface','JEdge5':'edge5'}[name])
 if a['state_sha256']!=item['state_sha256'] or a['cumulative_updates']!=3000 or a['parameters']!=PARAMETERS:raise ValueError('Endpoint identity differs')
 return model,a

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
 parser.add_argument('--device',default='cpu');args=parser.parse_args()
 model,audit=load(args.checkpoint,device=args.device);print(json.dumps(audit,indent=2))
