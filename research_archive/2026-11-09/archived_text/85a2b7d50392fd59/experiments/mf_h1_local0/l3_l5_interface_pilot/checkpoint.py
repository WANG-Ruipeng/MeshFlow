"""Explicit pilot checkpoints; unchanged Native T1+Geo inference architecture."""
from .common import *
from ..loss_screen.checkpoint import cpu
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256,_read_config,NATIVE_META
from native_t1.context_geometry import ContextGeometryEncoder,ENCODER_METADATA
from native_t1.model import NativeInpaintingModel
from models.equidit import DiT
from models.utils import get_embedder
SCHEMA='native_t1_geo_l3_l5_interface_pilot_v1'

def validate(p,protocol_sha=None):
 if p.get('schema')!=SCHEMA or p.get('arm') not in ARMS:raise ValueError('Wrong pilot schema/arm')
 if p.get('initial_checkpoint_sha256')!=INITIAL_SHA or p.get('initial_state_sha256')!=INITIAL_STATE:raise ValueError('Wrong G0_2500')
 step=p.get('additional_updates')
 if step not in (250,500) or p.get('base_cumulative_updates')!=2500 or p.get('cumulative_updates')!=2500+step:raise ValueError('Wrong update identity')
 if p.get('native_meta')!=NATIVE_META or p.get('config_sha256')!=config_sha256(DEFAULT_CONFIG):raise ValueError('Architecture differs')
 context=p.get('context',{})
 if any(context.get(k)!=v for k,v in ENCODER_METADATA.items()) or context.get('context_mode')!='geo' or context.get('additional_parameters')!=149888:raise ValueError('Geo differs')
 if p.get('batch_in_progress') is not False or p.get('optimizer_step_in_progress') is not False:raise ValueError('Incomplete update')
 if p.get('gate')!='FP32_t>=0.5' or p.get('ramp')!='min(additional_step/50,1)':raise ValueError('Schedule changed')
 if protocol_sha is not None and p.get('protocol_sha256')!=protocol_sha:raise ValueError('Protocol differs')
 if any(v.dtype!=torch.float32 for v in p['model'].values()):raise ValueError('FP32 state required')
 if p.get('auxiliary_network_parameters')!=0 or p.get('optimizer_reset_at_start') is not True:raise ValueError('Unexpected network/resume')
 if p['lambdas']['L3']!=LAMBDA3 or p['lambdas']['L5']!=LAMBDA5:raise ValueError('Historical fixed coefficient changed')
 if seal(p['lambdas'])!=p.get('coefficients_sha256'):raise ValueError('Coefficient seal changed')
 if p['arm']=='JEdge5' and (not isinstance(p['lambdas']['D_edge'],(int,float)) or not np.isfinite(p['lambdas']['D_edge']) or p['lambdas']['D_edge']<=0):raise ValueError('Invalid edge coefficient')
 start=p['stream_start'];end=p['stream_state']
 if seal(start)!=START_STREAM_SHA or (end['seed'],end['arm'])!=(1010,'Native_correct') or end['batch_index']!=1000+step or end['sample_index']!=8000+8*step:raise ValueError('Stream identity changed')
 return p

def save(path,model,optimizer,stream,*,arm,step,r):
 path=Path(path);tmp=path.with_suffix('.pt.partial')
 if path.exists() or tmp.exists():raise FileExistsError(path)
 payload=dict(schema=SCHEMA,arm=arm,native_meta=dict(NATIVE_META),model=cpu(model.state_dict()),
  model_state_sha256=state_sha256(model),context=model.context_encoder.metadata(),optimizer=cpu(optimizer.state_dict()),
  optimizer_reset_at_start=True,batch_in_progress=False,optimizer_step_in_progress=False,
  initial_checkpoint_sha256=INITIAL_SHA,initial_state_sha256=INITIAL_STATE,base_cumulative_updates=2500,
  additional_updates=step,cumulative_updates=2500+step,config_sha256=config_sha256(DEFAULT_CONFIG),
  input_sha256=r['assets']['inputs']['sha256'],stream_start=r['stream_start'],stream_state=stream.state_dict(),
  protocol_sha256=r['protocol_sha256'],lambdas={'L3':LAMBDA3,'L5':LAMBDA5,'D_edge':r['calibration']['lambda_edge']},
  coefficients_sha256=r['calibration']['coefficients_sha256'],gate='FP32_t>=0.5',ramp='min(additional_step/50,1)',auxiliary_network_parameters=0,
  loss_definition=r['protocol']['losses'][arm],supervision_ids_are_model_inputs=False)
 validate(payload,r['protocol_sha256']);torch.save(payload,tmp);tmp.replace(path)
 return dict(**record(path),state_sha256=payload['model_state_sha256'],schema=SCHEMA,cumulative_updates=2500+step)

def load(path,*,protocol_sha,device='cuda'):
 path=Path(path);p=torch.load(path,map_location='cpu',weights_only=True,mmap=True);validate(p,protocol_sha)
 config=_read_config(Path(DEFAULT_CONFIG));configure_stable_runtime()
 with torch.device('meta'):model=NativeInpaintingModel(DiT(**config),trainable=False)
 model.context_encoder=ContextGeometryEncoder('geo',seed=p['context']['initialization_seed'])
 model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
 keys=model.load_state_dict(p['model'],strict=True,assign=True);actual=state_sha256(model)
 if actual!=p['model_state_sha256'] or sum(v.numel() for v in model.parameters())!=PARAMETERS:raise ValueError('State/count differs')
 if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise FloatingPointError('Nonfinite endpoint')
 model.set_trainable(False).eval().to(device)
 return model,dict(**record(path),status='PASS',arm=p['arm'],state_sha256=actual,strict=True,
  missing_keys=list(keys.missing_keys),unexpected_keys=list(keys.unexpected_keys),parameters=PARAMETERS,cumulative_updates=p['cumulative_updates'],
  optimizer_restored=False,auxiliary_network_parameters=0,geometry_loss_modules_constructed=0,coefficients_sha256=p['coefficients_sha256'])

def load_endpoint(r,name,device='cuda'):
 info=r['assets']['endpoints'][name]
 if file_sha256(info['path'])!=info['sha256']:raise ValueError('Endpoint file changed')
 if name in OLD_STATES:
  from ..loss_screen.checkpoint import load as old_load
  model,audit=old_load(info['path'],device=device,protocol_sha=r['old_protocol_sha256'])
 else:
  model,audit=load(info['path'],device=device,protocol_sha=r['protocol_sha256'])
  if audit['coefficients_sha256']!=r['calibration']['coefficients_sha256']:raise ValueError('Endpoint coefficient differs from frozen calibration')
 if audit['state_sha256']!=info['state_sha256'] or audit['cumulative_updates']!=3000 or audit['parameters']!=PARAMETERS:
  raise ValueError('Strict endpoint identity differs')
 return model,audit
