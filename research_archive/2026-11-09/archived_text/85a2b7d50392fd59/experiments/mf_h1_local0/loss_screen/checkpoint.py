"""Explicit full-generator checkpoints for the geometry-loss screen."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import torch
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.model import NativeInpaintingModel
from native_t1.context_geometry import ContextGeometryEncoder,ENCODER_METADATA
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256,_read_config,NATIVE_META
from native_t1.artifacts import state_sha256,file_sha256
from .common import ARMS,INITIAL_SHA,INITIAL_STATE,PARAMETERS
SCHEMA='native_t1_geo_loss_screen_v1'
def cpu(value):
 if torch.is_tensor(value):return value.detach().cpu()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 if isinstance(value,list):return [cpu(v) for v in value]
 return value
def validate(p,protocol_sha=None):
 if p.get('schema')!=SCHEMA or p.get('arm') not in ARMS:raise ValueError('Explicit loss-screen schema/arm required')
 if p.get('initial_checkpoint_sha256')!=INITIAL_SHA or p.get('initial_state_sha256')!=INITIAL_STATE:raise ValueError('Wrong common G0_2500')
 step=p.get('additional_updates')
 if step not in (250,500) or p.get('base_cumulative_updates')!=2500 or p.get('cumulative_updates')!=2500+step:raise ValueError('Update identity changed')
 if p.get('native_meta')!=NATIVE_META or p.get('config_sha256')!=config_sha256(DEFAULT_CONFIG):raise ValueError('Architecture identity differs')
 context=p.get('context',{})
 if any(context.get(k)!=v for k,v in ENCODER_METADATA.items()) or context.get('context_mode')!='geo' or context.get('additional_parameters')!=149888:raise ValueError('Geo definition differs')
 if p.get('batch_in_progress') is not False or p.get('optimizer_step_in_progress') is not False:raise ValueError('Incomplete update')
 if p.get('gate')!='FP32_t>=0.5' or p.get('ramp')!='min(additional_step/50,1)':raise ValueError('Loss schedule differs')
 if protocol_sha is not None and p.get('protocol_sha256')!=protocol_sha:raise ValueError('Protocol differs')
 if not isinstance(p.get('model'),dict) or not p['model'] or any(not torch.is_tensor(v) or v.dtype!=torch.float32 for v in p['model'].values()):raise ValueError('Full FP32 state required')
 if p.get('auxiliary_network_parameters')!=0 or p.get('optimizer_reset_at_start') is not True:raise ValueError('Unexpected auxiliary network/optimizer resume')
 start=p['stream_start'];end=p['stream_state']
 if (end['seed'],end['arm'])!=(start['seed'],start['arm']) or end['batch_index']!=start['batch_index']+step or end['sample_index']!=start['sample_index']+8*step:raise ValueError('Input stream differs')
 return p
def save(path,model,optimizer,stream,*,arm,step,r):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.pt.partial')
 if path.exists() or tmp.exists():raise FileExistsError(path)
 payload=dict(schema=SCHEMA,arm=arm,native_meta=dict(NATIVE_META),model=cpu(model.state_dict()),
 model_state_sha256=state_sha256(model),context=model.context_encoder.metadata(),
 optimizer=cpu(optimizer.state_dict()),optimizer_reset_at_start=True,batch_in_progress=False,optimizer_step_in_progress=False,
 initial_checkpoint_sha256=INITIAL_SHA,initial_state_sha256=INITIAL_STATE,
 base_cumulative_updates=2500,additional_updates=step,cumulative_updates=2500+step,
 config_sha256=config_sha256(DEFAULT_CONFIG),input_sha256=r['assets']['inputs']['sha256'],
 stream_start=r['stream_start'],stream_state=stream.state_dict(),protocol_sha256=r['protocol_sha256'],
 lambda_geometry=0. if arm=='L0' else r['calibration']['candidates'][arm]['lambda'],
 gate='FP32_t>=0.5',ramp='min(additional_step/50,1)',auxiliary_network_parameters=0,
 loss_definition=r['protocol']['losses'][arm],supervision_ids_are_model_inputs=False)
 validate(payload,r['protocol_sha256'])
 torch.save(payload,tmp);tmp.replace(path)
 return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path),
 state_sha256=payload['model_state_sha256'],schema=SCHEMA,cumulative_updates=2500+step)
def load(path,*,device='cuda',protocol_sha=None):
 path=Path(path);p=torch.load(path,map_location='cpu',weights_only=True,mmap=True);validate(p,protocol_sha)
 config=_read_config(Path(DEFAULT_CONFIG));configure_stable_runtime()
 with torch.device('meta'):model=NativeInpaintingModel(DiT(**config),trainable=False)
 model.context_encoder=ContextGeometryEncoder('geo',seed=p['context']['initialization_seed'])
 model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
 keys=model.load_state_dict(p['model'],strict=True,assign=True)
 actual=state_sha256(model)
 if actual!=p['model_state_sha256'] or sum(x.numel() for x in model.parameters())!=PARAMETERS:raise ValueError('State hash/shape/count differs')
 if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):raise ValueError('Nonfinite generator')
 model.set_trainable(False).eval().to(device)
 return model,dict(status='PASS',path=str(path.resolve()),sha256=file_sha256(path),state_sha256=actual,
 arm=p['arm'],cumulative_updates=p['cumulative_updates'],strict=True,missing_keys=list(keys.missing_keys),
 unexpected_keys=list(keys.unexpected_keys),parameters=PARAMETERS,auxiliary_network_parameters=0,
 geometry_loss_modules_constructed=0,optimizer_restored=False)
