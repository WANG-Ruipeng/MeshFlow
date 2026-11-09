"""Explicit training schema and generator-only inference for representation pilot."""
from .runtime import configure_stable_runtime
from collections.abc import Mapping
from pathlib import Path
import torch
from models.equidit import DiT
from models.utils import get_embedder
from .model import NativeInpaintingModel
from .context_geometry import ContextGeometryEncoder, ENCODER_METADATA
from .checkpoint import DEFAULT_CONFIG
from .portable_checkpoint import _read_config, config_sha256
from .artifacts import file_sha256,state_sha256

SCHEMA='native_t1_geo_rep_training_v1'
ARMS=('G0','G_Angular','G_AngularScale')
INITIAL_SHA='c52e102530fe8d50d9f90e5d9df7a5dadff89655b792a019305d373cf83a0571'
INITIAL_STATE='1742a8aad06ed082dd7d21f91829f88d3ef118e0cfcf1d3c4cd8a6c853b3bf4e'
GENERATOR_PARAMETERS=130401155

def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return value

def validate(payload,config_path):
    if not isinstance(payload,Mapping) or payload.get('schema')!=SCHEMA:
        raise ValueError('Explicit representation training schema required')
    arm=payload.get('arm')
    if arm not in ARMS: raise ValueError('Unknown representation arm')
    if payload.get('initial_checkpoint_sha256')!=INITIAL_SHA or payload.get('initial_state_sha256')!=INITIAL_STATE:
        raise ValueError('Wrong authorized Geo2000 parent')
    step=payload.get('additional_updates')
    if step not in (250,500) or payload.get('cumulative_updates')!=2000+step:
        raise ValueError('Unexpected saved update boundary')
    if payload.get('feature_layer')!=9 or payload.get('method')!='analytic-geometry representation alignment pilot':
        raise ValueError('Wrong representation identity')
    if payload.get('optimizer_step_in_progress') is not False or payload.get('batch_in_progress') is not False:
        raise ValueError('Incomplete update')
    if payload.get('config_sha256')!=config_sha256(config_path):
        raise ValueError('Configuration changed')
    context=payload.get('context',{})
    if context.get('context_mode')!='geo' or context.get('additional_parameters')!=149888:
        raise ValueError('Wrong Geo identity')
    if any(context.get(k)!=v for k,v in ENCODER_METADATA.items()):
        raise ValueError('Input Geo formula changed')
    state=payload.get('generator_state')
    if not isinstance(state,Mapping) or not state or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 for v in state.values()):
        raise ValueError('Complete FP32 generator state required')
    if (payload.get('aux_projector_state') is None)!=(arm=='G0') or (payload.get('aux_scale_head_state') is None)!=(arm!='G_AngularScale'):
        raise ValueError('Auxiliary head identity mismatch')
    stream=payload['stream_state']
    if (stream['seed'],stream['batch_index'],stream['sample_index'],stream['arm'])!=(1010,500+step,4000+8*step,'Native_correct'):
        raise ValueError('Input stream progress differs')
    return payload

def save_training_checkpoint(path,model,heads,optimizer,stream,*,arm,step,stats,calibration,protocol_sha256,config_path=DEFAULT_CONFIG):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pt.partial')
    if path.exists() or temp.exists(): raise FileExistsError(path)
    payload=dict(schema=SCHEMA,arm=arm,method='analytic-geometry representation alignment pilot',
        generator_state=cpu(model.state_dict()),
        aux_projector_state=None if heads is None else cpu(heads.P.state_dict()),
        aux_scale_head_state=None if heads is None or heads.S is None else cpu(heads.S.state_dict()),
        generator_state_sha256=state_sha256(model),
        projector_state_sha256=None if heads is None else state_sha256(heads.P),
        scale_state_sha256=None if heads is None or heads.S is None else state_sha256(heads.S),
        context=model.context_encoder.metadata(),optimizer=cpu(optimizer.state_dict()),
        target_description_and_statistics=stats,calibration=calibration,
        feature_layer=9,ramp='min(additional_optimizer_step/50,1)',head_initialization_seed=20261001,
        optimizer_reset_at_start=True,optimizer_step_in_progress=False,batch_in_progress=False,
        stream_state=stream.state_dict(),initial_checkpoint_sha256=INITIAL_SHA,initial_state_sha256=INITIAL_STATE,
        additional_updates=step,cumulative_updates=2000+step,protocol_sha256=protocol_sha256,
        config_sha256=config_sha256(config_path))
    validate(payload,config_path)
    torch.save(payload,temp);temp.replace(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path),
        state_sha256=payload['generator_state_sha256'],projector_sha256=payload['projector_state_sha256'],
        scale_sha256=payload['scale_state_sha256'],cumulative_updates=2000+step,schema=SCHEMA)

def load_representation_generator(path,config_path=DEFAULT_CONFIG,*,device='cuda'):
    """Instantiate ONLY unchanged T1+Geo. No GT, P/S class or target computation."""
    path=Path(path);payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    validate(payload,config_path);config=_read_config(Path(config_path))
    configure_stable_runtime()
    with torch.device('meta'):
        model=NativeInpaintingModel(DiT(**config),trainable=False)
    model.context_encoder=ContextGeometryEncoder('geo',seed=payload['context']['initialization_seed'])
    model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
    keys=model.load_state_dict(payload['generator_state'],strict=True,assign=True)
    actual=state_sha256(model)
    if actual!=payload['generator_state_sha256'] or sum(p.numel() for p in model.parameters())!=GENERATOR_PARAMETERS:
        raise ValueError('Generator tensor identity mismatch')
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()): raise ValueError('Nonfinite generator')
    model.set_trainable(False).eval().to(device)
    return model,dict(status='PASS',path=str(path.resolve()),sha256=file_sha256(path),schema=SCHEMA,
        arm=payload['arm'],state_sha256=actual,strict=True,missing_keys=list(keys.missing_keys),
        unexpected_keys=list(keys.unexpected_keys),parameters=GENERATOR_PARAMETERS,auxiliary_modules_constructed=0,
        target_attributes_read_by_network=False,cumulative_updates=payload['cumulative_updates'])

def load_auxiliary_heads(path,*,device='cuda'):
    """Separate explicit training/teacher loader; never called by sampling."""
    from .geometry_representation import AuxiliaryHeads
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    validate(payload,DEFAULT_CONFIG)
    if payload['arm']=='G0': return None
    heads=AuxiliaryHeads(use_scale=payload['arm']=='G_AngularScale',seed=payload['head_initialization_seed'],device=device)
    heads.P.load_state_dict(payload['aux_projector_state'],strict=True)
    if heads.S is not None: heads.S.load_state_dict(payload['aux_scale_head_state'],strict=True)
    if state_sha256(heads.P)!=payload['projector_state_sha256'] or (heads.S is not None and state_sha256(heads.S)!=payload['scale_state_sha256']):
        raise ValueError('Strict auxiliary reload differs')
    return heads.eval()
