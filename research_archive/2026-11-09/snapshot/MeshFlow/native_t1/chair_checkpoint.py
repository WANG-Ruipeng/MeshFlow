"""Strict inference loading of the retained Chair Hybrid START checkpoint.

This is its own fixed historical identity, not the portable N112 Geo schema.
The original checkpoint remains intact, including AdamW/RNG/input-stream state.
Inference loads only its exact trained model and never restores an optimizer or
RNG, starts a training stream, or imports experiment orchestration.
"""
from .runtime import configure_stable_runtime

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

import torch
from models.equidit import DiT
from models.utils import get_embedder

from .artifacts import file_sha256, state_sha256
from .checkpoint import REPO, DEFAULT_CONFIG
from .chair_model import ChairModel
from .context_geometry import ContextGeometryEncoder
from .portable_checkpoint import _read_config, config_sha256


CHAIR_START_CHECKPOINT = REPO / 'native_t1/checkpoints/chair_hybrid_start_total5000.pt'
CHAIR_START_FILE_SHA256 = 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
CHAIR_START_STATE_SHA256 = '166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972'
SCHEMA = 'chair_hybrid_coupling_checkpoint_v1'
MODEL_TENSOR_COUNT = 194
MODEL_PARAMETER_COUNT = 130401155
GEO_PARAMETER_COUNT = 149888
EXPECTED_METADATA = dict(
    arm='OT_HYBRID', pilot_step=1000, conditional_total_step=5000,
    source_checkpoint_sha256='190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf',
    context_encoder='geo', loss_recipe='fm', lambda_variance=.25,
    model_state_sha256=CHAIR_START_STATE_SHA256,
    optimizer_state_sha256='7649fc3e4f65cbb2d468e3770ff503cba03bc095cbbcfba7284b9f27b762dd50',
    rng_sha256='4bac14063e6550712b062fba5b7cf728c82b11013067c500c938b458269619dc',
    stream_state_sha256='da8668d3cacfe6074f7bc45d3856d8e5fd5b3f5a9cb5708563f8c8dcf22f3ffa',
    protocol_sha256='bb5335ae1af1b57a98560ba933531c2c4d419cb2f720a85e8a029e4f28ab3a60',
    config_sha256='fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f',
    completed_updates=1000, full_recovery_state=True, optimizer_reset=False,
    model_parameters='FP32', optimizer_state='FP32',
)


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def validate_chair_start(payload, expected_config_sha256=None):
    """Validate the fixed saved identity without constructing or running a model.

    The loader additionally pins the complete original file SHA256 and verifies
    the actual model tensor hash. Saved AdamW is inspected but never restored.
    """
    if not isinstance(payload,Mapping) or payload.get('schema')!=SCHEMA:
        raise ValueError('Expected the fixed Chair Hybrid START checkpoint schema')
    if payload.get('update_in_progress') is not False:
        raise ValueError('Chair START must be saved at a completed update boundary')
    meta=payload.get('metadata')
    if not isinstance(meta,Mapping):raise ValueError('Chair START metadata is missing')
    for name,expected in EXPECTED_METADATA.items():
        actual=meta.get(name)
        if actual!=expected or type(actual) is not type(expected):
            raise ValueError('Chair START identity differs: '+name)
    if expected_config_sha256 is not None and meta['config_sha256']!=expected_config_sha256:
        raise ValueError('Chair START architecture config differs')
    for name in ('model','optimizer','rng','stream_state','protocol'):
        if not isinstance(payload.get(name),Mapping):raise ValueError('Chair START requires '+name+' mapping')
    if _json_sha256(payload['protocol'])!=meta['protocol_sha256']:
        raise ValueError('Chair START protocol hash differs')
    stream=payload['stream_state']
    if _json_sha256(stream)!=meta['stream_state_sha256']:
        raise ValueError('Chair START input-stream hash differs')
    expected_stream=dict(arm='OT_HYBRID',seed=20261004,batch_index=1000,sample_index=8000,
        lambda_value=.25,RNG_mode='stateless_per_sample_frozen_seeds',
        coupling_contract='original_preOT_cost_scale__FP32_final_target_x2__raw_iid_noise_slots_v1')
    if any(stream.get(k)!=v or type(stream.get(k)) is not type(v) for k,v in expected_stream.items()):
        raise ValueError('Chair START input-stream identity differs')
    if not {'python','numpy','torch_cpu','torch_cuda'}.issubset(payload['rng']):
        raise ValueError('Chair START full saved RNG state is missing')
    state=payload['model']
    if len(state)!=MODEL_TENSOR_COUNT or any(not isinstance(k,str) or not isinstance(v,torch.Tensor)
        or v.dtype!=torch.float32 or v.device.type!='cpu' for k,v in state.items()):
        raise ValueError('Chair START requires the complete CPU FP32 model state')
    if sum(v.numel() for v in state.values())!=MODEL_PARAMETER_COUNT:
        raise ValueError('Chair START parameter count differs')
    if any(k.startswith('relation_modules.') for k in state):
        raise ValueError('Chair START has no relation-attention parameters')
    optimizer=payload['optimizer'];groups=optimizer.get('param_groups');saved=optimizer.get('state')
    if not isinstance(groups,list) or len(groups)!=1 or not isinstance(saved,Mapping) or len(saved)!=MODEL_TENSOR_COUNT:
        raise ValueError('Chair START requires the original complete one-group AdamW state')
    group=groups[0];ids=group.get('params',[])
    if (not isinstance(ids,list) or len(ids)!=MODEL_TENSOR_COUNT or len(set(ids))!=len(ids)
        or set(ids)!=set(saved) or (group.get('lr'),tuple(group.get('betas',())),group.get('weight_decay'))!=(1e-5,(.9,.95),0.)):
        raise ValueError('Chair START AdamW ownership or settings differ')
    for parameter,sid in zip(state.values(),ids):
        entry=saved[sid]
        if not isinstance(entry,Mapping) or set(entry)!= {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Chair START AdamW state entry differs')
        step=entry['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or step.device.type!='cpu' or float(step)!=5000.:
            raise ValueError('Chair START AdamW must retain step5000 for every parameter')
        for key in ('exp_avg','exp_avg_sq'):
            value=entry[key]
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device.type!='cpu' or value.shape!=parameter.shape:
                raise ValueError('Chair START AdamW moment tensor differs')
    return dict(meta)


def load_chair_start(path=CHAIR_START_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device='cuda'):
    """Return the frozen ChairModel and audit; CPU loading performs no CUDA work."""
    tick=time.perf_counter();path=Path(path);config_path=Path(config_path)
    digest=file_sha256(path)
    if digest!=CHAIR_START_FILE_SHA256:
        raise ValueError('Expected the exact retained OT_HYBRID pilot1000/total5000 file SHA256')
    config=_read_config(config_path)
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    meta=validate_chair_start(payload,config_sha256(config_path))
    sidecar=path.with_suffix('.json')
    if sidecar.exists():
        recorded=json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if recorded.get('checkpoint_sha256')!=digest or recorded.get('metadata')!=meta:
            raise ValueError('Chair START sidecar identity differs')
    state=payload['model']
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Chair START has a nonfinite model tensor')
    configure_stable_runtime()
    with torch.device('meta'):
        model=ChairModel(DiT(**config),trainable=False)
    model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
    model.context_encoder=ContextGeometryEncoder('geo',seed=1010)
    keys=model.load_state_dict(state,strict=True,assign=True)
    model.set_trainable(False).eval()
    actual=state_sha256(model)
    if actual!=CHAIR_START_STATE_SHA256 or actual!=meta['model_state_sha256']:
        raise ValueError('Chair START actual model tensor hash differs')
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=MODEL_PARAMETER_COUNT or sum(p.numel() for p in model.context_encoder.parameters())!=GEO_PARAMETER_COUNT:
        raise ValueError('Loaded Chair START parameter count differs')
    audit=dict(status='PASS',path=str(path.resolve()),checkpoint=str(path.resolve()),
        sha256=digest,checkpoint_sha256=digest,state_sha256=actual,model_state_sha256=actual,
        schema=SCHEMA,checkpoint_schema=SCHEMA,profile='chair-hybrid',arm='OT_HYBRID',
        pilot_step=1000,completed_updates=1000,base_cumulative_updates=4000,
        conditional_total_step=5000,cumulative_updates=5000,
        context_encoder='geo',context=model.context_encoder.metadata(),
        loss_recipe='fm',lambda_variance=.25,training_objective={'name':'fm'},
        coupling_contract=payload['stream_state']['coupling_contract'],
        config_sha256=meta['config_sha256'],config_file_sha256=file_sha256(config_path),
        source_checkpoint_sha256=meta['source_checkpoint_sha256'],
        protocol_sha256=meta['protocol_sha256'],stream_state_sha256=meta['stream_state_sha256'],
        parameters=parameters,model_tensor_count=len(state),trainable_parameters=0,
        strict=True,missing_keys=list(keys.missing_keys),unexpected_keys=list(keys.unexpected_keys),
        total_faces_range_inclusive=[128,256],known_faces='0 < K < total N; separate valid/known masks',
        full_saved_recovery_state=True,saved_optimizer_state_sha256=meta['optimizer_state_sha256'],
        saved_optimizer_parameter_states=MODEL_TENSOR_COUNT,saved_optimizer_steps=[5000],
        optimizer_restored=False,rng_restored=False,optimizer_reset=False,
        model_forward_calls=0,model_backward_calls=0,optimizer_updates=0)
    del payload,state
    model.to(device)
    audit.update(device=str(device),seconds=time.perf_counter()-tick)
    return model,audit
