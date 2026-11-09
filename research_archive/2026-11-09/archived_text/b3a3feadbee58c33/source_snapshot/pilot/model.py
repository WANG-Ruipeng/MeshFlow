"""S8_W512_KD: exact old Chair decoder math, smaller independently loaded state.

make_student(seed=..., activation_checkpointing=False) -> (model, spec)
model(x_d: FP32[1,N,9], face_count: long[1]|None) -> delta only.
effective_proposal(model, x_d, face_count=None) -> (FP32 proposal, FP32 effective r).
Caller owns BF16 autocast, scaling, free mask/loss, C restoration and accounting.
No forward below uses AutoencoderKL.forward/encode, adds noise or detaches C.
Student construction/load never opens or loads the official teacher checkpoint.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from r2.pilots.anchor_aware_denoiser_v1.model import (
    _official_imports, load_official_wrapper, OFFICIAL_COMMIT,
    OFFICIAL_PARAMETERS, OFFICIAL_STATE_KEYS, OFFICIAL_SHA256,
)

STUDENT='S8_W512_KD'
SCHEMA='single_step_denoiser_student_inference_v1'
KWARGS=dict(hidden_dim=512, num_layers=8, num_heads=8, latent_channels=3,
            decoder_type='reg', num_bins=256, use_rmsnorm=True, face_bin=100,
            use_identity_encoder=True, fixed_std=.02)


def tensor_state_sha256(state: dict[str,torch.Tensor]) -> str:
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


class StudentDenoiser(nn.Module):
    def __init__(self, base: nn.Module, seed: int):
        super().__init__(); self.base=base; self.initialization_seed=int(seed)

    def forward(self, x_d: torch.Tensor, face_count: torch.Tensor|None=None) -> torch.Tensor:
        if x_d.dtype != torch.float32 or x_d.ndim!=3 or x_d.shape[0]!=1 or x_d.shape[-1]!=9:
            raise ValueError('Student accepts only FP32 unpadded [1,N,9]')
        n=int(x_d.shape[1])
        if not 1<=n<=800: raise ValueError('Complete input N must be in 1..800')
        if face_count is None: face_count=torch.tensor([n],device=x_d.device,dtype=torch.long)
        if face_count.dtype!=torch.long or face_count.shape!=(1,) or face_count.device!=x_d.device or int(face_count.item())!=n:
            raise ValueError('Face count must be complete N, long[1], on the input device')
        # One PE -> one projection -> unchanged official decoder. No role,
        # noise, padding mask, repeated projection or coordinate addition.
        z=self.base.pe_nerf(x_d.reshape(1,3*n,3))
        return self.base.decode(z,face_count,mask=None)


def audit_student(model: StudentDenoiser) -> dict[str,Any]:
    b=model.base; d=b.decoder; layers=d.layers
    assert len(layers)==8 and b.post_quant_linear.out_features==512
    assert all(layer.dim==512 and layer.num_heads==8 and layer.version==3 for layer in layers)
    assert b.encoder is None and b.quant_linear is None and b.use_identity_encoder
    assert d.decoder_type=='reg' and d.face_cond and d.face_bin==100 and d.num_classes==8 and d.uncond_y==801
    assert d.y_embedder.embedding_table.num_embeddings==9 and d.y_embedder.dropout_prob==0
    assert b.pe_nerf.num_freqs==12 and b.pe_nerf.include_input and b.pe_nerf.output_dim==75
    assert not hasattr(d,'t_embedder') and not hasattr(model,'e_known')
    assert all(p.dtype==torch.float32 for p in model.parameters())
    dropouts=[]
    for name,m in model.named_modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            dropouts.append(dict(name=name,p=m.p));assert m.p==0
        if hasattr(m,'dropout') and isinstance(m.dropout,(int,float)):
            dropouts.append(dict(name=name,p=m.dropout));assert m.dropout==0
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    unused_modules={'base.decoder.vertex_cross_attn':sum(p.numel() for p in d.vertex_cross_attn.parameters())}
    assert total<=50_000_000
    state=model.state_dict()
    return dict(student=STUDENT,actual_hidden_dim=512,actual_num_layers=len(layers),actual_num_heads=8,
        actual_version=d.version,actual_encoder='identity/no encoder parameters',decoder='regression residual',
        total_parameters=total,trainable_parameters=trainable,state_keys=len(state),
        parameter_fp32_tensor_bytes=4*total,state_tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        teacher_parameters=OFFICIAL_PARAMETERS,teacher_state_keys=OFFICIAL_STATE_KEYS,
        teacher_fp32_parameter_bytes=4*OFFICIAL_PARAMETERS,student_teacher_parameter_ratio=total/OFFICIAL_PARAMETERS,
        teacher_student_parameter_factor=OFFICIAL_PARAMETERS/total,
        unused_modules=unused_modules,unused_parameter_count=sum(unused_modules.values()),
        unused_path_explanation='version3 bypasses the parameter-free multiplicative vertex_cross_attn; no duplicate encoder/teacher',
        face_bin=100,max_N=800,face_condition='N//100, full N, AdaLN',
        face_embedding_rows=9,face_embedding_extra_row='row8 retained; original CFG row also accepts full N=800; not deleted',
        face_count_dropout_actual=0.,other_dropout_modules=dropouts,
        NeRF=dict(num_freqs=12,include_input=True,output_dim=75,frequency_bands=b.pe_nerf.freq_bands.tolist(),formula='sin/cos(x*2^k*pi), k=0..11'),
        layer_norms=[type(layers[0].norm1).__name__,type(layers[0].norm2).__name__],
        MLP_actual=str(layers[0].mlp),MLP_ratio=4.,MLP_activation='SiLU (actual DiTLayer default; config use_swiglu is not forwarded)',
        activation_checkpointing=[x.gradient_checkpointing for x in layers],
        initialization='official AutoencoderKL/Model initialization, random seed; no teacher tensor copied',
        post_quant_initialization='torch.nn.Linear default (outside Model.initialize_weights)',
        decoder_initialization='official Xavier Linear, N(0,.02) label embedding, last residual Linear exact zero',
        final_residual_zero=bool(torch.count_nonzero(d.final_layer.linear.weight).item()==0 and torch.count_nonzero(d.final_layer.linear.bias).item()==0),
        known_feature_detach=False,known_role=False,noise_in_forward=False,
        official_source_commit=OFFICIAL_COMMIT,init_kwargs=copy.deepcopy(KWARGS))


def make_student(seed: int, activation_checkpointing: bool=False) -> tuple[StudentDenoiser,dict[str,Any]]:
    # Reuse verified source code, not teacher parameters. Constructor defaults
    # deliberately allocate the legacy 9-row N table before dropout is disabled.
    with _official_imports() as (module,attention,manifest):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            base=module.AutoencoderKL(**KWARGS)
        base.decoder.y_embedder.dropout_prob=0.
        for layer in base.decoder.layers: layer.gradient_checkpointing=bool(activation_checkpointing)
        for layer in base.modules():
            if isinstance(layer,attention.SelfAttention):
                if layer.dropout!=0 or layer.causal:raise ValueError('Unexpected attention randomness/causality')
                layer.mixed_precision='fp32'  # SDPA math selector, not a cast.
            if isinstance(layer,nn.modules.dropout._DropoutNd):layer.p=0.
        model=StudentDenoiser(base,seed)
        spec=audit_student(model)
        spec.update(initialization_seed=int(seed),official_source_manifest=manifest,
                    initial_state_sha256=tensor_state_sha256(model.state_dict()),
                    caller_CPU_RNG_unchanged=True,teacher_checkpoint_loaded=False)
    return model,spec


def load_teacher(*, verify_checkpoint_hash: bool=True):
    model,spec=load_official_wrapper(anchor=False,activation_checkpointing=False,
                                     verify_checkpoint_hash=verify_checkpoint_hash)
    model.eval().requires_grad_(False)
    model.validate_parameters()
    spec.update(distillation_teacher='official EMA only; no adapted weights',frozen=True,
                official_parameters_trainable=False,trainable_parameters=0,face_count_dropout_eval=0.,
                additional_marker='fixed zero only; never receives known values')
    return model,spec


def effective_proposal(model: nn.Module, x_d: torch.Tensor, face_count: torch.Tensor|None=None):
    """Exactly one residual call plus FP32 add/sub; autograd is retained."""
    if isinstance(model,StudentDenoiser): delta=model(x_d,face_count)
    else:
        if bool(getattr(model,'anchor',True)):raise ValueError('Only zero-role official teacher is allowed')
        delta=model(x_d,None,face_count)
    with torch.autocast(device_type=x_d.device.type,enabled=False):
        proposal=x_d.float()+delta.float()
        residual=proposal-x_d.float()
    return proposal,residual


def export_student(path: str|Path, model: StudentDenoiser, step: int, extra: dict|None=None):
    """Only FP32 student state plus small metadata; writes exclusively new OUT."""
    from . import common as C
    path=C.checked(path)
    spec=audit_student(model)
    state={k:v.detach().cpu().float().contiguous() for k,v in model.state_dict().items()}
    payload=dict(schema=SCHEMA,student=STUDENT,step=int(step),initialization_seed=model.initialization_seed,
                 activation_checkpointing=bool(model.base.decoder.layers[0].gradient_checkpointing),
                 official_source_commit=OFFICIAL_COMMIT,model=state,spec=spec,extra=extra or {},
                 state_sha256=tensor_state_sha256(state))
    C.atomic(path,lambda f:torch.save(payload,f),binary=True)
    receipt=dict(status='PASS',path=str(path),sha256=C.sha(path),actual_file_bytes=path.stat().st_size,
                 step=int(step),state_sha256=payload['state_sha256'],**spec)
    C.write(path.with_suffix('.json'),receipt)
    return receipt


def load_student(path: str|Path, require_step: int|None=4000):
    """Strict independent CPU load. No teacher model/state is opened or created."""
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if payload['schema']!=SCHEMA or payload['student']!=STUDENT or payload['official_source_commit']!=OFFICIAL_COMMIT:
        raise ValueError('Wrong student export identity')
    if require_step is not None and int(payload['step'])!=require_step:raise ValueError('Wrong student endpoint')
    model,spec=make_student(payload['initialization_seed'],payload['activation_checkpointing'])
    state=payload['model']
    if any(x.dtype!=torch.float32 for x in state.values()):raise ValueError('Student export must be FP32')
    if tensor_state_sha256(state)!=payload['state_sha256']:raise ValueError('Student state hash mismatch')
    info=model.load_state_dict(state,strict=True)
    if info.missing_keys or info.unexpected_keys:raise ValueError('Non-strict student state')
    model.eval()
    spec.update(audit_student(model),teacher_checkpoint_loaded=False,independent_student_load=True,
                loaded_step=int(payload['step']),loaded_state_sha256=payload['state_sha256'])
    return model,spec,payload
