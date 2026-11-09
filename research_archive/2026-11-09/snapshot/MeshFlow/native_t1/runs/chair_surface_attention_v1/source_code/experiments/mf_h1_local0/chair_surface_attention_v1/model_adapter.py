"""Experiment-local instance adapter; old parameters and public source stay intact."""
from __future__ import annotations
import hashlib
import math
import time
from types import MethodType
import torch
from torch import nn
from native_t1.model import (assert_native_runtime,native_block_forward,native_final_forward,_corners,_face_select)
from native_t1.context_geometry import inject_context
from .attention import RelationMessageMLP,attention_with_relations

ARMS=('H_CONT','MOMENT_XY','SURFACE_XY')


def _state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(a.shape),str(a.dtype))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def selected_layers(model):
    L=len(model.backbone.layers)
    if L<4:raise ValueError('Need at least four compatible native blocks')
    return tuple(math.ceil(L*f/4) for f in (1,2,3,4))


def _profile_start(model,x,name):
    if not model._relation_profile_enabled:return None
    if x.device.type=='cuda':
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        return name,a,b
    return name,time.perf_counter(),None


def _profile_end(model,token):
    if token is None:return
    name,a,b=token
    if b is not None:b.record();model._relation_profile_events.append((name,a,b))
    else:model._relation_profile_events.append((name,(time.perf_counter()-a)*1000,None))


def set_profile(model,enabled=True,*,collect_stats=False):
    model._relation_profile_enabled=bool(enabled);model._relation_collect_stats=bool(collect_stats)


def profile_summary(model,*,synchronize=True):
    events=getattr(model,'_relation_profile_events',[])
    if synchronize and any(b is not None for _,_,b in events):torch.cuda.synchronize()
    rows=[dict(site=name,milliseconds=float(a.elapsed_time(b) if b is not None else a)) for name,a,b in events]
    return dict(relation_milliseconds=sum(r['milliseconds'] for r in rows if r['site']=='relations'),
                attention_correction_milliseconds=sum(r['milliseconds'] for r in rows if r['site'].startswith('attention_')),
                events=rows,relation_build_calls=getattr(model,'_relation_build_calls',0),
                note='CUDA-event scopes are within the full forward; do not add them to full-model elapsed time.')


def _relation_block(model,block,hidden,c_free,c_known,known_mask,valid_mask,relations,layer_index,observer,face_time):
    free_mods=block.adaLN_modulation(c_free).chunk(6,dim=1)
    known_mods=block.adaLN_modulation(c_known).chunk(6,dim=1)
    mods=tuple(_face_select(f,k,known_mask) for f,k in zip(free_mods,known_mods))
    shift_a,scale_a,gate_a,shift_m,scale_m,gate_m=mods
    if observer is not None:
        observer(dict(site='block',layer=layer_index,face_time=face_time,known_mask=known_mask,
            valid_mask=valid_mask,free_mods=free_mods,known_mods=known_mods,selected_mods=mods,hidden=hidden))
    B,corners,D=hidden.shape
    face_hidden=hidden.reshape(B,corners//3,3,D).mean(dim=2)
    face_input=block.norm1(face_hidden)*(1+scale_a)+shift_a
    base,delta,stats=attention_with_relations(block.attn,face_input,valid_mask,relations,
        model.relation_modules[f'layer_{layer_index:02d}'],query_chunk=48,collect_stats=model._relation_collect_stats,
        profile=((lambda:_profile_start(model,hidden,'attention_'+str(layer_index))), (lambda token:_profile_end(model,token))) if model._relation_profile_enabled else None)
    # Original gate multiplication and original broadcast retain their order.
    face_hidden=gate_a*base
    hidden=hidden+_corners(face_hidden)
    hidden=hidden+_corners(gate_a)*delta
    hidden=hidden+_corners(gate_m)*block.mlp(block.norm2(hidden)*(1+_corners(scale_m))+_corners(shift_m),mask=valid_mask)
    if model._relation_collect_stats:model.relation_observations.append(dict(layer=layer_index,batches=stats))
    return hidden


def _forward(self,x,t,y,valid_mask=None,known_mask=None,*,observer=None):
    assert_native_runtime(self,coordinates=(x,t))
    if x.ndim!=3 or x.shape[-1]!=9:raise ValueError('Expected FP32 x[B,padded_N,9]')
    B,N,_=x.shape
    if t.shape!=(B,) or y.shape!=(B,) or y.dtype!=torch.int64:raise ValueError('Free time/total N shape differs')
    if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()) or bool(((t<0)|(t>1)).any()):raise ValueError('Invalid coordinates/time')
    for mask in (valid_mask,known_mask):
        if mask is None or mask.dtype!=torch.bool or mask.shape!=(B,N) or mask.device!=x.device:raise ValueError('Expected separate colocated masks')
    if bool((known_mask&~valid_mask).any()):raise ValueError('Known faces must be valid')
    if not torch.equal(valid_mask.sum(1),y) or bool(((y<128)|(y>256)).any()):raise ValueError('N must count natural real faces')
    if bool((known_mask.sum(1)==0).any()) or bool(((valid_mask&~known_mask).sum(1)==0).any()):raise ValueError('Formal task needs C and free')
    self._relation_profile_events=[];self.relation_observations=[];self._relation_build_calls=0
    from .relations import build_relations
    token=_profile_start(self,x,'relations')
    relations=build_relations(x,t,known_mask,valid_mask,self._surface_attention_arm,
        sigma_z=self._relation_sigma_z,static_cache=self._relation_static_cache)
    self._relation_build_calls+=1;_profile_end(self,token)
    _,hidden=self.backbone.x_embedder(x);hidden=hidden.view(B,N*3,-1)
    role=self.role_embedding[known_mask.long()].to(hidden.dtype);hidden=hidden+_corners(role)
    if self.context_encoder is not None:
        geo=self.context_encoder(x,known_mask,valid_mask,observer=self.context_observer)
        hidden=inject_context(hidden,geo,known_mask,observer=self.context_observer)
    c_free=self.backbone.t_embedder(t);c_known=self.backbone.t_embedder(torch.ones_like(t))
    count=self.backbone.y_embedder(y//self.backbone.face_bin,False);c_free,c_known=c_free+count,c_known+count
    face_time=torch.where(known_mask,torch.ones_like(t[:,None]),t[:,None])
    if observer is not None:
        observer(dict(site='embedding',hidden=hidden,role=role,known_mask=known_mask,valid_mask=valid_mask,
            face_time=face_time,c_free=c_free,c_known=c_known))
    for index,block in enumerate(self.backbone.layers,1):
        if index in self._surface_attention_layers:
            hidden=_relation_block(self,block,hidden,c_free,c_known,known_mask,valid_mask,relations,index,observer,face_time)
        else:
            hidden=native_block_forward(block,hidden,c_free,c_known,known_mask,valid_mask,
                observer=observer,layer_index=index,face_time=face_time)
    result=native_final_forward(self.backbone.final_layer,hidden,c_free,c_known,known_mask,observer=observer,face_time=face_time)
    return result.view(B,N,9)


def attach(model,arm,seed=618205,*,profile=False,sigma_z=1.0,collect_stats=False,static_cache=True):
    """Attach after strict old-model load; retain every original Parameter object."""
    if arm not in ARMS:raise ValueError('Unregistered surface attention arm')
    if hasattr(model,'_surface_attention_arm'):raise ValueError('Model instance was already attached')
    old={n:id(p) for n,p in model.named_parameters()};old_keys=set(model.state_dict())
    layers=selected_layers(model);device=next(model.parameters()).device
    for index in layers:
        b=model.backbone.layers[index-1];a=b.attn
        if b.version!=3 or a.causal or a.dropout!=0 or a.input_dim!=a.hidden_dim or a.output_dim!=a.hidden_dim:
            raise ValueError('Local attention/block is incompatible with registered operator')
    model._surface_attention_arm=arm;model._surface_attention_layers=layers
    model._relation_sigma_z=float(sigma_z);model._relation_static_cache=[] if static_cache else None
    model._relation_profile_enabled=bool(profile);model._relation_collect_stats=bool(collect_stats)
    model._relation_profile_events=[];model._relation_build_calls=0;model.relation_observations=[]
    model.relation_modules=nn.ModuleDict()
    if arm!='H_CONT':
        for index in layers:
            a=model.backbone.layers[index-1].attn
            model.relation_modules[f'layer_{index:02d}']=RelationMessageMLP(a.num_heads,a.hidden_dim,seed=int(seed)+index).to(device)
        model._surface_original_forward=model.forward
        model.forward=MethodType(_forward,model)
    model.set_trainable(model.experiment_trainable).eval()
    actual={n:id(p) for n,p in model.named_parameters()}
    if any(actual.get(n)!=pid for n,pid in old.items()):raise RuntimeError('Old Parameter object was replaced')
    new={n:p for n,p in model.named_parameters() if n not in old}
    if any(not n.startswith('relation_modules.') for n in new):raise RuntimeError('Unexpected new parameter namespace')
    if set(model.state_dict())-old_keys!=set(new):raise RuntimeError('Unexpected added state keys')
    metadata=dict(arm=arm,seed=int(seed),layer_seed_rule='seed + one-based layer index',layers=list(layers),
        module_paths=[f'backbone.layers.{i-1}.attn' for i in layers],new_parameter_names=sorted(new),
        new_parameter_shapes={n:list(p.shape) for n,p in new.items()},new_parameter_count=sum(p.numel() for p in new.values()),
        initialization_sha256=_state_hash({n:p for n,p in new.items()}),old_parameter_objects_preserved=True,
        H_CONT_path='Original MVPModel.forward unchanged',relation_dtype='FP32',relation_query_chunk=48,
        carrier='Original SDPA projected face broadcast plus exact algebraic dw@V and rearranged pair-value corner correction',
        last_layer_zero_initialized=True,external_zero_gate=False,sigma_z=float(sigma_z),
        output_parameterization='Original MVP velocity head and wrapper unchanged')
    model.surface_attention_metadata=metadata
    return metadata
