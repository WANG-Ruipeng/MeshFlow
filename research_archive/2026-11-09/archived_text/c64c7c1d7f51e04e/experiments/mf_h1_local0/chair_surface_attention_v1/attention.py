"""Exact relation-attention correction around the untouched native SDPA message.

For each free corner, softmax still ranges over ALL valid source faces. The
content delta is (softmax(ell0+b)-softmax(ell0)) @ V. This algebraic difference
preserves the original SDPA baseline at zero initialization. Nonzero floating
arithmetic is explicitly checked; mathematical equivalence does not promise
bitwise equivalence between explicit FP32 softmax and a different backend.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.attention import attention as original_attention


class RelationMessageMLP(nn.Module):
    def __init__(self, heads, value_dim, *, seed):
        super().__init__()
        if value_dim % heads:raise ValueError('Value dimension must divide into heads')
        self.heads=int(heads);self.value_dim=int(value_dim)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.fc1=nn.Linear(11,32,device='cpu',dtype=torch.float32)
            self.fc2=nn.Linear(32,heads+value_dim,device='cpu',dtype=torch.float32)
            nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
        self.eval()

    def features(self,R):
        dtype=torch.float64 if R.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=R.device.type,enabled=False):
            return F.silu(F.linear(R.to(dtype),self.fc1.weight.to(dtype),self.fc1.bias.to(dtype)))


def original_face_attention(attn,x,valid_mask):
    """Literal protected SelfAttention operations, retaining original dimensions."""
    B,N,_=x.shape
    qkv=attn.qkv_proj(x).reshape(B,N,3,attn.num_heads,attn.head_dim)
    qkv=qkv.permute(2,0,1,3,4)
    q,k,v=qkv.chunk(3,dim=0)
    q,k=attn.q_norm(q[0]),attn.k_norm(k[0]);v=v[0]
    backend='pytorch-sdpa' if attn.mixed_precision=='fp32' else 'flash-attn'
    if q.dtype not in (torch.float16,torch.bfloat16):backend='pytorch-sdpa'
    message=original_attention(q,k,v,mask_q=valid_mask,mask_kv=valid_mask,
        dropout=attn.dropout,causal=attn.causal,backend=backend)
    return attn.out_proj(message.reshape(B,N,-1)),q,k,v


def relation_delta(q,k,v,valid,free_faces,known_faces,R,module,*,query_chunk=48,literal=False,collect_stats=False):
    """One batch item; no [3F,K,D] value tensor in the production branch.

    q/k/v are [N,H,d], R is [F,3,K,11]. The literal option is restricted to
    isolated small-module tests and is never passed by model_adapter.
    """
    if q.ndim!=3 or q.shape!=k.shape or k.shape!=v.shape:raise ValueError('Expected aligned [N,H,d] QKV')
    N,H,d=q.shape;Fcount=len(free_faces);K=len(known_faces)
    if valid.shape!=(N,) or valid.dtype!=torch.bool or (q.device.type=='cpu' and not bool(valid.any())):
        raise ValueError('At least one valid source key is required')
    if R.shape!=(Fcount,3,K,11):raise ValueError('Relation axes do not match free-corner/known identities')
    if H!=module.heads or H*d!=module.value_dim:raise ValueError('Relation module/value dimensions differ')
    if q.device.type=='cpu' and (not bool(valid[free_faces].all()) or not bool(valid[known_faces].all())):raise ValueError('Padding cannot be a relation endpoint')
    if q.device.type=='cpu' and set(free_faces.tolist()) & set(known_faces.tolist()):
        raise ValueError('Known and free relation sets overlap')
    dtype=torch.float64 if R.dtype==torch.float64 else torch.float32
    # No-known is the exact zero correction, not an all-negative-infinity softmax.
    if K==0 or Fcount==0:return torch.zeros((Fcount,3,H*d),dtype=dtype,device=q.device),{}
    if literal and 3*Fcount*K*H*d>200000:raise ValueError('Literal pair-value expansion is small-test only')
    all_queries=free_faces.repeat_interleave(3);flat=R.reshape(3*Fcount,K,11)
    outputs=[];stats=[]
    with torch.autocast(device_type=q.device.type,enabled=False):
        q,k,v=q.to(dtype),k.to(dtype),v.to(dtype)
        W=module.fc2.weight.to(dtype);bias=module.fc2.bias.to(dtype)
        for start in range(0,3*Fcount,int(query_chunk)):
            stop=min(start+int(query_chunk),3*Fcount)
            queries=q.index_select(0,all_queries[start:stop]).transpose(0,1)
            # Q/K have already passed the original q_norm/k_norm. Math SDPA's
            # defining content logit is q dot k / sqrt(head_dim), with key mask.
            logits=torch.matmul(queries,k.transpose(0,1).transpose(1,2))*(d**-.5)
            logits=logits.masked_fill(~valid[None,None,:],float('-inf'))
            phi=module.features(flat[start:stop])
            b=F.linear(phi,W[:H],bias[:H])
            dense_bias=torch.zeros_like(logits).index_copy(2,known_faces,b.permute(2,0,1))
            base_weights=torch.softmax(logits,dim=-1)
            weights=torch.softmax(logits+dense_bias,dim=-1)
            content=torch.matmul(weights-base_weights,v.transpose(0,1)).transpose(0,1)
            known_weights=weights.index_select(2,known_faces)
            if literal:
                pair=F.linear(phi,W[H:],bias[H:]).reshape(stop-start,K,H,d)
                extra=torch.einsum('hqk,qkhd->qhd',known_weights,pair)
            else:
                weighted_phi=torch.einsum('hqk,qkr->hqr',known_weights,phi)
                U=W[H:].reshape(H,d,32)
                extra=torch.einsum('hqr,hdr->qhd',weighted_phi,U)
                extra=extra+known_weights.sum(-1).transpose(0,1)[...,None]*bias[H:].reshape(H,d)[None]
            outputs.append((content+extra).reshape(stop-start,H*d))
            if collect_stats:
                stats.append(dict(logit_sumsq=b.detach().square().sum(),logit_count=b.numel(),
                    weighted_pair_value_sumsq=extra.detach().square().sum(),weighted_pair_value_count=extra.numel(),
                    content_delta_sumsq=content.detach().square().sum(),content_delta_count=content.numel()))
    result=torch.cat(outputs).reshape(Fcount,3,H*d)
    audit={}
    if stats:
        for prefix in ('logit','weighted_pair_value','content_delta'):
            audit[prefix+'_RMS']=(sum(s[prefix+'_sumsq'] for s in stats)/sum(s[prefix+'_count'] for s in stats)).sqrt()
    return result,audit


def attention_with_relations(attn,face_input,valid_mask,relations,module,*,query_chunk=48,collect_stats=False,profile=None):
    """Original projected face message and additive projected corner correction."""
    if attn.causal or float(attn.dropout)!=0.:raise ValueError('Registered native attention is noncausal with zero dropout')
    base,q,k,v=original_face_attention(attn,face_input,valid_mask)
    profile_token=profile[0]() if profile is not None else None
    B,N,_=face_input.shape
    if len(relations)!=B:raise ValueError('One relation bank per batch item required')
    all_delta=[];observations=[]
    for batch,r in enumerate(relations):
        delta,stats=relation_delta(q[batch],k[batch],v[batch],valid_mask[batch],
            r['free_face_indices'],r['known_indices'],r['R'],module,query_chunk=query_chunk,collect_stats=collect_stats)
        # Use the original projection weights without repeating its bias. This
        # projection obeys the surrounding native BF16/FP32 autocast context.
        projected=F.linear(delta.reshape(-1,module.value_dim),attn.out_proj.weight,None)
        corner_indices=(r['free_face_indices'][:,None]*3+torch.arange(3,device=q.device)[None]).reshape(-1)
        dense=projected.new_zeros((3*N,attn.output_dim)).index_copy(0,corner_indices,projected)
        all_delta.append(dense);observations.append(stats)
    stacked=torch.stack(all_delta)
    if profile is not None:profile[1](profile_token)
    return base,stacked,observations
