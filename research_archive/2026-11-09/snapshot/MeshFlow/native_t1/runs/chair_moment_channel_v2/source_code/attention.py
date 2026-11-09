"""V1 exact-difference attention, with compact valid-source relation arithmetic.

The old projected base SDPA is untouched. This is the existing 32-dimensional
pair-value rearrangement, not a claimed new source-partition optimization.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from experiments.mf_h1_local0.chair_surface_attention_v1.attention import RelationMessageMLP,original_face_attention


def relation_delta(q,k,v,valid,free_faces,known_faces,R,module,*,query_chunk=48,literal=False,
                   collect_stats=False,debug=None):
    if q.ndim!=3 or q.shape!=k.shape or k.shape!=v.shape:raise ValueError('Aligned [N,H,d] QKV required')
    N,H,d=q.shape;Fcount=len(free_faces);K=len(known_faces)
    if valid.shape!=(N,) or valid.dtype!=torch.bool:raise ValueError('Boolean source mask required')
    if R.shape!=(Fcount,3,K,11) or H!=module.heads or H*d!=module.value_dim:
        raise ValueError('Relation identities or module dimensions differ')
    if query_chunk<=0:raise ValueError('Positive corner-query chunk required')
    ids=torch.nonzero(valid,as_tuple=False).flatten()
    mapping=torch.full((N,),-1,dtype=torch.long,device=q.device)
    mapping=mapping.index_copy(0,ids,torch.arange(len(ids),device=q.device))
    known=mapping[known_faces]
    if q.device.type=='cpu' and (bool((known<0).any()) or bool((mapping[free_faces]<0).any())
        or set(free_faces.tolist())&set(known_faces.tolist())):raise ValueError('Invalid relation endpoint mapping')
    dtype=torch.float64 if R.dtype==torch.float64 else torch.float32
    if debug is not None:
        debug.update(valid_indices=ids.detach().cpu(),free_face_indices=free_faces.detach().cpu(),
            known_indices=known_faces.detach().cpu(),compact_q=q.index_select(0,ids).detach().cpu(),
            compact_k=k.index_select(0,ids).detach().cpu(),compact_v=v.index_select(0,ids).detach().cpu(),
            compact_valid_mask=torch.ones(len(ids),dtype=torch.bool))
    if K==0 or Fcount==0:return torch.zeros((Fcount,3,H*d),dtype=dtype,device=q.device),{}
    if not len(ids):raise ValueError('Nonempty correction needs valid source keys')
    if literal and 3*Fcount*K*H*d>200000:raise ValueError('Literal pair value is small-test only')
    queries=free_faces.repeat_interleave(3);flat=R.reshape(3*Fcount,K,11)
    outputs=[];stats=[];telemetry_phi=[];stages=[]
    with torch.autocast(device_type=q.device.type,enabled=False):
        q=q.to(dtype);k=k.index_select(0,ids).to(dtype);v=v.index_select(0,ids).to(dtype)
        W=module.fc2.weight.to(dtype);bias=module.fc2.bias.to(dtype)
        for start in range(0,3*Fcount,int(query_chunk)):
            stop=min(start+int(query_chunk),3*Fcount)
            current=q.index_select(0,queries[start:stop]).transpose(0,1)
            logits=torch.matmul(current,k.transpose(0,1).transpose(1,2))*(d**-.5)
            phi=module.features(flat[start:stop]);b=F.linear(phi,W[:H],bias[:H])
            # All compact keys are valid: no padded length, masked zeros, or
            # all-negative-infinity rows occur in this new-channel softmax.
            dense_bias=torch.zeros_like(logits).index_copy(2,known,b.permute(2,0,1))
            base_weights=torch.softmax(logits,-1);weights=torch.softmax(logits+dense_bias,-1)
            content=torch.matmul(weights-base_weights,v.transpose(0,1)).transpose(0,1)
            known_weights=weights.index_select(2,known)
            if literal:
                pair=F.linear(phi,W[H:],bias[H:]).reshape(stop-start,K,H,d)
                extra=torch.einsum('hqk,qkhd->qhd',known_weights,pair)
            else:
                weighted_phi=torch.einsum('hqk,qkr->hqr',known_weights,phi)
                extra=torch.einsum('hqr,hdr->qhd',weighted_phi,W[H:].reshape(H,d,32))
                extra=extra+known_weights.sum(-1).transpose(0,1)[...,None]*bias[H:].reshape(H,d)[None]
            outputs.append((content+extra).reshape(stop-start,H*d))
            if debug is not None:
                stages.append({name:value.detach().cpu() for name,value in dict(phi=phi,b=b,logits=logits,
                    base_weights=base_weights,weights=weights,content_delta=content,pair_extra=extra,
                    aggregate=content+extra).items()})
            if collect_stats:
                stats.append(dict(b_sumsq=b.detach().square().sum(),b_count=b.numel(),b_max=b.detach().abs().max(),
                    weighted_pair_value_sumsq=extra.detach().square().sum(),weighted_pair_value_count=extra.numel(),
                    weighted_pair_value_max=extra.detach().abs().max(),content_delta_sumsq=content.detach().square().sum(),
                    content_delta_count=content.numel(),content_delta_max=content.detach().abs().max()))
                telemetry_phi.append(phi.detach())
        audit={}
        if stats:
            for prefix in ('b','weighted_pair_value','content_delta'):
                audit[prefix+'_RMS']=(sum(s[prefix+'_sumsq'] for s in stats)/sum(s[prefix+'_count'] for s in stats)).sqrt()
                audit[prefix+'_max']=torch.stack([s[prefix+'_max'] for s in stats]).max()
            audit['logit_RMS']=audit['b_RMS'];audit['logit_max']=audit['b_max']
            # Exact effective pair-u statistics, detached and bounded to ONE
            # head's d channels. Never allocate the full [3F,K,D] pair values.
            with torch.no_grad():
                all_phi=torch.cat(telemetry_phi);squares=[];maxima=[];count=0
                for h in range(H):
                    lo=H+h*d;hi=lo+d
                    u=F.linear(all_phi,W[lo:hi].detach(),bias[lo:hi].detach())
                    squares.append(u.square().sum());maxima.append(u.abs().max());count+=u.numel()
                audit['u_RMS']=(sum(squares)/count).sqrt();audit['u_max']=torch.stack(maxima).max()
            audit.update(effective_relation_pairs=3*Fcount*K,effective_free_corners=3*Fcount,
                effective_known_sources=K,effective_all_sources=len(ids))
        if debug is not None:
            for name in stages[0]:debug[name]=torch.cat([s[name] for s in stages],dim=1 if name in ('logits','base_weights','weights') else 0)
    return torch.cat(outputs).reshape(Fcount,3,H*d),audit


def attention_with_relations(attn,face_input,valid_mask,relations,module,*,query_chunk=48,
                             collect_stats=False,profile=None,debug=None):
    if attn.causal or float(attn.dropout)!=0.:raise ValueError('Original registered noncausal zero-dropout attention required')
    base,q,k,v=original_face_attention(attn,face_input,valid_mask)
    token=profile[0]() if profile is not None else None
    B,N,_=face_input.shape;all_delta=[];observations=[]
    if len(relations)!=B:raise ValueError('One relation bank per sample required')
    for batch,r in enumerate(relations):
        stages={} if debug is not None else None
        delta,stats=relation_delta(q[batch],k[batch],v[batch],valid_mask[batch],r['free_face_indices'],
            r['known_indices'],r['R'],module,query_chunk=query_chunk,collect_stats=collect_stats,debug=stages)
        projected=F.linear(delta.reshape(-1,module.value_dim),attn.out_proj.weight,None)
        corner_indices=(r['free_face_indices'][:,None]*3+torch.arange(3,device=q.device)[None]).reshape(-1)
        dense=projected.new_zeros((3*N,attn.output_dim)).index_copy(0,corner_indices,projected)
        all_delta.append(dense);observations.append(stats)
        if debug is not None:
            stages['projected_delta']=projected.detach().cpu();stages['corner_scatter_indices']=corner_indices.detach().cpu()
            debug.append(stages)
    result=torch.stack(all_delta)
    if profile is not None:profile[1](token)
    return base,result,observations
