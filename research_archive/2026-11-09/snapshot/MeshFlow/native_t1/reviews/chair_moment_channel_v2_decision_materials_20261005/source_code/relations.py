"""MOMENT_XY only, compact valid identities in their original slot order.

There is no coordinate sorting, quadrature, nearest point, or learned cache.
Only known coordinates and masks bind the optional pure-geometric moment cache.
Current free coordinates and time are recomputed on every call.
"""
from __future__ import annotations
import math
import torch

FEATURE_NAMES=('c_minus_x_x','c_minus_x_y','c_minus_x_z','Sigma_xx','Sigma_yy','Sigma_zz',
               'Sigma_xy','Sigma_xz','Sigma_yz','log1p_negative_logmass','t')


def _moments(C):
    centroid=C.mean(1)
    centered=C-centroid[:,None,:]
    covariance=torch.einsum('kai,kaj->kij',centered,centered)/12.
    return centroid,covariance


def build_relations(x,t,known_mask,valid_mask,mode='MOMENT_XY',sigma_z=1.,*,
                    static_cache=None,reference=False,query_chunk=48,known_chunk=8):
    if mode!='MOMENT_XY':raise ValueError('V2 accepts only MOMENT_XY; no surface quadrature')
    if not math.isfinite(float(sigma_z)) or sigma_z<=0:raise ValueError('Positive global sigma_Z required')
    if x.ndim!=3 or x.shape[-1]!=9 or known_mask.shape!=x.shape[:2] or valid_mask.shape!=x.shape[:2]:
        raise ValueError('Expected x[B,N,9] and aligned masks')
    if known_mask.dtype!=torch.bool or valid_mask.dtype!=torch.bool:raise ValueError('Boolean masks required')
    if known_mask.device!=x.device or valid_mask.device!=x.device:raise ValueError('Colocated masks required')
    times=torch.as_tensor(t,device=x.device).reshape(-1)
    if times.numel()==1:times=times.expand(len(x))
    if times.numel()!=len(x):raise ValueError('One time per batch item required')
    if static_cache is not None:
        if not isinstance(static_cache,list):raise ValueError('Cache must be an explicit per-forward-owner list')
        if len(static_cache)!=len(x):static_cache[:]=[None]*len(x)
    dtype=torch.float64 if reference else torch.float32;result=[]
    with torch.autocast(device_type=x.device.type,enabled=False):
        for i in range(len(x)):
            valid=torch.nonzero(valid_mask[i],as_tuple=False).flatten()
            known=torch.nonzero(valid_mask[i]&known_mask[i],as_tuple=False).flatten()
            free=torch.nonzero(valid_mask[i]&~known_mask[i],as_tuple=False).flatten()
            mapping=torch.full((x.shape[1],),-1,dtype=torch.long,device=x.device)
            mapping=mapping.index_copy(0,valid,torch.arange(len(valid),device=x.device))
            C=x[i].index_select(0,known).reshape(-1,3,3).to(dtype)/float(sigma_z)
            q=x[i].index_select(0,free).reshape(-1,3).to(dtype)/float(sigma_z)
            time=times[i].to(dtype)
            if not bool(torch.isfinite(C).all() and torch.isfinite(q).all() and torch.isfinite(time)):
                raise ValueError('Nonfinite effective relation input')
            if not 0<=float(time.detach())<=1:raise ValueError('Time outside [0,1]')
            cache=None if static_cache is None else static_cache[i]
            reuse=(cache is not None and not C.requires_grad and cache['C'].shape==C.shape
                and cache['C'].dtype==C.dtype and cache['C'].device==C.device
                and cache['known_mask'].shape==known_mask[i].shape
                and torch.equal(cache['known_mask'],known_mask[i]) and torch.equal(cache['valid_mask'],valid_mask[i])
                and torch.equal(cache['C'],C))
            if reuse:centroid,covariance=cache['centroid'],cache['covariance']
            else:centroid,covariance=_moments(C)
            if static_cache is not None:
                static_cache[i]=None if C.requires_grad else dict(C=C.detach().clone(),known_mask=known_mask[i].clone(),
                    valid_mask=valid_mask[i].clone(),centroid=centroid.detach(),covariance=covariance.detach())
            difference=centroid[None]-q[:,None]
            six=torch.stack((covariance[:,0,0],covariance[:,1,1],covariance[:,2,2],
                covariance[:,0,1],covariance[:,0,2],covariance[:,1,2]),-1)
            negative_z=difference.square().sum(-1)/(2*((1-time).square()+.1**2))
            R=torch.cat((difference,six[None].expand(len(q),-1,-1),
                torch.log1p(negative_z.clamp_min(0))[...,None],time.expand(len(q),len(C),1)),dim=-1)
            result.append(dict(valid_indices=valid,free_face_indices=free,known_indices=known,
                free_compact_indices=mapping[free],known_compact_indices=mapping[known],
                R=R.reshape(len(free),3,len(known),11)))
    return result


compute_relations=build_relations
