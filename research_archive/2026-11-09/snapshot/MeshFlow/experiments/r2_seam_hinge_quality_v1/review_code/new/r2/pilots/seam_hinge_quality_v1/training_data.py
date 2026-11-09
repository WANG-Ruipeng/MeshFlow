"""Frozen source-edge transport and C-only prior transform; no neural calls."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import trainer_support as P
from r2.pilots.seam_quality_wave1_wave2_v1 import training_data as Q

ROOT=Path(__file__).resolve().parents[3]
ARMS=('CLEAN_EDGE_COH','SMOOTH_PRIOR')
BASE_STEP,UPDATES,BATCH=14000,1000,8
EDGE_LAMBDA,RHO=.1,.25
resolve,read,digest,source_arrays=P.resolve,P.read,Q.digest,Q.source_arrays

def write_new(path,value):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,ensure_ascii=False,indent=2,allow_nan=False)

def transform_bulk_noise(noise,bulk_face_mask,rho=RHO):
    """Identical FP32 transform for training Z/epsilon and inference Gaussian.

    Supports (...,3,3) triangle arrays, with one bool per leading face. It is
    invertible for rho>0; Gaussian residuals remain unbounded, not bounded.
    """
    z=np.asarray(noise)
    mask=np.asarray(bulk_face_mask,bool)
    if z.shape[-2:]!=(3,3) or mask.shape!=z.shape[:-2] or not 0<rho<=1:
        raise ValueError('Triangle noise and exact bulk mask required')
    if z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('Finite FP32 noise required')
    out=z.copy();selected=z[mask]
    mean=selected.mean(axis=-2,keepdims=True,dtype=np.float32)
    out[mask]=mean+np.float32(rho)*(selected-mean)
    return out

def transport_edges(sample,label):
    # canonical target corner -> source vertex; invert cached source face/corner maps.
    ids=np.asarray(sample['source_face_ids'],np.int64)
    inverse=np.argsort(ids)
    corner_inverse=np.argsort(np.asarray(sample['source_corner_permutations'],np.int64),axis=1)
    rows=np.asarray(label['clean_edges'],np.int64).reshape(-1,6)
    if not len(rows):return np.empty((0,4),np.int64)
    fa,fb= inverse[rows[:,0]],inverse[rows[:,1]]
    ix=np.stack((fa*3+corner_inverse[fa,rows[:,2]],fa*3+corner_inverse[fa,rows[:,3]],
                 fb*3+corner_inverse[fb,rows[:,4]],fb*3+corner_inverse[fb,rows[:,5]]),axis=1)
    target=sample['x1'].reshape(-1,3)
    if not np.array_equal(target[ix[:,0]],target[ix[:,2]]) or not np.array_equal(target[ix[:,1]],target[ix[:,3]]):
        raise ValueError('Shared endpoints lost under saved source/OT/corner mapping')
    return ix

def recover_existing_noise(sample,record):
    n,k=sample['N'],sample['K'];seeds=record['frozen_random_stream']
    z=np.random.RandomState(seeds['noise_seed']).randn(n-k,3,3).astype(np.float32)
    eps=np.random.RandomState(seeds['epsilon_seed']).randn(n-k,3,3).astype(np.float32)
    fullz=np.zeros((n,3,3),np.float32);fulleps=fullz.copy()
    fullz[k:]=z;fulleps[k:]=eps
    order=sample['face_permutation'];fullz=fullz[order];fulleps=fulleps[order]
    mixed=np.float32(np.sqrt(.75))*fullz+np.float32(.5)*fulleps
    fixed=sample['anchor_ids']>=0
    mixed[fixed]=sample['C_vertices'][sample['anchor_ids'][fixed]]
    if not np.array_equal(mixed,sample['x0']):raise ValueError('Frozen Gaussian replay does not reproduce existing x0 bitwise')
    return fullz,fulleps

def load_sample(record,root,arm,step,micro,labels):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    s=P.load_sample(record,root,'SEAM_SLOT_C200',step,micro,digest)
    label=labels[s['uid']]
    Q.label_sample(s,label)
    s['continuation_arm']=arm
    s['edge_indices']=transport_edges(s,label)
    s['edge_count']=len(s['edge_indices'])
    s['edge_categories']=label['edge_categories']
    if arm=='SMOOTH_PRIOR':
        z,eps=recover_existing_noise(s,record)
        bulk=~s['known_mask'] & s['free_corner_mask'].all(1)
        z=transform_bulk_noise(z,bulk);eps=transform_bulk_noise(eps,bulk)
        mixed=np.float32(np.sqrt(.75))*z+np.float32(.5)*eps
        fixed=s['anchor_ids']>=0;mixed[fixed]=s['C_vertices'][s['anchor_ids'][fixed]]
        if not np.array_equal(mixed[~bulk],s['x0'][~bulk]):raise ValueError('Smooth transform changed C/seam state')
        s['x0']=mixed;s['u']=s['x1']-mixed
        s['xt']=(np.float32(1)-s['t'])*mixed+s['t']*s['x1'];s['xt'][fixed]=mixed[fixed]
        if s['u'][fixed].any():raise ValueError('Smooth transform changed anchor velocity')
    return s

def batch_denominators(samples):
    if len(samples)!=8:raise ValueError('Whole effective batch8 required')
    return sum(int(s['free_coordinate_count']) for s in samples),sum(float(s['weighted_free_coordinate_count']) for s in samples),sum(s['edge_count'] for s in samples)

diagnostic=Q.diagnostic

def edge_coherence(prediction,sample,denominator):
    """Sum over model-unit squared endpoint+vector terms / whole-batch edges.

    Fixed corners are overwritten before lookup, so their velocity gradients
    are exactly zero. This is a quadratic relational loss, not a topological
    guarantee; the vector term is a combination of endpoint errors.
    """
    import torch
    velocity=prediction.float().reshape(-1,3)
    if denominator<sample['edge_count']:raise ValueError('Invalid full-batch edge denominator')
    if not denominator or not sample['edge_count']:
        return velocity.sum()*0.,dict(edge_count=sample['edge_count'],endpoint_sse=0.,vector_sse=0.)
    cv=lambda a,dt:torch.as_tensor(a,dtype=dt,device=velocity.device)
    xt=cv(sample['xt'].reshape(-1,3),torch.float32)
    endpoint=xt+(1-float(sample['t']))*velocity
    fixed=cv((sample['anchor_ids']>=0).reshape(-1,1),torch.bool)
    endpoint=torch.where(fixed,xt,endpoint)
    ix=cv(sample['edge_indices'],torch.long)
    d0=endpoint[ix[:,0]]-endpoint[ix[:,2]]
    d1=endpoint[ix[:,1]]-endpoint[ix[:,3]]
    endpoint_sse=.5*(d0.square().sum()+d1.square().sum())
    vector_sse=.5*(d1-d0).square().sum()
    loss=(endpoint_sse+vector_sse)/denominator
    return loss,dict(edge_count=sample['edge_count'],endpoint_sse=float(endpoint_sse.detach()),vector_sse=float(vector_sse.detach()))
