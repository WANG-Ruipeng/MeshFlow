"""Training-only analytic-geometry representation alignment.

Targets are fixed descriptors of clean free GT faces. They are never generator
inputs. P/S are separate auxiliary heads, never generator children or samplers.
The input Geo encoder and its feature definition are not changed here.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_LAYER = 9
FEATURE_DIM = 13
HIDDEN_DIM = 768
HEAD_SEED = 20261001
EPS = 1e-6
GROUP_SLICES = {'centroid':slice(0,3), 'log_edges':slice(3,6),
                'log_area':slice(6,7), 'normal_outer':slice(7,13)}
FORMULA_SPEC = dict(version='analytic_geometry_representation_v1',
    raw='centroid_xyz_sorted_edges_area_normal_outer_xx_yy_zz_xy_xz_yz',
    canonical_order='stable_lexicographic_xyz', descriptor_dtype='CPU_float64',
    length_log_floor=1e-8, area_log_floor=1e-12, true_degenerate_reject='zero_edge_or_area_le_zero',
    groups=[['centroid',0,3],['log_edges',3,6],['log_area',6,7],['normal_outer',7,13]],
    group_scale='centered_RMS_over_faces_and_group_dimensions', scale_floor=1e-6,
    normalization='(r_g-mu_g)/(max(s_g,1e-6)*sqrt(d_g))', training_target_dtype='float32',
    fit_K=[4,8,12], deduplication='parent_index_and_canonical_source_face_id',
    angular_epsilon=1e-6, invalid_direction='norm_q_le_1e-6_excluded_from_angular_only')


def _hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _hash_array(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def _triangles(value):
    a=np.asarray(value)
    if a.ndim==2 and a.shape[1]==9:a=a.reshape(-1,3,3)
    if a.ndim!=3 or a.shape[1:]!=(3,3) or len(a)==0 or a.dtype not in (np.float32,np.float64):
        raise ValueError('Expected nonempty FP32/FP64 triangles[F,3,3] or [F,9]')
    return a


def _canonical(value):
    a=_triangles(value)
    order=np.lexsort((a[:,:,2],a[:,:,1],a[:,:,0]),axis=1)
    return np.take_along_axis(a,order[:,:,None],axis=1)


def descriptor13(triangles):
    """Independent O(F) single-face formula; no adjacency, model, or autograd."""
    a=_canonical(triangles).astype(np.float64)
    if not np.isfinite(a).all():
        bad=np.flatnonzero(~np.isfinite(a).all(axis=(1,2))).tolist()
        raise ValueError('Nonfinite target triangle indices: '+str(bad))
    vectors=np.stack((a[:,1]-a[:,0],a[:,2]-a[:,0],a[:,2]-a[:,1]),axis=1)
    lengths=np.linalg.norm(vectors,axis=-1)
    cross=np.cross(vectors[:,0],vectors[:,1])
    twice_area=np.linalg.norm(cross,axis=-1);area=twice_area*.5
    invalid=(lengths<=0).any(axis=1)|(area<=0)|~np.isfinite(lengths).all(axis=1)|~np.isfinite(area)
    if invalid.any():
        raise ValueError('Degenerate/nonfinite target triangle indices: '+str(np.flatnonzero(invalid).tolist()))
    normal=cross/twice_area[:,None];nx,ny,nz=normal.T
    outer=np.stack((nx*nx,ny*ny,nz*nz,nx*ny,nx*nz,ny*nz),axis=1)
    raw=np.concatenate((a.mean(axis=1),np.sort(lengths,axis=1),area[:,None],outer),axis=1)
    transformed=raw.copy()
    transformed[:,3:6]=np.log(np.maximum(raw[:,3:6],1e-8))
    transformed[:,6]=np.log(np.maximum(raw[:,6],1e-12))
    if not np.isfinite(transformed).all():raise ValueError('Nonfinite transformed target descriptor')
    return dict(raw=raw,transformed=transformed,length_floor_elements=int((raw[:,3:6]<1e-8).sum()),
                area_floor_faces=int((raw[:,6]<1e-12).sum()),faces=len(a))


def _normalize(transformed,groups):
    result=np.empty_like(transformed,dtype=np.float64)
    for name,sl in GROUP_SLICES.items():
        info=groups[name]
        result[:,sl]=(transformed[:,sl]-np.asarray(info['mean'],np.float64))/info['denominator']
    return result


class GeometryTargets:
    """Frozen deduplicated training-face statistics and exact identity lookup."""
    @classmethod
    def from_archive(cls,archive):
        cases=archive.cases
        if {int(p) for p,k in cases if k in (4,8,12)}!={0,1}:
            raise ValueError('The registered fit uses exactly training parents0/1')
        unique={};case_counts=[]
        for parent in (0,1):
            for K in (4,8,12):
                case=cases[parent,K]
                ids=np.asarray(case.free_source_ids)
                free=_triangles(case.free_target)
                full=_triangles(case.full_target)
                if (ids.ndim!=1 or len(ids)!=len(free) or ids.dtype.kind not in 'iu'
                        or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(full))):
                    raise ValueError('Invalid free source-face identities for '+str((parent,K)))
                if free.dtype!=np.float32 or full.dtype!=np.float32 or free.tobytes()!=full[ids].tobytes():
                    raise ValueError('Free GT does not exactly follow archived source faces: '+str((parent,K)))
                for face,triangle in zip(ids,free):
                    key=(parent,int(face));canonical=_canonical(triangle[None])[0].copy()
                    if key in unique and unique[key].tobytes()!=canonical.tobytes():
                        raise ValueError('Conflicting GT for deduplicated identity '+str(key))
                    unique[key]=canonical
                case_counts.append(dict(parent=parent,K=K,free_faces=len(ids)))
        identities=sorted(unique)
        triangles=np.stack([unique[key] for key in identities])
        try:descriptors=descriptor13(triangles)
        except ValueError as error:
            raise ValueError(str(error)+'; ordered parent/source identities='+str(identities)) from error
        r=descriptors['transformed'];groups={}
        for name,sl in GROUP_SLICES.items():
            part=r[:,sl];mean=part.mean(axis=0);scale=float(np.sqrt(np.mean((part-mean)**2)))
            width=sl.stop-sl.start
            groups[name]=dict(mean=mean.tolist(),centered_RMS=scale,dimension=width,
                denominator=max(scale,1e-6)*math.sqrt(width),scale_floor_active=scale<1e-6)
        q64=_normalize(r,groups);q=q64.astype(np.float32)
        meta=dict(formula=dict(FORMULA_SPEC),fit_cases=case_counts,unique_faces=len(identities),
            groups=groups,source_archive_sha256=getattr(archive,'sha256',None),
            transformed_descriptor_sha256=_hash_array(r),normalized_target_sha256=_hash_array(q),
            q_mean_FP64=q.astype(np.float64).mean(axis=0).tolist(),
            q_variance_FP64=q.astype(np.float64).var(axis=0).tolist(),
            length_floor_elements=descriptors['length_floor_elements'],area_floor_faces=descriptors['area_floor_faces'],
            source_samples=[dict(parent=key[0],source_face_id=key[1],
                canonical_triangle_sha256=_hash_array(unique[key]),q=q[i].tolist())
                for i,key in enumerate(identities)])
        meta['statistics_sha256']=_hash_json(meta)
        return cls.from_metadata(meta)

    @classmethod
    def from_metadata(cls,metadata):
        meta=copy.deepcopy(metadata)
        digest=meta.pop('statistics_sha256',None)
        if digest!=_hash_json(meta) or meta.get('formula')!=FORMULA_SPEC:
            raise ValueError('Frozen target statistics hash or formula differs')
        if type(meta.get('unique_faces')) is not int or meta['unique_faces']<=0:
            raise ValueError('Empty target fit')
        for name,sl in GROUP_SLICES.items():
            item=meta['groups'][name];mean=np.asarray(item['mean'],np.float64)
            if mean.shape!=(sl.stop-sl.start,) or not np.isfinite(mean).all():
                raise ValueError('Invalid group mean')
            scale=item['centered_RMS'];den=item['denominator']
            if not np.isfinite(scale) or scale<0 or not np.isfinite(den) or den!=max(scale,1e-6)*math.sqrt(sl.stop-sl.start):
                raise ValueError('Invalid frozen group scale')
        obj=cls.__new__(cls);obj._lookup={}
        for row in meta['source_samples']:
            key=(row['parent'],row['source_face_id'])
            q=np.asarray(row['q'],dtype=np.float32)
            if key in obj._lookup or q.shape!=(13,) or not np.isfinite(q).all():
                raise ValueError('Invalid frozen target identity/value')
            q.setflags(write=False)
            obj._lookup[key]=(q,row['canonical_triangle_sha256'])
        if len(obj._lookup)!=meta['unique_faces']:raise ValueError('Target fit count differs')
        if _hash_array(np.stack([obj._lookup[key][0] for key in sorted(obj._lookup)]))!=meta['normalized_target_sha256']:
            raise ValueError('Frozen target tensor hash differs')
        meta['statistics_sha256']=digest;obj._metadata=meta
        return obj

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def for_sample(self,sample,device='cpu'):
        """Match actual free slots after face/angle permutation; never inspect known Y."""
        known=np.asarray(sample['known_mask']);valid=np.asarray(sample['valid_mask'])
        ids=np.asarray(sample['source_face_ids']);y=np.asarray(sample['x1'])
        if known.dtype!=bool or valid.dtype!=bool or known.ndim!=1 or valid.shape!=known.shape:
            raise ValueError('Expected original boolean masks')
        if np.any(known&~valid) or ids.shape!=known.shape or ids.dtype.kind not in 'iu':
            raise ValueError('Invalid source face map or known mask')
        if y.dtype!=np.float32 or y.shape!=(len(known),9):raise ValueError('Clean GT must be FP32[N,9]')
        free=valid&~known
        if not free.any():raise ValueError('No valid free targets')
        parent=int(sample['parent_index']);free_ids=ids[free];triangles=_canonical(y[free])
        values=[]
        for face,triangle in zip(free_ids,triangles):
            key=(parent,int(face))
            if key not in self._lookup:raise ValueError('Target outside the registered training fit: '+str(key))
            q,expected=self._lookup[key]
            if _hash_array(triangle)!=expected:
                raise ValueError('Actual free GT/source face mapping differs: '+str(key))
            values.append(q)
        q=np.stack(values).copy()
        # The direction predicate is made on the actual FP32 training target.
        target=torch.tensor(q,dtype=torch.float32,device=device)
        angular_valid=torch.linalg.vector_norm(target,dim=-1)>EPS
        count=int(angular_valid.sum())
        return dict(q=target,angular_valid=angular_valid,free_count=len(q),angular_count=count,
            invalid_angular_count=len(q)-count,source_face_ids=free_ids.astype(int).tolist(),
            parent=parent,labels_sha256=_hash_array(q),statistics_sha256=self._metadata['statistics_sha256'],
            target_requires_grad=False,known_targets_evaluated=0)


class AuxiliaryHeads(nn.Module):
    """Independent FP32 per-face P and optional S; never attached to generator."""
    def __init__(self,use_scale,seed=HEAD_SEED,device='cpu'):
        super().__init__()
        if type(use_scale) is not bool or type(seed) is not int or not 0<=seed<2**63:
            raise ValueError('Explicit scale switch and nonnegative integer seed required')
        self.use_scale,self.seed=use_scale,seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            kw=dict(device='cpu',dtype=torch.float32)
            self.P=nn.Sequential(nn.LayerNorm(768,eps=1e-5,**kw),nn.Linear(768,256,**kw),
                                 nn.SiLU(),nn.Linear(256,13,**kw))
            self.S=nn.Sequential(nn.Linear(13,64,**kw),nn.SiLU(),nn.Linear(64,13,**kw)) if use_scale else None
        self.to(device);self.eval()

    def metadata(self):
        return dict(schema='analytic_geometry_aux_heads_v1',use_scale=self.use_scale,seed=self.seed,
            feature_layer=FEATURE_LAYER,feature_dim=13,hidden_dim=768,dtype='torch.float32',eps=EPS,
            P_parameters=sum(p.numel() for p in self.P.parameters()),
            S_parameters=0 if self.S is None else sum(p.numel() for p in self.S.parameters()),
            total_parameters=sum(p.numel() for p in self.parameters()),normal_nonzero_initialization=True,
            generator_child=False)

    def forward(self,h9,valid_mask,known_mask):
        if h9.ndim!=3 or h9.shape[-1]!=768 or h9.shape[1]%3:
            raise ValueError('Expected actual H9[B,3N,768]')
        batch,_,width=h9.shape;faces=h9.shape[1]//3
        if (valid_mask.shape!=(batch,faces) or known_mask.shape!=valid_mask.shape
                or valid_mask.dtype!=torch.bool or known_mask.dtype!=torch.bool
                or valid_mask.device!=h9.device or known_mask.device!=h9.device):
            raise ValueError('Expected colocated boolean face masks')
        if bool((known_mask&~valid_mask).any()):raise ValueError('Known must be valid')
        if any(p.dtype!=torch.float32 or p.device!=h9.device for p in self.parameters()):
            raise ValueError('Auxiliary parameters must remain colocated FP32')
        free=valid_mask&~known_mask
        if not bool(free.any()):raise ValueError('No free feature faces')
        with torch.autocast(device_type=h9.device.type,enabled=False):
            face=h9.float().reshape(batch,faces,3,width).mean(dim=2)
            selected=face[free]
            z=self.P(selected);z_norm=F.normalize(z,dim=-1,eps=EPS)
            q_hat=self.S(z_norm) if self.S is not None else None
        return dict(z=z,z_norm=z_norm,q_hat=q_hat,H_face=selected)


def _denominator(value,count,name):
    if value is None:value=count
    if isinstance(value,bool) or not isinstance(value,Integral) or value<count or value<=0:
        raise ValueError(name+' must include all effective-batch eligible faces and be positive')
    return int(value)


def auxiliary_losses(predictions,q,*,angular_denominator=None,free_denominator=None):
    """Unweighted losses. Effective-batch denominators are the caller's contract."""
    z=predictions['z'];z_norm=predictions['z_norm'];q_hat=predictions['q_hat']
    if q.shape!=z.shape or q.ndim!=2 or q.shape[-1]!=13 or q.dtype!=torch.float32 or q.device!=z.device:
        raise ValueError('Expected colocated FP32 free q[F,13] matching P output')
    if not bool(torch.isfinite(q).all()):raise ValueError('Nonfinite fixed target')
    q=q.detach()
    with torch.autocast(device_type=z.device.type,enabled=False):
        target_norm=torch.linalg.vector_norm(q,dim=-1)
        valid=target_norm>EPS;angular_count=int(valid.sum());free_count=len(q)
        angular_den=_denominator(angular_denominator,angular_count,'Angular denominator')
        free_den=_denominator(free_denominator,free_count,'Free-face denominator')
        q_norm=F.normalize(q,dim=-1,eps=EPS)
        cosine=(z_norm*q_norm).sum(-1)
        angular_sum=(1-cosine[valid]).sum() if angular_count else z.sum()*0
        angular=angular_sum/angular_den
        error=None if q_hat is None else (q_hat-q).square()
        scale_sum=None if error is None else error.sum()
        scale=None if error is None else scale_sum/(13*free_den)
    return dict(angular=angular,scale=scale,angular_sum=angular_sum,scale_sum=scale_sum,
        angular_count=angular_count,invalid_angular_count=free_count-angular_count,free_count=free_count,
        angular_denominator=angular_den,free_denominator=free_den,angular_valid=valid,
        cosine=cosine,target_norm=target_norm,scale_squared_error=error,
        group_scale_MSE=None if error is None else {name:error[:,sl].mean() for name,sl in GROUP_SLICES.items()})
