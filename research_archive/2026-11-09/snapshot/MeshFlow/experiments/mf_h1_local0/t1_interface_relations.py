"""Trusted source-corner relations for the bounded T1 interface experiment.

Source IDs construct supervision and C-only topology features. IDs, free GT,
relation groups, and predicted nearest points are never model conditions.
Nothing in this module runs a model, optimizer, or inference repair.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from native_t1.data import array_hash, file_sha256

SUPPORTED_K=(2,4,8,12)


def _same(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def context_phi(source_faces):
    """Only supplied C topology: opposite-edge open flag and other-edge sum."""
    faces=np.asarray(source_faces,dtype=np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3 or any(len(set(f))!=3 for f in faces.tolist()):
        raise ValueError('C must contain valid indexed triangles.')
    opposite=[[(int(face[(a+1)%3]),int(face[(a+2)%3])) for a in range(3)] for face in faces]
    incidence=Counter(tuple(sorted(edge)) for row in opposite for edge in row)
    if any(n>2 for n in incidence.values()):
        raise ValueError('This fixed experiment has no overincident C edges.')
    opened=np.asarray([[incidence[tuple(sorted(edge))]==1 for edge in row] for row in opposite],dtype=np.float32)
    phi=np.stack([opened,opened.sum(axis=1,keepdims=True)-opened],axis=-1).astype(np.float32)
    return phi,dict(boundary_edges=sum(n==1 for n in incidence.values()),internal_edges=sum(n==2 for n in incidence.values()),
        corner_feature_counts=[dict(phi=list(key),count=count) for key,count in sorted(Counter(map(tuple,phi.reshape(-1,2).tolist())).items())],
        source='C source-index subset edge incidence only',uses_free_GT=False)


class SourceRelations:
    def __init__(self,parents,audit):
        self.parents=parents; self.audit=audit

    def condition_phi(self,parent,K):
        """Sampling feature for C in unchanged archive face/corner order."""
        case=self.parents[int(parent)]['cases'][int(K)]
        return case['phi'].copy()

    def prepare_sample_labels(self,sample):
        """Recover immutable target identities through the *actual* slot maps."""
        parent=int(sample['parent_index']); K=int(sample['K'])
        if int(sample['donor_index'])!=parent:
            raise ValueError('These three correct-context arms require matched C and target source identities.')
        entry=self.parents[parent]; case=entry['cases'][K]
        face_ids=np.asarray(sample['source_face_ids'],dtype=np.int64)
        corners=np.asarray(sample['corner_permutations'],dtype=np.int64)
        known=np.asarray(sample['known_mask'],dtype=bool); valid=np.asarray(sample['valid_mask'],dtype=bool)
        if face_ids.shape!=(112,) or sorted(face_ids.tolist())!=list(range(112)) or corners.shape!=(112,3):
            raise ValueError('Actual archived face IDs and corner permutations required.')
        if not np.all(np.sort(corners,axis=1)==np.arange(3)) or known.shape!=(112,) or valid.shape!=(112,) or not valid.all() or known.sum()!=K:
            raise ValueError('Unexpected corner map or actual N/K/mask.')
        ids=np.take_along_axis(entry['face_vertex_ids'][face_ids],corners,axis=1)
        coordinates=entry['vertices'][ids]
        target=np.asarray(sample['x1']).reshape(112,3,3)
        context=np.asarray(sample['context']).reshape(112,3,3)
        if not _same(coordinates[~known],target[~known]):
            raise RuntimeError('Free endpoint/source-corner mapping is not FP32 exact. OT may reorder Z only.')
        if not _same(coordinates[known],context[known]):
            raise RuntimeError('C/source-corner mapping is not FP32 exact.')
        known_target_ids=np.asarray(sample['known_target_ids'],dtype=np.int64)
        if known_target_ids.shape!=(112,) or sorted(known_target_ids[known].tolist())!=list(range(K)):
            raise ValueError('Known C slot mapping changed.')
        if not np.array_equal(face_ids[known],case['source_face_ids'][known_target_ids[known]]):
            raise RuntimeError('Known source-face identity does not match selected C.')
        phi=np.zeros((112,3,2),dtype=np.float32)
        for slot in np.flatnonzero(known):
            cindex=int(known_target_ids[slot])
            phi[slot]=case['phi'][cindex,corners[slot]]
        # This independent C-only recalculation proves that free topology was
        # not needed to obtain the emitted phi under face/corner permutations.
        direct,_=context_phi(ids[known])
        if not _same(phi[known],direct) or np.count_nonzero(phi[~known]):
            raise RuntimeError('Known-only phi permutation mapping failed.')
        labels=build_relation_labels(ids,known,context,valid)
        labels.update(phi=phi,corner_source_vertex_ids=ids.copy(),parent=parent,K=K,
            source_mapping_audit=dict(status='PASS',target_corners_FP32_exact=True,C_corners_FP32_exact=True,
                actual_source_ids_sha256=array_hash(ids),known_mask_sha256=array_hash(known),
                phi_sha256=array_hash(phi),GT_nearest_point_matching=False,OT_changes_target_order=False))
        return labels


def build_relation_labels(corner_source_vertex_ids,known_mask,context,valid_mask=None):
    """CPU group construction from trusted identities, not geometric proximity."""
    ids=np.asarray(corner_source_vertex_ids,dtype=np.int64)
    known=np.asarray(known_mask,dtype=bool)
    valid=np.ones(len(ids),dtype=bool) if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    C=np.asarray(context).reshape(len(ids),3,3)
    if ids.shape!=(len(known),3) or valid.shape!=known.shape or C.dtype!=np.float32:
        raise ValueError('Require source IDs [N,3], masks [N], FP32 context corners.')
    free_by_id=defaultdict(list); known_by_id=defaultdict(list)
    for face in np.flatnonzero(valid):
        for corner in range(3):
            group=known_by_id if known[face] else free_by_id
            group[int(ids[face,corner])].append(int(3*face+corner))
    flat=C.reshape(-1,3); fc=[]; ff=[]
    for source_id,indices in sorted(free_by_id.items()):
        if source_id in known_by_id:
            known_indices=known_by_id[source_id]; anchor=flat[known_indices[0]].copy()
            if any(not _same(flat[index],anchor) for index in known_indices):
                raise RuntimeError('One trusted known source vertex has contradictory C coordinates.')
            fc.append(dict(source_vertex_id=source_id,free_corner_indices=np.asarray(indices,dtype=np.int64),
                anchor=anchor,known_corner_indices=np.asarray(known_indices,dtype=np.int64)))
        if len(indices)>=2:
            ff.append(dict(source_vertex_id=source_id,free_corner_indices=np.asarray(indices,dtype=np.int64)))
    return dict(fc_groups=fc,ff_groups=ff,FC_count=len(fc),FF_count=len(ff),
        free_corner_count=sum(map(len,free_by_id.values())),
        known_anchor_deduplication=True,group_weighting='Equal weight per source vertex group; FC within-group corner mean, FF unordered-pair mean',
        loss_units='Raw model-coordinate MSE over x/y/z, no bbox normalization')


def prepare_relations(labels,device='cuda'):
    """Pre-pad group IDs outside forward; two batched gathers suffice per loss."""
    device=torch.device(device)
    result={}
    for prefix in ('fc','ff'):
        groups=labels[prefix+'_groups']; count=len(groups)
        width=max((len(g['free_corner_indices']) for g in groups),default=0)
        indices=np.zeros((count,width),dtype=np.int64)
        valid=np.zeros((count,width),dtype=bool)
        sizes=np.asarray([len(g['free_corner_indices']) for g in groups],dtype=np.int64)
        for row,group in enumerate(groups):
            n=len(group['free_corner_indices'])
            indices[row,:n]=group['free_corner_indices']; valid[row,:n]=True
        result[prefix+'_count']=count
        result[prefix+'_indices']=torch.as_tensor(indices,dtype=torch.long,device=device)
        result[prefix+'_valid']=torch.as_tensor(valid,dtype=torch.bool,device=device)
        result[prefix+'_sizes']=torch.as_tensor(sizes,dtype=torch.long,device=device)
    anchors=np.asarray([g['anchor'] for g in labels['fc_groups']],dtype=np.float32).reshape(-1,3)
    result['fc_anchors']=torch.as_tensor(anchors,dtype=torch.float32,device=device).detach()
    result.update(FC_count=result['fc_count'],FF_count=result['ff_count'],device=str(device))
    return result


def relation_sums(predicted_endpoint,prepared):
    """FC/FF group sums/counts, then caller weights across its effective batch.

    Two padded index-selects and fixed-axis reductions avoid per-group kernels
    and explicit scatter. GPU backward support is checked by the caller's strict
    deterministic gate. Padding contributes exactly zero. The FF group mean
    stays in the graph; its variance identity equals the unordered-pair MSE.
    """
    if not predicted_endpoint.is_floating_point() or predicted_endpoint.dtype not in (torch.float32,torch.float64):
        raise ValueError('Endpoint loss requires FP32 (FP64 allowed for CPU mathematical fixtures).')
    with torch.autocast(device_type=predicted_endpoint.device.type,enabled=False):
        positions=predicted_endpoint.reshape(-1,3)
        zero=positions.sum()*0
        fc=zero; ff=zero
        if prepared['FC_count']:
            idx=prepared['fc_indices']; count,width=idx.shape
            values=positions.index_select(0,idx.reshape(-1)).reshape(count,width,3)
            valid=prepared['fc_valid'][...,None].to(dtype=positions.dtype)
            residual=values-prepared['fc_anchors'].detach().to(dtype=positions.dtype)[:,None,:]
            sums=(residual.square()*valid).sum(dim=(1,2))
            fc=(sums/(3*prepared['fc_sizes'].to(dtype=positions.dtype))).sum()
        if prepared['FF_count']:
            idx=prepared['ff_indices']; count,width=idx.shape
            values=positions.index_select(0,idx.reshape(-1)).reshape(count,width,3)
            valid=prepared['ff_valid'][...,None].to(dtype=positions.dtype)
            sizes=prepared['ff_sizes'].to(dtype=positions.dtype)
            mean=(values*valid).sum(dim=1,keepdim=True)/sizes[:,None,None]
            deviation=values-mean
            sums=(deviation.square()*valid).sum(dim=(1,2))
            ff=(sums*(2/(3*(sizes-1)))).sum()
    return dict(FC_sum=fc,FC_count=prepared['FC_count'],FF_sum=ff,FF_count=prepared['FF_count'])


def endpoint_from_velocity(xt,velocity,t,known_mask,context):
    """Construct only free Yhat=Xt+(1-t)v; known C is copied detached."""
    if xt.shape!=velocity.shape or xt.ndim!=3 or xt.shape[-1]!=9 or context.shape!=xt.shape:
        raise ValueError('Require matching [B,N,9] states, velocities, and context.')
    if any(a.dtype!=torch.float32 for a in (xt,velocity,t,context)) or known_mask.dtype!=torch.bool or known_mask.shape!=xt.shape[:2] or t.shape!=(len(xt),):
        raise ValueError('FP32 states/times and boolean known masks required.')
    free=~known_mask
    remaining=(1-t[:,None].expand(xt.shape[:2]))[free][:,None]
    result=context.detach().clone()
    result[free]=xt[free]+remaining*velocity[free]
    return result


def load_source_relations(inputs_path,expected_sha256=None,source_manifest_path=None):
    """Verify frozen source identity maps; no approximate vertex association."""
    path=Path(inputs_path).resolve()
    manifest_path=Path(source_manifest_path).resolve() if source_manifest_path else path.with_name('results.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))['assets']
    expected=manifest['inputs']['sha256'] if expected_sha256 is None else expected_sha256
    actual=file_sha256(path)
    if actual!=expected or actual!=manifest['inputs']['sha256']:
        raise RuntimeError('Frozen source inputs hash differs.')
    parents={}; rows=[]
    with np.load(path,allow_pickle=False) as archive:
        for provenance in manifest['raw_source_topology']:
            parent=int(provenance['parent_index']); raw_record=provenance['pre_OT']['source']; raw_path=Path(raw_record['path'])
            if file_sha256(raw_path)!=raw_record['sha256']:
                raise RuntimeError('Actual raw source NPZ identity differs.')
            names=('source_vertices','source_face_vertex_ids','canonical_to_raw_face','target')
            a={name:archive[f'parent{parent}_{name}'].copy() for name in names}
            for name,value in a.items():
                if array_hash(value)!=manifest['inputs']['arrays'][f'parent{parent}_{name}']['sha256']:
                    raise RuntimeError('Source map tensor identity differs: '+name)
            V=a['source_vertices']; F=a['source_face_vertex_ids']; raw_map=a['canonical_to_raw_face']; target=a['target'].reshape(112,3,3)
            with np.load(raw_path,allow_pickle=False) as raw:
                raw_faces=raw['faces'].astype(np.int64); raw_uid=str(raw['uid'].item())
            permutations=np.asarray(provenance['raw_corner_permutations'],dtype=np.int64)
            reconstructed=np.take_along_axis(raw_faces[raw_map],permutations,axis=1)
            if raw_uid!=provenance['parent_uid'] or sorted(raw_map.tolist())!=list(range(112)) or not np.array_equal(F,reconstructed):
                raise RuntimeError('Raw face/source-corner identity is not an exact bijection.')
            if not _same(V[F],target):
                raise RuntimeError('Source corner coordinates do not exactly reconstruct archive.')
            cases={}; case_rows=[]
            for K in SUPPORTED_K:
                prefix=f'parent{parent}_K{K}_'; ids=archive[prefix+'source_face_ids'].copy(); C=archive[prefix+'constraints'].copy()
                for key,value in (('source_face_ids',ids),('constraints',C)):
                    if array_hash(value)!=manifest['inputs']['arrays'][prefix+key]['sha256']:
                        raise RuntimeError('C identity hash differs.')
                if not _same(V[F[ids]],C):
                    raise RuntimeError('C source corner identities differ.')
                phi,phi_audit=context_phi(F[ids])
                cases[K]=dict(source_face_ids=ids,C=C,phi=phi)
                case_rows.append(dict(K=K,phi_sha256=array_hash(phi),**phi_audit))
            parents[parent]=dict(vertices=V,face_vertex_ids=F,cases=cases)
            rows.append(dict(parent=parent,parent_uid=raw_uid,raw_source=raw_record,
                source_map_hashes={name:array_hash(value) for name,value in a.items()},
                all112_corners_exact=True,raw_to_archive_bijective=True,phi_cases=case_rows))
    audit=dict(status='PASS',inputs=dict(path=str(path),sha256=actual),
        source_manifest=dict(path=str(manifest_path),sha256=file_sha256(manifest_path)),parents=rows,
        trustworthy_identity_source='Actual raw vertex IDs mapped through frozen normalization, face ordering, and corner permutations',
        geometry_nearest_neighbor_identity=False,model_calls=0,optimizer_updates=0)
    return SourceRelations(parents,audit)


def run_cpu_tests():
    """Fixed pure mathematical fixtures; no model, optimizer, or real stream."""
    rng=np.random.RandomState(5021)
    ids=np.asarray([[0,1,2],[0,2,3],[0,3,4],[1,3,4]],dtype=np.int64)
    known=np.asarray([True,True,False,False]); C=np.zeros((4,3,3),dtype=np.float32)
    vertices=rng.randn(5,3).astype(np.float32); C[known]=vertices[ids[known]]
    labels=build_relation_labels(ids,known,C); pack=prepare_relations(labels,'cpu')
    assert labels['FC_count']==3 and labels['FF_count']==2
    assert next(g for g in labels['fc_groups'] if g['source_vertex_id']==0)['known_corner_indices'].size==2
    # u=Y-Z reconstructs the endpoint: no loss penalty for the correct geometry.
    Y=np.zeros((4,3,3),dtype=np.float32);Y[~known]=vertices[ids[~known]]
    Z=rng.randn(4,3,3).astype(np.float32);Z[known]=0
    t=torch.tensor([.375],dtype=torch.float32)
    xt=(1-t[:,None,None])*torch.from_numpy(Z.reshape(1,4,9))+t[:,None,None]*torch.from_numpy(Y.reshape(1,4,9))
    context=torch.from_numpy(C.reshape(1,4,9)).requires_grad_()
    mask=torch.from_numpy(known[None]); xt=torch.where(mask[...,None],context.detach(),xt)
    u=torch.from_numpy((Y-Z).reshape(1,4,9)).requires_grad_()
    endpoint=endpoint_from_velocity(xt,u,t,mask,context)
    truth=torch.from_numpy(vertices[ids].reshape(1,4,9))
    assert torch.allclose(endpoint,truth,rtol=0,atol=3e-7)
    zero_losses=relation_sums(endpoint,pack)
    assert float(zero_losses['FC_sum'])<1e-12 and float(zero_losses['FF_sum'])<1e-12
    # Positive nonzero perturbation: independently enumerate every unordered pair.
    pred=torch.from_numpy(rng.randn(4,3,3)).requires_grad_()
    got=relation_sums(pred,pack)
    explicit_fc=pred.sum()*0;explicit_ff=pred.sum()*0
    for g in labels['fc_groups']:
        values=pred.reshape(-1,3)[g['free_corner_indices']]
        explicit_fc=explicit_fc+(values-torch.from_numpy(g['anchor']).double()).square().mean()
    for g in labels['ff_groups']:
        values=pred.reshape(-1,3)[g['free_corner_indices']]
        pair=[(values[i]-values[j]).square().mean() for i in range(len(values)) for j in range(i+1,len(values))]
        explicit_ff=explicit_ff+torch.stack(pair).mean()
    assert torch.allclose(got['FC_sum'],explicit_fc,rtol=1e-12,atol=1e-12)
    assert torch.allclose(got['FF_sum'],explicit_ff,rtol=1e-12,atol=1e-12)
    actual_grad=torch.autograd.grad(got['FC_sum']+got['FF_sum'],pred,retain_graph=True)[0]
    reference_grad=torch.autograd.grad(explicit_fc+explicit_ff,pred)[0]
    assert torch.allclose(actual_grad,reference_grad,rtol=1e-12,atol=1e-12)
    # Prove C anchors are detached even if a caller supplied requires-grad anchors.
    pack['fc_anchors'].requires_grad_(True)
    detached=relation_sums(pred,pack)
    anchors=[pack['fc_anchors']]
    anchor_grads=torch.autograd.grad(detached['FC_sum'],anchors,allow_unused=True)
    assert all(g is None for g in anchor_grads)
    # Full synchronized arbitrary face and corner permutations preserve phi/loss.
    order=np.asarray([2,0,3,1]);corners=np.asarray([[2,0,1],[0,2,1],[1,0,2],[2,1,0]])
    permute=lambda a:np.take_along_axis(a[order],corners[...,None],axis=1) if a.ndim==3 else np.take_along_axis(a[order],corners,axis=1)
    ids2=permute(ids);C2=permute(C);pred2=torch.from_numpy(permute(pred.detach().numpy())).requires_grad_()
    lab2=build_relation_labels(ids2,known[order],C2);got2=relation_sums(pred2,prepare_relations(lab2,'cpu'))
    assert torch.allclose(got2['FC_sum'],got['FC_sum'],rtol=1e-12,atol=1e-12)
    assert torch.allclose(got2['FF_sum'],got['FF_sum'],rtol=1e-12,atol=1e-12)
    phi,_=context_phi(ids[known]);p2,_=context_phi(ids2[known[order]])
    complete_phi=np.zeros((4,3,2),dtype=np.float32);complete_phi[known]=phi
    assert _same(permute(complete_phi)[known[order]],p2)
    # Empty groups preserve an identically zero graph; no NaN/division by zero.
    empty_labels=build_relation_labels(np.arange(12).reshape(4,3),known,C)
    empty=relation_sums(pred,prepare_relations(empty_labels,'cpu'))
    empty_grad=torch.autograd.grad(empty['FC_sum']+empty['FF_sum'],pred)[0]
    assert empty['FC_count']==empty['FF_count']==0 and torch.count_nonzero(empty_grad)==0
    # Validate a group larger than2; pair-variance coefficient is then essential.
    large_ids=np.asarray([[0,1,2],[0,3,4],[0,5,6],[0,7,8]])
    large=build_relation_labels(large_ids,np.zeros(4,dtype=bool),np.zeros((4,3,3),dtype=np.float32))
    large_loss=relation_sums(pred,prepare_relations(large,'cpu'))['FF_sum']
    x=pred[:,0,:]; direct=torch.stack([(x[i]-x[j]).square().mean() for i in range(4) for j in range(i+1,4)]).mean()
    assert torch.allclose(large_loss,direct,rtol=1e-12,atol=1e-12)
    # First derivatives alone cannot detect a detached centroid because their
    # residuals sum to zero. Translation-direction Hessian detects that bug.
    gradient=torch.autograd.grad(large_loss,pred,create_graph=True)[0]
    translation_hessian=torch.autograd.grad(gradient.sum(),pred)[0]
    assert torch.allclose(translation_hessian,torch.zeros_like(translation_hessian),atol=1e-12,rtol=0)
    shifted_velocity=u+torch.full_like(u,.13)
    shifted=endpoint_from_velocity(xt,shifted_velocity,t,mask,context)
    shifted_loss=relation_sums(shifted,pack)
    velocity_grad,context_grad=torch.autograd.grad(shifted_loss['FC_sum']+shifted_loss['FF_sum'],(u,context),allow_unused=True)
    assert context_grad is None and torch.count_nonzero(velocity_grad[mask])==0 and torch.count_nonzero(velocity_grad[~mask])>0
    return dict(status='PASS',fixtures=['u_restores_Y_and_zero_loss','known_anchor_deduplication','explicit_unordered_pairs',
        'pair_gradient_matches_live_mean','C_anchor_detached','face_corner_permutation_invariance','C_only_phi_equivariance','empty_groups_zero_graph','four_corner_group_coefficient','FF_translation_Hessian_preserves_mean_graph','aux_gradient_to_free_velocity_not_C'],
        model_forwards=0,model_backwards=0,optimizer_updates=0,CPU_autograd_grad_calls=7,
        geometry_coordinate_losses_scope='Synthetic math fixtures only; no inference guidance or output repair')
