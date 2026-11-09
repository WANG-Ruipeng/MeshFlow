"""Incremental full-quadric transport from paired element motion only.

Neighbor IDs, weights and distance scale remain those of the initial frame.
Each step rebuilds the regression design from the current source positions.
The existing lsq16 estimator supplies every Jacobian and fallback decision.
"""
import copy
import numpy as np
from motion_jacobian import (estimate_motion, K, RANK_RTOL, RIDGE_TRACE_FACTOR)


def _positions(value, name):
    value=np.array(value,dtype=np.float64,copy=True)
    if value.ndim!=2 or value.shape[1]!=3 or not np.isfinite(value).all():
        raise ValueError(name+' must be finite (N,3)')
    return value


def _summary(value):
    value=np.asarray(value)
    finite=value[np.isfinite(value)]
    return dict(finite_count=int(finite.size),nonfinite_count=int(value.size-finite.size),
                minimum=float(finite.min()) if finite.size else None,
                mean=float(finite.mean()) if finite.size else None,
                maximum=float(finite.max()) if finite.size else None)


def refresh_neighborhood(x_current, frozen_nb):
    """Return an estimate_motion-compatible neighborhood for the current step.

    IDs/weights/h are copied exactly from frozen_nb; no nearest-neighbor search
    or weight/scale update occurs. x0 in this returned cache means the current
    regression source frame, not the original experiment frame.
    """
    current=_positions(x_current,'x_current')
    frozen=frozen_nb['arrays']
    ids=np.array(frozen['neighbor_indices'],copy=True)
    weights=np.array(frozen['weights'],copy=True)
    h=np.array(frozen['h'],copy=True)
    if ids.shape!=(len(current),K) or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('Frozen neighbor IDs must have shape (N,16)')
    if np.any(ids<0) or np.any(ids>=len(current)):
        raise ValueError('Frozen neighbor ID outside current element range')
    if weights.shape!=ids.shape or h.shape!=(len(current),):
        raise ValueError('Frozen weight/scale shape mismatch')
    if not np.isfinite(weights).all() or np.any(weights<0) or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError('Invalid frozen weights or scales; no repair is allowed')
    if not np.allclose(weights.sum(axis=1),1.,rtol=0.,atol=8*np.finfo(float).eps):
        raise ValueError('Frozen weights are not normalized')
    offsets=current[ids]-current[:,None,:]
    X=offsets/h[:,None,None]
    weighted=np.sqrt(weights)[...,None]*X
    U,S,Vh=np.linalg.svd(weighted,full_matrices=False)
    rank=np.sum(S>RANK_RTOL*S[:,:1],axis=1).astype(np.int64)
    G=np.einsum('nki,nk,nkj->nij',X,weights,X)
    lam=RIDGE_TRACE_FACTOR*np.trace(G,axis1=1,axis2=2)/3.
    arrays=dict(x0=current,neighbor_indices=ids,neighbor_distances=np.array(frozen['neighbor_distances'],copy=True),
                current_neighbor_distances=np.linalg.norm(offsets,axis=2),h=h,weights=weights,X=X,
                weighted_U=U,weighted_singular_values=S,weighted_Vh=Vh,rank=rank,G=G,ridge_lambda=lam)
    stats=copy.deepcopy(frozen_nb.get('stats',{}))
    stats.update(elements=len(current),k=K,rank_rtol=RANK_RTOL,
                 ridge_trace_factor=RIDGE_TRACE_FACTOR,
                 rank_counts={str(r):int(np.sum(rank==r)) for r in range(4)},
                 neighbors_frozen=True,weights_frozen=True,h_frozen=True,
                 regression_source='current step source positions',
                 design_svd_rank_gram_recomputed=True,no_neighbor_search=True,
                 neighbor_distances_field='original frozen distances',
                 current_neighbor_distances=_summary(arrays['current_neighbor_distances']),
                 h=_summary(h),weights_sum_max_error=float(np.max(abs(weights.sum(axis=1)-1))))
    return dict(arrays=arrays,stats=stats)


def step_transport(Q_world, x_current, x_next, frozen_nb):
    """Return (Q_next, H, estimate, current_nb), without mutating any input.

    H_i maps x to J_used_i x + b_i, b_i=x_next_i-J_used_i x_current_i.
    Q_next=H_i^{-T} Q_world H_i^{-1}. Thus an identity-J fallback still
    translates that element's complete quadric by its measured displacement.
    Raw/used sigma_min diagnostics are added to estimate['arrays']/['stats'].
    """
    current=_positions(x_current,'x_current')
    target=_positions(x_next,'x_next')
    if target.shape!=current.shape:
        raise ValueError('Current and next position shapes differ')
    Q=np.array(Q_world,dtype=np.float64,copy=True)
    if Q.shape!=(len(current),4,4) or not np.isfinite(Q).all():
        raise ValueError('Q_world must be finite (N,4,4); no repair is allowed')
    current_nb=refresh_neighborhood(current,frozen_nb)
    estimate=estimate_motion(current,target,current_nb,mode='lsq16')
    J=estimate['J_used']
    b=target-np.einsum('nij,nj->ni',J,current)
    H=np.broadcast_to(np.eye(4),(len(current),4,4)).copy()
    H[:,:3,:3]=J
    H[:,:3,3]=b
    inverse_J=np.linalg.inv(J)
    inverse_H=np.broadcast_to(np.eye(4),(len(current),4,4)).copy()
    inverse_H[:,:3,:3]=inverse_J
    inverse_H[:,:3,3]=-np.einsum('nij,nj->ni',inverse_J,b)
    Q_next=inverse_H.transpose(0,2,1)@Q@inverse_H
    if not np.isfinite(Q_next).all():
        raise ValueError('Transport produced nonfinite Q; no repair is allowed')
    used_sigma=np.linalg.svd(J,compute_uv=False)[:,-1]
    raw=estimate['J_raw']
    raw_sigma=np.full(len(raw),np.nan)
    finite=np.isfinite(raw).all(axis=(1,2))
    if finite.any():
        raw_sigma[finite]=np.linalg.svd(raw[finite],compute_uv=False)[:,-1]
    estimate['arrays'].update(raw_sigma_min=raw_sigma,used_sigma_min=used_sigma)
    estimate['stats'].update(raw_sigma_min=_summary(raw_sigma),used_sigma_min=_summary(used_sigma),
                             transport='full world homogeneous Q inverse congruence',
                             incremental=True,analytic_jacobian_used=False,
                             fallback_transport='identity Jacobian plus measured per-element translation',
                             Q_repair_performed=False)
    return Q_next,H,estimate,current_nb
