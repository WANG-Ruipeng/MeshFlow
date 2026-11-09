"""Estimate local deformation solely from paired element positions.

No reference surface, normal, quadric, analytic deformation, or target Jacobian
is accepted. Neighbors/weights are frozen at x0. Failed estimates use identity;
neither singular values nor determinants are clipped.
"""
import numpy as np
from scipy.spatial import cKDTree

K = 16
RANK_RTOL = 1e-6
RIDGE_TRACE_FACTOR = 1e-3
DET_FLOOR = 1e-8
COND_LIMIT = 1e4
REASON_CODES = {0:'accepted', 1:'lsq_rank_deficient', 2:'nonfinite_estimate_or_motion',
                3:'determinant_at_or_below_floor', 4:'condition_above_limit',
                5:'linear_algebra_failure'}


def _finite_stats(values):
    values=np.asarray(values)
    finite=values[np.isfinite(values)]
    return dict(finite_count=int(finite.size),nonfinite_count=int(values.size-finite.size),
                minimum=float(finite.min()) if finite.size else None,
                mean=float(finite.mean()) if finite.size else None,
                maximum=float(finite.max()) if finite.size else None)


def prepare_neighborhood(x0, k=K):
    """Return {'arrays': frozen neighborhood arrays, 'stats': JSON-safe metadata}."""
    x0=np.array(x0,dtype=np.float64,copy=True)
    if k!=K:
        raise ValueError('This protocol fixes k=16')
    if x0.ndim!=2 or x0.shape[1]!=3 or len(x0)<=k or not np.isfinite(x0).all():
        raise ValueError('x0 must be finite (N,3), with N>16')
    distance,index=cKDTree(x0).query(x0,k=k+1,workers=1)
    neighbors=np.empty((len(x0),k),dtype=np.int64)
    distances=np.empty((len(x0),k),dtype=np.float64)
    for i in range(len(x0)):
        mask=index[i]!=i
        neighbors[i]=index[i][mask][:k]
        distances[i]=distance[i][mask][:k]
    h=distances[:,-1]
    if np.any(h<=0) or not np.isfinite(h).all():
        raise ValueError('Zero/nonfinite 16th-neighbor distance; no scale repair permitted')
    weights=np.exp(-(distances/h[:,None])**2)
    weights/=weights.sum(axis=1,keepdims=True)
    X=(x0[neighbors]-x0[:,None,:])/h[:,None,None]
    WX=np.sqrt(weights)[...,None]*X
    U,S,Vh=np.linalg.svd(WX,full_matrices=False)
    rank=np.sum(S>RANK_RTOL*S[:,:1],axis=1).astype(np.int64)
    G=np.einsum('nki,nk,nkj->nij',X,weights,X)
    lam=RIDGE_TRACE_FACTOR*np.trace(G,axis1=1,axis2=2)/3.
    arrays=dict(x0=x0,neighbor_indices=neighbors,neighbor_distances=distances,h=h,
                weights=weights,X=X,weighted_U=U,weighted_singular_values=S,weighted_Vh=Vh,
                rank=rank,G=G,ridge_lambda=lam)
    stats=dict(elements=len(x0),k=K,rank_rtol=RANK_RTOL,ridge_trace_factor=RIDGE_TRACE_FACTOR,
               determinant_floor=DET_FLOOR,condition_limit=COND_LIMIT,
               rank_counts={str(r):int(np.sum(rank==r)) for r in range(4)},
               h=_finite_stats(h),weights_sum_max_error=float(np.max(abs(weights.sum(axis=1)-1))),
               neighbors_frozen=True,weights_frozen=True,
               tie_policy='cKDTree distance order; ties not permutation-certified',
               exact_adjacent_distance_ties=int(np.sum(np.diff(distances,axis=1)==0)))
    return dict(arrays=arrays,stats=stats)


def estimate_motion(x0, xt, neighborhood, mode='lsq16'):
    """Return J_raw/J_used, per-element arrays, and strictly JSON-safe stats.

    J_raw is NaN when LSQ rank<3 or an estimate cannot be formed. Infinite
    conditions and unavailable raw diagnostics stay only in numerical arrays;
    JSON summaries use null and explicit nonfinite counts.
    """
    if mode not in ('lsq16','ridge16'):
        raise ValueError('Unknown estimator mode')
    x0=np.asarray(x0,dtype=np.float64)
    xt=np.asarray(xt,dtype=np.float64)
    a=neighborhood['arrays']
    if x0.shape!=a['x0'].shape or not np.array_equal(x0,a['x0']) or xt.shape!=x0.shape:
        raise ValueError('Position order/shape differs from the frozen neighborhood')
    n=len(x0);indices=a['neighbor_indices'];X=a['X'];weights=a['weights']
    # Displacement-form regression has exactly zero RHS at t=0.
    D=((xt[indices]-xt[:,None,:])-(x0[indices]-x0[:,None,:]))/a['h'][:,None,None]
    J_raw=np.full((n,3,3),np.nan)
    J_used=np.broadcast_to(np.eye(3),(n,3,3)).copy()
    raw_det=np.full(n,np.nan);raw_cond=np.full(n,np.nan)
    raw_residual=np.full(n,np.nan);used_residual=np.full(n,np.nan)
    reason=np.zeros(n,dtype=np.int64)
    for i in range(n):
        if not np.isfinite(D[i]).all():
            reason[i]=2
            continue
        if mode=='lsq16' and a['rank'][i]<3:
            reason[i]=1
        else:
            try:
                if mode=='lsq16':
                    rhs=np.sqrt(weights[i])[:,None]*D[i]
                    B=a['weighted_Vh'][i].T@((a['weighted_U'][i].T@rhs)/a['weighted_singular_values'][i,:,None])
                else:
                    rhs=X[i].T@(weights[i,:,None]*D[i])
                    B=np.linalg.solve(a['G'][i]+a['ridge_lambda'][i]*np.eye(3),rhs)
                J_raw[i]=np.eye(3)+B.T
                raw_residual[i]=np.sqrt(np.sum(weights[i]*np.sum((D[i]-X[i]@B)**2,axis=1)))
                if not np.isfinite(J_raw[i]).all():
                    reason[i]=2
                else:
                    raw_det[i]=np.linalg.det(J_raw[i])
                    raw_cond[i]=np.linalg.cond(J_raw[i])
                    if not np.isfinite(raw_det[i]) or np.isnan(raw_cond[i]):
                        reason[i]=2
                    elif raw_det[i]<=DET_FLOOR:
                        reason[i]=3
                    elif raw_cond[i]>COND_LIMIT or not np.isfinite(raw_cond[i]):
                        reason[i]=4
                    else:
                        J_used[i]=J_raw[i]
            except np.linalg.LinAlgError:
                reason[i]=5
        B_used=(J_used[i]-np.eye(3)).T
        used_residual[i]=np.sqrt(np.sum(weights[i]*np.sum((D[i]-X[i]@B_used)**2,axis=1)))
    fallback=reason!=0
    arrays=dict(J_raw=J_raw,J_used=J_used,neighbor_indices=a['neighbor_indices'].copy(),
                neighbor_distances=a['neighbor_distances'].copy(),weights=a['weights'].copy(),h=a['h'].copy(),
                weighted_design_singular_values=a['weighted_singular_values'].copy(),rank=a['rank'].copy(),
                ridge_lambda=a['ridge_lambda'].copy(),used_fallback=fallback,reason_code=reason,
                raw_determinant=raw_det,raw_condition=raw_cond,
                raw_weighted_residual_normalized=raw_residual,used_weighted_residual_normalized=used_residual,
                raw_weighted_residual_world=raw_residual*a['h'],used_weighted_residual_world=used_residual*a['h'])
    stats=dict(mode=mode,elements=n,k=K,rank_rtol=RANK_RTOL,ridge_trace_factor=RIDGE_TRACE_FACTOR,
               determinant_floor=DET_FLOOR,condition_limit=COND_LIMIT,
               fallback_count=int(fallback.sum()),rank_deficient_count=int(np.sum(a['rank']<3)),
               reason_codes={str(k):v for k,v in REASON_CODES.items()},
               reason_counts={v:int(np.sum(reason==k)) for k,v in REASON_CODES.items()},
               raw_determinant=_finite_stats(raw_det),raw_condition=_finite_stats(raw_cond),
               raw_weighted_residual_normalized=_finite_stats(raw_residual),
               used_weighted_residual_normalized=_finite_stats(used_residual),
               J_raw_nonfinite_elements=int(np.sum(~np.isfinite(J_raw).all(axis=(1,2)))),
               nonfinite_motion_elements=int(np.sum(~np.isfinite(D).all(axis=(1,2)))),
               npz_nonfinite_policy='NaN marks unavailable raw J/diagnostics; inf may mark singular raw condition. JSON uses null/counts.',
               no_reference_geometry=True,no_analytic_jacobian=True,no_clipping=True,
               fallback='identity',unobserved_motion_prior='identity through zero displacement gradient',
               t0_exact_identity=bool(np.array_equal(xt,x0) and np.array_equal(J_used,np.broadcast_to(np.eye(3),(n,3,3)))))
    return dict(J_raw=J_raw,J_used=J_used,arrays=arrays,stats=stats)
