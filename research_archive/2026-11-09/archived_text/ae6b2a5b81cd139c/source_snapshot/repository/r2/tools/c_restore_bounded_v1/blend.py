"""Bounded local C-restoration blend; pure NumPy, no solve/model/RNG/IO.

The radius and free safety cap match the previous RBF diagnostic.  This
version uses nonnegative subconvex weights, including support fade when their
sum is below one.  These bounds are not a collision or connectivity guarantee.
"""
from __future__ import annotations
import numpy as np

VERSION='c_restore_bounded_wendland_blend_v1'
RADIUS_L=0.10
FREE_CAP_L=0.02

def definition():
    return dict(version=VERSION,kernel='Wendland C2: (1-q)^4*(4*q+1) for 0<=q<1, else 0',
        radius_L=RADIUS_L,free_cap_L=FREE_CAP_L,
        weights='Every source vertex ID has total observation weight 1; all proposal corner copies retained.',
        formula='b_i=C_i-C_hat_i; s_i(x)=k(x,C_hat_i)/multiplicity(source_id_i); alpha_i=s_i/max(1,sum_j s_j); u=sum_i alpha_i*b_i',
        support_fade='sum alpha = min(sum s,1); no support gives exactly zero displacement',
        arithmetic='FP64 kernel/weights/blend; FP32 output; no matrix solve and no random numbers',
        theoretical_bound='norm(u(x)) <= max(norm(b_i) for s_i(x)>0); maximum of empty set is zero',
        actual_bound='norm(F_out_FP32-F_hat_FP32) <= min(0.02L,max participating norm(b)); checked in FP64 after rounding',
        clipping='Radial projection to per-corner bound, then one conservative FP32 nextafter toward the original free corner only where rounding exceeds the bound. Failure raises; no fallback deformation.',
        output='Original C bytes copied unchanged; free faces and corner order unchanged',
        limitations='No source/free correspondence, shared-edge, winding, semantic-use, collision, or fold-free guarantee')

def kernel(x,centers,radius):
    x=np.asarray(x,dtype=np.float64).reshape(-1,3)
    centers=np.asarray(centers,dtype=np.float64).reshape(-1,3)
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive finite kernel radius required')
    q=np.linalg.norm(x[:,None,:]-centers[None,:,:],axis=2)/radius
    t=np.maximum(1.0-q,0.0)
    return np.where(q<1.0,t**4*(4.0*q+1.0),0.0)

def _stats(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
        p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)

def coordinate_restore(C,proposal,C_source_vertex_ids,*,L):
    """Return (output, diagnostic, details), matching the old geometry API.

    C and full unconstrained proposal must be FP32 triangle arrays (K,3,3)
    and (N,3,3), 0<K<N.  IDs must be original integers (K,3).  Details are
    arrays suitable for individual NPZ files; diagnostics contain JSON values.
    Caller arrays are read-only inputs.  Original C bytes are preserved.
    """
    C=np.asarray(C);proposal=np.asarray(proposal);ids=np.asarray(C_source_vertex_ids)
    if C.dtype!=np.float32 or proposal.dtype!=np.float32:raise TypeError('Require actual FP32 C and proposal')
    if C.ndim!=3 or C.shape[1:]!=(3,3) or len(C)==0:raise ValueError('C must be nonempty (K,3,3)')
    if proposal.ndim!=3 or proposal.shape[1:]!=(3,3) or len(proposal)<=len(C):raise ValueError('Require both C and free faces')
    if ids.shape!=C.shape[:2] or not np.issubdtype(ids.dtype,np.integer):raise ValueError('Original integer source IDs required')
    if not np.isfinite(C).all() or not np.isfinite(proposal).all() or not np.isfinite(L) or L<=0:raise ValueError('Nonfinite geometry/scale')
    K=len(C);radius=RADIUS_L*float(L);cap=FREE_CAP_L*float(L)
    if not np.isfinite(radius) or radius<=0 or not np.isfinite(cap) or cap<=0:raise ValueError('Invalid scaled radius/cap')
    centers=proposal[:K].reshape(-1,3).astype(np.float64);target=C.reshape(-1,3).astype(np.float64)
    flat_ids=ids.reshape(-1).astype(np.int64);unique,inverse,counts=np.unique(flat_ids,return_inverse=True,return_counts=True)
    weights=1.0/counts[inverse];b=target-centers;b_norm=np.linalg.norm(b,axis=1)
    original=C.reshape(-1,3)
    for j in range(len(unique)):
        ii=np.flatnonzero(inverse==j)
        if any(original[i].tobytes()!=original[ii[0]].tobytes() for i in ii):raise ValueError('Original source-ID coordinates disagree')
    free=proposal[K:].reshape(-1,3)
    values=kernel(free,centers,radius);s=values*weights[None,:]
    support_sum=s.sum(axis=1);denominator=np.maximum(1.0,support_sum)
    alpha=s/denominator[:,None];alpha_sum=alpha.sum(axis=1)
    supported=s>0.0;support_count=supported.sum(axis=1)
    max_b=np.max(np.where(supported,b_norm[None,:],0.0),axis=1)
    bound=np.minimum(cap,max_b);unclipped=alpha@b;norms=np.linalg.norm(unclipped,axis=1)
    if not np.isfinite(unclipped).all() or not np.isfinite(norms).all():raise ArithmeticError('Nonfinite local blend')
    # The subconvex bound is exact mathematically. The projection also removes
    # any floating-point accumulation excess before the output rounding check.
    clipped_mask=norms>bound
    factor=np.ones(len(free),dtype=np.float64)
    np.divide(bound,norms,out=factor,where=clipped_mask)
    clipped=unclipped*factor[:,None]
    corrected=free.copy();changed=np.any(clipped!=0,axis=1)
    corrected[changed]=(free[changed].astype(np.float64)+clipped[changed]).astype(np.float32)
    actual=corrected.astype(np.float64)-free.astype(np.float64)
    rounding_adjust=np.linalg.norm(actual,axis=1)>bound
    corrected[rounding_adjust]=np.nextafter(corrected[rounding_adjust],free[rounding_adjust])
    actual=corrected.astype(np.float64)-free.astype(np.float64);actual_norm=np.linalg.norm(actual,axis=1)
    if not np.isfinite(corrected).all() or np.any(actual_norm>bound):raise ArithmeticError('Actual FP32 displacement bound not satisfied')
    if np.any(actual[support_count==0]!=0):raise ArithmeticError('No-support corner moved')
    if np.any(alpha<0) or np.any(alpha_sum>1+8*np.finfo(np.float64).eps):raise ArithmeticError('Invalid subconvex blend weights')
    output=proposal.copy();output[:K]=C;output[K:]=corrected.reshape(-1,3,3)
    duplicate=[]
    for j,vid in enumerate(unique):
        ii=np.flatnonzero(inverse==j);cc=centers[ii]
        duplicate.append(dict(source_vertex_id=int(vid),copies=len(ii),total_weight=float(weights[ii].sum()),
            proposal_split_diameter=float(np.linalg.norm(cc[:,None,:]-cc[None,:,:],axis=2).max()),
            target_displacement_max=float(b_norm[ii].max())))
    diag=dict(status='PASS',definition=definition(),L=float(L),radius_model=radius,free_cap_model=cap,
        anchor_corners=len(centers),source_vertices=len(unique),free_corners=len(free),
        C_bytes_preserved=output[:K].tobytes()==C.tobytes(),all_original_corner_observations_used=True,
        duplicates=duplicate,duplicate_source_vertex_groups=int(sum(counts>1)),
        split_source_vertex_groups=sum(r['proposal_split_diameter']>0 for r in duplicate),
        support_sum=_stats(support_sum),alpha_sum=_stats(alpha_sum),
        supported_free_corners=int(sum(support_count>0)),no_support_free_corners=int(sum(support_count==0)),
        free_support_participating_b_max_norm=_stats(max_b),free_actual_bound_norm=_stats(bound),
        free_unclipped_displacement_norm=_stats(norms),free_applied_FP32_displacement_norm=_stats(actual_norm),
        clipped_free_corners=int(sum(clipped_mask)),safety_cap_active_free_corners=int(sum(norms>cap)),
        theoretical_bound_roundoff_clipped_free_corners=int(sum((norms>max_b)&(norms<=cap))),
        FP32_cap_rounding_adjusted_corners=int(rounding_adjust.sum()),
        removed_by_clipping_norm=_stats(np.linalg.norm(unclipped-clipped,axis=1)),
        minimum_actual_bound_margin=float(np.min(bound-actual_norm)),
        model_calls=0,new_noise=0,matrix_solves=0)
    details=dict(anchor_source_vertex_ids=flat_ids,anchor_proposal_model=centers,anchor_target_model=target,
        anchor_weight=weights,anchor_displacement_target=b,anchor_displacement_target_norm=b_norm,
        free_kernel=values,free_weighted_support=s,free_alpha=alpha,free_support_sum=support_sum,
        free_alpha_sum=alpha_sum,free_support_count=support_count,free_supported=supported,
        free_participating_b_max_norm=max_b,free_actual_bound=bound,
        free_displacement_unclipped=unclipped,free_displacement_clipped=clipped,
        free_clip_factor=factor,free_clipped=clipped_mask,free_safety_cap_active=norms>cap,
        free_FP32_cap_rounding_adjusted=rounding_adjust,free_actual_displacement_FP32=actual,
        free_actual_displacement_norm=actual_norm,free_actual_bound_margin=bound-actual_norm)
    return output,diag,details
