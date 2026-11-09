"""Fixed compact-RBF coordinated C restoration, pure NumPy CPU arithmetic.

Not exact interpolation, semantic attachment, or a fold-free deformation.
The original C triangle array is immutable and copied byte-for-byte to output.
"""
from __future__ import annotations
import numpy as np

VERSION='c_restore_weighted_wendland_c2_v1'
RADIUS_L=0.10
RIDGE=1e-6
FREE_CAP_L=0.02

def definition():
    return dict(version=VERSION,kernel='Wendland C2: (1-q)^4*(4*q+1) for 0<=q<1, else 0',
        radius_L=RADIUS_L,ridge=RIDGE,free_cap_L=FREE_CAP_L,
        weights='Each original source vertex ID has total observation weight 1; all proposal corner copies retained',
        objective='sum_i w_i ||(K a)_i-b_i||^2 + lambda trace(a^T K a)',
        solve='(K + lambda diag(1/w)) a = b; float64',
        anchor_residual='K a - b, before free-only clipping; not claimed zero',
        output='F_hat + radially clipped field at F_hat, rounded FP32; original C restored bytewise',
        clipping='Each free corner actual FP32 displacement from F_hat <=0.02L; one nextafter toward F_hat when rounding would exceed cap',
        limitations='No correspondence, connectivity, collision, semantic-use, or invertible-map guarantee')

def kernel(x,centers,radius):
    x=np.asarray(x,dtype=np.float64).reshape(-1,3)
    centers=np.asarray(centers,dtype=np.float64).reshape(-1,3)
    q=np.linalg.norm(x[:,None,:]-centers[None,:,:],axis=2)/radius
    t=np.maximum(1.0-q,0.0)
    return np.where(q<1.0,t**4*(4.0*q+1.0),0.0)

def _stats(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
        p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)

def coordinate_restore(C,proposal,C_source_vertex_ids,*,L):
    """Return (FP32 full output, JSON diagnostics, flat NPZ-ready details).

    C: original (K,3,3), proposal: unconstrained D0 (N,3,3), both FP32.
    IDs: original integer vertex IDs (K,3); L: frozen source-data scale.
    No model calls, no RNG, no IO, no mutation of caller-owned arrays.
    """
    C=np.asarray(C);proposal=np.asarray(proposal);ids=np.asarray(C_source_vertex_ids)
    if C.dtype!=np.float32 or proposal.dtype!=np.float32:raise TypeError('Require actual FP32 C and proposal')
    if C.ndim!=3 or C.shape[1:]!=(3,3) or len(C)==0:raise ValueError('C must be nonempty (K,3,3)')
    if proposal.ndim!=3 or proposal.shape[1:]!=(3,3) or len(proposal)<=len(C):raise ValueError('Require both C and free faces')
    if ids.shape!=C.shape[:2] or not np.issubdtype(ids.dtype,np.integer):raise ValueError('Original integer source IDs required')
    if not np.isfinite(C).all() or not np.isfinite(proposal).all() or not np.isfinite(L) or L<=0:raise ValueError('Nonfinite geometry/scale')
    K=len(C);radius=RADIUS_L*float(L);cap=FREE_CAP_L*float(L)
    centers=proposal[:K].reshape(-1,3).astype(np.float64);target=C.reshape(-1,3).astype(np.float64)
    flat_ids=ids.reshape(-1).astype(np.int64);unique,inverse,counts=np.unique(flat_ids,return_inverse=True,return_counts=True)
    weights=1.0/counts[inverse];b=target-centers
    # Same original vertex ID is a coordinate identity, not a proposed weld.
    original=C.reshape(-1,3)
    for j in range(len(unique)):
        ii=np.flatnonzero(inverse==j)
        if any(original[i].tobytes()!=original[ii[0]].tobytes() for i in ii):raise ValueError('Original source-ID coordinates disagree')
    gram=kernel(centers,centers,radius)
    system=gram+np.diag(RIDGE/weights)
    coefficients=np.linalg.solve(system,b)
    fit=gram@coefficients;residual=fit-b;normal_residual=system@coefficients-b
    free=proposal[K:].reshape(-1,3);unclipped=kernel(free,centers,radius)@coefficients
    norms=np.linalg.norm(unclipped,axis=1)
    factor=np.ones(len(free),dtype=np.float64)
    np.divide(cap,norms,out=factor,where=norms>cap)
    clipped=unclipped*factor[:,None]
    corrected=free.copy();changed=np.any(clipped!=0,axis=1)
    corrected[changed]=(free[changed].astype(np.float64)+clipped[changed]).astype(np.float32)
    actual=corrected.astype(np.float64)-free.astype(np.float64)
    rounding_adjust=np.linalg.norm(actual,axis=1)>cap
    corrected[rounding_adjust]=np.nextafter(corrected[rounding_adjust],free[rounding_adjust])
    actual=corrected.astype(np.float64)-free.astype(np.float64)
    if np.any(np.linalg.norm(actual,axis=1)>cap):raise ArithmeticError('FP32 clipping contract not satisfied')
    if not np.isfinite(coefficients).all() or not np.isfinite(corrected).all():raise ArithmeticError('Nonfinite solve/output; no fallback')
    output=proposal.copy();output[:K]=C;output[K:]=corrected.reshape(-1,3,3)
    duplicate=[]
    for j,vid in enumerate(unique):
        ii=np.flatnonzero(inverse==j);cc=centers[ii];bb=b[ii]
        duplicate.append(dict(source_vertex_id=int(vid),copies=len(ii),total_weight=float(weights[ii].sum()),
            proposal_split_diameter=float(np.linalg.norm(cc[:,None,:]-cc[None,:,:],axis=2).max()),
            proposal_split_rms=float(np.sqrt(np.mean(np.sum((cc-cc.mean(0))**2,axis=1)))),
            target_displacement_spread=float(np.linalg.norm(bb[:,None,:]-bb[None,:,:],axis=2).max()),
            fit_residual_max=float(np.linalg.norm(residual[ii],axis=1).max())))
    weighted_mse=float(np.sum(weights[:,None]*residual**2)/weights.sum())
    diag=dict(status='PASS',definition=definition(),L=float(L),radius_model=radius,free_cap_model=cap,
        anchor_corners=len(centers),source_vertices=len(unique),free_corners=len(free),
        C_bytes_preserved=output[:K].tobytes()==C.tobytes(),all_original_corner_observations_used=True,
        anchor_fit_residual_norm=_stats(np.linalg.norm(residual,axis=1)),anchor_weighted_vector_MSE=weighted_mse,
        system_residual_max=float(np.abs(normal_residual).max()),system_condition_number_2=float(np.linalg.cond(system)),
        duplicates=duplicate,duplicate_source_vertex_groups=sum(len(np.flatnonzero(inverse==j))>1 for j in range(len(unique))),
        split_source_vertex_groups=sum(r['proposal_split_diameter']>0 for r in duplicate),
        free_unclipped_displacement_norm=_stats(norms),free_applied_FP32_displacement_norm=_stats(np.linalg.norm(actual,axis=1)),
        clipped_free_corners=int(sum(norms>cap)),FP32_cap_rounding_adjusted_corners=int(rounding_adjust.sum()),
        removed_by_clipping_norm=_stats(np.linalg.norm(unclipped-clipped,axis=1)),
        model_calls=0,new_noise=0)
    details=dict(anchor_source_vertex_ids=flat_ids,anchor_proposal_model=centers,anchor_target_model=target,
        anchor_weight=weights,anchor_displacement_target=b,anchor_displacement_fit=fit,anchor_fit_residual=residual,
        coefficients=coefficients,free_displacement_unclipped=unclipped,free_displacement_clipped=clipped,
        free_clip_factor=factor,free_clipped=norms>cap,free_FP32_cap_rounding_adjusted=rounding_adjust,
        free_actual_displacement_FP32=actual)
    return output,diag,details
