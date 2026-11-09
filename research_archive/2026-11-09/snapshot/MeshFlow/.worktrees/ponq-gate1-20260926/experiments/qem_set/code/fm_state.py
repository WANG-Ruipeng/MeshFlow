"""Full centered-Q state encoding for a future FM candidate; no training.

The N x 22 layout is [x(3), raw n(3), principal PSD sqrt F(16)].
Encoding keeps the complete centered linear/constant terms and raw normals.
Decoding preserves raw normals by default; normalization is an explicit option.
"""
import numpy as np

NEGATIVE_SPECTRUM_TOLERANCE_ULPS = 64
STATE_DIM = 22


def _array(value, shape, name):
    result=np.array(value,dtype=np.float64,copy=True)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError(name+' has invalid shape or nonfinite entries')
    return result


def _translation(x):
    matrix=np.broadcast_to(np.eye(4),(len(x),4,4)).copy()
    matrix[:,:3,3]=x
    return matrix


def _congruence(Q,T):
    return T.transpose(0,2,1)@Q@T


def encode_state(params):
    """Return (float64 N x 22 state, JSON-safe spectrum/roundoff diagnostics).

    Only roundoff-sized antisymmetry is symmetrized; material asymmetry or a
    negative eigenvalue below -64 eps ||C||_F raises. No ridge is introduced.
    """
    raw_x=np.asarray(params['v'])
    if raw_x.ndim!=2 or raw_x.shape[1]!=3 or len(raw_x)==0:
        raise ValueError('v must have nonempty shape (N,3)')
    count=len(raw_x)
    x=_array(raw_x,(count,3),'v')
    normals=_array(params['n'],(count,3),'n')
    Q=_array(params['Q'],(count,4,4),'Q')
    C=_congruence(Q,_translation(x))
    if not np.isfinite(C).all():
        raise ValueError('Centering produced nonfinite Q')
    norm=np.linalg.norm(C,axis=(1,2))
    tolerance=NEGATIVE_SPECTRUM_TOLERANCE_ULPS*np.finfo(np.float64).eps*norm
    symmetric=(C+C.transpose(0,2,1))/2
    asymmetry=np.linalg.norm(C-C.transpose(0,2,1),axis=(1,2))
    if np.any(asymmetry>tolerance):
        raise ValueError('Materially asymmetric centered Q; no repair allowed')
    values,vectors=np.linalg.eigh(symmetric)
    if np.any(values[:,0]<-tolerance):
        indices=np.flatnonzero(values[:,0]<-tolerance)
        raise ValueError('Materially negative centered Q spectrum at elements '+str(indices.tolist()))
    negative=np.minimum(values,0.)
    clipped_counts=(values<0).sum(axis=1)
    factor=(vectors*np.sqrt(np.maximum(values,0.))[:,None,:])@vectors.transpose(0,2,1)
    reconstructed=factor@factor.transpose(0,2,1)
    error=np.linalg.norm(reconstructed-C,axis=(1,2))
    relative=np.divide(error,norm,out=np.zeros_like(error),where=norm!=0)
    state=np.concatenate([x,normals,factor.reshape(count,16)],axis=1)
    diagnostics=dict(
        state_layout=['x3','raw_normal3','principal_centered_psd_factor16'],
        state_dimension=STATE_DIM,elements=count,dtype='float64',
        centered_quadric_convention='C=T(x).T @ Q_world @ T(x); x_world=x+y_local',
        negative_tolerance_ulps=NEGATIVE_SPECTRUM_TOLERANCE_ULPS,
        tolerance_norm='Frobenius norm of full centered 4x4 C',
        psd_tolerance_per_matrix=tolerance.tolist(),
        negative_eigenvalues_clipped=int(clipped_counts.sum()),
        matrices_with_negative_eigenvalue_clipping=int(np.count_nonzero(clipped_counts)),
        negative_eigenvalue_count_per_matrix=clipped_counts.tolist(),
        negative_spectrum_clipped_l2_per_matrix=np.linalg.norm(negative,axis=1).tolist(),
        negative_spectrum_clipped_frobenius=float(np.linalg.norm(negative)),
        maximum_negative_eigenvalue_magnitude_clipped=float(max(0.,-values.min())),
        minimum_eigenvalue_before_clipping=float(values.min()),
        symmetrization_frobenius_per_matrix=np.linalg.norm(symmetric-C,axis=(1,2)).tolist(),
        centered_factor_reconstruction_max_abs=float(np.max(abs(reconstructed-C))),
        centered_factor_reconstruction_relative_frobenius_mean=float(relative.mean()),
        centered_factor_reconstruction_relative_frobenius_max=float(relative.max()),
        zero_reference_nonzero_reconstruction_count=int(np.sum((norm==0)&(error!=0))),
        raw_normal_magnitude_min=float(np.linalg.norm(normals,axis=1).min()),
        raw_normal_magnitude_max=float(np.linalg.norm(normals,axis=1).max()),
        normal_encoding_normalized=False,normal_decoding_default_normalized=False,
        normal_decoding_normalization_is_explicit_option=True,
        centered_linear_term_discarded=False,centered_constant_term_discarded=False,
        ridge_added=False,positive_eigenvalue_floor=0.)
    return state,diagnostics


def decode_state(state, *, normalize_normals=False):
    """Decode x/raw-n/F into v=p=x, full world Q and A spectrum.

    Normal magnitudes are preserved unless normalize_normals=True is explicitly
    selected. That option is a deployment change, not a strict target replay.

    Reject zero normals and a zero spatial-Q normalization spectrum. No spectral
    floor, latent clipping, positional correction or quadric repair is used.
    """
    state=np.array(state,dtype=np.float64,copy=True)
    if state.ndim!=2 or state.shape[1]!=STATE_DIM or len(state)==0 or not np.isfinite(state).all():
        raise ValueError('state must be finite nonempty (N,22)')
    x=state[:,:3].copy()
    raw_n=state[:,3:6].copy()
    lengths=np.linalg.norm(raw_n,axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths==0):
        raise ValueError('Cannot unit-normalize a zero/nonfinite normal')
    n=raw_n/lengths[:,None] if normalize_normals else raw_n
    factor=state[:,6:].reshape(-1,4,4)
    C=factor@factor.transpose(0,2,1)
    Q=_congruence(C,_translation(-x))
    if not np.isfinite(Q).all():
        raise ValueError('Decoded full world Q is nonfinite')
    eigs=np.linalg.svd(Q[:,:3,:3],compute_uv=False)
    if np.any(eigs[:,0]<=0):
        raise ValueError('Decoded spatial Q has zero normalization eigenvalue')
    return dict(v=x,p=x.copy(),n=n,Q=Q,eigs=eigs)
