"""CPU-only arithmetic on saved FP32 tensors; safe to import in GPU producers.

This module never loads a model, samples noise, or changes process environment.
Effective residual subtraction and residual differences are FP32. Reductions
are FP64 over those saved FP32 values. Real examples have no reconstruction GT.
"""
from __future__ import annotations
import collections
import numpy as np

VERSION = 'single_step_effective_residual_fidelity_v1'
LOW_SIGNAL = 1e-14
SCALE = .95 * .3762
L = 9.208138264587333


def definition():
    return dict(version=VERSION, residual='FP32(proposal_d - input_d)',
        difference='FP32(r_S_d - r_T_d), then FP64 square/sum',
        denominator='9 * number of valid free faces; C excluded',
        E='sum squared residual error / free scalar count',
        B='sum squared teacher effective residual / free scalar count',
        R='E/B; identity R=1 only when B>1e-14',
        LOW_SIGNAL_teacher_mean_square_max=LOW_SIGNAL,
        pooling='pooled E and B use all free scalars; pooled R=sum error SSE/sum teacher energy, subject to the same pooled B low-signal rule',
        vector_cosine='one flattened free residual vector; NA for low teacher signal or zero student energy',
        coordinate_RMS='sqrt(mean(square(all coordinate differences)))',
        corner_distance='Euclidean norm of each free corner displacement, quantiles over corners',
        units=dict(denoiser_scale=SCALE, model_reference_L=L),
        synthetic_target='same-shape diagnostic only, FP64 difference of saved FP32 proposal and target; never real DEV reconstruction')


def _tensor(a, name, shape=None):
    a = np.asarray(a)
    if a.dtype != np.float32 or a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError(name + ' must be actual FP32 [N,3,3]')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' shape mismatch')
    if not np.isfinite(a).all():
        raise ValueError(name + ' is not finite')
    return a


def _distance_fields(delta, prefix):
    a = np.asarray(delta, dtype=np.float64)
    norms = np.linalg.norm(a, axis=-1).reshape(-1)
    return {prefix + 'coordinate_RMS': float(np.sqrt(np.mean(a*a))),
        prefix + 'corner_L2_RMS': float(np.sqrt(np.mean(norms*norms))),
        prefix + 'corner_L2_mean': float(np.mean(norms)),
        prefix + 'corner_L2_median': float(np.median(norms)),
        prefix + 'corner_L2_p95': float(np.quantile(norms, .95)),
        prefix + 'corner_L2_max': float(np.max(norms))}


def residual_metrics(input_d, proposal_T_d, proposal_S_d, K,
                     teacher_model=None, student_model=None, target_d=None):
    x = _tensor(input_d, 'input_d')
    t = _tensor(proposal_T_d, 'proposal_T_d', x.shape)
    s = _tensor(proposal_S_d, 'proposal_S_d', x.shape)
    K = int(K)
    if not 0 <= K < len(x):
        raise ValueError('No valid free faces or invalid K')
    rt = np.subtract(t, x, dtype=np.float32)[K:]
    rs = np.subtract(s, x, dtype=np.float32)[K:]
    error = np.subtract(rs, rt, dtype=np.float32).astype(np.float64)
    td, sd = rt.astype(np.float64), rs.astype(np.float64)
    e, b, se = float(np.sum(error*error)), float(np.sum(td*td)), float(np.sum(sd*sd))
    dot = float(np.sum(td*sd)); den = 9*(len(x)-K)
    signal = b/den > LOW_SIGNAL
    out = dict(metric_version=VERSION, N=len(x), K=K, free_faces=len(x)-K,
        free_scalar_count=den, free_corner_count=3*(len(x)-K), error_SSE=e,
        teacher_energy_sum=b, student_energy_sum=se, teacher_student_dot_sum=dot,
        E=e/den, B=b/den, student_mean_square=se/den,
        signal_status='OK' if signal else 'LOW_SIGNAL', R=e/b if signal else None,
        residual_norm_ratio=float(np.sqrt(se/b)) if signal else None,
        residual_cosine=float(np.clip(dot/np.sqrt(b*se), -1, 1)) if signal and se>0 else None,
        cosine_status='LOW_SIGNAL' if not signal else ('ZERO_STUDENT' if se==0 else 'OK'))
    dd = np.subtract(s, t, dtype=np.float32)[K:]
    out.update(_distance_fields(dd, 'pre_GEO_d_'))
    if (teacher_model is None) != (student_model is None):
        raise ValueError('Both actual model-unit proposals are required together')
    if teacher_model is not None:
        tm = _tensor(teacher_model, 'teacher_model', x.shape)
        sm = _tensor(student_model, 'student_model', x.shape)
        dm = np.subtract(sm, tm, dtype=np.float32)[K:]
        out['model_distance_source'] = 'ACTUAL_SAVED_MODEL_PROPOSALS'
    else:
        dm = dd.astype(np.float64)/SCALE
        out['model_distance_source'] = 'DERIVED_D_DIFFERENCE_DIVIDED_BY_SCALE'
    out.update(_distance_fields(dm, 'pre_GEO_model_'))
    out.update(_distance_fields(np.asarray(dm,dtype=np.float64)/L, 'pre_GEO_L_'))
    out['target_status'] = 'NA_REAL_NO_UNIQUE_GT' if target_d is None else 'SYNTHETIC_SAME_SHAPE_DIAGNOSTIC'
    if target_d is not None:
        y = _tensor(target_d, 'target_d', x.shape)
        for tag, pred in (('teacher', t), ('student', s), ('identity', x)):
            delta = pred[K:].astype(np.float64)-y[K:].astype(np.float64)
            out[tag+'_target_MSE_d'] = float(np.mean(delta*delta))
            out.update(_distance_fields(delta, tag+'_target_d_'))
    return out


def summarize(rows, keys=()):
    groups=collections.defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output=[]
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        count=sum(int(r['free_scalar_count']) for r in group)
        error=sum(float(r['error_SSE']) for r in group)
        energy=sum(float(r['teacher_energy_sum']) for r in group)
        se=sum(float(r['student_energy_sum']) for r in group)
        dot=sum(float(r['teacher_student_dot_sum']) for r in group)
        signal=energy/count>LOW_SIGNAL
        row=dict(zip(keys,key))
        row.update(outputs=len(group), LOW_SIGNAL_outputs=sum(r['signal_status']=='LOW_SIGNAL' for r in group),
            free_scalar_count=count, error_SSE=error, teacher_energy_sum=energy, student_energy_sum=se,
            pooled_E=error/count, pooled_B=energy/count, pooled_R=error/energy if signal else None,
            pooled_signal_status='OK' if signal else 'LOW_SIGNAL',
            pooled_residual_norm_ratio=float(np.sqrt(se/energy)) if signal else None,
            pooled_residual_cosine=float(np.clip(dot/np.sqrt(energy*se),-1,1)) if signal and se>0 else None)
        numeric=('E','B','R','residual_norm_ratio','residual_cosine',
            'pre_GEO_d_coordinate_RMS','pre_GEO_d_corner_L2_p95','pre_GEO_d_corner_L2_max',
            'pre_GEO_model_coordinate_RMS','pre_GEO_model_corner_L2_p95','pre_GEO_model_corner_L2_max',
            'pre_GEO_L_coordinate_RMS','pre_GEO_L_corner_L2_p95','pre_GEO_L_corner_L2_max',
            'teacher_target_MSE_d','student_target_MSE_d','identity_target_MSE_d')
        for field in numeric:
            vals=[float(r[field]) for r in group if r.get(field) is not None]
            row[field+'_valid_n']=len(vals)
            for name,fn in (('mean',np.mean),('median',np.median),('p05',lambda v:np.quantile(v,.05)),
                            ('p95',lambda v:np.quantile(v,.95)),('max',np.max)):
                row[field+'_'+name]=float(fn(vals)) if vals else None
        output.append(row)
    return output


def self_check():
    x=np.zeros((3,3,3),np.float32);t=np.ones_like(x);t[0]=7
    exact=residual_metrics(x,t,t,1)
    identity=residual_metrics(x,t,x,1)
    half=residual_metrics(x,t,t*.5,1)
    opposite=residual_metrics(x,t,-t,1)
    low=residual_metrics(x,x,t,1)
    assert exact['R']==0 and abs(exact['residual_cosine']-1)<1e-15
    assert identity['R']==1 and identity['residual_cosine'] is None
    assert half['R']==.25 and half['residual_norm_ratio']==.5
    assert opposite['R']==4 and opposite['residual_cosine']==-1
    assert low['signal_status']=='LOW_SIGNAL' and low['R'] is None and low['E']==1
    pooled=summarize([exact,identity,low])[0]
    assert pooled['pooled_R']==1 and pooled['LOW_SIGNAL_outputs']==1
    changed=t.copy();changed[0]=1000
    assert residual_metrics(x,t,changed,1)['E']==0
    return dict(status='PASS',checks=7,models_loaded=0,new_noise=0,definition=definition())
