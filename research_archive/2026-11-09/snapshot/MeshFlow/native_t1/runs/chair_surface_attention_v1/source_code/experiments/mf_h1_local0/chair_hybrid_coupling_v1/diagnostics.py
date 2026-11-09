"""Finite-edit symmetric response decomposition on frozen Euler trajectories.

I/P are algebraic symmetric allocations, not infinitesimal Jacobians or unique
causal attributions. Model loading, GPU locking and durable call limits belong
to the root worker; every additional real model call goes through its Budget.
"""
from __future__ import annotations

import numpy as np
from .sampling import validate_bank, array_digest

DELTA = .1
DT = 1 / 50


def rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    aa, bb = float(a @ a), float(b @ b)
    if aa == 0.0 or bb == 0.0:
        return None, 'UNDEFINED'
    return float(np.clip(float(a @ b) / np.sqrt(aa * bb), -1, 1)), 'DEFINED'


def decompose(vpp, vpm, vmp, vmm):
    vpp, vpm, vmp, vmm = (np.asarray(v, dtype=np.float64) for v in (vpp, vpm, vmp, vmm))
    if len({v.shape for v in (vpp, vpm, vmp, vmm)}) != 1:
        raise ValueError('Four velocities must have identical free slots')
    if any(not np.isfinite(v).all() for v in (vpp, vpm, vmp, vmm)):
        raise FloatingPointError('Nonfinite response velocity')
    return ((vpp-vpm)+(vmp-vmm))/(4*DELTA), ((vpp-vmp)+(vpm-vmm))/(4*DELTA)


def select_diagnostic_curves(panels):
    """Preselect 2 lexical train + 2 lexical validation UIDs, noise0/1."""
    rows = panels['B'] if isinstance(panels, dict) else panels
    selected = []
    for split in ('train', 'validation'):
        parents = sorted({r['uid'] for r in rows if r['split'] == split})
        if len(parents) != 4:
            raise ValueError('Original B must contain exactly four parents in each split')
        for uid in parents[:2]:
            for noise in (0, 1):
                rr = [r for r in rows if r['uid'] == uid and r['noise_index'] == noise]
                by_alpha = {float(r['alpha']): r for r in rr}
                if len(rr) != 3 or set(by_alpha) != {.9, 1., 1.1}:
                    raise ValueError('Original B curve must retain three alpha requests')
                selected.append(dict(uid=uid, split=split, noise_index=noise,
                                     minus_record=by_alpha[.9], base_record=by_alpha[1.],
                                     plus_record=by_alpha[1.1],
                                     fp32=(uid == parents[0] and noise == 0)))
    return selected


def validate_pair(pair):
    minus, plus = pair['minus_record'], pair['plus_record']
    for key in ('uid', 'split', 'noise_index', 'N', 'K', 'patch_id'):
        if key in minus and key in plus and minus[key] != plus[key]:
            raise ValueError('Cross-alpha registered identity differs: '+key)
    if float(minus['alpha']) != .9 or float(plus['alpha']) != 1.1:
        raise ValueError('Diagnostic endpoints must be the registered .9/1.1')
    zm, cm, km = validate_bank(pair['minus_bank'])
    zp, cp, kp = validate_bank(pair['plus_bank'])
    if zm.tobytes() != zp.tobytes() or not np.array_equal(km, kp):
        raise ValueError('Cross-alpha Gaussian/masks differ')
    for key in ('C_source_face_ids', 'known_source_face_ids', 'source_face_ids', 'free_source_ids', 'known_ids', 'gaussian'):
        bm, bp = pair['minus_bank'], pair['plus_bank']
        if key in bm or key in bp:
            if key not in bm or key not in bp or not np.array_equal(bm[key], bp[key]):
                raise ValueError('Cross-alpha source face identity differs: '+key)
    N, K = zm.shape[1], len(cm)
    traces = []
    for label, c in (('minus', cm), ('plus', cp)):
        result = pair[label+'_result']
        states, velocities = np.asarray(result['states']), np.asarray(result['velocities'])
        if states.shape != (51, N, 9) or velocities.shape != (50, N, 9):
            raise ValueError('Need 51 actual accepted states and 50 actual velocities')
        if states.dtype != np.float32 or velocities.dtype != np.float32:
            raise ValueError('Stored states/float-cast actual velocities must be FP32')
        if not np.isfinite(states).all() or not np.isfinite(velocities).all():
            raise FloatingPointError('Nonfinite frozen trajectory')
        if any(s[:K].tobytes() != c.tobytes() for s in states):
            raise ValueError('Frozen trajectory changed C')
        if states[0, K:].tobytes() != zm[0, K:].tobytes():
            raise ValueError('Frozen trajectory starts from different noise slots')
        if result.get('raw') is not None and states[-1].tobytes() != np.asarray(result['raw']).tobytes():
            raise ValueError('Trajectory endpoint does not match formal output')
        traces.append((states, velocities))
    return zm, cm, cp, km, traces


def _forward(model, state, condition, step, budget, context, *, fp32=False):
    """One actual forward; known coordinates are replaced, and Geo recomputes."""
    import torch
    from native_t1.model import native_execution

    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Diagnostic real forward requires the caller-held GPU lock')
    x_np = np.asarray(state, dtype=np.float32).copy()
    C = np.asarray(condition, dtype=np.float32).reshape(-1, 9)
    N, K = len(x_np), len(C)
    x_np[:K] = C
    x = torch.from_numpy(x_np)[None].to(device)
    t = torch.full((1,), step/50, dtype=torch.float32, device=device)
    y = torch.tensor([N], dtype=torch.long, device=device)
    known = torch.arange(N, device=device)[None] < K
    valid = torch.ones_like(known)
    expected_C = x[0, :K].clone()
    dtype_records, geometry_records = [], []

    def geo_hook(module, args):
        if not torch.equal(args[0][0, :K], expected_C):
            raise RuntimeError('Cross-condition Geo did not receive substituted known coordinates')
        geometry_records.append(dict(condition_sha256=array_digest(C), known_count=K,
                                     dtype=str(args[0].dtype)))

    def dtype_hook(name):
        def hook(module, args, returned):
            dtype_records.append(dict(module=name, input_dtype=str(args[0].dtype),
                                      output_dtype=str(returned.dtype),
                                      parameter_dtype=str(module.weight.dtype)))
            if fp32 and (args[0].dtype != torch.float32 or returned.dtype != torch.float32 or module.weight.dtype != torch.float32):
                raise RuntimeError('FP32 diagnostic encountered non-FP32 Linear arithmetic')
        return hook

    if getattr(model, 'context_encoder', None) is None:
        raise ValueError('Native+Geo diagnostic requires actual Geo encoder')
    handles = [model.context_encoder.register_forward_pre_hook(geo_hook)]
    # FP32 verifies actual interior arithmetic, not merely the final output cast.
    # BF16 verifies the actual final projection dtype with no extra model calls.
    modules = [(n,m) for n,m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    for name, module in modules:
        if fp32 or module is model.backbone.final_layer.linear:
            handles.append(module.register_forward_hook(dtype_hook(name)))
    try:
        with torch.no_grad(), native_execution(model, coordinates=(x,t), autocast=not fp32) as facts:
            if torch.is_autocast_enabled('cuda') == fp32:
                raise RuntimeError('Actual autocast does not match requested diagnostic precision')
            returned = budget.call('diagnostic_fp32' if fp32 else 'diagnostic_bf16',
                                   model, x, t, y, valid, known, context=context)
            expected_dtype = torch.float32 if fp32 else torch.bfloat16
            if returned.dtype != expected_dtype:
                raise RuntimeError('Actual model return precision mismatch')
            velocity = returned.float()[0].cpu().numpy().copy()
        if len(geometry_records) != 1 or not dtype_records:
            raise RuntimeError('Missing actual Geo/precision observation')
        if not np.isfinite(velocity).all() or velocity.shape != x_np.shape:
            raise FloatingPointError('Nonfinite or malformed diagnostic velocity')
        if not np.array_equal(x[0].cpu().numpy(), x_np):
            raise RuntimeError('Diagnostic forward mutated frozen input state')
        return velocity, dict(runtime=facts, actual_output_dtype=str(returned.dtype),
                              linear_dtype_observations=dtype_records,
                              geometry_observations=geometry_records,
                              all_observed_linears_fp32=(all(r['input_dtype']=='torch.float32' and
                                   r['output_dtype']=='torch.float32' and r['parameter_dtype']=='torch.float32'
                                   for r in dtype_records) if fp32 else None),
                              input_state_sha256=array_digest(x_np))
    finally:
        for handle in handles:
            handle.remove()


def diagnose_pair(model, pair, budget, context=None):
    z, cm, cp, known, traces = validate_pair(pair)
    (sm, vm), (sp, vp) = traces
    K = len(cm)
    identity = {key: pair['minus_record'][key] for key in ('uid','split','noise_index')}
    identity.update(context or {})
    S = (sp[:,K:].astype(np.float64)-sm[:,K:].astype(np.float64))/(2*DELTA)
    if np.count_nonzero(S[0]):
        raise ValueError('Frozen paired initial free response must be exactly zero')
    Is, Ps, cross_pm, cross_mp, rows, fp32_rows, fp32_vectors = [], [], [], [], [], [], {}
    call_audits = []
    ci = np.zeros_like(S[0]); cpp = np.zeros_like(S[0])
    for step in range(50):
        base_context = dict(identity, euler_step=step)
        vpm, audit_pm = _forward(model, sp[step], cm, step, budget, dict(base_context, combination='pm'))
        vmp, audit_mp = _forward(model, sm[step], cp, step, budget, dict(base_context, combination='mp'))
        vpp, vmm = vp[step,K:], vm[step,K:]
        free_pm, free_mp = vpm[K:], vmp[K:]
        I, P = decompose(vpp, free_pm, free_mp, vmm)
        ci += DT*I; cpp += DT*P
        residual = S[step+1]-S[step]-DT*(I+P)
        algebra = I+P-(vpp.astype(np.float64)-vmm.astype(np.float64))/(2*DELTA)
        closure = S[step+1]-S[0]-ci-cpp
        cos, status = cosine(I,P)
        cisc, cisc_status = cosine(ci,S[step+1])
        cppsc, cppsc_status = cosine(cpp,S[step+1])
        rows.append(dict(identity, step=step, t=step/50, valid_free_coordinates=I.size,
                         I_RMS=rms(I), P_RMS=rms(P), S_RMS=rms(S[step]), S_next_RMS=rms(S[step+1]),
                         cosine_IP=cos, cosine_IP_status=status, opposing=(cos < 0 if cos is not None else None),
                         euler_residual_RMS=rms(residual), euler_residual_maxabs=float(np.max(np.abs(residual))),
                         algebra_identity_maxabs=float(np.max(np.abs(algebra))),
                         cumulative_I_RMS=rms(ci), cumulative_P_RMS=rms(cpp),
                         cumulative_closure_RMS=rms(closure), cumulative_closure_maxabs=float(np.max(np.abs(closure))),
                         cumulative_I_cosine_S=cisc, cumulative_I_cosine_S_status=cisc_status,
                         cumulative_P_cosine_S=cppsc, cumulative_P_cosine_S_status=cppsc_status))
        Is.append(I); Ps.append(P); cross_pm.append(free_pm); cross_mp.append(free_mp)
        call_audits.extend([dict(step=step, combination='pm', **audit_pm), dict(step=step, combination='mp', **audit_mp)])
        if pair.get('fp32',False) and step in (0,25,49):
            fp32_v = {}
            bf16_v = dict(pp=vpp, pm=free_pm, mp=free_mp, mm=vmm)
            for combination, state, C in (('pp',sp[step],cp),('pm',sp[step],cm),
                                           ('mp',sm[step],cp),('mm',sm[step],cm)):
                velocity, audit = _forward(model,state,C,step,budget,
                                           dict(base_context,combination=combination),fp32=True)
                fp32_v[combination] = velocity[K:]
                fp32_vectors[f'k{step}_{combination}'] = velocity[K:]
                reference = bf16_v[combination]
                fp32_rows.append(dict(identity,step=step,t=step/50,combination=combination,
                                       kind='velocity_precision',bf16_RMS=rms(reference),fp32_RMS=rms(velocity[K:]),
                                       difference_RMS=rms(velocity[K:].astype(np.float64)-reference),
                                       cosine=cosine(reference,velocity[K:])[0],
                                       cosine_status=cosine(reference,velocity[K:])[1],audit=audit))
            fi, fp = decompose(*(fp32_v[k] for k in ('pp','pm','mp','mm')))
            fcos, fstatus = cosine(fi,fp)
            fp32_rows.append(dict(identity,step=step,t=step/50,combination='symmetric_decomposition',
                                   kind='response_precision',bf16_I_RMS=rms(I),fp32_I_RMS=rms(fi),
                                   bf16_P_RMS=rms(P),fp32_P_RMS=rms(fp),
                                   I_difference_RMS=rms(fi-I),P_difference_RMS=rms(fp-P),
                                   bf16_cosine_IP=cos,bf16_cosine_IP_status=status,
                                   fp32_cosine_IP=fcos,fp32_cosine_IP_status=fstatus,
                                   opposing_sign_agrees=(bool(cos < 0)==bool(fcos < 0)
                                                         if cos is not None and fcos is not None else None)))
    defined = [r for r in rows if r['cosine_IP_status']=='DEFINED']
    sum_cos, sum_status = cosine(ci,cpp)
    summary = dict(identity,steps=50,extra_bf16_forwards=100,
                   extra_fp32_forwards=12 if pair.get('fp32',False) else 0,
                   reused_actual_sampling_velocities=100,S0_RMS=rms(S[0]),S50_RMS=rms(S[-1]),
                   integrated_I_RMS=rms(ci),integrated_P_RMS=rms(cpp),
                   integrated_IP_cosine=sum_cos,integrated_IP_cosine_status=sum_status,
                   closure_RMS=rms(S[-1]-S[0]-ci-cpp),
                   closure_maxabs=float(np.max(np.abs(S[-1]-S[0]-ci-cpp))),
                   defined_cosine_steps=len(defined),undefined_cosine_steps=50-len(defined),
                   opposing_steps=sum(r['opposing'] for r in defined),
                   opposing_fraction=(sum(r['opposing'] for r in defined)/len(defined) if defined else None),
                   max_step_euler_residual_RMS=max(r['euler_residual_RMS'] for r in rows),
                   minus_case_id=pair['minus_record']['case_id'],plus_case_id=pair['plus_record']['case_id'],
                   inference='Finite-edit symmetric algebraic allocation; not a unique causal or infinitesimal Jacobian decomposition')
    vectors = dict(S=S,I=np.stack(Is),P=np.stack(Ps),v_pm=np.stack(cross_pm),v_mp=np.stack(cross_mp),
                   integrated_I=ci,integrated_P=cpp,**fp32_vectors)
    return dict(rows=rows,summary=summary,fp32=fp32_rows,vectors=vectors,call_audits=call_audits)


def run_diagnostics(model, paired_trajectories, budget, context=None):
    """Run exactly 8 frozen pairs; caller persists each returned artifact."""
    if len(paired_trajectories) != 8 or sum(bool(p.get('fp32')) for p in paired_trajectories) != 2:
        raise ValueError('Each checkpoint needs exactly eight pairs and two FP32-selected pairs')
    keys = {(p['minus_record']['uid'],p['minus_record']['noise_index']) for p in paired_trajectories}
    if len(keys) != 8:
        raise ValueError('Duplicate diagnostic pair')
    for split in ('train','validation'):
        rr = [p for p in paired_trajectories if p['minus_record']['split']==split]
        parents = sorted({p['minus_record']['uid'] for p in rr})
        if len(rr)!=4 or len(parents)!=2:
            raise ValueError('Need two parents and two original noises per split')
        for p in rr:
            r=p['minus_record']
            if bool(p.get('fp32')) != (r['uid']==parents[0] and r['noise_index']==0):
                raise ValueError('FP32 selection differs from frozen first-parent/noise0 rule')
    results=[]
    for pair in paired_trajectories:
        results.append(diagnose_pair(model,pair,budget,context))
    return dict(rows=[r for p in results for r in p['rows']],
                summaries=[p['summary'] for p in results],
                fp32=[r for p in results for r in p['fp32']],
                pairs=results,
                planned_extra_bf16_forwards=800,planned_extra_fp32_forwards=24)


def cpu_checks():
    """Algebra-only tests: no torch import, real model, CUDA, or forward."""
    rng=np.random.default_rng(20261004)
    vpp,vpm,vmp,vmm=rng.normal(size=(4,7,9))
    I,P=decompose(vpp,vpm,vmp,vmm)
    error=float(np.max(np.abs(I+P-(vpp-vmm)/(2*DELTA))))
    assert error < 1e-13
    assert cosine(np.zeros((2,3)),np.ones((2,3))) == (None,'UNDEFINED')
    assert cosine(np.ones(3),-np.ones(3))[0] == -1.0
    # Field v=A*F+B*C: symmetric decomposition recovers each finite term.
    fm,fp,cm,cp=rng.normal(size=(4,7,9)); A,B=.25,1.75
    ii,pp=decompose(A*fp+B*cp,A*fp+B*cm,A*fm+B*cp,A*fm+B*cm)
    assert np.allclose(ii,B*(cp-cm)/(2*DELTA),rtol=0,atol=1e-14)
    assert np.allclose(pp,A*(fp-fm)/(2*DELTA),rtol=0,atol=1e-14)
    S=(fp-fm)/(2*DELTA)
    sn=((fp+DT*(A*fp+B*cp))-(fm+DT*(A*fm+B*cm)))/(2*DELTA)
    closure=float(np.max(np.abs(sn-S-DT*(ii+pp))))
    assert closure < 1e-13
    # Same state/condition no-response yields exactly zero terms and undefined cosine.
    zi,zp=decompose(vpp,vpp,vpp,vpp)
    assert np.count_nonzero(zi)==0 and np.count_nonzero(zp)==0
    return dict(status='PASS',algebra_maxabs=error,linear_field_euler_closure_maxabs=closure,
                zero_norm_cosine='UNDEFINED',real_model_forwards=0)
