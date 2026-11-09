"""Independent teacher-only decomposition of the unchanged FC/FF objectives.

No models, optimizers, geometry matching, or artifact writes are used here.
Real output-space gradients are analytic. Only run_cpu_tests uses autograd;
the caller must explicitly invoke it and account for its returned call counts.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import numpy as np
import torch


NUMERICAL_POLICY = dict(
    fp64_fixture_atol=1e-12, fp64_fixture_rtol=1e-10,
    actual_fp32_loss_atol=5e-7, actual_fp32_loss_rtol=5e-6,
    output_dot_negative_atol=1e-12, output_dot_negative_norm_product_rtol=1e-8,
    FM_ratio_norm_floor=1e-12, parameter_gram_chunk_elements=262144,
    squared_explicit_time_weights={"0.1": .81, "0.3": .49, "0.5": .25, "0.7": .09, "0.9": .01},
)
SUBSETS = ('all', 'interface', 'free_only', 'shared', 'singleton',
           'interface_shared', 'interface_singleton', 'free_only_shared', 'free_only_singleton')


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def _tokens(value):
    value = _array(value)
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] == 9:
        value = value[0]
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError('Require one sample of [N,9], [1,N,9], or [N,3,3] coordinates')
    if value.dtype not in (np.float32, np.float64) or not np.isfinite(value).all():
        raise ValueError('Finite FP32 coordinates required; FP64 allowed for mathematical fixtures')
    return value


def _record(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str((array.dtype, array.shape)).encode())
    h.update(array.tobytes())
    return dict(dtype=str(array.dtype), shape=list(array.shape), sha256=h.hexdigest())


def _residual(array):
    value = np.asarray(array, dtype=np.float64)
    return dict(elements=int(value.size), RMS=float(np.sqrt(np.mean(value**2))) if value.size else None,
                max_abs=float(np.max(np.abs(value))) if value.size else None)


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _prepare(sample, velocity, labels):
    arrays = {key: _tokens(sample[key]) for key in ('xt', 'x0', 'x1', 'u', 'context')}
    arrays['v'] = _tokens(velocity)
    shape = arrays['xt'].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError('All teacher coordinate arrays must have the same sample shape')
    n = shape[0]
    known = _array(sample['known_mask']).reshape(-1)
    valid = _array(sample['valid_mask']).reshape(-1)
    if known.shape != (n,) or valid.shape != (n,) or known.dtype != np.bool_ or valid.dtype != np.bool_:
        raise ValueError('Separate boolean valid and known masks required')
    if np.any(known & ~valid):
        raise ValueError('Known mask includes invalid faces')
    free = valid & ~known
    if not free.any():
        raise ValueError('At least one free face is required')
    tv = _array(sample['t']).reshape(-1)
    if len(tv) != 1 or not 0 < float(tv[0]) < 1:
        raise ValueError('A single teacher time strictly inside (0,1) is required')
    dtype = arrays['xt'].dtype
    if any(value.dtype != dtype for value in arrays.values()):
        raise ValueError('Explicitly cast model velocity to the FP32 state dtype before analysis')
    t = float(tv[0]); alpha = float(np.asarray(1, dtype=dtype) - np.asarray(t, dtype=dtype))
    if not _same(arrays['xt'][known], arrays['context'][known]):
        raise ValueError('Actual teacher Xt known coordinates do not match C bitwise')
    key = 'corner_source_vertex_ids' if 'corner_source_vertex_ids' in labels else 'corner_source_ids'
    ids = _array(labels[key])
    if ids.shape != (n, 3) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Trusted corner source vertex IDs [N,3] required')
    by_free = defaultdict(list); by_known = defaultdict(list)
    for face in np.flatnonzero(valid):
        target = by_known if known[face] else by_free
        for corner in range(3):
            target[int(ids[face, corner])].append(int(3 * face + corner))
    fc_old = {int(g['source_vertex_id']): g for g in labels['fc_groups']}
    ff_old = {int(g['source_vertex_id']): g for g in labels['ff_groups']}
    if len(fc_old) != len(labels['fc_groups']) or len(ff_old) != len(labels['ff_groups']):
        raise ValueError('Duplicate old relation group IDs')
    groups = []
    yflat, cflat = arrays['x1'].reshape(-1, 3), arrays['context'].reshape(-1, 3)
    for gid, indices in sorted(by_free.items()):
        idx = np.asarray(indices, dtype=np.int64); y = yflat[idx[0]]
        if any(not _same(yflat[i], y) for i in idx):
            raise ValueError('Same trusted source vertex has inconsistent free target coordinates')
        interface = gid in by_known; m = len(idx)
        if interface:
            if gid not in fc_old or sorted(map(int, fc_old[gid]['free_corner_indices'])) != indices:
                raise ValueError('Independent interface grouping differs from unchanged FC labels')
            anchor = _array(fc_old[gid]['anchor'])
            if not _same(anchor, y) or any(not _same(cflat[i], y) for i in by_known[gid]):
                raise ValueError('FC anchor, known C, and free target source coordinates disagree')
        if m >= 2:
            if gid not in ff_old or sorted(map(int, ff_old[gid]['free_corner_indices'])) != indices:
                raise ValueError('Independent shared grouping differs from unchanged FF labels')
        groups.append(dict(source_vertex_id=gid, indices=idx, m=m, interface=interface,
                           target=y.astype(np.float64), anchor=y.astype(np.float64) if interface else None))
    if set(fc_old) != {g['source_vertex_id'] for g in groups if g['interface']}:
        raise ValueError('Old FC group population differs')
    if set(ff_old) != {g['source_vertex_id'] for g in groups if g['m'] >= 2}:
        raise ValueError('Old FF group population differs')
    if labels['FC_count'] != len(fc_old) or labels['FF_count'] != len(ff_old):
        raise ValueError('Old FC/FF group denominator differs')
    endpoint = arrays['context'].copy()
    endpoint[free] = arrays['xt'][free] + np.asarray(alpha, dtype=dtype) * arrays['v'][free]
    oracle = arrays['context'].copy()
    oracle[free] = arrays['xt'][free] + np.asarray(alpha, dtype=dtype) * arrays['u'][free]
    epsilon = arrays['v'].astype(np.float64) - arrays['u'].astype(np.float64)
    return dict(arrays=arrays, known=known, valid=valid, free=free, ids=ids, groups=groups,
                t=t, alpha=alpha, endpoint=endpoint, oracle=oracle, epsilon=epsilon,
                FM_denominator=int(free.sum()) * 9, FC_denominator=len(fc_old), FF_denominator=len(ff_old))


def _decompose(values, target):
    center = values.mean(axis=0)
    position = float(np.mean((center - target)**2))
    spread = float(np.mean((values - center)**2))
    endpoint = float(np.mean((values - target)**2))
    return dict(Position=position, Spread=spread, EndpointError=endpoint,
                center=center.tolist(), identity_residual=endpoint-position-spread)


def _in_subset(g, subset):
    tags = {'all', 'interface' if g['interface'] else 'free_only', 'shared' if g['m'] >= 2 else 'singleton'}
    tags.add(('interface' if g['interface'] else 'free_only') + ('_shared' if g['m'] >= 2 else '_singleton'))
    return subset in tags


def _subset(rows, subset):
    selected = [g for g in rows if _in_subset(g, subset)]
    count = len(selected); corners = sum(g['m'] for g in selected)
    result = dict(group_count=count, corner_count=corners, status='AVAILABLE' if count else 'EMPTY_SUBSET')
    for space in ('endpoint', 'epsilon'):
        sums = {key:sum(g[space][key] for g in selected) for key in ('Position','Spread','EndpointError')}
        weighted = {key:sum(g['m']*g[space][key] for g in selected) for key in sums}
        result[space] = dict(group_sums=sums,
            group_equal={key:value/count if count else None for key,value in sums.items()},
            corner_weighted={key:value/corners if corners else None for key,value in weighted.items()})
    return result


def analyze_teacher(sample, velocity, labels, *, legacy_losses=None):
    """JSON-safe decomposition; no autograd and no call to a legacy loss.

    legacy_losses, when supplied, are values already computed by the original
    implementation on this same single state, with FM/FC/FF or L_FM/L_FC/L_FF
    keys. Group sums/counts are exposed for aggregation without reweighting K.
    """
    p = _prepare(sample, velocity, labels); a = p['arrays']; free = p['free']
    endpoint = p['endpoint'].astype(np.float64).reshape(-1, 3)
    eps = p['epsilon'].reshape(-1, 3); rows = []
    nf, nfc, nff = (p[k+'_denominator'] for k in ('FM','FC','FF'))
    for g in p['groups']:
        d = _decompose(endpoint[g['indices']], g['target'])
        e = _decompose(eps[g['indices']], np.zeros(3, dtype=np.float64))
        coefficient = 2*g['m']/(g['m']-1) if g['m'] >= 2 else None
        pair = coefficient*d['Spread'] if coefficient is not None else None
        epair = coefficient*e['Spread'] if coefficient is not None else None
        rows.append(dict(source_vertex_id=g['source_vertex_id'], m=g['m'], interface=g['interface'],
            group_type='interface' if g['interface'] else 'free_only', shared=g['m']>=2,
            free_corner_indices=g['indices'].tolist(), target_position=g['target'].tolist(),
            known_anchor=None if g['anchor'] is None else g['anchor'].tolist(), endpoint=d, epsilon=e,
            pairwise_coefficient=coefficient, pairwise_endpoint_MSE=pair, pairwise_epsilon_MSE=epair,
            FC_global_denominator_contribution=d['EndpointError']/nfc if g['interface'] else 0.,
            FF_global_denominator_contribution=pair/nff if pair is not None else 0.))
    interface = [g for g in rows if g['interface']]; shared = [g for g in rows if g['shared']]
    fc_sum = sum(g['endpoint']['EndpointError'] for g in interface)
    ff_sum = sum(g['pairwise_endpoint_MSE'] for g in shared)
    fc_pos = sum(g['endpoint']['Position'] for g in interface)/max(nfc,1)
    fc_spread = sum(g['endpoint']['Spread'] for g in interface)/max(nfc,1)
    ff_interface = sum(g['FF_global_denominator_contribution'] for g in shared if g['interface'])
    ff_free_only = sum(g['FF_global_denominator_contribution'] for g in shared if not g['interface'])
    epsilon_losses = dict(FM=float(np.mean(p['epsilon'][free]**2)),
        FC=sum(g['epsilon']['EndpointError'] for g in interface)/max(nfc,1),
        FF=sum(g['pairwise_epsilon_MSE'] for g in shared)/max(nff,1))
    losses = dict(FM=epsilon_losses['FM'],FC=fc_sum/max(nfc,1),FF=ff_sum/max(nff,1),
        FC_sum=fc_sum,FF_sum=ff_sum,FC_position_component=fc_pos,FC_spread_component=fc_spread,
        FF_interface_contribution=ff_interface,FF_free_only_contribution=ff_free_only,
        denominators=dict(FM_free_coordinates=nf,FC_interface_groups=nfc,FF_shared_groups=nff),
        normalization='Per-state group-equal FC/FF; original effective-batch training sums groups across all 8 samples before division',
        auxiliary_weighted=.1*(fc_sum/max(nfc,1)+ff_sum/max(nff,1)))
    xt64=a['xt'].astype(np.float64); y64=a['x1'].astype(np.float64); z64=a['x0'].astype(np.float64)
    actual64=p['endpoint'].astype(np.float64); v64=a['v'].astype(np.float64)
    path64=(1-p['t'])*z64+p['t']*y64
    dt=a['xt'].dtype; path_actual=(np.asarray(1,dtype=dt)-np.asarray(p['t'],dtype=dt))*a['x0']+np.asarray(p['t'],dtype=dt)*a['x1']
    endpoint64=a['context'].astype(np.float64).copy()
    endpoint64[free]=xt64[free]+(1-p['t'])*v64[free]
    residuals=dict(
        Xt_vs_recomputed_state_dtype_equation=_residual((a['xt']-path_actual)[free]),
        Xt_vs_FP64_linear_path_from_stored_arrays=_residual((xt64-path64)[free]),
        u_vs_FP64_Y_minus_Z=_residual((a['u'].astype(np.float64)-(y64-z64))[free]),
        oracle_v_equals_u_endpoint_error=_residual((p['oracle'].astype(np.float64)-y64)[free]),
        predicted_endpoint_state_dtype_vs_FP64_arithmetic=_residual((actual64-endpoint64)[free]),
        endpoint_error_vs_actual_alpha_times_epsilon=_residual((actual64-y64-p['alpha']*p['epsilon'])[free]),
        endpoint_error_vs_FP64_one_minus_t_times_epsilon=_residual((actual64-y64-(1-p['t'])*p['epsilon'])[free]),
        epsilon_state_dtype_vs_FP64_subtraction=_residual(((a['v']-a['u']).astype(np.float64)-p['epsilon'])[free]))
    scaling = {key:dict(endpoint=losses[key],direct_velocity=epsilon_losses[key],
        expected_explicit_scaled=p['alpha']**2*epsilon_losses[key],
        difference=losses[key]-p['alpha']**2*epsilon_losses[key]) for key in ('FC','FF')}
    group_scale = {key:max((abs(g['endpoint'][key]-p['alpha']**2*g['epsilon'][key]) for g in rows),default=0.)
                   for key in ('Position','Spread','EndpointError')}
    reconcile = dict(status='NOT_SUPPLIED',note='No legacy loss rerun is hidden in this analysis')
    if legacy_losses is not None:
        entries={}
        for key in ('FM','FC','FF'):
            value=legacy_losses.get(key,legacy_losses.get('L_'+key))
            if value is None:raise ValueError('Supply all three original per-state losses')
            actual=float(_array(value).reshape(())); expected=losses[key]
            tolerance=NUMERICAL_POLICY['actual_fp32_loss_atol']+NUMERICAL_POLICY['actual_fp32_loss_rtol']*abs(expected)
            entries[key]=dict(original=actual,independent=expected,difference=actual-expected,
                              tolerance=tolerance,passed=abs(actual-expected)<=tolerance)
        reconcile=dict(status='PASS' if all(x['passed'] for x in entries.values()) else 'FAIL_DEFINITION_OR_NUMERICS',values=entries)
    return dict(schema='c_relation_teacher_decomposition_v1',groups=rows,
        subsets={key:_subset(rows,key) for key in SUBSETS},losses=losses,velocity_error_losses=epsilon_losses,
        time=dict(stored_t=p['t'],actual_state_dtype_one_minus_t=p['alpha'],actual_explicit_squared_weight=p['alpha']**2,
                  FP64_explicit_squared_weight=(1-p['t'])**2,declared_grid_weights=NUMERICAL_POLICY['squared_explicit_time_weights'],
                  interpretation='Explicit coefficients are not measured gradient ratios or training-time expectations'),
        identities=dict(max_abs_endpoint_minus_Position_minus_Spread=max(abs(g['endpoint']['identity_residual']) for g in rows),
            max_abs_epsilon_endpoint_minus_Position_minus_Spread=max(abs(g['epsilon']['identity_residual']) for g in rows),
            FC_minus_Position_plus_Spread=losses['FC']-fc_pos-fc_spread,
            FF_minus_global_partition_contributions=losses['FF']-ff_interface-ff_free_only,
            group_explicit_time_scaling_max_abs_residual=group_scale),
        explicit_time_scaling=scaling,floating_point_audit=residuals,legacy_loss_reconciliation=reconcile,
        array_records={**{key:_record(value) for key,value in a.items()},'known_mask':_record(p['known']),
            'valid_mask':_record(p['valid']),'corner_source_vertex_ids':_record(p['ids']),
            'Yhat_actual_state_dtype':_record(p['endpoint']),'Yhat_FP64_arithmetic':_record(endpoint64)},
        coordinate_units='Original model coordinates; MSE_xyz divides squared 3-vector norm by 3; no bbox normalization',
        decomposition_reduction_dtype='float64 on decoded actual state-dtype endpoint',
        source_mapping_status='PASS_EXACT_GROUP_TARGET_AND_ANCHOR_CHECK',model_calls=0,autograd_calls=0)


def gradient_norm_dot_statistics(fm, fc, ff, *, names=None, chunk_elements=262144):
    """Accumulate six Gram entries by tensor/chunk, never concatenate gradients.

    Inputs are aligned parameter-gradient sequences (None means zero). GPU
    tensors stay on their device; at most three chunks are cast to FP64 at once.
    Only six final scalar entries are copied to the host. Names describe the
    exact supplied subset; a caller must not relabel a subset as the full model.
    """
    lists=[list(fm),list(fc),list(ff)]
    if not len(lists[0])==len(lists[1])==len(lists[2]):raise ValueError('Aligned gradient sequences required')
    if names is not None and len(names)!=len(lists[0]):raise ValueError('Names must align with gradient tensors')
    if not isinstance(chunk_elements,int) or chunk_elements<=0:raise ValueError('Positive chunk size required')
    total=None;device=None;elements=0;chunks=0;none=[0,0,0]
    with torch.no_grad():
        for triple in zip(*lists):
            nonempty=[x for x in triple if x is not None]
            if not nonempty:
                none=[x+1 for x in none];continue
            converted=[None if x is None else (x.detach() if isinstance(x,torch.Tensor) else torch.as_tensor(x)) for x in triple]
            ref=next(x for x in converted if x is not None)
            if any(x is not None and (x.shape!=ref.shape or x.device!=ref.device) for x in converted):raise ValueError('Gradient shape/device mismatch')
            if total is None:device=ref.device;total=torch.zeros(6,dtype=torch.float64,device=device)
            if ref.device!=device:raise ValueError('One statistics call requires one device')
            flats=[None if x is None else x.reshape(-1) for x in converted];elements+=ref.numel()
            for i,x in enumerate(flats):none[i]+=x is None
            for start in range(0,ref.numel(),chunk_elements):
                end=min(start+chunk_elements,ref.numel());chunks+=1
                values=[None if x is None else x[start:end].to(dtype=torch.float64) for x in flats]
                def dot(i,j):
                    return total.new_zeros(()) if values[i] is None or values[j] is None else (values[i]*values[j]).sum()
                total.add_(torch.stack([dot(0,0),dot(1,1),dot(2,2),dot(0,1),dot(0,2),dot(1,2)]))
        gram=[0.]*6 if total is None else total.cpu().tolist()
    if not all(math.isfinite(value) for value in gram):raise ValueError('Nonfinite gradient Gram entries')
    aa,bb,cc,ab,ac,bc=gram
    norms=dict(FM=math.sqrt(max(aa,0)),FC=math.sqrt(max(bb,0)),FF=math.sqrt(max(cc,0)))
    auxsq=.01*(bb+cc+2*bc);auxnorm=math.sqrt(max(auxsq,0.))
    def pair(dot,n0,n1):
        return dict(dot=dot,cosine=None if n0==0 or n1==0 else dot/(n0*n1),
                    status='UNDEFINED_ZERO_NORM' if n0==0 or n1==0 else 'DEFINED',norm_product=n0*n1)
    directions=dict(FM_vs_FC=pair(ab,norms['FM'],norms['FC']),FM_vs_FF=pair(ac,norms['FM'],norms['FF']),
                    FC_vs_FF=pair(bc,norms['FC'],norms['FF']),FM_vs_weighted_aux=pair(.1*(ab+ac),norms['FM'],auxnorm))
    near=norms['FM']<=NUMERICAL_POLICY['FM_ratio_norm_floor']
    return dict(unweighted_norms=norms,weighted_norms=dict(FM=norms['FM'],FC=.1*norms['FC'],FF=.1*norms['FF'],combined_aux=auxnorm),
        unweighted_gram=dict(FM_FM=aa,FC_FC=bb,FF_FF=cc,FM_FC=ab,FM_FF=ac,FC_FF=bc),
        directions=directions,weighted_dots=dict(FM_vs_0p1FC=.1*ab,FM_vs_0p1FF=.1*ac,weighted_FC_vs_FF=.01*bc),
        ratios_to_FM=dict(FC=None if near else .1*norms['FC']/norms['FM'],FF=None if near else .1*norms['FF']/norms['FM'],
                          combined_aux=None if near else auxnorm/norms['FM']),
        ratio_status='UNDEFINED_FM_NEAR_ZERO' if near else 'DEFINED',FM_near_zero=near,
        FM_ratio_norm_floor=NUMERICAL_POLICY['FM_ratio_norm_floor'],combined_aux_squared_norm_unclamped=auxsq,
        coverage=dict(tensors=len(lists[0]),elements_with_any_gradient=elements,None_counts=dict(zip(('FM','FC','FF'),none)),
                      names=None if names is None else list(names),device=str(device),chunk_elements=chunk_elements,chunks=chunks),
        accumulation_dtype='float64',autograd_calls=0)


def _analytic_output_gradients(p, ideal=False):
    shape=p['arrays']['v'].shape; gradients={key:np.zeros(shape,dtype=np.float64) for key in ('FM','FC','FF')}
    gradients['FM'][p['free']]=2*p['epsilon'][p['free']]/p['FM_denominator']
    endpoint=p['endpoint'].astype(np.float64).reshape(-1,3)
    epsilon=p['epsilon'].reshape(-1,3);alpha=p['alpha']
    fc=gradients['FC'].reshape(-1,3);ff=gradients['FF'].reshape(-1,3)
    for g in p['groups']:
        idx=g['indices'];m=g['m']
        error=alpha*epsilon[idx] if ideal else endpoint[idx]-g['target']
        if g['interface']:fc[idx]=2*alpha*error/(3*m*p['FC_denominator'])
        if m>=2:ff[idx]=4*alpha*(error-error.mean(axis=0))/(3*(m-1)*p['FF_denominator'])
    return gradients


def output_gradient_diagnostics(sample, velocity, labels):
    """Analytic velocity-space gradients, distinct from parameter gradients."""
    p=_prepare(sample,velocity,labels);result={}
    for ideal in (False,True):
        gradients=_analytic_output_gradients(p,ideal)
        stats=gradient_norm_dot_statistics(*[[gradients[k][p['free']]] for k in ('FM','FC','FF')],names=['free_velocity_coordinates'])
        psd={}
        for key in ('FM_vs_FC','FM_vs_FF','FM_vs_weighted_aux'):
            d=stats['directions'][key]
            tolerance=NUMERICAL_POLICY['output_dot_negative_atol']+NUMERICAL_POLICY['output_dot_negative_norm_product_rtol']*d['norm_product']
            psd[key]=dict(dot=d['dot'],negative_tolerance=tolerance,significant_negative=d['dot'] < -tolerance)
        stats['PSD_output_error_check']=psd
        result['ideal_linear_error_reference' if ideal else 'actual_endpoint_analytic']=stats
    result.update(space='Output free velocity, not model parameters',analytic_evaluations=1,autograd_calls=0,model_calls=0,
        formulas=dict(FM='2*epsilon/(3*free_corner_count)',FC='2*(1-t)*(Yhat_i-c_g)/(3*m_g*FC_group_count)',
                      FF='4*(1-t)*(Yhat_i-Ybar_g)/(3*(m_g-1)*FF_group_count)'),
        numerical_note='Actual uses decoded FP32 Yhat and actual FP32 (1-t); ideal uses (1-t)*epsilon. Both reductions are FP64; neither is a new model backward.')
    return result


def run_cpu_tests():
    """Explicit fixed fixtures; called once by the experiment owner, not on import."""
    from .t1_interface_relations import build_relation_labels,prepare_relations,relation_sums,endpoint_from_velocity
    counts=dict(model_forwards=0,model_backwards=0,optimizer_updates=0,CPU_autograd_grad_calls=0,finite_difference_loss_evaluations=0,legacy_relation_loss_evaluations=0)
    def grad(*args,**kwargs):
        counts['CPU_autograd_grad_calls']+=1
        return torch.autograd.grad(*args,**kwargs)
    ids=np.asarray([[0,1,2],[0,3,4],[0,3,5],[1,3,6]],dtype=np.int64)
    known=np.asarray([True,False,False,False]);valid=np.ones(4,dtype=bool)
    vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.2,.3,.4],[-.2,.1,.7],[.6,.4,.1],[.3,-.5,.2]],dtype=np.float32)
    Y=vertices[ids];context=np.zeros_like(Y);context[known]=Y[known]
    x1=Y.copy();x1[known]=0
    Z=np.arange(36,dtype=np.float32).reshape(4,3,3)/31-.5;Z[known]=0
    t=np.float32(.5);xt=(1-t)*Z+t*x1;xt[known]=context[known]
    sample=dict(x0=Z.reshape(4,9),x1=x1.reshape(4,9),u=(x1-Z).reshape(4,9),xt=xt.reshape(4,9),t=t,
                context=context.reshape(4,9),known_mask=known,valid_mask=valid)
    labels=build_relation_labels(ids,known,context);labels['corner_source_vertex_ids']=ids
    # FP64 ideal input fixtures retain exactly corresponding target coordinates.
    double={k:np.asarray(v,dtype=np.float64) if k in ('x0','x1','u','xt','t','context') else v.copy() for k,v in sample.items()}
    double['u']=double['x1']-double['x0'];double['xt']=.5*double['x0']+.5*double['x1'];double['xt'][known]=double['context'][known]
    dl=build_relation_labels(ids,known,context);dl['corner_source_vertex_ids']=ids
    for g in dl['fc_groups']:g['anchor']=g['anchor'].astype(np.float64)
    velocity=double['u']+np.sin(np.arange(36)).reshape(4,9)*.15
    analyzed=analyze_teacher(double,velocity,dl)
    tol=NUMERICAL_POLICY['fp64_fixture_atol']
    assert analyzed['identities']['max_abs_endpoint_minus_Position_minus_Spread']<tol
    assert abs(analyzed['identities']['FF_minus_global_partition_contributions'])<tol
    assert max(analyzed['identities']['group_explicit_time_scaling_max_abs_residual'].values())<tol
    oracle=analyze_teacher(double,double['u'],dl)
    assert oracle['losses']['FC']<tol and oracle['losses']['FF']<tol
    # Enumerate unordered pairs independently from the variance identity.
    p=_prepare(double,velocity,dl);end=p['endpoint'].reshape(-1,3)
    explicit=[]
    for g in p['groups']:
        if g['m']>=2:
            x=end[g['indices']];pairs=[np.mean((x[i]-x[j])**2) for i in range(len(x)) for j in range(i+1,len(x))]
            explicit.append(float(np.mean(pairs)))
    assert abs(np.mean(explicit)-analyzed['losses']['FF'])<tol
    # Analytic output gradients versus independent small FP64 autograd.
    v=torch.tensor(velocity,dtype=torch.float64,requires_grad=True)
    e=torch.tensor(double['xt'])+.5*v;eflat=e.reshape(-1,3);uf=torch.tensor(double['u'])
    fm=((v-uf)[torch.from_numpy(~known)]**2).mean();fc=[];ff=[]
    for g in p['groups']:
        x=eflat[g['indices']]
        if g['interface']:fc.append(((x-torch.tensor(g['target']))**2).mean())
        if g['m']>=2:ff.append(torch.stack([((x[i]-x[j])**2).mean() for i in range(len(x)) for j in range(i+1,len(x))]).mean())
    losses=(fm,torch.stack(fc).mean(),torch.stack(ff).mean());analytic=_analytic_output_gradients(p)
    for index,(key,loss) in enumerate(zip(('FM','FC','FF'),losses)):
        got=grad(loss,v,retain_graph=index<2)[0].numpy().reshape(4,3,3)
        assert np.allclose(got,analytic[key],atol=tol,rtol=NUMERICAL_POLICY['fp64_fixture_rtol'])
    # Directional finite differences, no autodiff reuse.
    direction=np.cos(np.arange(36)).reshape(4,9);h=1e-5
    plus=analyze_teacher(double,velocity+h*direction,dl)['losses'];minus=analyze_teacher(double,velocity-h*direction,dl)['losses']
    counts['finite_difference_loss_evaluations']+=2
    for key in ('FM','FC','FF'):
        numerical=(plus[key]-minus[key])/(2*h);expected=float(np.sum(analytic[key]*direction.reshape(4,3,3)))
        assert np.isclose(numerical,expected,atol=tol,rtol=NUMERICAL_POLICY['fp64_fixture_rtol'])
    # Actual unchanged FP32 endpoint/loss code, with detached C and anchors.
    pack=prepare_relations(labels,'cpu');pack['fc_anchors'].requires_grad_(True)
    c=torch.tensor(sample['context'][None],requires_grad=True);vf=torch.tensor(velocity.astype(np.float32)[None],requires_grad=True)
    real=endpoint_from_velocity(torch.tensor(sample['xt'][None]),vf,torch.tensor([t]),torch.tensor(known[None]),c)
    terms=relation_sums(real,pack);counts['legacy_relation_loss_evaluations']+=1
    oracle_actual=endpoint_from_velocity(torch.tensor(sample['xt'][None]),torch.tensor(sample['u'][None]),torch.tensor([t]),torch.tensor(known[None]),c)
    oracle_terms=relation_sums(oracle_actual,pack);counts['legacy_relation_loss_evaluations']+=1
    assert float(oracle_terms['FC_sum'].detach())<1e-12 and float(oracle_terms['FF_sum'].detach())<1e-12
    detached=grad(terms['FC_sum']+terms['FF_sum'],(vf,c,pack['fc_anchors']),allow_unused=True)
    assert detached[1] is None and detached[2] is None and torch.count_nonzero(detached[0][:,known])==0
    original=dict(FC=float(terms['FC_sum'].detach())/terms['FC_count'],FF=float(terms['FF_sum'].detach())/terms['FF_count'],
                  FM=float(((vf.detach()[0,~known]-torch.tensor(sample['u'][~known]))**2).mean()))
    fp32=analyze_teacher(sample,vf.detach(),labels,legacy_losses=original)
    assert fp32['legacy_loss_reconciliation']['status']=='PASS'
    # Face/corner permutations carry trusted identities and all arrays together.
    order=np.asarray([2,0,3,1]);corners=np.asarray([[2,0,1],[1,2,0],[2,1,0],[0,2,1]])
    def perm(a):return np.take_along_axis(a.reshape(4,3,3)[order],corners[...,None],axis=1).reshape(4,9)
    moved={k:perm(val) if k in ('x0','x1','u','xt','context') else val for k,val in double.items()}
    moved['known_mask']=known[order];moved['valid_mask']=valid[order]
    ids2=np.take_along_axis(ids[order],corners,axis=1);ct=perm(context).reshape(4,3,3).astype(np.float32)
    labels2=build_relation_labels(ids2,known[order],ct);labels2['corner_source_vertex_ids']=ids2
    for g in labels2['fc_groups']:g['anchor']=g['anchor'].astype(np.float64)
    moved_result=analyze_teacher(moved,perm(velocity),labels2)
    assert all(np.isclose(moved_result['losses'][key],analyzed['losses'][key],atol=tol,rtol=1e-10) for key in ('FM','FC','FF'))
    stats=output_gradient_diagnostics(double,velocity,dl)
    assert not any(x['significant_negative'] for x in stats['ideal_linear_error_reference']['PSD_output_error_check'].values())
    zeros=gradient_norm_dot_statistics([np.zeros(3)],[np.ones(3)],[None],names=['fixture'],chunk_elements=2)
    assert zeros['ratio_status']=='UNDEFINED_FM_NEAR_ZERO' and zeros['directions']['FM_vs_FC']['cosine'] is None
    probe=[np.arange(7,dtype=np.float64)/7,np.sin(np.arange(7)),np.cos(np.arange(7))]
    gram=gradient_norm_dot_statistics(*[[x] for x in probe],names=['nonzero_fixture'],chunk_elements=2)
    for key,i,j in [('FM_FM',0,0),('FC_FC',1,1),('FF_FF',2,2),('FM_FC',0,1),('FM_FF',0,2),('FC_FF',1,2)]:
        assert np.isclose(gram['unweighted_gram'][key],np.dot(probe[i],probe[j]),atol=tol,rtol=1e-10)
    assert np.isclose(gram['weighted_norms']['combined_aux'],np.linalg.norm(.1*probe[1]+.1*probe[2]),atol=tol,rtol=1e-10)
    return dict(status='PASS',fixture_families=9,numerical_policy=NUMERICAL_POLICY,counts=counts,
        fixtures=['group_decomposition_and_partition','oracle_endpoint','explicit_unordered_pairs','analytic_output_gradients',
                  'directional_finite_difference','legacy_FP32_losses_and_detached_anchors','face_corner_equivariance',
                  'output_PSD_quadratic','chunked_gradient_zero_norm_policy'])
