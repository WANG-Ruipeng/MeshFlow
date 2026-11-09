"""Vectorized current-coordinate GPU energies; no model, GT or CPU geometry.

Production uses FP32 CUDA tensors. CPU FP64 is supported solely by the fixed
mathematical probes. Hard closest-feature/minimum switches are piecewise
differentiable, not a smooth collision barrier. The caller owns the common
per-vertex displacement clipping and the original FP32 Euler operation order.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import numpy as np
import torch


NUMERICAL_POLICY = dict(q_cut=10., area_floor=1e-10, q_floor=1e-12,
    actual_near_degenerate_area_threshold=1e-12, distance_denominator_floor=1e-30,
    sigma=.005, finite_difference_h=1e-6, fp64_gradient_atol=2e-7,
    fp64_gradient_rtol=2e-4, gpu_energy_atol=2e-6, gpu_energy_rtol=2e-5,
    gpu_gradient_atol=2e-5, gpu_gradient_rtol=2e-3,
    GPU_permutation_energy_atol=2e-6, GPU_permutation_energy_rtol=2e-5,
    GPU_permutation_gradient_atol=2e-5, GPU_permutation_gradient_rtol=2e-3,
    coordinate_scope='Y=X_free/L; Cnorm=C/L; original C is never overwritten',
    smoothness='Piecewise differentiable min and nearest-feature switches; no collision barrier',
    near_degeneracy_is_not_area_floor=True)


def initial_energy_ledger():
    return dict(energy_forward_attempts=0, energy_forward_calls=0,
        energy_A_evaluations=0, energy_R_evaluations=0,
        vertex_gaussian_evaluations=0, geometry_backward_attempts=0,
        geometry_backward_calls=0, CPU_energy_forward_calls=0,
        CUDA_energy_forward_calls=0, CPU_geometry_backward_calls=0,
        CUDA_geometry_backward_calls=0, finite_difference_energy_evaluations=0,
        offline_primary_pair_tests=0, offline_independent_pair_tests=0,
        fixture_runs=0, model_forwards=0, model_backwards=0, optimizer_updates=0)


def _inc(ledger, key, amount=1):
    if ledger is not None:
        ledger[key] = ledger.get(key, 0) + amount


def _validate(Y, Cnorm):
    if not isinstance(Y, torch.Tensor) or not isinstance(Cnorm, torch.Tensor):
        raise TypeError('Energies require tensors; the sampling API performs no CPU conversions')
    if Y.ndim != 3 or Y.shape[1:] != (3, 3) or len(Y) == 0:
        raise ValueError('Y must be nonempty [N_free,3,3]')
    if Cnorm.ndim != 3 or Cnorm.shape[1:] != (3, 3) or len(Cnorm) == 0:
        raise ValueError('Cnorm must be nonempty [K,3,3]')
    if Y.device != Cnorm.device or Y.dtype != Cnorm.dtype:
        raise ValueError('Y and Cnorm must share device and floating dtype')
    if Y.dtype not in (torch.float32, torch.float64):
        raise ValueError('FP32 production or FP64 mathematical probes only')
    if Cnorm.requires_grad:
        raise ValueError('Cnorm is a read-only constant, never a gradient input')


@contextmanager
def _geometry_context(Y):
    # Re-enable only this small geometry graph even inside caller no_grad and
    # BF16 native autocast. No model or parameter tensors enter these functions.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(Y.device.type, enabled=False):
        yield


def point_to_finite_triangles_squared(points, triangles):
    """All point/finite-triangle distances [P,K], entirely tensor operations.

    Interior orthogonal projection competes with all three finite segments.
    Degenerate target triangles use segment candidates, never an invented
    normal. Denominator clamps only prevent zero divisions; no signed SDF.
    """
    A, B, C = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = B - A, C - A
    normal = torch.linalg.cross(ab, ac, dim=-1)
    normal2 = normal.square().sum(-1)
    safe_normal2 = normal2.clamp_min(NUMERICAL_POLICY['distance_denominator_floor'])
    relative = points[:, None, :] - A[None, :, :]
    height_coefficient = (relative * normal[None]).sum(-1) / safe_normal2[None]
    projected_relative = relative - height_coefficient[..., None] * normal[None]
    b = (torch.linalg.cross(projected_relative, ac[None], dim=-1) * normal[None]).sum(-1) / safe_normal2[None]
    c = (torch.linalg.cross(ab[None], projected_relative, dim=-1) * normal[None]).sum(-1) / safe_normal2[None]
    interior = (b >= 0) & (c >= 0) & (b + c <= 1) & (normal2[None] > NUMERICAL_POLICY['distance_denominator_floor'])
    plane2 = (relative - projected_relative).square().sum(-1)
    plane2 = torch.where(interior, plane2, torch.full_like(plane2, float('inf')))
    start = triangles
    end = triangles[:, (1, 2, 0), :]
    edge = end - start
    edge2 = edge.square().sum(-1).clamp_min(NUMERICAL_POLICY['distance_denominator_floor'])
    segment_relative = points[:, None, None, :] - start[None]
    alpha = (segment_relative * edge[None]).sum(-1) / edge2[None]
    nearest = start[None] + alpha.clamp(0, 1)[..., None] * edge[None]
    segment2 = (points[:, None, None, :] - nearest).square().sum(-1).min(-1).values
    return torch.minimum(plane2, segment2)


def _seven_points(Y):
    return torch.cat((Y, (Y + Y[:, (1, 2, 0), :]) * .5,
                      Y.mean(dim=1, keepdim=True)), dim=1)


def energy_terms(Y, Cnorm, mode='GAR', ledger=None):
    """Selected energy terms as GPU tensors; no transfers or scalar reads.

    Call with detached leaf FP32 Y.requires_grad_(True), constant Cnorm, then
    gradients_from_terms. Only selected terms are evaluated. For GAR both
    components are normalized by N_free; R additionally averages its7 points.
    """
    _validate(Y, Cnorm)
    if mode not in ('GA', 'GR', 'GAR'):
        raise ValueError('Registered modes are GA, GR, GAR; G0 skips geometry')
    _inc(ledger, 'energy_forward_attempts')
    terms = {}
    with _geometry_context(Y):
        if mode in ('GA', 'GAR'):
            _inc(ledger, 'energy_A_evaluations')
            edges = Y[:, (1, 2, 0), :] - Y
            area = .5 * torch.linalg.vector_norm(torch.linalg.cross(Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 0], dim=-1), dim=-1)
            edge_squared_sum = edges.square().sum(dim=(1, 2))
            q_unclipped = edge_squared_sum / (4 * math.sqrt(3) * area.clamp_min(NUMERICAL_POLICY['area_floor']))
            q = q_unclipped.clamp_min(NUMERICAL_POLICY['q_floor'])
            excess = torch.relu(torch.log(q / NUMERICAL_POLICY['q_cut']))
            terms.update(E_A=(.5 * excess.square()).mean(), q=q, area=area,
                area_floor_count=(area < NUMERICAL_POLICY['area_floor']).sum(),
                exact_zero_area_count=(area == 0).sum(),
                near_degenerate_count=(area <= NUMERICAL_POLICY['actual_near_degenerate_area_threshold']).sum(),
                q_above_cut_count=(q > NUMERICAL_POLICY['q_cut']).sum(),
                q_numeric_floor_count=(q_unclipped < NUMERICAL_POLICY['q_floor']).sum())
        if mode in ('GR', 'GAR'):
            _inc(ledger, 'energy_R_evaluations')
            points = _seven_points(Y)
            d2 = point_to_finite_triangles_squared(points.reshape(-1, 3), Cnorm).min(-1).values.reshape(len(Y), 7)
            gaussian = torch.exp(-d2 / NUMERICAL_POLICY['sigma'] ** 2)
            terms.update(E_R=gaussian.mean(), point_distance_sq=d2,
                gaussian_per_point=gaussian, gaussian_per_face=gaussian.mean(-1))
        terms['E_selected'] = terms['E_A'] + terms['E_R'] if mode == 'GAR' else terms['E_A' if mode == 'GA' else 'E_R']
    _inc(ledger, 'energy_forward_calls')
    _inc(ledger, 'CUDA_energy_forward_calls' if Y.is_cuda else 'CPU_energy_forward_calls')
    return terms


def _gradient(energy, Y, ledger, retain_graph=False):
    _inc(ledger, 'geometry_backward_attempts')
    gradient = torch.autograd.grad(energy, Y, retain_graph=retain_graph, create_graph=False, allow_unused=False)[0]
    _inc(ledger, 'geometry_backward_calls')
    _inc(ledger, 'CUDA_geometry_backward_calls' if Y.is_cuda else 'CPU_geometry_backward_calls')
    return gradient.detach()


def gradients_from_terms(terms, Y, mode='GAR', ledger=None):
    """Geometry-only reverse passes; GAR uses2 passes and sums the gradients.

    The caller must clip g_selected once. Component cos=0 is a placeholder
    when cosine_defined=False, not evidence of orthogonality. All diagnostics
    remain on device and must be transferred only after a complete trajectory.
    """
    if mode not in ('GA', 'GR', 'GAR') or not Y.is_leaf or not Y.requires_grad:
        raise ValueError('Require registered mode and detached leaf Y with gradients')
    result = {}
    with _geometry_context(Y):
        if mode in ('GA', 'GAR'):
            result['g_A'] = _gradient(terms['E_A'], Y, ledger, retain_graph=mode == 'GAR')
            result['norm_A'] = torch.linalg.vector_norm(result['g_A'])
        if mode in ('GR', 'GAR'):
            result['g_R'] = _gradient(terms['E_R'], Y, ledger)
            result['norm_R'] = torch.linalg.vector_norm(result['g_R'])
        g = result['g_A'] + result['g_R'] if mode == 'GAR' else result['g_A' if mode == 'GA' else 'g_R']
        result.update(g_selected=g, norm_selected=torch.linalg.vector_norm(g),
            rms_selected=g.square().mean().sqrt(), gradient_finite=torch.isfinite(g).all(),
            energy_finite=torch.isfinite(terms['E_selected']).all())
        if mode == 'GAR':
            denominator = result['norm_A'] * result['norm_R']
            defined = denominator > 0
            dot = (result['g_A'] * result['g_R']).sum()
            result.update(gradient_dot=dot, cosine_defined=defined,
                cosine=torch.where(defined, dot / denominator.clamp_min(1e-30), torch.zeros_like(dot)))
    return {key: value.detach() for key, value in result.items()}


def vertex_gaussian_terms(Y, Cnorm, ledger=None):
    """Probe-only original vertex Gaussian; never silently used as formal GR."""
    _validate(Y, Cnorm)
    _inc(ledger, 'energy_forward_attempts')
    _inc(ledger, 'vertex_gaussian_evaluations')
    with _geometry_context(Y):
        d2 = point_to_finite_triangles_squared(Y.reshape(-1, 3), Cnorm).min(-1).values.reshape(len(Y), 3)
        kernel = torch.exp(-d2 / NUMERICAL_POLICY['sigma'] ** 2)
        result = dict(V_vertex_gaussian=kernel.mean(), vertex_distance_sq=d2,
                      gaussian_per_vertex=kernel)
    _inc(ledger, 'energy_forward_calls')
    _inc(ledger, 'CUDA_energy_forward_calls' if Y.is_cuda else 'CPU_energy_forward_calls')
    return result


def diagnostic_snapshot(Y, Cnorm, mode='GAR', ledger=None):
    """One forward plus geometry reverse pass(es), with detached GPU outputs.

    This helper is suitable for root-owned GT compatibility diagnostics; the
    returned gradient is never a training or sampling decision input there.
    """
    working = Y.detach().clone().requires_grad_(True)
    terms = energy_terms(working, Cnorm.detach(), mode, ledger)
    gradient = gradients_from_terms(terms, working, mode, ledger)
    return dict(terms={key: value.detach() for key, value in terms.items()}, gradients=gradient)


def _numpy_point_triangle(point, tri):
    """Independent scalar FP64 closest-region reference for probes only."""
    a,b,c=tri;ab=b-a;ac=c-a;ap=point-a
    d1=float(ab@ap);d2=float(ac@ap)
    if d1<=0 and d2<=0:return float(ap@ap)
    bp=point-b;d3=float(ab@bp);d4=float(ac@bp)
    if d3>=0 and d4<=d3:return float(bp@bp)
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        v=d1/(d1-d3);delta=point-(a+v*ab);return float(delta@delta)
    cp=point-c;d5=float(ab@cp);d6=float(ac@cp)
    if d6>=0 and d5<=d6:return float(cp@cp)
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        w=d2/(d2-d6);delta=point-(a+w*ac);return float(delta@delta)
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        w=(d4-d3)/((d4-d3)+(d5-d6));delta=point-(b+w*(c-b));return float(delta@delta)
    denom=va+vb+vc
    if denom<=0:raise ValueError('Independent smooth reference only accepts nondegenerate C')
    v=vb/denom;w=vc/denom;delta=point-(a+ab*v+ac*w)
    return max(0.,float(delta@delta))


def _numpy_energy(Y,C,mode):
    """Independent FP64 reference, deliberately scalar and probe-only."""
    Y=np.asarray(Y,np.float64);C=np.asarray(C,np.float64);value=0.
    if mode in ('GA','GAR'):
        areas=.5*np.linalg.norm(np.cross(Y[:,1]-Y[:,0],Y[:,2]-Y[:,0]),axis=-1)
        edges=Y[:,[1,2,0]]-Y;s=np.sum(edges*edges,axis=(1,2))
        q=np.maximum(s/(4*math.sqrt(3)*np.maximum(areas,1e-10)),1e-12)
        value+=float(np.mean(.5*np.maximum(np.log(q/10),0)**2))
    if mode in ('GR','GAR'):
        points=np.concatenate((Y,(Y+Y[:,[1,2,0]])*.5,Y.mean(1,keepdims=True)),axis=1)
        d2=[min(_numpy_point_triangle(p,t) for t in C) for p in points.reshape(-1,3)]
        value+=float(np.exp(-np.array(d2)/.005**2).mean())
    return value


def _finite_difference(Y,C,mode,ledger):
    gradient=np.empty_like(Y);h=NUMERICAL_POLICY['finite_difference_h']
    for index in np.ndindex(Y.shape):
        plus=Y.copy();minus=Y.copy();plus[index]+=h;minus[index]-=h
        gradient[index]=(_numpy_energy(plus,C,mode)-_numpy_energy(minus,C,mode))/(2*h)
        _inc(ledger,'finite_difference_energy_evaluations',2)
    return gradient


def _serial(value):
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):return value.item() if value.ndim==0 else value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:_serial(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_serial(item) for item in value]
    return value


def run_fixed_probes(ledger=None, gpu=True):
    """Fixed Stage A/B probes, including CPU FP64 and actual CUDA FP32.

    No model or optimizer is imported or invoked. Counterexamples pass when
    faithfully reproduced. All geometry-classifier calls are confined here.
    Formal root execution may invoke this same fixed suite once and record it
    separately from implementation checks; no new fixtures are selected.
    """
    from itertools import permutations
    from .guided_native_geometry_geometry import verified_pair
    if ledger is None:ledger=initial_energy_ledger()
    before=dict(ledger);_inc(ledger,'fixture_runs')
    C=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],np.float64)
    equilateral=np.array([[[.1,.1,.2],[.4,.1,.2],[.25,.1+math.sqrt(3)*.15,.2]]])
    thin=np.array([[[.12,.18,.0031],[.83,.18,.0031],[.16,.1803,.0031]]])
    visible=np.array([[[.2,.2,.02],[.6,.2,-.02],[.2,.6,.02]]])
    blind=np.array([[[.99,.002,.3],[.99,.002,-.7],[1.4,.002,.3]]])
    shared=np.array([[[0.,0.,0.],[1.,0.,0.],[.5,-.004,.002]]])
    folded=np.array([[[0.,0.,0.],[1.,0.,0.],[.5,0.,.004]]])
    exact=np.array([[[.2,.2,0.],[.4,.2,0.],[.2,.4,0.]]])
    good1=np.array([[-.5,-.3,0.],[.5,-.3,0.],[0.,.566025403784,0.]])
    good2=np.array([[-.5,0.,-.3],[.5,0.,-.3],[0.,0.,.566025403784]])
    arrays=dict(C=C,equilateral=equilateral,thin=thin,visible_puncture=visible,
        seven_point_blind_puncture=blind,shared_edge=shared,folded_edge=folded,
        exact_surface=exact,good_intersecting_pair=np.stack((good1,good2)))
    fixtures=[]
    def snapshot(free,mode='GAR',device='cpu',dtype=torch.float64,context=C):
        result=diagnostic_snapshot(torch.tensor(free,device=device,dtype=dtype),
            torch.tensor(context,device=device,dtype=dtype),mode,ledger)
        return _serial(result)
    def classify(A,B):
        local={};v=verified_pair(A,B,1.,local)
        _inc(ledger,'offline_primary_pair_tests',local.get('primary_pair_tests',0))
        _inc(ledger,'offline_independent_pair_tests',local.get('independent_pair_tests',0))
        return v
    eq=snapshot(equilateral,'GA')
    np.testing.assert_allclose(eq['terms']['q'],[1.],atol=1e-12,rtol=1e-12)
    assert eq['terms']['E_A']==0
    fixtures.append(dict(name='equilateral_q_one',status='PASS',observed=eq))
    ref=snapshot(thin,'GA');angle=.37
    rotation=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1.]])
    transforms=[('rigid',thin@rotation.T+np.array([.7,-.8,.3])),('scale',thin*2.3)]
    transforms += [('vertex_permutation_'+''.join(map(str,p)),thin[:,p,:]) for p in permutations(range(3))]
    for name,free in transforms:
        value=snapshot(free,'GA')
        np.testing.assert_allclose(value['terms']['q'],ref['terms']['q'],atol=1e-8,rtol=1e-8)
    assert ref['terms']['q'][0]>10 and ref['terms']['E_A']>0
    fixtures.append(dict(name='shape_rigid_scale_corner_invariance_and_thin_active',status='PASS',variants=len(transforms),thin_reference=ref))
    collapsed=np.array([[[.2,.2,0.]]*3,[[.1,.1,0.],[.2,.1,0.],[.3,.1,0.]]])
    deg=snapshot(collapsed,'GA')
    assert deg['terms']['area_floor_count']==2 and deg['terms']['exact_zero_area_count']==2
    assert np.isfinite(np.array(deg['gradients']['g_selected'])).all()
    assert np.array_equal(np.array(deg['gradients']['g_selected'])[0],np.zeros((3,3)))
    fixtures.append(dict(name='coincident_and_collinear_degeneracy_not_rescued',status='PASS_COUNTEREXAMPLE',observed=deg))
    intersection=classify(good1,good2);good=snapshot(np.stack((good1,good2)),'GA')
    assert intersection['classification']=='proper_transverse_penetration' and good['terms']['E_A']==0
    fixtures.append(dict(name='good_shape_energy_zero_despite_penetration',status='PASS_COUNTEREXAMPLE',intersection=intersection,observed=good))
    cpu_reference={}
    for mode in ('GA','GR','GAR'):
        value=snapshot(thin,mode);gradient=np.array(value['gradients']['g_selected'])
        finite_difference=_finite_difference(thin,C,mode,ledger)
        np.testing.assert_allclose(gradient,finite_difference,atol=NUMERICAL_POLICY['fp64_gradient_atol'],rtol=NUMERICAL_POLICY['fp64_gradient_rtol'])
        independent=_numpy_energy(thin,C,mode);_inc(ledger,'finite_difference_energy_evaluations')
        np.testing.assert_allclose(value['terms']['E_selected'],independent,atol=1e-11,rtol=1e-11)
        cpu_reference[mode]=value
        fixtures.append(dict(name='independent_finite_difference_'+mode,status='PASS',maximum_absolute_gradient_error=float(np.max(np.abs(gradient-finite_difference))),observed=value))
    for name,free in [('vertex_miss_seven_points_detect',visible),('both_vertex_and_seven_points_miss',blind)]:
        value=snapshot(free,'GR');y=torch.tensor(free,dtype=torch.float64,requires_grad=True)
        old=vertex_gaussian_terms(y,torch.tensor(C,dtype=torch.float64),ledger)
        old_grad=_gradient(old['V_vertex_gaussian'],y,ledger)
        crossing=classify(C[0],free[0]);assert crossing['classification']=='proper_transverse_penetration'
        oldE=float(old['V_vertex_gaussian'].detach());newE=value['terms']['E_R']
        if name.startswith('vertex_miss'):
            assert oldE<1e-6 and newE>.2
        else:
            assert oldE<1e-50 and newE<1e-50
        fixtures.append(dict(name=name,status='PASS_COUNTEREXAMPLE',intersection=crossing,
            vertex_energy=oldE,vertex_gradient=_serial(old_grad),seven_point=value))
    surface=snapshot(exact,'GR')
    assert surface['terms']['E_R']==1. and np.linalg.norm(surface['gradients']['g_selected'])<1e-12
    fixtures.append(dict(name='surface_kernel_maximum_zero_gradient',status='PASS_COUNTEREXAMPLE',observed=surface))
    for name,free in [('legal_shared_edge',shared),('legal_90_degree_fold',folded)]:
        value=snapshot(free,'GR');kind=classify(C[0],free[0]);assert kind['resolved'] and not kind['illegal']
        assert value['terms']['E_R']>0 and value['gradients']['norm_selected']>0
        fixtures.append(dict(name=name,status='PASS_COUNTEREXAMPLE',intersection=kind,observed=value))
    sides=np.array([[[.2,.2,.004],[.6,.2,-.004],[.2,.6,.004]]])
    # Retain the actual leaf; never detach sampled points from the free vertices.
    sy=torch.tensor(sides,dtype=torch.float64,requires_grad=True)
    side_old=vertex_gaussian_terms(sy,torch.tensor(C,dtype=torch.float64),ledger)
    sg=_gradient(side_old['V_vertex_gaussian'],sy,ledger).numpy()
    moved=sides-1e-5*sg;side_after=vertex_gaussian_terms(torch.tensor(moved,dtype=torch.float64),torch.tensor(C,dtype=torch.float64),ledger)
    side_before_class=classify(C[0],sides[0]);side_after_class=classify(C[0],moved[0])
    assert side_before_class['illegal'] and side_after_class['classification']=='proper_transverse_penetration'
    assert float(side_after['V_vertex_gaussian'])<float(side_old['V_vertex_gaussian'].detach())
    fixtures.append(dict(name='two_sides_repulsion_energy_decrease_still_intersection',status='PASS_COUNTEREXAMPLE',step=1e-5,
        before_energy=float(side_old['V_vertex_gaussian'].detach()),after_energy=float(side_after['V_vertex_gaussian']),gradient=sg.tolist(),after_class=side_after_class))
    point=torch.tensor([[1.2,1.2,0.]],dtype=torch.float64)
    finite=float(point_to_finite_triangles_squared(point,torch.tensor(C,dtype=torch.float64))[0,0])
    assert abs(finite-.98)<1e-12
    fixtures.append(dict(name='finite_triangle_not_infinite_plane',status='PASS',point=[1.2,1.2,0.],finite_distance_squared=finite,infinite_plane_distance_squared=0.))
    gpu_rows=[]
    if gpu:
        if not torch.cuda.is_available():raise RuntimeError('GPU FP32 probes required; no CPU substitution')
        for mode in ('GA','GR','GAR'):
            actual=snapshot(thin,mode,'cuda',torch.float32)
            reference=cpu_reference[mode]
            np.testing.assert_allclose(actual['terms']['E_selected'],reference['terms']['E_selected'],atol=NUMERICAL_POLICY['gpu_energy_atol'],rtol=NUMERICAL_POLICY['gpu_energy_rtol'])
            np.testing.assert_allclose(actual['gradients']['g_selected'],reference['gradients']['g_selected'],atol=NUMERICAL_POLICY['gpu_gradient_atol'],rtol=NUMERICAL_POLICY['gpu_gradient_rtol'])
            repeat=snapshot(thin,mode,'cuda',torch.float32)
            assert actual==repeat
            gpu_rows.append(dict(mode=mode,energy_absolute_error=abs(actual['terms']['E_selected']-reference['terms']['E_selected']),
                gradient_max_absolute_error=float(np.max(np.abs(np.array(actual['gradients']['g_selected'])-reference['gradients']['g_selected']))),repeat_bitwise=True,observed=actual))
        mixed=np.concatenate((thin,visible,equilateral));base=snapshot(mixed,'GAR','cuda',torch.float32)
        order=[2,0,1];corner=[2,0,1];permuted=snapshot(mixed[order][:,corner],'GAR','cuda',torch.float32)
        restored=np.array(permuted['gradients']['g_selected'])[np.argsort(order)][:,np.argsort(corner)]
        np.testing.assert_allclose(permuted['terms']['E_selected'],base['terms']['E_selected'],atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(restored,base['gradients']['g_selected'],atol=2e-5,rtol=2e-3)
        gpu_rows.append(dict(mode='GAR_face_and_corner_permutation',energy_absolute_error=abs(permuted['terms']['E_selected']-base['terms']['E_selected']),
            restored_gradient_max_absolute_error=float(np.max(np.abs(restored-base['gradients']['g_selected'])))))
        for name,free in [('collapsed',collapsed),('visible_puncture',visible),('blind_puncture',blind),('surface',exact),('shared',shared),('folded',folded)]:
            value=snapshot(free,'GAR','cuda',torch.float32)
            assert value['gradients']['gradient_finite'] and value['gradients']['energy_finite']
            gpu_rows.append(dict(mode='FP32_'+name,observed=value))
    return dict(status='PASS',policy=dict(NUMERICAL_POLICY),fixtures=fixtures,GPU_probes=gpu_rows,
        GPU_probe_status='PASS' if gpu else 'NOT_RUN',ledger_delta={k:v-before.get(k,0) for k,v in ledger.items()},
        counters_are_not_geometry_safety_certificates=True,
        fixture_design_note='Seven-point blind fixture z=.3/-.7/.3 fixed analytically before tests; initial draft z=+/-1 rejected before execution because midpoint would hit C',
        model_forwards=0,model_backwards=0,optimizer_updates=0),arrays
