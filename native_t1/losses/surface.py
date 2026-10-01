"""Training-only area/unsigned-normal surface MMD (the fixed L5 objective).

Three barycentric points per face and spatial scales .02/.05/.10 are fixed.
Predicted areas, locations and unsigned normal matrices retain gradients.
Both measures divide by the fixed GT total area; prediction mass is not
independently normalized. This does not certify connectivity or intersections.
Only detached GT geometry/kernel-self constants may be cached, never predictions.
The schema/formula strings and zero-valued legacy work-counter fields are kept
for exact diagnostic compatibility; no transport solver is implemented here.
"""
from contextlib import nullcontext
import torch
NORMAL_EPSILON = 1e-8
SPATIAL_SCALES = (0.02, 0.05, 0.10)
NORMAL_SCALE = 0.5
SCHEMA = 'loss_screen_distribution_target_v1'
FORMULA = 'complete_KL_symmetric_log_sinkhorn96_and_area_normal_MMD_v1'
_WORK_COUNTS = dict(sinkhorn_solve_attempts=0, sinkhorn_solve_returns=0, sinkhorn_iterations=0,
                    kernel_matrix_attempts=0, kernel_matrix_returns=0)

def work_counters():
    """Process-local actual attempts/returns, including failed calls; no reset."""
    return dict(_WORK_COUNTS)

def _tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(name + ' must be an FP32/FP64 tensor.')
    if value.ndim != 3 or value.shape[1:] != (3, 3) or len(value) == 0:
        raise ValueError(name + ' must have nonempty [F,3,3] shape.')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' contains nonfinite coordinates.')
    return value

def _autocast_off(device):
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ('cpu', 'cuda') else nullcontext()

def _quadrature(triangles):
    bary = triangles.new_tensor(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    return (triangles[:, None, :, :] * bary[None, :, :, None]).sum(2).reshape(-1, 3)

def _geometry(triangles):
    cross = torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], dim=-1)
    twice_area = torch.linalg.vector_norm(cross, dim=-1)
    area = twice_area * 0.5
    normal = cross / twice_area.clamp_min(NORMAL_EPSILON)[:, None]
    Q = (normal[:, :, None] * normal[:, None, :]).reshape(-1, 9)
    return dict(points=_quadrature(triangles), areas=area, Q=Q.repeat_interleave(3, dim=0),
                normal_floor_count=int((twice_area < NORMAL_EPSILON).sum().detach()),
                zero_area_count=int((area == 0).sum().detach()))

def _squared_distances(x, y):
    # Direct differences preserve an exact zero on the diagonal, without
    # clamping cancellation from x^2 + y^2 - 2xy or differentiating sqrt(0).
    return (x[:, None, :] - y[None, :, :]).square().sum(-1)

def _kernel_values(points_a, Q_a, points_b, Q_b):
    position_cost = _squared_distances(points_a, points_b)
    normal_cost = _squared_distances(Q_a, Q_b)
    result=[]
    for scale in SPATIAL_SCALES:
        _WORK_COUNTS['kernel_matrix_attempts'] += 1
        result.append((-position_cost/(2*scale*scale)-normal_cost/(2*NORMAL_SCALE*NORMAL_SCALE)).exp())
        _WORK_COUNTS['kernel_matrix_returns'] += 1
    return result

def _weighted_kernel_values(x, wx, y, wy):
    return torch.stack([(wx[:, None]*wy[None, :]*kernel).sum()
        for kernel in _kernel_values(x['points'], x['Q'], y['points'], y['Q'])])

def prepare_distribution_target(free_gt, bbox_L):
    """Detach/clone fixed GT only. Preparation performs zero kernel evaluations.

    The returned dictionary may be retained across updates for the same target.
    Cache entries contain only detached GT measures, GT-GT constants and their
    diagnostics. Every loss result counts actual kernel matrices evaluated
    in that call, distinguishing cached GT self terms.
    """
    free_gt = _tensor(free_gt, 'free_gt')
    length = torch.as_tensor(bbox_L, dtype=free_gt.dtype, device=free_gt.device).detach().clone()
    if length.numel()!=1 or not bool(torch.isfinite(length)) or float(length)<=0:
        raise ValueError('bbox_L must be a finite positive fixed scalar.')
    return dict(schema=SCHEMA, formula=FORMULA, free_gt=free_gt.detach().clone(), bbox_L=length.reshape(()),
        preparation_counts=dict(sinkhorn_solves=0, sinkhorn_iterations=0, kernel_matrix_evaluations=0),
        _cache={})

def _target_on(target, prediction):
    if target.get('schema')!=SCHEMA or target.get('formula')!=FORMULA:
        raise ValueError('Unexpected distribution target schema/formula.')
    if target['free_gt'].requires_grad or target['bbox_L'].requires_grad:
        raise ValueError('GT and bbox supervision must remain detached.')
    key=(str(prediction.device),str(prediction.dtype))
    if key not in target['_cache']:
        with torch.no_grad():
            length=target['bbox_L'].to(device=prediction.device,dtype=prediction.dtype)
            triangles=target['free_gt'].to(device=prediction.device,dtype=prediction.dtype)/length
            geometry=_geometry(triangles)
            geometry['total_area']=geometry['areas'].sum()
            geometry['bbox_L']=length
            geometry['kernel_self']=None
            target['_cache'][key]=geometry
    return target['_cache'][key]

def distribution_loss(name, pred_free, target):
    """Return one unweighted per-sample mean, leaving g(t)/ramp/lambda outside.

    All predicted positions, areas, probabilities and unsigned normals retain
    their full gradient. FP32 production and FP64 fixtures use the same formula.
    Negative finite divergence/MMD values are reported as-is, never clamped.
    """
    if name != 'L5':raise ValueError('Only the fixed L5 objective is available.')
    pred_free=_tensor(pred_free,'pred_free')
    counts=dict(sinkhorn_solves=0,sinkhorn_iterations=0,GTGT_sinkhorn_solves=0,GTGT_sinkhorn_cache_hits=0,
        kernel_matrix_evaluations=0,GTGT_kernel_matrix_evaluations=0,GTGT_kernel_cache_hits=0)
    with _autocast_off(pred_free.device):
        gt=_target_on(target,pred_free)
        pred=_geometry(pred_free/gt['bbox_L'])
        common=dict(formula=FORMULA,coordinate_normalization='fixed training bbox diagonal',
            predicted_faces=len(pred_free),GT_faces=len(target['free_gt']),quadrature_points_per_face=3,
            predicted_total_area_normalized=float(pred['areas'].sum().detach()),GT_total_area_normalized=float(gt['total_area']),
            predicted_zero_area_count=pred['zero_area_count'],GT_zero_area_count=gt['zero_area_count'],
            predicted_normal_floor_count=pred['normal_floor_count'],GT_normal_floor_count=gt['normal_floor_count'],
            fixed_GT_requires_grad=False,negative_value_clamped=False)
        if float(gt['total_area'])<=0:
            return dict(loss=pred_free.sum()*0,applicable=False,counts=counts,
                diagnostics=dict(**common,status='NOT_APPLICABLE_ZERO_GT_AREA'))
        wp=(pred['areas']/(3*gt['total_area'])).repeat_interleave(3)
        wg=(gt['areas']/(3*gt['total_area'])).repeat_interleave(3)
        pp=_weighted_kernel_values(pred,wp,pred,wp)
        pg=_weighted_kernel_values(pred,wp,gt,wg)
        counts['kernel_matrix_evaluations']=6
        if gt['kernel_self'] is None:
            with torch.no_grad():gt['kernel_self']=_weighted_kernel_values(gt,wg,gt,wg)
            counts['kernel_matrix_evaluations']+=3;counts['GTGT_kernel_matrix_evaluations']=3
        else:counts['GTGT_kernel_cache_hits']=1
        gg=gt['kernel_self'];per_scale=pp+gg-2*pg;loss=per_scale.mean()
        diagnostics=dict(**common,spatial_scales=list(SPATIAL_SCALES),normal_scale=NORMAL_SCALE,normal_epsilon=NORMAL_EPSILON,
            predicted_measure_mass=float(wp.sum().detach()),GT_measure_mass=float(wg.sum()),
            predicted_mass_independently_normalized=False,area_floor_applied=False,
            per_scale_MMD2=per_scale.detach().cpu().tolist(),pred_pred=pp.detach().cpu().tolist(),
            pred_GT=pg.detach().cpu().tolist(),GT_GT=gg.detach().cpu().tolist(),negative_value=bool(loss.detach()<0))
        if not bool(torch.isfinite(loss)):raise FloatingPointError(name+' produced a nonfinite loss.')
        return dict(loss=loss,applicable=True,counts=counts,diagnostics=diagnostics)
