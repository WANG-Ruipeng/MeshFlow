"""Training-only finite GT interface edge supervision, without new model inputs.

Only an undirected source edge having exactly two incident faces, one known
and one free, is eligible. GT identities select the corresponding predicted
free corners, never a nearest predicted edge. Fixed GT lengths weight both
finite-segment coverage directions. No projection modifies the prediction.

Both query and segment coordinates are normalized by the fixed GT bbox L.
Segment projection is centered at the midpoint and uses clamp(length^2,1e-12),
then clamps its centered parameter to [-.5,.5]. Swapping endpoints therefore
has the same distance and the corresponding swapped gradients. At an exactly
zero predicted edge, the projection point is its midpoint: the two endpoints
share midpoint gradients. A point collapsed at the symmetric GT midpoint may
have positive uncovered-length loss and zero gradient; this is a documented
stationary degeneracy, not a guarantee that the loss expands collapsed edges.
Short predicted edges are retained and counted. A GT interface with normalized
length squared <=1e-12 is rejected, never skipped or repaired.

D_edge only supervises geometry of GT-corresponding finite edges. It does not
create shared IDs, guarantee watertightness, or prevent global intersections.
"""
from collections.abc import Mapping
from collections import defaultdict
import torch

from .relations import prepare_relations

EPSILON = 1e-12
S_VALUES = (0., .25, .5, .75, 1.)
QUADRATURE_WEIGHTS = (.125, .25, .25, .25, .125)
SCHEMA = 'finite_interface_edge_target_v1'
FORMULA_SPEC = dict(schema=SCHEMA,coordinate_scale='fixed_full_GT_bbox_diagonal',
    s=list(S_VALUES),weights=list(QUADRATURE_WEIGHTS),direction_weights=[.5,.5],
    weights_from='fixed_GT_eligible_edge_length',
    eligibility='one_undirected_edge_exactly_one_known_plus_one_free_global_incidence',
    segment='midpoint_centered_projection_clamped_to_finite_segment',
    denominator_epsilon=EPSILON,GT_interface_length_squared_reject_le=EPSILON,
    predicted_short_edge='retained_and_counted; exact_zero_uses_midpoint_shared_endpoint_gradient',
    unavailable='graph_connected_zero_NOT_APPLICABLE')


def prepare_edge_target(label_or_free_gt, known=None, free_ids=None, known_ids=None, bbox_L=None):
    """Accept a validated training geometry label, or five explicit positional arguments.

    label keys: free_gt, known, free_ids, known_ids, bbox_L. All source IDs must
    follow actual target face/corner permutations; nested OT only permutes Z.
    Fixed relation preparation validates exact source-ID correspondence;
    source labels are supervision only and never model arguments.
    """
    if isinstance(label_or_free_gt,Mapping):
        if any(x is not None for x in (known,free_ids,known_ids,bbox_L)):
            raise ValueError('Use either one label mapping or five explicit inputs')
        label=label_or_free_gt
        free_gt,known,free_ids,known_ids,bbox_L=(label[k] for k in ('free_gt','known','free_ids','known_ids','bbox_L'))
    else:
        free_gt=label_or_free_gt
        if any(x is None for x in (known,free_ids,known_ids,bbox_L)):
            raise ValueError('Five explicit geometry/ID/scale inputs are required')
    base=prepare_relations(free_gt,known,free_ids,known_ids,bbox_L)
    fids=torch.as_tensor(free_ids).detach().cpu().tolist()
    kids=torch.as_tensor(known_ids).detach().cpu().tolist()
    incidences=defaultdict(list)
    for is_free,rows in ((True,fids),(False,kids)):
        for face,ids in enumerate(rows):
            for i,j in ((0,1),(1,2),(2,0)):
                incidences[tuple(sorted((ids[i],ids[j])))].append((is_free,face))
    counts=dict(free_faces=len(fids),known_faces=len(kids),source_edges=len(incidences),
        eligible_interface_edges=0,excluded_open_free_edges=0,excluded_open_known_edges=0,
        excluded_two_free_edges=0,excluded_two_known_edges=0,
        excluded_nonmanifold_mixed_edges=0,excluded_nonmanifold_free_only_edges=0,
        excluded_nonmanifold_known_only_edges=0,excluded_nonmanifold_known_free_pairs=0,
        max_source_edge_incidence=max(map(len,incidences.values())),
        original_known_free_pair_relations=base['counts']['known_free_interface_relations'])
    eligible=set()
    for edge,rows in incidences.items():
        free_count=sum(is_free for is_free,_ in rows);known_count=len(rows)-free_count
        if len(rows)==2 and free_count==known_count==1:
            eligible.add(edge);counts['eligible_interface_edges']+=1
        elif len(rows)==1:
            counts['excluded_open_free_edges' if free_count else 'excluded_open_known_edges']+=1
        elif len(rows)==2:
            counts['excluded_two_free_edges' if free_count else 'excluded_two_known_edges']+=1
        else:
            key='excluded_nonmanifold_mixed_edges' if free_count and known_count else (
                'excluded_nonmanifold_free_only_edges' if free_count else 'excluded_nonmanifold_known_only_edges')
            counts[key]+=1
            counts['excluded_nonmanifold_known_free_pairs']+=free_count*known_count
    selected=[];seen=set()
    faces=base['interface_free_faces'].cpu().tolist()
    corners=base['interface_free_corners'].cpu().tolist()
    for index,(face,pair) in enumerate(zip(faces,corners)):
        edge=tuple(sorted(fids[face][corner] for corner in pair))
        if edge not in eligible:continue
        if edge in seen:raise ValueError('An eligible undirected interface was registered twice')
        selected.append(index);seen.add(edge)
    if seen!=eligible:raise ValueError('Source relation mapping omitted an eligible edge')
    choose=torch.tensor(selected,dtype=torch.long,device=base['free_gt'].device)
    endpoints=base['interface_endpoints'][choose].detach().clone()
    lengths=base['interface_lengths'][choose].detach().clone()
    # Exact fixed threshold is in normalized coordinates, before any loss call.
    if len(lengths) and (not bool(torch.isfinite(lengths).all()) or bool((lengths.square()<=EPSILON).any())):
        raise ValueError('Abnormal GT interface: normalized length squared <=1e-12 or nonfinite; input investigation required')
    counts['excluded_known_free_pair_relations']=base['counts']['known_free_interface_relations']-len(selected)
    counts['quadrature_points_per_direction']=5*len(selected)
    return dict(schema=SCHEMA,bbox_L=base['bbox_L'],free_shape=tuple(base['free_gt'].shape),
        free_faces=base['interface_free_faces'][choose].detach().clone(),
        free_corners=base['interface_free_corners'][choose].detach().clone(),
        GT_endpoints=endpoints,GT_lengths=lengths,edge_ids=[list(x) for x in sorted(eligible)],
        counts=counts,GT_and_C_detached=True)


def centered_segment_distance_sq(points, segments):
    """points[E,Q,3] -> squared distance to corresponding segments[E,2,3].

    Piecewise differentiable closest-point choice; no distance or prediction is
    detached. The epsilon and zero-edge convention are identical in FP32/FP64.
    """
    if points.ndim!=3 or points.shape[-1]!=3 or segments.shape!=(len(points),2,3):
        raise ValueError('Expected corresponding points[E,Q,3] and segments[E,2,3]')
    if points.dtype!=segments.dtype or points.device!=segments.device:
        raise ValueError('Colocated same-dtype geometry is required')
    midpoint=(segments[:,0]+segments[:,1])*.5
    direction=segments[:,1]-segments[:,0]
    squared_length=direction.square().sum(-1)
    parameter=((points-midpoint[:,None])*direction[:,None]).sum(-1)/squared_length.clamp_min(EPSILON)[:,None]
    closest=midpoint[:,None]+parameter.clamp(-.5,.5)[...,None]*direction[:,None]
    return (points-closest).square().sum(-1)


def edge_loss(pred_free,target):
    """One sample mean only; caller applies lambda, flow-time gate, ramp and /8."""
    if not isinstance(pred_free,torch.Tensor) or pred_free.dtype not in (torch.float32,torch.float64):
        raise ValueError('FP32 production or FP64 fixture tensor required')
    if target.get('schema')!=SCHEMA or tuple(pred_free.shape)!=target['free_shape']:
        raise ValueError('Prediction layout differs from fixed source targets')
    if not bool(torch.isfinite(pred_free).all()):raise FloatingPointError('Nonfinite predicted free geometry')
    counts=dict(target['counts'])
    if not counts['eligible_interface_edges']:
        return dict(loss=pred_free.sum()*0,applicable=False,counts=counts,
                    diagnostics=dict(status='NOT_APPLICABLE',GT_and_C_detached=True))
    def fixed(key):
        value=target[key]
        return value.detach().to(device=pred_free.device,dtype=pred_free.dtype if value.is_floating_point() else value.dtype)
    with torch.autocast(device_type=pred_free.device.type,enabled=False):
        free=pred_free/target['bbox_L']
        predicted=free[fixed('free_faces')[:,None],fixed('free_corners')]
        known=fixed('GT_endpoints');length=fixed('GT_lengths')
        s=pred_free.new_tensor(S_VALUES)[None,:,None]
        w=pred_free.new_tensor(QUADRATURE_WEIGHTS)[None]
        q_gt=(1-s)*known[:,:1]+s*known[:,1:]
        q_pred=(1-s)*predicted[:,:1]+s*predicted[:,1:]
        gt_to_pred=(centered_segment_distance_sq(q_gt,predicted)*w).sum(-1)
        pred_to_gt=(centered_segment_distance_sq(q_pred,known)*w).sum(-1)
        GT_to_pred=(length*gt_to_pred).sum()/length.sum()
        pred_to_GT=(length*pred_to_gt).sum()/length.sum()
        value=.5*(GT_to_pred+pred_to_GT)
        squared_length=(predicted[:,1]-predicted[:,0]).square().sum(-1)
        counts['short_predicted_edges']=(squared_length<=EPSILON).sum().detach()
        counts['zero_predicted_edges']=(squared_length==0).sum().detach()
    if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite finite-interface edge loss')
    return dict(loss=value,applicable=True,counts=counts,
        diagnostics=dict(GT_to_pred_mean=GT_to_pred.detach(),pred_to_GT_mean=pred_to_GT.detach(),
          GT_total_length_normalized=length.sum().detach(),GT_and_C_detached=True,
          predicted_edges_retained=True,endpoint_symmetric=True,denominator_epsilon=EPSILON,
          zero_edge_gradient_convention='midpoint: equal endpoint gradients; symmetric centered collapse can remain stationary'))
