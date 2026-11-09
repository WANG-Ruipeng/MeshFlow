"""Fixed L1/L2/L3/L6 supervision from trusted, permuted GT source vertex IDs.

These are training-only objectives. No source identity enters a generator.
Coordinates in every objective are divided by the fixed full-object bbox L.
Targets are detached; prediction geometry is never detached or cached.

Blind spots: L1 only tests the GT-selected adjacent local half-space (coplanar
relations are inapplicable). L2 cannot detect translations, adjacency breaks,
or normal-sign/convexity changes. L3 permits tangential motion with weight .05
and does not guarantee exact shared edges. L6 samples a finite local boundary
and barycentric band, measures only to the free surface, and does not ensure
global coverage or topology. None is a general collision or watertightness loss.

Numerical policy is fixed before fixtures: side ambiguity <=1e-6 normalized
length; predicted normals use cross/clamp(norm,1e-8); finite-surface distance
uses all three finite edges and an interior plane candidate. Cross norm squared
<=1e-24 disables only the interior candidate, not the degenerate triangle or
its edges. Squared edge denominators use the same 1e-24 lower bound. No NaN,
Inf, or zero GT edge/triangle is accepted. Nonmanifold edge incidence is kept:
each incident-face pair with at least one free face contributes independently.
"""
from itertools import combinations
import math

import numpy as np
import torch


SIDE_EPS = 1e-6
NORMAL_EPS = 1e-8
DISTANCE_DENOM_EPS = 1e-24
TANGENTIAL_WEIGHT = 0.05
S_VALUES = (0., .25, .5, .75, 1.)
R_VALUES = (.1, .2)
FORMULA_SPEC = dict(
    version="gt_relation_losses_v1", coordinate_scale="fixed_full_GT_bbox_diagonal",
    side_ambiguity_le=SIDE_EPS, normal_cross_norm_epsilon=NORMAL_EPS,
    surface_cross_squared_epsilon=DISTANCE_DENOM_EPS,
    surface_edge_squared_epsilon=DISTANCE_DENOM_EPS,
    tangential_weight=TANGENTIAL_WEIGHT, gamma_s=list(S_VALUES),
    band_r=list(R_VALUES), L6_weights=[.5, .5],
    shared_edge_policy="all_incident_face_pairs_with_at_least_one_free",
    L1_weighting="equal_applicable_interface_relations_and_seven_points",
    L2_L3_L6_weighting="fixed_GT_edge_length_normalized_within_sample")


def _geometry(value, name, *, dtype=None, device=None):
    value = torch.as_tensor(value, dtype=dtype, device=device).detach().clone()
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError(name + " must be [faces,3,3]")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + " must use FP32 or FP64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    return value


def _ids(value, faces, name):
    value = torch.as_tensor(value).detach().cpu()
    if value.shape != (faces, 3) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(name + " must be integer [faces,3]")
    array = value.numpy().astype(np.int64, copy=False)
    if np.any(array < 0) or any(len(set(row.tolist())) != 3 for row in array):
        raise ValueError(name + " needs three distinct nonnegative source IDs per face")
    return array


def _cross(triangles):
    return torch.linalg.cross(triangles[:, 1]-triangles[:, 0],
                              triangles[:, 2]-triangles[:, 0], dim=-1)


def _normal(triangles):
    cross = _cross(triangles)
    length = torch.linalg.vector_norm(cross, dim=-1)
    return cross / length.clamp_min(NORMAL_EPS)[:, None], length


def prepare_relations(free_gt, known, free_vertex_ids, known_vertex_ids, bbox_L):
    """Prepare detached targets; IDs must follow actual target face/corner slots.

    May run on CPU and the returned target can be reused for this exact label
    layout. relation_loss transfers only fixed target tensors to pred's device.
    Source IDs are used only here. Same-ID coordinates must agree bitwise.
    """
    free = _geometry(free_gt, "free_gt")
    fixed = _geometry(known, "known", dtype=free.dtype, device=free.device)
    if not len(free):
        raise ValueError("At least one free face is required")
    scale = float(bbox_L.detach().cpu()) if isinstance(bbox_L, torch.Tensor) else float(bbox_L)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("bbox_L must be a positive finite fixed scalar")
    fids = _ids(free_vertex_ids, len(free), "free_vertex_ids")
    kids = _ids(known_vertex_ids, len(fixed), "known_vertex_ids")
    all_ids = np.concatenate((fids, kids), axis=0)
    source_triangles = torch.cat((free, fixed), dim=0)
    coordinates = {}
    edge_faces = {}
    for face, ids in enumerate(all_ids):
        for corner, vid in enumerate(ids):
            old = coordinates.setdefault(int(vid), source_triangles[face, corner])
            if not torch.equal(old, source_triangles[face, corner]):
                raise ValueError("Same source vertex ID has inconsistent coordinates")
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(ids[a]), int(ids[b]))))
            edge_faces.setdefault(key, []).append(face)
    normalized = source_triangles / scale
    clean_cross = _cross(normalized)
    cross_length = torch.linalg.vector_norm(clean_cross, dim=-1)
    if bool((cross_length == 0).any()):
        raise ValueError("Degenerate clean GT face cannot define the registered relation")
    normals = clean_cross / cross_length[:, None]
    pairs, lengths, reference_cos2 = [], [], []
    interfaces, interface_corners, endpoints, tangents = [], [], [], []
    interface_lengths, directions, applicable, gamma, band = [], [], [], [], []
    nfree = len(free)
    for edge in sorted(edge_faces):
        a, b = (coordinates[v] / scale for v in edge)
        length = torch.linalg.vector_norm(b-a)
        if float(length) == 0:
            raise ValueError("Zero GT edge cannot define a relation")
        for i, j in combinations(sorted(edge_faces[edge]), 2):
            if i >= nfree and j >= nfree:
                continue
            pairs.append((i, j)); lengths.append(length)
            reference_cos2.append(torch.dot(normals[i], normals[j]).square())
            if (i < nfree) == (j < nfree):
                continue
            fi, ki = (i, j) if i < nfree else (j, i)
            corners = [int(np.flatnonzero(all_ids[fi] == vid)[0]) for vid in edge]
            off = next(k for k in range(3) if k not in corners)
            third = normalized[fi, off]
            # GT normals are true units even for clean small triangles; the
            # prediction epsilon belongs to L2, not the L1 side convention.
            known_unit = _cross(normalized[ki:ki+1])[0] / cross_length[ki]
            signed = torch.dot(third-a, known_unit)
            good = abs(float(signed)) > SIDE_EPS
            interfaces.append(fi); interface_corners.append(corners)
            endpoints.append(torch.stack((a, b))); tangents.append((b-a)/length)
            interface_lengths.append(length)
            directions.append(known_unit * torch.sign(signed)); applicable.append(good)
            ss = free.new_tensor(S_VALUES)
            edge_points = (1-ss[:, None])*a + ss[:, None]*b
            gamma.append(edge_points)
            band.append(torch.cat([(1-r)*edge_points+r*third for r in R_VALUES], dim=0))

    def floats(values, shape):
        return torch.stack(values).detach() if values else free.new_empty(shape)
    def integers(values, shape):
        return torch.tensor(values, dtype=torch.long, device=free.device).reshape(shape)
    interface_count = len(interfaces)
    counts = dict(free_faces=nfree, known_faces=len(fixed), source_edges=len(edge_faces),
                  boundary_source_edges=sum(len(v)==1 for v in edge_faces.values()),
                  nonmanifold_source_edges=sum(len(v)>2 for v in edge_faces.values()),
                  max_source_edge_incidence=max(map(len, edge_faces.values())),
                  adjacent_pairs_at_least_one_free=len(pairs),
                  free_free_pairs=sum(i<nfree and j<nfree for i,j in pairs),
                  known_free_interface_relations=interface_count,
                  L1_applicable_relations=sum(applicable),
                  L1_inapplicable_coplanar_or_ambiguous=interface_count-sum(applicable),
                  gamma_points=5*interface_count, band_points=10*interface_count,
                  clean_normal_epsilon_faces=int((cross_length <= NORMAL_EPS).sum()),
                  duplicate_face_vertex_sets=len(all_ids)-len({tuple(sorted(row)) for row in all_ids}))
    return dict(formula_version=FORMULA_SPEC["version"], bbox_L=scale,
                free_gt=(free/scale).detach(), known=(fixed/scale).detach(),
                known_normals=normals[nfree:].detach(),
                pair_faces=integers(pairs, (-1,2)),
                pair_lengths=floats(lengths, (0,)),
                pair_gt_cos2=floats(reference_cos2, (0,)),
                interface_free_faces=integers(interfaces, (-1,)),
                interface_free_corners=integers(interface_corners, (-1,2)),
                interface_endpoints=floats(endpoints, (0,2,3)),
                interface_tangents=floats(tangents, (0,3)),
                interface_lengths=floats(interface_lengths, (0,)),
                interface_correct_normals=floats(directions, (0,3)),
                interface_applicable=torch.tensor(applicable,dtype=torch.bool,device=free.device),
                gamma=floats(gamma, (0,5,3)), band=floats(band, (0,10,3)), counts=counts)


def point_triangle_distance_sq(points, triangles):
    """Differentiable [Q,F] squared distances to finite triangle surfaces.

    The selected closest feature/triangle is piecewise differentiable. Exact
    degenerate triangles remain finite segments/points; they are never dropped.
    Coordinates are already bbox-normalized. No CPU geometry library is used.
    """
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 3 or triangles.shape[1:] != (3,3):
        raise ValueError("Expected points[Q,3], triangles[F,3,3]")
    a, b, c = (triangles[:, j][None] for j in range(3))
    p = points[:, None]
    edge_values = []
    for start, end in ((a,b),(b,c),(c,a)):
        edge = end-start
        fraction = ((p-start)*edge).sum(-1) / edge.square().sum(-1).clamp_min(DISTANCE_DENOM_EPS)
        closest = start + fraction.clamp(0,1)[...,None]*edge
        edge_values.append((p-closest).square().sum(-1))
    ab, ac, ap = b-a, c-a, p-a
    cross = torch.linalg.cross(ab, ac, dim=-1)
    denominator = cross.square().sum(-1)
    safe = denominator.clamp_min(DISTANCE_DENOM_EPS)
    d00, d01, d11 = ab.square().sum(-1), (ab*ac).sum(-1), ac.square().sum(-1)
    d20, d21 = (ap*ab).sum(-1), (ap*ac).sum(-1)
    u = (d11*d20-d01*d21)/safe
    v = (d00*d21-d01*d20)/safe
    interior = (denominator > DISTANCE_DENOM_EPS) & (u >= 0) & (v >= 0) & (u+v <= 1)
    plane = (ap*cross).sum(-1).square()/safe
    plane = torch.where(interior, plane, torch.full_like(plane, float('inf')))
    return torch.stack(edge_values+[plane], dim=-1).min(dim=-1).values


def relation_loss(name, pred_free, target):
    """Return one sample mean; caller alone applies time gate/ramp/lambda/1/8.

    Counts are small Python integers, except dynamic prediction-degeneracy
    counters, which stay scalar tensors on-device until the caller records them.
    """
    if name not in ('L1','L2','L3','L6'):
        raise ValueError("Unsupported registered relation loss: "+str(name))
    if not isinstance(pred_free, torch.Tensor) or pred_free.dtype not in (torch.float32,torch.float64):
        raise ValueError("Prediction must be a FP32/FP64 tensor")
    if pred_free.shape != target['free_gt'].shape:
        raise ValueError("Prediction face/corner layout differs from prepared GT labels")
    if target.get('formula_version') != FORMULA_SPEC['version']:
        raise ValueError("Wrong fixed formula version")
    counts = dict(target['counts'])
    count = counts['L1_applicable_relations'] if name=='L1' else (
        counts['adjacent_pairs_at_least_one_free'] if name=='L2' else counts['known_free_interface_relations'])
    if count == 0:
        return dict(loss=pred_free.sum()*0, applicable=False, counts=counts)
    def fixed(key):
        value=target[key]
        return value.detach().to(device=pred_free.device, dtype=pred_free.dtype if value.is_floating_point() else value.dtype)
    with torch.autocast(device_type=pred_free.device.type, enabled=False):
        x = pred_free / target['bbox_L']
        if name == 'L1':
            selected = fixed('interface_free_faces')[fixed('interface_applicable')]
            triangle = x[selected]
            points = torch.cat((triangle, (triangle+triangle.roll(-1,dims=1))*.5,
                                triangle.mean(dim=1,keepdim=True)), dim=1)
            a=fixed('interface_endpoints')[fixed('interface_applicable'),0]
            normal=fixed('interface_correct_normals')[fixed('interface_applicable')]
            signed=((points-a[:,None])*normal[:,None]).sum(-1)
            loss=torch.relu(-signed).square().mean()
        elif name == 'L2':
            free_normal,length = _normal(x)
            normal=torch.cat((free_normal,fixed('known_normals')),dim=0)
            pair=fixed('pair_faces')
            value=(normal[pair[:,0]]*normal[pair[:,1]]).sum(-1).square()
            weight=fixed('pair_lengths')
            loss=((value-fixed('pair_gt_cos2')).square()*weight).sum()/weight.sum()
            counts['prediction_normal_epsilon_faces']=(length[:len(x)]<=NORMAL_EPS).sum().detach()
        elif name == 'L3':
            face=fixed('interface_free_faces');corners=fixed('interface_free_corners')
            residual=x[face[:,None],corners]-fixed('interface_endpoints')
            tangent=fixed('interface_tangents')[:,None]
            parallel=(residual*tangent).sum(-1,keepdim=True)*tangent
            value=((residual-parallel).square().sum(-1)+TANGENTIAL_WEIGHT*parallel.square().sum(-1)).mean(-1)
            weight=fixed('interface_lengths')
            loss=(value*weight).sum()/weight.sum()
        else:
            points=torch.cat((fixed('gamma'),fixed('band')),dim=1)
            distance=point_triangle_distance_sq(points.reshape(-1,3),x).min(dim=-1).values.reshape(-1,15)
            weight=fixed('interface_lengths')
            gamma=(distance[:,:5].mean(-1)*weight).sum()/weight.sum()
            band=(distance[:,5:].mean(-1)*weight).sum()/weight.sum()
            loss=.5*gamma+.5*band
            counts['prediction_surface_edge_fallback_faces']=(_cross(x).square().sum(-1)<=DISTANCE_DENOM_EPS).sum().detach()
    return dict(loss=loss, applicable=True, counts=counts)
