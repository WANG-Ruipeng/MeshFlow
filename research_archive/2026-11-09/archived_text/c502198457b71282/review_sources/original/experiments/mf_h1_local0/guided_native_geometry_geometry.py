"""Faithful original potentials and independent CPU triangle verification.

Original V/S are deliberately not promoted into separation guarantees. The
independent detector uses bounded edge/triangle intersections and barycentric
interior witnesses; it does not reuse the primary detector's plane-cut interval
construction or its coplanar polygon clipping. Unknown cases remain unknown.
"""
from __future__ import annotations

import itertools
import json
import math
import time
import numpy as np
import torch

from .native_patch_seams import (
    classify_triangle_pair as _primary_classifier,
    extract_boundary_segments, nearest_segments,
    triangle_soup_topology, point_to_triangle_surface_squared,
)


ILLEGAL_CLASSES = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))
DETECTOR_POLICY = {
    'dtype': 'float64', 'device': 'cpu',
    'distance_epsilon_bbox': 1e-9,
    'primary_area_epsilon_bbox_squared': 1e-12,
    'independent_near_degenerate_area_bbox_squared': 1e-11,
    'positive_coplanar_area_epsilon_bbox_squared': 1e-12,
    'relative_parallel_determinant_epsilon': 1e-12,
    'independent_algorithm': 'Six finite edges against opposite triangles, barycentric interior witness; coplanar edge intersections and contained vertices with convex hull area',
    'primary_algorithm': 'Unchanged native_patch_seams plane-cut intervals and 2D polygon clipping',
    'uncertain_policy': 'UNRESOLVED; never implicitly disjoint; verifier rejects classification disagreement',
    'continuous_collision_safety': 'NOT_RUN',
}


def initial_geometry_ledger():
    return dict(
        original_V_evaluations=0, original_S_evaluations=0,
        geometry_autograd_attempts=0, geometry_autograd_calls=0,
        primary_pair_tests=0, independent_pair_tests=0, verified_pair_tests=0,
        detector_disagreements=0, detector_unresolved=0,
        independent_AABB_disjoint=0, fixture_runs=0,
        model_forwards=0, model_backward_calls=0, optimizer_updates=0,
    )


def _bump(ledger, name, count=1):
    if ledger is not None:
        ledger[name] = ledger.get(name, 0) + count


def _numpy(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('This module accepts CPU geometry only')
        value = value.detach().numpy()
    return np.asarray(value)


def _triangles(value):
    array = _numpy(value)
    if array.ndim == 2 and array.shape[-1] == 9:
        array = array.reshape(-1, 3, 3)
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        raise ValueError('Expected free/context triangles [N,3,3] or [N,9]')
    if not np.isfinite(array).all():
        raise ValueError('Geometry coordinates must be finite')
    return array


def _grad(energy, variable, ledger, **kwargs):
    _bump(ledger, 'geometry_autograd_attempts')
    gradient, = torch.autograd.grad(energy, variable, create_graph=False, **kwargs)
    _bump(ledger, 'geometry_autograd_calls')
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError('Nonfinite CPU geometry gradient')
    return gradient


def original_vertex_repel(free, context, epsilon, ledger=None):
    """V_repel_original, exactly sum inside*relu(epsilon-|signed d|)^2.

    Barycentric membership is a hard current-coordinate indicator. It is not a
    triangle penetration predicate. The abs derivative at exact d=0 is whatever
    the actual local torch FP64 autograd returns (zero in the registered test).
    ``gradient_np`` is detached, CPU, FP64, shape [Nfree,3,3].
    """
    _bump(ledger, 'original_V_evaluations')
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    f, c = _triangles(free), _triangles(context)
    if not len(c):
        raise ValueError('Original vertex repel needs at least one known triangle')
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cpu', enabled=False):
        x = torch.tensor(f, dtype=torch.float64, device='cpu', requires_grad=True)
        target = torch.tensor(c, dtype=torch.float64, device='cpu')
        ab, ac = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
        cross = torch.linalg.cross(ab, ac)
        normal_squared = cross.square().sum(-1)
        if bool((normal_squared <= torch.finfo(torch.float64).tiny).any()):
            raise ValueError('UNRESOLVED: zero-area C has no defined original plane normal')
        normal = cross / normal_squared.sqrt()[:, None]
        relative = x[:, :, None] - target[None, None, :, 0]
        distance = (relative * normal[None, None]).sum(-1)
        aa, bb, cc = ab.square().sum(-1), (ab * ac).sum(-1), ac.square().sum(-1)
        ap, cp = (relative * ab[None, None]).sum(-1), (relative * ac[None, None]).sum(-1)
        bary_b = (cc[None, None] * ap - bb[None, None] * cp) / normal_squared[None, None]
        bary_c = (aa[None, None] * cp - bb[None, None] * ap) / normal_squared[None, None]
        inside = (bary_b >= 0) & (bary_c >= 0) & (bary_b + bary_c <= 1)
        band = torch.relu(epsilon - distance.abs())
        term = inside.to(torch.float64) * band.square()
        energy = term.sum()
        gradient = _grad(energy, x, ledger)
        active = inside & (distance.abs() < epsilon)
        # Independent decomposition of the original derivative, useful when
        # auditing the full sum's update without attributing summed forces to
        # a single pair. This does not add autograd calls.
        derivative = -2 * inside.to(torch.float64) * band * distance.sign()
        pair_gradient = derivative[..., None] * normal[None, None]
        total_reference = pair_gradient.sum(2)
        error = float((gradient - total_reference).abs().max()) if gradient.numel() else 0.0
        if not torch.allclose(gradient, total_reference, atol=1e-10, rtol=1e-10):
            raise RuntimeError('Original V closed derivative disagrees with actual autograd')
        result = dict(
            name='V_repel_original', energy=float(energy.detach()),
            gradient_np=gradient.detach().numpy().copy(),
            gradient_norm=float(torch.linalg.vector_norm(gradient)),
            active_vertex_count=int(active.any(2).sum()),
            active_vertex_triangle_pair_count=int(active.sum()),
            inside_projection_count=int(inside.sum()), epsilon=epsilon,
            pair_energy=term.sum(1).detach().numpy().copy(),
            pair_active_vertex_count=active.sum(1).detach().numpy().copy(),
            pair_gradient_norm=pair_gradient.permute(0, 2, 1, 3).square().sum((-1, -2)).sqrt().detach().numpy().copy(),
            signed_distance=distance.detach().numpy().copy(),
            inside_projection=inside.detach().numpy().copy(),
            analytic_vs_autograd_max_absolute_error=error,
            actual_coordinate_dtype='torch.float64', actual_gradient_dtype='torch.float64',
            device='cpu', geometry_autograd_calls=1, model_backward_calls=0,
            triangle_intersection_is_not_tested_by_this_energy=True,
        )
    return result


def original_seam_near(free, boundary_segments, near_delta, ledger=None):
    """S_near_original: current near-vertex indicator times min segment d^2."""
    _bump(ledger, 'original_S_evaluations')
    f = _triangles(free)
    segments = _numpy(boundary_segments)
    if segments.ndim != 3 or segments.shape[-2:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError('Finite boundary segments [E,2,3] required')
    near_delta = float(near_delta)
    if not math.isfinite(near_delta) or near_delta <= 0:
        raise ValueError('near_delta must be positive and finite')
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cpu', enabled=False):
        x = torch.tensor(f, dtype=torch.float64, device='cpu', requires_grad=True)
        if len(segments):
            s = torch.tensor(segments, dtype=torch.float64, device='cpu')
            direction = s[:, 1] - s[:, 0]
            denominator = direction.square().sum(-1)
            if bool((denominator <= 0).any()):
                raise ValueError('Zero-length segment in fixed boundary')
            delta = x[:, :, None] - s[None, None, :, 0]
            alpha = ((delta * direction[None, None]).sum(-1) / denominator[None, None]).clamp(0, 1)
            closest = s[None, None, :, 0] + alpha[..., None] * direction[None, None]
            squared = (x[:, :, None] - closest).square().sum(-1)
            minimum, which = squared.min(-1)
            active = minimum < near_delta * near_delta
            energy = (minimum * active.to(torch.float64)).sum()
            nearest_np = torch.gather(closest, 2, which[..., None, None].expand(-1, -1, 1, 3)).squeeze(2).detach().numpy().copy()
            distance_np = minimum.sqrt().detach().numpy().copy()
            which_np = which.detach().numpy().copy()
        else:
            active = torch.zeros(x.shape[:2], dtype=torch.bool)
            energy = (x * 0).sum()
            nearest_np = distance_np = which_np = None
        gradient = _grad(energy, x, ledger)
        result = dict(
            name='S_near_original', energy=float(energy.detach()),
            gradient_np=gradient.detach().numpy().copy(),
            gradient_norm=float(torch.linalg.vector_norm(gradient)),
            active_vertex_count=int(active.sum()), near_delta=near_delta,
            active_mask=active.detach().numpy().copy(),
            nearest_point=nearest_np, nearest_segment=which_np,
            vertex_distance=distance_np,
            no_boundary=len(segments) == 0, no_candidates=int(active.sum()) == 0,
            actual_coordinate_dtype='torch.float64', actual_gradient_dtype='torch.float64',
            device='cpu', geometry_autograd_calls=1, model_backward_calls=0,
            coverage_is_not_tested_by_this_energy=True,
        )
    return result


def classify_triangle_pair_primary(first, second, bbox_diagonal, ledger=None):
    _bump(ledger, 'primary_pair_tests')
    result = dict(_primary_classifier(first, second, bbox_diagonal))
    resolved = result['classification'] not in ('degenerate_pair', 'near_parallel_unresolved', 'UNRESOLVED')
    result.update(resolved=resolved,
                  illegal=result['classification'] in ILLEGAL_CLASSES if resolved else None,
                  detector='unchanged_primary')
    return result


def _barycentric(point, triangle):
    ab, ac, ap = triangle[1] - triangle[0], triangle[2] - triangle[0], point - triangle[0]
    aa, bb, cc = float(ab @ ab), float(ab @ ac), float(ac @ ac)
    denominator = float(np.cross(ab, ac) @ np.cross(ab, ac))
    u = (cc * float(ap @ ab) - bb * float(ap @ ac)) / denominator
    v = (aa * float(ap @ ac) - bb * float(ap @ ab)) / denominator
    return np.array([1 - u - v, u, v])


def _inside_distances(point, triangle):
    weights = _barycentric(point, triangle)
    double_area = float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
    opposite_edges = np.array([np.linalg.norm(triangle[2] - triangle[1]),
                               np.linalg.norm(triangle[0] - triangle[2]),
                               np.linalg.norm(triangle[1] - triangle[0])])
    return weights * double_area / opposite_edges


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _point_on_segment2(point, start, end, epsilon):
    direction = end - start
    denominator = float(direction @ direction)
    if denominator == 0:
        return 0.0 if np.linalg.norm(point - start) <= epsilon else None
    value = float((point - start) @ direction / denominator)
    clamped = min(1.0, max(0.0, value))
    return clamped if np.linalg.norm(point - start - clamped * direction) <= epsilon else None


def _segment_intersections2(a, b, c, d, epsilon):
    """Return intersection points and A-segment parameters; include overlap ends."""
    u, v = b - a, d - c
    denominator = _cross2(u, v)
    scale = float(np.linalg.norm(u) * np.linalg.norm(v))
    rows = []
    if abs(denominator) > 1e-12 * scale:
        relative = c - a
        s, t = _cross2(relative, v) / denominator, _cross2(relative, u) / denominator
        # Geometric epsilon is converted into segment parameter tolerance.
        su = epsilon / max(float(np.linalg.norm(u)), np.finfo(float).tiny)
        tv = epsilon / max(float(np.linalg.norm(v)), np.finfo(float).tiny)
        if -su <= s <= 1 + su and -tv <= t <= 1 + tv:
            s = min(1.0, max(0.0, s))
            rows.append((a + s * u, s))
    else:
        for point in (a, b, c, d):
            s = _point_on_segment2(point, a, b, epsilon)
            t = _point_on_segment2(point, c, d, epsilon)
            if s is not None and t is not None:
                rows.append((a + s * u, s))
    return rows


def _deduplicate(points, epsilon):
    result = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if all(float(np.linalg.norm(point - old)) > epsilon for old in result):
            result.append(point.copy())
    return result


def _convex_hull2(points):
    unique = sorted({tuple(map(float, p)) for p in points})
    if len(unique) <= 1:
        return [np.asarray(p) for p in unique]
    lower, upper = [], []
    for sequence, destination in ((unique, lower), (unique[::-1], upper)):
        for p in sequence:
            p = np.asarray(p)
            while len(destination) >= 2 and _cross2(destination[-1] - destination[-2], p - destination[-1]) <= 0:
                destination.pop()
            destination.append(p)
    return lower[:-1] + upper[:-1]


def _coplanar_intersection(a, b, normal, epsilon):
    keep = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
    hits = []
    for point in a:
        if np.min(_inside_distances(point, b)) >= -epsilon:
            hits.append(point)
    for point in b:
        if np.min(_inside_distances(point, a)) >= -epsilon:
            hits.append(point)
    for i in range(3):
        for j in range(3):
            for _, alpha in _segment_intersections2(a[i, keep], a[(i + 1) % 3, keep],
                                                    b[j, keep], b[(j + 1) % 3, keep], epsilon):
                hits.append(a[i] + alpha * (a[(i + 1) % 3] - a[i]))
    hits = _deduplicate(hits, epsilon)
    hull = _convex_hull2([p[keep] for p in hits])
    area_projected = 0.0
    if len(hull) >= 3:
        translated = [p - hull[0] for p in hull]
        area_projected = abs(sum(_cross2(translated[i], translated[(i + 1) % len(hull)]) for i in range(len(hull)))) / 2
    drop = int(np.argmax(np.abs(normal)))
    return hits, area_projected / abs(float(normal[drop]))


def _edge_triangle_hits(start, end, triangle, epsilon):
    """Finite-segment barycentric solve, including segment-in-plane contacts."""
    direction = end - start
    e1, e2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
    normal = np.cross(e1, e2)
    pvec = np.cross(direction, e2)
    determinant = float(e1 @ pvec)
    scale = float(np.linalg.norm(direction) * np.linalg.norm(normal))
    if abs(determinant) > 1e-12 * scale:
        relative = start - triangle[0]
        qvec = np.cross(relative, e1)
        u = float(relative @ pvec) / determinant
        v = float(direction @ qvec) / determinant
        alpha = float(e2 @ qvec) / determinant
        edge_tolerance = epsilon / max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
        if -edge_tolerance <= alpha <= 1 + edge_tolerance:
            alpha = min(1.0, max(0.0, alpha))
            point = start + alpha * direction
            if np.min(_inside_distances(point, triangle)) >= -epsilon:
                return [point]
        return []
    unit = normal / np.linalg.norm(normal)
    if abs(float((start - triangle[0]) @ unit)) > epsilon or abs(float((end - triangle[0]) @ unit)) > epsilon:
        return []
    keep = [i for i in range(3) if i != int(np.argmax(np.abs(unit)))]
    hits = []
    for point in (start, end):
        if np.min(_inside_distances(point, triangle)) >= -epsilon:
            hits.append(point)
    for i in range(3):
        for _, alpha in _segment_intersections2(start[keep], end[keep], triangle[i, keep], triangle[(i + 1) % 3, keep], epsilon):
            hits.append(start + alpha * direction)
    return hits


def _shared_count(a, b, epsilon):
    allowed = np.linalg.norm(a[:, None] - b[None], axis=-1) <= epsilon
    return max(sum(bool(allowed[i, p[i]]) for i in range(3)) for p in itertools.permutations(range(3)))


def classify_triangle_pair_independent(first, second, bbox_diagonal, ledger=None):
    _bump(ledger, 'independent_pair_tests')
    a0, b0 = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    length = float(bbox_diagonal)
    if a0.shape != (3, 3) or b0.shape != (3, 3) or not np.isfinite(a0).all() or not np.isfinite(b0).all() or not math.isfinite(length) or length <= 0:
        raise ValueError('Independent detector needs finite triangles and positive bbox scale')
    origin = a0[0].copy()
    a, b = (a0 - origin) / length, (b0 - origin) / length
    eps = 1e-9
    result = dict(classification='disjoint', resolved=True, illegal=False,
                  proper_transverse=False, intersection_length=0.0, intersection_points=[],
                  coplanar_overlap_area=0.0, contact_dimension=None,
                  detector='independent_edge_triangle_barycentric', normalized_by_bbox=True,
                  epsilon=eps * length, reason='No bounded triangle intersection')

    def unresolved(reason):
        result.update(classification='UNRESOLVED', resolved=False, illegal=None,
                      proper_transverse=False, reason=reason)
        _bump(ledger, 'detector_unresolved')
        return result

    n_a, n_b = np.cross(a[1] - a[0], a[2] - a[0]), np.cross(b[1] - b[0], b[2] - b[0])
    norm_a, norm_b = float(np.linalg.norm(n_a)), float(np.linalg.norm(n_b))
    if min(norm_a, norm_b) / 2 <= 1e-11:
        return unresolved('Near-degenerate triangle area <=1e-11 L^2; no clean classification')
    if np.any(a.max(0) < b.min(0) - eps) or np.any(b.max(0) < a.min(0) - eps):
        _bump(ledger, 'independent_AABB_disjoint')
        return result
    n_a, n_b = n_a / norm_a, n_b / norm_b
    coplanar = (np.max(np.abs((b - a[0]) @ n_a)) <= eps
                and np.max(np.abs((a - b[0]) @ n_b)) <= eps)
    shared = _shared_count(a, b, eps)
    result['shared_vertices_within_epsilon'] = int(shared)
    if coplanar:
        hits, area = _coplanar_intersection(a, b, n_a, eps)
        result.update(intersection_points=[(p * length + origin).tolist() for p in hits],
                      coplanar_overlap_area=float(area * length * length))
        if area > 1e-12:
            result.update(classification='coplanar_positive_area_overlap', illegal=True,
                          contact_dimension=2, reason='Independent coplanar intersection hull has positive area')
            return result
        if not hits:
            return result
        diameter = max((float(np.linalg.norm(p - q)) for p in hits for q in hits), default=0.0)
        result.update(intersection_length=diameter * length,
                      contact_dimension=1 if diameter > eps else 0)
        if shared >= 2:
            label = 'shared_edge_contact'
        elif shared == 1:
            label = 'shared_vertex_contact'
        else:
            label = 'coplanar_boundary_or_tolerance_contact'
        result.update(classification=label, reason='Coplanar boundary-only contact; inspect contact dimension for partial-edge/T-junction contacts')
        return result
    if float(np.linalg.norm(np.cross(n_a, n_b))) <= 1e-12:
        # Avoid treating a nearly parallel ill-conditioned pair as clear.
        return unresolved('Non-coplanar classification with near-parallel plane normals')
    hits = []
    for i in range(3):
        hits.extend(_edge_triangle_hits(a[i], a[(i + 1) % 3], b, eps))
        hits.extend(_edge_triangle_hits(b[i], b[(i + 1) % 3], a, eps))
    hits = _deduplicate(hits, eps)
    if not hits:
        return result
    result['intersection_points'] = [(p * length + origin).tolist() for p in hits]
    if len(hits) == 1:
        result.update(classification='shared_vertex_contact' if shared else 'point_or_tangent_contact',
                      contact_dimension=0, reason='Only a point contact is independently witnessed')
        return result
    _, i, j = max((float(np.linalg.norm(p - q)), i, j) for i, p in enumerate(hits) for j, q in enumerate(hits))
    diameter = float(np.linalg.norm(hits[i] - hits[j]))
    result['intersection_length'] = diameter * length
    if diameter <= eps:
        result.update(classification='shared_vertex_contact' if shared else 'point_or_tangent_contact',
                      contact_dimension=0, reason='Intersection point diameter is below distance tolerance')
        return result
    middle = (hits[i] + hits[j]) / 2
    interior_a, interior_b = _inside_distances(middle, a), _inside_distances(middle, b)
    result.update(interior_witness=(middle * length + origin).tolist(),
                  witness_edge_distances_first=(interior_a * length).tolist(),
                  witness_edge_distances_second=(interior_b * length).tolist(), contact_dimension=1)
    if np.min(interior_a) > eps and np.min(interior_b) > eps:
        result.update(classification='proper_transverse_penetration', illegal=True,
                      proper_transverse=True, reason='Positive-length segment has a witness strictly inside both triangles')
    else:
        result.update(classification='shared_edge_contact' if shared >= 2 else 'noncoplanar_edge_or_tangent_contact',
                      reason='Intersection remains on at least one triangle boundary')
    return result


def verified_pair(first, second, bbox_diagonal, ledger=None):
    """Both independent constructions must resolve and agree before acceptance."""
    _bump(ledger, 'verified_pair_tests')
    primary = classify_triangle_pair_primary(first, second, bbox_diagonal, ledger)
    independent = classify_triangle_pair_independent(first, second, bbox_diagonal, ledger)
    resolved = bool(primary['resolved'] and independent['resolved'])
    agrees = bool(resolved and primary['classification'] == independent['classification'])
    if resolved and not agrees:
        _bump(ledger, 'detector_disagreements')
    if not agrees:
        return dict(classification='UNRESOLVED', resolved=False, verified=False, illegal=None,
                    reason='Primary/independent classification disagreement or uncertainty',
                    primary=primary, independent=independent)
    return dict(classification=primary['classification'], resolved=True, verified=True,
                illegal=primary['illegal'], proper_transverse=primary['proper_transverse'],
                intersection_length=independent['intersection_length'],
                intersection_points=independent['intersection_points'],
                primary=primary, independent=independent)


def _potential_summary(result):
    return {key: value for key, value in result.items()
            if key not in ('gradient_np', 'pair_energy', 'pair_active_vertex_count', 'pair_gradient_norm',
                           'signed_distance', 'inside_projection', 'active_mask',
                           'nearest_point', 'nearest_segment', 'vertex_distance')}


def _triangle_areas(triangles):
    t = np.asarray(triangles, dtype=np.float64)
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=-1) / 2


def run_fixed_fixtures(ledger=None):
    """Return (JSON-safe nine-family report, small NPZ-ready arrays).

    Reproducing original-potential counterexamples is PASS. The run must stop
    later stages only if an analytic detector expectation fails, not because a
    proposed energy proves insufficient. No model/GPU/optimizer is used.
    """
    if ledger is None:
        ledger = initial_geometry_ledger()
    before = dict(ledger)
    _bump(ledger, 'fixture_runs')
    tick = time.perf_counter()
    length, epsilon = math.sqrt(2), .1
    c = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    f1 = np.array([[[.2, .2, 1], [.6, .2, -1], [.2, .6, 1]]], dtype=np.float64)
    segments, boundary = extract_boundary_segments(c)
    rows, arrays = [], {'context': c.copy(), 'boundary_segments': segments.copy()}

    def expect_pair(first, second, expected):
        result = verified_pair(first, second, length, ledger)
        if expected == 'UNRESOLVED':
            assert not result['resolved'] and result['illegal'] is None, result
        else:
            assert result['verified'] and result['classification'] == expected, result
        return result

    # 1: Whole-triangle penetration with every vertex outside the activation band.
    detector1 = expect_pair(c[0], f1[0], 'proper_transverse_penetration')
    potential1 = original_vertex_repel(f1, c, epsilon, ledger)
    assert potential1['energy'] == 0 and potential1['gradient_norm'] == 0
    assert potential1['active_vertex_count'] == 0
    rows.append(dict(id=1, name='outside_band_vertices_inside_triangle_penetration', status='PASS_COUNTEREXAMPLE',
                     potential=_potential_summary(potential1), detector=detector1,
                     conclusion='Full triangle intersects even though original V has zero energy and gradient'))
    arrays['fixture1_free'] = f1.copy()

    # 2: One faithful original update decreases V without eliminating penetration.
    f2 = f1.copy(); f2[0, :, 2] = [.02, -.02, .02]
    v2 = original_vertex_repel(f2, c, epsilon, ledger)
    after2 = f2 - .02 * 20 * v2['gradient_np']
    v2_after = original_vertex_repel(after2, c, epsilon, ledger)
    d2_before = expect_pair(c[0], f2[0], 'proper_transverse_penetration')
    d2_after = expect_pair(c[0], after2[0], 'proper_transverse_penetration')
    assert v2_after['energy'] < v2['energy']
    rows.append(dict(id=2, name='lower_original_energy_persistent_penetration', status='PASS_COUNTEREXAMPLE',
                     h=.02, lambda_value=20, before=_potential_summary(v2), after=_potential_summary(v2_after),
                     detector_before=d2_before, detector_after=d2_after,
                     area_before=float(_triangle_areas(f2)[0]), area_after=float(_triangle_areas(after2)[0]),
                     conclusion='Energy decrease is not defect decrease'))
    arrays.update(fixture2_before=f2.copy(), fixture2_after=after2.copy())

    # 3: Actual abs autograd at the plane, rather than a selected subgradient.
    f3 = np.array([[[.2, .2, 0], [.3, .2, 0], [.2, .3, 0]]], dtype=np.float64)
    v3 = original_vertex_repel(f3, c, epsilon, ledger)
    assert v3['energy'] > 0 and v3['gradient_norm'] == 0
    rows.append(dict(id=3, name='exact_plane_abs_autograd', status='PASS_COUNTEREXAMPLE',
                     potential=_potential_summary(v3), actual_gradient=v3['gradient_np'].tolist(),
                     conclusion='Positive original V has zero local autograd gradient at d=0'))
    arrays['fixture3_free'] = f3.copy()

    # 4: Both endpoints clear and inactive; the specified linear motion crosses.
    start4, end4 = f3 + [0, 0, .2], f3 + [0, 0, -.2]
    start_v = original_vertex_repel(start4, c, epsilon, ledger)
    end_v = original_vertex_repel(end4, c, epsilon, ledger)
    start_d = expect_pair(c[0], start4[0], 'disjoint')
    end_d = expect_pair(c[0], end4[0], 'disjoint')
    middle_d = expect_pair(c[0], ((start4 + end4) / 2)[0], 'coplanar_positive_area_overlap')
    assert start_v['energy'] == end_v['energy'] == 0
    rows.append(dict(id=4, name='step_interior_crossing_clear_endpoints', status='PASS_COUNTEREXAMPLE',
                     endpoint_before=start_d, endpoint_after=end_d, analytic_midpoint=middle_d,
                     start_potential=_potential_summary(start_v), end_potential=_potential_summary(end_v),
                     crossing_fraction=.5, crossing_fact='Each vertex z(t)=.2-.4*t changes sign',
                     continuous_collision_detection='NOT_RUN',
                     conclusion='Endpoint acceptance cannot certify a collision-free full step'))
    arrays.update(fixture4_start=start4.copy(), fixture4_end=end4.copy())

    # 5: Three legal-contact constructions must remain non-illegal.
    legal = {
        'coplanar_shared_edge': (np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]], dtype=float), 'shared_edge_contact'),
        'right_angle_shared_edge': (np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float), 'shared_edge_contact'),
        'only_shared_vertex': (np.array([[0, 0, 0], [-1, 0, 1], [0, -1, 1]], dtype=float), 'shared_vertex_contact'),
    }
    legal_results = {}
    for label, (triangle, expected) in legal.items():
        legal_results[label] = expect_pair(c[0], triangle, expected)
        assert legal_results[label]['illegal'] is False
        arrays['fixture5_' + label] = triangle[None].copy()
    rows.append(dict(id=5, name='legal_contacts', status='PASS_DETECTORS', cases=legal_results))

    # 6: Positive-area coplanar overlap is illegal, despite no transverse cut.
    d6 = expect_pair(c[0], f3[0], 'coplanar_positive_area_overlap')
    assert d6['independent']['coplanar_overlap_area'] > 0 and d6['illegal']
    rows.append(dict(id=6, name='coplanar_positive_area_overlap', status='PASS_DETECTORS', detector=d6))

    # 7: C has AB; free triangles have AM and MB. Coordinate coverage is exact,
    # but the unmodified triangle soup does not acquire a common AB edge.
    f7 = np.array([[[0, 0, 0], [.5, 0, 0], [0, -.5, 0]],
                   [[.5, 0, 0], [1, 0, 0], [1, -.5, 0]]], dtype=np.float64)
    junction_pairs = [expect_pair(c[0], triangle, 'shared_vertex_contact') for triangle in f7]
    probes = np.column_stack([np.linspace(0, 1, 33), np.zeros(33), np.zeros(33)])
    distance7 = point_to_triangle_surface_squared(probes, f7, length)
    topology7 = triangle_soup_topology(np.concatenate([c, f7]), length)
    assert float(distance7.max()) < 1e-24 and topology7['boundary_edges'] == 9
    rows.append(dict(id=7, name='T_junction_zero_gap_not_shared_discrete_edge', status='PASS_COUNTEREXAMPLE',
                     AB_probe_count=33, AB_max_surface_distance=float(np.sqrt(distance7.max())),
                     topology=topology7, detectors=junction_pairs,
                     fixed_edge='AB', free_edges=['AM', 'MB'], topology_closed=False,
                     conclusion='Geometric line coverage does not split AB or weld the indexed connectivity'))
    arrays['fixture7_free'] = f7.copy()

    # 8: A full nearest-line attraction step flattens a genuine triangle.
    f8 = np.array([[[.2, .02, .04], [.4, .03, .05], [.6, .01, .025]]], dtype=np.float64)
    s8 = original_seam_near(f8, segments, .1, ledger)
    after8 = f8 - .5 * s8['gradient_np']
    s8_after = original_seam_near(after8, segments, .1, ledger)
    assert float(_triangle_areas(f8)[0]) > 0 and float(_triangle_areas(after8)[0]) <= 1e-24
    assert s8_after['energy'] == 0
    unresolved8 = expect_pair(c[0], after8[0], 'UNRESOLVED')
    rows.append(dict(id=8, name='full_attraction_projection_collapses_area', status='PASS_COUNTEREXAMPLE',
                     original_seam_before=_potential_summary(s8), original_seam_after=_potential_summary(s8_after),
                     diagnostic_gradient_step=.5, area_before=float(_triangle_areas(f8)[0]),
                     area_after=float(_triangle_areas(after8)[0]), detector_after=unresolved8,
                     note='Counterexample full attraction step; not the registered bounded S beta=.25 corrector'))
    arrays.update(fixture8_before=f8.copy(), fixture8_after=after8.copy())

    # 9: Both forces have normal components; local weighted scalar equilibrium
    # stays away from the plane. Verify actual torch derivative of that model.
    f9 = np.array([[[.2, .02, .03], [.4, .025, .03], [.6, .015, .03]]], dtype=np.float64)
    v9 = original_vertex_repel(f9, c, epsilon, ledger)
    s9 = original_seam_near(f9, segments, .2, ledger)
    repel, attract = -20 * v9['gradient_np'], -10 * s9['gradient_np']
    flat_r, flat_s = repel.reshape(-1), attract.reshape(-1)
    cosine = float(flat_r @ flat_s / (np.linalg.norm(flat_r) * np.linalg.norm(flat_s)))
    angle = math.degrees(math.acos(max(-1, min(1, cosine))))
    equilibrium = 20 * epsilon / (20 + 10)
    with torch.enable_grad(), torch.autocast('cpu', enabled=False):
        z = torch.tensor(equilibrium, dtype=torch.float64, requires_grad=True)
        scalar_energy = 20 * torch.relu(epsilon - z.abs()).square() + 10 * z.square()
        scalar_gradient = _grad(scalar_energy, z, ledger)
    assert angle > 90 and equilibrium > 0 and abs(float(scalar_gradient)) < 1e-12
    rows.append(dict(id=9, name='repulsion_attraction_conflict_and_nonzero_equilibrium', status='PASS_COUNTEREXAMPLE',
                     lambda_repel=20, lambda_seam=10, force_angle_degrees=angle,
                     repel_normal_components=repel[..., 2].tolist(),
                     attract_normal_components=attract[..., 2].tolist(),
                     combined_force=(repel + attract).tolist(),
                     scalar_equilibrium_distance=equilibrium,
                     scalar_equilibrium_gradient=float(scalar_gradient),
                     scalar_formula='20*relu(.1-|z|)^2+10*z^2',
                     conclusion='Opposing forces can balance at nonzero gap; they are not assumed orthogonal'))
    arrays['fixture9_free'] = f9.copy()

    # Independent detector condition-number guard is part of the fixed suite.
    near_degenerate = np.array([[.2, .2, -.1], [.2 + 1e-12, .2, .1], [.2, .2 + 1e-12, .1]])
    near_result = expect_pair(c[0], near_degenerate, 'UNRESOLVED')
    arrays['near_degenerate_detector_probe'] = near_degenerate[None].copy()
    # Same analytic proper triangle, six vertex permutations on each side.
    permutation_calls = 0
    for first in itertools.permutations(range(3)):
        for second in itertools.permutations(range(3)):
            expect_pair(c[0, list(first)], f1[0, list(second)], 'proper_transverse_penetration')
            permutation_calls += 1
    report = dict(
        status='PASS', stage='A_pure_geometry', fixed_fixture_families=9, fixtures=rows,
        detector_validation=dict(status='PASS', near_degenerate=near_result,
                                 proper_double_vertex_permutations=permutation_calls),
        detector_policy=dict(DETECTOR_POLICY), fixture_epsilon=.1,
        fixture_bbox_diagonal=length, boundary=boundary,
        real_original_V_epsilon_fraction=.005, real_original_S_near_delta_fraction=.005,
        counterexample_reproduction_is_success=True, actual_counts={
            key: ledger.get(key, 0) - before.get(key, 0) for key in ledger},
        model_forwards=0, model_backward_calls=0, optimizer_updates=0,
        continuous_collision_safety='NOT_RUN', seconds=time.perf_counter() - tick,
    )
    json.dumps(report, allow_nan=False)
    return report, arrays


# Explicit aliases retain clear original-formula names in callers and reports.
V_repel_original = original_vertex_repel
S_near_original = original_seam_near
run_stage_a_fixed_fixtures = run_fixed_fixtures


if __name__ == '__main__':
    result, _ = run_fixed_fixtures()
    print(json.dumps(result, indent=2))
