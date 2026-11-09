"""Independent CPU triangle verifier; original finite geometry unchanged."""
import itertools
import math
import numpy as np
from .seams import classify_triangle_pair as _primary_classifier

def _numpy(value):
    return np.asarray(value)

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


def _bump(ledger, name, count=1):
    if ledger is not None:
        ledger[name] = ledger.get(name, 0) + count


def _triangles(value):
    array = _numpy(value)
    if array.ndim == 2 and array.shape[-1] == 9:
        array = array.reshape(-1, 3, 3)
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        raise ValueError('Expected free/context triangles [N,3,3] or [N,9]')
    if not np.isfinite(array).all():
        raise ValueError('Geometry coordinates must be finite')
    return array


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
