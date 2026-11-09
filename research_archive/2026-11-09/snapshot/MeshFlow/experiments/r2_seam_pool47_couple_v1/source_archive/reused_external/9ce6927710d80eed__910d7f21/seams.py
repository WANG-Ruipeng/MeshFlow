"""CPU-only patch boundary, C/free contact, and optional export diagnostics.

No training, model, optimizer, torch, mesh repairs, or file mutation occurs here.
The intersection diagnostic covers C-versus-free pairs only, not free/free
self-intersections or a complete topology certificate. Its plane/interval test
follows the geometric construction in Moller (1997); the contact taxonomy and
fixed tolerances below are explicit additions, not exact-predicate guarantees.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import time
import numpy as np


NUMERIC_POLICY = {
    'computation_dtype': 'float64',
    'intersection_distance_epsilon_bbox': 1e-9,
    'degenerate_triangle_area_epsilon_bbox_squared': 1e-12,
    'coplanar_positive_overlap_area_epsilon_bbox_squared': 1e-12,
    'parallel_unit_normal_cross_epsilon': 1e-12,
    'near_boundary_threshold_bbox': 1e-3,
    'near_boundary_comparison': 'strictly less than threshold',
    'boundary_sampling_target': 512,
    'minimum_samples_per_segment': 2,
    'boundary_sample_seed': 9919,
    'exact_topology_vertex_key': 'Exact numeric FP64 coordinate tuple; signed zeros compare equal; no tolerance weld',
    'reference': 'https://web.stanford.edu/class/cs277/resources/papers/Moller1997b.pdf',
}


def _triangles(value):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[-1] == 9:
        return a.reshape(-1, 3, 3)
    if a.ndim == 3 and a.shape[-2:] == (3, 3):
        return a
    raise ValueError('Expected triangles [N,9] or [N,3,3]')


def _bbox_scale(value):
    length = float(value)
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Require a positive finite fixed complete-target bbox diagonal')
    return length


def _hash(a):
    value = np.ascontiguousarray(a)
    return hashlib.sha256(str((str(value.dtype), value.shape)).encode() + value.tobytes()).hexdigest()


def _index_soup(triangles):
    triangles = _triangles(triangles)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite coordinates cannot define exact-coordinate topology')
    vertices, mapping, faces = [], {}, []
    for triangle in triangles:
        face = []
        for vertex in triangle:
            key = tuple(float(q) for q in vertex)
            if key not in mapping:
                mapping[key] = len(vertices)
                vertices.append(vertex.copy())
            face.append(mapping[key])
        faces.append(face)
    v = np.asarray(vertices, dtype=triangles.dtype).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    incidences = defaultdict(list)
    for face_index, face in enumerate(f):
        for corner in range(3):
            a, b = int(face[corner]), int(face[(corner + 1) % 3])
            if a != b:
                incidences[tuple(sorted((a, b)))].append((face_index, a, b))
    return v, f, incidences


def triangle_soup_topology(triangles, bbox_diagonal=None):
    """Exact-coordinate indexing audit, not a watertight/manifold certification."""
    t = _triangles(triangles)
    v, f, edges = _index_soup(t)
    counts = Counter(len(incident) for incident in edges.values())
    repeated_vertex = sum(len(set(face.tolist())) < 3 for face in f)
    duplicates = Counter(tuple(sorted(face.tolist())) for face in f)
    same_direction = sum(
        len(entries) == 2 and entries[0][1:] == entries[1][1:]
        for entries in edges.values())
    area_degenerate = None
    if bbox_diagonal is not None:
        length = _bbox_scale(bbox_diagonal)
        area = np.linalg.norm(np.cross(t[:, 1].astype(float) - t[:, 0],
                                       t[:, 2].astype(float) - t[:, 0]), axis=-1) / 2
        area_degenerate = int((area <= 1e-12 * length * length).sum())
    adjacency = [set() for _ in range(len(f))]
    for entries in edges.values():
        face_ids = {q[0] for q in entries}
        for a in face_ids:
            adjacency[a].update(face_ids - {a})
    visited, components = set(), 0
    for start in range(len(f)):
        if start in visited:
            continue
        components += 1
        pending = [start]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency[current] - visited)
    return dict(
        status='PASS_EXACT_COORDINATE_AUDIT', faces=len(f), unique_vertices=len(v),
        unique_undirected_edges=len(edges), boundary_edges=counts.get(1, 0),
        two_incident_face_edges=counts.get(2, 0),
        nonmanifold_edges=sum(n > 2 for n in (len(q) for q in edges.values())),
        same_orientation_two_face_edges=int(same_direction),
        incidence_histogram={str(k): int(n) for k, n in sorted(counts.items())},
        exact_repeated_vertex_faces=int(repeated_vertex),
        duplicate_faces_ignoring_winding=sum(n - 1 for n in duplicates.values()),
        near_zero_area_faces=area_degenerate, exact_edge_connected_face_components=components,
        indexing=NUMERIC_POLICY['exact_topology_vertex_key'],
        vertices_merged_by_tolerance=False,
        watertight_status='NOT_CERTIFIED', manifold_status='NOT_CERTIFIED',
        full_self_intersection_status='NOT_RUN',
    )


def extract_boundary_segments(context):
    """Return actual one-incident-face C edges, removing shared interior edges.

    Returns ``(segments[E,2,3], audit)``. No convex hull, bbox boundary, nearest
    vertex shortcut, or inferred patch outline substitutes for source edges.
    """
    c = _triangles(context)
    v, faces, incidences = _index_soup(c)
    boundary = [(edge, entries[0]) for edge, entries in sorted(incidences.items()) if len(entries) == 1]
    segments = np.asarray([v[list(edge)] for edge, _ in boundary], dtype=c.dtype).reshape(-1, 2, 3)
    lengths = np.linalg.norm(segments[:, 1].astype(float) - segments[:, 0], axis=-1)
    if np.any(lengths <= 0):
        raise RuntimeError('Exact non-self boundary edges must have nonzero lengths')
    return segments, dict(
        method='Count exact shared C edges; retain precisely incidence=1',
        context_faces=len(faces), boundary_segments=len(segments),
        interior_two_face_edges=sum(len(q) == 2 for q in incidences.values()),
        nonmanifold_context_edges=sum(len(q) > 2 for q in incidences.values()),
        segment_lengths=lengths.tolist(), total_boundary_length=float(lengths.sum()),
        segment_source_faces=[int(entry[0]) for _, entry in boundary],
        segment_exact_vertex_ids=[list(edge) for edge, _ in boundary],
        includes_hole_and_disconnected_boundaries=True,
        source_coordinates_unchanged=True,
    )


def nearest_segments(points, segments):
    """Exact finite-segment nearest points in FP64; never infinite lines."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    s = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if not len(s):
        return None, None, None
    start, direction = s[:, 0], s[:, 1] - s[:, 0]
    denominator = np.einsum('si,si->s', direction, direction)
    numerator = np.einsum('psi,si->ps', p[:, None] - start[None], direction)
    alpha = np.divide(numerator, denominator[None], out=np.zeros_like(numerator), where=denominator[None] > 0)
    alpha = np.clip(alpha, 0, 1)
    closest = start[None] + alpha[..., None] * direction[None]
    squared = ((p[:, None] - closest) ** 2).sum(-1)
    which = squared.argmin(1)
    return squared[np.arange(len(p)), which], closest[np.arange(len(p)), which], which


def point_to_triangle_surface_squared(points, triangles, bbox_diagonal):
    """Minimum to finite triangle interiors/edges; degenerates use their edges."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = _triangles(triangles).astype(np.float64, copy=False)
    length = _bbox_scale(bbox_diagonal)
    if not len(t):
        return None
    a, ab, ac = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    normal = np.cross(ab, ac)
    n2 = np.einsum('ti,ti->t', normal, normal)
    stable = n2 > (2e-12 * length * length) ** 2
    pa = p[:, None] - a[None]
    d00 = np.einsum('ti,ti->t', ab, ab)
    d01 = np.einsum('ti,ti->t', ab, ac)
    d11 = np.einsum('ti,ti->t', ac, ac)
    d20 = np.einsum('pti,ti->pt', pa, ab)
    d21 = np.einsum('pti,ti->pt', pa, ac)
    v = np.divide(d11[None] * d20 - d01[None] * d21, n2[None],
                  out=np.zeros_like(d20), where=stable[None])
    w = np.divide(d00[None] * d21 - d01[None] * d20, n2[None],
                  out=np.zeros_like(d21), where=stable[None])
    inside = stable[None] & (v >= 0) & (w >= 0) & (v + w <= 1)
    signed_n = np.einsum('pti,ti->pt', pa, normal)
    planar = np.divide(signed_n * signed_n, n2[None],
                       out=np.full_like(signed_n, np.inf), where=stable[None])
    best = np.where(inside, planar, np.inf).min(1)
    for edge in ((0, 1), (1, 2), (2, 0)):
        d2, _, _ = nearest_segments(p, t[:, list(edge)])
        best = np.minimum(best, d2)
    return np.maximum(best, 0)


def _sample_boundary(segments, seed, target_count):
    s = np.asarray(segments, dtype=np.float64)
    lengths = np.linalg.norm(s[:, 1] - s[:, 0], axis=-1)
    if not len(s):
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0), []
    counts = np.maximum(2, np.ceil(int(target_count) * lengths / lengths.sum()).astype(np.int64))
    rng = np.random.RandomState(int(seed))
    samples, identities, point_weights = [], [], []
    for index, (segment, count, segment_length) in enumerate(zip(s, counts, lengths)):
        # Different independent uniform draws per segment; endpoints are not
        # forced into an otherwise sampled maximum or treated as full coverage.
        alpha = rng.uniform(size=int(count))
        samples.append(segment[0] + alpha[:, None] * (segment[1] - segment[0]))
        identities.extend([index] * int(count))
        point_weights.extend([float(segment_length / lengths.sum() / count)] * int(count))
    return np.concatenate(samples), np.asarray(identities), np.asarray(point_weights), counts.tolist()


def _distance_stats(distances, length):
    d = np.asarray(distances, dtype=np.float64)
    if not len(d):
        return None
    return dict(count=int(d.size), rms=float(np.sqrt(np.mean(d * d))), max=float(d.max()),
                rms_bbox_fraction=float(np.sqrt(np.mean(d * d)) / length),
                max_bbox_fraction=float(d.max() / length))


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _coplanar_clip(a, b, unit_normal):
    drop = int(np.argmax(np.abs(unit_normal)))
    keep = [i for i in range(3) if i != drop]
    # Translate first to limit cancellation for coordinates far from the origin.
    origin = a[0]
    aa, bb = (a - origin)[:, keep], (b - origin)[:, keep]
    orientation = 1.0 if _cross2(bb[1] - bb[0], bb[2] - bb[0]) >= 0 else -1.0
    polygon = [q.copy() for q in aa]
    for edge in range(3):
        p, q = bb[edge], bb[(edge + 1) % 3]
        if not polygon:
            break
        old, polygon = polygon, []
        previous = old[-1]
        old_sign = orientation * _cross2(q - p, previous - p)
        for current in old:
            sign = orientation * _cross2(q - p, current - p)
            old_inside, inside = old_sign >= 0, sign >= 0
            if old_inside != inside:
                ratio = old_sign / (old_sign - sign)
                polygon.append(previous + ratio * (current - previous))
            if inside:
                polygon.append(current.copy())
            previous, old_sign = current, sign
    if len(polygon) < 3:
        area = 0.0
    else:
        v = np.asarray(polygon) - polygon[0]
        projected = abs(sum(_cross2(v[i], v[(i + 1) % len(v)]) for i in range(len(v)))) / 2
        area = projected / abs(float(unit_normal[drop]))
    return polygon, float(area)


def _segment_pair_distance(a, b, c, d):
    endpoints = np.stack([a, b, c, d])
    da, _, _ = nearest_segments(endpoints[:2], np.stack([c, d])[None])
    db, _, _ = nearest_segments(endpoints[2:], np.stack([a, b])[None])
    best = min(float(da.min()), float(db.min()))
    u, v, w = b - a, d - c, a - c
    aa, ab, bb = float(u @ u), float(u @ v), float(v @ v)
    denominator = float(np.cross(u, v) @ np.cross(u, v))
    if denominator > np.finfo(np.float64).eps * aa * bb:
        aw, bw = float(u @ w), float(v @ w)
        s, t = (ab * bw - bb * aw) / denominator, (aa * bw - ab * aw) / denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            best = min(best, float(np.sum((a + s * u - c - t * v) ** 2)))
    return math.sqrt(max(best, 0.0))


def _plane_cut(triangle, distances, epsilon):
    points = [triangle[i].copy() for i in range(3) if abs(float(distances[i])) <= epsilon]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if (distances[i] > epsilon and distances[j] < -epsilon) or (distances[j] > epsilon and distances[i] < -epsilon):
            points.append(triangle[i] + distances[i] / (distances[i] - distances[j]) * (triangle[j] - triangle[i]))
    return points


def _maximum_unique_matches(allowed):
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.shape != (3, 3):
        raise ValueError('Triangle vertex matching must be 3 by 3')
    return max(sum(bool(allowed[i, p[i]]) for i in range(3))
               for p in itertools.permutations(range(3)))


def classify_triangle_pair(first, second, bbox_diagonal):
    """Classify bounded FP64 triangle contacts without calling a model.

    Proper transverse penetration means a positive-length intersection of the
    relative interiors of both noncoplanar triangles. Shared vertices/edges,
    boundary tangencies and positive-area coplanar overlap are separate labels.
    Degenerate and ill-conditioned pairs are reported, never counted as clean.
    """
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Pair classifier needs two finite [3,3] triangles')
    length = _bbox_scale(bbox_diagonal)
    eps, area_eps = 1e-9 * length, 1e-12 * length * length
    n_a, n_b = np.cross(a[1] - a[0], a[2] - a[0]), np.cross(b[1] - b[0], b[2] - b[0])
    norm_a, norm_b = float(np.linalg.norm(n_a)), float(np.linalg.norm(n_b))
    result = dict(classification='disjoint', proper_transverse=False,
                  intersection_length=0.0, coplanar_overlap_area=0.0,
                  epsilon=eps, area_epsilon=area_eps)
    if norm_a / 2 <= area_eps or norm_b / 2 <= area_eps:
        result.update(classification='degenerate_pair',
                      first_degenerate=norm_a / 2 <= area_eps, second_degenerate=norm_b / 2 <= area_eps)
        return result
    if np.any(a.max(0) < b.min(0) - eps) or np.any(b.max(0) < a.min(0) - eps):
        return result
    n_a, n_b = n_a / norm_a, n_b / norm_b
    b_to_a = (b - a[0]) @ n_a
    a_to_b = (a - b[0]) @ n_b
    same_side = lambda d: bool(np.all(d > eps) or np.all(d < -eps))
    if same_side(a_to_b) or same_side(b_to_a):
        return result
    distances = np.linalg.norm(a[:, None] - b[None], axis=-1)
    # A tiny edge may put two A vertices within epsilon of one B vertex.
    # Count a maximum one-to-one matching, never two matches to one vertex.
    shared_a = _maximum_unique_matches(distances <= eps)
    exact_shared = _maximum_unique_matches(np.all(a[:, None] == b[None], axis=-1))
    result.update(shared_vertices_within_epsilon=shared_a, exact_shared_vertices=exact_shared,
                  shared_vertex_count_is_one_to_one=True)
    coplanar = bool(np.max(np.abs(b_to_a)) <= eps and np.max(np.abs(a_to_b)) <= eps)
    if coplanar:
        polygon, overlap = _coplanar_clip(a, b, n_a)
        result['coplanar_overlap_area'] = overlap
        if overlap > area_eps:
            result['classification'] = 'coplanar_positive_area_overlap'
            return result
        contact_distance = min(_segment_pair_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])
                               for i in range(3) for j in range(3))
        if contact_distance > eps:
            return result
        if shared_a >= 2:
            result['classification'] = 'shared_edge_contact'
        elif shared_a == 1:
            result['classification'] = 'shared_vertex_contact'
        else:
            result['classification'] = 'coplanar_boundary_or_tolerance_contact'
        result['minimum_edge_contact_distance'] = float(contact_distance)
        return result
    line = np.cross(n_a, n_b)
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-12:
        result['classification'] = 'near_parallel_unresolved'
        return result
    line /= line_norm
    cut_a, cut_b = _plane_cut(a, a_to_b, eps), _plane_cut(b, b_to_a, eps)
    if not cut_a or not cut_b:
        return result
    interval_a = (np.asarray(cut_a) - a[0]) @ line
    interval_b = (np.asarray(cut_b) - a[0]) @ line
    overlap = min(float(interval_a.max()), float(interval_b.max())) - max(float(interval_a.min()), float(interval_b.min()))
    if overlap < -eps:
        return result
    result['intersection_length'] = max(overlap, 0.0)
    straddles = lambda d: bool(d.min() < -eps and d.max() > eps)
    if overlap > eps and straddles(a_to_b) and straddles(b_to_a):
        result.update(classification='proper_transverse_penetration', proper_transverse=True)
    elif overlap > eps:
        result['classification'] = 'shared_edge_contact' if shared_a >= 2 else 'noncoplanar_edge_or_tangent_contact'
    else:
        result['classification'] = 'shared_vertex_contact' if shared_a else 'point_or_tangent_contact'
    return result
