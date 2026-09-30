"""Independent CPU geometry primitives used by pure Native T1 evaluation.

The arithmetic and RNG order are preserved from the archived core evaluator.
There are no model, training, guidance, welding, or geometry-repair operations.
Surface scoring uses trimesh finite-triangle closest points; boundary scoring
uses the original independent FP64 finite-triangle and finite-segment formula.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import numpy as np
import trimesh

def surface_distance_squared(points, triangles, chunk=32):
    """Trimesh's independent triangle closest-point implementation, not G helper."""
    points, triangles = np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.float64)
    if not np.isfinite(triangles).all() or not len(triangles):
        return np.full(len(points), np.inf)
    values = []
    for start in range(0, len(points), chunk):
        current = points[start:start+chunk]
        repeated_triangles = np.tile(triangles, (len(current), 1, 1))
        repeated_points = np.repeat(current, len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(repeated_triangles, repeated_points)
        d2 = np.sum((closest-repeated_points)**2, axis=-1).reshape(len(current), len(triangles))
        # Trimesh can produce NaN on collapsed triangles. Evaluate the finite
        # boundary segments independently and use them for those entries.
        bad = ~np.isfinite(d2)
        if bad.any():
            for qi, fi in np.argwhere(bad):
                tri, p = triangles[fi], current[qi]
                candidates = []
                for i, j in ((0,1),(1,2),(2,0)):
                    edge = tri[j]-tri[i]
                    length = np.dot(edge, edge)
                    u = np.clip(np.dot(p-tri[i], edge)/length, 0, 1) if length > 0 else 0.
                    candidates.append(np.sum((p-(tri[i]+u*edge))**2))
                d2[qi,fi] = min(candidates)
        values.extend(np.min(d2, axis=1))
    return np.asarray(values)


def sample_mesh_surface(triangles, count, rng):
    tri = np.asarray(triangles, dtype=np.float64)
    areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=-1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        return None
    selected = tri[rng.choice(len(tri), count, p=areas/areas.sum())]
    uv = rng.rand(count,2); root = np.sqrt(uv[:,:1])
    return ((1-root)*selected[:,0] + root*(1-uv[:,1:])*selected[:,1] + root*uv[:,1:]*selected[:,2])


def raw_quality(tokens):
    tri = np.asarray(tokens, dtype=np.float64).reshape(-1,3,3)
    finite = np.isfinite(tri).all()
    result = dict(raw_faces=len(tri), nan_count=int(np.isnan(tri).sum()), inf_count=int(np.isinf(tri).sum()),
                  self_intersection_status="NOT_RUN", watertight_status="NOT_RUN", manifold_status="NOT_RUN")
    if not finite:
        result.update(degenerate_fraction=None, min_angle_lt5_fraction=None, angles_p05=None)
        return result
    edge_lengths = np.stack([np.linalg.norm(tri[:,(i+1)%3]-tri[:,i],axis=-1) for i in range(3)],axis=-1)
    area2 = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),axis=-1)
    scale = max(np.linalg.norm(np.ptp(tri.reshape(-1,3),axis=0)), 1e-12)
    degenerate = area2 <= scale**2 * 1e-12
    angles = []
    for i in range(3):
        u, v = tri[:,(i+1)%3]-tri[:,i], tri[:,(i+2)%3]-tri[:,i]
        denominator = np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        cosine = np.divide((u*v).sum(-1), denominator, out=np.ones(len(tri)), where=denominator>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angles = np.stack(angles,-1)
    result.update(degenerate_fraction=float(degenerate.mean()),
                  min_angle_lt5_fraction=float((angles.min(-1)<5).mean()),
                  angles_p05=float(np.quantile(angles,.05)))
    return result


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


def _increment(ledger, key, amount=1):
    if ledger is not None:
        ledger[key] = ledger.get(key, 0) + amount


def _sha_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((str(value.dtype), value.shape)).encode() + value.tobytes()).hexdigest()


def boundary_metrics(context, free, length, ledger=None):
    segments, boundary = extract_boundary_segments(context)
    points, ids, weights, counts = _sample_boundary(segments, 9919, 512)
    d2 = point_to_triangle_surface_squared(points, free, length)
    _increment(ledger, "boundary_distance_evaluations")
    if d2 is None or len(d2) == 0:
        return dict(status="NO_BOUNDARY_OR_FREE", rms=None, p95=None, maximum=None,
                    no_candidates_is_not_zero=True)
    distances = np.sqrt(d2)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, .95 * cumulative[-1], side="left")), len(order)-1)
    p95 = float(distances[order[position]])
    rms = float(np.sqrt(np.sum(weights*d2)))
    maximum = float(distances.max())
    near = distances < .001 * length
    return dict(status="PASS", length_weighted=True, point_count=len(points), point_seed=9919,
                point_sha256=_sha_array(points), weight_sha256=_sha_array(weights),
                rms=rms, p95=p95, maximum=maximum, rms_bbox_pct=100*rms/length,
                p95_bbox_pct=100*p95/length, max_bbox_pct=100*maximum/length,
                coverage_threshold_bbox_fraction=.001,
                coverage_fraction=float(weights[near].sum()),
                weighted_p95_definition="Smallest sampled distance with cumulative arc-length weights >=95%",
                maximum_scope="Sample maximum; not exact Hausdorff", boundary=boundary,
                sample_counts_per_segment=counts)
