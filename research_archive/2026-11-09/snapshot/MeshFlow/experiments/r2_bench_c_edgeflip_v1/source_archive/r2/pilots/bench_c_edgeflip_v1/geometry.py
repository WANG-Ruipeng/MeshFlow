"""CPU-only deterministic C edge flip; no Y, model, RNG or vertex writes.

Only the first legal original internal edge in vertex-ID lexicographic order
is flipped.  The two replacement faces occupy the same two row positions.
Face IDs at those rows describe new triangles, not unchanged source faces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np

from r2.pilots.bench_c_tessellation_v1.geometry import canonical_faces

VERSION = "bench_C_single_planar_edgeflip_v1"
CANONICAL_SOURCE = "r2/data_tools/bench_user_v2/build.py:canonical"
SHAPE_TOL = 1e-10
AREA_RTOL = 3e-6
NORMAL_COS_TOL = 3e-6


def _input(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3 or not len(v):
        raise ValueError("vertices must be a nonempty FP32 (V,3) array")
    if f.dtype != np.int64 or f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("faces_original must be a nonempty int64 (K,3) array")
    if not np.isfinite(v).all() or (f < 0).any() or (f >= len(v)).any():
        raise ValueError("Nonfinite coordinates or invalid face indices")
    return v, f


def _edges(f):
    result = {}
    for row, face in enumerate(f):
        for k in range(3):
            u, w = int(face[k]), int(face[(k + 1) % 3])
            result.setdefault(tuple(sorted((u, w))), []).append((row, u, w))
    return result


def _boundary(f):
    return sorted((rows[0][1], rows[0][2]) for rows in _edges(f).values() if len(rows) == 1)


def _geometry(v, f):
    t = v[f].astype(np.float64)
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normal = np.divide(cross, norm[:, None], out=np.zeros_like(cross), where=norm[:, None] > 0)
    return t, norm / 2, normal


def _duplicates(v, f):
    keys = []
    for tri in v[f]:
        # Coordinate equality, including distinct vertex IDs, ignores winding.
        ordered = tri[np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))].copy()
        ordered[ordered == 0] = 0  # +0 and -0 are the same physical coordinate.
        keys.append(ordered.tobytes())
    return sum(n - 1 for n in Counter(keys).values())


def _minimum_angle(triangles):
    values = []
    for tri in triangles:
        for k in range(3):
            a = tri[(k + 1) % 3] - tri[k]
            b = tri[(k + 2) % 3] - tri[k]
            values.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))
    return float(min(values))


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _point_in_2d(p, tri, tol):
    signs = [_cross2(tri[(i + 1) % 3] - tri[i], p - tri[i]) for i in range(3)]
    area_tol = tol * max(float(np.linalg.norm(np.roll(tri, -1, axis=0) - tri, axis=1).max()), tol)
    return min(signs) >= -area_tol or max(signs) <= area_tol


def _segment_parameters_2d(p, q, a, b, tol):
    """Parameters on pq for its intersection with ab, including contacts."""
    r, s = q - p, b - a
    lr, ls = float(np.linalg.norm(r)), float(np.linalg.norm(s))
    if lr <= tol:
        return []
    den = _cross2(r, s)
    area_tol = tol * max(lr, ls, tol)
    if abs(den) > area_tol:
        t, u = _cross2(a - p, s) / den, _cross2(a - p, r) / den
        if -tol / lr <= t <= 1 + tol / lr and -tol / max(ls, tol) <= u <= 1 + tol / max(ls, tol):
            return [float(np.clip(t, 0, 1))]
        return []
    if abs(_cross2(a - p, r)) > area_tol:
        return []
    rr = float(np.dot(r, r))
    lo, hi = sorted((float(np.dot(a - p, r) / rr), float(np.dot(b - p, r) / rr)))
    lo, hi = max(0., lo), min(1., hi)
    return [lo, hi] if lo <= hi + tol / lr else []


def _segment_triangle_points(p, q, tri, tol):
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    dp, dq = float(np.dot(p - tri[0], normal)), float(np.dot(q - tri[0], normal))
    axes = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
    t2 = tri[:, axes]
    if abs(dp) <= tol and abs(dq) <= tol:
        params = [x for x, pt in ((0., p), (1., q)) if _point_in_2d(pt[axes], t2, tol)]
        for i in range(3):
            params.extend(_segment_parameters_2d(p[axes], q[axes], t2[i], t2[(i + 1) % 3], tol))
        return [p + t * (q - p) for t in params]
    if (dp > tol and dq > tol) or (dp < -tol and dq < -tol):
        return []
    den = dp - dq
    if abs(den) <= np.finfo(np.float64).tiny:
        return []
    t = dp / den
    if not 0 <= t <= 1:
        return []
    point = p + t * (q - p)
    return [point] if _point_in_2d(point[axes], t2, tol) else []


def _illegal_pair(v64, a, b, tol):
    """Intersection beyond the exact shared-ID vertex/edge is illegal.

    This conservative local gate is not the production evaluation scorer.
    It checks all triangle-edge intersections, including coplanar overlap.
    """
    ta, tb = v64[a], v64[b]
    if np.any(ta.max(0) < tb.min(0) - tol) or np.any(tb.max(0) < ta.min(0) - tol):
        return False
    shared = sorted(set(map(int, a)) & set(map(int, b)))
    if len(shared) == 3:
        return True
    points = []
    for first, second in ((ta, tb), (tb, ta)):
        for k in range(3):
            points.extend(_segment_triangle_points(first[k], first[(k + 1) % 3], second, tol))
    for p in points:
        if not shared:
            return True
        if len(shared) == 1:
            if np.linalg.norm(p - v64[shared[0]]) > 4 * tol:
                return True
        else:
            u, w = v64[shared]
            delta = w - u
            t = float(np.dot(p - u, delta) / np.dot(delta, delta))
            closest = u + np.clip(t, 0, 1) * delta
            if np.linalg.norm(p - closest) > 4 * tol:
                return True
    return False


def _candidate(v, f, edge, rows, old_edges, tolerance):
    i, u, w = rows[0]
    j, u2, w2 = rows[1]
    result = dict(edge=list(edge), face_rows=[i, j], status="REJECTED")

    def reject(reason):
        result["reason"] = reason
        return None, result

    if (u, w) != (w2, u2):
        return reject("shared_edge_winding_not_opposite")
    a = next(int(x) for x in f[i] if x not in edge)
    b = next(int(x) for x in f[j] if x not in edge)
    if len({u, w, a, b}) != 4:
        return reject("not_four_distinct_vertex_ids")
    alternative = tuple(sorted((a, b)))
    result["new_edge"] = list(alternative)
    if alternative in old_edges:
        return reject("alternative_diagonal_already_exists")
    local = np.asarray([f[i], f[j]], dtype=np.int64)
    oldtri, oldarea, oldnormal = _geometry(v, local)
    # Orient from the first source row: (u,w,a), (w,u,b).
    replacement = np.asarray(((a, u, b), (a, b, w)), dtype=np.int64)
    newtri, newarea, newnormal = _geometry(v, replacement)
    quad = v[[a, u, b, w]].astype(np.float64)
    diameter = float(np.linalg.norm(quad[:, None] - quad[None, :], axis=-1).max())
    scale2 = diameter * diameter
    if scale2 <= 0 or np.min(newarea) <= 0:
        return reject("degenerate_new_triangle")
    shapes = 2 * np.concatenate((oldarea, newarea)) / scale2
    result["minimum_normalized_double_area"] = float(shapes.min())
    if shapes.min() < SHAPE_TOL:
        return reject("local_shape_below_tolerance")
    normal = oldnormal[0]
    heights = (quad - quad[0]) @ normal
    plane_error = float(np.abs(heights).max())
    surface_bound = float(np.ptp(heights))
    min_cos = float(np.min(np.vstack((oldnormal, newnormal)) @ normal))
    result.update(maximum_plane_distance=plane_error, surface_distance_upper_bound=surface_bound,
                  minimum_normal_cosine=min_cos)
    if plane_error > tolerance or surface_bound > tolerance:
        return reject("not_coplanar_within_tolerance")
    if min_cos < 1 - NORMAL_COS_TOL:
        return reject("face_normal_not_consistent")
    turns = np.asarray([np.dot(np.cross(quad[(k + 1) % 4] - quad[k],
                                       quad[(k + 2) % 4] - quad[(k + 1) % 4]), normal)
                        for k in range(4)]) / scale2
    result["minimum_normalized_convex_turn"] = float(turns.min())
    if turns.min() < SHAPE_TOL:
        return reject("quadrilateral_not_strictly_convex")
    before_angle, after_angle = _minimum_angle(oldtri), _minimum_angle(newtri)
    threshold = min(before_angle, 5.)
    result.update(original_local_minimum_angle_deg=before_angle,
                  new_local_minimum_angle_deg=after_angle, minimum_angle_required_deg=threshold)
    if after_angle + 1e-10 < threshold:
        return reject("new_minimum_angle_below_frozen_threshold")
    relative = abs(float(newarea.sum() - oldarea.sum())) / float(oldarea.sum())
    result["local_area_relative_error"] = relative
    if relative > AREA_RTOL:
        return reject("local_area_not_preserved")
    g = f.copy()
    g[[i, j]] = replacement
    if _boundary(g) != _boundary(f):
        return reject("directed_boundary_changed")
    if _duplicates(v, g):
        return reject("duplicate_coordinate_face")
    new_edges = _edges(g)
    removed, added = set(old_edges) - set(new_edges), set(new_edges) - set(old_edges)
    if removed != {edge} or added != {alternative}:
        return reject("not_exactly_one_diagonal_exchange")
    for e in set(old_edges) & set(new_edges):
        if len(old_edges[e]) != len(new_edges[e]):
            return reject("external_edge_incidence_changed")
    pairs = sorted({tuple(sorted((row, other))) for row in (i, j) for other in range(len(f)) if row != other})
    illegal = [[x, y] for x, y in pairs if _illegal_pair(v.astype(np.float64), g[x], g[y], tolerance)]
    result.update(local_pair_tests=len(pairs), illegal_new_local_pairs=illegal)
    if illegal:
        return reject("new_patch_has_illegal_triangle_intersection")
    result.update(status="ACCEPTED", reason="first_legal_original_internal_edge",
                  old_local_faces=local.tolist(), new_local_faces=replacement.tolist(),
                  original_local_area=float(oldarea.sum()), new_local_area=float(newarea.sum()),
                  projected_domain_equal=True)
    return g, result


def flip_one(vertices, faces_original):
    """Return ``(new_faces_original, audit)``; BLOCKED returns unchanged copy.

    The input vertex array is never mutated or re-encoded.  No source/Y data
    is accepted.  Surface equality uses the same convex projected domain and
    a reported upper bound on the two piecewise-planar height differences.
    No output or model result is involved in selecting the diagonal.
    """
    v, f = _input(vertices, faces_original)
    original_vertex_bytes, original_face_bytes = v.tobytes(), f.tobytes()
    scale = max(float(np.ptp(v.astype(np.float64), axis=0).max()), float(np.abs(v).max()), 1.)
    tolerance = 8 * np.finfo(np.float32).eps * scale
    old_edges = _edges(f)
    _, oldarea, _ = _geometry(v, f)
    degenerate = int(np.count_nonzero(oldarea <= 0))
    duplicates = _duplicates(v, f)
    audit = dict(schema=VERSION, status="BLOCKED", selection_rule="first_legal_lexicographic_original_vertex_ID_edge",
                 canonical_source=CANONICAL_SOURCE, original_vertices=len(v), new_vertices=len(v),
                 original_faces=len(f), new_faces=len(f), flips=0, model_calls=0, random_draws=0,
                 uses_Y=False, coordinate_tolerance=tolerance, shape_tolerance=SHAPE_TOL,
                 area_relative_tolerance=AREA_RTOL, normal_cosine_tolerance=NORMAL_COS_TOL,
                 minimum_angle_rule_deg="new >= min(original_local_minimum, 5)",
                 original_degenerate_faces=degenerate, original_duplicate_coordinate_faces=duplicates,
                 original_vertex_sha256=hashlib.sha256(original_vertex_bytes).hexdigest(),
                 original_face_sha256=hashlib.sha256(original_face_bytes).hexdigest(),
                 original_directed_boundary_edges=[list(e) for e in _boundary(f)],
                 candidates=[], original_area=float(oldarea.sum()))
    g = f.copy()
    if degenerate or duplicates:
        audit["reason"] = "source_contains_degenerate_or_duplicate_face"
    else:
        selected = False
        for edge in sorted(old_edges):
            rows = old_edges[edge]
            if len(rows) == 1:
                continue
            if selected:
                audit["candidates"].append(dict(edge=list(edge), status="NOT_EVALUATED", reason="stopped_after_first_legal_edge"))
                continue
            if len(rows) != 2:
                audit["candidates"].append(dict(edge=list(edge), status="REJECTED", reason="edge_not_two_manifold", incidence=len(rows)))
                continue
            candidate, record = _candidate(v, f, edge, rows, old_edges, tolerance)
            audit["candidates"].append(record)
            if candidate is not None:
                g, selected = candidate, True
                audit.update(status="PASS", reason="one_legal_internal_edge_flipped", flips=1,
                             selected_edge=record["edge"], new_edge=record["new_edge"],
                             selected_face_rows=record["face_rows"], selected=record)
        if not selected:
            audit["reason"] = "no_eligible_internal_edge"
    _, newarea, _ = _geometry(v, g)
    changed = np.flatnonzero(np.any(f != g, axis=1)).tolist()
    model, p = canonical_faces(v, g) if not degenerate else (None, None)
    audit.update(new_face_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                 original_vertex_bytes_preserved=v.tobytes() == original_vertex_bytes,
                 input_faces_not_mutated=f.tobytes() == original_face_bytes,
                 face_count_preserved=len(g) == len(f), vertex_count_preserved=True,
                 changed_face_rows=changed, unchanged_face_rows_exact=True,
                 new_directed_boundary_edges=[list(e) for e in _boundary(g)],
                 directed_boundary_exact=_boundary(g) == _boundary(f),
                 new_degenerate_faces=int(np.count_nonzero(newarea <= 0)),
                 new_duplicate_coordinate_faces=_duplicates(v, g), new_area=float(newarea.sum()),
                 surface_certificate="convex projected domain equality plus plane-height range bound; unchanged faces byte exact",
                 intersection_scope="all new local faces versus every C face; unrelated old pairs unchanged",
                 source_corner_roundtrip_exact=None if p is None else bool(np.array_equal(np.take_along_axis(model, np.argsort(p, axis=1), axis=1), g)))
    if not audit["original_vertex_bytes_preserved"] or not audit["input_faces_not_mutated"]:
        raise AssertionError("Input mutation")
    if audit["status"] == "PASS" and (len(changed) != 2 or not audit["directed_boundary_exact"]):
        raise AssertionError("Invalid edge-flip invariant")
    return g, audit
