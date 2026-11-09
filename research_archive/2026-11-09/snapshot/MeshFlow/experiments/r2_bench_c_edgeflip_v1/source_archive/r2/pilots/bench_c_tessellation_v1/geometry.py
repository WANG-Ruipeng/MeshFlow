"""CPU-only, surface-preserving C subdivision for a frozen-generator probe.

No model, source completion, or random state is used.  Original vertices keep
their exact FP32 bytes.  New midpoint coordinates are rounded once to FP32.
Canonical corner ordering reproduces bench_user_v2/build.py:canonical using
model-space coordinates (its positive global scale does not change ordering).
"""
from __future__ import annotations

from collections import Counter
import numpy as np

VERSION = "bench_C_midpoint4_surface_probe_v1"
CANONICAL_SOURCE = "r2/data_tools/bench_user_v2/build.py:canonical"


def _input(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("C_vertices must be an FP32 (V,3) array")
    if f.dtype != np.int64 or f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("C_faces_original must be a nonempty int64 (K,3) array")
    if not np.isfinite(v).all() or (f < 0).any() or (f >= len(v)).any():
        raise ValueError("Nonfinite coordinates or invalid face indices")
    if np.any(np.diff(np.sort(f, axis=1), axis=1) == 0):
        raise ValueError("A source C face repeats a vertex ID")
    return v, f


def canonical_faces(vertices, faces):
    """Return model face indices and P with model[j] = original[P[j]]."""
    v, f = _input(vertices, faces)
    ranks = np.argsort(np.lexsort((v[:, 0], v[:, 2], v[:, 1])))
    p = np.argsort(ranks[f], axis=1, kind="stable")
    pre = (v / np.float32(2))[np.take_along_axis(f, p, axis=1)]
    begin = np.lexsort((pre[:, :, 2], pre[:, :, 1], pre[:, :, 0]), axis=1)[:, 0]
    p = np.take_along_axis(p, (begin[:, None] + np.arange(3)[None, :]) % 3, axis=1)
    return np.take_along_axis(f, p, axis=1), p.astype(np.int64)


def _edge_rows(faces):
    rows = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            u, v = int(face[i]), int(face[(i + 1) % 3])
            rows.setdefault((min(u, v), max(u, v)), []).append((fi, u, v))
    return rows


def midpoint4(C_vertices, C_faces_original):
    """Split each source triangle into four, sharing source-edge midpoints.

    Child order is the three original-vertex corners followed by the center.
    Source face order and original vertex indices are unchanged.  The new
    representation has new input-face identities; child_parent_face_ids is
    provenance, not a reusable unique source-face-ID array.
    """
    v, f = _input(C_vertices, C_faces_original)
    edges = np.asarray(sorted(_edge_rows(f)), dtype=np.int64).reshape(-1, 2)
    midpoint_ids = np.arange(len(v), len(v) + len(edges), dtype=np.int64)
    mid = ((v[edges[:, 0]].astype(np.float64) + v[edges[:, 1]].astype(np.float64)) * .5).astype(np.float32)
    refined_v = np.concatenate((v.copy(), mid), axis=0)
    lookup = {tuple(e): int(i) for e, i in zip(edges.tolist(), midpoint_ids)}
    refined_f = []
    for a0, b0, c0 in f:
        a, b, c = int(a0), int(b0), int(c0)
        ab = lookup[min(a, b), max(a, b)]
        bc = lookup[min(b, c), max(b, c)]
        ca = lookup[min(c, a), max(c, a)]
        refined_f.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    refined_f = np.asarray(refined_f, dtype=np.int64)
    model, p = canonical_faces(refined_v, refined_f)
    result = dict(
        C_vertices=refined_v, C_faces_original=refined_f, C_faces_model=model,
        child_parent_face_ids=np.repeat(np.arange(len(f), dtype=np.int64), 4),
        child_local_ids=np.tile(np.arange(4, dtype=np.int64), len(f)),
        midpoint_origin_edges=edges, midpoint_vertex_ids=midpoint_ids,
        corner_permutations=p, C_source_corner_permutations=p.copy(),
        original_vertex_count=np.asarray(len(v), dtype=np.int64),
        original_face_count=np.asarray(len(f), dtype=np.int64),
    )
    audit_subdivision(dict(C_vertices=v, C_faces_original=f), result)
    return result


def _geometry(v, f):
    tri = v[f].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Degenerate/nonfinite C triangle")
    return tri, .5 * norm, cross / norm[:, None]


def _coordinate_duplicate_count(v, f):
    # Ignore corner winding, but do not coordinate-weld the mesh.
    keys = []
    for tri in v[f]:
        ordered = tri[np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))]
        keys.append(ordered.tobytes())
    return sum(n - 1 for n in Counter(keys).values())


def audit_subdivision(original, refined):
    """Check finite geometry and physical boundary preservation; return facts.

    FP32 midpoint roundoff is explicitly tolerated and measured.  No claim of
    bitwise preservation of newly constructed surface coordinates is made.
    Topology checks certify this subdivision, not all source-mesh legality.
    """
    v, f = _input(original["C_vertices"], original["C_faces_original"])
    w, g = _input(refined["C_vertices"], refined["C_faces_original"])
    if len(g) != 4 * len(f) or int(refined["original_vertex_count"]) != len(v):
        raise ValueError("Incorrect subdivision cardinality")
    if w[:len(v)].tobytes() != v.tobytes():
        raise ValueError("Original C vertex bytes changed")
    parents = np.asarray(refined["child_parent_face_ids"])
    if not np.array_equal(parents, np.repeat(np.arange(len(f)), 4)):
        raise ValueError("Child/source-face identity mapping differs")
    if not np.array_equal(refined["child_local_ids"], np.tile(np.arange(4), len(f))):
        raise ValueError("Child local order differs")
    if int(refined["original_face_count"]) != len(f):
        raise ValueError("Original source face count differs")
    tri, area, normal = _geometry(v, f)
    rtri, rarea, rnormal = _geometry(w, g)
    parent_sum = np.bincount(parents, weights=rarea, minlength=len(f))
    relative_area = np.abs(parent_sum - area) / area
    if not np.allclose(parent_sum, area, rtol=3e-6, atol=1e-12):
        raise ValueError("Subdivision changes per-source-face surface area")
    cosine = np.sum(normal[parents] * rnormal, axis=1)
    if np.min(cosine) < 1 - 3e-6:
        raise ValueError("Subdivision flips/distorts a child normal")
    extent = max(float(np.ptp(v, axis=0).max()), 1.)
    tolerance = 8 * np.finfo(np.float32).eps * max(float(np.abs(v).max()), extent)
    plane_distance = np.abs(np.sum((rtri - tri[parents, :1]) * normal[parents, None, :], axis=2))
    if float(plane_distance.max()) > tolerance:
        raise ValueError("Child vertex left its source triangle plane")
    if not np.array_equal(w.min(0), v.min(0)) or not np.array_equal(w.max(0), v.max(0)):
        raise ValueError("Source C bbox changed")
    if _coordinate_duplicate_count(v, f) or _coordinate_duplicate_count(w, g):
        raise ValueError("Duplicate source or subdivided C triangle")
    model, p = canonical_faces(w, g)
    if not np.array_equal(model, refined["C_faces_model"]):
        raise ValueError("Canonical model ordering differs from legacy formula")
    if not np.array_equal(p, refined["corner_permutations"]):
        raise ValueError("Incorrect source-to-model corner permutation")
    inverse = np.argsort(p, axis=1)
    if not np.array_equal(np.take_along_axis(model, inverse, axis=1), g):
        raise ValueError("Source corner restoration failed")

    old_edges, new_edges = _edge_rows(f), _edge_rows(g)
    edges, ids = np.asarray(refined["midpoint_origin_edges"]), np.asarray(refined["midpoint_vertex_ids"])
    if not np.array_equal(edges, np.asarray(sorted(old_edges), np.int64)):
        raise ValueError("Midpoint source-edge map differs")
    if not np.array_equal(ids, np.arange(len(v), len(w))) or len(edges) != len(ids):
        raise ValueError("Midpoints must have unique appended vertex IDs")
    exact_mid = (v[edges[:, 0]].astype(np.float64) + v[edges[:, 1]].astype(np.float64)) * .5
    midpoint_error = np.linalg.norm(w[ids].astype(np.float64) - exact_mid, axis=1)
    if midpoint_error.max(initial=0) > tolerance:
        raise ValueError("Midpoint coordinate error exceeds FP32 roundoff")
    lookup = {tuple(e): int(i) for e, i in zip(edges.tolist(), ids)}
    expected_boundary = set()
    old_boundary_length = new_boundary_length = 0.
    for edge, rows in old_edges.items():
        a, b = edge
        m = lookup[edge]
        segments = ((min(a, m), max(a, m)), (min(m, b), max(m, b)))
        if any(len(new_edges.get(e, ())) != len(rows) for e in segments):
            raise ValueError("Original edge incidence was not preserved by two segments")
        if len(rows) == 1:
            expected_boundary.update(segments)
            old_boundary_length += float(np.linalg.norm(v[a].astype(np.float64) - v[b]))
            for e in segments:
                new_boundary_length += float(np.linalg.norm(w[e[0]].astype(np.float64) - w[e[1]]))
            # Preserve the directed physical contour, not just incidence.
            _, u, z = rows[0]
            required = {(u, m), (m, z)}
            actual = {(new_edges[e][0][1], new_edges[e][0][2]) for e in segments}
            if required != actual:
                raise ValueError("An open source boundary reversed direction")
    actual_boundary = {e for e, rows in new_edges.items() if len(rows) == 1}
    if actual_boundary != expected_boundary:
        raise ValueError("Subdivision created or removed an open boundary")
    if not np.isclose(old_boundary_length, new_boundary_length, rtol=3e-6, atol=1e-12):
        raise ValueError("Physical boundary length changed")

    return dict(
        status="PASS", schema=VERSION, canonical_source=CANONICAL_SOURCE,
        original_vertices=len(v), refined_vertices=len(w), original_faces=len(f), refined_faces=len(g),
        original_vertex_bytes_preserved=True, child_parent_mapping_exact=True,
        source_corner_roundtrip_exact=True, bbox_exact=True, coordinate_duplicate_faces=0,
        degenerate_faces=0, original_area=float(area.sum()), refined_area=float(rarea.sum()),
        maximum_parent_area_relative_error=float(relative_area.max()), minimum_normal_cosine=float(cosine.min()),
        maximum_child_plane_distance=float(plane_distance.max()), maximum_midpoint_roundoff=float(midpoint_error.max(initial=0)),
        coordinate_tolerance=float(tolerance), original_boundary_edges=len(expected_boundary)//2,
        refined_boundary_edges=len(actual_boundary), physical_boundary_preserved=True,
        boundary_winding_preserved=True, original_boundary_length=old_boundary_length,
        refined_boundary_length=new_boundary_length,
        boundary_note="Identical physical contour within reported FP32 tolerance; each original boundary edge becomes two index edges.",
        model_calls=0, random_draws=0,
    )
