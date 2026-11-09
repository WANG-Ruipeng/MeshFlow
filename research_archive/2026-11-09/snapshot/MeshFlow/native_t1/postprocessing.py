"""Optional CPU-only GEO_C cleanup of a saved FP32 triangle soup.

The radius is an engineering convention, not an exact reproduction of the
paper's implementation. Original model coordinates are never normalized again.
Ported from CHAIR_START_POSTPROCESS_COMPARISON_V1 without changing geometry.
The first K triangles are C; their bytes do not encode an original vertex-ID graph.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import time

import numpy as np


RADIUS = .015 / (.95 * .3762)
FIXED_CONFIG = {
    "schema": "chair_start_c_protected_geometry_v1",
    "arm": "GEO_C",
    "dataset_std": .3762,
    "reference_half_extent": .95,
    "reference_radius": .015,
    "radius_model": RADIUS,
    "scale_contract": "x_model=2*x_unitbox/.3762; x_reference=.95*.3762*x_model",
    "radius_limit": "Single engineering threshold; original paper runtime units are not fully traceable",
    "normalization": "NONE; never normalize the generated bbox",
    "known_vertex_identity": "FP32 coordinate bytes, including signed zero",
    "anchor_rule": "one distinct numeric C coordinate within radius; conflicting coordinates leave free corner fixed",
    "free_rule": "lexicographic original-point representatives; nearest existing representative within radius; no transitive unions or means",
    "degenerate_rule": "free only: exact repeated coordinate or exactly zero FP64 cross product",
    "duplicate_rule": "free only: same three numeric coordinates ignoring winding; C wins, then first retained free source face",
    "hole_fill": False,
    "smoothing": False,
    "quantization": False,
    "clipping": False,
    "model_calls": 0,
}


def _hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _bits(p):
    return tuple(int(v) for v in np.asarray(p, np.float32).view(np.uint32))


def _numeric(p):
    return tuple(float(v) for v in p)


def _sort_key(p):
    return _numeric(p) + _bits(p)


def _face_key(tri):
    return tuple(sorted(_numeric(p) for p in tri))


def _degenerate(tri):
    if len({_numeric(p) for p in tri}) != 3:
        return "exact_repeated_vertex"
    t = tri.astype(np.float64)
    if np.all(np.cross(t[1] - t[0], t[2] - t[0]) == 0):
        return "exact_collinear"
    return None


def process(raw, K, radius=RADIUS):
    """Return corrected triangle soup, real indexed topology, and source maps.

    Only FP32 raw[N,3,3], known-face count and the frozen radius are accepted.
    No GT, UID, score, model, random number, or filesystem access enters this
    function. All N original corners keep their records even if a free face is
    deleted. ``corner_target_input_ids`` indexes raw.reshape(-1,3); -1 in
    ``corner_vertex_ids`` means that coordinate is absent from the kept mesh.
    """
    started = time.perf_counter()
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32:
        raise TypeError("raw must be an existing FP32 numpy array")
    if raw.ndim != 3 or raw.shape[1:] != (3, 3) or not np.isfinite(raw).all():
        raise ValueError("raw must be finite [N,3,3]")
    if type(K) is not int or not 0 < K < len(raw):
        raise ValueError("Require integer 0 < K < N")
    if not np.isscalar(radius) or not np.isfinite(radius) or float(radius) != RADIUS:
        raise ValueError("Only the preregistered engineering radius is allowed")
    radius = float(radius)
    original_hash = _hash(raw)
    original = np.ascontiguousarray(raw).copy()
    points = original.reshape(-1, 3)
    corrected = points.copy()
    known_count = 3 * K
    n_corners = len(points)
    target = np.arange(n_corners, dtype=np.int64)
    kind = np.full(n_corners, "known", dtype="U32")
    anchor_candidates = np.zeros(n_corners, dtype=np.int64)
    radius2 = radius * radius

    # Signed-zero variants remain separate indexed C vertices. For proximity
    # conflicts, however, they are one numeric position, not different anchors.
    anchors_by_numeric = {}
    for i in range(known_count):
        anchors_by_numeric.setdefault(_numeric(points[i]), []).append(i)
    anchor_rows = []
    for ids in anchors_by_numeric.values():
        # Stable bit-pattern preference, with source id only a provenance tie.
        anchor_rows.append(min(ids, key=lambda i: (_bits(points[i]), i)))
    anchor_rows.sort(key=lambda i: _sort_key(points[i]))
    anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
    anchor_points = points[anchor_rows].astype(np.float64)
    free_for_clustering = []
    for i in range(known_count, n_corners):
        delta = anchor_points - points[i].astype(np.float64)
        d2 = np.einsum("ij,ij->i", delta, delta)
        hits = np.flatnonzero(d2 <= radius2)
        anchor_candidates[i] = len(hits)
        if len(hits) == 1:
            j = int(anchor_rows[hits[0]])
            corrected[i] = points[j]
            target[i] = j
            kind[i] = "unique_C_anchor"
        elif len(hits) > 1:
            # This fixed corner cannot later be moved by the free/free stage.
            kind[i] = "ambiguous_C_unchanged"
        else:
            free_for_clustering.append(i)

    representatives = []
    for i in sorted(free_for_clustering, key=lambda j: (_sort_key(points[j]), j)):
        if representatives:
            rp = points[np.asarray(representatives)].astype(np.float64)
            delta = rp - points[i].astype(np.float64)
            distances = np.einsum("ij,ij->i", delta, delta)
            hits = np.flatnonzero(distances <= radius2)
        else:
            hits = []
        if len(hits):
            # Representatives already have the fixed lexicographic ordering.
            position = min(hits, key=lambda h: (float(distances[h]), int(h)))
            j = representatives[int(position)]
            corrected[i] = points[j]
            target[i] = j
            kind[i] = "free_to_representative"
        else:
            representatives.append(i)
            kind[i] = "free_representative"

    corrected = corrected.reshape(original.shape)
    displacement_vector = corrected.astype(np.float64) - original.astype(np.float64)
    displacement2 = np.einsum("nci,nci->nc", displacement_vector, displacement_vector)
    if np.any(displacement2 > radius2):
        raise AssertionError("FP32 writeback exceeded the fixed radius")
    if corrected[:K].tobytes() != original[:K].tobytes():
        raise AssertionError("Known triangles changed before face cleanup")

    keep = np.ones(len(original), dtype=bool)
    reasons = np.full(len(original), "kept", dtype="U32")
    duplicate_of = np.full(len(original), -1, dtype=np.int64)
    seen = {}
    known_degenerate = 0
    for i in range(K):
        known_degenerate += _degenerate(corrected[i]) is not None
        seen.setdefault(_face_key(corrected[i]), i)
    for i in range(K, len(original)):
        reason = _degenerate(corrected[i])
        key = _face_key(corrected[i])
        if reason is None and key in seen:
            duplicate_of[i] = seen[key]
            reason = "duplicate_C" if seen[key] < K else "duplicate_free"
        if reason is not None:
            keep[i] = False
            reasons[i] = reason
        else:
            seen[key] = i

    source_faces = np.flatnonzero(keep).astype(np.int64)
    output = corrected[keep].copy()
    vertices = []
    vertex_sources = []
    vertex_index = {}
    faces = np.empty((len(output), 3), dtype=np.int64)
    for row, source in enumerate(source_faces):
        for c in range(3):
            p = output[row, c]
            key = _bits(p)
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(p.copy())
                vertex_sources.append(int(target[3 * source + c]))
            faces[row, c] = vertex_index[key]
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    corner_vertex_ids = np.asarray(
        [vertex_index.get(_bits(p), -1) for p in corrected.reshape(-1, 3)],
        dtype=np.int64).reshape(-1, 3)
    face_records = [dict(source_face_id=i, known=i < K, kept=bool(keep[i]),
                         reason=str(reasons[i]), duplicate_of_source_face=int(duplicate_of[i]))
                    for i in range(len(original))]
    if vertices[faces].tobytes() != output.tobytes():
        raise AssertionError("Indexed topology changed triangle coordinates")
    if output[:K].tobytes() != original[:K].tobytes():
        raise AssertionError("Known triangle count/order/winding/bytes changed")
    if _hash(raw) != original_hash:
        raise AssertionError("Caller input was modified")
    moved = np.any(corrected.view(np.uint32) != original.view(np.uint32), axis=2)
    stats = dict(
        status="PASS", input_faces=len(original), output_faces=len(output), known_faces=K,
        input_triangle_soup_vertices=n_corners, output_indexed_vertices=len(vertices),
        deleted_free_faces=int((~keep).sum()), deleted_by_reason=dict(Counter(str(r) for r in reasons[~keep])),
        corner_status_counts=dict(Counter(str(s) for s in kind[known_count:])),
        moved_free_corners=int(moved[K:].sum()),
        max_corner_displacement=float(np.sqrt(displacement2.max())), radius_model=radius,
        known_degenerate_faces_retained=int(known_degenerate),
        known_all_bytes_unchanged=True, indexed_roundtrip_bytes_unchanged=True,
        original_input_unchanged=True, all_corner_displacements_within_radius=True,
        input_sha256=original_hash, output_sha256=_hash(output),
        vertices_sha256=_hash(vertices), faces_sha256=_hash(faces),
        postprocessing_scope="C protected; free exact degenerate/duplicate deletion and bounded vertex consolidation",
        geometry_repair_guarantees="No manifold, orientation, self-intersection, or watertightness guarantee",
        seconds=time.perf_counter() - started, model_calls=0,
    )
    return dict(output=output, vertices=vertices, faces=faces,
                source_face_ids=source_faces, source_corner_ids=np.tile(np.arange(3), (len(output), 1)),
                vertex_source_input_corner_ids=np.asarray(vertex_sources, dtype=np.int64),
                corner_displacement=np.sqrt(displacement2), corner_displacement_vector=displacement_vector,
                corner_target_input_ids=target.reshape(-1, 3), corner_vertex_ids=corner_vertex_ids,
                corner_status=kind.reshape(-1, 3), corner_C_candidate_count=anchor_candidates.reshape(-1, 3),
                source_face_kept=keep, face_records=face_records, stats=stats)
