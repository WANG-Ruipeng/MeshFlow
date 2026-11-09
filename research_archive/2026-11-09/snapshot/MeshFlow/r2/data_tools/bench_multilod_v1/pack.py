"""Versioned Native packing for fixed-C, interface-preserving free simplification.

Pure CPU array construction.  ``new_vertices`` is FP32 model-space geometry in
the ORIGINAL vertex-slot domain (unused slots are retained). ``new_faces`` is
oriented, not the model's canonical corner order. ``kept_face_ids`` records
which original face slot produced each survivor; it is lineage, not the new
face identity.  Source supervision is recomputed by the existing finite-face
detector and is never an inference feature.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

VERSION = "BENCH_FIXED_C_MULTILOD_PACK_V1"
N_MIN, N_MAX = 128, 799
C_IMMUTABLE = (
    "C", "C_pre_ot", "C_vertices", "C_faces_original", "C_faces_model",
    "C_source_vertex_ids", "C_source_corner_permutations", "C_source_faces_original",
)


def _require(ok, text):
    if not bool(ok):
        raise ValueError(text)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _coordinates(original, vertices):
    """Invert the recorded normalization only; never renormalize the new LOD."""
    tr = json.loads(str(np.asarray(original["transform_json"]).item()))
    required = {"center1", "center2", "center3", "divisor1", "divisor3", "dataset_std", "model_scale"}
    _require(required <= set(tr), "Unknown source transform: cannot infer an inverse")
    c1, c2, c3 = [np.asarray(tr[k], np.float64) for k in ("center1", "center2", "center3")]
    _require(all(c.shape == (3,) and np.isfinite(c).all() for c in (c1, c2, c3)), "Invalid recorded transform centers")
    d1, d3, std, scale = [float(tr[k]) for k in ("divisor1", "divisor3", "dataset_std", "model_scale")]
    _require(all(np.isfinite(x) and x > 0 for x in (d1, d3, std, scale)), "Invalid recorded transform scales")
    _require(scale == 2. and std == .3762, "This Native packing requires the recorded 2/.3762 contract")
    old = np.asarray(original["model_vertices"])
    changed = np.any(vertices.view(np.uint32) != old.view(np.uint32), axis=1)
    norm = np.asarray(original["normalized_vertices"]).copy()
    source = np.asarray(original["source_vertices"]).copy()
    _require(norm.shape == source.shape == vertices.shape, "Coordinate domains must retain original vertex slots")
    _require(norm.dtype == source.dtype == np.float32, "Source and normalized coordinate domains must be FP32")
    # Preserve all unmodified rows byte-for-byte, including their rounding.
    norm[changed] = vertices[changed] * np.float32(std / scale)
    source[changed] = ((norm[changed].astype(np.float64) * d3 + c3 + c2) * d1 + c1).astype(np.float32)
    return norm, source, changed


def _canonical(vertices, normalized, faces):
    """Same per-face canonicalization as bench_user_v2.build.canonical."""
    rank = np.argsort(np.lexsort((normalized[:, 0], normalized[:, 2], normalized[:, 1])))
    permutation = np.argsort(rank[faces], axis=1, kind="stable").astype(np.int64)
    half = vertices / np.float32(2)
    pre = half[np.take_along_axis(faces, permutation, axis=1)]
    first = np.lexsort((pre[:, :, 2], pre[:, :, 1], pre[:, :, 0]), axis=1)[:, 0]
    permutation = np.take_along_axis(permutation, (first[:, None] + np.arange(3)[None, :]) % 3, axis=1)
    return np.take_along_axis(faces, permutation, axis=1), permutation


def _features(vertices, faces, edges, n):
    """Production Natural-N's 24 metadata fields; no neural import/call."""
    L = float(np.sqrt(3) * 2 / .3762)
    k = len(faces)
    if not len(edges):
        return np.empty((0, 24), np.float32)
    center, span = (vertices.min(0) + vertices.max(0)) * .5, np.ptp(vertices, axis=0)
    degree = np.bincount(edges[:, 1:].reshape(-1), minlength=len(vertices))
    out = []
    for fi, u, w in edges:
        a, b = vertices[u], vertices[w]
        delta = b - a
        normal = np.cross(vertices[faces[fi, 1]] - vertices[faces[fi, 0]],
                          vertices[faces[fi, 2]] - vertices[faces[fi, 0]])
        norm = np.linalg.norm(normal)
        _require(norm >= 1e-12, "Degenerate C has no defined boundary feature")
        out.append(np.r_[(a+b)*.5/L, delta/L, np.abs(delta)/L, normal/norm,
                         span/L, center/L, np.linalg.norm(delta)/L, k/256., n/256.,
                         degree[u]/8., degree[w]/8., 1.])
    return np.asarray(out, np.float32)


def validate_case(case, original=None):
    """CPU identity checks; not a whole-mesh quality certificate or OT run."""
    n, k = int(case["N"]), int(case["K"])
    _require(N_MIN <= n <= N_MAX and 2 <= k < n, "Native N128..799 / K contract")
    v, f, canonical, p = (case[x] for x in ("model_vertices", "source_faces", "source_face_vertex_ids", "source_corner_permutations"))
    known, free = case["known_ids"], case["free_ids"]
    _require(_same(np.sort(np.r_[known, free]), np.arange(n, dtype=np.int64)), "C/Y partition is not exhaustive/disjoint")
    _require(np.array_equal(np.sort(p, 1), np.tile(np.arange(3), (n, 1))), "Invalid corner permutation")
    _require(_same(np.take_along_axis(f, p, 1), canonical), "Canonical/source winding mapping broken")
    _require(_same(v[canonical], case["full_target"]), "Model coordinate roundtrip broken")
    _require(_same(case["pre_ot_full"] * np.float32(2), case["full_target"]), "Pre-OT scale broken")
    _require(_same(case["C"], case["full_target"][known]), "C not the exact target subset")
    _require(_same(case["Y"], case["full_target"][free]), "Y not the exact target complement")
    _require(_same(case["C_source_faces_original"], f[known]), "C oriented source identities broken")
    _require(_same(case["C_vertices"], v[case["C_source_vertex_ids"]]), "C vertex domain changed")
    _require(_same(case["C_source_corner_permutations"], p[known]), "C source permutation mismatch")
    _require(_same(case["C_vertices"][case["C_faces_model"]], case["C"]), "C local model indices broken")
    _require(_same(np.take_along_axis(case["C_faces_original"], p[known], 1), case["C_faces_model"]), "C source/model winding inverse broken")
    _require(_same(case["source_face_ids"], known) and _same(case["free_source_face_ids"], free), "Face IDs must refer to the new geometry version")
    own, alias, supported = (case[x] for x in ("occupied_labels", "source_alias_labels", "source_supported_labels"))
    _require(all(x.dtype == bool and x.shape == (len(case["edge_records"]),) for x in (own, alias, supported)), "Source label shape/dtype")
    _require(np.array_equal(own | alias, supported) and not np.any(own & alias), "Source role partition")
    owner = case["edge_owner_face"]
    _require(np.all(owner[~supported] == -1) and np.isin(owner[supported], free).all(), "Stale source interface face ID")
    _require(len(np.unique(owner[own])) == int(own.sum()), "A target face is assigned multiple owner slots")
    for fi in np.unique(owner[supported]):
        es = np.flatnonzero(supported & (owner == fi))
        _require(own[es[0]] and own[es].sum() == 1, "Lowest-edge owner/alias policy differs")
    if original is not None:
        for name in C_IMMUTABLE + ("transform_json",):
            _require(_same(case[name], original[name]), "Immutable field changed: " + name)
    return dict(status="PASS", N=n, K=k, C_byte_identity=True,
                same_version_C_Y_partition=True, inverse_corner_order_exact=True,
                source_face_identity_domain="target_version_id", model_calls=0, OT_calls=0)


def pack_case(original, new_vertices, new_faces, kept_face_ids):
    """Package one candidate LOD and return (case, identity/lineage audit).

    Interface legality labels are recomputed with the retained source scorer.
    The caller must separately enforce simplification/fidelity/quality gates.
    """
    v, f, kept = map(np.asarray, (new_vertices, new_faces, kept_face_ids))
    old_v, old_f = original["model_vertices"], original["source_faces"]
    _require(v.dtype == np.float32 and v.shape == old_v.shape and np.isfinite(v).all(), "Expected FP32 original vertex-slot domain")
    _require(f.dtype == np.int64 and f.ndim == 2 and f.shape[1:] == (3,), "Expected oriented int64[N,3] source faces")
    _require(kept.dtype == np.int64 and kept.shape == (len(f),), "Expected one original face lineage ID per new face")
    _require(len(set(kept.tolist())) == len(kept) and np.isin(kept, np.arange(len(old_f))).all(), "Survivor face lineage must be a unique subset")
    _require(len(f) and f.min() >= 0 and f.max() < len(v), "Face vertex IDs outside original domain")
    _require(np.all(np.diff(np.sort(f, axis=1), axis=1) != 0), "Repeated vertex in a face")
    old_known = np.asarray(original["known_ids"], np.int64)
    inverse = {int(old): new for new, old in enumerate(kept)}
    _require(all(int(x) in inverse for x in old_known), "A fixed C face was deleted")
    known = np.asarray([inverse[int(x)] for x in old_known], np.int64)
    free = np.setdiff1d(np.arange(len(f), dtype=np.int64), known)
    _require(_same(f[known], old_f[old_known]), "A fixed C face or its source winding was changed")
    cvid = original["C_source_vertex_ids"]
    _require(_same(v[cvid], old_v[cvid]), "A fixed C vertex was moved")
    norm, source, moved = _coordinates(original, v)
    canonical, perm = _canonical(v, norm, f)
    # C is an immutable USER array, including internal canonical corner order.
    canonical[known] = original["source_face_vertex_ids"][old_known]
    perm[known] = original["source_corner_permutations"][old_known]
    geometry_hash = _digest(v, f, kept)
    version = VERSION + "_" + geometry_hash[:16]
    result = dict(model_vertices=v.copy(), source_vertices=source, normalized_vertices=norm,
                  transform_json=original["transform_json"].copy(), source_faces=f.copy(),
                  source_face_vertex_ids=canonical, source_corner_permutations=perm,
                  full_target=v[canonical].copy(), pre_ot_full=(v / np.float32(2))[canonical],
                  target_version_id=np.asarray(version), geometry_version=np.asarray(version),
                  known_ids=known, free_ids=free, source_face_ids=known.copy(), free_source_face_ids=free.copy(),
                  N=np.asarray(len(f), np.int64), K=np.asarray(len(known), np.int64))
    for name in C_IMMUTABLE:
        result[name] = original[name].copy()
    result["Y"] = result["full_target"][free].copy()
    result["Y_pre_ot"] = result["pre_ot_full"][free].copy()
    # Preserve existing detector definitions rather than equating index contact
    # with legal finite-face contact.
    from r2.data_tools.bench_parts_v1.quality import SourceAuditor
    auditor = SourceAuditor(result)
    quality, supervision = auditor.candidate(known, dict(group="TRAIN", candidate_id=version))
    for name in C_IMMUTABLE:
        if name in supervision:
            _require(_same(supervision[name], result[name]), "Source auditor changed fixed C domain: " + name)
    result.update(supervision)
    result["source_third_in_C"] = np.asarray([
        bool(ok and int(triple[2]) in set(cvid.tolist()))
        for ok, triple in zip(result["source_supported_labels"], result["edge_source_triple"])
    ], bool)
    result["edge_features"] = _features(result["C_vertices"], result["C_faces_original"], result["edge_records"], len(f))
    result["multilod_parent_target_version_id"] = original["target_version_id"].copy()
    result["multilod_parent_face_slot_ids"] = kept.copy()
    result["multilod_geometry_sha256"] = np.asarray(geometry_hash)
    result["multilod_pack_schema"] = np.asarray(VERSION)
    changed_faces = np.asarray([not _same(f[i], old_f[old]) or not _same(v[f[i]], old_v[old_f[old]])
                                for i, old in enumerate(kept)], bool)
    result["multilod_face_geometry_changed"] = changed_faces
    audit = validate_case(result, original)
    audit.update(schema=VERSION, target_version_id=version, geometry_sha256=geometry_hash,
        parent_target_version_id=str(np.asarray(original["target_version_id"]).item()),
        old_N=len(old_f), new_N=len(f), old_known_face_ids=old_known.tolist(), new_known_face_ids=known.tolist(),
        parent_face_slot_ids=kept.tolist(), removed_parent_face_slots=np.setdiff1d(np.arange(len(old_f)), kept).tolist(),
        changed_new_face_slots=np.flatnonzero(changed_faces).tolist(), moved_vertex_slots=np.flatnonzero(moved).tolist(),
        source_coordinate_policy="Original recorded inverse transform; unchanged vertex rows retained bitwise; no renormalization",
        source_label_contract="Recomputed original finite detector; lowest C-edge ID owner, remaining supported edges alias",
        source_owner_count=int(result["occupied_labels"].sum()), source_alias_count=int(result["source_alias_labels"].sum()),
        source_null_count=int((~result["source_supported_labels"]).sum()),
        source_interface_rows=quality["source_interface_rows"], source_C_geometry_status=quality["geometry_status"],
        source_C_geometry_reasons=quality["reasons"], detector_calls=dict(auditor.calls),
        lineage_scope="Survivor face slots and changed geometry only; full vertex-collapse ancestry belongs to simplifier ledger",
        quality_scope="Packing/source boundary supervision only; separate complete-target fidelity and quality gates required")
    return result, audit
