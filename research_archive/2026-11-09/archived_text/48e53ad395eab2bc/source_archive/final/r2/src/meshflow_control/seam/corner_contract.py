"""Pure NumPy corner contract for the seam-slot CPU prototype.

Single-sample shapes are x[N,3,3], face masks[N], anchor_ids[N,3],
C_vertices[V_C,3].  Anchor IDs address the preserved C vertex table; -1
means an independently evolving corner.  No network, GPU, welding, source
completion lookup, or occupancy prediction is performed here.

An inactive candidate slot is an ordinary three-corner bulk triangle.  Its
face remains valid and counts towards N.  A caller must supply the active
occupancy decision explicitly; this module never infers it from geometry.
"""
from __future__ import annotations

import numpy as np

VERSION = "seam_slot_corner_contract_numpy_v1"


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def _bool_array(value, shape, name):
    a = np.asarray(value)
    if a.dtype != np.bool_ or a.shape != shape:
        raise ValueError(f"{name} must be bool with shape {shape}")
    return a


def _ids(value, shape, name):
    a = np.asarray(value)
    if a.dtype != np.int64 or a.shape != shape or np.any(a < -1):
        raise ValueError(f"{name} must be int64 with shape {shape}, values >= -1")
    return a


def _coordinates(value, shape, name):
    a = np.asarray(value)
    if a.dtype not in (np.dtype("float32"), np.dtype("float64")) or a.shape != shape:
        raise ValueError(f"{name} must be float32/float64 with shape {shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def _face_masks(known_face_mask, valid_face_mask):
    known = np.asarray(known_face_mask)
    if known.ndim != 1:
        raise ValueError("Face masks must be one-dimensional single-sample arrays")
    known = _bool_array(known, known.shape, "known_face_mask")
    valid = _bool_array(valid_face_mask, known.shape, "valid_face_mask")
    if np.any(known & ~valid):
        raise ValueError("Every original C face must be valid")
    return known, valid


def corner_masks(known_face_mask, valid_face_mask, anchor_vertex_ids):
    """Validate identity masks and return valid/known/free corner partitions.

    Original C faces require all three IDs.  Padding requires all IDs=-1.
    Extra fixed references on free faces are permitted, including a third C
    vertex reference; the active seam builder below creates only two.
    """
    known, valid = _face_masks(known_face_mask, valid_face_mask)
    ids = _ids(anchor_vertex_ids, (len(known), 3), "anchor_vertex_ids")
    anchored = ids >= 0
    if not anchored[known].all():
        raise ValueError("Original C faces require three preserved vertex references")
    if anchored[~valid].any():
        raise ValueError("Padding cannot contain anchor references")
    for row in ids:
        active = row[row >= 0]
        if len(np.unique(active)) != len(active):
            raise ValueError("A triangle cannot reference the same C vertex twice")
    valid_corner = np.broadcast_to(valid[:, None], ids.shape).copy()
    free = valid_corner & ~anchored
    return dict(valid_corner_mask=valid_corner, known_corner_mask=anchored.copy(),
                free_corner_mask=free, free_scalar_count=int(free.sum()) * 3,
                valid_face_count=int(valid.sum()), C_face_count=int(known.sum()))


def build_anchor_ids(known_face_mask, valid_face_mask, c_face_vertex_ids,
                     seam_endpoint_ids, occupied):
    """Create anchor IDs from original C plus explicitly active seam slots.

    c_face_vertex_ids[N,3] must be -1 outside C.  seam_endpoint_ids[N,2]
    holds ordered C IDs (caller chooses opposite C winding), or [-1,-1]
    for noncandidates.  occupied[N] is bool.  False candidate occupancy
    returns that slot to bulk with all three corner IDs=-1.
    """
    known, valid = _face_masks(known_face_mask, valid_face_mask)
    n = len(known)
    cids = _ids(c_face_vertex_ids, (n, 3), "c_face_vertex_ids")
    edge = _ids(seam_endpoint_ids, (n, 2), "seam_endpoint_ids")
    occ = _bool_array(occupied, (n,), "occupied")
    if np.any(cids[~known] != -1) or np.any(cids[known] < 0):
        raise ValueError("C vertex references must occur on original C faces only")
    has = edge >= 0
    if np.any(has[:, 0] != has[:, 1]):
        raise ValueError("Candidate seam requires both endpoint IDs")
    candidates = has.all(axis=1)
    if np.any(candidates & (known | ~valid)):
        raise ValueError("Only valid non-C faces can be candidate seam slots")
    if np.any(candidates & (edge[:, 0] == edge[:, 1])):
        raise ValueError("Candidate seam endpoints must have distinct IDs")
    if np.any(occ & ~candidates):
        raise ValueError("Occupied noncandidate/C/padding slot")
    keys = [tuple(sorted(map(int, row))) for row in edge[candidates]]
    if len(keys) != len(set(keys)):
        raise ValueError("At most one candidate slot may own each original C edge")
    result = np.full((n, 3), -1, dtype=np.int64)
    result[known] = cids[known]
    result[occ, :2] = edge[occ]
    corner_masks(known, valid, result)
    return result


def validate_contract(x, known_face_mask, valid_face_mask, anchor_vertex_ids,
                      c_vertices, *, require_anchors_exact=False):
    """Validate arrays without silently coercing dtypes, IDs, or masks."""
    masks = corner_masks(known_face_mask, valid_face_mask, anchor_vertex_ids)
    n = len(np.asarray(known_face_mask))
    a = _coordinates(x, (n, 3, 3), "x")
    c = np.asarray(c_vertices)
    if c.ndim != 2 or c.shape[1:] != (3,):
        raise ValueError("c_vertices must have shape [V_C,3]")
    c = _coordinates(c, c.shape, "c_vertices")
    if a.dtype != c.dtype:
        raise ValueError("x and c_vertices must share dtype for exact preservation")
    ids = np.asarray(anchor_vertex_ids)
    fixed = masks["known_corner_mask"]
    if np.any(ids[fixed] >= len(c)):
        raise ValueError("Anchor ID outside preserved C vertex table")
    if np.any(a[~masks["valid_corner_mask"]] != 0):
        raise ValueError("Padding coordinates must be exact zeros")
    exact = _same_bits(a[fixed], c[ids[fixed]])
    if require_anchors_exact and not exact:
        raise ValueError("Anchor coordinates differ from preserved C references")
    return dict(**masks, anchors_exact=bool(exact), dtype=str(a.dtype))


def corner_conditioning(t, known_face_mask, valid_face_mask, anchor_vertex_ids):
    """Select time/role values; no embeddings or neural forward are executed.

    Corner MLP/final modulation: anchors use 1, independent corners use t.
    Face attention modulation: original C uses 1; seam and bulk use t.
    Padding uses time=0/role=0 and remains excluded by valid_face_mask.
    This is not a claim that face attention has per-corner time modulation.
    """
    if np.asarray(t).shape != () or not np.isfinite(t) or not 0 <= float(t) <= 1:
        raise ValueError("t must be a finite scalar in [0,1]")
    known, valid = _face_masks(known_face_mask, valid_face_mask)
    masks = corner_masks(known, valid, anchor_vertex_ids)
    fixed = masks["known_corner_mask"]
    times = np.where(fixed, 1.0, float(t))
    times[~masks["valid_corner_mask"]] = 0.0
    roles = fixed.astype(np.int64)
    face_time = np.where(known, 1.0, float(t))
    face_time[~valid] = 0.0
    return dict(corner_time=times, corner_role_ids=roles, face_attention_time=face_time,
                free_corner_mask=masks["free_corner_mask"])


def clamp_coordinates(x, known_face_mask, valid_face_mask, anchor_vertex_ids, c_vertices):
    """Copy x and overwrite anchored corners from the immutable C table."""
    facts = validate_contract(x, known_face_mask, valid_face_mask, anchor_vertex_ids, c_vertices)
    result = np.array(x, copy=True)
    fixed = facts["known_corner_mask"]
    result[fixed] = np.asarray(c_vertices)[np.asarray(anchor_vertex_ids)[fixed]]
    validate_contract(result, known_face_mask, valid_face_mask, anchor_vertex_ids,
                      c_vertices, require_anchors_exact=True)
    return result


def masked_velocity_loss(prediction, target, known_face_mask, valid_face_mask,
                         anchor_vertex_ids, *, denominator=None):
    """Mean square on independent valid coordinates, with optional batch divisor.

    Target anchored/padding velocities must be zero.  Predictions there are
    ignored, not penalized towards zero.  Accumulation is float64 in this CPU
    contract; production BF16/FP32 arithmetic is not tested here.
    """
    facts = corner_masks(known_face_mask, valid_face_mask, anchor_vertex_ids)
    shape = (len(np.asarray(known_face_mask)), 3, 3)
    pred = _coordinates(prediction, shape, "prediction")
    truth = _coordinates(target, shape, "target")
    free = facts["free_corner_mask"]
    if np.any(truth[~free] != 0):
        raise ValueError("Fixed and padding target velocities must be exactly zero")
    count = facts["free_scalar_count"]
    if count <= 0:
        raise ValueError("FM loss requires at least one independent coordinate")
    if denominator is None:
        divisor = count
    else:
        if isinstance(denominator, (bool, np.bool_)) or not isinstance(denominator, (int, np.integer)):
            raise ValueError("denominator must be an integer scalar count")
        divisor = int(denominator)
    if divisor < count:
        raise ValueError("Batch denominator cannot be smaller than this microbatch count")
    error = pred[free].astype(np.float64) - truth[free].astype(np.float64)
    sse = float(np.square(error).sum(dtype=np.float64))
    return dict(loss=sse / divisor, squared_error_sum=sse,
                free_scalar_count=count, denominator=divisor)


def euler_step(x, velocity, dt, known_face_mask, valid_face_mask,
               anchor_vertex_ids, c_vertices):
    """Integrate only independent corners, then restore all exact C references."""
    facts = validate_contract(x, known_face_mask, valid_face_mask, anchor_vertex_ids,
                              c_vertices, require_anchors_exact=True)
    v = _coordinates(velocity, np.asarray(x).shape, "velocity")
    if v.dtype != np.asarray(x).dtype:
        raise ValueError("State and integrated velocity must share dtype")
    if np.asarray(dt).shape != () or not np.isfinite(dt) or float(dt) < 0:
        raise ValueError("dt must be a finite nonnegative scalar")
    out = np.array(x, copy=True)
    free = facts["free_corner_mask"]
    out[free] = out[free] + np.asarray(dt, dtype=out.dtype) * v[free]
    return clamp_coordinates(out, known_face_mask, valid_face_mask, anchor_vertex_ids, c_vertices)


def straight_endpoint_estimate(x, velocity, t, known_face_mask, valid_face_mask,
                               anchor_vertex_ids, c_vertices):
    """Yhat=X+(1-t)v on independent corners of a straight FM path only."""
    if np.asarray(t).shape != () or not np.isfinite(t) or not 0 <= float(t) <= 1:
        raise ValueError("t must be a finite scalar in [0,1]")
    return euler_step(x, velocity, 1.0 - float(t), known_face_mask, valid_face_mask,
                      anchor_vertex_ids, c_vertices)


def decode_shared_indices(x, known_face_mask, valid_face_mask, anchor_vertex_ids,
                          c_vertices):
    """Decode exact anchor IDs, leaving every free corner with its own new ID.

    No coordinate-based deduplication/welding, face deletion, normal change,
    or topology guarantee is hidden here.  Even an independent corner whose
    coordinates equal C retains its independent ID.  The caller's downstream
    cleanup must preserve the explicit anchor provenance.
    """
    facts = validate_contract(x, known_face_mask, valid_face_mask, anchor_vertex_ids,
                              c_vertices, require_anchors_exact=True)
    n = len(np.asarray(known_face_mask))
    ids = np.array(anchor_vertex_ids, copy=True)
    free = facts["free_corner_mask"]
    ids[free] = np.arange(int(free.sum()), dtype=np.int64) + len(c_vertices)
    vertices = np.concatenate((np.asarray(c_vertices), np.asarray(x)[free]), axis=0)
    valid = np.asarray(valid_face_mask)
    faces = ids[valid].copy()
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Decoded face indices outside vertex table")
    if not _same_bits(vertices[faces], np.asarray(x)[valid]):
        raise ValueError("Shared-index decode changed triangle coordinates")
    if ids.shape != (n, 3):
        raise AssertionError("Internal corner mapping shape changed")
    return dict(vertices=vertices, faces=faces, corner_to_vertex=ids,
                valid_face_indices=np.flatnonzero(valid).astype(np.int64),
                independent_free_vertices=int(free.sum()),
                fixed_reference_corners=int(facts["known_corner_mask"].sum()))
