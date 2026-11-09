"""Finite-triangle references and fixed-partition refits for clay diagnostics.

The reference is the declared piecewise-linear (PL) mesh, not the continuous
deformation evaluated at arbitrary samples. Samples retain their old Voronoi
labels and are mapped by barycentric coordinates through shared-edge midpoint
subdivision. No layout fitting, vertex relocation, repair, or extraction occurs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def shrink_plane_quadrics(quadrics, scale):
    """Transform unit-normal plane Q under x'=scale*x, preserving unit normals.

    This is scale**2 * H**(-T) @ Q @ H**(-1), H=diag(scale,scale,scale,1).
    Thus A remains unchanged; linear blocks scale once and the constant twice.
    This common scale choice must also be used by the other experimental arms.
    """
    q = np.asarray(quadrics, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (4, 4) or not np.isfinite(q).all():
        raise ValueError("Expected finite K x 4 x 4 quadrics")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Shrink must be finite and positive")
    d = np.array([1.0, 1.0, 1.0, float(scale)])
    return q * d[None, :, None] * d[None, None, :]


def _validate_mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError("Expected finite N x 3 vertices")
    if f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("Expected integer M x 3 faces")
    f = f.astype(np.int64, copy=False)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("Face indices out of range")
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or np.any(area2 <= 0):
        raise ValueError("Reference has nonfinite or zero-area triangles")
    return v, f, area2, cross / area2[:, None]


def replay_source_samples(vertices, faces, count, seed):
    """Replay common.sample exactly, additionally exposing its barycentric data."""
    v, f, area2, normals = _validate_mesh(vertices, faces)
    if int(count) != count or count < 1:
        raise ValueError("Positive integer sample count required")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(f), int(count), p=area2 / area2.sum())
    uv = rng.random((int(count), 2))
    root = np.sqrt(uv[:, 0])
    bary = np.c_[1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]]
    points = np.einsum("ij,ijk->ik", bary, v[f[ids]])
    return dict(points=points, normals=normals[ids], face_ids=ids,
                barycentric=bary, source_double_areas=area2)


def _subdivide_once(vertices, faces):
    """Four children per parent, one shared midpoint per indexed undirected edge."""
    count = len(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    midpoint = (vertices[unique[:, 0]] + vertices[unique[:, 1]]) * 0.5
    new_vertices = np.concatenate([vertices, midpoint])
    ab = inverse[:count] + len(vertices)
    bc = inverse[count:2 * count] + len(vertices)
    ca = inverse[2 * count:] + len(vertices)
    a, b, c = faces.T
    children = np.stack([np.c_[a, ab, ca], np.c_[ab, b, bc],
                         np.c_[ca, bc, c], np.c_[ab, bc, ca]], axis=1)
    return new_vertices, children.reshape(-1, 3)


def _descend_samples(face_ids, barycentric):
    """Associate samples with one of the four oriented children deterministically."""
    w = barycentric
    # Ties on shared child edges select the earliest corner child.
    child = np.full(len(w), 3, dtype=np.int64)
    child[w[:, 2] >= 0.5] = 2
    child[w[:, 1] >= 0.5] = 1
    child[w[:, 0] >= 0.5] = 0
    local = 2 * w.copy()
    for corner in range(3):
        mask = child == corner
        local[mask, corner] -= 1
    mid = child == 3
    # Child vertices are (ab, bc, ca), in this order.
    local[mid] = np.c_[1 - 2 * w[mid, 2], 1 - 2 * w[mid, 0], 1 - 2 * w[mid, 1]]
    if float(local.min()) < -1e-12 or float(local.max()) > 1 + 1e-12:
        raise RuntimeError("Invalid child barycentric association")
    return 4 * face_ids + child, local


def prepare_reference(vertices, faces, base_p, object_index, levels, shrink=0.7,
                      samples_fit=65536, fit_seed=1701):
    """Build a refined PL reference and immutable sample/partition association.

    Parameters refer to the unshrunk archived clean reference and archived fit p.
    `levels` applies shared-edge midpoint subdivision before the common shrink.
    Returned vertices and association.base_samples are already shrunk exactly once.
    """
    source_v, source_f, _, _ = _validate_mesh(vertices, faces)
    p = np.asarray(base_p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or not np.isfinite(p).all():
        raise ValueError("Expected finite K x 3 archived sampling points p")
    if int(levels) != levels or levels < 0:
        raise ValueError("Nonnegative integer subdivision level required")
    if not np.isfinite(shrink) or shrink <= 0:
        raise ValueError("Positive finite common shrink required")
    sample = replay_source_samples(source_v, source_f, samples_fit, fit_seed + int(object_index))
    labels = cKDTree(p).query(sample["points"], workers=1)[1]
    counts = np.bincount(labels, minlength=len(p))
    if np.any(counts == 0):
        raise ValueError("Frozen source Voronoi partition contains empty elements")
    refined_v, refined_f = source_v.copy(), source_f.copy()
    child_ids, bary = sample["face_ids"].copy(), sample["barycentric"].copy()
    parent_ids = np.arange(len(source_f), dtype=np.int64)
    for _ in range(int(levels)):
        refined_v, refined_f = _subdivide_once(refined_v, refined_f)
        child_ids, bary = _descend_samples(child_ids, bary)
        parent_ids = np.repeat(parent_ids, 4)
    refined_v = refined_v * float(shrink)
    refined_v, refined_f, base_area2, _ = _validate_mesh(refined_v, refined_f)
    base_pl_samples = np.einsum("ij,ijk->ik", bary, refined_v[refined_f[child_ids]])
    base_samples = sample["points"] * float(shrink)
    errors = np.linalg.norm(base_pl_samples - base_samples, axis=1)
    if float(errors.max()) > 1e-11 * max(1.0, float(np.max(np.abs(refined_v)))):
        raise RuntimeError("Subdivision association changed the source surface samples")
    association = dict(
        sample_child_faces=child_ids, sample_barycentric=bary, labels=labels,
        source_face_ids=sample["face_ids"], source_barycentric=sample["barycentric"],
        source_samples=sample["points"], old_normals=sample["normals"],
        base_samples=base_samples, base_pl_samples=base_pl_samples,
        base_child_double_areas=base_area2, base_reference_vertices=refined_v.copy(),
        reference_faces=refined_f.copy(), child_parent_faces=parent_ids,
        base_p=p.copy() * float(shrink), element_count=len(p),
        source_count=len(sample["points"]), object_index=int(object_index),
        seed=fit_seed + int(object_index), shrink=float(shrink), levels=int(levels))
    diagnostics = dict(source_faces=len(source_f), reference_faces=len(refined_f),
                       source_vertices=len(source_v), reference_vertices=len(refined_v),
                       levels=int(levels), shrink=float(shrink),
                       maximum_barycentric_reconstruction_error=float(errors.max()),
                       rms_barycentric_reconstruction_error=float(np.sqrt(np.mean(errors**2))),
                       minimum_sample_barycentric=float(bary.min()),
                       maximum_sample_barycentric_sum_error=float(np.max(np.abs(bary.sum(1) - 1))),
                       minimum_samples_per_frozen_element=int(counts.min()),
                       source_face_association_matches=bool(np.array_equal(parent_ids[child_ids], sample["face_ids"])))
    return dict(vertices=refined_v, faces=refined_f, association=association,
                diagnostics=diagnostics)


def refit_frame(deformed_ref_vertices, faces, association):
    """Reestimate Q/n on the current PL surface using frozen sample partitions.

    Each old area-uniform sample retains its refined child and barycentric
    coordinates. Its importance weight is the child area ratio. New sample normals
    are current child unit normals. Q is their weighted mean homogeneous plane Q;
    n is the weighted mean unit normal, not forcibly renormalized.
    """
    v, f, area2, normals = _validate_mesh(deformed_ref_vertices, faces)
    if not np.array_equal(f, association["reference_faces"]):
        raise ValueError("Reference connectivity must remain frozen")
    if v.shape != association["base_reference_vertices"].shape:
        raise ValueError("Reference vertex shape changed")
    ids = association["sample_child_faces"]
    bary = association["sample_barycentric"]
    labels = association["labels"]
    count = association["element_count"]
    points = np.einsum("ij,ijk->ik", bary, v[f[ids]])
    sample_normals = normals[ids]
    weights = (area2 / association["base_child_double_areas"])[ids]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid PL surface area transport weights")
    denom = np.bincount(labels, weights=weights, minlength=count)
    if np.any(denom <= 0):
        raise ValueError("Empty weighted frozen partition")
    planes = np.c_[sample_normals, -np.einsum("ij,ij->i", sample_normals, points)]
    quadrics = np.zeros((count, 4, 4), dtype=np.float64)
    means = np.zeros((count, 3), dtype=np.float64)
    for i in range(4):
        for j in range(i, 4):
            value = np.bincount(labels, weights=weights * planes[:, i] * planes[:, j], minlength=count) / denom
            quadrics[:, i, j] = value
            quadrics[:, j, i] = value
    for i in range(3):
        means[:, i] = np.bincount(labels, weights=weights * sample_normals[:, i], minlength=count) / denom
    if not np.isfinite(quadrics).all() or not np.isfinite(means).all():
        raise RuntimeError("Refit generated nonfinite parameters")
    diagnostics = dict(sample_count=len(points), element_count=count,
                       minimum_area_ratio=float(weights.min()), maximum_area_ratio=float(weights.max()),
                       mean_area_ratio=float(weights.mean()),
                       minimum_weighted_partition_mass=float(denom.min()),
                       minimum_mean_normal_length=float(np.linalg.norm(means, axis=1).min()),
                       reference_total_area=float(area2.sum() / 2),
                       reference_minimum_triangle_area=float(area2.min() / 2),
                       frozen_labels=True, relocated_vertices=False,
                       samples_on_declared_pl_reference=True)
    return dict(Q=quadrics, n=means, samples=points, sample_normals=sample_normals,
                sample_weights=weights, diagnostics=diagnostics)
