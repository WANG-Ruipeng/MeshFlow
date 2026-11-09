"""Independent surface and mesh-quality measurements for layout-value experiments.

Nothing here repairs, welds, projects, clips, or otherwise changes the supplied
mesh. Surface error uses area-uniform samples and nearest *triangle* distances,
not nearest vertices or a nearest sampled-point surrogate. The two directed
mean-square errors receive equal weight in the symmetric RMS.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import trimesh

from ..mesh_ops import topology
from ..legacy.exact_accel import audit as exact_audit


def _arrays(vertices, faces, *, allow_empty=False):
    v = np.asarray(vertices, dtype=np.float64)
    original_f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if original_f.ndim != 2 or original_f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.isfinite(v).all():
        raise ValueError("nonfinite vertex coordinates")
    if not np.issubdtype(original_f.dtype, np.integer):
        raise ValueError("face indices must have integer dtype")
    f = np.asarray(original_f, dtype=np.int64)
    if not allow_empty and (not len(v) or not len(f)):
        raise ValueError("empty surface")
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError("face index out of bounds")
    return v, f


def _native(value: Any):
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return value


def mesh_quality(v, f):
    """Quality of actual 3D triangles; q05 is over per-face minimum angles.

    Degeneracy uses the inherited topology auditor's scale-relative threshold.
    Degenerate faces get minimum angle and normalized shape quality zero.
    They stay in every quantile's denominator.
    """
    v, f = _arrays(v, f)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    area2_tolerance = max(diagonal**2, 1e-300) * 1e-14
    degenerate = area2 <= area2_tolerance
    angle_columns = []
    for i in range(3):
        a = tri[:, (i + 1) % 3] - tri[:, i]
        b = tri[:, (i + 2) % 3] - tri[:, i]
        angle_columns.append(np.rad2deg(np.arctan2(
            np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b)
        )))
    angles = np.stack(angle_columns, axis=1)
    minimum = angles.min(axis=1)
    minimum[degenerate] = 0.0
    edge_lengths = np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2)
    squared_sum = np.square(edge_lengths).sum(axis=1)
    shape_quality = np.divide(2 * np.sqrt(3) * area2, squared_sum,
                              out=np.zeros_like(area2), where=squared_sum > 0)
    shape_quality[degenerate] = 0.0
    # Avoid cancellation from a large translation in the signed-volume sum.
    centre = (v.min(axis=0) + v.max(axis=0)) * 0.5
    shifted = tri - centre
    signed_volume = np.einsum("ij,ij->i", shifted[:, 0],
                             np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6.0
    return {
        "face_count": int(len(f)),
        "min_angle_deg": float(minimum.min()),
        "angle_q05_deg": float(np.quantile(minimum, 0.05)),
        "angle_median_deg": float(np.median(minimum)),
        "angle_q05_definition": "linear quantile of one minimum angle per 3D triangle",
        "degenerate_count": int(degenerate.sum()),
        "area2_degeneracy_tolerance": float(area2_tolerance),
        "area_min": float(0.5 * area2.min()),
        "area_total": float(0.5 * area2.sum()),
        "shape_quality_min": float(shape_quality.min()),
        "shape_quality_q05": float(np.quantile(shape_quality, 0.05)),
        "shape_quality_median": float(np.median(shape_quality)),
        "shape_quality_definition": "4*sqrt(3)*area / sum(squared edge lengths); equilateral=1",
        "signed_volume": float(signed_volume),
        "bbox_diagonal": diagonal,
    }


def _surface_samples(v, f, count, rng):
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = area2.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("surface has no finite positive area")
    face_ids = rng.choice(len(f), size=count, p=area2 / total)
    uniform = rng.random((count, 2))
    root = np.sqrt(uniform[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uniform[:, 1]), root * uniform[:, 1]))
    return np.einsum("nij,ni->nj", tri[face_ids], bary)


def _triangle_distances(mesh, points, chunk=1024):
    try:
        import rtree  # noqa: F401 -- explicit dependency, no vertex-distance fallback
    except ImportError as exc:
        raise RuntimeError("Triangle proximity requires rtree; install rtree in this environment") from exc
    distances = []
    for start in range(0, len(points), chunk):
        _, distance, _ = trimesh.proximity.closest_point(mesh, points[start:start + chunk])
        distances.append(distance)
    answer = np.concatenate(distances)
    if not np.isfinite(answer).all():
        raise FloatingPointError("nonfinite point-to-triangle distance")
    return answer


def evaluate_mesh(v, f, ref_v, ref_f, points=8192, seed=72101):
    """Two independently sampled area integrals of point-to-triangle distances.

    Symmetric RMS = sqrt((mean(d_mesh_to_ref**2)+mean(d_ref_to_mesh**2))/2).
    Percent normalization uses ONLY the reference surface bounding-box diagonal.
    Sampled maxima are diagnostics, never exact/continuous Hausdorff distances.
    Sampling uses local RNGs and cannot perturb training or optimization RNG.
    """
    v, f = _arrays(v, f)
    ref_v, ref_f = _arrays(ref_v, ref_f)
    if isinstance(points, bool) or int(points) != points or points < 1:
        raise ValueError("points must be a positive integer per surface")
    points = int(points)
    used_reference = ref_v[np.unique(ref_f)]
    lower, upper = used_reference.min(axis=0), used_reference.max(axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("reference bounding-box diagonal must be positive")
    centre = (lower + upper) * 0.5
    # A common transform preserves geometric comparisons and improves numerical scale.
    mesh_v = (v - centre) / diagonal
    reference_v = (ref_v - centre) / diagonal
    sequences = np.random.SeedSequence(seed).spawn(2)
    mesh_points = _surface_samples(mesh_v, f, points, np.random.default_rng(sequences[0]))
    reference_points = _surface_samples(reference_v, ref_f, points, np.random.default_rng(sequences[1]))
    mesh = trimesh.Trimesh(vertices=mesh_v, faces=f, process=False, validate=False)
    reference = trimesh.Trimesh(vertices=reference_v, faces=ref_f, process=False, validate=False)
    d_forward = _triangle_distances(reference, mesh_points)
    d_reverse = _triangle_distances(mesh, reference_points)
    forward_mse = float(np.square(d_forward).mean())
    reverse_mse = float(np.square(d_reverse).mean())
    symmetric_rms = float(np.sqrt(0.5 * (forward_mse + reverse_mse)))
    return {
        "rms_bbox_percent": 100.0 * symmetric_rms,
        "mesh_to_reference_rms_bbox_percent": 100.0 * float(np.sqrt(forward_mse)),
        "reference_to_mesh_rms_bbox_percent": 100.0 * float(np.sqrt(reverse_mse)),
        "rms_world_units": diagonal * symmetric_rms,
        "mesh_to_reference_sampled_max_bbox_percent": 100.0 * float(d_forward.max()),
        "reference_to_mesh_sampled_max_bbox_percent": 100.0 * float(d_reverse.max()),
        "sampled_symmetric_max_bbox_percent": 100.0 * float(max(d_forward.max(), d_reverse.max())),
        "sampled_max_is_true_hausdorff": False,
        "reference_bbox_diagonal": diagonal,
        "points_per_direction": points,
        "seed": int(seed),
        "sampler": "independent local SeedSequence streams; area-uniform triangle sampling",
        "distance": "trimesh accelerated nearest point on full triangle surface",
        "symmetric_rms_definition": "sqrt(0.5*(mean(mesh_to_ref_distance^2)+mean(ref_to_mesh_distance^2)))",
        "normalization": "100/reference surface bbox diagonal; no per-mesh rescaling",
    }


def audit_mesh(v, f, exact=True):
    """Inherited topology plus exact rational endpoint intersection audit.

    ``valid`` is null when exact=False: a topology-only check cannot certify
    absence of intersections. The experiment's valid scope requires one closed,
    oriented, genus-zero component, with no duplicate or degenerate faces.
    Exactness is for the stored float64 endpoint coordinates, not trajectories.
    """
    started = time.perf_counter()
    try:
        v, f = _arrays(v, f)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"valid": False, "closed": False, "oriented": False,
                "manifold": False, "exact_requested": bool(exact),
                "exact_executed": False, "error": str(exc),
                "audit_seconds": float(time.perf_counter() - started)}
    tp = _native(topology(v, f))
    evidence = _native(exact_audit(v, f)) if exact else None
    closed = tp["boundary_edges"] == 0 and tp["nonmanifold_edges"] == 0
    manifold = closed and tp["bad_vertex_links"] == 0
    oriented = closed and tp["orientation_errors"] == 0
    topology_pass = bool(tp["closed_oriented_g0"])
    pair_count = int(evidence["pair_count"]) if evidence is not None else None
    intersection_faces = (sorted({face for pair in evidence["improper_intersection_pairs"] for face in pair})
                          if evidence is not None else None)
    return {
        "valid": bool(topology_pass and evidence["embedded_endpoint_pass"]) if exact else None,
        "topology_pass": topology_pass,
        "closed": bool(closed), "oriented": bool(oriented), "manifold": bool(manifold),
        "components": int(tp["components"]), "euler": int(tp["euler"]),
        "degenerate_count": int(tp["degenerate_faces_tolerance"]),
        "exact_degenerate_count": len(evidence["degenerate_faces"]) if evidence is not None else None,
        "selfintersection_count": pair_count,
        "selfintersection_face_count": len(intersection_faces) if intersection_faces is not None else None,
        "selfintersection_face_fraction": len(intersection_faces) / len(f) if intersection_faces is not None else None,
        "exact_requested": bool(exact), "exact_executed": bool(exact),
        "valid_scope": "single closed oriented genus-zero surface; stored endpoints only",
        "topology": tp, "exact": evidence,
        "audit_seconds": float(time.perf_counter() - started),
    }
