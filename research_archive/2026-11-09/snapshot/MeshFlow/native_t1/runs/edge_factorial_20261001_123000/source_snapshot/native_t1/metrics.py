"""Pure Native T1 offline metrics, independent of the experiment code tree.

A fixed known mask extracts free faces once. Every target and swapped-context
score uses that same free set. Target geometry is used only for CPU evaluation.
"""
from __future__ import annotations
import json
import hashlib
import time
import numpy as np
from .metric_geometry import (
    _triangles, sample_mesh_surface, surface_distance_squared, raw_quality,
    boundary_metrics,
)

SURFACE_POINTS = 512
SURFACE_SEED = 9911
COMMON_POINTS = 512
COMMON_POINT_SEED = 9973
COMMON_REFERENCE_K = 12

def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def symmetric_surface_rms(predicted, target, bbox_diagonal, seed=SURFACE_SEED):
    """Original 512-point scoring stream, exposing both directional means.

    Fresh seed9911 per pair reproduces evaluate.completion_rms exactly. The
    prediction and target consume separate draws of the same local RNG stream.
    Cross-target calls use identical prediction-side samples for a fixed free
    array. Each target's own full bbox is the normalization denominator.
    """
    predicted = _triangles(predicted).astype(np.float64, copy=False)
    target = _triangles(target).astype(np.float64, copy=False)
    diagonal = float(bbox_diagonal)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Need positive finite target full bbox diagonal")
    result = dict(
        status="PASS", points_per_surface=SURFACE_POINTS, seed=seed,
        target_full_bbox_diagonal=diagonal, predicted_faces=len(predicted), target_faces=len(target),
        sampling="Area-uniform, independent RNG draws per direction; original local RandomState(seed) order",
        predicted_to_target_rms=None, target_to_predicted_rms=None, symmetric_rms=None,
        predicted_to_target_rms_bbox_pct=None, target_to_predicted_rms_bbox_pct=None,
        symmetric_rms_bbox_pct=None)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        result["status"] = "NONFINITE"
        return result
    rng = np.random.RandomState(seed)
    a = sample_mesh_surface(predicted, SURFACE_POINTS, rng)
    b = sample_mesh_surface(target, SURFACE_POINTS, rng)
    if a is None or b is None:
        result["status"] = "ZERO_OR_INVALID_SURFACE_AREA"
        return result
    ab, ba = surface_distance_squared(a, target), surface_distance_squared(b, predicted)
    mean_ab, mean_ba = float(ab.mean()), float(ba.mean())
    if not (np.isfinite(mean_ab) and np.isfinite(mean_ba)):
        result["status"] = "NONFINITE_DISTANCE"
        return result
    forward, backward = np.sqrt(mean_ab), np.sqrt(mean_ba)
    symmetric = np.sqrt((mean_ab + mean_ba) / 2)
    result.update(
        predicted_to_target_mean_squared=mean_ab, target_to_predicted_mean_squared=mean_ba,
        predicted_to_target_rms=float(forward), target_to_predicted_rms=float(backward),
        symmetric_rms=float(symmetric), predicted_to_target_rms_bbox_pct=float(100 * forward / diagonal),
        target_to_predicted_rms_bbox_pct=float(100 * backward / diagonal),
        symmetric_rms_bbox_pct=float(100 * symmetric / diagonal),
        prediction_sample_sha256=array_sha(a), target_sample_sha256=array_sha(b))
    return result


def _areas(triangles):
    triangles = _triangles(triangles).astype(np.float64, copy=False)
    return .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=-1)


def prepare_common_reference(common_remainders, bbox_diagonals, seed=COMMON_POINT_SEED):
    """Build one immutable point bank for each parent's fixed unknown R12.

    The same returned bank must be passed to every K/model/noise score. This
    helper reads geometry only, never generated outputs. Seed9973/+parent are
    CPU surface-sampling seeds, not new model-generation noise seeds.
    """
    if len(common_remainders) != 2 or len(bbox_diagonals) != 2:
        raise ValueError("Exactly two parent reference remainders are required")
    rows = []
    for parent, (target, diagonal) in enumerate(zip(common_remainders, bbox_diagonals)):
        reference = _triangles(target).copy()
        diagonal = float(diagonal)
        if not np.isfinite(reference).all() or not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("Finite reference and positive full-target bbox required")
        points = sample_mesh_surface(reference, COMMON_POINTS, np.random.RandomState(seed + parent))
        if points is None:
            raise ValueError("Common R12 must have positive surface area")
        rows.append(dict(target=reference, points=points.copy(), audit=dict(
            parent_index=parent, common_reference_K=COMMON_REFERENCE_K,
            reference_faces=len(reference), reference_sha256=array_sha(reference),
            point_count=COMMON_POINTS, point_seed=seed + parent, point_sha256=array_sha(points),
            target_full_bbox_diagonal=diagonal,
            reference_surface_area=float(_areas(reference).sum()),
            source="Fixed R12 target geometry only; never selected from outputs",
            usage="R12-to-predicted-free one-way coverage, identical target points across K/models/noises",
            not_model_generation_noise=True)))
    return dict(references=rows, audit=dict(
        schema="native_patch_common_reference_v1", common_K=COMMON_REFERENCE_K,
        point_count=COMMON_POINTS, point_seed_base=seed,
        references=[row["audit"] for row in rows],
        scoring_is_asymmetric=True,
        excludes_prediction_to_R12_penalty=True,
        rationale="K4/K8 legitimately generate faces later included in C12; a symmetric prediction-to-R12 penalty would change what is penalized across K",
        model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0))


def common_unknown_coverage(free_tokens, reference):
    """Fixed R12 target samples -> all extracted predicted free triangles."""
    free = _triangles(free_tokens)
    audit, points, target = reference["audit"], reference["points"], reference["target"]
    if array_sha(points) != audit["point_sha256"] or array_sha(target) != audit["reference_sha256"]:
        raise AssertionError("Fixed common unknown reference changed")
    distances = surface_distance_squared(points, free)
    if not np.isfinite(distances).all():
        raise ValueError("Common coverage returned nonfinite distances")
    diagonal = audit["target_full_bbox_diagonal"]
    rms = float(np.sqrt(distances.mean()))
    return dict(
        status="PASS", direction="Fixed target R12 -> predicted free surface only",
        target_index=audit["parent_index"], common_reference_K=COMMON_REFERENCE_K,
        target_point_count=len(points), target_point_seed=audit["point_seed"],
        target_point_sha256=audit["point_sha256"], target_reference_sha256=audit["reference_sha256"],
        target_full_bbox_diagonal=diagonal, predicted_free_sha256=array_sha(free),
        mean_squared=float(distances.mean()), rms=rms, rms_bbox_pct=100 * rms / diagonal,
        q95_bbox_pct=float(100 * np.quantile(np.sqrt(distances), .95) / diagonal),
        fraction_target_points_within_1pct_bbox=float((distances <= (.01 * diagonal) ** 2).mean()),
        prediction_to_common_R12_error="INTENTIONALLY_NOT_COMPUTED",
        extraction_or_assignment_performed=False,
        caveat="One-way coverage controls target-region support across K; it does not measure extra/incorrect predicted area or by itself establish causal conditioning")


def _distribution(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    finite=values[np.isfinite(values)]
    return dict(count=len(values),finite_count=len(finite),positive_inf_count=int(np.isposinf(values).sum()),
        negative_inf_count=int(np.isneginf(values).sum()),nan_count=int(np.isnan(values).sum()),
        mean=float(finite.mean()) if len(finite) else None,
        median=float(np.median(finite)) if len(finite) else None,
        minimum=float(finite.min()) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,
        p05=float(np.quantile(finite,.05)) if len(finite) else None,
        p95=float(np.quantile(finite,.95)) if len(finite) else None,
        finite_subset_statistics=True)


def shape_quality(tokens, bbox_diagonal):
    """Actual area/shape q and angles, with degeneracy kept explicit."""
    triangles = _triangles(tokens).astype(np.float64, copy=False)
    length = float(bbox_diagonal)
    if length <= 0 or not np.isfinite(length) or not np.isfinite(triangles).all():
        raise ValueError("Finite triangles and a positive target bbox diagonal are required")
    normalized = triangles / length
    edges = np.stack([normalized[:, (i + 1) % 3] - normalized[:, i] for i in range(3)], axis=1)
    area = .5 * np.linalg.norm(np.cross(normalized[:, 1] - normalized[:, 0], normalized[:, 2] - normalized[:, 0]), axis=-1)
    squared = (edges * edges).sum(-1).sum(-1)
    q = np.divide(squared, 4 * np.sqrt(3) * area, out=np.full_like(squared, np.inf), where=area > 0)
    q[(area == 0) & (squared == 0)] = np.nan
    angles = []
    for i in range(3):
        a = normalized[:, (i + 1) % 3] - normalized[:, i]
        b = normalized[:, (i + 2) % 3] - normalized[:, i]
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cosine = np.divide((a * b).sum(-1), denominator, out=np.ones_like(denominator), where=denominator > 0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    angles = np.stack(angles, axis=-1)
    return dict(faces=len(triangles), bbox_diagonal=length,
        q_definition="sum squared edge lengths/(4*sqrt(3)*triangle area)",
        q_actual=_distribution(q), q_actual_gt10_count=int((q > 10).sum()),
        q_actual_gt10_fraction=float((q > 10).mean()),
        area_normalized=_distribution(area), total_area=float(area.sum() * length * length),
        exact_zero_area_count=int((area == 0).sum()),
        near_degenerate_area_threshold=1e-12,
        near_degenerate_area_count=int((area <= 1e-12).sum()),
        min_angle_lt5_count=int((angles.min(-1) < 5).sum()),
        min_angle_lt5_fraction=float((angles.min(-1) < 5).mean()),
        angle_p05_degrees=float(np.quantile(angles, .05)))


def extract_prediction(tokens, context, known_mask=None):
    """Extract 100 free faces exactly once using the actual N112/K12 mask."""
    full = _triangles(tokens)
    context = _triangles(context)
    if full.shape != (112, 3, 3) or context.shape != (12, 3, 3):
        raise ValueError("This evaluator supports fixed N112/K12 only")
    if full.dtype != np.float32 or context.dtype != np.float32:
        raise ValueError("Evaluate original FP32 output and C")
    if not np.isfinite(full).all() or not np.isfinite(context).all():
        raise ValueError("Nonfinite geometry cannot be scored")
    if known_mask is None:
        known_mask = np.arange(112) < 12
    mask = np.asarray(known_mask)
    if mask.dtype != np.bool_ or mask.shape not in ((112,), (1, 112)) or mask.sum() != 12:
        raise ValueError("Require a bool mask selecting exactly12 of112 faces")
    mask = mask.reshape(112)
    if full[mask].tobytes() != context.tobytes():
        raise ValueError("Known slots must preserve original C bytes and corner order")
    free = full[~mask].copy()
    return dict(full=full, free=free, context=context, known_mask=mask.copy(), audit=dict(
        mode="known_mask", input_faces=112, free_faces=100,
        excluded_face_indices=np.flatnonzero(mask).tolist(), free_face_indices=np.flatnonzero(~mask).tolist(),
        full_sha256=array_sha(full), free_sha256=array_sha(free), actual_context_sha256=array_sha(context),
        known_mask_sha256=array_sha(mask), extraction_count=1, cross_target_rematching=False,
        source_target_used_for_extraction=False))


def evaluate_output(tokens, context, full_target, remainder_targets,
                    bbox_diagonals, parent_index, known_mask=None,
                    common_reference=None):
    """Score one raw Native output against both immutable target remainders.

    APIs intentionally accept target geometry only here, after generation.
    The original 512-point pairwise RMS stream and fixed R12 reference stream
    are preserved. This small package does not implement full intersection or
    vertex-link audits; retain the historical independent audit for those.
    """
    tick = time.perf_counter()
    parent_index = int(parent_index)
    if parent_index not in (0, 1) or len(remainder_targets) != 2 or len(bbox_diagonals) != 2:
        raise ValueError("Exactly two fixed target parents are required")
    prediction = extract_prediction(tokens, context, known_mask)
    full, free, desired = prediction["full"], prediction["free"], prediction["context"]
    target = _triangles(full_target)
    if target.shape != (112, 3, 3) or not np.isfinite(target).all():
        raise ValueError("Expected a finite full112-face target")
    remainders = [_triangles(value) for value in remainder_targets]
    if any(value.shape != (100, 3, 3) for value in remainders):
        raise ValueError("Each R12 must have100 faces")
    common = common_reference if common_reference is not None else prepare_common_reference(remainders, bbox_diagonals)
    cross, common_cross = [], []
    for target_index in (0, 1):
        score = symmetric_surface_rms(free, remainders[target_index], bbox_diagonals[target_index])
        score.update(target_index=target_index,
            predicted_free_sha256=prediction["audit"]["free_sha256"],
            target_remainder_sha256=array_sha(remainders[target_index]), rematching_performed=False)
        cross.append(score)
        reference = common["references"][target_index]
        if reference["audit"]["target_full_bbox_diagonal"] != float(bbox_diagonals[target_index]):
            raise ValueError("Fixed reference bbox does not match the target row")
        if reference["audit"]["reference_sha256"] != array_sha(remainders[target_index]):
            raise ValueError("Fixed reference geometry differs from R12")
        common_cross.append(common_unknown_coverage(free, reference))
    if cross[0]["prediction_sample_sha256"] != cross[1]["prediction_sample_sha256"]:
        raise AssertionError("Cross targets must reuse the same prediction-side sample bank")
    length = float(bbox_diagonals[parent_index])
    context_area, full_area = float(_areas(desired).sum()), float(_areas(target).sum())
    if full_area <= 0:
        raise ValueError("Full target must have positive area")
    result = dict(schema="native_t1_metrics_v1", variant="raw", parent_index=parent_index,
        K=12, total_faces=112, free_faces=100,
        known_fidelity=dict(context_bitwise_present=True,
            context_original_vertex_order_preserved=True,
            preservation_interpretation="Exact clamping by construction; not evidence of learned completion"),
        extraction=prediction["audit"],
        context_coverage=dict(known_faces=12, full_faces=112, face_fraction=12/112,
            context_surface_area=context_area, full_target_surface_area=full_area,
            surface_area_fraction=context_area/full_area),
        remainder=dict(own=cross[parent_index], cross=cross,
            fixed_free_sha256=prediction["audit"]["free_sha256"], extraction_count=1,
            rematching_for_cross_scores=False),
        common_unknown_R12=dict(own=common_cross[parent_index], cross=common_cross,
            direction="R12 to predicted free only"),
        whole=dict(surface=symmetric_surface_rms(full, target, length), quality=raw_quality(full)),
        free_quality=raw_quality(free), free_shape_quality=shape_quality(free, length),
        boundary=boundary_metrics(desired, free, length),
        independent_full_geometry_audit=dict(status="NOT_RUN",
            C_free_intersections="NOT_RUN", free_free_intersections="NOT_RUN",
            exact_vertex_link_topology="NOT_RUN", orientability="NOT_RUN",
            continuous_collision_safety="NOT_RUN",
            reason="This isolated core evaluator has no full intersection or vertex-link implementation; prior independent audit outputs remain preserved externally"),
        geometry_postprocessing="NONE", model_forwards=0, model_backwards=0,
        geometry_autograd_calls=0, optimizer_updates=0,
        scoring_seconds=time.perf_counter()-tick)
    if array_sha(full) != prediction["audit"]["full_sha256"] or array_sha(free) != prediction["audit"]["free_sha256"]:
        raise AssertionError("Read-only evaluation modified geometry")
    json.dumps(result, allow_nan=False)
    return result


def evaluate_case(tokens, cases, parent_index, known_mask=None):
    """Convenience adapter for data.PatchCase mappings keyed by (parent,K)."""
    parent_index = int(parent_index)
    case = cases[parent_index, 12]
    return evaluate_output(tokens, case.constraints, case.full_target,
        [cases[p, 12].free_target for p in (0, 1)],
        [cases[p, 12].bbox_diagonal for p in (0, 1)], parent_index, known_mask)


def binding_four_grid(context0_metrics, context1_metrics):
    """No extra sampling: use previously scored fixed free sets in four cells."""
    if context0_metrics["K"] != 12 or context1_metrics["K"] != 12:
        raise ValueError("Require K12 results for both contexts")
    if context0_metrics["parent_index"] != 0 or context1_metrics["parent_index"] != 1:
        raise ValueError("Expected context0 then context1 result")
    a, b = context0_metrics["remainder"]["cross"], context1_metrics["remainder"]["cross"]
    e00, e10 = a[0]["symmetric_rms_bbox_pct"], a[1]["symmetric_rms_bbox_pct"]
    e01, e11 = b[0]["symmetric_rms_bbox_pct"], b[1]["symmetric_rms_bbox_pct"]
    delta0, delta1 = e01-e00, e10-e11
    common_a, common_b = context0_metrics["common_unknown_R12"]["cross"], context1_metrics["common_unknown_R12"]["cross"]
    c00, c10 = common_a[0]["rms_bbox_pct"], common_a[1]["rms_bbox_pct"]
    c01, c11 = common_b[0]["rms_bbox_pct"], common_b[1]["rms_bbox_pct"]
    return dict(E00=e00, E01=e01, E11=e11, E10=e10,
        E_R0_F_C0=e00, E_R0_F_C1=e01, E_R1_F_C1=e11, E_R1_F_C0=e10,
        Delta0=delta0, Delta1=delta1, B=(delta0+delta1)/2,
        direction0_bbox_percentage_points=delta0, direction1_bbox_percentage_points=delta1,
        mean_binding_bbox_percentage_points=(delta0+delta1)/2,
        direction0_positive=delta0>0, direction1_positive=delta1>0,
        both_directions_positive=delta0>0 and delta1>0,
        normalization="Each target row uses that parent's fixed full bbox; positive means own context improves its target score",
        common_unknown_R12_binding=dict(E_R0_F_C0=c00, E_R0_F_C1=c01, E_R1_F_C1=c11, E_R1_F_C0=c10,
            direction0_bbox_percentage_points=c01-c00, direction1_bbox_percentage_points=c10-c11,
            mean_binding_bbox_percentage_points=((c01-c00)+(c10-c11))/2,
            both_directions_positive=(c01-c00)>0 and (c10-c11)>0),
        extraction_or_assignment_performed=False, geometry_evaluations=0, model_forwards=0)


cross_binding = binding_four_grid
