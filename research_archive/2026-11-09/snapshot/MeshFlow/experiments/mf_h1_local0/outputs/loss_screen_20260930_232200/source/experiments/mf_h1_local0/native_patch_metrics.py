"""Offline metrics for nested connected K=4/8/12 native-context diagnostics.

The K-specific target remainder changes with K. Therefore its RMS trend is not
by itself a causal estimate of context benefit. A fixed R12-to-predicted-free
coverage diagnostic uses the identical target point bank for every K.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .evaluate import raw_quality, sample_mesh_surface, surface_distance_squared
from .native_inpainting_metrics import (
    array_sha, known_fidelity, near_context_diagnostic, symmetric_surface_rms,
    binding_four_grid,
)


TRAINED_K = (4, 8, 12)
EVALUATED_K = (2, 4, 8, 12)
REGRESSION_SEEDS = (9201, 9202, 9203, 9204)
COMMON_REFERENCE_K = 12
COMMON_POINTS = 512
COMMON_POINT_SEED = 9973


def _triangles(value):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError(f"Expected [N,9] or [N,3,3], got {value.shape}")
    return value


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


def extract_patch_prediction(tokens, context, known_mask):
    """Extract all valid free slots once, without any geometry-based selection.

    Supports the declared K4/8/12 family and optional nested-K2 evaluation only.
    The latter must be labeled separately by the caller; it is not historical
    disjoint K2 and is outside the mixed-K4/8/12 training distribution.
    """
    tick = time.perf_counter()
    full, desired = _triangles(tokens).copy(), _triangles(context).copy()
    known = np.asarray(known_mask)
    if known.dtype != np.bool_ or known.shape not in ((len(full),), (1, len(full))):
        raise ValueError("known_mask must be bool[N] or bool[1,N]")
    known = known.reshape(-1)
    K = len(desired)
    if K not in (2, 4, 8, 12) or int(known.sum()) != K or len(full) <= K:
        raise ValueError("Require declared K, exactly K known slots, and nonempty free geometry")
    if not (np.isfinite(full).all() and np.isfinite(desired).all()):
        raise ValueError("Cannot score nonfinite output or context")
    free = full[~known].copy()
    return dict(full=full, free=free, context=desired, known_mask=known.copy(), audit=dict(
        schema="native_patch_extraction_v1", mode="known_mask", K=K,
        input_faces=len(full), free_faces=len(free),
        excluded_face_indices=np.flatnonzero(known).tolist(), free_face_indices=np.flatnonzero(~known).tolist(),
        actual_context_sha256=array_sha(desired), full_sha256=array_sha(full), free_sha256=array_sha(free),
        known_mask_sha256=array_sha(known), source_target_used_for_extraction=False,
        extraction_count=1, cross_target_rematching=False,
        seconds=time.perf_counter() - tick))


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


def score_patch_extracted(prediction, context, full_target, remainder_targets,
                          bbox_diagonals, parent_index, common_reference,
                          seam_result=None, variant="raw"):
    """Score one free set against same-K R0/R1 and fixed common R12 banks."""
    tick = time.perf_counter()
    parent_index = int(parent_index)
    if parent_index not in (0, 1) or len(remainder_targets) != 2 or len(bbox_diagonals) != 2:
        raise ValueError("Exactly two same-K target parents required")
    if variant != "raw":
        raise ValueError("This authorized run is raw-only; projected geometry is not evaluated")
    full, free, desired = prediction["full"], prediction["free"], _triangles(context)
    if array_sha(desired) != prediction["audit"]["actual_context_sha256"]:
        raise AssertionError("Scoring context differs from actual generated context")
    original_free_sha = prediction["audit"]["free_sha256"]
    if array_sha(free) != original_free_sha:
        raise AssertionError("Extracted free set changed before scoring")
    K = len(desired)
    cross, common_cross = [], []
    for target_index in (0, 1):
        remainder = _triangles(remainder_targets[target_index])
        if len(remainder) != len(free):
            raise ValueError("Same-K target and predicted remainders must have N-K faces")
        value = symmetric_surface_rms(free, remainder, bbox_diagonals[target_index])
        value.update(target_index=target_index, predicted_free_sha256=original_free_sha,
                     target_remainder_sha256=array_sha(remainder), rematching_performed=False)
        cross.append(value)
        reference = common_reference["references"][target_index]
        if reference["audit"]["target_full_bbox_diagonal"] != float(bbox_diagonals[target_index]):
            raise AssertionError("Common coverage and same-K score use different bbox normalization")
        common_cross.append(common_unknown_coverage(free, reference))
    if cross[0].get("prediction_sample_sha256") != cross[1].get("prediction_sample_sha256"):
        raise AssertionError("Same free geometry must have the same prediction-side sampling bank")
    diagonal = float(bbox_diagonals[parent_index])
    target = _triangles(full_target)
    context_area, total_area = float(_areas(desired).sum()), float(_areas(target).sum())
    if total_area <= 0:
        raise ValueError("Full target must have positive surface area")
    result = dict(
        schema="native_patch_metrics_v1", variant=variant, parent_index=parent_index,
        K=K, total_faces=len(full), free_faces=len(free),
        K_scope="mixed_training_K4_K8_K12" if K in TRAINED_K else "nested_K2_untrained_reference_not_historical_disjoint_K2",
        known_fidelity=known_fidelity(full, desired, diagonal), extraction=prediction["audit"],
        context_coverage=dict(known_faces=K, full_faces=len(target), face_fraction=K / len(target),
            context_surface_area=context_area, full_target_surface_area=total_area,
            surface_area_fraction=context_area / total_area,
            rule="Actual triangle areas in the shared coordinates; not K/112 as a proxy for covered surface area"),
        remainder=dict(own=cross[parent_index], cross=cross, fixed_free_sha256=original_free_sha,
            extraction_count=1, rematching_for_cross_scores=False,
            across_K_warning="R_K changes when K changes. Lower R_K RMS can reflect removal of difficult target regions; do not interpret alone as causal context improvement"),
        common_unknown_R12=dict(own=common_cross[parent_index], cross=common_cross,
            normalization="Each target parent's fixed full bbox; common R12 and its target samples held identical for all K",
            direction="R12 to predicted free only"),
        whole=dict(surface=symmetric_surface_rms(full, target, diagonal), quality=raw_quality(full)),
        free_quality=raw_quality(free), near_context=near_context_diagnostic(free, desired, diagonal),
        seam=dict(status="NOT_RUN", reason="Independent seam diagnostic has not been attached") if seam_result is None else seam_result,
        scoring_seconds=time.perf_counter() - tick,
        geometry_postprocessing="NONE",
        model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0)
    if array_sha(free) != original_free_sha or array_sha(full) != prediction["audit"]["full_sha256"]:
        raise AssertionError("CPU scoring mutated generated geometry")
    json.dumps(result, allow_nan=False)
    return result


def evaluate_patch_output(tokens, context, full_target, remainder_targets,
                          common_reference, bbox_diagonals, parent_index,
                          known_mask, seam_result=None, variant="raw", extraction=None):
    """Return (JSON metrics, reusable extracted array package).

    No projection is implemented or accepted in this raw-only experiment.
    """
    prediction = extraction or extract_patch_prediction(tokens, context, known_mask)
    if array_sha(_triangles(tokens)) != prediction["audit"]["full_sha256"]:
        raise AssertionError("Provided extraction belongs to a different output")
    score = score_patch_extracted(prediction, context, full_target, remainder_targets,
                                  bbox_diagonals, parent_index, common_reference, seam_result, variant)
    return score, prediction


def patch_binding_four_grid(context0_metrics, context1_metrics):
    """Same-K binding from fixed free sets, plus common-region binding."""
    if context0_metrics["K"] != context1_metrics["K"]:
        raise ValueError("Binding must compare the same K")
    if context0_metrics["variant"] != context1_metrics["variant"]:
        raise ValueError("Raw and projected variants cannot be pooled")
    result = binding_four_grid(context0_metrics, context1_metrics)
    a, b = context0_metrics["common_unknown_R12"]["cross"], context1_metrics["common_unknown_R12"]["cross"]
    e00, e10, e01, e11 = a[0]["rms_bbox_pct"], a[1]["rms_bbox_pct"], b[0]["rms_bbox_pct"], b[1]["rms_bbox_pct"]
    d0, d1 = e01 - e00, e10 - e11
    result["common_unknown_R12_binding"] = dict(
        E_R0_F_C0=e00, E_R0_F_C1=e01, E_R1_F_C1=e11, E_R1_F_C0=e10,
        direction0_bbox_percentage_points=d0, direction1_bbox_percentage_points=d1,
        mean_binding_bbox_percentage_points=(d0 + d1) / 2,
        both_directions_positive=d0 > 0 and d1 > 0,
        direction="Fixed common target R12 -> free coverage, target samples identical across K")
    result["variant"] = context0_metrics["variant"]
    return result


def plot_patch_fixed(outputs, contexts, savepath):
    """Fixed parent0/seed9201: full and free rows for each K, shared bounds.

    outputs={2:{'Native0':package,'T1':package,'T2':package},4:{...},8:{...},12:{...}}.
    All rows use one camera/range. The C column is blank in free-only rows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if tuple(sorted(outputs)) != EVALUATED_K or tuple(sorted(contexts)) != EVALUATED_K:
        raise ValueError("Fixed view requires K2 reference plus every trained K4/8/12")
    groups = ("Native0", "T1", "T2")
    vertices = []
    for K in EVALUATED_K:
        if set(outputs[K]) != set(groups):
            raise ValueError("Each K must include Native0, T1 and T2")
        vertices.append(_triangles(contexts[K]).reshape(-1, 3))
        vertices.extend(outputs[K][group]["full"].reshape(-1, 3) for group in groups)
    all_vertices = np.concatenate(vertices)
    all_vertices = all_vertices[np.isfinite(all_vertices).all(axis=1)]
    low, high = all_vertices.min(axis=0), all_vertices.max(axis=0)
    center, half = (low + high) / 2, max(float((high - low).max()) * .55, 1e-3)
    limits = np.stack((center - half, center + half), axis=1)
    figure = plt.figure(figsize=(15, 26), constrained_layout=True)
    for ki, K in enumerate(EVALUATED_K):
        desired = _triangles(contexts[K])
        for row, part in enumerate(("full", "free")):
            for column, group in enumerate(("C",) + groups):
                axis = figure.add_subplot(8, 4, (2 * ki + row) * 4 + column + 1, projection="3d")
                triangles = (desired if row == 0 else np.empty((0, 3, 3))) if group == "C" else outputs[K][group][part]
                finite = triangles[np.isfinite(triangles).all(axis=(1, 2))]
                if len(finite):
                    closed = np.concatenate((finite, finite[:, :1]), axis=1)
                    axis.add_collection3d(Line3DCollection(closed, colors="#58798f", linewidths=.55, alpha=.75))
                if row == 0:
                    axis.add_collection3d(Poly3DCollection(desired, facecolors="#ed2727", edgecolors="#b00000", linewidths=.9, alpha=.3))
                axis.set_title(f"K={K} | {group} | {part}" + (" [NONFINITE]" if len(finite) != len(triangles) else ""), fontsize=10)
                axis.set_xlim(limits[0]); axis.set_ylim(limits[1]); axis.set_zlim(limits[2])
                axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=22, azim=35)
                axis.tick_params(labelsize=7)
    figure.suptitle("Fixed parent0 / reused regression noise9201 | nested K2 reference (not trained), connected K4/8/12\n"
                   "Red: input C. Full/free pairs share one camera and complete coordinate range; no C overlay on free rows.", fontsize=14)
    path = Path(savepath); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150); plt.close(figure)
    return dict(path=str(path), fixed_parent=0, fixed_regression_seed=9201, K=list(EVALUATED_K),
                groups=list(groups), shared_limits=limits.tolist(), camera=dict(elevation=22, azimuth=35),
                context_overlay_in_free_rows=False, postprocessing="NONE")


def run_cpu_tests():
    """Small mathematical fixtures; no official model, GPU, or gradient calls."""
    base = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    target0 = np.stack([base + np.asarray([2. * (i % 4), 2. * (i // 4), .1 * (i % 2)], dtype=np.float32) for i in range(16)])
    target1 = target0 + np.asarray([.2, -.1, .4], dtype=np.float32)
    diagonal = [float(np.linalg.norm(np.ptp(target.reshape(-1, 3), axis=0))) for target in (target0, target1)]
    common = prepare_common_reference([target0[12:], target1[12:]], diagonal)
    common_hash_before = json.dumps(common["audit"], sort_keys=True)
    scores = []
    for K in TRAINED_K:
        known = np.arange(len(target0)) < K
        metric, package = evaluate_patch_output(target0, target0[:K], target0,
            [target0[K:], target1[K:]], common, diagonal, 0, known)
        assert len(package["free"]) == 16 - K
        assert metric["known_fidelity"]["context_bitwise_present"]
        assert metric["known_fidelity"]["thresholds"][-1]["all_targets_all_vertices_pass"]
        assert metric["remainder"]["own"]["symmetric_rms_bbox_pct"] < 1e-12
        assert metric["common_unknown_R12"]["own"]["rms_bbox_pct"] < 1e-12
        assert metric["remainder"]["cross"][0]["predicted_free_sha256"] == metric["remainder"]["cross"][1]["predicted_free_sha256"]
        assert metric["common_unknown_R12"]["own"]["prediction_to_common_R12_error"] == "INTENTIONALLY_NOT_COMPUTED"
        json.dumps(metric, allow_nan=False)
        scores.append(metric)
    hashes = [metric["common_unknown_R12"]["own"]["target_point_sha256"] for metric in scores]
    assert len(set(hashes)) == 1
    assert json.dumps(common["audit"], sort_keys=True) == common_hash_before
    irregular = target0.copy(); irregular[0, 1:] = irregular[0, 0] + .1 * (irregular[0, 1:] - irregular[0, 0])
    metric, _ = evaluate_patch_output(irregular, irregular[:4], irregular,
        [irregular[4:], target1[4:]], common, diagonal, 0, np.arange(16) < 4)
    assert abs(metric["context_coverage"]["surface_area_fraction"] - 4 / 16) > .01
    k2, _ = evaluate_patch_output(target0, target0[:2], target0, [target0[2:], target1[2:]],
                                  common, diagonal, 0, np.arange(16) < 2)
    assert k2["K_scope"] == "nested_K2_untrained_reference_not_historical_disjoint_K2"
    try:
        patch_binding_four_grid(scores[0], scores[1])
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-K binding accepted")
    return dict(status="PASS", tests=[
        "K4_K8_K12_known_mask_counts", "generic_Hungarian_fidelity_up_to_K12",
        "fixed_free_set_cross_scoring", "fixed_common_R12_points_across_all_K",
        "zero_exact_surface_and_common_coverage", "no_prediction_to_R12_penalty",
        "actual_surface_area_not_face_count_fraction", "nested_K2_explicit_untrained_scope",
        "cross_K_binding_rejected", "reference_arrays_unchanged", "JSON_safe"],
        model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    print(json.dumps(run_cpu_tests(), sort_keys=True))
