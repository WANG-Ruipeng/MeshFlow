"""CPU-only, aligned context/remainder metrics for native inpainting.

Remainders are extracted once using the actual input context, then held fixed
for both target scores. No target mesh participates in choosing excluded faces.
No model, optimizer, geometry repair, or topology certification is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluate import raw_quality, sample_mesh_surface, surface_distance_squared
from .soft_matching_metrics import (
    PERMUTATIONS, RELATIVE_THRESHOLDS, exact_face_distances,
    _matched_rows, _matching_summary,
)


GROUPS = ("B0", "B1", "B2", "B3", "B4", "B5", "T1", "T2")
NEAR_CONTEXT_FRACTION = 1e-3
SURFACE_SEED = 9911
SURFACE_POINTS = 512


def _triangles(value):
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[-1] == 9:
        array = array.reshape(-1, 3, 3)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"Expected [N,9] or [N,3,3], got {array.shape}")
    return array


def array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _assignment(tokens, context):
    distance, permutation, _ = exact_face_distances(tokens, context)
    face_ids, target_ids = linear_sum_assignment(distance)
    chosen = np.empty(len(context), dtype=np.int64)
    chosen[target_ids] = face_ids
    if len(set(chosen.tolist())) != len(context):
        raise AssertionError("Context assignment must use distinct generated faces")
    return chosen, distance, permutation


def extract_prediction(tokens, context, mode="assigned", known_mask=None, fixed_excluded=None):
    """Return an internal array package: full, free, context, and JSON audit.

    ``assigned``: Base/Soft; choose once from actual C using six permutations.
    ``known_mask``: native/clamp; exclude exactly the known slots.
    ``fixed_excluded``: copy baseline; reuse original Base's chosen slots.
    Arrays are copied; neither source nor C is modified. Subsequent scores must
    call score_extracted(), which never performs another remainder extraction.
    """
    tick = time.perf_counter()
    full, desired = _triangles(tokens).copy(), _triangles(context).copy()
    if not np.isfinite(full).all() or not np.isfinite(desired).all():
        raise ValueError("Cannot assign/extract nonfinite output or context")
    if len(desired) not in (1, 2) or len(full) <= len(desired):
        raise ValueError("Require K in {1,2} and nonempty free remainder")
    if mode == "assigned":
        if known_mask is not None or fixed_excluded is not None:
            raise ValueError("Assigned extraction must not receive alternate exclusions")
        excluded, _, _ = _assignment(full, desired)
        rule = "One six-permutation Hungarian assignment using actual input C only"
    elif mode == "known_mask":
        if known_mask is None or fixed_excluded is not None:
            raise ValueError("Native extraction requires known_mask only")
        known = np.asarray(known_mask)
        if known.dtype != np.bool_ or known.shape not in ((len(full),), (1, len(full))):
            raise ValueError("known_mask must be bool[N] or bool[1,N]")
        excluded = np.flatnonzero(known.reshape(-1))
        if len(excluded) != len(desired):
            raise ValueError("known_mask count differs from K")
        rule = "Exactly known_mask slots; no distance-based selection"
    elif mode == "fixed_excluded":
        if known_mask is not None or fixed_excluded is None:
            raise ValueError("Copy extraction requires its pre-copy Base exclusions")
        excluded = np.asarray(fixed_excluded, dtype=np.int64)
        rule = "Reuse pre-copy Base assignment; no matching after replacement"
    else:
        raise ValueError(f"Unknown extraction mode {mode}")
    if excluded.shape != (len(desired),) or len(set(excluded.tolist())) != len(desired):
        raise ValueError("Exactly K distinct excluded indices required")
    if np.any(excluded < 0) or np.any(excluded >= len(full)):
        raise ValueError("Excluded index out of bounds")
    keep = np.ones(len(full), dtype=bool)
    keep[excluded] = False
    free_indices = np.flatnonzero(keep)
    free = full[keep].copy()
    return dict(full=full, free=free, context=desired, audit=dict(
        mode=mode, rule=rule, input_faces=len(full), K=len(desired), free_faces=len(free),
        excluded_face_indices=excluded.tolist(), free_face_indices=free_indices.tolist(),
        actual_context_sha256=array_sha(desired), full_sha256=array_sha(full), free_sha256=array_sha(free),
        source_target_used_for_extraction=False, extraction_count=1,
        cross_target_rematching=False, seconds=time.perf_counter() - tick))


def best_match_copy(base_tokens, context):
    """Explicit copy baseline; return (copied tokens, extracted package, audit).

    This is declared postprocessing, never described as learned preservation.
    Every nonselected face must remain byte-identical to the original Base.
    """
    tick = time.perf_counter()
    original = np.asarray(base_tokens)
    baseline = extract_prediction(base_tokens, context, mode="assigned")
    desired = _triangles(context)
    if baseline["full"].dtype != desired.dtype:
        raise ValueError("Copy cannot silently cast context coordinates")
    copied = baseline["full"].copy()
    selected = baseline["audit"]["excluded_face_indices"]
    copied[selected] = desired
    prediction = extract_prediction(copied, context, mode="fixed_excluded", fixed_excluded=selected)
    same = array_sha(prediction["free"]) == array_sha(baseline["free"])
    if not same:
        raise AssertionError("Copy changed Base remainder")
    audit = dict(
        postprocessing="Replace K assigned existing Base faces with exact C; output face count unchanged",
        baseline_extraction=baseline["audit"], selected_face_indices=selected,
        free_bitwise_equal_to_base_extraction=same,
        copy_context_bitwise_preserved=copied[selected].tobytes() == desired.tobytes(),
        seconds=time.perf_counter() - tick, model_forwards=0, geometry_autograd_calls=0)
    return copied.reshape(original.shape), prediction, audit


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


def known_fidelity(tokens, context, bbox_diagonal):
    full, desired = _triangles(tokens), _triangles(context)
    chosen, distances, permutations = _assignment(full, desired)
    rows = _matched_rows(full, desired, distances, permutations, chosen, float(bbox_diagonal))
    result = _matching_summary(rows, float(bbox_diagonal))
    exact_ids = []
    for face in desired:
        candidates = [i for i, output in enumerate(full)
                      if output.dtype == face.dtype and output.tobytes() == face.tobytes()]
        exact_ids.append(candidates)
    result.update(
        selected_face_indices=chosen.tolist(), distinct_faces=True,
        context_bitwise_present=all(bool(ids) for ids in exact_ids),
        context_original_vertex_order_exact_face_ids=exact_ids,
        preservation_interpretation="Copy/clamp exactness follows the construction rule; it is not evidence of learned completion")
    return result


def near_context_diagnostic(free_tokens, context, bbox_diagonal):
    """Declared geometric overlap diagnostics, never a topology validity test."""
    free, desired = _triangles(free_tokens), _triangles(context)
    threshold = NEAR_CONTEXT_FRACTION * float(bbox_diagonal)
    distances, permutation_indices, _ = exact_face_distances(free, desired)
    nearest_context = np.argmin(distances, axis=1)
    face_ids = np.arange(len(free))
    selected_permutation = PERMUTATIONS[permutation_indices[face_ids, nearest_context]]
    aligned = desired[nearest_context[:, None], selected_permutation]
    exact_vertex_distances = np.linalg.norm(free.astype(np.float64) - aligned, axis=-1)
    all_vertex_matched_near = np.max(exact_vertex_distances, axis=1) <= threshold
    vertex_surface_d2 = surface_distance_squared(free.reshape(-1, 3), desired).reshape(len(free), 3)
    vertices_near_surface = vertex_surface_d2 <= threshold * threshold
    samples = sample_mesh_surface(free, SURFACE_POINTS, np.random.RandomState(9951))
    sampled_point_fraction = None
    if samples is not None:
        sampled_point_fraction = float((surface_distance_squared(samples, desired) <= threshold * threshold).mean())
    return dict(
        relative_bbox_threshold=NEAR_CONTEXT_FRACTION, coordinate_threshold=threshold,
        exact_permutation_duplicate_face_count=int((exact_vertex_distances.max(axis=1) == 0).sum()),
        near_complete_context_face_fraction=float(all_vertex_matched_near.mean()),
        free_vertices_near_context_surface_fraction=float(vertices_near_surface.mean()),
        free_faces_all_vertices_near_context_surface_fraction=float(vertices_near_surface.all(axis=1).mean()),
        free_surface_sample_points_near_context_fraction=sampled_point_fraction,
        sampled_points=SURFACE_POINTS, sample_seed=9951,
        interpretation="Geometric proximity/duplicate coverage only. Nearby free geometry is not automatically an error; no self-intersection, watertightness or stitching certification.",
        self_intersection_status="NOT_RUN", watertight_status="NOT_RUN", manifold_status="NOT_RUN")


def score_extracted(prediction, context, full_target, remainder_targets, bbox_diagonals,
                    target_index, expected_copy_free=None):
    """Score one immutable extracted free set against BOTH remainder targets."""
    tick = time.perf_counter()
    full, free, desired = prediction["full"], prediction["free"], _triangles(context)
    target_index = int(target_index)
    if target_index not in (0, 1) or len(remainder_targets) != 2 or len(bbox_diagonals) != 2:
        raise ValueError("This comparison has exactly two target parents")
    if array_sha(desired) != prediction["audit"]["actual_context_sha256"]:
        raise AssertionError("Scoring context differs from actual extraction context")
    before = array_sha(free)
    if before != prediction["audit"]["free_sha256"]:
        raise AssertionError("Extracted free geometry changed before scoring")
    if expected_copy_free is not None and array_sha(_triangles(expected_copy_free)) != before:
        raise AssertionError("Copy free set is not byte-identical to Base extraction")
    cross = []
    for index in range(2):
        remainder = _triangles(remainder_targets[index])
        if len(remainder) != len(free):
            raise ValueError("Every target remainder must have N-K faces")
        score = symmetric_surface_rms(free, remainder, bbox_diagonals[index])
        score.update(target_index=index, predicted_free_sha256=before,
                     target_remainder_sha256=array_sha(remainder), rematching_performed=False)
        cross.append(score)
    if cross[0].get("prediction_sample_sha256") != cross[1].get("prediction_sample_sha256"):
        raise AssertionError("Cross-scoring changed prediction-side surface sampling")
    own_bbox = float(bbox_diagonals[target_index])
    result = dict(
        schema="native_inpainting_metrics_v1", target_index=target_index, K=len(desired),
        total_faces=len(full), free_faces=len(free),
        known_fidelity=known_fidelity(full, desired, own_bbox),
        extraction=prediction["audit"],
        remainder=dict(own=cross[target_index], cross=cross,
                       fixed_free_sha256=before, extraction_count=1, rematching_for_cross_scores=False),
        whole=dict(surface=symmetric_surface_rms(full, full_target, own_bbox), quality=raw_quality(full)),
        free_quality=raw_quality(free),
        near_context=near_context_diagnostic(free, desired, own_bbox),
        copy_free_bitwise_sanity=None if expected_copy_free is None else True,
        scoring_seconds=time.perf_counter() - tick,
        model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0)
    if array_sha(free) != before or array_sha(full) != prediction["audit"]["full_sha256"]:
        raise AssertionError("Offline scoring mutated output geometry")
    json.dumps(result, allow_nan=False)
    return result


def evaluate_all_output(tokens, context, full_target, remainder_targets, bbox_diagonals,
                        target_index, mode="assigned", known_mask=None, fixed_excluded=None,
                        expected_copy_free=None, extraction=None):
    """Convenience API returning (JSON metrics, reusable array package).

    For B3 pass the package returned by best_match_copy as ``extraction`` and
    original Base.free as ``expected_copy_free``. No second assignment occurs.
    """
    prediction = extraction
    if prediction is None:
        prediction = extract_prediction(tokens, context, mode, known_mask, fixed_excluded)
    elif array_sha(_triangles(tokens)) != prediction["audit"]["full_sha256"]:
        raise AssertionError("Provided extraction is from a different output")
    return score_extracted(prediction, context, full_target, remainder_targets, bbox_diagonals,
                           target_index, expected_copy_free), prediction


def binding_four_grid(context0_metrics, context1_metrics):
    """Each fixed free set has already been scored; no geometry selection here."""
    a, b = context0_metrics["remainder"]["cross"], context1_metrics["remainder"]["cross"]
    e00, e10 = a[0]["symmetric_rms_bbox_pct"], a[1]["symmetric_rms_bbox_pct"]
    e01, e11 = b[0]["symmetric_rms_bbox_pct"], b[1]["symmetric_rms_bbox_pct"]
    direction0 = None if e00 is None or e01 is None else e01 - e00
    direction1 = None if e11 is None or e10 is None else e10 - e11
    return dict(
        E_R0_F_C0=e00, E_R0_F_C1=e01, E_R1_F_C1=e11, E_R1_F_C0=e10,
        direction0_bbox_percentage_points=direction0, direction1_bbox_percentage_points=direction1,
        mean_binding_bbox_percentage_points=None if direction0 is None or direction1 is None else (direction0 + direction1) / 2,
        both_directions_positive=None if direction0 is None or direction1 is None else direction0 > 0 and direction1 > 0,
        normalization="Each row uses that target parent's full bbox; positive means its own context improves that target remainder score",
        caveat="Base and Copy extraction itself depends on C; their cross-score change is not learned network conditioning")


def plot_fixed(outputs, contexts, savepath):
    """Fixed P0/K2/seed9101: C + all eight groups, full and free-only rows.

    ``outputs`` maps B0..T2 to extract_prediction packages. ``contexts`` is the
    fixed [2,3,3] context array (or {"C": array}). It is not selected by quality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    desired = _triangles(contexts["C"] if isinstance(contexts, dict) else contexts)
    if len(desired) != 2 or set(outputs) != set(GROUPS):
        raise ValueError("Fixed comparison requires K=2 and all eight groups")
    vertices = np.concatenate([desired.reshape(-1, 3)] + [
        outputs[group][part].reshape(-1, 3) for group in GROUPS for part in ("full", "free")])
    finite_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if not len(finite_vertices):
        raise ValueError("No finite geometry to plot")
    low, high = finite_vertices.min(axis=0), finite_vertices.max(axis=0)
    center, half = (low + high) / 2, max(float((high - low).max()) * .55, 1e-3)
    limits = np.stack((center - half, center + half), axis=1)
    figure = plt.figure(figsize=(28, 7.4), constrained_layout=True)
    labels = dict(B0="Base", B1="Soft5", B2="Soft20", B3="Copy", B4="Global clamp",
                  B5="Native step0", T1="Native correct", T2="Native independent")
    for row, part in enumerate(("full", "free")):
        for column, group in enumerate(("C",) + GROUPS):
            axis = figure.add_subplot(2, 9, row * 9 + column + 1, projection="3d")
            if group == "C":
                triangle = desired if row == 0 else np.empty((0, 3, 3))
                color = "#dc2727"
                title = "Input C (K=2)" if row == 0 else "C excluded from free score"
            else:
                triangle = outputs[group][part]
                color = "#526c86" if row == 0 else "#006c94"
                title = labels[group] + (" | full" if row == 0 else " | free only")
            valid_tri = triangle[np.isfinite(triangle).all(axis=(1, 2))]
            if len(valid_tri):
                closed = np.concatenate((valid_tri, valid_tri[:, :1]), axis=1)
                axis.add_collection3d(Line3DCollection(closed, colors=color, linewidths=.55, alpha=.75))
            if row == 0:
                axis.add_collection3d(Poly3DCollection(desired, facecolors="#ed2222",
                                                      edgecolors="#b00000", linewidths=1.1, alpha=.28))
            if len(valid_tri) != len(triangle):
                title += " [NONFINITE]"
            axis.set_title(title, fontsize=9)
            axis.set_xlim(limits[0]); axis.set_ylim(limits[1]); axis.set_zlim(limits[2])
            axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=22, azim=35)
            axis.tick_params(labelsize=6)
    figure.suptitle("Fixed parent0 / K2 / seed9101 | shared camera and full coordinate bounds\n"
                   "Top: full 112 faces and red input C. Bottom: actual fixed extracted free sets; C is not overlaid.", fontsize=14)
    path = Path(savepath); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150); plt.close(figure)
    return dict(path=str(path), fixed_parent_index=0, K=2, seed=9101,
                groups=list(GROUPS), rows=["full", "free"], shared_limits=limits.tolist(),
                camera=dict(elevation=22, azimuth=35), selection="Predeclared, no quality selection",
                geometry_postprocessing="NONE", free_row_context_overlay=False)


def run_cpu_tests():
    """Bounded fixtures that check extraction invariance and aligned scoring."""
    from .evaluate import completion_rms
    a = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    b, c, d = a + [0., 0., 2.], a + [3., 0., 0.], a + [0., 3., 0.]
    full = np.stack((a[[2, 0, 1]], b, c, d)).astype(np.float32)
    context = a[None].copy()
    extraction = extract_prediction(full, context)
    assert extraction["audit"]["excluded_face_indices"] == [0]
    assert np.array_equal(extraction["free"], full[1:])
    copied, package, copy_audit = best_match_copy(full, context)
    assert copied[0].tobytes() == context[0].tobytes()
    assert copy_audit["free_bitwise_equal_to_base_extraction"]
    assert package["free"].tobytes() == extraction["free"].tobytes()
    diagonals = [float(np.linalg.norm(np.ptp(full.reshape(-1, 3), axis=0))), 10.]
    remainder0, remainder1 = full[1:].copy(), (full[1:] + [1., -.3, .2]).astype(np.float32)
    score, same_package = evaluate_all_output(copied, context, full, [remainder0, remainder1], diagonals, 0,
                                             extraction=package, expected_copy_free=extraction["free"])
    assert score["known_fidelity"]["context_bitwise_present"]
    assert score["known_fidelity"]["thresholds"][-1]["all_targets_all_vertices_pass"]
    assert score["remainder"]["own"]["symmetric_rms_bbox_pct"] < 1e-12
    cross = score["remainder"]["cross"]
    assert cross[0]["predicted_free_sha256"] == cross[1]["predicted_free_sha256"]
    assert cross[0]["prediction_sample_sha256"] == cross[1]["prediction_sample_sha256"]
    assert cross[0]["target_full_bbox_diagonal"] != cross[1]["target_full_bbox_diagonal"]
    assert not cross[0]["rematching_performed"] and same_package is package
    direct = symmetric_surface_rms(remainder0, remainder1, diagonals[1])
    legacy = completion_rms(remainder0, remainder1, diagonals[1], seed=SURFACE_SEED)
    assert abs(direct["symmetric_rms_bbox_pct"] - legacy) < 1e-12
    assert abs(direct["symmetric_rms"] ** 2 -
               (direct["predicted_to_target_rms"] ** 2 + direct["target_to_predicted_rms"] ** 2) / 2) < 1e-12
    known_mask = np.asarray([True, False, False, False])
    native = extract_prediction(copied, context, "known_mask", known_mask=known_mask)
    assert native["free"].tobytes() == extraction["free"].tobytes()
    context2 = np.stack((a, b)).astype(np.float32)
    two = extract_prediction(full, context2)
    assert len(two["free"]) == 2 and set(two["audit"]["excluded_face_indices"]) == {0, 1}
    overlap = near_context_diagnostic(np.stack((a, c)).astype(np.float32), context, diagonals[0])
    assert overlap["exact_permutation_duplicate_face_count"] == 1
    assert overlap["near_complete_context_face_fraction"] == .5
    try:
        score_extracted(package, context + 1, full, [remainder0, remainder1], diagonals, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Changed cross-scoring context was accepted")
    json.dumps(score, allow_nan=False)
    return dict(status="PASS", tests=[
        "six_permutation_context_only_extraction", "copy_reuses_base_excluded_set",
        "copy_free_byte_identity", "original_context_vertex_order_exact_copy",
        "known_mask_excludes_only_known", "K1_K2_remainder_counts",
        "cross_scores_same_free_array_no_rematching", "cross_scores_same_prediction_samples",
        "target_specific_full_bbox_normalization", "symmetric_matches_original_completion",
        "both_direction_algebra", "declared_duplicate_geometry_diagnostic",
        "wrong_context_rejected", "JSON_safe"],
        model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    print(json.dumps(run_cpu_tests(), sort_keys=True))
