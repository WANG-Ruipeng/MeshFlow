"""Independent CPU metrics and fixed views for training-free triangle capture.

No model calls, geometry repair, or coordinate transformations occur here.
Distances are in model coordinates and relative to the supplied GT bbox diagonal;
they have no physical micrometre interpretation.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluate import completion_rms, raw_quality, surface_distance_squared


PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=np.int64)
RELATIVE_THRESHOLDS = (1e-3, 1e-4, 1e-5)


def _triangles(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError(f"Expected [N,9] or [N,3,3], got {value.shape}")
    return value


def exact_face_distances(tokens, targets):
    """Return D[N,K], argmin permutation[N,K], all permutation distances.

    D is the sum of three squared Euclidean vertex distances, with an exact
    NumPy minimum over all six permutations. This independently checks the
    PyTorch guidance implementation. Vertex RMS is sqrt(D / 3).
    """
    generated, desired = _triangles(tokens), _triangles(targets)
    if not (np.isfinite(generated).all() and np.isfinite(desired).all()):
        raise ValueError("Exact matching requires finite coordinates")
    reordered = desired[:, PERMUTATIONS, :]
    difference = generated[:, None, None, :, :] - reordered[None, :, :, :, :]
    all_distances = np.sum(difference * difference, axis=(-1, -2))
    argmin = np.argmin(all_distances, axis=-1)
    return np.min(all_distances, axis=-1), argmin, all_distances


def independent_check_points(targets, count=128, seed=29317):
    """Fixed uniform interior points; RNG is independent of model/noise/output.

    The same barycentric bank applies to each target. All weights are strictly
    positive, so target vertices/edges are not reused as evaluation points.
    """
    desired = _triangles(targets)
    uv = np.random.RandomState(seed).uniform(1e-9, 1 - 1e-9, (count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]), axis=1)
    if not np.all(weights > 0):
        raise AssertionError("Independent check points must be strictly interior")
    return np.einsum("pa,kad->kpd", weights, desired)


def _matched_rows(generated, desired, distances, permutation_indices, indices, diagonal):
    rows = []
    for target_index, face_index in enumerate(indices):
        permutation_index = int(permutation_indices[face_index, target_index])
        permutation = PERMUTATIONS[permutation_index]
        vertex_errors = np.linalg.norm(
            generated[face_index] - desired[target_index, permutation], axis=-1)
        rms = float(np.sqrt(distances[face_index, target_index] / 3))
        maximum = float(vertex_errors.max())
        rows.append(dict(
            target_index=target_index, face_index=int(face_index),
            target_vertex_permutation=permutation.tolist(),
            sum_squared_vertex_distance=float(distances[face_index, target_index]),
            vertex_rms=rms, vertex_rms_bbox_fraction=rms / diagonal,
            vertex_rms_bbox_pct=100 * rms / diagonal,
            max_vertex_error=maximum,
            max_vertex_error_bbox_fraction=maximum / diagonal,
            max_vertex_error_bbox_pct=100 * maximum / diagonal,
            per_vertex_error=vertex_errors.tolist()))
    return rows


def _matching_summary(rows, diagonal):
    rms = np.asarray([row["vertex_rms"] for row in rows])
    maximum = np.asarray([row["max_vertex_error"] for row in rows])
    return dict(
        rows=rows, mean_target_vertex_rms=float(rms.mean()),
        aggregate_vertex_rms=float(np.sqrt(np.mean(rms * rms))),
        mean_target_vertex_rms_bbox_pct=float(100 * rms.mean() / diagonal),
        aggregate_vertex_rms_bbox_pct=float(100 * np.sqrt(np.mean(rms * rms)) / diagonal),
        maximum_vertex_error_bbox_pct=float(100 * maximum.max() / diagonal),
        thresholds=[dict(
            relative_bbox_threshold=threshold,
            coordinate_threshold=threshold * diagonal,
            target_count_rms_pass=int(np.sum(rms <= threshold * diagonal)),
            target_count_all_vertices_pass=int(np.sum(maximum <= threshold * diagonal)),
            all_targets_rms_pass=bool(np.all(rms <= threshold * diagonal)),
            all_targets_all_vertices_pass=bool(np.all(maximum <= threshold * diagonal)))
            for threshold in RELATIVE_THRESHOLDS])


def _displacement(generated, base, excluded_indices, diagonal):
    keep = np.ones(len(generated), dtype=bool)
    keep[np.asarray(sorted(set(excluded_indices)), dtype=np.int64)] = False
    difference = generated[keep] - base[keep]
    result = dict(
        excluded_face_indices=sorted(set(map(int, excluded_indices))),
        remaining_face_count=int(keep.sum()),
        correspondence="Original Euler face and vertex indices; no rematching")
    if not len(difference):
        result.update(vertex_rms=None, vertex_rms_bbox_pct=None,
                      max_vertex_displacement=None, max_vertex_displacement_bbox_pct=None,
                      unchanged_face_count=0, unchanged_face_fraction=None)
        return result
    vertex_distances = np.linalg.norm(difference, axis=-1)
    rms = float(np.sqrt(np.mean(vertex_distances * vertex_distances)))
    unchanged = np.all(generated[keep] == base[keep], axis=(1, 2))
    result.update(
        vertex_rms=rms, vertex_rms_bbox_pct=100 * rms / diagonal,
        max_vertex_displacement=float(vertex_distances.max()),
        max_vertex_displacement_bbox_pct=float(100 * vertex_distances.max() / diagonal),
        median_face_vertex_rms_bbox_pct=float(
            100 * np.median(np.sqrt(np.mean(vertex_distances * vertex_distances, axis=1))) / diagonal),
        unchanged_face_count=int(unchanged.sum()),
        unchanged_face_fraction=float(unchanged.mean()))
    return result


def evaluate_output(tokens, targets, full_target, bbox_diagonal,
                    base_tokens=None, check_points=None):
    """JSON-safe exact matching, surface, quality, and Base-displacement scores.

    A one-to-one assignment minimizes the sum of D, not independently selected
    nearest targets. Both RMS and maximum-vertex threshold checks are retained:
    a low interior point-to-surface distance alone does not prove face capture.
    """
    generated, desired, reference = map(_triangles, (tokens, targets, full_target))
    diagonal = float(bbox_diagonal)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("bbox_diagonal must be finite and positive")
    if len(desired) not in (1, 2) or len(generated) < len(desired):
        raise ValueError("This diagnostic requires K in {1,2}, N >= K")
    if not (np.isfinite(desired).all() and np.isfinite(reference).all()):
        raise ValueError("Input target geometry must be finite")
    result = dict(
        schema="soft_matching_metrics_v1", num_faces=len(generated),
        num_targets=len(desired), bbox_diagonal=diagonal,
        units="Model coordinates; percentage of supplied GT bbox diagonal; no physical unit claim",
        quality=raw_quality(generated),
        independent_matching=None, one_to_one_matching=None,
        interior_surface=None, completion_rms_bbox_pct=None,
        nonwinning_displacement_vs_same_noise_base=None)
    if not np.isfinite(generated).all():
        result["status"] = "NONFINITE_OUTPUT"
        return result

    distances, permutation_indices, _ = exact_face_distances(generated, desired)
    closest = np.argmin(distances, axis=0)
    independent_rows = _matched_rows(
        generated, desired, distances, permutation_indices, closest, diagonal)
    independent = _matching_summary(independent_rows, diagonal)
    independent.update(
        selected_face_indices=closest.tolist(),
        unique_selected_face_count=len(set(closest.tolist())),
        collision_count=len(desired) - len(set(closest.tolist())),
        has_collision=len(set(closest.tolist())) < len(desired),
        collision_target_groups=[
            np.flatnonzero(closest == face).tolist()
            for face in sorted(set(closest.tolist()))
            if np.sum(closest == face) > 1])

    assigned_faces, assigned_targets = linear_sum_assignment(distances)
    by_target = np.empty(len(desired), dtype=np.int64)
    by_target[assigned_targets] = assigned_faces
    if len(set(by_target.tolist())) != len(desired):
        raise AssertionError("Assignment must select distinct generated faces")
    assigned_rows = _matched_rows(
        generated, desired, distances, permutation_indices, by_target, diagonal)
    assigned = _matching_summary(assigned_rows, diagonal)
    assigned.update(
        objective="Minimum sum of six-permutation squared vertex distances D",
        selected_face_indices=by_target.tolist(),
        distinct_faces=True,
        total_sum_squared_vertex_distance=float(distances[by_target, np.arange(len(desired))].sum()))

    points = independent_check_points(desired) if check_points is None else np.asarray(check_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[0] != len(desired) or points.shape[-1] != 3:
        raise ValueError("check_points must have shape [K,P,3]")
    if not np.isfinite(points).all() or points.shape[1] == 0:
        raise ValueError("check_points must contain finite points")
    per_target = []
    squared = []
    for target_index, target_points in enumerate(points):
        d2 = surface_distance_squared(target_points, generated)
        squared.append(d2)
        rms = float(np.sqrt(d2.mean()))
        per_target.append(dict(
            target_index=target_index, check_point_count=len(target_points),
            rms=rms, rms_bbox_pct=100 * rms / diagonal,
            max_bbox_pct=float(100 * np.sqrt(d2.max()) / diagonal)))
    result.update(
        status="PASS", independent_matching=independent, one_to_one_matching=assigned,
        interior_surface=dict(
            check_point_rule="Fixed uniform strictly interior barycentric samples, output-independent",
            check_point_seed=29317 if check_points is None else None,
            point_bank="generated_independent" if check_points is None else "caller_supplied",
            per_target=per_target,
            rms_bbox_pct=float(100 * np.sqrt(np.mean([v.mean() for v in squared])) / diagonal)),
        completion_rms_bbox_pct=completion_rms(generated, reference, diagonal, seed=9911),
        completion_rule="Existing symmetric surface RMS; 512 area-uniform points each direction; seed 9911")
    if base_tokens is not None:
        base = _triangles(base_tokens)
        if base.shape != generated.shape or not np.isfinite(base).all():
            raise ValueError("Base must be finite with the same ordered token shape")
        result["nonwinning_displacement_vs_same_noise_base"] = dict(
            excluding_independent_winners=_displacement(generated, base, closest, diagonal),
            excluding_one_to_one_assignment=_displacement(generated, base, by_target, diagonal),
            all_faces=_displacement(generated, base, [], diagonal))
    return result


def plot_fixed(outputs, targets, savepath):
    """Plot fixed K=1/2 rows and Base/lambda5/lambda20 columns, one shared range.

    outputs: {1: {'Base': array, 'lambda5': array, 'lambda20': array}, 2: ...}
    targets: {1: [1,3,3], 2: [2,3,3]} (a two-element list is also accepted).
    The caller selects the predeclared parent/noise, never a visually best case.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if not isinstance(targets, dict):
        targets = {1: targets[0], 2: targets[1]}
    columns = ("Base", "lambda5", "lambda20")
    arrays = [_triangles(outputs[k][name]) for k in (1, 2) for name in columns]
    arrays.extend(_triangles(targets[k]) for k in (1, 2))
    finite_vertices = np.concatenate([a.reshape(-1, 3) for a in arrays])
    finite_vertices = finite_vertices[np.isfinite(finite_vertices).all(axis=1)]
    if not len(finite_vertices):
        raise ValueError("Cannot plot a completely nonfinite collection")
    low, high = finite_vertices.min(axis=0), finite_vertices.max(axis=0)
    center = (low + high) / 2
    half_extent = max(float((high - low).max()) * .55, 1e-3)
    bounds = np.stack((center - half_extent, center + half_extent), axis=1)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    panels = []
    for row, k in enumerate((1, 2)):
        desired = _triangles(targets[k])
        for column, name in enumerate(columns):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            tri = _triangles(outputs[k][name])
            finite = np.isfinite(tri).all()
            valid_tri = tri[np.isfinite(tri).all(axis=(1, 2))]
            closed = np.concatenate((valid_tri, valid_tri[:, :1]), axis=1)
            ax.add_collection3d(Line3DCollection(closed, colors="#687787", linewidths=.30, alpha=.38))
            ax.scatter(valid_tri[:, :, 0].ravel(), valid_tri[:, :, 1].ravel(),
                       valid_tri[:, :, 2].ravel(), s=.65, c="#65788a", alpha=.35)
            chosen = []
            if finite:
                d, _, _ = exact_face_distances(tri, desired)
                face_ids, target_ids = linear_sum_assignment(d)
                chosen = face_ids[np.argsort(target_ids)].tolist()
                best = tri[chosen]
                ax.add_collection3d(Poly3DCollection(
                    best, facecolors="#1ba3ba", edgecolors="#006782", linewidths=1.6, alpha=.45))
                ax.scatter(best[:, :, 0].ravel(), best[:, :, 1].ravel(),
                           best[:, :, 2].ravel(), s=12, c="#007e99")
            ax.add_collection3d(Poly3DCollection(
                desired, facecolors="#ff3030", edgecolors="#c80000", linewidths=2, alpha=.22))
            ax.scatter(desired[:, :, 0].ravel(), desired[:, :, 1].ravel(),
                       desired[:, :, 2].ravel(), s=28, c="#de1010", marker="x", depthshade=False)
            ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1]); ax.set_zlim(bounds[2])
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            suffix = "" if finite else " [NONFINITE: finite faces only shown]"
            label = name.replace("lambda", "lambda=")
            ax.set_title(f"K={k} | {label}{suffix}")
            panels.append(dict(k=k, column=name, assigned_highlight_faces=chosen,
                               nonfinite=not finite))
    fig.suptitle("Fixed parent / fixed noise | red: target; cyan: distinct best matches\n"
                 "Identical camera and full shared bounds; no geometry repair or clipping", fontsize=13)
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=180)
    plt.close(fig)
    return dict(path=str(savepath), panels=panels, shared_xyz_limits=bounds.tolist(),
                camera=dict(elevation=22, azimuth=35),
                extent_rule="All finite coordinates of all six outputs and both target sets; shared cubic bounds",
                nonfinite_policy="Explicit panel annotation; show finite faces only; preserve original raw arrays")


def cpu_fixture_checks():
    """Small mathematical fixtures, with no official model or GPU calls."""
    import json
    import tempfile

    a = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    b = a + np.asarray([0., 0., 2.])
    c = a + np.asarray([3., 0., 0.])
    generated = np.stack((a[[2, 0, 1]], b[[1, 2, 0]], c))
    desired = np.stack((a, b))
    diag = float(np.linalg.norm(np.ptp(generated.reshape(-1, 3), axis=0)))
    pristine = generated.copy()
    result = evaluate_output(generated, desired, generated, diag, generated)
    assert result["independent_matching"]["selected_face_indices"] == [0, 1]
    assert result["one_to_one_matching"]["aggregate_vertex_rms"] == 0
    assert result["one_to_one_matching"]["thresholds"][-1]["all_targets_all_vertices_pass"]
    assert result["interior_surface"]["rms_bbox_pct"] < 1e-12
    assert result["completion_rms_bbox_pct"] < 1e-12
    assert result["nonwinning_displacement_vs_same_noise_base"]["all_faces"]["vertex_rms"] == 0
    assert np.array_equal(generated, pristine)

    collision = evaluate_output(generated, np.stack((a, a)), generated, diag)
    assert collision["independent_matching"]["has_collision"]
    assert collision["independent_matching"]["collision_target_groups"] == [[0, 1]]
    assert collision["independent_matching"]["aggregate_vertex_rms"] == 0
    assert collision["one_to_one_matching"]["aggregate_vertex_rms"] > 0
    assert not collision["one_to_one_matching"]["thresholds"][0]["all_targets_all_vertices_pass"]

    shifted = generated.copy(); shifted[2] += np.asarray([.1, -.2, .3])
    displacement = evaluate_output(shifted, desired, generated, diag, generated)
    expected = float(np.linalg.norm([.1, -.2, .3]))
    measured = displacement["nonwinning_displacement_vs_same_noise_base"]["excluding_one_to_one_assignment"]
    assert measured["remaining_face_count"] == 1
    assert abs(measured["vertex_rms"] - expected) < 1e-12
    assert measured["unchanged_face_count"] == 0

    translation = np.asarray([.03, -.04, .05])
    single = evaluate_output((a + translation)[None], a[None], a[None], 1.)
    assert abs(single["independent_matching"]["rows"][0]["vertex_rms"] - np.linalg.norm(translation)) < 1e-12
    nan_output = generated.copy(); nan_output[0, 0, 0] = np.nan
    nonfinite = evaluate_output(nan_output, desired, generated, diag)
    assert nonfinite["status"] == "NONFINITE_OUTPUT"
    assert nonfinite["quality"]["nan_count"] == 1
    for value in (result, collision, displacement, single, nonfinite):
        json.dumps(value, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix="softmatch_metrics_") as directory:
        path = Path(directory) / "fixture.png"
        figure = plot_fixed(
            {k: {name: generated for name in ("Base", "lambda5", "lambda20")} for k in (1, 2)},
            {1: a[None], 2: desired}, path)
        assert path.is_file() and path.stat().st_size > 10000
        assert len(figure["panels"]) == 6
    return dict(status="PASS", tests=[
        "six_permutation_exact_zero", "distinct_assignment_exact_zero",
        "duplicate_target_collision_detected", "one_to_one_rejects_shared_exact_face",
        "independent_interior_surface_zero", "original_completion_zero",
        "same_noise_nonwinning_displacement", "known_translation_rms",
        "nonfinite_is_explicit", "json_safe", "raw_arrays_unchanged",
        "fixed_six_panel_png"], model_forwards=0, backward_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    import json
    print(json.dumps(cpu_fixture_checks(), sort_keys=True))
