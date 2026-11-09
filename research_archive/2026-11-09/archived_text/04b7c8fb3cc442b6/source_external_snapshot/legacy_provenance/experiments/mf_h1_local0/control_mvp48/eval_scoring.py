"""Control MVP48 CPU-only, variable-N adapter over the original geometry scorer.

No model, matching, OT, repair, or target-dependent prediction extraction occurs
here. R12 scalar names are retained for compatibility; they mean this case's own
GT free complement, not a fixed twelve-face condition or another condition's GT.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from experiments.mf_h1_local0.evaluate import sample_mesh_surface, surface_distance_squared, raw_quality
from experiments.mf_h1_local0.native_inpainting_metrics import array_sha
from experiments.mf_h1_local0.native_patch_seams import (
    extract_boundary_segments, _sample_boundary, point_to_triangle_surface_squared,
)
from experiments.mf_h1_local0.guided_native_geometry_audit import pair_scan, PROPER, COPLANAR
from experiments.mf_h1_local0.gpu_guidance_evaluation import shape_quality, _detector_view

ROOT = Path(__file__).resolve().parents[3]
EDGE_SOURCE = ROOT / "native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py"
SURFACE_POINTS = 512
SURFACE_SEED = 9911
COVERAGE_SEED = 9973
BOUNDARY_SEED = 9919

METRIC_SPECS = {}
for _name in (
    "remainder_symmetric_RMS_bbox_pct", "free_to_R12_RMS_bbox_pct", "R12_to_free_RMS_bbox_pct",
    "fixed_R12_coverage_RMS_bbox_pct", "boundary_RMS_bbox_pct", "boundary_p95_bbox_pct",
    "Gamma_to_free_edges_RMS_bbox_pct", "Gamma_to_free_edges_p95_bbox_pct",
):
    METRIC_SPECS[_name] = dict(direction="lower", unit="% L")
for _name in (
    "fixed_R12_coverage_within_1pct_fraction", "boundary_coverage_fraction",
    "Gamma_to_free_edges_coverage_0.001_fraction",
):
    METRIC_SPECS[_name] = dict(direction="higher", unit="fraction")
for _name in (
    "CF_proper_pair_count", "CF_coplanar_pair_count", "CF_unresolved_or_degenerate_pair_count",
    "FF_proper_pair_count", "FF_coplanar_pair_count", "FF_illegal_pair_count",
    "FF_unresolved_or_degenerate_pair_count", "free_near_degenerate_count",
    "free_exact_zero_area_count", "free_nan_coordinate_count", "free_inf_coordinate_count",
):
    METRIC_SPECS[_name] = dict(direction="lower", unit="count")
for _name in (
    "CF_proper_affected_free_fraction", "FF_proper_affected_free_fraction",
    "FF_illegal_affected_free_fraction", "free_minimum_angle_lt5_fraction",
    "free_q_gt10_fraction", "free_near_degenerate_fraction", "free_exact_zero_area_fraction",
):
    METRIC_SPECS[_name] = dict(direction="lower", unit="fraction")
METRIC_SPECS["free_angle_p05_degrees"] = dict(direction="higher", unit="degrees")
METRIC_SPECS["free_q_mean"] = dict(direction="lower", unit="dimensionless")
METRIC_SPECS["C_preserved"] = dict(direction="descriptive", unit="bool")
for _name in ("actual_N", "actual_K", "actual_free_faces"):
    METRIC_SPECS[_name] = dict(direction="descriptive", unit="faces")
for _name in ("free_total_surface_area", "GT_free_total_surface_area"):
    METRIC_SPECS[_name] = dict(direction="descriptive", unit="model coordinate squared")
METRIC_SPECS["free_area_over_GT_free_area"] = dict(direction="descriptive", unit="ratio")


def _tri(value):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("Expected [N,9] or [N,3,3] unpadded triangles")
    return value


def _inc(ledger, key, amount=1):
    if ledger is not None:
        ledger[key] = ledger.get(key, 0) + int(amount)


def source_identity():
    files = [Path(__file__), EDGE_SOURCE]
    files += [ROOT / "experiments/mf_h1_local0" / name for name in (
        "evaluate.py", "native_inpainting_metrics.py", "native_patch_seams.py",
        "guided_native_geometry_audit.py", "guided_native_geometry_geometry.py",
        "gpu_guidance_evaluation.py",
    )]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _edge_module():
    # Execute the original standalone helper from its retained source snapshot.
    spec = importlib.util.spec_from_file_location("control_mvp48_original_edge_metric", EDGE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_reference(known, gt_free, L, *, gamma_points=None, gamma_weights=None,
                      surface_seed=SURFACE_SEED, coverage_seed=COVERAGE_SEED,
                      boundary_seed=BOUNDARY_SEED):
    """Freeze target-only query arrays once, then share across all model scores.

    The original pairwise stream samples prediction first and GT second. Its
    weighted RandomState.choice consumes 512 uniforms regardless of face count
    or probabilities, followed by 512x2 barycentric uniforms. Sampling GT once
    into a discarded buffer advances exactly that stream without reading any
    prediction; the second draw is the unchanged original GT query set.
    """
    C, target, L = _tri(known), _tri(gt_free), float(L)
    if not len(C) or not len(target) or not np.isfinite(C).all() or not np.isfinite(target).all():
        raise ValueError("Finite, nonempty C and own GT free complement required")
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Positive full-GT bbox diagonal required")
    rng = np.random.RandomState(int(surface_seed))
    if sample_mesh_surface(target, SURFACE_POINTS, rng) is None:
        raise ValueError("GT free complement has no positive finite surface area")
    pairwise = sample_mesh_surface(target, SURFACE_POINTS, rng)
    coverage = sample_mesh_surface(target, SURFACE_POINTS, np.random.RandomState(int(coverage_seed)))
    if (gamma_points is None) != (gamma_weights is None):
        raise ValueError("Supply both frozen Gamma points and their arc-length weights")
    segments, boundary = extract_boundary_segments(C)
    if gamma_points is None:
        gamma_points, _, gamma_weights, counts = _sample_boundary(segments, int(boundary_seed), SURFACE_POINTS)
        gamma_source = "Original exact C boundary extraction and arc-length sampling"
    else:
        gamma_points = np.asarray(gamma_points, dtype=np.float64).reshape(-1, 3)
        gamma_weights = np.asarray(gamma_weights, dtype=np.float64).reshape(-1)
        counts, gamma_source = None, "Caller-provided frozen Gamma arrays"
    if gamma_points.shape != (len(gamma_weights), 3) or not np.isfinite(gamma_points).all():
        raise ValueError("Gamma shape/nonfinite mismatch")
    if not np.isfinite(gamma_weights).all() or np.any(gamma_weights < 0):
        raise ValueError("Finite nonnegative Gamma weights required")
    if len(gamma_weights) and not np.isclose(gamma_weights.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError("Original Gamma weights must sum to one; do not renormalize after selection")
    arrays = dict(pairwise_target_points=pairwise, coverage_points=coverage,
                  gamma_points=np.array(gamma_points, copy=True), gamma_weights=np.array(gamma_weights, copy=True))
    for value in arrays.values():
        value.setflags(write=False)
    return dict(arrays=arrays, audit=dict(
        schema="control_mvp48_reference_v1", known_sha256=array_sha(C), GT_free_sha256=array_sha(target),
        L=L, K=len(C), GT_free_faces=len(target), surface_points=SURFACE_POINTS,
        surface_seed=int(surface_seed), coverage_seed=int(coverage_seed), boundary_seed=int(boundary_seed),
        array_sha256={key: array_sha(value) for key, value in arrays.items()},
        boundary=boundary, boundary_sample_counts=counts, gamma_source=gamma_source,
        target_scope="This condition's own GT free complement; C excluded",
        prediction_rng="Independent local RandomState(surface_seed), original area-weighted protocol",
        GT_pairwise_rng="Second sample after original fixed-sized first draw; independent of prediction geometry",
        model_forwards=0, OT_calls=0))


def _check_reference(reference, C, target, L):
    audit = reference["audit"]
    if (audit["known_sha256"] != array_sha(C) or audit["GT_free_sha256"] != array_sha(target)
            or audit["L"] != float(L)):
        raise ValueError("Reference belongs to a different C, own GT complement, or bbox")
    for key, expected in audit["array_sha256"].items():
        if array_sha(reference["arrays"][key]) != expected:
            raise ValueError("Frozen reference array changed: " + key)


def prepare_reference_from_sample(sample, *, surface_seed=SURFACE_SEED,
                                  coverage_seed=COVERAGE_SEED, boundary_seed=BOUNDARY_SEED):
    """Freeze A's true known/free crossing Gamma; never assign generated faces."""
    full = _tri(sample["full_target"])
    C, target = _tri(sample["constraints"]), _tri(sample["free_target"])
    interface = sample["interface"]
    known_ids = np.asarray(interface["known_source_face_ids"], dtype=np.int64)
    free_ids = np.asarray(interface["free_source_face_ids"], dtype=np.int64)
    known_corners = np.asarray(interface["known_corner_ids"], dtype=np.int64)
    free_corners = np.asarray(interface["free_corner_ids"], dtype=np.int64)
    vertex_ids = np.asarray(interface["edge_source_vertex_ids"], dtype=np.int64)
    count = len(known_ids)
    if count == 0 or free_ids.shape != (count,) or known_corners.shape != (count, 2) or free_corners.shape != (count, 2) or vertex_ids.shape != (count, 2):
        raise ValueError("Nonempty registered interface edge arrays required")
    selected = np.asarray(sample["source_face_ids"], dtype=np.int64)
    complement = np.asarray(sample["free_source_ids"], dtype=np.int64)
    if full[selected].tobytes() != C.tobytes() or full[complement].tobytes() != target.tobytes():
        raise ValueError("A's C / own free complement differs from its full target")
    if set(selected.tolist()) & set(complement.tolist()) or sorted(np.r_[selected, complement].tolist()) != list(range(len(full))):
        raise ValueError("Known and free source IDs must partition this parent exactly")
    if not set(known_ids.tolist()) <= set(selected.tolist()) or not set(free_ids.tolist()) <= set(complement.tolist()):
        raise ValueError("Gamma endpoints do not cross this condition's known/free sets")
    segments = full[known_ids[:, None], known_corners]
    other = full[free_ids[:, None], free_corners]
    if segments.tobytes() != other.tobytes():
        raise ValueError("Registered GT interface corner coordinates disagree")
    lengths = np.linalg.norm(segments[:, 1].astype(np.float64)-segments[:, 0], axis=-1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0) or len({tuple(x) for x in vertex_ids}) != count:
        raise ValueError("Registered interface must contain unique positive-length finite edges")
    points, _, weights, counts = _sample_boundary(segments, boundary_seed, SURFACE_POINTS)
    result = prepare_reference(C, target, sample["bbox_diagonal"], gamma_points=points, gamma_weights=weights,
                               surface_seed=surface_seed, coverage_seed=coverage_seed, boundary_seed=boundary_seed)
    result["audit"].update(gamma_source="Session A registered source-index known/free interface; unchanged length-weighted sampler",
        boundary_sample_counts=counts, gamma_edge_count=count, gamma_segment_sha256=array_sha(segments),
        gamma_source_vertex_ids_sha256=array_sha(vertex_ids), gamma_total_length=float(lengths.sum()),
        gamma_excludes_source_open_edges=True, generated_GT_correspondence_used=False)
    return result


def _weighted_surface(free, points, weights, L, ledger):
    if not len(points):
        return dict(status="NO_BOUNDARY", rms_bbox_pct=None, p95_bbox_pct=None,
                    coverage_fraction=None, per_point_distances=[])
    _inc(ledger, "Gamma_surface_calls_attempted")
    squared = point_to_triangle_surface_squared(points, free, L)
    _inc(ledger, "Gamma_surface_point_triangle_pairs", len(points) * len(free))
    if squared is None or not np.isfinite(squared).all():
        raise ValueError("Gamma surface distance unavailable; absent candidates are not zero")
    distances = np.sqrt(squared)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, .95 * cumulative[-1], side="left")), len(order) - 1)
    rms, p95 = float(np.sqrt(np.sum(weights * squared))), float(distances[order[position]])
    _inc(ledger, "Gamma_surface_calls_returned")
    return dict(status="PASS", rms=rms, p95=p95, rms_bbox_pct=100*rms/L,
                p95_bbox_pct=100*p95/L, coverage_fraction=float(weights[distances < .001*L].sum()),
                coverage_threshold_bbox_fraction=.001, comparison="strict_less_than",
                point_count=len(points), point_sha256=array_sha(points), weight_sha256=array_sha(weights),
                per_point_distances=distances.tolist(), length_weighted=True)


def score_free(free, known, gt_free, L, *, reference, ledger=None, include_pair_audit=True):
    """Score exactly the supplied free set; suitable for offline four-cell reuse.

    include_pair_audit=False permits only reused-output distance comparisons.
    It must not be used for the 96 primary endpoint scores.
    """
    tick = time.perf_counter()
    if ledger is None:
        ledger = {}
    before_ledger = dict(ledger)
    free, C, target, L = _tri(free), _tri(known), _tri(gt_free), float(L)
    _check_reference(reference, C, target, L)
    result = dict(schema="control_mvp48_score_v1", status="PASS", failure_reason=None,
                  metrics={key: None for key in METRIC_SPECS}, detail={}, ledger={})
    m, detail = result["metrics"], result["detail"]
    m.update(actual_K=len(C), actual_free_faces=len(free), actual_N=len(C)+len(free),
             free_nan_coordinate_count=int(np.isnan(free).sum()), free_inf_coordinate_count=int(np.isinf(free).sum()))
    detail.update(reference=reference["audit"], predicted_free_sha256=array_sha(free),
                  prediction_extraction="Caller-supplied free triangles, with copied C excluded",
                  free_quality=raw_quality(free) if len(free) else dict(status="EMPTY_FREE"),
                  exact_topology="NOT_RUN", watertight_certification="NOT_RUN", exact_predicates="NOT_RUN",
                  geometry_modified=False, GT_face_correspondence_used=False,
                  model_forwards=0, model_backwards=0, geometry_autograd_calls=0, OT_calls=0)
    _inc(ledger, "endpoint_scores_attempted")
    original = free.tobytes()
    if not len(free) or not np.isfinite(free).all():
        result.update(status="INVALID_OUTPUT", failure_reason="EMPTY_FREE" if not len(free) else "NONFINITE_FREE")
        detail["pair_audit_status"] = "NOT_RUN_INVALID_OUTPUT"
    else:
        arrays = reference["arrays"]
        predicted = sample_mesh_surface(free, SURFACE_POINTS, np.random.RandomState(reference["audit"]["surface_seed"]))
        _inc(ledger, "prediction_surface_sample_calls")
        if predicted is None:
            result.update(status="INVALID_OUTPUT", failure_reason="ZERO_OR_INVALID_SURFACE_AREA")
        else:
            d2a = surface_distance_squared(predicted, target)
            d2b = surface_distance_squared(arrays["pairwise_target_points"], free)
            _inc(ledger, "remainder_surface_distance_calls", 2)
            _inc(ledger, "remainder_point_triangle_pairs", SURFACE_POINTS * (len(target) + len(free)))
            if not np.isfinite(d2a).all() or not np.isfinite(d2b).all():
                result.update(status="INVALID_OUTPUT", failure_reason="NONFINITE_SURFACE_DISTANCE")
            else:
                m.update(remainder_symmetric_RMS_bbox_pct=float(100*np.sqrt((d2a.mean()+d2b.mean())/2)/L),
                         free_to_R12_RMS_bbox_pct=float(100*np.sqrt(d2a.mean())/L),
                         R12_to_free_RMS_bbox_pct=float(100*np.sqrt(d2b.mean())/L))
                detail["remainder"] = dict(prediction_sample_sha256=array_sha(predicted),
                    target_sample_sha256=array_sha(arrays["pairwise_target_points"]),
                    predicted_to_target_mean_squared=float(d2a.mean()), target_to_predicted_mean_squared=float(d2b.mean()),
                    points_per_direction=SURFACE_POINTS, predicted_faces=len(free), target_faces=len(target),
                    free_to_GT_distances=np.sqrt(d2a).tolist(), GT_to_free_distances=np.sqrt(d2b).tolist())
        d2c = surface_distance_squared(arrays["coverage_points"], free)
        _inc(ledger, "fixed_GT_coverage_distance_calls")
        _inc(ledger, "fixed_GT_coverage_point_triangle_pairs", SURFACE_POINTS * len(free))
        if np.isfinite(d2c).all():
            m.update(fixed_R12_coverage_RMS_bbox_pct=float(100*np.sqrt(d2c.mean())/L),
                     fixed_R12_coverage_within_1pct_fraction=float((d2c <= (.01*L)**2).mean()))
            detail["fixed_GT_coverage"] = dict(points_sha256=array_sha(arrays["coverage_points"]),
                per_point_distances=np.sqrt(d2c).tolist(), comparison="less_than_or_equal", threshold_bbox_fraction=.01)
        else:
            reason = "NONFINITE_FIXED_GT_COVERAGE"
            if result["failure_reason"]:
                reason = result["failure_reason"] + ";" + reason
            result.update(status="INVALID_OUTPUT", failure_reason=reason)
            detail["fixed_GT_coverage"] = dict(status="NONFINITE_DISTANCE",
                points_sha256=array_sha(arrays["coverage_points"]),
                nonfinite_distance_count=int((~np.isfinite(d2c)).sum()),
                comparison="less_than_or_equal", threshold_bbox_fraction=.01)
        boundary = _weighted_surface(free, arrays["gamma_points"], arrays["gamma_weights"], L, ledger)
        detail["boundary"] = boundary
        m.update(boundary_RMS_bbox_pct=boundary["rms_bbox_pct"], boundary_p95_bbox_pct=boundary["p95_bbox_pct"],
                 boundary_coverage_fraction=boundary["coverage_fraction"])
        if len(arrays["gamma_points"]):
            edge_module = _edge_module()
            edge = edge_module.gamma_to_free_edges(free, arrays["gamma_points"], arrays["gamma_weights"], L, ledger=ledger)
            ordering = edge_module.check_surface_lower_bound(edge["per_point_distances"], boundary["per_point_distances"], L)
            if ordering["status"] != "PASS":
                raise AssertionError("Finite-edge distance fell below surface beyond original tolerance")
            detail.update(finite_edges=edge, edge_surface_ordering=ordering)
            m.update(Gamma_to_free_edges_RMS_bbox_pct=edge["RMS_bbox_pct"],
                     Gamma_to_free_edges_p95_bbox_pct=edge["p95_bbox_pct"],
                     **{"Gamma_to_free_edges_coverage_0.001_fraction": edge["coverage_by_bbox_fraction"]["0.001"]})
        q = shape_quality(free, L)
        q.pop("guarded_EA_CPU_formula_reference", None)
        q.pop("EA_scope", None)
        detail["free_shape_quality"] = q
        target_area = float(shape_quality(target, L)["total_area"])
        m.update(free_minimum_angle_lt5_fraction=q["min_angle_lt5_fraction"], free_angle_p05_degrees=q["angle_p05_degrees"],
                 free_q_gt10_fraction=q["q_actual_gt10_fraction"], free_q_mean=q["q_actual"]["mean"],
                 free_near_degenerate_count=q["near_degenerate_area_count"], free_near_degenerate_fraction=q["near_degenerate_area_count"]/len(free),
                 free_exact_zero_area_count=q["exact_zero_area_count"], free_exact_zero_area_fraction=q["exact_zero_area_count"]/len(free),
                 free_total_surface_area=q["total_area"], GT_free_total_surface_area=target_area,
                 free_area_over_GT_free_area=q["total_area"]/target_area)
        _inc(ledger, "free_shape_quality_calls")
        _inc(ledger, "GT_free_area_calculations")
        if include_pair_audit:
            cf = pair_scan(C, free, L, ledger=ledger, independent=True)
            ff = pair_scan(free, free, L, ledger=ledger, self_pairs=True, independent=True)
            if cf["tested_pairs"] != len(C)*len(free) or ff["tested_pairs"] != len(free)*(len(free)-1)//2:
                raise AssertionError("Variable-N complete pair-scan denominator mismatch")
            illegal_ids = sorted({index for row in ff["contacts"] if row["classification"] in (PROPER, COPLANAR) for index in row["pair"]})
            ff.update(illegal_affected_faces=illegal_ids, illegal_affected_face_fraction=len(illegal_ids)/len(free))
            detail.update(CF=cf, FF=ff, pair_audit_status="COMPLETE_NUMERICAL_SCAN",
                detector_views={scope: {which: _detector_view(scan, which) for which in ("primary", "independent", "verified")}
                                for scope, scan in (("CF", cf), ("FF", ff))})
            for scope, scan in (("CF", cf), ("FF", ff)):
                m.update({scope+"_proper_pair_count": scan["proper_pair_count"],
                          scope+"_coplanar_pair_count": scan["coplanar_positive_area_pairs"],
                          scope+"_proper_affected_free_fraction": scan["proper_affected_face_fraction"],
                          scope+"_unresolved_or_degenerate_pair_count": scan["unresolved_or_degenerate_pairs"]})
            m.update(FF_illegal_pair_count=ff["proper_pair_count"]+ff["coplanar_positive_area_pairs"],
                     FF_illegal_affected_free_fraction=ff["illegal_affected_face_fraction"])
            unresolved = cf["unresolved_or_degenerate_pairs"] + ff["unresolved_or_degenerate_pairs"]
            illegal = cf["proper_pair_count"]+cf["coplanar_positive_area_pairs"]+m["FF_illegal_pair_count"]
            detail["geometry_status"] = "FAIL_DETECTED_ILLEGAL_INTERSECTIONS" if illegal else "UNRESOLVED" if unresolved else "NO_ILLEGAL_CF_FF_INTERSECTIONS_DETECTED"
        else:
            detail["pair_audit_status"] = "NOT_RUN_DISTANCE_ONLY_REUSE"
    if free.tobytes() != original:
        raise AssertionError("Read-only scoring mutated raw free geometry")
    _inc(ledger, "endpoint_scores_returned")
    result["ledger"] = {key: value-before_ledger.get(key, 0) for key, value in ledger.items() if value-before_ledger.get(key, 0)}
    result["scoring_seconds"] = time.perf_counter()-tick
    json.dumps(result, allow_nan=False)
    return result


def score_output(raw, known, gt_free, L, *, known_mask=None, reference, ledger=None):
    """Validate FP32 original endpoint and extract exactly the actual known mask.

    Pass the unpadded N faces. A non-prefix known mask is supported; padding is
    never treated as geometry. Failures remain rows, with no seed substitution.
    """
    full, C, target = _tri(raw), _tri(known), _tri(gt_free)
    if full.dtype != np.float32 or C.dtype != np.float32:
        raise ValueError("Original FP32 raw endpoint and C required")
    mask = np.arange(len(full)) < len(C) if known_mask is None else np.asarray(known_mask)
    if mask.dtype != np.bool_ or mask.shape not in ((len(full),), (1, len(full))):
        raise ValueError("Actual bool[N] known mask required")
    mask = mask.reshape(-1)
    if int(mask.sum()) != len(C):
        raise ValueError("Known mask count differs from C")
    preserved = full[mask].tobytes() == C.tobytes()
    result = score_free(full[~mask], C, target, L, reference=reference, ledger=ledger)
    result["metrics"].update(C_preserved=int(preserved), actual_N=len(full))
    result["detail"].update(raw_sha256=array_sha(full), known_mask_sha256=array_sha(mask),
        known_slots=np.flatnonzero(mask).tolist(), free_slots=np.flatnonzero(~mask).tolist(),
        C_preserved=preserved, C_original_corner_order_preserved=preserved,
        prediction_extraction="Actual bool known mask only; no GT face IDs, matching, or reassignment",
        extraction_count=1, expected_N=len(C)+len(target), actual_N=len(full))
    errors = []
    if not preserved:
        errors.append("C_NOT_PRESERVED")
    if len(full) != len(C)+len(target):
        errors.append("ACTUAL_N_MISMATCH")
    if errors:
        result.update(status="INVALID_OUTPUT", failure_reason=";".join(([result["failure_reason"]] if result["failure_reason"] else [])+errors))
    json.dumps(result, allow_nan=False)
    return result
