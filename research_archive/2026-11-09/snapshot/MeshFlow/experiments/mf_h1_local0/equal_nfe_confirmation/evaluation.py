"""Isolated, single-thread CPU scoring of registered Equal-NFE endpoints.

The historical 54-scalar scorer is imported unchanged. Cross-condition rows
reuse its two target remainder scores and add only fixed Gamma distances.
No model, optimizer, OT, sampling, repair, or new topology audit is run here.
"""
from __future__ import annotations

import os

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments.mf_h1_local0.jedge5_diagnosis.common import (
    ROOT, EVALUATION_SHA, read_json, write_json, file_sha256,
)
from experiments.mf_h1_local0.jedge5_diagnosis.scoring import (
    REFERENCE, banks_with_context_audit, reference_evaluator, score_one,
)

IDENTITY_FIELDS = ("task_id", "kind", "model", "solver", "parent", "patch", "seed")
SOLVERS = ("Euler50", "Midpoint25", "E_quad50")


def scalar_schema():
    """Obtain the schema from the read-only scorer, without scoring geometry."""
    ref = reference_evaluator()
    blank_coverage = {str(t): None for t in ref.THRESHOLDS}
    skeleton = dict(remainder=dict(own={}), whole=dict(surface={}),
        edge_factorial_boundary=dict(surface=dict(coverage_by_bbox_fraction=blank_coverage),
            edges=dict(RMS_bbox_pct=None, p95_bbox_pct=None, zero_length_edge_count=None,
                       coverage_by_bbox_fraction=blank_coverage)))
    result = ref.scalar_metrics(skeleton)
    if len(result) != 54 or any(value is not None for value in result.values()):
        raise ValueError("Frozen 54-scalar schema changed")
    return result


def _stem(value):
    if not isinstance(value, str) or not value or any(x in value for x in ("/", "\\", "..")):
        raise ValueError("Expected a single isolated filename stem")
    return value


def _sha_array(value):
    from experiments.mf_h1_local0.native_inpainting_metrics import array_sha
    return array_sha(value)


def _source_identity():
    files = [Path(__file__), REFERENCE / "evaluation.py", REFERENCE / "edge_metric.py"]
    files.extend(ROOT / "experiments/mf_h1_local0" / filename for filename in (
        "jedge5_diagnosis/scoring.py", "tgeo_m10_evaluation.py", "a_then_fc_evaluation.py",
        "t1_gated_factorial_evaluation.py", "t1_native_confirmation_evaluation.py",
        "guided_native_geometry_audit.py", "native_patch_metrics.py", "native_patch_seams.py",
        "native_inpainting_metrics.py", "gpu_guidance_evaluation.py", "evaluate.py"))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in files}


def _normalized_area(free, length):
    """Same arithmetic as archived shape_quality.total_area, without extra audit."""
    normalized = np.asarray(free, dtype=np.float64).reshape(-1, 3, 3) / float(length)
    area = .5 * np.linalg.norm(np.cross(normalized[:, 1] - normalized[:, 0],
                                      normalized[:, 2] - normalized[:, 0]), axis=-1)
    return float(area.sum() * length * length)


def _fallback_metadata(free, length):
    triangles = np.asarray(free, dtype=np.float64).reshape(-1, 3, 3)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    n2 = np.einsum("ti,ti->t", normal, normal)
    fallback = n2 <= (2e-12 * length * length) ** 2
    return dict(distance_fallback="Degenerate triangle interiors use their original finite edges",
                fallback_triangle_count=int(fallback.sum()),
                fallback_triangle_ids=np.flatnonzero(fallback).tolist(),
                original_stable_normal_squared_threshold=float((2e-12 * length * length) ** 2))


def _foreign_boundary(free, bank, target, patch, ledger):
    """Apply unchanged boundary/edge formulas to a target's frozen Gamma only."""
    from experiments.mf_h1_local0.guided_native_geometry_audit import point_to_triangle_surface_squared
    ref = reference_evaluator()
    edge_module = sys.modules["experiments.mf_h1_local0.jedge5_diagnosis.edge_metric"]
    pre = f"parent{target}_patch{patch}_"
    points = bank["arrays"][pre + "boundary_points"]
    weights = bank["arrays"][pre + "boundary_weights"]
    length = bank["cases_by_key"][target, 12]["bbox_diagonal"]
    ledger["binding_foreign_Gamma_surface_calls_attempted"] = ledger.get("binding_foreign_Gamma_surface_calls_attempted", 0) + 1
    squared = point_to_triangle_surface_squared(points, free, length)
    if squared is None or not len(squared):
        raise ValueError("Foreign target Gamma distance unavailable; no candidates is not zero")
    distances = np.sqrt(squared)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, .95 * cumulative[-1], side="left")), len(order) - 1)
    surface = dict(rms_bbox_pct=100 * float(np.sqrt(np.sum(weights * squared))) / length,
                   p95_bbox_pct=100 * float(distances[order[position]]) / length,
                   coverage_fraction=float(weights[distances < .001 * length].sum()))
    coverage = {str(t): float(weights[distances < t * length].sum()) for t in ref.THRESHOLDS}
    edge = edge_module.gamma_to_free_edges(free, points, weights, length, ledger=ledger)
    edge_distances = np.asarray(edge["per_point_distances"], dtype=np.float64)
    violations = edge_distances + ref.DISTANCE_ORDER_TOL_BBOX * length < distances
    if bool(violations.any()):
        raise ValueError("Foreign Gamma edge distance below surface beyond frozen tolerance")
    ledger["binding_foreign_Gamma_surface_calls_returned"] = ledger.get("binding_foreign_Gamma_surface_calls_returned", 0) + 1
    return surface, dict(surface=dict(coverage_by_bbox_fraction=coverage,
                                     per_point_distances=distances.tolist()),
        edges=edge, ordering=dict(status="PASS", violations=int(violations.sum()),
            tolerance_bbox_fraction=ref.DISTANCE_ORDER_TOL_BBOX,
            minimum_edge_minus_surface_bbox=float((edge_distances - distances).min() / length)),
        fallback=_fallback_metadata(free, length))


def cross_binding_rows(raw, metrics, bank, task, ledger):
    """Target-indexed E_target(F_prediction); copied C never enters F."""
    parent, patch = int(task["parent"]), int(task["patch"])
    free = raw.reshape(112, 3, 3)[12:].astype(np.float64)
    rows, detail = {}, {}
    for target in (0, 1):
        remainder = metrics["remainder"]["cross"][target]
        common = metrics["common_unknown_R12"]["cross"][target]
        if target == parent:
            surface = metrics["guided_geometry_audit"]["boundary"]
            extension = metrics["edge_factorial_boundary"]
            ledger["binding_own_Gamma_scores_reused"] = ledger.get("binding_own_Gamma_scores_reused", 0) + 1
        else:
            surface, extension = _foreign_boundary(free, bank, target, patch, ledger)
        ledger["binding_existing_remainder_scores_reused"] = ledger.get("binding_existing_remainder_scores_reused", 0) + 1
        values = dict(remainder_symmetric_RMS_bbox_pct=remainder["symmetric_rms_bbox_pct"],
            free_to_R12_RMS_bbox_pct=remainder["predicted_to_target_rms_bbox_pct"],
            R12_to_free_RMS_bbox_pct=remainder["target_to_predicted_rms_bbox_pct"],
            fixed_R12_coverage_RMS_bbox_pct=common["rms_bbox_pct"],
            fixed_R12_coverage_within_1pct_fraction=common["fraction_target_points_within_1pct_bbox"],
            boundary_RMS_bbox_pct=surface["rms_bbox_pct"], boundary_p95_bbox_pct=surface["p95_bbox_pct"],
            boundary_coverage_fraction=surface["coverage_fraction"],
            Gamma_to_free_edges_RMS_bbox_pct=extension["edges"]["RMS_bbox_pct"],
            Gamma_to_free_edges_p95_bbox_pct=extension["edges"]["p95_bbox_pct"])
        for name, source in (("Gamma_surface", extension["surface"]), ("Gamma_to_free_edges", extension["edges"])):
            for threshold, value in source["coverage_by_bbox_fraction"].items():
                values[f"{name}_coverage_{threshold}_fraction"] = value
        pre = f"parent{target}_patch{patch}_"
        identity = dict(target_bbox_diagonal=bank["cases_by_key"][target, 12]["bbox_diagonal"],
            target_pairwise_points_sha256=_sha_array(bank["arrays"][pre + "pairwise_target_points"]),
            target_coverage_points_sha256=_sha_array(bank["arrays"][pre + "coverage_points"]),
            target_Gamma_points_sha256=_sha_array(bank["arrays"][pre + "boundary_points"]),
            target_Gamma_weights_sha256=_sha_array(bank["arrays"][pre + "boundary_weights"]),
            predicted_free_sha256=_sha_array(raw.reshape(112, 3, 3)[12:]), predicted_faces=100,
            copied_C_included=False, region="Target's own R12 complement and fixed Gamma")
        rows[str(target)] = dict(target_parent=target, prediction_parent=parent,
            source_task_id=task["task_id"], status="PASS", metrics=values, identity=identity)
        detail[str(target)] = dict(remainder=remainder, common_unknown_R12=common,
            surface=surface, extension=extension,
            fallback=_fallback_metadata(free, identity["target_bbox_diagonal"]))
    return rows, detail


def validate_endpoint(path, expected_sha, task, bank):
    """Reject incomplete/altered trajectories before endpoint metrics are called."""
    actual_sha = file_sha256(path)
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("Raw SHA256 missing or different from worker record")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    stage_key = "stage_states" if "stage_states" in arrays else "stages"
    for key in ("raw", "states", "C", stage_key):
        if key not in arrays:
            raise ValueError("Missing trajectory field: " + key)
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError("Nonfinite or non-FP32 trajectory field: " + key)
    raw = arrays["raw"].reshape(112, 3, 3)
    accepted = 26 if task["solver"] == "Midpoint25" else 51
    states = arrays["states"].reshape(-1, 112, 3, 3)
    stages = arrays[stage_key].reshape(-1, 112, 3, 3)
    expected_C = bank["cases_by_key"][int(task["parent"]), 12]["constraints"]
    if len(states) != accepted or len(stages) != 50:
        raise ValueError("Incomplete trajectory: accepted/query counts differ")
    if arrays["C"].reshape(12, 3, 3).tobytes() != expected_C.tobytes():
        raise ValueError("Registered C identity differs")
    if states[-1].tobytes() != raw.tobytes():
        raise ValueError("Raw endpoint differs from final accepted state")
    if any(x[:12].tobytes() != expected_C.tobytes() for x in np.concatenate((states, stages))):
        raise ValueError("C changed in accepted or query state")
    time_key = next((k for k in ("accepted_times_FP32", "accepted_times_control_FP64",
                               "accepted_times", "control_times", "times") if k in arrays), None)
    if time_key is None:
        raise ValueError("Accepted endpoint time evidence absent")
    times = arrays[time_key].reshape(-1)
    if (len(times) != accepted or not np.isfinite(times).all() or float(times[0]) != 0.0
            or float(times[-1]) != 1.0 or not np.all(np.diff(times) > 0)):
        raise ValueError("Trajectory did not reach registered t=1 endpoint")
    return raw, dict(status="PASS", raw_file_sha256=actual_sha,
        raw_array_sha256=_sha_array(raw), C_array_sha256=_sha_array(expected_C),
        predicted_free_sha256=_sha_array(raw[12:]), accepted_count=accepted,
        query_stage_count=50, t_final=float(times[-1]), accepted_time_field=time_key,
        known_slots=list(range(12)),
        free_slots=list(range(12, 112)), C_preserved_all_accepted_and_query_states=True)


def _registered_group(protocol, group_id):
    groups = protocol["groups"]
    if isinstance(groups, dict):
        groups = [dict(value, group_id=key) for key, value in groups.items()]
    selected = [group for group in groups if group["group_id"] == group_id]
    if len(selected) != 1:
        raise ValueError("Expected exactly one registered group: " + group_id)
    tasks = selected[0]["tasks"]
    ids = [task["task_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate registered task ID")
    for task in tasks:
        _stem(task["task_id"])
        if task["kind"] == "formal" and (task["model"] not in ("B", "J") or task["solver"] not in SOLVERS):
            raise ValueError("Unknown formal model/solver")
    return [task for task in tasks if task["kind"] == "formal"]


def score_group(out, group_id):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(out).resolve()
    group_id = _stem(group_id)
    result_path = out / "scores" / (group_id + ".json")
    if result_path.exists():
        raise FileExistsError("Scoring attempts are retained; refusing to overwrite " + str(result_path))
    tick, cpu_tick = time.perf_counter(), time.process_time()
    protocol_path = out / "registration/protocol.json"
    tasks = _registered_group(read_json(protocol_path), group_id)
    if not tasks:
        raise ValueError("No formal tasks in registered group; regressions/repeats are not endpoint-scored")
    worker_path = out / "workers" / (group_id + ".json")
    worker = read_json(worker_path)
    worker_rows = worker["rows"]
    indexed = {row["task_id"]: row for row in worker_rows}
    if len(indexed) != len(worker_rows):
        raise ValueError("Duplicate worker task row")
    result = dict(schema="equal_nfe_cpu_scores_v1", status="RUNNING", group_id=group_id,
        rows=[], requested_count=len(tasks), ledger={}, model_forwards=0, OT_calls=0,
        additional_intersection_scans_for_binding=0, topology_certification="NOT_RUN",
        exact_predicate_certification="NOT_RUN", CPU_threads=1,
        protocol_sha256=file_sha256(protocol_path), worker_sha256=file_sha256(worker_path),
        evaluation_inputs_sha256=EVALUATION_SHA, metric_source_sha256=_source_identity())
    write_json(result_path, result)
    banks_tick = time.perf_counter()
    banks = banks_with_context_audit(out)
    result["bank_load_and_identity_seconds"] = time.perf_counter() - banks_tick
    schema = scalar_schema()
    for task in tasks:
        task_id = task["task_id"]
        detail_path = out / "scores" / (task_id + ".json")
        if detail_path.exists():
            raise FileExistsError(detail_path)
        row = dict(task, status="NOT_SCORED", sampling_status=None, scalars54=dict(schema),
            derived={}, cross_binding={}, score_path=str(detail_path.relative_to(out)))
        detail = dict(row=row, metrics=None, cross_binding_details=None, ledger={})
        row_tick, row_cpu = time.perf_counter(), time.process_time()
        stage = "worker_identity"
        try:
            source = indexed.get(task_id)
            if source is None:
                row["status"] = "NOT_SCORED_MISSING_WORKER"
            else:
                row["sampling_status"] = source.get("status")
                row["sampling"] = source
                for key in IDENTITY_FIELDS:
                    if source.get(key) != task.get(key):
                        raise ValueError("Worker task identity differs: " + key)
                if source.get("status") != "PASS":
                    row["status"] = "NOT_SCORED_SAMPLING_FAILURE"
                else:
                    raw_path = reference_evaluator()._path(source.get("path") or source.get("raw_path") or f"raw/{task_id}.npz")
                    if not raw_path.is_absolute():
                        raw_path = out / raw_path
                    raw_path = raw_path.resolve()
                    raw_path.relative_to((out / "raw").resolve())
                    row["raw_path"] = str(raw_path.relative_to(out))
                    row["raw_sha256"] = source.get("sha256") or source.get("raw_sha256")
                    stage = "endpoint_validation"
                    raw, row["raw_identity"] = validate_endpoint(raw_path, row["raw_sha256"], task, banks[int(task["patch"])])
                    stage = "original_54_metrics"
                    detail["ledger"]["original_endpoint_scores_attempted"] = 1
                    metrics, scalars = score_one(raw, banks[int(task["patch"])], int(task["parent"]), int(task["patch"]), detail["ledger"])
                    detail["metrics"] = metrics
                    if set(scalars) != set(schema):
                        raise ValueError("Original scalar schema changed")
                    row["scalars54"] = scalars
                    detail["ledger"]["original_endpoint_scores_returned"] = 1
                    bank = banks[int(task["patch"])]
                    case = bank["cases_by_key"][int(task["parent"]), 12]
                    area = metrics["t1_native_confirmation"]["free_shape_quality"]["total_area"]
                    gt_area = _normalized_area(case["free_target"], case["bbox_diagonal"])
                    row["derived"] = dict(free_total_area=area, GT_free_total_area=gt_area,
                        free_area_over_GT_free_area=area / gt_area if gt_area > 0 else None,
                        formula="Original shape_quality FP64 bbox-normalized area arithmetic; GT denominator uses own R12",
                        role="Offline descriptive analysis only; no universal improvement direction")
                    detail["ledger"]["GT_free_area_calculations"] = 1
                    stage = "cross_binding_distances"
                    row["cross_binding"], detail["cross_binding_details"] = cross_binding_rows(raw, metrics, bank, task, detail["ledger"])
                    row["status"] = "PASS"
                    row["geometry_status"] = metrics["t1_native_confirmation"]["EMBEDDED_GEOMETRY_VALIDITY_STATUS"]
                    row["unresolved_pair_count"] = sum(view["verified"]["unresolved_count"] for view in metrics["t1_native_confirmation"]["detector_views"].values())
        except Exception as exc:
            row.update(status="SCORE_FAILED", failure_stage=stage, error=repr(exc), traceback=traceback.format_exc())
        finally:
            row["seconds"] = time.perf_counter() - row_tick
            row["cpu_process_seconds"] = time.process_time() - row_cpu
            for key, value in detail["ledger"].items():
                result["ledger"][key] = result["ledger"].get(key, 0) + value
            detail["row"] = row
            write_json(detail_path, detail)
            result["rows"].append(row)
            result["completed"] = len(result["rows"])
            result["seconds"] = time.perf_counter() - tick
            result["cpu_process_seconds"] = time.process_time() - cpu_tick
            write_json(result_path, result)
            print("score", group_id, len(result["rows"]), "/", len(tasks), task_id, row["status"], flush=True)
    result["status"] = "PASS" if all(row["status"] == "PASS" for row in result["rows"]) else "PARTIAL_OR_FAILED"
    result["success_count"] = sum(row["status"] == "PASS" for row in result["rows"])
    result["failure_or_missing_count"] = len(tasks) - result["success_count"]
    result["seconds"] = time.perf_counter() - tick
    result["cpu_process_seconds"] = time.process_time() - cpu_tick
    write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args(argv)
    result = score_group(args.out, args.group)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
