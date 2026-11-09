"""CPU-only analysis of four frozen two-triangle START control outputs.

No torch/model imports, random draws, repair, welding, or mesh reconstruction.
All widths describe stored free triangle corners, never the C-inclusive mesh.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


VERSION = "chair_two_triangle_control_analysis_v1"
METRICS = (
    "free_x_bbox_width", "free_x_q05_q95_width", "seat_band_free_x_width",
    "boundary_to_free_median", "boundary_to_free_p95",
    "boundary_to_free_le_003_fraction",
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bytes_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(out, path):
    p = Path(path)
    return p if p.is_absolute() else out / p


def point_triangle_distances(points, triangles):
    """Minimum Euclidean distances to actual triangle interiors and edges.

    All operations are FP64. An interior orthogonal plane projection is a
    candidate only when its barycentric coordinates lie in the closed triangle;
    otherwise the three clamped segment projections determine the distance.
    Exactly degenerate triangles contribute their segments/vertices, not NaN.
    Tiny nonzero triangles are retained rather than repaired or thresholded away.
    """
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError("Expected points[Q,3] and triangles[F,3,3]")
    if not len(t) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Distances require nonempty finite triangles and points")
    answer = np.full(len(p), np.inf, dtype=np.float64)
    for start in range(0, len(t), 128):
        tri = t[start:start + 128]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        v0, v1 = b - a, c - a
        n = np.cross(v0, v1)
        n2 = np.einsum("fi,fi->f", n, n)
        ap = p[:, None, :] - a[None, :, :]
        signed = np.einsum("qfi,fi->qf", ap, n)
        # Cross-product barycentric formulas avoid subtracting nearly equal
        # Gram products. Clamped edge distances remain candidates everywhere.
        u_num = np.einsum("qfi,fi->qf", np.cross(ap, v1[None]), n)
        v_num = np.einsum("qfi,fi->qf", np.cross(v0[None], ap), n)
        regular = n2 > 0
        u = np.divide(u_num, n2[None], out=np.zeros_like(u_num), where=regular[None])
        v = np.divide(v_num, n2[None], out=np.zeros_like(v_num), where=regular[None])
        inside = regular[None] & (u >= 0) & (v >= 0) & (u + v <= 1)
        best = np.divide(signed * signed, n2[None], out=np.full_like(signed, np.inf), where=inside)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x
            edge2 = np.einsum("fi,fi->f", edge, edge)
            xp = p[:, None, :] - x[None]
            dot = np.einsum("qfi,fi->qf", xp, edge)
            fraction = np.divide(dot, edge2[None], out=np.zeros_like(dot), where=edge2[None] > 0)
            fraction = np.clip(fraction, 0, 1)
            residual = xp - fraction[:, :, None] * edge[None]
            d2 = np.einsum("qfi,qfi->qf", residual, residual)
            best = np.minimum(best, d2)
        answer = np.minimum(answer, best.min(axis=1))
    return np.sqrt(np.maximum(answer, 0))


def rectangle_boundary(condition, count=256):
    """Uniform arclength on the outer rectangle, excluding its diagonal."""
    corners = np.asarray(condition, dtype=np.float64).reshape(-1, 3)
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    zmin, zmax = corners[:, 2].min(), corners[:, 2].max()
    if not np.all(corners[:, 1] == corners[0, 1]):
        raise ValueError("The registered C must be horizontal")
    width, depth = xmax - xmin, zmax - zmin
    if width <= 0 or depth <= 0:
        raise ValueError("The registered C rectangle is degenerate")
    y = corners[0, 1]
    vertices = np.array([[xmin, y, zmin], [xmax, y, zmin], [xmax, y, zmax], [xmin, y, zmax]])
    lengths = np.array([width, depth, width, depth])
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    samples = np.arange(count, dtype=np.float64) * cumulative[-1] / count
    index = np.searchsorted(cumulative[1:], samples, side="right")
    fraction = (samples - cumulative[index]) / lengths[index]
    result = vertices[index] + fraction[:, None] * (vertices[(index + 1) % 4] - vertices[index])
    return result


def definitions():
    return {
        "version": VERSION,
        "coordinates": "Original model coordinates; no re-normalization, registration, repair, welding, or duplicate-corner removal.",
        "free_domain": "Stored output[K:N], flattened to 3*(N-K) triangle corners. C is excluded from every response-width metric.",
        "free_x_bbox_width": "max(x)-min(x) across all stored free corners.",
        "free_x_q05_q95_width": "numpy.quantile(x,0.95,method='linear') minus quantile(x,0.05,method='linear'); unweighted triangle corners, not area-weighted surface samples.",
        "seat_band_free_x_width": "max(x)-min(x) across free corners with abs(y-seat_y)<=0.25; NA when fewer than3 stored corners qualify. Repeated corners retain their original multiplicity.",
        "boundary_samples": "256 uniform-arclength points around the outer rectangle of the actual FP32 C: begin at(xmin,y,zmin), follow+x,+z,-x,-z, omit repeated end point. Internal shared diagonal excluded. Wide/narrow rectangles each use their own perimeter.",
        "boundary_distance": "FP64 Euclidean minimum to all original free triangle interiors/edges/vertices; no vertex-only approximation. Exactly degenerate faces contribute their segments/vertices.",
        "boundary_to_free_median": "Median of256 distances in original coordinate units.",
        "boundary_to_free_p95": "Linear-interpolated95th percentile of256 distances in original coordinate units.",
        "boundary_to_free_le_003_fraction": "Count(distance<=0.03)/256, inclusive threshold in original coordinate units. A proximity indicator, not watertightness or topological attachment.",
        "C_bitwise_all51": "Independently compare every saved path[s,:K] with the input C using identical dtype,shape,bytes for all51 states. Also preserve the official sampler audit as distinct evidence.",
        "paired_response": "Same registered seed and exact full initial_gaussian bytes/shape/dtype required. Delta=wide-minus-narrow. Width response slope=delta_width/(actual_C_width_wide-actual_C_width_narrow). No target1.0 success threshold is imposed.",
        "limits": "Two paired seeds are descriptive only. No hypothesis significance test, usability judgement, GT comparison, or claim that larger free width is better.",
    }


def analyze_one(out, request):
    request_id = request["id"]
    output_dir = resolve(out, request["output_dir"])
    raw_path, run_path = output_dir / "raw.npz", output_dir / "run.json"
    row = {"id": request_id, "condition": request["condition"], "seed": request["seed"], "status": "BLOCKED", "N": request["N"], "K": request["K"], "requested_C_width": request["width"], "depth": request["depth"], "seat_y": request["seat_y"], "raw_path": str(raw_path), "run_path": str(run_path)}
    missing = [str(p) for p in [raw_path, run_path, resolve(out, request["C"]), resolve(out, request["input_npz"])] if not p.is_file()]
    if missing:
        row["missing"] = missing
        return row, None
    run_record = json.loads(run_path.read_text(encoding="utf-8-sig"))
    condition = np.load(resolve(out, request["C"]), allow_pickle=False)
    with np.load(resolve(out, request["input_npz"]), allow_pickle=False) as inp:
        registered_noise = inp["initial_gaussian"]
    with np.load(raw_path, allow_pickle=False) as archive:
        output, path, noise, stored_C = (archive[k] for k in ["output", "path", "initial_gaussian", "C"])
    n, k = int(request["N"]), int(request["K"])
    expected_C = condition.reshape(k, 9)
    expected_shapes = output.shape == (n, 9) and path.shape == (51, n, 9) and noise.shape == (1, n, 9)
    if not expected_shapes:
        row.update(status="FAIL", error="Unexpected saved shape", output_shape=list(output.shape), path_shape=list(path.shape), noise_shape=list(noise.shape))
        return row, None
    known_checks = [same_bytes(path[s, :k], expected_C) for s in range(51)]
    sampling = run_record.get("sampling", {})
    normals = np.cross(condition[:, 1] - condition[:, 0], condition[:, 2] - condition[:, 0])
    actual_C_width = float(np.ptp(condition[:, :, 0]))
    checks = {
        "run_status_PASS": run_record.get("status") == "PASS",
        "raw_finite": bool(np.isfinite(output).all() and np.isfinite(path).all()),
        "input_C_is_FP32_K2": condition.dtype == np.float32 and condition.shape == (2, 3, 3),
        "C_positive_Y_normals": bool((normals[:, 1] > 0).all() and (normals[:, [0, 2]] == 0).all()),
        "saved_C_bitwise_input": same_bytes(stored_C, condition),
        "output_C_bitwise_input": same_bytes(output[:k], expected_C),
        "C_bitwise_all51": all(known_checks),
        "output_bitwise_final_state": same_bytes(output, path[-1]),
        "noise_bitwise_registered": same_bytes(noise, registered_noise),
        "noise_sha256_registered": bytes_hash(noise) == request["noise_sha256"],
        "official_known_all51_bitwise": sampling.get("known_all51_bitwise") is True,
    }
    row.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks, C_known_state_checks=51, C_bitwise_all51=all(known_checks), C_state_checks=known_checks, official_known_all51_bitwise=sampling.get("known_all51_bitwise"), official_known_state_checks=sampling.get("known_state_checks"), actual_C_width=actual_C_width, noise_sha256=bytes_hash(noise), raw_sha256=digest(raw_path), run_sha256=digest(run_path), C_file_sha256=digest(resolve(out, request["C"])), input_npz_sha256=digest(resolve(out, request["input_npz"])))
    if not checks["raw_finite"]:
        row["error"] = "Nonfinite stored output/path; metrics unavailable"
        return row, noise
    free = output[k:].reshape(-1, 3, 3)
    vertices = free.reshape(-1, 3).astype(np.float64)
    selected = vertices[np.abs(vertices[:, 1] - float(request["seat_y"])) <= .25]
    boundary = rectangle_boundary(condition)
    distances = point_triangle_distances(boundary, free)
    row.update(free_faces=len(free), free_corner_count=len(vertices), free_x_bbox_width=float(np.ptp(vertices[:, 0])), free_x_q05_q95_width=float(np.quantile(vertices[:, 0], .95, method="linear") - np.quantile(vertices[:, 0], .05, method="linear")), seat_band_free_corner_count=len(selected), seat_band_free_x_width=float(np.ptp(selected[:, 0])) if len(selected) >= 3 else None, seat_band_status="PASS" if len(selected) >= 3 else "NA_INSUFFICIENT_CORNERS", boundary_point_count=len(boundary), boundary_to_free_median=float(np.median(distances)), boundary_to_free_p95=float(np.quantile(distances, .95, method="linear")), boundary_to_free_le_003_fraction=float(np.mean(distances <= .03)))
    np.savez_compressed(out / "analysis" / (request_id + "_boundary_distances.npz"), boundary_points=boundary, boundary_to_free_distance=distances)
    return row, noise


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run(out):
    out = Path(out).resolve()
    started = time.perf_counter()
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8-sig"))
    requests = plan["requests"]
    if len(requests) != 4 or len({r["id"] for r in requests}) != 4:
        raise ValueError("Expected exactly four unique frozen requests")
    rows, noises = [], {}
    for request in requests:
        row, noise = analyze_one(out, request)
        rows.append(row)
        noises[request["id"]] = noise
    pairs = []
    for seed in sorted({r["seed"] for r in requests}):
        selected = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["requested_C_width"])
        if len(selected) != 2:
            pairs.append({"seed": seed, "status": "BLOCKED", "reason": "Expected exactly two conditions for this seed"})
            continue
        low, high = selected
        a, b = noises[low["id"]], noises[high["id"]]
        matched = a is not None and b is not None and same_bytes(a, b)
        valid = matched and low["status"] == high["status"] == "PASS"
        delta_C = high.get("actual_C_width", high["requested_C_width"]) - low.get("actual_C_width", low["requested_C_width"])
        pair = {"seed": seed, "narrow_id": low["id"], "wide_id": high["id"], "status": "PASS" if valid else "BLOCKED", "same_initial_gaussian_bitwise": matched, "delta_C_width": delta_C, "noise_sha256": low.get("noise_sha256") if matched else None}
        for metric in METRICS:
            x, y = low.get(metric), high.get(metric)
            difference = y - x if valid and x is not None and y is not None else None
            pair[metric + "_narrow"] = x
            pair[metric + "_wide"] = y
            pair[metric + "_delta_wide_minus_narrow"] = difference
            if "width" in metric:
                pair[metric + "_response_per_C_width"] = difference / delta_C if difference is not None and delta_C > 0 else None
        pairs.append(pair)
    complete = all(r["status"] == "PASS" for r in rows) and len(pairs) == 2 and all(r["status"] == "PASS" for r in pairs)
    summary = {"version": VERSION, "status": "PASS" if complete else "BLOCKED", "plan_sha256": digest(out / "plan.json"), "analysis_source_sha256": digest(__file__), "outputs": len(rows), "passed_outputs": sum(r["status"] == "PASS" for r in rows), "paired_seeds": len(pairs), "passed_pairs": sum(r["status"] == "PASS" for r in pairs), "definitions": definitions(), "per_output": rows, "paired_responses": pairs, "seconds": time.perf_counter() - started, "model_calls": 0, "sampling_calls": 0, "new_random_draws": 0}
    write_json(analysis / "metric_definitions.json", definitions())
    write_json(analysis / "per_output.json", rows)
    write_json(analysis / "paired_response.json", pairs)
    write_csv(analysis / "per_output.csv", rows)
    write_csv(analysis / "paired_response.csv", pairs)
    write_json(analysis / "analysis_summary.json", summary)
    return summary


def self_test():
    tri = np.array([[[0., 0, 0], [2, 0, 0], [0, 2, 0]]])
    points = np.array([[.25, .25, 3.], [1., 1., 0.], [2., 2., 0.], [-1., -1., 0.], [3., 0., 0.]])
    expected = np.array([3., 0., math.sqrt(2), math.sqrt(2), 1.])
    measured = point_triangle_distances(points, tri)
    degenerate = np.array([[[0., 0, 0], [1., 0, 0], [2., 0, 0]], [[4., 0, 0], [4., 0, 0], [4., 0, 0]]])
    measured_degenerate = point_triangle_distances(np.array([[1., 2, 0], [4., 0, 3]]), degenerate)
    checks = {"triangle_interior_edge_vertex_distances": bool(np.allclose(measured, expected, atol=1e-14, rtol=0)), "degenerate_segments_and_points": bool(np.allclose(measured_degenerate, [2., 3.], atol=1e-14, rtol=0))}
    C = np.array([[[-1., 0, -1], [1., 0, 1], [1., 0, -1]], [[-1., 0, -1], [-1., 0, 1], [1., 0, 1]]], dtype=np.float32)
    boundary = rectangle_boundary(C)
    checks["closed_rectangle_256_unique_points"] = len(np.unique(boundary, axis=0)) == 256
    checks["boundary_to_covering_triangles_zero"] = bool(np.max(point_triangle_distances(boundary, C)) == 0)
    checks["byte_comparison_catches_signed_zero"] = not same_bytes(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32))
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "PASS", "checks": checks, "actual": measured.tolist(), "expected": expected.tolist(), "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        write_json(args.out / "analysis" / "analysis_self_test.json", result)
    else:
        result = run(args.out)
    print(json.dumps({k: result[k] for k in ["status", "outputs", "passed_outputs", "paired_seeds", "passed_pairs", "seconds"] if k in result}, indent=2))
