"""Read-only frozen R2 quality mathematics, adapted to variable-face-count POST.

Only this module and explicitly supplied details_dir are new outputs. No model,
GT reconstruction, fresh C queries, fresh F surface samples, or topology repair.
score_quality(record, processed_npz_path, details_dir) returns JSON-safe evidence.
record needs request_id, label, raw_path, raw_sha256; other registered metadata
is checked when supplied. POST is FP32 ordered C-prefix triangles at model scale.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Must be set by the CPU worker before importing NumPy. Keep caller thread budget.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np

VERSION = "r2_postprocess_frozen_quality_variable_N_v1"
REPO = Path(__file__).resolve().parents[3]
MIX_ROOT = REPO / "experiments/r2_user_mix_from_official_v1"
REGISTRATION = MIX_ROOT / "evaluation/registration.json"
OLD_ADAPTER = REPO / "r2/pilots/user_condition_finetune_v1/evaluation.py"
MIX_ADAPTER = REPO / "r2/pilots/user_mix_from_official_v1/evaluation.py"
L = float(np.sqrt(3.) * 2 / .3762)


def resolve(value):
    value = str(value)
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        return Path("/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/"))
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def digest(path):
    result = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def _old():
    name = "_r2_post_quality_readonly_old_adapter"
    if name not in sys.modules:
        # Original adapter fixes threads at two. Restore caller's environment
        # immediately; it imports no new NumPy instance or model.
        threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        spec = importlib.util.spec_from_file_location(name, OLD_ADAPTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for key, value in threads.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return sys.modules[name]


def protected_paths():
    """Source files and frozen query/input assets, not model checkpoints."""
    reg = read(REGISTRATION)
    paths = {str(REGISTRATION), str(OLD_ADAPTER), str(MIX_ADAPTER)}
    paths.update(reg["scorer_sources"])
    paths.add(reg["old_registration_path"])
    paths.update(q["path"] for q in reg["queries"])
    paths.update(q["input_path"] for q in reg["requests"])
    return sorted(str(resolve(p)) for p in paths)


def source_hashes():
    reg = read(REGISTRATION)
    if digest(MIX_ADAPTER) != reg["adapter_sha256"] or digest(OLD_ADAPTER) != reg["old_adapter_sha256"]:
        raise ValueError("Original adapter source differs from frozen registration")
    for path, expected in reg["scorer_sources"].items():
        if digest(path) != expected:
            raise ValueError("Original quality mathematics changed: " + path)
    if digest(reg["old_registration_path"]) != reg["old_registration_sha256"]:
        raise ValueError("Original underlying evaluation registration changed")
    return {**reg["scorer_sources"], str(MIX_ADAPTER): digest(MIX_ADAPTER),
            str(OLD_ADAPTER): digest(OLD_ADAPTER), str(Path(__file__).resolve()): digest(__file__)}


def definition():
    reg = read(REGISTRATION)
    return {
        "version": VERSION,
        "source_metric_version": reg["definition"]["source_metric_version"],
        "original_definition": reg["definition"]["original_definition"],
        "L": L,
        "scale_source": "Fixed canonical training cube diagonal sqrt(3)*2/.3762; unchanged from RAW",
        "stage": "POST with preserved ordered C prefix; variable actual free-face count",
        "CF_FF": "Unmodified complete independently verified original pair_scan, all C/F and unique F/F pairs",
        "denominator": "CF/FF affected-free fractions use actual POST len(F); I retains full original C2048 query weight",
        "I": "Original ConflictCov_C cross/dup union, bounds, uncertainty and low-support status",
        "queries": "Read exact frozen per-condition C2048 NPZ and verify file/array hashes; never resample",
        "small_angles": "Unmodified shape_quality; free minimum triangle angle strictly <5 degrees",
        "topology": "Not inferred here; actual indexed topology must be reported separately by caller",
        "GT": "NA_NOT_UNIQUE_GENERATIVE_TARGET", "S": "NA_NO_REGISTERED_REQUIRED_INTERFACE",
        "E": "NA_NO_EXPOSURE_TASK", "D_free": "NOT_RUN_NO_NEW_PREDICTION_QUERIES",
        "empty_free_extension": "Zero pair/area counts; free-face fractions NA with denominator0; I computed with frozen C queries and empty F",
        "new_C_queries": 0, "new_F_queries": 0, "model_calls": 0,
        "software_versions": reg["definition"]["software_versions"],
    }


def _triangles(path):
    with np.load(resolve(path), allow_pickle=False) as loaded:
        payload = {k: loaded[k].copy() for k in loaded.files}
    key = next((k for k in ("triangles", "output", "raw", "post") if k in payload), None)
    if key is None:
        if "vertices" not in payload or "faces" not in payload:
            raise ValueError("POST NPZ needs triangles/output/raw/post or vertices+faces")
        faces = payload["faces"]
        if faces.dtype.kind not in "iu" or faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("POST faces must be integer [N,3]")
        vertices = payload["vertices"]
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("POST indexed mesh invalid")
        tri = vertices[faces]
    else:
        tri = payload[key]
    if tri.ndim == 2 and tri.shape[1:] == (9,):
        tri = tri.reshape(-1, 3, 3)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or tri.dtype != np.float32 or not np.isfinite(tri).all():
        raise ValueError("POST requires finite original-model-coordinate FP32 [N,3,3]")
    if "vertices" in payload and "faces" in payload:
        indexed = payload["vertices"][payload["faces"]]
        if indexed.shape != tri.shape or indexed.dtype != tri.dtype or indexed.tobytes() != tri.tobytes():
            raise ValueError("POST indexed geometry differs from triangle payload")
    return np.ascontiguousarray(tri), payload


def _empty_scans(K):
    common = dict(status="PASS_STATIC_SCAN", tested_pairs=0, expected_full_pairs=0,
                  affected_subset=None, classification_counts={}, proper_pair_count=0,
                  proper_affected_faces=[], proper_affected_face_count=0,
                  proper_affected_face_denominator=0, proper_affected_face_fraction=None,
                  coplanar_positive_area_pairs=0, unresolved_or_degenerate_pairs=0,
                  contacts=[], independent_verification=True, continuous_collision_safety="NOT_RUN")
    return {**common, "scope": "all C/free pairs"}, {**common, "scope": "all unique free/free pairs"}


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite result must be explicitly status-coded")
    return value


def score_quality(raw_record, processed_npz_path, details_dir):
    """Score one registered output; caller owns registration, scheduling and cache.

    Returns status PASS with metric-specific NA/PARTIAL retained. Errors raise:
    caller must persist BLOCKED/FAILED receipts instead of dropping the request.
    This function writes only pairs.json, vectors.npz and shape_per_face.json in
    details_dir. It never writes into any historical experiment/source folder.
    """
    started = time.perf_counter()
    reg = read(REGISTRATION)
    sources = source_hashes()
    old = _old()
    matches = [q for q in reg["requests"] if q["request_id"] == raw_record["request_id"]]
    if len(matches) != 1 or raw_record["label"] not in reg["labels"]:
        raise ValueError("Output outside frozen48 panel")
    q = matches[0]
    for name in old.META:
        if name in raw_record:
            actual, expected = raw_record[name], q[name]
            if isinstance(expected, int):
                actual = int(actual)
            elif isinstance(expected, float):
                actual = float(actual)
            if actual != expected:
                raise ValueError("Registered metadata differs: " + name)
    if digest(raw_record["raw_path"]) != raw_record["raw_sha256"]:
        raise ValueError("Protected RAW hash mismatch")
    query = next(a for a in reg["queries"] if a["condition_id"] == q["condition_id"])
    inputs = old.read_input(q)
    reference = old.load_queries(query)
    C, K = inputs["C"], int(q["K"])
    original, _ = _triangles(raw_record["raw_path"])
    if len(original) != int(q["N"]) or not old.byte_equal(original[:K], C):
        raise ValueError("Original RAW C/N differs from frozen input")
    tri, payload = _triangles(processed_npz_path)
    if len(tri) < K or not old.byte_equal(tri[:K], C):
        raise ValueError("POST did not preserve ordered FP32 C prefix")
    if "C" in payload and not old.byte_equal(payload["C"].reshape(K, 3, 3), C):
        raise ValueError("POST C field differs from input")
    F = tri[K:]
    if "free" in payload and not old.byte_equal(payload["free"].reshape(-1, 3, 3), F):
        raise ValueError("POST free field differs from ordered suffix")
    mod = old.cpu_math()
    if len(F):
        metrics, cf, ff = mod["geometry"].scan_geometry(C, F, L)
    else:
        cf, ff = _empty_scans(K)
        metrics = {name + "_" + field: value for name in ("CF", "FF")
                   for field, value in (("proper_pair_count", 0), ("coplanar_pair_count", 0),
                                        ("proper_affected_free_fraction", None), ("unresolved_or_degenerate_pair_count", 0))}
        metrics.update(FF_illegal_pair_count=0, FF_illegal_affected_free_fraction=None,
                       free_minimum_angle_lt5_fraction=None, free_angle_p05_degrees=None,
                       free_q_gt10_fraction=None, free_q_mean=None, free_near_degenerate_count=0,
                       free_near_degenerate_fraction=None, free_exact_zero_area_count=0,
                       free_exact_zero_area_fraction=None, free_total_surface_area=0.)
    trusted = mod["control"].trusted_cf(C, F, L, cf_scan=cf)
    im, vectors = mod["control"].conflict_metrics(F, {"C": C, "L0": L}, reference, trusted)
    if abs(im["ConflictCov_C"] - im["ConflictCov_C_cross"] - im["ConflictCov_C_dup"] + im["ConflictCov_C_cross_and_dup"]) > 1e-12:
        raise ValueError("I cross/dup union identity failed")
    metrics.update(im)
    illegal = {"proper_transverse_penetration", "coplanar_positive_area_overlap"}
    cf_free = {row["pair"][1] for row in cf["contacts"] if row["classification"] in illegal}
    metrics["CF_illegal_pair_count"] = sum(row["classification"] in illegal for row in cf["contacts"])
    metrics["CF_illegal_affected_free_fraction"] = len(cf_free) / len(F) if len(F) else None
    cq = mod["shape"].shape_quality(C, L)
    fq = mod["shape"].shape_quality(F, L)
    whole = mod["shape"].shape_quality(tri, L)
    metrics.update(C_total_surface_area=cq["total_area"], C_near_degenerate_count=cq["near_degenerate_area_count"],
                   C_exact_zero_area_count=cq["exact_zero_area_count"], whole_total_surface_area=whole["total_area"],
                   whole_near_degenerate_count=whole["near_degenerate_area_count"],
                   whole_minimum_angle_lt5_fraction=whole["min_angle_lt5_fraction"])
    details = resolve(details_dir).resolve()
    repo = REPO.resolve()
    if not details.is_relative_to(repo) or details.is_relative_to(MIX_ROOT.resolve()):
        raise ValueError("Details must be in a new independent folder inside E:/MeshFlow")
    pairs_path, vectors_path = details / "pairs.json", details / "vectors.npz"
    shape_path = details / "shape_per_face.json"
    old.write(pairs_path, _clean({"CF": cf, "FF": ff}))
    old.save_npz(vectors_path, vectors)
    old.write(shape_path, _clean({"C": cq, "free": fq, "whole": whole}))
    result = {k: q[k] for k in old.META}
    result.update(
        version=VERSION, label=raw_record["label"], stage="POST", status="PASS", metrics=metrics,
        identity={"raw_path": str(resolve(raw_record["raw_path"])), "raw_sha256": raw_record["raw_sha256"],
                  "processed_path": str(resolve(processed_npz_path)), "processed_sha256": digest(processed_npz_path),
                  "input_sha256": q["input_sha256"], "query_path": query["path"], "query_sha256": query["sha256"],
                  "original_registration_path": str(REGISTRATION), "original_registration_sha256": digest(REGISTRATION),
                  "adapter_sha256": digest(__file__), "scorer_sources": sources},
        definition=definition(), source_hashes=sources,
        contracts={"C_output_bitwise": True, "finite": True, "FP32": True,
                   "actual_POST_N": len(tri), "raw_N": int(q["N"]), "actual_free_N": len(F), "K": K},
        metric_status={"ConflictCov_C": im["ConflictCov_C_status"],
                       "free_face_fractions": "PASS" if len(F) else "NA_EMPTY_FREE_DENOMINATOR",
                       "SeamCov": "NA_NO_REGISTERED_REQUIRED_INTERFACE", "ExposedRetain": "NA_NO_EXPOSURE_TASK",
                       "GT_reconstruction": "NA_NOT_UNIQUE_GENERATIVE_TARGET", "topology": "CALLER_SUPPLIED_SEPARATELY",
                       "D_free": "NOT_RUN_NO_NEW_PREDICTION_QUERIES", "human_usability": "NOT_RUN"},
        C_query_count=len(reference["arrays"]["C_points"]),
        C_query_denominator=float(reference["arrays"]["C_weights"].sum()),
        free_face_denominator=len(F), full_CF_scans=1, full_FF_scans=1, I_reuses_CF=True,
        CF_tested_pair_denominator=K * len(F), FF_tested_pair_denominator=len(F) * (len(F) - 1) // 2,
        pair_audit_path=str(pairs_path), vectors_path=str(vectors_path), shape_details_path=str(shape_path),
        files={str(p): digest(p) for p in (pairs_path, vectors_path, shape_path)},
        seconds=time.perf_counter() - started, model_calls=0, new_C_queries=0, new_F_queries=0)
    result = _clean(result)
    json.dumps(result, allow_nan=False)
    return result