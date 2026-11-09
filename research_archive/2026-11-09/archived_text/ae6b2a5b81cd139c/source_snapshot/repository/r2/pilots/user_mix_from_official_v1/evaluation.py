"""USER-only fixed-input evaluation for the single official-init R2 USER mix arm.

Only sample may import a generator, and it requires a controller-owned dispatch.
The 12 inputs, six C-query banks and 36 historical RAWs are read-only reuse.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import html
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

VERSION = "r2_user_mix_from_official_evaluation_v1"
ROOT_NAME = "r2_user_mix_from_official_v1"
NEW_LABEL = "R2_USER_MIX"
HISTORICAL_LABELS = ("R2_FROZEN", "R2_CONT_OLD", "R2_CONT_USER")
LABELS = HISTORICAL_LABELS + (NEW_LABEL,)
R2 = Path(__file__).resolve().parents[2]
REPO = R2.parent
OLD_ADAPTER = R2 / "pilots/user_condition_finetune_v1/evaluation.py"
_spec = importlib.util.spec_from_file_location("_r2_mix_readonly_old_evaluation", OLD_ADAPTER)
_old = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _old
_spec.loader.exec_module(_old)

# These functions have no experiment-root dependency and do not import torch.
resolve, digest, read, arrays = _old.resolve, _old.digest, _old.read, _old.arrays
write, save_npz, write_csv = _old.write, _old.save_npz, _old.write_csv
array_sha, byte_equal = _old.array_sha, _old.byte_equal
source_module, cpu_math = _old.source_module, _old.cpu_math
scorer_sources = _old.scorer_sources
read_input, verify_output, load_queries = _old.read_input, _old.verify_output, _old.load_queries
CellBudget = _old.CellBudget
META, L = _old.META, _old.L
HISTORY = resolve("E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1")
DEFAULT_OUT = REPO / "experiments" / ROOT_NAME


def root(path):
    path = resolve(path).resolve()
    if path != DEFAULT_OUT.resolve():
        raise ValueError("This adapter writes only the registered experiment root")
    return path


def definition():
    value = copy.deepcopy(_old.definition())
    value.update(version=VERSION, aggregation="NEWDEV only: average two noises within each condition, then equal six parents; seat/back and tier are descriptive subgroups",
                 raw_geometry_scores_recomputed_uniformly=12,
                 historical_geometry_scores_verified_reused=36,
                 no_new_C_queries=True, no_new_inference_Gaussian=True,
                 no_OLDREG=True, historical_comparisons="Descriptive: old training populations and update totals differ; not a controlled sole-factor attribution",
                 semantic_use="AI_PROXY visual review separately identified; no objective semantic-use score is inferred from C copy, CF/FF or I")
    return value


def _historical_score_path(label, request_id):
    return HISTORY / "evaluation/scores" / label / (request_id + ".json")


def _verify_historical_score(q, receipt, query, registration, path):
    score = read(path)
    expected = {"version": _old.VERSION, "registration_sha256": digest(HISTORY / "evaluation/registration.json"),
                "label": receipt["label"], "request_id": q["request_id"], "raw_sha256": receipt["raw_sha256"],
                "input_sha256": q["input_sha256"], "query_sha256": query["sha256"],
                "scorer_sources": registration["scorer_sources"]}
    if score["identity"] != expected or score["status"] != "PASS":
        raise ValueError("Historical score identity/status mismatch: " + str(path))
    for field in META:
        if score[field] != q[field]:
            raise ValueError("Historical score request metadata changed: " + field)
    if score["C_query_count"] != 2048 or score["free_face_denominator"] != q["N"] - q["K"]:
        raise ValueError("Historical denominator mismatch")
    if not score.get("files") or not score.get("vectors_path") or not score.get("pair_audit_path"):
        raise ValueError("Historical score details missing")
    for filename, sha in score["files"].items():
        if digest(filename) != sha:
            raise ValueError("Historical per-point/pair detail changed")
    m = score["metrics"]
    if abs(m["ConflictCov_C"] - m["ConflictCov_C_cross"] - m["ConflictCov_C_dup"] + m["ConflictCov_C_cross_and_dup"]) > 1e-12:
        raise ValueError("Historical I union identity")
    return score


def prepare(out=DEFAULT_OUT):
    out = root(out)
    destination = out / "evaluation/registration.json"
    if destination.exists():
        return load_registration(out)
    if any((out / "evaluation/outputs").glob("*/*/calls.jsonl")):
        raise ValueError("Registration must precede every new rollout")
    previous_path = HISTORY / "evaluation/registration.json"
    previous = read(previous_path)
    if previous["adapter_sha256"] != digest(OLD_ADAPTER) or previous["scorer_sources"] != scorer_sources():
        raise ValueError("Read-only original scoring source drift")
    if previous["definition"] != _old.definition():
        raise ValueError("Original scoring definition/software environment drift")
    requests = sorted((copy.deepcopy(q) for q in previous["requests"] if q["panel"] == "NEWDEV"), key=lambda q: q["request_id"])
    ids = {q["condition_id"] for q in requests}
    queries = [copy.deepcopy(q) for q in previous["queries"] if q["condition_id"] in ids]
    if len(requests) != 12 or len(ids) != 6 or len(queries) != 6 or len({q["parent_id"] for q in requests}) != 6:
        raise ValueError("Exactly six existing USER conditions and two noises required")
    if collections.Counter(q["region_type"] for q in requests) != {"seat": 4, "back": 8}:
        raise ValueError("Registered 2-seat/4-back panel changed")
    for q in requests:
        read_input(q)
    by_query = {q["condition_id"]: q for q in queries}
    for q in queries:
        load_queries(q)
        pair = [r for r in requests if r["condition_id"] == q["condition_id"]]
        if len(pair) != 2 or {r["candidate"] for r in pair} != {0, 1} or len({r["C_sha256"] for r in pair}) != 1:
            raise ValueError("Broken frozen noise pairing")
    historical = []
    checkpoints = {}
    for label in HISTORICAL_LABELS:
        binding_path = HISTORY / "evaluation/checkpoints" / (label + ".json")
        binding = read(binding_path)
        checkpoints[label] = binding
        for q in requests:
            receipt_path = HISTORY / "evaluation/outputs" / label / q["request_id"] / "receipt.json"
            receipt = read(receipt_path)
            _, facts = verify_output(q, receipt, binding["sha256"])
            score_path = _historical_score_path(label, q["request_id"])
            score = _verify_historical_score(q, receipt, by_query[q["condition_id"]], previous, score_path)
            if score["contracts"] != facts:
                raise ValueError("Historical RAW versus score contract mismatch")
            reused = copy.deepcopy(receipt)
            reused.update(status="VERIFIED_CACHE", generation_status="VERIFIED_HISTORICAL_RAW_REUSE", new_rollouts=0,
                          historical_receipt_path=str(receipt_path), historical_receipt_sha256=digest(receipt_path),
                          historical_score_path=str(score_path), historical_score_sha256=digest(score_path), contracts=facts)
            write(out / "evaluation/outputs" / label / q["request_id"] / "receipt.json", reused)
            historical.append({"label": label, "request_id": q["request_id"], "receipt_path": str(receipt_path),
                               "receipt_sha256": digest(receipt_path), "score_path": str(score_path), "score_sha256": digest(score_path)})
    for path, sha in scorer_sources().items():
        target = out / "source_external_snapshot" / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha:
            raise ValueError("Read-only scorer snapshot mismatch")
        if not target.exists():
            shutil.copyfile(path, target)
    registration = {"version": VERSION, "status": "FROZEN", "adapter_sha256": digest(__file__),
                    "old_adapter_sha256": digest(OLD_ADAPTER), "old_registration_path": str(previous_path),
                    "old_registration_sha256": digest(previous_path), "scorer_sources": scorer_sources(),
                    "render_helper_sha256": digest(R2 / "data_tools/user_conditions_v1/render.py"),
                    "sampling_sha256": digest(R2 / "src/meshflow_control/sampling.py"),
                    "definition": definition(), "labels": list(LABELS), "requests": requests, "queries": queries,
                    "historical_scores": historical, "historical_checkpoints": checkpoints,
                    "budget": {"logical_outputs": 48, "new_rollouts": 12, "sampling_forwards": 600, "historical_raw_reuses": 36,
                               "new_C_query_arrays": 0, "new_Gaussian_arrays": 0, "new_OLDREG_rollouts": 0, "PNG_count": 24},
                    "query_count_total": 6 * 2048, "model_calls": 0,
                    "panel_status": "REPEATED_DEVELOPMENT_PANEL_NOT_UNSEEN_TEST", "no_unique_GT_ranking": True}
    write(destination, registration)
    write(out / "evaluation/reuse_receipt.json", {"status": "COMPLETE", "historical_outputs": 36, "historical_scores": 36,
                                               "new_C_queries": 0, "new_Gaussians": 0, "model_calls": 0, "rows": historical})
    return registration


def load_registration(out):
    out = root(out)
    reg = read(out / "evaluation/registration.json")
    checks = {"version": VERSION, "adapter_sha256": digest(__file__), "old_adapter_sha256": digest(OLD_ADAPTER),
              "scorer_sources": scorer_sources(), "definition": definition(), "labels": list(LABELS),
              "render_helper_sha256": digest(R2 / "data_tools/user_conditions_v1/render.py"),
              "sampling_sha256": digest(R2 / "src/meshflow_control/sampling.py")}
    for key, value in checks.items():
        if reg[key] != value:
            raise ValueError("Evaluation frozen identity changed: " + key)
    if digest(reg["old_registration_path"]) != reg["old_registration_sha256"] or len(reg["requests"]) != 12 or len(reg["queries"]) != 6:
        raise ValueError("Frozen source panel changed")
    for q in reg["requests"]:
        read_input(q)
    for q in reg["queries"]:
        load_queries(q)
    return reg


def bind_checkpoint(out, label, path):
    out = root(out)
    reg = load_registration(out)
    path = resolve(path).resolve()
    if label != NEW_LABEL or path != (out / "inference" / (NEW_LABEL + "_generator_step6000.pt")).resolve():
        raise ValueError("Only registered new endpoint may be bound for model calls")
    # Hash binding loads no model and performs no forward. The trusted trainer
    # validates tensor/config/protocol identity on its sole sampling load.
    value = {"label": label, "path": str(path), "sha256": digest(path), "global_step": 6000,
             "evaluation_registration_sha256": digest(out / "evaluation/registration.json"),
             "load_contract": "trainer.load_generator validates arm/global_step/schema/state identity before first forward"}
    write(out / "evaluation/checkpoints" / (label + ".json"), value)
    return value


def sample_job(out, label=NEW_LABEL, attempt="initial", controller_owned=False):
    if not controller_owned:
        raise ValueError("Only explicit resource-aware root dispatcher may own GPU")
    if label != NEW_LABEL or attempt != "initial":
        raise ValueError("Only one new arm and one initial dispatch are registered; no implicit retries")
    out = root(out)
    reg = load_registration(out)
    binding = read(out / "evaluation/checkpoints" / (label + ".json"))
    if binding["evaluation_registration_sha256"] != digest(out / "evaluation/registration.json") or digest(binding["path"]) != binding["sha256"]:
        raise ValueError("Model binding changed")
    folder = out / "evaluation/sampling" / label
    folder.mkdir(parents=True, exist_ok=True)
    dispatch = folder / "initial_dispatch.json"
    # A complete cache may be inspected without reopening a dispatch or loading a model.
    receipts = [out / "evaluation/outputs" / label / q["request_id"] / "receipt.json" for q in reg["requests"]]
    if all(p.exists() for p in receipts):
        for q, p in zip(reg["requests"], receipts):
            verify_output(q, read(p), binding["sha256"])
        return {"status": "VERIFIED_CACHE", "label": label, "registered": 12, "completed": 12, "new_rollouts": 0, "model_calls": 0}
    with dispatch.open("x", encoding="utf8") as f:
        json.dump({"attempt": attempt, "binding": binding, "pid": os.getpid(), "started": time.time()}, f)
    done = []
    started = time.perf_counter()
    try:
        if any(p.exists() or p.with_name("calls.jsonl").exists() for p in receipts):
            raise ValueError("Partially attempted sampling requires an explicit audited recovery decision")
        sys.path.insert(0, str(R2 / "src"))
        from meshflow_control import runtime
        import torch
        torch.set_num_threads(2)
        runtime.configure_stable_runtime()
        from meshflow_control.sampling import sample
        trainer = source_module("_r2_mix_evaluation_trainer", Path(__file__).with_name("trainer.py"))
        model, identity = trainer.load_generator(resolve(binding["path"]), device="cuda", expected_sha256=binding["sha256"])
        if identity["arm"] != label or identity["global_step"] != 6000:
            raise ValueError("Wrong official-init training endpoint")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        write(folder / "generator_identity.json", identity)
        for q in reg["requests"]:
            b = read_input(q)
            dest = out / "evaluation/outputs" / label / q["request_id"]
            context = {"label": label, "request_id": q["request_id"], "attempt": attempt, "phase": "USER_ONLY_EVAL",
                       "input_sha256": q["input_sha256"], "checkpoint_sha256": binding["sha256"]}
            budget = CellBudget(dest / "calls.jsonl", context)
            _, _, audit = sample(model, b["C"], q["N"], seed=q.get("noise_seed"), z=b["z"], budget=budget, out=dest, context=context)
            receipt = copy.deepcopy(q)
            receipt.update(label=label, status="COMPLETE", checkpoint_sha256=binding["sha256"], generation_status="NEW_EULER50", new_rollouts=1, attempt=attempt)
            for name, filename in (("raw", "raw.npz"), ("ply", "raw.ply"), ("result", "run.json")):
                receipt[name + "_path"] = str(dest / filename)
                receipt[name + "_sha256"] = digest(dest / filename)
            _, facts = verify_output(q, receipt, binding["sha256"])
            receipt["contracts"] = facts
            write(dest / "receipt.json", receipt)
            done.append(receipt)
            write(folder / "progress.json", {"status": "RUNNING", "label": label, "completed": len(done), "registered": 12,
                                            "seconds": time.perf_counter() - started, "cuda_peak_allocated_bytes": audit["cuda_peak_allocated_bytes"],
                                            "cuda_peak_reserved_bytes": audit["cuda_peak_reserved_bytes"], "peak_scope": audit["peak_scope"]}, freeze=False)
        result = {"status": "COMPLETE", "label": label, "rows": done, "registered": 12, "new_rollouts": 12,
                  "sampling_forwards": 600, "seconds": time.perf_counter() - started}
        write(folder / "completed.json", result)
        return result
    except BaseException as exc:
        write(folder / "initial_failure.json", {"status": "OOM" if "out of memory" in str(exc).lower() else "FAILED",
                                              "label": label, "attempt": attempt, "completed": len(done), "error": repr(exc),
                                              "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started})
        raise


def collect_outputs(out, require_complete=True):
    out = root(out)
    reg = load_registration(out)
    rows, missing = [], []
    for label in LABELS:
        for q in reg["requests"]:
            path = out / "evaluation/outputs" / label / q["request_id"] / "receipt.json"
            if not path.exists():
                missing.append({"label": label, "request_id": q["request_id"]})
                continue
            receipt = read(path)
            if any(receipt[field] != q[field] for field in (*META, "input_sha256")) or receipt["label"] != label:
                raise ValueError("Receipt input/label identity mismatch")
            sha = reg["historical_checkpoints"][label]["sha256"] if label in HISTORICAL_LABELS else read(out / "evaluation/checkpoints" / (label + ".json"))["sha256"]
            verify_output(q, receipt, sha)
            rows.append(receipt)
    result = {"status": "COMPLETE" if not missing else "PENDING", "registered": 48, "completed": len(rows), "missing": missing, "rows": rows}
    write(out / "evaluation/raw_index.json", result, freeze=False)
    write_csv(out / "evaluation/raw_index.csv", [{k: r.get(k) for k in (*META, "label", "generation_status", "new_rollouts", "input_path", "input_sha256", "raw_path", "raw_sha256", "checkpoint_sha256")} for r in rows])
    if missing and require_complete:
        raise ValueError("All 48 registered RAWs must be saved/verified before score/render; missing=" + str(len(missing)))
    return rows


def score_one(out, q, receipt, query, registration_sha):
    """Independent root adapter; mathematical scorer is the unchanged old code."""
    out = root(out)
    destination = out / "evaluation/scores" / receipt["label"] / (q["request_id"] + ".json")
    identity = {"version": VERSION, "registration_sha256": registration_sha, "label": receipt["label"], "request_id": q["request_id"],
                "raw_sha256": receipt["raw_sha256"], "input_sha256": q["input_sha256"], "query_sha256": query["sha256"], "scorer_sources": scorer_sources()}
    if destination.exists():
        result = read(destination)
        if result["identity"] != identity:
            raise ValueError("Scoring identity changed")
        for p, sha in result.get("files", {}).items():
            if digest(p) != sha:
                raise ValueError("Score vectors changed")
        return result
    if receipt["label"] in HISTORICAL_LABELS:
        historical_path = resolve(receipt["historical_score_path"])
        if digest(historical_path) != receipt["historical_score_sha256"]:
            raise ValueError("Historical score changed after registration")
        result = copy.deepcopy(_verify_historical_score(q, receipt, query, read(HISTORY / "evaluation/registration.json"), historical_path))
        old_identity = result.pop("identity")
        result.update(identity=identity, score_execution="VERIFIED_EXACT_DEFINITION_HISTORICAL_CACHE", historical_identity=old_identity,
                      historical_score_path=str(historical_path), historical_score_sha256=digest(historical_path), new_cpu_geometry_scans=0)
        write(destination, result)
        return result
    started = time.perf_counter()
    result = {k: q[k] for k in META}
    result.update(label=receipt["label"], identity=identity, status="BLOCKED", metrics={}, files={})
    try:
        raw, facts = verify_output(q, receipt)
        b = read_input(q)
        reference = load_queries(query)
        C, F = b["C"], raw["raw"][q["K"]:]
        mod = cpu_math()
        metrics, cf, ff = mod["geometry"].scan_geometry(C, F, L)
        trusted = mod["control"].trusted_cf(C, F, L, cf_scan=cf)
        im, iv = mod["control"].conflict_metrics(F, {"C": C, "L0": L}, reference, trusted)
        if abs(im["ConflictCov_C"] - im["ConflictCov_C_cross"] - im["ConflictCov_C_dup"] + im["ConflictCov_C_cross_and_dup"]) > 1e-12:
            raise ValueError("I union identity")
        metrics.update(im)
        metrics.update(mod["geometry"].topology(raw["raw"], q["K"]))
        illegal = {"proper_transverse_penetration", "coplanar_positive_area_overlap"}
        cf_free = {r["pair"][1] for r in cf["contacts"] if r["classification"] in illegal}
        metrics["CF_illegal_pair_count"] = sum(r["classification"] in illegal for r in cf["contacts"])
        metrics["CF_illegal_affected_free_fraction"] = len(cf_free) / len(F)
        cq = mod["shape"].shape_quality(C, L)
        whole = mod["shape"].shape_quality(raw["raw"], L)
        metrics.update(C_total_surface_area=cq["total_area"], C_near_degenerate_count=cq["near_degenerate_area_count"],
                       C_exact_zero_area_count=cq["exact_zero_area_count"], whole_total_surface_area=whole["total_area"],
                       whole_near_degenerate_count=whole["near_degenerate_area_count"], whole_minimum_angle_lt5_fraction=whole["min_angle_lt5_fraction"])
        qp = mod["surface"].sample_mesh_surface(F, 2048, np.random.RandomState(mod["quality"].seed("Q_F", q["condition_id"], q["candidate"], 2048)))
        if qp is None:
            raise ValueError("No finite positive free surface")
        folder = out / "evaluation/score_details" / receipt["label"] / q["request_id"]
        pairpath, vectorpath = folder / "pairs.json", folder / "vectors.npz"
        write(pairpath, {"CF": cf, "FF": ff})
        save_npz(vectorpath, dict(iv, Q_F2048=qp))
        result.update(status="PASS", metrics=metrics, contracts=facts, files={str(pairpath): digest(pairpath), str(vectorpath): digest(vectorpath)},
                      vectors_path=str(vectorpath), pair_audit_path=str(pairpath), metric_status={"ConflictCov_C": im["ConflictCov_C_status"],
                      "SeamCov": "NA_NO_REGISTERED_REQUIRED_INTERFACE", "ExposedRetain": "NA_NO_EXPOSURE_TASK", "GT_reconstruction": "NA_NOT_UNIQUE_GENERATIVE_TARGET",
                      "watertight_certification": "NOT_IMPLEMENTED", "human_usability": "NOT_RUN"}, C_query_denominator=float(reference["arrays"]["C_weights"].sum()),
                      C_query_count=2048, free_face_denominator=len(F), full_CF_scans=1, full_FF_scans=1, I_reuses_CF=True,
                      score_execution="NEW_UNCHANGED_CPU_SCORER", new_cpu_geometry_scans=1)
    except Exception as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc())
    result["seconds"] = time.perf_counter() - started
    write(destination, result)
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _mean_complete(values):
    return float(np.mean(values)) if values and all(_number(v) for v in values) else None


def _summary(records, numeric, label, group, value):
    pp = [r for r in records if r["label"] == label and (group == "ALL" or r[group] == value)]
    pids = sorted({r["parent_id"] for r in pp})
    result = {"panel": "NEWDEV", "label": label, "grouping": group, "group": value,
              "registered_conditions": len(pp), "registered_parents": len(pids), "registered_outputs": 2 * len(pp)}
    for key in numeric:
        result[key] = _mean_complete([_mean_complete([r[key] for r in pp if r["parent_id"] == pid]) for pid in pids])
    return result


def score_all(out):
    out = root(out)
    reg = load_registration(out)
    receipts = {(r["label"], r["request_id"]): r for r in collect_outputs(out)}
    queries = {q["condition_id"]: q for q in reg["queries"]}
    rows, started = [], time.perf_counter()
    for label in LABELS:
        for q in reg["requests"]:
            row = score_one(out, q, receipts[label, q["request_id"]], queries[q["condition_id"]], digest(out / "evaluation/registration.json"))
            rows.append(row)
            write(out / "evaluation/scoring_progress.json", {"completed": len(rows), "registered": 48, "seconds": time.perf_counter() - started}, freeze=False)
            print(json.dumps({"stage": "score", "label": label, "request_id": q["request_id"], "status": row["status"], "completed": len(rows)}), flush=True)
    write(out / "evaluation/scores_all.json", {"status": "COMPLETE" if all(r["status"] == "PASS" for r in rows) else "PARTIAL_BLOCKED", "rows": rows, "registered": 48})
    write_csv(out / "evaluation/per_output.csv", [{**{k: r.get(k) for k in (*META, "label", "status", "reason", "score_execution", "C_query_count", "C_query_denominator", "free_face_denominator")}, **r["metrics"]} for r in rows])
    numeric = sorted({k for r in rows for k, v in r["metrics"].items() if _number(v)})
    by = {(r["label"], r["request_id"]): r for r in rows}
    conditions, paired_noise, paired_conditions = [], [], []
    ids = sorted({q["condition_id"] for q in reg["requests"]})
    for cid in ids:
        qs = [q for q in reg["requests"] if q["condition_id"] == cid]
        for label in LABELS:
            rr = [by[label, q["request_id"]] for q in qs]
            record = {k: qs[0][k] for k in ("panel", "condition_id", "parent_id", "region_type", "tier")}
            record.update(label=label, registered_noises=2, scored_outputs=sum(r["status"] == "PASS" for r in rr))
            record.update({k: _mean_complete([r["metrics"].get(k) if r["status"] == "PASS" else None for r in rr]) for k in numeric})
            conditions.append(record)
    by_condition = {(r["label"], r["condition_id"]): r for r in conditions}
    for baseline in HISTORICAL_LABELS:
        comparison = NEW_LABEL + " minus " + baseline
        for q in reg["requests"]:
            a, b = by[NEW_LABEL, q["request_id"]], by[baseline, q["request_id"]]
            paired_noise.append({**{k: q[k] for k in META}, "comparison": comparison,
                                 **{k: a["metrics"][k] - b["metrics"][k] if a["status"] == b["status"] == "PASS" and _number(a["metrics"].get(k)) and _number(b["metrics"].get(k)) else None for k in numeric}})
        for cid in ids:
            a, b = by_condition[NEW_LABEL, cid], by_condition[baseline, cid]
            paired_conditions.append({**{k: a[k] for k in ("panel", "condition_id", "parent_id", "region_type", "tier")}, "comparison": comparison,
                                      **{k: a[k] - b[k] if _number(a[k]) and _number(b[k]) else None for k in numeric}})
    write_csv(out / "evaluation/per_condition.csv", conditions)
    write_csv(out / "evaluation/paired_noise.csv", paired_noise)
    write_csv(out / "evaluation/paired_conditions.csv", paired_conditions)
    write_csv(out / "evaluation/summary_equal_parent.csv", [_summary(conditions, numeric, label, "ALL", "ALL") for label in LABELS])
    write_csv(out / "evaluation/summary_by_region.csv", [_summary(conditions, numeric, label, "region_type", region) for region in ("seat", "back") for label in LABELS])
    write_csv(out / "evaluation/summary_by_tier.csv", [_summary(conditions, numeric, label, "tier", tier) for tier in sorted({q["tier"] for q in reg["requests"]}) for label in LABELS])
    distributions, directions = [], []
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        for k in numeric:
            values = [r["metrics"].get(k) if r["status"] == "PASS" else None for r in group]
            good = [v for v in values if _number(v)]
            distributions.append({"label": label, "metric": k, "registered_outputs": len(group), "finite_values": len(good), "missing": len(group) - len(good),
                                  "min": float(np.min(good)) if good else None, "median": float(np.median(good)) if good else None,
                                  "p90": float(np.quantile(good, .9)) if good else None, "max": float(np.max(good)) if good else None,
                                  "scope": "Observed values; missing counts explicit, primary means require completeness"})
    for baseline in HISTORICAL_LABELS:
        comparison = NEW_LABEL + " minus " + baseline
        for level, source in (("condition", paired_conditions), ("noise", paired_noise)):
            group = [r for r in source if r["comparison"] == comparison]
            for k in numeric:
                values = [r[k] for r in group]
                good = [v for v in values if _number(v)]
                directions.append({"comparison": comparison, "level": level, "metric": k, "registered": len(group), "finite": len(good),
                                   "negative": sum(v < 0 for v in good), "zero": sum(v == 0 for v in good), "positive": sum(v > 0 for v in good),
                                   "mean_delta": _mean_complete(values), "median_delta": float(np.median(good)) if good else None,
                                   "min_delta": min(good) if good else None, "max_delta": max(good) if good else None,
                                   "interpretation": "Direction only; not all metrics have a preferred smaller value"})
    write_csv(out / "evaluation/geometry_distributions.csv", distributions)
    write_csv(out / "evaluation/paired_direction_counts.csv", directions)
    diversity = []
    for p in conditions:
        qs = sorted((q for q in reg["requests"] if q["condition_id"] == p["condition_id"]), key=lambda q: q["candidate"])
        a, b = [by[p["label"], q["request_id"]] for q in qs]
        record = {k: p[k] for k in ("label", "panel", "condition_id", "parent_id", "region_type", "tier")}
        if a["status"] == b["status"] == "PASS":
            av, bv = arrays(a["vectors_path"])["Q_F2048"], arrays(b["vectors_path"])["Q_F2048"]
            ar, _ = verify_output(qs[0], receipts[p["label"], qs[0]["request_id"]])
            br, _ = verify_output(qs[1], receipts[p["label"], qs[1]["request_id"]])
            surface = cpu_math()["surface"]
            d1 = surface.surface_distance_squared(av, br["raw"][qs[1]["K"]:])
            d2 = surface.surface_distance_squared(bv, ar["raw"][qs[0]["K"]:])
            record.update(status="PASS", D_free=float((d1.mean() + d2.mean()) / (2 * L**2)), noise_pairs=1)
        else:
            record.update(status="BLOCKED", D_free=None, noise_pairs=1)
        diversity.append(record)
    write_csv(out / "evaluation/diversity.csv", diversity)
    return {"registered": 48, "scored": sum(r["status"] == "PASS" for r in rows), "new_score_rows": 12, "historical_score_reuses": 36, "seconds": time.perf_counter() - started}


def render_all(out):
    out = root(out)
    reg = load_registration(out)
    receipts = {(r["label"], r["request_id"]): r for r in collect_outputs(out)}
    helper = source_module("_r2_mix_readonly_render", R2 / "data_tools/user_conditions_v1/render.py")
    plt = helper.plt
    pages, bounds, started = [], {}, time.perf_counter()
    geometry = {}
    for q in reg["requests"]:
        for label in LABELS:
            geometry[label, q["request_id"]] = verify_output(q, receipts[label, q["request_id"]])[0]["raw"]
    for cid in sorted({q["condition_id"] for q in reg["requests"]}):
        qs = [q for q in reg["requests"] if q["condition_id"] == cid]
        points = np.concatenate([geometry[label, q["request_id"]].reshape(-1, 3) for q in qs for label in LABELS])
        lo, hi = points.min(0), points.max(0)
        bounds[cid] = {"center_model_xyz": ((lo + hi) / 2).tolist(), "span": float(np.max(hi - lo)) * 1.14,
                       "scope": "Union C+RAW all4models/all2noises; no crop, display-only", "projection": "orthographic",
                       "axis_transform": "model[x,y,z]->plot[x,z,y]"}
    for q in reg["requests"]:
        for mode in ("neutral", "colored"):
            path = out / "gallery" / mode / (q["request_id"] + ".png")
            sidecar = path.with_suffix(".png.json")
            identity = {"version": VERSION, "evaluation_registration_sha256": digest(out / "evaluation/registration.json"),
                        "request_id": q["request_id"], "condition_id": q["condition_id"], "panel": "NEWDEV", "mode": mode,
                        "columns": list(LABELS), "cameras": [[22, -60], [22, 120]], "bounds": bounds[q["condition_id"]],
                        "raw_sha256": {label: receipts[label, q["request_id"]]["raw_sha256"] for label in LABELS},
                        "source_sha256": digest(__file__), "renderer_helper_sha256": digest(R2 / "data_tools/user_conditions_v1/render.py")}
            if path.exists() and sidecar.exists():
                page = read(sidecar)
                if page["identity"] != identity or page["image_sha256"] != digest(path):
                    raise ValueError("Gallery identity changed")
                pages.append(page)
                continue
            if path.exists() or sidecar.exists():
                raise ValueError("Incomplete gallery artifact")
            fig = plt.figure(figsize=(18, 9), facecolor="white")
            for col, label in enumerate(LABELS):
                tri = geometry[label, q["request_id"]]
                colors = helper._shaded(tri, "#acb3bb" if mode == "neutral" else "#de9758")
                if mode == "colored":
                    colors[:q["K"]] = helper._shaded(tri[:q["K"]], "#268cca")
                for rr, view in enumerate(((22, -60), (22, 120))):
                    ax = fig.add_subplot(2, 4, rr * 4 + col + 1, projection="3d")
                    helper._draw(ax, tri, colors, bounds[q["condition_id"]], view)
                    ax.set_title(label + " | " + ("front oblique" if rr == 0 else "opposite oblique"), fontsize=10)
            fig.suptitle("USER DEVELOPMENT | " + q["parent_id"] + " | " + q["region_type"] + "/" + q["tier"] +
                         " | n" + str(q["candidate"]) + " | N=" + str(q["N"]) + " K=" + str(q["K"]), fontsize=11, y=.985)
            fig.text(.5, .018, "RAW ONLY | " + ("BLUE=C / ORANGE=generated free" if mode == "colored" else "Neutral geometry") +
                     " | Same condition bounds across all models/noises; historical comparisons are descriptive", ha="center", fontsize=9)
            fig.subplots_adjust(left=0, right=1, top=.92, bottom=.05, wspace=0, hspace=.03)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".png.tmp")
            fig.savefig(temp, format="png", dpi=150, facecolor="white")
            plt.close(fig)
            temp.replace(path)
            page = {"status": "COMPLETE", "identity": identity, "image": str(path), "image_sha256": digest(path), "AI_PROXY_review": "PENDING"}
            write(sidecar, page)
            pages.append(page)
        write(out / "gallery/progress.json", {"pages": len(pages), "target": 24, "seconds": time.perf_counter() - started}, freeze=False)
    write(out / "gallery/manifest.json", {"status": "COMPLETE", "pages": pages, "requests": 12, "model_outputs": 48,
                                        "PNG_count": 24, "model_calls": 0, "GPU": 0})
    template = out / "gallery/AI_PROXY_review_template.csv"
    if not template.exists():
        write_csv(template, [{**{k: q[k] for k in META}, "label": label, "reviewer": "AI_PROXY", "actually_viewed": False,
                              "neutral_image": str(out / "gallery/neutral" / (q["request_id"] + ".png")),
                              "colored_image": str(out / "gallery/colored" / (q["request_id"] + ".png")),
                              "overall_observation": "", "C_role_compatibility": "", "visible_fragments_or_misuse": "",
                              "comparison_improved_similar_degraded_uncertain": "", "uncertainty": "", "human_rating": False}
                             for q in reg["requests"] for label in LABELS])
    sections = []
    for q in reg["requests"]:
        rid = html.escape(q["request_id"])
        sections.append('<section id="' + rid + '"><h2>' + html.escape(q["parent_id"] + " | " + q["region_type"] + "/" + q["tier"] + " | noise " + str(q["candidate"])) + '</h2>' +
                        ''.join('<a href="' + mode + '/' + rid + '.png"><img loading="lazy" src="' + mode + '/' + rid + '.png" alt="' + mode + '"></a>' for mode in ("colored", "neutral")) + '</section>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>R2 USER mix development gallery</title><style>body{font:16px system-ui;margin:24px;background:#fafafa}img{width:100%;height:auto}section{margin:36px 0;border-top:1px solid #bbb}h2{font-size:18px}a{color:#145ea8}</style><h1>R2 USER mix: all 12 frozen requests</h1><p>Columns: R2_FROZEN / R2_CONT_OLD / R2_CONT_USER / R2_USER_MIX. RAW only. Blue is preserved C; orange is free. Two cameras are views of one output, not independent samples. Six repeated development parents, two fixed noises; no unique-GT ranking. Historical training populations and update totals differ. Visual review is separate AI_PROXY evidence.</p>' + ''.join(sections) + '</html>\n'
    index = out / "gallery/index.html"
    if index.exists() and index.read_text(encoding="utf8") != document:
        raise ValueError("Gallery index content changed")
    if not index.exists():
        index.write_text(document, encoding="utf8")
    return {"status": "COMPLETE", "PNG_count": 24, "model_outputs": 48, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "bind", "sample", "collect", "score", "render"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--label", choices=LABELS, default=NEW_LABEL)
    parser.add_argument("--checkpoint")
    parser.add_argument("--attempt", default="initial", choices=("initial",))
    parser.add_argument("--controller-owned", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        reg = prepare(args.out)
        result = {"status": reg["status"], "requests": len(reg["requests"]), "conditions": len(reg["queries"]), "budget": reg["budget"], "model_calls": 0}
    elif args.stage == "bind":
        if not args.checkpoint:
            parser.error("bind needs --checkpoint")
        result = bind_checkpoint(args.out, args.label, args.checkpoint)
    elif args.stage == "sample":
        result = sample_job(args.out, args.label, args.attempt, args.controller_owned)
        result = {k: v for k, v in result.items() if k != "rows"}
    elif args.stage == "collect":
        rows = collect_outputs(args.out, False)
        result = {"completed": len(rows), "registered": 48, "status": "COMPLETE" if len(rows) == 48 else "PENDING"}
    elif args.stage == "score":
        result = score_all(args.out)
    else:
        result = render_all(args.out)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
