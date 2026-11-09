#!/usr/bin/env python3
"""Isolated recipe-factorial input, sampling, cache and gallery controller.

No model is imported until sample_label. Historical files are read-only. Every
phase owns frozen labels, input identities and a durable reserve-before-call log.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from meshflow_control.evaluation.io import arrays, array_sha, digest, read_json, resolve, save_npz, write_json

VERSION = "chair_condition_recipe_factorial_evaluation_v1"
ROOT_NAME = "chair_condition_recipe_factorial_v1"
DEFAULT_OUT = "/mnt/e/MeshFlow-Control-artifacts/experiments/" + ROOT_NAME
LABELS = ("START5000", "HIST_FM6000", "R1_40_H_ALL", "R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE")
NOISES = (0, 1, 6, 7)
L_CANON = float(np.sqrt(3.) * 2. / .3762)
META = ("request_id", "label", "parent_id", "condition_id", "candidate", "ratio", "condition_cohort", "N", "K")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Unsafe registered identifier")
    return value


def root(path):
    p = resolve(path).resolve()
    if p.name != ROOT_NAME or p.parent.name != "experiments":
        raise ValueError("Wrong authorized experiment directory")
    return p


def phase_root(out, phase):
    return root(out) / "phases" / safe_name(phase)


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def read_input(q):
    if digest(q["input_path"]) != q["input_sha256"]:
        raise ValueError("Frozen input changed")
    b = arrays(q["input_path"])
    if not {"C", "z", "N", "K"} <= set(b) or set(b) - {"C", "z", "N", "K", "known_mask", "valid_mask"}:
        raise ValueError("Inference bank has unauthorized fields")
    n, k = int(b["N"]), int(b["K"])
    if (n, k) != (q["N"], q["K"]) or not 0 < k < n or not 128 <= n <= 256:
        raise ValueError("Input face budget differs")
    if b["C"].dtype != np.float32 or b["C"].shape != (k, 3, 3) or b["z"].dtype != np.float32 or b["z"].shape != (1, n, 9):
        raise ValueError("Input must preserve FP32 C and Gaussian shapes")
    if not np.isfinite(b["C"]).all() or not np.isfinite(b["z"]).all() or np.any(b["z"][:, :k]):
        raise ValueError("Nonfinite input or nonzero known Gaussian")
    for key, value in (("C_sha256", b["C"]), ("Gaussian_sha256", b["z"])):
        if key in q and q[key] != array_sha(value):
            raise ValueError("Input array identity differs")
    return b


def derive_a40_noise(a20, k40):
    """Absolute slots K40:N, not the first N-K40 old free slots; no RNG."""
    z = np.asarray(a20)
    if z.dtype != np.float32 or z.ndim != 3 or z.shape[0] != 1 or z.shape[2] != 9 or not 0 < k40 < z.shape[1]:
        raise ValueError("Malformed existing Gaussian")
    result = np.zeros_like(z)
    result[:, k40:] = z[:, k40:]
    return result


def validate_requests(rows, *, main=True):
    ids = [safe_name(r["request_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request")
    groups = {}
    for r in rows:
        safe_name(r["condition_id"])
        if r["ratio"] not in (.2, .4):
            raise ValueError("Only registered20/40 conditions")
        key = (r["parent_id"], r["ratio"])
        groups.setdefault(key, []).append(r)
    for rr in groups.values():
        if len(rr) != 4 or len({r["candidate"] for r in rr}) != 4:
            raise ValueError("Four distinct noises required per condition")
        for key in ("condition_id", "N", "K", "C_sha256", "condition_cohort"):
            if len({r[key] for r in rr}) != 1:
                raise ValueError("Condition identity changes across noise")
        if main and set(r["candidate"] for r in rr) != set(NOISES):
            raise ValueError("Main panel must preserve n0/n1/n6/n7")
    if main:
        parents = {r["parent_id"] for r in rows}
        cohorts = Counter(rr[0]["condition_cohort"] for key, rr in groups.items() if key[1] == .4)
        if len(rows) != 64 or len(parents) != 8 or set(groups) != {(p, r) for p in parents for r in (.2, .4)}:
            raise ValueError("Expected8 parents x2 ratios x4 noises")
        if cohorts != {"ORIGINAL6": 6, "EXTENDED2": 2}:
            raise ValueError("A40 original6/extended2 split differs")
    return rows


def prepare_inputs(out, artifacts="/mnt/e/MeshFlow-Control-artifacts"):
    out = root(out); art = resolve(artifacts)
    src20 = art / "experiments/chair_contextual_alignment_sie_v1/data/evaluation/requests.json"
    src40 = art / "experiments/chair_early_alignment_schedule40_v1/evaluation/requests.json"
    old20 = [r for r in read_json(src20)["rows"] if r["panel"] == "P1"]
    old40 = read_json(src40)["rows"]
    rows = []
    for old in old20:
        q = dict(old); q.pop("historical_start", None)
        b = read_input(q)
        q.update(ratio=.2, condition_cohort="A20", C_sha256=array_sha(b["C"]), Gaussian_sha256=array_sha(b["z"]),
                 condition_construction="EXISTING_A20", source_manifest=str(src20), source_manifest_sha256=digest(src20),
                 noise_provenance="EXISTING_A20_ACTUAL_ARRAY", auxiliary_reference_status="A20_REFERENCE_DIAGNOSTICS_ONLY")
        rows.append(q)
    by = {(r["parent_id"], r["candidate"]): r for r in rows}
    for old in old40:
        b40 = read_input(old)
        cohort = "ORIGINAL6" if old["condition_construction"].startswith("EXISTING") else "EXTENDED2"
        for candidate in NOISES:
            q = dict(old)
            if candidate != 0:
                source = by[old["parent_id"], candidate]; b20 = read_input(source)
                if int(b20["N"]) != old["N"] or int(b20["K"]) >= old["K"]:
                    raise ValueError("Cross-ratio N/K contract differs")
                z = derive_a40_noise(b20["z"], old["K"])
                rid = old["parent_id"] + "__C40__n" + str(candidate)
                path = out / "evaluation/inputs" / (rid + ".npz")
                save_npz(path, dict(C=b40["C"], z=z, N=b40["N"], K=b40["K"]))
                q.update(request_id=rid, candidate=candidate, input_path=str(path), input_sha256=digest(path),
                         noise_seed=source.get("noise_seed"), Gaussian_sha256=array_sha(z),
                         noise_provenance="A20_ABSOLUTE_FREE_SLOTS_K40_TO_N", noise_source_path=source["input_path"],
                         noise_source_sha256=source["input_sha256"], noise_source_Gaussian_sha256=source["Gaussian_sha256"],
                         noise_slice=[old["K"], old["N"]])
            else:
                q["noise_provenance"] = "EXISTING_A40_N0_DIFFERENT_FROM_A20_N0"
            q.update(ratio=.4, condition_cohort=cohort, source_manifest=str(src40), source_manifest_sha256=digest(src40),
                     auxiliary_reference_status="NA_NO_TRUSTED_A40_INTERFACE", cross_ratio_pure_added_C=False)
            rows.append(q)
    rows.sort(key=lambda r: (r["ratio"], r["parent_id"], r["candidate"]))
    validate_requests(rows)
    for q in rows:
        read_input(q)
    manifest = dict(version=VERSION, rows=rows, labels=list(LABELS), outputs_per_model=64, L_canonical=L_CANON,
                    model_calls=0, new_Gaussian_draws=0, condition_ratio_comparison="Different completion tasks; not pure added-C intervention")
    write_json(out / "evaluation/requests.json", manifest, freeze=True)
    register_confirmation_plan(out, rows)
    from score_recipe_factorial_conflict import prepare_queries
    return dict(status="FROZEN", requests=64, query_registration=prepare_queries(out, out / "evaluation/requests.json"), model_calls=0)



def register_confirmation_plan(out, rows):
    """Freeze identities/seeds only; no confirmation noise arrays or scores."""
    namespace = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1/confirmation_noise/v1"
    planned = []
    for pid in sorted({q["parent_id"] for q in rows}):
        for index in range(4):
            seed_text = namespace + "/" + pid + "/n" + str(index)
            seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf8")).digest()[:4], "little")
            for ratio in (.2, .4):
                q = next(q for q in rows if q["parent_id"] == pid and q["ratio"] == ratio)
                planned.append(dict(parent_id=pid, ratio=ratio, condition_id=q["condition_id"], C_sha256=q["C_sha256"],
                                    N=q["N"], K=q["K"], candidate=8+index, seed=seed, seed_text=seed_text))
    value = dict(status="PREREGISTERED_NOT_MATERIALIZED", namespace=namespace, rows=planned,
                 method="MT19937 full[1,N,9] standardGaussian FP32 then known prefix zero; same parent/noise absolute slots across ratios",
                 candidate_model_identity_in_seed=False, model_pair="NOT_SELECTED", expected_outputs=128,
                 main_max_outputs=384, course_max_new_outputs=128, total_registered_output_ceiling=640,
                 selection_independent_noises=True, new_parent_generalization=False,
                 actual_array_collision_audit="REQUIRED_ON_MATERIALIZATION", Gaussian_arrays_created=0, model_calls=0)
    write_json(root(out) / "evaluation/confirmation_plan.json", value, freeze=True)
    return value


def register_phase(out, phase, models, requests_path=None, kind="main"):
    p = phase_root(out, phase)
    rp = resolve(requests_path) if requests_path else root(out) / "evaluation/requests.json"
    rows = validate_requests(read_json(rp)["rows"], main=kind != "confirm")
    labels = [safe_name(m["label"]) for m in models]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Duplicate/empty labels")
    if kind == "main" and tuple(labels) != LABELS:
        raise ValueError("Main labels must be the six frozen recipes in order")
    if kind == "course" and (not 1 <= len(labels) <= 2 or set(labels) & set(LABELS)):
        raise ValueError("Course phase registers only one or two new labels; prior baselines stay in their completed main phase")
    if kind == "confirm" and (len(labels) != 2 or len(rows) != 64 or rp.resolve() == (root(out) / "evaluation/requests.json").resolve()):
        raise ValueError("Confirm requires two independent-repeat labels and independent requests")
    if kind not in ("main", "course", "confirm"):
        raise ValueError("Unregistered phase kind")
    prior = [read_json(file) for file in (root(out) / "phases").glob("*/registration.json") if file.parent != p]
    if kind == "main" and any(r["kind"] == "main" for r in prior):
        raise ValueError("Only one main phase is authorized")
    if kind == "course":
        extra = set(labels) - set(LABELS)
        for r in prior:
            if r["kind"] == "course":
                if set(labels) & set(r["labels"]):
                    raise ValueError("A course label cannot reserve a second64-output phase")
                extra.update(set(r["labels"]) - set(LABELS))
        if len(extra) > 2:
            raise ValueError("Across this run at most two additional course labels")
    if kind == "confirm":
        if any(r["kind"] == "confirm" for r in prior):
            raise ValueError("Only one independent confirmation pair is authorized")
        plan = read_json(root(out) / "evaluation/confirmation_plan.json")
        planned = {(r["parent_id"],r["ratio"],r["candidate"]):r for r in plan["rows"]}
        if set(planned) != {(r["parent_id"],r["ratio"],r["candidate"]) for r in rows}:
            raise ValueError("Confirmation requests differ from preregistered holdout identities")
        for q in rows:
            r = planned[q["parent_id"],q["ratio"],q["candidate"]]
            if any(q[key] != r[key] for key in ("N","K","C_sha256","condition_id")) or q.get("noise_seed") != r["seed"]:
                raise ValueError("Confirmation conditions/noise seed differ from preregistration")
            read_input(q)
        if not read_json(rp).get("holdout_noise_collision_audit_passed"):
            raise ValueError("Confirmation needs actual-array collision audit before registration")
    frozen = []
    for m in models:
        if m["loader"] not in ("start", "historical", "schedule40", "recipe_factorial"):
            raise ValueError("Unknown checkpoint loader")
        frozen.append({k: m[k] for k in ("label", "checkpoint_path", "loader")})
    scientific = dict(version=VERSION, phase=phase, kind=kind, models=frozen, labels=labels,
                      requests_path=str(rp), requests_sha256=digest(rp), registered_per_model=len(rows),
                      expected_outputs=len(rows) * len(labels), nfe_per_output=50, max_cell_attempts=1,
                      max_label_dispatch_attempts=2, actual_checkpoint_hashes="Separately frozen before first sample")
    write_json(p / "registration.json", scientific, freeze=True)
    from score_recipe_factorial_conflict import prepare_queries
    prepare_queries(out, rp)
    return scientific


def phase_registration(out, phase):
    p = phase_root(out, phase); s = read_json(p / "registration.json")
    if s["version"] != VERSION or digest(s["requests_path"]) != s["requests_sha256"]:
        raise ValueError("Phase registration changed")
    rows = validate_requests(read_json(s["requests_path"])["rows"], main=s["kind"] != "confirm")
    return p, s, rows


def bind_checkpoint(out, phase, label):
    p, s, _ = phase_registration(out, phase)
    model = next(m for m in s["models"] if m["label"] == label)
    binding = dict(model, checkpoint_sha256=digest(model["checkpoint_path"]), registration_sha256=digest(p / "registration.json"))
    write_json(p / "checkpoints" / (label + ".json"), binding, freeze=True)
    return binding


def canonical_raw(raw, n, k):
    return dict(raw=np.asarray(raw.get("raw", raw.get("output"))).reshape(n, 3, 3),
                trajectory=np.asarray(raw.get("trajectory", raw.get("path"))).reshape(51, n, 3, 3),
                C=np.asarray(raw["C"]).reshape(k, 3, 3),
                z=np.asarray(raw.get("z", raw.get("initial_gaussian"))).reshape(1, n, 9))


def verify_output(q, receipt, checkpoint_sha=None):
    from meshflow_control.mesh_io import read_ply
    if checkpoint_sha and receipt.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Checkpoint binding differs")
    if receipt.get("status") not in (None, "COMPLETE", "VERIFIED_CACHE"):
        raise ValueError("Incomplete output receipt")
    for field in ("raw", "ply", "result"):
        if digest(receipt[field + "_path"]) != receipt[field + "_sha256"]:
            raise ValueError("Output " + field + " changed")
    bank = read_input(q); n, k = q["N"], q["K"]
    a = canonical_raw(arrays(receipt["raw_path"]), n, k)
    ply, _ = read_ply(resolve(receipt["ply_path"]))
    result = read_json(receipt["result_path"])
    facts = dict(C_output_bitwise=byte_equal(a["raw"][:k], bank["C"]),
                 C_all51_bitwise=all(byte_equal(x[:k], bank["C"]) for x in a["trajectory"]),
                 actual_Gaussian_bitwise=byte_equal(a["z"], bank["z"]), NPZ_C_bitwise=byte_equal(a["C"], bank["C"]),
                 PLY_C_bitwise=byte_equal(ply[:k], bank["C"]), PLY_full_bitwise=byte_equal(ply, a["raw"]),
                 trajectory_final_is_RAW=byte_equal(a["trajectory"][-1], a["raw"]),
                 initial_free_is_Gaussian=byte_equal(a["trajectory"][0, k:].reshape(1, n-k, 9), bank["z"][:, k:]),
                 N_preserved=a["raw"].shape == (n, 3, 3), finite=bool(np.isfinite(a["trajectory"]).all()),
                 FP32=a["raw"].dtype == a["trajectory"].dtype == a["z"].dtype == np.float32)
    runtime = result.get("sampling", result)
    if runtime.get("nfe") != 50:
        raise ValueError("Runtime proof missing or sampler did not use50 evaluations")
    dtype = str(runtime.get("dtype", "")).upper()
    if "BF16" not in dtype or "FP32" not in dtype or runtime.get("readout_mode") != "none":
        raise ValueError("Runtime proof must establish BF16/FP32 and no inference readout")
    if not all(facts.values()):
        raise ValueError("Hard output contract failed: " + str(facts))
    return a, facts


def reuse_outputs(out, phase, label, source_manifest, source_label):
    p, _, qs = phase_registration(out, phase); binding = bind_checkpoint(out, phase, label)
    source = read_json(source_manifest); sources = source["rows"] if isinstance(source, dict) else source
    reused = []; rejected = []
    for q in qs:
        candidates = [r for r in sources if r.get("label") == source_label and r.get("parent_id") == q["parent_id"]
                      and r.get("candidate") == q["candidate"] and r.get("N") == q["N"] and r.get("K") == q["K"]]
        if len(candidates) != 1:
            continue
        src = dict(candidates[0]); dst = p / "outputs" / label / q["request_id"] / "receipt.json"
        try:
            if src.get("checkpoint_sha256") != binding["checkpoint_sha256"]:
                raise ValueError("Historical file hash is not current checkpoint")
            # Readonly historical PLY may be next to RAW when old manifest omitted it.
            src.setdefault("ply_path", str(resolve(src["raw_path"]).with_suffix(".ply")))
            src.setdefault("ply_sha256", digest(src["ply_path"]))
            src.setdefault("result_sha256", digest(src["result_path"]))
            a, facts = verify_output(q, src, binding["checkpoint_sha256"])
            receipt = dict(q, label=label, status="COMPLETE", checkpoint_sha256=binding["checkpoint_sha256"],
                           generation_status="VERIFIED_HISTORICAL_RAW_REUSE", new_rollouts=0, contracts=facts,
                           source_manifest=str(source_manifest), source_manifest_sha256=digest(source_manifest),
                           source_label=source_label, source_request_id=src["request_id"],
                           **{k: src[k] for k in ("raw_path", "raw_sha256", "ply_path", "ply_sha256", "result_path", "result_sha256")})
            write_json(dst, receipt, freeze=True); reused.append(q["request_id"])
        except (ValueError, FileNotFoundError, KeyError) as exc:
            rejected.append(dict(request_id=q["request_id"], reason=str(exc)))
    audit = dict(label=label, reused=reused, rejected=rejected, source_manifest_sha256=digest(source_manifest), new_rollouts=0)
    write_json(p / "sampling" / label / ("reuse_" + digest(source_manifest)[:12] + ".json"), audit, freeze=True)
    return audit


class CellBudget:
    def __init__(self, path, context):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = context; self.counts = {}
        # Exclusive creation is also the permanent no-retry marker for this cell.
        with self.path.open("x", encoding="utf8") as f:
            f.write(json.dumps(dict(event="CELL_REGISTERED", context=context)) + "\n"); f.flush(); os.fsync(f.fileno())

    def event(self, event, **kwargs):
        with self.path.open("a", encoding="utf8") as f:
            f.write(json.dumps(dict(time_unix=time.time(), event=event, **kwargs), allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())

    def reserve(self, kind, context=None):
        count = self.counts.get(kind, 0) + 1
        if kind not in ("rollout_attempt", "sampling_forward") or count > {"rollout_attempt": 1, "sampling_forward": 50}[kind]:
            raise RuntimeError("Registered cell budget exceeded")
        self.event("RESERVE", kind=kind, ordinal=count, context=context or self.context)
        self.counts[kind] = count

    def call(self, kind, fn, context=None):
        self.reserve(kind, context); tick = time.perf_counter()
        try:
            value = fn()
        except BaseException as exc:
            self.event("ERROR", kind=kind, error=repr(exc)); raise
        self.event("RETURN", kind=kind, seconds=time.perf_counter()-tick)
        return value


def sample_label(out, phase, label, attempt="initial", controller_lock_owned=False):
    if not controller_lock_owned:
        raise RuntimeError("Sampling requires resource-aware parent controller owning shared GPU lock")
    if attempt not in ("initial", "serial_retry", "oom_serial_retry"):
        raise ValueError("At most initial and one serial_retry dispatch")
    p, s, qs = phase_registration(out, phase); binding = bind_checkpoint(out, phase, label)
    folder = p / "sampling" / label; folder.mkdir(parents=True, exist_ok=True)
    if attempt in ("serial_retry", "oom_serial_retry") and not (folder / "initial_failure.json").exists():
        raise ValueError("Serial retry requires previous failed dispatch")
    dispatch_key = "initial" if attempt == "initial" else "serial_retry"
    with (folder / (dispatch_key + "_dispatch.json")).open("x", encoding="utf8") as f:
        json.dump(dict(binding, attempt=attempt, started=time.time()), f); f.flush(); os.fsync(f.fileno())
    completed = []; blocked = []; model = None; tick = time.perf_counter()
    try:
        # Validate/reuse complete cells before constructing any real model.
        pending = []
        for q in qs:
            dst = p / "outputs" / label / q["request_id"]
            if (dst / "receipt.json").exists():
                r = read_json(dst / "receipt.json"); verify_output(q, r, binding["checkpoint_sha256"]); completed.append(r)
            elif (dst / "calls.jsonl").exists():
                blocked.append(q["request_id"])
            else:
                read_input(q); pending.append(q)
        if pending and s["kind"] == "course" and label in LABELS:
            raise RuntimeError("Course baseline must reuse verified main outputs; no duplicate baseline rollouts")
        if pending:
            import torch
            from meshflow_control import runtime
            from meshflow_control.sampling import sample
            torch.set_num_threads(2); runtime.configure_stable_runtime()
            if binding["loader"] == "start":
                from meshflow_control.checkpoints import load_start
                model, _, _ = load_start(resolve(binding["checkpoint_path"]), device="cuda", trainable=False)
            else:
                import importlib
                module = {"historical": "context_checkpoint", "schedule40": "schedule40", "recipe_factorial": "recipe_factorial"}[binding["loader"]]
                model, _ = importlib.import_module("meshflow_control.training." + module).load_generator(resolve(binding["checkpoint_path"]), device="cuda")
            for q in pending:
                dst = p / "outputs" / label / q["request_id"]; b = read_input(q)
                context = dict(label=label, request_id=q["request_id"], attempt=attempt, phase=phase,
                               checkpoint_sha256=binding["checkpoint_sha256"], input_sha256=q["input_sha256"])
                budget = CellBudget(dst / "calls.jsonl", context)
                _, _, audit = sample(model, b["C"], q["N"], q.get("noise_seed"), b["z"], budget=budget, out=dst, context=context)
                r = dict(q, label=label, status="COMPLETE", checkpoint_sha256=binding["checkpoint_sha256"],
                         generation_status="NEW_EULER50", new_rollouts=1, attempt=attempt,
                         **{k: v for name, file in (("raw", "raw.npz"), ("ply", "raw.ply"), ("result", "run.json")) for k, v in ((name+"_path", str(dst/file)), (name+"_sha256", digest(dst/file)))})
                _, facts = verify_output(q, r, binding["checkpoint_sha256"]); r["contracts"] = facts
                write_json(dst / "receipt.json", r, freeze=True); completed.append(r)
                write_json(folder / "progress.json", dict(status="RUNNING", label=label, attempt=attempt, completed=len(completed), blocked=blocked,
                           elapsed_seconds=time.perf_counter()-tick, **{k: audit[k] for k in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "peak_scope")}))
        result = dict(status="COMPLETE" if len(completed) == len(qs) else "PARTIAL_BLOCKED", label=label, rows=completed,
                      blocked=blocked, registered=len(qs), seconds=time.perf_counter()-tick)
        write_json(folder / "completed.json", result); return result
    except BaseException as exc:
        result = dict(status="OOM" if "out of memory" in str(exc).lower() else "FAILED", label=label, attempt=attempt,
                      completed=len(completed), blocked=blocked, error=repr(exc), traceback=traceback.format_exc(), seconds=time.perf_counter()-tick)
        write_json(folder / (attempt + "_failure.json"), result); raise


def completed_rows(out, phase):
    p, s, qs = phase_registration(out, phase); rows = []
    for label in s["labels"]:
        binding = read_json(p / "checkpoints" / (label + ".json"))
        for q in qs:
            rp = p / "outputs" / label / q["request_id"] / "receipt.json"
            if not rp.exists():
                raise RuntimeError("ALL_PHASE_RAW_REQUIRED_BEFORE_SCORING: " + label + "/" + q["request_id"])
            r = read_json(rp)
            if any(r.get(k) != q[k] for k in ("request_id", "input_sha256", "N", "K", "candidate", "condition_id")) or r["label"] != label:
                raise ValueError("Completed receipt identifies different request")
            verify_output(q, r, binding["checkpoint_sha256"]); rows.append(r)
    write_json(p / "outputs/all_completed.json", dict(status="COMPLETE", rows=rows, registration_sha256=digest(p / "registration.json")), freeze=True)
    return rows


def render(out, phase):
    p, s, qs = phase_registration(out, phase); rows = completed_rows(out, phase)
    from evaluate_schedule40 import render_page
    ordered = sorted(s["labels"], key=lambda label: hashlib.sha256((VERSION+"/"+phase+"/"+label).encode()).hexdigest())
    aliases = {label: "Method " + str(i+1) for i, label in enumerate(ordered)}
    write_json(p / "gallery/private_method_mapping.json", aliases, freeze=True)
    by = {(r["label"], r["request_id"]): r for r in rows}; pages = []; cameras = {}
    for cid in sorted({q["condition_id"] for q in qs}):
        group = [q for q in qs if q["condition_id"] == cid]; loaded = {}
        C = read_input(group[0])["C"]
        for q in group:
            for label in ordered:
                r = by[label, q["request_id"]]
                loaded[label, q["candidate"]] = canonical_raw(arrays(r["raw_path"]), q["N"], q["K"])["raw"]
        points = np.concatenate([C.reshape(-1, 3)] + [v.reshape(-1, 3) for v in loaded.values()])
        bounds = (points.min(0), points.max(0))
        cameras[cid] = dict(bounds=[v.tolist() for v in bounds], all_labels_and_noises=True,
                            raw_hashes={label+"/n"+str(q["candidate"]): by[label, q["request_id"]]["raw_sha256"] for q in group for label in ordered})
        for q in group:
            columns = [("Input C", C, q["K"])] + [(aliases[label], loaded[label, q["candidate"]], q["K"]) for label in ordered]
            for mode in ("colored", "neutral"):
                path = p / "gallery" / mode / (q["request_id"] + ".png")
                render_page(path, columns, bounds, q["parent_id"]+" | C="+str(q["ratio"])+" | n"+str(q["candidate"])+" | "+mode, mode == "neutral")
                pages.append(dict(request_id=q["request_id"], condition_id=cid, candidate=q["candidate"], mode=mode, path=str(path), sha256=digest(path), AI_VISUAL_REVIEW="PENDING"))
    write_json(p / "gallery/cameras.json", cameras, freeze=True)
    manifest = dict(status="COMPLETE", pages=pages, page_count=len(pages), expected_pages=2*len(qs), all_registered_outputs_shown=True,
                    five_views_per_model=True, model_labels_blinded=True, AI_VISUAL_REVIEW="PENDING", human_usability="NOT_RUN")
    write_json(p / "gallery/gallery.json", manifest)
    links = "\n".join('<li><a href="'+html.escape(str(Path(r["path"]).relative_to(p / "gallery")).replace("\\", "/"))+'">'+html.escape(r["request_id"]+" / "+r["mode"])+'</a></li>' for r in pages)
    (p / "gallery/INDEX.html").write_text('<!doctype html><meta charset="utf-8"><title>All registered outputs</title><h1>Blind model labels; complete fixed-noise gallery</h1><ul>'+links+'</ul>', encoding="utf8")
    return {k: v for k, v in manifest.items() if k != "pages"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "prepare-inputs", "register-phase", "bind-checkpoint", "reuse", "sample", "collect", "score", "render"))
    parser.add_argument("--out", default=DEFAULT_OUT); parser.add_argument("--artifacts", default="/mnt/e/MeshFlow-Control-artifacts")
    parser.add_argument("--phase", default="main"); parser.add_argument("--kind", default="main", choices=("main", "course", "confirm"))
    parser.add_argument("--models"); parser.add_argument("--requests"); parser.add_argument("--label"); parser.add_argument("--attempt", default="initial")
    parser.add_argument("--source-manifest"); parser.add_argument("--source-label"); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--controller-lock-owned", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in ("prepare", "prepare-inputs"): result = prepare_inputs(args.out, args.artifacts)
    elif args.stage == "register-phase": result = register_phase(args.out, args.phase, read_json(args.models), args.requests, args.kind)
    elif args.stage == "bind-checkpoint": result = bind_checkpoint(args.out, args.phase, args.label)
    elif args.stage == "reuse": result = reuse_outputs(args.out, args.phase, args.label, args.source_manifest, args.source_label)
    elif args.stage == "sample": result = sample_label(args.out, args.phase, args.label, args.attempt, args.controller_lock_owned)
    elif args.stage == "collect": result = dict(status="COMPLETE", outputs=len(completed_rows(args.out, args.phase)))
    elif args.stage == "render": result = render(args.out, args.phase)
    else:
        from score_recipe_factorial_conflict import score
        result = score(args.out, args.phase, args.workers)
    print(json.dumps(result), flush=True)
    return 2 if result.get("status") in ("PARTIAL_BLOCKED", "FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
