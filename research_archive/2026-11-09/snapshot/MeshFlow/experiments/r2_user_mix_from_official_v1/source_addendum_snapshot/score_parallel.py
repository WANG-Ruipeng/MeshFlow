"""Two CPU workers for the 12 registered new RAWs; historical scores stay cached.

Registration imports no model and performs no score. The root dispatcher starts
the explicit run stage only after all 48 RAWs have been saved and verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

VERSION = "r2_user_mix_parallel_cpu_score_v1"
EVALUATION_PATH = Path(__file__).with_name("evaluation.py")
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "experiments/r2_user_mix_from_official_v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, freeze=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and freeze:
        if read(path) != value:
            raise ValueError("Frozen CPU helper artifact changed: " + str(path))
        return
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf8")
    temporary.replace(path)


def root(path):
    path = Path(path).resolve()
    if path != DEFAULT_OUT.resolve():
        raise ValueError("Only the registered USER mix root is writable")
    return path


def resources():
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "MemFree"):
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    return {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(),
            "logical_cpu_count": os.cpu_count(), "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            "load_average_1_5_15": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "memory": memory, "threads_per_worker": 2, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}


def stable_identity(out):
    return {"version": VERSION, "helper_path": str(Path(__file__).resolve()), "helper_sha256": digest(__file__),
            "evaluation_path": str(EVALUATION_PATH.resolve()), "evaluation_sha256": digest(EVALUATION_PATH),
            "evaluation_registration_sha256": digest(out / "evaluation/registration.json"),
            "registered_new_outputs": 12, "historical_outputs": 36, "maximum_workers": 2,
            "threads_per_worker": 2, "new_model_calls": 0, "new_queries": 0, "new_Gaussians": 0,
            "execution": "Process-local evaluation modules; disjoint per-request score outputs; serial deterministic cache aggregation"}


def register(out):
    out = root(out)
    identity = stable_identity(out)
    write(out / "audit/score_helper_identity.json", identity)
    target = out / "audit/score_helper_registration_resources.json"
    if not target.exists():
        write(target, resources())
    return {"status": "REGISTERED", **identity, "scoring_executed": False}


def load_evaluation():
    name = "_r2_user_mix_cpu_scoring_process_" + str(os.getpid())
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EVALUATION_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only scorer unexpectedly imported torch")
    return module


def score_request(out_string, request_id, expected_identity):
    """Process-owned module, RNG and one unique score path; never catches failure."""
    out = root(out_string)
    if stable_identity(out) != expected_identity:
        raise ValueError("Worker source/registration identity changed")
    evaluator = load_evaluation()
    registration = evaluator.load_registration(out)
    matches = [q for q in registration["requests"] if q["request_id"] == request_id]
    if len(matches) != 1:
        raise ValueError("Request not uniquely registered")
    q = matches[0]
    queries = [q0 for q0 in registration["queries"] if q0["condition_id"] == q["condition_id"]]
    if len(queries) != 1:
        raise ValueError("Query not uniquely registered")
    receipt = evaluator.read(out / "evaluation/outputs" / evaluator.NEW_LABEL / request_id / "receipt.json")
    destination = out / "evaluation/scores" / evaluator.NEW_LABEL / (request_id + ".json")
    cached_before = destination.exists()
    started = time.perf_counter()
    result = evaluator.score_one(out, q, receipt, queries[0], expected_identity["evaluation_registration_sha256"])
    if result["status"] != "PASS":
        raise RuntimeError("Scorer did not PASS: " + request_id + " | " + result.get("reason", result["status"]))
    if "torch" in sys.modules:
        raise RuntimeError("Geometry scoring unexpectedly imported torch")
    return {"status": "PASS", "request_id": request_id, "pid": os.getpid(), "cached_before": cached_before,
            "new_geometry_scans": 0 if cached_before else 1, "score_path": str(destination),
            "score_sha256": digest(destination), "seconds": time.perf_counter() - started, "model_calls": 0}


def choose_workers(snapshot, requested, remaining):
    if requested not in (1, 2):
        raise ValueError("Only one or two CPU workers are registered")
    cores = snapshot["cpu_affinity_count"] or 1
    average = (snapshot["load_average_1_5_15"] or [0])[0]
    available = snapshot["memory"].get("MemAvailable_bytes")
    if available is None:
        raise RuntimeError("Cannot schedule CPU workers without available-RAM observation")
    if available < 1024**3:
        raise RuntimeError("Less than 1 GiB available RAM; wait instead of dispatching")
    count = min(requested, remaining)
    reason = "Two workers / two threads each fit observed CPU and RAM headroom"
    if count > 1 and (cores - average < 6 or available < 2 * 1024**3):
        count = 1
        reason = "One CPU worker: retain CPU/RAM headroom under current measured load"
    return max(1, count), reason


def run(out, requested_workers=2, controller_owned=False):
    if not controller_owned:
        raise ValueError("Root dispatcher must explicitly start CPU scoring with --controller-owned")
    out = root(out)
    identity = stable_identity(out)
    if read(out / "audit/score_helper_identity.json") != identity:
        raise ValueError("CPU scoring helper identity differs from registration")
    evaluator = load_evaluation()
    registration = evaluator.load_registration(out)
    # All RAWs must exist, including historical read-only references, before any
    # geometry score starts. This also checks every model/request/hash contract.
    raw = evaluator.collect_outputs(out, require_complete=True)
    if len(raw) != 48:
        raise ValueError("Expected exactly 48 verified RAWs")
    requests = sorted(q["request_id"] for q in registration["requests"])
    if len(requests) != 12:
        raise ValueError("Only 12 NEWDEV requests are authorized")
    observed = resources()
    workers, reason = choose_workers(observed, requested_workers, len(requests))
    audit = out / "audit"
    tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = audit / "cpu_score_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    dispatch = {"status": "DISPATCHED", "identity": identity, "resources": observed,
                "requested_workers": requested_workers, "actual_workers": workers, "threads_per_worker": 2,
                "reason": reason, "registered_requests": requests, "historical_scans": 0}
    write(run_dir / "dispatch.json", dispatch)
    started = time.perf_counter()
    completed, pending = [], {}
    queue = iter(requests)
    try:
        # Spawn gives each worker a clean module/RNG namespace. At most the
        # chosen number of jobs are live; a failure stops dispatching new jobs.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for _ in range(workers):
                rid = next(queue, None)
                if rid is not None:
                    pending[pool.submit(score_request, str(out), rid, identity)] = rid
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]):
                    rid = pending.pop(future)
                    result = future.result()
                    completed.append(result)
                    write(run_dir / "progress.json", {"status": "RUNNING", "completed": len(completed), "registered": 12,
                                                      "results": completed, "seconds": time.perf_counter() - started}, freeze=False)
                    print(json.dumps({"stage": "parallel_score", **result, "completed": len(completed)}, allow_nan=False), flush=True)
                for _ in range(workers - len(pending)):
                    rid = next(queue, None)
                    if rid is not None:
                        pending[pool.submit(score_request, str(out), rid, identity)] = rid
        if len(completed) != 12 or len({r["request_id"] for r in completed}) != 12:
            raise ValueError("Incomplete new-score coverage")
        # Every new-row score now exists, so score_all can only verify its cache.
        # Historical rows follow the immutable historical-cache branch. The
        # unchanged aggregator computes summaries/diversity, not geometry scans.
        new_hashes = {r["score_path"]: r["score_sha256"] for r in completed}
        aggregated = evaluator.score_all(out)
        if any(digest(path) != sha for path, sha in new_hashes.items()):
            raise ValueError("Serial aggregation changed a parallel score")
        if aggregated["scored"] != 48:
            raise ValueError("48-row aggregation is incomplete")
        all_scores = evaluator.read(out / "evaluation/scores_all.json")["rows"]
        historical = [r for r in all_scores if r["label"] in evaluator.HISTORICAL_LABELS]
        if len(historical) != 36 or any(r.get("score_execution") != "VERIFIED_EXACT_DEFINITION_HISTORICAL_CACHE" or r.get("new_cpu_geometry_scans") != 0 for r in historical):
            raise ValueError("Historical results were not uniformly reused")
        result = {"status": "COMPLETE", "actual_workers": workers, "threads_per_worker": 2, "results": sorted(completed, key=lambda r: r["request_id"]),
                  "new_geometry_scans_this_invocation": sum(r["new_geometry_scans"] for r in completed), "historical_geometry_scans": 0,
                  "aggregation": aggregated, "model_calls": 0, "seconds": time.perf_counter() - started, "resources_after": resources()}
        write(run_dir / "completed.json", result)
        write(audit / "cpu_score_parallel_result.json", result, freeze=False)
        return {k: value for k, value in result.items() if k != "results"}
    except BaseException as exc:
        for future in pending:
            future.cancel()
        write(run_dir / "failure.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                                         "completed_results": completed, "pending_requests": sorted(pending.values()),
                                         "seconds": time.perf_counter() - started, "model_calls": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--controller-owned", action="store_true")
    args = parser.parse_args()
    result = register(args.out) if args.stage == "register" else run(args.out, args.workers, args.controller_owned)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
