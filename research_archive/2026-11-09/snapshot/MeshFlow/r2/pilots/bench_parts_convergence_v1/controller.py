"""Read-only CPU coordinator for the matched Bench convergence experiment.

This module makes plans and verifies resumable cursors. It NEVER starts a GPU
process. Root execution owns dispatch; ``status`` is the default CLI command.
Both continuation arms must reach the same fixed milestone before the common
stopping decision is made. A plateau in one arm does not shorten that arm.
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import subprocess
import time
from pathlib import Path

from .convergence import assess_plateau


VERSION = "bench_parts_matched_controller_cpu_v1"
ARMS = ("PLAIN", "SEAM")
SAFE_STATUSES = {"COMPLETE", "SEGMENT_COMPLETE", "SAFE_SEGMENT", "INTERRUPTED_SAFE",
                 "PAUSED_SAFE", "CHECKPOINTED", "READY"}
ACTIVE_STATUSES = {"LOADING", "TRAINING", "RUNNING", "PROBING", "SAVING"}
FAIL_STATUSES = {"BLOCKED", "OOM", "FAILED", "ERROR", "INTERRUPTED_UNSAFE"}


def read(path, *, timeout=2.0):
    """Bounded retries for metadata replacement visibility, never for jobs."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            if isinstance(exc, OSError) and exc.errno not in {
                errno.ENOENT, errno.EACCES, errno.EPERM, errno.EBUSY, errno.ESTALE
            }:
                raise
            if time.monotonic() >= deadline:
                raise
            if attempts == 0:
                print(json.dumps(dict(event="METADATA_READ_VISIBILITY_RETRY", path=str(path),
                                      error=repr(exc), timeout_seconds=timeout,
                                      scientific_job_retry=False)), flush=True)
            attempts += 1
            time.sleep(min(.02 * attempts, .1))


def resolve_path(path, root):
    text = str(path)
    if os.name != "nt":
        text = text.replace("E:\\MeshFlow", "/mnt/e/MeshFlow").replace("E:/MeshFlow", "/mnt/e/MeshFlow")
        text = text.replace("\\", "/")
    p = Path(text)
    return p if p.is_absolute() else Path(root)/p


def _step(payload):
    value = payload.get("global_step", payload.get("step"))
    if value is None or isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError("An explicit nonnegative global_step/step is required")
    return int(value)


def load_probe_history(root, arm):
    """Read registered statistics, not models; refuse conflicting duplicate steps."""
    root = Path(root)
    paths = set((root/"training"/arm/"fixed_loss").glob("*.json"))
    paths.update((root/"fixed_loss"/arm).glob("*.json"))
    by_step = {}
    for path in sorted(paths):
        payload = read(path)
        # Non-probe receipts may share the directory; record recognition is
        # structural, never based on a favourable numeric value.
        if "train" not in payload or "probe_manifest_sha256" not in payload:
            continue
        recorded_arm = payload.get("arm", arm)
        # Frozen COMMON worker records the actual Native model mode as PLAIN.
        # Directory ownership remains the training-stage identity; do not edit
        # already frozen workers or rewrite their historical probe JSON.
        if recorded_arm != arm and not (arm == "COMMON" and recorded_arm == "PLAIN"):
            raise ValueError("Probe arm identity mismatch: "+str(path))
        row = dict(payload)
        row["job_arm"] = arm
        row["recorded_model_mode"] = recorded_arm
        if arm == "COMMON" and recorded_arm == "PLAIN":
            row["arm_identity_note"] = "COMMON directory is stage identity; frozen payload arm=PLAIN denotes Native forward mode"
        row["step"] = _step(payload)
        compare = {k: v for k, v in row.items() if k not in ("source_path", "utc", "unix", "wall_seconds")}
        if row["step"] in by_step:
            old = by_step[row["step"]]
            if compare != old["comparison"]:
                raise ValueError("Conflicting fixed-probe records at step "+str(row["step"]))
            old["paths"].append(str(path))
        else:
            by_step[row["step"]] = {"row": row, "comparison": compare, "paths": [str(path)]}
    return [dict(item["row"], source_paths=item["paths"]) for _, item in sorted(by_step.items())]


def recovery_precheck(root, arm):
    """Check file/cursor safety without loading tensors or replaying an update.

    Worker checkpoint validation remains authoritative for tensor hashes,
    optimizer/RNG identity and pending optimizer calls. This lightweight check
    must not be represented as a complete checkpoint-content audit.
    """
    root = Path(root)
    folder = root/"training"/arm
    path = folder/"status.json"
    result = {"arm": arm, "status_path": str(path), "model_loaded": False,
              "content_hash_verified": False, "replay_authorized": False}
    if not folder.exists():
        result.update(state="ABSENT", safe_to_start=True, orphan_artifacts=[], step=None)
        return result
    try:
        payload = read(path)
    except FileNotFoundError:
        loose = [str(p) for p in (folder/"rolling.pt", folder/"ledger.jsonl", folder/"training_log.jsonl") if p.exists()]
        result.update(state="ABSENT" if not loose else "AMBIGUOUS_ORPHAN_ARTIFACTS",
                      safe_to_start=not loose, orphan_artifacts=loose, step=None)
        return result
    if payload.get("arm", arm) != arm:
        raise ValueError("Status arm mismatch")
    step = _step(payload)
    status = str(payload.get("status", "UNKNOWN")).upper()
    result.update(state=status, step=step, safe_to_start=False, pid=payload.get("pid"))
    if status in ACTIVE_STATUSES:
        result["action"] = "WAIT_OR_VERIFY_EXISTING_OWNER; never duplicate dispatch from a stale-looking status"
        return result
    if status in FAIL_STATUSES or status not in SAFE_STATUSES:
        result["action"] = "EXPLICIT_RECOVERY_REQUIRED; no automatic replay"
        return result
    checkpoint = payload.get("checkpoint", payload.get("checkpoint_path"))
    if isinstance(checkpoint, dict):
        cp_path = checkpoint.get("path")
        expected_sha = checkpoint.get("sha256")
    else:
        cp_path = checkpoint
        expected_sha = payload.get("checkpoint_sha256")
    cp = resolve_path(cp_path, root) if cp_path else folder/"rolling.pt"
    if not cp.exists() or cp.stat().st_size == 0:
        result.update(state="MISSING_RESUME_CHECKPOINT", checkpoint=str(cp))
        return result
    sidecar = cp.with_suffix(".json")
    if sidecar.exists():
        receipt = read(sidecar)
        if "step" in receipt or "global_step" in receipt:
            if _step(receipt) != step:
                result.update(state="CHECKPOINT_CURSOR_MISMATCH", checkpoint=str(cp), receipt=receipt)
                return result
        if expected_sha and receipt.get("sha256") and expected_sha != receipt["sha256"]:
            result.update(state="CHECKPOINT_RECEIPT_HASH_MISMATCH", checkpoint=str(cp))
            return result
        expected_sha = expected_sha or receipt.get("sha256")
    result.update(safe_to_start=True, checkpoint=str(cp), checkpoint_bytes=cp.stat().st_size,
                  expected_sha256=expected_sha, checkpoint_sidecar=str(sidecar) if sidecar.exists() else None,
                  worker_must_verify="Exact saved global cursor, complete AdamW/RNG, registration and checkpoint hash; unresolved calls forbid replay")
    return result


def paired_stage_decision(histories, *, stage_start_step, min_updates, max_updates,
                          probe_interval, major_parts=()):
    """Pure decision from completed fixed probes, with one shared endpoint."""
    missing = [a for a in ARMS if not histories.get(a)]
    if missing:
        return {"status": "WAIT_FOR_BOTH_PROBE_HISTORIES", "missing": missing,
                "stop_training": False, "choose_best_checkpoint": False}
    latest = {a: _step(histories[a][-1]) for a in ARMS}
    if len(set(latest.values())) != 1:
        target = max(latest.values())
        return {"status": "WAIT_FOR_MATCHED_MILESTONE", "latest": latest,
                "target_global_step": target, "lagging_arms": [a for a in ARMS if latest[a] < target],
                "stop_training": False, "choose_best_checkpoint": False}
    assessments = {a: assess_plateau(histories[a], min_updates=min_updates, max_updates=max_updates,
                                     stage_start_step=stage_start_step, probe_interval=probe_interval,
                                     major_parts=major_parts) for a in ARMS}
    current = latest["PLAIN"]
    both = all(a["status"] == "PLATEAU" for a in assessments.values())
    exhausted = current-stage_start_step >= max_updates
    status = "PAIRED_LOSS_PLATEAU" if both else "BUDGET_EXHAUSTED_NOT_CONVERGED" if exhausted else "CONTINUE_BOTH"
    target = None if both or exhausted else min(current+probe_interval, stage_start_step+max_updates)
    return {"status": status, "current_global_step": current,
            "stage_updates": current-stage_start_step, "stop_training": both or exhausted,
            "next_common_target": target, "assessments": assessments,
            "choose_best_checkpoint": False, "quality_promotion": False,
            "interpretation": "Both arms keep the same update budget; DEV warnings do not select an endpoint"}


def resource_dispatch_plan(snapshot, *, authorized_pending_tasks, active_workers=0,
                           observed_worker_process_peak_mib=None, observed_worker_ram_mib=None,
                           cpu_threads_per_worker=2, process_peak_scope=None):
    """Resource-based starting plan; returns counts, never launches processes.

    snapshot: gpu_total_mib/free_mib/gpu_utilization, cpu_logical/available_cores,
    memory_available_mib/cpu_utilization; include gpu_processes as evidence.
    PyTorch allocated/reserved alone is not a process VRAM peak.
    """
    pending = int(authorized_pending_tasks)
    if pending < 0 or active_workers < 0 or cpu_threads_per_worker < 1:
        raise ValueError("Invalid worker or pending-task counts")
    total = float(snapshot["gpu_total_mib"])
    free = float(snapshot["gpu_free_mib"])
    headroom = max(2048., .1*total)
    reasons = []
    if pending == 0:
        slots = 0; reasons.append("NO_REMAINING_AUTHORIZED_TASK")
    elif free <= headroom:
        slots = 0; reasons.append("NO_ADDITIONAL_MEMORY_HEADROOM")
    elif observed_worker_process_peak_mib is None:
        slots = 1 if active_workers == 0 else 0
        reasons.append("MEASURE_FIRST_ALREADY_AUTHORIZED_TASK_THROUGH_BACKWARD_AND_OPTIMIZER; no synthetic probe")
    else:
        peak = float(observed_worker_process_peak_mib)
        if peak <= 0 or not process_peak_scope:
            raise ValueError("Positive comparable process peak with explicit scope required")
        if process_peak_scope in ("pytorch_allocated_only", "pytorch_reserved_only"):
            raise ValueError("Allocator-only memory is not a comparable process peak")
        slots = int(max(0., free-headroom)//(1.2*peak))
    available_cores = max(0, int(snapshot.get("available_cores", snapshot.get("cpu_logical", 1)))-active_workers*cpu_threads_per_worker)
    slots = min(slots, available_cores//cpu_threads_per_worker, pending)
    if observed_worker_ram_mib:
        slots = min(slots, int(float(snapshot["memory_available_mib"])//(1.2*float(observed_worker_ram_mib))))
    if float(snapshot.get("cpu_utilization", 0)) >= 90:
        slots = 0; reasons.append("CPU_MATCHING_AUDIT_LOAD_HIGH")
    if float(snapshot.get("gpu_utilization", 0)) >= 95 and active_workers > 0:
        slots = 0; reasons.append("GPU_COMPUTE_ALREADY_SATURATED; concurrency has no measured throughput benefit")
    return {"schema": VERSION, "additional_workers": slots,
            "launch_policy": "Incrementally launch at most one, remeasure representative process peak before the next",
            "authorized_pending_tasks": pending, "active_workers": active_workers,
            "headroom_mib": headroom, "safety_factor": 1.2,
            "observed_worker_process_peak_mib": observed_worker_process_peak_mib,
            "process_peak_scope": process_peak_scope, "snapshot": snapshot, "reasons": reasons,
            "stops_unrelated_processes": False, "changes_batch_precision_or_budget": False}


def resource_snapshot():
    """Read-only OS/GPU telemetry; querying nvidia-smi is not model execution."""
    def command(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": repr(exc)}
    gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    procs = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    result = {"unix": time.time(), "gpu_raw": gpu, "gpu_processes": procs,
              "cpu_logical": os.cpu_count(), "model_calls": 0}
    if gpu.get("returncode") == 0:
        values = [x.strip() for x in gpu["stdout"].splitlines()[0].split(",")]
        result.update(gpu_name=values[0], gpu_total_mib=float(values[1]), gpu_used_mib=float(values[2]),
                      gpu_free_mib=float(values[3]), gpu_utilization=float(values[4]))
    try:
        import psutil
        result.update(memory_available_mib=psutil.virtual_memory().available/(1024**2),
                      cpu_utilization=psutil.cpu_percent(interval=.1),
                      load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else None)
    except ImportError:
        result["cpu_RAM_detail_status"] = "NA_PSUTIL_UNAVAILABLE; do not install automatically"
    return result


def experiment_status(root, *, common_updates=12000, min_stage_updates=3000,
                      max_stage_updates=9000, probe_interval=1000, major_parts=()):
    root = Path(root)
    registered = root/"registration.json"
    if registered.exists():
        reg = read(registered)
        if reg.get("status") != "FROZEN":
            raise ValueError("Status plans require the frozen experiment registration")
        declared = (reg["common_updates"], reg["minimum_stage_updates"],
                    reg["maximum_stage_updates"], reg["stopping"]["probe_interval"])
        if declared != (common_updates, min_stage_updates, max_stage_updates, probe_interval):
            raise ValueError("Requested controller budget differs from frozen registration")
        if not major_parts:
            major_parts = tuple(reg["major_parts"])
        elif set(major_parts) != set(reg["major_parts"]):
            raise ValueError("Major-part convergence guard differs from registration")
        if (reg["stopping"]["absolute_relative_change_lt"], reg["stopping"]["major_part_worsening_gt"],
                reg["stopping"]["major_part_worsening_intervals"]) != (.02, .05, 3):
            raise ValueError("Convergence rule differs; do not silently reuse default thresholds")
    if common_updates <= 0 or common_updates % probe_interval or max_stage_updates % probe_interval:
        raise ValueError("Registered budgets must land on fixed probe milestones")
    states = {a: recovery_precheck(root, a) for a in ("COMMON", *ARMS)}
    output = {"schema": VERSION, "root": str(root), "states": states,
              "GPU_dispatch_enabled": False, "automatic_model_calls": 0,
              "budgets": {"common_updates": common_updates, "minimum_stage_updates": min_stage_updates,
                          "maximum_stage_updates": max_stage_updates, "probe_interval": probe_interval},
              "proposed_commands": []}
    common = states["COMMON"]
    if common["state"] == "ABSENT":
        output["phase"] = "COMMON_NOT_STARTED"
        output["proposed_commands"].append(["--root", str(root), "--arm", "COMMON", "--target", str(common_updates)])
        return output
    if common["step"] is None or common["step"] < common_updates:
        output["phase"] = "COMMON_IN_PROGRESS_OR_RECOVERY_REQUIRED"
        if common["safe_to_start"]:
            output["proposed_commands"].append(["--root", str(root), "--arm", "COMMON", "--target", str(common_updates), "--resume", common["checkpoint"]])
        return output
    if common["step"] != common_updates or not common["safe_to_start"]:
        output["phase"] = "COMMON_FORK_NOT_VERIFIED"
        return output
    histories = {a: [h for h in load_probe_history(root, a) if _step(h) >= common_updates] for a in ARMS}
    output["probe_steps"] = {a: [h["step"] for h in rows] for a, rows in histories.items()}
    if all(histories.values()):
        decision = paired_stage_decision(histories, stage_start_step=common_updates,
                                        min_updates=min_stage_updates, max_updates=max_stage_updates,
                                        probe_interval=probe_interval, major_parts=major_parts)
    else:
        decision = {"status": "PAIRED_STAGE_NOT_READY", "stop_training": False,
                    "next_common_target": common_updates+probe_interval}
    output["decision"] = decision
    if decision.get("stop_training"):
        # Stale probes cannot close a run whose worker cursor disagrees.
        endpoint = decision["current_global_step"]
        if all(states[a]["step"] == endpoint and states[a]["safe_to_start"] for a in ARMS):
            output["phase"] = "TRAINING_COMPLETE_READY_FOR_FIXED_EVALUATION"
        else:
            output["phase"] = "PROBE_WORKER_CURSOR_MISMATCH"
        return output
    active = [a for a in ARMS if states[a]["state"] in ACTIVE_STATUSES]
    output["phase"] = "PAIRED_STAGE_ACTIVE" if active else "PAIRED_STAGE_READY_OR_RECOVERY_REQUIRED"
    target = decision.get("next_common_target", decision.get("target_global_step"))
    if target is None:
        current = max([common_updates]+[states[a]["step"] for a in ARMS if states[a]["step"] is not None])
        target = min(common_updates+max_stage_updates, common_updates+probe_interval*math.ceil((current-common_updates)/probe_interval))
    for arm in ARMS:
        state = states[arm]
        base = ["--root", str(root), "--arm", arm, "--target", str(target)]
        if state["state"] == "ABSENT":
            output["proposed_commands"].append(base+["--fork", common["checkpoint"]])
        elif state["safe_to_start"] and state["step"] < target:
            output["proposed_commands"].append(base+["--resume", state["checkpoint"]])
    output["dispatch_note"] = "Commands are reviewable plans only; root must remeasure resources and dispatch incrementally"
    return output


def analytic_fixture_checks():
    def history(value):
        p = lambda x: {"sse": x, "count": 1}
        return [{"step": 12000+i*1000, "probe_manifest_sha256": "fixed",
                 "train": {"ALL_free": p(value**i), "bulk": p(1.), "seam_tip": p(1.), "parts": {}}}
                for i in range(4)]
    both = {"PLAIN": history(1), "SEAM": history(1)}
    args = dict(stage_start_step=12000, min_updates=3000, max_updates=9000, probe_interval=1000)
    assert paired_stage_decision(both, **args)["status"] == "PAIRED_LOSS_PLATEAU"
    mixed = {"PLAIN": history(1), "SEAM": history(.95)}
    r = paired_stage_decision(mixed, **args)
    assert r["status"] == "CONTINUE_BOTH" and r["next_common_target"] == 16000
    args["max_updates"] = 3000
    assert paired_stage_decision(mixed, **args)["status"] == "BUDGET_EXHAUSTED_NOT_CONVERGED"
    short = {"PLAIN": history(1), "SEAM": history(1)[:-1]}
    assert paired_stage_decision(short, **args)["status"] == "WAIT_FOR_MATCHED_MILESTONE"
    snap = {"gpu_total_mib": 16303, "gpu_free_mib": 13000, "gpu_utilization": 5,
            "cpu_logical": 24, "cpu_utilization": 10, "memory_available_mib": 60000}
    r = resource_dispatch_plan(snap, authorized_pending_tasks=2, observed_worker_process_peak_mib=4000,
                               process_peak_scope="observed_single_worker_process_estimate_including_context")
    assert r["additional_workers"] == 2 and r["headroom_mib"] == 2048
    assert resource_dispatch_plan(snap, authorized_pending_tasks=2)["additional_workers"] == 1
    snap["gpu_free_mib"] = 3000
    assert resource_dispatch_plan(snap, authorized_pending_tasks=2, observed_worker_process_peak_mib=4000,
                                   process_peak_scope="observed_process")["additional_workers"] == 0
    return {"status": "PASS", "schema": VERSION,
            "checks": ["both_plateau_required", "one_plateau_keeps_equal_updates", "shared_budget_cap",
                       "milestone_barrier", "memory_headroom_slots", "first_authorized_task_only", "zero_slots_wait"],
            "model_calls": 0, "GPU_dispatches": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("status", "self-test"), default="status")
    parser.add_argument("--root", type=Path, default=Path("experiments/r2_bench_parts_convergence_v1"))
    parser.add_argument("--common-updates", type=int, default=12000)
    parser.add_argument("--min-stage-updates", type=int, default=3000)
    parser.add_argument("--max-stage-updates", type=int, default=9000)
    parser.add_argument("--probe-interval", type=int, default=1000)
    parser.add_argument("--major-part", action="append", default=[])
    parser.add_argument("--resources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analytic_fixture_checks() if args.command == "self-test" else experiment_status(
        args.root, common_updates=args.common_updates, min_stage_updates=args.min_stage_updates,
        max_stage_updates=args.max_stage_updates, probe_interval=args.probe_interval, major_parts=args.major_part)
    if args.resources:
        result["resources"] = resource_snapshot()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError("Preserve existing controller evidence: "+str(args.output))
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
