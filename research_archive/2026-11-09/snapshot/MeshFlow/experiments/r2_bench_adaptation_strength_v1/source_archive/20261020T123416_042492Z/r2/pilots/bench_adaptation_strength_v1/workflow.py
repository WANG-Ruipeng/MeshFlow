"""Finite, recoverable dispatch controller; default status never starts work.

The controller adopts existing A/B/RAW workers. It may start the one remaining C
arm, the registered eight RAW labels, one D0 per output after the 224 RAW barrier,
and one CPU pool at a time. It never restarts a failed scientific attempt.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import ROOT, OUT, ARMS, read, resolve, sha, write_new, atomic_write, resources
from . import resume as R

FLOW = OUT / "workflow"
PEAK_MIB = {"train": 4500., "raw": 2800., "post": 3500.}
GPU_STAGES = tuple(PEAK_MIB)
FINAL_TRAIN = 1500
CPU_STAGES = ("cpu_raw", "cpu_post", "collect", "gallery")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def alive(job):
    if not R.alive(job):
        return False
    import psutil
    try:
        return psutil.Process(job["pid"]).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def records():
    paths = sorted((OUT / "dispatch").glob("job_*.json"))
    paths += sorted((FLOW / "jobs").glob("job_*.json"))
    return [(path, read(path)) for path in paths]


def label_specs():
    return read(OUT / "registration.json")["evaluation_labels"]


def completion_path(stage, identity):
    if stage == "train":
        return OUT / "training" / identity / "status.json"
    if stage in ("raw", "post"):
        subset = label_specs()[identity]["subset"]
        return OUT / "evaluation" / identity / (stage + "_complete_" + subset + ".json")
    if stage in ("cpu_raw", "cpu_post"):
        mode = stage.removeprefix("cpu_")
        return OUT / "evaluation/completions" / (mode + "__" + identity + ".json")
    if stage == "collect":
        return OUT / "results/completion.json"
    if stage == "gallery":
        return OUT / "gallery/completion.json"
    raise ValueError("Unknown registered stage: " + stage)


def complete(stage, identity):
    p = completion_path(stage, identity)
    if not p.exists():
        return False
    value = read(p)
    if stage == "train":
        return value.get("status") == "COMPLETE" and int(value.get("local_step", -1)) == FINAL_TRAIN
    if stage in ("raw", "post"):
        return value.get("status") == "COMPLETE" and value.get("outputs") == label_specs()[identity]["requests"]
    if stage == "gallery":
        return value.get("status") == "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW" and value.get("sheets") == 144
    return value.get("status") == "COMPLETE"


def ownership(stage, identity, jobs):
    matches = [(p, j) for p, j in jobs if (j["stage"], j["identity"]) == (stage, identity)]
    return matches


def blockers(jobs):
    """Dead incomplete jobs are evidence boundaries, not requests for replay."""
    issues = []
    for p, j in jobs:
        stage, identity = j["stage"], j["identity"]
        if stage not in (*GPU_STAGES, *CPU_STAGES):
            issues.append(dict(kind="UNKNOWN_JOB", path=str(p), stage=stage))
            continue
        if complete(stage, identity):
            continue
        if not alive(j):
            issues.append(dict(kind="DEAD_INCOMPLETE_WORKER", stage=stage, identity=identity,
                               job=str(p), log=j.get("log"), automatic_retry=False))
    for arm in ARMS:
        p = OUT / "training" / arm / "status.json"
        if p.exists():
            s = read(p)
            if s.get("status") in ("FAILED", "BLOCKED", "INTERRUPTED_UNSAFE", "INTERRUPTED_SAFE", "SAFE_SEGMENT"):
                issues.append(dict(kind="TRAINING_NOT_AUTOMATICALLY_RESUMABLE", arm=arm,
                                   status=s.get("status"), path=str(p)))
    # A/B are already owned by the authorized manual initial dispatch. Absence
    # is not license for this remaining-stage controller to start them again.
    for arm in ARMS[:2]:
        if not complete("train", arm) and not ownership("train", arm, jobs):
            issues.append(dict(kind="MISSING_ORIGINAL_A_B_DISPATCH", arm=arm))
    # A worker may still be exiting after writing failure. Stop new dispatches
    # immediately while leaving all active and unrelated processes untouched.
    patterns = (
        "training/*/attempts/*/failure.json",
        "evaluation/*/sampling_attempt_*/failure.json",
        "evaluation/*/postprocess_attempt_*/failure.json",
        "cpu/*/*/failure.json",
    )
    for pattern in patterns:
        for p in OUT.glob(pattern):
            issues.append(dict(kind="PRESERVED_FAILURE_RECORD", path=str(p)))
    for arm in ARMS:
        directory = OUT / "training" / arm
        if directory.exists() and not complete("train", arm) and not ownership("train", arm, jobs):
            issues.append(dict(kind="UNOWNED_TRAINING_ATTEMPT", arm=arm, path=str(directory)))
    for label in label_specs():
        for stage, paths in (
            ("raw", list((OUT / "evaluation" / label).glob("sampling_attempt_*"))),
            ("post", list((OUT / "evaluation" / label).glob("postprocess_attempt_*"))),
            ("cpu_raw", list((OUT / "evaluation/scores" / label).glob("*/RAW"))),
            ("cpu_post", list((OUT / "cpu" / label).glob("*"))),
        ):
            if paths and not complete(stage, label) and not ownership(stage, label, jobs):
                issues.append(dict(kind="UNOWNED_PARTIAL_ATTEMPT", stage=stage, identity=label,
                                   paths=[str(p) for p in paths], automatic_retry=False))
    return issues


def status():
    jobs = records()
    specs = label_specs()
    return dict(utc=utc(), experiment="R2_BENCH_ADAPTATION_STRENGTH_V1",
        controller="READ_ONLY_STATUS", blockers=blockers(jobs),
        training={a: dict(complete=complete("train", a),
                         status=read(OUT / "training" / a / "status.json") if (OUT / "training" / a / "status.json").exists() else None)
                  for a in ARMS},
        labels={lab: dict(expected=s["requests"], RAW=complete("raw", lab), POST=complete("post", lab),
                          RAW_scored=complete("cpu_raw", lab), FINAL_scored=complete("cpu_post", lab))
                for lab, s in specs.items()},
        jobs=[dict(path=str(p), stage=j["stage"], identity=j["identity"], pid=j["pid"],
                   alive=alive(j), complete=complete(j["stage"], j["identity"]), log=j.get("log")) for p, j in jobs],
        collected=complete("collect", "ALL"), rendered=complete("gallery", "ALL"),
        evaluation_complete=(OUT / "completion.json").exists())


def gpu_numbers(snapshot):
    gpu = snapshot["gpu"]
    if gpu["returncode"]:
        raise RuntimeError("GPU resource query failed")
    fields = gpu["stdout"].strip().splitlines()[0].split(",")
    total, used, free, util = (float(fields[i].strip().split()[0]) for i in (1, 2, 3, 4))
    return dict(total_mib=total, used_mib=used, free_mib=free, utilization=util)


def representative(job):
    """Require actual formal work, including backward/optimizer for training."""
    stage, identity = job["stage"], job["identity"]
    if stage == "train":
        p = OUT / "training" / identity / "status.json"
        if not p.exists():
            return False
        s = read(p)
        last = s.get("last", {})
        return int(s.get("local_step", 0)) >= 1 and last.get("peak_reserved_bytes", 0) > 0
    if stage == "raw":
        return any((OUT / "evaluation" / identity).glob("*/raw/receipt.json"))
    if stage == "post":
        return any((OUT / "evaluation" / identity).glob("*/post/proposal_receipt.json"))
    return True


def gpu_gate(stage, jobs, snapshot, history):
    g = gpu_numbers(snapshot)
    now = time.time()
    current = [j for _, j in jobs if j["stage"] in GPU_STAGES and alive(j)]
    latest_start = max((j["process_created"] for _, j in jobs if j["stage"] in GPU_STAGES), default=0.)
    headroom = max(2048., .1 * g["total_mib"])
    peak = PEAK_MIB[stage]
    slots = int(max(0., g["free_mib"] - headroom) // (1.2 * peak))
    checks = dict(stage=stage, conservative_process_peak_mib=peak,
        headroom_mib=headroom, peak_margin=1.2, incremental_slots=slots,
        RAM_available=snapshot["RAM_available"], GPU=g,
        seconds_since_latest_GPU_launch=now-latest_start,
        active_owned_GPU_workers=len(current),
        unmeasured_active=[dict(stage=j["stage"], identity=j["identity"], pid=j["pid"])
                           for j in current if not representative(j)])
    if slots < 1:
        return False, "VRAM_HEADROOM_WAIT", checks
    if snapshot["RAM_available"] < 8 * (1 << 30):
        return False, "RAM_WAIT", checks
    if now - latest_start < 30:
        return False, "MODEL_LOADING_SPACING_WAIT", checks
    if checks["unmeasured_active"]:
        return False, "WAIT_FOR_REPRESENTATIVE_FORMAL_PEAK", checks
    recent = [(t, u) for t, u in history if now-t <= 75]
    sustained = len(recent) >= 3 and recent[-1][0] - recent[0][0] >= 30 and all(u > 85 for _, u in recent)
    if sustained or (current and g["utilization"] > 85):
        return False, "GPU_COMPUTE_BUSY_WAIT", checks
    return True, "PASS", checks


def cpu_workers(snapshot, jobs, maximum):
    ntrain = sum(j["stage"] == "train" and alive(j) for _, j in jobs)
    logical = int(snapshot["CPU_cores"])
    # Keep cores for OT and user work. The scorer also measures its first real
    # authorized cell and may reduce this cap based on RSS/load.
    cap = min(maximum, 4 if ntrain else 6, max(1, logical - 2 - 2*ntrain))
    if snapshot["RAM_available"] < 8 * (1 << 30) or snapshot["CPU_percent"] > 90:
        return 0
    return max(1, min(cap, int(snapshot["RAM_available"] // (2 * (1 << 30)))))


def event(kind, **fields):
    FLOW.mkdir(parents=True, exist_ok=True)
    value = dict(utc=utc(), event=kind)
    value.update(fields)
    with (FLOW / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def spawn_cpu(stage, identity, workers, snapshot):
    if stage in ("cpu_raw", "cpu_post"):
        mode = stage.removeprefix("cpu_")
        argv = ["evaluation", mode, "--labels", identity, "--workers", str(workers)]
    elif stage == "collect":
        argv = ["evaluation", "collect"]
    elif stage == "gallery":
        argv = ["gallery", "run", "--workers", str(workers)]
    else:
        raise ValueError(stage)
    stamp = str(time.time_ns())
    folder = FLOW / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / (stage + "_" + identity + "_" + stamp + ".log")
    command = [sys.executable, "-B", "-u", "-m", __package__ + "." + argv[0], *argv[1:]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/"r2/src"))
    with log.open("x", encoding="utf-8") as f:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    import psutil
    record = dict(stage=stage, identity=identity, pid=p.pid,
                  process_created=psutil.Process(p.pid).create_time(),
                  argv=command, log=str(log), resources=snapshot, workers_cap=workers,
                  CPU_pool_count=1, model_calls=0, automatic_retry=False)
    write_new(folder / ("job_" + stage + "_" + identity + "_" + stamp + ".json"), record)
    return record


def eligible_gpu(jobs):
    specs = label_specs()
    pending = []
    # Only C is new training for this controller. A/B are adopted, never replayed.
    arm = "C_OFFICIAL_LR2E6"
    if not complete("train", arm) and not ownership("train", arm, jobs):
        pending.append(("train", arm))
    for label, spec in specs.items():
        export = resolve(spec["generator_path"])
        if (not complete("raw", label) and not ownership("raw", label, jobs)
                and export.exists() and export.with_suffix(".json").exists()):
            pending.append(("raw", label))
    if all(complete("raw", label) for label in specs):
        for label in specs:
            if not complete("post", label) and not ownership("post", label, jobs):
                pending.append(("post", label))
    return pending


def eligible_cpu(jobs):
    if any(j["stage"] in CPU_STAGES and alive(j) for _, j in jobs):
        return None
    specs = label_specs()
    # Final processing is valuable once D0 finishes; RAW scoring can overlap
    # the earlier registered GPU work without changing its input/query budget.
    for stage, dependency in (("cpu_post", "post"), ("cpu_raw", "raw")):
        for label in specs:
            if complete(dependency, label) and not complete(stage, label) and not ownership(stage, label, jobs):
                return stage, label
    all_scored = all(complete(stage, label) for stage in ("cpu_raw", "cpu_post") for label in specs)
    if all_scored and not complete("collect", "ALL") and not ownership("collect", "ALL", jobs):
        return "collect", "ALL"
    if complete("collect", "ALL") and not complete("gallery", "ALL") and not ownership("gallery", "ALL", jobs):
        return "gallery", "ALL"
    return None


@contextlib.contextmanager
def controller_lock():
    if os.name != "posix":
        raise RuntimeError("Execute this controller in the maintained WSL environment")
    import fcntl
    import psutil
    FLOW.mkdir(parents=True, exist_ok=True)
    with (FLOW / "controller.lock").open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another workflow controller owns dispatch")
        atomic_write(FLOW / "owner.json", dict(pid=os.getpid(),
            process_created=psutil.Process().create_time(), utc=utc(), automatic_retry=False))
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def execute(poll_seconds=15, max_cpu_workers=6):
    if not 5 <= poll_seconds <= 60 or not 1 <= max_cpu_workers <= 6:
        raise ValueError("Poll must be 5..60s; CPU cap 1..6")
    with controller_lock():
        run_id = str(time.time_ns())
        start = time.perf_counter()
        history = []
        previous_wait = None
        event("CONTROLLER_STARTED", run_id=run_id, source_sha256=sha(__file__),
              adopts_existing_GPU_workers=True, no_training_replay=True, max_cpu_workers=max_cpu_workers)
        try:
            while True:
                jobs = records()
                issues = blockers(jobs)
                if issues:
                    result = dict(status="BLOCKED", run_id=run_id, utc=utc(), reasons=issues,
                                  running_workers_left_untouched=[dict(stage=j["stage"], identity=j["identity"], pid=j["pid"])
                                                                 for _, j in jobs if alive(j)],
                                  no_automatic_retry=True)
                    atomic_write(FLOW / "status.json", result)
                    write_new(FLOW / ("blocked_" + str(time.time_ns()) + ".json"), result)
                    event("BLOCKED", **result)
                    return result
                if (all(complete("train", arm) for arm in ARMS)
                        and complete("collect", "ALL") and complete("gallery", "ALL")):
                    snapshot = status()
                    completions = [completion_path("train", a) for a in ARMS]
                    completions += [completion_path(stage, lab) for stage in ("raw", "post", "cpu_raw", "cpu_post")
                                    for lab in label_specs()]
                    completions += [completion_path(stage, "ALL") for stage in ("collect", "gallery")]
                    result = dict(status="COMPLETE_COMPUTE_PENDING_ACTUAL_VISUAL_REVIEW_AND_REPORT",
                                  run_id=run_id, utc=utc(), seconds_this_controller=time.perf_counter()-start,
                                  updates=4500, RAW=224, D0=224, scored_rows=672,
                                  rendered_sheets=144, no_default_change=True,
                                  automatic_extension=False, current=snapshot,
                                  completion_receipts={str(p): sha(p) for p in completions},
                                  workflow_source_sha256=sha(__file__),
                                  registration_sha256=sha(OUT / "registration.json"))
                    atomic_write(FLOW / "status.json", result)
                    path = FLOW / "completion.json"
                    if not path.exists():
                        write_new(path, result)
                    event("COMPUTE_COMPLETE", run_id=run_id, elapsed_seconds=result["seconds_this_controller"])
                    return result
                snapshot = resources()
                g = gpu_numbers(snapshot)
                now = time.time()
                history.append((now, g["utilization"]))
                history = [(t, u) for t, u in history if now-t <= 90]
                pending_gpu = eligible_gpu(jobs)
                launched = None
                waits = []
                for stage, identity in pending_gpu:
                    # A new measurement is taken again by resume.execute directly
                    # before Popen. Our gate additionally enforces load spacing.
                    allowed, reason, check = gpu_gate(stage, jobs, snapshot, history)
                    if not allowed:
                        waits.append(dict(stage=stage, identity=identity, reason=reason, checks=check))
                        continue
                    event("GPU_DISPATCH_INTENT", stage=stage, identity=identity, checks=check, resources=snapshot)
                    try:
                        record = R.execute(stage, identity, False, PEAK_MIB[stage])
                    except RuntimeError as exc:
                        if str(exc).startswith("RESOURCE_WAIT:"):
                            waits.append(dict(stage=stage, identity=identity, reason=str(exc)))
                            break
                        raise
                    launched = dict(stage=stage, identity=identity, pid=record.get("pid"))
                    event("GPU_DISPATCHED", **launched)
                    break  # No simultaneous model-loading burst.
                jobs = records()
                cpu_task = eligible_cpu(jobs)
                if cpu_task:
                    # This pool is independent of GPU parameters/RNG/writers.
                    cpu_snapshot = resources()
                    workers = cpu_workers(cpu_snapshot, jobs, max_cpu_workers)
                    if workers:
                        stage, identity = cpu_task
                        event("CPU_DISPATCH_INTENT", stage=stage, identity=identity, workers=workers, resources=cpu_snapshot)
                        record = spawn_cpu(stage, identity, workers, cpu_snapshot)
                        event("CPU_DISPATCHED", stage=stage, identity=identity, pid=record["pid"], workers=workers)
                live = [dict(stage=j["stage"], identity=j["identity"], pid=j["pid"]) for _, j in records() if alive(j)]
                value = dict(status="RUNNING", run_id=run_id, utc=utc(), active=live,
                             dispatched_GPU=launched, GPU_waits=waits,
                             completed_training=[a for a in ARMS if complete("train", a)],
                             completed_RAW=[l for l in label_specs() if complete("raw", l)],
                             completed_POST=[l for l in label_specs() if complete("post", l)],
                             current_resources=snapshot, no_automatic_retry=True)
                atomic_write(FLOW / "status.json", value)
                wait_key = json.dumps([(w["stage"], w["identity"], w["reason"]) for w in waits])
                if wait_key != previous_wait:
                    event("DISPATCH_WAIT_CHANGE", waits=waits)
                    previous_wait = wait_key
                time.sleep(poll_seconds)
        except BaseException as exc:
            result = dict(status="BLOCKED_CONTROLLER", run_id=run_id, utc=utc(), error=repr(exc),
                          traceback=traceback.format_exc(), science_workers_not_terminated=True,
                          no_automatic_retry=True)
            atomic_write(FLOW / "status.json", result)
            write_new(FLOW / ("controller_failure_" + str(time.time_ns()) + ".json"), result)
            event("CONTROLLER_EXCEPTION", error=repr(exc), run_id=run_id)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--max-cpu-workers", type=int, default=6)
    a = p.parse_args()
    result = execute(a.poll_seconds, a.max_cpu_workers) if a.execute else status()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
