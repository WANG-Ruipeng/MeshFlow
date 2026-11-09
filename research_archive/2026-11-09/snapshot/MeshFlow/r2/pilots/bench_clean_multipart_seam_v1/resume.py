"""Recoverable entry point: inspect by default; explicit bounded stage dispatch.

Workers run in detached sessions with durable logs. Failure never triggers an
implicit retry, checkpoint rollback, or new scientific budget.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, OUT, read, sha, write_new, atomic_write, resources

LABELS = ("NEW_HEAD", "OLD_S15000_LAYOUT")


def status():
    t = read(OUT / "training/status.json") if (OUT / "training/status.json").exists() else {}
    return dict(experiment="R2_BENCH_CLEAN_MULTIPART_SEAM_V1",
                training=t.get("status", "NOT_STARTED"), local_updates=t.get("local_step", 0),
                target=1500, historical_complete=(OUT / "historical_completion.json").exists(),
                RAW={x: len(list((OUT / "evaluation" / x).glob("*/raw/receipt.json"))) for x in LABELS},
                POST={x: len(list((OUT / "evaluation" / x).glob("*/post/receipt.json"))) for x in LABELS},
                report=(OUT / "REPORT.md").exists(),
                completion=(OUT / "completion.json").exists())


def alive(record):
    import psutil
    try:
        p = psutil.Process(record["pid"])
        return p.is_running() and abs(p.create_time() - record["process_created"]) < 2
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def execute(stage, *, resume_training=False):
    if stage == "train":
        if status()["training"] == "COMPLETE":
            return status()
        if not (OUT / "training_gate.json").exists():
            raise RuntimeError("Frozen training gate missing")
        if resume_training and status()["training"] not in ("INTERRUPTED_SAFE", "SAFE_SEGMENT"):
            raise RuntimeError("Only exact safely saved training state can resume")
        if not resume_training and status()["training"] != "NOT_STARTED":
            raise RuntimeError("Existing training attempt requires review, not implicit restart")
        command = ["training", "--execute"] + (["--resume"] if resume_training else [])
    elif stage in ("raw-new", "raw-old", "post-new", "post-old"):
        if status()["training"] != "COMPLETE":
            raise RuntimeError("Final training endpoint required")
        label = LABELS[0 if stage.endswith("new") else 1]
        command = ["sample" if stage.startswith("raw") else "post_d0", "--label", label]
    else:
        raise ValueError(stage)
    jobs = OUT / "dispatch"
    jobs.mkdir(parents=True, exist_ok=True)
    records = [read(p) for p in jobs.glob("job_*.json")]
    if any(alive(r) and r["stage"] == stage for r in records):
        raise RuntimeError("This stage already has an active owned writer")
    snapshot = resources()
    gpu_line = snapshot["gpu"]["stdout"].strip().splitlines()[0].split(",")
    free_mib = float(gpu_line[3].strip().split()[0])
    total_mib = float(gpu_line[1].strip().split()[0])
    headroom = max(2048., .1 * total_mib)
    # No reliable per-process WDDM reading: first authorized worker is observed.
    # Additional GPU dispatch remains an explicit measured root decision.
    minimum = headroom + (4096 if stage == "train" else 3500 if stage.startswith("post") else 2800)
    if free_mib < minimum:
        raise RuntimeError(f"RESOURCE_WAIT: free={free_mib}MiB < required initial envelope={minimum}MiB")
    if stage == "train":
        go = OUT / "training_dispatch_go.json"
        if not go.exists():
            write_new(go, dict(authorized=True, updates=1500,
                              training_registration_sha256=sha(OUT / "training_registration.json"),
                              resources=snapshot, first_authorized_worker_observation=True,
                              process_peak_reliability="WDDM per-process value unavailable; actual aggregate and tensor peaks logged",
                              concurrency=1, unrelated_user_processes_untouched=True))
    stamp = str(time.time_ns())
    log = jobs / f"{stage}_{stamp}.log"
    argv = [sys.executable, "-B", "-u", "-m", __package__ + "." + command[0], *command[1:]]
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2" if stage == "train" else "1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="2" if stage == "train" else "1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "r2/src")
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    import psutil
    record = dict(stage=stage, argv=argv, pid=process.pid,
                  process_created=psutil.Process(process.pid).create_time(),
                  log=str(log), resources=snapshot, automatic_retry=False, owned=True)
    write_new(jobs / f"job_{stage}_{stamp}.json", record)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-stage", choices=("train", "raw-new", "raw-old", "post-new", "post-old"))
    p.add_argument("--resume-training", action="store_true")
    a = p.parse_args()
    print(json.dumps(execute(a.execute_stage, resume_training=a.resume_training)
                     if a.execute_stage else status(), ensure_ascii=False, indent=2), flush=True)
