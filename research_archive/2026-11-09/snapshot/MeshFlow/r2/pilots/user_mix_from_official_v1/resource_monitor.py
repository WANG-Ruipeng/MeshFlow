"""Read-only nvidia-smi observations around the sole registered sampling phase.

Start this monitor before sample; its ready receipt follows the first resource
observation. No torch, model, generation, process termination or retry is used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

VERSION = "r2_user_mix_sampling_nvidia_monitor_v1"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "experiments/r2_user_mix_from_official_v1"
INTERVAL_SECONDS = 2.0
TIMEOUT_SECONDS = 600.0
GPU_COMMAND = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
PROCESS_COMMAND = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def root(path):
    path = Path(path).resolve()
    if path != DEFAULT_OUT.resolve():
        raise ValueError("Wrong experiment output root")
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("Refuse overwrite of existing sampling resource record")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def identity():
    return {"version": VERSION, "source_path": str(Path(__file__).resolve()), "source_sha256": sha256(__file__),
            "interval_seconds": INTERVAL_SECONDS, "maximum_wall_seconds": TIMEOUT_SECONDS,
            "gpu_command": GPU_COMMAND, "process_command": PROCESS_COMMAND,
            "sampler_pid_source": "evaluation/sampling/R2_USER_MIX/initial_dispatch.json:pid",
            "process_attribution": "Exact sampler PID match in nvidia-smi process table; unreported/N/A is UNAVAILABLE, never zero",
            "device_memory_scope": "Whole GPU including background processes and driver/context allocations",
            "memory_rule": "Observed device used and observed per-process used are separate; allocated/reserved never added",
            "model_calls": 0, "process_termination": False}


def register(out):
    out = root(out)
    value = identity()
    write(out / "audit/sampling_resource_monitor_identity.json", value)
    return {"status": "REGISTERED", "monitor_started": False, **value}


def _number(raw):
    try:
        value = float(raw.strip())
        if not (float("-inf") < value < float("inf")):
            return None
        return value
    except (ValueError, TypeError):
        return None


def query(command, remaining_seconds):
    started = time.monotonic()
    timeout = max(.001, min(8., remaining_seconds))
    actual_command = list(command)
    if shutil.which(actual_command[0]) is None and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        actual_command[0] = "/usr/lib/wsl/lib/nvidia-smi"
    try:
        result = subprocess.run(actual_command, text=True, encoding="utf8", errors="replace", capture_output=True, timeout=timeout, check=False)
        return {"status": "PASS" if result.returncode == 0 else "ERROR", "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "command": actual_command, "seconds": time.monotonic() - started}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "ERROR", "error": repr(exc), "stdout": "", "stderr": "", "returncode": None,
                "command": actual_command, "seconds": time.monotonic() - started}


def snapshot(out, deadline, reason, sequence):
    observed_unix = time.time()
    dispatch_path = out / "evaluation/sampling/R2_USER_MIX/initial_dispatch.json"
    dispatch_error = None
    try:
        dispatch = read(dispatch_path) if dispatch_path.exists() else None
    except (json.JSONDecodeError, OSError) as exc:
        # A partially written dispatch is pending, never a guessed sampler PID.
        dispatch, dispatch_error = None, repr(exc)
    gpu = query(GPU_COMMAND, deadline - time.monotonic())
    process = query(PROCESS_COMMAND, deadline - time.monotonic())
    devices, processes, parse_errors = [], [], []
    if gpu["status"] == "PASS":
        for fields in csv.reader(io.StringIO(gpu["stdout"]), skipinitialspace=True):
            if len(fields) != 7:
                parse_errors.append({"table": "gpu", "row": fields})
                continue
            index, uuid, name, total, used, free, utilization = (f.strip() for f in fields)
            devices.append({"index": index, "uuid": uuid, "name": name, "total_MiB": _number(total),
                            "used_MiB": _number(used), "free_MiB": _number(free), "utilization_percent": _number(utilization)})
    if process["status"] == "PASS":
        for fields in csv.reader(io.StringIO(process["stdout"]), skipinitialspace=True):
            if not fields:
                continue
            if len(fields) != 4:
                parse_errors.append({"table": "compute_processes", "row": fields})
                continue
            pid, name, used, uuid = (f.strip() for f in fields)
            parsed_pid = int(pid) if pid.isdigit() else None
            processes.append({"pid": parsed_pid, "pid_raw": pid, "process_name": name,
                              "used_MiB": _number(used), "used_MiB_raw": used, "gpu_uuid": uuid,
                              "matches_sampler_pid": bool(dispatch is not None and parsed_pid == dispatch["pid"])})
    return {"sequence": sequence, "reason": reason, "time_unix": observed_unix,
            "time_utc": dt.datetime.fromtimestamp(observed_unix, dt.timezone.utc).isoformat(),
            "sampler_pid": dispatch.get("pid") if dispatch else None,
            "sampler_dispatch_started_unix": dispatch.get("started") if dispatch else None,
            "sampler_dispatch_path": str(dispatch_path), "dispatch_seen": dispatch is not None, "dispatch_read_error": dispatch_error,
            "devices": devices, "compute_processes": processes, "parse_errors": parse_errors,
            "gpu_query": gpu, "process_query": process,
            "status": "PASS" if devices and not parse_errors and gpu["status"] == process["status"] == "PASS" else "PARTIAL_UNAVAILABLE"}


def summarize(rows, terminal, started_unix, elapsed, completion, failure):
    # PID may first become known after model loading starts. Reattribute earlier
    # rows by the eventual exact PID without inventing unreported memory values.
    pids = sorted({row["sampler_pid"] for row in rows if row["sampler_pid"] is not None})
    if len(pids) > 1:
        raise ValueError("Sampler PID changed within a single registered phase")
    sampler_pid = pids[0] if pids else None
    processes = [proc for row in rows for proc in row["compute_processes"] if sampler_pid is not None and proc["pid"] == sampler_pid]
    process_values = [proc["used_MiB"] for proc in processes if proc["used_MiB"] is not None]
    devices = []
    for uuid in sorted({gpu["uuid"] for row in rows for gpu in row["devices"]}):
        values = [gpu for row in rows for gpu in row["devices"] if gpu["uuid"] == uuid]
        used = [gpu["used_MiB"] for gpu in values if gpu["used_MiB"] is not None]
        free = [gpu["free_MiB"] for gpu in values if gpu["free_MiB"] is not None]
        utilization = [gpu["utilization_percent"] for gpu in values if gpu["utilization_percent"] is not None]
        devices.append({"uuid": uuid, "name": values[0]["name"], "total_MiB": values[0]["total_MiB"],
                        "observed_peak_used_MiB": max(used) if used else None, "observed_minimum_free_MiB": min(free) if free else None,
                        "observed_peak_utilization_percent": max(utilization) if utilization else None,
                        "observed_mean_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
                        "scope": "Whole-device observations including background; not sampler-only process peak"})
    dispatch_starts = [row["sampler_dispatch_started_unix"] for row in rows if row["sampler_dispatch_started_unix"] is not None]
    first_observation = rows[0]["time_unix"] if rows else None
    before_dispatch = bool(dispatch_starts and first_observation is not None and first_observation <= min(dispatch_starts))
    return {"status": terminal, "version": VERSION, "started_unix": started_unix, "wall_seconds": elapsed,
            "interval_seconds": INTERVAL_SECONDS, "samples": len(rows), "sampling_pid": sampler_pid,
            "first_observation_before_sampling_dispatch": before_dispatch,
            "loading_observation_status": "STARTED_BEFORE_DISPATCH_INCLUDES_MODEL_LOAD_WINDOW" if before_dispatch else "UNVERIFIED_OR_STARTED_AFTER_DISPATCH",
            "completion_receipt_seen": completion, "failure_receipt_seen": failure,
            "device_summary": devices,
            "sampler_process_memory": {"status": "OBSERVED" if process_values else "UNAVAILABLE",
                                       "observed_peak_used_MiB": max(process_values) if process_values else None,
                                       "exact_pid_rows": len(processes), "numeric_memory_rows": len(process_values),
                                       "reason": None if process_values else "nvidia-smi did not report numeric used memory for the exact sampler PID",
                                       "scope": "Per-process used VRAM sampled every 2s; an observed maximum, not a guaranteed exact peak"},
            "memory_semantics": "Device total/used/free and per-process used kept separate; no sum of PyTorch allocated and reserved",
            "model_calls": 0, "observations": rows}


def monitor(out, controller_owned=False):
    if not controller_owned:
        raise ValueError("Root must explicitly start this phase observer with --controller-owned")
    out = root(out)
    frozen = identity()
    if read(out / "audit/sampling_resource_monitor_identity.json") != frozen:
        raise ValueError("Resource-monitor source identity changed")
    audit = out / "audit"
    result_path = audit / "sampling_resources.json"
    rows_path = audit / "sampling_resources_samples.jsonl"
    if result_path.exists() or rows_path.exists():
        raise ValueError("A resource observation already exists; no silent overwrite/restart")
    phase = out / "evaluation/sampling/R2_USER_MIX"
    completion_path, failure_path = phase / "completed.json", phase / "initial_failure.json"
    started_unix, started = time.time(), time.monotonic()
    deadline = started + TIMEOUT_SECONDS
    next_observation = started
    rows = []
    with rows_path.open("x", encoding="utf8") as handle:
        try:
            terminal = "TIMEOUT"
            while time.monotonic() < deadline:
                completion, failure = completion_path.exists(), failure_path.exists()
                reason = "TERMINAL_FOLLOWUP" if completion or failure else "PERIODIC"
                row = snapshot(out, deadline, reason, len(rows))
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                if len(rows) == 1:
                    write(audit / "sampling_resources_monitor_ready.json", {"status": "READY" if row["status"] == "PASS" else "READY_WITH_UNAVAILABLE_FIELDS",
                                                                          "pid": os.getpid(), "first_observation_unix": row["time_unix"],
                                                                          "source_sha256": frozen["source_sha256"], "samples_path": str(rows_path)})
                    print(json.dumps({"status": "MONITOR_READY", "pid": os.getpid(), "first_observation_status": row["status"]}), flush=True)
                if completion or failure:
                    terminal = "SAMPLING_COMPLETE" if completion and not failure else "SAMPLING_FAILED"
                    break
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    next_observation += INTERVAL_SECONDS
                    delay = max(0., next_observation - time.monotonic())
                    time.sleep(min(delay, remaining))
            result = summarize(rows, terminal, started_unix, time.monotonic() - started, completion_path.exists(), failure_path.exists())
            result.update(identity=frozen, samples_path=str(rows_path), samples_sha256=sha256(rows_path))
            write(result_path, result)
            return {key: value for key, value in result.items() if key != "observations"}
        except BaseException as exc:
            write(audit / "sampling_resource_monitor_failure.json", {"status": "MONITOR_FAILED", "error": repr(exc),
                                                                    "traceback": traceback.format_exc(), "samples_written": len(rows),
                                                                    "wall_seconds": time.monotonic() - started, "model_calls": 0})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "monitor"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--controller-owned", action="store_true")
    args = parser.parse_args()
    result = register(args.out) if args.stage == "register" else monitor(args.out, args.controller_owned)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
