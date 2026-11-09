"""Isolated evidence paths; no model construction or forward on import."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_state_schedule_v1"
OLD = ROOT / "experiments/r2_bench_clean_multipart_seam_v1"
ARMS = ("PERTURBED_STATE", "ASYNC_LEAD_SEAM")
EXPERIMENT = "R2_BENCH_STATE_SCHEDULE_V1"


def resolve(path):
    text = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_new(path, obj):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write(path, obj):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.partial")
    write_new(tmp, obj)
    # Retry only filesystem replace sharing races, never scientific operations.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def resources():
    import psutil
    def query(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    active = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            args = p.info.get("cmdline") or []
            if any("python" in s or "r2." in s for s in args):
                active.append(dict(pid=p.pid, name=p.info["name"], command=args,
                                   rss=p.info["memory_info"].rss if p.info["memory_info"] else None))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    m = psutil.virtual_memory()
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                gpu=query(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]),
                gpu_processes=query(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
                CPU_cores=psutil.cpu_count(), CPU_percent=psutil.cpu_percent(interval=.2),
                RAM_total=m.total, RAM_available=m.available,
                load=os.getloadavg() if hasattr(os, "getloadavg") else None,
                science_and_python_processes=active,
                GPU_process_visibility="WDDM may omit host processes; aggregate VRAM plus scoped allocator and observation retained")
