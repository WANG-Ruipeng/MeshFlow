"""Shared paths and append-only evidence utilities for the fixed pilot."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_clean_multipart_seam_v1"
SOURCE_BENCH = ROOT / "experiments/r2_bench_parts_convergence_v1"
LABEL = "BENCH_CLEAN_MULTIPART_S1500"


def resolve(path):
    text = str(path).replace("\\", "/")
    if text.startswith("E:/MeshFlow/"):
        return ROOT / text[len("E:/MeshFlow/"):]
    if text.startswith("/mnt/e/MeshFlow/"):
        return ROOT / text[len("/mnt/e/MeshFlow/"):]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def atomic_write(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def resources():
    import psutil
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    active = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            args = p.info.get("cmdline") or []
            if any("r2." in s or "python" in s for s in args):
                active.append(dict(pid=p.pid, name=p.info["name"], command=args,
                                   rss=p.info["memory_info"].rss if p.info["memory_info"] else None))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    mem = psutil.virtual_memory()
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                gpu=command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader"]),
                gpu_processes=command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"]),
                CPU_cores=psutil.cpu_count(), CPU_percent=psutil.cpu_percent(interval=.2),
                RAM_total=mem.total, RAM_available=mem.available,
                load=os.getloadavg() if hasattr(os, "getloadavg") else None,
                science_and_python_processes=active,
                GPU_process_visibility="WDDM may omit host graphics processes; aggregate VRAM is authoritative")
