"""Bounded CGAL classic-plane QEM baseline; no repair or budget coercion.

Build dependencies (Ubuntu): build-essential cmake libcgal-dev libeigen3-dev.
All runtime inputs/outputs and metadata are retained under work_dir. Compile is
an environment/setup cost and is separately timed; run time includes process
startup, serialization, native input audit, guards, final audit and output I/O.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

SOURCE = Path(__file__).resolve().parent / "cgal"

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_cgal(build_dir: str | Path | None = None) -> tuple[Path, dict]:
    """Build native code if absent/stale. Never install dependencies implicitly."""
    build = Path(build_dir) if build_dir else SOURCE / "build"
    binary = build / "guarded_qem"
    sources = [SOURCE / "guarded_qem.cpp", SOURCE / "CMakeLists.txt"]
    needed = not binary.exists() or any(p.stat().st_mtime_ns > binary.stat().st_mtime_ns for p in sources)
    begin = time.perf_counter()
    logs = []
    if needed:
        for command in [
            ["cmake", "-S", str(SOURCE), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "--parallel", "2"],
        ]:
            result = subprocess.run(command, capture_output=True, text=True, timeout=240)
            logs.append((result.stdout + result.stderr)[-8000:])
            if result.returncode:
                raise RuntimeError("CGAL build failed: " + logs[-1])
    return binary, {
        "build_performed": needed,
        "build_seconds": time.perf_counter() - begin,
        "source_sha256": {p.name: _sha256(p) for p in sources},
        "binary_sha256": _sha256(binary),
        "build_log_tail": "\n".join(logs)[-8000:],
    }

def _write_off(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as out:
        out.write(f"OFF\n{len(v)} {len(f)} 0\n")
        np.savetxt(out, v, fmt="%.17g")
        np.savetxt(out, np.column_stack((np.full(len(f), 3), f)), fmt="%d")

def _read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if lines[0] != "OFF":
        raise ValueError("CGAL output is not plain OFF")
    nv, nf, _ = map(int, lines[1].split())
    vertices = np.asarray([list(map(float, line.split())) for line in lines[2:2+nv]], dtype=np.float64)
    records = np.asarray([list(map(int, line.split())) for line in lines[2+nv:2+nv+nf]], dtype=np.int64)
    if records.ndim != 2 or records.shape[1] != 4 or not np.all(records[:, 0] == 3):
        raise ValueError("CGAL output contains non-triangles")
    return vertices, records[:, 1:]

def run_cgal(v, f, target: int, min_angle_deg: float, angle_q05_deg: float,
             work_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reduce same source mesh using CGAL plane QEM + explicit guards.

    Local changed-face minimum angle is enforced. Global minimum/q05 and exact
    budget are audited at endpoint and can fail; outputs are never repaired.
    CGAL preserves closed manifold topology by its native collapse link checks.
    Prospective changed/rest AND changed/changed triangle pairs use exact
    intersection constructions; their only permitted contact is the shared
    topological vertex or edge. Starting input is checked for self intersection.
    """
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected Vx3 vertices and Fx3 faces")
    if not np.isfinite(v).all() or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid mesh arrays")
    if target < 4 or target % 2 or target > len(f):
        raise ValueError("Closed triangular target must be even, >=4, <=input faces")
    if not 0 <= min_angle_deg <= 60 or not 0 <= angle_q05_deg <= 60:
        raise ValueError("Angle thresholds must be in [0,60]")
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    input_path, output_path, native_report = [directory / n for n in ("input.off", "output.off", "native_report.json")]
    if any(p.exists() for p in (input_path, output_path, native_report)):
        raise FileExistsError("CGAL run path already contains input/output; use a new directory")
    binary, build_metadata = build_cgal()
    begin = time.perf_counter()
    _write_off(input_path, v, f)
    command = [str(binary), str(input_path), str(output_path), str(native_report),
               str(target), repr(float(min_angle_deg)), repr(float(angle_q05_deg))]
    result = subprocess.run(command, capture_output=True, text=True, timeout=360)
    log = (result.stdout + result.stderr)[-8000:]
    (directory / "native.log").write_text(log, encoding="utf8")
    if result.returncode:
        failure = {"status": "FAILED", "returncode": result.returncode, "log_tail": log,
                   "command": command, "runtime_seconds": time.perf_counter()-begin, "build": build_metadata}
        (directory / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf8")
        raise RuntimeError(f"CGAL run failed ({result.returncode}): {log}")
    vo, fo = _read_off(output_path)
    metadata = json.loads(native_report.read_text(encoding="utf8"))
    metadata.update({
        "status": "COMPLETED",
        "seconds_export": time.perf_counter() - begin,
        "timing_scope": "Python OFF serialization + native process startup/read/preaudit/QEM/guards/postaudit/write + Python output read; build excluded",
        "build": build_metadata,
        "command": command,
        "log_tail": log,
        "budget_pass": len(fo) == target,
        "all_output_constraints_pass": len(fo) == target and metadata["endpoint_quality_pass"]
            and metadata["closed"] and not metadata["final_self_intersection"],
        "global_q05_enforced_per_step": False,
        "initial_low_quality_faces_may_persist": True,
    })
    (directory / "runner_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return vo, fo, metadata
