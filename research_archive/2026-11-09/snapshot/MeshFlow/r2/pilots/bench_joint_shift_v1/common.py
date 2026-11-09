"""Isolated Bench joint-translation experiment paths; no model or CUDA calls."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_joint_shift_v1"
EXPERIMENT = "R2_BENCH_JOINT_SHIFT_V1"
ARM = "JOINT_SHIFT_SEAM"
ARMS = (ARM,)
LABEL = "JOINT_SHIFT_S1500"
BASELINE_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
BASELINE_LABEL = "A_S1500"

