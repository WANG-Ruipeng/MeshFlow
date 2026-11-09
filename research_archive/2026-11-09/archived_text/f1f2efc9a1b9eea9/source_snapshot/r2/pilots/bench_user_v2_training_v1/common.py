"""Isolated Bench coverage experiment paths; no model or CUDA calls."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_user_v2_training_v1"
EXPERIMENT = "R2_BENCH_USER_V2_TRAINING_V1"
ARM = "USER_V2_SEAM"
ARMS = (ARM,)
LABEL = "USER_V2_S1500"
BASELINE_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
BASELINE_LABEL = "A_S1500"

