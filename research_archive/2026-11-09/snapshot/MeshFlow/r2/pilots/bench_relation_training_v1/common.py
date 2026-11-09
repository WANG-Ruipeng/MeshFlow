"""Isolated Bench coverage experiment paths; no model or CUDA calls."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_relation_training_v1"
EXPERIMENT = "R2_BENCH_RELATION_TRAINING_V1"
ARM = "RELATION_V1_SEAM"
ARMS = (ARM,)
LABEL = "RELATION_V1_S1500"
BASELINE_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
BASELINE_LABEL = "A_S1500"

