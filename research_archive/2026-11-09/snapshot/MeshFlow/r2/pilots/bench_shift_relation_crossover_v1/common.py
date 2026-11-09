"""Independent evidence directory; no model construction on import."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve, read, sha, write_new, atomic_write, resources
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_shift_relation_crossover_v1'
EXPERIMENT = 'R2_BENCH_SHIFT_RELATION_CROSSOVER_V1'
LABEL = 'SHIFT_RELATION_T050'
ARM = LABEL
ARMS = (ARM,)
