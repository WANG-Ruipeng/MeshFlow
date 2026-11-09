"""Independent experiment identities; completed B/W assets are read-only."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)
from r2.pilots.bench_high_joint_coverage_v1.common import (
    SOURCE_ROOT, SOURCE, SOURCE_INFERENCE, SOURCE_IDENTITIES,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_group_weight_match_v1'
EXPERIMENT = 'R2_BENCH_GROUP_WEIGHT_MATCH_V1'
ARMS = ('W_MATCHED',)
BASE_STEP = 21000
UPDATES = 1500
WIDE_ROOT = ROOT / 'experiments/r2_bench_high_joint_coverage_v1'
NARROW_ROOT = ROOT / 'experiments/r2_bench_natural_n_recipe_v1'
MANIFEST = WIDE_ROOT / 'data/manifest.json'
SCHEDULE = WIDE_ROOT / 'data/schedule_WIDE_HIGH.json'
NARROW_MANIFEST = NARROW_ROOT / 'data/manifest.json'
NARROW_SCHEDULE = NARROW_ROOT / 'data/schedule_NATURAL_HIGH.json'
COEFFICIENTS = OUT / 'data/coefficient_schedule.json'
LOSS_NORMALIZATION = 'narrow_reference_group_MSE_coefficients_v1'
MODULE = 'r2.pilots.bench_group_weight_match_v1.'
immutable = write_new
