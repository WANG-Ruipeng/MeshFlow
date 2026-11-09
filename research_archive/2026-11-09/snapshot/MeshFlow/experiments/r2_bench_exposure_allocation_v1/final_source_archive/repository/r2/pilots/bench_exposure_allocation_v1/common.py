"""Independent experiment identities; historical inputs remain read-only."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_exposure_allocation_v1'
EXPERIMENT = 'R2_BENCH_EXPOSURE_ALLOCATION_V1'
ARMS = ('A_50_50', 'B_75_25')
BASE_STEP = 19500
SOURCE_ROOT = ROOT / 'experiments/r2_bench_seat_coverage_v1'
SOURCE = SOURCE_ROOT / 'training/NATIVE/checkpoint_step1500.pt'
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256='68703028dc20d84f210daa5b8c6eb4468299bf5aa2a5bace8881e8bd500fcf10',
    source_generator_sha256='f5a36b4f2bad3671f14fab8b906cfc4c47e924b817d34c33431a5fb5c49d99ac',
    source_optimizer_sha256='e1138e192e0ee9dbbd69a1797173952f2e76b1ba4a7018b1ef0e1d8d0ee1f758',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8',
)
MODULE = 'r2.pilots.bench_exposure_allocation_v1.'
