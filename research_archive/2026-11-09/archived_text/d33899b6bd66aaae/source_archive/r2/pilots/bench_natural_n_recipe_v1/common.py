"""Fixed experiment identity. Historical models/data remain read-only."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import (
    resolve, read, sha, write_new, atomic_write, resources,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_natural_n_recipe_v1'
EXPERIMENT = 'R2_BENCH_NATURAL_N_RECIPE_V1'
ARMS = ('CONTROL_OLD', 'NATURAL_HIGH')
BASE_STEP = 21000
UPDATES = 1500
SOURCE_ROOT = ROOT / 'experiments/r2_bench_exposure_allocation_v1'
SOURCE = SOURCE_ROOT / 'training/B_75_25/checkpoint_step1500.pt'
SOURCE_INFERENCE = SOURCE_ROOT / 'training/B_75_25/generator_step1500.pt'
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256='e22c72b26ef6648336a0f14f98f371db9f5d4a86696a9f197680593baf303443',
    source_generator_sha256='5962e74f8bc16408b336c1502aba6935548ce28f41fb03f8d1c508ea39b0ba55',
    source_optimizer_sha256='dc7268094520266fa68daa63a2691f881b5cee93ca579560755e442ba3b931e1',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8',
)
MODULE = 'r2.pilots.bench_natural_n_recipe_v1.'
immutable = write_new
