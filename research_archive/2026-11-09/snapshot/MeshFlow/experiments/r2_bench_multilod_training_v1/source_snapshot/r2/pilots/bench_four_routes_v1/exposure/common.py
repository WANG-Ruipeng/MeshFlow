"""Fixed identity of M continuation; historical inputs remain read-only."""
from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve,read,sha,write_new,atomic_write,resources

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/r2_bench_four_routes_v1/exposure'
MASTER=OUT.parent
EXPERIMENT='R2_BENCH_FOUR_ROUTES_EXPOSURE_V1'
ARMS=('M_PLUS3000',)
BASE_STEP=22500
UPDATES=3000
M_ROOT=ROOT/'experiments/r2_bench_group_weight_match_v1'
SOURCE_ROOT=ROOT/'experiments/r2_bench_exposure_allocation_v1'
SOURCE=M_ROOT/'training/W_MATCHED/checkpoint_step1500.pt'
SOURCE_INFERENCE=M_ROOT/'training/W_MATCHED/generator_step1500.pt'
SOURCE_IDENTITIES=dict(source_checkpoint_sha256='0fc20a5414d7e247de0e877844528dfaa9fdf777062d92c7e215bd3bb3309575',
    source_generator_sha256='8a0fca7a8d882a9c246f86237763ec0178690274ef037b2bd6700d368db2cc9a',
    source_optimizer_sha256='d20350baac7379aa2d39bf21d44256c3a4b409bc810aa44fcb739a15da9c64b4',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8')
WIDE_ROOT=ROOT/'experiments/r2_bench_high_joint_coverage_v1'
NARROW_ROOT=ROOT/'experiments/r2_bench_natural_n_recipe_v1'
MANIFEST=WIDE_ROOT/'data/manifest.json'
EVAL_MANIFEST=WIDE_ROOT/'data/eval_manifest.json'
ORIGINAL_SCHEDULE=WIDE_ROOT/'data/schedule_WIDE_HIGH.json'
ORIGINAL_COEFFICIENTS=M_ROOT/'data/coefficient_schedule.json'
SCHEDULE=OUT/'data/schedule_M_PLUS3000.json'
COEFFICIENTS=OUT/'data/coefficient_schedule.json'
LOSS_NORMALIZATION='narrow_reference_group_MSE_coefficients_v1'
immutable=write_new
