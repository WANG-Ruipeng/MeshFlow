from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve, read, sha, write_new, atomic_write, resources

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/r2_bench_four_routes_v1/seam'
EXPERIMENT = 'R2_BENCH_FOUR_ROUTES_V1_SEAM'
ARMS = ('SEAM1500',)
BASE_STEP = 22500
UPDATES = 1500
SOURCE_ROOT = ROOT / 'experiments/r2_bench_natural_n_recipe_v1'
SOURCE = SOURCE_ROOT / 'training/NATURAL_HIGH/checkpoint_step1500.pt'
EVAL_SOURCE = ROOT / 'experiments/r2_bench_group_weight_match_v1/data/eval_manifest.json'
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256='111d2142780ddb21c9aa94cef9174571fbb22f0602d561c663c4af92b4a9e17d',
    source_generator_sha256='6e40603c2d16b156ca1237592fc1bbfffa081e4140136e18823d081b0fa75c2d',
    source_optimizer_sha256='f2319e8410131d7163091116de0aa6ef650cc30b9f13a91b1c29bc251df3a42a',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8')
MODULE = 'r2.pilots.bench_four_routes_v1.seam.'
immutable = write_new
