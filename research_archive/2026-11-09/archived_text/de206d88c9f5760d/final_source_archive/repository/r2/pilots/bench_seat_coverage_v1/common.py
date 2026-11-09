from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve, read, sha, write_new, atomic_write, resources

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_seat_coverage_v1'
EXPERIMENT = 'R2_BENCH_SEAT_COVERAGE_V1'
STAGE = 'NATIVE'
BASE_STEP = 18000
SOURCE_ROOT = ROOT / 'experiments/r2_bench_full_seat_curriculum_v1'
SOURCE = SOURCE_ROOT / 'training/NATIVE/checkpoint_step6000.pt'
SOURCE_IDENTITIES = dict(
    source_checkpoint_sha256='31aef34d1126c4ecb7a22201ec3b874288417d7b57dda8e56f7dfe7eb49a8bfe',
    source_generator_sha256='f0b9a24db061e7486bdec82a9fa2eb09ca94b644340f42f2c7b1886696ff10b2',
    source_optimizer_sha256='5ebcfb68d276a270a068cab1980ef7c2f1fecea0a34478c09fdc83a31c2ca7b5',
    source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8')
MODULE = 'r2.pilots.bench_seat_coverage_v1.'
