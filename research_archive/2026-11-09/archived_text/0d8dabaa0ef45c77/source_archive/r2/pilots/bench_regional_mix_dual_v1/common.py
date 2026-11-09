"""Local paths; reuse immutable helpers, never redirect historical outputs."""
from r2.pilots.bench_four_routes_v1.hybrid.common import (
    ROOT, L, resolve, read, sha, write, arrays, npz, csvwrite, verify, utc, clean,
)

OUT = ROOT / 'experiments/r2_bench_regional_mix_dual_v1'
CODE = ROOT / 'r2/pilots/bench_regional_mix_dual_v1'
OLD = ROOT / 'experiments/r2_bench_four_routes_v1/hybrid'
