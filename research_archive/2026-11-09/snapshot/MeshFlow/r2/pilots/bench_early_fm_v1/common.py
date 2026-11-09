"""Independent early-time FM experiment; historical assets are read-only."""
from pathlib import Path
from r2.pilots.bench_trajectory_crossover_v1.common import (
    resolve, read, sha, arrays, save, array_file, event, csvwrite, resources, verify)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_early_fm_v1'
OLD_TRAIN = ROOT / 'experiments/r2_bench_user_v2_training_v1'
OLD = ROOT / 'experiments/r2_bench_trajectory_diagnostic_v1'
CROSS = ROOT / 'experiments/r2_bench_trajectory_crossover_v1'
BASELINE_ROOT = ROOT / 'experiments/r2_bench_adaptation_strength_v1'
ARM = 'EARLY_FM'
ARMS = (ARM,)
EXPERIMENT = 'R2_BENCH_EARLY_FM_V1'

def time_weight(t):
    if not 0 <= float(t) <= 1:
        raise ValueError('Time outside [0,1]')
    return 1.5 if float(t) < .5 else .5

def registration(): return read(OUT / 'registration.json')
