from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import resolve, read, sha, write_new, atomic_write, resources
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_newdata_joint_v1'
EXPERIMENT = 'R2_BENCH_NEWDATA_JOINT_V1'
ARM = 'NEW_DATA_JOINT'
ARMS = (ARM,)
LABEL = 'NEWDATA_JOINT_S1500'
BASELINE_ROOT = ROOT / 'experiments/r2_bench_adaptation_strength_v1'
REFERENCE_ROOT = ROOT / 'experiments/r2_bench_relation_training_v1'
REFERENCE_LABEL = 'RELATION_V1_S1500'
