from pathlib import Path
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.common import (
 ROOT, START, START_SHA, GPU_LOCK, read_json,digest,json_hash,array_hash,atomic_json,freeze_json,append_event,ledger_rows)
OUT=ROOT/'native_t1/runs/chair_capped500_half_v1'
PREVIOUS=ROOT/'native_t1/runs/chair_capped500_gradbudget_v1'
CAPPED=ROOT/'native_t1/runs/chair_control_feedback_auto_v1/checkpoints/explore_TRAJECTORY_REPLAY_CAPPED_500.pt'
CAPPED_SHA='058168b3126d26707ac57470613ac4e914ccc9fc2a34d119816cc1327b358c35'
LABEL='HALF_CAPPED500'
LABELS=('START','HIST_CAPPED500',LABEL)
WEIGHT=OUT/'checkpoints/half_capped500_alpha050.pt'
