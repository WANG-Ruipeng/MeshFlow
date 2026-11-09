from experiments.mf_h1_local0.chair_control_feedback_auto_v1.common import (
    ROOT, SOURCE, CHAIR, START, START_SHA, GPU_LOCK, read_json, digest,
    json_hash, array_hash, atomic_json, freeze_json, append_event, ledger_rows, save_npz)
OLD = ROOT / 'native_t1/runs/chair_control_feedback_auto_v1'
OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
ARMS = ('T', 'L', 'G')
ARM_NAMES = dict(T='TEACHER_LCAP', L='CAPPED_REPLAY', G='GRAD_BUDGET')
LABELS = ('START', 'HIST_TEACHER500', 'HIST_CAPPED500', 'T500', 'T1000', 'L500', 'L1000', 'G500', 'G1000')
AUX_SEED = 20261008

from pathlib import Path
