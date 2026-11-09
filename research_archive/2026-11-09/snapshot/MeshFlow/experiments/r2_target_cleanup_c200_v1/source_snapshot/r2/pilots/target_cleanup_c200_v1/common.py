"""Private output binding; the historical helper and assets stay read-only."""
from pathlib import Path
import importlib.util
import sys

_root = Path(__file__).resolve().parents[3]
_name = '_target_cleanup_c200_common_private'
_spec = importlib.util.spec_from_file_location(_name, _root/'r2/pilots/full_seat_learning_curve_v1/common.py')
_m = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _m
_spec.loader.exec_module(_m)
_m.OUT = _root/'experiments/r2_target_cleanup_c200_v1'
_m.CODE = Path(__file__).resolve().parent
_m.VERSION = 'R2_TARGET_CLEANUP_C200_V1'
_m.SOURCE = _root/'experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt'
_m.EVALUATION_LABELS = ['ORIGINAL_C200', 'DEDUP_C200', 'FULL_S6000']
_m.EVALUATION_STEPS = [200]
globals().update({k:v for k,v in vars(_m).items() if not k.startswith('__')})
ARMS = ('ORIGINAL_C200', 'DEDUP_C200')
BASE_STEP = 12000
UPDATES = 200
