"""Independent output binding; historical code and evidence are read-only."""
from pathlib import Path
import importlib.util
import sys

_root = Path(__file__).resolve().parents[3]
_name = '_dedup_n_sensitivity_common_private'
_spec = importlib.util.spec_from_file_location(_name, _root/'r2/pilots/full_seat_learning_curve_v1/common.py')
_m = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _m
_spec.loader.exec_module(_m)
_m.OUT = _root/'experiments/r2_dedup_n_sensitivity_v1'
_m.CODE = Path(__file__).resolve().parent
_m.VERSION = 'R2_DEDUP_N_SENSITIVITY_V1'
_m.EVALUATION_LABELS = ['ORIGINAL_LOW_N','DEDUP_LOW_N','ORIGINAL_ORIG_N','DEDUP_ORIG_N']
globals().update({k:v for k,v in vars(_m).items() if not k.startswith('__')})
ARMS = ('ORIGINAL_LOW_N','DEDUP_LOW_N')
REFERENCES = ('ORIGINAL_ORIG_N','DEDUP_ORIG_N')
BASE_STEP = 12200
UPDATES = 0
