from pathlib import Path
from experiments.mf_h1_local0.chair_surface_attention_v1.common import (
    ROOT,ARMS,START,START_SHA,INIT_SEED,read_json,digest,json_hash,array_hash,
    atomic_json,freeze_json,append_event,ledger_rows,save_npz)
OLD=ROOT/'native_t1/runs/chair_surface_attention_v1'
OUT=ROOT/'native_t1/runs/chair_surface_attention_v1r1'
