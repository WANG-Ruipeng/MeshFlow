"""Experiment-local identities with shared durable helpers from the read-only pilot."""
from pathlib import Path
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import (
    read_json,digest,json_hash,array_hash,atomic_json,freeze_json,append_event,ledger_rows,save_npz)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'native_t1/runs/control_mvp48_v1'
CHAIR = ROOT/'native_t1/runs/chair_domain_control_v1'
HYBRID = ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
OUT = ROOT/'native_t1/runs/chair_moment_channel_v2'
OLD = ROOT/'native_t1/runs/chair_surface_attention_v1'
R1 = ROOT/'native_t1/runs/chair_surface_attention_v1r1'
START = HYBRID/'checkpoints/OT_HYBRID_pilot1000_total5000.pt'
START_SHA = 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
ARMS = ('H_CONT','MOMENT_XY')
LABELS = ('START','H_CONT_END','MOMENT_XY_END')
GPU_LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
TRAIN_SEED = 20261006
INIT_SEED = 618205

def resolve(path, base=CHAIR):
    p=Path(path)
    return p if p.is_absolute() else Path(base)/p

def records():
    panels=read_json(CHAIR/'panels.json')
    return [dict(r,table=table) for table in ('A','B','C') for r in panels[table]]
