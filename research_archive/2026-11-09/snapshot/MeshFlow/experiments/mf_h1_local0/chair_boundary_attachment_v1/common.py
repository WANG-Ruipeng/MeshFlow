"""Shared paths and immutable scientific identities; no model side effects."""
from pathlib import Path
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import (
    read_json, digest, json_hash, array_hash, atomic_json, freeze_json, append_event, ledger_rows)

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'native_t1/runs/chair_boundary_attachment_v1'
SOURCE=ROOT/'native_t1/runs/control_mvp48_v1'
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
HYBRID=ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
QUICK=ROOT/'native_t1/runs/chair_control_usability_quick_v1'
START=HYBRID/'checkpoints/OT_HYBRID_pilot1000_total5000.pt'
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
ARMS=('A_FM','B_BOUNDARY','C_SEAM','D_BOUNDARY_SEAM')
LABELS=('START_HYBRID',)+tuple(a+'_END' for a in ARMS)
BOUNDARY_ARMS=('B_BOUNDARY','D_BOUNDARY_SEAM')
SEAM_ARMS=('C_SEAM','D_BOUNDARY_SEAM')
GPU_LOCK='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
SEED=20261007


def resolve(value):
    value=str(value).replace('\\','/')
    for prefix in ('E:/MeshFlow/','e:/MeshFlow/','/mnt/e/MeshFlow/'):
        if value.startswith(prefix):return ROOT/value[len(prefix):]
    p=Path(value)
    return p if p.is_absolute() else ROOT/p
