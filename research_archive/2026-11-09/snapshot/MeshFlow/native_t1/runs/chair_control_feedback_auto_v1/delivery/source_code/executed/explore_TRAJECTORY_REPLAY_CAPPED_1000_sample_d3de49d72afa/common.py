from pathlib import Path
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import (
    read_json,digest,json_hash,array_hash,atomic_json,freeze_json,append_event,ledger_rows,save_npz as _old_save_npz)
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'native_t1/runs/control_mvp48_v1'
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
OUT=ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
START=ROOT/'native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt'
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
GPU_LOCK='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
ARMS=('FM_CONTROL','REPLAY_FM_CONTROL','TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED')
START_UTC='2026-10-06T09:18:49Z'
SOFT_DEADLINE_UTC='2026-10-06T13:18:49Z'

def save_npz(filename, **arrays):
    # Unlike the historical helper, accepts an array named path (Euler states).
    import numpy as np, os
    target=Path(filename);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.tmp.'+str(os.getpid()))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**arrays);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,target)
    return digest(target)