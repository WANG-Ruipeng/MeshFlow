"""CPU-only shared helpers and exact historical identities."""
from pathlib import Path
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import (
    read_json,digest,json_hash,array_hash,atomic_json,freeze_json,append_event,ledger_rows,save_npz)

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'native_t1/runs/chair_bounded_attachment_v1'
BOUNDARY=ROOT/'native_t1/runs/chair_boundary_attachment_v1'
QUICK=ROOT/'native_t1/runs/chair_control_usability_quick_v1'
HYBRID=ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
SOURCE=ROOT/'native_t1/runs/control_mvp48_v1'
START=HYBRID/'checkpoints/OT_HYBRID_pilot1000_total5000.pt'
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
LABELS=('RAW','ENDPOINT','TRAJECTORY')
GPU_LOCK='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'


def resolve(value):
    s=str(value).replace('\\','/')
    for prefix in ('E:/MeshFlow/','e:/MeshFlow/','/mnt/e/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def result_path(out,row,label=None):
    rid=row['request_id'] if label is None else label+'__'+row.get('base_id',row['base_request_id'])
    return Path(out)/'outputs'/rid/'result.json'


def arrays(path):
    import numpy as np
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write_mesh(path,raw):
    import numpy as np
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    a=np.asarray(raw).reshape(-1,3,3)
    # Triangle soup, independent index triplet per original face; no welding.
    lines=['ply','format ascii 1.0','element vertex '+str(a.size//3),
        'property float x','property float y','property float z','element face '+str(len(a)),
        'property list uchar int vertex_indices','end_header']
    lines+=[' '.join(format(float(x),'.9g') for x in v) for v in a.reshape(-1,3)]
    lines+=['3 '+str(3*i)+' '+str(3*i+1)+' '+str(3*i+2) for i in range(len(a))]
    p.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return digest(p)
