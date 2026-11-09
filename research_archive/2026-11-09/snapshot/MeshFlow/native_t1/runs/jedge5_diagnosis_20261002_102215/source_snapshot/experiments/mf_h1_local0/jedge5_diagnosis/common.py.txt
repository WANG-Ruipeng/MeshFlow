"""Shared frozen assets and JSON IO for the isolated JEdge5 diagnosis."""
from native_t1 import runtime as _runtime
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
from native_t1.artifacts import file_sha256, state_sha256, array_hash
from native_t1.data import PatchCase
from native_t1.geometry_checkpoint import load_geometry_checkpoint

ROOT=Path(__file__).resolve().parents[3]
TRAIN=ROOT/'output/t1_publish_20260930/chair_n112.npz'
EVALUATION=ROOT/'native_t1/runs/edge_factorial_20261001_123000/registration/evaluation_inputs.npz'
TRAIN_SHA='7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
EVALUATION_SHA='2154881d77b946a2a18959503930fd1d405f88f0d1f56d64a8f0d86710ed01b4'
ASSETS={
 'B':dict(path=ROOT/'native_t1/checkpoints/ablations/fm_cumulative3000.pt',recipe='fm',state='f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b'),
 'J':dict(path=ROOT/'native_t1/checkpoints/jedge5_cumulative3000.pt',recipe='edge5',state='b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3'),
 'G0':dict(path=ROOT/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt',recipe='fm',state='6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392')}

def _json(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,default=_json,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def load_model(label,device='cuda',checkpoint=None):
    spec=ASSETS[label]
    with torch.random.fork_rng(devices=[]):
        model,identity=load_geometry_checkpoint(checkpoint or spec['path'],device=device,expected_recipe=spec['recipe'])
    if identity['state_sha256']!=spec['state']:
        raise ValueError('Frozen '+label+' generator identity mismatch')
    return model,identity

def load_cases():
    if file_sha256(EVALUATION)!=EVALUATION_SHA:raise ValueError('Evaluation asset changed')
    with np.load(EVALUATION,allow_pickle=False) as z:
        reg=json.loads(str(z['registration_json'].item()))
        arrays={k:z[k].copy() for k in z.files if k!='registration_json'}
    if reg['status']!='PASS' or set(reg['arrays'])!=set(arrays):raise ValueError('Incomplete registration')
    for key,a in arrays.items():
        # The original frozen registration uses dtype/shape followed by raw bytes.
        from experiments.mf_h1_local0.tgeo_m10_evaluation import _array_hash
        rec=reg['arrays'][key]
        if _array_hash(a)!=rec['sha256'] or list(a.shape)!=rec['shape'] or str(a.dtype)!=rec['dtype']:
            raise ValueError('Registered array changed: '+key)
    cases={}
    for p in (0,1):
        for j in range(5):
            prefix=f'parent{p}_patch{j}_';Y=arrays[f'parent{p}_target']
            C=arrays[prefix+'constraints'];free=arrays[prefix+'free_target']
            ids=arrays[prefix+'source_face_ids'];freeids=arrays[prefix+'free_source_ids']
            pre=arrays[prefix+'pre_ot_free'];L=float(np.linalg.norm(np.ptp(Y.reshape(-1,3),axis=0)))
            case=PatchCase(p,12,C,Y,free,ids,freeids,L,pre)
            row=dict(parent=p,patch=j,C=C,full=Y,free=free,pre_ot_free=pre,
                source_face_ids=ids,free_source_ids=freeids,bbox=L,case=case,
                source_face_vertex_ids=arrays[f'parent{p}_source_face_vertex_ids'],
                source_vertices=arrays[f'parent{p}_source_vertices'])
            for suffix in ('source_corner_ids','corner_permutations','boundary_segments','boundary_points','boundary_weights','boundary_point_segment_ids','pairwise_target_points','coverage_points'):
                row[suffix]=arrays[prefix+suffix]
            cases[p,j]=row
    return cases

def noise_bank(out):
    with np.load(Path(out)/'registration/noises.npz',allow_pickle=False) as z:
        return {int(k.removeprefix('seed_')):torch.from_numpy(z[k].copy()) for k in z.files}

def registered_seeds(out):
    return read_json(Path(out)/'registration/protocol.json')['generation_seeds']
