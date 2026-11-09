"""Provenance, atomic files and safe NPZ/JSON helpers. No silent overwrite."""
from __future__ import annotations
import hashlib,json,os,tempfile,random
from pathlib import Path
import numpy as np
import torch

def file_hash(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def array_hash(*values:np.ndarray) -> str:
    h=hashlib.sha256()
    for v in values:
        v=np.ascontiguousarray(v)
        h.update(str(v.dtype).encode());h.update(json.dumps(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()

def model_hash(model:torch.nn.Module)->str:
    h=hashlib.sha256()
    for key,v in sorted(model.state_dict().items()):
        h.update(key.encode());a=v.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def json_bytes(obj)->bytes:
    def conv(x):
        if isinstance(x,Path):return str(x)
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=conv)+'\n').encode()

def write_json(path,obj,*,replace=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(p)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:f.write(json_bytes(obj));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def new_dir(path)->Path:
    p=Path(path);p.mkdir(parents=True,exist_ok=False);return p

def save_npz(path,**arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**arrays)

def atomic_torch(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy_name':n[0],'numpy_keys':torch.tensor(n[1].astype(np.int64)),
            'numpy_pos':n[2],'numpy_has_gauss':n[3],'numpy_cached':n[4],
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state((s['numpy_name'],s['numpy_keys'].numpy().astype(np.uint32),s['numpy_pos'],s['numpy_has_gauss'],s['numpy_cached']))
    torch.set_rng_state(s['torch_cpu'])
    if s['torch_cuda']:
        if not torch.cuda.is_available():raise RuntimeError('CUDA RNG checkpoint requires CUDA')
        torch.cuda.set_rng_state_all(s['torch_cuda'])

def read_mesh(path):
    with np.load(path,allow_pickle=False) as z:
        v=z['vertices'].astype(np.float64);f=z['faces'].astype(np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or len(f)==0:raise ValueError('Expected triangular mesh')
    if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Nonfinite coordinates or invalid indices')
    return v,f
