from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import csv, datetime, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
CODE=Path(__file__).parent
OUT=ROOT/'experiments/r2_bench_four_routes_v1/hybrid'
OLD=ROOT/'experiments/r2_bench_group_weight_match_v1'
sys.path.insert(0,str(ROOT/'r2/src'))
L=float(np.sqrt(3.)*2/.3762)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def resolve(path):
    s=str(path).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    p=Path(s);return p if p.is_absolute() else ROOT/p
def read(path):return json.loads(resolve(path).read_text('utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x
def write(path,obj):
    p=resolve(path);obj=clean(obj)
    if p.exists():
        if read(p)!=obj:raise ValueError('Immutable JSON differs: '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def npz(path,values):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**values)
def csvwrite(path,rows):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def verify(book):
    for p,h in book.items():
        if sha(p)!=h:raise ValueError('Frozen hash changed: '+str(p))
def resources():
    from r2.pilots.bench_adaptation_strength_v1.common import resources as original
    return original()
