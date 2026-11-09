from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_c_tessellation_v1'
OLD = ROOT / 'experiments/r2_bench_four_routes_v1/exposure'
WEIGHT = OLD / 'training/M_PLUS3000/generator_step3000.pt'
WEIGHT_SHA = '70e51d7913f99ad017d023e47965387b28692075ed9964b61d9d55dd1f0ff6b3'
PREFIXES = ('459', 'f495', 'bcb')

def resolve(p):
    s=str(p).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if s.startswith(prefix): return ROOT/s[len(prefix):]
    p=Path(s); return p if p.is_absolute() else ROOT/p

def read(p): return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p=resolve(p); p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)
    if p.exists():
        if read(p)!=json.loads(text): raise ValueError('Immutable result differs: '+str(p))
        return
    with p.open('x',encoding='utf-8') as f: f.write(text)
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def npz(p,v):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        a=arrays(p)
        if set(a)!=set(v) or any(a[k].dtype!=np.asarray(v[k]).dtype or not np.array_equal(a[k],v[k]) for k in a):
            raise ValueError('Immutable arrays differ: '+str(p))
        return
    with p.open('xb') as f: np.savez_compressed(f,**v)
def verify(book):
    for p,h in book.items():
        if sha(p)!=h: raise ValueError('Protected file changed: '+str(p))
def csvwrite(p,rows):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
