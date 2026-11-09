"""Small portable evaluation IO; no model imports."""
from pathlib import Path
import csv,hashlib,json,os
import numpy as np
from .hashing import array_sha

def resolve(value):
    text=str(value).replace('\\','/')
    if os.name!='nt' and len(text)>2 and text[1]==':':text='/mnt/'+text[0].lower()+text[2:]
    if os.name=='nt' and text.startswith('/mnt/') and len(text)>7:text=text[5].upper()+':/'+text[7:]
    return Path(text)
def digest(path):return hashlib.sha256(resolve(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'INF' if x>0 else '-INF' if x<0 else None
    return x
def write_json(path,value,freeze=False):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True);value=clean(value)
    if freeze and path.exists():
        if read_json(path)!=value:raise ValueError('Frozen JSON differs: '+str(path))
        return
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def save_npz(path,values):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=arrays(p)
        if set(old)!=set(values) or any(array_sha(old[k])!=array_sha(v) for k,v in values.items()):raise ValueError('Frozen NPZ differs')
    else:np.savez_compressed(p,**values)
def write_csv(path,rows):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else clean(v) for k,v in r.items()})
def load_manifest(value):
    d=read_json(value) if isinstance(value,(str,Path)) else value
    return d['rows'] if isinstance(d,dict) else d
