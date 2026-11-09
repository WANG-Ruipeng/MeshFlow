from pathlib import Path
import numpy as np
import json, hashlib, os, time
ROOT=Path(__file__).resolve().parents[1]

def write_json(path, obj):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default))

def sample_surface(v, f, n, rng):
    tri=v[f]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); area=np.linalg.norm(cross,axis=1)
    if not np.isfinite(area).all() or area.sum()<=0: raise ValueError('Invalid surface')
    ids=rng.choice(len(f), size=n, p=area/area.sum())
    uv=rng.random((n,2)); r=np.sqrt(uv[:,0]); bary=np.c_[1-r,r*(1-uv[:,1]),r*uv[:,1]]
    pts=(tri[ids]*bary[:,:,None]).sum(1)
    normals=cross[ids]/np.maximum(area[ids,None],1e-300)
    return pts,normals,ids

def face_quality(v,f):
    t=v[f]; e=np.roll(t,-1,axis=1)-t; l=np.linalg.norm(e,axis=2)
    a2=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    q=2*np.sqrt(3)*a2/np.maximum((l*l).sum(1),1e-300)
    angles=[]
    for i in range(3):
        a=t[:,(i+1)%3]-t[:,i]; b=t[:,(i+2)%3]-t[:,i]
        cs=(a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-300)
        angles.append(np.rad2deg(np.arccos(np.clip(cs,-1,1))))
    return {'min_angle_deg':float(np.min(angles)), 'q_min':float(q.min()),'q05':float(np.quantile(q,.05)), 'q50':float(np.median(q)), 'area_min':float(.5*a2.min()),'small_area_count_1e-10':int((a2<2e-10).sum()),'area_total':float(.5*a2.sum()),'volume_signed':float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)}

def save_obj(path,v,f):
    with open(path,'w') as s:
        for p in v: s.write('v '+' '.join(f'{x:.17g}' for x in p)+'\n')
        for a in f: s.write('f '+' '.join(str(int(i)+1) for i in a)+'\n')
