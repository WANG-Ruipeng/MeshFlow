"""CPU-only geometry and statistics. No learned model, no radial reference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import json, hashlib, math
from collections import defaultdict, Counter
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    def conv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(str(type(x)))
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=conv,allow_nan=False)+'\n')

def save_mesh(p,v,f):
    # 17 significant decimal digits roundtrip binary64 exactly.
    with open(p,'w') as out:
        for x in v: out.write('v '+' '.join(format(float(a),'.17g') for a in x)+'\n')
        for t in f: out.write('f '+' '.join(str(int(i)+1) for i in t)+'\n')

def load_mesh(p):
    m=trimesh.load(p,process=False,force='mesh');return np.asarray(m.vertices),np.asarray(m.faces)

def sample(v,f,n,seed):
    rng=np.random.default_rng(seed);tri=v[f]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);a=np.linalg.norm(cross,axis=1)
    idx=rng.choice(len(f),n,p=a/a.sum());uv=rng.random((n,2));r=np.sqrt(uv[:,0])
    w=np.c_[1-r,r*(1-uv[:,1]),r*uv[:,1]]
    return np.einsum('ij,ijk->ik',w,tri[idx]),cross[idx]/a[idx,None],idx

class SurfaceDistance:
    """Point-to-triangle distance, conservative centroid radius candidate search."""
    def __init__(self,v,f):
        self.v=v;self.f=f;self.t=v[f];self.c=self.t.mean(1)
        self.r=np.linalg.norm(self.t-self.c[:,None],axis=2).max(1)
        self.vtree=cKDTree(v);self.tree=cKDTree(self.c)
        cross=np.cross(self.t[:,1]-self.t[:,0],self.t[:,2]-self.t[:,0])
        self.norm=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-300)
    def query(self,p):
        dd=[];ii=[];cp=[]
        for st in range(0,len(p),128):
            q=p[st:st+128];bound=self.vtree.query(q)[0]
            # A nearest mesh vertex gives an upper bound for nearest-surface distance.
            # Every face which could beat it lies in this centroid ball.
            lists=self.tree.query_ball_point(q,bound+self.r.max()+1e-12)
            cnt=np.array([len(x) for x in lists]);pid=np.repeat(np.arange(len(q)),cnt)
            tid=np.concatenate(lists).astype(int)
            keep=np.linalg.norm(self.c[tid]-q[pid],axis=1)<=bound[pid]+self.r[tid]+1e-12
            pid=pid[keep];tid=tid[keep]
            near=trimesh.triangles.closest_point(self.t[tid],q[pid])
            d2=np.sum((near-q[pid])**2,axis=1)
            order=np.lexsort((tid,d2,pid));uniq=np.r_[True,np.diff(pid[order])!=0]
            ix=order[uniq];assert len(ix)==len(q)
            dd.extend(np.sqrt(d2[ix]));ii.extend(tid[ix]);cp.extend(near[ix])
        return np.array(dd),np.array(ii),np.array(cp)

def metrics(v,f,gv,gf,gt_samples,gt_normals,seed,n=8192):
    q,qn,_=sample(v,f,n,seed)
    gd=SurfaceDistance(gv,gf);pd=SurfaceDistance(v,f)
    d0,i0,_=gd.query(q);d1,i1,_=pd.query(gt_samples)
    diag=np.linalg.norm(gv.max(0)-gv.min(0))
    cos0=np.clip(np.einsum('ij,ij->i',qn,gd.norm[i0]),-1,1)
    cos1=np.clip(np.einsum('ij,ij->i',gt_normals,pd.norm[i1]),-1,1)
    angles=np.degrees(np.arccos(np.abs(np.r_[cos0,cos1])))
    signed_angles=np.degrees(np.arccos(np.r_[cos0,cos1]))
    return dict(rms_pct_bboxdiag=100*np.sqrt(.5*(np.mean(d0*d0)+np.mean(d1*d1)))/diag,
                p95_pct_bboxdiag=100*np.quantile(np.r_[d0,d1],.95)/diag,
                pred_to_gt_rms_pct=100*np.sqrt(np.mean(d0*d0))/diag,
                gt_to_pred_rms_pct=100*np.sqrt(np.mean(d1*d1))/diag,
                normal_abs_mean_deg=angles.mean(),normal_abs_p95_deg=np.quantile(angles,.95),
                normal_oriented_mean_deg=signed_angles.mean(),samples_per_direction=n)

def mesh_audit(v,f):
    nv=len(v);nf=len(f)
    edgefaces=defaultdict(list); balance=Counter(); vf=defaultdict(list)
    for j,(a,b,c) in enumerate(f):
        for a0,b0 in [(a,b),(b,c),(c,a)]:
            e=tuple(sorted((int(a0),int(b0))));edgefaces[e].append(j);balance[e]+=1 if a0<b0 else -1
        for x in (a,b,c): vf[int(x)].append(j)
    edges=list(edgefaces);ei={e:i for i,e in enumerate(edges)}
    counts=np.array([len(edgefaces[e]) for e in edges]);E=len(edges)
    rr=np.array(edges,dtype=int)
    graph=coo_matrix((np.ones(E*2),(np.r_[rr[:,0],rr[:,1]],np.r_[rr[:,1],rr[:,0]])),shape=(nv,nv)).tocsr()
    b0,lab=connected_components(graph,directed=False)
    # GF(2) rank of boundary_2, with one integer bit column per face.
    piv={}
    for a,b,c in f:
        col=0
        for x,y in [(a,b),(b,c),(c,a)]:col^=1<<ei[tuple(sorted((int(x),int(y))))]
        while col:
            k=col.bit_length()-1
            if k in piv:col^=piv[k]
            else:piv[k]=col;break
    rank2=len(piv);b1=E-(nv-b0)-rank2;b2=nf-rank2
    nonman_v=[]
    for x,faces in vf.items():
        link=defaultdict(list)
        for j in faces:
            other=[int(a) for a in f[j] if a!=x]
            if len(other)!=2:continue
            a,b=other;link[a].append(b);link[b].append(a)
        if not link:nonman_v.append(x);continue
        vis=set();stack=[next(iter(link))]
        while stack:
            a=stack.pop()
            if a in vis:continue
            vis.add(a);stack.extend(link[a])
        degrees=sorted(len(a) for a in link.values())
        cycle=all(d==2 for d in degrees)
        path=degrees.count(1)==2 and all(d in (1,2) for d in degrees)
        if len(vis)!=len(link) or not(cycle or path): nonman_v.append(x)
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);doublearea=np.linalg.norm(cross,axis=1)
    ang=[]
    for i in range(3):
        a=tri[:,(i+1)%3]-tri[:,i];b=tri[:,(i+2)%3]-tri[:,i]
        den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        ang.extend(np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/np.maximum(den,1e-300),-1,1))))
    bad_or=sum(counts[i]==2 and balance[e]!=0 for i,e in enumerate(edges))
    closed_man=bool(np.all(counts==2) and not nonman_v and bad_or==0 and np.all(doublearea>0))
    return dict(vertices=nv,faces=nf,edges=E,boundary_edges=int(np.sum(counts==1)),
                nonmanifold_edges=int(np.sum(counts>2)),nonmanifold_vertices=len(nonman_v),
                inconsistent_oriented_edges=int(bad_or),duplicate_faces=nf-len(np.unique(np.sort(f,axis=1),axis=0)),
                float_zero_area_faces=int(np.sum(doublearea==0)),near_degenerate_faces=int(np.sum(doublearea<1e-14)),
                beta0=int(b0),beta1=int(b1),beta2=int(b2),euler=int(nv-E+nf),
                closed_oriented_2manifold=closed_man,genus_sum=int(b1//2) if closed_man else None,
                signed_volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6),
                angle_q05_deg=float(np.quantile(ang,.05)),angle_min_deg=float(min(ang)),
                nonmanifold_vertex_ids=nonman_v)
