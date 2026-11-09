"""All three internal arms read exactly the same saved mainfield; no target input."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from .legacy.geometry_task import fibonacci,hull_faces,ray_owner
from .legacy.chart import dets,solid_angles
from .legacy.networks import DomainVelocity
from .legacy.deform import deform
from .legacy.common import sample_surface

ASSETS=Path(__file__).resolve().parent/'assets'
class RadialSupport:
    def __init__(self,path=ASSETS/'support.npz'):
        with np.load(path,allow_pickle=False) as s:self.u=np.asarray(s['u'],np.float64);self.faces=np.asarray(s['faces'],np.int64)
        self.inv=np.linalg.inv(self.u[self.faces]);cc=self.u[self.faces].mean(1);cc/=np.linalg.norm(cc,axis=1,keepdims=True);self.tree=cKDTree(cc)
    def query(self,nodal_logr,q):
        y=np.asarray(nodal_logr,np.float64);q=np.asarray(q,np.float64)
        if y.shape!=(len(self.u),) or not np.isfinite(y).all():raise ValueError('Invalid node vector')
        with np.errstate(over='raise',invalid='raise',under='ignore'):rs=np.exp(y)
        if not np.isfinite(rs).all() or (rs<=0).any():raise FloatingPointError('Invalid positive radius; do not clamp or retry')
        _,cand=self.tree.query(q,k=12);bb=np.einsum('ni,nkij->nkj',q,self.inv[cand]);ok=bb.min(-1)>=-1e-12
        pick=ok.argmax(1);fi=cand[np.arange(len(q)),pick];be=bb[np.arange(len(q)),pick]
        missing=~ok.any(1)
        if missing.any():
            fi[missing]=ray_owner(self.u,self.faces,q[missing])
            if (fi[missing]<0).any():raise ValueError('support chart gap')
            be[missing]=np.einsum('ni,nij->nj',q[missing],self.inv[fi[missing]])
        den=(be/rs[self.faces[fi]]).sum(-1)
        if not np.isfinite(den).all() or (den<=0).any():raise ValueError('Invalid query')
        return 1/den
    def mesh(self,y):
        y=np.asarray(y,np.float64)
        if y.shape!=(len(self.u),) or not np.isfinite(y).all():raise ValueError('bad support nodes')
        with np.errstate(over='raise',invalid='raise'):r=np.exp(y)
        if (r<=0).any():raise ValueError('underflow radius')
        return self.u*r[:,None],self.faces

class MeshExporter:
    def __init__(self,a_seed=10,assets=ASSETS):
        self.support=RadialSupport(Path(assets)/'support.npz');self.flow=DomainVelocity()
        self.flow.load_state_dict(torch.load(Path(assets)/f'original_A_{a_seed}.pt',map_location='cpu',weights_only=True))
        self.flow.eval().requires_grad_(False);self.templates={};self.a_seed=a_seed
    def template(self,F):
        if F<4 or F%2:raise ValueError('Even F>=4 required')
        if F not in self.templates:
            u=fibonacci(F//2+2);self.templates[F]=(u,hull_faces(u))
        return self.templates[F]
    def proxy(self,y):
        v,f=self.support.mesh(y);return sample_surface(v,f,256,np.random.default_rng(4501))[0]
    def export(self,y,F,arm,proxy=None):
        t=time.perf_counter()
        if arm=='qem':
            v,f=self.support.mesh(y);v,f=vtk_qem(v,f,F)
            return {'vertices':v,'faces':f},{'arm':arm,'requested_faces':F,'actual_faces':len(f),'budget_match':len(f)==F,'seconds':time.perf_counter()-t}
        base,f=self.template(F)
        if arm=='fixed':u=base.copy();alpha=np.ones(1)
        elif arm=='a':u,path,alpha,_=deform(self.flow,self.proxy(y) if proxy is None else proxy,base,f,True,32)
        else:raise ValueError(arm)
        r=self.support.query(y,u);v=u*r[:,None]
        return {'vertices':v,'faces':f,'u':u,'r':r,'alpha':alpha},{'arm':arm,'requested_faces':F,'actual_faces':len(f),'budget_match':len(f)==F,'seconds':time.perf_counter()-t,'min_alpha':float(alpha.min()),'shrunk_steps':int((alpha<1).sum()),'min_direction_det':float(dets(u,f).min()),'solid_angle_sum':float(solid_angles(u,f).sum())}

def vtk_qem(v,f,target):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    if target>len(f):raise ValueError('QEM target exceeds support faces')
    pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(v,np.float64),deep=True))
    ca=vtk.vtkCellArray();ca.SetData(numpy_to_vtk(np.arange(0,len(f)*3+1,3,dtype=np.int64),deep=True),numpy_to_vtk(np.asarray(f,np.int64).ravel(),deep=True))
    pd=vtk.vtkPolyData();pd.SetPoints(pts);pd.SetPolys(ca)
    q=vtk.vtkQuadricDecimation();q.SetInputData(pd);q.SetTargetReduction(1-target/len(f));q.AttributeErrorMetricOff();q.VolumePreservationOff()
    if hasattr(q,'SetRegularize'):q.SetRegularize(False)
    q.Update();out=q.GetOutput()
    if out.GetNumberOfPoints()==0:raise RuntimeError('Empty QEM output')
    off=vtk_to_numpy(out.GetPolys().GetOffsetsArray())
    if not np.all(np.diff(off)==3):raise RuntimeError('Nontriangular QEM output')
    v=vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64);f=vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1,3).astype(np.int64)
    # Remove only unreferenced vertices; no capping, retries, or target-face adjustment.
    used,inv=np.unique(f,return_inverse=True);return v[used],inv.reshape(-1,3)
