import numpy as np
from scipy.spatial import ConvexHull
from numba import njit
from .chart import dets

def fibonacci(n):
 i=np.arange(n);z=1-2*(i+.5)/n;t=i*np.pi*(3-np.sqrt(5));s=np.sqrt(1-z*z)
 return np.c_[s*np.cos(t),s*np.sin(t),z]


def hull_faces(u):
 f=ConvexHull(u).simplices.copy();neg=dets(u,f)<0;f[neg]=f[neg][:,[0,2,1]]
 # deterministic ordered face list, without changing orientation
 f=np.array([np.roll(t,-np.argmin(t)) for t in f]);f=f[np.lexsort((f[:,2],f[:,1],f[:,0]))]
 assert len(f)==2*len(u)-4 and dets(u,f).min()>0
 return f


def params(pair,family):
 rng=np.random.default_rng(1709700+int(pair));R,_=np.linalg.qr(rng.normal(size=(3,3)))
 if np.linalg.det(R)<0:R[:,0]*=-1
 axes=rng.uniform([1.03,.91,.80],[1.25,1.08,.98]);amps=rng.uniform(.55,.95,3) if family else np.zeros(3)
 sx=rng.uniform(.10,.17,3);sy=sx*rng.uniform(1.,2.0,3)
 # Three separated features and their tangent frames share the global rotation.
 centers=R.T.copy();t1=np.roll(R.T,1,axis=0);t2=np.cross(centers,t1)
 return dict(pair=int(pair),family=int(family),R=R,axes=axes,amps=amps,sx=sx,sy=sy,centers=centers,t1=t1,t2=t2,scale=1.)


def radius_np(u,p):
 loc=u@p['R'];r=1/np.sqrt(((loc/p['axes'])**2).sum(-1))
 a=u@p['t1'].T/p['sx'];b=u@p['t2'].T/p['sy'];c=(1-u@p['centers'].T)/np.sqrt(p['sx']*p['sy'])
 s=a*a+b*b+c*c;r=r+(p['amps']*np.exp(-.5*(s+1e-6)**.7)).sum(-1)
 return r/float(p['scale'])


@njit(cache=True)
def ray_owner(u,f,q):
 cr=np.empty((len(f),3,3))
 for j in range(len(f)):
  a,b,c=u[f[j,0]],u[f[j,1]],u[f[j,2]]
  cr[j,0]=np.cross(b,c);cr[j,1]=np.cross(c,a);cr[j,2]=np.cross(a,b)
 out=np.empty(len(q),np.int64)
 for k in range(len(q)):
  found=-1
  for j in range(len(f)):
   if np.dot(q[k],cr[j,0])>=-1e-13 and np.dot(q[k],cr[j,1])>=-1e-13 and np.dot(q[k],cr[j,2])>=-1e-13:
    found=j;break
  out[k]=found
 return out


def radial_at_rays(x,f,q,owner):
 tr=x[f[owner]];nor=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);return (nor*tr[:,0]).sum(1)/(nor*q).sum(1)
