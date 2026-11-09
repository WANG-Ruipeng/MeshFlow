import numpy as np
import torch
import time
from .networks import point_features
from .chart import retract

@torch.no_grad()
def deform(A,P,base,faces,guarded,steps=32):
    pf=torch.from_numpy(point_features(P))[None];context=A.context(pf);u=base.copy();path=[u.copy()];alphas=[]
    t0=time.perf_counter()
    for it in range(steps):
        ut=torch.tensor(u,dtype=torch.float32)[None];tt=torch.tensor([[it/steps]],dtype=torch.float32)
        delta=A(ut,tt,pf,context)[0].numpy().astype('float64')/steps
        delta-=np.sum(delta*u,axis=-1,keepdims=True)*u
        if guarded:u,alpha=retract(u,delta,faces)
        else:
            u=u+delta;u/=np.linalg.norm(u,axis=1,keepdims=True);alpha=1.
        path.append(u.copy());alphas.append(alpha)
    return u,np.array(path),np.array(alphas),time.perf_counter()-t0
