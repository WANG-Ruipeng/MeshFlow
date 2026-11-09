import numpy as np
import torch
from torch import nn

class DomainVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,48),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(3+3+96+18,128),nn.SiLU(),nn.Linear(128,96),nn.SiLU(),nn.Linear(96,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def context(self,points):
        z=self.encoder(points)
        return torch.cat((z.mean(1),z.amax(1)),-1)
    def forward(self,u,t,points,context=None):
        # All query-dependent features are smooth in u: no query kNN or nearest-face selection.
        dirs,lr=points[...,:3],points[...,3]
        dot=torch.einsum('bnd,bmd->bnm',u,dirs)
        local=[]
        for beta in (8.,32.,128.):
            w=torch.softmax(beta*dot,-1)
            avg=torch.einsum('bnm,bm->bn',w,lr)
            second=torch.einsum('bnm,bm->bn',w,lr.square())-avg.square()
            mean_dir=torch.einsum('bnm,bmd->bnd',w,dirs)
            cov=torch.einsum('bnm,bmd->bnd',w,lr[...,None]*dirs)-avg[...,None]*mean_dir
            cov=cov-(cov*u).sum(-1,keepdim=True)*u
            density=torch.logsumexp(beta*(dot-1),-1)/beta
            local.extend([avg[...,None],second[...,None],cov,density[...,None]])
        if context is None:context=self.context(points)
        if t.ndim==1:t=t[:,None]
        time=torch.cat((t,torch.sin(torch.pi*t),torch.cos(torch.pi*t)),-1)[:,None,:].expand(-1,u.shape[1],-1)
        h=torch.cat((u,time,context[:,None,:].expand(-1,u.shape[1],-1),*local),-1)
        raw=self.head(h)
        return raw-(raw*u).sum(-1,keepdim=True)*u


def point_features(points):
    rad=np.linalg.norm(points,axis=-1,keepdims=True)
    return np.concatenate((points/rad,np.log(rad)),-1).astype('float32')
