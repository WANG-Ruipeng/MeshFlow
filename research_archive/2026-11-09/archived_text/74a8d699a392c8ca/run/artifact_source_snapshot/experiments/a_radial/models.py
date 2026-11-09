"""Locked width1024/z64 ShapeAE; archived state_dict keys remain compatible."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class ShapeAE(nn.Module):
    def __init__(self,n=2050,d=64,width=1024):
        super().__init__();self.d=d
        self.register_buffer('mean',torch.zeros(n));self.register_buffer('scale',torch.ones(()))
        self.register_buffer('basis',torch.zeros(n,d));self.register_buffer('zscale',torch.ones(d))
        self.enc=nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,d))
        self.dec=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,n))
        for last in (self.enc[-1],self.dec[-1]):nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    def encode(self,logr):
        x=(logr-self.mean)/self.scale
        return (x@self.basis)/self.zscale+self.enc(x)
    def decode_nodes(self,z):
        return self.mean+self.scale*((z*self.zscale)@self.basis.T+self.dec(z))
    def forward(self,y):
        z=self.encode(y);return self.decode_nodes(z),z

@torch.no_grad()
def fit_pca_train_only(model:ShapeAE,y:torch.Tensor)->dict:
    """Handles N=16: rank<=15, zero-pads unavailable modes; no noisy fake dimensions."""
    a=y.detach().cpu().numpy().astype(np.float64)
    if a.ndim!=2 or len(a)<2 or a.shape[1]!=len(model.mean) or not np.isfinite(a).all():raise ValueError('Invalid training-only PCA input')
    mean=a.mean(0);scale=float(np.sqrt(np.mean((a-mean)**2)))
    if scale<=1e-12:raise ValueError('All training observations identical')
    x=(a-mean)/scale;u,s,vh=np.linalg.svd(x,full_matrices=False)
    tol=max(x.shape)*np.finfo(np.float64).eps*s[0]
    rank=int(min(np.count_nonzero(s>tol),len(a)-1,model.d))
    basis=np.zeros((a.shape[1],model.d));zs=np.ones(model.d)
    for k in range(rank):
        vec=vh[k].copy();j=np.argmax(np.abs(vec));vec*=1 if vec[j]>=0 else -1
        basis[:,k]=vec;zs[k]=max(float(np.std(x@vec)),1e-3)
    for name,value in [('mean',mean),('scale',scale),('basis',basis),('zscale',zs)]:
        getattr(model,name).copy_(torch.as_tensor(value,dtype=getattr(model,name).dtype,device=getattr(model,name).device))
    return {'n_training_objects':len(a),'numeric_rank':rank,'latent_dim':model.d,'inactive_pca_columns':model.d-rank,
            'rank_threshold':float(tol),'fit_scope':'provided training objects only','closed_set_caveat':'rank15 PCA can interpolate16 training vectors; not evidence of generalization'}

class ShapeFM(nn.Module):
    def __init__(self,d=64,width=128):
        super().__init__();self.register_buffer('mu',torch.zeros(d));self.register_buffer('sd',torch.ones(d))
        self.net=nn.Sequential(nn.Linear(d+5,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,d))
    def forward(self,x,t):
        t=t.to(device=x.device,dtype=x.dtype).reshape(len(x),1)
        tt=torch.cat([t,torch.sin(torch.pi*t),torch.cos(torch.pi*t),torch.sin(2*torch.pi*t),torch.cos(2*torch.pi*t)],1)
        return self.net(torch.cat([x,tt],1))
    def standardize(self,z):return (z-self.mu)/self.sd
    def unstandardize(self,z):return z*self.sd+self.mu
    @torch.no_grad()
    def sample_standardized(self,noise,steps=32):
        if steps<1:raise ValueError('steps must be positive')
        x=noise.clone()
        for i in range(steps):x=x+self(x,torch.full((len(x),),i/steps,device=x.device,dtype=x.dtype))/steps
        if not torch.isfinite(x).all():raise FloatingPointError('FM nonfinite sample; record failure, do not resample')
        return x
    @torch.no_grad()
    def fit_normalization(self,train_codes,floor=1e-3):
        self.mu.copy_(train_codes.mean(0));raw=train_codes.std(0,unbiased=False)
        self.sd.copy_(raw.clamp_min(floor))
        return {'floor':floor,'floor_dimensions':int((raw<floor).sum()),'n_training_codes':len(train_codes),'mean_square':float(train_codes.square().mean())}
