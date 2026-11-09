"""Permutation-equivariant velocity models for the closed-set FM pilot.

State order is physical [x(3), mean normal(3), centered factor F(16)].
The complete local quadric is C=F F^T. Network inputs/outputs use z=(s-mean)/scale.
Transport is computed in physical units and converted back to chart velocity.
"""
import torch
from torch import nn

STATE_DIM = 22
K = 16
RIDGE_FACTOR = 1e-3


class DeepSetsVelocity(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = int(width)
        self.encoder = nn.Sequential(nn.Linear(STATE_DIM+1,width),nn.SiLU(),
                                     nn.Linear(width,width),nn.SiLU())
        self.decoder = nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),
                                     nn.Linear(width,width),nn.SiLU(),
                                     nn.Linear(width,STATE_DIM))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, z, t):
        if z.ndim != 3 or z.shape[-1] != STATE_DIM:
            raise ValueError('z must be (batch, elements, 22)')
        if t.numel() != z.shape[0]:
            raise ValueError('t must contain one FM time per batch item')
        times = t.reshape(-1,1,1).expand(-1,z.shape[1],1)
        point = self.encoder(torch.cat((z,times),dim=-1))
        mean = point.mean(dim=1,keepdim=True).expand_as(point)
        maximum = point.amax(dim=1,keepdim=True).expand_as(point)
        return self.decoder(torch.cat((point,mean,maximum),dim=-1))


def validate_chart(mean, scale):
    if mean.shape != (STATE_DIM,) or scale.shape != (STATE_DIM,):
        raise ValueError('mean/scale must have shape (22,)')
    if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale>0).all()):
        raise ValueError('Chart must be finite with strictly positive scales')
    for start,end in [(0,3),(3,6),(6,22)]:
        if not torch.allclose(scale[start:end],scale[start].expand(end-start),rtol=1e-6,atol=0):
            raise ValueError('Each x/n/F block must have one scalar scale')


def local_velocity_gradient(x, velocity, k=K, ridge_factor=RIDGE_FACTOR):
    """Differentiable ridge fit of v_j-v_i = L_i (x_j-x_i).

Neighbor identities come from current x, excluding self; h is kth distance.
There is no determinant guard: L is a velocity gradient, not a finite motion J.
No reference positions, target velocities, analytic Jacobians, or mesh enter.
"""
    if x.shape != velocity.shape or x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] <= k:
        raise ValueError('x/velocity must be matched (B,N,3), N>k')
    if not bool(torch.isfinite(x).all() and torch.isfinite(velocity).all()):
        raise FloatingPointError('Nonfinite positions or predicted velocity')
    distance = torch.cdist(x,x)
    mask = torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
    distance = distance.masked_fill(mask,float('inf'))
    distances,indices = torch.topk(distance,k,dim=-1,largest=False,sorted=True)
    h = distances[...,-1]
    if not bool(torch.isfinite(h).all() and (h>0).all()):
        raise FloatingPointError('Zero/nonfinite kth-neighbor scale; no repair')
    batch = torch.arange(x.shape[0],device=x.device)[:,None,None]
    dx = x[batch,indices]-x[:,:,None,:]
    dv = velocity[batch,indices]-velocity[:,:,None,:]
    X = dx/h[...,None,None]; V = dv/h[...,None,None]
    weights = torch.exp(-(distances/h[...,None])**2)
    weights = weights/weights.sum(dim=-1,keepdim=True)
    XtW = X.transpose(-1,-2)*weights[...,None,:]
    G = XtW@X
    lam = ridge_factor*G.diagonal(dim1=-2,dim2=-1).sum(dim=-1)/3
    eye = torch.eye(3,device=x.device,dtype=x.dtype)
    B = torch.linalg.solve(G+lam[...,None,None]*eye,XtW@V)
    L = B.transpose(-1,-2)
    if not bool(torch.isfinite(L).all()):
        raise FloatingPointError('Nonfinite ridge velocity gradient; no repair')
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(G.detach())
        # Clamping is diagnostic only; neither G nor the fit is modified.
        singular = eigenvalues.clamp_min(0).sqrt()
        rank = (singular>1e-6*singular[...,-1:]).sum(dim=-1)
        condition = singular[...,-1]/singular[...,0]
    return L,dict(neighbor_indices=indices,h=h,weights=weights,ridge_lambda=lam,
                  design_rank=rank,design_condition=condition)


def compose_velocity(raw, z, mean, scale, arm):
    """Return (total chart velocity, diagnostics); both arms execute the fit.

The transport term uses this same model's predicted physical position velocity,
with gradients retained. For direct it is measured but not added to the loss.
"""
    if arm not in ('direct','transport_residual'):
        raise ValueError('Unknown arm')
    if raw.shape != z.shape:
        raise ValueError('Raw velocity and state shapes differ')
    physical = z*scale+mean
    velocity_x = raw[...,:3]*scale[:3]
    L,diagnostics = local_velocity_gradient(physical[...,:3],velocity_x)
    F = physical[...,6:].reshape(*physical.shape[:-1],4,4)
    top = -L.transpose(-1,-2)@F[...,:3,:]
    physical_transport = torch.cat((top,torch.zeros_like(F[...,3:4,:])),dim=-2)
    chart_transport_F = physical_transport.flatten(start_dim=-2)/scale[6:]
    transport = torch.cat((torch.zeros_like(raw[...,:6]),chart_transport_F),dim=-1)
    total = raw+transport if arm=='transport_residual' else raw
    diagnostics.update(L=L,transport=transport,raw_velocity=raw,physical_F=F)
    return total,diagnostics


def total_velocity(model, z, t, mean, scale, arm):
    return compose_velocity(model(z,t),z,mean,scale,arm)
