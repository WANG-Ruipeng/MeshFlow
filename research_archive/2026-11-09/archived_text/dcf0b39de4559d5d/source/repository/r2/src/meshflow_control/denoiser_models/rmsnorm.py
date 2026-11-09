"""Local implementation of RMS normalization; not a copied Llama class.

Keep checkpoint parameter name `weight`, epsilon and FP32 reduction/cast order
identical to the audited denoiser. There is no fairscale dependency.
"""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def forward(self,values):
        work=values.float()
        reciprocal_rms=torch.rsqrt(work.pow(2).mean(-1,keepdim=True)+self.eps)
        normalized=(work*reciprocal_rms).to(values.dtype)
        return normalized*self.weight
