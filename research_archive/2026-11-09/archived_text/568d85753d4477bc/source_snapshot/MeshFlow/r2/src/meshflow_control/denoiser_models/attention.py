# Adapted from MeshFlow db86bc32423c316c00bcc2f645a26243e8010f20.
# Copyright (c) 2026 qiisun. See LICENSE_MeshFlow and PROVENANCE.md.
"""Unmasked SDPA-only backend. No FlashAttention import or global module alias."""
import torch
from torch import nn
from torch.nn import functional as F

def attention(q,k,v,mask_q=None,mask_kv=None,dropout=0.0,causal=False,backend='pytorch-sdpa'):
    if backend!='pytorch-sdpa' or mask_q is not None or mask_kv is not None or dropout!=0 or causal:
        raise ValueError('Official single-mesh denoising requires unmasked noncausal dropout-free SDPA')
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]:
        raise ValueError('Expected compatible [batch,sequence,head,channel] tensors')
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        attn_mask=None,dropout_p=0.0,is_causal=False).transpose(1,2)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, output_dim=None, dropout=0, causal=False, mixed_precision='bf16'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.mixed_precision = mixed_precision
    
    def forward(self, x, mask=None):
        # x: [B, N, C]
        # mask: [B, N]
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.chunk(3, dim=0) # [3, B, N, H, D] -> 3 * [1, B, N, H, D]
        x = attention(q[0], k[0], v[0], mask_q=mask, mask_kv=mask, dropout=self.dropout, causal=self.causal, backend='pytorch-sdpa' if self.mixed_precision=='fp32' else 'flash-attn') # [B, N, H, D]
        x = self.out_proj(x.reshape(B, N, -1))
        return x
