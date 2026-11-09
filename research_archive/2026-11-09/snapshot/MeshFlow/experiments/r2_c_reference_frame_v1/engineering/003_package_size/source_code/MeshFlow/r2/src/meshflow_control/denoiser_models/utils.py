"""Local standard sine/cosine positional formula; no copied IO utilities."""
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim,pos):
    if embed_dim%2:raise ValueError('Embedding dimension must be even')
    frequencies=np.arange(embed_dim//2,dtype=np.float64)/(embed_dim/2.)
    frequencies=1./10000**frequencies
    phase=np.einsum('m,d->md',np.asarray(pos).reshape(-1),frequencies)
    return np.concatenate([np.sin(phase),np.cos(phase)],axis=1)

def get_2d_sincos_pos_embed_from_grid(embed_dim,grid):
    if embed_dim%2:raise ValueError('Embedding dimension must be even')
    return np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[0]),
                           get_1d_sincos_pos_embed_from_grid(embed_dim//2,grid[1])],axis=1)

def get_2d_sincos_pos_embed(embed_dim,grid_size,cls_token=False,extra_tokens=0):
    axis=np.arange(grid_size,dtype=np.float32)
    grid=np.stack(np.meshgrid(axis,axis),axis=0).reshape(2,1,grid_size,grid_size)
    encoded=get_2d_sincos_pos_embed_from_grid(embed_dim,grid)
    if cls_token and extra_tokens>0:encoded=np.concatenate([np.zeros((extra_tokens,embed_dim)),encoded],axis=0)
    return encoded
