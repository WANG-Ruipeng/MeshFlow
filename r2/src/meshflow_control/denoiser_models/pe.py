# Adapted from MeshFlow db86bc32423c316c00bcc2f645a26243e8010f20.
# Copyright (c) 2026 qiisun. See LICENSE_MeshFlow and PROVENANCE.md.
import torch
from torch import nn
import numpy as np

class NeRFEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        self.output_dim = (3 * (2 * num_freqs + (1 if include_input else 0)))

    def forward(self, x):
        embed_fns = []
        
        if self.include_input:
            embed_fns.append(x)
            
        for freq in self.freq_bands:
            embed_fns.append(torch.sin(x * freq * np.pi))
            embed_fns.append(torch.cos(x * freq * np.pi))
        return torch.cat(embed_fns, dim=-1)
