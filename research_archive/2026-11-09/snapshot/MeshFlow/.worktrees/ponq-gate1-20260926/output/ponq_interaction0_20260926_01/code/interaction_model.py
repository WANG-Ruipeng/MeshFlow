"""One zero-initialized pre-pooling residual block for SINGLE_STATE_INTERACTION0.

Original encoder/decoder parameter names are preserved. No slot/position IDs,
reference state, geometry, neighbor selection or target features are consumed.
"""
import torch
from torch import nn
from fm_model import DeepSetsVelocity,STATE_DIM

MODULE_INIT_SEED=260930
ARMS=('B0','B1','B2')


class PointResidual(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.linear1=nn.Linear(width,2*width);self.linear2=nn.Linear(2*width,width)
        self.activation=nn.SiLU()
        nn.init.zeros_(self.linear2.weight);nn.init.zeros_(self.linear2.bias)
    def forward(self,h):return h+self.linear2(self.activation(self.linear1(self.norm(h))))


class AttentionResidual(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True,bias=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
    def forward(self,h):
        x=self.norm(h);attention,_=self.attention(x,x,x,need_weights=False)
        return h+attention


class InteractionVelocity(DeepSetsVelocity):
    def __init__(self,arm='B0',width=128):
        if arm not in ARMS:raise ValueError('Unknown interaction arm')
        if width!=128:raise ValueError('Frozen interaction width is 128')
        # All modules initialize on CPU. Only the CPU default generator is seeded
        # inside this fork; CUDA RNG and the dedicated data generator are untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(MODULE_INIT_SEED)
            super().__init__(width=width)
            self.interaction=nn.Identity() if arm=='B0' else PointResidual(width) if arm=='B1' else AttentionResidual(width)
        self.arm=arm

    def forward_with_hidden(self,z,t):
        if z.ndim!=3 or z.shape[-1]!=STATE_DIM:raise ValueError('z must be (batch,elements,22)')
        if t.numel()!=z.shape[0]:raise ValueError('One FM time is needed per batch item')
        times=t.reshape(-1,1,1).expand(-1,z.shape[1],1)
        shared_hidden=self.interaction(self.encoder(torch.cat((z,times),dim=-1)))
        mean=shared_hidden.mean(dim=1,keepdim=True).expand_as(shared_hidden)
        maximum=shared_hidden.amax(dim=1,keepdim=True).expand_as(shared_hidden)
        raw_velocity=self.decoder(torch.cat((shared_hidden,mean,maximum),dim=-1))
        return raw_velocity,shared_hidden

    def forward(self,z,t):return self.forward_with_hidden(z,t)[0]

    def parameter_counts(self):
        total=sum(p.numel() for p in self.parameters());added=sum(p.numel() for p in self.interaction.parameters())
        return dict(total=total,original=total-added,added=added)