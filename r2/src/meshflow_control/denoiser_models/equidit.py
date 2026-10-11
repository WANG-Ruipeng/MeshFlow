# Adapted from MeshFlow db86bc32423c316c00bcc2f645a26243e8010f20.
# Copyright (c) 2026 qiisun. See LICENSE_MeshFlow and PROVENANCE.md.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .pos_embed import get_embedder
from .rmsnorm import RMSNorm
from .attention import SelfAttention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, nonlin_type="swiglu"):
        super().__init__()
        mlp_hidden_dim = int(dim * mult)
        if nonlin_type == "geglu":
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim * 2),
                GEGLU(),
                nn.Linear(mlp_hidden_dim, dim)
            )
        elif nonlin_type == "silu":
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, dim)
            )
        elif nonlin_type == "swiglu":
            raise NotImplementedError("The fixed official denoiser uses SiLU, not SwiGLU")
        else:
            raise ValueError(f"Invalid nonlin_type: {nonlin_type}")

    def forward(self, x, mask=None, version=3):
        if version == 3:
            if mask is not None:
                B, N, D = x.shape
                # Maks is (B, N), X is (B, N, D)
                x = x.reshape(B, N//3, 3, D)[mask]
            y = self.net(x)
            if mask is not None:
                y_ = torch.zeros(B, N//3, 3, D, device=x.device, dtype=y.dtype)
                y_[mask] = y
                y = y_.reshape(B, N, D)
            return y
        else:
            B, N, D = x.shape
            x = x[mask]
            y = self.net(x)
            y_ = torch.zeros(B, N, D, device=x.device, dtype=y.dtype)
            y_[mask] = y
            y = y_.reshape(B, N, D)
            return y

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class XEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, pe_freq, use_coord_encoding=True, version=2):
        super().__init__()
        self.version = version
        embed_fn, frequency_embedding_size = get_embedder(pe_freq, input_dims=3 if version > 1 else 9)  # 10 -> 20
        self.embed_fn = embed_fn
        self.input_dim = frequency_embedding_size if use_coord_encoding else 9
        self.use_coord_encoding = use_coord_encoding
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.hidden_size = hidden_size

    def x_embedding(self, x, max_period=1.): # [N, 9] -> [N, 180]
        return self.embed_fn(x / max_period + 0.5) # renormalize to [0, 1]
    
    def forward(self, x):
        if self.version > 1:
            b, n, _ = x.shape
            x = x.view(b, n*3, 3)
        if self.use_coord_encoding:
            x = self.x_embedding(x=x)
        x_emb = self.mlp(x) # [bs, N*3, hidden_dim]
        if self.version > 1:
            x_emb = x_emb.view(b, n, 3, self.hidden_size)
            x_emb_2 = x_emb.view(b*n, 3, self.hidden_size) # keep the order 
            x_emb_1 = x_emb.mean(dim=2) # pooled face feature
            return x_emb_1, x_emb_2
        else:
            return x_emb

class VertexCrossAttention(nn.Module):
    def __init__(self, modulation_type="add", hidden_dim=1024):
        super().__init__()
        self.modulation_type = modulation_type
        if self.modulation_type == "concat":
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
    
    def forward(self, x, x_):
        # [B, 3, C] x [B, 1, C] -> [B, 3, C]
        if self.modulation_type == "add":
            return  x + x_.repeat(1, 3, 1)
        elif self.modulation_type == "mult":
            return  x * x_.repeat(1, 3, 1)
        elif self.modulation_type == "concat":
            return self.linear(torch.cat([x, x_.repeat(1, 3, 1)], dim=-1))
        else:
            raise ValueError(f"Invalid modulation type: {self.modulation_type}")

class DiTLayer(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True, mixed_precision='bf16', 
                version=3, 
                nonlin_type="silu", # 'swiglu'
                use_rmsnorm=False,
                mlp_ratio=4.0,
                use_qknorm=False, # used for stablizing training
                ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.version = version

        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        
        else:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
            
        self.attn = SelfAttention(dim, num_heads, mixed_precision=mixed_precision)
        self.mlp = FeedForward(dim, mult=mlp_ratio, nonlin_type=nonlin_type)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
    def forward(self, x, c, mask=None):
        if self.version <= 2:
            if self.training and self.gradient_checkpointing:
                return checkpoint(self._forward, x, c, mask=mask, use_reentrant=False)
            elif self.version <= 2:
                return self._forward(x, c, mask=mask)
        else:
            if self.training and self.gradient_checkpointing:
                return checkpoint(self._forward_v3, x, c, mask=mask, use_reentrant=False)
            else:
                return self._forward_v3(x, c, mask=mask)
    
    def _forward(self, x, c, mask=None):
        # x: [B, N, C], hidden states
        # t_adaln: [B, 6, C], timestep embedding of adaln
        # return: [B, N, C], updated hidden states
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=mask) # masking padded tokens to avoid the attention to see the future
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp),
                                                mask=mask, version =self.version)
        return x
    
    def _forward_v3(self, x, c, mask=None):
        # x: [B, N*3, C], hidden states
        # t_adaln: [B, 6, C], timestep embedding of adaln
        # return: [B, N, 3, C], updated hidden states
        b, n_vertex, _ = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)                
        x_face = x.reshape(b, n_vertex//3, 3, -1).mean(dim=2) # [B, n_face, C]
        x_face = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x_face), shift_msa, scale_msa), mask=mask) 
        x = x + x_face.unsqueeze(2).repeat(1,1,3,1).reshape(b, n_vertex, self.dim) # [B, n_face, 3, C]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp),
                                                mask=mask) # [B, n_vertex, C]
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels, input_size=None, use_rmsnorm=False):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(input_size)
        self.linear = nn.Linear(input_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * input_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
