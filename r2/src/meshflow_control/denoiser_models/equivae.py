# Adapted from MeshFlow db86bc32423c316c00bcc2f645a26243e8010f20.
# Copyright (c) 2026 qiisun. See LICENSE_MeshFlow and PROVENANCE.md.
import torch
from torch import nn
import numpy as np
from .utils import get_2d_sincos_pos_embed
from .equidit import LabelEmbedder, XEmbedder, VertexCrossAttention, DiTLayer, FinalLayer

class AutoencoderKL(nn.Module):
    def __init__(self, 
                 hidden_dim=768,
                 num_layers=6,
                num_heads=12,
                 latent_channels=3,
                 decoder_type="reg",
                 num_bins=256,
                 use_rmsnorm=True,
                 face_bin=10,
                 deterministic=False,
                 use_identity_encoder=False,
                 fixed_std=0.0):
        super().__init__()
        if not use_identity_encoder:
            raise ValueError("Only the released identity-encoder denoiser is supported")
        
        self.latent_channels = latent_channels
        self.num_bins = num_bins
        
        self.use_identity_encoder = use_identity_encoder
        self.fixed_std = fixed_std
        self.deterministic = deterministic

           
        if self.use_identity_encoder:
            self.encoder = None 
            self.quant_linear = None 
            self.input_proj = nn.Identity()
            from .pe import NeRFEncoding
            self.pe_nerf = NeRFEncoding(num_freqs=12)
        else:
            self.encoder = Model(hidden_dim=hidden_dim, model_type='encoder', num_bins=num_bins, use_rmsnorm=use_rmsnorm, face_bin=face_bin,
                                 num_layers=num_layers,
                             num_heads=num_heads) # encoder 
            self.quant_linear = nn.Linear(hidden_dim, 2 * latent_channels) # 9 -> 4*2
        
        self.post_quant_linear = nn.Linear(latent_channels if not self.use_identity_encoder else self.pe_nerf.output_dim, hidden_dim)
        self.decoder = Model(hidden_dim=hidden_dim, model_type='decoder', decoder_type=decoder_type, num_bins=num_bins, use_rmsnorm=use_rmsnorm, face_bin=face_bin,
                             use_identity_encoder=use_identity_encoder,
                             num_layers=num_layers,
                             num_heads=num_heads,) # decoder   

    def decode(self, z, cond, mask):
        """
        解码过程：输入 Latent -> 重建图像
        """
        z = self.post_quant_linear(z)
        dec = self.decoder(z, cond, mask)
        return dec


class Model(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, max_length=800,
                 num_layers=6, gradient_checkpointing=True, 
                 class_dropout_prob=0.1, use_coord_encoding=True, 
                 version=3, pe_freq=20, mixed_precision='bf16',
                 use_dit_like_pe=False, face_cond=True, face_bin=100,
                 use_rmsnorm=False, 
                 model_type='encoder',
                 decoder_type="reg",
                 num_bins=512,
                 use_identity_encoder=False):
        super().__init__()
        self.model_type = model_type
        self.decoder_type = decoder_type
        self.use_dit_like_pe = use_dit_like_pe
        self.kl_weight = 1e-6
        self.use_l1_loss = True
        self.use_identity_encoder = use_identity_encoder

        input_dim = 3 if version > 1 else 9
        self.version = version
        self.use_coord_encoding = use_coord_encoding
        self.hidden_size = hidden_dim
        # project input
        if model_type == 'encoder':
            self.x_embedder = XEmbedder(hidden_dim, pe_freq=pe_freq, use_coord_encoding=use_coord_encoding, version=version)
            
        # positional encoding (just use a learnable positional encoding)
        if use_dit_like_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) / hidden_dim ** 0.5)
        
        # timestep encoding
        self.face_cond = face_cond
        if face_cond:
            self.uncond_y = max_length + 1
            self.num_classes = max_length//face_bin
            self.y_embedder = LabelEmbedder(max_length//face_bin, hidden_dim, class_dropout_prob) # 16+1
            self.face_bin = face_bin

        # transformer layers
        self.layers = nn.ModuleList([DiTLayer(hidden_dim, num_heads, gradient_checkpointing, mixed_precision=mixed_precision, version=version, use_rmsnorm=use_rmsnorm) for _ in range(num_layers)])

        # project out        
        modulation_type = "mult" 
        self.vertex_cross_attn = VertexCrossAttention(
            modulation_type=modulation_type, hidden_dim=hidden_dim)
        if model_type == 'decoder':
            if decoder_type == "cls":
                self.num_bins = num_bins
                self.final_layer = FinalLayer(hidden_dim, input_dim*num_bins, use_rmsnorm=use_rmsnorm) # input_dim: 9 for v2, 3 for v3
            elif decoder_type == "reg":
                self.final_layer = FinalLayer(hidden_dim, input_dim, use_rmsnorm=use_rmsnorm) # input_dim: 9 for v2, 3 for v3
        self.initialize_weights()
    

    def initialize_weights(self):
        # Initialize transformer layers:
        # 对于VAE来说，不能0初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.use_dit_like_pe:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 1)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        if self.face_cond:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize embedding MLP:   
        if self.model_type == "encoder":     
            nn.init.normal_(self.x_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp[2].weight, std=0.02)
        
        if self.use_identity_encoder:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, y, mask=None):
        # x: [B, N, C], hidden states
        # c: [B, M, C]
        # y: [B,], face number
        # mask: [B, N]
        # return: [B, N, C], updated hidden states
        if self.model_type == 'decoder':
            B, _, C = x.shape
            x = x.view(B, -1, 3, C)
            N = x.shape[1]
            x_ = x.view(-1, 3, C) # keep the order 
            x = x.mean(dim=2) # pooled face feature
        else:
            B, N, _ = x.shape
            if self.version > 1:
                x, x_ = self.x_embedder(x) # [B, N, C], [B*N, 3, C]
            else:
                x = self.x_embedder(x) # [B, N, C]
            
        # positional encoding (default: None)
        if self.use_dit_like_pe: 
            x = x + self.pos_embed[:, :N, :]

        # only face encoding
        if self.face_cond:
            y = self.y_embedder(y // self.face_bin, self.training)
            c = y

        # transformer layers
        if self.version > 2: # v3
            x = x_.reshape(B, N*3, -1) # [B, N*3, C]
            
        for layer in self.layers:
            x = layer(x, c, mask=mask)
        
        # project out
        if self.version == 2:
            x = x.view(B*N, 1, self.hidden_size)
            x = self.vertex_cross_attn(x_, x) # [B*N, 3, C]
            x = x.view(B, N*3, -1)
        
        if self.model_type == 'encoder':
            return x # [b, N*3, C]
        elif self.model_type == 'decoder':
            x = self.final_layer(x, c) # v2: [B, N*3, 3]
            if self.decoder_type == "cls":
                x = x.view(B, N, 9, self.num_bins) # [B, N, 9, num_bins]
            elif self.decoder_type == "reg":
                x = x.view(B, N, 9) # [B, N, 9, num_bins]
            return x
