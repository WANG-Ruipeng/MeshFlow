"""Standard 32-point static cross-attention for the authorized 112-face sandbox.

The 32 observed points remain 32 tokens. Primitive groups supply only the
edge/face type lookup and the fixed sampling prior -log(points in that group).
There is no primitive pooling, group embedding, dynamic value or geometry query.
"""
from __future__ import annotations

from typing import Mapping
import torch
from torch import Tensor, nn

from .adapters import AdapterLayer, EncodedPrompts, point_features


class S32StaticAdapter(nn.Module):
    """OfficialWrapper-compatible four-site static 4-head attention branch."""
    def __init__(self, width: int = 768, layers: int = 4, query_chunk: int = 256):
        super().__init__()
        if width != 768 or layers != 4 or query_chunk < 1:
            raise ValueError("S32_static requires width768, four injection sites and a positive query chunk")
        self.arm = "S32_static"
        self.condition_encoder = nn.Sequential(nn.Linear(41, 128), nn.SiLU(), nn.Linear(128, 128))
        # The audited old static layer already has exactly LN/Wq/Wk/Wv/zero-Wo.
        # Its static branch ignores xt/t; it constructs no dynamic MLP.
        self.layers = nn.ModuleList(AdapterLayer("S_static", width, query_chunk) for _ in range(layers))
        self.call_count = 0
        self.encoder_call_count = 0

    def encode(self, prompts: Mapping[str, Tensor]) -> EncodedPrompts:
        self.encoder_call_count += 1
        points, vertices = prompts["points"], prompts["vertices"]
        arity = prompts["arity"].long()
        group = prompts["point_primitive"].long()
        point_mask = prompts["point_mask"].bool()
        primitive_mask = prompts["primitive_mask"].bool()
        if points.ndim != 3 or points.shape[1:] != (32, 3):
            raise ValueError("S32_static accepts exactly 32 point-token slots per condition")
        batch, primitive_count = arity.shape
        if (points.shape[0] != batch or vertices.shape != (batch, primitive_count, 3, 3)
                or group.shape != (batch, 32) or point_mask.shape != (batch, 32)
                or primitive_mask.shape != (batch, primitive_count)):
            raise ValueError("Malformed S32_static geometry/mask shapes")
        if primitive_count < 1:
            raise ValueError("S32_static requires declared primitive groups; empty conditions use false masks")
        if (((arity != 2) & (arity != 3)) & primitive_mask).any():
            raise ValueError("Valid primitive arity must be edge2 or face3")
        if (((group < 0) | (group >= primitive_count)) & point_mask).any():
            raise ValueError("Valid points require in-range primitive groups")
        safe_group = torch.where(point_mask, group, torch.zeros_like(group))
        point_mask = point_mask & primitive_mask.gather(1, safe_group)
        clean_points = torch.where(point_mask.unsqueeze(-1), points, torch.zeros_like(points))
        clean_arity = torch.where(primitive_mask, arity, torch.full_like(arity, 2))
        types = clean_arity.gather(1, safe_group)
        tokens = self.condition_encoder(point_features(clean_points, types))
        tokens = torch.where(point_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        # Exact integer counts avoid a feature reduction entirely. Each group's
        # point priors sum to one, independently of its 8/8/16 sample count.
        counts = torch.zeros((batch, primitive_count), dtype=torch.long, device=points.device)
        counts = counts.scatter_add(1, safe_group, point_mask.long())
        per_point_count = counts.gather(1, safe_group).clamp_min(1).to(points.dtype)
        log_weights = -per_point_count.log()
        log_weights = torch.where(point_mask, log_weights, torch.zeros_like(log_weights))
        clean_vertices = torch.where(primitive_mask[..., None, None], vertices, torch.zeros_like(vertices))
        # Compatibility-only geometry fields are never consumed by S_static.
        centers = vertices.new_zeros((batch, primitive_count, 3))
        return EncodedPrompts(tokens, point_mask, log_weights, clean_vertices,
                              clean_arity, centers, clean_points)

    def forward_layer(self, index, h, xt, t, encoded, mask=None, geometry_cache=None):
        if geometry_cache is not None:
            raise ValueError("S32_static has no geometry cache or closest-point path")
        self.call_count += 1
        return self.layers[index](h, xt, t, encoded, mask)

    def architecture_summary(self):
        return dict(arm=self.arm, point_tokens=32, condition_input_features=41,
            condition_widths=[41, 128, 128], condition_activation="SiLU",
            query_normalization="LayerNorm(768)", query_projection=[768, 128],
            key_projection=[128, 128], value_projection=[128, 128],
            heads=4, head_dim=32, output_projection=[128, 768],
            injection_after_complete_blocks=[3, 6, 9, 12], residual_scale=1.,
            point_prior="-log(valid point count of the point's primitive)",
            primitive_group_embedding=False, primitive_token_pooling=False,
            dynamic_value=False, relative_geometry=False, nearest_point=False,
            null_token=False, learned_gate=False, backbone_copy=False,
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            expected_parameters=950144,
            parameter_groups=dict(condition_encoder=sum(p.numel() for p in self.condition_encoder.parameters()),
                injection_layers=[sum(p.numel() for p in layer.parameters()) for layer in self.layers]))
