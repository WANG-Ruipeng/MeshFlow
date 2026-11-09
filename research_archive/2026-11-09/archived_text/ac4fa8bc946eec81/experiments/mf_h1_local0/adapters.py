"""The exact S/Q/G/P control branches from MF-H1-LOCAL0, with no backbone.

Prompt dictionaries contain only observed geometry: vertices [B,K,3,3], arity
[B,K] (2 or 3), primitive_mask [B,K], points [B,M,3], point_primitive [B,M], and
point_mask [B,M]. Boolean masks are True for valid slots. Point grouping is not
a semantic identifier. Q/G/P state dictionaries have identical keys and shapes;
construct once and copy the state for exactly matched initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .geometry import closest_points_primitives, primitive_centroids, relation_features

ARMS = ("S_static", "Q_center", "G_closest", "P_point")


@dataclass
class EncodedPrompts:
    tokens: Tensor
    token_mask: Tensor
    log_weights: Tensor
    vertices: Tensor
    arity: Tensor
    centers: Tensor
    points: Tensor


def point_features(points: Tensor, arity: Tensor) -> Tensor:
    bands = torch.arange(6, dtype=points.dtype, device=points.device)
    phase = points.unsqueeze(-1) * (torch.pi * 2.0**bands)
    types = torch.stack((arity == 2, arity == 3), -1).to(points.dtype)
    return torch.cat((points, phase.sin().flatten(-2), phase.cos().flatten(-2), types), -1)


def time_features(t: Tensor, dtype: torch.dtype) -> Tensor:
    """Fixed 32-D sinusoidal encoding, frequencies exp(-log(10000)*k/16)."""
    frequency = torch.exp(-math.log(10000) * torch.arange(16, device=t.device, dtype=torch.float32) / 16)
    phase = 1000 * t.float().reshape(-1, 1) * frequency
    return torch.cat((phase.cos(), phase.sin()), -1).to(dtype)


class AdapterLayer(nn.Module):
    def __init__(self, arm: str, width: int = 768, query_chunk: int = 256):
        super().__init__()
        self.arm = arm
        self.query_chunk = query_chunk
        self.query_norm = nn.LayerNorm(width)
        self.query_projection = nn.Linear(width, 128)
        self.key_projection = nn.Linear(128, 128)
        self.value_projection = nn.Linear(128, 128)
        if arm != "S_static":
            self.dynamic_mlp = nn.Sequential(nn.Linear(316, 128), nn.SiLU(), nn.Linear(128, 128))
        self.output_projection = nn.Linear(128, width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def aggregate(self, h: Tensor, xt: Tensor, t: Tensor, encoded: EncodedPrompts,
                  mask: Tensor | None = None) -> Tensor:
        """Return pre-output-projection messages; useful for nonvacuous tests."""
        batch, queries, _ = h.shape
        if encoded.tokens.shape[1] == 0:
            return h.new_zeros((batch, queries, 128))
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h))
            xt = torch.where(mask.unsqueeze(-1), xt, torch.zeros_like(xt))
        q = self.query_projection(self.query_norm(h))
        key = self.key_projection(encoded.tokens).reshape(batch, -1, 4, 32)
        static_value = self.value_projection(encoded.tokens)
        tau = time_features(t, q.dtype)
        valid = encoded.token_mask[:, None, None, :]
        has_condition = encoded.token_mask.any(-1)[:, None, None, None]
        chunks = []
        for start in range(0, queries, self.query_chunk):
            stop = min(start + self.query_chunk, queries)
            query = q[:, start:stop]
            logits = torch.einsum("bqhd,bahd->bhqa", query.reshape(batch, -1, 4, 32), key) / math.sqrt(32)
            logits = logits + encoded.log_weights[:, None, None, :]
            logits = logits.masked_fill(~valid, -torch.inf)
            # All-masked rows enter softmax as zeros and are then explicitly zero.
            logits = torch.where(has_condition, logits, torch.zeros_like(logits))
            alpha = logits.softmax(-1).masked_fill(~valid, 0)
            if self.arm == "S_static":
                message = torch.einsum("bhqa,bahd->bqhd", alpha,
                                       static_value.reshape(batch, -1, 4, 32))
            else:
                current = xt[:, start:stop]
                if self.arm == "Q_center":
                    delta = current.unsqueeze(-2) - encoded.centers.unsqueeze(1)
                elif self.arm == "G_closest":
                    closest = closest_points_primitives(current, encoded.vertices, encoded.arity)
                    delta = current.unsqueeze(-2) - closest
                else:
                    delta = current.unsqueeze(-2) - encoded.points.unsqueeze(1)
                count = encoded.tokens.shape[1]
                rel = relation_features(delta).to(query.dtype)
                dynamic_input = torch.cat((query.unsqueeze(2).expand(-1, -1, count, -1),
                                           encoded.tokens.unsqueeze(1).expand(-1, stop-start, -1, -1),
                                           rel, tau[:, None, None, :].expand(-1, stop-start, count, -1)), -1)
                value = self.dynamic_mlp(dynamic_input) + static_value.unsqueeze(1)
                message = torch.einsum("bhqa,bqahd->bqhd", alpha,
                                       value.reshape(batch, stop-start, count, 4, 32))
            chunks.append(message.flatten(-2))
        result = torch.cat(chunks, 1) if chunks else q.new_zeros((batch, 0, 128))
        if mask is not None:
            result = torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result))
        return result

    def forward(self, h: Tensor, xt: Tensor, t: Tensor, encoded: EncodedPrompts,
                mask: Tensor | None = None) -> Tensor:
        message = self.aggregate(h, xt, t, encoded, mask)
        residual = self.output_projection(message)
        # A trained output bias must not create control for an empty prompt.
        valid = encoded.token_mask.any(-1)[:, None, None]
        if mask is not None:
            valid = valid & mask.unsqueeze(-1)
        return torch.where(valid, residual, torch.zeros_like(residual))


class ControlAdapter(nn.Module):
    def __init__(self, arm: str, width: int = 768, layers: int = 4, query_chunk: int = 256):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.arm = arm
        self.condition_encoder = nn.Sequential(nn.Linear(41, 128), nn.SiLU(), nn.Linear(128, 128))
        self.layers = nn.ModuleList(AdapterLayer(arm, width, query_chunk) for _ in range(layers))
        self.call_count = 0
        self.encoder_call_count = 0

    def encode(self, prompts: Mapping[str, Tensor]) -> EncodedPrompts:
        """Encode static conditions once per forward/train graph or sample path."""
        self.encoder_call_count += 1
        vertices = prompts["vertices"]
        arity = prompts["arity"].long()
        primitive_mask = prompts["primitive_mask"].bool()
        points = prompts["points"]
        group = prompts["point_primitive"].long()
        point_mask = prompts["point_mask"].bool()
        batch, primitive_count = arity.shape
        if vertices.shape != (batch, primitive_count, 3, 3):
            raise ValueError("vertices must have shape [B,K,3,3]")
        if ((arity != 2) & (arity != 3) & primitive_mask).any():
            raise ValueError("Valid primitive arity must be 2 or 3")
        if primitive_count == 0:
            empty_tokens = points.new_zeros((batch, 0, 128))
            empty_mask = torch.zeros((batch, 0), dtype=torch.bool, device=points.device)
            return EncodedPrompts(empty_tokens, empty_mask, points.new_zeros((batch, 0)),
                                  vertices, arity, vertices.new_zeros((batch, 0, 3)), points[:, :0])
        if (((group < 0) | (group >= primitive_count)) & point_mask).any():
            raise ValueError("Valid points require an in-range primitive group")
        safe_group = torch.where(point_mask, group, torch.zeros_like(group))
        point_mask = point_mask & primitive_mask.gather(1, safe_group)
        points = torch.where(point_mask.unsqueeze(-1), points, torch.zeros_like(points))
        vertices = torch.where(primitive_mask[..., None, None], vertices, torch.zeros_like(vertices))
        arity = torch.where(primitive_mask, arity, torch.full_like(arity, 2))
        types = arity.gather(1, safe_group)
        features = self.condition_encoder(point_features(points, types))
        features = torch.where(point_mask.unsqueeze(-1), features, torch.zeros_like(features))
        sums = features.new_zeros((batch, primitive_count, 128)).scatter_add(
            1, safe_group.unsqueeze(-1).expand(-1, -1, 128), features)
        counts = features.new_zeros((batch, primitive_count)).scatter_add(1, safe_group, point_mask.to(features.dtype))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        primitive_mask = primitive_mask & (counts > 0)
        centers = primitive_centroids(vertices, arity)
        if self.arm == "P_point":
            tokens = features + means.gather(1, safe_group.unsqueeze(-1).expand(-1, -1, 128))
            token_mask = point_mask
            log_weights = -counts.gather(1, safe_group).clamp_min(1).log()
        else:
            tokens = means
            token_mask = primitive_mask
            log_weights = torch.zeros_like(counts)
        tokens = torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return EncodedPrompts(tokens, token_mask, log_weights, vertices, arity, centers, points)

    def forward_layer(self, index: int, h: Tensor, xt: Tensor, t: Tensor,
                      encoded: EncodedPrompts, mask: Tensor | None = None) -> Tensor:
        self.call_count += 1
        return self.layers[index](h, xt, t, encoded, mask)
