"""C-only edge-context owner head; no source completion enters inference.

Feature 27 explicitly repeats the existing world-y midpoint feature so the
requested height channel is transparent, not described as new information.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
from meshflow_control.seam.slot_data import edge_features, inference_layout, L

HEAD_CONFIG = dict(input_dim=28, width=32, heads=4, layers=2, dropout=.1,
                   feature_schema='C_edge24_relative_bbox3_world_y1_v1')
THRESHOLD_LOGIT = 0.


def features(C_vertices, C_faces_original, total_N):
    """Unchanged 24 edge features + bbox-relative midpoint xyz + world y/L."""
    v = np.asarray(C_vertices, np.float32)
    edges, base = edge_features(v, C_faces_original, int(total_N))
    mid = (v[edges[:, 1]] + v[edges[:, 2]]) * np.float32(.5)
    center = (v.min(0) + v.max(0)) * np.float32(.5)
    denominator = np.maximum(np.ptp(v, axis=0), np.float32(1e-6 * L))
    relative = (mid - center) / denominator
    return np.concatenate((base, relative, mid[:, 1:2] / np.float32(L)), axis=1).astype(np.float32)


class ContextBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width), nn.Dropout(dropout))

    def forward(self, x, mask):
        batch, edges, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, edges, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        scores = (q.float() @ k.float().transpose(-1, -2)) / ((width // self.heads) ** .5)
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = self.attention_dropout(torch.softmax(scores, dim=-1).to(v.dtype))
        mixed = (attention @ v).transpose(1, 2).reshape(batch, edges, width)
        x = x + self.output_dropout(self.proj(mixed))
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(~mask[:, :, None], 0.)


class MultipartOccupancyHead(nn.Module):
    def __init__(self, input_dim=28, width=32, heads=4, layers=2, dropout=.1,
                 feature_schema='C_edge24_relative_bbox3_world_y1_v1'):
        super().__init__()
        config = dict(input_dim=input_dim, width=width, heads=heads, layers=layers,
                      dropout=dropout, feature_schema=feature_schema)
        if config != HEAD_CONFIG:
            raise ValueError('Unregistered multipart head configuration')
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU())
        self.blocks = nn.ModuleList([ContextBlock(width, heads, dropout) for _ in range(layers)])
        self.output = nn.Linear(width, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != HEAD_CONFIG['input_dim']:
            raise ValueError('Expected [E,28] or [B,E,28] head input')
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:2] or not mask.any(dim=1).all():
            raise ValueError('Invalid or empty edge mask')
        h = self.input(x)
        for block in self.blocks:
            h = block(h, mask)
        logits = self.output(h).squeeze(-1).float().masked_fill(~mask, 0.)
        return logits[0] if single else logits


def new_head(seed):
    # Parameter initialization must not consume the inherited generator RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return MultipartOccupancyHead(**HEAD_CONFIG)


@torch.no_grad()
def predict_layout(head, bundle):
    required = {'C_vertices', 'C_faces_original', 'C_faces_model', 'total_N'}
    if set(bundle) != required:
        raise ValueError('Inference accepts only the four registered C-only fields')
    if head.training:
        raise ValueError('Occupancy inference requires eval mode (dropout disabled)')
    feat = features(bundle['C_vertices'], bundle['C_faces_original'], bundle['total_N'])
    tensor = torch.from_numpy(feat).to(next(head.parameters()).device)
    logits = head(tensor).float().cpu().numpy()
    return inference_layout(**bundle, occupied=logits >= THRESHOLD_LOGIT), logits


def predict(head, bundle):
    layout, logits = predict_layout(head, bundle)
    return dict(active=layout['occupied'], logits=logits, layout=layout)


def load_head(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['head_config'] != HEAD_CONFIG or payload['threshold_logit'] != THRESHOLD_LOGIT:
        raise ValueError('Head schema or threshold mismatch')
    head = MultipartOccupancyHead(**payload['head_config'])
    head.load_state_dict(payload['head_state'], strict=True)
    return head.to(device).eval().requires_grad_(False), payload
