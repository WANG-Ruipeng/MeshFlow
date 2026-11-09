"""Independent initial-state pointer/Null head; no generator/Y lookup.

Frozen contract: unweighted multiclass CE over all valid bulk corners;
Null is class zero, predicted by a separate per-query MLP.  No threshold scan,
positive-class correction, or quality-feedback gradient is used.
"""
from __future__ import annotations

from meshflow_control import runtime as _stable
import torch
from torch import nn
from torch.nn import functional as F
from .shared_tip import PAIR_DIM, QUERY_DIM

VERSION = 'seam_abc_initial_tip_pointer_null_v1'
INITIALIZATION_SEED = 2026101703
HIDDEN = 64


class TipPointerHead(nn.Module):
    def __init__(self, seed=INITIALIZATION_SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.pair_net = nn.Sequential(nn.Linear(PAIR_DIM, HIDDEN, device='cpu', dtype=torch.float32),
                                          nn.SiLU(), nn.Linear(HIDDEN, 1, device='cpu', dtype=torch.float32))
            self.null_net = nn.Sequential(nn.Linear(QUERY_DIM, HIDDEN, device='cpu', dtype=torch.float32),
                                          nn.SiLU(), nn.Linear(HIDDEN, 1, device='cpu', dtype=torch.float32))
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        self.initialization_seed = int(seed)

    def forward(self, pair_features, null_features):
        if (pair_features.ndim != 3 or pair_features.shape[-1] != PAIR_DIM or
                null_features.shape != (pair_features.shape[0], QUERY_DIM)):
            raise ValueError('Pointer feature schema mismatch')
        if (pair_features.device != next(self.parameters()).device or
                null_features.device != pair_features.device or
                not bool(torch.isfinite(pair_features).all() and torch.isfinite(null_features).all())):
            raise ValueError('Pointer features must be finite and colocated')
        pair = self.pair_net(pair_features.detach().float()).squeeze(-1)
        null = self.null_net(null_features.detach().float())
        return torch.cat((null, pair), dim=1)


def pointer_ce_loss(logits, labels, *, denominator):
    if labels.ndim != 1 or labels.shape != logits.shape[:1] or labels.dtype != torch.long:
        raise ValueError('Pointer labels must be int64 one class per query')
    if denominator < len(labels) or not len(labels):
        raise ValueError('Pointer effective-batch corner denominator invalid')
    return F.cross_entropy(logits.float(), labels.detach(), reduction='sum') / int(denominator)
