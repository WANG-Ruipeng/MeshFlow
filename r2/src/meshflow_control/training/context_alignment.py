"""Training-only single-query alignment of actual fourth-block attention messages.

No generator arithmetic is replaced. The hook returns None and its tensor keeps
autograd until the current microbatch finishes. No teacher/model runs on import.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .context_contract import (NAMESPACE, HEAD_SEED, HEAD_PARAMETERS, ARMS, DOMAINS, HEAD_CONFIG, validate_config)

def select_messages(messages, known_mask, valid_mask, domain):
    if messages.ndim != 3 or messages.shape[-1] != 768:
        raise ValueError("Capture must be face messages [B,N,768]")
    if domain not in ("free", "known"):
        raise ValueError("Select free or known faces")
    if (known_mask.dtype != torch.bool or valid_mask.dtype != torch.bool
            or known_mask.shape != messages.shape[:2] or valid_mask.shape != known_mask.shape
            or known_mask.device != messages.device or valid_mask.device != messages.device
            or bool((known_mask & ~valid_mask).any())):
        raise ValueError("Invalid separate masks")
    selected = valid_mask & (known_mask if domain == "known" else ~known_mask)
    if not bool(selected.any(1).all()):
        raise ValueError("Empty alignment domain")
    return [messages[i, selected[i]] for i in range(len(messages))], selected


class AlignmentHead(nn.Module):
    """One FP32 query; source token order and padding are not model inputs."""
    def __init__(self, seed=HEAD_SEED):
        super().__init__()
        if seed != HEAD_SEED:
            raise ValueError("Initialization seed is frozen")
        # CPU-only isolated RNG, including nn.Linear constructors.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.q = nn.Parameter(torch.empty(1, 768, dtype=torch.float32))
            self.source_norm = nn.LayerNorm(768, eps=1e-5, elementwise_affine=False)
            self.query_norm = nn.LayerNorm(768, eps=1e-5, elementwise_affine=False)
            self.output_norm = nn.LayerNorm(768, eps=1e-5, elementwise_affine=False)
            self.W_Q = nn.Linear(768, 768, bias=False, dtype=torch.float32)
            self.W_K = nn.Linear(768, 768, bias=False, dtype=torch.float32)
            self.W_V = nn.Linear(768, 768, bias=False, dtype=torch.float32)
            self.W_O = nn.Linear(768, 768, bias=False, dtype=torch.float32)
            self.linear_1 = nn.Linear(768, 1024, bias=True, dtype=torch.float32)
            self.linear_2 = nn.Linear(1024, 768, bias=True, dtype=torch.float32)
            nn.init.normal_(self.q, mean=0., std=.02)
            for layer in (self.W_Q, self.W_K, self.W_V, self.W_O, self.linear_1, self.linear_2):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.eval()
        if sum(p.numel() for p in self.parameters()) != HEAD_PARAMETERS:
            raise AssertionError("Alignment head capacity differs")

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != 768 or len(tokens) == 0:
            raise ValueError("Head accepts only nonempty selected face messages [L,768]")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Alignment head must remain FP32")
        with torch.autocast(tokens.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            s = self.source_norm(tokens.float())  # Conversion keeps the generator graph.
            q0 = self.query_norm(self.q)
            q = self.W_Q(q0).reshape(1, 1, 24, 32).transpose(1, 2)
            k = self.W_K(s).reshape(1, len(s), 24, 32).transpose(1, 2)
            v = self.W_V(s).reshape(1, len(s), 24, 32).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
            z = self.q + self.W_O(attended.transpose(1, 2).reshape(1, 768))
            return self.linear_2(F.silu(self.linear_1(self.output_norm(z)))).squeeze(0)


def alignment_loss(prediction, target):
    """Raw sample loss; target is always detached. Batch coefficient is .1/4."""
    if prediction.shape != (768,) or target.shape != (768,):
        raise ValueError("Prediction/teacher must be global 768-vectors")
    with torch.autocast(prediction.device.type, enabled=False):
        p, t = prediction.float(), target.detach().to(device=prediction.device, dtype=torch.float32)
        pn, tn = p.norm(), t.norm()
        if not bool(torch.isfinite(p).all() and torch.isfinite(t).all()) or float(pn) <= 1e-12 or float(tn) <= 1e-12:
            raise FloatingPointError("Nonfinite or zero global feature")
        cosine = (F.normalize(p, dim=0, eps=1e-12) * F.normalize(t, dim=0, eps=1e-12)).sum()
        loss = 1. - cosine
        return loss, dict(prediction_norm=float(pn.detach()), teacher_norm=float(tn.detach()),
                          cosine=float(cosine.detach()))


class AttentionCapture(AbstractContextManager):
    """Capture the sole actual layer-4 attention output; no replacement/detach."""
    def __init__(self, model):
        if getattr(model, "readout_mode", None) != "none":
            raise ValueError("This experiment requires no inference readout")
        self.module = model.backbone.layers[3].attn
        self.tensor = None
        self.calls = 0
        self.handle = None

    def _hook(self, module, args, output):
        if self.tensor is not None or not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] != 768:
            raise RuntimeError("Expected exactly one face-level attention capture per microbatch")
        self.tensor = output
        self.calls += 1
        return None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Capture already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def get(self):
        if self.tensor is None or self.calls != 1:
            raise RuntimeError("Fourth attention message was not captured exactly once")
        return self.tensor

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.tensor = None
        return False
