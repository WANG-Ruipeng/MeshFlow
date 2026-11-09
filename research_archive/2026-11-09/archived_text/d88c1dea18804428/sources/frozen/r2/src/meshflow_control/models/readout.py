"""Direct per-corner context readout, shared by MIXED and CLEAN.

This is a local experimental addition, not an upstream MeshFlow method.
Production arithmetic is FP32 throughout the new branch. The reference method
is CPU-only FP64 and exists only for compact mathematical tests.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

READOUT_LAYERS = (3, 6, 9, 12)
READOUT_DIM = 192
READOUT_HEADS = 4
READOUT_PARAMETERS = 2359296


def corner_indices(face_mask):
    """Ascending original slot order; retain all three corner instances."""
    return torch.nonzero(face_mask[:, None].expand(-1, 3).reshape(-1), as_tuple=False).flatten()


def gather_known_corners(hidden, known_mask, valid_mask):
    """Fresh differentiable copies; neither detach nor cross-forward caching."""
    return tuple(hidden[b].index_select(0, corner_indices(known_mask[b] & valid_mask[b]))
                 for b in range(hidden.shape[0]))


class CornerReadout(nn.Module):
    def __init__(self, hidden_dim=768, readout_dim=READOUT_DIM, heads=READOUT_HEADS):
        super().__init__()
        if hidden_dim < 1 or readout_dim < 1 or heads < 1 or readout_dim % heads:
            raise ValueError("Invalid readout dimensions")
        self.hidden_dim, self.readout_dim, self.heads = hidden_dim, readout_dim, heads
        self.head_dim = readout_dim // heads
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-5)
        opts = dict(bias=False, device="cpu", dtype=torch.float32)
        self.W_Q = nn.Linear(hidden_dim, readout_dim, **opts)
        self.W_K = nn.Linear(hidden_dim, readout_dim, **opts)
        self.W_V = nn.Linear(hidden_dim, readout_dim, **opts)
        self.W_O = nn.Linear(readout_dim, hidden_dim, **opts)
        for module in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.W_O.weight)

    def _compute(self, query, memory, dtype, observer=None):
        if query.ndim != 2 or memory.ndim != 2 or query.shape[1] != self.hidden_dim or memory.shape[1] != self.hidden_dim:
            raise ValueError("Readout expects query[Q,D] and memory[K,D]")
        if query.device != memory.device or any(p.device != query.device or p.dtype != dtype for p in self.parameters()):
            raise ValueError("Readout parameters, inputs, and declared precision differ")
        if query.shape[0] == 0 or memory.shape[0] == 0:
            return query.to(dtype) * 0
        with torch.autocast(device_type=query.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            q0, m0 = self.norm(query.to(dtype)), self.norm(memory.to(dtype))
            q = self.W_Q(q0).reshape(-1, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            k = self.W_K(m0).reshape(-1, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            v = self.W_V(m0).reshape(-1, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            # Compact sequences have no padding mask and never include an empty
            # source. In SDPA a bool mask, if present, means True == allowed.
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                dropout_p=0.0, is_causal=False)
            value = value[0].transpose(0, 1).reshape(query.shape[0], self.readout_dim)
            residual = self.W_O(value)
            if observer is not None:
                # Limited summaries only; no attention matrix is retained.
                with torch.no_grad():
                    logits = q.detach() @ k.detach().transpose(-2, -1) / math.sqrt(self.head_dim)
                    logp = logits.log_softmax(dim=-1)
                    entropy = -(logp.exp() * logp).sum(-1).mean()
                    rms = lambda a: a.detach().double().square().mean().sqrt()
                    hidden_rms, residual_rms = rms(query), rms(residual)
                    observer(dict(site="readout", query_corners=query.shape[0],
                        source_corners=memory.shape[0], residual_rms=residual_rms,
                        hidden_rms=hidden_rms,
                        residual_to_hidden_rms=residual_rms / hidden_rms.clamp_min(1e-30),
                        attention_entropy=entropy.detach(),
                        branch_dtype=str(residual.dtype), hidden_dtype=str(query.dtype)))
        return residual

    def forward(self, query, memory, *, observer=None):
        return self._compute(query, memory, torch.float32, observer)

    def forward_reference(self, query, memory):
        if query.device.type != "cpu" or memory.device.type != "cpu":
            raise ValueError("FP64 reference is restricted to CPU synthetic tests")
        if query.dtype != torch.float64 or memory.dtype != torch.float64:
            raise ValueError("Reference inputs must be FP64")
        return self._compute(query, memory, torch.float64)


def make_readouts(seed, *, layers=READOUT_LAYERS):
    if type(seed) is not int or not 0 <= seed < 2**63 or tuple(layers) != READOUT_LAYERS:
        raise ValueError("Registered seed/layers differ")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        modules = nn.ModuleDict({str(layer): CornerReadout() for layer in layers})
    if sum(p.numel() for p in modules.parameters()) != READOUT_PARAMETERS:
        raise RuntimeError("Registered readout capacity differs")
    return modules


def apply_readout(hidden, known_mask, valid_mask, module, *, memory=None,
                  observer=None, layer=None, source_mode=None, reference=False,
                  addition_mode="on", capture=None):
    """Only valid free corners receive residuals; all other bytes are copied.

    index_select preserves the original instance order. index_copy uses unique
    destination indices, so no reduction or arbitrary coordinate sorting occurs.
    Existing H0 and stored CLEAN memory are never modified in place.
    """
    if addition_mode not in ("on", "off", "zero"):
        raise ValueError("Unknown readout addition mode")
    if capture is not None and not callable(capture):
        raise TypeError("capture must be callable or None")
    if hidden.ndim != 3 or hidden.shape[1] != 3 * known_mask.shape[1] or known_mask.shape != valid_mask.shape:
        raise ValueError("Hidden and face mask dimensions differ")
    if known_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
        raise ValueError("Separate boolean masks required")
    if known_mask.device != hidden.device or valid_mask.device != hidden.device:
        raise ValueError("Masks must be colocated")
    if addition_mode == "off":
        # The original backbone/Geo/role/CLEAN-H0 path still executes.
        return hidden
    if memory is None:
        memory = gather_known_corners(hidden, known_mask, valid_mask)
    if len(memory) != hidden.shape[0]:
        raise ValueError("One memory tensor is required per sample")
    rows = []
    for b in range(hidden.shape[0]):
        indices = corner_indices(valid_mask[b] & ~known_mask[b])
        if indices.numel() == 0 or memory[b].shape[0] == 0:
            rows.append(hidden[b])
            continue
        query = hidden[b].index_select(0, indices)
        observe = None
        if observer is not None:
            def observe(record, b=b):
                observer(dict(record, batch_index=b, layer=layer, source_mode=source_mode))
        if reference:
            residual = module.forward_reference(query, memory[b])
        else:
            residual = module(query, memory[b], observer=observe)
        cast = residual.to(dtype=hidden.dtype)
        applied = torch.zeros_like(cast) if addition_mode == "zero" else cast
        updated = query + applied
        rows.append(hidden[b].index_copy(0, indices, updated))
        if capture is not None:
            # Actual free-only operands, after the output row has been copied.
            # Retain real r/cast under ZERO and separately expose applied zeros.
            capture(dict(site="readout_addition", batch_index=b, layer=layer,
                source_mode=source_mode, addition_mode=addition_mode,
                free_corner_indices=indices.detach(), before=query.detach(),
                residual=residual.detach(), cast=cast.detach(),
                applied=applied.detach(), after=updated.detach()))
    return torch.stack(rows)
