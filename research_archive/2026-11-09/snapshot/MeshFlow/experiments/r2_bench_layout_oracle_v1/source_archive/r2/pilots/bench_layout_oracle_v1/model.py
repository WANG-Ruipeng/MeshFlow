"""A coarse, privileged layout input for the unchanged Native face generator.

The only new operation is one FP32 cross-attention residual after Native's
coordinate/role/Geo embedding.  Layout tokens are NOT generated triangles,
do not consume face slots, and contain no target face/corner correspondences.
An all-missing layout is explicitly distinct from an observed absent part.
No inherited model or module-global function is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn

from r2.pilots.bench_natural_n_recipe_v1 import runtime as native
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, assert_native_runtime, native_block_forward,
    native_final_forward, native_execution,
)

VERSION = "bench_coarse_source_layout_embedding_readout_v1"
LAYOUT_DIM = 8
LAYOUT_SLOTS = 2
READOUT_DIM = 64
READOUT_HEADS = 4
LAYOUT_SEED = 20261030
L = math.sqrt(3.0) * 2.0 / 0.3762
TOKEN_FIELDS = (
    "center_x_relative_to_C_area_centroid_over_L",
    "center_y_relative_to_C_area_centroid_over_L",
    "center_z_relative_to_C_area_centroid_over_L",
    "extent_x_over_L", "extent_y_over_L", "extent_z_over_L",
    "is_BACK_REGION", "is_SUPPORT_REGION",
)


def c_area_centroid(x, known_mask):
    """Surface-area centroid of C in the unmodified world/model coordinates."""
    faces = x.float().reshape(x.shape[0], x.shape[1], 3, 3)
    a, b, c = faces.unbind(dim=2)
    area2 = torch.linalg.vector_norm(torch.cross(b - a, c - a, dim=-1), dim=-1)
    weight = area2 * known_mask.to(area2.dtype)
    denom = weight.sum(dim=1)
    if bool((denom <= 0).any()) or not bool(torch.isfinite(denom).all()):
        raise ValueError("C must have positive finite total area")
    return ((a + b + c) / 3.0 * weight[..., None]).sum(dim=1) / denom[:, None]


class LayoutReadout(nn.Module):
    """Small geometry-aware cross-attention; exactly zero-initialized exit."""

    def __init__(self, hidden_dim=768, *, seed=LAYOUT_SEED):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        # CPU-only isolated initialization also works inside an outer meta scope.
        # It never advances the inherited training stream's CPU RNG state.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            # torch.manual_seed also touches CUDA generators if already active;
            # seed only the isolated CPU default generator here.
            torch.default_generator.manual_seed(self.seed)
            self.query_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            self.query_hidden = nn.Linear(self.hidden_dim, READOUT_DIM, bias=False)
            self.query_position = nn.Linear(3, READOUT_DIM, bias=False)
            self.layout_embed = nn.Sequential(
                nn.Linear(LAYOUT_DIM, READOUT_DIM), nn.SiLU(),
                nn.Linear(READOUT_DIM, READOUT_DIM),
            )
            self.key = nn.Linear(READOUT_DIM, READOUT_DIM, bias=False)
            self.value = nn.Linear(READOUT_DIM, READOUT_DIM, bias=False)
            self.output = nn.Linear(READOUT_DIM, self.hidden_dim, bias=False)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.zeros_(self.output.weight)

    def metadata(self):
        return dict(version=VERSION, token_fields=list(TOKEN_FIELDS),
            slot_order=["BACK_REGION", "SUPPORT_REGION"], slots=LAYOUT_SLOTS,
            token_dimension=LAYOUT_DIM, hidden_dimension=self.hidden_dim,
            readout_dimension=READOUT_DIM, heads=READOUT_HEADS,
            output_initialization="zeros", initialization_seed=self.seed,
            parameters=sum(p.numel() for p in self.parameters()),
            injection="once after coordinate/role/Geo embedding; valid free only",
            arithmetic="FP32 branch, add and cast to inherited hidden dtype",
            center="C triangle area-weighted centroid", scale_L=L,
            layout_semantics="privileged coarse source region bbox, not C-only prediction",
            missing_semantics="mask false is unknown/dropped, not observed absence",
            generated_face_slots_added=0)

    def forward(self, hidden, x, known_mask, valid_mask, tokens, layout_valid):
        batch, faces = x.shape[:2]
        if hidden.shape != (batch, 3 * faces, self.hidden_dim):
            raise ValueError("Unexpected inherited corner hidden shape")
        if tokens.shape != (batch, LAYOUT_SLOTS, LAYOUT_DIM) or tokens.dtype != torch.float32:
            raise ValueError("layout_tokens must be FP32 [B,2,8]")
        if layout_valid.shape != (batch, LAYOUT_SLOTS) or layout_valid.dtype != torch.bool:
            raise ValueError("layout_valid must be bool [B,2]")
        if tokens.device != x.device or layout_valid.device != x.device:
            raise ValueError("Layout and coordinates must be colocated")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Layout tokens, including masked placeholders, must be finite")
        if bool((tokens[..., 3:6][layout_valid] < 0).any()):
            raise ValueError("Observed layout extents must be nonnegative")
        expected_roles = torch.eye(2, device=tokens.device, dtype=tokens.dtype)[None].expand(batch, -1, -1)
        if not torch.equal(tokens[..., 6:8][layout_valid], expected_roles[layout_valid]):
            raise ValueError("Observed token type must match fixed BACK/SUPPORT slot")
        with torch.autocast(device_type=x.device.type, enabled=False):
            center = c_area_centroid(x, known_mask)
            position = (x.float().reshape(batch, faces * 3, 3) - center[:, None, :]) / L
            query = self.query_hidden(self.query_norm(hidden.float())) + self.query_position(position)
            # Invalid rows cannot leak values through the embedding biases or keys.
            clean_tokens = torch.where(layout_valid[..., None], tokens, torch.zeros_like(tokens))
            encoded = self.layout_embed(clean_tokens)
            width = READOUT_DIM // READOUT_HEADS
            q = query.reshape(batch, 3 * faces, READOUT_HEADS, width).transpose(1, 2)
            k = self.key(encoded).reshape(batch, LAYOUT_SLOTS, READOUT_HEADS, width).transpose(1, 2)
            v = self.value(encoded).reshape(batch, LAYOUT_SLOTS, READOUT_HEADS, width).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(width)
            any_layout = layout_valid.any(dim=1)
            scores = scores.masked_fill(~layout_valid[:, None, None, :], -torch.inf)
            # Avoid NaN in all-missing softmax; its output is multiplied by zero.
            scores = torch.where(any_layout[:, None, None, None], scores, torch.zeros_like(scores))
            attention = torch.softmax(scores, dim=-1) * layout_valid[:, None, None, :].to(scores.dtype)
            context = (attention @ v).transpose(1, 2).reshape(batch, 3 * faces, READOUT_DIM)
            delta = self.output(context)
            free = (valid_mask & ~known_mask).repeat_interleave(3, dim=1)
            active = free & any_layout[:, None]
            delta = torch.where(active[..., None], delta, torch.zeros_like(delta))
            # Keeping the old tensor explicitly for known/padding/missing rows
            # guarantees no quantization change for these rows.
            result = torch.where(active[..., None], (hidden.float() + delta).to(hidden.dtype), hidden)
        return result, delta


class LayoutNative(NativeModel):
    """Native with an instance-local, scoped optional coarse layout input."""

    def configure_layout(self, tokens, layout_valid):
        """Replace this worker's current microbatch P without touching weights."""
        if getattr(self, "_layout_context_active", False):
            raise RuntimeError("Cannot replace layout inside a scoped rollout")
        if tokens is None or layout_valid is None:
            raise ValueError("Use clear_layout for no input; NULL uses explicit false validity")
        self._layout_payload = (tokens, layout_valid)
        return self

    def clear_layout(self):
        if getattr(self, "_layout_context_active", False):
            raise RuntimeError("Cannot clear layout inside a scoped rollout")
        self._layout_payload = None
        return self

    @contextmanager
    def layout_context(self, tokens, layout_valid):
        """Scope a fixed P to a rollout; no mutable parameter/state-dict changes.

        Use a distinct model instance per worker. Nested contexts are refused so
        an accidental stale/scope-overlapping input cannot silently be reused.
        """
        if getattr(self, "_layout_payload", None) is not None:
            raise RuntimeError("A layout context is already active on this model")
        if tokens is None or layout_valid is None:
            raise ValueError("Pass explicit finite tokens and validity, including for NULL")
        self._layout_payload = (tokens, layout_valid)
        self._layout_context_active = True
        try:
            yield self
        finally:
            self._layout_payload = None
            self._layout_context_active = False

    def layout_metadata(self):
        return self.layout_branch.metadata()

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None,
                layout_tokens=None, layout_valid=None):
        payload = getattr(self, "_layout_payload", None)
        if layout_tokens is not None or layout_valid is not None:
            if layout_tokens is None or layout_valid is None or payload is not None:
                raise ValueError("Provide one complete layout input, explicit or scoped")
            payload = (layout_tokens, layout_valid)
        if payload is None:
            # Exact inherited function when no layout is requested.
            return super().forward(x, t, y, valid_mask=valid_mask,
                                   known_mask=known_mask, observer=observer)
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected free t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Free time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated separate bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 799)).any()):
            raise ValueError("y must count real faces only, N128..799")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        if self.readout_mode != "none" or len(self.readouts):
            raise ValueError("Layout experiment requires original no-readout Native")
        hidden, role = self._embed(x, known_mask, valid_mask)
        original_hidden = hidden
        hidden, delta = self.layout_branch(hidden, x, known_mask, valid_mask, *payload)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="layout_embedding", original_hidden=original_hidden,
                delta=delta, hidden=hidden, layout_valid=payload[1]))
            observer(dict(site="embedding", hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)


def attach_layout(model, *, seed=LAYOUT_SEED):
    """Add the fresh branch to an owned base instance after base optimizer load."""
    if hasattr(model, "layout_branch"):
        raise ValueError("Layout branch already exists")
    if type(model) is not NativeModel:
        raise TypeError("Expected an owned, unmodified natural-N Native instance")
    branch = LayoutReadout(int(model.role_embedding.shape[-1]), seed=seed)
    device = model.role_embedding.device
    if device.type != "meta":
        branch.to(device=device, dtype=torch.float32)
    branch.requires_grad_(model.experiment_trainable)
    # Native's deterministic contract requires eval even while gradients train.
    branch.eval()
    model.__class__ = LayoutNative
    model.layout_branch = branch
    model._layout_payload = None
    model._layout_context_active = False
    return model


def build_model(model_config, readout_mode="none", readout_seed=20261007,
                trainable=True, *, layout_seed=LAYOUT_SEED):
    model = native.build_model(model_config, readout_mode=readout_mode,
        readout_seed=readout_seed, trainable=trainable)
    return attach_layout(model, seed=layout_seed)


def split_base_state(state):
    """Split full checkpoint keys for strict inherited/new-module loading."""
    prefix = "layout_branch."
    base = {k: v for k, v in state.items() if not k.startswith(prefix)}
    layout = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not base or not layout:
        raise ValueError("Expected a full layout candidate state, not a base-only checkpoint")
    return base, layout


def layout_parameters(model):
    return list(model.layout_branch.parameters())


def base_parameters(model):
    return [p for name, p in model.named_parameters() if not name.startswith("layout_branch.")]
