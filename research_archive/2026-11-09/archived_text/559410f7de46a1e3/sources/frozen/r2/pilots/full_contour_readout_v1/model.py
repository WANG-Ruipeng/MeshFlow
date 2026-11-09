"""One independent FULL-contour readout after complete Native block 6.

Old parameter objects, names and traversal order are retained.  No hook, loss,
projection, coordinate change, source-Y information or hidden cache is added.
"""
from __future__ import annotations

import math
from meshflow_control import runtime as _stable  # CUBLAS policy before CUDA.
import torch
from torch import nn
from meshflow_control.models.native import (
    NativeModel, assert_native_runtime, build_model, native_block_forward,
    native_final_forward,
)
from meshflow_control.models.readout import corner_indices

from .relations import (
    FEATURE_NAMES, LENGTH_SCALE, RELATION_DIM, TIE_EPS_FACTOR,
    relation_features,
)

READOUT_LAYER = 6
READOUT_DIM = 128
READOUT_HEADS = 4
EXTRA_PARAMETERS = 412032
INITIALIZATION_SEED = 20261014


class ContourReadout(nn.Module):
    def __init__(self, hidden_dim=768, readout_dim=READOUT_DIM, heads=READOUT_HEADS):
        super().__init__()
        if hidden_dim < 1 or readout_dim < 1 or heads < 1 or readout_dim % heads:
            raise ValueError("Invalid readout dimensions")
        self.hidden_dim, self.readout_dim, self.heads = hidden_dim, readout_dim, heads
        self.head_dim = readout_dim // heads
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-5)
        linear = dict(bias=False, device="cpu", dtype=torch.float32)
        self.W_Q = nn.Linear(hidden_dim, readout_dim, **linear)
        self.W_K = nn.Linear(hidden_dim, readout_dim, **linear)
        self.W_V = nn.Linear(hidden_dim, readout_dim, **linear)
        self.phi = nn.Sequential(
            nn.Linear(RELATION_DIM, readout_dim, bias=True, device="cpu", dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(readout_dim, readout_dim, bias=True, device="cpu", dtype=torch.float32),
        )
        self.W_O = nn.Linear(readout_dim, hidden_dim, **linear)
        for module in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.W_O.weight)

    def forward(self, query, memory, points, triangles, boundary_edge_mask, time, *, observer=None):
        if query.ndim != 2 or memory.ndim != 2 or query.shape[1] != self.hidden_dim or memory.shape[1] != self.hidden_dim:
            raise ValueError("Expected free query[Q,D] and C face memory[K,D]")
        if query.shape[0] != points.shape[0] or memory.shape[0] != triangles.shape[0]:
            raise ValueError("Hidden and geometry identities differ")
        if memory.shape[0] == 0:
            raise ValueError("Contour readout requires at least one valid C face")
        if memory.device != query.device or any(p.device != query.device or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Contour branch parameters must be colocated FP32")
        with torch.autocast(device_type=query.device.type, enabled=False):
            q0, s0 = self.norm(query.float()), self.norm(memory.float())
            q = self.W_Q(q0).reshape(-1, self.heads, self.head_dim).transpose(0, 1)
            k = self.W_K(s0).reshape(-1, self.heads, self.head_dim).transpose(0, 1)
            v = self.W_V(s0).reshape(-1, self.heads, self.head_dim).transpose(0, 1)
            weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
            descriptors = relation_features(points, triangles, boundary_edge_mask, time)
            pair_value = self.phi(descriptors).reshape(query.shape[0], memory.shape[0], self.heads, self.head_dim).permute(2, 0, 1, 3)
            # Geometry remains query-dependent through the sum over C.  There
            # is no free-dimension average, distance logit bias, or top-k mask.
            message = weights @ v + (weights[..., None] * pair_value).sum(-2)
            message = message.transpose(0, 1).reshape(query.shape[0], self.readout_dim)
            residual = self.W_O(message)
            if observer is not None:
                observer(dict(site="contour_readout", query_corners=query.shape[0],
                              source_faces=memory.shape[0], branch_dtype=str(residual.dtype),
                              residual=residual.detach(), relation_features=descriptors.detach()))
        return residual


def apply_contour_readout(hidden, memory, x, t, known_mask, valid_mask,
                          boundary_edge_mask, module, *, observer=None):
    """FP32(hidden + delta), then cast, and copy only valid free corners."""
    batch, faces = known_mask.shape
    rows = []
    for b in range(batch):
        free_indices = corner_indices(valid_mask[b] & ~known_mask[b])
        source_faces = torch.nonzero(known_mask[b] & valid_mask[b], as_tuple=False).flatten()
        query = hidden[b].index_select(0, free_indices)
        coordinates = x[b].reshape(faces * 3, 3).index_select(0, free_indices)
        triangles = x[b].reshape(faces, 3, 3).index_select(0, source_faces)
        mask = boundary_edge_mask[b].index_select(0, source_faces)
        def observe(record):
            if observer is not None:
                observer(dict(record, batch_index=b, layer=READOUT_LAYER))
        residual = module(query, memory[b], coordinates, triangles, mask, t[b],
                          observer=observe if observer is not None else None)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            updated = (query.float() + residual).to(hidden.dtype)
        rows.append(hidden[b].index_copy(0, free_indices, updated))
        if observer is not None:
            observer(dict(site="contour_addition", layer=READOUT_LAYER, batch_index=b,
                          free_corner_indices=free_indices.detach(), before=query.detach(),
                          residual=residual.detach(), after=updated.detach()))
    return torch.stack(rows)


class ContourModel(NativeModel):
    """Independent forward with the original Native operations left intact."""
    def contour_metadata(self):
        return dict(version="FULL_CONTOUR_READOUT_V1", insertion="after_complete_block",
                    layer=READOUT_LAYER, hidden_dimension=768, readout_dimension=READOUT_DIM,
                    heads=READOUT_HEADS, head_dimension=READOUT_DIM // READOUT_HEADS,
                    feature_names=list(FEATURE_NAMES), relation_dimension=RELATION_DIM,
                    fixed_length_scale=LENGTH_SCALE, length_scale_formula="sqrt(3)*2/0.3762",
                    boundary_domain="per_C_face_source_ID_incidence_1_edges",
                    edge_order=[[0, 1], [1, 2], [2, 0]], topology_ids_are_features=False,
                    boundary_ties="symmetric_mean_closest_vector_and_tangent_dyad",
                    boundary_distance="actual_minimum_distance_not_norm_of_mean_vector",
                    tie_rounding_tolerance="8*fp32_eps*max(min_distance_squared,fp32_eps*L_squared)",
                    tie_eps_factor=TIE_EPS_FACTOR, missing_boundary_features="finite_zero_with_exists_0",
                    face_memory="current_forward_H0_after_coordinate_role_Geo_three_corner_FP32_mean",
                    detached_memory=False, detached_query=False, cross_forward_cache=False,
                    attention_domain="all_valid_C_faces", geometry_attention_bias=False,
                    top_k=False, query_dependent_pair_value=True, trainable_multiplicative_gate=False,
                    branch_dtype="float32_autocast_disabled", residual_add="cast_original(float32(hidden)+delta_fp32)",
                    qkv_bias=False, output_bias=False, mlp_bias=True,
                    layernorm_affine=False, layernorm_eps=1e-5,
                    layernorm="same_nonaffine_LN_applied_separately_to_query_and_face_memory",
                    mlp_widths=[17, 128, 128], mlp_activation="SiLU",
                    qkv_initialization="xavier_uniform", mlp_initialization="torch_Linear_default",
                    output_initialization="zeros", initialization_seed=self.contour_seed,
                    extra_parameters=sum(p.numel() for p in self.contour_readout.parameters()),
                    expected_extra_parameters=EXTRA_PARAMETERS,
                    extra_parameter_names=[n for n, _ in self.named_parameters() if n.startswith("contour_readout.")])

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *,
                boundary_edge_mask=None, observer=None):
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
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError("y must count real faces only, N128..256")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        if boundary_edge_mask is None or boundary_edge_mask.dtype != torch.bool or boundary_edge_mask.shape != (batch, faces, 3) or boundary_edge_mask.device != x.device:
            raise ValueError("Actual-FULL source boundary_edge_mask[B,padded_N,3] is required")
        if bool((boundary_edge_mask & ~known_mask[..., None]).any()):
            raise ValueError("Source contour metadata must be false outside C")
        hidden, role = self._embed(x, known_mask, valid_mask)
        # The current embedding remains differentiable back to original params.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            face_memory = hidden.float().reshape(batch, faces, 3, -1).mean(dim=2)
        memory = tuple(face_memory[b][known_mask[b] & valid_mask[b]] for b in range(batch))
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                          known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                          c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
            if index == READOUT_LAYER:
                before = hidden
                first_records = [] if isinstance(getattr(self, "initial_zero_observations", None), list) else None
                def observe_branch(record):
                    if self.contour_observer is not None:
                        self.contour_observer(record)
                    if first_records is not None and record["site"] == "contour_addition":
                        # First authorized batch only; no retained graph/tensor.
                        residual = record["residual"]
                        first_records.append(dict(
                            zero_residual_exact=bool((residual == 0).all()),
                            finite_residual=bool(torch.isfinite(residual).all()),
                            free_corners=int(residual.shape[0]),
                            branch_dtype=str(residual.dtype)))
                hidden = apply_contour_readout(hidden, memory, x, t, known_mask, valid_mask,
                    boundary_edge_mask, self.contour_readout,
                    observer=observe_branch if first_records is not None or self.contour_observer is not None else None)
                if first_records is not None:
                    self.initial_zero_observations.append(dict(
                        layer=READOUT_LAYER,
                        zero_residual_exact=all(r["zero_residual_exact"] for r in first_records),
                        unchanged_hidden_exact=torch.equal(before, hidden),
                        finite_residual=all(r["finite_residual"] for r in first_records),
                        branch_dtype="torch.float32",
                        free_corners=sum(r["free_corners"] for r in first_records)))
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)


def upgrade_model(base, seed=INITIALIZATION_SEED):
    """Add a branch without recreating any old parameter or changing its name.

    Restore the old optimizer before appending the new parameter group.  CPU
    fork_rng protects initialization; the caller restores the source checkpoint
    RNG after its complete model/optimizer construction and device transfer.
    """
    if type(base) is not NativeModel or base.readout_mode != "none" or len(base.readouts):
        raise ValueError("Upgrade requires a clean original NativeModel without readouts")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid contour initialization seed")
    original_parameters = tuple((n, id(p)) for n, p in base.named_parameters())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        branch = ContourReadout()
    if sum(p.numel() for p in branch.parameters()) != EXTRA_PARAMETERS:
        raise RuntimeError("Registered contour capacity differs")
    branch.requires_grad_(base.experiment_trainable)
    branch.eval()
    device = next(base.parameters()).device
    if device.type not in ("cpu", "meta"):
        branch.to(device)
    base.__class__ = ContourModel
    base.contour_readout = branch
    base.contour_seed = seed
    base.contour_observer = None
    base.initial_zero_observations = None
    if tuple((n, id(p)) for n, p in base.named_parameters() if not n.startswith("contour_readout.")) != original_parameters:
        raise RuntimeError("Adding the branch changed old parameter identity/order")
    return base


def build_candidate(model_config, trainable=True, seed=INITIALIZATION_SEED):
    """Construction only.  No weights, model forward, or CUDA initialization."""
    return upgrade_model(build_model(model_config, readout_mode="none", trainable=trainable), seed=seed)
