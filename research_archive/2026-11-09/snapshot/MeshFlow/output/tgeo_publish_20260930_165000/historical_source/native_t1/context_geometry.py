"""C-only face geometry and one-layer context encoding for Native T1.

Geometry is recomputed on every forward. Exact FP32 coordinate equality recovers
shared edges; it cannot identify topologically separate coincident vertices.
The production branch is FP32 even inside the backbone BF16 autocast context.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

FEATURE_DIM = 13
HIDDEN_DIM = 128
OUTPUT_DIM = 768
INITIALIZATION_SEED = 1010
AREA_MIN = 1e-12
FORMULA_ATOL = 2e-6
FORMULA_RTOL = 2e-5
ENCODER_METADATA = dict(
    schema="native_t1_geom_context_v1", feature_dim=13, hidden_dim=128,
    graph_layers=1, branch_dtype="torch.float32", output_dim=768,
    injection="after_coordinate_embedding_and_role_before_blocks",
    geometry="centroid_sorted_edges_area_unsigned_normal_outer_six",
    graph="exact_FP32_shared_whole_edge_binary_row_normalize_A_plus_I",
    area_reject_le=AREA_MIN, layer_norm_eps=1e-5, initialization_seed=1010)
GEOMETRY_SPEC = ENCODER_METADATA


def _canonical_corners(c):
    # Stable lexicographic x/y/z order removes S3 arithmetic-order differences.
    result = c
    for axis in (2, 1, 0):
        order = torch.argsort(result[..., axis], dim=1, stable=True)
        result = torch.gather(result, 1, order[..., None].expand(-1, -1, 3))
    return result


def face_geometry(c):
    """Return phi[K,13], binary A[K,K], and geometry facts from C alone.

    Production input is FP32; FP64 is admitted solely for small numerical
    reference tests. No free coordinates or source topology are accepted.
    AREA_MIN is in original model coordinate units squared, with no bbox scaling.
    Nonmanifold shared edges remain binary neighbors and are explicitly counted.
    """
    if (not isinstance(c, torch.Tensor) or c.ndim != 3 or c.shape[1:] != (3, 3)
            or c.shape[0] < 1 or c.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Expected nonempty floating C[K,3,3]")
    with torch.autocast(device_type=c.device.type, enabled=False):
        if not bool(torch.isfinite(c).all()):
            raise ValueError("Nonfinite known coordinate")
        ordered = _canonical_corners(c)
        vectors = torch.stack((ordered[:, 1] - ordered[:, 0],
                               ordered[:, 2] - ordered[:, 0],
                               ordered[:, 2] - ordered[:, 1]), dim=1)
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        if bool((lengths == 0).any()):
            raise ValueError("Known triangle has a zero-length edge")
        cross = torch.linalg.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        twice_area = torch.linalg.vector_norm(cross, dim=-1)
        area = twice_area * 0.5
        if bool((area <= AREA_MIN).any()):
            raise ValueError("Known triangle area is at or below 1e-12 model units squared")
        normal = cross / twice_area[:, None]
        nx, ny, nz = normal.unbind(-1)
        unsigned = torch.stack((nx*nx, ny*ny, nz*nz, nx*ny, nx*nz, ny*nz), -1)
        phi = torch.cat((ordered.mean(1), lengths.sort(dim=1).values,
                         area[:, None], unsigned), -1)
        if not bool(torch.isfinite(phi).all()):
            raise ValueError("Known geometry overflows the selected floating-point precision")
        k = c.shape[0]
        same_face = (ordered[:, None] == ordered[None, :]).all(-1).all(-1)
        eye = torch.eye(k, device=c.device, dtype=torch.bool)
        if bool((same_face & ~eye).any()):
            raise ValueError("Duplicate known face (including permuted corners)")
        # Canonical vertex ordering also canonically orders each whole edge.
        edges = ordered[:, ((0, 1), (0, 2), (1, 2)), :]
        same_edge = (edges[:, None, :, None] == edges[None, :, None, :]).all(-1).all(-1)
        adjacency = same_edge.any(-1).any(-1) & ~eye
        flat = edges.reshape(3*k, 2, 3)
        edge_equal = (flat[:, None] == flat[None, :]).all(-1).all(-1)
        incidences = edge_equal.sum(-1)
        unique_first = ~torch.tril(edge_equal, diagonal=-1).any(-1)
        return phi, adjacency, dict(
            area=area, sorted_edge_lengths=lengths.sort(dim=1).values,
            nonmanifold_edge_count=((incidences > 2) & unique_first).sum(),
            unique_edge_count=unique_first.sum(),
            coordinate_topology_limitation="coincident but topologically separate vertices are indistinguishable")


def _snapshot(record):
    # Observation is read-only. Detached references avoid per-forward tensor
    # copies for timing-only observers; a retaining observer owns its own copies.
    return {key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in record.items()}


class ContextGeometryEncoder(nn.Module):
    """Same 149888 parameters for identity (geo) and C-adjacency (graph)."""
    def __init__(self, mode="geo", *, seed=INITIALIZATION_SEED):
        super().__init__()
        if mode not in ("geo", "graph"):
            raise ValueError("Context mode must be geo or graph")
        if type(seed) is not int:
            raise ValueError("Initialization seed must be an integer")
        self.mode, self.initialization_seed = mode, seed
        # Explicit CPU construction does not consume the caller's CPU or CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            opts = dict(device="cpu", dtype=torch.float32)
            self.mlp = nn.Sequential(nn.Linear(13, 128, **opts), nn.SiLU(),
                                     nn.Linear(128, 128, **opts), nn.SiLU())
            self.W_self = nn.Linear(128, 128, **opts)
            self.W_neighbor = nn.Linear(128, 128, **opts)
            self.norm = nn.LayerNorm(128, eps=1e-5, **opts)
            self.W_out = nn.Linear(128, 768, bias=False, **opts)
            nn.init.zeros_(self.W_out.weight)
        self.eval()

    def metadata(self):
        return dict(copy.deepcopy(ENCODER_METADATA), context_mode=self.mode,
                    initialization_seed=self.initialization_seed,
                    additional_parameters=sum(p.numel() for p in self.parameters()))

    def forward(self, x, known_mask, valid_mask, *, observer=None):
        if observer is not None:
            observer(dict(site="geometry_start", batch_index=None, known_count=None, batch_size=x.shape[0]))
        if (x.ndim != 3 or x.shape[-1] != 9 or x.dtype != torch.float32
                or known_mask.shape != x.shape[:2] or valid_mask.shape != x.shape[:2]
                or known_mask.dtype != torch.bool or valid_mask.dtype != torch.bool
                or known_mask.device != x.device or valid_mask.device != x.device):
            raise ValueError("Expected FP32 x[B,N,9] and colocated bool masks")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if any(p.dtype != torch.float32 or p.device != x.device for p in self.parameters()):
            raise ValueError("Context parameters must be colocated FP32")
        batch, faces, _ = x.shape
        output = []
        known_counts = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for b in range(batch):
                indices = torch.nonzero(known_mask[b], as_tuple=False).flatten()
                known_counts.append(indices.numel())
                if indices.numel() == 0:
                    raise ValueError("At least one known face is required")
                c = x[b].reshape(faces, 3, 3).index_select(0, indices)
                phi, adjacency, facts = face_geometry(c)
                identity = torch.eye(indices.numel(), dtype=torch.float32, device=x.device)
                if self.mode == "graph":
                    p = adjacency.to(torch.float32) + identity
                    p = p / p.sum(dim=-1, keepdim=True)
                else:
                    p = identity
                h0 = self.mlp(phi)
                message = p @ h0
                h1 = h0 + torch.nn.functional.silu(self.W_self(h0) + self.W_neighbor(message))
                delta = self.W_out(self.norm(h1))
                # Dense selection has unique slots, no scatter reduction, and no
                # dependency on any unselected free/padding coordinate value.
                placement = (torch.arange(faces, device=x.device)[:, None] == indices[None, :]).float()
                dense = placement @ delta
                dense = torch.where(known_mask[b, :, None], dense, torch.zeros_like(dense))
                output.append(dense)
                if observer is not None and getattr(observer, "requires_geometry_details", True):
                    observer(_snapshot(dict(site="geometry", batch_index=b, known_indices=indices,
                        phi=phi, adjacency=adjacency, P=p, h0=h0, h1=h1,
                        delta=delta, nonmanifold_edge_count=facts["nonmanifold_edge_count"],
                        geometry_dtype=str(phi.dtype), encoding_dtype=str(delta.dtype))))
        result = torch.stack(output)
        if observer is not None:
            observer(dict(site="geometry_end", batch_index=None, known_count=known_counts, batch_size=batch))
        return result


def make_context_encoder(mode, seed=INITIALIZATION_SEED, device="cpu"):
    """Construct matching seeded branches without advancing caller RNG."""
    return ContextGeometryEncoder(mode, seed=seed).to(device)


def attach_context_encoder(model, mode, *, seed=INITIALIZATION_SEED):
    """Explicitly add the branch; retain the caller's trainable/frozen policy."""
    if getattr(model, "context_encoder", None) is not None:
        raise ValueError("Context encoder is already attached")
    encoder = ContextGeometryEncoder(mode, seed=seed).to(model.role_embedding.device)
    model.context_encoder = encoder
    model.set_trainable(model.experiment_trainable).eval()
    return encoder.metadata()


def inject_context(hidden, delta, known_mask, *, observer=None):
    """Broadcast FP32 face residual to corners after casting to hidden.dtype."""
    batch, faces, width = delta.shape
    if hidden.shape != (batch, 3*faces, width):
        raise ValueError("Hidden and context residual dimensions differ")
    cast = delta.to(dtype=hidden.dtype)
    corner_delta = cast.unsqueeze(2).repeat(1, 1, 3, 1).reshape_as(hidden)
    after = hidden + corner_delta
    if observer is not None and getattr(observer, "requires_injection_details", True):
        observer(_snapshot(dict(site="geometry_injection", hidden_before=hidden,
            hidden_after=after, delta=delta, cast_corner_delta=corner_delta,
            known_mask=known_mask, hidden_dtype=str(hidden.dtype),
            residual_dtype=str(delta.dtype), after_dtype=str(after.dtype))))
    return after


class ContextGeometryObserver:
    """Optional selected-step diagnostics; one call observes one actual forward.

    No new model/encoder forward is issued. CPU scalar reads/synchronization are
    confined to this explicitly enabled observer. Raw geometry is opt-in.
    """
    def __init__(self, *, retain_geometry=False):
        self.retain_geometry = retain_geometry
        self.geometry = []
        self.injections = []

    def __call__(self, record):
        if record["site"] in ("geometry_start", "geometry_end"):
            return
        if record["site"] == "geometry":
            item = dict(batch_index=record["batch_index"],
                        known_count=int(record["phi"].shape[0]),
                        nonmanifold_edge_count=int(record["nonmanifold_edge_count"]),
                        geometry_dtype=record["geometry_dtype"], encoding_dtype=record["encoding_dtype"])
            if self.retain_geometry:
                item.update({key: record[key].cpu().tolist()
                             for key in ("known_indices", "phi", "P", "h1", "delta")})
            self.geometry.append(item)
        elif record["site"] == "geometry_injection":
            before = record["hidden_before"].float()
            after = record["hidden_after"].float()
            original = record["delta"][:, :, None, :].expand(-1, -1, 3, -1).reshape_as(before)
            effective = after - before
            known = record["known_mask"][:, :, None].expand(-1, -1, 3).reshape(before.shape[:2])
            nonzero = (original != 0) & known[..., None]
            swallowed = nonzero & (after == before)
            n_nonzero = int(nonzero.sum())
            rms = lambda a: float(a.double().square().mean().sqrt())
            self.injections.append(dict(
                delta_RMS_known=rms(original[known]), delta_RMS_all=rms(original),
                effective_RMS_known=rms(effective[known]),
                effective_minus_delta_RMS_known=rms((effective-original)[known]),
                nonzero_delta_elements=n_nonzero, swallowed_nonzero_elements=int(swallowed.sum()),
                swallowed_nonzero_fraction=(float(swallowed.sum()) / n_nonzero if n_nonzero else None),
                free_direct_delta_exact_zero=bool((original[~known] == 0).all()),
                free_hidden_unchanged=bool((after[~known] == before[~known]).all()),
                hidden_dtype=record["hidden_dtype"], residual_dtype=record["residual_dtype"],
                after_dtype=record["after_dtype"]))
        else:
            raise ValueError("Unknown context diagnostic event")

    def as_dict(self):
        return dict(geometry=self.geometry, injections=self.injections,
                    additional_encoder_forwards=0, additional_model_forwards=0)


GeometryObserver = ContextGeometryObserver
