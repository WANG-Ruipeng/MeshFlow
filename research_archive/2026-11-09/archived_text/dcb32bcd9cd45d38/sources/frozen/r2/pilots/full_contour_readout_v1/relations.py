"""Finite C-surface and per-face source-contour descriptors, version 1.

This module never infers topology by welding coordinates.  The caller supplies
one boolean for each local edge (01, 12, 20), established from C-only source-ID
incidence and transported with the actual current corner ordering.
"""
from __future__ import annotations

import math
import torch

LENGTH_SCALE = math.sqrt(3.0) * 2.0 / 0.3762
RELATION_DIM = 17
EDGE_CORNERS = ((0, 1), (1, 2), (2, 0))
TIE_EPS_FACTOR = 8.0
DYAD_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
FEATURE_NAMES = (
    "surface_dx_over_L", "surface_dy_over_L", "surface_dz_over_L",
    "surface_distance_over_L", "boundary_dx_over_L", "boundary_dy_over_L",
    "boundary_dz_over_L", "boundary_distance_over_L", "tangent_xx",
    "tangent_yy", "tangent_zz", "tangent_xy", "tangent_xz", "tangent_yz",
    "boundary_exists", "boundary_edge_count_over_3", "free_time",
)


def _lexicographic_rows(value):
    """Canonical arithmetic order only; it neither welds nor changes topology."""
    result = value
    for coordinate in reversed(range(value.shape[-1])):
        order = torch.argsort(result[..., coordinate], dim=-1, stable=True)
        result = result.gather(-2, order[..., None].expand_as(result))
    return result


def _segment(points, a, b):
    # Endpoints supplied in a deterministic geometry order by callers.
    edge = b - a
    denominator = edge.square().sum(-1, keepdim=True)
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    fraction = (((points - a) * edge).sum(-1, keepdim=True) / safe).clamp(0, 1)
    return a + fraction * edge


def closest_point_triangle(points, triangles):
    """FP32 queries [Q,3] and finite triangles [K,3,3] -> [Q,K,3].

    Uses plane projection only when inside a numerically resolved triangle,
    otherwise its finite edges.  The symmetric 64*eps**2 area test and finite
    edge fallback follow the earlier G_closest implementation, copied into this
    independent pilot rather than importing a retired activity package.
    """
    vertices = _lexicographic_rows(triangles)
    a, b, c = (v[None] for v in vertices.unbind(-2))
    query = points[:, None, :]
    ab, ac, bc = b - a, c - a, c - b
    normal = torch.linalg.cross(ab, ac, dim=-1)
    area_squared = normal.square().sum(-1, keepdim=True)
    longest_squared = torch.stack((ab.square().sum(-1), ac.square().sum(-1),
                                   bc.square().sum(-1)), -1).amax(-1, keepdim=True)
    eps = torch.finfo(points.dtype).eps
    usable = area_squared > 64 * eps**2 * longest_squared.square()
    safe_area = torch.where(usable, area_squared, torch.ones_like(area_squared))
    projection = query - ((query - a) * normal).sum(-1, keepdim=True) / safe_area * normal
    offset = projection - a
    coord_b = (torch.linalg.cross(offset, ac.expand_as(offset), dim=-1) * normal).sum(-1, keepdim=True) / safe_area
    coord_c = (torch.linalg.cross(ab.expand_as(offset), offset, dim=-1) * normal).sum(-1, keepdim=True) / safe_area
    inside = usable & (coord_b >= 0) & (coord_c >= 0) & (coord_b + coord_c <= 1)
    candidates = torch.stack((projection, _segment(query, a, b),
                              _segment(query, a, c), _segment(query, b, c)), -2)
    distances = (candidates - query[..., None, :]).square().sum(-1)
    distances = torch.cat((distances[..., :1].masked_fill(~inside, torch.inf),
                           distances[..., 1:]), -1)
    # Canonical triangle arithmetic removes any input-order tie preference.
    selected = distances.argmin(-1, keepdim=True)
    return candidates.gather(-2, selected[..., None].expand(*selected.shape, 3)).squeeze(-2)


def _canonical_edges(triangles, boundary_edge_mask):
    starts = triangles[:, [0, 1, 2], :]
    ends = triangles[:, [1, 2, 0], :]
    endpoints = _lexicographic_rows(torch.stack((starts, ends), -2))
    flattened = endpoints.flatten(-2)
    mask = boundary_edge_mask
    # A canonical arithmetic order makes summing tied candidates invariant to
    # corner permutations.  IDs never enter this ordering or the descriptors.
    for coordinate in reversed(range(6)):
        order = torch.argsort(flattened[..., coordinate], dim=-1, stable=True)
        flattened = flattened.gather(-2, order[..., None].expand_as(flattened))
        mask = mask.gather(-1, order)
    return flattened.reshape(-1, 3, 2, 3), mask


def relation_features(points, triangles, boundary_edge_mask, time):
    """Return [Q,K,17] finite FP32 pair descriptors without changing geometry.

    B_j contains only current C face j's incidence-one edges.  For numerically
    equal nearest edges, average closest-point vectors and unsigned tangent
    dyads symmetrically.  Distance remains the actual minimum distance, not the
    norm of the averaged vector.  The tie tolerance is a fixed FP32 rounding
    rule, 8*eps*max(d_min**2, eps*L**2), not a spatial support hyperparameter.
    """
    if points.ndim != 2 or points.shape[-1] != 3 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected points[Q,3], triangles[K,3,3]")
    if points.dtype != torch.float32 or triangles.dtype != torch.float32:
        raise ValueError("Relation coordinates must be the actual FP32 state")
    if boundary_edge_mask.dtype != torch.bool or boundary_edge_mask.shape != triangles.shape[:2]:
        raise ValueError("Expected boolean source-topology boundary mask[K,3]")
    if not (points.device == triangles.device == boundary_edge_mask.device):
        raise ValueError("Relation inputs must be colocated")
    if not isinstance(time, torch.Tensor) or time.numel() != 1 or time.device != points.device or time.dtype != torch.float32:
        raise ValueError("Expected one colocated FP32 free time")
    with torch.autocast(device_type=points.device.type, enabled=False):
        surface_vector = closest_point_triangle(points, triangles) - points[:, None, :]
        surface_distance = torch.linalg.vector_norm(surface_vector, dim=-1, keepdim=True)
        edges, mask = _canonical_edges(triangles, boundary_edge_mask)
        a, b = edges[:, :, 0, :], edges[:, :, 1, :]
        edge = b - a
        edge_length = torch.linalg.vector_norm(edge, dim=-1, keepdim=True)
        unit = edge / torch.where(edge_length > 0, edge_length, torch.ones_like(edge_length))
        ux, uy, uz = unit.unbind(-1)
        dyad = torch.stack((ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz), -1)
        closest = _segment(points[:, None, None, :], a[None], b[None])
        vectors = closest - points[:, None, None, :]
        distances_squared = vectors.square().sum(-1)
        has_boundary = mask.any(-1)
        minimum = distances_squared.masked_fill(~mask[None], torch.inf).amin(-1)
        # Remove infinities before computing ties, distances or MLP inputs.
        minimum = torch.where(has_boundary[None], minimum, torch.zeros_like(minimum))
        eps = torch.finfo(torch.float32).eps
        tolerance = TIE_EPS_FACTOR * eps * minimum.abs().clamp_min(eps * LENGTH_SCALE**2)
        tied = mask[None] & ((distances_squared - minimum[..., None]).abs() <= tolerance[..., None])
        weights = tied.float() / tied.sum(-1, keepdim=True).clamp_min(1)
        boundary_vector = (vectors * weights[..., None]).sum(-2)
        boundary_distance = minimum.clamp_min(0).sqrt()[..., None]
        boundary_dyad = (dyad[None] * weights[..., None]).sum(-2)
        shape = (*surface_vector.shape[:2], 1)
        exists = has_boundary.float()[None, :, None].expand(shape)
        count = (mask.sum(-1).float() / 3.0)[None, :, None].expand(shape)
        free_time = time.reshape(1, 1, 1).expand(shape)
        result = torch.cat((surface_vector / LENGTH_SCALE, surface_distance / LENGTH_SCALE,
                            boundary_vector / LENGTH_SCALE, boundary_distance / LENGTH_SCALE,
                            boundary_dyad, exists, count, free_time), -1)
    return result


def boundary_mask_from_vertex_ids(vertex_ids):
    """Small CPU helper for explicit C topology, also used by unit tests.

    No coordinate inference, source Y or ID magnitudes are used.  An edge with
    incidence > 2 is not open; this descriptor does not repair nonmanifold C.
    """
    if vertex_ids.device.type != "cpu" or vertex_ids.ndim != 2 or vertex_ids.shape[1] != 3 or vertex_ids.dtype != torch.int64:
        raise ValueError("Expected CPU int64 C vertex IDs[K,3]")
    rows = vertex_ids.tolist()
    incidence = {}
    for row in rows:
        for a, b in EDGE_CORNERS:
            key = tuple(sorted((row[a], row[b])))
            incidence[key] = incidence.get(key, 0) + 1
    return torch.tensor([[incidence[tuple(sorted((row[a], row[b])))] == 1
                          for a, b in EDGE_CORNERS] for row in rows], dtype=torch.bool).reshape(-1, 3)
