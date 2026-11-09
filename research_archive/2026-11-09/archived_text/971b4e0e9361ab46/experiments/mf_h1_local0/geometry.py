"""Finite, unoriented primitive geometry used only to form control features.

Nothing in this module changes the evolving mesh. Inputs and returned closest
points stay in the same global object frame. The evaluation oracle must remain
independent of this training implementation.

Degenerate policy: zero length segments return their endpoint; triangles with
area below floating point resolution use the closest of their three finite
edges. Lexicographic vertex ordering makes edge ties and the fallback invariant
to every permutation of the input vertices. No signed normal is exposed.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _canonical_vertices(vertices: Tensor) -> Tensor:
    """Stable xyz lexicographic order; last two axes are vertex and xyz."""
    result = vertices
    for coordinate in (2, 1, 0):
        order = torch.argsort(result[..., coordinate], dim=-1, stable=True)
        result = result.gather(-2, order.unsqueeze(-1).expand_as(result))
    return result


def closest_point_segment(points: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Closest point on finite [a,b], broadcasting all leading dimensions."""
    a, b = torch.broadcast_tensors(a, b)
    endpoints = _canonical_vertices(torch.stack((a, b), dim=-2))
    a, b = endpoints.unbind(-2)
    edge = b - a
    denominator = edge.square().sum(-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator,
                                   torch.ones_like(denominator))
    fraction = ((points - a) * edge).sum(-1, keepdim=True) / safe_denominator
    fraction = fraction.clamp(0, 1)
    return a + fraction * edge


def closest_point_triangle(points: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Closest point on the *finite* triangle, including degenerate triangles.

    The candidates are the plane projection (only inside the triangle) and the
    three finite edge projections. Squared area is computed with a cross product
    to avoid subtracting nearly equal Gram determinant terms. The resolution
    threshold is 64*eps**2*max_edge_length**4; this is a symmetric numerical
    fallback, not a training prompt filter.
    """
    a, b, c = torch.broadcast_tensors(a, b, c)
    vertices = _canonical_vertices(torch.stack((a, b, c), dim=-2))
    a, b, c = vertices.unbind(-2)
    ab, ac, bc = b - a, c - a, c - b
    normal = torch.linalg.cross(ab, ac, dim=-1)
    area_squared = normal.square().sum(-1, keepdim=True)
    longest_squared = torch.stack((ab.square().sum(-1), ac.square().sum(-1),
                                    bc.square().sum(-1)), -1).amax(-1, keepdim=True)
    epsilon = torch.finfo(points.dtype).eps
    usable = area_squared > 64 * epsilon**2 * longest_squared.square()
    safe_area = torch.where(usable, area_squared, torch.ones_like(area_squared))
    offset = points - a
    projection = points - (offset * normal).sum(-1, keepdim=True) / safe_area * normal
    projected_offset = projection - a
    coord_b = (torch.linalg.cross(projected_offset, ac.expand_as(projected_offset), dim=-1) * normal).sum(-1, keepdim=True) / safe_area
    coord_c = (torch.linalg.cross(ab.expand_as(projected_offset), projected_offset, dim=-1) * normal).sum(-1, keepdim=True) / safe_area
    inside = usable & (coord_b >= 0) & (coord_c >= 0) & (coord_b + coord_c <= 1)
    candidates = torch.stack((projection, closest_point_segment(points, a, b),
                              closest_point_segment(points, a, c),
                              closest_point_segment(points, b, c)), -2)
    distances = (candidates - points.unsqueeze(-2)).square().sum(-1)
    distances = torch.cat((distances[..., :1].masked_fill(~inside, torch.inf),
                           distances[..., 1:]), -1)
    selected = distances.argmin(-1, keepdim=True)
    return candidates.gather(-2, selected.unsqueeze(-1).expand(*selected.shape, 3)).squeeze(-2)


def primitive_centroids(vertices: Tensor, arity: Tensor) -> Tensor:
    """[B,K,3,3] + [B,K] arities 2/3 -> [B,K,3] geometric centroids."""
    keep = torch.arange(3, device=vertices.device) < arity.unsqueeze(-1)
    clean = torch.where(keep.unsqueeze(-1), vertices, torch.zeros_like(vertices))
    return clean.sum(-2) / arity.clamp_min(1).unsqueeze(-1)


def closest_points_primitives(points: Tensor, vertices: Tensor, arity: Tensor) -> Tensor:
    """[B,Q,3] query corners to [B,K,3,3] -> [B,Q,K,3].

    Arity 2 uses only the first two vertices, including when the unused third
    slot contains arbitrary padding. Callers must mask padded primitives.
    """
    query = points.unsqueeze(-2)
    a = vertices[:, None, :, 0, :]
    b = vertices[:, None, :, 1, :]
    c = vertices[:, None, :, 2, :]
    c = torch.where((arity == 2)[:, None, :, None], a, c)
    segment = closest_point_segment(query, a, b)
    triangle = closest_point_triangle(query, a, b, c)
    return torch.where((arity == 2)[:, None, :, None], segment, triangle)


def relation_features(delta: Tensor) -> Tensor:
    """R(delta): xyz, squared norm, and four pi*2**k sin/cos bands (28)."""
    bands = torch.arange(4, dtype=delta.dtype, device=delta.device)
    phase = delta.unsqueeze(-1) * (torch.pi * 2.0**bands)
    return torch.cat((delta, delta.square().sum(-1, keepdim=True),
                      phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
