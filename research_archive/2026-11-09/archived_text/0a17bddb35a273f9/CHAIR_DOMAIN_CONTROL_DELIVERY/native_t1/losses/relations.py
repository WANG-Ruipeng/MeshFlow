"""Detached source-ID interface targets; no model-visible topology or loss search.

This minimal builder preserves the established endpoint ordering and validation.
It prepares known/free shared-edge pairs only. The finite-edge objective then
requires global incidence exactly two; no coordinate welding infers identities.
"""
from itertools import combinations
import math
import numpy as np
import torch

def _geometry(value, name, *, dtype=None, device=None):
    value = torch.as_tensor(value, dtype=dtype, device=device).detach().clone()
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError(name + " must be [faces,3,3]")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(name + " must use FP32 or FP64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    return value

def _ids(value, faces, name):
    value = torch.as_tensor(value).detach().cpu()
    if value.shape != (faces, 3) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(name + " must be integer [faces,3]")
    array = value.numpy().astype(np.int64, copy=False)
    if np.any(array < 0) or any(len(set(row.tolist())) != 3 for row in array):
        raise ValueError(name + " needs three distinct nonnegative source IDs per face")
    return array

def _cross(triangles):
    return torch.linalg.cross(triangles[:, 1]-triangles[:, 0],
                              triangles[:, 2]-triangles[:, 0], dim=-1)

def prepare_relations(free_gt, known, free_vertex_ids, known_vertex_ids, bbox_L):
    """Validate fixed geometry and source IDs in actual target face/corner order."""
    free = _geometry(free_gt, "free_gt")
    fixed = _geometry(known, "known", dtype=free.dtype, device=free.device)
    if not len(free):
        raise ValueError("At least one free face is required")
    scale = float(bbox_L.detach().cpu()) if isinstance(bbox_L, torch.Tensor) else float(bbox_L)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("bbox_L must be a positive finite fixed scalar")
    fids = _ids(free_vertex_ids, len(free), "free_vertex_ids")
    kids = _ids(known_vertex_ids, len(fixed), "known_vertex_ids")
    all_ids = np.concatenate((fids, kids), axis=0)
    source_triangles = torch.cat((free, fixed), dim=0)
    coordinates = {}
    edge_faces = {}
    for face, ids in enumerate(all_ids):
        for corner, vid in enumerate(ids):
            old = coordinates.setdefault(int(vid), source_triangles[face, corner])
            if not torch.equal(old, source_triangles[face, corner]):
                raise ValueError("Same source vertex ID has inconsistent coordinates")
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(ids[a]), int(ids[b]))))
            edge_faces.setdefault(key, []).append(face)
    normalized = source_triangles / scale
    cross_length = torch.linalg.vector_norm(_cross(normalized), dim=-1)
    if bool((cross_length == 0).any()):
        raise ValueError("Degenerate clean GT face cannot define the registered relation")
    interfaces, interface_corners, endpoints, interface_lengths = [], [], [], []
    nfree = len(free)
    for edge in sorted(edge_faces):
        a, b = (coordinates[v] / scale for v in edge)
        length = torch.linalg.vector_norm(b-a)
        if float(length) == 0:
            raise ValueError("Zero GT edge cannot define a relation")
        for i, j in combinations(sorted(edge_faces[edge]), 2):
            if (i < nfree) == (j < nfree):
                continue
            fi = i if i < nfree else j
            corners = [int(np.flatnonzero(all_ids[fi] == vid)[0]) for vid in edge]
            interfaces.append(fi)
            interface_corners.append(corners)
            endpoints.append(torch.stack((a, b)))
            interface_lengths.append(length)
    def floats(values, shape):
        return torch.stack(values).detach() if values else free.new_empty(shape)
    def integers(values, shape):
        return torch.tensor(values, dtype=torch.long, device=free.device).reshape(shape)
    return dict(bbox_L=scale, free_gt=(free/scale).detach(),
        interface_free_faces=integers(interfaces, (-1,)),
        interface_free_corners=integers(interface_corners, (-1,2)),
        interface_endpoints=floats(endpoints, (0,2,3)),
        interface_lengths=floats(interface_lengths, (0,)),
        counts=dict(known_free_interface_relations=len(interfaces)))
