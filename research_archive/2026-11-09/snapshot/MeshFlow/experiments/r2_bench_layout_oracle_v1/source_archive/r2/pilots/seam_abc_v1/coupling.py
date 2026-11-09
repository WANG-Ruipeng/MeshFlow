"""Training-only source-vertex tip/bulk endpoint coupling, version 1.

The source identity is used only after existing OT/corner permutation. No target
geometry, identity or neighborhood is supplied at inference. Each eligible seam
tip has equal weight irrespective of its bulk vertex valence. Coordinates and
loss use the existing model unit; lambda=0.1 is applied by the trainer.
"""
from __future__ import annotations

import numpy as np

VERSION = "source_tip_bulk_endpoint_mse_v1"
LAMBDA = 0.1


def transported_source_vertices(sample, source_face_vertex_ids, source_faces=None):
    faces = np.asarray(sample["source_face_ids"], np.int64)
    corners = np.asarray(sample["source_corner_permutations"], np.int64)
    table = np.asarray(source_face_vertex_ids, np.int64)
    if corners.shape != (len(faces), 3) or not np.array_equal(
        np.sort(corners, axis=1), np.tile(np.arange(3), (len(faces), 1))
    ):
        raise ValueError("Invalid transported source corner permutations")
    if (faces < 0).any() or (faces >= len(table)).any():
        raise ValueError("Source face outside registered parent")
    current = table[faces[:, None], corners]
    if source_faces is not None:
        winding = np.asarray(sample["source_winding_permutation"], np.int64)
        restored = np.take_along_axis(current, winding, axis=1)
        if not np.array_equal(restored, np.asarray(source_faces)[faces]):
            raise ValueError("OT-transported identity disagrees with source winding")
    return current


def build_pairs(sample, source_face_vertex_ids, source_faces=None):
    """Return flattened tip/bulk corner pairs and inverse-valence weights."""
    vertices = transported_source_vertices(sample, source_face_vertex_ids, source_faces)
    third = np.asarray(sample["seam_third_corner_mask"], bool)
    free = np.asarray(sample["free_corner_mask"], bool)
    known = np.asarray(sample["known_mask"], bool)
    if third.shape != vertices.shape or np.any(third & ~free) or third[known].any():
        raise ValueError("Seam tip mask must select genuinely free owner corners")
    # A Null boundary slot is an ordinary all-free bulk face. All active owner
    # faces are excluded, including their fixed corners and their free tips.
    bulk_faces = ~known & ~third.any(axis=1) & free.all(axis=1)
    bulk_flat = np.flatnonzero(np.broadcast_to(bulk_faces[:, None], vertices.shape).reshape(-1))
    all_tips = np.flatnonzero(third.reshape(-1))
    ids = vertices.reshape(-1)
    tip_indices, bulk_indices, weights, eligible = [], [], [], []
    for tip in all_tips:
        neighbors = bulk_flat[ids[bulk_flat] == ids[tip]]
        if not len(neighbors):
            continue
        eligible.append(int(tip))
        tip_indices.extend([int(tip)] * len(neighbors))
        bulk_indices.extend(neighbors.tolist())
        weights.extend([1.0 / len(neighbors)] * len(neighbors))
    result = dict(tip_indices=np.asarray(tip_indices, np.int64),
                  bulk_indices=np.asarray(bulk_indices, np.int64),
                  pair_weights=np.asarray(weights, np.float64),
                  eligible_tip_indices=np.asarray(eligible, np.int64),
                  eligible_tip_count=len(eligible), total_tip_count=len(all_tips),
                  unmatched_tip_count=len(all_tips) - len(eligible),
                  pair_count=len(tip_indices), version=VERSION)
    if len(tip_indices):
        x1 = np.asarray(sample["x1"]).reshape(-1, 3)
        if not np.array_equal(x1[result["tip_indices"]], x1[result["bulk_indices"]]):
            raise ValueError("Same-source tip and bulk targets differ after OT/augmentation")
        if not np.isclose(sum(weights), len(eligible), atol=1e-12, rtol=0):
            raise ValueError("Per-tip equal weighting failed")
    return result


def endpoint_coupling(prediction, sample, pairs, *, eligible_tip_denominator):
    """Use one existing velocity forward; return unweighted loss and diagnostics.

    denominator is the number of eligible tips in the full effective batch,
    not the number in this microbatch. Both tip and bulk receive gradients.
    """
    import torch

    den = int(eligible_tip_denominator)
    if den < pairs["eligible_tip_count"] or den < 0:
        raise ValueError("Effective-batch eligible-tip denominator is invalid")
    velocity = prediction.float().reshape(-1, 3)
    factor = 1.0 - float(sample["t"])
    facts = dict(version=VERSION, eligible_tip_count=pairs["eligible_tip_count"],
                 total_tip_count=pairs["total_tip_count"], unmatched_tip_count=pairs["unmatched_tip_count"],
                 pair_count=pairs["pair_count"], batch_tip_denominator=den,
                 scalar_denominator=3 * den, time=float(sample["t"]), units="model_coordinate_squared")
    if not den or not pairs["pair_count"]:
        loss = velocity.sum() * 0.0
        facts.update(loss=0.0, endpoint_weighted_sse=0.0, velocity_gradient_l2_analytic=0.0)
        return loss, facts
    cv = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=velocity.device)
    tips = cv(pairs["tip_indices"], torch.long)
    bulk = cv(pairs["bulk_indices"], torch.long)
    weights = cv(pairs["pair_weights"], torch.float32)
    xt = cv(np.asarray(sample["xt"]).reshape(-1, 3), torch.float32)
    endpoint = xt + factor * velocity
    delta = endpoint[tips] - endpoint[bulk]
    sse = (delta.square().sum(-1) * weights).sum()
    loss = sse / (3 * den)
    # Exact velocity-space gradient of this loss, not parameter-gradient norm.
    with torch.no_grad():
        pair_grad = (delta.detach().cpu().numpy().astype(np.float64)
                     * (2.0 * factor / (3 * den)) * pairs["pair_weights"][:, None])
        grad = np.zeros((velocity.shape[0], 3), np.float64)
        np.add.at(grad, pairs["tip_indices"], pair_grad)
        np.add.at(grad, pairs["bulk_indices"], -pair_grad)
        grad_norm = float(np.linalg.norm(grad))
    facts.update(loss=float(loss.detach()), endpoint_weighted_sse=float(sse.detach()),
                 velocity_gradient_l2_analytic=grad_norm)
    return loss, facts
