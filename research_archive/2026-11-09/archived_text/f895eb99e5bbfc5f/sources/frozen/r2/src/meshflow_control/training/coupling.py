"""Exact historical free-only nested OT, independently packaged.

Adapted from chair_hybrid_coupling_v1/coupling.py; provenance hashes are
recorded by import_legacy_data. Costs retain scipy cdist's Euclidean default,
six corner permutations and Hungarian face assignment. No model is imported.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

CONTRACT = "original_preOT_cost_scale__FP32_final_target_x2__raw_iid_noise_slots_v1"

def validate_map(face, corner):
    face, corner = np.asarray(face), np.asarray(corner)
    n = len(face)
    if face.dtype != np.int64 or face.shape != (n,) or not np.array_equal(np.sort(face), np.arange(n)):
        raise ValueError("Invalid face bijection")
    if corner.dtype != np.int64 or corner.shape != (n, 3) or not np.array_equal(np.sort(corner, axis=1), np.tile(np.arange(3), (n, 1))):
        raise ValueError("Invalid corner bijection")
    return face, corner

def nested_ot_map(pre_target, raw_noise):
    data, noise = np.asarray(pre_target), np.asarray(raw_noise)
    if data.ndim != 3 or data.shape[1:] != (3, 3) or data.shape != noise.shape or not len(data):
        raise ValueError("Expected nonempty aligned free triangles")
    if not np.isfinite(data).all() or not np.isfinite(noise).all():
        raise ValueError("Nonfinite OT input")
    n = len(data)
    permutations = np.array(list(itertools.permutations([0, 1, 2])), dtype=np.int32)
    costs = np.stack([cdist(data.reshape(n, -1), noise[:, p].reshape(n, -1)) for p in permutations])
    row, col = linear_sum_assignment(np.min(costs, axis=0))
    if not np.array_equal(row, np.arange(n)):
        raise ValueError("Assignment lost ordered target rows")
    corner = permutations[np.argmin(costs, axis=0)[row, col]].astype(np.int64)
    return validate_map(col.astype(np.int64), corner)

def to_target_slots(common_slots, face, corner):
    face, corner = validate_map(face, corner)
    a = np.asarray(common_slots)
    if a.shape[:2] != (len(face), 3):
        raise ValueError("Common slot shape differs")
    return a[face][np.arange(len(face))[:, None], corner].copy()

def to_noise_slots(target_slots, face, corner):
    face, corner = validate_map(face, corner)
    a = np.asarray(target_slots)
    if a.shape[:2] != (len(face), 3):
        raise ValueError("Target slot shape differs")
    inverse_face = np.argsort(face)
    inverse_corner = np.argsort(corner, axis=1)[inverse_face]
    return a[inverse_face][np.arange(len(face))[:, None], inverse_corner].copy()

def coupling_path(raw_noise, target_in_noise_slots, epsilon, t, lam=.25):
    if float(lam) != .25:
        raise ValueError("This experiment fixes HYBRID lambda=.25")
    z, y, eps = [np.asarray(a, np.float32) for a in (raw_noise, target_in_noise_slots, epsilon)]
    if z.shape != y.shape or z.shape != eps.shape or z.ndim != 3 or z.shape[1:] != (3, 3):
        raise ValueError("Mismatched free triangles")
    t = np.asarray(t, np.float32)
    if t.shape != () or not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Invalid scalar time")
    mixed = np.float32(np.sqrt(.75)) * z + np.float32(.5) * eps
    state = (np.float32(1) - t) * mixed + t * y
    velocity = y - mixed
    if not all(np.isfinite(a).all() for a in (z, y, eps, mixed, state, velocity)):
        raise ValueError("Nonfinite coupling")
    return mixed, state, velocity
