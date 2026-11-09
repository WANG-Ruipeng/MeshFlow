"""CPU-only exact nested-OT maps and common iid Gaussian slot coupling.

The protected implementation solves cdist(pre_OT_target, raw_noise[:, perm])
and returns noise[col_ind][best_perm].  Its target is multiplied by two only
AFTER matching.  We preserve those costs and recover the exact inverse map;
we never claim that target-ordered noise has an iid marginal.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

CONTRACT = 'original_preOT_cost_scale__FP32_final_target_x2__raw_iid_noise_slots_v1'


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_map(face, corner):
    face, corner = np.asarray(face), np.asarray(corner)
    n = len(face)
    if face.dtype != np.int64 or face.shape != (n,) or not np.array_equal(np.sort(face), np.arange(n)):
        raise ValueError('OT face mapping is not an int64 bijection')
    if corner.dtype != np.int64 or corner.shape != (n, 3) or not np.array_equal(np.sort(corner, axis=1), np.tile(np.arange(3), (n, 1))):
        raise ValueError('OT corner mapping is not an int64 bijection')
    return face, corner


def nested_ot_map(pre_target, raw_noise):
    """Exact protected arithmetic; face[i],corner[i,c] locate Z for target i,c."""
    data, noise = np.asarray(pre_target), np.asarray(raw_noise)
    if data.ndim != 3 or data.shape[1:] != (3, 3) or data.shape != noise.shape or not len(data):
        raise ValueError('Require matching nonempty [free_faces,3,3] arrays')
    if not np.isfinite(data).all() or not np.isfinite(noise).all():
        raise ValueError('Nonfinite OT inputs')
    n = data.shape[0]
    permutations = np.array(list(itertools.permutations([0, 1, 2])), dtype=np.int32)
    cost_matrices = []
    for perm in permutations:
        cost_matrices.append(cdist(data.reshape(n, -1), noise[:, perm].reshape(n, -1)))
    cost_matrices = np.stack(cost_matrices)
    cost_matrix = np.min(cost_matrices, axis=0)
    best_perm_indices = np.argmin(cost_matrices, axis=0)
    row, col = linear_sum_assignment(cost_matrix)
    if not np.array_equal(row, np.arange(n)):
        raise ValueError('Square assignment no longer has every ordered target row')
    corner = permutations[best_perm_indices[row, col]].astype(np.int64)
    return validate_map(col.astype(np.int64), corner)


def to_target_slots(common_slots, face, corner):
    """Gather original Z/common slots into the historical target slots."""
    face, corner = validate_map(face, corner)
    a = np.asarray(common_slots)
    if a.shape[:2] != (len(face), 3):
        raise ValueError('Common slot shape differs from OT map')
    return a[face][np.arange(len(face))[:, None], corner].copy()


def to_noise_slots(target_slots, face, corner):
    """Invert face AND within-face corner maps, retaining triangular grouping."""
    face, corner = validate_map(face, corner)
    a = np.asarray(target_slots)
    if a.shape[:2] != (len(face), 3):
        raise ValueError('Target slot shape differs from OT map')
    inverse_face = np.argsort(face)
    inverse_corner = np.argsort(corner, axis=1)[inverse_face]
    return a[inverse_face][np.arange(len(face))[:, None], inverse_corner].copy()


def coupling_path(raw_noise, target_in_noise_slots, epsilon, t, lam):
    """FP32 FM arithmetic; lambda=0 has an explicit bit-preserving branch."""
    if not 0 <= float(lam) <= 1:
        raise ValueError('Invalid variance mixture fraction')
    z = np.asarray(raw_noise, dtype=np.float32)
    y = np.asarray(target_in_noise_slots, dtype=np.float32)
    if z.shape != y.shape or z.ndim != 3 or z.shape[1:] != (3, 3):
        raise ValueError('Coupling requires aligned free triangle arrays')
    if float(lam) == 0:
        mixed = z.copy()  # epsilon deliberately not read or evaluated.
    else:
        eps = np.asarray(epsilon, dtype=np.float32)
        if eps.shape != z.shape:
            raise ValueError('Independent epsilon shape differs')
        mixed = eps.copy() if float(lam) == 1 else (
            np.float32(np.sqrt(1 - lam)) * z + np.float32(np.sqrt(lam)) * eps)
    time = np.asarray(t, dtype=np.float32)
    if time.shape != () or not np.isfinite(time) or not 0 <= time <= 1:
        raise ValueError('t must be a finite scalar in [0,1]')
    velocity = y - mixed
    state = (np.float32(1) - time) * mixed + time * y
    if not all(np.isfinite(a).all() for a in (z, y, mixed, velocity, state)):
        raise ValueError('Nonfinite coupling arrays')
    return mixed, state, velocity


def mapping_key(pre, noise, source_ids, corner_ids, known_ids, alpha, implementation_hash):
    data = dict(contract=CONTRACT, implementation_sha256=implementation_hash,
                alpha=float(alpha), pre_target=array_hash(pre), raw_noise=array_hash(noise),
                free_source_ids=array_hash(source_ids), source_corner_ids=array_hash(corner_ids),
                known_source_ids=array_hash(known_ids))
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), data


def to_old_slots(array, sample):
    """Restore an [N,9] or [N,3,3] tensor to the original full randomized slots.

    This is a permutation only. Used for CPU exact path/label tests and the
    separately budgeted real-model forward equivariance check.
    """
    original = np.asarray(array)
    n, k = int(sample['y']), int(sample['K'])
    if original.shape not in ((n, 9), (n, 3, 3)):
        raise ValueError('Expected one unpadded sample array')
    order = sample['face_permutation']
    known_first = original.reshape(n, 3, 3)[np.argsort(order)].copy()
    known_first[k:] = to_target_slots(known_first[k:], sample['ot_face_map'], sample['ot_corner_map'])
    return known_first[order].reshape(original.shape)


def old_slot_sample(sample):
    if sample['lambda'] != 0:
        raise ValueError('Historical baseline comparison requires lambda=0')
    result = dict(sample)
    for key in ('x0', 'x1', 'u', 'xt', 'context'):
        result[key] = to_old_slots(sample[key], sample)
    n, k = int(sample['y']), int(sample['K'])
    order = sample['face_permutation']
    ids = sample['source_face_ids'][np.argsort(order)].copy()
    corners = sample['corner_permutations'][np.argsort(order)].copy()
    ids[k:] = ids[k:][sample['ot_face_map']]
    corners[k:] = to_target_slots(corners[k:], sample['ot_face_map'], sample['ot_corner_map'])
    result['source_face_ids'], result['corner_permutations'] = ids[order], corners[order]
    return result


def map_velocity_old_to_common(sample, velocity):
    """Map one NumPy [N,9]/[N,3,3] historical prediction into common slots."""
    original = np.asarray(velocity)
    n, k = int(sample['y']), int(sample['K'])
    if original.shape not in ((n,9),(n,3,3)):
        raise ValueError('Expected one unpadded NumPy velocity')
    order = sample['face_permutation']
    first = original.reshape(n,3,3)[np.argsort(order)].copy()
    first[k:] = to_noise_slots(first[k:],sample['ot_face_map'],sample['ot_corner_map'])
    return first[order].reshape(original.shape)
