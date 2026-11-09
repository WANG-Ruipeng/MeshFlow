"""Read-only transport of static source quality through the saved TRAIN47 OT.

No new matching, noise, augmentation, model construction or geometry scoring.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from r2.pilots.seam_slot_c200_v1 import trainer_support as prior

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('CLEAN_BULK_FM', 'SURGICAL_LATE4')
UPDATES, BASE_STEP, BATCH = 1000, 13000, 8
DIRTY_WEIGHT = .2
resolve, read = prior.resolve, prior.read


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def require_cleanup(root):
    receipt = resolve(root) / 'cleanup/completion.json'
    if not receipt.exists() or not str(read(receipt).get('status', '')).startswith('COMPLETE'):
        raise RuntimeError('Stage 0 cleanup must complete before computation')


@lru_cache(maxsize=64)
def source_arrays(path, sha):
    if digest(path) != sha:
        raise ValueError('Source array changed: ' + str(path))
    with np.load(resolve(path), allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in handle.files}


def label_sample(sample, labels):
    """Source face IDs already include OT+slot permutation; corner weights agree."""
    n = sample['N']
    ids = np.asarray(sample['source_face_ids'], np.int64)
    if ids.shape != (n,) or not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError('Saved slot/source face bijection changed')
    if n != labels['N'] or sample['uid'] != labels['parent_id']:
        raise ValueError('Parent/N label identity mismatch')
    C_rows = np.flatnonzero(sample['known_mask'])
    if sorted(map(int, ids[C_rows])) != labels['known_source_ids']:
        raise ValueError('Original C source identities changed')
    seam = np.asarray(sample['seam_third_corner_mask'], bool)
    seam_faces = seam.any(1)
    if sorted(map(int, ids[seam_faces])) != labels['owner_source_ids']:
        raise ValueError('Source owner identity differs from static quality labels')
    corrupt = np.isin(ids, np.asarray(labels['corrupted_source_ids'], np.int64))
    free = sample['free_corner_mask'] & sample['valid_mask'][:, None]
    bulk = free & ~seam_faces[:, None]
    dirty = bulk & corrupt[:, None]
    clean = bulk & ~corrupt[:, None]
    if (seam & ~free).any() or not np.array_equal(seam | dirty | clean, free):
        raise ValueError('Seam/clean-bulk/dirty-bulk must partition free corners')
    weights = np.ones((n, 3), np.float32)
    weights[dirty] = np.float32(DIRTY_WEIGHT)
    weights[~free] = 0
    if np.any(weights[seam] != 1) or np.any(weights[~free] != 0):
        raise ValueError('Owner third points must stay unit weight; anchors excluded')
    sample.update(quality_weight=weights, quality_dirty_bulk_mask=dirty,
                  quality_clean_bulk_mask=clean, quality_seam_mask=seam,
                  corrupted_source_face_mask=corrupt,
                  weighted_free_coordinate_count=float(weights.astype(np.float64).sum() * 3))
    return sample


def load_sample(record, root, arm, step, micro, labels_by_parent):
    if arm not in ARMS:
        raise ValueError('Unregistered training arm')
    sample = prior.load_sample(record, root, 'SEAM_SLOT_C200', step, micro, digest)
    sample['continuation_arm'] = arm
    return label_sample(sample, labels_by_parent[sample['uid']])


def batch_denominators(samples):
    if len(samples) != BATCH or len({s['continuation_arm'] for s in samples}) != 1:
        raise ValueError('One registered arm and exactly eight examples required')
    plain = sum(int(s['free_coordinate_count']) for s in samples)
    weighted = sum(float(s['weighted_free_coordinate_count']) for s in samples)
    if not 0 < weighted <= plain:
        raise ValueError('Invalid whole-batch weighted scalar denominator')
    return plain, weighted


def diagnostic(sample, prediction, bins):
    out = prior.diagnostic(sample, prediction, bins)
    squared = (np.asarray(prediction, np.float64).reshape(sample['N'], 3, 3)
               - sample['u'].astype(np.float64)) ** 2
    for name, key in (('quality_clean_bulk', 'quality_clean_bulk_mask'),
                      ('quality_dirty_bulk', 'quality_dirty_bulk_mask'),
                      ('quality_seam_third', 'quality_seam_mask')):
        mask = sample[key]
        count = int(mask.sum()) * 3
        sse = float(squared[mask].sum())
        out.update({name + '_scalar_count': count, name + '_velocity_sse': sse,
                    name + '_velocity_mse': sse / count if count else None,
                    name + '_status': 'PASS' if count else 'NA_NO_COORDINATES'})
    out['quality_weighted_velocity_sse'] = float((squared * sample['quality_weight'][..., None]).sum())
    if not np.isclose(sum(out[key + '_velocity_sse'] for key in
                         ('quality_clean_bulk', 'quality_dirty_bulk', 'quality_seam_third')),
                      out['all_free_velocity_sse'], atol=1e-8, rtol=1e-12):
        raise ValueError('Unweighted quality partitions do not sum to original FM')
    return out
