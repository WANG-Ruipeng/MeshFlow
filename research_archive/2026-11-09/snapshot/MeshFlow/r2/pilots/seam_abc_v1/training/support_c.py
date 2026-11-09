"""C-only teacher-reference adapter to the unchanged shared 6400 OT stream."""
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import trainer_support as prior
from .support import parent_arrays
from ..shared_tip import prepare_training

ROOT = Path(__file__).resolve().parents[4]
ARMS = ('C_SHARED_TIP',)
BASE_STEP, HEAD_BASE_STEP, UPDATES = 12200, 200, 800
L = prior.L
resolve, read, diagnostic = prior.resolve, prior.read, prior.diagnostic


def load_sample(record, root, arm, step, micro, digest):
    if arm not in ARMS: raise ValueError('Only C_SHARED_TIP accepted')
    old = prior.load_sample(record, root, 'SEAM_SLOT_C200', step, micro, digest)
    canonical, source = parent_arrays(record['parent_npz'], record['parent_sha256'])
    sample = prepare_training(old, canonical)
    sample['continuation_arm'] = arm
    restored = np.take_along_axis(canonical[sample['source_face_ids'][:, None], sample['source_corner_permutations']],
                                 sample['source_winding_permutation'], 1)
    if not np.array_equal(restored, source[sample['source_face_ids']]):
        raise ValueError('C alias source transport/winding mismatch')
    return sample


def validate_batch(samples):
    prior.validate_batch(samples)
    if {s['continuation_arm'] for s in samples} != {'C_SHARED_TIP'}:
        raise ValueError('Mixed reference training batch')
    return sum(s['train_free_coordinate_count'] for s in samples), sum(len(s['occupied_labels']) for s in samples)
