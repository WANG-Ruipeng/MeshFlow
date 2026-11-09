"""Shared continuation stream and unchanged SEAM contract validation."""
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import trainer_support as prior

ROOT = Path(__file__).resolve().parents[4]
ARMS = ('LONG', 'ANTI_COVER', 'LOCAL_NOISE')
BASE_STEP = 12200
HEAD_BASE_STEP = 200
UPDATES = 800
L = prior.L
resolve, read = prior.resolve, prior.read
validate_batch, diagnostic = prior.validate_batch, prior.diagnostic


def load_sample(record, root, arm, step, micro, digest):
    if arm not in ARMS:
        raise ValueError('Unregistered continuation arm')
    sample = prior.load_sample(record, root, 'SEAM_SLOT_C200', step, micro, digest)
    if arm == 'LOCAL_NOISE':
        fixed = sample['anchor_ids'] >= 0
        slots = np.flatnonzero(~sample['known_mask'] & fixed[:, 0] & fixed[:, 1] & ~fixed[:, 2])
        before = {k: sample[k].copy() for k in ('x0', 'x1', 'xt', 'u')}
        midpoint = sample['C_vertices'][sample['anchor_ids'][slots, :2]].mean(axis=1)
        sample['x0'][slots, 2] += midpoint
        sample['u'] = sample['x1'] - sample['x0']
        t = np.float32(sample['t'])
        sample['xt'] = (np.float32(1) - t) * sample['x0'] + t * sample['x1']
        sample['xt'][fixed] = sample['x0'][fixed]
        untouched = np.ones((sample['N'], 3), bool)
        untouched[slots, 2] = False
        for key in ('x0', 'x1', 'xt', 'u'):
            if not np.array_equal(sample[key][untouched], before[key][untouched]):
                raise ValueError('Local-noise transform changed a non-tip coordinate')
        sample['local_noise_midpoint_shift'] = midpoint
    sample['continuation_arm'] = arm
    return sample
