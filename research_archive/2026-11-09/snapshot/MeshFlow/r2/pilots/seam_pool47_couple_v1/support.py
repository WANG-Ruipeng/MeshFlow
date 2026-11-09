"""Shared TRAIN47 loader; old data and coupling definitions are read-only."""
from pathlib import Path
from r2.pilots.seam_abc_v1.training import support as old
from r2.pilots.seam_slot_c200_v1 import trainer_support as prior
from r2.pilots.seam_abc_v1.coupling import build_pairs

ROOT=Path(__file__).resolve().parents[3]
ARMS=('PLAIN3000','COUPLE3000')
BASE_STEP,HEAD_BASE_STEP,UPDATES=13000,1000,3000
resolve,read,diagnostic=prior.resolve,prior.read,prior.diagnostic
L=prior.L

def load_sample(record,root,arm,step,micro,digest):
    if arm not in ARMS:raise ValueError('Unknown arm')
    sample=prior.load_sample(record,root,'SEAM_SLOT_C200',step,micro,digest)
    # Both arms carry identical transported correspondence metadata; only the
    # COUPLE arm contributes its existing A loss to backward.
    canonical,source=old.parent_arrays(record['parent_npz'],record['parent_sha256'])
    sample['coupling_pairs']=build_pairs(sample,canonical,source)
    sample['continuation_arm']=arm
    return sample

def validate_batch(samples):
    if len(samples)!=8 or len({s['continuation_arm'] for s in samples})!=1:
        raise ValueError('Exactly eight samples from one arm required')
    return sum(int(s['free_coordinate_count']) for s in samples),sum(len(s['occupied_labels']) for s in samples)
