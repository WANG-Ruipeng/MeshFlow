"""ABC-only stream adapter; prior data and contracts remain read-only."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from meshflow_control.data.io import digest
from r2.pilots.seam_slot_c200_v1 import trainer_support as prior
from ..coupling import build_pairs

ROOT = Path(__file__).resolve().parents[4]
ARMS = ("A_COUPLE", "B_POOL47")
BASE_STEP, HEAD_BASE_STEP, UPDATES = 12200, 200, 800
L = prior.L
resolve, read, diagnostic = prior.resolve, prior.read, prior.diagnostic


@lru_cache(maxsize=128)
def parent_arrays(path, sha):
    p = resolve(path)
    if digest(p) != sha:
        raise ValueError("Registered source parent bytes changed")
    with np.load(p, allow_pickle=False) as z:
        return z["source_face_vertex_ids"].copy(), z["source_faces"].copy()


def load_sample(record, root, arm, step, micro, digest):
    if arm not in ARMS:
        raise ValueError("Unregistered A/B arm")
    sample = prior.load_sample(record, root, "SEAM_SLOT_C200", step, micro, digest)
    if arm == "A_COUPLE":
        canonical, source = parent_arrays(record["parent_npz"], record["parent_sha256"])
        sample["coupling_pairs"] = build_pairs(sample, canonical, source)
    sample["continuation_arm"] = arm
    return sample


def validate_batch(samples):
    arms = {s["continuation_arm"] for s in samples}
    if len(arms) != 1 or len(samples) != 8:
        raise ValueError("One registered arm and eight microbatches required")
    if arms == {"A_COUPLE"}:
        return prior.validate_batch(samples)
    if arms != {"B_POOL47"}:
        raise ValueError("Unknown A/B batch")
    # B expands the balanced parent stream; the head still uses the inherited
    # fixed class weights, with a sum divided by the actual batch edge count.
    return sum(int(s["free_coordinate_count"]) for s in samples), sum(len(s["occupied_labels"]) for s in samples)
