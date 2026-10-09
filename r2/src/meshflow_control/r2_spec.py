"""Public R2 recipe and retained identities; standard library only."""
from copy import deepcopy

RECIPE = "R2_MIX_H_ALL"
SPEC = {
    "recipe": RECIPE,
    "model": "Native T1 + zero-initialized role + C-only Geo (seed 1010)",
    "initialization": "official Chair EMA; fresh AdamW; not START continuation",
    "total_updates": 6000,
    "batch": 8,
    "microbatch": 1,
    "distinct_parents_per_batch": 8,
    "conditions_per_batch": {"20_percent_clean": 2, "20_percent_width_augmented": 2,
                             "40_percent_clean": 2, "40_percent_width_augmented": 2},
    "width_augmentation": [0.9, 1.1],
    "hybrid_lambda": 0.25,
    "hybrid_schedule": "all",
    "matching": "free-only nested OT before HYBRID mixing; no rematching",
    "loss": "free-coordinate FM / whole effective batch valid free scalar count",
    "optimizer": {"name": "AdamW", "lr": 1e-5, "betas": [0.9, 0.95],
                  "weight_decay": 0.0, "global_clip": 1.0},
    "precision": "BF16 forward; FP32 parameters, moments and integration",
    "readout": "none",
    "teacher": False,
    "auxiliary_loss": False,
    "new_EMA": False,
    "sampling": "standard Gaussian; Euler50; C hard-preserved at all 51 states",
    "total_faces": [128, 256],
    "data_contract": "historical 32-parent/453-task import; R2 uses 26 parents/411 tasks",
    "limitations": [
        "20%/40% count faces, not surface area; historical selector rules differ",
        "new parent pools and new condition construction require a new data schema",
        "resume continues an interrupted registered 6000-update run, not a new post-6000 study",
        "recipe selection does not choose between the main and confirmation endpoints",
    ],
}

RETAINED = {
    "main": {
        "generator_sha256": "6770a575e03e7845a7b0f491d201dc61c3fc42a254fa949b84c498eb8ea149e8",
        "training_sha256": "89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541",
    },
    "confirmation": {
        "generator_sha256": "9dc93885043121ea82287a105cb9ad3ab373827ce4c3b41802fcfe3b746f59a8",
        "training_sha256": "df9ddfe8a9735180cb79b4bc4193d633ea8f0e1e990f8f186ccf07399f351939",
    },
}


def describe():
    return deepcopy(SPEC)
