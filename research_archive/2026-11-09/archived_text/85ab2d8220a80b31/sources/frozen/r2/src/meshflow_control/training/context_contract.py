"""Frozen lightweight identity for contextual alignment; no model imports."""
from __future__ import annotations
import copy
NAMESPACE = "CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1"
HEAD_SEED = 17006844
HEAD_PARAMETERS = 3934720
ARMS = {"A": "A_FM", "B": "B_FREE_ALIGN", "C": "C_KNOWN_ALIGN"}
DOMAINS = {"A": None, "B": "free", "C": "known"}
HEAD_CONFIG = dict(dimension=768, heads=24, head_dimension=32, mlp_dimension=1024,
    layer_1based=4, layernorm_eps=1e-5, layernorm_affine=False,
    qkvo_bias=False, mlp_bias=True, dropout=0., query_init_std=.02,
    linear_init="xavier_uniform; MLP biases zero", seed=HEAD_SEED,
    head_dtype="float32", capture="backbone.layers[3].attn actual output before external gate",
    target_gradient=False, align_weight=.1, align_batch_denominator=4)


def validate_config(value):
    config = copy.deepcopy(value)
    arm = config.get("arm")
    if arm not in ARMS:
        raise ValueError("Contextual alignment arm must be A, B, or C")
    expected = dict(experiment=NAMESPACE, label=ARMS[arm], readout_mode="none",
        alignment_domain=DOMAINS[arm], head=HEAD_CONFIG,
        training=dict(updates=1000, effective_batch=8, microbatch=1, generator_lr=1e-5,
            aligner_lr=1e-4, betas=[.9, .95], weight_decay=0., generator_clip=1.,
            aligner_clip=1., save_steps=[500, 1000], alignment_weight=.1,
            alignment_denominator=4, update_observation_steps=[1, 100, 500, 1000],
            gradient_observation_steps=[1, 100, 1000]))
    for key, item in expected.items():
        config.setdefault(key, copy.deepcopy(item))
        if config[key] != item:
            raise ValueError("Frozen contextual configuration differs: " + key)
    if set(config) != {"arm", *expected}:
        raise ValueError("Unknown contextual configuration fields")
    return config


