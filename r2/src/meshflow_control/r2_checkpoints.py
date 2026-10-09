"""R2-only adapters for the two retained historical checkpoint schemas."""
from pathlib import Path

from .r2_spec import RECIPE

SCHEMAS = {
    "chair_condition_recipe_factorial_training_v1": ("main", "training"),
    "chair_condition_recipe_factorial_generator_v1": ("main", "generator"),
    "chair_recipe_confirmation_training_v1": ("confirmation", "training"),
    "chair_recipe_confirmation_generator_v1": ("confirmation", "generator"),
}


def identify(value):
    identity = SCHEMAS.get(value.get("schema"))
    if identity is None or value.get("recipe") != RECIPE:
        raise ValueError("Expected an R2 training checkpoint or generator export")
    protocol = value.get("protocol", {})
    if (protocol.get("condition_schedule") != "MIX"
            or protocol.get("hybrid_schedule") != "all"
            or protocol.get("hybrid_lambda") != 0.25):
        raise ValueError("Checkpoint is not full-course MIX + HYBRID")
    return identity


def backend(stage):
    if stage == "main":
        from .training import recipe_factorial
        return recipe_factorial
    if stage == "confirmation":
        from .training import recipe_confirmation
        return recipe_confirmation
    raise ValueError("Unknown R2 stage")


def read(path, expected_sha256=None):
    from . import runtime  # Establish CUBLAS policy before torch/CUDA use.
    import torch
    from .data.io import digest
    path = Path(path).resolve()
    actual = digest(path)
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise ValueError("Checkpoint file SHA256 differs")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    stage, kind = identify(value)
    return value, stage, kind, actual


def inspect_checkpoint(path, expected_sha256=None):
    """CPU validation; no model forward, optimizer update, CUDA or new weights."""
    from . import runtime
    import torch
    value, stage, kind, sha = read(path, expected_sha256)
    module = backend(stage)
    if kind == "training":
        module.validate_payload(value)
        validation = "Full tensor/optimizer/RNG/stream hashes and scientific protocol"
    else:
        del value
        model, value = module.load_generator(path, device="cpu", expected_sha256=sha)
        validation = "Strict CPU generator load, tensor hash and scientific protocol"
        del model
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU inspection unexpectedly initialized CUDA")
    keys = ("schema", "recipe", "global_step", "generator_state_sha256",
            "optimizer_state_sha256", "rng_sha256", "stream_state_sha256",
            "protocol_sha256")
    return dict(stage=stage, kind=kind, file_sha256=sha,
                validation=validation, model_forwards=0, cuda_initialized=False,
                **{key: value[key] for key in keys if key in value})


def load_generator(path, device="cpu", expected_sha256=None):
    value, stage, kind, sha = read(path, expected_sha256)
    if kind != "generator":
        raise ValueError("Sampling requires the generator export, not an optimizer checkpoint")
    del value
    return backend(stage).load_generator(path, device=device, expected_sha256=sha)
