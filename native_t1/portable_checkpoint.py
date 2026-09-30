"""Portable, strict Native T1 loaders; checkpoints are always external assets.

Official initialization accepts only the audited chair EMA. Trained endpoints
must carry the portable schema and Native identity; arbitrary state dictionaries
and historical experiment schemas are deliberately not inferred.
"""
from .runtime import configure_stable_runtime

from collections.abc import Mapping
import hashlib
from pathlib import Path
import re
import time

import torch
import yaml
from models.equidit import DiT
from models.utils import get_embedder

from .artifacts import file_sha256, state_sha256
from .checkpoint import DEFAULT_CONFIG
from .model import NativeInpaintingModel

OFFICIAL_FILE_SHA256 = "bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d"
OFFICIAL_BACKBONE_SHA256 = "b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222"
OFFICIAL_NATIVE_SHA256 = "89783df7227019c4d1dd42d3190922c5eda2555acddfc6de023c617b0279410b"
OFFICIAL_CONFIG_SHA256 = "fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f"
TRAINING_SCHEMA = "native_t1_training_v1"
NATIVE_META = {"architecture": "native_t1", "num_faces": 112,
               "role_shape": [2, 768], "known_time": 1.0, "prediction": "velocity"}


def config_sha256(path):
    """Canonical UTF-8 YAML identity; LF and CRLF clones have the same hash."""
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_training_metadata(payload, config_sha256=None):
    """Check identity and completed-update accounting without loading a model.

    File identity for user-created checkpoints is reported, not pinned to a
    private endpoint. The embedded tensor hash and supplied YAML hash are checked
    separately by ``load_trained_checkpoint``.
    """
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError("Expected native_t1_training_v1 checkpoint schema")
    meta = payload.get("native_meta")
    if not isinstance(meta, Mapping) or any(meta.get(k) != v for k, v in NATIVE_META.items()):
        raise ValueError("Checkpoint Native T1 identity differs")
    for name in ("model", "optimizer", "stream_state"):
        if not isinstance(payload.get(name), Mapping):
            raise ValueError(f"Checkpoint requires {name} mapping")
    for name in ("completed_updates", "base_cumulative_updates", "cumulative_updates", "stream_seed"):
        value = payload.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {name}")
    if payload["cumulative_updates"] != payload["base_cumulative_updates"] + payload["completed_updates"]:
        raise ValueError("Cumulative update accounting differs")
    if (payload.get("loss") != "masked_free_fm" or payload.get("num_faces") != 112
            or payload.get("train_K") != [4, 8, 12]
            or payload.get("optimizer_reset_at_start") is not True):
        raise ValueError("Portable training definition differs")
    if any(payload.get(k) is not False for k in ("optimizer_step_in_progress", "batch_in_progress")):
        raise ValueError("Checkpoint was not saved at a completed update boundary")
    for name in ("config_sha256", "model_state_sha256", "input_sha256"):
        _sha(payload.get(name), name)
    if config_sha256 is not None and payload["config_sha256"] != config_sha256:
        raise ValueError("Checkpoint configuration SHA256 differs")
    return {"schema": TRAINING_SCHEMA, "native_meta": dict(meta),
            "completed_updates": payload["completed_updates"],
            "base_cumulative_updates": payload["base_cumulative_updates"],
            "cumulative_updates": payload["cumulative_updates"],
            "stream_seed": payload["stream_seed"], "input_sha256": payload["input_sha256"]}


def _read_config(path):
    if config_sha256(path) != OFFICIAL_CONFIG_SHA256:
        raise ValueError("Official chair configuration canonical SHA256 differs")
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    model = dict(config["model"])
    required = dict(model_type="equidit", version=3, hidden_dim=768, num_layers=12,
                    num_heads=12, max_length=800, face_bin=20, pe_freq=20,
                    use_qknorm=True, use_rmsnorm=True, use_coord_encoding=True,
                    use_dit_like_pe=False, face_cond=True, gradient_checkpointing=False)
    if any(model.get(k) != v for k, v in required.items()):
        raise ValueError("Expected official chair Native T1 architecture configuration")
    if config.get("transport", {}).get("prediction") != "velocity":
        raise ValueError("Native T1 requires velocity prediction")
    model.pop("model_type")
    return model


def _ema_state(payload):
    if not isinstance(payload, Mapping) or not isinstance(payload.get("ema"), Mapping):
        raise ValueError("Official checkpoint must contain the EMA mapping")
    state = payload["ema"]
    if not state or not all(isinstance(k, str) for k in state):
        raise ValueError("Official EMA state is empty or has invalid keys")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed official EMA module prefixes")
    return {key[7:] if all(prefixed) else key: value for key, value in state.items()}


def _strict_native(state, config, expected_state, *, expected_backbone=None):
    if not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                        or v.device.type != "cpu" or v.dtype != torch.float32
                        for k, v in state.items()):
        raise ValueError("Native state must contain CPU FP32 tensors only")
    configure_stable_runtime()
    # Meta construction performs no random CPU/CUDA parameter initialization.
    with torch.device("meta"):
        model = NativeInpaintingModel(DiT(**config), trainable=False)
    # Official positional encoding captures unregistered constants in a closure.
    # Recreate them on CPU before assigning the complete parameter state.
    model.backbone.x_embedder.embed_fn, _ = get_embedder(config["pe_freq"], input_dims=3)
    keys = model.load_state_dict(state, strict=True, assign=True)
    model.set_trainable(False).eval()
    actual = state_sha256(model)
    backbone = state_sha256(model.backbone)
    if actual != expected_state:
        raise ValueError("Native model_state_sha256 differs from loaded tensors")
    if expected_backbone is not None and backbone != expected_backbone:
        raise ValueError("Official EMA backbone tensor identity differs")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint tensor")
    return model, dict(state_sha256=actual, backbone_sha256=backbone, strict=True,
                      missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys),
                      parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=0, role_shape=list(model.role_embedding.shape),
                      role_nonzero=int(torch.count_nonzero(model.role_embedding)),
                      parameter_dtype="torch.float32", eval_mode=True)


def load_official_initialization(checkpoint_path, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Strictly initialize from the fixed official chair EMA and zero Native roles.

    No model/optimizer fallback exists. The returned model is frozen and in eval
    mode; an explicit training caller may call ``set_trainable(True)``.
    """
    tick = time.perf_counter()
    path, config_path = Path(checkpoint_path), Path(config_path)
    actual_file = file_sha256(path)
    if actual_file != OFFICIAL_FILE_SHA256:
        raise ValueError("Expected the verified official chair EMA checkpoint file")
    config = _read_config(config_path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ema = _ema_state(payload)
    state = {"backbone." + key: value for key, value in ema.items()}
    state["role_embedding"] = torch.zeros((2, 768), dtype=torch.float32)
    model, audit = _strict_native(state, config, OFFICIAL_NATIVE_SHA256,
                                  expected_backbone=OFFICIAL_BACKBONE_SHA256)
    del payload, ema, state
    model.to(device)
    audit.update(status="PASS", checkpoint=str(path.resolve()), checkpoint_sha256=actual_file,
                 config=str(config_path.resolve()), config_sha256=config_sha256(config_path),
                 config_file_sha256=file_sha256(config_path),
                 checkpoint_schema="official_chair_ema", official_EMA_loaded=True,
                 selected_state="ema", role_initialization="exact_zero",
                 cumulative_updates=0, completed_updates=0, device=str(device),
                 seconds=time.perf_counter() - tick)
    return model, audit


def load_trained_checkpoint(checkpoint_path, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Load an explicitly identified portable full Native T1 training checkpoint.

    Optimizer and stream metadata are checked/preserved in the file, but this
    inference loader does not instantiate an optimizer or restore random state.
    It does not accept adapter-only or historical experiment checkpoints.
    """
    tick = time.perf_counter()
    path, config_path = Path(checkpoint_path), Path(config_path)
    config = _read_config(config_path)
    config_sha = config_sha256(config_path)
    actual_file = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    identity = validate_training_metadata(payload, config_sha)
    model, audit = _strict_native(payload["model"], config, payload["model_state_sha256"])
    del payload
    model.to(device)
    audit.update(identity)
    audit.update(status="PASS", checkpoint=str(path.resolve()), checkpoint_sha256=actual_file,
                 config=str(config_path.resolve()), config_sha256=config_sha,
                 config_file_sha256=file_sha256(config_path),
                 checkpoint_schema=TRAINING_SCHEMA, official_EMA_loaded=False,
                 selected_state="model", device=str(device), seconds=time.perf_counter() - tick)
    return model, audit
