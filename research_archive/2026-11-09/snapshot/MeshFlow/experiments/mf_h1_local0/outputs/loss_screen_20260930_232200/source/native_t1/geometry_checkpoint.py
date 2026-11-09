"""Portable Geo-only checkpoints, with explicit legacy Geo compatibility.

No checkpoint, data path, update budget, or initialization seed is implied by
this public format. Graph checkpoints are rejected before model construction.
"""
from .runtime import configure_stable_runtime
from collections.abc import Mapping
from pathlib import Path
import re
import time

import torch
from models.equidit import DiT
from models.utils import get_embedder
from .artifacts import file_sha256, state_sha256
from .checkpoint import DEFAULT_CONFIG
from .model import NativeInpaintingModel
from .portable_checkpoint import _read_config, config_sha256, NATIVE_META
from .context_geometry import ContextGeometryEncoder, ENCODER_METADATA

SCHEMA = "native_t1_geo_training_v1"
LEGACY_SCHEMA = "native_t1_geom_context_v1"
ENCODER_PARAMETER_COUNT = 149888

# Actual historical metadata, retained without translating or guessing keys.
LEGACY_ENCODER_METADATA = dict(
    schema=LEGACY_SCHEMA, feature_dim=13, hidden_dim=128, graph_layers=1,
    branch_dtype="torch.float32", output_dim=768,
    injection="after_coordinate_embedding_and_role_before_blocks",
    geometry="centroid_sorted_edges_area_unsigned_normal_outer_six",
    graph="exact_FP32_shared_whole_edge_binary_row_normalize_A_plus_I",
    area_reject_le=1e-12, layer_norm_eps=1e-5, initialization_seed=1010)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(name + " must be a lowercase SHA256")


def validate_geometry_checkpoint(payload, expected_config_sha256=None):
    """Validate the public or explicitly historical Geo identity, never Graph.

    Full key/shape, finite-tensor, and actual state-hash checks additionally run
    in the loader. Input RNG may start after an earlier training segment; this
    segment's progress is recorded separately from cumulative model updates.
    """
    if not isinstance(payload, Mapping) or payload.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Expected portable Geo or explicit legacy Geo schema")
    legacy = payload["schema"] == LEGACY_SCHEMA
    if payload.get("native_meta") != NATIVE_META:
        raise ValueError("Native metadata differs")
    context = payload.get("context")
    if not isinstance(context, Mapping) or context.get("context_mode") != "geo":
        raise ValueError("Only Geo checkpoints are supported; Graph is not supported")
    seed = _integer(context.get("initialization_seed"), "context initialization_seed")
    definition = LEGACY_ENCODER_METADATA if legacy else ENCODER_METADATA
    for key, expected in definition.items():
        if key == "initialization_seed" and not legacy:
            continue
        if context.get(key) != expected:
            raise ValueError("Geometry definition differs: " + key)
    if (type(context.get("additional_parameters")) is not int
            or context["additional_parameters"] != ENCODER_PARAMETER_COUNT):
        raise ValueError("Geometry parameter count differs")
    if seed >= 2**63:
        raise ValueError("Context seed is outside the supported range")
    for key in ("model", "optimizer", "stream_state", "base_identity"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError("Checkpoint requires " + key + " mapping")
    state = payload["model"]
    if not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                        or v.dtype != torch.float32 or v.device.type == "meta"
                        for k, v in state.items()):
        raise ValueError("A complete FP32 parameter state is required")
    base = _integer(payload.get("base_cumulative_updates"), "base_cumulative_updates")
    completed = _integer(payload.get("completed_updates"), "completed_updates")
    cumulative = _integer(payload.get("cumulative_updates"), "cumulative_updates")
    if cumulative != base + completed:
        raise ValueError("Cumulative update accounting differs")
    base_identity = payload["base_identity"]
    if "cumulative_updates" in base_identity:
        if _integer(base_identity["cumulative_updates"], "base identity cumulative_updates") != base:
            raise ValueError("Base checkpoint progress differs from the saved segment")
    elif legacy:
        raise ValueError("Legacy base checkpoint progress is missing")
    if (payload.get("loss") != "masked_free_fm" or payload.get("train_K") != [4, 8, 12]
            or type(payload.get("num_faces")) is not int or payload["num_faces"] != 112
            or payload.get("optimizer_reset_at_start") is not True):
        raise ValueError("Pure FM training definition differs")
    if any(payload.get(k) is not False for k in ("optimizer_step_in_progress", "batch_in_progress")):
        raise ValueError("Checkpoint must be saved at a completed update boundary")
    stream_seed = _integer(payload.get("stream_seed"), "stream_seed")
    stream = payload["stream_state"]
    if (_integer(stream.get("seed"), "stream seed") != stream_seed
            or stream.get("arm") != "Native_correct"):
        raise ValueError("Input stream identity differs")
    batch = _integer(stream.get("batch_index"), "stream batch_index")
    sample = _integer(stream.get("sample_index"), "stream sample_index")
    if sample != 8 * batch or batch < completed:
        raise ValueError("Input stream progress differs")
    if legacy:
        if base != 1500 or completed not in (250, 500) or stream_seed != 1010 or batch != completed:
            raise ValueError("Explicit legacy Geo training identity differs")
    else:
        start_batch = _integer(payload.get("stream_start_batch_index"), "stream_start_batch_index")
        start_sample = _integer(payload.get("stream_start_sample_index"), "stream_start_sample_index")
        if start_batch + completed != batch or start_sample != 8 * start_batch:
            raise ValueError("Recorded input-stream segment start differs")
    for key in ("input_sha256", "config_sha256", "model_state_sha256"):
        _sha(payload.get(key), key)
    for key in ("checkpoint_sha256", "state_sha256"):
        _sha(base_identity.get(key), "base_identity." + key)
    if expected_config_sha256 is not None and payload["config_sha256"] != expected_config_sha256:
        raise ValueError("Config identity differs")
    return dict(context)


def save_geometry_checkpoint(path, model, optimizer, stream, archive, *,
                             config_path, base_identity, completed, base_updates=None):
    """Save one completed portable Geo segment; never infer a training budget."""
    path = Path(path)
    temporary = path.with_suffix(".pt.partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    if not isinstance(base_identity, Mapping):
        raise ValueError("Base checkpoint identity must be supplied")
    base = _integer(base_identity.get("cumulative_updates") if base_updates is None else base_updates,
                    "base_cumulative_updates")
    completed = _integer(completed, "completed_updates")
    encoder = getattr(model, "context_encoder", None)
    if encoder is None or encoder.metadata().get("context_mode") != "geo":
        raise ValueError("Only an attached Geo encoder can use this checkpoint writer")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Geo training requires a freshly constructed AdamW")
    parameters = list(model.parameters())
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}:
        raise ValueError("Optimizer must own every Geo model parameter exactly once")
    if any((g["lr"], tuple(g["betas"]), g["weight_decay"]) != (1e-5, (.9, .95), 0.)
           for g in optimizer.param_groups):
        raise ValueError("Geo AdamW settings differ")
    _read_config(Path(config_path))
    stream_state = stream.state_dict()
    start_batch = _integer(stream_state.get("batch_index"), "stream batch_index") - completed
    payload = dict(schema=SCHEMA, native_meta=dict(NATIVE_META), context=encoder.metadata(),
        model=model.state_dict(), optimizer=optimizer.state_dict(), stream_state=stream_state,
        stream_seed=stream.seed, stream_start_batch_index=start_batch,
        stream_start_sample_index=8*start_batch, input_sha256=archive.sha256,
        base_identity=dict(base_identity), config_sha256=config_sha256(config_path),
        base_cumulative_updates=base, completed_updates=completed, cumulative_updates=base+completed,
        loss="masked_free_fm", num_faces=112, train_K=[4,8,12], optimizer_reset_at_start=True,
        optimizer_step_in_progress=False, batch_in_progress=False,
        model_state_sha256=state_sha256(model))
    validate_geometry_checkpoint(payload, config_sha256(config_path))
    if any(not bool(torch.isfinite(v).all()) for v in payload["model"].values()):
        raise ValueError("Nonfinite model cannot be saved")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest,
                checkpoint_sha256=digest, model_state_sha256=payload["model_state_sha256"],
                cumulative_updates=base+completed, completed_updates=completed,
                schema=SCHEMA, context=payload["context"])


def load_geometry_checkpoint(path, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Strict FP32 Geo load, including explicitly validated old Geo checkpoints."""
    tick = time.perf_counter()
    path, config_path = Path(path), Path(config_path)
    digest = file_sha256(path)
    config = _read_config(config_path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    context = validate_geometry_checkpoint(payload, config_sha256(config_path))
    state = payload["model"]
    if any(v.device.type != "cpu" for v in state.values()):
        raise ValueError("Loading requires CPU FP32 checkpoint tensors")
    configure_stable_runtime()
    with torch.device("meta"):
        model = NativeInpaintingModel(DiT(**config), trainable=False)
    model.context_encoder = ContextGeometryEncoder("geo", seed=context["initialization_seed"])
    model.backbone.x_embedder.embed_fn, _ = get_embedder(config["pe_freq"], input_dims=3)
    keys = model.load_state_dict(state, strict=True, assign=True)
    model.set_trainable(False).eval()
    actual = state_sha256(model)
    if actual != payload["model_state_sha256"]:
        raise ValueError("Full Geo model tensor hash differs")
    if sum(p.numel() for p in model.context_encoder.parameters()) != context["additional_parameters"]:
        raise ValueError("Encoder parameter count differs")
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint tensor")
    audit = dict(status="PASS", path=str(path.resolve()), checkpoint=str(path.resolve()),
        sha256=digest, checkpoint_sha256=digest, state_sha256=actual,
        completed_updates=payload["completed_updates"], base_cumulative_updates=payload["base_cumulative_updates"],
        cumulative_updates=payload["cumulative_updates"], schema=payload["schema"],
        checkpoint_schema=payload["schema"], context=context, base_identity=dict(payload["base_identity"]),
        input_sha256=payload["input_sha256"], stream_seed=payload["stream_seed"],
        config_sha256=config_sha256(config_path), config_file_sha256=file_sha256(config_path),
        strict=True, missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys),
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
        legacy_schema=payload["schema"] == LEGACY_SCHEMA, optimizer_restored=False)
    del payload, state
    model.to(device)
    audit["seconds"] = time.perf_counter()-tick
    return model, audit
