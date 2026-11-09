"""Strict checkpoint identity for the fixed one-layer C-geometry pilot."""
from .runtime import configure_stable_runtime
from pathlib import Path
import time
import re
from collections.abc import Mapping
import torch
from models.equidit import DiT
from models.utils import get_embedder
from .artifacts import file_sha256, state_sha256
from .checkpoint import DEFAULT_CONFIG
from .model import NativeInpaintingModel
from .portable_checkpoint import _read_config, config_sha256, NATIVE_META
from .context_geometry import ContextGeometryEncoder, ENCODER_METADATA

SCHEMA = "native_t1_geom_context_v1"


def save_geometry_checkpoint(path, model, optimizer, stream, archive, *, base_identity, completed):
    path = Path(path)
    if path.exists() or path.with_suffix(".pt.partial").exists():
        raise FileExistsError(path)
    encoder = model.context_encoder
    if encoder is None or completed not in (250, 500):
        raise ValueError("Only registered geometry pilot checkpoints are allowed")
    payload = dict(schema=SCHEMA, native_meta=dict(NATIVE_META), context=encoder.metadata(),
        model=model.state_dict(), optimizer=optimizer.state_dict(), stream_state=stream.state_dict(),
        stream_seed=stream.seed, input_sha256=archive.sha256, base_identity=base_identity,
        config_sha256=config_sha256(DEFAULT_CONFIG), base_cumulative_updates=1500,
        completed_updates=completed, cumulative_updates=1500+completed, loss="masked_free_fm",
        num_faces=112, train_K=[4,8,12], optimizer_reset_at_start=True,
        optimizer_step_in_progress=False, batch_in_progress=False,
        model_state_sha256=state_sha256(model))
    temporary = path.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path),
                model_state_sha256=payload["model_state_sha256"], cumulative_updates=1500+completed,
                schema=SCHEMA, context=payload["context"])


def validate_geometry_checkpoint(payload):
    if payload.get("schema") != SCHEMA:
        raise ValueError("Expected native_t1_geom_context_v1")
    if payload.get("native_meta") != NATIVE_META:
        raise ValueError("Native metadata differs")
    context = payload.get("context", {})
    if any(context.get(key) != value for key,value in ENCODER_METADATA.items()):
        raise ValueError("Geometry feature/width/precision/injection definition differs")
    if context.get("context_mode") not in ("geo","graph") or context.get("initialization_seed") != 1010:
        raise ValueError("Geometry context mode/initialization differs")
    if (payload.get("base_cumulative_updates") != 1500 or payload.get("completed_updates") not in (250,500)
            or payload.get("cumulative_updates") != 1500+payload.get("completed_updates",0)):
        raise ValueError("Geometry pilot checkpoint progress differs")
    if (payload.get("loss") != "masked_free_fm" or payload.get("train_K") != [4,8,12]
            or payload.get("num_faces") != 112 or payload.get("optimizer_reset_at_start") is not True):
        raise ValueError("Training definition differs")
    if any(payload.get(key) is not False for key in ("optimizer_step_in_progress","batch_in_progress")):
        raise ValueError("Incomplete update checkpoint is not supported")
    if not isinstance(payload.get("base_identity"),dict) or payload["base_identity"].get("cumulative_updates") != 1500:
        raise ValueError("Common initialization identity missing")
    for key in ("optimizer","stream_state"):
        if not isinstance(payload.get(key),Mapping):
            raise ValueError("Geometry checkpoint requires "+key)
    if (payload.get("stream_seed") != 1010 or payload["stream_state"].get("seed") != 1010
            or payload["stream_state"].get("arm") != "Native_correct"
            or payload["stream_state"].get("batch_index") != payload["completed_updates"]
            or payload["stream_state"].get("sample_index") != 8*payload["completed_updates"]):
        raise ValueError("Registered new input stream identity/progress differs")
    hashes=[payload.get(k) for k in ("input_sha256","config_sha256","model_state_sha256")]
    hashes += [payload["base_identity"].get(k) for k in ("checkpoint_sha256","state_sha256")]
    if any(not isinstance(v,str) or re.fullmatch(r"[0-9a-f]{64}",v) is None for v in hashes):
        raise ValueError("Checkpoint hash identity is incomplete")
    return context


def load_geometry_checkpoint(path, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Full new schema only, strict keys/shapes/hash; frozen inference default."""
    tick=time.perf_counter()
    path,config_path=Path(path),Path(config_path)
    file_hash=file_sha256(path)
    config=_read_config(config_path)
    payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    context=validate_geometry_checkpoint(payload)
    if payload.get("config_sha256") != config_sha256(config_path):
        raise ValueError("Config identity differs")
    state=payload["model"]
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or v.device.type!="cpu" for v in state.values()):
        raise ValueError("Complete FP32 Native/geometry parameters required")
    configure_stable_runtime()
    with torch.device("meta"):
        model=NativeInpaintingModel(DiT(**config),trainable=False)
    model.context_encoder=ContextGeometryEncoder(context["context_mode"],seed=1010)
    model.backbone.x_embedder.embed_fn,_=get_embedder(config["pe_freq"],input_dims=3)
    keys=model.load_state_dict(state,strict=True,assign=True)
    model.set_trainable(False).eval()
    actual=state_sha256(model)
    if actual != payload.get("model_state_sha256"):
        raise ValueError("Full geometry model tensor hash differs")
    if sum(p.numel() for p in model.context_encoder.parameters()) != context["additional_parameters"]:
        raise ValueError("Encoder parameter count differs")
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint tensor")
    audit=dict(status="PASS",path=str(path.resolve()),sha256=file_hash,state_sha256=actual,
               cumulative_updates=payload["cumulative_updates"],schema=SCHEMA,context=context,
               base_identity=payload["base_identity"],strict=True,missing_keys=list(keys.missing_keys),
               unexpected_keys=list(keys.unexpected_keys),seconds=time.perf_counter()-tick)
    del payload,state
    model.to(device)
    return model,audit
