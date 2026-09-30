"""Strict full T1 checkpoint loading without adapters or historical imports."""
from .runtime import configure_stable_runtime
from pathlib import Path
import time
import torch
import yaml
from models.equidit import DiT
from models.utils import get_embedder
from .model import NativeInpaintingModel
from .artifacts import file_sha256, state_sha256

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/snet/base-120m-ot-v-chair.yaml"
DEFAULT_CHECKPOINT = REPO / "experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt"
T1_FILE_SHA256 = "7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634"
T1_STATE_SHA256 = "37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172"
T1_BACKBONE_SHA256 = "e342012659cf7b747b6426a500e496895ca2b334c056e2ddab280eb131a2ea0f"


def load_t1(checkpoint_path=DEFAULT_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Load the verified T1 step500 endpoint. Frozen by default; no EMA needed.

    This endpoint includes the entire fine-tuned backbone and role embedding.
    The YAML supplies model structure only: its general-purpose CFG/optimizer
    settings are not applied to the Native sampler or training API.
    """
    tick = time.perf_counter()
    configure_stable_runtime()
    checkpoint_path, config_path = Path(checkpoint_path), Path(config_path)
    actual_file_sha = file_sha256(checkpoint_path)
    if actual_file_sha != T1_FILE_SHA256:
        raise ValueError("Expected the verified Native_correct step500 checkpoint")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    if model_config.pop("model_type") != "equidit":
        raise ValueError("Expected official equidit configuration")
    required = dict(version=3, hidden_dim=768, num_layers=12, num_heads=12,
                    max_length=800, face_bin=20, pe_freq=20, use_qknorm=True,
                    use_rmsnorm=True, use_coord_encoding=True,
                    use_dit_like_pe=False, face_cond=True)
    if any(model_config.get(k) != v for k, v in required.items()):
        raise ValueError("T1 architecture configuration differs")
    if config["transport"]["prediction"] != "velocity":
        raise ValueError("T1 requires velocity prediction")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("schema"), payload.get("arm"), payload.get("completed_updates")) != (
            "native_connected_patch_v1", "Native_correct", 500):
        raise ValueError("T1 endpoint metadata differs")
    # Avoid random initialization and avoid loading an unused official EMA.
    with torch.device("meta"):
        model = NativeInpaintingModel(DiT(**model_config), trainable=False)
    # The official positional embedder closes over non-buffer constants.
    # Recreate these constants on CPU exactly as the audited original loader.
    model.backbone.x_embedder.embed_fn, _ = get_embedder(model_config["pe_freq"], input_dims=3)
    keys = model.load_state_dict(payload["model"], strict=True, assign=True)
    del payload
    model.set_trainable(False).eval().to(device)
    state = state_sha256(model)
    backbone = state_sha256(model.backbone)
    if state != T1_STATE_SHA256 or backbone != T1_BACKBONE_SHA256:
        raise RuntimeError("Loaded T1 tensor hashes differ")
    return model, dict(status="PASS", checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=actual_file_sha, config=str(config_path.resolve()),
        config_sha256=file_sha256(config_path), state_sha256=state,
        backbone_sha256=backbone, strict=True, missing_keys=list(keys.missing_keys),
        unexpected_keys=list(keys.unexpected_keys), official_EMA_loaded=False,
        checkpoint_schema="native_connected_patch_v1", completed_updates=500,
        role_shape=list(model.role_embedding.shape), role_nonzero=int(torch.count_nonzero(model.role_embedding)),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-tick)
