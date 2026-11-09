"""Pinned working baseline: pure-FM A_continue, cumulative 1500 updates.

Historical checkpoint.load_t1 remains pinned to the original step500.
Importing this module performs no checkpoint IO, CUDA initialization or training.
"""
from .runtime import configure_stable_runtime
from pathlib import Path
import time
import torch
import yaml
from models.equidit import DiT
from models.utils import get_embedder
from .model import NativeInpaintingModel
from .artifacts import file_sha256, state_sha256
from .checkpoint import DEFAULT_CONFIG, REPO

BASELINE_CHECKPOINT = REPO / "experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_continue/additional_step500_cumulative_step1500.pt"
BASELINE_FILE_SHA256 = "d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400"
BASELINE_STATE_SHA256 = "f162c79cc0d0d8dbd3ff208fecc6f433f348096d950652678a06cfaa328cc90b"
BASELINE_BACKBONE_SHA256 = "25d5414e6a3af1cbf1f0f71f27a68010f28318578aaf092af150d80b00e4a1a2"
BASELINE_CONFIG_SHA256 = "d29f1e43ac468047b025751b9795b0c36a64dcf9f9e02dea381b57d54eed6442"


def validate_baseline_metadata(payload):
    """Reject other branches or update counts before model construction."""
    expected = dict(schema="a_then_fc_branch_v1", arm="A_continue",
                    base_cumulative_updates=1000, additional_updates=500,
                    cumulative_updates=1500, loss_switches={"FC": False, "FF": False})
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected pure-FM A_continue cumulative1500 metadata")
    if payload.get("optimizer_step_in_progress") or payload.get("batch_in_progress"):
        raise ValueError("Working baseline must be a completed checkpoint")


def load_baseline(checkpoint_path=BASELINE_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device="cuda"):
    """Strictly load the existing endpoint, frozen FP32/eval; no optimizer restore."""
    tick = time.perf_counter()
    checkpoint_path, config_path = Path(checkpoint_path), Path(config_path)
    actual_file_sha = file_sha256(checkpoint_path)
    if actual_file_sha != BASELINE_FILE_SHA256:
        raise ValueError("Expected the pinned A_continue cumulative1500 checkpoint")
    actual_config_sha = file_sha256(config_path)
    if actual_config_sha not in (BASELINE_CONFIG_SHA256, "fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f"):
        raise ValueError("Working baseline architecture configuration differs")
    configure_stable_runtime()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    if model_config.pop("model_type") != "equidit" or config["transport"]["prediction"] != "velocity":
        raise ValueError("Expected official equidit velocity configuration")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    validate_baseline_metadata(payload)
    with torch.device("meta"):
        model = NativeInpaintingModel(DiT(**model_config), trainable=False)
    # These official embedding constants are closures, not registered buffers.
    model.backbone.x_embedder.embed_fn, _ = get_embedder(model_config["pe_freq"], input_dims=3)
    keys = model.load_state_dict(payload["model"], strict=True, assign=True)
    del payload
    model.set_trainable(False).eval().to(device)
    state = state_sha256(model)
    backbone = state_sha256(model.backbone)
    if state != BASELINE_STATE_SHA256 or backbone != BASELINE_BACKBONE_SHA256:
        raise RuntimeError("Loaded A_continue tensor hashes differ")
    return model, dict(status="PASS", profile="a-continue", checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=actual_file_sha, config=str(config_path.resolve()),
        config_sha256=actual_config_sha, state_sha256=state, backbone_sha256=backbone,
        strict=True, missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys),
        official_EMA_loaded=False, optimizer_restored=False, checkpoint_schema="a_then_fc_branch_v1",
        arm="A_continue", base_cumulative_updates=1000, additional_updates=500, cumulative_updates=1500,
        loss_switches={"FC": False, "FF": False},
        role_shape=list(model.role_embedding.shape), role_nonzero=int(torch.count_nonzero(model.role_embedding)),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-tick)
