"""Portable, fixed official Chair D0 proposal inference.

This module returns the complete unconstrained proposal, including moved C.
C restoration, GEO_C and source-winding export belong to the caller. Loading
uses only the official EMA checkpoint; inference never calls an encoder,
posterior sampler, generator, or a conditional-guidance branch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

# Set before importing torch / creating a CUDA context. Refuse silent changes
# to a process which has already initialized CUDA with an unknown setting.
_CUBLAS = ":4096:8"
_prior_torch = sys.modules.get("torch")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, _CUBLAS):
    raise RuntimeError("Denoising requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
    if _prior_torch is not None and _prior_torch.cuda.is_initialized():
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .denoiser_models.attention import SelfAttention
from .denoiser_models.equivae import AutoencoderKL

SCALE = 0.95 * 0.3762
OFFICIAL_SHA = "6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36"
OFFICIAL_BYTES = 7289979546
OFFICIAL_COMMIT = "db86bc32423c316c00bcc2f645a26243e8010f20"
EXPECTED_PARAMETERS = 455592963
EXPECTED_STATE_KEYS = 296
INIT_KWARGS = dict(hidden_dim=1024, num_layers=24, num_heads=16,
    latent_channels=3, decoder_type="reg", num_bins=256, use_rmsnorm=True,
    face_bin=100, use_identity_encoder=True, fixed_std=0.02)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_state() -> dict[str, Any]:
    return dict(cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())


def _assert_runtime() -> dict[str, Any]:
    state = _runtime_state()
    expected = dict(cublas=_CUBLAS, deterministic=True, warn_only=False,
        matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False,
        flash_sdp=False, mem_efficient_sdp=False, math_sdp=True, cudnn_sdp=False)
    if state != expected:
        raise RuntimeError(f"Denoiser numerical runtime changed: {state}")
    return state


def configure_runtime() -> dict[str, Any]:
    """Install the audited deterministic BF16/math-SDPA inference contract."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS:
        raise RuntimeError("CUBLAS configuration changed after import")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE=1 conflicts with the strict denoiser runtime")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return _assert_runtime()


def _checkpoint_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model, data = config["model"], config["data"]
    values = dict(hidden_dim=model["hidden_dim"], num_layers=model["num_layers"],
        num_heads=model["num_heads"], latent_channels=model["latent_channels"],
        decoder_type=model["decoder_type"], num_bins=data["num_bins"],
        use_rmsnorm=model["use_rms"], face_bin=model["face_bin"],
        use_identity_encoder=model["use_identity_encoder"], fixed_std=model["fixed_std"])
    if values != INIT_KWARGS or "03001627" not in str(data["data_path"]):
        raise ValueError("Checkpoint does not match the official Chair D0 configuration")
    return values


def _source_identity() -> dict[str, Any]:
    directory = Path(__file__).with_name("denoiser_models")
    manifest_path = directory / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["vendored_file_sha256"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"Packaged denoiser source hash mismatch: {name}")
    return dict(official_commit=OFFICIAL_COMMIT,
        source_manifest_sha256=_sha256(manifest_path),
        public_wrapper_sha256=_sha256(Path(__file__)),
        vendored_file_sha256=manifest["vendored_file_sha256"])


def load_denoiser(checkpoint: str | Path, device: str | torch.device = "cuda",
                  expected_sha256: str = OFFICIAL_SHA) -> tuple[AutoencoderKL, dict[str, Any]]:
    """Strictly load the fixed official EMA; no checkpoint download or fallback.

    Parameters remain FP32. The unused parameter-free version-2 cross-attention branch
    and the extra face-count embedding row are retained so the
    complete 296-key EMA is loaded without dropping anything. The checkpoint
    itself remains external to the package. CPU loading is supported for
    inspection, while proposal inference requires CUDA BF16.
    """
    if expected_sha256 != OFFICIAL_SHA:
        raise ValueError("This loader only supports the fixed official Chair D0 SHA256")
    path = Path(checkpoint).expanduser().resolve(strict=True)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before[0] != OFFICIAL_BYTES or _sha256(path) != expected_sha256:
        raise ValueError("Official checkpoint size or SHA256 mismatch")
    source = _source_identity()
    runtime = configure_runtime()
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Supported denoiser devices are cpu (inspection) and cuda")
    if target_device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Official D0 inference requires CUDA BF16 support")
    checkpoint_object = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    kwargs = _checkpoint_kwargs(checkpoint_object["config"])
    if "ema" not in checkpoint_object:
        raise ValueError("The required EMA state is absent; no model-state fallback")
    original = checkpoint_object["ema"]
    state = {key.removeprefix("module."): value for key, value in original.items()}
    if len(state) != len(original) or len(state) != EXPECTED_STATE_KEYS:
        raise ValueError("EMA prefix collision or unexpected state tensor count")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           for value in state.values()):
        raise ValueError("The full official EMA must contain FP32 tensors")
    # Initialization is overwritten by EMA and must not perturb the caller's
    # random stream. The original initialization formulas are retained.
    with torch.random.fork_rng(devices=[]):
        model = AutoencoderKL(**kwargs)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("EMA load was not strict")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS or len(model.state_dict()) != EXPECTED_STATE_KEYS:
        raise ValueError("Packaged architecture differs from the official 24-layer model")
    model.eval().requires_grad_(False)
    attention_count = 0
    for module in model.modules():
        if isinstance(module, SelfAttention):
            if module.dropout != 0 or module.causal:
                raise ValueError("Unexpected stochastic or causal attention")
            module.mixed_precision = "fp32"  # SDPA selector, NOT autocast dtype
            attention_count += 1
    if attention_count != 24:
        raise ValueError("Expected exactly 24 decoder self-attention layers")
    model = model.to(target_device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Denoiser parameters must remain FP32")
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Checkpoint changed during loading")
    identity = dict(status="PASS", checkpoint_path=str(path), checkpoint_sha256=expected_sha256,
        checkpoint_bytes=before[0], state_branch="ema", parameters=parameters,
        parameter_fp32_bytes=parameters * 4, state_keys=EXPECTED_STATE_KEYS,
        strict_missing_keys=[], strict_unexpected_keys=[], init_kwargs=kwargs,
        coordinate_scale=SCALE, precision="BF16 forward; FP32 parameters/residual/scaling/output",
        face_embedding_rows=int(model.decoder.y_embedder.embedding_table.weight.shape[0]),
        attention_backend="pytorch-sdpa/math", mask=None, device=str(target_device),
        torch_version=torch.__version__, runtime=runtime, source=source)
    model._meshflow_denoiser_identity = identity
    _assert_runtime()
    return model, identity


def _validate_raw(raw: np.ndarray) -> int:
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32:
        raise TypeError("RAW must be a numpy float32 array; no implicit coordinate cast")
    if raw.ndim != 3 or raw.shape[1:] != (3, 3) or not 1 <= len(raw) <= 800:
        raise ValueError("RAW must have shape (N,3,3), with total N in 1..800")
    if not np.isfinite(raw).all():
        raise ValueError("RAW contains nonfinite coordinates")
    return len(raw)


def _integrate_residual(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Exactly one FP32 addition followed by fixed inverse scaling."""
    if x.shape != residual.shape or x.dtype != torch.float32:
        raise ValueError("Residual and FP32 scaled input must have identical shape")
    return (residual.float() + x.float()) / SCALE


def predict_proposal(model: AutoencoderKL, raw: np.ndarray) -> np.ndarray:
    """Return one complete unconstrained FP32 proposal, with input order intact.

    ``raw`` is the whole C-union-F mesh in model coordinates, shaped (N,3,3).
    No C restoration, geometry cleanup, new noise, padding or face sorting is
    performed. The caller must keep the original C and restore it explicitly.
    The process runtime is checked before and after the single decoder call;
    any random-state consumption is an error. No independent RNG is created.
    """
    number = _validate_raw(raw)
    if not isinstance(model, AutoencoderKL) or not hasattr(model, "_meshflow_denoiser_identity"):
        raise TypeError("Use load_denoiser to obtain the verified official model")
    if any(module.training for module in model.modules()):
        raise ValueError("Denoiser must remain in eval mode")
    parameters = list(model.parameters())
    device = parameters[0].device
    if device.type != "cuda" or any(p.device != device or p.dtype != torch.float32 for p in parameters):
        raise ValueError("Proposal inference requires one CUDA device and all-FP32 parameters")
    if any(m.mixed_precision != "fp32" or m.dropout != 0 or m.causal
           for m in model.modules() if isinstance(m, SelfAttention)):
        raise ValueError("The verified attention configuration was changed")
    _assert_runtime()
    x = torch.from_numpy(np.ascontiguousarray(raw).reshape(1, number, 9)).to(device) * SCALE
    condition = torch.tensor([number], device=device, dtype=torch.long)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.pe_nerf(x.reshape(1, 3 * number, 3))
        residual = model.decode(z, condition, mask=None)
    proposal = _integrate_residual(x, residual).reshape(number, 3, 3).cpu().numpy().copy()
    if proposal.dtype != np.float32 or not np.isfinite(proposal).all():
        raise ValueError("Denoiser returned invalid coordinates")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError("Official proposal inference unexpectedly consumed RNG")
    _assert_runtime()
    return proposal
