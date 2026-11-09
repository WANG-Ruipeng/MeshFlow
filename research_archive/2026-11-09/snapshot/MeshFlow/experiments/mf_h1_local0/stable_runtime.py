"""Shared strict execution policy for new training, teacher and sampling entrypoints.

Import this module BEFORE importing torch or model modules in an entrypoint.
Use ``with stable_execution(backbone, adapter):`` around actual computation;
the context preserves the caller's gradient mode and never casts parameters.
No exception is caught to retry an unsupported deterministic operation.
"""
from __future__ import annotations

import os
import sys

_CUBLAS_CONFIG = ":4096:8"
_existing_torch = sys.modules.get("torch")
_existing_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_config not in (None, _CUBLAS_CONFIG):
    raise RuntimeError("Stable execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8; conflicting value found")
if (_existing_config is None and _existing_torch is not None
        and _existing_torch.cuda.is_initialized()):
    raise RuntimeError("Import stable_runtime before CUDA initialization; CUBLAS configuration was absent")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_CONFIG

from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_ACTIVE_DEPTH = ContextVar("meshflow_stable_execution_depth", default=0)


def _parameter_summary(modules):
    count = 0
    seen = set()
    for module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                raise RuntimeError(f"Stable execution requires FP32 parameters: {name} is {parameter.dtype}")
            count += parameter.numel()
    return count


def configure_stable_runtime():
    """Set the process policy explicitly; this does not initialize CUDA."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG changed after stable_runtime import")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE=1 conflicts with the stable TF32-off policy")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from models import attention
    if attention.FLASH_ATTN_AVAILABLE:
        raise RuntimeError("External FlashAttention is available; the audited SDPA-only environment is required")
    return assert_stable_runtime(require_context=False)


def assert_stable_runtime(*modules, require_context=True, coordinates=None):
    """Fail closed and return JSON-safe facts; call inside actual execution paths."""
    from models import attention
    coordinate_tensors = () if coordinates is None else (
        (coordinates,) if isinstance(coordinates, torch.Tensor) else tuple(coordinates))
    for coordinate in coordinate_tensors:
        if not isinstance(coordinate, torch.Tensor) or coordinate.dtype != torch.float32:
            raise RuntimeError("Stable execution requires FP32 integration coordinates")
    facts = dict(
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        external_flash_available=attention.FLASH_ATTN_AVAILABLE,
        sdpa_math_enabled=torch.backends.cuda.math_sdp_enabled(),
        sdpa_flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
        sdpa_cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled(),
        cuda_autocast_enabled=torch.is_autocast_enabled("cuda"),
        cuda_autocast_dtype=str(torch.get_autocast_dtype("cuda")),
        gradient_enabled=torch.is_grad_enabled(),
        parameter_dtype="torch.float32",
        parameter_elements_checked=_parameter_summary(modules),
        coordinate_tensors_checked=len(coordinate_tensors),
        active_context_depth=_ACTIVE_DEPTH.get(),
    )
    required = (facts["cublas_workspace_config"] == _CUBLAS_CONFIG
                and facts["deterministic_algorithms"] and not facts["deterministic_warn_only"]
                and not facts["tf32_matmul"] and not facts["tf32_cudnn"]
                and not facts["cudnn_benchmark"] and not facts["external_flash_available"])
    if require_context:
        required = (required and facts["active_context_depth"] > 0
                    and facts["sdpa_math_enabled"] and not facts["sdpa_flash_enabled"]
                    and not facts["sdpa_mem_efficient_enabled"] and not facts["sdpa_cudnn_enabled"]
                    and facts["cuda_autocast_enabled"]
                    and torch.get_autocast_dtype("cuda") == torch.bfloat16)
    if not required:
        raise RuntimeError(f"Stable runtime assertion failed: {facts}")
    return facts


@contextmanager
def stable_execution(*modules):
    """Strict deterministic CUDA BF16/SDPA-math execution; grad mode is unchanged.

    Deterministic/TF32 settings remain strict after exit. Autocast and backend
    contexts unwind normally, including on exceptions. Parameters are checked,
    never converted. Use for a training forward/backward, teacher, or sampler.
    """
    configure_stable_runtime()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        token = _ACTIVE_DEPTH.set(_ACTIVE_DEPTH.get() + 1)
        try:
            facts = assert_stable_runtime(*modules)
            yield facts
            assert_stable_runtime(*modules)
        finally:
            _ACTIVE_DEPTH.reset(token)
