"""Set the pre-CUDA environment without importing PyTorch or a model."""
import os
import sys

CUBLAS_CONFIG = ":4096:8"


def ensure_cublas_environment():
    """Retain the original import-order guard for model and CPU-only entrypoints."""
    existing_torch = sys.modules.get("torch")
    existing_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_config not in (None, CUBLAS_CONFIG):
        raise RuntimeError("Stable execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8; conflicting value found")
    if (existing_config is None and existing_torch is not None
            and existing_torch.cuda.is_initialized()):
        raise RuntimeError("Import stable_runtime before CUDA initialization; CUBLAS configuration was absent")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG
