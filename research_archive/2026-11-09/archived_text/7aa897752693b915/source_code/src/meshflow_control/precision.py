"""Explicit, scoped precision/readout controls for fixed-weight diagnostics.

Default model calls retain the original BF16 path. This module never loads
weights, changes parameters, initializes CUDA or invokes a model on import.
"""
from __future__ import annotations

# Set the existing deterministic import policy before torch.
from . import runtime as _stable

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import torch
from torch import nn

PRECISIONS = ("LEGACY_BF16", "FP32_REFERENCE")
READOUT_SWITCHES = ("on", "off", "zero")
_POLICY = ContextVar("meshflow_precision_readout_policy", default=None)


def current_precision():
    """None outside this explicit experimental context."""
    policy = _POLICY.get()
    return None if policy is None else policy["precision"]


def readout_execution_options():
    """No options outside the context, preserving default call semantics."""
    policy = _POLICY.get()
    if policy is None:
        return {}
    return {"addition_mode": policy["readout"], "capture": policy["capture"]}


@contextmanager
def precision_execution(model, precision="LEGACY_BF16", readout="on", *,
                        coordinates=None, capture=None):
    """Execute the unchanged model under one declared, fixed-weight policy.

    OFF skips the four extra readout computations/additions only. ZERO computes
    their real residuals, casts as usual, then adds an all-zero tensor. CLEAN
    H0 construction, masks, Geo, role and original blocks are untouched.

    capture(record) is optional. At each actual free-corner addition it gets
    detached before/residual/cast/applied/after tensors and original corner
    indices. In ZERO, residual and cast remain the real branch result; applied
    is the zero tensor actually added. The callback must treat them read-only.
    No CPU copies are made here. The caller owns storage/statistical reduction.

    Parameters stay FP32 and frozen. The caller owns inference/no_grad mode;
    this context does not perform a forward or backward. Autocast, policy and
    caller matmul-precision selection unwind even on exceptions.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")
    if readout not in READOUT_SWITCHES:
        raise ValueError(f"Unknown readout switch: {readout}")
    if capture is not None and not callable(capture):
        raise TypeError("capture must be callable or None")
    if getattr(model, "experiment_trainable", False) or any(
            p.requires_grad for p in model.parameters()):
        raise RuntimeError("Precision-path experiment requires frozen parameters")
    if any(module.training for module in model.modules()):
        raise RuntimeError("Precision-path experiment requires all modules eval")
    if any(p.is_floating_point() and p.dtype != torch.float32
           for p in model.parameters()):
        raise RuntimeError("Precision-path experiment requires FP32 parameters")
    from .models.native import native_execution

    old_matmul = torch.get_float32_matmul_precision()
    policy_token = _POLICY.set(dict(precision=precision, readout=readout, capture=capture))
    fp32 = precision == "FP32_REFERENCE"
    try:
        if fp32:
            torch.set_float32_matmul_precision("highest")
        # The native context controls CUDA autocast. Also defeat a caller's CPU
        # autocast in the explicit FP32 route (useful for compact CPU contracts).
        with (torch.autocast("cpu", enabled=False) if fp32 else nullcontext()):
            with native_execution(model, coordinates=coordinates, autocast=not fp32) as facts:
                facts.update(precision=precision, readout_switch=readout,
                    readout_computation_skipped=(readout == "off"),
                    actual_zero_addition=(readout == "zero"),
                    float32_matmul_precision=torch.get_float32_matmul_precision())
                yield facts
    finally:
        _POLICY.reset(policy_token)
        if fp32:
            torch.set_float32_matmul_precision(old_matmul)


def _tensor_facts(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return [{"path": prefix, "dtype": str(value.dtype),
                 "shape": list(value.shape), "device": str(value.device),
                 "floating": value.is_floating_point()}]
    if isinstance(value, dict):
        return [item for key, child in value.items()
                for item in _tensor_facts(child, f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(value, (tuple, list)):
        return [item for i, child in enumerate(value)
                for item in _tensor_facts(child, f"{prefix}[{i}]")]
    return []


class DtypeAudit:
    """Bounded dtype-only hooks: no tensor values, reductions or device copies.

    Use once around an already budgeted forward, then retain records/summary().
    Pass observer=audit.observe to NativeModel.forward to include its embedding
    role/Geo fusion, free/known modulation, hidden streams and CLEAN memory.
    Optionally call audit.observe(record) from the readout capture callback.
    These observations do not replace model-call accounting or finite checks.
    """
    _CLASSES = {"XEmbedder", "TimestepEmbedder", "LabelEmbedder",
                "ContextGeometryEncoder", "RMSNorm", "PrecisionSafeLayerNorm",
                "SelfAttention", "FeedForward", "SwiGLUFFN", "CornerReadout",
                "FinalLayer", "NativeModel"}

    def __init__(self, model, precision="LEGACY_BF16", *, strict=True):
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision: {precision}")
        self.model, self.precision, self.strict = model, precision, bool(strict)
        self.records = []
        self._handles = []

    def _record(self, site, value, **fields):
        facts = _tensor_facts(value)
        bad = [x for x in facts if x["floating"] and x["dtype"] != "torch.float32"]
        row = dict(site=site, tensors=facts, **fields)
        self.records.append(row)
        if self.strict and self.precision == "FP32_REFERENCE" and bad:
            raise RuntimeError(f"FP32_REFERENCE dtype violation at {site}: {bad}")

    def observe(self, record):
        """NativeModel observer / readout-capture-compatible metadata consumer."""
        self._record("native_observer", record,
                     observed_site=record.get("site"), layer=record.get("layer"))

    def __enter__(self):
        if self._handles:
            raise RuntimeError("DtypeAudit is already active")
        for name, module in self.model.named_modules():
            if not (isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm))
                    or type(module).__name__ in self._CLASSES):
                continue
            def before(mod, args, kwargs, name=name):
                self._record("module_input", {"args": args, "kwargs": kwargs},
                             module=name, module_type=type(mod).__name__)
            def after(mod, args, kwargs, output, name=name):
                self._record("module_output", output,
                             module=name, module_type=type(mod).__name__)
            self._handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self._handles.append(module.register_forward_hook(after, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self):
        tensors = [t for row in self.records for t in row["tensors"] if t["floating"]]
        dtypes = sorted({t["dtype"] for t in tensors})
        bad = sum(t["dtype"] != "torch.float32" for t in tensors)
        return dict(precision=self.precision, records=len(self.records),
                    floating_tensor_observations=len(tensors), floating_dtypes=dtypes,
                    fp32_contract_violations=bad if self.precision == "FP32_REFERENCE" else None,
                    status=("NO_OBSERVATIONS" if not tensors else
                            "FAIL" if self.precision == "FP32_REFERENCE" and bad else "PASS"),
                    value_copies=0, model_calls_performed_by_auditor=0)
