"""Isolated rank-four policies sharing the frozen FULL_S6000 base.

Only block attention qkv/output and SiLU MLP input/output linears receive LoRA.
Native arithmetic is copied below solely to replace its all-or-none parameter
guard.  No maintained model/runtime module is patched or monkeypatched.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import copy
import hashlib
import math

from meshflow_control import runtime as _stable  # Before any CUDA initialization.
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash
from meshflow_control.models.native import (
    NativeModel, build_model, native_block_forward, native_final_forward,
)

VERSION = "terminal_reward_nft_lora_v1"
RANK = 4
ALPHA = 4.0
TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.net.0", "mlp.net.2")
EXPECTED_MODULES = 48
EXPECTED_TRAINABLE = 589824
_DEPTH = ContextVar("terminal_reward_nft_execution_depth", default=0)
_AUTOCAST = ContextVar("terminal_reward_nft_autocast", default=True)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class AdapterLinear(nn.Module):
    """One untouched base Linear, current factors and detached old factors."""
    def __init__(self, base, rank=RANK, alpha=ALPHA):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.dtype != torch.float32:
            raise ValueError("LoRA requires a loaded FP32 Linear")
        if rank != RANK or alpha != ALPHA:
            raise ValueError("This pilot fixes rank=4 and alpha=4")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        self.in_features, self.out_features = base.in_features, base.out_features
        self.current_a = nn.Parameter(torch.empty(rank, base.in_features,
                                                   dtype=torch.float32, device=base.weight.device))
        self.current_b = nn.Parameter(torch.zeros(base.out_features, rank,
                                                   dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.current_a, a=math.sqrt(5))
        self.register_buffer("old_a", self.current_a.detach().clone())
        self.register_buffer("old_b", self.current_b.detach().clone())
        self.active_adapter = "current"
        self.eval()

    def forward(self, x):
        result = self.base(x)
        if self.active_adapter == "reference":
            return result
        if self.active_adapter == "current":
            a, b = self.current_a, self.current_b
        elif self.active_adapter == "old":
            a, b = self.old_a, self.old_b
        else:
            raise RuntimeError("Unknown adapter")
        return result + F.linear(F.linear(x, a), b) * self.scale


def assert_nft_runtime(model, coordinates=None, *, require_context=True):
    facts = _stable.assert_stable_runtime(model, require_context=False,
                                         coordinates=coordinates)
    if any(m.training for m in model.modules()):
        raise RuntimeError("NFT policy and all submodules must remain eval")
    expected = set(model.current_parameter_names())
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != expected:
        raise RuntimeError("Only the registered current LoRA factors may require gradients")
    if len(model.adapter_modules()) != EXPECTED_MODULES:
        raise RuntimeError("LoRA module count changed")
    if sum(p.numel() for p in model.current_parameters()) != EXPECTED_TRAINABLE:
        raise RuntimeError("LoRA trainable parameter count changed")
    if any(layer.gradient_checkpointing for layer in model.backbone.layers):
        raise RuntimeError("Gradient checkpointing must remain disabled")
    for module in model.modules():
        if hasattr(module, "dropout") and isinstance(module.dropout, (float, int)):
            if module.dropout != 0:
                raise RuntimeError("Attention dropout must remain zero")
        if isinstance(module, AdapterLinear):
            if module.active_adapter != model.active_adapter:
                raise RuntimeError("Mixed current/old/reference adapter selection")
            if module.old_a.dtype != torch.float32 or module.old_b.dtype != torch.float32:
                raise RuntimeError("Old factors must remain FP32")
    if require_context:
        if _DEPTH.get() < 1:
            raise RuntimeError("NFT forward must execute in nft_execution")
        if not (torch.backends.cuda.math_sdp_enabled()
                and not torch.backends.cuda.flash_sdp_enabled()
                and not torch.backends.cuda.mem_efficient_sdp_enabled()
                and not torch.backends.cuda.cudnn_sdp_enabled()):
            raise RuntimeError("NFT requires SDPA math only")
        if torch.is_autocast_enabled("cuda") != _AUTOCAST.get():
            raise RuntimeError("Unexpected CUDA autocast state")
        if _AUTOCAST.get() and torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise RuntimeError("NFT forward precision must be BF16")
        if coordinates is not None:
            values = (coordinates,) if isinstance(coordinates, torch.Tensor) else coordinates
            if any(v.device.type == "cuda" for v in values) and not _AUTOCAST.get():
                raise RuntimeError("FP32 CUDA model forwards are outside this pilot")
    facts.update(runtime_version=VERSION, active_adapter=model.active_adapter,
                 trainable_parameters=EXPECTED_TRAINABLE, all_modules_eval=True,
                 lora_rank=RANK, lora_alpha=ALPHA, lora_dropout=0,
                 old_is_buffer=True, reference_shared_frozen_base=True)
    return facts


@contextmanager
def nft_execution(model, *, coordinates=None, autocast=True):
    """Independent LoRA guard with the maintained deterministic math policy.

CPU structural/unit checks may explicitly use autocast=False. All actual GPU
training and generation use BF16 forward with FP32 parameters and coordinates.
The caller owns gradient mode and GPU-budget accounting.
"""
    _stable.configure_stable_runtime()
    if autocast and next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Use autocast=False for an explicitly authorized CPU check")
    with sdpa_kernel(SDPBackend.MATH), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bool(autocast)):
        d = _DEPTH.set(_DEPTH.get() + 1)
        a = _AUTOCAST.set(bool(autocast))
        try:
            yield assert_nft_runtime(model, coordinates)
            assert_nft_runtime(model, coordinates)
        finally:
            _AUTOCAST.reset(a)
            _DEPTH.reset(d)


class LoRAPolicy(NativeModel):
    def adapter_modules(self):
        return {n: m for n, m in self.named_modules() if isinstance(m, AdapterLinear)}

    def current_parameter_names(self):
        return [n for n, _ in self.named_parameters()
                if n.endswith((".current_a", ".current_b"))]

    def current_parameters(self):
        names = set(self.current_parameter_names())
        return [p for n, p in self.named_parameters() if n in names]

    def set_trainable(self, value):
        raise RuntimeError("NFT parameter ownership is fixed; use torch.no_grad for sampling")

    @contextmanager
    def activate(self, adapter):
        if adapter not in ("current", "old", "reference"):
            raise ValueError("Unknown adapter")
        previous = self.active_adapter
        self.active_adapter = adapter
        for module in self.adapter_modules().values():
            module.active_adapter = adapter
        try:
            yield self
        finally:
            self.active_adapter = previous
            for module in self.adapter_modules().values():
                module.active_adapter = previous

    @torch.no_grad()
    def update_old(self, decay=0.0):
        """This pilot freezes an exact round-start snapshot, not an EMA."""
        if float(decay) != 0.0:
            raise ValueError("The registered pilot requires old snapshot decay=0")
        for module in self.adapter_modules().values():
            module.old_a.copy_(module.current_a)
            module.old_b.copy_(module.current_b)

    def adapter_state_dict(self):
        return {f"{name}.{key}": getattr(module, key).detach().cpu().clone()
                for name, module in self.adapter_modules().items()
                for key in ("current_a", "current_b", "old_a", "old_b")}

    @torch.no_grad()
    def load_adapter_state_dict(self, state):
        expected = self.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError("Adapter checkpoint keys differ")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != torch.float32:
                raise ValueError("Adapter tensor shape/dtype differs: " + name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite adapter tensor: " + name)
        modules = self.adapter_modules()
        for name, value in state.items():
            module_name, key = name.rsplit(".", 1)
            getattr(modules[module_name], key).copy_(value)

    def base_state_dict(self):
        """Recover original Native names for source-identity auditing."""
        out = {}
        for name, value in self.state_dict().items():
            if name.endswith((".current_a", ".current_b", ".old_a", ".old_b")):
                continue
            out[name.replace(".base.weight", ".weight").replace(".base.bias", ".bias")] = value
        return out

    def metadata(self):
        return dict(version=VERSION, rank=RANK, alpha=ALPHA, dropout=0,
                    target_suffixes=list(TARGET_SUFFIXES),
                    target_names=list(self.adapter_modules()),
                    trainable_parameters=EXPECTED_TRAINABLE, init_seed=self.init_seed,
                    base_file_sha256=self.base_file_sha256,
                    base_state_sha256=self.base_state_sha256,
                    old_update="exact_round_start_snapshot", reference="disabled_adapter",
                    time_orientation="noise_0_endpoint_1")

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        assert_nft_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError("y must count real faces only, N128..256")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        if self.readout_mode != "none" or len(self.readouts):
            raise RuntimeError("NFT base must have no context readout")
        hidden, role = self._embed(x, known_mask, valid_mask)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                          known_mask=known_mask, valid_mask=valid_mask,
                          face_time=face_time, c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)


def load_policy(generator_path, expected_sha256=None, device="cpu", init_seed=20261015):
    """Strictly load FULL_S6000 without old optimizer, then attach fresh LoRA."""
    path = Path(generator_path)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("FULL_S6000 file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("schema") != "r2_full_seat_learning_curve_generator_v1"
            or payload.get("local_step") != 6000 or payload.get("global_step") != 12000
            or payload.get("EMA") is not False or payload.get("model_config") != MODEL_CONFIG):
        raise ValueError("Expected original live FULL_S6000 generator export")
    state = payload["generator_state"]
    base_hash = state_hash(state)
    if base_hash != payload["generator_state_sha256"]:
        raise ValueError("FULL_S6000 internal generator hash differs")
    if len(state) != 194 or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Base must contain original finite FP32 parameters")
    with torch.device("meta"):
        model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=False)
    model.load_state_dict(state, strict=True, assign=True)
    model.__class__ = LoRAPolicy
    model.active_adapter = "current"
    model.base_file_sha256, model.base_state_sha256 = digest, base_hash
    model.init_seed = int(init_seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(init_seed))
        for layer in model.backbone.layers:
            for suffix in TARGET_SUFFIXES:
                parent_name, child_name = suffix.rsplit(".", 1)
                parent = layer.get_submodule(parent_name)
                setattr(parent, child_name, AdapterLinear(getattr(parent, child_name)))
    model.to(device).eval()
    _stable.configure_stable_runtime()
    assert_nft_runtime(model, require_context=False)
    if state_hash(model.base_state_dict()) != base_hash:
        raise ValueError("Attaching LoRA changed original base parameters")
    return model
