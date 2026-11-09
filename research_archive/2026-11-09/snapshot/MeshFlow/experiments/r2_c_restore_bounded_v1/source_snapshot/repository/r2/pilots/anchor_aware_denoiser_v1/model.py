"""Versioned, minimal known-face readout for the released Chair denoiser.

Public API:
    model, spec = load_official_wrapper(anchor=True)
    model.train()  # inference uses model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        residual = model(x_d, known_faces, face_count)
    proposal_d = x_d.float() + residual.float()  # exactly once, by caller

``x_d`` is FP32 [1,N,9] in denoiser coordinates; ``known_faces`` is bool
[1,N]; ``face_count`` is long [1], equal to N. No padding mask is accepted.
The caller owns noise, coordinate scaling, free-only MSE, C restoration,
optimizer/RNG state, accounting and filesystem writes. This module writes no
files, calls no encode/sample path, and never selects or starts a GPU task.
The official model is exposed as ``model.base`` (and read-only ``model.model``
property); ``model.e_known`` is the only additional state tensor.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Iterator

import torch
from torch import nn

VERSION = "anchor_aware_denoiser_wrapper_v1"
REPO = Path(__file__).resolve().parents[3]
PREVIOUS = REPO / "experiments/r2_postprocess_avmr_pmr_v1"
OFFICIAL_SOURCE = PREVIOUS / "source_snapshot/official_denoiser_db86bc3"
OFFICIAL_CHECKPOINT = PREVIOUS / "assets/chair_denoiser_0030000.pt"
OFFICIAL_COMMIT = "db86bc32423c316c00bcc2f645a26243e8010f20"
OFFICIAL_SHA256 = "6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36"
OFFICIAL_BYTES = 7_289_979_546
OFFICIAL_PARAMETERS = 455_592_963
OFFICIAL_STATE_KEYS = 296
SCALE = 0.95 * 0.3762


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _official_imports() -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Import exact old sources without writing old pycache or runtime shims.

    The official source uses absolute ``models.*`` imports. Temporarily own
    those names and restore prior modules afterwards; do not poison another
    generator's models package. The constructor executes inside this context
    because it lazily imports the official PE implementation.
    """
    manifest = _read(PREVIOUS / "asset_audit/official_denoiser_source_manifest.json")
    if manifest["commit"] != OFFICIAL_COMMIT:
        raise ValueError("Wrong official source commit")
    for row in manifest["files"]:
        path = (OFFICIAL_SOURCE / row["path"]).resolve()
        if not path.is_relative_to(OFFICIAL_SOURCE.resolve()):
            raise ValueError("Official source manifest escapes snapshot")
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Official source changed: {row['path']}")

    old = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    old_no_bytecode = sys.dont_write_bytecode
    for name in old:
        del sys.modules[name]
    sys.dont_write_bytecode = True
    try:
        package = types.ModuleType("models")
        package.__path__ = [str(OFFICIAL_SOURCE / "models")]
        package.__file__ = str(OFFICIAL_SOURCE / "models/__init__.py")
        sys.modules["models"] = package

        # Exact same narrowly extracted compatibility helpers as the audited
        # prior runner, held only in memory. Neither dependency alters math.
        source = (OFFICIAL_SOURCE / "models/rmsnorm.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RMSNorm")
        rms_code = ast.get_source_segment(source, node)
        rms = types.ModuleType("models.rmsnorm")
        rms.__dict__.update(torch=torch, nn=nn)
        exec(compile(rms_code, str(OFFICIAL_SOURCE / "models/rmsnorm.py") + "[RMSNorm only]", "exec"), rms.__dict__)
        sys.modules["models.rmsnorm"] = rms

        source = (OFFICIAL_SOURCE / "models/utils.py").read_text(encoding="utf-8")
        wanted = {"get_2d_sincos_pos_embed", "get_2d_sincos_pos_embed_from_grid", "get_1d_sincos_pos_embed_from_grid"}
        fragments = [ast.get_source_segment(source, n) for n in ast.parse(source).body
                     if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if len(fragments) != 3:
            raise ValueError("Official positional helpers incomplete")
        utils_code = "import numpy as np\n\n" + "\n\n".join(fragments) + "\n"
        utils = types.ModuleType("models.utils")
        exec(compile(utils_code, str(OFFICIAL_SOURCE / "models/utils.py") + "[sincos only]", "exec"), utils.__dict__)
        sys.modules["models.utils"] = utils
        module = importlib.import_module("models.equivae")
        attention = importlib.import_module("models.attention")
        if Path(module.__file__).resolve() != (OFFICIAL_SOURCE / "models/equivae.py").resolve():
            raise ValueError("Wrong official model imported")
        yield module, attention, manifest
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        sys.modules.update(old)
        sys.dont_write_bytecode = old_no_bytecode


class AnchorAwareDenoiser(nn.Module):
    """PE -> projection -> role add -> unchanged official decoder residual.

    Role addition is in the activation dtype in both arms. Known features are
    neither frozen nor detached, so the free reconstruction loss can propagate
    through attention to the known coordinates' feature path.
    """

    def __init__(self, base: nn.Module, *, anchor: bool):
        super().__init__()
        self.base = base
        self.anchor = bool(anchor)
        hidden = int(base.post_quant_linear.out_features)
        zero = torch.zeros(hidden, dtype=torch.float32)
        if self.anchor:
            self.e_known = nn.Parameter(zero)
        else:
            self.register_buffer("e_known", zero, persistent=True)
        self.last_role_stats: dict[str, Any] | None = None

    @property
    def model(self) -> nn.Module:
        return self.base

    def _inputs(self, x_d: torch.Tensor, known_faces: torch.Tensor | None,
                face_count: torch.Tensor | None) -> tuple[int, torch.Tensor]:
        if x_d.ndim != 3 or x_d.shape[0] != 1 or x_d.shape[2] != 9:
            raise ValueError("Only unpadded microbatch1 [1,N,9] is supported")
        if x_d.dtype != torch.float32:
            raise ValueError("Coordinate input must remain FP32; use BF16 autocast externally")
        n = int(x_d.shape[1])
        if n < 1:
            raise ValueError("Empty mesh")
        if known_faces is not None:
            if known_faces.dtype != torch.bool or known_faces.shape != (1, n) or known_faces.device != x_d.device:
                raise ValueError("known_faces must be bool [1,N] on the input device")
        elif self.anchor:
            raise ValueError("Anchor arm requires explicit known_faces")
        if face_count is None:
            face_count = torch.tensor([n], dtype=torch.long, device=x_d.device)
        if face_count.shape != (1,) or face_count.dtype != torch.long or face_count.device != x_d.device:
            raise ValueError("face_count must be long [1] on the input device")
        if int(face_count.item()) != n:
            raise ValueError("N must count the complete input mesh faces")
        return n, face_count

    def official_residual(self, x_d: torch.Tensor, face_count: torch.Tensor | None = None) -> torch.Tensor:
        """Exact reference PE/decode path, for separately budgeted prechecks."""
        # The reference does not need a known mask, even for an anchor wrapper.
        n = int(x_d.shape[1]) if x_d.ndim == 3 else 0
        dummy = torch.zeros((1, n), dtype=torch.bool, device=x_d.device)
        n, face_count = self._inputs(x_d, dummy, face_count)
        z = self.base.pe_nerf(x_d.reshape(1, 3 * n, 3))
        return self.base.decode(z, face_count, mask=None)

    def forward(self, x_d: torch.Tensor, known_faces: torch.Tensor | None,
                face_count: torch.Tensor | None = None, *,
                capture_role_stats: bool = False) -> torch.Tensor:
        n, face_count = self._inputs(x_d, known_faces, face_count)
        z = self.base.pe_nerf(x_d.reshape(1, 3 * n, 3))
        h = self.base.post_quant_linear(z)
        if self.anchor:
            marker = known_faces.repeat_interleave(3, dim=1).unsqueeze(-1).to(dtype=h.dtype)
        else:
            # No-role does not read the values of the true known mask. Both
            # arms still execute identically placed/dtyped multiply and add.
            marker = torch.zeros((1, 3 * n, 1), dtype=h.dtype, device=h.device)
        role = marker * self.e_known.to(dtype=h.dtype).view(1, 1, -1)
        conditioned = h + role
        if capture_role_stats:
            with torch.no_grad():
                actual = conditioned.detach().float() - h.detach().float()
                self.last_role_stats = {
                    "activation_dtype": str(h.dtype),
                    "role_parameter_dtype": str(self.e_known.dtype),
                    "role_l2": float(self.e_known.detach().float().norm().item()),
                    "requested_add_l2": float(role.detach().float().norm().item()),
                    "effective_add_l2": float(actual.norm().item()),
                    "effective_add_max_abs": float(actual.abs().max().item()),
                    "changed_hidden_scalars": int(torch.count_nonzero(actual).item()),
                    "total_hidden_scalars": actual.numel(),
                }
        return self.base.decoder(conditioned, face_count, mask=None)

    def validate_parameters(self) -> None:
        if any(p.dtype != torch.float32 for p in self.base.parameters()) or self.e_known.dtype != torch.float32:
            raise ValueError("Parameters and role must stay FP32")
        if self.e_known.shape != (self.base.post_quant_linear.out_features,):
            raise ValueError("Role shape mismatch")
        if not self.anchor and bool(torch.count_nonzero(self.e_known).item()):
            raise ValueError("No-role state must remain exactly zero")


def load_official_wrapper(*, anchor: bool, activation_checkpointing: bool = True,
                          face_count_dropout: float = 0.1,
                          verify_checkpoint_hash: bool = True) -> tuple[AnchorAwareDenoiser, dict[str, Any]]:
    """Strict CPU EMA load; returns an FP32 trainable wrapper and audit spec.

    Model construction is inside CPU fork_rng: neither official random init
    (subsequently overwritten) nor zero-role creation shifts caller data RNG.
    Activation checkpointing is the official non-reentrant per-block path;
    its default preserve_rng_state=True is untouched. Caller must save the
    returned spec and use matched CPU/CUDA RNG streams between both arms.
    ``verify_checkpoint_hash=False`` is only for a caller that has already
    independently hashed this protected asset in the current run.
    """
    if face_count_dropout != 0.1:
        raise ValueError("Registered pilot uses actual official train dropout0.1 in both arms")
    if OFFICIAL_CHECKPOINT.stat().st_size != OFFICIAL_BYTES:
        raise ValueError("Official checkpoint size mismatch")
    if verify_checkpoint_hash and sha256(OFFICIAL_CHECKPOINT) != OFFICIAL_SHA256:
        raise ValueError("Official checkpoint hash mismatch")
    expected = _read(PREVIOUS / "strict_load.json")
    if expected["checkpoint_sha256"] != OFFICIAL_SHA256 or expected["state_branch"] != "ema":
        raise ValueError("Unexpected prior strict-load identity")
    checkpoint = torch.load(OFFICIAL_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["config"]
    if config != expected["config"]:
        raise ValueError("Official checkpoint config differs from audited config")
    cfg, data = config["model"], config["data"]
    kwargs = dict(hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
                  latent_channels=cfg["latent_channels"], decoder_type=cfg["decoder_type"], num_bins=data["num_bins"],
                  use_rmsnorm=cfg["use_rms"], face_bin=cfg["face_bin"],
                  use_identity_encoder=cfg["use_identity_encoder"], fixed_std=cfg["fixed_std"])
    if kwargs != expected["init_kwargs"] or cfg["fixed_std"] != 0.02 or "03001627" not in data["data_path"]:
        raise ValueError("Wrong Chair model initialization contract")
    if "ema" not in checkpoint:
        raise ValueError("Official EMA branch missing; no fallback")
    state = {key.removeprefix("module."): value for key, value in checkpoint["ema"].items()}
    if len(state) != len(checkpoint["ema"]) or len(state) != OFFICIAL_STATE_KEYS:
        raise ValueError("EMA state count/prefix collision")
    with _official_imports() as (module, attention, manifest):
        with torch.random.fork_rng(devices=[]):
            base = module.AutoencoderKL(**kwargs)
        result = base.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("Official state was not strictly loaded")
        if sum(p.numel() for p in base.parameters()) != OFFICIAL_PARAMETERS:
            raise ValueError("Official parameter count mismatch")
        if any(p.dtype != torch.float32 for p in base.parameters()):
            raise ValueError("Official parameters not FP32")
        original_checkpoint_flags = [layer.gradient_checkpointing for layer in base.decoder.layers]
        for layer in base.decoder.layers:
            layer.gradient_checkpointing = bool(activation_checkpointing)
        base.decoder.y_embedder.dropout_prob = face_count_dropout
        attentions = 0
        for layer in base.modules():
            if isinstance(layer, attention.SelfAttention):
                if layer.dropout != 0 or layer.causal:
                    raise ValueError("Unexpected stochastic or causal attention")
                layer.mixed_precision = "fp32"  # SDPA backend selector, not activation cast.
                attentions += 1
        base.requires_grad_(True)
        wrapper = AnchorAwareDenoiser(base, anchor=anchor)
        wrapper.validate_parameters()
    spec = {
        "wrapper_version": VERSION, "wrapper_sha256": sha256(Path(__file__)),
        "official_source_commit": OFFICIAL_COMMIT, "official_source_manifest": manifest,
        "checkpoint_path": str(OFFICIAL_CHECKPOINT), "checkpoint_bytes": OFFICIAL_BYTES,
        "checkpoint_sha256": OFFICIAL_SHA256, "checkpoint_hash_verified_by_loader": verify_checkpoint_hash,
        "state_branch": "ema", "official_parameters": OFFICIAL_PARAMETERS, "official_state_keys": len(state),
        "strict_missing_keys": [], "strict_unexpected_keys": [], "init_kwargs": kwargs,
        "anchor": bool(anchor), "e_known_shape": list(wrapper.e_known.shape),
        "e_known_trainable": bool(anchor), "e_known_initialization": "exact FP32 zero, no RNG",
        "trainable_parameters": sum(p.numel() for p in wrapper.parameters() if p.requires_grad),
        "wrapper_state_keys": len(wrapper.state_dict()), "official_parameters_trainable": True,
        "role_location": "after official PE/post_quant_linear, before unchanged decoder",
        "role_add_dtype": "post_quant_linear activation dtype; BF16 under caller BF16 autocast",
        "known_feature_detach": False, "network_output": "residual only; no projection/scaling/residual addition",
        "padding_mask": None, "microbatch": 1, "face_condition": "original N//face_bin and AdaLN retained",
        "actual_original_checkpoint_flags": original_checkpoint_flags,
        "activation_checkpointing": bool(activation_checkpointing), "checkpoint_use_reentrant": False,
        "checkpoint_preserve_rng_state": True, "face_count_dropout_training": face_count_dropout,
        "face_count_dropout_eval": 0.0, "attention_dropout": 0.0, "SDPA_attention_modules": attentions,
        "fixed_std": float(cfg["fixed_std"]), "fixed_std_units": "denoiser coordinates",
        "noise_in_network": False, "coordinate_scale": SCALE, "caller_CPU_rng_preserved_by_initialization": True,
        "checkpoint_config_gradient_checkpointing_is_not_forwarded_by_AutoencoderKL": cfg["gradient_checkpointing"],
    }
    del state, checkpoint
    gc.collect()
    return wrapper, spec
