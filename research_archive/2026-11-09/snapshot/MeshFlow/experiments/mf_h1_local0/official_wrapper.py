"""Strict local EMA loading and residual hooks; upstream source is untouched."""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn
import yaml
from .adapters import ForwardGeometryCache


def state_sha256(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_official(checkpoint_path, config_path, device="cpu"):
    """Never deserialize arbitrary pickle, relax strictness, or guess EMA keys."""
    from models.equidit import DiT
    from models.utils import get_embedder
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    model_config.pop("model_type")
    required = dict(version=3, hidden_dim=768, num_layers=12, num_heads=12,
                    max_length=800, face_bin=20, use_qknorm=True,
                    use_rmsnorm=True, use_coord_encoding=True, use_dit_like_pe=False,
                    face_cond=True)
    for key, expected in required.items():
        if model_config.get(key) != expected:
            raise ValueError(f"Official configuration mismatch: {key}")
    if config["transport"]["prediction"] != "velocity":
        raise ValueError("Only audited velocity checkpoint is supported")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["ema"]
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed module prefixes are not accepted")
    if all(prefixed):
        state = {key[7:]: value for key, value in state.items()}
    with torch.device("meta"):
        model = DiT(**model_config)
    # Official Embedder closes over unregistered frequency tensors. Recreate
    # only these deterministic constants on CPU; checkpoint tensors stay exact.
    embed_fn, _ = get_embedder(model_config["pe_freq"], input_dims=3)
    model.x_embedder.embed_fn = embed_fn
    result = model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).eval()
    model.to(device)
    audit = {"status": "PASS", "top_level_keys": list(checkpoint), "key": "ema",
             "strict": True, "assign": True, "missing_keys": result.missing_keys,
             "unexpected_keys": result.unexpected_keys, "config": config,
             "parameters": sum(p.numel() for p in model.parameters()),
             "meta_frequency_constants_recreated_on_cpu": True}
    return model, audit


class OfficialWrapper(nn.Module):
    """Call the actual official forward once, with hooks after audited layers.

    No target argument exists. Geometry can read only this call's x_t and prompts.
    Freezing parameters does not disable autograd through later backbone layers.
    A wrapper owns its hooks; close() before attaching another to the same base.
    """
    def __init__(self, backbone, adapter):
        super().__init__()
        if backbone.version != 3 or backbone.hidden_size != 768 or len(backbone.layers) != 12:
            raise ValueError("Unexpected official architecture")
        self.backbone = backbone.requires_grad_(False).eval()
        self.adapter = adapter
        self.counts = dict(wrapper_forward=0, backbone_forward=0, adapter_layer=0,
                           control_disabled=0, condition_encoding=0,
                           geometry_requests=0, helper_calls=0, computations=0, cache_hits=0)
        self._context = None
        self.last_geometry_cache_summary = dict(enabled=False,entries_created=0,entries_after_cleanup=0)
        self._handles = [backbone.register_forward_pre_hook(self._count_backbone)]
        for index, layer_number in enumerate((3, 6, 9, 12)):
            self._handles.append(backbone.layers[layer_number - 1].register_forward_hook(self._hook(index)))

    def _count_backbone(self, module, args):
        self.counts["backbone_forward"] += 1

    def _hook(self, index):
        def residual_hook(module, args, output):
            if self._context is None:
                return output
            xt, t, encoded, mask, scale, geometry_cache = self._context
            self.counts["adapter_layer"] += 1
            return output + scale * self.adapter.forward_layer(index, output, xt, t, encoded, mask,
                                                               geometry_cache=geometry_cache)
        return residual_hook

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, prompts):
        self.counts["condition_encoding"] += 1
        return self.adapter.encode(prompts)

    def forward(self, x_t, t, y, mask=None, prompts=None, prompt_mask=None,
                control_scale=1.0, encoded=None, inference_geometry_cache=False):
        if self._context is not None:
            raise RuntimeError("Reentrant wrapper invocation is not supported")
        if inference_geometry_cache:
            if self.adapter.arm != "G_closest":
                raise ValueError("Inference geometry cache is only defined for G_closest")
            if self.training or self.adapter.training or self.backbone.training or torch.is_grad_enabled():
                raise RuntimeError("Geometry caching requires eval mode and disabled gradients; training is unvalidated")
        self.counts["wrapper_forward"] += 1
        geometry_cache = None
        self.last_geometry_cache_summary = dict(enabled=False,entries_created=0,entries_after_cleanup=0)
        try:
            if prompt_mask is not None:
                prompts = dict(prompts, primitive_mask=prompt_mask)
            enabled = control_scale != 0 and (prompts is not None or encoded is not None)
            if enabled:
                if encoded is None:
                    encoded = self.encode(prompts)
                corner_mask = None if mask is None else mask.repeat_interleave(3, dim=1)
                xt = x_t.reshape(x_t.shape[0], -1, 3)
                if self.adapter.arm == "G_closest":
                    geometry_cache = ForwardGeometryCache(inference_geometry_cache, self.counts,
                                                          xt, encoded.vertices, encoded.arity)
                self._context = (xt, t.reshape(x_t.shape[0], -1)[:, 0],
                                 encoded, corner_mask, control_scale, geometry_cache)
            else:
                self.counts["control_disabled"] += 1
            return self.backbone(x_t, t, y, mask=mask)
        finally:
            self._context = None
            if geometry_cache is not None:
                self.last_geometry_cache_summary = geometry_cache.close()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
