"""Isolated C-reference coordinates with unchanged Native/T1 + world-Geo weights.

Only the coordinate embedder reads local coordinates.  The existing Geo encoder
reads explicit original-world C, at exactly the same face/corner slots.  No new
parameters, buffers, learned heads, rotations, scales, clamps, or noise demeaning
are introduced.  The original Native block/final arithmetic is imported intact.
"""
from __future__ import annotations

from meshflow_control import runtime  # Establish CUBLAS before importing torch.
from pathlib import Path
import json
import time
import numpy as np
import torch

from meshflow_control.models.native import (
    NativeModel, _corners, assert_native_runtime, native_execution,
    native_block_forward, native_final_forward, build_model as _build_native,
)
from meshflow_control.models.geometry import inject_context
from meshflow_control.mesh_io import validate_C, write_raw
from meshflow_control.sampling import make_noise

FRAME_VERSION = "c_reference_area_fp32_world_geo_v1"


def area_weighted_center(C):
    """Return FP32 area-weighted surface center of augmented C, without Y.

    Input face order is retained, and every arithmetic/reduction is FP32.  The
    caller computes this once after whole-pair augmentation, before random slot
    permutations.  This is a translation only, not a triangulation-invariance or
    exact floating-point permutation-invariance promise.
    """
    C = np.asarray(C)
    if C.dtype != np.float32 or C.ndim not in (2, 3) or C.shape[1:] not in ((9,), (3, 3)):
        raise ValueError("Center requires FP32 C[K,9] or C[K,3,3]")
    C = C.reshape(-1, 3, 3)
    if len(C) == 0 or not np.isfinite(C).all():
        raise ValueError("Center requires nonempty finite C")
    cross = np.cross(C[:, 1] - C[:, 0], C[:, 2] - C[:, 0])
    areas = np.sqrt(np.sum(cross * cross, axis=1, dtype=np.float32)) * np.float32(0.5)
    if not np.isfinite(areas).all() or np.any(areas <= np.float32(1e-12)):
        raise ValueError("C contains a degenerate/nonfinite face (same area floor as Geo)")
    centroids = np.mean(C, axis=1, dtype=np.float32)
    center = np.sum(areas[:, None] * centroids, axis=0, dtype=np.float32) / np.sum(areas, dtype=np.float32)
    if not np.isfinite(center).all():
        raise ValueError("Nonfinite C area center")
    return np.ascontiguousarray(center, dtype=np.float32)


area_center = area_weighted_center


def to_local(world, center):
    """FP32 world-to-local subtraction, retaining input shape and face order."""
    world = np.asarray(world)
    center = np.asarray(center)
    if world.dtype != np.float32 or center.dtype != np.float32 or center.shape != (3,):
        raise ValueError("Expected FP32 coordinates and FP32 center[3]")
    if world.shape[-1] not in (3, 9) or not np.isfinite(world).all() or not np.isfinite(center).all():
        raise ValueError("Malformed coordinates/center")
    return np.ascontiguousarray((world.reshape(-1, 3) - center).reshape(world.shape))


def world_trajectory(local_trajectory, C, center):
    """Translate free corners only; copy original C bytes in all 51 states."""
    local = np.asarray(local_trajectory)
    C = np.asarray(C)
    center = np.asarray(center)
    if local.dtype != np.float32 or local.ndim != 3 or local.shape[-1] != 9:
        raise ValueError("Expected FP32 local trajectory[states,N,9]")
    if C.dtype != np.float32 or C.ndim not in (2, 3) or C.shape[1:] not in ((9,), (3, 3)):
        raise ValueError("Expected FP32 source C")
    if center.dtype != np.float32 or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Expected finite FP32 center[3]")
    K = len(C)
    if not 0 < K < local.shape[1] or not np.isfinite(local).all() or not np.isfinite(C).all():
        raise ValueError("Malformed local trajectory or C")
    world = local.copy().reshape(len(local), local.shape[1], 3, 3)
    world[:, K:] = world[:, K:] + center
    world[:, :K] = C.reshape(K, 3, 3)[None]
    return np.ascontiguousarray(world.reshape(local.shape))


class CReferenceModel(NativeModel):
    """Same state/parameters as NativeModel, with explicit dual-coordinate IO."""

    def _embed_dual(self, x_local, geo_world, known_mask, valid_mask):
        batch, faces, _ = x_local.shape
        _, hidden = self.backbone.x_embedder(x_local)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(hidden.dtype)
        hidden = hidden + _corners(role)
        # Never cache/detach this: the inherited Geo remains fully trainable.
        delta = self.context_encoder(geo_world, known_mask, valid_mask,
                                     observer=self.context_observer)
        hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)
        return hidden, role

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, geo_world, observer=None):
        assert_native_runtime(self, coordinates=(x, t, geo_world))
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 local x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if geo_world.shape != x.shape or geo_world.device != x.device:
            raise ValueError("Explicit world-Geo coordinates must match local shape/device/slot order")
        if self.readout_mode != "none" or len(self.readouts) or self.context_encoder is None:
            raise ValueError("This pilot requires Native/T1 + Geo and no READOUT")
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected free t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all() and torch.isfinite(geo_world).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Free time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated separate bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
            raise ValueError("y must count real faces only, N128..256")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        hidden, role = self._embed_dual(x, geo_world, known_mask, valid_mask)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=observer, layer_index=index, face_time=face_time)
        output = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=observer, face_time=face_time)
        return output.view(batch, faces, 9)


def as_c_reference_model(model):
    """Promote an experiment-owned model in place; preserve all Parameter IDs.

    This adds no registered module prefix, so strict checkpoints and the cloned
    AdamW parameter ordering remain exactly those of the maintained NativeModel.
    The caller must not use the promoted object as a world-coordinate baseline.
    """
    if type(model) not in (NativeModel, CReferenceModel):
        raise TypeError("Expected the experiment-owned exact NativeModel")
    if model.readout_mode != "none" or len(model.readouts) or model.context_encoder is None:
        raise ValueError("C-reference pilot permits only existing Native/T1 + Geo")
    if type(model) is CReferenceModel:
        return model
    keys = tuple(model.state_dict())
    params = tuple((name, id(p)) for name, p in model.named_parameters())
    model.__class__ = CReferenceModel
    if tuple(model.state_dict()) != keys or tuple((name, id(p)) for name, p in model.named_parameters()) != params:
        raise RuntimeError("Promotion changed parameter/state identity")
    return model


def build_model(model_config, readout_mode="none", readout_seed=20261007, trainable=True):
    return as_c_reference_model(_build_native(model_config, readout_mode=readout_mode,
        readout_seed=readout_seed, trainable=trainable))


def sample(model, C, N, seed=None, z=None, *, budget, out=None, context=None):
    """Exactly Euler50 in local coordinates; export world RAW for unchanged D0.

    z is the original frozen standard Gaussian, not translated or demeaned.
    Returned trajectory and RAW are WORLD coordinates.  raw.npz additionally
    holds local_trajectory, center and local_C for exact local Euler auditing.
    The budget owner can record the returned BF16 velocity in budget.call.
    """
    if type(model) is not CReferenceModel:
        raise TypeError("Use an explicitly promoted CReferenceModel for local sampling")
    N = int(N)
    C = validate_C(C, N)
    K = len(C)
    center = area_weighted_center(C)
    local_C = to_local(C, center)
    ctx = context or {}
    if z is None:
        z = make_noise(seed, N, K)
    z = np.asarray(z, dtype=np.float32)
    if z.shape == (N - K, 3, 3):
        full = np.zeros((1, N, 9), np.float32)
        full[:, K:] = z.reshape(1, N - K, 9)
        z = full
    if z.shape != (1, N, 9) or not np.isfinite(z).all():
        raise ValueError("Malformed frozen local Gaussian")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Freeze model for sampling")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Real Euler50 sampling requires explicitly owned CUDA")
    budget.reserve("rollout_attempt", context=ctx)
    state = torch.from_numpy(z.copy()).to(device)
    known = torch.zeros((1, N), dtype=torch.bool, device=device)
    known[:, :K] = True
    valid = torch.ones_like(known)
    y = torch.tensor([N], device=device, dtype=torch.long)
    cond = torch.from_numpy(local_C.reshape(K, 9).copy()).to(device)
    geo_world = torch.zeros_like(state)
    geo_world[:, :K] = torch.from_numpy(C.reshape(K, 9).copy()).to(device)[None]
    state[:, :K] = cond[None]
    path = [state[0].cpu().numpy().copy()]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    with torch.no_grad(), native_execution(model, coordinates=(state, geo_world)):
        for step in range(50):
            t = torch.full((1,), step / 50, dtype=torch.float32, device=device)
            velocity = budget.call("sampling_forward",
                lambda: model(state, t, y, valid, known, geo_world=geo_world),
                context={**ctx, "euler_step": step})
            if velocity.dtype != torch.bfloat16 or velocity.shape != state.shape or not torch.isfinite(velocity).all():
                raise RuntimeError("Invalid BF16 velocity")
            next_state = state.clone()
            next_state[:, K:] = state[:, K:] + velocity.float()[:, K:] * (1 / 50)
            next_state[:, :K] = cond[None]
            state = next_state
            if not torch.isfinite(state).all() or not torch.equal(state[0, :K], cond):
                raise RuntimeError("Local Euler/C contract failed")
            path.append(state[0].cpu().numpy().copy())
    torch.cuda.synchronize()
    local = np.stack(path)
    trajectory = world_trajectory(local, C, center)
    raw = trajectory[-1].copy()
    if any(p[:K].tobytes() != C.tobytes() for p in trajectory):
        raise RuntimeError("World C bit pattern not preserved")
    if any(p[:K].tobytes() != local_C.tobytes() for p in local):
        raise RuntimeError("Local C bit pattern not preserved")
    if local[0, K:].tobytes() != z[0, K:].tobytes():
        raise RuntimeError("Local starting Gaussian changed")
    audit = dict(nfe=50, known_all51_bitwise=True, known_coordinates_bitwise=True,
        known_state_checks=51, local_known_all51_bitwise=True, N=N, K=K, seed=seed,
        seconds=time.perf_counter() - tick, readout_mode=model.readout_mode,
        model_inputs=["C_local", "C_world_for_existing_Geo", "N", "local_Gaussian", "time"],
        dtype="BF16 backbone/FP32 Geo/FP32 parameters+local integration+world restore",
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope="one sampling rollout, resident weights included",
        human_usability="NOT_RUN", request=ctx, frame_version=FRAME_VERSION,
        center=center.tolist(), center_source="augmented_C_only_FP32_area_weighted",
        center_is_time_independent=True, geometry_coordinates="original_world_C",
        backbone_coordinates="C_relative", integration_coordinates="C_relative",
        exported_coordinates="original_world", local_initial_noise_unchanged=True,
        world_initial_free="z_local + center", external_C="original byte copy",
        extra_npz_fields=["local_trajectory", "center", "local_C"])
    budget.event("ROLLOUT_COMPLETE", context=ctx,
        **{k: audit[k] for k in ("nfe", "seconds", "known_all51_bitwise")})
    if out:
        out = Path(out)
        write_raw(out, raw, C, z, trajectory, audit)
        # Extend this NEW output only; maintained writer and all older outputs
        # remain unchanged.  No completion record is issued before this returns.
        with np.load(out / "raw.npz", allow_pickle=False) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        fields.update(local_trajectory=local, center=center, local_C=local_C)
        np.savez_compressed(out / "raw.npz", **fields)
        with np.load(out / "raw.npz", allow_pickle=False) as saved:
            if any(saved[name].tobytes() != value.tobytes() for name, value in fields.items()):
                raise RuntimeError("Extended local/world export changed bytes")
        audit["local_world_export_bitwise"] = True
        (out / "run.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return raw, trajectory, audit
