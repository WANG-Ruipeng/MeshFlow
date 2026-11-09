"""Parameter-free ASYNC_LEAD_SEAM contract, local-velocity formulation.

The only changed mechanism is the seam-tip clock s(t)=min(t/tau,1).
Targets keep their original local-time units U=Y-Z_H. Active-tip and bulk
errors share the ORIGINAL structural-free scalar denominator; inactive tips
have no direct velocity loss. The physical ODE uses dX_tip/dt=s'(t)*v_tip.

Never turn a generated tip into an input-C identity or pass it into Geo.
No model is constructed, loaded or executed by importing this module.
"""
from __future__ import annotations

import json
import math
import operator
from pathlib import Path
import time

from meshflow_control import runtime as _runtime  # establish policy before torch
import numpy as np
import torch

from meshflow_control.models.geometry import inject_context
from meshflow_control.models.native import assert_native_runtime, native_execution, _corners
from meshflow_control.seam.model_adapter import (
    seam_forward, corner_masked_fm_loss, _validate_corner_masks,
)

TAU = 0.7
SCHEMA = "ASYNC_LEAD_SEAM_LOCAL_VELOCITY_V1"


def _tau(tau):
    tau = float(tau)
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must be finite and in (0,1]")
    return tau


def seam_clock(t, tau=TAU):
    """Same FP32 clock for NumPy preparation and torch forward."""
    tau = _tau(tau)
    if isinstance(t, torch.Tensor):
        if not t.is_floating_point():
            raise ValueError("Clock requires a floating-point time tensor")
        if not bool(torch.isfinite(t).all()) or bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Time outside [0,1]")
        return (t / t.new_tensor(tau)).clamp(max=1)
    a = np.asarray(t, np.float32)
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("Time outside [0,1]")
    return np.minimum(a / np.float32(tau), np.float32(1))


def mask_partition(valid, C_mask, fixed):
    """Return structural free, seam faces, tip corners and bulk corners.

    Supports reordered faces and arbitrary positions of the one free seam
    corner. The current owner-only contract permits 0/2/3 fixed corners:
    zero=bulk, two=seam, three=original C. Padding never contributes.
    """
    if (valid.dtype != torch.bool or C_mask.dtype != torch.bool or fixed.dtype != torch.bool
            or C_mask.shape != valid.shape or fixed.shape != (*valid.shape, 3)):
        raise ValueError("Expected bool[B,N] valid/C and bool[B,N,3] fixed")
    if any(v.device != valid.device for v in (C_mask, fixed)):
        raise ValueError("Masks must be colocated")
    if bool((C_mask & ~valid).any()) or bool((fixed & ~valid[..., None]).any()):
        raise ValueError("Invalid known/padding relation")
    if bool((C_mask[..., None] & ~fixed).any()):
        raise ValueError("Original C must have all three fixed corners")
    count = fixed.sum(-1)
    if bool((valid & ~C_mask & (count != 0) & (count != 2)).any()):
        raise ValueError("Unsupported non-C corner layout")
    seam = valid & ~C_mask & (count == 2)
    free = valid[..., None] & ~fixed
    tips = seam[..., None] & free
    bulk = free & ~tips
    return free, seam, tips, bulk


def transform_sample(sample, tau=TAU):
    """Transform one already-matched sample; do not mutate any input array.

    Existing x0, x1, u, OT maps, permutations, source IDs and the original
    free_coordinate_count remain unchanged. Parent code may continue to call
    its existing ``tensors(sample)`` converter. Extra keys are audit metadata.
    """
    tau = _tau(tau)
    t = np.float32(np.asarray(sample["t"]).item())
    s = np.float32(seam_clock(t, tau))
    valid = np.asarray(sample["valid_mask"], bool)
    known = np.asarray(sample["known_mask"], bool)
    fixed = np.asarray(sample["anchor_ids"]) >= 0
    n = int(sample["N"])
    if valid.shape != (n,) or known.shape != (n,) or fixed.shape != (n, 3):
        raise ValueError("Sample mask shape differs")
    if np.any(known & ~valid) or np.any(fixed & ~valid[:, None]) or np.any(known[:, None] & ~fixed):
        raise ValueError("Invalid source masks")
    count = fixed.sum(1)
    if np.any(valid & ~known & (count != 0) & (count != 2)):
        raise ValueError("Unsupported non-C corner layout")
    free = valid[:, None] & ~fixed
    seam = valid & ~known & (count == 2)
    tips = seam[:, None] & free
    scalar_count = int(free.sum()) * 3
    if scalar_count != int(sample["free_coordinate_count"]) or scalar_count <= 0:
        raise ValueError("Original free-coordinate denominator differs")
    result = dict(sample)
    active = free.copy()
    if tau != 1 and seam.any():
        x0 = np.asarray(sample["x0"], np.float32)
        x1 = np.asarray(sample["x1"], np.float32)
        if x0.shape != (n, 3, 3) or x1.shape != x0.shape:
            raise ValueError("Expected sample endpoints [N,3,3]")
        xt = np.asarray(sample["xt"], np.float32).copy()
        xt[tips] = ((np.float32(1)-s)*x0[tips] + s*x1[tips])
        xt[fixed] = x0[fixed]
        result["xt"] = xt
        if t >= np.float32(tau):
            active[tips] = False
    result.update(async_schema=SCHEMA, async_tau=np.asarray(tau, np.float32),
                  seam_time=np.asarray(s, np.float32), seam_face_mask=seam,
                  seam_tip_mask=tips, active_free_corner_mask=active,
                  original_free_scalar_count=np.asarray(scalar_count, np.int64),
                  async_active_tip_scalar_count=np.asarray(int((active & tips).sum())*3, np.int64),
                  async_late_tip_teacher_forced=np.asarray(bool(t >= np.float32(tau) and seam.any() and tau != 1)))
    return result


def _select_face(bulk_value, known_value, seam_value, C_mask, seam):
    return torch.where(C_mask[..., None], known_value[:, None],
                       torch.where(seam[..., None], seam_value[:, None], bulk_value[:, None]))


def _select_corner(bulk_value, known_value, seam_value, fixed, tips):
    selected = torch.where(fixed[..., None], known_value[:, None, None],
                           torch.where(tips[..., None], seam_value[:, None, None],
                                       bulk_value[:, None, None]))
    return selected.reshape(selected.shape[0], -1, selected.shape[-1])


def forward(model, x, t, y, valid_mask, C_face_mask, corner_known, *, tau=TAU):
    """Drop-in seam_forward signature; time changes, parameter/role/Geo do not."""
    tau = _tau(tau)
    _validate_corner_masks(x, valid_mask, C_face_mask, corner_known)
    _, seam, tips, _ = mask_partition(valid_mask, C_face_mask, corner_known)
    if tau == 1 or not bool(seam.any()):
        return seam_forward(model, x, t, y, valid_mask, C_face_mask, corner_known)
    assert_native_runtime(model, coordinates=(x, t))
    if getattr(model, "readout_mode", "none") != "none" or len(getattr(model, "readouts", {})):
        raise ValueError("Async accepts Native/T1 + role + Geo without READOUT")
    batch, faces, _ = x.shape
    if (t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64
            or t.device != x.device or y.device != x.device):
        raise ValueError("Expected colocated t[B] and int64 total N[B]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite coordinates")
    s = seam_clock(t, tau)
    if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
        raise ValueError("Real total N must lie in N128..256")
    if bool((C_face_mask.sum(1) == 0).any()):
        raise ValueError("Original input C is required")

    backbone = model.backbone
    _, hidden = backbone.x_embedder(x)
    hidden = hidden.view(batch, faces * 3, -1)
    role = model.role_embedding[corner_known.long()].to(hidden.dtype)
    hidden = hidden + role.reshape_as(hidden)
    if model.context_encoder is not None:
        delta = model.context_encoder(x, C_face_mask, valid_mask, observer=model.context_observer)
        hidden = inject_context(hidden, delta, C_face_mask, observer=model.context_observer)
    c_bulk = backbone.t_embedder(t)
    c_known = backbone.t_embedder(torch.ones_like(t))
    c_seam = backbone.t_embedder(s)
    count = backbone.y_embedder(y // backbone.face_bin, False)
    c_bulk, c_known, c_seam = c_bulk + count, c_known + count, c_seam + count
    for block in backbone.layers:
        bulk_mod = block.adaLN_modulation(c_bulk).chunk(6, dim=1)
        known_mod = block.adaLN_modulation(c_known).chunk(6, dim=1)
        seam_mod = block.adaLN_modulation(c_seam).chunk(6, dim=1)
        shift_a, scale_a, gate_a = (
            _select_face(b, k, s0, C_face_mask, seam)
            for b, k, s0 in zip(bulk_mod[:3], known_mod[:3], seam_mod[:3]))
        shift_m, scale_m, gate_m = (
            _select_corner(b, k, s0, corner_known, tips)
            for b, k, s0 in zip(bulk_mod[3:], known_mod[3:], seam_mod[3:]))
        face_hidden = hidden.reshape(batch, faces, 3, -1).mean(dim=2)
        face_hidden = gate_a * block.attn(
            block.norm1(face_hidden) * (1 + scale_a) + shift_a, mask=valid_mask)
        hidden = hidden + _corners(face_hidden)
        hidden = hidden + gate_m * block.mlp(
            block.norm2(hidden) * (1 + scale_m) + shift_m, mask=valid_mask)
    final = backbone.final_layer
    bulk_mod = final.adaLN_modulation(c_bulk).chunk(2, dim=1)
    known_mod = final.adaLN_modulation(c_known).chunk(2, dim=1)
    seam_mod = final.adaLN_modulation(c_seam).chunk(2, dim=1)
    shift, scale = (
        _select_corner(b, k, s0, corner_known, tips)
        for b, k, s0 in zip(bulk_mod, known_mod, seam_mod))
    hidden = final.norm_final(hidden) * (1 + scale) + shift
    return final.linear(hidden).view(batch, faces, 9)


def active_masks(b, tau=TAU):
    """b uses existing tensors(sample) keys: t/valid/known/fixed."""
    tau = _tau(tau)
    free, seam, tips, bulk = mask_partition(b["valid"], b["known"], b["fixed"])
    if b["t"].shape != (free.shape[0],):
        raise ValueError("Expected one time per sample")
    seam_clock(b["t"], tau)  # bounds check
    active_tips = tips if tau == 1 else tips & (b["t"] < b["t"].new_tensor(tau))[:, None, None]
    return free, seam, tips, bulk, active_tips, bulk | active_tips


def loss_and_stats(pred, b, *, denominator=None, tau=TAU):
    """Return (loss, JSON-safe statistics); b is the old tensors(sample) dict.

    ``denominator`` must be the ORIGINAL effective-batch free scalar count.
    all_free_* is retained as an alias for the active numerator/original
    denominator, and is explicitly marked NONCOMPARABLE_TO_BASELINE_TOTAL.
    """
    tau = _tau(tau)
    if pred.shape != b["target"].shape or pred.ndim != 3 or pred.shape[-1] != 9:
        raise ValueError("Prediction and original local target must be [B,N,9]")
    free, seam, tips, bulk, active_tips, active = active_masks(b, tau)
    original = int(free.sum()) * 3
    divisor = original if denominator is None else operator.index(denominator)
    if original <= 0 or divisor < original:
        raise ValueError("Original effective-batch denominator differs")
    error = (pred.float() - b["target"].float()).reshape(*free.shape, 3)
    if tau == 1 or not bool(seam.any()):
        loss = corner_masked_fm_loss(pred, b["target"], b["valid"], b["fixed"], denominator=divisor)
    else:
        loss = error[active].square().sum() / divisor
    with torch.no_grad():
        squared = error.detach().double().square()
        sum_error = lambda m: float(squared[m].sum().cpu())
        bulk_sse, tip_sse = sum_error(bulk), sum_error(active_tips)
        active_sse = bulk_sse + tip_sse
        stats = dict(
            schema=SCHEMA, tau=tau,
            all_free_velocity_sse=active_sse, all_free_scalar_count=original,
            total_loss_comparison="NONCOMPARABLE_TO_BASELINE_TOTAL",
            active_local_velocity_sse=active_sse, original_free_scalar_count=original,
            active_free_scalar_count=int(active.sum())*3,
            seam_tip_velocity_sse=tip_sse, seam_tip_scalar_count=int(active_tips.sum())*3,
            structural_seam_tip_scalar_count=int(tips.sum())*3,
            inactive_seam_tip_scalar_count=int((tips & ~active_tips).sum())*3,
            bulk_velocity_sse=bulk_sse, bulk_scalar_count=int(bulk.sum())*3,
            equivalent_active_global_velocity_sse=bulk_sse + tip_sse/(tau*tau),
            effective_batch_original_denominator=divisor,
            tip_error_scope="ACTIVE_LOCAL_TIME_ONLY; LATE_TEACHER_FORCED_TIPS_EXCLUDED",
        )
    return loss, stats


def endpoint_prediction(pred, b, *, tau=TAU):
    """Local-velocity one-step probe; late training tip is a tautology, not fit."""
    _, _, tips, _, _, _ = active_masks(b, tau)
    t = b["t"]
    s = seam_clock(t, tau)
    remaining = (1-t)[:, None, None].expand_as(tips)
    remaining = torch.where(tips, (1-s)[:, None, None], remaining)
    remaining = torch.where(b["fixed"], torch.zeros_like(remaining), remaining)
    x = b["x"].reshape(*tips.shape, 3)
    return (x + remaining[..., None]*pred.float().reshape_as(x)).reshape_as(b["x"])


def integration_step(state, velocity, anchor_values, valid, C_mask, fixed,
                     step, *, steps=50, tau=TAU):
    """One FP32 Euler step on [B,N,3,3]; zero-Delta-s tips are bitwise retained."""
    tau = _tau(tau)
    step, steps = operator.index(step), operator.index(steps)
    if steps <= 0 or not 0 <= step < steps:
        raise ValueError("Invalid registered integration step")
    if state.dtype != torch.float32 or state.shape != (*valid.shape, 3, 3):
        raise ValueError("Expected FP32 [B,N,3,3] integration state")
    if anchor_values.shape != state.shape or velocity.numel() != state.numel():
        raise ValueError("Velocity/anchor shape differs")
    free, seam, tips, bulk = mask_partition(valid, C_mask, fixed)
    v = velocity.float().reshape_as(state)
    nxt = state.clone()
    dt = 1/steps
    if tau == 1 or not bool(seam.any()):
        nxt[free] = state[free] + v[free]*dt
    else:
        nxt[bulk] = state[bulk] + v[bulk]*dt
        # Piecewise-clock difference handles a step crossing tau exactly.
        ds = min((step+1)/steps/tau, 1.) - min(step/steps/tau, 1.)
        if ds > 0:
            nxt[tips] = state[tips] + v[tips]*ds
    nxt[fixed] = anchor_values[fixed]
    return nxt


def _event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(event=event, unix=time.time(), **fields), allow_nan=False)+"\n")
        stream.flush()


def rollout(model, layout, noise, arm, ledger, *, tau=TAU):
    """Old rollout interface; caller owns dispatch, weights, noise and outputs.

    This is the only GPU entry in the module and runs only when explicitly
    called. Exactly 50 generator calls; no teacher, source Y, or extra noise.
    """
    tau = _tau(tau)
    if arm not in ("SEAM", "ASYNC", "ASYNC_LEAD_SEAM"):
        raise ValueError("Only the registered asynchronous SEAM arm is supported")
    n, k = int(layout["N"]), int(layout["K"])
    if np.asarray(noise).shape != (n-k, 3, 3) or np.asarray(noise).dtype != np.float32:
        raise ValueError("Expected original frozen FP32 noise")
    fixed_np = np.asarray(layout["anchor_ids"]) >= 0
    initial = np.zeros((n, 3, 3), np.float32)
    initial[k:] = noise
    initial[fixed_np] = layout["C_vertices"][layout["anchor_ids"][fixed_np]]
    state = torch.from_numpy(initial[None]).cuda()
    anchors = state.clone()
    fixed = torch.from_numpy(fixed_np[None]).cuda()
    C_mask = torch.from_numpy(layout["known_mask"][None]).cuda()
    valid = torch.ones_like(C_mask)
    y = torch.tensor([n], dtype=torch.int64, device=state.device)
    _, seam, tips, _ = mask_partition(valid, C_mask, fixed)
    trajectory = [initial.copy()]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    frozen_tips = None
    with torch.no_grad(), native_execution(model, coordinates=state):
        for step in range(50):
            t = torch.full((1,), step/50, dtype=torch.float32, device=state.device)
            _event(ledger, "FORWARD_STARTED", step=step, schema=SCHEMA)
            v = forward(model, state.reshape(1, n, 9), t, y, valid, C_mask, fixed, tau=tau)
            torch.cuda.synchronize()
            _event(ledger, "FORWARD_COMPLETE", step=step, schema=SCHEMA)
            if not bool(torch.isfinite(v).all()):
                raise FloatingPointError("Nonfinite async velocity")
            state = integration_step(state, v, anchors, valid, C_mask, fixed, step, tau=tau)
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError("Nonfinite async state")
            if (step+1)/50 >= tau:
                if frozen_tips is None:
                    frozen_tips = state[tips].clone()
                elif not torch.equal(state[tips], frozen_tips):
                    raise ValueError("Generated tip moved after local-clock completion")
            trajectory.append(state[0].cpu().numpy().copy())
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise RuntimeError("Unexpected sampling RNG")
    trajectory = np.stack(trajectory)
    if any(not np.array_equal(a[fixed_np], initial[fixed_np]) for a in trajectory):
        raise ValueError("Original anchor moved")
    return trajectory[-1], trajectory, dict(
        schema=SCHEMA, tau=tau, generator_forwards=50, seconds=time.perf_counter()-tick,
        anchored_all51_exact=True, tips_frozen_after_global_step=math.ceil(50*tau),
        tip_freeze_value="ACTUAL_GENERATED_STATE; NO_SOURCE_Y",
        frozen_tip_coordinates_exact=True, active_seam_faces=int(seam.sum()),
        velocity_units="LOCAL_TIME; INTEGRATION_USES_DELTA_S",
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
