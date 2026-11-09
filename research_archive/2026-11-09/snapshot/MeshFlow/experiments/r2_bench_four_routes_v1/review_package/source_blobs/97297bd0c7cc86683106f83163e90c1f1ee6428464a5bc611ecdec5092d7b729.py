"""Corner-time adapter for a supplied, unmodified NativeModel.

This module never constructs or loads a generator.  Call ``seam_forward`` only
inside the original ``native_execution`` context.  C faces alone enter Geo and
receive known-time face attention; seam anchor corners receive known roles and
known-time MLP/final modulation.  Attention on every non-C face remains at t.
Clamping/integration and indexed decoding remain the caller's responsibility.

The adapter adds no parameters: restore the generator's matching AdamW state.
The separate occupancy head must have a fresh, independent optimizer state.
"""
from __future__ import annotations

import operator

from meshflow_control import runtime as _stable  # policy before torch import
import torch
from meshflow_control.models.geometry import inject_context
from meshflow_control.models.native import assert_native_runtime, _corners, _face_select


def _corner_select(free, known, corner_known):
    """Select [B,D] modulation for each of the B x N x 3 corners."""
    selected = torch.where(
        corner_known[..., None], known[:, None, None, :], free[:, None, None, :])
    return selected.reshape(selected.shape[0], -1, selected.shape[-1])


def _validate_corner_masks(x, valid_mask, C_face_mask, corner_known):
    if x.ndim != 3 or x.shape[-1] != 9 or x.dtype != torch.float32:
        raise ValueError("Expected FP32 x[B,padded_N,9]")
    shape = x.shape[:2]
    for name, value, wanted in (
        ("valid_mask", valid_mask, shape),
        ("C_face_mask", C_face_mask, shape),
        ("corner_known", corner_known, (*shape, 3)),
    ):
        if (value.dtype != torch.bool or value.shape != wanted
                or value.device != x.device):
            raise ValueError(f"{name} must be colocated boolean{wanted}")
    if bool((C_face_mask & ~valid_mask).any()):
        raise ValueError("C faces must be valid")
    if bool((corner_known & ~valid_mask[..., None]).any()):
        raise ValueError("Padded corners cannot be known")
    if bool((C_face_mask[..., None] & ~corner_known).any()):
        raise ValueError("Every C corner must be known")


def seam_forward(model, x, t, y, valid_mask, C_face_mask, corner_known):
    """Reuse original Native parameters with mixed known/free seam corners.

    ``corner_known`` is bool[B,N,3], independent of the C-face context mask.
    The unmodified no-seam branch is a literal call to the original model, so
    it preserves that model's arithmetic and validation rather than emulating
    a supposedly equivalent forward. No production forward is run by tests.
    """
    _validate_corner_masks(x, valid_mask, C_face_mask, corner_known)
    ordinary = C_face_mask[..., None].expand_as(corner_known)
    if torch.equal(corner_known, ordinary):
        return model(x, t, y, valid_mask=valid_mask, known_mask=C_face_mask)

    assert_native_runtime(model, coordinates=(x, t))
    if getattr(model, "readout_mode", "none") != "none" or len(getattr(model, "readouts", {})):
        raise ValueError("This prototype accepts Native/T1 + role + Geo without READOUT")
    batch, faces, _ = x.shape
    if (t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64
            or t.device != x.device or y.device != x.device):
        raise ValueError("Colocated free t[B] and integer total N[B] required")
    if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
        raise ValueError("Nonfinite coordinates/time")
    if bool(((t < 0) | (t > 1)).any()):
        raise ValueError("Free time outside [0,1]")
    if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
        raise ValueError("y must count real faces only, N128..256")
    if bool((C_face_mask.sum(1) == 0).any()):
        raise ValueError("Every sample requires original C faces")
    if bool(((valid_mask[..., None] & ~corner_known).sum((1, 2)) == 0).any()):
        raise ValueError("Every sample requires a free coordinate")

    backbone = model.backbone
    _, hidden = backbone.x_embedder(x)
    hidden = hidden.view(batch, faces * 3, -1)
    role = model.role_embedding[corner_known.long()].to(hidden.dtype)
    hidden = hidden + role.reshape_as(hidden)
    if model.context_encoder is not None:
        delta = model.context_encoder(
            x, C_face_mask, valid_mask, observer=model.context_observer)
        hidden = inject_context(hidden, delta, C_face_mask, observer=model.context_observer)

    c_free = backbone.t_embedder(t)
    c_known = backbone.t_embedder(torch.ones_like(t))
    count = backbone.y_embedder(y // backbone.face_bin, False)
    c_free, c_known = c_free + count, c_known + count
    for block in backbone.layers:
        free_mods = block.adaLN_modulation(c_free).chunk(6, dim=1)
        known_mods = block.adaLN_modulation(c_known).chunk(6, dim=1)
        shift_a, scale_a, gate_a = (
            _face_select(f, k, C_face_mask)
            for f, k in zip(free_mods[:3], known_mods[:3]))
        shift_m, scale_m, gate_m = (
            _corner_select(f, k, corner_known)
            for f, k in zip(free_mods[3:], known_mods[3:]))
        face_hidden = hidden.reshape(batch, faces, 3, -1).mean(dim=2)
        face_hidden = gate_a * block.attn(
            block.norm1(face_hidden) * (1 + scale_a) + shift_a, mask=valid_mask)
        hidden = hidden + _corners(face_hidden)
        hidden = hidden + gate_m * block.mlp(
            block.norm2(hidden) * (1 + scale_m) + shift_m, mask=valid_mask)

    final = backbone.final_layer
    free_mods = final.adaLN_modulation(c_free).chunk(2, dim=1)
    known_mods = final.adaLN_modulation(c_known).chunk(2, dim=1)
    shift, scale = (
        _corner_select(f, k, corner_known) for f, k in zip(free_mods, known_mods))
    hidden = final.norm_final(hidden) * (1 + scale) + shift
    return final.linear(hidden).view(batch, faces, 9)


def corner_masked_fm_loss(velocity, target, valid_mask, corner_known, *, denominator=None):
    """Original summed squared FM error, excluding every anchored coordinate.

    Pass the sum of all effective-batch free scalar coordinates as denominator
    when accumulating microbatches.  Known corners and padding never enter
    either the numerator or default denominator.  No reweighting is introduced.
    """
    if velocity.shape != target.shape or velocity.ndim != 3 or velocity.shape[-1] != 9:
        raise ValueError("Velocity and target must share [B,N,9]")
    if (valid_mask.shape != velocity.shape[:2] or valid_mask.dtype != torch.bool
            or corner_known.shape != (*velocity.shape[:2], 3)
            or corner_known.dtype != torch.bool):
        raise ValueError("Boolean face-valid and corner-known masks required")
    if any(v.device != velocity.device for v in (target, valid_mask, corner_known)):
        raise ValueError("FM arrays must be colocated")
    mask = valid_mask[..., None] & ~corner_known
    count = int(mask.sum()) * 3
    divisor = count if denominator is None else operator.index(denominator)
    if count <= 0 or divisor < count:
        raise ValueError("Invalid effective-batch free-coordinate denominator")
    difference = (velocity.float() - target.float()).reshape(*mask.shape, 3)
    return difference[mask].square().sum() / divisor


masked_corner_fm_loss = corner_masked_fm_loss


def cpu_contract_checks():
    """Small tensor algebra/autograd checks; zero generator construction/calls."""
    valid = torch.tensor([[True, True, True, False]])
    known = torch.tensor([[[True, True, True], [True, True, False],
                           [False, False, False], [False, False, False]]])
    prediction = torch.arange(36, dtype=torch.float32).reshape(1, 4, 9).requires_grad_()
    target = torch.zeros_like(prediction)
    loss = corner_masked_fm_loss(prediction, target, valid, known)
    loss.backward()
    active = (valid[..., None] & ~known)[..., None].expand(1, 4, 3, 3).reshape_as(prediction)
    assert torch.equal(prediction.grad[~active], torch.zeros_like(prediction.grad[~active]))
    assert bool((prediction.grad[active] != 0).all())
    assert torch.equal(loss.detach(), prediction.detach()[active].square().sum() / 12)
    external = corner_masked_fm_loss(prediction, target, valid, known, denominator=24)
    assert torch.equal(external.detach(), loss.detach() / 2)
    free, context = torch.tensor([[1., 2.]]), torch.tensor([[3., 4.]])
    selected = _corner_select(free, context, known)
    assert torch.equal(selected[0, 3], context[0])
    assert torch.equal(selected[0, 5], free[0])
    face_mask = torch.tensor([[True, False, False, False]])
    face_only = _face_select(free, context, face_mask)
    assert torch.equal(face_only[0, 1], free[0])
    return dict(status="PASS", generator_constructions=0, generator_forwards=0,
                backward_scope="12 active toy scalar coordinates only",
                masked_scalar_coordinates=24, active_scalar_coordinates=12,
                known_and_padding_gradient_zero=True,
                effective_batch_denominator_checked=True,
                production_mixed_corner_forward_executed=False,
                no_seam_original_forward_equivalence="STATIC_DIRECT_DELEGATION_ONLY")
