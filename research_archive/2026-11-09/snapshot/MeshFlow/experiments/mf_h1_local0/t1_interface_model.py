"""Known-corner boundary features for the bounded Native T1 continuation.

The only new parameter is a zero-initialized Linear(2, 768, bias=False).
No target geometry, source face IDs, projection, hooks, or geometry cache enter
this model. Native T1's official submodules and arithmetic order are retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from native_t1.model import (
    NativeInpaintingModel, assert_native_runtime, _corners,
    native_block_forward, native_final_forward,
)
import torch
from torch import nn


_CONDITION = ContextVar("t1_interface_boundary_condition", default=None)


def _validate_phi(phi, batch, device):
    if not isinstance(phi, torch.Tensor):
        raise ValueError("Boundary conditioning requires an explicit phi tensor")
    if phi.dtype != torch.float32 or phi.shape != (batch, 112, 3, 2):
        raise ValueError("Boundary phi must be FP32[B,112,3,2]")
    if phi.device != device:
        raise ValueError("Boundary phi and model coordinates must use the same device")
    if not bool(torch.isfinite(phi).all()):
        raise ValueError("Boundary phi must be finite")


class BoundaryNativeT1(NativeInpaintingModel):
    """Native T1 with an additive known-corner input feature.

Use ``boundary_phi=phi`` on individual training forwards. For the unchanged
five-positional-argument sampler call, surround the complete sampling call by
``with model.boundary_condition(phi):``. The condition is recomputed through
the trainable linear map on every forward; only the fixed input phi is held.
"""

    def __init__(self, backbone, trainable=True):
        super().__init__(backbone, trainable=trainable)
        device = self.role_embedding.device
        # Linear's ordinary initializer samples random numbers. Construct its
        # throw-away initialization on meta, materialize empty storage, then
        # zero it. Neither the CPU nor any CUDA RNG stream is consumed.
        self.boundary_proj = nn.Linear(2, 768, bias=False, device="meta", dtype=torch.float32)
        self.boundary_proj.to_empty(device=device)
        with torch.no_grad():
            self.boundary_proj.weight.zero_()
        self.boundary_proj.requires_grad_(self.experiment_trainable)
        self.eval()

    @contextmanager
    def boundary_condition(self, phi, *, injection_observer=None):
        """Bind one immutable fixed phi; restore the context even on failure.

        Phi is neither detached nor encoded here. This context must be closed
        before reusing a model for a different batch or conditioning sample.
        The optional observer is intended only for predeclared diagnostic steps.
        """
        if _CONDITION.get() is not None:
            raise RuntimeError("Boundary condition contexts cannot be nested")
        batch = phi.shape[0] if isinstance(phi, torch.Tensor) and phi.ndim else -1
        _validate_phi(phi, batch, self.role_embedding.device)
        version = phi._version
        token = _CONDITION.set((self, phi, version, injection_observer))
        try:
            yield self
            if phi._version != version:
                raise RuntimeError("Bound phi was mutated during boundary conditioning")
        finally:
            _CONDITION.reset(token)

    def _resolve_condition(self, boundary_phi, injection_observer, batch, device):
        bound = _CONDITION.get()
        if boundary_phi is not None:
            if bound is not None:
                raise RuntimeError("Use explicit phi or the condition context, not both")
            phi = boundary_phi
        else:
            if bound is None or bound[0] is not self:
                raise RuntimeError("BoundaryNativeT1 forward requires boundary_phi or its condition context")
            _, phi, version, bound_observer = bound
            if phi._version != version:
                raise RuntimeError("Bound phi was mutated before forward")
            if injection_observer is not None and bound_observer is not None:
                raise RuntimeError("Choose one explicit or context injection observer")
            if injection_observer is None:
                injection_observer = bound_observer
        _validate_phi(phi, batch, device)
        return phi, injection_observer

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None,
                boundary_phi=None, injection_observer=None):
        assert_native_runtime(self, coordinates=(x, t))
        if x.ndim != 3 or x.shape[1:] != (112, 9):
            raise ValueError('Native task requires x[B,112,9]')
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,):
            raise ValueError('Global free time and total face counts must have shape [B]')
        if not bool((y == 112).all()):
            raise ValueError('The face condition is total N=112, never N_free')
        if valid_mask is None or known_mask is None:
            raise ValueError('Separate valid_mask and known_mask are required')
        if (valid_mask.shape != (batch, faces) or known_mask.shape != (batch, faces)
                or valid_mask.dtype != torch.bool or known_mask.dtype != torch.bool):
            raise ValueError('Both masks must be bool[B,112]')
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError('Known faces must be valid attention context')
        if not bool(valid_mask.all()):
            raise ValueError('All 112 faces are valid in this registered experiment')
        phi, injection_observer = self._resolve_condition(
            boundary_phi, injection_observer, batch, x.device)
        _, hidden = self.backbone.x_embedder(x)
        hidden = hidden.view(batch, faces * 3, -1)
        role = self.role_embedding[known_mask.long()].to(dtype=hidden.dtype)
        hidden = hidden + _corners(role)
        # The sole new arithmetic site: after coordinate embedding plus role,
        # before the original per-face time conditioning and every DiT block.
        projected = self.boundary_proj(phi)
        boundary = (projected * known_mask[:, :, None, None].to(projected.dtype)).reshape(batch, faces * 3, 768)
        before_boundary = hidden
        hidden = hidden + boundary.to(dtype=hidden.dtype)
        if injection_observer is not None:
            injection_observer(dict(site='boundary_injection', hidden_before=before_boundary,
                projected=projected, boundary=boundary, hidden_after=hidden,
                known_mask=known_mask, phi=phi))
        # Two original embedder calls preserve official GEMM dimensions.
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site='embedding', hidden=hidden, role=role,
                          known_mask=known_mask, valid_mask=valid_mask,
                          face_time=face_time, c_free=c_free, c_known=c_known))
        for index, block in enumerate(self.backbone.layers, 1):
            hidden = native_block_forward(
                block, hidden, c_free, c_known, known_mask, valid_mask,
                observer=observer, layer_index=index, face_time=face_time)
        result = native_final_forward(
            self.backbone.final_layer, hidden, c_free, c_known, known_mask,
            observer=observer, face_time=face_time)
        output = result.view(batch, faces, 9)
        if injection_observer is not None:
            injection_observer(dict(site='boundary_output', output=output))
        return output


def add_boundary_model(base_native):
    """Transfer the backbone and copy role values; add exactly 1536 zeros.

    The original wrapper must be discarded after this transfer: the returned
    wrapper owns the same backbone object. All original state values remain
    unchanged, the small role Parameter is copied, and no RNG is advanced.
    This helper does not load a checkpoint, transfer device, or run a forward.
    """
    if type(base_native) is not NativeInpaintingModel:
        raise TypeError("Expected an unmodified NativeInpaintingModel wrapper")
    if base_native.role_embedding.dtype != torch.float32:
        raise ValueError("Expected FP32 common parameters")
    result = BoundaryNativeT1(base_native.backbone, trainable=base_native.experiment_trainable)
    with torch.no_grad():
        result.role_embedding.copy_(base_native.role_embedding)
    return result


class BoundaryInjectionStats:
    """Read-only, detached scalar diagnostics for a few scheduled forwards.

    Passing this collector never triggers another model evaluation or backward.
    Supply an already computed reference_output for a same-input velocity
    comparison; without it, output difference is explicitly NOT_RUN.
    Records contain only scalars/strings, never live autograd tensors.
    """

    def __init__(self, *, tag=None, reference_output=None):
        self.tag = tag
        self.reference_output = None if reference_output is None else reference_output.detach()
        self.records = []

    @staticmethod
    def _rms(value):
        return float(value.detach().float().square().mean().sqrt()) if value.numel() else None

    @torch.no_grad()
    def __call__(self, event):
        site = event['site']
        if site == 'boundary_injection':
            before, after, boundary = event['hidden_before'], event['hidden_after'], event['boundary']
            known_corners = event['known_mask'].repeat_interleave(3, dim=1)
            effective = after.float() - before.float()
            nonzero = boundary != 0
            n = int(nonzero.sum())
            before_rms = self._rms(before)
            boundary_rms = self._rms(boundary)
            self.records.append(dict(site=site, tag=self.tag,
                hidden_before_dtype=str(before.dtype), projection_dtype=str(event['projected'].dtype),
                boundary_dtype=str(boundary.dtype), hidden_after_dtype=str(after.dtype),
                next_native_block_input_dtype=str(after.dtype),
                hidden_RMS=before_rms, boundary_RMS=boundary_rms,
                boundary_to_hidden_RMS=None if before_rms == 0 else boundary_rms / before_rms,
                known_corner_boundary_RMS=self._rms(boundary[known_corners]),
                free_corner_boundary_RMS=self._rms(boundary[~known_corners]),
                effective_RMS=self._rms(effective),
                rounding_difference_RMS=self._rms(effective - boundary.float()),
                nonzero_boundary_elements=n,
                unchanged_among_nonzero=None if n == 0 else float(((after == before) & nonzero).sum()) / n))
        elif site == 'boundary_output':
            output = event['output']
            record = dict(site=site, tag=self.tag, output_dtype=str(output.dtype),
                          output_RMS=self._rms(output), reference_difference_status='NOT_RUN')
            if self.reference_output is not None:
                reference = self.reference_output
                if reference.shape != output.shape or reference.device != output.device:
                    raise ValueError('Reference output must match the actual output shape and device')
                difference = output.float() - reference.float()
                record.update(reference_difference_status='OBSERVED_FROM_SUPPLIED_OUTPUT',
                    output_difference_RMS=self._rms(difference),
                    output_difference_max_abs=float(difference.abs().max()),
                    output_equal_reference=bool(torch.equal(output, reference)))
            self.records.append(record)
        else:
            raise ValueError(f'Unexpected injection statistics site: {site}')
