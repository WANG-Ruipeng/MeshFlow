"""Two trained shared-core passes with unchanged Native parameter identities.

This is a bounded deep-supervision experiment, not a LiFT implementation.
The original twelve-layer pass is preserved as v1.  For v2, zero-based layers
8..11 are applied again with their full residual updates, using the same t/N/C.
Known token states are refreshed from this call's core entrance (after layer 7);
that reference retains autograd and is not claimed to be a C-only encoding.
No public model function, weights, or module globals are patched.
"""
from __future__ import annotations

import torch

from r2.pilots.bench_natural_n_recipe_v1 import runtime as native
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, assert_native_runtime, native_block_forward,
    native_final_forward, native_execution,
)
from meshflow_control.seam.model_adapter import corner_masked_fm_loss

VERSION = "bench_native_trained_loop2_deepsup_v1"
CORE_START = 8
CORE_STOP = 12


def _rounds(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("Only the registered one-pass bypass or two-pass core is allowed")
    return value


class LoopNative(NativeModel):
    """Same state_dict and named-parameter order as the natural-N Native model."""

    loop_rounds = 2
    loop_runtime_version = VERSION

    def set_loop_rounds(self, rounds):
        self.loop_rounds = _rounds(rounds)
        return self

    def loop_metadata(self):
        return dict(version=VERSION, rounds=_rounds(self.loop_rounds),
            core_zero_based=list(range(CORE_START, CORE_STOP)),
            core_residual_scale=1., known_refresh="current_call_core_entrance",
            known_reference_detached=False, new_parameters=0,
            supervision_weights=[.5, .5], last_round_only_inference=True,
            block_calls_per_forward=12 if self.loop_rounds == 1 else 16,
            final_readouts_training=1 if self.loop_rounds == 1 else 2,
            final_readouts_inference=1, complete_LiFT_reproduction=False)

    def forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        if _rounds(self.loop_rounds) == 1:
            # Strict original route: no alternate arithmetic, extra head or cache.
            return super().forward(x, t, y, valid_mask=valid_mask,
                                   known_mask=known_mask, observer=observer)
        return self._two_rounds(x, t, y, valid_mask, known_mask,
                                observer=observer, first_readout=False)[1]

    def forward_train(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
        """Return (v1, v2); caller combines both losses and calls backward once."""
        if _rounds(self.loop_rounds) == 1:
            value = super().forward(x, t, y, valid_mask=valid_mask,
                                    known_mask=known_mask, observer=observer)
            return value, value
        return self._two_rounds(x, t, y, valid_mask, known_mask,
                                observer=observer, first_readout=True)

    def _two_rounds(self, x, t, y, valid_mask, known_mask, *, observer, first_readout):
        assert_native_runtime(self, coordinates=(x, t))
        # Keep the inherited Native128..799 contract; no small-model bypass here.
        if x.ndim != 3 or x.shape[-1] != 9:
            raise ValueError("Expected FP32 x[B,padded_N,9]")
        batch, faces, _ = x.shape
        if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
            raise ValueError("Expected free t[B] and integer total N[B]")
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
            raise ValueError("Nonfinite coordinates/time")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("Free time outside [0,1]")
        for mask in (valid_mask, known_mask):
            if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
                raise ValueError("Colocated separate bool[B,padded_N] masks required")
        if bool((known_mask & ~valid_mask).any()):
            raise ValueError("Known faces must be valid")
        if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 799)).any()):
            raise ValueError("y must count real faces only, N128..799")
        if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
            raise ValueError("Each sample requires known and free faces")
        if self.readout_mode != "none" or len(self.readouts) or len(self.backbone.layers) != 12:
            raise ValueError("Loop candidate requires the unmodified 12-layer no-readout Native layout")

        hidden, role = self._embed(x, known_mask, valid_mask)
        c_free = self.backbone.t_embedder(t)
        c_known = self.backbone.t_embedder(torch.ones_like(t))
        count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
        c_free, c_known = c_free + count, c_known + count
        face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
        if observer is not None:
            observer(dict(site="embedding", hidden=hidden, role=role,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))

        def observe(round_number):
            if observer is None:
                return None
            return lambda record: observer(dict(record, loop_round=round_number))

        first_observer = observe(1)
        reference = None
        for index, block in enumerate(self.backbone.layers, 1):
            if index == CORE_START + 1:
                reference = hidden
            hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
                valid_mask, observer=first_observer, layer_index=index, face_time=face_time)
        v1 = None
        if first_readout:
            v1 = native_final_forward(self.backbone.final_layer, hidden, c_free,
                c_known, known_mask, observer=first_observer, face_time=face_time).view(batch, faces, 9)

        known_corners = known_mask.repeat_interleave(3, dim=1)
        previous = hidden
        # Out-of-place selection: neither source nor recurrent state is mutated.
        hidden = torch.where(known_corners[..., None], reference, previous)
        second_observer = observe(2)
        if observer is not None:
            observer(dict(site="loop_core_entry", loop_round=2, hidden=hidden,
                reference=reference, previous=previous, known_corner_mask=known_corners,
                known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
                c_free=c_free, c_known=c_known))
        for index in range(CORE_START, CORE_STOP):
            hidden = native_block_forward(self.backbone.layers[index], hidden, c_free,
                c_known, known_mask, valid_mask, observer=second_observer,
                layer_index=index + 1, face_time=face_time)
        v2 = native_final_forward(self.backbone.final_layer, hidden, c_free,
            c_known, known_mask, observer=second_observer, face_time=face_time).view(batch, faces, 9)
        return v1, v2


def build_model(model_config, readout_mode="none", readout_seed=20261007,
                trainable=True, *, rounds=2):
    """Use the real natural-N constructor, then specialize only its new instance.

    Strict load_state_dict and inherited AdamW parameter names/order remain valid.
    This function never takes ownership of or mutates an existing public model.
    """
    rounds = _rounds(rounds)
    model = native.build_model(model_config, readout_mode=readout_mode,
                               readout_seed=readout_seed, trainable=trainable)
    model.__class__ = LoopNative
    model.set_loop_rounds(rounds)
    return model


def deep_supervision_loss(v1, v2, target, valid_mask, corner_known, *, denominator=None):
    """Return (0.5*FM1+0.5*FM2, FM1, FM2), retaining both gradient paths."""
    first = corner_masked_fm_loss(v1, target, valid_mask, corner_known, denominator=denominator)
    second = corner_masked_fm_loss(v2, target, valid_mask, corner_known, denominator=denominator)
    return .5 * first + .5 * second, first, second
