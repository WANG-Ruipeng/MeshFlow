"""Read-only observation of the actual residual addition in OfficialWrapper.

Install AFTER constructing the wrapper. Each capture surrounds exactly one
forward and takes its face mask. Hooks return None and retain only detached CPU
values; they never recompute or substitute a model activation. Layer 12's next
module is final_layer. Others are the following backbone transformer layer.

Example::

    with ResidualObserver(wrapper) as observer:
        with observer.capture("correct", face_mask):
            correct = wrapper(...)
        with observer.capture("mismatch", face_mask):
            mismatch = wrapper(...)
        rows = observer.report()
        comparison = observer.compare_effective("correct", "mismatch")

Keep each observer bounded to one microbatch if retaining effective residuals;
release_effective() frees those arrays without removing the scalar report.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import torch

LAYERS = (3, 6, 9, 12)


def _cpu(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Residual observation requires a tensor activation")
    return tensor.detach().cpu().clone()


def _moments(values):
    values = values.double()
    count = values.numel()
    if not count:
        return dict(elements=0, square_sum=0., rms=None, max_abs=None)
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite activation in residual observation")
    square_sum = float(values.square().sum())
    return dict(elements=count, square_sum=square_sum,
                rms=math.sqrt(square_sum/count), max_abs=float(values.abs().max()))


def _ratio(numerator, denominator):
    return numerator/denominator if numerator is not None and denominator else None


class ResidualObserver:
    """Observe pre-add h, actual r, post-add h_after and next-module input.

    The wrapper's insertion hook must already exist. A prepended backbone hook
    sees h before that hook; an appended hook sees its actual returned h_after.
    Adapter hooks capture r as actually returned, without dtype conversion until
    the explicitly defined CPU FP32 diagnostic subtraction.
    """
    def __init__(self, wrapper, retain_effective=True):
        self.wrapper = wrapper
        self.retain_effective = retain_effective
        self.records = []
        self._effective = {}
        self._handles = []
        self._active = None
        self._pending = {}
        self._installed = False

    def __enter__(self):
        if self._installed:
            raise RuntimeError("ResidualObserver cannot be installed twice")
        if len(self.wrapper.adapter.layers) != 4 or len(self.wrapper.backbone.layers) != 12:
            raise ValueError("ResidualObserver requires the audited four-injection 12-layer wrapper")
        if not self.wrapper._handles:
            raise RuntimeError("Install ResidualObserver on a live wrapper with its insertion hooks")
        try:
            for index, number in enumerate(LAYERS):
                block = self.wrapper.backbone.layers[number-1]
                following = (self.wrapper.backbone.layers[number] if number < 12
                             else self.wrapper.backbone.final_layer)
                self._handles.append(block.register_forward_hook(self._before(index), prepend=True))
                self._handles.append(self.wrapper.adapter.layers[index].register_forward_hook(self._residual(index)))
                self._handles.append(block.register_forward_hook(self._after(index)))
                self._handles.append(following.register_forward_pre_hook(self._next(index)))
        except BaseException:
            self.close()
            raise
        self._installed = True
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

    @contextmanager
    def capture(self, label, face_mask):
        """Capture one forward; face_mask is the actual [B,N] valid-face mask."""
        if not self._installed or self._active is not None:
            raise RuntimeError("Capture requires an installed, inactive observer")
        if any(row["label"] == label for row in self.records):
            raise ValueError("Capture labels must be unique within an observer")
        if face_mask.ndim != 2 or face_mask.dtype != torch.bool:
            raise ValueError("face_mask must be a boolean [batch, faces] tensor")
        record = dict(label=label, status="RUNNING", layers=[], event_order=[])
        self.records.append(record)
        self._active = dict(record=record, mask=_cpu(face_mask).repeat_interleave(3, dim=1))
        self._pending = {}
        try:
            yield self
            if [row["layer_one_based"] for row in record["layers"]] != list(LAYERS):
                raise RuntimeError("Capture did not observe exactly one forward through all four injection sites")
            record["status"] = "PASS"
        except BaseException as error:
            record.update(status="FAIL", error=repr(error))
            raise
        finally:
            self._active = None
            self._pending.clear()

    def _event(self, index, name):
        self._active["record"]["event_order"].append(dict(layer_one_based=LAYERS[index], event=name))

    def _before(self, index):
        def hook(module, args, output):
            if self._active is None:
                return None
            if index in self._pending:
                raise RuntimeError("Repeated injection site in one residual capture")
            context = self.wrapper._context
            scale = 0. if context is None else float(context[4])
            if scale not in (0., 1.):
                raise RuntimeError("Learning diagnosis only permits original control scale 0 or 1")
            h = _cpu(output)
            if h.ndim != 3 or h.shape[:2] != self._active["mask"].shape:
                raise RuntimeError("Actual hidden shape does not match the captured face mask")
            self._pending[index] = dict(h=h, scale=scale, r=None)
            self._event(index, "h_before_insertion")
            return None
        return hook

    def _residual(self, index):
        def hook(module, args, output):
            if self._active is None:
                return None
            pending = self._pending[index]
            if pending["r"] is not None:
                raise RuntimeError("Repeated adapter execution in one residual capture")
            if not torch.equal(_cpu(args[0]), pending["h"]):
                raise RuntimeError("Observed adapter input differs from pre-insertion hidden feature")
            pending["r"] = _cpu(output)
            self._event(index, "actual_adapter_residual")
            return None
        return hook

    def _after(self, index):
        def hook(module, args, output):
            if self._active is None:
                return None
            self._pending[index]["after"] = _cpu(output)
            self._event(index, "actual_h_after_insertion")
            return None
        return hook

    def _next(self, index):
        def hook(module, args):
            if self._active is None:
                return None
            pending = self._pending[index]
            following = _cpu(args[0])
            self._event(index, "actual_next_module_input")
            self._summarize(index, pending, following)
            return None
        return hook

    def _summarize(self, index, pending, following):
        h, after, residual = pending["h"], pending["after"], pending["r"]
        mask = self._active["mask"]
        if after.shape != h.shape or following.shape != h.shape:
            raise RuntimeError("Residual boundary shape changed")
        if residual is None and pending["scale"] != 0.:
            raise RuntimeError("Enabled branch did not execute its adapter")
        if residual is not None and residual.shape != h.shape:
            raise RuntimeError("Residual shape does not match hidden feature")
        raw = torch.zeros_like(h, dtype=torch.float32) if residual is None else residual.float()
        effective = after.float()-h.float()
        requested = raw*pending["scale"]
        error = effective-requested
        h_stats = _moments(h[mask])
        r_stats = _moments(raw[mask])
        eff_stats = _moments(effective[mask])
        error_stats = _moments(error[mask])
        nonzero = raw[mask] != 0
        unchanged = after[mask].float() == h[mask].float()
        nonzero_count = int(nonzero.sum())
        lost_count = int((nonzero & unchanged).sum())
        row = dict(
            layer_one_based=LAYERS[index], control_scale=pending["scale"],
            shape=list(h.shape), valid_hidden_elements=h_stats["elements"],
            actual_dtypes=dict(h=str(h.dtype), r=None if residual is None else str(residual.dtype),
                               h_after=str(after.dtype), next_module_input=str(following.dtype)),
            next_module=(f"layers.{LAYERS[index]}" if LAYERS[index] < 12 else "final_layer"),
            next_input_equals_h_after=(following.dtype == after.dtype and torch.equal(following, after)),
            residual_observed=residual is not None,
            h=h_stats, r=r_stats, r_eff=eff_stats, quantization_error=error_stats,
            residual_to_feature_rms=_ratio(r_stats["rms"], h_stats["rms"]),
            effective_to_raw_residual_rms=_ratio(eff_stats["rms"], r_stats["rms"]),
            quantization_error_to_raw_residual_rms=_ratio(error_stats["rms"], r_stats["rms"]),
            nonzero_r_elements=nonzero_count, nonzero_r_addition_unchanged_elements=lost_count,
            nonzero_r_addition_unchanged_fraction=lost_count/nonzero_count if nonzero_count else None,
            effective_nonzero_elements=int((effective[mask] != 0).sum()),
        )
        self._active["record"]["layers"].append(row)
        if self.retain_effective:
            key = (self._active["record"]["label"], index)
            self._effective[key] = (effective[mask].clone(), mask.clone())
        # Release detached boundary snapshots as soon as the next module sees them.
        pending.clear()

    def compare_effective(self, correct_label, mismatch_label):
        """Elementwise correct-minus-mismatch r_eff under identical valid masks."""
        rows = []
        for index, number in enumerate(LAYERS):
            a, mask_a = self._effective[(correct_label, index)]
            b, mask_b = self._effective[(mismatch_label, index)]
            if a.shape != b.shape or not torch.equal(mask_a, mask_b):
                raise RuntimeError("Effective residual comparison requires identical shapes and masks")
            rows.append(dict(layer_one_based=number, difference=_moments(a-b),
                             bitwise_equal=torch.equal(a, b)))
        return dict(correct_label=correct_label, mismatch_label=mismatch_label,
                    definition="r_eff(correct) - r_eff(mismatch), identical valid hidden elements", layers=rows)

    def report(self):
        return dict(captures=copy.deepcopy(self.records), hooks_return_values=False,
                    statistics_device="cpu", diagnostic_subtraction_dtype="torch.float32",
                    effective_definition="h_after.float() - h.float()",
                    quantization_error_definition="r_eff - control_scale * r.float()",
                    off_residual_note="No adapter residual is observed when control is off; zero is used for diagnostics")

    def release_effective(self):
        self._effective.clear()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = None
        self._pending.clear()
        self._installed = False
