"""Independent, temporary v3 Pre-FFN insertion; official source is untouched.

H_a is the official face-attention broadcast residual sum. The one change is
H_pre = H_a + Adapter(H_a, C), used by BOTH norm2/FFN and the residual main path.
The query therefore observes H_a, not the Post wrapper's completed-block H_b.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from types import MethodType
import weakref

import torch
from torch import nn
from models.equidit import DiTLayer, modulate

from .learning_residuals import _cpu, _moments, _ratio

LAYERS = (3, 6, 9, 12)
OWNER_ATTRIBUTE = "_mf_pre_injection_owner"


def active_pre_owner(backbone):
    reference = getattr(backbone, OWNER_ATTRIBUTE, None)
    return reference() if isinstance(reference, weakref.ReferenceType) else reference


def post_injection_hooks(backbone):
    return [(number, key) for number, layer in enumerate(backbone.layers, 1)
            for key, hook in layer._forward_hooks.items()
            if getattr(hook, "__name__", None) == "residual_hook"
            and getattr(hook, "__module__", "").endswith("official_wrapper")]


class PreFFNWrapper(nn.Module):
    """OfficialWrapper-compatible S32 control at four real Pre-FFN sites.

    No backbone/adapter parameter is created or loaded here. Method replacement
    exists only during a forward and is restored in finally, including failures.
    The weak ownership marker survives until close() to reject stacked owners.
    """
    def __init__(self, backbone, adapter):
        super().__init__()
        if backbone.version != 3 or backbone.hidden_size != 768 or len(backbone.layers) != 12:
            raise ValueError("Pre-FFN requires the audited official version3 backbone")
        if adapter.arm != "S32_static" or sum(p.numel() for p in adapter.parameters()) != 950144:
            raise ValueError("Pre-FFN changes only the insertion site of the exact S32_static adapter")
        if active_pre_owner(backbone) is not None or post_injection_hooks(backbone):
            raise RuntimeError("Close the existing Post/Pre controller before constructing PreFFNWrapper")
        if backbone.training or any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("Freeze the official backbone and keep it in eval; no checkpointing change is allowed")
        self.backbone, self.adapter = backbone, adapter
        self._context = None
        self._in_forward = False
        self._closed = False
        self._saved_methods = []
        self._observers = []
        self.counts = dict(wrapper_forward=0, backbone_forward=0, adapter_layer=0,
            control_disabled=0, condition_encoding=0, pre_site_visits=0,
            geometry_requests=0, helper_calls=0, computations=0, cache_hits=0)
        self._assert_original_methods()
        setattr(backbone, OWNER_ATTRIBUTE, weakref.ref(self))
        self._handles = [backbone.register_forward_pre_hook(self._count_and_guard)]

    def _assert_original_methods(self):
        for number in LAYERS:
            block = self.backbone.layers[number-1]
            if block.version != 3 or getattr(block._forward_v3, "__func__", None) is not DiTLayer._forward_v3:
                raise RuntimeError("An unknown or stale version3 forward replacement is already installed")

    def _assert_exclusive(self):
        if self._closed or active_pre_owner(self.backbone) is not self:
            raise RuntimeError("PreFFNWrapper is closed or no longer owns this backbone")
        if post_injection_hooks(self.backbone):
            raise RuntimeError("Post and Pre controls cannot execute on the same backbone")
        if self.backbone.training or any(layer.training for layer in self.backbone.layers):
            raise RuntimeError("Pre-FFN requires frozen eval blocks; checkpointed training is forbidden")

    def _count_and_guard(self, module, args):
        self.counts["backbone_forward"] += 1
        self._assert_exclusive()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, prompts):
        self._assert_exclusive()
        self.counts["condition_encoding"] += 1
        return self.adapter.encode(prompts)

    def _v3(self, index, block, x, c, mask=None):
        # Keep every official operation, broadcasting method and arithmetic order.
        b, n_vertex, _ = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6, dim=1)
        x_face = x.reshape(b, n_vertex//3, 3, -1).mean(dim=2)
        x_face = gate_msa.unsqueeze(1) * block.attn(modulate(block.norm1(x_face), shift_msa, scale_msa), mask=mask)
        x = x + x_face.unsqueeze(2).repeat(1,1,3,1).reshape(b, n_vertex, block.dim)
        self.counts["pre_site_visits"] += 1
        h_a = x
        residual = None
        if self._context is not None:
            xt, t, encoded, corner_mask = self._context
            self.counts["adapter_layer"] += 1
            residual = self.adapter.forward_layer(index, h_a, xt, t, encoded, corner_mask)
            x = h_a + residual
        h_pre = x
        for observer in tuple(self._observers):
            observer._boundary(index, h_a, residual, h_pre)
        # h_pre is explicitly the operand of BOTH norm2 and the residual sum.
        branch = gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(h_pre), shift_mlp, scale_mlp), mask=mask)
        x = h_pre + branch
        for observer in tuple(self._observers):
            observer._result(index, h_pre, x)
        return x

    def _install_methods(self):
        self._assert_original_methods()
        for index, number in enumerate(LAYERS):
            block = self.backbone.layers[number-1]
            had_instance = "_forward_v3" in block.__dict__
            previous = block.__dict__.get("_forward_v3")
            self._saved_methods.append((block, had_instance, previous))
            def patched(block_self, x, c, mask=None, _index=index):
                return self._v3(_index, block_self, x, c, mask=mask)
            block._forward_v3 = MethodType(patched, block)

    def _restore_methods(self):
        while self._saved_methods:
            block, had_instance, previous = self._saved_methods.pop()
            if had_instance:
                block._forward_v3 = previous
            else:
                delattr(block, "_forward_v3")

    def forward(self, x_t, t, y, mask=None, prompts=None, prompt_mask=None,
                control_scale=1., encoded=None, inference_geometry_cache=False,
                control_enabled=None):
        self._assert_exclusive()
        if self._in_forward:
            raise RuntimeError("Reentrant PreFFNWrapper invocation is forbidden")
        if inference_geometry_cache:
            raise ValueError("S32_static has no geometry cache")
        if control_scale not in (0., 1.):
            raise ValueError("Internal scale is fixed at1; use final-velocity guidance outside this wrapper")
        self.counts["wrapper_forward"] += 1
        self._in_forward = True
        try:
            if prompt_mask is not None:
                prompts = dict(prompts, primitive_mask=prompt_mask)
            enabled = (control_enabled is not False and control_scale != 0
                       and (prompts is not None or encoded is not None))
            if enabled:
                if encoded is None:
                    encoded = self.encode(prompts)
                self._context = (x_t.reshape(x_t.shape[0], -1, 3),
                    t.reshape(x_t.shape[0], -1)[:,0], encoded,
                    None if mask is None else mask.repeat_interleave(3, dim=1))
            else:
                self.counts["control_disabled"] += 1
            self._install_methods()
            return self.backbone(x_t, t, y, mask=mask)
        finally:
            self._context = None
            self._restore_methods()
            self._in_forward = False

    def close(self):
        if self._in_forward:
            raise RuntimeError("Cannot close a controller while its forward is running")
        self._context = None
        self._restore_methods()
        for observer in tuple(self._observers):
            observer.close()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if active_pre_owner(self.backbone) is self:
            delattr(self.backbone, OWNER_ATTRIBUTE)
        self._closed = True


class PreResidualObserver:
    """Observe actual H_a/r/H_pre and the real norm2 input, without changing it."""
    def __init__(self, wrapper, retain_effective=True):
        if not isinstance(wrapper, PreFFNWrapper):
            raise TypeError("Use PreResidualObserver only with the Pre-FFN controller")
        self.wrapper, self.retain_effective = wrapper, retain_effective
        self.records = []
        self._handles, self._pending, self._effective = [], {}, {}
        self._active = None
        self._installed = False

    def __enter__(self):
        if self._installed:
            raise RuntimeError("Observer is already installed")
        self.wrapper._assert_exclusive()
        try:
            for index, number in enumerate(LAYERS):
                self._handles.append(self.wrapper.backbone.layers[number-1].norm2.register_forward_pre_hook(self._norm_input(index)))
            self.wrapper._observers.append(self)
            self._installed = True
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()
        return False

    @contextmanager
    def capture(self, label, face_mask):
        if not self._installed or self._active is not None:
            raise RuntimeError("Capture needs an installed inactive Pre observer")
        if any(row["label"] == label for row in self.records):
            raise ValueError("Capture labels must be unique")
        if face_mask.ndim != 2 or face_mask.dtype != torch.bool:
            raise ValueError("Capture needs the actual boolean [B,N] face mask")
        record = dict(label=label,status="RUNNING",layers=[],event_order=[])
        self.records.append(record)
        self._active = dict(record=record,mask=_cpu(face_mask).repeat_interleave(3,1))
        self._pending = {}
        try:
            yield self
            if ([row["layer_one_based"] for row in record["layers"]] != list(LAYERS)
                    or not all(row["main_path_uses_same_h_pre"] for row in record["layers"])):
                raise RuntimeError("Exactly four complete actual Pre-FFN boundaries must be observed")
            record["status"] = "PASS"
        except BaseException as error:
            record.update(status="FAIL",error=repr(error))
            raise
        finally:
            self._active = None
            self._pending.clear()

    def _event(self, index, name):
        self._active["record"]["event_order"].append(dict(layer_one_based=LAYERS[index],event=name))

    def _boundary(self, index, h_a, residual, h_pre):
        if self._active is None:
            return
        if index in self._pending or h_a.shape[:2] != self._active["mask"].shape:
            raise RuntimeError("Repeated Pre injection or mismatched face-mask shape")
        self._pending[index] = dict(h=_cpu(h_a), r=None if residual is None else _cpu(residual),
                                    after=_cpu(h_pre), after_id=id(h_pre), row=None)
        for name in ("actual_h_a_after_attention_broadcast", "actual_adapter_r", "actual_h_pre_after_addition"):
            self._event(index, name)

    def _norm_input(self, index):
        def hook(module,args):
            if self._active is None:
                return None
            pending = self._pending[index]
            following = _cpu(args[0])
            h, after, residual = pending["h"],pending["after"],pending["r"]
            mask = self._active["mask"]
            raw = torch.zeros_like(h,dtype=torch.float32) if residual is None else residual.float()
            effective = after.float()-h.float()
            error = effective-raw
            hs,rs,es,qs = (_moments(v[mask]) for v in (h,raw,effective,error))
            nonzero = raw[mask] != 0
            unchanged = after[mask].float() == h[mask].float()
            nz = int(nonzero.sum()); lost = int((nonzero & unchanged).sum())
            row = dict(layer_one_based=LAYERS[index],position="after_attention_broadcast_before_norm2",
                control_scale=1. if residual is not None else 0.,shape=list(h.shape),valid_hidden_elements=hs["elements"],
                actual_dtypes=dict(h=str(h.dtype),r=None if residual is None else str(residual.dtype),
                    h_after=str(after.dtype),next_module_input=str(following.dtype)),
                next_module=f"layers.{LAYERS[index]-1}.norm2",
                next_input_equals_h_after=following.dtype==after.dtype and torch.equal(following,after),
                norm2_input_is_same_h_pre_object=id(args[0])==pending["after_id"],
                main_path_uses_same_h_pre=False,residual_observed=residual is not None,
                h=hs,r=rs,r_eff=es,quantization_error=qs,
                residual_to_feature_rms=_ratio(rs["rms"],hs["rms"]),
                effective_to_raw_residual_rms=_ratio(es["rms"],rs["rms"]),
                quantization_error_to_raw_residual_rms=_ratio(qs["rms"],rs["rms"]),
                nonzero_r_elements=nz,nonzero_r_addition_unchanged_elements=lost,
                nonzero_r_addition_unchanged_fraction=lost/nz if nz else None,
                effective_nonzero_elements=int((effective[mask]!=0).sum()))
            if not row["next_input_equals_h_after"] or not row["norm2_input_is_same_h_pre_object"]:
                raise RuntimeError("norm2 did not consume the actual injected H_pre")
            self._active["record"]["layers"].append(row)
            self._event(index,"actual_norm2_input")
            pending["row"] = row
            if self.retain_effective:
                self._effective[(self._active["record"]["label"],index)] = (effective[mask].clone(),mask.clone())
            del pending["h"],pending["r"],pending["after"]
            return None
        return hook

    def _result(self,index,main_path,output):
        if self._active is None:
            return
        pending=self._pending[index]
        pending["row"]["main_path_uses_same_h_pre"] = id(main_path)==pending["after_id"]
        self._event(index,"same_h_pre_used_in_ffn_residual_sum")

    def compare_effective(self,correct_label,mismatch_label):
        rows=[]
        for index,number in enumerate(LAYERS):
            a,ma=self._effective[(correct_label,index)];b,mb=self._effective[(mismatch_label,index)]
            if a.shape!=b.shape or not torch.equal(ma,mb):
                raise RuntimeError("Effective residual comparison requires identical masks")
            rows.append(dict(layer_one_based=number,difference=_moments(a-b),bitwise_equal=torch.equal(a,b)))
        return dict(correct_label=correct_label,mismatch_label=mismatch_label,layers=rows,
                    definition="Actual Pre r_eff(correct)-r_eff(mismatch), same valid hidden elements")

    def report(self):
        return dict(captures=copy.deepcopy(self.records),hooks_return_values=False,statistics_device="cpu",
            diagnostic_subtraction_dtype="torch.float32",effective_definition="H_pre.float()-H_a.float()",
            quantization_error_definition="r_eff-r.float()",insertion_position="after_attention_broadcast_before_norm2",
            query_note="Pre query sees H_a; Post query sees completed-block H_b, which generally differs.")

    def release_effective(self):
        self._effective.clear()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self in self.wrapper._observers:
            self.wrapper._observers.remove(self)
        self._active=None;self._pending.clear();self._installed=False
