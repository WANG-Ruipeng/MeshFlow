"""Full-parameter FP32 gradient carrier for the frozen T/L/G experiment.

No forward, backward, optimizer step, random draw or model loading occurs here.
The caller accumulates eight primary backwards, captures gF, then executes one
UNWEIGHTED auxiliary backward. All three arms use exactly this carrier.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import torch

RHO = 1.0
LOSS_CAP = 0.1
LOSS_EPS = 1e-30
GRAD_EPS = 1e-12
ARM_NAMES = {"T": "T", "L": "L", "G": "G", "TEACHER_LCAP": "T",
             "CAPPED_REPLAY": "L", "GRAD_BUDGET": "G"}


def _named(named_parameters):
    entries = [(str(n), p) for n, p in named_parameters if p.requires_grad]
    if not entries:
        raise ValueError("The carrier requires the complete nonempty trainable parameter set.")
    if len({n for n, _ in entries}) != len(entries) or len({id(p) for _, p in entries}) != len(entries):
        raise ValueError("Duplicate parameter names/objects are not allowed.")
    device = entries[0][1].device
    if any(p.dtype != torch.float32 or p.device != device for _, p in entries):
        raise TypeError("All trainable parameters must be FP32 on one device.")
    return entries


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_grad(p, g):
    if g is not None and (g.is_sparse or g.dtype != torch.float32 or g.device != p.device or g.shape != p.shape):
        raise TypeError("Dense FP32 parameter-shaped gradients on the parameter device are required.")


def _norm(tensors, device):
    norms = [torch.linalg.vector_norm(g.detach()) for g in tensors if g is not None]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros((), device=device, dtype=torch.float32)


def coefficients(primary_loss, auxiliary_loss, norm_f, norm_a, arm):
    """Detached Python scalars; zero-gF override is the registered G rule only."""
    arm = ARM_NAMES[arm]
    f, a, nf, na = map(float, (primary_loss, auxiliary_loss, norm_f, norm_a))
    if not all(math.isfinite(x) for x in (f, a, nf, na)):
        raise FloatingPointError("Nonfinite loss or full-gradient norm.")
    if min(f, a, nf, na) < 0:
        raise ValueError("MSE and gradient norms must be nonnegative.")
    loss_ratio = f / max(a, LOSS_EPS)
    loss_weight = min(LOSS_CAP, loss_ratio)
    grad_weight = RHO * nf / (na + GRAD_EPS)
    effective = min(loss_weight, grad_weight) if arm == "G" else loss_weight
    if arm == "G" and nf == 0.0:
        effective = 0.0
    return dict(arm=arm, primary_loss=f, auxiliary_loss=a, rho=RHO,
        lambda_loss=loss_weight, lambda_grad=grad_weight, actual_lambda=effective,
        lambda_loss_hex=loss_weight.hex(), lambda_grad_hex=grad_weight.hex(), actual_lambda_hex=effective.hex(),
        loss_cap_active=loss_weight < LOSS_CAP,
        gradient_cap_active=arm == "G" and grad_weight < loss_weight,
        primary_gradient_zero=nf == 0.0, auxiliary_gradient_zero=na == 0.0,
        zero_primary_override=arm == "G" and nf == 0.0,
        coefficient_detached=True, zero_auxiliary_contribution=na == 0.0 or effective == 0.0)


class GradientCarrier:
    """One extra gF buffer; gA stays in normal .grad until assembly.

    None is zero for all norms/dot/combination. Both-None remains None in the
    final .grad to retain AdamW's existing unused-parameter skip semantics.
    An explicitly zero tensor remains a tensor and keeps optimizer semantics.
    """
    def __init__(self, named_parameters):
        self.named_parameters = _named(named_parameters)
        self.params = [p for _, p in self.named_parameters]
        self.device = self.params[0].device
        self._identities = [(id(p), tuple(p.shape), p.device, p.dtype) for p in self.params]
        self._primary = None
        self._capture = None
        description = [(n, tuple(p.shape), p.numel(), str(p.dtype)) for n, p in self.named_parameters]
        self.parameter_set_sha256 = hashlib.sha256(json.dumps(description, separators=(",", ":")).encode()).hexdigest()

    def _check_identity(self):
        if any(not p.requires_grad or (id(p), tuple(p.shape), p.device, p.dtype) != x for p, x in zip(self.params, self._identities)):
            raise RuntimeError("Trainable parameter identity changed after carrier construction.")

    @torch.no_grad()
    def capture_primary(self, *, clear=True):
        if not clear:
            raise ValueError("The registered carrier must clear .grad before raw auxiliary backward.")
        if self._primary is not None:
            raise RuntimeError("Primary gradients already captured; combine once before recapture.")
        self._check_identity()
        _sync(self.device)
        started = time.perf_counter()
        for p in self.params:
            _validate_grad(p, p.grad)
        norm = float(_norm([p.grad for p in self.params], self.device))
        if not math.isfinite(norm):
            raise FloatingPointError("Primary gradient contains nonfinite values or its FP32 norm overflowed.")
        self._primary = [None if p.grad is None else p.grad.detach().clone() for p in self.params]
        missing = [n for (n, _), g in zip(self.named_parameters, self._primary) if g is None]
        buffer_bytes = sum(g.numel() * g.element_size() for g in self._primary if g is not None)
        for p in self.params:
            p.grad = None
        _sync(self.device)
        self._capture = dict(gF_norm=norm, primary_none_names=missing, primary_none_count=len(missing),
            primary_buffer_bytes=buffer_bytes, buffer_dtype="torch.float32", capture_seconds=time.perf_counter()-started,
            parameter_count=len(self.params), parameter_elements=sum(p.numel() for p in self.params),
            parameter_set_sha256=self.parameter_set_sha256, primary_finite=True)
        return dict(self._capture)

    @torch.no_grad()
    def combine(self, *, primary_loss, auxiliary_loss, arm):
        if self._primary is None:
            raise RuntimeError("capture_primary must precede the unweighted auxiliary backward.")
        self._check_identity()
        _sync(self.device)
        started = time.perf_counter()
        gf, ga = self._primary, [p.grad for p in self.params]
        for p, g in zip(self.params, ga):
            _validate_grad(p, g)
        # All raw gradients remain FP32. Reductions match full-vector FP32
        # arithmetic, with a deterministic parameter order (not a layer proxy).
        na_t = _norm(ga, self.device)
        dots = [torch.dot(f.reshape(-1), a.detach().reshape(-1)) for f, a in zip(gf, ga) if f is not None and a is not None]
        dot_t = torch.stack(dots).sum() if dots else torch.zeros((), device=self.device, dtype=torch.float32)
        na, dot = [float(x) for x in torch.stack((na_t, dot_t)).cpu()]
        nf = self._capture["gF_norm"]
        if not math.isfinite(na) or not math.isfinite(dot):
            raise FloatingPointError("Auxiliary gradient or full-gradient dot is nonfinite.")
        balance = coefficients(primary_loss, auxiliary_loss, nf, na, arm)
        weight = balance["actual_lambda"]
        cosine = dot/(nf*na) if nf > 0 and na > 0 else None
        if cosine is not None and not math.isfinite(cosine):
            raise FloatingPointError("Nonfinite full-gradient cosine.")
        missing_a = [n for (n, _), g in zip(self.named_parameters, ga) if g is None]
        both_missing = [n for (n, _), f, a in zip(self.named_parameters, gf, ga) if f is None and a is None]
        reduction_seconds = time.perf_counter()-started
        assembly_started = time.perf_counter()
        # Scale the UNWEIGHTED auxiliary exactly once. No extra backward and
        # no gradient-norm differentiation, projection or optimizer operation.
        for a in ga:
            if a is not None:
                a.mul_(weight)
        actual_aux_norm = float(_norm(ga, self.device))
        for p, f, a in zip(self.params, gf, ga):
            if a is None:
                p.grad = f  # transfer ownership of gF; None stays None
            elif f is not None:
                a.add_(f)
                p.grad = a
            else:
                p.grad = a
        total_norm = float(_norm([p.grad for p in self.params], self.device))
        if not math.isfinite(actual_aux_norm) or not math.isfinite(total_norm):
            raise FloatingPointError("Nonfinite assembled gradient; caller must not update.")
        _sync(self.device)
        assembly_seconds = time.perf_counter()-assembly_started
        result = dict(self._capture, **balance, gA_norm=na, gF_dot_gA=dot,
            cosine=cosine, cosine_status="OBSERVED" if cosine is not None else "UNDEFINED_ZERO_NORM",
            weighted_aux_norm=actual_aux_norm,
            weighted_aux_over_FM_norm=actual_aux_norm/nf if nf > 0 else None,
            weighted_aux_over_FM_norm_status="DEFINED" if nf > 0 else "UNDEFINED_ZERO_PRIMARY",
            combination_preclip_norm=total_norm,
            predicted_preclip_norm_squared=nf*nf+weight*weight*na*na+2*weight*dot,
            theoretical_weighted_aux_norm=weight*na,
            theoretical_budget_excess=max(0.0, weight*na-RHO*nf) if balance["arm"] == "G" else None,
            actual_FP32_budget_excess=max(0.0, actual_aux_norm-RHO*nf) if balance["arm"] == "G" else None,
            budget_inequality_scope="Pre-combination full auxiliary gradient; FP32 multiply/reduction rounding reported separately; not AdamW parameter updates.",
            auxiliary_none_names=missing_a, auxiliary_none_count=len(missing_a),
            both_none_names=both_missing, both_none_count=len(both_missing),
            none_semantics="Zero contribution to arithmetic; both-None retained for original optimizer skip semantics.",
            auxiliary_finite=True, combination_finite=True,
            weighted_auxiliary_loss_value=weight*float(auxiliary_loss),
            weighted_auxiliary_loss_value_scope="Detached Python diagnostic; backward was raw L_A and weighting was applied once to FP32 gradients.",
            scalar_reduction_seconds=reduction_seconds, combination_seconds=assembly_seconds,
            carrier_seconds=self._capture["capture_seconds"]+reduction_seconds+assembly_seconds,
            gA_storage="Existing parameter .grad; no second persistent gradient copy",
            primary_buffer_released=True)
        self._primary = None
        self._capture = None
        return result


def should_snapshot(step):
    step = int(step)
    return step == 1 or step == 501 or (step > 0 and step % 25 == 0)


class ParameterSnapshot:
    """Chunked CPU FP32 pre-update values; exact saved-float delta norm in FP64.

    Construct immediately before the caller's one authorized AdamW.step, then
    call measure_and_release afterwards. It never clones model/optimizer state
    onto the GPU and never performs an optimizer step itself.
    """
    @torch.no_grad()
    def __init__(self, named_parameters, *, chunk_elements=1048576, parameter_groups=None):
        if int(chunk_elements) <= 0:
            raise ValueError("Positive chunk size required.")
        self.named_parameters = _named(named_parameters)
        self.device = self.named_parameters[0][1].device
        self.chunk_elements = int(chunk_elements)
        self.groups = parameter_groups or {n:"all" for n, _ in self.named_parameters}
        if set(self.groups) != {n for n, _ in self.named_parameters}:
            raise ValueError("Group mapping must cover every trainable parameter exactly.")
        self._pieces = []
        _sync(self.device)
        started = time.perf_counter()
        for name, p in self.named_parameters:
            value = p.detach().reshape(-1)
            chunks = [value[i:i+self.chunk_elements].to(device="cpu", dtype=torch.float32, copy=True)
                      for i in range(0, value.numel(), self.chunk_elements)]
            if any(not bool(torch.isfinite(c).all()) for c in chunks):
                raise FloatingPointError("Nonfinite parameter before the optimizer update.")
            self._pieces.append((name, p, tuple(p.shape), chunks))
        _sync(self.device)
        self.capture_seconds = time.perf_counter()-started
        self.bytes = sum(c.numel()*c.element_size() for _, _, _, chunks in self._pieces for c in chunks)

    @torch.no_grad()
    def measure_and_release(self):
        if self._pieces is None:
            raise RuntimeError("This parameter snapshot has already been consumed.")
        _sync(self.device)
        started = time.perf_counter()
        totals = {}
        per_parameter = []
        for name, p, shape, chunks in self._pieces:
            if tuple(p.shape) != shape or p.dtype != torch.float32:
                raise RuntimeError("Parameter identity/shape changed across update.")
            values = p.detach().reshape(-1)
            before_sq, after_sq, delta_sq, changed, maximum = [], [], [], 0, 0.0
            for index, before in enumerate(chunks):
                after = values[index*self.chunk_elements:index*self.chunk_elements+before.numel()].to(device="cpu", dtype=torch.float32, copy=True)
                if not bool(torch.isfinite(after).all()):
                    raise FloatingPointError("Nonfinite parameter after optimizer update.")
                b, a = before.double(), after.double()
                d = a-b
                before_sq.append(float(torch.dot(b,b)))
                after_sq.append(float(torch.dot(a,a)))
                delta_sq.append(float(torch.dot(d,d)))
                changed += int(torch.count_nonzero(after != before))
                if d.numel():
                    maximum = max(maximum, float(d.abs().max()))
                chunks[index] = None  # free consumed CPU FP32 storage promptly
            b2, a2, d2 = map(math.fsum, (before_sq, after_sq, delta_sq))
            record = dict(name=name, group=str(self.groups[name]), elements=p.numel(),
                parameter_before_norm=math.sqrt(b2), parameter_after_norm=math.sqrt(a2),
                actual_update_norm=math.sqrt(d2), actual_update_relative_norm=math.sqrt(d2/b2) if b2 else None,
                relative_status="DEFINED" if b2 else "UNDEFINED_ZERO_PARAMETER_NORM", changed_elements=changed, max_abs_update=maximum)
            per_parameter.append(record)
            entry = totals.setdefault(str(self.groups[name]), dict(before=[], after=[], delta=[], changed=0, elements=0, maximum=0.))
            entry["before"].append(b2);entry["after"].append(a2);entry["delta"].append(d2)
            entry["changed"] += changed;entry["elements"] += p.numel();entry["maximum"] = max(entry["maximum"],maximum)
        def pack(name, b2, a2, d2, changed, elements, maximum):
            return dict(group=name, parameter_before_norm=math.sqrt(b2), parameter_after_norm=math.sqrt(a2), actual_update_norm=math.sqrt(d2),
                actual_update_relative_norm=math.sqrt(d2/b2) if b2 else None,
                relative_status="DEFINED" if b2 else "UNDEFINED_ZERO_PARAMETER_NORM", changed_elements=changed, elements=elements,max_abs_update=maximum)
        group_rows = [pack(k,math.fsum(v["before"]),math.fsum(v["after"]),math.fsum(v["delta"]),v["changed"],v["elements"],v["maximum"]) for k,v in sorted(totals.items())]
        total = pack("ALL", math.fsum(math.fsum(v["before"]) for v in totals.values()), math.fsum(math.fsum(v["after"]) for v in totals.values()),
            math.fsum(math.fsum(v["delta"]) for v in totals.values()),sum(v["changed"] for v in totals.values()),sum(v["elements"] for v in totals.values()),max(v["maximum"] for v in totals.values()))
        self._pieces = None
        _sync(self.device)
        total.update(status="PASS", groups=group_rows, per_parameter=per_parameter, cpu_snapshot_bytes=self.bytes,
            chunk_elements=self.chunk_elements, snapshot_seconds=self.capture_seconds, measure_seconds=time.perf_counter()-started,
            measurement="Actual FP32 parameter values before/after the caller's optimizer step; chunked FP64 CPU sum of squared differences.",
            no_gradient_proxy=True, released=True, model_calls=0, optimizer_updates_by_helper=0)
        return total
