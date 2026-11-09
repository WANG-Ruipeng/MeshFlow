"""Model-free FP32 diagnostics for the registered local Euler experiment.

This module never calls a neural model, RNG, OT, optimizer, CUDA initialization,
backward, or a geometry evaluator. Integration and differences use actual FP32
operands. Scalar reductions are evaluated in float64 to avoid introducing a
second low-precision reduction into the reported diagnostics.
"""
from __future__ import annotations
import math
from itertools import combinations
import torch

RESOLUTION_ULPS = 8.0
RESOLUTION_RULE = {
    "name": "eight_rms_coordinate_ulps",
    "multiplier": RESOLUTION_ULPS,
    "formula": "8 * RMS(max_operand(max(nextafter(x,+inf)-x, x-nextafter(x,-inf))))",
    "scope": "free coordinates; each nextafter and operand is FP32; scalar reduction FP64",
    "interpretation": "Conservative preregistered representation-resolution gate; not a neural forward error bound.",
    "unresolved": "denominator <= floor, or denominator is zero; report absolute values and null ratios",
}


def _validate(x, known):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.ndim < 2:
        raise ValueError("FP32 tensor with a final coordinate dimension required")
    if not isinstance(known, torch.Tensor) or known.dtype != torch.bool or known.shape != x.shape[:-1]:
        raise ValueError("known must be bool with shape x.shape[:-1]")
    if known.device != x.device or x.shape[-1] % 3 or not bool((~known).any()):
        raise ValueError("Shared device, xyz coordinate tuples and nonempty free set required")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite coordinate or velocity")


def _compatible(reference, value, known):
    _validate(value, known)
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError("Tensor shape/device mismatch")


def bits_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype == torch.float32 and
            a.device == b.device and torch.equal(a.contiguous().view(torch.int32),
                                               b.contiguous().view(torch.int32)))


def free_values(value, known):
    _validate(value, known)
    return value[~known]


def _rms(value):
    value = value.double()
    return float(torch.sqrt(torch.mean(value * value)).item())


def vector_stats(value, known, bbox=1.0):
    """Raw model units and fixed-bbox units; coordinate and point RMS differ."""
    if not math.isfinite(float(bbox)) or float(bbox) <= 0:
        raise ValueError("Positive finite fixed GT bbox required")
    v = free_values(value, known).double()
    coord = _rms(v)
    points = v.reshape(-1, 3)
    point = float(torch.sqrt(points.square().sum(-1).mean()).item())
    maximum = float(v.abs().max().item())
    norm = float(torch.linalg.vector_norm(v).item())
    return {"coordinate_count": v.numel(), "point_count": points.shape[0],
            "coordinate_rms": coord, "point_rms": point, "l2_norm": norm,
            "max_abs": maximum, "coordinate_rms_bbox": coord / float(bbox),
            "point_rms_bbox": point / float(bbox), "max_abs_bbox": maximum / float(bbox),
            "bbox": float(bbox), "reduction_dtype": "float64"}


@torch.no_grad()
def euler_free(x, velocity, dt, known):
    """Exactly FP32 x + velocity * dt on free slots; C bits are copied untouched."""
    _validate(x, known)
    _compatible(x, velocity, known)
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("Positive finite dt required")
    out = x.clone()
    out[~known] = x[~known] + velocity[~known] * float(dt)
    if not bits_equal(out[known], x[known]) or not bool(torch.isfinite(out).all()):
        raise RuntimeError("Euler update changed C bits or produced a nonfinite state")
    return out


@torch.no_grad()
def local_predictors(x, v0, known, h=0.01):
    return {"x_E": euler_free(x, v0, h, known),
            "x_half": euler_free(x, v0, h / 2, known),
            "x_quarter": euler_free(x, v0, h / 4, known)}


@torch.no_grad()
def free_difference(left, right, known):
    """Actual FP32 subtraction; known slots are diagnostic zeros, never integrated."""
    _validate(left, known)
    _compatible(left, right, known)
    out = torch.zeros_like(left)
    out[~known] = left[~known] - right[~known]
    return out


@torch.no_grad()
def resolution_floor(*states, known):
    """Eight RMS coordinate ULPs, fixed before observing experiment results."""
    if not states:
        raise ValueError("At least one actual operand is required")
    maxima = None
    for state in states:
        _compatible(states[0], state, known)
        v = state[~known]
        up = torch.nextafter(v, torch.full_like(v, math.inf)).double() - v.double()
        down = v.double() - torch.nextafter(v, torch.full_like(v, -math.inf)).double()
        spacing = torch.maximum(up, down)
        if not bool(torch.isfinite(spacing).all()):
            raise ValueError("ULP is nonfinite at the FP32 extreme")
        maxima = spacing if maxima is None else torch.maximum(maxima, spacing)
    return RESOLUTION_ULPS * _rms(maxima)


def _ratio(numerator, denominator, floor):
    return (float(numerator / denominator) if denominator > floor and denominator > 0
            else None)


@torch.no_grad()
def local_differences(x, v0, v_mid, v_quarter, known, h=0.01, bbox=1.0):
    states = local_predictors(x, v0, known, h)
    states["x_2E"] = euler_free(states["x_half"], v_mid, h / 2, known)
    states["x_E_small"] = states["x_half"]
    states["x_2E_small"] = euler_free(states["x_quarter"], v_quarter, h / 4, known)
    states["delta_h"] = free_difference(states["x_2E"], states["x_E"], known)
    states["delta_half"] = free_difference(states["x_2E_small"], states["x_E_small"], known)
    dh, ds = vector_stats(states["delta_h"], known, bbox), vector_stats(states["delta_half"], known, bbox)
    floor_h = resolution_floor(states["x_E"], states["x_2E"], known=known)
    floor_half = resolution_floor(states["x_E_small"], states["x_2E_small"], known=known)
    ratio = _ratio(dh["coordinate_rms"], ds["coordinate_rms"], floor_half)
    states["summary"] = {"h": h, "delta_h": dh, "delta_half": ds,
                         "delta_h_over_delta_half_rms": ratio,
                         "delta_half_over_delta_h_rms": _ratio(ds["coordinate_rms"], dh["coordinate_rms"], floor_h),
                         "ratio_status": "RESOLVED" if ratio is not None else "DENOMINATOR_UNRESOLVED",
                         "delta_h_resolution_floor": floor_h,
                         "delta_half_resolution_floor": floor_half,
                         "definition": "two-half-step minus one-step; not exact local truncation error"}
    return states


def cosine_stats(a, b, known):
    av, bv = free_values(a, known).double().flatten(), free_values(b, known).double().flatten()
    na, nb = float(torch.linalg.vector_norm(av)), float(torch.linalg.vector_norm(bv))
    if na == 0 or nb == 0:
        return {"cosine": None, "angle_degrees": None, "status": "ZERO_VECTOR_UNRESOLVED"}
    raw = float(torch.dot(av, bv)) / (na * nb)
    cosine = max(-1.0, min(1.0, raw))
    return {"cosine": cosine, "raw_cosine": raw,
            "angle_degrees": math.degrees(math.acos(cosine)), "status": "RESOLVED"}


@torch.no_grad()
def decompose_velocity(v0, v_mid, v_space, v_time, known, bbox=1.0):
    for v in (v_mid, v_space, v_time):
        _compatible(v0, v, known)
    total = free_difference(v_mid, v0, known)
    space = free_difference(v_space, v0, known)
    time = free_difference(v_time, v0, known)
    cross = free_difference(free_difference(total, space, known), time, known)
    reconstructed = space + time + cross
    residual = free_difference(reconstructed, total, known)
    terms = {"space": space, "time": time, "cross": cross, "total": total}
    summaries = {name: vector_stats(value, known, bbox) for name, value in terms.items()}
    angles = {f"{a}__{b}": cosine_stats(terms[a], terms[b], known)
              for a, b in combinations(terms, 2)}
    summed_norm = sum(summaries[key]["l2_norm"] for key in ("space", "time", "cross"))
    total_norm = summaries["total"]["l2_norm"]
    # Conservative per-coordinate reconstruction tolerance for fixed FP32 operations.
    magnitude = sum(value[~known].double().abs() for value in (space, time, cross, total))
    tolerance = 16 * torch.finfo(torch.float32).eps * magnitude + torch.finfo(torch.float32).tiny
    passed = bool((residual[~known].double().abs() <= tolerance).all())
    return {"terms": terms, "identity_residual": residual,
            "summary": {"terms": summaries, "pairwise_angles": angles,
                        "identity_residual": vector_stats(residual, known, bbox),
                        "identity_max_coordinate_tolerance": float(tolerance.max()),
                        "identity_within_tolerance": passed,
                        "total_norm_over_sum_term_norms": total_norm / summed_norm if summed_norm else None,
                        "cancellation_fraction": 1 - total_norm / summed_norm if summed_norm else None,
                        "norm_note": "Vector terms may cancel; their norms are not additive."}}


@torch.no_grad()
def make_direction_probe(x, delta, known, eta, reference_states=None):
    """Normalize the actual defect, then form two actual FP32 perturbed states.

    A None tensor pair means no neural calls should be dispatched. Scale is
    max(RMS(x_free), 1 model coordinate unit), independent of fixed GT bbox.
    """
    _validate(x, known)
    _compatible(x, delta, known)
    if eta not in (1e-4, 5e-5):
        raise ValueError("Only preregistered eta=1e-4 or 5e-5")
    references = tuple(reference_states) if reference_states is not None else (x,)
    floor = resolution_floor(*references, known=known)
    delta_rms = _rms(delta[~known])
    base = {"eta": eta, "delta_coordinate_rms": delta_rms, "direction_resolution_floor": floor,
            "status": "DIRECTION_UNRESOLVED", "x_plus": None, "x_minus": None,
            "direction": None}
    if delta_rms <= floor or delta_rms == 0:
        return base
    scale = max(_rms(x[~known]), 1.0)
    direction = torch.zeros_like(x)
    direction[~known] = delta[~known] / delta_rms
    shift = direction[~known] * (eta * scale)
    plus, minus = x.clone(), x.clone()
    plus[~known], minus[~known] = x[~known] + shift, x[~known] - shift
    if not bits_equal(plus[known], x[known]) or not bits_equal(minus[known], x[known]):
        raise RuntimeError("Perturbation changed C bits")
    difference = free_difference(plus, minus, known)
    denominator = _rms(difference[~known])
    pair_floor = resolution_floor(plus, minus, known=known)
    base.update({"status": "RESOLVED" if denominator > pair_floor else "PERTURBATION_UNRESOLVED",
                 "x_plus": plus, "x_minus": minus, "direction": direction,
                 "direction_actual_coordinate_rms": _rms(direction[~known]), "scale": scale,
                 "actual_denominator_coordinate_rms": denominator,
                 "theoretical_denominator_coordinate_rms": 2 * eta * scale,
                 "actual_denominator_resolution_floor": pair_floor})
    if base["status"] != "RESOLVED":
        base["x_plus"] = base["x_minus"] = None
    return base


@torch.no_grad()
def direction_gain(f_plus, f_minus, probe, known, bbox=1.0):
    if probe["status"] != "RESOLVED":
        return {"status": probe["status"], "gain": None}
    difference = free_difference(f_plus, f_minus, known)
    summary = vector_stats(difference, known, bbox)
    denominator = probe["actual_denominator_coordinate_rms"]
    ratio = _ratio(summary["coordinate_rms"], denominator, probe["actual_denominator_resolution_floor"])
    return {"status": "RESOLVED" if ratio is not None else "DENOMINATOR_UNRESOLVED",
            "gain": ratio, "velocity_difference": summary,
            "actual_denominator_coordinate_rms": denominator,
            "theoretical_denominator_coordinate_rms": probe["theoretical_denominator_coordinate_rms"],
            "definition": "Finite-difference gain along the local-defect direction only"}


@torch.no_grad()
def propagation_summary(initial_delta, final_delta, known, bbox=1.0, reference_states=None):
    if reference_states is None:
        raise ValueError("Initial actual endpoint operands are required for the denominator resolution gate")
    initial = vector_stats(initial_delta, known, bbox)
    final = vector_stats(final_delta, known, bbox)
    floor = resolution_floor(*reference_states, known=known)
    amplification = _ratio(final["coordinate_rms"], initial["coordinate_rms"], floor)
    return {"initial_difference": initial, "final_difference": final,
            "amplification": amplification, "denominator_resolution_floor": floor,
            "status": "RESOLVED" if amplification is not None else "DENOMINATOR_UNRESOLVED",
            "definition": "Fixed Euler100 suffix-map coordinate amplification; not exact continuous ODE stability"}
