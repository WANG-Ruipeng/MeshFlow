"""CPU-only off-path endpoint corrective regression for the frozen Bench stream.

This is NOT the CFM derivative of the perturbed interpolation.  The experiment
deliberately supervises a one-step return to the unchanged target.  Only valid
free corners change; the source layout, OT assignment and loss denominator do
not.  This module has no torch/model imports and touches no global RNG.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

VERSION = "bench_off_path_endpoint_corrective_v1"
SALT = "R2_BENCH_STATE_SCHEDULE_V1/PERTURBED_ENDPOINT_CORRECTIVE/v1"
PROBABILITY = 0.25
TIME_MIN = 0.3
TIME_MAX = 0.9
ALPHA_MIN = 0.3
ALPHA_MAX = 0.8


def _json_value(value):
    """Canonicalize row metadata, never model/sample arrays."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Schedule row keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("Schedule row must contain finite JSON metadata")


def derive_row_randomness(row):
    """Independent, reproducible gate/alpha uniforms from the frozen row.

    The row is canonical JSON (sorted keys); separate SHA256 domains supply
    independent streams.  The upper 53 bits map to [0,1) exactly as a binary64
    uniform.  No seed is derived from worker ID, arm, completion order, or a
    mutable random state.  Caller must pass the original frozen schedule row.
    """
    if not isinstance(row, Mapping) or not row:
        raise ValueError("A nonempty frozen schedule row is required")
    payload = json.dumps(
        _json_value(row), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    row_sha = hashlib.sha256(payload).hexdigest()

    def uniform(domain):
        digest = hashlib.sha256(
            SALT.encode("utf-8") + b"\0" + domain.encode("ascii") + b"\0" + payload
        ).digest()
        return (int.from_bytes(digest[:8], "big") >> 11) * (2.0 ** -53)

    gate, alpha_uniform = uniform("gate"), uniform("alpha")
    return dict(
        salt=SALT, canonical_row_sha256=row_sha, gate_uniform=gate,
        alpha_uniform=alpha_uniform,
        alpha_candidate=ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * alpha_uniform,
        selected_by_probability=gate < PROBABILITY,
    )


def _validated_arrays(sample):
    arrays = {}
    for name in ("xt", "x1", "u"):
        value = np.asarray(sample[name])
        if (value.dtype != np.float32 or value.ndim != 3
                or value.shape[1:] != (3, 3) or not np.isfinite(value).all()):
            raise ValueError(f"{name} must be finite FP32 [faces,3,3]")
        arrays[name] = value
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("State, target and velocity shapes differ")
    faces = len(arrays["xt"])
    valid = np.asarray(sample["valid_mask"])
    known = np.asarray(sample["known_mask"])
    anchors = np.asarray(sample["anchor_ids"])
    free_recorded = np.asarray(sample["free_corner_mask"])
    if (valid.dtype != np.bool_ or known.dtype != np.bool_
            or valid.shape != (faces,) or known.shape != (faces,)):
        raise ValueError("Expected boolean face-valid and original-C masks")
    if anchors.shape != (faces, 3) or not np.issubdtype(anchors.dtype, np.integer):
        raise ValueError("Expected integer anchor IDs [faces,3]")
    if np.any(known & ~valid) or not known.any():
        raise ValueError("Original C must contain valid faces")
    if np.any(anchors[known] < 0):
        raise ValueError("Every original C corner must be anchored")
    fixed = anchors >= 0
    if np.any(fixed & ~valid[:, None]):
        raise ValueError("Padding cannot contain an anchored corner")
    free = valid[:, None] & ~fixed
    if (free_recorded.dtype != np.bool_ or free_recorded.shape != (faces, 3)
            or not np.array_equal(free_recorded, free)):
        raise ValueError("Recorded free-corner mask differs from valid anchors")
    denominator = int(np.asarray(sample["free_coordinate_count"]).item())
    if not free.any() or denominator != int(free.sum()) * 3:
        raise ValueError("Original free scalar denominator is inconsistent")
    if (not np.array_equal(arrays["xt"][fixed], arrays["x1"][fixed])
            or np.any(arrays["u"][fixed] != 0)):
        raise ValueError("Original fixed coordinates/zero target are violated")
    time_array = np.asarray(sample["t"])
    if time_array.size != 1:
        raise ValueError("Expected one sample time")
    time = float(time_array.item())
    if not math.isfinite(time) or not 0 <= time <= 1:
        raise ValueError("Time must lie in [0,1]")
    return arrays, valid, known, fixed, free, denominator, time


def apply_perturbation(sample, row, *, enabled=True):
    """Return ``(sample_copy, audit)`` with frozen row-wise corruption policy.

    At eligible times and the deterministic 25% gate, compute
        d = alpha * (mean(original C corners) - mean(target free corners))
        xt'[free] = xt[free] + t*d
        u'[free]  = (x1[free] - xt'[free])/(1-t).
    Means accumulate in FP64; d and state/target arithmetic are FP32, matching
    the existing stream's coordinate precision.  The means use neither unique
    vertex de-duplication nor surface-area weighting.  Source target geometry is
    training-only, never passed to inference.

    Disabled/ineligible/gate-rejected paths preserve EVERY original array
    byte, including u (do not recompute its nominal straight-path identity).
    """
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be an explicit boolean")
    arrays, valid, known, fixed, free, denominator, time = _validated_arrays(sample)
    random = derive_row_randomness(row)
    # Compare against the same FP32 boundary representation as stored times.
    eligible = float(np.float32(TIME_MIN)) <= time <= float(np.float32(TIME_MAX))
    applied = bool(enabled and eligible and random["selected_by_probability"])
    before = arrays["u"][free].astype(np.float64)
    old_sse = float(np.square(before).sum())
    audit = dict(
        version=VERSION, objective="OFF_PATH_ENDPOINT_CORRECTIVE_REGRESSION_NOT_PATH_CFM",
        enabled=bool(enabled), eligible=bool(eligible), applied=applied, t=time,
        probability=PROBABILITY, time_interval=[TIME_MIN, TIME_MAX],
        alpha_range=[ALPHA_MIN, ALPHA_MAX], **random,
        free_coordinate_count=denominator, free_corner_count=int(free.sum()),
        fixed_corner_count=int(fixed.sum()), original_C_face_count=int(known.sum()),
        valid_face_count=int(valid.sum()),
        alpha=None, mu_C=None, mu_Y_free=None, displacement_model_xyz=[0.0, 0.0, 0.0],
        d_norm_model=0.0, state_shift_norm_model=0.0,
        original_target_velocity_sse=old_sse, target_velocity_sse=old_sse,
        target_velocity_l2_gain=1.0 if old_sse > 0 else None,
        target_velocity_mse_gain=1.0 if old_sse > 0 else None,
        target_norm_gain_status="DEFINED" if old_sse > 0 else "NA_ZERO_ORIGINAL_NORM",
        fixed_coordinates_unchanged=True, fixed_velocity_target_zero=True,
        free_denominator_unchanged=True, endpoint_identity_max_abs=None,
        center_definition="original_valid_C_corner_mean; target_valid_free_corner_mean",
        global_rng_touched=False,
    )
    copied = dict(sample)
    if not applied:
        audit["reason"] = (
            "DISABLED" if not enabled else "OUTSIDE_TIME_WINDOW" if not eligible
            else "PROBABILITY_GATE_NOT_SELECTED"
        )
        return copied, audit

    alpha = float(random["alpha_candidate"])
    center_C = arrays["x1"][known].astype(np.float64).mean(axis=(0, 1))
    center_Y = arrays["x1"][free].astype(np.float64).mean(axis=0)
    delta = np.asarray(alpha * (center_C - center_Y), dtype=np.float32)
    time32 = np.float32(time)
    remainder = np.float32(1) - time32
    if remainder <= 0:
        raise ValueError("Corrective target requires a strictly positive remaining time")
    shift = np.asarray(time32 * delta, dtype=np.float32)
    xt_new, u_new = arrays["xt"].copy(), arrays["u"].copy()
    xt_new[free] = arrays["xt"][free] + shift
    u_new[free] = (arrays["x1"][free] - xt_new[free]) / remainder
    if not np.isfinite(xt_new).all() or not np.isfinite(u_new).all():
        raise FloatingPointError("Nonfinite off-path state or target")
    if (not np.array_equal(xt_new[~free], arrays["xt"][~free])
            or not np.array_equal(u_new[~free], arrays["u"][~free])):
        raise AssertionError("Correction changed a fixed or padded corner")
    copied["xt"], copied["u"] = xt_new, u_new
    new_sse = float(np.square(u_new[free].astype(np.float64)).sum())
    residual = (
        xt_new[free].astype(np.float64)
        + float(remainder) * u_new[free].astype(np.float64)
        - arrays["x1"][free].astype(np.float64)
    )
    ratio = new_sse / old_sse if old_sse > 0 else None
    audit.update(
        reason="APPLIED", alpha=alpha, mu_C=center_C.tolist(), mu_Y_free=center_Y.tolist(),
        displacement_model_xyz=delta.tolist(),
        d_norm_model=float(np.linalg.norm(delta.astype(np.float64))),
        state_shift_norm_model=float(np.linalg.norm(shift.astype(np.float64))),
        target_velocity_sse=new_sse,
        target_velocity_l2_gain=math.sqrt(ratio) if ratio is not None else None,
        target_velocity_mse_gain=ratio,
        endpoint_identity_max_abs=float(np.abs(residual).max()),
        actual_fp32_remainder=float(remainder),
    )
    return copied, audit


perturb_sample = apply_perturbation
