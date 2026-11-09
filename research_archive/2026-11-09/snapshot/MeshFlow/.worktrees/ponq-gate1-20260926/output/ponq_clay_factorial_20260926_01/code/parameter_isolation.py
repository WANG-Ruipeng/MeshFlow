"""Pure parameter controls for the existing full-quadric PoNQ experiment.

No extraction, fitting, files, torch, or SVD vertex relocation occurs here.
The supplied ``v`` is the original extraction site, computed upstream using
truncated SVD. It need not minimize the full quadric: its centered linear
term can be nonzero. Both that term and the constant residual are retained.

``fixed_v_metric_*`` names the intervention, not a promise that the centered linear
term is unchanged. A full local congruence sends (A,g,c) to
(S.T A S,S.T g,c), while the extraction sites and normals remain fixed.
"""

import numpy as np


STATES = (
    "roundtrip",
    "fixed_v_metric_light",
    "fixed_v_metric_strong",
    "position_matched_q_light",
    "position_matched_q_strong",
)


def _finite_array(value, name, shape=None):
    array = np.array(value, dtype=np.float64, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite values")
    return array


def _homogeneous_translation(delta):
    matrix = np.broadcast_to(np.eye(4), (len(delta), 4, 4)).copy()
    matrix[:, :3, 3] = delta
    return matrix


def _congruence(matrix, transform):
    return transform.transpose(0, 2, 1) @ matrix @ transform


def _relative_frobenius(new, old):
    difference = np.linalg.norm(new - old, axis=(-2, -1))
    denominator = np.linalg.norm(old, axis=(-2, -1))
    # Zero/zero is zero. A nonzero difference against a zero reference is
    # reported separately rather than generating infinity in JSON audits.
    ratio = np.divide(difference, denominator, out=np.zeros_like(difference),
                      where=denominator != 0)
    return dict(mean=float(ratio.mean()), maximum=float(ratio.max()),
                nonzero_change_against_zero_reference=int(np.sum(
                    (denominator == 0) & (difference != 0))))


def _full_quadric_legality_audit(quadrics):
    symmetric = (quadrics + quadrics.transpose(0, 2, 1)) / 2
    values = np.linalg.eigvalsh(symmetric)
    scale = np.maximum(np.linalg.norm(quadrics, axis=(1, 2)),
                       np.finfo(np.float64).tiny)
    tolerance = 128 * np.finfo(np.float64).eps * scale
    asymmetry = np.max(abs(quadrics - quadrics.transpose(0, 2, 1)), axis=(1, 2))
    if np.any(asymmetry > tolerance):
        raise ValueError("Full Q is materially asymmetric; no repair was applied")
    if np.any(values[:, 0] < -tolerance):
        raise ValueError("Full Q is materially indefinite; no repair was applied")
    return dict(min_symmetric_eigenvalue=float(values.min()),
                max_asymmetry=float(asymmetry.max()),
                negative_eigenvalues_including_roundoff=int(np.sum(values < 0)),
                per_matrix_psd_roundoff_tolerance=tolerance.tolist(),
                psd_within_recorded_float64_roundoff_tolerance=True,
                eigenvalue_repair_performed=False)


def principal_psd_sqrt(matrix):
    """Return the principal PSD square root and a transparent float audit.

    Only roundoff-scale negative eigenvalues are clipped for forming F.
    Encoding stores the residual separately, so this clipping never silently
    regularizes or replaces the original A. Material indefiniteness is an
    error. No positive eigenvalue floor or rank promotion is applied.
    """
    matrix = _finite_array(matrix, "A")
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3) or len(matrix) == 0:
        raise ValueError("A must have nonempty shape (N,3,3)")
    scale = np.maximum(np.linalg.norm(matrix, axis=(1, 2)),
                       np.finfo(np.float64).tiny)
    tolerance = 128 * np.finfo(np.float64).eps * scale
    asymmetry = np.max(abs(matrix - matrix.transpose(0, 2, 1)), axis=(1, 2))
    if np.any(asymmetry > tolerance):
        raise ValueError("A is materially asymmetric")
    symmetric = (matrix + matrix.transpose(0, 2, 1)) / 2
    values, vectors = np.linalg.eigh(symmetric)
    if np.any(values[:, 0] < -tolerance):
        raise ValueError("A is materially indefinite; no repair was applied")
    factor = (vectors * np.sqrt(np.maximum(values, 0))[:, None, :]) @ vectors.transpose(0, 2, 1)
    residual = matrix - factor @ factor.transpose(0, 2, 1)
    audit = dict(
        input_A_min_eigenvalue=float(values.min()),
        input_A_max_asymmetry=float(asymmetry.max()),
        negative_eigenvalues_clipped_in_factor_only=int(np.sum(values < 0)),
        maximum_negative_eigenvalue_magnitude_clipped=float(max(0., -values.min())),
        factor_residual_max_abs=float(np.max(abs(residual))),
        factor_residual_relative_frobenius=_relative_frobenius(
            factor @ factor.transpose(0, 2, 1), matrix),
        positive_eigenvalue_floor=0.,
        original_A_retained_via_explicit_residual=True,
    )
    return factor, audit


def encode_centered(quadrics, sites):
    """Encode Q around sites as (F,g,c,explicit roundoff residual).

    For x=v+y, [x,1]^T Q [x,1] = [y,1]^T C [y,1]. The full C is
    reconstructed as [[F F.T,g],[g.T,c]] + residual. The residual preserves
    sqrt roundoff and any roundoff asymmetry; g and c are never discarded.
    """
    sites = _finite_array(sites, "v")
    if sites.ndim != 2 or sites.shape[1] != 3 or len(sites) == 0:
        raise ValueError("v must have nonempty shape (N,3)")
    quadrics = _finite_array(quadrics, "Q", (len(sites), 4, 4))
    centered = _congruence(quadrics, _homogeneous_translation(sites))
    factor, audit = principal_psd_sqrt(centered[:, :3, :3])
    linear = centered[:, :3, 3].copy()
    constant = centered[:, 3, 3].copy()
    reconstructed = np.zeros_like(centered)
    reconstructed[:, :3, :3] = factor @ factor.transpose(0, 2, 1)
    reconstructed[:, :3, 3] = linear
    reconstructed[:, 3, :3] = linear
    reconstructed[:, 3, 3] = constant
    residual = centered - reconstructed
    encoding = dict(v=sites, F=factor, g=linear, c=constant,
                    residual=residual)
    audit.update(
        centered_linear_norm_mean=float(np.linalg.norm(linear, axis=1).mean()),
        centered_linear_norm_max=float(np.linalg.norm(linear, axis=1).max()),
        centered_constant_min=float(constant.min()),
        centered_constant_max=float(constant.max()),
        centered_full_residual_max_abs=float(np.max(abs(residual))),
        center_is_prescribed_truncated_svd_site=True,
        centered_linear_term_forced_to_zero=False,
        constant_term_forced_to_zero=False,
    )
    return encoding, audit


def decode_centered(encoding):
    """Reconstitute the full Q, retaining the explicit encoding residual."""
    sites = _finite_array(encoding["v"], "v")
    count = len(sites)
    factor = _finite_array(encoding["F"], "F", (count, 3, 3))
    linear = _finite_array(encoding["g"], "g", (count, 3))
    constant = _finite_array(encoding["c"], "c", (count,))
    residual = _finite_array(encoding["residual"], "residual", (count, 4, 4))
    centered = np.zeros((count, 4, 4), dtype=np.float64)
    centered[:, :3, :3] = factor @ factor.transpose(0, 2, 1)
    centered[:, :3, 3] = linear
    centered[:, 3, :3] = linear
    centered[:, 3, 3] = constant
    centered += residual
    return _congruence(centered, _homogeneous_translation(-sites))


def make_parameters(base, state, obj_index=0, *, source_params=None,
                    seed=271828, svd_eps=.05):
    """Return (params, audit) for one predeclared isolation control.

    ``base`` is a mapping with p, v, n, Q, eigs (as in fit_256.npz).
    Position-matched states require the corresponding saved original qem
    arm's parameters in ``source_params``. Only its v is used. Strong and
    light A interventions share their random spatial-transform direction.
    Returned arrays are copies; neither input mapping is mutated.
    """
    if state not in STATES:
        raise ValueError(f"Unknown isolation state {state!r}")
    count = len(base["v"])
    shapes = dict(p=(count, 3), v=(count, 3), n=(count, 3),
                  Q=(count, 4, 4), eigs=(count, 3))
    params = {key: np.array(value, copy=True) for key, value in base.items()}
    for key, shape in shapes.items():
        params[key] = _finite_array(base[key], key, shape)
    original = {key: params[key].copy() for key in shapes}
    input_Q_legality = _full_quadric_legality_audit(original["Q"])
    encoding, encoding_audit = encode_centered(original["Q"], original["v"])
    roundtrip = decode_centered(encoding)
    centered_original = _congruence(original["Q"],
                                   _homogeneous_translation(original["v"]))
    alpha = 0.
    transform = None
    expected_centered = centered_original

    if state == "roundtrip":
        params["Q"] = roundtrip
    elif state.startswith("fixed_v_metric_"):
        alpha = .10 if state.endswith("strong") else .02
        rng = np.random.default_rng(int(seed) + 100 * int(obj_index) + 11)
        direction = rng.normal(size=(count, 3, 3))
        direction *= (np.sqrt(3.) / np.linalg.norm(direction, axis=(1, 2)))[:, None, None]
        transform = np.eye(3)[None] + alpha * direction
        homogeneous = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        homogeneous[:, :3, :3] = transform
        expected_centered = _congruence(centered_original, homogeneous)
        params["Q"] = _congruence(expected_centered,
                                   _homogeneous_translation(-original["v"]))
        # The extractor uses the A singular spectrum for normalization. This
        # computation does NOT recompute or relocate the extraction sites.
        params["eigs"] = np.linalg.svd(params["Q"][:, :3, :3], compute_uv=False)
        params["spatial_linear_transform"] = transform
    else:
        if source_params is None:
            raise ValueError("Position-matched control requires saved source_params")
        source_sites = _finite_array(source_params["v"], "source.v", (count, 3))
        delta = source_sites - original["v"]
        params["Q"] = _congruence(original["Q"], _homogeneous_translation(-delta))
        # Direct copy matches the original arm bit for bit, avoiding a second
        # subtraction/addition round trip in the prescribed extraction sites.
        params["v"] = source_sites.copy()
        params["matched_position_delta"] = delta

    output_Q_legality = _full_quadric_legality_audit(params["Q"])
    final_encoding, final_encoding_audit = encode_centered(params["Q"], params["v"])
    params.update(local_factor=final_encoding["F"],
                  local_linear=final_encoding["g"],
                  local_constant=final_encoding["c"],
                  local_reconstruction_residual=final_encoding["residual"])
    actual_centered = _congruence(params["Q"],
                                 _homogeneous_translation(params["v"]))
    site_shift = np.linalg.norm(params["v"] - original["v"], axis=1)
    a_original, a_output = original["Q"][:, :3, :3], params["Q"][:, :3, :3]
    original_eigs, output_eigs = original["eigs"], params["eigs"]
    roundtrip_spectrum = np.linalg.svd(roundtrip[:, :3, :3], compute_uv=False)
    original_spectrum = np.linalg.svd(a_original, compute_uv=False)
    original_rank = np.sum(original_eigs > svd_eps * original_eigs[:, :1], axis=1)
    output_rank = np.sum(output_eigs > svd_eps * output_eigs[:, :1], axis=1)
    audit = dict(
        version="PONQ_PARAMETER_ISOLATION_V1", state=state,
        elements=count, seed=int(seed), object_index=int(obj_index),
        dose_spatial_transform_relative_frobenius=alpha,
        dose_comparable_to_old_full_quadric_factor_noise=False,
        svd_vertex_relocation_performed=False,
        v_is_not_assumed_exact_argmin=True,
        source_encoding=encoding_audit, output_encoding=final_encoding_audit,
        input_Q_legality=input_Q_legality, output_Q_legality=output_Q_legality,
        zero_noise_Q_roundtrip_max_abs=float(np.max(abs(roundtrip - original["Q"]))),
        zero_noise_Q_roundtrip_relative_frobenius=_relative_frobenius(roundtrip, original["Q"]),
        zero_noise_prescribed_vertex_roundtrip_max_abs_error=float(np.max(abs(encoding["v"] - original["v"]))),
        zero_noise_A_spectrum_roundtrip_max_abs_error=float(np.max(abs(roundtrip_spectrum - original_spectrum))),
        zero_noise_A_spectrum_max_abs_difference_from_saved=float(np.max(abs(roundtrip_spectrum - original_eigs))),
        output_Q_relative_frobenius=_relative_frobenius(params["Q"], original["Q"]),
        output_A_relative_frobenius=_relative_frobenius(a_output, a_original),
        output_Q_min_symmetric_eigenvalue=float(np.linalg.eigvalsh(
            (params["Q"] + params["Q"].transpose(0, 2, 1)) / 2).min()),
        output_Q_max_asymmetry=float(np.max(abs(params["Q"] - params["Q"].transpose(0, 2, 1)))),
        prescribed_centered_congruence_max_abs_error=float(np.max(abs(actual_centered - expected_centered))),
        extraction_sites_bitwise_unchanged=bool(np.array_equal(params["v"], original["v"])),
        fitting_points_bitwise_unchanged=bool(np.array_equal(params["p"], original["p"])),
        normals_bitwise_unchanged=bool(np.array_equal(params["n"], original["n"])),
        A_bitwise_unchanged=bool(np.array_equal(a_output, a_original)),
        extraction_site_shift_mean=float(site_shift.mean()),
        extraction_site_shift_max=float(site_shift.max()),
        singular_spectrum_recomputed_without_relocation=state.startswith("fixed_v_metric_"),
        truncated_rank_threshold=float(svd_eps),
        thresholded_A_rank_changes=int(np.sum(output_rank != original_rank)),
    )
    if transform is not None:
        ratios = np.linalg.norm(transform - np.eye(3)[None], axis=(1, 2)) / np.sqrt(3.)
        audit.update(
            actual_spatial_transform_relative_frobenius_mean=float(ratios.mean()),
            actual_spatial_transform_relative_frobenius_max=float(ratios.max()),
            spatial_transform_min_singular_value=float(np.linalg.svd(transform, compute_uv=False).min()),
            centered_constant_change_max_abs=float(np.max(abs(actual_centered[:, 3, 3] - centered_original[:, 3, 3]))),
            centered_linear_change_norm_mean=float(np.linalg.norm(actual_centered[:, :3, 3] - centered_original[:, :3, 3], axis=1).mean()),
            centered_linear_change_norm_max=float(np.linalg.norm(actual_centered[:, :3, 3] - centered_original[:, :3, 3], axis=1).max()),
            centered_linear_is_transformed_not_discarded=True,
            dose_definition="||S-I||_F / ||I_3||_F; spatial linear map, not Q factor noise",
            interpretation="local full-Q congruence at fixed extraction sites; both A and residual g transform",
        )
    elif state.startswith("position_matched_q_"):
        audit.update(
            matched_source_sites_bitwise_equal=bool(np.array_equal(params["v"], source_params["v"])),
            matched_source_delta_max_abs_error=float(np.max(abs(
                params["v"] - original["v"] - params["matched_position_delta"]))),
            matched_source_full4_factor_relative_frobenius=.10 if state.endswith("strong") else .02,
            matched_source_dose_definition="source full4 principal-factor relative Frobenius noise; this arm applies only the saved source site displacement",
            interpretation="original full Q translated to match source qem-arm sites; original A and normals retained",
        )
    else:
        audit["interpretation"] = "zero-noise centered reencoding including g, c and explicit factor roundoff residual"
    return params, audit
