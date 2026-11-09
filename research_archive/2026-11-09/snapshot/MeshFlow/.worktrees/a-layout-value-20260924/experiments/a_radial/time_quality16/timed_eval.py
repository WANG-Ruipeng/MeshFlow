"""Synchronous, uncached CPU audit of a fixed TTQ observation.

This function has no file writes and no model. The optimizer never receives its
Monte Carlo results. References are the same frozen radial-field full surface;
identity and reference-array equality are bound by the runner before timing.
"""
from __future__ import annotations
import time
import numpy as np
from ..layout_student.deployment import named_quality, qualification
from ..layout_value.geometry import audit_mesh, evaluate_mesh

ANGLES = ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg')
MIN_GAIN_FRACTION = 0.02


def evaluate_candidate(u, faces, logr, support, ref_vertices, ref_faces, *,
                       floor=None, baseline_metrics=None, seeds, points=8192):
    """Return fresh ``(mesh, report)``; propagate errors to the case ledger.

    The baseline call supplies neither floor nor baseline_metrics. Every call,
    including a quality failure and repeated identical geometry, performs fresh
    radius queries, endpoint exact audit and both independent MC evaluations.
    Report timing excludes runner model/optimizer time and subsequent disk IO.
    """
    started = time.perf_counter()
    if (floor is None) != (baseline_metrics is None):
        raise ValueError('Supply both floor and baseline_metrics, or neither for baseline')
    seed_values = tuple(seeds)
    if len(seed_values) != 2 or any(isinstance(s, (bool, np.bool_)) or
            not isinstance(s, (int, np.integer)) or s < 0 for s in seed_values):
        raise ValueError('Exactly two nonnegative integer evaluation seeds required')
    if seed_values[0] == seed_values[1]:
        raise ValueError('Two independent MC seeds must differ')
    seed_values = tuple(int(s) for s in seed_values)
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or points <= 0:
        raise ValueError('Positive integer MC sample count required')
    points = int(points)
    directions = np.array(u, dtype=np.float64, copy=True)
    input_faces = np.asarray(faces)
    if directions.shape != (258, 3) or not np.isfinite(directions).all():
        raise ValueError('Finite original 258-node directions required; no repair is applied')
    if input_faces.ndim != 2 or input_faces.shape[1] != 3 or not np.issubdtype(input_faces.dtype, np.integer):
        raise ValueError('Integer triangle faces required')
    mesh_faces = np.array(input_faces, dtype=np.int64, copy=True)
    field = np.asarray(logr, dtype=np.float64)
    reference_v = np.asarray(ref_vertices, dtype=np.float64)
    reference_f = np.asarray(ref_faces)
    if field.shape != (2050,) or reference_v.shape != (2050, 3) or reference_f.shape != (4096, 3):
        raise ValueError('Frozen 2050-node / 4096-face same-field reference required')
    if not np.isfinite(field).all() or not np.isfinite(reference_v).all():
        raise ValueError('Nonfinite field or reference')
    baseline_call = baseline_metrics is None
    if not baseline_call:
        if len(baseline_metrics) != 2:
            raise ValueError('Both baseline MC measurements required')
        for seed, metric in zip(seed_values, baseline_metrics):
            if metric['seed'] != seed or metric['points_per_direction'] != points:
                raise ValueError('Baseline MC seeds/count differ from this observation')
            if not np.isfinite(metric['rms_bbox_percent']) or metric['rms_bbox_percent'] < 0:
                raise ValueError('Invalid baseline RMS')
        if any(not np.isfinite(floor[key]) for key in ANGLES):
            raise ValueError('Nonfinite quality floor')

    query_start = time.perf_counter()
    radii = np.asarray(support.query(field, directions), dtype=np.float64)
    if radii.shape != (258,) or not np.isfinite(radii).all():
        raise ValueError('Nonfinite or wrong-shaped accepted-direction radius query')
    vertices = directions * radii[:, None]
    mesh = {'u': directions, 'r': radii, 'vertices': vertices, 'faces': mesh_faces}
    query_seconds = time.perf_counter() - query_start
    quality_start = time.perf_counter()
    quality = named_quality(vertices, mesh_faces)
    quality_seconds = time.perf_counter() - quality_start
    exact_start = time.perf_counter()
    audit = audit_mesh(vertices, mesh_faces, exact=True)
    exact_seconds = time.perf_counter() - exact_start
    active_floor = dict(quality) if baseline_call else dict(floor)
    gate = qualification(quality, audit, active_floor, requested_faces=512)
    reasons = list(gate['failure_reasons'])
    if audit.get('exact_executed') is not True:
        reasons.append('exact_audit_not_executed')
    if len(mesh_faces) != 512 and 'face_budget' not in reasons:
        reasons.append('face_budget')
    # No early return on failed quality: preserve the complete fixed denominator.
    metrics, mc_seconds = [], []
    for seed in seed_values:
        mc_start = time.perf_counter()
        metrics.append(evaluate_mesh(vertices, mesh_faces, reference_v, reference_f, points=points, seed=seed))
        mc_seconds.append(time.perf_counter() - mc_start)
    for seed, metric in zip(seed_values, metrics):
        if metric['seed'] != seed or metric['points_per_direction'] != points:
            raise ValueError('Metric implementation returned unexpected seed/count')
        if not np.isfinite(metric['rms_bbox_percent']) or metric['rms_bbox_percent'] < 0:
            raise ValueError('Invalid measured RMS')
    denominator_defined = [True, True] if baseline_call else [m['rms_bbox_percent'] > 0 for m in baseline_metrics]
    gains = [0.0, 0.0] if baseline_call else [
        1.0 - measured['rms_bbox_percent'] / base['rms_bbox_percent'] if defined else None
        for measured, base, defined in zip(metrics, baseline_metrics, denominator_defined)]
    qualified = not reasons
    target_pass = bool(not baseline_call and qualified and all(
        gain is not None and gain >= MIN_GAIN_FRACTION for gain in gains))
    timing = {'fresh_radius_query_seconds': float(query_seconds),
              'quality_seconds': float(quality_seconds),
              'exact_audit_seconds': float(exact_seconds),
              'mc_seconds': [float(x) for x in mc_seconds],
              'total_seconds': float(time.perf_counter() - started)}
    report = {'quality': quality, 'audit': audit, 'quality_floor': active_floor,
              'qualified': qualified, 'failure_reasons': reasons,
              'metrics': metrics, 'metric_seeds': list(seed_values), 'points_each_direction': points,
              'gain_fractions': gains, 'rms_denominator_defined': denominator_defined,
              'minimum_gain_fraction': MIN_GAIN_FRACTION,
              'target_pass': target_pass, 'is_fixed_baseline': baseline_call,
              'actual_face_count': int(len(mesh_faces)), 'timing': timing,
              'surface_reference': 'same frozen radial mainfield full2050-node4096-face surface; not raw asset GT',
              'MC_cache_used': False, 'MC_used_for_optimizer_selection_or_stopping': False,
              'final_accepted_radius_freshly_queried': True,
              'timing_scope': 'input validation + fresh radius/vertices + quality + exact endpoint audit + two MC calls + gate; excludes model/optimizer and runner file writes'}
    return mesh, report
