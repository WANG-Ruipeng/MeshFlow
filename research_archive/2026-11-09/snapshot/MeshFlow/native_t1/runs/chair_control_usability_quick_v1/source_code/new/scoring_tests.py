"""Small analytic CPU tests for only the new quick-pilot scoring definitions."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import tempfile
import numpy as np
from .scoring import (pr_metrics, surface_distance_squared, sample_mesh_surface,
    interface_metrics, gamma_points, nearest_segments, write_json, seed, failure_metrics)
from .aggregation import condition_summary, mean_fixed


def run(out):
    checks = []
    target = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    q = sample_mesh_surface(target, 128, np.random.RandomState(12))
    own = surface_distance_squared(q, target)
    m = pr_metrics(own, own, 1.)
    assert m['D_sym2048'] < 1e-25 and m['Fscore2048_0.005'] == 1.
    checks.append('identical_GT_free_has_zero_distance_and_unit_Fscore')
    shifted = target + np.array([0., 0., .015])
    p = sample_mesh_surface(shifted, 128, np.random.RandomState(12))
    a, b = surface_distance_squared(q, shifted), surface_distance_squared(p, target)
    m = pr_metrics(a, b, 1.)
    assert np.allclose([m['GT_to_F_MSE2048_L0'], m['F_to_GT_MSE2048_L0']], .015**2)
    assert m['Fscore2048_0.01'] == 0. and m['Fscore2048_0.02'] == 1.
    checks.append('normal_translation_two_directions_and_squared_threshold')
    scaled = pr_metrics(a*10000, b*10000, 100.)
    assert np.isclose(scaled['D_sym2048'], m['D_sym2048']) and scaled['Fscore2048_0.01'] == m['Fscore2048_0.01']
    checks.append('coordinate_and_L0_scale_invariance')
    # Copying a distinct C cannot enter the free-only nearest surface target.
    copied_C = target.copy(); bad_free = target + np.array([0., 0., 1.])
    assert surface_distance_squared(q, copied_C).max() < 1e-25
    assert surface_distance_squared(q, bad_free).min() > .99
    assert pr_metrics(surface_distance_squared(q, bad_free), surface_distance_squared(q+np.array([0., 0., 1.]), target), 1.)['Fscore2048_0.05'] == 0.
    checks.append('copied_C_cannot_receive_free_quality_credit')
    d, nearest, _ = nearest_segments(np.array([[2., 1., 0.], [.5, 1., 0.]]), np.array([[[0., 0., 0.], [1., 0., 0.]]]))
    assert np.allclose(d, [2., 1.]) and np.allclose(nearest[1], [.5, 0., 0.])
    checks.append('finite_segments_not_infinite_lines_or_endpoints')
    edges = np.array([[[0., 0., 0.], [1., 0., 0.]], [[0., 0., 0.], [0., 2., 0.]]])
    gp, w = gamma_points(edges, 32)
    assert len(gp) == 64 and np.isclose(w.sum(), 3.) and np.isclose(w[32:].sum(), 2.)
    assert np.allclose(gp[0], [1/64, 0., 0.])
    gm, _ = interface_metrics(target, edges[:1], 1., 32)
    assert gm['Gamma32_edge_RMS_L0'] < 1e-15 and gm['Gamma32_edge_AttachCov_0.001'] == 1.
    assert interface_metrics(target, np.empty((0, 2, 3)), 1.)[0]['Gamma32_status'] == 'NOT_APPLICABLE'
    checks.append('Gamma_uniform_midpoints_arclength_weights_and_missing_NA')
    rows = []
    for candidate, distance in enumerate((.3, .2, .1, .4)):
        rows.append(dict(uid='fixture', condition_id='fixture_C1', context_level=1, candidate=candidate,
            arm='OT_BASE', panel='main', status='PASS', D_sym2048=distance,
            generation_seconds=1., **{'Fscore2048_0.01': 10*distance}))
    # Disable only geometry for the algebra fixture by withholding vector paths.
    with tempfile.TemporaryDirectory(dir=Path(out)) as temporary:
        c, _ = condition_summary(temporary, rows, 4)
        assert c['GT_oracle_Best_at2_candidate'] == 1 and c['GT_oracle_Best_at4_candidate'] == 2
        assert c['GT_oracle_Best_at4_Fscore2048_0.01'] == 1.
        assert np.isclose(c['D_sym2048_mean'], .25)
        incomplete, _ = condition_summary(temporary, rows[:2], 4)
        assert incomplete['GT_oracle_Best_at4_D_sym2048'] is None and incomplete['D_sym2048_mean'] is None
        failed = [dict(r) for r in rows]
        failed[3] = failure_metrics(dict(failed[3], status='GENERATION_FAILED'))
        # Separate fixture folder avoids a cache with intentionally changed data.
        cfail, _ = condition_summary(str(Path(temporary)/'failure'), failed, 4)
        assert np.isinf(cfail['D_sym2048_mean']) and cfail['failed_candidates'] == 1
        assert cfail['GT_oracle_Best_at4_candidate'] == 2
    checks.append('fixed_oracle_winner_all_metrics_missing_and_failure_denominators')
    assert mean_fixed([1., 3.], 2) == 2. and mean_fixed([1.], 2) is None
    assert seed('Q_Y', 'same', 2048) == seed('Q_Y', 'same', 2048)
    assert 'torch' not in sys.modules
    checks.append('fixed_parent_denominator_stateless_queries_no_torch')
    result = dict(status='PASS', checks=checks, model_calls=0, backward=0, optimizer_updates=0, torch_imported=False)
    write_json(Path(out)/'scoring_cpu_tests.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True); print(run(a.out))
