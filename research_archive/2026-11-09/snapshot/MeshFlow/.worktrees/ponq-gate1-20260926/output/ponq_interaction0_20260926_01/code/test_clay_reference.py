"""Focused contracts for frozen-sample PL reference transport; no extraction."""
from pathlib import Path
import unittest
import numpy as np
from common import sample, load_mesh
from clay_reference import prepare_reference, refit_frame, replay_source_samples, shrink_plane_quadrics


class ClayReferenceTests(unittest.TestCase):
    def test_common_sample_replay_is_exact(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        expected = sample(v, f, 1024, 1701)
        actual = replay_source_samples(v, f, 1024, 1701)
        for x, y in zip(expected, [actual['points'], actual['normals'], actual['face_ids']]):
            np.testing.assert_array_equal(x, y)

    def test_shared_midpoint_connectivity_and_nonlinear_pl_samples(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        f = np.array([[0, 1, 2], [1, 3, 2]])
        prepared = prepare_reference(v, f, np.array([[.5, .5, 0.]]), 0, 3, samples_fit=2048)
        self.assertEqual(len(prepared['faces']), 2 * 4**3)
        self.assertEqual(len(prepared['vertices']), 81)  # Shared-edge 9 x 9 grid.
        a = prepared['association']
        self.assertLess(prepared['diagnostics']['maximum_barycentric_reconstruction_error'], 1e-14)
        self.assertTrue(prepared['diagnostics']['source_face_association_matches'])
        moved = prepared['vertices'].copy()
        moved[:, 2] = moved[:, 0]**2
        result = refit_frame(moved, prepared['faces'], a)
        tri = moved[prepared['faces'][a['sample_child_faces']]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        off_plane = np.sum((result['samples'] - tri[:, 0]) * normal, axis=1)
        self.assertLess(float(np.max(abs(off_plane))), 1e-16)
        # Analytical warp of the old point differs from the declared flat triangle.
        analytic = a['base_samples'].copy()
        analytic[:, 2] = analytic[:, 0]**2
        self.assertGreater(float(np.max(np.linalg.norm(analytic - result['samples'], axis=1))), 1e-5)
        np.testing.assert_array_equal(a['labels'], np.zeros(2048, dtype=int))

    def test_initial_refit_matches_three_archived_fits_after_common_shrink(self):
        output = Path(__file__).resolve().parents[3] / 'output'
        cases = [('ponq_gate1_synthetic_20260926_01', 'sphere', 0, 0),
                 ('ponq_gate1_synthetic_20260926_01', 'thin_u', 2, 4),
                 ('ponq_gate1_real_20260926_01', 'bottle_local_1', 6, 1)]
        for study, name, oi, levels in cases:
            with self.subTest(object=name):
                directory = output / study / 'inputs' / name
                self.assertTrue(directory.is_dir(), f'Required frozen fixture missing: {directory}')
                v, f = load_mesh(directory / 'reference.obj')
                with np.load(directory / 'fit_256.npz', allow_pickle=False) as archive:
                    p, Q, n, counts = [archive[key].copy() for key in ['p', 'Q', 'n', 'count']]
                prepared = prepare_reference(v, f, p, oi, levels)
                a = prepared['association']
                np.testing.assert_array_equal(np.bincount(a['labels'], minlength=len(p)), counts)
                current = refit_frame(prepared['vertices'], prepared['faces'], a)
                np.testing.assert_allclose(current['Q'], shrink_plane_quadrics(Q, .7), rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(current['n'], n, rtol=2e-10, atol=2e-12)
                np.testing.assert_array_equal(current['sample_weights'], np.ones(65536))
                self.assertLess(prepared['diagnostics']['maximum_barycentric_reconstruction_error'], 1e-13)
                print(name, 'Q_max_abs', float(np.max(abs(current['Q'] - shrink_plane_quadrics(Q, .7)))),
                      'n_max_abs', float(np.max(abs(current['n'] - n))),
                      'bary_max', prepared['diagnostics']['maximum_barycentric_reconstruction_error'], flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
