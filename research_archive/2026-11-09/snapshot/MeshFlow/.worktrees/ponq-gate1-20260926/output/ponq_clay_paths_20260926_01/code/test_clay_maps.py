"""Pure analytical-map tests; no meshing, training, or filesystem output."""

import json
import unittest
import numpy as np

from clay_maps import MAP_NAMES, TIME_STEPS, build_map, deform


def reference_box():
    signs = np.array([[a, b, c] for a in [-1., 1.]
                      for b in [-1., 1.] for c in [-1., 1.]])
    return signs * [.5, .7, .4] + [.08, -.03, .12]


class ClayMapTests(unittest.TestCase):
    def test_identity_all_maps_is_bitwise(self):
        points = reference_box()
        for name in MAP_NAMES:
            spec = build_map(name, points)
            moved, jacobian = deform(points, 0., spec)
            np.testing.assert_array_equal(moved, points)
            np.testing.assert_array_equal(jacobian,
                                          np.broadcast_to(np.eye(3), (len(points), 3, 3)))
            self.assertIsNot(moved, points)
            json.dumps(spec, allow_nan=False)
            self.assertEqual(spec["time_steps"], list(TIME_STEPS))

    def test_exact_jacobians_against_centered_finite_differences(self):
        points = reference_box()
        # Include points slightly outside the reference domain, without repair.
        probes = np.vstack([points, points * 1.03, np.array([[.123, .217, -.091]])])
        epsilon = 1e-6
        for name in MAP_NAMES:
            spec = build_map(name, points)
            for time in [0., .37, 1.]:
                _, jacobian = deform(probes, time, spec)
                numerical = np.empty_like(jacobian)
                for axis in range(3):
                    delta = np.eye(3)[axis] * epsilon
                    plus, _ = deform(probes + delta, time, spec)
                    minus, _ = deform(probes - delta, time, spec)
                    numerical[..., axis] = (plus - minus) / (2 * epsilon)
                np.testing.assert_allclose(jacobian, numerical, rtol=2e-9, atol=2e-10,
                                           err_msg=f"{name} at t={time}")

    def test_positive_determinants_in_reference_bbox(self):
        ref = reference_box()
        rng = np.random.default_rng(20260926)
        points = np.vstack([ref, rng.uniform(ref.min(0), ref.max(0), size=(100, 3))])
        for name in MAP_NAMES:
            spec = build_map(name, ref)
            for time in TIME_STEPS:
                _, jacobian = deform(points, time, spec)
                determinants = np.linalg.det(jacobian)
                self.assertGreaterEqual(determinants.min(), .35 - 1e-15)
                if name == "sphere_squash":
                    np.testing.assert_allclose(determinants, (1 + .2 * time) * (1 - .45 * time))
                elif name == "bottle_bend":
                    np.testing.assert_allclose(determinants, 1 + .25 * time)

    def test_thin_u_outside_domain_is_not_clipped(self):
        ref = np.array([[-1., 0., -.1], [1., 2., .1]])
        spec = build_map("thin_u_squeeze", ref)
        moved, jacobian = deform(np.array([[1., 4., 0.]]), 1., spec)
        self.assertAlmostEqual(moved[0, 0], -1.6)
        self.assertAlmostEqual(np.linalg.det(jacobian)[0], -1.6)
        self.assertEqual(moved[0, 1], 4.)

    def test_bottle_axis_frame_center_and_full_length(self):
        center = np.array([5., -2., .7])
        ref = np.array([[-1., -3., -2.], [1., 3., 2.]]) + center
        spec = build_map("bottle_bend", ref)
        self.assertEqual(spec["axial_axis"], 1)
        self.assertEqual(spec["frame_axis_order"], [2, 0, 1])
        self.assertEqual(spec["length"], 6.)
        self.assertEqual(np.linalg.det(np.asarray(spec["frame_basis_columns"])), 1.)
        probes = np.array([center, center + [0., 3., 0.], center - [0., 3., 0.]])
        moved, _ = deform(probes, 1., spec)
        np.testing.assert_array_equal(moved[0], center)
        np.testing.assert_allclose(moved[1:, 1] - center[1], [3.75, -3.75])
        np.testing.assert_allclose(moved[1:, 2] - center[2], [.055 * 6, .055 * 6])
        tied = build_map("bottle_bend", np.array([[-1., -1., -1.], [1., 1., 1.]]))
        self.assertEqual(tied["axial_axis"], 0)

    def test_batch_shapes_and_input_immutability(self):
        ref = reference_box()
        points = ref.reshape((2, 4, 3)).copy()
        original = points.copy()
        for name in MAP_NAMES:
            spec = build_map(name, ref)
            old_spec = json.dumps(spec, sort_keys=True)
            moved, jacobian = deform(points, .6, spec)
            self.assertEqual(moved.shape, (2, 4, 3))
            self.assertEqual(jacobian.shape, (2, 4, 3, 3))
            single, single_j = deform(points[0, 0], .6, spec)
            np.testing.assert_array_equal(single, moved[0, 0])
            np.testing.assert_array_equal(single_j, jacobian[0, 0])
            self.assertEqual(json.dumps(spec, sort_keys=True), old_spec)
        np.testing.assert_array_equal(points, original)

    def test_local_affine_full_quadric_energy_transport(self):
        # For nonlinear paths this tests the affine map defined by phi(v),
        # J(v), NOT an exact finite-neighborhood nonlinear pullback claim.
        ref = reference_box()
        rng = np.random.default_rng(271828)
        sites = rng.uniform(ref.min(0), ref.max(0), size=(20, 3))
        factors = rng.normal(size=(20, 4, 4))
        quadrics = factors @ factors.transpose(0, 2, 1)
        offsets = rng.normal(size=(20, 3)) * .05
        for name in MAP_NAMES:
            spec = build_map(name, ref)
            destination, jacobian = deform(sites, .7, spec)
            H = np.broadcast_to(np.eye(4), (len(sites), 4, 4)).copy()
            H[:, :3, :3] = jacobian
            H[:, :3, 3] = destination - np.einsum("nij,nj->ni", jacobian, sites)
            inverse = np.linalg.inv(H)
            transported = inverse.transpose(0, 2, 1) @ quadrics @ inverse
            original_probe = np.c_[sites + offsets, np.ones(len(sites))]
            affine_probe = np.c_[destination + np.einsum("nij,nj->ni", jacobian, offsets),
                                  np.ones(len(sites))]
            old = np.einsum("ni,nij,nj->n", original_probe, quadrics, original_probe)
            new = np.einsum("ni,nij,nj->n", affine_probe, transported, affine_probe)
            np.testing.assert_allclose(new, old, rtol=2e-14, atol=2e-14)

    def test_invalid_inputs_and_times_rejected(self):
        ref = reference_box()
        with self.assertRaises(ValueError):
            build_map("unknown", ref)
        with self.assertRaises(ValueError):
            build_map("thin_u_squeeze", np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            build_map("bottle_bend", np.zeros((3, 3)))
        spec = build_map("sphere_squash", ref)
        for bad_time in [-.1, 1.1, np.nan, [0., 1.]]:
            with self.assertRaises(ValueError):
                deform(ref, bad_time, spec)
        with self.assertRaises(ValueError):
            deform(np.array([[np.nan, 0., 0.]]), .5, spec)


if __name__ == "__main__":
    unittest.main()
