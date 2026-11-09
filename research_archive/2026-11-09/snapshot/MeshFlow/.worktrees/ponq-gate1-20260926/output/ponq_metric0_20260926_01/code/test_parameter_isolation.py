"""Pure-mathematics tests: no extractor, fitting, filesystem, or GPU."""
import unittest
import numpy as np

from parameter_isolation import (
    encode_centered, decode_centered, make_parameters, principal_psd_sqrt,
)


def example_base():
    # Second eigenvalue is deliberately below the frozen .05 SVD threshold.
    # Thus v[1] stays at p[1] although Q's true minimizer has y=-.4.
    A = np.array([np.diag([1., .01, 0.]), np.diag([1., .5, .2])])
    minimizer = np.array([[.25, -.4, .7], [-.1, .3, .2]])
    Q = np.zeros((2, 4, 4))
    Q[:, :3, :3] = A
    b = -np.einsum("nij,nj->ni", A, minimizer)
    Q[:, :3, 3] = b
    Q[:, 3, :3] = b
    Q[:, 3, 3] = np.einsum("ni,nij,nj->n", minimizer, A, minimizer) + [.2, .03]
    return dict(p=np.array([[0., 0., .3], [.1, .2, .4]]),
                v=np.array([[.25, 0., .3], [-.1, .3, .2]]),
                n=np.array([[0., 0., .7], [.3, .2, .1]]),
                Q=Q, eigs=np.array([[1., .01, 0.], [1., .5, .2]]))


def score(Q, xyz):
    homogeneous = np.c_[xyz, np.ones(len(xyz))]
    return np.einsum("ni,nij,nj->n", homogeneous, Q, homogeneous)


class ParameterIsolationTests(unittest.TestCase):
    def test_roundtrip_keeps_truncated_linear_and_constant_residual(self):
        base = example_base()
        encoding, audit = encode_centered(base["Q"], base["v"])
        self.assertAlmostEqual(encoding["g"][0, 1], .004)
        self.assertAlmostEqual(encoding["c"][0], .2016)
        self.assertGreater(audit["centered_linear_norm_max"], .0039)
        np.testing.assert_allclose(decode_centered(encoding), base["Q"], rtol=0, atol=2e-16)
        params, result = make_parameters(base, "roundtrip")
        np.testing.assert_array_equal(params["v"], base["v"])
        np.testing.assert_allclose(params["Q"], base["Q"], rtol=0, atol=2e-16)
        self.assertFalse(result["svd_vertex_relocation_performed"])

    def test_principal_factor_keeps_rank_and_explicit_roundoff(self):
        A = np.array([np.diag([1., 0., 0.]), np.diag([1., .01, -1e-16])])
        factor, audit = principal_psd_sqrt(A)
        self.assertEqual(audit["negative_eigenvalues_clipped_in_factor_only"], 1)
        self.assertEqual(factor[0, 1, 1], 0.)
        self.assertEqual(factor[0, 2, 2], 0.)
        Q = np.zeros((2, 4, 4)); Q[:, :3, :3] = A
        encoding, _ = encode_centered(Q, np.zeros((2, 3)))
        np.testing.assert_array_equal(decode_centered(encoding), Q)
        with self.assertRaisesRegex(ValueError, "indefinite"):
            principal_psd_sqrt(np.array([np.diag([1., .01, -.001])]))

    def test_fixed_v_metric_is_full_congruence_at_fixed_sites(self):
        base = example_base()
        params, audit = make_parameters(base, "fixed_v_metric_strong", obj_index=2)
        for key in ["v", "p", "n"]:
            np.testing.assert_array_equal(params[key], base[key])
        self.assertFalse(audit["A_bitwise_unchanged"])
        S = params["spatial_linear_transform"]
        expected_A = S.transpose(0, 2, 1) @ base["Q"][:, :3, :3] @ S
        np.testing.assert_allclose(params["Q"][:, :3, :3], expected_A, rtol=0, atol=2e-16)
        old_encoding, _ = encode_centered(base["Q"], base["v"])
        expected_g = np.einsum("nji,nj->ni", S, old_encoding["g"])
        np.testing.assert_allclose(params["local_linear"], expected_g, rtol=0, atol=3e-16)
        np.testing.assert_allclose(params["local_constant"], old_encoding["c"], rtol=0, atol=3e-16)
        self.assertGreaterEqual(audit["output_Q_min_symmetric_eigenvalue"], -1e-14)
        self.assertAlmostEqual(audit["actual_spatial_transform_relative_frobenius_mean"], .1)
        np.testing.assert_allclose(params["eigs"], np.linalg.svd(expected_A, compute_uv=False))
        xyz = np.array([[.2, .1, -.3], [-.2, .4, .7]])
        transformed = base["v"] + np.einsum("nij,nj->ni", S, xyz - base["v"])
        np.testing.assert_allclose(score(params["Q"], xyz), score(base["Q"], transformed), atol=3e-16)

    def test_a_strong_and_light_share_direction(self):
        base = example_base()
        light, _ = make_parameters(base, "fixed_v_metric_light", obj_index=1)
        strong, _ = make_parameters(base, "fixed_v_metric_strong", obj_index=1)
        np.testing.assert_allclose((light["spatial_linear_transform"] - np.eye(3)) / .02,
                                   (strong["spatial_linear_transform"] - np.eye(3)) / .1,
                                   rtol=0, atol=6e-15)

    def test_position_matches_saved_sites_and_retains_A_and_normals(self):
        base = example_base()
        delta = np.array([[.013, -.002, .007], [-.008, .004, .001]])
        source = dict(v=base["v"] + delta, Q=np.full((2, 4, 4), 99.))
        params, audit = make_parameters(base, "position_matched_q_light", source_params=source)
        np.testing.assert_array_equal(params["v"], source["v"])
        np.testing.assert_array_equal(params["Q"][:, :3, :3], base["Q"][:, :3, :3])
        np.testing.assert_array_equal(params["n"], base["n"])
        np.testing.assert_array_equal(params["eigs"], base["eigs"])
        xyz = np.array([[.2, .1, -.3], [-.2, .4, .7]])
        measured_delta = source["v"] - base["v"]
        np.testing.assert_allclose(score(params["Q"], xyz), score(base["Q"], xyz - measured_delta), atol=2e-16)
        self.assertTrue(audit["matched_source_sites_bitwise_equal"])
        self.assertLess(audit["prescribed_centered_congruence_max_abs_error"], 2e-16)

    def test_inputs_are_not_mutated(self):
        base = example_base()
        originals = {key: value.copy() for key, value in base.items()}
        params, _ = make_parameters(base, "fixed_v_metric_light")
        params["v"][:] = 0.
        for key, value in base.items():
            np.testing.assert_array_equal(value, originals[key])

    def test_invalid_state_source_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            make_parameters(example_base(), "undeclared")
        with self.assertRaisesRegex(ValueError, "source_params"):
            make_parameters(example_base(), "position_matched_q_strong")
        indefinite = example_base(); indefinite["Q"][0, 3, 3] = -1.
        with self.assertRaisesRegex(ValueError, "Full Q is materially indefinite"):
            make_parameters(indefinite, "roundtrip")
        bad = example_base(); bad["Q"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            make_parameters(bad, "roundtrip")


if __name__ == "__main__":
    unittest.main()
