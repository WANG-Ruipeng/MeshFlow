"""CPU-only arithmetic fixtures; no real models, CUDA work, OT, or training."""
import json
import math
from pathlib import Path
import sys
import time
import unittest
import torch
from experiments.mf_h1_local0.jedge5_local_error_B.math_ops import (
    RESOLUTION_RULE, bits_equal, vector_stats, euler_free, local_predictors,
    local_differences, free_difference, resolution_floor, decompose_velocity,
    make_direction_probe, direction_gain, propagation_summary)


class LocalArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(1, 37, dtype=torch.float32).reshape(1, 4, 9) / 7
        self.x[0, 0, 0] = -0.0
        self.known = torch.tensor([[True, False, False, False]])

    def test_euler_production_order_and_negative_zero_context(self):
        x = self.x.clone()
        v = torch.sin(x)
        expected = x.clone()
        expected[:, 1:] = x[:, 1:] + v[:, 1:] * 0.01
        actual = euler_free(x, v, .01, self.known)
        self.assertTrue(bits_equal(actual, expected))
        self.assertTrue(bits_equal(x, self.x))
        self.assertTrue(torch.signbit(actual[0, 0, 0]))
        self.assertFalse(bits_equal(torch.tensor([-0.0]), torch.tensor([0.0])))

    def test_coordinate_and_point_rms_distinct(self):
        v = torch.ones_like(self.x)
        result = vector_stats(v, self.known, bbox=2)
        self.assertEqual(result['coordinate_rms'], 1)
        self.assertAlmostEqual(result['point_rms'], math.sqrt(3))
        self.assertEqual(result['coordinate_rms_bbox'], .5)
        self.assertEqual(result['coordinate_count'], 27)
        self.assertEqual(result['point_count'], 9)

    def test_affine_autonomous_field_defect_scales_four(self):
        # x'=2x+1: defect = h^2 * 2 * (2x+1) / 4.
        h, a = .02, 2.0
        f = lambda t, x: a * x + 1
        v0 = f(0, self.x)
        p = local_predictors(self.x, v0, self.known, h)
        result = local_differences(self.x, v0, f(h/2, p['x_half']),
                                   f(h/4, p['x_quarter']), self.known, h, 3)
        expected = h*h*a*v0[~self.known]/4
        torch.testing.assert_close(result['delta_h'][~self.known], expected, rtol=.001, atol=6e-7)
        self.assertAlmostEqual(result['summary']['delta_h_over_delta_half_rms'], 4, delta=.015)
        self.assertTrue(bits_equal(result['x_2E'][self.known], self.x[self.known]))

    def test_pure_time_field_defect_and_zero_space(self):
        h, t = .02, .35
        f = lambda time, x: torch.full_like(x, 3*time+2)
        v0 = f(t, self.x)
        p = local_predictors(self.x, v0, self.known, h)
        vmid = f(t+h/2, p['x_half'])
        result = local_differences(self.x, v0, vmid, f(t+h/4, p['x_quarter']), self.known, h)
        torch.testing.assert_close(result['delta_h'][~self.known],
                                   torch.full_like(v0[~self.known], 3*h*h/4), rtol=.002, atol=6e-7)
        self.assertAlmostEqual(result['summary']['delta_h_over_delta_half_rms'], 4, delta=.035)
        d = decompose_velocity(v0, vmid, f(t, p['x_half']), f(t+h/2, self.x), self.known)
        self.assertEqual(d['summary']['terms']['space']['coordinate_rms'], 0)
        self.assertEqual(d['summary']['terms']['cross']['coordinate_rms'], 0)
        self.assertTrue(d['summary']['identity_within_tolerance'])

    def test_linear_time_dependent_field_cross_and_identity(self):
        t, h, a, b, c = .5, .01, 2., 3., 4.
        f = lambda t, x: (a+b*t)*x+c*t
        v0 = f(t, self.x)
        p = local_predictors(self.x, v0, self.known, h)
        d = decompose_velocity(v0, f(t+h/2, p['x_half']),
                               f(t, p['x_half']), f(t+h/2, self.x), self.known)
        expected = b * (h/2) * (p['x_half']-self.x)
        torch.testing.assert_close(d['terms']['cross'][~self.known], expected[~self.known],
                                   rtol=.02, atol=1e-5)
        self.assertTrue(d['summary']['identity_within_tolerance'])
        self.assertLess(d['summary']['identity_residual']['max_abs'], 1e-5)

    def test_directional_cancellation_not_norm_sum(self):
        z = torch.zeros_like(self.x)
        space = torch.ones_like(self.x)
        time = -torch.ones_like(self.x)
        d = decompose_velocity(z, z, space, time, self.known)
        self.assertEqual(d['summary']['terms']['total']['l2_norm'], 0)
        self.assertEqual(d['summary']['total_norm_over_sum_term_norms'], 0)
        self.assertEqual(d['summary']['cancellation_fraction'], 1)
        self.assertAlmostEqual(d['summary']['pairwise_angles']['space__time']['cosine'], -1)
        self.assertEqual(d['summary']['pairwise_angles']['space__cross']['cosine'], None)

    def test_actual_fp32_perturbation_denominator(self):
        x = torch.full_like(self.x, 123.123)
        x[self.known] = self.x[self.known]
        delta = torch.arange(1, 37, dtype=torch.float32).reshape_as(x) / 17
        p = make_direction_probe(x, delta, self.known, 5e-5, reference_states=(x, x))
        self.assertEqual(p['status'], 'RESOLVED')
        actual_difference = (p['x_plus']-p['x_minus'])[~self.known].double()
        actual_rms = float(actual_difference.square().mean().sqrt())
        self.assertEqual(p['actual_denominator_coordinate_rms'], actual_rms)
        self.assertNotEqual(p['actual_denominator_coordinate_rms'], p['theoretical_denominator_coordinate_rms'])
        self.assertTrue(bits_equal(p['x_plus'][self.known], x[self.known]))
        self.assertTrue(bits_equal(p['x_minus'][self.known], x[self.known]))
        self.assertAlmostEqual(p['direction_actual_coordinate_rms'], 1, places=6)
        gain = direction_gain(p['x_plus'], p['x_minus'], p, self.known)
        self.assertEqual(gain['gain'], 1)

    def test_zero_and_subresolution_direction_not_replaced(self):
        x = torch.ones_like(self.x)
        floor = resolution_floor(x, known=self.known)
        self.assertEqual(floor, 8*torch.finfo(torch.float32).eps)
        for scale in (0., floor/2):
            p = make_direction_probe(x, torch.full_like(x, scale), self.known, 1e-4)
            self.assertEqual(p['status'], 'DIRECTION_UNRESOLVED')
            self.assertIsNone(p['x_plus'])
            self.assertIsNone(p['x_minus'])
            self.assertIsNone(p['direction'])

    def test_ulp_rule_uses_actual_operands_and_is_sign_symmetric(self):
        x = torch.ones_like(self.x)
        y = torch.full_like(x, 1024.)
        self.assertEqual(resolution_floor(x, known=self.known), 8*2**-23)
        self.assertEqual(resolution_floor(x, y, known=self.known), 8*2**-13)
        self.assertEqual(resolution_floor(-y, known=self.known), resolution_floor(y, known=self.known))
        self.assertGreater(resolution_floor(torch.zeros_like(x), known=self.known), 0)

    def test_small_amplification_denominator_is_unresolved(self):
        initial = torch.zeros_like(self.x)
        final = torch.ones_like(self.x)
        p = propagation_summary(initial, final, self.known, 2, (self.x, self.x))
        self.assertIsNone(p['amplification'])
        self.assertEqual(p['status'], 'DENOMINATOR_UNRESOLVED')
        self.assertEqual(p['final_difference']['coordinate_rms_bbox'], .5)

    def test_resolved_amplification_uses_coordinate_rms(self):
        initial = torch.full_like(self.x, 1e-3)
        final = initial*3
        p = propagation_summary(initial, final, self.known, 2, (self.x, self.x))
        self.assertEqual(p['status'], 'RESOLVED')
        self.assertAlmostEqual(p['amplification'], 3, places=6)
        self.assertAlmostEqual(p['initial_difference']['point_rms'], math.sqrt(3)*1e-3, places=9)

    def test_invalid_shapes_dtype_nonfinite_and_empty_free_rejected(self):
        for x, known in ((self.x.double(), self.known), (self.x, self.known[0]),
                         (self.x, torch.ones_like(self.known)), (self.x*math.nan, self.known)):
            with self.assertRaises(ValueError):
                vector_stats(x, known)
        with self.assertRaises(ValueError):
            vector_stats(self.x, self.known, 0)
        with self.assertRaises(ValueError):
            make_direction_probe(self.x, self.x, self.known, 1e-3)


if __name__ == '__main__':
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LocalArithmeticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"status": "PASS" if result.wasSuccessful() else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "elapsed_seconds": time.perf_counter()-start,
              "scope": "CPU arithmetic fixtures with analytic fields; no real model calls",
              "cuda_initialized": torch.cuda.is_initialized(), "neural_forwards": 0,
              "model_backwards": 0, "optimizer_updates": 0, "OT_calls": 0,
              "resolution_rule": RESOLUTION_RULE}
    Path(__file__).with_name('math_cpu_test_results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
    sys.exit(0 if result.wasSuccessful() else 1)
