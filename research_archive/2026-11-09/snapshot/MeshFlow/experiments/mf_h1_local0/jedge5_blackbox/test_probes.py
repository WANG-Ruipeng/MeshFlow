"""CPU NumPy fixtures; no torch, model, OT, scoring, or autograd calls."""
import importlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from . import probes as p


class ProbeFixtures(unittest.TestCase):
    def test_registered_budget(self):
        self.assertEqual(len(p.PATCHES) * len(p.NOISE_SEEDS) * 3 * 2 * (1 + 2 * 2), 120)
        self.assertEqual(2 * p.MODEL_FORWARD_CAP, p.TOTAL_FORWARD_CAP)

    def test_gaussian_offsets_are_not_normalized(self):
        offsets = p.registered_directions()
        for seed, a in offsets.items():
            expected = (np.random.RandomState(seed).randn(100, 9) * 1e-4).astype(np.float32)
            self.assertTrue(p._bits_equal(a, expected))
            self.assertNotEqual(p._rms(a), 1e-4)
        self.assertFalse(p._bits_equal(*offsets.values()))

    def test_direction_rng_is_private_and_repeatable(self):
        before = np.random.get_state()
        a, b = p.registered_directions(), p.registered_directions()
        self.assertTrue(p._numpy_rng_same(before, np.random.get_state()))
        self.assertTrue(all(p._bits_equal(a[k], b[k]) for k in a))

    def test_three_named_selection_criteria(self):
        a = np.zeros((51, 112, 9), np.float32)
        b = np.zeros((101, 112, 9), np.float32)
        b[12::2, 12, 0] = 10  # D increase is between coarse states 5 and 6.
        v = np.zeros((50, 112, 9), np.float32)
        v[20:, 12, 0] = 3
        selected = p.select_probe_steps(a, b, v)
        self.assertEqual([x['k'] for x in selected['steps']], [5, 20, 25])
        self.assertEqual(selected['D_increment'][5], 10)

    def test_duplicate_selection_fills_with_stable_ties(self):
        a = np.zeros((51, 112, 9), np.float32)
        b = np.zeros((101, 112, 9), np.float32)
        b[52::2, 12, 0] = 5
        v = np.zeros((50, 112, 9), np.float32)
        v[25:, 12, 0] = 2
        selected = p.select_probe_steps(a, b, v)
        self.assertEqual([x['k'] for x in selected['steps']], [25, 0, 1])
        self.assertEqual(len(selected['steps'][0]['reasons']), 3)

    def test_selection_ignores_known_velocity_predictions(self):
        a = np.zeros((51, 112, 9), np.float32)
        b = np.zeros((101, 112, 9), np.float32)
        v = np.zeros((50, 112, 9), np.float32)
        v[20:, 12, 0] = 1
        plain = p.select_probe_steps(a, b, v)
        v[30:, :12] = 1e10  # Clamped C slots can have nonzero predicted velocities.
        self.assertEqual(plain, p.select_probe_steps(a, b, v))

    def test_perturbation_preserves_C_bits_and_records_FP32_rounding(self):
        x = np.ones((112, 9), np.float32)
        x[0, 0] = np.float32(-0.0)
        offset = p.registered_directions()[p.DIRECTION_SEEDS[0]]
        plus, minus, audit = p.perturbed_states(x, offset)
        self.assertTrue(p._bits_equal(plus[:12], x[:12]))
        self.assertTrue(p._bits_equal(minus[:12], x[:12]))
        self.assertGreater(audit['plus_actual_delta_RMS'], 0)
        self.assertNotEqual(audit['requested_offset_RMS'], audit['plus_actual_delta_RMS'])

    def test_linear_directional_gain(self):
        x = np.ones((112, 9), np.float32)
        plus, minus, _ = p.perturbed_states(x, p.registered_directions()[61401])
        response = p.directional_statistics(x, plus, minus, 2 * x, 2 * plus, 2 * minus)
        self.assertAlmostEqual(response['free_central_directional_L2_gain'], 2, places=13)
        self.assertAlmostEqual(response['free_plus_directional_L2_gain'], 2, places=13)
        self.assertEqual(response['known_predicted_velocity_plus_response_RMS'], 0)

    def test_constant_response_and_vanished_offset_guard(self):
        x = np.ones((112, 9), np.float32)
        plus, minus, _ = p.perturbed_states(x, p.registered_directions()[61402])
        v = np.full_like(x, 7)
        response = p.directional_statistics(x, plus, minus, v, v, v)
        self.assertEqual(response['free_central_directional_L2_gain'], 0)
        self.assertIsNone(response['free_even_to_half_chord_ratio'])
        with self.assertRaises(RuntimeError):
            p.perturbed_states(x, np.zeros((100, 9), np.float32))

    def test_saved_trajectory_schema_and_C_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.npz'
            states = np.zeros((51, 112, 9), np.float32)
            velocity = np.ones((50, 112, 9), np.float32)  # Includes nonzero known velocity.
            arrays = dict(raw=states[-1].copy(), states=states, velocities=velocity,
                          times=np.asarray([k / 50 for k in range(50)], np.float32),
                          state_indices=np.arange(51), C=states[0, :12].reshape(12, 3, 3))
            np.savez(path, **arrays)
            loaded, info = p.load_trajectory(path, 50)
            self.assertEqual(info['nfe'], 50)
            self.assertTrue(p._bits_equal(loaded['velocities'], velocity))
            arrays['states'][1, 0, 0] = 1
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                p.load_trajectory(path, 50)


if __name__ == '__main__':
    unittest.main()
