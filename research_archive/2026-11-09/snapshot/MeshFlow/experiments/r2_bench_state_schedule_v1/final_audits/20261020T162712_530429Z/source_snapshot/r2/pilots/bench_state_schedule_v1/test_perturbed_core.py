"""CPU tests for the isolated corrective-state contract; no model/torch imports."""
from __future__ import annotations

import copy
import random
import unittest

import numpy as np

from r2.pilots.bench_state_schedule_v1.perturbed_core import (
    apply_perturbation, derive_row_randomness,
)


def fixture(time=0.5):
    # Original C is deliberately NOT the leading face.  Seam anchors and padding
    # have large values so accidentally including them in mean(Y_free) is visible.
    target = np.asarray([
        [[10., 0., 0.], [12., 0., 0.], [2., -1., 0.]],
        [[10., 0., 0.], [12., 0., 0.], [10., 2., 0.]],
        [[-1., -3., 2.], [1., -3., 2.], [0., -1., 2.]],
        [[99., 99., 99.], [99., 99., 99.], [99., 99., 99.]],
    ], np.float32)
    anchor = np.asarray([[0, 1, -1], [0, 1, 2], [-1, -1, -1], [-1, -1, -1]])
    valid = np.asarray([True, True, True, False])
    known = np.asarray([False, True, False, False])
    free = valid[:, None] & (anchor < 0)
    x0 = np.zeros_like(target)
    x0[anchor >= 0] = target[anchor >= 0]
    u = target - x0
    u[~valid] = 0
    t = np.asarray(time, np.float32)
    xt = (np.float32(1) - t) * x0 + t * target
    xt[anchor >= 0] = target[anchor >= 0]
    return dict(xt=xt, x1=target, x0=x0, u=u, t=t, anchor_ids=anchor,
                valid_mask=valid, known_mask=known, free_corner_mask=free,
                free_coordinate_count=np.asarray(int(free.sum()) * 3),
                source_face_ids=np.asarray([1, 0, 2, -1]),
                source_corner_permutations=np.tile(np.arange(3), (4, 1)))


def row(selected=True):
    for number in range(1000):
        candidate = dict(sample_index=number, noise_seed=1101 + number,
                         epsilon_seed=2201 + number, time_seed=3301 + number,
                         permutation_seed=4401 + number,
                         scale_xyz=[0.9, 1.0, 1.1], parent_id="synthetic_cpu_only")
        if derive_row_randomness(candidate)["selected_by_probability"] == selected:
            return candidate
    raise AssertionError("Could not find deterministic test row")


def assert_arrays_equal(test, first, second):
    for name, value in first.items():
        if isinstance(value, np.ndarray):
            test.assertEqual(value.dtype, second[name].dtype, name)
            test.assertEqual(value.shape, second[name].shape, name)
            test.assertEqual(value.tobytes(), second[name].tobytes(), name)


class PerturbedCoreTests(unittest.TestCase):
    def test_endpoint_identity_and_only_free_state_target_change(self):
        sample = fixture()
        before = copy.deepcopy(sample)
        changed, audit = apply_perturbation(sample, row())
        self.assertTrue(audit["applied"])
        self.assertFalse(np.array_equal(changed["xt"], sample["xt"]))
        free = sample["free_corner_mask"]
        np.testing.assert_allclose(
            changed["xt"][free] + (np.float32(1) - sample["t"]) * changed["u"][free],
            sample["x1"][free], rtol=0, atol=2e-6,
        )
        self.assertLess(audit["endpoint_identity_max_abs"], 2e-6)
        self.assertEqual(changed["xt"][~free].tobytes(), sample["xt"][~free].tobytes())
        self.assertEqual(changed["u"][~free].tobytes(), sample["u"][~free].tobytes())
        for name in sample.keys() - {"xt", "u"}:
            self.assertIs(changed[name], sample[name], name)
        self.assertEqual(int(changed["free_coordinate_count"]), 12)
        assert_arrays_equal(self, sample, before)

    def test_center_masks_include_tip_exclude_anchors_and_padding(self):
        sample = fixture()
        _, audit = apply_perturbation(sample, row())
        np.testing.assert_allclose(audit["mu_C"], [32 / 3, 2 / 3, 0])
        np.testing.assert_allclose(audit["mu_Y_free"], [.5, -2, 1.5])
        expected = np.asarray(
            audit["alpha"] * (np.asarray(audit["mu_C"]) - np.asarray(audit["mu_Y_free"])),
            np.float32,
        )
        np.testing.assert_array_equal(audit["displacement_model_xyz"], expected)

    def test_disabled_gate_rejected_and_ineligible_are_byte_identity(self):
        for t, enabled, selected, reason in (
            (.5, False, True, "DISABLED"),
            (.5, True, False, "PROBABILITY_GATE_NOT_SELECTED"),
            (.29, True, True, "OUTSIDE_TIME_WINDOW"),
            (.91, True, True, "OUTSIDE_TIME_WINDOW"),
            (0., True, True, "OUTSIDE_TIME_WINDOW"),
            (1., True, True, "OUTSIDE_TIME_WINDOW"),
        ):
            with self.subTest(t=t, enabled=enabled, selected=selected):
                sample = fixture(t)
                changed, audit = apply_perturbation(sample, row(selected), enabled=enabled)
                self.assertFalse(audit["applied"])
                self.assertEqual(audit["reason"], reason)
                assert_arrays_equal(self, sample, changed)

    def test_time_boundaries_are_the_recorded_fp32_values(self):
        for t in (.3, .9):
            _, audit = apply_perturbation(fixture(t), row())
            self.assertTrue(audit["eligible"])
            self.assertTrue(audit["applied"])

    def test_determinism_key_order_and_global_rng_untouched(self):
        original_row = row()
        swapped = dict(reversed(list(original_row.items())))
        self.assertEqual(derive_row_randomness(original_row), derive_row_randomness(swapped))
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        first, first_audit = apply_perturbation(fixture(), original_row)
        # Deliberately call another row in between: no order-based state.
        apply_perturbation(fixture(), row(False))
        second, second_audit = apply_perturbation(fixture(), original_row)
        assert_arrays_equal(self, first, second)
        self.assertEqual(first_audit, second_audit)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_closed_C_without_any_seam_is_supported(self):
        sample = fixture()
        sample["anchor_ids"][0] = -1
        sample["free_corner_mask"] = sample["valid_mask"][:, None] & (sample["anchor_ids"] < 0)
        sample["free_coordinate_count"] = np.asarray(int(sample["free_corner_mask"].sum()) * 3)
        changed, audit = apply_perturbation(sample, row())
        self.assertTrue(audit["applied"])
        self.assertEqual(audit["free_corner_count"], 6)
        self.assertEqual(int(changed["free_coordinate_count"]), 18)

    def test_norm_gain_matches_direct_computation(self):
        sample = fixture(.8)
        changed, audit = apply_perturbation(sample, row())
        free = sample["free_corner_mask"]
        ratio = (np.linalg.norm(changed["u"][free].astype(np.float64))
                 / np.linalg.norm(sample["u"][free].astype(np.float64)))
        self.assertAlmostEqual(audit["target_velocity_l2_gain"], ratio, places=12)
        self.assertAlmostEqual(audit["target_velocity_mse_gain"], ratio * ratio, places=12)

    def test_wrong_mask_denominator_or_nonzero_anchor_velocity_rejected(self):
        for field in ("free_corner_mask", "free_coordinate_count", "u"):
            sample = fixture()
            if field == "free_corner_mask":
                sample[field][0, 0] = True
            elif field == "free_coordinate_count":
                sample[field] = np.asarray(99)
            else:
                sample[field][1, 0, 0] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_perturbation(sample, row())

    def test_explicit_off_path_label_and_nonfinite_row_rejected(self):
        _, audit = apply_perturbation(fixture(), row())
        self.assertEqual(audit["objective"], "OFF_PATH_ENDPOINT_CORRECTIVE_REGRESSION_NOT_PATH_CFM")
        with self.assertRaises(TypeError):
            derive_row_randomness(dict(noise_seed=1, invalid=float("nan")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
