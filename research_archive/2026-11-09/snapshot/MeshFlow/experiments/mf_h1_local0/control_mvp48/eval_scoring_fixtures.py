"""CPU fixtures: python -B -m experiments.mf_h1_local0.control_mvp48.eval_scoring_fixtures."""
from __future__ import annotations
import json
import math
import time
import unittest
import numpy as np
from . import eval_scoring as s
from experiments.mf_h1_local0.native_inpainting_metrics import symmetric_surface_rms
from experiments.mf_h1_local0.guided_native_geometry_audit import boundary_metrics


def triangle(offset=(0., 0., 0.)):
    return np.asarray([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=np.float32) + np.asarray(offset, dtype=np.float32)


class ScoringFixtures(unittest.TestCase):
    def test_legacy_rng_and_directional_surface_exact_agreement(self):
        C = triangle((-2., 0., 0.))
        target = np.concatenate([triangle(), triangle((0., 0., .5))])
        reference = s.prepare_reference(C, target, 4.)
        for free in (target + np.float32(.1), np.concatenate([target, triangle((2., 0., .2))])):
            new = s.score_free(free, C, target, 4., reference=reference, include_pair_audit=False)
            old = symmetric_surface_rms(free, target, 4.)
            self.assertEqual(new["detail"]["remainder"]["target_sample_sha256"], old["target_sample_sha256"])
            self.assertEqual(new["detail"]["remainder"]["prediction_sample_sha256"], old["prediction_sample_sha256"])
            self.assertEqual(new["metrics"]["remainder_symmetric_RMS_bbox_pct"], old["symmetric_rms_bbox_pct"])
            self.assertEqual(new["metrics"]["free_to_R12_RMS_bbox_pct"], old["predicted_to_target_rms_bbox_pct"])
            self.assertEqual(new["metrics"]["R12_to_free_RMS_bbox_pct"], old["target_to_predicted_rms_bbox_pct"])

    def test_original_boundary_agreement_and_finite_edge_ordering(self):
        C, free, L = triangle(), triangle((0., 0., .2)), 2.
        reference = s.prepare_reference(C, free, L)
        new = s.score_free(free, C, free, L, reference=reference, include_pair_audit=False)
        old = boundary_metrics(C, free, L)
        for key in ("rms_bbox_pct", "p95_bbox_pct", "coverage_fraction"):
            self.assertEqual(new["detail"]["boundary"][key], old[key])
        self.assertEqual(new["detail"]["edge_surface_ordering"]["status"], "PASS")
        edge = s._edge_module().gamma_to_free_edges(triangle().astype(float), np.array([[2., 0., 0.]]), np.ones(1), 2.)
        self.assertEqual(edge["RMS"], 1.)

    def test_variable_N_nonprefix_mask_and_C_exclusion(self):
        C, target, L = triangle((0., 0., 5.)), np.concatenate([triangle(), triangle((2., 0., 0.)), triangle((4., 0., 0.))]), 8.
        raw = np.concatenate([target[:1], C, target[1:]])
        mask = np.array([False, True, False, False])
        reference = s.prepare_reference(C, target, L)
        result = s.score_output(raw, C, target, L, known_mask=mask, reference=reference)
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["metrics"]["remainder_symmetric_RMS_bbox_pct"], 1e-12)
        self.assertEqual(result["metrics"]["actual_N"], 4)
        self.assertEqual(result["metrics"]["actual_free_faces"], 3)
        self.assertEqual(result["detail"]["CF"]["tested_pairs"], 3)
        self.assertEqual(result["detail"]["FF"]["tested_pairs"], 3)
        self.assertEqual(result["detail"]["free_slots"], [0, 2, 3])
        self.assertEqual(result["detail"]["watertight_certification"], "NOT_RUN")

    def test_reference_rejects_wrong_complement_and_mutation(self):
        C, target = triangle(), triangle((0., 0., 2.))
        reference = s.prepare_reference(C, target, 4.)
        with self.assertRaisesRegex(ValueError, "different C"):
            s.score_free(target, C, target + 1, 4., reference=reference)
        with self.assertRaises(ValueError):
            reference["arrays"]["coverage_points"][0, 0] = 999
        changed = dict(reference, arrays=dict(reference["arrays"]))
        changed["arrays"]["coverage_points"] = reference["arrays"]["coverage_points"] + 1
        with self.assertRaisesRegex(ValueError, "reference array changed"):
            s.score_free(target, C, target, 4., reference=changed)

    def test_N_and_bitwise_C_failure_kept(self):
        C, target = triangle(), triangle((0., 0., 2.))
        reference = s.prepare_reference(C, target, 4.)
        raw = np.concatenate([C, target, triangle((3., 0., 0.))])
        result = s.score_output(raw, C, target, 4., reference=reference)
        self.assertEqual(result["status"], "INVALID_OUTPUT")
        self.assertIn("ACTUAL_N_MISMATCH", result["failure_reason"])
        raw = np.concatenate([C, target])
        raw[0, 0, 0] = np.float32(-0.)
        result = s.score_output(raw, C, target, 4., reference=reference)
        self.assertEqual(result["metrics"]["C_preserved"], 0)
        self.assertIn("C_NOT_PRESERVED", result["failure_reason"])

    def test_nonfinite_failure_without_pair_scan(self):
        C, target = triangle(), triangle((0., 0., 2.))
        reference = s.prepare_reference(C, target, 4.)
        raw = np.concatenate([C, target])
        raw[1, 0, 0], raw[1, 0, 1] = np.nan, np.inf
        result = s.score_output(raw, C, target, 4., reference=reference)
        self.assertEqual(result["status"], "INVALID_OUTPUT")
        self.assertEqual(result["metrics"]["free_nan_coordinate_count"], 1)
        self.assertEqual(result["metrics"]["free_inf_coordinate_count"], 1)
        self.assertIsNone(result["metrics"]["FF_illegal_pair_count"])
        self.assertNotIn("verified_pair_tests", result["ledger"])
        json.dumps(result, allow_nan=False)

    def test_degeneracy_and_UNRESOLVED_retained(self):
        C = triangle()
        target = np.concatenate([triangle((0., 0., 2.)), triangle((2., 0., 2.))])
        free = target.copy()
        free[0] = 0
        reference = s.prepare_reference(C, target, 5.)
        result = s.score_output(np.concatenate([C, free]), C, target, 5., reference=reference)
        self.assertEqual(result["metrics"]["free_exact_zero_area_count"], 1)
        self.assertEqual(result["metrics"]["free_near_degenerate_fraction"], .5)
        self.assertGreater(result["metrics"]["CF_unresolved_or_degenerate_pair_count"], 0)
        self.assertGreater(result["metrics"]["FF_unresolved_or_degenerate_pair_count"], 0)
        self.assertEqual(result["detail"]["geometry_status"], "UNRESOLVED")

    def test_unique_affected_face_fractions(self):
        C = triangle()
        crossing = np.array([[[.2, .2, 1.], [.6, .2, -1.], [.2, .6, 1.]]], dtype=np.float32)
        free = np.concatenate([crossing, triangle((3., 0., 2.)), triangle((5., 0., 2.))])
        reference = s.prepare_reference(C, free, 8.)
        result = s.score_output(np.concatenate([C, free]), C, free, 8., reference=reference)
        self.assertEqual(result["metrics"]["CF_proper_pair_count"], 1)
        self.assertEqual(result["metrics"]["CF_proper_affected_free_fraction"], 1/3)
        free2 = np.concatenate([crossing, crossing, triangle((5., 0., 2.))])
        result2 = s.score_free(free2, C, free, 8., reference=reference)
        self.assertEqual(result2["metrics"]["FF_illegal_pair_count"], 1)
        self.assertEqual(result2["metrics"]["FF_illegal_affected_free_fraction"], 2/3)

    def test_coverage_inequalities(self):
        free = triangle().astype(np.float64)
        points = np.array([[.25, .25, .001]], dtype=np.float64)
        boundary = s._weighted_surface(free, points, np.ones(1), 1., {})
        self.assertEqual(boundary["coverage_fraction"], 0.)
        edge = s._edge_module().gamma_to_free_edges(free, np.array([[.25, 0., .001]]), np.ones(1), 1.)
        self.assertEqual(edge["coverage_by_bbox_fraction"]["0.001"], 0.)
        d2 = s.surface_distance_squared(np.array([[.25, .25, .01]]), free)
        self.assertEqual(float((d2 <= .01**2).mean()), 1.)

    def test_nonfinite_fixed_GT_coverage_is_invalid_output(self):
        from unittest.mock import patch
        C, target = triangle(), triangle((0., 0., 2.))
        reference = s.prepare_reference(C, target, 4.)
        original = s.surface_distance_squared

        def injected_distance(points, triangles):
            if points is reference["arrays"]["coverage_points"]:
                return np.full(len(points), np.inf)
            return original(points, triangles)

        with patch.object(s, "surface_distance_squared", side_effect=injected_distance):
            result = s.score_free(target, C, target, 4., reference=reference, include_pair_audit=False)
        self.assertEqual(result["status"], "INVALID_OUTPUT")
        self.assertEqual(result["failure_reason"], "NONFINITE_FIXED_GT_COVERAGE")
        self.assertIsNone(result["metrics"]["fixed_R12_coverage_RMS_bbox_pct"])
        self.assertIsNone(result["metrics"]["fixed_R12_coverage_within_1pct_fraction"])
        self.assertEqual(result["detail"]["fixed_GT_coverage"]["nonfinite_distance_count"], 512)
        self.assertEqual(result["ledger"]["fixed_GT_coverage_distance_calls"], 1)
        json.dumps(result, allow_nan=False)

    def test_Gamma_is_true_crossing_interface(self):
        full = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                         [[1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]], dtype=np.float32)
        sample = dict(full_target=full, constraints=full[:1], free_target=full[1:], bbox_diagonal=math.sqrt(2),
            source_face_ids=[0], free_source_ids=[1], interface=dict(known_source_face_ids=[0], free_source_face_ids=[1],
                known_corner_ids=[[1, 2]], free_corner_ids=[[0, 2]], edge_source_vertex_ids=[[1, 2]]))
        reference = s.prepare_reference_from_sample(sample)
        self.assertEqual(reference["audit"]["gamma_edge_count"], 1)
        self.assertEqual(len(reference["arrays"]["gamma_points"]), 512)
        self.assertTrue(np.allclose(reference["arrays"]["gamma_points"][:, :2].sum(1), 1))
        self.assertTrue(reference["audit"]["gamma_excludes_source_open_edges"])


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU fixture process already initialized CUDA")
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScoringFixtures))
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU fixtures initialized CUDA")
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), seconds=time.perf_counter()-started,
        model_forwards=0, CUDA_initialized=False, threads=1, source_identity=s.source_identity()), indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
