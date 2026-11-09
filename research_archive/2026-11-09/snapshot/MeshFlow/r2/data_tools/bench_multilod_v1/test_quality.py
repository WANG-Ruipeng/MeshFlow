"""Small CPU fixtures for independent lower-LOD acceptance gates."""
import json
import sys
import unittest
from unittest import mock
import numpy as np

from . import quality as Q


def grid(n=8):
    v = np.asarray([(x / n, y / n, 0.) for y in range(n + 1) for x in range(n + 1)], np.float32)
    f = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            f.extend(((a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)))
    return v, np.asarray(f, np.int64)


class QualityTests(unittest.TestCase):
    def evaluate(self, v, f, nv=None, nf=None, lineage=None, known=None, L=2.):
        return Q.evaluate(v, f, v.copy() if nv is None else nv,
                          f.copy() if nf is None else nf,
                          np.arange(len(f)) if lineage is None else lineage,
                          np.asarray([0, 1], np.int64) if known is None else known, L)

    def test_identity_128_is_pass_and_cpu_only(self):
        v, f = grid()
        before = v.tobytes(), f.tobytes()
        r = self.evaluate(v, f)
        self.assertEqual(r["status"], "PASS_PENDING_VISUAL", r)
        self.assertEqual((v.tobytes(), f.tobytes()), before)
        self.assertEqual(r["finite_pairs"]["by_partition"]["CF"]["new_count"], 0)
        self.assertEqual(r["gpu_calls"], 0)
        self.assertNotIn("torch", sys.modules)
        json.dumps(r, allow_nan=False)

    def test_C_and_interface_face_indices_protected(self):
        v, f = grid(2)
        nv = v.copy(); nv[0, 2] = .001
        r = self.evaluate(v, f, nv=nv)
        self.assertIn("C_AND_SOURCE_CF_INTERFACE_UNCHANGED", r["rejection_reasons"])
        nf = f.copy(); nf[3] = nf[3, ::-1]
        r = self.evaluate(v, f, nf=nf)
        self.assertIn("C_AND_SOURCE_CF_INTERFACE_UNCHANGED", r["rejection_reasons"])
        self.assertIn(3, r["checks"]["C_AND_SOURCE_CF_INTERFACE_UNCHANGED"]["changed_face_ids"])

    def test_duplicate_and_collapse_are_not_hidden_by_counts(self):
        v, f = grid(2)
        nf = f.copy(); nf[-1] = nf[-2]
        r = self.evaluate(v, f, nf=nf)
        self.assertIn("NO_NEW_EXACT_DUPLICATE", r["rejection_reasons"])
        nf[-1] = [0, 0, 1]
        r = self.evaluate(v, f, nf=nf)
        self.assertIn("NO_NEW_DEGENERATE", r["rejection_reasons"])

    def test_deleting_tiny_component_fails_even_if_global_area_passes(self):
        v, f = grid(2)
        nv = np.vstack((v, [[2., 0., 0.], [2.01, 0., 0.], [2., .01, 0.]])).astype(np.float32)
        sf = np.vstack((f, [len(v), len(v) + 1, len(v) + 2]))
        r = self.evaluate(nv, sf, nf=f, lineage=np.arange(len(f)), L=3.)
        self.assertTrue(r["checks"]["TOTAL_AREA_RATIO"]["pass_"])
        self.assertFalse(r["checks"]["COMPONENT_PRESERVATION"]["pass_"])
        self.assertEqual(r["checks"]["COMPONENT_PRESERVATION"]["per_component"][-1]["area_ratio"], 0)

    def test_lineage_and_vertex_domain_are_required(self):
        v, f = grid(2)
        r = self.evaluate(v, f, lineage=np.zeros(len(f), np.int64))
        self.assertEqual(r["status"], "BLOCKED")
        r = self.evaluate(v, f, nv=np.vstack((v, [0., 0., 0.])).astype(np.float32))
        self.assertEqual(r["status"], "BLOCKED")

    def test_detects_new_illegal_FF_and_allows_identical_old_pair(self):
        v = np.asarray([[0,0,0],[1,0,0],[0,1,0], [3,0,0],[4,0,0],[3,1,0],
                        [0,0,1],[1,0,1],[0,1,1], [.25,-.2,2],[.25,.8,2],[.25,.3,3]], np.float64)
        f = np.arange(12).reshape(-1,3)
        nv = v.copy(); nv[-3:,2] -= 1.5
        r = self.evaluate(v, f, nv=nv, L=5.)
        self.assertGreater(r["finite_pairs"]["by_partition"]["FF"]["new_count"], 0)
        self.assertIn("NO_NEW_ILLEGAL_FF", r["rejection_reasons"])
        r = self.evaluate(nv, f, L=5.)
        self.assertGreater(r["finite_pairs"]["by_partition"]["FF"]["original_count"], 0)
        self.assertEqual(r["finite_pairs"]["by_partition"]["FF"]["new_count"], 0)

    def test_detector_failure_is_BLOCKED_not_zero(self):
        v, f = grid(2)
        with mock.patch.object(Q, "_scan_cached", side_effect=RuntimeError("fixture missing detector")):
            r = self.evaluate(v, f)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIsNone(r["finite_pairs"]["new_FF_pairs"])

    def test_sampled_finite_distance_hits_surface_interior(self):
        tri = np.asarray([[[0,0,0],[1,0,0],[0,1,0]]], float)
        self.assertEqual(len(Q.probes(tri)), 13)
        d = Q.distances(np.asarray([[.25,.25,.2],[.8,.8,0.]]), tri)
        self.assertAlmostEqual(d[0], .2)
        self.assertAlmostEqual(d[1], np.sqrt(.18))

    def test_global_surface_and_bbox_failure(self):
        v, f = grid(2)
        nv = v.copy(); nv[-1,2] = .1
        r = self.evaluate(v, f, nv=nv)
        self.assertIn("SAMPLED_SURFACE_AND_BBOX", r["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
