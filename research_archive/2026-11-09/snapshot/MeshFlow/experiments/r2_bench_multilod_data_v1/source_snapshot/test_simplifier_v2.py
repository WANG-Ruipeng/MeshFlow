"""Engineering-v2 CPU fixtures; real-parent admission gates are independent."""
import unittest
from unittest.mock import patch

import numpy as np

from r2.data_tools.bench_multilod_v1 import test_simplifier as original_tests
from r2.data_tools.bench_multilod_v1.simplifier import simplify as simplify_v1
from r2.data_tools.bench_multilod_v1.simplifier_v2 import simplify


def closed_cylinder(segments=24):
    angles = np.arange(segments) * (2 * np.pi / segments)
    lower = np.column_stack((np.cos(angles), -np.ones(segments), np.sin(angles)))
    upper = lower.copy()
    upper[:, 1] = 1
    v = np.vstack((lower, upper, [[0, -1, 0], [0, 1, 0]])).astype(np.float32)
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        f.extend(((i, i + segments, j + segments), (i, j + segments, j),
                  (2 * segments, i, j), (2 * segments + 1, j + segments, i + segments)))
    return v, np.asarray(f, np.int64)


class V2InheritedContractTests(original_tests.ProtectedSimplifierTests):
    """Run the unchanged three v1 contract fixtures against the v2 engine."""

    def run(self, result=None):
        with patch.object(original_tests, "simplify", simplify):
            return super().run(result)


class V2EngineeringTests(unittest.TestCase):
    def test_curved_sharp_rim_no_longer_freezes_all_vertices(self):
        v, f = closed_cylinder()
        old = simplify_v1(v, f, [], 80, 4.0)
        new = simplify(v, f, [], 80, 4.0)
        self.assertGreater(old[3]["sharp_edge_count"], 0)
        self.assertEqual(old[3]["accepted_collapses"], 0)
        self.assertGreater(new[3]["accepted_collapses"], 0)
        self.assertLess(len(new[1]), len(old[1]))
        self.assertGreaterEqual(len(new[1]), 80)
        self.assertEqual(old[3]["sharp_edge_count"], new[3]["sharp_edge_count"])
        self.assertLess(len(new[3]["fixed_vertex_ids"]), len(old[3]["fixed_vertex_ids"]))
        # This fixture establishes proposal availability, not real-surface admission.

    def test_one_fixed_endpoint_keeps_its_original_position_and_id(self):
        v, f = original_tests.grid(8)
        vv, ff, ids, ledger = simplify(v, f, [], len(f) - 2, 15.0)
        self.assertEqual(ledger["accepted_collapses"], 1)
        collapse = ledger["collapses"][0]
        self.assertTrue(collapse["keep_is_fixed"])
        self.assertFalse(collapse["drop_is_fixed"])
        self.assertEqual(collapse["position_choice"], -1)
        keep = collapse["keep_original_vertex_id"]
        drop = collapse["removed_original_vertex_id"]
        self.assertEqual(vv[keep].tobytes(), v[keep].tobytes())
        self.assertFalse(np.any(ff == drop))
        self.assertTrue(np.any(ff == keep))
        fixed = ledger["fixed_vertex_ids"]
        self.assertEqual(vv[fixed].tobytes(), v[fixed].tobytes())


if __name__ == "__main__":
    unittest.main()
