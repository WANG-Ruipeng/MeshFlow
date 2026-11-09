"""Small CPU fixtures for free-only binding and endpoint eligibility."""
import tempfile
import unittest
from pathlib import Path

from . import evaluation as e
import numpy as np


class EvaluationFixtures(unittest.TestCase):
    def test_frozen_null_schema(self):
        values = e.scalar_schema()
        self.assertEqual(len(values), 54)
        self.assertTrue(all(value is None for value in values.values()))
        self.assertIn("exact_C_free_shared_edge_count", values)
        self.assertIn("Gamma_to_free_edges_coverage_0.001_fraction", values)

    def test_foreign_gamma_uses_free_and_target_scale(self):
        triangle = np.array([[[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]])
        points = np.array([[.2, .2, 0.], [.3, .3, 0.]])
        weights = np.array([.5, .5])
        bank = dict(arrays={}, cases_by_key={})
        for p, length in ((0, 1.), (1, 2.)):
            pre = f"parent{p}_patch0_"
            bank["arrays"].update({pre + "boundary_points": points, pre + "boundary_weights": weights})
            bank["cases_by_key"][p, 12] = dict(bbox_diagonal=length)
        ledger = {}
        a, ea = e._foreign_boundary(triangle, bank, 0, 0, ledger)
        b, eb = e._foreign_boundary(triangle, bank, 1, 0, ledger)
        self.assertAlmostEqual(a["rms_bbox_pct"], 200.)
        self.assertAlmostEqual(b["rms_bbox_pct"], 100.)
        self.assertEqual(a["coverage_fraction"], 0.)
        self.assertEqual(ledger["binding_foreign_Gamma_surface_calls_returned"], 2)
        self.assertGreater(ea["edges"]["RMS_bbox_pct"], a["rms_bbox_pct"])
        self.assertEqual(eb["fallback"]["fallback_triangle_count"], 0)
        degenerate = np.zeros((1, 3, 3))
        _, fallback = e._foreign_boundary(degenerate, bank, 1, 0, ledger)
        self.assertEqual(fallback["fallback"]["fallback_triangle_count"], 1)

    def test_incomplete_or_changed_trajectory_rejected(self):
        task = dict(solver="Midpoint25", parent=0)
        C = np.zeros((12, 3, 3), dtype=np.float32)
        raw = np.zeros((112, 3, 3), dtype=np.float32)
        states = np.repeat(raw[None], 26, axis=0)
        stages = np.repeat(raw[None], 50, axis=0)
        times = np.linspace(0, 1, 26)
        bank = dict(cases_by_key={(0, 12): dict(constraints=C)})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.npz"
            def save():
                np.savez(path, raw=raw, states=states, stage_states=stages, C=C,
                         accepted_times_FP32=times.astype(np.float32),
                         accepted_times_control_FP64=np.linspace(0, 1, 26))
                return e.file_sha256(path)
            endpoint, identity = e.validate_endpoint(path, save(), task, bank)
            self.assertEqual(identity["accepted_count"], 26)
            self.assertEqual(endpoint.shape, (112, 3, 3))
            times[-1] = .96
            with self.assertRaisesRegex(ValueError, "t=1"):
                e.validate_endpoint(path, save(), task, bank)
            times[-1] = 1.
            stages[4, 0, 0, 0] = 1.
            with self.assertRaisesRegex(ValueError, "C changed"):
                e.validate_endpoint(path, save(), task, bank)

    def test_regressions_excluded_from_formal_denominator(self):
        formal = dict(task_id="formal", kind="formal", model="B", solver="Euler50")
        extra = dict(task_id="repeat", kind="repeat", model="B", solver="Euler50")
        protocol = dict(groups=[dict(group_id="block00", tasks=[formal, extra])])
        self.assertEqual(e._registered_group(protocol, "block00"), [formal])


if __name__ == "__main__":
    unittest.main()
