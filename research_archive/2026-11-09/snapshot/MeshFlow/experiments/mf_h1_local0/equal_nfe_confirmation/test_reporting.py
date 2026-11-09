"""Small CPU tests for scientific aggregation, not implementation-mirroring tests."""
import unittest

from .reporting import make_binding, make_effects, summarize


class ReportingTests(unittest.TestCase):
    def test_median_is_of_paired_differences(self):
        rows = []
        for seed, a, b in zip((1, 2, 3), (0, 100, 101), (0, 1, 101)):
            for solver, value in (("Euler50", a), ("Midpoint25", b)):
                rows.append(dict(model="B", solver=solver, parent=0, patch=0, seed=seed, values={"RMS": value}))
        effects = make_effects(rows, ["RMS"])
        diff = {(r["parent"], r["patch"], r["seed"]): r["differences"]["RMS"] for r in effects if r["effect"] == "Delta_B_M"}
        stats = summarize(diff, "RMS", True)
        self.assertEqual(stats["paired_median_delta"], 0)  # difference of marginal medians would be -99
        self.assertEqual((stats["improved"], stats["equal"], stats["worsened"]), (1, 2, 0))

    def test_coverage_improvement_sign(self):
        values = {(0, 0, 1): .2, (0, 0, 2): -.1, (0, 1, 1): 0}
        result = summarize(values, "boundary_coverage_fraction", True)
        self.assertEqual((result["improved"], result["equal"], result["worsened"]), (1, 1, 1))
        self.assertEqual(result["worst_case"]["seed"], 2)

    def test_missing_does_not_become_success_subset_mean(self):
        values = {(0, 0, 1): 1, (0, 0, 2): None, (1, 0, 1): 10, (1, 0, 2): 20}
        stats = summarize(values, "RMS", True)
        self.assertIsNone(stats["hierarchical_mean"])
        self.assertEqual(stats["available_complete_cell_mean"], 15)
        self.assertEqual((stats["valid"], stats["requested"], stats["cells_complete"]), (3, 4, 1))
        self.assertEqual(stats["paired_median_delta"], 10)

    def test_hierarchy_equal_weights_cells(self):
        values = {(0, 0, 1): 0, (0, 0, 2): 2, (1, 0, 1): 10}
        stats = summarize(values, "RMS")
        self.assertEqual(stats["hierarchical_mean"], 5.5)
        self.assertEqual(stats["available_case_mean"], 4)

    def test_interaction_requires_all_four_operands(self):
        rows = [dict(model=m, solver=s, parent=0, patch=0, seed=1, values={"RMS": v})
                for m, s, v in (("B", "Euler50", 4), ("B", "Midpoint25", 2), ("J", "Euler50", 10), ("J", "Midpoint25", 3))]
        effect = {r["effect"]: r["differences"]["RMS"] for r in make_effects(rows, ["RMS"])}
        self.assertEqual((effect["Delta_B_M"], effect["Delta_J_M"], effect["A_M"], effect["I_M"]), (-2, -7, 1, -5))
        rows.pop()
        effect = {r["effect"]: r["differences"]["RMS"] for r in make_effects(rows, ["RMS"])}
        self.assertIsNone(effect["I_M"])
        self.assertIsNone(effect["A_M"])

    def test_binding_keeps_both_directions(self):
        rows = []
        # P0 correct 2 / P1 foreign 3 => +1; P1 correct 10 / P0 foreign 9 => -1.
        for parent, vals in ((0, (2, 9)), (1, (3, 10))):
            rows.append(dict(model="B", solver="Euler50", parent=parent, patch=0, seed=1, task_id=f"source{parent}",
                             cross_binding={str(p): {"metrics": {"RMS": v}} for p, v in enumerate(vals)}))
        row = make_binding(rows, [1])[0]
        self.assertEqual(row["metrics"]["RMS"]["mean"], 0)
        self.assertEqual(row["metrics"]["RMS"]["delta_a"], 1)
        self.assertEqual(row["metrics"]["RMS"]["delta_b"], -1)
        self.assertFalse(row["metrics"]["RMS"]["both_directions_positive"])
        self.assertEqual(row["source_P1"], "source1")


if __name__ == "__main__":
    unittest.main()
