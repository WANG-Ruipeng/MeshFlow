"""CPU accounting tests; no Torch, model construction, OT, or CUDA calls."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from . import training as T


def allocation_fixture(original_counts, actual_counts, groups, original_gains=None):
    original_gains = original_gains or [1.] * 8
    keys = ("OLD", "FOCUS", "HIGH")
    ref_d = sum(original_counts)
    ref_weights = {g: sum(n * w for n, w, group in zip(original_counts, original_gains, groups)
                          if group == g) / ref_d for g in keys}
    actual_group_counts = {g: sum(n for n, group in zip(actual_counts, groups) if group == g) for g in keys}
    total = sum(actual_counts)
    coefficient = dict(stage_step=1, D_W=total, D_B=ref_d,
                       micro_groups=groups[:], micro_free_scalar_counts=actual_counts[:],
                       actual_group_scalar_counts=actual_group_counts,
                       reference_group_mse_weights=ref_weights,
                       micro_gains=[ref_weights[g] * total / actual_group_counts[g] for g in groups])
    samples = [dict(N=n//9+4, K=4, free_coordinate_count=n) for n in actual_counts]
    entry = dict(stage_step=1, samples=[dict(replay_group=g) for g in groups])
    return coefficient, samples, entry


def ledger_fixture(step):
    counts = dict(T.limits(), train_forward=step*8, train_backward=step*8, optimizer_update=step)
    rows = []
    for s in range(1, step+1):
        for micro in range(8):
            for kind in ("train_forward", "train_backward"):
                rows.append(dict(event="CALL_BEGIN", kind=kind, context=dict(T.cursor(s), microbatch=micro)))
        rows.append(dict(event="CALL_BEGIN", kind="optimizer_update", context=T.cursor(s)))
    return SimpleNamespace(pending={}, failed={}, limits=T.limits(), attempted=counts.copy(),
                           successful=counts.copy(), rows=rows)


class GroupAccounting(unittest.TestCase):
    def test_plain_reproduces_existing_group_objective(self):
        groups = ["OLD"]*4 + ["HIGH"]*4
        counts = [900, 900, 1800, 1800, 3600, 3600, 4500, 4500]
        # Existing exposure law: high group has half its coordinate mass weight.
        d_actual = sum(counts)
        d_reference = sum(counts[:4]) + sum(counts[4:])*.5
        gains = [d_actual/d_reference] * 4 + [.5*d_actual/d_reference] * 4
        coefficient, samples, entry = allocation_fixture(counts, counts, groups, gains)
        T.allocation_for_update(coefficient, samples, entry)
        for old, new in zip(gains, coefficient["micro_gains"]):
            self.assertAlmostEqual(old, new, places=13)

    def test_old_low_n_preserves_focus_weight_without_gain_one_shortcut(self):
        groups = ["OLD"]*4 + ["FOCUS"]*4
        original = [1800]*4 + [3600]*4
        actual = [1440]*4 + [3600]*4
        coefficient, samples, entry = allocation_fixture(original, actual, groups)
        T.allocation_for_update(coefficient, samples, entry)
        self.assertTrue(all(g != 1. for g in coefficient["micro_gains"]))
        mse = {"OLD": 2., "FOCUS": 7.}
        objective = sum(n * mse[group] * gain / coefficient["D_W"]
                        for n, group, gain in zip(actual, groups, coefficient["micro_gains"]))
        self.assertAlmostEqual(objective, 2./3. + 14./3., places=13)

    def test_low_n_in_both_groups_preserves_reference_group_mse(self):
        groups = ["OLD"]*4 + ["HIGH"]*4
        original = [1800]*4 + [5400]*4
        actual = [1440]*4 + [4320]*4
        coefficient, samples, entry = allocation_fixture(original, actual, groups)
        T.allocation_for_update(coefficient, samples, entry)
        sse = [n * (i+1.) for i, n in enumerate(actual)]
        objective = sum(e*g/coefficient["D_W"] for e, g in zip(sse, coefficient["micro_gains"]))
        expected = sum(coefficient["reference_group_mse_weights"][g] *
                       sum(e for e, group in zip(sse, groups) if group == g) /
                       coefficient["actual_group_scalar_counts"][g] for g in ("OLD", "HIGH"))
        self.assertAlmostEqual(objective, expected, places=13)

    def test_stale_original_denominator_and_mask_count_rejected(self):
        coefficient, samples, entry = allocation_fixture([1800]*8, [1440]*8, ["OLD"]*4+["HIGH"]*4)
        bad = copy.deepcopy(coefficient)
        bad["D_W"] = 1800*8
        with self.assertRaises(ValueError):
            T.allocation_for_update(bad, samples, entry)
        samples[0]["K"] += 1
        with self.assertRaises(ValueError):
            T.allocation_for_update(coefficient, samples, entry)

    def test_unregistered_gain_or_group_rejected(self):
        coefficient, samples, entry = allocation_fixture([1800]*8, [1440]*8, ["OLD"]*4+["HIGH"]*4)
        bad = copy.deepcopy(coefficient)
        bad["micro_gains"][0] *= 2.
        with self.assertRaises(ValueError):
            T.allocation_for_update(bad, samples, entry)
        entry["samples"][0]["replay_group"] = "FOCUS"
        with self.assertRaises(ValueError):
            T.allocation_for_update(coefficient, samples, entry)


class BudgetAndRecovery(unittest.TestCase):
    def test_cursor_limits(self):
        self.assertEqual(T.cursor(1500)["global_step"], 27000)
        self.assertEqual(T.cursor(1500)["native_total_step"], 15000)
        self.assertEqual(T.limits()["train_forward"], 12000)
        for value in (-1, 1501, 1.0, True):
            with self.assertRaises(ValueError):
                T.cursor(value)

    def test_exact_completed_update_resume_and_pending_refusal(self):
        ledger = ledger_fixture(2)
        T.require_cursor(ledger, 2)
        with self.assertRaises(RuntimeError):
            T.require_cursor(ledger, 1)
        ledger.pending = {"unfinished_call": {}}
        with self.assertRaises(RuntimeError):
            T.require_cursor(ledger, 2)

    def test_same_count_reordered_microbatch_refused(self):
        ledger = ledger_fixture(2)
        ledger.rows[0]["context"]["microbatch"] = 7
        with self.assertRaises(RuntimeError):
            T.require_cursor(ledger, 2)


if __name__ == "__main__":
    unittest.main()
