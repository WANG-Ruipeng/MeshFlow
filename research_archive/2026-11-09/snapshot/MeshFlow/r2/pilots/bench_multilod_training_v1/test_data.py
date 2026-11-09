"""CPU schedule and algebra gates; no model, OT, training or CUDA work."""
import copy
import math
import unittest
import numpy as np

from . import common as C
from . import data as D


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.groups = ["OLD"]*6+["HIGH"]*2
        self.actual = [100]*6+[400]*2
        ref = [100]*6+[200]*2
        dw, db = sum(self.actual), sum(ref)
        self.original = dict(micro_groups=self.groups, micro_free_scalar_counts=self.actual,
            micro_gains=[dw/db]*6+[dw/db*.5]*2, reference_micro_free_scalar_counts=ref,
            D_W=dw, D_B=db, branch="HIGH")

    def test_group_weights_survive_multiple_variable_groups(self):
        # Use valid scalar counts divisible by nine; the original weighted masses
        # are arbitrary algebra fixtures, independent of array/OT construction.
        counts = [90]*6+[180]*2
        entry = dict(stage_step=1, samples=[dict(replay_group=g, N=c//9+2, K=2)
                    for c,g in zip(counts, self.groups)])
        a = D.allocation(entry, self.original)
        self.assertAlmostEqual(a["reference_group_mse_weights"]["OLD"], .6)
        self.assertAlmostEqual(a["reference_group_mse_weights"]["HIGH"], .4)
        losses = np.arange(1., 9.)
        sse = losses*np.asarray(counts)
        actual = sum(q*s for q,s in zip(a["micro_gains"], sse))/sum(counts)
        expected = .6*sum(sse[:6])/sum(counts[:6]) + .4*sum(sse[6:])/sum(counts[6:])
        self.assertAlmostEqual(actual, expected, places=14)
        self.assertAlmostEqual(a["group_mse_weights_sum"], 1.)

    def test_wrong_group_order_fails(self):
        entry = dict(stage_step=1, samples=[dict(replay_group="OLD", N=128, K=2)]*8)
        with self.assertRaises(ValueError): D.allocation(entry, self.original)


class RealScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.schedules, cls.coefficients = D.make()

    def test_full_pair_and_case_cache_contract(self):
        self.assertEqual(len(self.manifest["train"]), 39)
        self.assertEqual(len({r["case_id"] for r in self.manifest["train"]}), 39)
        self.assertEqual(len({r["condition_id"] for r in self.manifest["train"]}), 30)
        self.assertEqual(D.validate_pair(self.schedules,self.manifest)["status"], "PASS")
        for arm in C.ARMS:
            self.assertEqual(len(self.schedules[arm]["steps"]), 1500)
            for e in self.schedules[arm]["steps"]:
                for j in range(0,8,2):
                    a,b=e["samples"][j:j+2]
                    self.assertEqual(a["case_id"], b["case_id"])
                    self.assertEqual(a["variant"],a["variant_kind"])

    def test_all_reference_group_weights_and_plain_match(self):
        p, m = (self.coefficients[a]["steps"] for a in C.ARMS)
        for ap,am in zip(p,m):
            self.assertEqual(ap["reference_group_mse_weights"], am["reference_group_mse_weights"])
            self.assertLess(max(abs(a-b) for a,b in zip(ap["micro_gains"], ap["original_micro_gains"])),1e-12)
            for a in (ap,am):
                self.assertEqual(sum(a["micro_free_scalar_counts"]), a["D_W"])
                self.assertLess(abs(math.fsum(a["effective_group_mse_weights"].values())-1.),1e-12)

    def test_scientific_seed_fields_are_shared_and_new(self):
        old=D.C.read(D.SOURCE_SCHEDULE)
        oldsets={f:{r[f] for e in old["steps"] for r in e["samples"]} for f in D.RANDOM_FIELDS}
        for ep,em in zip(*(self.schedules[a]["steps"] for a in C.ARMS)):
            for rp,rm in zip(ep["samples"],em["samples"]):
                for f in D.RANDOM_FIELDS:
                    self.assertEqual(rp[f],rm[f]);self.assertNotIn(rp[f],oldsets[f])
        mutated=copy.deepcopy(self.schedules)
        mutated[C.ARMS[1]]["steps"][0]["samples"][0]["noise_seed"] += 1
        with self.assertRaises(ValueError): D.validate_pair(mutated,self.manifest)


if __name__ == "__main__": unittest.main()
