"""Small CPU accounting checks; no experiment/model/scorer calls."""
import unittest

from .run import statistics_for, pair_rows, pair_summaries, training_windows, panel_rows


class AggregationContract(unittest.TestCase):
    def test_parent_condition_weighting_not_request_weighting(self):
        rows = [dict(parent_id="a", condition_id="a1", x=0),
                dict(parent_id="a", condition_id="a1", x=0),
                dict(parent_id="a", condition_id="a2", x=6),
                dict(parent_id="b", condition_id="b1", x=9)]
        got = statistics_for(rows, "x")
        self.assertEqual(got["parent_macro"], 6)
        self.assertEqual(got["request_mean"], 3.75)
        self.assertEqual(got["valid_conditions"], 3)

    def test_missing_does_not_become_zero(self):
        rows = [dict(parent_id="a", condition_id="c", x=None),
                dict(parent_id="a", condition_id="c", x=4)]
        got = statistics_for(rows, "x")
        self.assertEqual(got["parent_macro"], 4)
        self.assertEqual(got["missing_requests"], 1)
        self.assertIsNone(statistics_for(rows[:1], "x")["parent_macro"])

    def test_pair_I_PASS_intersection_preserves_low_support(self):
        rows = []
        for label, values, statuses in (("REF", [0.2,0.3], ["PASS","LOW_SUPPORT"]),
                                       ("RELATION_V1_S1500", [0.4,0.1], ["PASS","PASS"])):
            for n in range(2):
                rows.append(dict(label=label, stage="RAW", request_id=str(n), condition_id="c",
                                 parent_id="p", noise_index=n, N=150,K=2,part_type="SEAT",
                                 split="DEV", I_status=statuses[n],ConflictCov_C=values[n]))
        pairs = pair_rows(rows,["ConflictCov_C"],references=("REF",))
        self.assertEqual(len(pairs),2)
        out = pair_summaries(pairs,["ConflictCov_C"])
        selected = [r for r in out if r["panel"] == "DEV" and r["part_type"] == "ALL_PARTS"]
        both = next(r for r in selected if r["pair_support"] == "BOTH_PASS_ONLY")
        all_rows = next(r for r in selected if r["pair_support"] == "ALL_PAIRED_SUPPORT")
        self.assertEqual(both["valid_requests"],1)
        self.assertAlmostEqual(both["parent_macro"],0.2)
        self.assertEqual(all_rows["valid_requests"],2)
        self.assertAlmostEqual(all_rows["parent_macro"],0)

    def test_all_is_all_parents_not_two_split_means(self):
        rows = [dict(parent_id=str(n),condition_id=str(n),split="TRAIN" if n == 0 else "DEV",part_type="SEAT",x=n*2) for n in range(3)]
        identity,subset=next(panel_rows(rows))
        self.assertEqual(identity["panel"],"ALL")
        self.assertEqual(statistics_for(subset,"x")["parent_macro"],2)

    def test_learning_curve_sse_denominator(self):
        rows=[]
        for step,denom,sse in ((1,1,1),(2,9,18)):
            r=dict(local_step=step,objective=step,update_seconds=1,generator_gradient_norm=1)
            for prefix in ("all_free","bulk","seam_tip"):
                r[prefix+"_scalar_count"]=denom
                r[prefix+"_velocity_sse"]=sse
            r["endpoint_sse"] = sse / step**2
            r["bulk_endpoint_sse"] = sse / step**2
            r["seam_tip_endpoint_sse"] = sse / step**2
            rows.append(r)
        got=training_windows(rows)[0]
        self.assertEqual(got["all_free_velocity_MSE"],1.9)
        self.assertEqual(got["objective_mean"],1.5)
        self.assertEqual(got["one_step_endpoint_sse"],5.5)
        self.assertEqual(got["one_step_endpoint_scalar_count"],10)
        self.assertEqual(got["one_step_endpoint_MSE"],0.55)
        self.assertEqual(got["bulk_endpoint_MSE"],0.55)
        self.assertEqual(got["seam_tip_endpoint_MSE"],0.55)


if __name__ == "__main__":
    unittest.main()
