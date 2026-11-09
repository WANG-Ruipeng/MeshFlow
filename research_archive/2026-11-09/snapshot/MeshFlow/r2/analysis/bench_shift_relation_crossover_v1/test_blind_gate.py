"""Synthetic gate checks; never reads the actual ongoing anonymous review."""
import tempfile
import json
import unittest
from pathlib import Path
from . import blind_gate as G


def rows():
    return [dict(request_id=f"r{i}",parent_id=f"p{i//4}",condition_id=f"c{i}",part_type="SEAT_SURFACE",
                 **{G.REF+"__C_use":"UNCERTAIN",G.SHIFT+"__C_use":"UNCERTAIN"}) for i in range(32)]


class GateTests(unittest.TestCase):
    def test_two_parents_and_strict_total(self):
        r=rows()
        for i in (0,4):r[i].update({G.REF+"__C_use":"REPLACED_OR_IGNORED",G.SHIFT+"__C_use":"PLAUSIBLE"})
        d=G.gate_from_pairs(r)
        self.assertEqual(d["status"],"PASS");self.assertEqual(len(d["anonymous_strict_gains"]),2)

    def test_uncertain_gains_do_not_meet_strict_parent_gate(self):
        r=rows()
        for i in (0,4):r[i][G.SHIFT+"__C_use"]="PLAUSIBLE"
        d=G.gate_from_pairs(r)
        self.assertEqual(d["status"],"FAIL");self.assertEqual(len(d["anonymous_shift_unique_positives"]),2)
        self.assertEqual(d["anonymous_strict_gains"],[])

    def test_two_requests_one_parent_fail(self):
        r=rows()
        for i in (0,1):r[i].update({G.REF+"__C_use":"REPLACED_OR_IGNORED",G.SHIFT+"__C_use":"PLAUSIBLE"})
        self.assertEqual(G.gate_from_pairs(r)["status"],"FAIL")

    def test_equal_adoption_totals_fail_despite_two_parent_gains(self):
        r=rows()
        for i in (0,4):r[i].update({G.REF+"__C_use":"REPLACED_OR_IGNORED",G.SHIFT+"__C_use":"PLAUSIBLE"})
        for i in (8,12):r[i][G.REF+"__C_use"]="PLAUSIBLE"
        self.assertEqual(G.gate_from_pairs(r)["status"],"FAIL")

    def test_original_bothpositive_membership(self):
        r=rows();r[0].update({G.REF+"__C_use":"PLAUSIBLE",G.SHIFT+"__C_use":"PLAUSIBLE"})
        d=G.gate_from_pairs(r)
        self.assertEqual(len(d["anonymous_both_positive"]),1)
        self.assertEqual(d["anonymous_shift_unique_positives"],[])

    def test_default_does_not_read_partial_or_write_outputs(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);q=root/"blind_review/review_records/review.json"
            q.parent.mkdir(parents=True);q.write_text("intentionally invalid partial",encoding="utf-8")
            before=q.read_bytes();d=G.run(root)
            self.assertFalse(d["review_read"]);self.assertEqual(q.read_bytes(),before)
            self.assertFalse((root/"blind_review/gate.json").exists())

    def test_gate_stop_refuses_dispatched_workflow(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);(root/"workflow").mkdir();(root/"workflow/state.json").write_text("{}")
            with self.assertRaises(ValueError):G.close_gate_failure(root,dict(status="FAIL"))
            self.assertFalse((root/"COMPLETE.json").exists())

    def test_gate_stop_allows_prepared_cpu_layout_receipts(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p)
            layout=root/"evaluation"/G.A.CANDIDATE/"layouts"/"condition"/"receipt.json"
            layout.parent.mkdir(parents=True);layout.write_text('{"CPU_only":true}')
            gate=dict(status="FAIL",counts={G.REF:2,G.SHIFT:2},strict_gain_parents=[])
            gp=root/"blind_review/gate.json";gp.parent.mkdir(parents=True)
            gp.write_text(json.dumps(gate))
            result=G.close_gate_failure(root,gate)
            self.assertEqual(result["status"],"COMPLETE_GATE_STOP_BEFORE_GPU")
            self.assertEqual(result["new_generator_forwards"],0)
            self.assertEqual(layout.read_text(),'{"CPU_only":true}')

    def test_gate_stop_refuses_real_raw_receipt_next_to_layout(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);candidate=root/"evaluation"/G.A.CANDIDATE
            layout=candidate/"layouts"/"condition"/"receipt.json"
            layout.parent.mkdir(parents=True);layout.write_text('{}')
            raw=candidate/"request"/"raw"/"receipt.json"
            raw.parent.mkdir(parents=True);raw.write_text('{}')
            with self.assertRaises(ValueError):G.close_gate_failure(root,dict(status="FAIL"))
            self.assertFalse((root/"COMPLETE.json").exists())

    def test_gate_stop_refuses_started_sampling_shard(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p)
            (root/"evaluation"/G.A.CANDIDATE/"sampling_shard_0").mkdir(parents=True)
            with self.assertRaises(ValueError):G.close_gate_failure(root,dict(status="FAIL"))
            self.assertFalse((root/"COMPLETE.json").exists())


if __name__=="__main__":unittest.main()
