"""Small tests for scientific accounting, support and immutable output behavior."""
import tempfile
import unittest
from pathlib import Path
from . import run as A
from . import closeout as Z


def row(label,rid,parent,value,status="PASS"):
    return dict(label=label,stage="CLEAN_FINAL_POST",request_id=rid,parent_id=parent,
                condition_id=parent+"_C",noise_index=int(rid[-1]),N=128,K=2,
                part_type="SEAT_SURFACE",split="DEV",I_status=status,
                seam_to_bulk_status="PASS",seam_applicability="APPLICABLE",ConflictCov_C=value)


class AnalysisTests(unittest.TestCase):
    def test_parent_macro_not_request_mean(self):
        rows=[row(A.CANDIDATE,"r0","p1",0),row(A.CANDIDATE,"r1","p1",0),row(A.CANDIDATE,"r2","p2",1)]
        stat=A.core.statistics_for(rows,"ConflictCov_C")
        self.assertEqual(stat["parent_macro"],.5)
        self.assertAlmostEqual(stat["request_mean"],1/3)

    def test_missing_never_becomes_zero(self):
        rows=[row(A.CANDIDATE,"r0","p1",None)]
        stat=A.core.statistics_for(rows,"ConflictCov_C")
        self.assertIsNone(stat["parent_macro"])
        self.assertEqual(stat["missing_requests"],1)

    def test_correct_candidate_and_both_pass_domain(self):
        rows=[]
        for label in A.LABELS:
            rows.append(row(label,"r0","p1",.2 if label==A.CANDIDATE else .1))
            rows.append(row(label,"r1","p2",.9 if label==A.CANDIDATE else .8,
                            "LOW_SUPPORT" if label==A.CANDIDATE else "PASS"))
        paired=A.core.pair_rows(rows,["ConflictCov_C"],A.CANDIDATE,A.REFERENCES)
        summaries=A.pair_summaries(paired,["ConflictCov_C"])
        self.assertTrue(all(r["candidate"]==A.CANDIDATE for r in summaries))
        valid=[r for r in summaries if r["panel"]=="ALL" and r["part_type"]=="ALL_PARTS"
               and r["pair_support"]=="BOTH_PASS_ONLY"]
        self.assertEqual(len(valid),2)
        self.assertTrue(all(r["valid_requests"]==1 for r in valid))
        self.assertTrue(all(abs(r["parent_macro"]-.1)<1e-12 for r in valid))

    def test_pair_identity_mismatch_rejected(self):
        rows=[row(l,"r0","p1",.1) for l in A.LABELS]
        rows[-1]["N"]=129
        with self.assertRaises(ValueError):
            A.core.pair_rows(rows,["ConflictCov_C"],A.CANDIDATE,A.REFERENCES)

    def test_waiting_produces_no_results(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.assertEqual(A.run(path)["status"],"WAITING_FOR_COMPLETED_COLLECTOR")
            self.assertEqual(Z.run(path)["status"],"WAITING")
            self.assertEqual(list(path.iterdir()),[])

    def test_fixed_sets_preserve_exact_membership(self):
        _,sets=A.fixed_sets()
        self.assertEqual(len(sets["gain9"]),9)
        self.assertEqual(len(sets["original_positive8"]),8)
        self.assertFalse(sets["gain9"] & sets["original_positive8"])

    def test_output_never_silently_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"x.json"
            A.core.write_json(path,{"x":1})
            A.core.write_json(path,{"x":1})
            with self.assertRaises(ValueError):A.core.write_json(path,{"x":2})

    def test_fixed_visual_set_not_reselected_from_candidate(self):
        records=[]
        enums={k:v[0] for k,v in Z.AXES.items()}
        for label in A.LABELS:
            r=row(label,"r0","p1",.1)
            r.update(enums,note="Synthetic test only")
            if label==A.CANDIDATE:r["C_use"]="REPLACED_OR_IGNORED"
            records.append(r)
        _,_,fixed=Z.review_tables(records,{"gain9":{"r0"},"original_positive8":set()})
        self.assertEqual(len(fixed),2)
        self.assertTrue(all(r["fixed_set"]=="gain9" and r["C_use__candidate"]=="REPLACED_OR_IGNORED" for r in fixed))


if __name__=="__main__":unittest.main()
