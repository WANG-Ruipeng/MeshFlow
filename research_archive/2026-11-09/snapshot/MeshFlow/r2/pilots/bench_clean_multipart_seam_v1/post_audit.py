"""Independent read-only original-124 metric and frozen geometry audit."""
from __future__ import annotations
import argparse
import numpy as np
from . import post_cpu as P
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
C=P.C;ROOT=P.ROOT


def run(partial=False):
    dest=ROOT/("historical_metric_audit_partial.json" if partial else "historical_metric_audit.json")
    if dest.exists():
        if partial:raise RuntimeError("Partial snapshot already exists")
        result=C.read(dest);P.E.verify(result["files"]);return result
    fields=C.read(C.ROOT/"experiments/r2_bench_control_transfer_v1/evaluation/summary_completion.json")["numeric_fields"]
    assert len(fields)==124
    mismatch=[];compared=0;cells=[];checks=[]
    for job in P.jobs():
        q,label=job["q"],job["label"];cell=C.resolve(job["dest"])
        if not (cell/"complete.json").exists():
            if partial:continue
            raise RuntimeError("Missing registered cell "+str(cell))
        base=C.arrays(job["base_path"]);pairs=C.read(job["pairs_path"])
        score=C.read(C.resolve(job["pairs_path"]).parent/"score.json")
        row=metric_row(score,base,pairs)[0]
        new=C.read(cell/"BASE_POST/scores/row.json")
        for key in fields:
            a,b=row.get(key),new.get(key);compared+=1
            valid=(a is None and b is None) or (a is not None and b is not None and np.isclose(float(a),float(b),atol=1e-10,rtol=1e-10))
            if not valid:mismatch.append(dict(label=label,request_id=q["request_id"],metric=key,old=a,new=b))
        final=C.arrays(cell/"CLEAN_FINAL_POST/post.npz")
        K=q["K"]
        assert base["C"].tobytes()==final["C"].tobytes()
        assert base["faces"][:K].tobytes()==final["faces"][:K].tobytes()
        assert len(final["faces"])>=len(base["faces"])
        safety=C.read(cell/"base_final_paired_ledger.json")
        for key in ("CF_lost_legal_edges","FF_lost_legal_edges","CF_new_illegal_pairs","FF_new_illegal_pairs","deleted_source_faces"):
            assert safety["summary"][key]==0
        cells.append(str(cell/"complete.json"))
    result=dict(status="PASS" if not mismatch else "FAIL",scope="completed subset snapshot" if partial else "full registered144",
        cells=len(cells),metric_fields=124,scalar_comparisons=compared,mismatches=mismatch,
        original_C_coordinates_winding_and_indices_exact=True,no_added_bad_pairs_or_lost_edges=True,
        model_calls=0,new_scores=0,new_geometry=0,
        files={p:C.sha(p) for p in cells},code_sha256=C.sha(__file__))
    C.write(dest,result)
    if mismatch:raise AssertionError("Original124 metric mismatch "+repr(mismatch[:5]))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--partial",action="store_true");a=p.parse_args()
    r=run(a.partial);print({k:v for k,v in r.items() if k!="files"})
