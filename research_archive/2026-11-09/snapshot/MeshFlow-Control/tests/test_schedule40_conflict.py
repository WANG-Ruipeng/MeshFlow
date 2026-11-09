"""CPU tests of original I reuse and source-free frozen query contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "schedule40_conflict", Path(__file__).parents[1]/"tools/score_schedule40_conflict.py")
supplement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supplement)


def request_fixture(tmp_path, condition="c", K=2):
    triangle = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32)
    C = np.stack([triangle + np.array([i%2,i//2,0.],np.float32) for i in range(K)])
    path=tmp_path/(condition+".npz")
    bank=dict(C=C,N=np.array(128,np.int64),K=np.array(K,np.int64),
              z=np.zeros((1,128,9),np.float32))
    supplement.save_npz(path,bank)
    return dict(request_id=condition,parent_id=condition,condition_id=condition,
                N=128,K=K,input_path=str(path),input_sha256=supplement.digest(path),
                C_sha256=supplement.array_sha(C),condition_construction="TEST_ONLY"),C


def test_query_bank_matches_original_and_is_reused_exactly(tmp_path,monkeypatch):
    request,C=request_fixture(tmp_path)
    row=supplement.freeze_query_bank(request,tmp_path/"query")
    original=supplement.open_reference(dict(C=C,L0=supplement.L_CANON),request["condition_id"])
    loaded=supplement.load_query_bank(row)
    for key,array in original["arrays"].items():
        np.testing.assert_array_equal(loaded["arrays"][key],array)
    # Scoring-side reuse must load saved values, never resample.
    monkeypatch.setattr(supplement.control,"sample_C",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("resampled")))
    reused=supplement.load_query_bank(row)
    np.testing.assert_array_equal(reused["arrays"]["C_points"],original["arrays"]["C_points"])
    assert reused["arrays"]["C_points"].shape==(2048,3)
    assert row["weight_denominator"]==1.


def test_query_tampering_is_rejected(tmp_path):
    request,_=request_fixture(tmp_path)
    row=supplement.freeze_query_bank(request,tmp_path/"query")
    values=supplement.arrays(row["query_path"])
    values["C_points"][0,0]+=1.
    np.savez_compressed(row["query_path"],**values)
    with pytest.raises(ValueError,match="query file changed"):
        supplement.load_query_bank(row)


def test_no_gt_or_gamma_allowed_in_input_bank(tmp_path):
    request,C=request_fixture(tmp_path)
    values=supplement.arrays(request["input_path"])
    values["GT_free"]=C
    np.savez_compressed(request["input_path"],**values)
    request["input_sha256"]=supplement.digest(request["input_path"])
    with pytest.raises(ValueError,match="no GT or Gamma"):
        supplement.read_input(request)


def test_original_duplicate_I_operates_without_GT(tmp_path):
    request,C=request_fixture(tmp_path,K=1)
    row=supplement.freeze_query_bank(request,tmp_path/"query")
    ref=supplement.load_query_bank(row)
    F=C.copy();F[:,:,2]+=.001
    cf=supplement.control.trusted_cf(C,F,supplement.L_CANON)
    metrics,vectors=supplement.control.conflict_metrics(F,dict(C=C,L0=supplement.L_CANON),ref,cf)
    assert metrics["ConflictCov_C_cross"]==0.
    assert metrics["ConflictCov_C_dup"]>.25
    assert metrics["ConflictCov_C"]==metrics["ConflictCov_C_dup"]
    assert metrics["ConflictCov_C"]==pytest.approx(
        metrics["ConflictCov_C_cross"]+metrics["ConflictCov_C_dup"]-
        metrics["ConflictCov_C_cross_and_dup"])
    assert vectors["conflict_union"].shape==(2048,)
    assert "GT_free" not in ref and "gamma_edges" not in ref["arrays"]


def test_missing_pair_members_remain_in_denominators():
    rows=[]
    for parent in ("a","b"):
        for label in supplement.LABELS:
            scored=not(parent=="a" and label=="EARLY")
            rows.append(dict(parent_id=parent,label=label,status="SCORED" if scored else "BLOCKED",
                metric_status="PASS" if scored else "NA_BLOCKED",
                metrics={k:.1 for k in supplement.PRIMARY} if scored else {}))
    summaries,pairs=supplement.aggregate(rows)
    early=next(r for r in summaries if r["label"]=="EARLY")
    assert early["registered"]==8
    assert early["scored"]==1
    assert early["ConflictCov_C_all8_mean"] is None
    assert len(pairs)==30
    assert sum(r["status"]=="BLOCKED_PAIR" for r in pairs)==5


def test_registration_refuses_existing_rollout_artifacts(tmp_path):
    out=tmp_path/"experiments"/supplement.ROOT_NAME
    requests=[]
    for i in range(8):
        row,_=request_fixture(tmp_path,condition="c"+str(i),K=51)
        requests.append(row)
    supplement.write_json(out/"evaluation/requests.json",dict(rows=requests,labels=supplement.LABELS))
    marker=out/"outputs/EARLY/c0/started.json"
    marker.parent.mkdir(parents=True);marker.write_text("{}")
    with pytest.raises(RuntimeError,match="after rollout artifacts"):
        supplement.prepare(out)

def test_failed_revalidation_never_overwrites_completed_result(tmp_path):
    path=tmp_path/"score.json"
    original=dict(status="SCORED",metrics={"ConflictCov_C":.25})
    supplement.write_json(path,original)
    supplement.persist_row(path,dict(status="BLOCKED",reason="asset changed"))
    assert supplement.read_json(path)==original
    errors=list(tmp_path.glob("score.validation_error.*.json"))
    assert len(errors)==1
    assert supplement.read_json(errors[0])["previous_completed_score_preserved"]==str(path)

def test_trusted_cached_CF_does_not_rescan_pairs(tmp_path,monkeypatch):
    request,C=request_fixture(tmp_path,K=1)
    F=C.copy();F[:,:,2]+=.1
    original=supplement.control.trusted_cf(C,F,supplement.L_CANON)
    monkeypatch.setattr(supplement.control,"pair_scan",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("full CF rescan")))
    reused=supplement.control.trusted_cf(C,F,supplement.L_CANON,cf_scan=original["scan"])
    np.testing.assert_array_equal(reused["codes"],original["codes"])
    assert reused["scan"] is original["scan"]
