"""CPU-only scientific-contract checks; no real model or historical scoring."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
ev = importlib.import_module("evaluate_recipe_factorial")
sc = importlib.import_module("score_recipe_factorial_conflict")
from meshflow_control.mesh_io import write_raw


def request(tmp_path, *, candidate=0, ratio=.2, parent="parent", cohort="A20"):
    n = 128; k = round(n*ratio)
    C = np.tile(np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]], np.float32), (k,1,1))
    C[:, :, 2] = np.arange(k, dtype=np.float32)[:,None]*.1
    z = np.arange(n*9, dtype=np.float32).reshape(1,n,9)/100 + candidate
    z[:, :k] = 0
    rid = parent + "__C" + str(int(ratio*100)) + "__n" + str(candidate)
    path = tmp_path / (rid + ".npz")
    ev.save_npz(path,dict(C=C,z=z,N=np.array(n),K=np.array(k)))
    return dict(request_id=rid,parent_id=parent,condition_id=parent+"_C"+str(int(ratio*100)),candidate=candidate,
                ratio=ratio,condition_cohort=cohort,N=n,K=k,C_sha256=ev.array_sha(C),Gaussian_sha256=ev.array_sha(z),
                input_path=str(path),input_sha256=ev.digest(path),panel="DEV",cohort="SEEN_DEV")


def synthetic_output(tmp_path, q):
    bank = ev.read_input(q); n,k = q["N"],q["K"]
    raw = bank["z"][0].copy(); raw[:k] = bank["C"].reshape(k,9)
    trajectory = np.repeat(raw[None],51,axis=0)
    dst = tmp_path / "output"
    write_raw(dst,raw,bank["C"],bank["z"],trajectory,
              dict(nfe=50,dtype="BF16 backbone/FP32 parameters+integration",readout_mode="none"))
    r = dict(q,label="MODEL",status="COMPLETE",checkpoint_sha256="weight",
             **{key:value for name,file in (("raw","raw.npz"),("ply","raw.ply"),("result","run.json"))
                for key,value in ((name+"_path",str(dst/file)),(name+"_sha256",ev.digest(dst/file)))})
    return r


def test_a40_noise_uses_absolute_slots_without_cast_or_rng():
    z = np.arange(1*128*9,dtype=np.float32).reshape(1,128,9)
    z[:,90,1] = -0.
    x = ev.derive_a40_noise(z,51)
    assert x[:,51:].tobytes() == z[:,51:].tobytes()
    assert not np.any(x[:,:51]) and np.signbit(x[:,90,1]).all()
    assert z[0,0,1] == 1
    with pytest.raises(ValueError):
        ev.derive_a40_noise(z.astype(np.float64),51)


def test_actual_all51_and_initial_noise_are_checked(tmp_path):
    q = request(tmp_path); r = synthetic_output(tmp_path,q)
    _,facts = ev.verify_output(q,r,"weight")
    assert all(facts.values())
    raw = ev.arrays(r["raw_path"]); raw["trajectory"][17,0,0] += 1
    np.savez_compressed(r["raw_path"],**raw); r["raw_sha256"] = ev.digest(r["raw_path"])
    with pytest.raises(ValueError,match="C_all51_bitwise"):
        ev.verify_output(q,r,"weight")


def test_cache_rejects_missing_runtime_even_with_valid_arrays(tmp_path):
    q = request(tmp_path); r = synthetic_output(tmp_path,q)
    ev.write_json(r["result_path"],dict(cache_reused=True,known_all51_bitwise=True))
    r["result_sha256"] = ev.digest(r["result_path"])
    with pytest.raises(ValueError,match="Runtime proof"):
        ev.verify_output(q,r,"weight")


def test_checkpoint_mismatch_prevents_reuse(tmp_path):
    q = request(tmp_path); r = synthetic_output(tmp_path,q)
    with pytest.raises(ValueError,match="Checkpoint"):
        ev.verify_output(q,r,"another_weight")


def test_cell_attempt_is_durable_and_never_replayed(tmp_path):
    path = tmp_path / "calls.jsonl"; budget = ev.CellBudget(path,dict(request_id="cell"))
    budget.reserve("rollout_attempt")
    with pytest.raises(RuntimeError): budget.reserve("rollout_attempt")
    for _ in range(50): budget.call("sampling_forward",lambda:None)
    with pytest.raises(RuntimeError): budget.call("sampling_forward",lambda:None)
    with pytest.raises(FileExistsError): ev.CellBudget(path,{})
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert sum(e.get("kind")=="sampling_forward" and e["event"]=="RESERVE" for e in events)==50
    assert sum(e.get("kind")=="sampling_forward" and e["event"]=="RETURN" for e in events)==50


def test_aggregate_noise_first_parent_equal_and_no_shrinking(tmp_path):
    qs=[]; rows=[]
    for parent,cohort,base in (("p0","ORIGINAL6",1.),("p1","EXTENDED2",9.)):
        for n in ev.NOISES:
            q=request(tmp_path,candidate=n,parent=parent,ratio=.4,cohort=cohort); qs.append(q)
            rows.append(dict(q,label="M",status="PASS",metrics={"ConflictCov_C":base+n}))
    pp,summary,_,_=sc.aggregate(rows,["M"],qs)
    allrow=next(r for r in summary if r["condition_cohort"]=="ALL")
    assert allrow["ConflictCov_C_equal_parent_mean"]==8.5
    rows[-1]["status"]="BLOCKED"
    pp,summary,_,_=sc.aggregate(rows,["M"],qs)
    allrow=next(r for r in summary if r["condition_cohort"]=="ALL")
    assert allrow["registered_outputs"]==8 and allrow["scored_outputs"]==7
    assert allrow["ConflictCov_C_equal_parent_mean"] is None
    assert allrow["ConflictCov_C_complete_parents"]==1
    assert next(r for r in pp if r["parent_id"]=="p1")["ConflictCov_C"] is None


def test_all_phase_barrier_precedes_any_score(monkeypatch,tmp_path):
    monkeypatch.setattr(sc,"phase_registration",lambda *a:(tmp_path,{"labels":["M"]},[]))
    def missing(*args): raise RuntimeError("ALL_PHASE_RAW_REQUIRED_BEFORE_SCORING")
    monkeypatch.setattr(sc,"completed_rows",missing)
    monkeypatch.setattr(sc,"score_one",lambda *a:pytest.fail("Scoring crossed RAW barrier"))
    with pytest.raises(RuntimeError,match="ALL_PHASE_RAW_REQUIRED"):
        sc.score(tmp_path,"main",1)


def test_A40_reference_metrics_are_not_invented(tmp_path):
    q=request(tmp_path,ratio=.4,cohort="ORIGINAL6")
    aux=sc.auxiliary_reference_score(q,{},tmp_path)
    assert aux["status"]=="NA_NOT_APPLICABLE" and not aux["promotion_gate"] and aux["metrics"]=={}


def test_confirm_plan_is_only_registration_and_model_independent(tmp_path):
    out=tmp_path/"experiments"/ev.ROOT_NAME
    rows=[request(tmp_path,ratio=r,parent=p,cohort="A20" if r==.2 else "ORIGINAL6") for p in ("p0","p1") for r in (.2,.4)]
    plan=ev.register_confirmation_plan(out,rows)
    assert plan["Gaussian_arrays_created"]==0 and plan["model_pair"]=="NOT_SELECTED"
    assert not list(out.rglob("*.npz"))
    pair=[r for r in plan["rows"] if r["parent_id"]=="p0" and r["candidate"]==8]
    assert len(pair)==2 and pair[0]["seed"]==pair[1]["seed"]
    assert not plan["candidate_model_identity_in_seed"]


def test_diversity_keeps_all_six_pairs_when_one_noise_is_blocked(tmp_path):
    qs=[request(tmp_path,candidate=n) for n in ev.NOISES]
    pairs,summary=sc.diversity([], ["M"],qs)
    assert len(pairs)==6 and all(r["status"]=="BLOCKED_MISSING_MEMBER" for r in pairs)
    assert summary[0]["D_free_mean"] is None and summary[0]["expected_pairs"]==6


def test_prepare_freezes64_inputs_and16_shared_query_banks_without_confirmation_arrays(tmp_path):
    art=tmp_path/"artifacts"; source=tmp_path/"source"
    old20=[]; old40=[]
    for i in range(8):
        pid="parent"+str(i)
        for n in ev.NOISES:
            q=request(source,candidate=n,parent=pid)
            q.update(panel="P1",noise_seed=100+n)
            old20.append(q)
        q=request(source,candidate=0,parent=pid,ratio=.4,cohort="ORIGINAL6" if i<6 else "EXTENDED2")
        q["condition_construction"]="EXISTING_A40_CONNECTED_CONDITION" if i<6 else "OLD_C_PLUS_NEAREST_FREE_CENTROIDS_EXACT_COORDINATES"
        old40.append(q)
    ev.write_json(art/"experiments/chair_contextual_alignment_sie_v1/data/evaluation/requests.json",dict(rows=old20))
    ev.write_json(art/"experiments/chair_early_alignment_schedule40_v1/evaluation/requests.json",dict(rows=old40))
    out=art/"experiments"/ev.ROOT_NAME
    result=ev.prepare_inputs(out,art)
    assert result["requests"]==64 and result["model_calls"]==0
    rows=ev.read_json(out/"evaluation/requests.json")["rows"]
    qr=ev.read_json(result["query_registration"])
    assert len(qr["rows"])==16
    assert len(list((out/"evaluation/inputs").glob("*.npz")))==24
    assert all(r["all_four_noises_share_same_bank"] for r in qr["rows"])
    assert ev.read_json(out/"evaluation/confirmation_plan.json")["Gaussian_arrays_created"]==0
    q=next(r for r in rows if r["parent_id"]=="parent0" and r["candidate"]==6 and r["ratio"]==.4)
    src=next(r for r in rows if r["parent_id"]=="parent0" and r["candidate"]==6 and r["ratio"]==.2)
    assert ev.read_input(q)["z"][:,q["K"]:].tobytes()==ev.read_input(src)["z"][:,q["K"]:].tobytes()
    assert ev.prepare_inputs(out,art)["requests"]==64
    models=[dict(label=label,loader="start" if label=="START5000" else "recipe_factorial",checkpoint_path=str(out/(label+".pt"))) for label in ev.LABELS]
    phase=ev.register_phase(out,"main",models)
    assert phase["expected_outputs"]==384
    assert not any(Path(m["checkpoint_path"]).exists() for m in models)
    with pytest.raises(FileNotFoundError):
        ev.bind_checkpoint(out,"main","START5000")
