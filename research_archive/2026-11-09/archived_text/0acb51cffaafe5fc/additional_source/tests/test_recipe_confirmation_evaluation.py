"""Synthetic confirmation contracts only; no model or real experiment writes."""
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
a=importlib.import_module("evaluate_recipe_confirmation")
base=a.base
sc=importlib.import_module("score_recipe_factorial_conflict")


def synthetic_fixture(tmp_path):
    out=tmp_path/"experiments"/base.ROOT_NAME
    rows=[]
    for p in range(8):
        for ratio in (.2,.4):
            n=128;k=round(ratio*n);pid="p"+str(p);cid=pid+"_C"+str(int(ratio*100))
            C=np.arange(k*9,dtype=np.float32).reshape(k,3,3)/100+p
            for candidate in base.NOISES:
                z=np.arange(n*9,dtype=np.float32).reshape(1,n,9)/100+p+candidate
                z[:,:k]=0;rid=cid+"__n"+str(candidate);path=out/"synthetic_main_inputs"/(rid+".npz")
                base.save_npz(path,dict(C=C,z=z,N=np.array(n),K=np.array(k)))
                rows.append(dict(request_id=rid,parent_id=pid,condition_id=cid,ratio=ratio,candidate=candidate,
                    N=n,K=k,condition_cohort="A20" if ratio==.2 else "ORIGINAL6" if p<6 else "EXTENDED2",
                    C_sha256=base.array_sha(C),Gaussian_sha256=base.array_sha(z),input_path=str(path),
                    input_sha256=base.digest(path),panel="P1"))
    requests=out/"evaluation/requests.json";base.write_json(requests,dict(rows=rows))
    plan=base.register_confirmation_plan(out,rows)
    query_rows=[]
    for cid,q in {q["condition_id"]:q for q in rows}.items():
        qp=out/"synthetic_query_banks"/(cid+".npz")
        values=dict(C_points=np.zeros((2048,3),np.float32),C_weights=np.ones(2048,np.float64))
        base.save_npz(qp,values)
        query_rows.append(dict(condition_id=cid,parent_id=q["parent_id"],ratio=q["ratio"],
            C_sha256=q["C_sha256"],query_path=str(qp),query_sha256=base.digest(qp),
            query_array_sha256={k:base.array_sha(v) for k,v in values.items()},query_count=2048,
            identity=dict(synthetic_fixture=True),all_four_noises_share_same_bank=True))
    base.write_json(sc.query_registration_path(out,requests),dict(status="FROZEN",rows=query_rows,
        identity=dict(requests_sha256=base.digest(requests),definition=sc.definition(),original_sources=sc.sources())))
    main=base.phase_root(out,"main")
    base.write_json(main/"registration.json",dict(kind="main",expected_outputs=384))
    base.write_json(main/"evaluation/scoring_receipt.json",dict(status="COMPLETE",scored=384,registered=384,synthetic_fixture=True))
    base.write_json(main/"gallery/gallery.json",dict(status="COMPLETE",page_count=128,all_registered_outputs_shown=True,synthetic_fixture=True))
    selected=["R4_MIX_H_LATE","R3_40_H_LATE"]
    source_paths=("src/meshflow_control/training/recipe_confirmation.py",
                  "src/meshflow_control/data/recipe_confirmation.py","tools/train_recipe_confirmation.py")
    auth=dict(schema=a.AUTH_SCHEMA,experiment=a.EXPERIMENT,stage="independent_confirmation",
        selected_pair=selected,authorized_recipes=selected,hybrid_schedules={r:"late" for r in selected},
        namespace=a.EXPERIMENT+"/confirmation/synthetic_only",out=str(out/"confirmation"),
        experiment_root=str(out),updates_per_recipe=6000,effective_batch=8,microbatch=1,max_training_recipes=2,
        source_hashes={r:base.digest(base.REPO/r) for r in source_paths},synthetic_fixture=True)
    ap=out/"synthetic_authorization.json";base.write_json(ap,auth)
    return out,ap,auth,rows,plan


def test_prepare_requires_explicit_selected_pair_before_any_noise(tmp_path):
    out,ap,auth,_,_=synthetic_fixture(tmp_path)
    auth["selected_pair"]=[];base.write_json(ap,auth)
    with pytest.raises(ValueError,match="Exactly two"):
        a.prepare(out,ap)
    assert not (out/"phases/confirm").exists()


def test_prepare_waits_for_complete_main_and_gallery(tmp_path):
    out,ap,_,_,_=synthetic_fixture(tmp_path)
    path=out/"phases/main/gallery/gallery.json"
    gallery=base.read_json(path);gallery["page_count"]=127;base.write_json(path,gallery)
    with pytest.raises(ValueError,match="Complete main gallery"):
        a.prepare(out,ap)
    assert not (out/"phases/confirm").exists()


def test_materialize_registered_noise_preserves_absolute_slots_and_no_collisions(tmp_path):
    out,ap,auth,main,plan=synthetic_fixture(tmp_path)
    original={str(p):base.digest(p) for p in out.rglob("*") if p.is_file()}
    reg=a.prepare(out,ap)
    assert reg["labels"]==auth["selected_pair"] and reg["expected_outputs"]==128
    assert all(m["loader"]=="recipe_confirmation" and m["checkpoint_path"].startswith(auth["out"]) for m in reg["models"])
    rows=base.read_json(reg["requests_path"])["rows"];assert len(rows)==64
    assert {q["candidate"] for q in rows}==set(a.NOISES)
    for parent in {q["parent_id"] for q in rows}:
        for candidate in a.NOISES:
            q20=next(q for q in rows if (q["parent_id"],q["ratio"],q["candidate"])==(parent,.2,candidate))
            q40=next(q for q in rows if (q["parent_id"],q["ratio"],q["candidate"])==(parent,.4,candidate))
            b20,b40=base.read_input(q20),base.read_input(q40)
            assert base.byte_equal(b20["z"][:,q40["K"]:],b40["z"][:,q40["K"]:])
            full=np.random.RandomState(q20["noise_seed"]).standard_normal((1,q20["N"],9)).astype(np.float32)
            assert base.byte_equal(b20["z"][:,q20["K"]:],full[:,q20["K"]:])
    main_q=sc.load_registration(out,out/"evaluation/requests.json")
    confirm_q=sc.load_registration(out,reg["requests_path"])
    assert main_q["rows"]==confirm_q["rows"]
    audit=base.read_json(out/"phases/confirm/evaluation/holdout_noise_collision_audit.json")
    assert audit["status"]=="PASS" and len(audit["rows"])==64
    assert all(r["main_same_N_comparisons"]==64 for r in audit["rows"])
    assert len({r["full_N_Gaussian_sha256"] for r in audit["rows"]})==32
    assert all("full[1,N,9]" in r["full_N_source_definition"] for r in audit["rows"])
    assert all(base.digest(p)==h for p,h in original.items())
    assert a.prepare(out,ap)==reg


def test_collision_checks_actual_full_and_common_free_suffix(tmp_path):
    _,_,_,main,plan=synthetic_fixture(tmp_path)
    rows,banks,_=a.materialize_arrays(plan,main)
    old={q["request_id"]:base.read_input(q) for q in main}
    q=rows[0];source=next(r for r in main if r["parent_id"]==q["parent_id"] and r["ratio"]==.4)
    # Collision only in common free slots must still fail even when K differs.
    banks[q["request_id"]]["z"][:,source["K"]:]=old[source["request_id"]]["z"][:,source["K"]:]
    with pytest.raises(ValueError,match="collides"):
        a.collision_audit(rows,banks,main,old)


def test_seed_and_model_independence_contract(tmp_path):
    _,_,_,main,plan=synthetic_fixture(tmp_path)
    a.validate_plan(plan,main)
    changed=copy.deepcopy(plan);changed["rows"][0]["seed"]+=1
    with pytest.raises(ValueError,match="seed differs"):a.validate_plan(changed,main)
    assert len({(q["parent_id"],q["candidate"]) for q in plan["rows"]})==32
    assert not plan["new_parent_generalization"]


def test_confirm_query_tamper_and_second_phase_are_rejected(tmp_path):
    out,ap,_,_,_=synthetic_fixture(tmp_path)
    reg=a.prepare(out,ap)
    with pytest.raises(ValueError,match="Only one128"):
        a.prepare(out,ap,"second_confirm")
    qp=sc.query_registration_path(out,reg["requests_path"])
    saved=base.read_json(qp);saved["rows"][0]["query_sha256"]="wrong";base.write_json(qp,saved)
    with pytest.raises(ValueError,match="query registration changed"):
        a.phase_registration(out,"confirm")


def test_all128_raw_barrier_before_score_or_render(tmp_path):
    out,ap,_,_,_=synthetic_fixture(tmp_path)
    reg=a.prepare(out,ap)
    for model in reg["models"]:
        cp=Path(model["checkpoint_path"]);cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b"synthetic-no-model")
        base.bind_checkpoint(out,"confirm",model["label"])
    for stage in ("collect","score","render"):
        with pytest.raises(RuntimeError,match="ALL_PHASE_RAW_REQUIRED_BEFORE_SCORING"):
            a.after_raw(out,"confirm",stage)
    assert not (out/"phases/confirm/evaluation/scoring_receipt.json").exists()


def test_generator_metadata_requires_confirmation_pair_and_authorization(tmp_path):
    _,ap,auth,_,_=synthetic_fixture(tmp_path);recipe=auth["selected_pair"][0]
    identity=dict(schema=a.GENERATOR_SCHEMA,experiment=a.EXPERIMENT,global_step=6000,recipe=recipe,segment=recipe,
        protocol=dict(input_namespace=auth["namespace"],training_stage="independent_confirmation",
                      independent_training_input_stream=True,independent_model_initialization=False,hybrid_schedule="late"),
        identities=dict(authorization_sha256=base.digest(ap),selected_pair=auth["selected_pair"],confirmation_namespace=auth["namespace"]))
    a.validate_generator_identity(identity,auth,recipe,base.digest(ap))
    identity["identities"]["authorization_sha256"]="another"
    with pytest.raises(ValueError,match="authorization lineage"):
        a.validate_generator_identity(identity,auth,recipe,base.digest(ap))


def test_no_sample_without_controller_lock_and_no_second_cell_attempt(tmp_path):
    with pytest.raises(RuntimeError,match="controller"):
        a.sample_label(tmp_path,"confirm","R4_MIX_H_LATE")
    path=tmp_path/"calls.jsonl";budget=base.CellBudget(path,dict(synthetic_fixture=True))
    budget.reserve("rollout_attempt")
    for _ in range(50):budget.call("sampling_forward",lambda:None)
    with pytest.raises(RuntimeError):budget.reserve("sampling_forward")
    with pytest.raises(FileExistsError):base.CellBudget(path,{})



def test_missing_existing_queries_never_generates_replacements(tmp_path):
    out,ap,_,_,_=synthetic_fixture(tmp_path)
    path=sc.query_registration_path(out,out/"evaluation/requests.json")
    path.unlink()
    banks={str(p):base.digest(p) for p in (out/"synthetic_query_banks").glob("*.npz")}
    with pytest.raises(FileNotFoundError):
        a.prepare(out,ap)
    assert not (out/"phases/confirm").exists()
    assert all(base.digest(p)==h for p,h in banks.items())


def test_namespace_matches_training_nested_suffix_contract(tmp_path):
    out,ap,auth,_,_=synthetic_fixture(tmp_path)
    auth["namespace"]=a.EXPERIMENT+"/confirmation/input_repeat1/runA"
    base.write_json(ap,auth)
    assert a.check_authorization(out,ap)["namespace"]==auth["namespace"]
    assert not (out/"phases/confirm").exists()
    for suffix in ("", "_leading", "/leading", "input.repeat1/runA"):
        auth["namespace"]=a.EXPERIMENT+"/confirmation/"+suffix
        base.write_json(ap,auth)
        with pytest.raises(ValueError,match="namespace suffix"):
            a.check_authorization(out,ap)
