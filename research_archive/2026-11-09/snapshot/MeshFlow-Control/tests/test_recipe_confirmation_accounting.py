"""Synthetic metadata/ledger checks; no real confirmation files or model work."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"tools/audit_recipe_confirmation_training.py"
spec = importlib.util.spec_from_file_location("confirmation_accounting", SOURCE)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n", encoding="utf8")


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf8")


def plan_and_pairs():
    raw = []; pairs = []
    for i in range(48):
        ratio = .2 if i%8 < 4 else .4
        row = dict(sample_index=i, origin_sample_index=i, uid="p"+str(i%16), task_id="t"+str(i%16),
            N=200, K=round(200*ratio), ratio=ratio, alpha=1. if i%4 < 2 else 1.05)
        row.update({field:a.derived_seed(a.NAMESPACE, domain, i) for field,domain in
            (("gaussian_seed","gaussian"),("epsilon_seed","epsilon"),("permutation_seed","permutation"),("t_seed","time"))})
        raw.append(row)
        pairs.append(dict(sample_index=i, origin_sample_index=i, uid=row["uid"], task_id=row["task_id"],
            N=200, K=row["K"], ratio=ratio, alpha=row["alpha"], step=i//8+1,
            free_coordinate_count=9*(200-row["K"]), lambda_value=.25, shared_input_sha256="new_s"+str(i),
            input_bytehash="new_i"+str(i), label_bytehash="new_l"+str(i), ot_cache_key="new_o"+str(i)))
    first = [i for i in range(32) if raw[i]["ratio"] == .2]
    second = [i for i in range(32) if raw[i]["ratio"] == .4]
    plan = dict(namespace=a.NAMESPACE, selected_pair=list(a.ARMS), mixed_records=raw,
        curriculum_orders={a.ARMS[0]:first+second+list(range(32,48))}, data_manifest_sha256="synthetic",
        eligible_parents=["p"+str(i) for i in range(16)],
        base_domain_seeds={d:a.derived_seed(a.NAMESPACE,d) for d in ("parent_schedule","task40","alpha","alpha_slot_shuffle","gaussian","epsilon","permutation","time")})
    docs = {}
    for arm in a.ARMS:
        records = a.confirmation_records(plan, arm)
        docs[arm] = dict(schema="meshflow_recipe_confirmation_paired_inputs_v1", recipe=arm,
            namespace=a.NAMESPACE, selected_pair=list(a.ARMS), hybrid_schedule="all", historical_paired_inputs_reused=False,
            records=[dict(pairs[r["sample_index"]],step=r["step"]) for r in records])
    return plan, docs


def fixture(root, arm, replay=False):
    plan, docs = plan_and_pairs(); records = a.confirmation_records(plan, arm)
    put(root/"streams/confirmation_plan.json", plan)
    for label, doc in docs.items():
        doc["plan_sha256"] = a.file_hash(root/"streams/confirmation_plan.json")
        put(root/"streams/paired_inputs"/(label+".json"), doc)
    auth = dict(updates_per_recipe=6); auth_path = root/"future_authorization.json"; put(auth_path, auth)
    attempts = [("initial", [1,2,3,4]), ("oom_serial_retry", [4,5,6])] if replay else [("initial", list(range(1,7)))]
    last_final = None
    for attempt, steps in attempts:
        folder = root/"training"/arm/"attempts"/attempt; folder.mkdir(parents=True)
        calls=[]; ex=[]; logs=[]; events=[]; diagnostics=[]; call_id=0
        if attempt != "initial": events.append(dict(event="RESUME", step=3, optimizer_reset=False))
        for step in steps:
            batch = records[(step-1)*8:step*8]; current=[]
            for micro, r in enumerate(batch):
                pair = docs[a.BASELINE]["records"][r["sample_index"]]
                row = dict(r,recipe=arm,attempt_id=attempt,free_coordinates=pair["free_coordinate_count"],lambda_value=.25,t=.5,
                    ot_cache_status="HIT",**{k:pair[k] for k in ("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key")})
                current.append(row)
                for kind in ("train_forward", "train_backward"):
                    call_id += 1
                    call = dict(attempt_id=attempt,segment=arm,call_id=call_id,kind=kind,step=step,microbatch=micro,
                        sample_index=r["sample_index"],input_bytehash=pair["input_bytehash"],label_bytehash=pair["label_bytehash"],
                        time_unix=step*100+call_id+(10000 if attempt != "initial" else 0))
                    calls += [dict(call,event="CALL_BEGIN"),dict(call,event="CALL_DONE")]
            call_id += 1
            call = dict(attempt_id=attempt,segment=arm,call_id=call_id,kind="optimizer_update",step=step,time_unix=step*100+call_id+(10000 if attempt != "initial" else 0))
            calls += [dict(call,event="CALL_BEGIN"),dict(call,event="CALL_DONE")]; ex += current
            hashes={k:a.json_hash([r[s] for r in current]) for k,s in a.BATCH_HASH.items()}
            denom=sum(r["free_coordinates"] for r in current)
            logs.append(dict(recipe=arm,attempt_id=attempt,step=step,lambda_value=.25,FM=.2,total_loss=.2,generator_gradient_norm=.1,
                generator_adamw_step=step,ratio20_samples=sum(r["ratio"]==.2 for r in current),ratio40_samples=sum(r["ratio"]==.4 for r in current),
                free_coordinate_denominator=denom,free20_coordinates=sum(r["free_coordinates"] for r in current if r["ratio"]==.2),
                free40_coordinates=sum(r["free_coordinates"] for r in current if r["ratio"]==.4),input_seconds=.1,
                update_seconds=.6,forward_seconds=.2,backward_seconds=.3,optimizer_seconds=.1,cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,**hashes))
            events += [dict(event="INPUT_DONE",step=step,**hashes),dict(event="UPDATE_DONE",step=step,generator_adamw_step=step)]
            diagnostics.append(dict(event="SAMPLE_LOSSES",step=step,lambda_value=.25,values=[dict(sample_index=r["sample_index"],
                alpha=r["alpha"],FM_contribution=.025,raw_align=None,alignment_eligible=False,batch_free_denominator=denom) for r in current]))
        a.write_csv(folder/"training_log.csv",logs); a.write_csv(folder/"exposure.csv",ex)
        lines(folder/"calls.jsonl",calls); lines(folder/"events.jsonl",events); lines(folder/"diagnostics.jsonl",diagnostics)
        last_final=dict(status="OOM" if replay and attempt=="initial" else "COMPLETE",physical_optimizer_updates_this_attempt=len(steps),
            successful_calls=dict(train_forward=8*len(steps),train_backward=8*len(steps),optimizer_update=len(steps)),wall_seconds=len(steps))
        put(folder/"final_status.json",last_final)
    receipts=[]
    for path in (root/"checkpoints"/(arm+"_step6000.pt"),root/"inference"/(arm+"_generator_step6000.pt")):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"synthetic opaque bytes, not a tensor")
        receipts.append(dict(path=str(path),sha256=a.file_hash(path),global_step=6,recipe=arm,generator_state_sha256="same"))
    put((root/"checkpoints"/(arm+"_step6000.pt")).with_suffix(".json"),receipts[0])
    put(root/"training"/arm/"completed.json",dict(status="COMPLETE",step=6,identities=dict(
        plan_sha256=a.file_hash(root/"streams/confirmation_plan.json"),paired_inputs_sha256=a.file_hash(root/"streams/paired_inputs"/(arm+".json")),
        authorization_sha256=a.file_hash(auth_path),data_manifest_sha256="synthetic",selected_pair=list(a.ARMS),confirmation_namespace=a.NAMESPACE),
        checkpoint=receipts[0],inference_export=receipts[1]))
    return plan,auth,auth_path


def audit_fixture(root,arm,replay=False):
    plan,auth,path=fixture(root,arm,replay); audit=a.Audit()
    tables={k:[] for k in ("updates","exposures","groups","calls","attempts")}
    result=a.audit_arm(root,arm,plan,auth,audit,tables,{"t"+str(i):"synthetic" for i in range(16)},path)
    return result,audit,tables


@pytest.mark.parametrize("arm",a.ARMS)
def test_candidate_and_new_control_complete_ledger(tmp_path,arm):
    result,audit,tables=audit_fixture(tmp_path,arm)
    assert result["unique_successful_updates"]==result["physical_successful_optimizer_updates"]==6
    assert sum(r["succeeded"] for r in tables["calls"] if r["kind"]=="train_forward")==48
    assert all(c["status"]=="PASS" for c in audit.checks),audit.checks


@pytest.mark.parametrize("arm",a.ARMS)
def test_replay_keeps_physical_counts_and_logical_multiset(tmp_path,arm):
    result,audit,tables=audit_fixture(tmp_path,arm,replay=True)
    assert result["physical_successful_optimizer_updates"]==7 and result["unique_successful_updates"]==6
    assert result["successful_replayed_updates"]==1
    assert sum(r["final_logical_trajectory"] for r in tables["exposures"])==48
    assert len(tables["exposures"])==56
    assert all(c["status"]=="PASS" for c in audit.checks),audit.checks


def test_pairing_allows_only_registered_step_reorder():
    plan,docs=plan_and_pairs()
    records=a.paired_contract(plan,a.ARMS[0],docs[a.ARMS[0]],docs[a.BASELINE])
    assert records[4]["sample_index"]==8 and records[4]["step"]==1
    changed=copy.deepcopy(docs[a.ARMS[0]]); changed["records"][4]["input_bytehash"]="bad"
    with pytest.raises(ValueError,match="exact input bytes"):a.paired_contract(plan,a.ARMS[0],changed,docs[a.BASELINE])
    changed=copy.deepcopy(docs[a.ARMS[0]]);changed["historical_paired_inputs_reused"]=True
    with pytest.raises(ValueError,match="identity"):a.paired_contract(plan,a.ARMS[0],changed,docs[a.BASELINE])


def test_rejects_changed_suffix_or_duplicate_origin():
    plan,docs=plan_and_pairs()
    plan["curriculum_orders"][a.ARMS[0]][-1],plan["curriculum_orders"][a.ARMS[0]][-2]=46,47
    with pytest.raises(ValueError,match="suffix"):a.confirmation_records(plan,a.ARMS[0])
    plan,docs=plan_and_pairs(); plan["mixed_records"][1]["sample_index"]=0
    with pytest.raises(ValueError,match="origin"):a.confirmation_records(plan,a.BASELINE)


@pytest.mark.parametrize("fault",("denominator","auxiliary","sum"))
def test_sample_diagnostics_detect_invalid_FM_contract(tmp_path,fault):
    arm=a.ARMS[0];plan,auth,auth_path=fixture(tmp_path,arm)
    path=tmp_path/"training"/arm/"attempts/initial/diagnostics.jsonl"
    rows=[json.loads(r) for r in path.read_text().splitlines()]
    sample=rows[0]["values"][0]
    if fault=="denominator":sample["batch_free_denominator"]//=8
    elif fault=="auxiliary":sample["alignment_eligible"]=True
    else:sample["FM_contribution"]=.04
    lines(path,rows)
    audit=a.Audit();tables={k:[] for k in ("updates","exposures","groups","calls","attempts")}
    a.audit_arm(tmp_path,arm,plan,auth,audit,tables,{"t"+str(i):"synthetic" for i in range(16)},auth_path)
    assert any(c["status"]=="FAIL" and "sample_FM" in c["item"] for c in audit.checks)


def test_namespace_seeds_and_historical_cache_reuse_checks():
    plan,docs=plan_and_pairs(); oldplan=dict(namespace="historical",eligible_parents=plan["eligible_parents"])
    historical=copy.deepcopy(docs[a.BASELINE])
    for r in historical["records"]:
        for k in ("shared_input_sha256","input_bytehash","ot_cache_key"):r[k]="old_"+r[k]
    audit=a.Audit();a.independent_input_checks(plan,oldplan,docs,historical,audit)
    assert all(c["status"]=="PASS" for c in audit.checks)
    plan["mixed_records"][0]["gaussian_seed"]+=1
    historical["records"][0]["ot_cache_key"]=docs[a.BASELINE]["records"][0]["ot_cache_key"]
    audit=a.Audit();a.independent_input_checks(plan,oldplan,docs,historical,audit)
    assert sum(c["status"]=="FAIL" for c in audit.checks)==2


def test_missing_actual_training_returns_pending_without_writes(tmp_path):
    result,target=a.run_audit(tmp_path)
    assert result["status"]=="PENDING" and target is None
    assert not list(tmp_path.iterdir())


def test_import_has_no_torch_numpy_or_model_loading():
    code="import runpy,sys;runpy.run_path("+repr(str(SOURCE))+");assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable,"-B","-c",code],check=True)


def test_transitive_source_metadata_does_not_import_trainer():
    sources=a.expected_training_sources()
    assert len(sources)==26 and "training/recipe_confirmation.py" in sources


def test_two_arm_final_multiset_and_suffix(tmp_path):
    plan,docs=plan_and_pairs(); tables={"exposures":[]}
    for arm in a.ARMS:
        for row in docs[arm]["records"]:
            tables["exposures"].append(dict(row,recipe=arm,final_logical_trajectory=True,
                free_coordinates=row["free_coordinate_count"],t=.5))
    audit=a.Audit();a.final_pair_checks(plan,tables,audit)
    assert all(c["status"]=="PASS" for c in audit.checks)
    tables["exposures"][-1]["input_bytehash"]="changed_suffix"
    audit=a.Audit();a.final_pair_checks(plan,tables,audit)
    failed={c["item"] for c in audit.checks if c["status"]=="FAIL"}
    assert failed=={"actual_same_input_byte_multiset","actual_same_final_2000_update_suffix"}


def registration_values(root):
    out=root/"confirmation"
    auth=dict(schema="chair_recipe_confirmation_training_authorization_v1",experiment="CHAIR_CONDITION_RECIPE_FACTORIAL_V1",
        stage="independent_confirmation",selected_pair=list(a.ARMS),authorized_recipes=list(a.ARMS),hybrid_schedules={r:"all" for r in a.ARMS},
        namespace=a.NAMESPACE,out=str(out),experiment_root=str(root),updates_per_recipe=6000,effective_batch=8,microbatch=1,max_training_recipes=2)
    plan=dict(schema="meshflow_recipe_confirmation_plan_v1",namespace=a.NAMESPACE,selected_pair=list(a.ARMS),hybrid_schedules=auth["hybrid_schedules"],
        total_updates=6000,effective_batch_count=6000,batch_size=8,microbatch_size=1,sample_count=48000,
        independent_training_input_stream=True,independent_model_initialization=False,Geo_seed=1010,no_historical_pair_or_OT_reuse=True,
        coupling_contract=a.CONTRACT,FM_normalization="Unchanged effective-batch valid free scalar count",mixed_records=[{}]*48000,
        base_records=[{}]*48000,eligible_parents=list(range(26)))
    return out,auth,plan


@pytest.mark.parametrize("fault",("namespace","pair","budget","historical","model_seed","none"))
def test_explicit_final_repeat_authorization_and_plan(tmp_path,fault):
    out,auth,plan=registration_values(tmp_path)
    if fault=="namespace":auth["namespace"]="CHAIR_CONDITION_RECIPE_FACTORIAL_V1/stage1"
    elif fault=="pair":auth["selected_pair"].reverse()
    elif fault=="budget":auth["updates_per_recipe"]=6001
    elif fault=="historical":plan["base_plan_sha256"]="old"
    elif fault=="model_seed":plan["independent_model_initialization"]=True
    if fault=="none":a.validate_registration(auth,plan,tmp_path,out)
    else:
        with pytest.raises(ValueError):a.validate_registration(auth,plan,tmp_path,out)


def test_duplicate_forward_slot_is_not_hidden_by_eight_total_calls(tmp_path):
    arm=a.BASELINE;plan,auth,path=fixture(tmp_path,arm)
    callpath=tmp_path/"training"/arm/"attempts/initial/calls.jsonl"
    rows=[json.loads(r) for r in callpath.read_text().splitlines()]
    for r in rows:
        if r["step"]==1 and r["kind"]=="train_forward" and r.get("sample_index")==1:
            r.update(sample_index=0,microbatch=0,input_bytehash="new_i0",label_bytehash="new_l0")
    lines(callpath,rows)
    audit=a.Audit();tables={k:[] for k in ("updates","exposures","groups","calls","attempts")}
    a.audit_arm(tmp_path,arm,plan,auth,audit,tables,{"t"+str(i):"synthetic" for i in range(16)},path)
    assert any(r["audit_status"]=="FAIL" and "per_registered_sample_count" in r["audit_errors"] for r in tables["updates"])


@pytest.mark.parametrize("fresh",(True,False))
def test_endpoint_receipts_fresh_initialization_and_schema(tmp_path,fresh):
    arm=a.ARMS[0];folder=tmp_path/"training"/arm;hash64="a"*64
    init=dict(selected_state="ema",official_file_sha256=a.OFFICIAL_SHA,generator_initial_sha256=a.INITIAL_SHA,
        role_nonzero=0,geo_exit_nonzero=0,Geo_seed=1010,fresh_AdamW=fresh,new_optimizer_step=0,teacher_loaded=False,no_auxiliary_head_constructed=True)
    plan=dict(namespace=a.NAMESPACE,implementation_sources={"fake.py":hash64});sources={"fake.py":hash64}
    completed=dict(recipe=arm,segment=arm,namespace=a.NAMESPACE,hybrid_schedule="all",teacher_forwards=0,head_forwards=0,
        alignment_active=False,attempt_id="initial",lineage=init,identities=dict(training_sources=sources,stream_implementation_sha256=a.json_hash(plan["implementation_sources"])),
        protocol_sha256=hash64,checkpoint=dict(schema="chair_recipe_confirmation_training_v1",protocol_sha256=hash64,
            optimizer_state_sha256=hash64,rng_sha256=hash64,stream_state_sha256=hash64),
        inference_export=dict(schema="chair_recipe_confirmation_generator_v1",protocol_sha256=hash64))
    put(folder/"completed.json",completed);put(folder/"initialization_audit.json",init);put(folder/"attempts/initial/final_status.json",completed)
    audit=a.Audit();a.verify_endpoint_metadata(tmp_path,arm,plan,tmp_path/"auth.json",audit,sources)
    assert any(c["status"]=="FAIL" for c in audit.checks)==(not fresh)


def test_resources_only_within_confirmation_training_worker_intervals(tmp_path):
    arm=a.ARMS[0]
    start=lambda phase,wave,t:dict(event="WORKER_START",phase=phase,wave=wave,name=arm,attempt="initial",time_unix=t,worker_count=1)
    end=lambda phase,wave,t:dict(event="WORKER_END",phase=phase,wave=wave,name=arm,attempt="initial",time_unix=t,wall_seconds=10,returncode=0)
    resource=lambda t:dict(event="RESOURCES",time_unix=t,active=[arm],snapshot=dict(gpu=None,cpu_cores=8,ram={},loadavg=[0]))
    events=[start("training","curriculum_training",0),resource(5),end("training","curriculum_training",10),
        start("training","confirmation_training",20),resource(25),end("training","confirmation_training",30),resource(35),
        start("sampling","confirmation_sampling",40),resource(45),end("sampling","confirmation_sampling",50)]
    lines(tmp_path/"audit/controller.jsonl",events)
    tables={"resources":[],"dispatch":[]};audit=a.Audit()
    value=a.resources_and_dispatch(tmp_path,audit,tables)
    assert [r["time_unix"] for r in tables["resources"]]==[25]
    assert len(value["intervals"])==1 and value["intervals"][0]["wave"]=="confirmation_training"
