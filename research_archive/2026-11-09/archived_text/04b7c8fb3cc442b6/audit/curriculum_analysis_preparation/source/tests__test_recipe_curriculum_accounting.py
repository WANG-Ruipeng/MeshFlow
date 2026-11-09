"""Synthetic curriculum accounting only; never reads actual logs or tensors."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"tools/audit_recipe_curriculum_training.py"
spec=importlib.util.spec_from_file_location("course_accounting",SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def plan_and_pairs():
    raw=[];pairs=[]
    for i in range(48):
        slot=i%8;ratio=.2 if slot<4 else .4
        r=dict(sample_index=i,origin_sample_index=i,uid="p"+str(i%16),task_id="t"+str(i%16),
               N=200,K=int(200*ratio),ratio=ratio,alpha=1. if slot%4<2 else 1.05)
        raw.append(r)
        pairs.append(dict(sample_index=i,origin_sample_index=i,task_id=r["task_id"],free_coordinate_count=9*(r["N"]-r["K"]),
                          lambda_value=.25,shared_input_sha256="s"+str(i),input_bytehash="i"+str(i),label_bytehash="l"+str(i),ot_cache_key="o"+str(i)))
    first=[i for i in range(32) if raw[i]["ratio"]==.2];second=[i for i in range(32) if raw[i]["ratio"]==.4]
    plan=dict(mixed_records=raw,curriculum_orders={a.ARMS[0]:first+second+list(range(32,48)),
                                                a.ARMS[1]:second+first+list(range(32,48))},
              data_manifest_sha256="synthetic")
    return plan,dict(records=pairs)

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+"\n",encoding="utf8")

def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")

def fixture(root,replay=False,complete=True,arm=None):
    arm=arm or a.ARMS[0];plan,r2=plan_and_pairs();records=a.curriculum_records(plan,arm)
    put(root/"streams/recipe_plan.json",plan);put(root/"streams/paired_inputs"/(a.BASELINE+".json"),r2)
    pair_doc=dict(schema="recipe_factorial_paired_inputs_v1",recipe=arm+"__H_ALL",
                  plan_sha256=a.file_hash(root/"streams/recipe_plan.json"),
                  records=[r2["records"][i] for i in plan["curriculum_orders"][arm]])
    pp=root/"streams/paired_inputs"/(arm+"__H_ALL.json");put(pp,pair_doc)
    auth=dict(updates_per_recipe=6);put(root/"registrations/curriculum_training.json",auth)
    attempts=[("initial",list(range(1,5))),("oom_serial_retry",[4,5,6])] if replay else [("initial",list(range(1,7)))]
    for attempt,steps in attempts:
        folder=root/"training"/arm/"attempts"/attempt;folder.mkdir(parents=True)
        calls=[];exposures=[];logs=[];events=[];call_id=0
        if attempt!="initial":events.append(dict(event="RESUME",step=3,optimizer_reset=False))
        for step in steps:
            batch=records[(step-1)*8:step*8];ex=[]
            for micro,r in enumerate(batch):
                pair=r2["records"][r["sample_index"]]
                row=dict(r,recipe=arm,attempt_id=attempt,free_coordinates=pair["free_coordinate_count"],lambda_value=.25,
                         t=.5,ot_cache_status="HIT",**{k:pair[k] for k in ("shared_input_sha256","input_bytehash","label_bytehash","ot_cache_key")})
                ex.append(row)
                for kind in ("train_forward","train_backward"):
                    call_id+=1;c=dict(attempt_id=attempt,segment=arm,call_id=call_id,kind=kind,step=step,
                        microbatch=micro,sample_index=r["sample_index"],input_bytehash=pair["input_bytehash"],
                        label_bytehash=pair["label_bytehash"],time_unix=step*100+call_id+(10000 if attempt!="initial" else 0))
                    calls += [dict(c,event="CALL_BEGIN"),dict(c,event="CALL_DONE")]
            call_id+=1;c=dict(attempt_id=attempt,segment=arm,call_id=call_id,kind="optimizer_update",step=step,time_unix=step*100+call_id+(10000 if attempt!="initial" else 0))
            calls += [dict(c,event="CALL_BEGIN"),dict(c,event="CALL_DONE")];exposures+=ex
            hashes={k:a.json_hash([r[s] for r in ex]) for k,s in a.BATCH_HASH.items()}
            logs.append(dict(recipe=arm,attempt_id=attempt,step=step,lambda_value=.25,FM=.2,total_loss=.2,generator_gradient_norm=.1,
                generator_adamw_step=step,ratio20_samples=sum(r["ratio"]==.2 for r in ex),ratio40_samples=sum(r["ratio"]==.4 for r in ex),
                free_coordinate_denominator=sum(r["free_coordinates"] for r in ex),
                free20_coordinates=sum(r["free_coordinates"] for r in ex if r["ratio"]==.2),
                free40_coordinates=sum(r["free_coordinates"] for r in ex if r["ratio"]==.4),
                input_seconds=.1,update_seconds=.6,forward_seconds=.2,backward_seconds=.3,optimizer_seconds=.1,
                cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,**hashes))
            events += [dict(event="INPUT_DONE",step=step,**hashes),dict(event="UPDATE_DONE",step=step,generator_adamw_step=step)]
        a.write_csv(folder/"training_log.csv",logs);a.write_csv(folder/"exposure.csv",exposures)
        lines(folder/"calls.jsonl",calls);lines(folder/"events.jsonl",events)
        if complete:
            put(folder/"final_status.json",dict(status="OOM" if replay and attempt=="initial" else "COMPLETE",
                physical_optimizer_updates_this_attempt=len(steps),successful_calls=dict(train_forward=8*len(steps),train_backward=8*len(steps),optimizer_update=len(steps)),wall_seconds=len(steps)))
    if complete:
        receipts=[]
        for path in (root/"checkpoints"/(arm+"_step6000.pt"),root/"inference"/(arm+"_generator_step6000.pt")):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"opaque synthetic file: never deserialized")
            receipts.append(dict(path=str(path),sha256=a.file_hash(path),global_step=6,recipe=arm,generator_state_sha256="same"))
        put((root/"checkpoints"/(arm+"_step6000.pt")).with_suffix(".json"),receipts[0])
        put(root/"training"/arm/"completed.json",dict(status="COMPLETE",step=6,identities=dict(
            plan_sha256=a.file_hash(root/"streams/recipe_plan.json"),paired_inputs_sha256=a.file_hash(pp),
            authorization_sha256=a.file_hash(root/"registrations/curriculum_training.json"),data_manifest_sha256="synthetic"),
            checkpoint=receipts[0],inference_export=receipts[1]))
    return arm,plan,auth

def audit_fixture(root,replay=False,complete=True,arm=None):
    arm,plan,auth=fixture(root,replay,complete,arm)
    audit=a.Audit();tables={k:[] for k in ("updates","exposures","groups","calls","attempts")}
    result=a.audit_arm(root,arm,plan,auth,audit,tables,{"t"+str(i):"synthetic" for i in range(16)})
    return result,audit,tables

@pytest.mark.parametrize("arm",a.ARMS)
def test_both_course_order_and_all_hybrid(arm):
    plan,r2=plan_and_pairs();order=plan["curriculum_orders"][arm]
    pair=dict(schema="recipe_factorial_paired_inputs_v1",recipe=arm+"__H_ALL",records=[r2["records"][i] for i in order])
    records=a.paired_contract(plan,arm,pair,r2)
    assert [r["sample_index"] for r in records]==order
    assert records[4]["sample_index"]!=4
    assert a.lam(arm,1)==a.lam(arm,4000)==a.lam(arm,4001)==.25
    assert [r["sample_index"] for r in records[-16:]]==list(range(32,48))

def test_rejects_nonpermutation_changed_suffix_and_nonidentical_paired():
    plan,r2=plan_and_pairs()
    changed=copy.deepcopy(plan);changed["curriculum_orders"][a.ARMS[0]][0]=1
    with pytest.raises(ValueError,match="multiset"):a.curriculum_records(changed,a.ARMS[0])
    changed=copy.deepcopy(plan);changed["curriculum_orders"][a.ARMS[0]][-1],changed["curriculum_orders"][a.ARMS[0]][-2]=46,47
    with pytest.raises(ValueError,match="suffix"):a.curriculum_records(changed,a.ARMS[0])
    pair=dict(schema="recipe_factorial_paired_inputs_v1",recipe=a.ARMS[0]+"__H_ALL",records=copy.deepcopy([r2["records"][i] for i in plan["curriculum_orders"][a.ARMS[0]]]))
    pair["records"][0]["input_bytehash"]="wrong"
    with pytest.raises(ValueError,match="exact ordered"):a.paired_contract(plan,a.ARMS[0],pair,r2)

@pytest.mark.parametrize("arm",a.ARMS)
def test_complete_accounting_accepts_reordered_stable_origins(tmp_path,arm):
    result,audit,tables=audit_fixture(tmp_path,arm=arm)
    assert result["unique_successful_updates"]==6 and result["physical_successful_optimizer_updates"]==6
    assert result["successful_replayed_updates"]==0
    assert all(c["status"]=="PASS" for c in audit.checks),audit.checks
    assert len(tables["exposures"])==48
    assert all(r["final_logical_trajectory"] for r in tables["exposures"])

def test_replay_physical_count_and_unique_exposure_separated(tmp_path):
    result,audit,tables=audit_fixture(tmp_path,replay=True)
    assert result["physical_successful_optimizer_updates"]==7 and result["unique_successful_updates"]==6
    assert result["successful_replayed_updates"]==1
    assert sum(r["final_logical_trajectory"] for r in tables["exposures"])==48
    assert len(tables["exposures"])==56 and all(c["status"]=="PASS" for c in audit.checks)

def test_partial_is_pending_without_reading_latest(tmp_path):
    result,audit,tables=audit_fixture(tmp_path,complete=False)
    assert result["completion_state"]=="PENDING" and result["endpoint_files"]==[]
    assert any(c["status"]=="PENDING" for c in audit.checks)
    assert not any(c["status"]=="FAIL" for c in audit.checks)
    assert not any(Path(r["path"]).name=="latest.pt" for r in audit.sources)

def test_course_position_mismatch_and_hash_mutation_caught(tmp_path):
    arm,plan,auth=fixture(tmp_path)
    callpath=tmp_path/"training"/arm/"attempts/initial/calls.jsonl"
    rr=[json.loads(x) for x in callpath.read_text().splitlines()]
    # Stable origin8 belongs in step1/micro4, not origin-derived step2/micro0.
    chosen=next(r for r in rr if r.get("sample_index")==8)
    chosen["step"]=2;lines(callpath,rr)
    audit=a.Audit();tables={k:[] for k in ("updates","exposures","groups","calls","attempts")}
    a.audit_arm(tmp_path,arm,plan,auth,audit,tables,{"t"+str(i):"synthetic" for i in range(16)})
    assert any(c["status"]=="FAIL" for c in audit.checks)
    with pytest.raises(ValueError,match="Mutable"):a.verify_file_receipt(tmp_path/"latest.pt",{},audit,arm,"checkpoint")

