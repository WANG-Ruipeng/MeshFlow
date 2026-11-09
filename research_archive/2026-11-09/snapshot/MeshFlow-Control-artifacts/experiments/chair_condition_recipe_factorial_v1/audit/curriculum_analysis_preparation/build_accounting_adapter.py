import ast
from pathlib import Path
repo=Path("/mnt/e/MeshFlow-Control")
old=repo/"tools/audit_recipe_factorial_training.py"
source=old.read_text(encoding="utf-8-sig")
node=ast.parse(source)
func={n.name:ast.get_source_segment(source,n) for n in node.body if isinstance(n,ast.FunctionDef)}
header='''#!/usr/bin/env python3
"""CPU-only curriculum training accounting. No model/OT or checkpoint loading.

Scan/update accounting is adapted from the preserved factorial auditor, with
explicit curriculum position maps, H_ALL and course receipt identities.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

_spec=importlib.util.spec_from_file_location("_factorial_accounting_helpers",Path(__file__).with_name("audit_recipe_factorial_training.py"))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
Audit=base.Audit
file_hash,json_hash,read=base.file_hash,base.json_hash,base.read
csv_rows,write_csv=base.csv_rows,base.write_csv
verify_file_receipt,condition_family,group_key=base.verify_file_receipt,base.condition_family,base.group_key
BATCH_HASH,EXPECTED_CALLS=base.BATCH_HASH,base.EXPECTED_CALLS
ARMS=("C20_40_MIX","C40_20_MIX")
BASELINE="R2_MIX_H_ALL"
DEFAULT=base.DEFAULT
VERSION="chair_recipe_curriculum_training_accounting_v1"


def lam(arm, step):
    if arm not in ARMS or type(step) is not int or step < 1:
        raise ValueError("Unregistered curriculum/step")
    return .25


def curriculum_records(plan, arm):
    raw=plan["mixed_records"];order=plan["curriculum_orders"][arm]
    if len(order)!=len(raw) or sorted(order)!=list(range(len(raw))):
        raise ValueError("Course is not an exact R2 multiset permutation")
    count=len(raw);third=count//3
    if count%24 or order[2*third:]!=list(range(2*third,count)):
        raise ValueError("Last2000-update MIX suffix order differs")
    first=.2 if arm==ARMS[0] else .4
    for pos,index in enumerate(order):
        if raw[index]["sample_index"]!=index or raw[index].get("origin_sample_index",index)!=index:
            raise ValueError("Stable origin identity differs")
        ratio=float(raw[index]["ratio"])
        if pos<third and ratio!=first or third<=pos<2*third and ratio!=.6-first:
            # Compare to an exact literal, not .6-.2 roundoff.
            required=first if pos<third else (.4 if first==.2 else .2)
            if pos<2*third and ratio!=required:raise ValueError("Pure-ratio course segment differs")
    records=[dict(raw[index],origin_sample_index=index,step=pos//8+1,optimizer_step=pos//8+1,
                  effective_batch_index=pos//8,slot_index=pos%8) for pos,index in enumerate(order)]
    for start in range(0,len(records),8):
        batch=records[start:start+8]
        if len({r["uid"] for r in batch})!=8 or sum(float(r["alpha"])==1 for r in batch)!=4:
            raise ValueError("Course batch must retain eight parents/four clean")
        if start>=2*third and sum(float(r["ratio"])==.2 for r in batch)!=4:
            raise ValueError("Final segment must retain MIX4+4")
    return records


def paired_contract(plan,arm,pair_doc,r2_doc):
    records=curriculum_records(plan,arm)
    order=plan["curriculum_orders"][arm]
    if pair_doc.get("schema")!="recipe_factorial_paired_inputs_v1" or pair_doc.get("recipe")!=arm+"__H_ALL":
        raise ValueError("Curriculum paired schema/recipe differs")
    if pair_doc["records"]!=[r2_doc["records"][i] for i in order]:
        raise ValueError("Course paired inputs are not exact ordered R2 rows")
    if any(float(r["lambda_value"])!=.25 for r in pair_doc["records"]):
        raise ValueError("Course must retain H_ALL")
    return records


def planned_groups(arm,records,families):
    groups=defaultdict(Counter)
    segment=max(1,len(records)//24)
    for r in records:
        c=groups[group_key(r,families)]
        c["samples"]+=1;c["free"]+=9*(r["N"]-r["K"])
        c["clean"]+=float(r["alpha"])==1;c["augmented"]+=float(r["alpha"])!=1
        c["segment"+str((r["optimizer_step"]-1)//segment)]+=1
    result=[]
    for (uid,ratio,family),c in groups.items():
        r=dict(recipe=arm,uid=uid,ratio=ratio,condition_family=family,planned_samples=c["samples"],
               planned_free_scalar_supervision=c["free"],planned_clean_samples=c["clean"],planned_augmented_samples=c["augmented"])
        for i,label in enumerate(("1_2000","2001_4000","4001_6000")):r["planned_step_"+label+"_samples"]=c["segment"+str(i)]
        result.append(r)
    return result

'''
scan=func["scan_calls"].replace('if index // 8 + 1 != step or int(row["microbatch"]) != index % 8:',
                               'if expected["_optimizer_step"] != step or int(row["microbatch"]) != expected["_slot_index"]:')
compare=func["compare_exposure"].replace('if int(row["step"]) != int(row["sample_index"]) // 8 + 1:',
                                        'if int(row["step"]) != planned["optimizer_step"]:')
arm=func["audit_arm"]
arm=arm.replace('def audit_arm(run, arm, plan, base, authorization, planned_groups, audit, out_rows, task_families):',
'''def audit_arm(run, arm, plan, authorization, audit, out_rows, task_families):''')
arm=arm.replace('(arm + ".json")','(arm + "__H_ALL.json")')
arm=arm.replace('    paired = {r["sample_index"]: r for r in pair_doc["records"]}\n    records = base["records"] if arm == ARMS[1] else plan["mixed_records"]',
'''    r2_path=run/"streams/paired_inputs"/(BASELINE+".json")
    audit.record(r2_path);r2_doc=read(r2_path)
    records=paired_contract(plan,arm,pair_doc,r2_doc)
    paired={r["sample_index"]:dict(r,_optimizer_step=pos//8+1,_slot_index=pos%8)
            for pos,r in enumerate(pair_doc["records"])}
    planned=planned_groups(arm,records,task_families)''')
arm=arm.replace('for row in planned_groups:', 'for row in planned:')
arm=arm.replace('if e.get("step") in (4000, 4001)', 'if e.get("step") in (2000, 2001, 4000, 4001)')
arm=arm.replace('key=lambda r: int(r["sample_index"])','key=lambda r: by_index[int(r["sample_index"])]["slot_index"]')
arm=arm.replace('expected_indices = list(range((step - 1) * 8, step * 8))',
                'expected_indices = [r["sample_index"] for r in records[(step-1)*8:step*8]]')
arm=arm.replace('str((int(sample["step"]) - 1) // 2000)', 'str((int(sample["step"]) - 1) // max(1,total//3))')
arm=arm.replace('for step in (4000, 4001):','for step in (2000, 2001, 4000, 4001):')
arm=arm.replace('run / "registrations/stage1_training.json"','run / "registrations/curriculum_training.json"')
arm=arm.replace('lambda_boundary=boundary, switch_and_resume_events=event_switch',
                'curriculum_boundaries=boundary, switch_and_resume_events=event_switch')
# State the actual course slot and ratio exposure in boundary diagnostics.
arm=arm.replace('attempt_id=selected_row[1] if selected_row else None,\n            status=', '''attempt_id=selected_row[1] if selected_row else None,
            observed_ratio20_samples=int(selected_row[2]["ratio20_samples"]) if selected_row else None,
            observed_ratio40_samples=int(selected_row[2]["ratio40_samples"]) if selected_row else None,
            status=''')
resource=func["resources_and_dispatch"].replace('if e.get("event") == "RESOURCES":',
    'if e.get("event") == "RESOURCES" and any(arm in e.get("active", []) for arm in ARMS):')
tail='''def run_audit(run,destination=None):
    started=time.time();run=Path(run).resolve()
    target=Path(destination) if destination else run/"audit/curriculum_training_accounting"
    if target.exists():raise FileExistsError("Choose a fresh audit destination")
    auth_path=run/"registrations/curriculum_training.json";plan_path=run/"streams/recipe_plan.json"
    auth,plan=read(auth_path),read(plan_path)
    if (auth.get("schema")!="chair_condition_recipe_factorial_training_authorization_v1" or
        auth.get("experiment")!="CHAIR_CONDITION_RECIPE_FACTORIAL_V1" or
        tuple(auth.get("authorized_recipes",[]))!=ARMS or auth.get("hybrid_schedules")!={a:"all" for a in ARMS} or
        auth.get("namespace")!=plan.get("namespace") or auth.get("out")!=str(run) or
        (auth.get("updates_per_recipe"),auth.get("effective_batch"),auth.get("microbatch"))!=(6000,8,1)):
        raise ValueError("Only the prospectively authorized two H_ALL curricula are in scope")
    if plan.get("schema")!="meshflow_recipe_factorial_plan_v1" or len(plan["mixed_records"])!=48000:
        raise ValueError("Frozen48000-slot recipe plan required")
    # Preserve a pending receipt rather than reading mutable latest checkpoint or live logs.
    missing=[str(run/"training"/a/"completed.json") for a in ARMS if not (run/"training"/a/"completed.json").exists()]
    if missing:return dict(schema=VERSION,status="PENDING",missing_completed_files=missing,scientific_calls=0),None
    audit=Audit();tables={k:[] for k in ("updates","exposures","groups","calls","attempts","resources","dispatch")}
    for p in (auth_path,plan_path,Path(__file__),Path(base.__file__)):audit.record(p)
    repo=Path(__file__).resolve().parents[1]
    for rel,h in auth["source_hashes"].items():
        p=repo/rel;audit.record(p);audit.check("","","frozen_training_source_hash",p.is_file() and file_hash(p)==h,rel)
    construction_path=run/"audit/condition_construction_audit.json";construction=read(construction_path)
    manifest=construction["source_files"]["imported_manifest"];mp=Path(manifest["path"])
    audit.record(construction_path);audit.record(mp)
    audit.check("","","task_manifest_hash",file_hash(mp)==manifest["sha256"]==plan["data_manifest_sha256"])
    families={t["task_id"]:condition_family(t) for t in read(mp)["tasks"] if t["ratio"] in (.2,.4)}
    paired_hash_path=run/"registrations/paired_hashes.json";paired_hashes=read(paired_hash_path);audit.record(paired_hash_path)
    for name in (BASELINE,*(a+"__H_ALL" for a in ARMS)):
        relative="streams/paired_inputs/"+name+".json";p=run/relative
        audit.check(name,"","registered_paired_table_hash",file_hash(p)==paired_hashes[relative])
    results=[audit_arm(run,a,plan,auth,audit,tables,families) for a in ARMS]
    initial={}
    for arm in (*ARMS,BASELINE):
        p=run/"training"/arm/"initialization_audit.json";audit.record(p);i=read(p);initial[arm]=i
        audit.check(arm,"","fresh_official_initialization",i.get("fresh_AdamW") is True and i.get("new_optimizer_step")==0
            and i.get("selected_state")=="ema" and i.get("role_nonzero")==0 and i.get("geo_exit_nonzero")==0
            and i.get("teacher_loaded") is False and i.get("no_auxiliary_head_constructed") is True)
    audit.check("","","same_official_initialization_as_R2",len({i["generator_initial_sha256"] for i in initial.values()})==1
        and len({i["official_file_sha256"] for i in initial.values()})==1)
    # Final selected exposures must exactly recover every stable origin and the fixed suffix.
    for arm in ARMS:
        ex=[r for r in tables["exposures"] if r["recipe"]==arm and r["final_logical_trajectory"]]
        records=curriculum_records(plan,arm)
        order={(r["step"],r["sample_index"]):pos for pos,r in enumerate(records)}
        ex.sort(key=lambda r:order[int(r["step"]),int(r["sample_index"])])
        audit.check(arm,"","actual_final_exact_origin_order",
                    [int(r["sample_index"]) for r in ex]==plan["curriculum_orders"][arm])
        audit.check(arm,"","actual_final_same48000_multiset",
                    sorted(int(r["sample_index"]) for r in ex)==list(range(48000)))
        audit.check(arm,"","actual_last16000_slots_exact_R2",
                    [int(r["sample_index"]) for r in ex[32000:]]==list(range(32000,48000)))
        audit.check(arm,"","actual_free_scalar_total",sum(int(r["free_coordinates"]) for r in ex)==
                    sum(9*(r["N"]-r["K"]) for r in plan["mixed_records"]))
    dispatch=resources_and_dispatch(run,audit,tables)
    status="FAIL" if any(c["status"]=="FAIL" for c in audit.checks) else "PENDING" if any(c["status"]=="PENDING" for c in audit.checks) else "PASS"
    result=dict(schema=VERSION,status=status,run=str(run),arms=results,checks=audit.checks,inputs=audit.sources,initializations=initial,
        new_unique_logical_updates=sum(r["unique_successful_updates"] for r in results),
        new_physical_optimizer_updates=sum(r["physical_successful_optimizer_updates"] for r in results),
        replayed_successful_updates=sum(r["successful_replayed_updates"] for r in results),
        authorized_new_updates=12000,baseline_R2_new_updates=0,baseline_R2_historical_updates=6000,
        dispatch=dispatch,scientific_calls=0,torch_imported="torch" in sys.modules,elapsed_seconds=time.time()-started,
        evidence_limits=["No tensor deserialization. Internal model/AdamW/RNG hashes are receipt claims bound to immutable actual file SHA.",
        "Actual order/multiset/free supervision is reconstructed from successful logical updates; every failed and replayed physical attempt remains counted.",
        "Concurrent elapsed times include contention and are not isolated compute benchmarks.",
        "R2 contributes only prior small initialization/paired receipts and no new training to this course audit."])
    target.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items():write_csv(target/(name+".csv"),rows)
    write_csv(target/"checks.csv",audit.checks)
    (target/"accounting.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\\n",encoding="utf8")
    lines=["# Curriculum training accounting","","Status: "+status,"",
           "Exactly two H_ALL curricula; R2 contributes zero new updates. No model/OT/checkpoint deserialization.","",
           "| Recipe | Unique | Physical | Completed replay |","|---|---:|---:|---:|"]
    lines += ["| %s | %s | %s | %s |"%(r["recipe"],r["unique_successful_updates"],r["physical_successful_optimizer_updates"],r["successful_replayed_updates"]) for r in results]
    lines += ["",*result["evidence_limits"],"",dispatch["timing_note"],"",dispatch["memory_note"]]
    lines += ["",*["- "+str(c) for c in audit.checks if c["status"]!="PASS"]]
    (target/"ACCOUNTING.md").write_text("\\n".join(lines)+"\\n",encoding="utf8")
    return result,target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,default=DEFAULT);parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv);result,target=run_audit(args.run,args.output)
    print(json.dumps(dict(status=result["status"],output=str(target) if target else None,
                         scientific_calls=0,missing=result.get("missing_completed_files",[]))),flush=True)
    return 0 if result["status"]=="PASS" else 2 if result["status"]=="PENDING" else 1


if __name__=="__main__":raise SystemExit(main())
'''
out=repo/"tools/audit_recipe_curriculum_training.py"
out.write_text(header+"\n\n"+scan+"\n\n"+compare+"\n\n"+arm+"\n\n"+resource+"\n\n"+tail,encoding="utf8")
print(out)

