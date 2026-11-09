"""Completed main sampling bookkeeping only; no model, score or render calls."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path("/mnt/e/MeshFlow-Control")
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
DEST=RUN/"audit/sampling_accounting"
sys.path.insert(0,str(REPO/"tools"))
import evaluate_recipe_factorial as base

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def csv_write(p,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols or ["status"]);w.writeheader();w.writerows(rows)

def ledger(path,context):
    counts=Counter();pending=False;pending_step=None;errors=[];steps=[];rollout_open=False
    data=path.read_bytes()
    if data and not data.endswith(b"\n"):errors.append("truncated ledger tail")
    for i,line in enumerate(data.splitlines(),1):
        try:r=json.loads(line)
        except (ValueError,TypeError) as e:
            errors.append("parse line "+str(i)+": "+repr(e));continue
        event=r.get("event");counts[event]+=1
        if "context" in r and any(r["context"].get(k)!=v for k,v in context.items()):
            errors.append("context line "+str(i))
        if event=="CELL_REGISTERED":
            if i!=1:errors.append("late/duplicate CELL_REGISTERED")
        elif event=="RESERVE":
            kind=r.get("kind");counts[kind+"_reserved"]+=1
            if r.get("ordinal")!=counts[kind+"_reserved"]:errors.append("ordinal line "+str(i))
            if kind=="rollout_attempt":
                if rollout_open:errors.append("second rollout")
                rollout_open=True
            elif kind=="sampling_forward":
                if pending or not rollout_open:errors.append("invalid forward nesting")
                pending=True;pending_step=r.get("context",{}).get("euler_step")
                steps.append(pending_step)
            else:errors.append("unknown reserve kind")
        elif event in ("RETURN","ERROR"):
            kind=r.get("kind");counts[kind+("_returned" if event=="RETURN" else "_failed")]+=1
            if kind=="sampling_forward":
                if not pending:errors.append("unpaired forward terminal")
                pending=False
            elif kind=="rollout_attempt":
                rollout_open=False
            else:errors.append("unknown terminal kind")
        elif event=="ROLLOUT_COMPLETE":
            if not rollout_open or pending:errors.append("unpaired rollout completion")
            rollout_open=False
            if r.get("nfe")!=50 or r.get("known_all51_bitwise") is not True:
                errors.append("rollout final proof")
        else:errors.append("unknown event "+str(event))
    if pending:counts["forward_unfinished"]+=1
    if rollout_open:counts["rollout_unfinished"]+=1
    if counts["CELL_REGISTERED"]!=1:errors.append("cell registration count")
    if counts["rollout_attempt_reserved"]!=1:errors.append("rollout budget")
    if counts["ROLLOUT_COMPLETE"]!=1:errors.append("rollout completed count")
    if counts["sampling_forward_reserved"]!=50 or counts["sampling_forward_returned"]!=50:
        errors.append("forward budget")
    if steps!=list(range(50)):errors.append("Euler step sequence")
    if counts["ERROR"] or counts["forward_unfinished"] or counts["rollout_unfinished"]:
        errors.append("failed or unclosed call")
    return dict(counts),errors,hashlib.sha256(data).hexdigest()

def main():
    started=time.time()
    if any(p.name!="audit_sampling.py" for p in DEST.iterdir()):
        raise FileExistsError("Preserve prior sampling accounting output")
    phase=RUN/"phases/main"
    registration_path=phase/"registration.json"
    registration=base.read_json(registration_path)
    request_path=base.resolve(registration["requests_path"])
    requests=base.read_json(request_path)["rows"]
    failures=[];rows=[];sources=[];all_calls={};matched_receipts=set();matched_calls=set()
    def record(path,kind):
        p=Path(path);s=dict(path=str(p),kind=kind,sha256=sha(p),bytes=p.stat().st_size)
        sources.append(s);return s["sha256"]
    if sha(request_path)!=registration["requests_sha256"]:
        failures.append(dict(kind="request registration hash"))
    for path,kind in ((registration_path,"phase_registration"),(request_path,"registered_inputs"),(Path(__file__),"auditor")):
        record(path,kind)
    frozen=base.read_json(RUN/"registrations/source_hashes.json")
    for rel in ("tools/evaluate_recipe_factorial.py","src/meshflow_control/evaluation/io.py",
                "src/meshflow_control/evaluation/hashing.py","src/meshflow_control/mesh_io.py"):
        h=record(REPO/rel,"frozen_contract_source")
        if h!=frozen[rel]:failures.append(dict(kind="frozen source hash",path=rel))
    expected={(label,q["request_id"]) for label in registration["labels"] for q in requests}
    for label in registration["labels"]:
        binding_path=phase/"checkpoints"/(label+".json")
        binding=base.read_json(binding_path);record(binding_path,"checkpoint_binding_metadata_only")
        for q in requests:
            output=phase/"outputs"/label/q["request_id"]
            rp=output/"receipt.json";cp=output/"calls.jsonl"
            row=dict(label=label,request_id=q["request_id"],parent_id=q["parent_id"],
                     condition_id=q["condition_id"],ratio=q["ratio"],candidate=q["candidate"],
                     N=q["N"],K=q["K"],status="PASS",errors="")
            errors=[]
            try:
                r=base.read_json(rp);matched_receipts.add(rp.resolve())
                row["receipt_path"]=str(rp);row["receipt_sha256"]=record(rp,"output_receipt")
                for k in ("request_id","parent_id","condition_id","ratio","candidate","N","K",
                          "input_sha256","C_sha256","Gaussian_sha256"):
                    if r.get(k)!=q[k]:errors.append("receipt_"+k)
                if r.get("label")!=label or r.get("status")!="COMPLETE":errors.append("label/status")
                row.update(generation_status=r["generation_status"],new_rollouts=r["new_rollouts"],
                    checkpoint_sha256=r["checkpoint_sha256"],input_sha256=r["input_sha256"],
                    C_sha256=r["C_sha256"],Gaussian_sha256=r["Gaussian_sha256"],input_path=q["input_path"])
                # This verifies actual RAW/PLY/51-state/input bytes and runtime dtype/nfe.
                arrays,facts=base.verify_output(q,r,binding["checkpoint_sha256"])
                row["actual_trajectory_states"]=int(arrays["trajectory"].shape[0])
                if row["actual_trajectory_states"]!=51:errors.append("trajectory_state_count")
                row.update(facts);del arrays
                if facts!=r.get("contracts"):errors.append("saved_contracts_disagree")
                for kind in ("raw","ply","result"):
                    row[kind+"_path"]=r[kind+"_path"];row[kind+"_sha256"]=r[kind+"_sha256"]
                    # verify_output already calculated and matched the actual file SHA.
                    sources.append(dict(path=r[kind+"_path"],kind=kind,sha256=r[kind+"_sha256"],
                                        bytes=base.resolve(r[kind+"_path"]).stat().st_size,
                                        verification="actual digest matched by frozen verify_output"))
                runtime=base.read_json(r["result_path"])
                runtime=runtime.get("sampling",runtime)
                row.update(runtime_nfe=runtime.get("nfe"),runtime_dtype=runtime.get("dtype"),
                           runtime_readout_mode=runtime.get("readout_mode"))
                if r["generation_status"]=="NEW_EULER50":
                    if r["new_rollouts"]!=1 or r.get("attempt")!="initial":errors.append("new_output_attempt")
                    context=dict(label=label,request_id=q["request_id"],attempt=r.get("attempt"),
                        phase="main",checkpoint_sha256=r["checkpoint_sha256"],input_sha256=q["input_sha256"])
                    counts,errs,h=ledger(cp,context);errors.extend(errs)
                    matched_calls.add(cp.resolve());all_calls[str(cp)]=counts
                    row.update(calls_path=str(cp),calls_sha256=h,
                        rollout_reserved=counts.get("rollout_attempt_reserved",0),
                        rollout_completed=counts.get("ROLLOUT_COMPLETE",0),
                        sampling_forward_reserved=counts.get("sampling_forward_reserved",0),
                        sampling_forward_succeeded=counts.get("sampling_forward_returned",0),
                        sampling_forward_failed=counts.get("sampling_forward_failed",0),
                        forward_unfinished=counts.get("forward_unfinished",0),
                        rollout_unfinished=counts.get("rollout_unfinished",0),
                        retry_count=max(0,counts.get("rollout_attempt_reserved",0)-1))
                    sources.append(dict(path=str(cp),kind="new_sampling_ledger",sha256=h,bytes=cp.stat().st_size))
                elif r["generation_status"]=="VERIFIED_HISTORICAL_RAW_REUSE":
                    if r["new_rollouts"]!=0:errors.append("reused charged new rollout")
                    if cp.exists():errors.append("reused cell has new-run calls")
                    row.update(rollout_reserved=0,rollout_completed=0,sampling_forward_reserved=0,
                        sampling_forward_succeeded=0,sampling_forward_failed=0,forward_unfinished=0,
                        rollout_unfinished=0,retry_count=0,
                        reused_cost_scope="Historical forwards excluded from this run")
                else:errors.append("unknown generation status")
            except Exception as e:
                errors.append(type(e).__name__+": "+str(e))
            if errors:
                row["status"]="FAIL";row["errors"]="; ".join(errors)
                failures.append(dict(label=label,request_id=q["request_id"],errors=errors))
            rows.append(row)
        print("audited",label,sum(r["label"]==label for r in rows),flush=True)
    # Detect orphan receipts/calls; no unregistered attempt is silently ignored.
    extra_receipts=[p for p in (phase/"outputs").glob("*/*/receipt.json") if p.resolve() not in matched_receipts]
    extra_calls=[p for p in (phase/"outputs").glob("*/*/calls.jsonl") if p.resolve() not in matched_calls]
    for p in extra_receipts:failures.append(dict(kind="unregistered or unread receipt",path=str(p)))
    for p in extra_calls:
        c,e,h=ledger(p,{})
        all_calls[str(p)]=c;record(p,"orphan_or_unmatched_calls")
        failures.append(dict(kind="orphan or unmatched ledger",path=str(p),errors=e))
    totals=Counter()
    for c in all_calls.values():totals.update(c)
    if len(rows)!=384 or len(expected)!=384:
        failures.append(dict(kind="registered output count",actual=len(rows),expected=384))
    new=sum(r.get("generation_status")=="NEW_EULER50" for r in rows)
    reused=sum(r.get("generation_status")=="VERIFIED_HISTORICAL_RAW_REUSE" for r in rows)
    if new!=312 or reused!=72 or totals["sampling_forward_returned"]!=15600:
        failures.append(dict(kind="expected main execution budget differs",new_outputs=new,reused_outputs=reused,actual_new_forwards=totals["sampling_forward_returned"]))
    groups=[]
    for label in registration["labels"]:
        selected=[r for r in rows if r["label"]==label]
        groups.append(dict(label=label,outputs=len(selected),
            new_outputs=sum(r.get("generation_status")=="NEW_EULER50" for r in selected),
            reused_outputs=sum(r.get("generation_status")=="VERIFIED_HISTORICAL_RAW_REUSE" for r in selected),
            actual_new_rollouts=sum(r.get("rollout_reserved",0) for r in selected),
            actual_successful_new_forwards=sum(r.get("sampling_forward_succeeded",0) for r in selected),
            failures=sum(r["status"]!="PASS" for r in selected)))
    failure_markers=list((phase/"sampling").glob("*/*failure.json"))
    retry_markers=list((phase/"sampling").glob("*/serial_retry_dispatch.json"))
    for p in failure_markers+retry_markers:
        record(p,"failure_or_retry_marker");failures.append(dict(kind="failure or retry marker",path=str(p)))
    summary=dict(schema="recipe_factorial_sampling_accounting_v1",status="FAIL" if failures else "PASS",
        scope="Completed main execution ledger and immutable output contracts; no quality/anonymous mapping read",
        outputs=len(rows),new_outputs=new,reused_outputs=reused,actual_new_ledger_files=len(all_calls),
        actual_rollouts_reserved=totals["rollout_attempt_reserved"],actual_rollouts_completed=totals["ROLLOUT_COMPLETE"],
        actual_new_sampling_forwards_reserved=totals["sampling_forward_reserved"],
        actual_new_sampling_forwards_succeeded=totals["sampling_forward_returned"],
        failed_call_events=totals["ERROR"],unfinished_forward_calls=totals["forward_unfinished"],
        unfinished_rollouts=totals["rollout_unfinished"],
        cell_retry_count=sum(max(0,c.get("rollout_attempt_reserved",0)-1) for c in all_calls.values()),
        dispatch_retry_markers=len(retry_markers),failure_markers=len(failure_markers),
        all_actual_output_contracts_passed=all(r["status"]=="PASS" for r in rows),
        expected_forward_budget=15600,expected_forward_budget_matches_actual=totals["sampling_forward_returned"]==15600,
        per_label=groups,exceptions=failures,source_files=sources,
        reused_forward_cost_new_run=0,torch_imported="torch" in sys.modules,
        model_operations=0,new_scoring_calls=0,new_render_calls=0,elapsed_seconds=time.time()-started)
    csv_write(DEST/"per_output.csv",rows);csv_write(DEST/"per_label.csv",groups)
    csv_write(DEST/"exceptions.csv",failures);csv_write(DEST/"source_hashes.csv",sources)
    (DEST/"accounting.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    md=["# Main sampling accounting","","Status: "+summary["status"],"",
        "Actual ledger totals: %s new rollouts, %s completed rollouts, %s successful sampling forwards; %s reused outputs contribute zero new forwards."%(summary["actual_rollouts_reserved"],summary["actual_rollouts_completed"],summary["actual_new_sampling_forwards_succeeded"],reused),
        "","All 384 registered outputs were checked against actual C/Gaussian input bytes, FP32 RAW and trajectory, PLY, 51 states, nfe50 and BF16/FP32 runtime proof.","",
        "The expected15600 is recorded separately from actual ledger counting. Failure, unfinished call and retry records are retained in accounting.json and exceptions.csv.","",
        "No model, scoring, rendering, quality summary or anonymous method mapping was used."]
    (DEST/"ACCOUNTING.md").write_text("\n".join(md)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_files","per_label","exceptions")}),flush=True)
    return 0 if not failures else 1

if __name__=="__main__":sys.exit(main())
