"""Read-only confirmation sampling accounting; no model or quality evaluation."""
from collections import Counter
import importlib.util
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

REPO=Path("/mnt/e/MeshFlow-Control")
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
DEST=RUN/"audit/confirmation_sampling_accounting"
PAIR=("C20_40_MIX","R2_MIX_H_ALL")
OLD_AUDITOR=RUN/"audit/sampling_accounting/audit_sampling.py"
OLD_AUDITOR_SHA="c3e973ed1c42fa7820ad76d34e8b58dabfc66a87ff73983d3ff1f128736ce7e7"
if hashlib.sha256(OLD_AUDITOR.read_bytes()).hexdigest()!=OLD_AUDITOR_SHA:
    raise ValueError("Frozen original ledger parser changed before import")
_spec=importlib.util.spec_from_file_location("_completed_main_sampling_ledger",OLD_AUDITOR)
_prior=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_prior)
sha,csv_write,ledger,base=_prior.sha,_prior.csv_write,_prior.ledger,_prior.base
import evaluate_recipe_confirmation as confirmation


def receipt_identity_errors(q,receipt,label,output,completed_row):
    errors=[]
    for key,value in q.items():
        if receipt.get(key)!=value:errors.append("receipt_"+key)
    if receipt!=completed_row:errors.append("completed_worker_row_disagrees")
    if receipt.get("label")!=label or receipt.get("status")!="COMPLETE":
        errors.append("label/status")
    if receipt.get("generation_status")!="NEW_EULER50" or receipt.get("new_rollouts")!=1:
        errors.append("confirmation_requires_one_new_rollout")
    for kind,name in (("raw","raw.npz"),("ply","raw.ply"),("result","run.json")):
        path=receipt.get(kind+"_path")
        if not path or base.resolve(path).resolve()!=(Path(output)/name).resolve():
            errors.append("output_path_"+kind)
    return errors


def readiness(phase):
    missing=[]
    for label in PAIR:
        done=phase/"sampling"/label/"completed.json"
        if not done.is_file():
            missing.append(str(done));continue
        if base.read_json(done).get("status")!="COMPLETE":
            missing.append(str(done)+": not COMPLETE")
    if len(list((phase/"outputs").glob("*/*/receipt.json")))<128:
        missing.append("128 immutable output receipt files not yet present")
    return missing

def audit(run=RUN,dest=None):
    run=Path(run).resolve()
    if run!=RUN.resolve():raise ValueError("Wrong experiment root")
    destination=Path(dest).resolve() if dest else DEST
    if not destination.is_relative_to(run/"audit"):raise ValueError("Audit output must remain below run/audit")
    started=time.time()
    phase=run/"phases/confirm"
    missing=readiness(phase)
    if missing:
        print(json.dumps(dict(status="PENDING",missing=missing,model_operations=0)),flush=True)
        return 2
    if destination.exists():
        raise FileExistsError("Preserve prior confirmation sampling accounting output")
    destination.mkdir(parents=True)
    registration_path=phase/"registration.json"
    registration=base.read_json(registration_path)
    if (registration.get("kind") != "confirm" or registration.get("phase") != "confirm"
        or tuple(registration.get("labels",())) != PAIR
        or registration.get("expected_outputs") != 128
        or registration.get("registered_per_model") != 64
        or registration.get("nfe_per_output") != 50
        or registration.get("max_cell_attempts") != 1):
        raise ValueError("Frozen confirmation phase registration differs")
    checked_phase,checked_registration,checked_requests=confirmation.phase_registration(run,"confirm")
    if checked_phase!=phase or checked_registration!=registration:
        raise ValueError("Confirmation preparation binding differs")
    authorization_path=run/"registrations/confirmation_training.json"
    authorization=confirmation.check_authorization(run,authorization_path)
    if tuple(authorization["selected_pair"])!=PAIR:
        raise ValueError("Selected confirmation pair differs")
    request_path=base.resolve(registration["requests_path"])
    requests=base.read_json(request_path)["rows"]
    if requests!=checked_requests or {q["candidate"] for q in requests}!={8,9,10,11}:
        raise ValueError("Confirmation64 heldout input identities differ")
    failures=[];rows=[];sources=[];all_calls={};matched_receipts=set();matched_calls=set()
    def record(path,kind):
        p=Path(path);s=dict(path=str(p),kind=kind,sha256=sha(p),bytes=p.stat().st_size)
        sources.append(s);return s["sha256"]
    if len(requests)!=64 or len({q["request_id"] for q in requests})!=64:
        failures.append(dict(kind="64 unique requests required"))
    if sha(OLD_AUDITOR)!=OLD_AUDITOR_SHA:
        failures.append(dict(kind="original pure-ledger source hash"))
    record(OLD_AUDITOR,"reused_main_ledger_source")
    record(authorization_path,"confirmation_training_authorization")
    record(REPO/"tools/evaluate_recipe_confirmation.py","confirmation_sampling_adapter")
    record(phase/"preparation_binding.json","confirmation_preparation_binding")
    for label in PAIR:
        record(phase/"sampling"/label/"completed.json","sampling_worker_completion")
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
        registered_model=next(m for m in registration["models"] if m["label"]==label)
        identity_path=phase/"checkpoints"/(label+"_generator_identity.json")
        identity=base.read_json(identity_path);record(identity_path,"generator_identity_metadata_only")
        confirmation.validate_generator_identity(identity,authorization,label,sha(authorization_path))
        done=base.read_json(phase/"sampling"/label/"completed.json")
        done_rows={r["request_id"]:r for r in done.get("rows",[])}
        if done.get("registered")!=64 or len(done.get("rows",[]))!=64 or len(done_rows)!=64:
            failures.append(dict(label=label,kind="sampling worker completion denominator"))
        if binding.get("registration_sha256")!=sha(registration_path):
            failures.append(dict(label=label,kind="checkpoint binding registration hash"))
        if any(binding.get(k)!=registered_model.get(k) for k in ("label","recipe","loader","checkpoint_path")):
            failures.append(dict(label=label,kind="checkpoint binding model metadata"))
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
                errors.extend(receipt_identity_errors(q,r,label,output,done_rows.get(q["request_id"])))
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
                    attempt=r.get("attempt")
                    row["dispatch_attempt"]=attempt
                    if r["new_rollouts"]!=1 or attempt not in ("initial","serial_retry","oom_serial_retry"):
                        errors.append("new_output_attempt")
                    if attempt!="initial":
                        dispatch=phase/"sampling"/label/"serial_retry_dispatch.json"
                        initial_failure=phase/"sampling"/label/"initial_failure.json"
                        if (not dispatch.is_file() or not initial_failure.is_file()
                            or base.read_json(dispatch).get("attempt")!=attempt):
                            errors.append("unregistered serial dispatch attempt")
                    context=dict(label=label,request_id=q["request_id"],attempt=r.get("attempt"),
                        phase="confirm",checkpoint_sha256=r["checkpoint_sha256"],input_sha256=q["input_sha256"])
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
    extra_receipts=[p for p in (phase/"outputs").rglob("receipt.json") if p.resolve() not in matched_receipts]
    extra_calls=[p for p in (phase/"outputs").rglob("calls.jsonl") if p.resolve() not in matched_calls]
    for p in extra_receipts:failures.append(dict(kind="unregistered or unread receipt",path=str(p)))
    for p in extra_calls:
        c,e,h=ledger(p,{})
        all_calls[str(p)]=c;record(p,"orphan_or_unmatched_calls")
        failures.append(dict(kind="orphan or unmatched ledger",path=str(p),errors=e))
    totals=Counter()
    for c in all_calls.values():totals.update(c)
    if len(rows)!=128 or len(expected)!=128:
        failures.append(dict(kind="registered output count",actual=len(rows),expected=128))
    new=sum(r.get("generation_status")=="NEW_EULER50" for r in rows)
    reused=sum(r.get("generation_status")=="VERIFIED_HISTORICAL_RAW_REUSE" for r in rows)
    if new!=128 or reused!=0 or totals["sampling_forward_returned"]!=6400:
        failures.append(dict(kind="expected confirmation execution budget differs",new_outputs=new,reused_outputs=reused,actual_new_forwards=totals["sampling_forward_returned"]))
    groups=[]
    for label in registration["labels"]:
        selected=[r for r in rows if r["label"]==label]
        groups.append(dict(label=label,outputs=len(selected),
            new_outputs=sum(r.get("generation_status")=="NEW_EULER50" for r in selected),
            reused_outputs=sum(r.get("generation_status")=="VERIFIED_HISTORICAL_RAW_REUSE" for r in selected),
            actual_new_rollouts=sum(r.get("rollout_reserved",0) for r in selected),
            actual_successful_new_forwards=sum(r.get("sampling_forward_succeeded",0) for r in selected),
            failures=sum(r["status"]!="PASS" for r in selected)))
    failure_markers=list((phase/"sampling").rglob("*failure.json"))
    retry_markers=list((phase/"sampling").rglob("serial_retry_dispatch.json"))
    dispatch_events=[]
    for p in failure_markers+retry_markers:
        record(p,"failure_or_retry_marker")
        dispatch_events.append(dict(path=str(p),record=base.read_json(p)))
    for label in PAIR:
        folder=phase/"sampling"/label
        dispatches=list(folder.glob("*_dispatch.json"))
        for dp in dispatches: record(dp,"sampling_label_dispatch")
        if len(dispatches)>2 or not (folder/"initial_dispatch.json").is_file():
            failures.append(dict(label=label,kind="label dispatch budget"))
        if (folder/"serial_retry_dispatch.json").exists() and not (folder/"initial_failure.json").exists():
            failures.append(dict(label=label,kind="retry without initial failure"))
    if len(all_calls)!=128 or totals["rollout_attempt_reserved"]!=128 or totals["ROLLOUT_COMPLETE"]!=128:
        failures.append(dict(kind="exact128 unique completed rollout ledgers",actual=len(all_calls),
                             reserved=totals["rollout_attempt_reserved"],completed=totals["ROLLOUT_COMPLETE"]))
    summary=dict(schema="recipe_confirmation_sampling_accounting_v1",status="FAIL" if failures else "PASS",
        scope="Completed confirmation execution ledger and immutable output contracts; no quality/anonymous mapping read",
        outputs=len(rows),new_outputs=new,reused_outputs=reused,actual_new_ledger_files=len(all_calls),
        actual_rollouts_reserved=totals["rollout_attempt_reserved"],actual_rollouts_completed=totals["ROLLOUT_COMPLETE"],
        actual_new_sampling_forwards_reserved=totals["sampling_forward_reserved"],
        actual_new_sampling_forwards_succeeded=totals["sampling_forward_returned"],
        failed_call_events=totals["ERROR"],unfinished_forward_calls=totals["forward_unfinished"],
        unfinished_rollouts=totals["rollout_unfinished"],
        cell_retry_count=sum(max(0,c.get("rollout_attempt_reserved",0)-1) for c in all_calls.values()),
        dispatch_retry_markers=len(retry_markers),failure_markers=len(failure_markers),
        dispatch_event_records=dispatch_events,
        dispatch_retry_is_not_cell_retry=True,
        dispatch_authorization_scope="Markers preserved; resource/retry authorization independently audited by the controller",
        all_actual_output_contracts_passed=all(r["status"]=="PASS" for r in rows),
        expected_forward_budget=6400,expected_forward_budget_matches_actual=totals["sampling_forward_returned"]==6400,
        per_label=groups,exceptions=failures,source_files=sources,
        reused_forward_cost_new_run=0,torch_imported="torch" in sys.modules,
        model_operations=0,new_scoring_calls=0,new_render_calls=0,elapsed_seconds=time.time()-started)
    csv_write(destination/"per_output.csv",rows);csv_write(destination/"per_label.csv",groups)
    csv_write(destination/"exceptions.csv",failures);csv_write(destination/"source_hashes.csv",sources)
    (destination/"accounting.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    md=["# Confirmation sampling accounting","","Status: "+summary["status"],"",
        "Actual ledger totals: %s new rollouts, %s completed rollouts, %s successful sampling forwards; %s reused outputs contribute zero new forwards."%(summary["actual_rollouts_reserved"],summary["actual_rollouts_completed"],summary["actual_new_sampling_forwards_succeeded"],reused),
        "","All 128 registered outputs were checked against actual C/Gaussian input bytes, FP32 RAW and trajectory, PLY, 51 states, nfe50 and BF16/FP32 runtime proof.","",
        "The expected6400 is recorded separately from actual ledger counting. Failure, unfinished call and retry records are retained in accounting.json and exceptions.csv.","",
        "Dispatch failure/retry markers are preserved separately from per-cell attempts. A continued label may start untouched cells, but no cell may have a second rollout.",
        "No model, scoring, rendering, quality summary or anonymous method mapping was used."]
    (destination/"ACCOUNTING.md").write_text("\n".join(md)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_files","per_label","exceptions")}),flush=True)
    return 0 if not failures else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=RUN);p.add_argument("--output",type=Path)
    a=p.parse_args(argv);return audit(a.run,a.output)

if __name__=="__main__":sys.exit(main())
