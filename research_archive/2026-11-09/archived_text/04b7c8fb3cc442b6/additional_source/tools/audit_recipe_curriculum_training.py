#!/usr/bin/env python3
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
        if pos<2*third:
            required=first if pos<third else (.4 if first==.2 else .2)
            if ratio!=required:raise ValueError("Pure-ratio course segment differs")
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



def scan_calls(path, paired, arm, attempt, audit, complete=False):
    """Stream call ledger with per-call pairing and per-step count validation."""
    pending = {}
    steps = defaultdict(Counter)
    counts = Counter()
    malformed = []
    h = hashlib.sha256()
    bytes_read = 0
    if not Path(path).exists():
        return steps, counts, []
    previous_id = 0
    with Path(path).open("rb") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.endswith(b"\n"):
                if complete:
                    malformed.append("incomplete final line")
                break
            h.update(raw); bytes_read += len(raw)
            try:
                row = json.loads(raw)
                event, kind, ident, step = row["event"], row["kind"], int(row["call_id"]), int(row["step"])
                if row.get("attempt_id") != attempt or row.get("segment") != arm:
                    malformed.append("identity at line " + str(line_no))
                key = (ident, kind, step)
                counts[(kind, event)] += 1
                steps[step][kind + ":" + event] += 1
                if event == "CALL_DONE" and kind in ("train_forward", "train_backward"):
                    steps[step][kind + "_sample_" + str(row["sample_index"])] += 1
                if event == "CALL_BEGIN":
                    if ident in pending or ident <= previous_id:
                        malformed.append("non-increasing/duplicate begin " + str(ident))
                    pending[ident] = key
                    previous_id = ident
                elif event in ("CALL_DONE", "CALL_FAILED"):
                    if pending.pop(ident, None) != key:
                        malformed.append("unpaired terminal " + str(ident))
                else:
                    malformed.append("unknown event " + event)
                if kind not in EXPECTED_CALLS:
                    malformed.append("unauthorized call kind " + kind)
                if kind in ("train_forward", "train_backward"):
                    index = int(row["sample_index"])
                    expected = paired[index]
                    if expected["_optimizer_step"] != step or int(row["microbatch"]) != expected["_slot_index"]:
                        malformed.append("slot cursor " + str(index))
                    if any(row.get(k) != expected[k] for k in ("input_bytehash", "label_bytehash")):
                        malformed.append("input hash " + str(index))
                if event == "CALL_DONE" and kind == "optimizer_update":
                    steps[step]["optimizer_time_unix"] = float(row["time_unix"])
            except (ValueError, KeyError, TypeError) as e:
                malformed.append("line %s: %r" % (line_no, e))
    audit.sources.append(dict(path=str(path), sha256=h.hexdigest(), bytes=bytes_read,
                             immutable_expected=complete, snapshot_semantics="stream of complete lines"))
    audit.check(arm, attempt, "call_pairing_and_registered_input_hashes", not malformed, malformed[:15])
    audit.check(arm, attempt, "all_calls_have_terminal_event", not pending,
                "unfinished calls=" + str(len(pending)), pending=bool(pending) and not complete)
    return steps, counts, list(pending)

def compare_exposure(row, planned, paired, arm):
    errors = []
    ints = ("sample_index", "origin_sample_index", "N", "K")
    expected = dict(planned, origin_sample_index=planned["sample_index"],
                    free_coordinates=9 * (planned["N"] - planned["K"]),
                    lambda_value=lam(arm, int(row["step"])))
    for k in ints + ("free_coordinates",):
        if int(row[k]) != int(expected[k]):
            errors.append(k)
    for k in ("uid", "task_id"):
        if row[k] != expected[k]:
            errors.append(k)
    for k in ("ratio", "alpha", "lambda_value"):
        if float(row[k]) != float(expected[k]):
            errors.append(k)
    for k in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key"):
        if row[k] != paired[k]:
            errors.append(k)
    if int(row["free_coordinates"]) != paired["free_coordinate_count"]:
        errors.append("paired_free_coordinate_count")
    if float(row["lambda_value"]) != paired["lambda_value"]:
        errors.append("paired_lambda_value")
    if row["ot_cache_status"] != "HIT":
        errors.append("OT_not_cache_hit")
    if not math.isfinite(float(row["t"])) or not 0 <= float(row["t"]) <= 1:
        errors.append("t")
    if int(row["step"]) != planned["optimizer_step"]:
        errors.append("step")
    return errors

def audit_arm(run, arm, plan, authorization, audit, out_rows, task_families):
    folder = run / "training" / arm
    completed_path = folder / "completed.json"
    completed = read(completed_path) if completed_path.exists() else {}
    is_complete = completed.get("status") == "COMPLETE"
    paired_path = run / "streams/paired_inputs" / (arm + "__H_ALL.json")
    audit.record(paired_path)
    pair_doc = read(paired_path)
    r2_path=run/"streams/paired_inputs"/(BASELINE+".json")
    audit.record(r2_path);r2_doc=read(r2_path)
    records=paired_contract(plan,arm,pair_doc,r2_doc)
    paired={r["sample_index"]:dict(r,_optimizer_step=pos//8+1,_slot_index=pos%8)
            for pos,r in enumerate(pair_doc["records"])}
    planned=planned_groups(arm,records,task_families)
    total = int(authorization["updates_per_recipe"])
    audit.check(arm, "", "registered_48000_distinct_slots", len(paired) == total * 8 == len(records))
    audit.check(arm, "", "paired_plan_sha", pair_doc["plan_sha256"] == file_hash(run / "streams/recipe_plan.json"))
    audit.check(arm, "", "complete_receipt", is_complete, completed.get("status", "not completed"), pending=not is_complete)
    by_index = {int(r["sample_index"]): r for r in records}
    attempt_results = []
    candidates = defaultdict(list)
    all_exposures = []
    group_actual = defaultdict(Counter)
    group_planned = {}
    for row in planned:
        if row["recipe"] == arm:
            group_planned[(row["uid"], float(row["ratio"]), row["condition_family"])] = row
    event_switch = []
    optimizer_step_counts = Counter()
    for attempt_dir in sorted((folder / "attempts").glob("*")) if (folder / "attempts").exists() else []:
        if not attempt_dir.is_dir():
            continue
        attempt = attempt_dir.name
        final_path = attempt_dir / "final_status.json"
        final = read(final_path) if final_path.exists() else {}
        ended = bool(final)
        audit.record(final_path, stable=ended)
        logs = csv_rows(audit, attempt_dir / "training_log.csv", ended)
        exposure = csv_rows(audit, attempt_dir / "exposure.csv", ended)
        events_text = audit.snapshot(attempt_dir / "events.jsonl", ended)
        events = [json.loads(line) for line in events_text.splitlines() if line]
        event_by_step = defaultdict(list)
        for e in events:
            if "step" in e:
                event_by_step[int(e["step"])].append(e)
            if e.get("step") in (2000, 2001, 4000, 4001) or e.get("event") == "RESUME":
                event_switch.append(dict(recipe=arm, attempt_id=attempt, **e))
        steps, counts, unfinished = scan_calls(attempt_dir / "calls.jsonl", paired, arm, attempt, audit, ended)
        success_steps = {s for s, c in steps.items() if c["optimizer_update:CALL_DONE"] > 0}
        for successful_step in success_steps:
            optimizer_step_counts[successful_step] += steps[successful_step]["optimizer_update:CALL_DONE"]
        exp_by_step = defaultdict(list)
        exposure_errors = []
        for row in exposure:
            index = int(row["sample_index"]); step = int(row["step"])
            errs = compare_exposure(row, by_index[index], paired[index], arm)
            if row["recipe"] != arm or row["attempt_id"] != attempt:
                errs.append("recipe_or_attempt")
            if errs:
                exposure_errors.append(dict(sample_index=index, errors=errs))
            row = dict(row, condition_family=task_families[row["task_id"]],
                       paired_hash_status="FAIL" if errs else "PASS",
                       optimizer_update_succeeded=step in success_steps,
                       forward_succeeded=bool(steps[step]["train_forward_sample_" + str(index)]),
                       backward_succeeded=bool(steps[step]["train_backward_sample_" + str(index)]),
                       final_logical_trajectory=False)
            exp_by_step[step].append(row); all_exposures.append(row)
        audit.check(arm, attempt, "exposure_vs_plan_and_paired_bytes", not exposure_errors,
                    exposure_errors[:15])
        log_steps = [int(r["step"]) for r in logs]
        audit.check(arm, attempt, "unique_training_log_steps", len(log_steps) == len(set(log_steps)))
        for row in logs:
            step = int(row["step"]); ex = sorted(exp_by_step[step], key=lambda r: by_index[int(r["sample_index"])]["slot_index"])
            errs = []
            if len(ex) != 8 or len({r["uid"] for r in ex}) != 8:
                errs.append("eight_distinct_parents")
            expected_indices = [r["sample_index"] for r in records[(step-1)*8:step*8]]
            if [int(r["sample_index"]) for r in ex] != expected_indices:
                errs.append("sample_slots")
            expected_fields = dict(free_coordinate_denominator=sum(int(r["free_coordinates"]) for r in ex),
                ratio20_samples=sum(float(r["ratio"]) == .2 for r in ex),
                ratio40_samples=sum(float(r["ratio"]) == .4 for r in ex),
                free20_coordinates=sum(int(r["free_coordinates"]) for r in ex if float(r["ratio"]) == .2),
                free40_coordinates=sum(int(r["free_coordinates"]) for r in ex if float(r["ratio"]) == .4),
                generator_adamw_step=step)
            for key, expected in expected_fields.items():
                if int(row[key]) != expected: errs.append(key)
            if float(row["lambda_value"]) != lam(arm, step): errs.append("lambda")
            for key, samplekey in BATCH_HASH.items():
                if row[key] != json_hash([r[samplekey] for r in ex]): errs.append(key)
            for kind, expected in EXPECTED_CALLS.items():
                if steps[step][kind + ":CALL_DONE"] != expected: errs.append(kind + "_calls")
            if not all(math.isfinite(float(row[k])) for k in ("FM", "total_loss", "generator_gradient_norm")):
                errs.append("nonfinite")
            if float(row["FM"]) != float(row["total_loss"]): errs.append("unexpected_aux_loss")
            done = [e for e in event_by_step[step] if e["event"] == "UPDATE_DONE"]
            inputs = [e for e in event_by_step[step] if e["event"] == "INPUT_DONE"]
            if len(done) != 1 or done[0].get("generator_adamw_step") != step: errs.append("UPDATE_DONE_cursor")
            if len(inputs) != 1 or any(inputs[0].get(k) != row[k] for k in BATCH_HASH): errs.append("INPUT_DONE_hash")
            row = dict(row, audit_status="FAIL" if errs else "PASS", audit_errors=";".join(errs),
                       final_logical_trajectory=False)
            out_rows["updates"].append(row)
            candidates[step].append((steps[step].get("optimizer_time_unix", 0), attempt, row, ex))
        bad = [r for r in out_rows["updates"] if r["recipe"] == arm and r["attempt_id"] == attempt and r["audit_status"] != "PASS"]
        audit.check(arm, attempt, "all_update_rows_vs_calls_inputs_denominators", not bad, len(bad))
        missing_logs = success_steps - set(log_steps)
        audit.check(arm, attempt, "successful_optimizer_calls_have_log_rows", not missing_logs,
                    sorted(missing_logs)[:15], pending=bool(missing_logs) and not ended)
        if ended:
            reported = int(final.get("physical_optimizer_updates_this_attempt", -1))
            audit.check(arm, attempt, "final_status_optimizer_count", reported == counts[("optimizer_update", "CALL_DONE")])
            for kind in EXPECTED_CALLS:
                audit.check(arm, attempt, "final_status_calls_" + kind,
                    int(final.get("successful_calls", {}).get(kind, 0)) == counts[(kind, "CALL_DONE")])
        resume = [e for e in events if e["event"] == "RESUME"]
        audit.check(arm, attempt, "resume_preserves_optimizer", all(e.get("optimizer_reset") is False for e in resume))
        first_step = int(resume[0]["step"]) if resume else 0
        audit.check(arm, attempt, "attempt_steps_contiguous_from_resume",
                    log_steps == list(range(first_step + 1, max(log_steps, default=first_step) + 1)))
        audit.check(arm, attempt, "one_resume_at_most", len(resume) <= 1)
        checkpoint_events = [e for e in events if e["event"] == "CHECKPOINT_DONE"]
        expected_checkpoint_steps = {step for step in log_steps if step % 100 == 0}
        audit.check(arm, attempt, "periodic_checkpoint_receipts",
                    expected_checkpoint_steps.issubset({e["step"] for e in checkpoint_events}),
                    pending=not ended)
        audit.check(arm, attempt, "checkpoint_receipt_cursors",
                    all(e["step"] == e["checkpoint"]["global_step"] for e in checkpoint_events))
        for kind in sorted({k for k, event in counts}):
            out_rows["calls"].append(dict(recipe=arm, attempt_id=attempt, kind=kind,
                begun=counts[(kind, "CALL_BEGIN")], succeeded=counts[(kind, "CALL_DONE")],
                failed=counts[(kind, "CALL_FAILED")],
                unfinished=counts[(kind, "CALL_BEGIN")] - counts[(kind, "CALL_DONE")] - counts[(kind, "CALL_FAILED")]))
        attempt_result = dict(recipe=arm, attempt_id=attempt, status=final.get("status", "RUNNING_OR_NOT_FINALIZED"),
            logical_first_logged_step=min(log_steps, default=None), logical_last_logged_step=max(log_steps, default=None),
            successful_physical_optimizer_updates=counts[("optimizer_update", "CALL_DONE")],
            loaded_sample_rows=len(exposure), successful_forward_calls=counts[("train_forward", "CALL_DONE")],
            successful_backward_calls=counts[("train_backward", "CALL_DONE")],
            failed_calls=sum(v for (k, e), v in counts.items() if e == "CALL_FAILED"),
            unfinished_calls=len(unfinished), resume_count=len(resume),
            resumed_from_step=resume[0]["step"] if resume else 0,
            wall_seconds=final.get("wall_seconds"),
            input_seconds_sum=sum(float(r["input_seconds"]) for r in logs),
            update_seconds_sum=sum(float(r["update_seconds"]) for r in logs),
            forward_seconds_sum=sum(float(r["forward_seconds"]) for r in logs),
            backward_seconds_sum=sum(float(r["backward_seconds"]) for r in logs),
            optimizer_seconds_sum=sum(float(r["optimizer_seconds"]) for r in logs),
            cuda_peak_allocated_bytes=max((int(r["cuda_peak_allocated_bytes"]) for r in logs), default=0),
            cuda_peak_reserved_bytes=max((int(r["cuda_peak_reserved_bytes"]) for r in logs), default=0))
        attempt_results.append(attempt_result); out_rows["attempts"].append(attempt_result)
    selected = {}
    for step, values in candidates.items():
        selected[step] = max(values, key=lambda v: v[0])
        _, attempt, row, ex = selected[step]
        row["final_logical_trajectory"] = True
        for sample in ex:
            sample["final_logical_trajectory"] = True
            key = group_key(sample, task_families)
            c = group_actual[key]
            c["samples"] += 1; c["free"] += int(sample["free_coordinates"])
            c["clean"] += float(sample["alpha"]) == 1.
            c["augmented"] += float(sample["alpha"]) != 1.
            c["segment" + str((int(sample["step"]) - 1) // max(1,total//3))] += 1
    for sample in all_exposures:
        key = group_key(sample, task_families)
        group_actual[key]["loaded"] += 1
        if sample["forward_succeeded"]:
            group_actual[key]["forward_samples"] += 1
            group_actual[key]["forward_free"] += int(sample["free_coordinates"])
        if sample["backward_succeeded"]:
            group_actual[key]["backward_samples"] += 1
            group_actual[key]["backward_free"] += int(sample["free_coordinates"])
        if sample["optimizer_update_succeeded"]:
            group_actual[key]["physical_samples"] += 1
            group_actual[key]["physical_free"] += int(sample["free_coordinates"])
    out_rows["exposures"].extend(all_exposures)
    for key, planned in group_planned.items():
        actual = group_actual[key]
        row = dict(recipe=arm, uid=key[0], ratio=key[1], condition_family=key[2],
            planned_samples=int(planned["planned_samples"]), actual_unique_successful_samples=actual["samples"],
            loaded_sample_rows=actual["loaded"], physical_optimizer_sample_rows=actual["physical_samples"],
            planned_free_scalar_supervision=int(planned["planned_free_scalar_supervision"]),
            actual_unique_free_scalar_supervision=actual["free"], physical_optimizer_free_scalar_supervision=actual["physical_free"],
            successful_forward_sample_rows=actual["forward_samples"], successful_backward_sample_rows=actual["backward_samples"],
            successful_forward_free_scalars=actual["forward_free"], successful_backward_free_scalars=actual["backward_free"],
            planned_clean_samples=int(planned["planned_clean_samples"]), actual_clean_samples=actual["clean"],
            planned_augmented_samples=int(planned["planned_augmented_samples"]), actual_augmented_samples=actual["augmented"])
        expected_equal = row["planned_samples"] == row["actual_unique_successful_samples"] and row["planned_free_scalar_supervision"] == row["actual_unique_free_scalar_supervision"] and row["planned_clean_samples"] == row["actual_clean_samples"] and row["planned_augmented_samples"] == row["actual_augmented_samples"]
        for i, label in enumerate(("1_2000", "2001_4000", "4001_6000")):
            row["planned_step_" + label + "_samples"] = int(planned["planned_step_" + label + "_samples"])
            row["actual_step_" + label + "_samples"] = actual["segment" + str(i)]
            expected_equal &= row["planned_step_" + label + "_samples"] == row["actual_step_" + label + "_samples"]
        row["status"] = "PASS" if expected_equal else "FAIL" if is_complete else "PENDING"
        out_rows["groups"].append(row)
    audit.check(arm, "", "logical_steps_cover_1_to_6000", set(selected) == set(range(1, total + 1)),
                "unique=" + str(len(selected)), pending=not is_complete)
    audit.check(arm, "", "actual_parent_family_ratio_free_counts_match_full_plan",
                all(r["status"] == "PASS" for r in out_rows["groups"] if r["recipe"] == arm), pending=not is_complete)
    boundary = []
    for step in (2000, 2001, 4000, 4001):
        selected_row = selected.get(step)
        boundary.append(dict(step=step, expected_lambda=lam(arm, step),
            observed_lambda=float(selected_row[2]["lambda_value"]) if selected_row else None,
            generator_adamw_step=int(selected_row[2]["generator_adamw_step"]) if selected_row else None,
            attempt_id=selected_row[1] if selected_row else None,
            observed_ratio20_samples=int(selected_row[2]["ratio20_samples"]) if selected_row else None,
            observed_ratio40_samples=int(selected_row[2]["ratio40_samples"]) if selected_row else None,
            status="PASS" if selected_row else "PENDING"))
    for step in range(2, total + 1):
        if step in selected and step - 1 in selected:
            if int(selected[step][2]["generator_adamw_step"]) != int(selected[step - 1][2]["generator_adamw_step"]) + 1:
                audit.check(arm, "", "optimizer_cursor_continuity", False, step)
    receipts = []
    if is_complete:
        audit.record(completed_path)
        expected_ids = dict(plan_sha256=file_hash(run / "streams/recipe_plan.json"),
            paired_inputs_sha256=file_hash(paired_path),
            authorization_sha256=file_hash(run / "registrations/curriculum_training.json"),
            data_manifest_sha256=plan["data_manifest_sha256"])
        audit.check(arm, "", "completed_input_identities", all(completed.get("identities", {}).get(k) == v for k, v in expected_ids.items()))
        audit.check(arm, "", "completed_step_6000", completed.get("step") == total)
        for kind, receipt in (("checkpoint", completed.get("checkpoint", {})),
                              ("inference_export", completed.get("inference_export", {}))):
            endpoint = run / ("checkpoints" if kind == "checkpoint" else "inference") / (arm + ("_step6000.pt" if kind == "checkpoint" else "_generator_step6000.pt"))
            sidecar = endpoint.with_suffix(".json")
            if kind == "checkpoint":
                audit.record(sidecar)
                side = read(sidecar) if sidecar.exists() else {}
                audit.check(arm, "", kind + "_receipt_sidecar", side == receipt)
            # The worker stores export metadata in completed.json only; no export sidecar is promised.
            audit.check(arm, "", kind + "_receipt_cursor_identity",
                receipt.get("global_step") == total and receipt.get("recipe", receipt.get("segment")) == arm)
            receipts.append(verify_file_receipt(endpoint, receipt, audit, arm, kind))
        if len(receipts) == 2:
            audit.check(arm, "", "export_same_generator_receipt", receipts[0].get("generator_state_sha256_receipt") == receipts[1].get("generator_state_sha256_receipt"))
    physical = sum(r["successful_physical_optimizer_updates"] for r in attempt_results)
    replayed = sum(max(0, n - 1) for n in optimizer_step_counts.values())
    audit.check(arm, "", "at_most_one_recovery_attempt", len(attempt_results) <= 2)
    audit.check(arm, "", "replay_completed_updates_within_99", replayed <= 99)
    audit.check(arm, "", "successful_optimizer_steps_have_log_evidence", set(optimizer_step_counts) == set(selected), pending=not is_complete)
    return dict(recipe=arm, completion_state="COMPLETE" if is_complete else "PENDING",
        unique_successful_updates=len(selected), physical_successful_optimizer_updates=physical,
        successful_replayed_updates=replayed, unique_successful_optimizer_call_steps=len(optimizer_step_counts), attempts=attempt_results,
        failed_attempts=sum(r["status"] in ("OOM", "BLOCKED", "INTERRUPTED_SAFE") for r in attempt_results),
        curriculum_boundaries=boundary, switch_and_resume_events=event_switch, endpoint_files=receipts)

def resources_and_dispatch(run, audit, out_rows):
    rows = audit.snapshot(run / "audit/controller.jsonl")
    events = [json.loads(x) for x in rows.splitlines() if x]
    starts = {}; intervals = []; pending_starts = []
    for e in events:
        if e.get("phase") == "training" and e.get("name") in ARMS:
            key = (e.get("wave"), e["name"], e.get("attempt"))
            if e["event"] == "WORKER_START":
                starts[key] = e
            if e["event"] == "WORKER_END":
                s = starts.pop(key, None)
                intervals.append(dict(recipe=e["name"], attempt_id=e["attempt"], wave=e.get("wave"),
                    start_unix=s.get("time_unix") if s else None, end_unix=e["time_unix"],
                    wall_seconds=e["wall_seconds"], returncode=e["returncode"],
                    workers_at_start=s.get("worker_count") if s else None))
        if e.get("event") == "RESOURCES" and any(arm in e.get("active", []) for arm in ARMS):
            s = e["snapshot"]; g = s.get("gpu") or {}
            out_rows["resources"].append(dict(time_unix=e["time_unix"], reason=e.get("reason"),
                active=";".join(e.get("active", [])), gpu_name=g.get("name"),
                gpu_total_mib=g.get("total_mib"), gpu_used_mib=g.get("used_mib"),
                gpu_free_mib=g.get("free_mib"), gpu_utilization=g.get("utilization"),
                cpu_cores=s.get("cpu_cores"), available_ram_bytes=s.get("ram", {}).get("MemAvailable"),
                loadavg1=s.get("loadavg", [None])[0]))
    for key, s in starts.items():
        pending_starts.append(dict(recipe=key[1], attempt_id=key[2], start_unix=s["time_unix"]))
    out_rows["dispatch"].extend(intervals)
    receipts = []
    for p in sorted((run / "audit").glob("dispatch_*.json")):
        d = read(p)
        if d.get("wave") and any(arm in d.get("completed", []) for arm in ARMS):
            audit.record(p); receipts.append(dict(path=str(p), **d))
    return dict(intervals=intervals, running_or_unclosed=pending_starts, phase_receipts=receipts,
                oom_downgrade_events=sum(e["event"] == "OOM_DOWNGRADE" for e in events),
                recovery_events=[e for e in events if e["event"] == "RECOVERY_ACCOUNTING"],
                timing_note="Concurrent elapsed wall times include contention. Summed worker wall/update times are not isolated GPU benchmarks or total phase elapsed time.",
                memory_note="PyTorch peaks cover reset-per-effective-update forward/backward/optimizer, exclude context and desktop. Device-used samples include all GPU processes; neither is a per-process OS VRAM measurement.")

def run_audit(run,destination=None):
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
    (target/"accounting.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    lines=["# Curriculum training accounting","","Status: "+status,"",
           "Exactly two H_ALL curricula; R2 contributes zero new updates. No model/OT/checkpoint deserialization.","",
           "| Recipe | Unique | Physical | Completed replay |","|---|---:|---:|---:|"]
    lines += ["| %s | %s | %s | %s |"%(r["recipe"],r["unique_successful_updates"],r["physical_successful_optimizer_updates"],r["successful_replayed_updates"]) for r in results]
    lines += ["",*result["evidence_limits"],"",dispatch["timing_note"],"",dispatch["memory_note"]]
    lines += ["",*["- "+str(c) for c in audit.checks if c["status"]!="PASS"]]
    (target/"ACCOUNTING.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    return result,target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,default=DEFAULT);parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv);result,target=run_audit(args.run,args.output)
    print(json.dumps(dict(status=result["status"],output=str(target) if target else None,
                         scientific_calls=0,missing=result.get("missing_completed_files",[]))),flush=True)
    return 0 if result["status"]=="PASS" else 2 if result["status"]=="PENDING" else 1


if __name__=="__main__":raise SystemExit(main())

