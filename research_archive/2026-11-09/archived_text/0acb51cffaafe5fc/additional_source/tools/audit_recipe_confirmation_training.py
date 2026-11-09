#!/usr/bin/env python3
"""Stdlib-only audit of the final authorized independent-input confirmation.

No tensor deserialization, training import, model/OT call, or mutable latest.pt read.
The call/exposure accounting loop is independently adapted from the preserved
curriculum auditor; its read-only helpers are reused without changing globals.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import importlib.util
import json
import hashlib
import math
from pathlib import Path
import sys
import time

_spec = importlib.util.spec_from_file_location("_confirmation_course_helpers", Path(__file__).with_name("audit_recipe_curriculum_training.py"))
course = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(course)
base = course.base
Audit = base.Audit
file_hash, json_hash, read = base.file_hash, base.json_hash, base.read
csv_rows, write_csv = base.csv_rows, base.write_csv
verify_file_receipt, condition_family, group_key = base.verify_file_receipt, base.condition_family, base.group_key
BATCH_HASH, EXPECTED_CALLS = base.BATCH_HASH, base.EXPECTED_CALLS
scan_calls, planned_groups = course.scan_calls, course.planned_groups
ARMS = ("C20_40_MIX", "R2_MIX_H_ALL")
BASELINE = ARMS[1]
DEFAULT = base.DEFAULT
VERSION = "chair_recipe_confirmation_training_accounting_v1"
NAMESPACE = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1/confirmation/input_repeat1"
CONTRACT = "original_preOT_cost_scale__FP32_final_target_x2__raw_iid_noise_slots_v1"
REPO = Path(__file__).resolve().parents[1]
INITIAL_SHA = "792ad8d6249ed4ccd1b5c5a81dfba0e527fc61911be96144729f8a2fc7f4811a"
OFFICIAL_SHA = "bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d"


def lam(arm, step):
    if arm not in ARMS or type(step) is not int or step < 1:
        raise ValueError("Unregistered confirmation recipe/step")
    return .25


def confirmation_records(plan, arm):
    if arm not in ARMS:
        raise ValueError("Only the selected confirmation pair is allowed")
    if arm == ARMS[0]:
        return course.curriculum_records(plan, arm)
    raw = plan["mixed_records"]
    if len(raw) % 24:
        raise ValueError("MIX stream does not contain complete segments")
    rows = []
    for index, r in enumerate(raw):
        if r["sample_index"] != index or r.get("origin_sample_index", index) != index:
            raise ValueError("MIX stable origin differs")
        rows.append(dict(r, origin_sample_index=index, step=index//8+1, optimizer_step=index//8+1,
                         effective_batch_index=index//8, slot_index=index%8))
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        if len({r["uid"] for r in batch}) != 8 or any(sum(r["ratio"] == ratio and (r["alpha"] == 1) == clean for r in batch) != 2 for ratio in (.2, .4) for clean in (False, True)):
            raise ValueError("MIX needs eight parents and two clean/two augmented per ratio")
    return rows


def paired_contract(plan, arm, pair_doc, control_doc):
    rows = confirmation_records(plan, arm)
    for label, doc in ((arm, pair_doc), (BASELINE, control_doc)):
        if (doc.get("schema") != "meshflow_recipe_confirmation_paired_inputs_v1" or doc.get("recipe") != label
            or doc.get("namespace") != plan["namespace"] or doc.get("selected_pair") != list(ARMS)
            or doc.get("hybrid_schedule") != "all" or doc.get("historical_paired_inputs_reused") is not False):
            raise ValueError("Independent paired-input identity differs")
    source = control_doc["records"]
    if len(source) != len(rows) or [r["sample_index"] for r in source] != list(range(len(rows))):
        raise ValueError("Control needs the complete ordered multiset")
    # The new data writer registers current step as well as immutable origin.
    expected = [dict(source[r["sample_index"]], step=r["step"]) for r in rows]
    if pair_doc["records"] != expected:
        raise ValueError("Selected recipes do not share exact input bytes under their registered order")
    for row, pair in zip(rows, pair_doc["records"]):
        if float(pair["lambda_value"]) != .25 or pair["free_coordinate_count"] != 9*(row["N"]-row["K"]):
            raise ValueError("H_ALL/free scalar paired denominator differs")
        for key in ("sample_index", "origin_sample_index", "task_id", "uid", "N", "K", "ratio", "alpha", "step"):
            if pair.get(key) != row[key]:
                raise ValueError("Paired plan field differs: " + key)
    return rows


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

def audit_arm(run, arm, plan, authorization, audit, out_rows, task_families, authorization_path):
    folder = run / "training" / arm
    completed_path = folder / "completed.json"
    completed = read(completed_path) if completed_path.exists() else {}
    is_complete = completed.get("status") == "COMPLETE"
    paired_path = run / "streams/paired_inputs" / (arm + ".json")
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
    audit.check(arm, "", "paired_plan_sha", pair_doc["plan_sha256"] == file_hash(run / "streams/confirmation_plan.json"))
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
        check_sample_losses(attempt_dir / "diagnostics.jsonl", logs, exp_by_step, arm, attempt, audit, ended)
        log_steps = [int(r["step"]) for r in logs]
        audit.check(arm, attempt, "unique_training_log_steps", len(log_steps) == len(set(log_steps)))
        for row in logs:
            step = int(row["step"]); ex = sorted(exp_by_step[step], key=lambda r: by_index[int(r["sample_index"])]["slot_index"])
            errs = []
            if len(ex) != 8 or len({r["uid"] for r in ex}) != 8:
                errs.append("eight_distinct_parents")
            for sample in ex:
                for kind in ("train_forward", "train_backward"):
                    if steps[step][kind+"_sample_"+str(sample["sample_index"])] != 1:
                        errs.append(kind+"_per_registered_sample_count")
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
        expected_ids = dict(plan_sha256=file_hash(run / "streams/confirmation_plan.json"),
            paired_inputs_sha256=file_hash(paired_path),
            authorization_sha256=file_hash(authorization_path),
            data_manifest_sha256=plan["data_manifest_sha256"],
            selected_pair=list(ARMS), confirmation_namespace=plan["namespace"])
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
        if e.get("phase") == "training" and e.get("name") in ARMS and "confirm" in str(e.get("wave", "")):
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
    windows = [(r["start_unix"], r["end_unix"]) for r in intervals if r["start_unix"] is not None]
    windows += [(r["start_unix"], math.inf) for r in pending_starts]
    out_rows["resources"] = [r for r in out_rows["resources"] if any(start <= r["time_unix"] <= end for start, end in windows)]
    receipts = []
    for p in sorted((run / "audit").glob("dispatch_*.json")):
        d = read(p)
        if "confirm" in str(d.get("wave", "")) and "training" in str(d.get("wave", "")) and any(arm in d.get("completed", []) for arm in ARMS):
            audit.record(p); receipts.append(dict(path=str(p), **d))
    return dict(intervals=intervals, running_or_unclosed=pending_starts, phase_receipts=receipts,
                oom_downgrade_events=sum(e["event"] == "OOM_DOWNGRADE" for e in events),
                recovery_events=[e for e in events if e["event"] == "RECOVERY_ACCOUNTING"],
                timing_note="Concurrent elapsed wall times include contention. Summed worker wall/update times are not isolated GPU benchmarks or total phase elapsed time.",
                memory_note="PyTorch peaks cover reset-per-effective-update forward/backward/optimizer, exclude context and desktop. Device-used samples include all GPU processes; neither is a per-process OS VRAM measurement.")

def check_sample_losses(path, logs, exposure_by_step, arm, attempt, audit, complete):
    rows = [json.loads(line) for line in audit.snapshot(path, complete).splitlines() if line]
    by = defaultdict(list)
    for row in rows:
        by[int(row["step"])].append(row)
    problems = []
    for log in logs:
        step = int(log["step"]); matches = by[step]
        if len(matches) != 1:
            problems.append((step, "missing/duplicate sample diagnostics")); continue
        record = matches[0]; values = record.get("values", [])
        ex = exposure_by_step[step]
        denominator = sum(int(r["free_coordinates"]) for r in ex)
        if record.get("event") != "SAMPLE_LOSSES" or record.get("lambda_value") != .25:
            problems.append((step, "diagnostic event/lambda"))
        if len(values) != 8 or Counter(v.get("sample_index") for v in values) != Counter(int(r["sample_index"]) for r in ex):
            problems.append((step, "sample diagnostic identities"))
        for value in values:
            if (value.get("batch_free_denominator") != denominator or value.get("raw_align") is not None
                or value.get("alignment_eligible") is not False
                or not math.isfinite(float(value.get("FM_contribution", float("nan"))))):
                problems.append((step, "free scalar denominator/auxiliary/nonfinite"))
        if not math.isclose(sum(float(v.get("FM_contribution", float("nan"))) for v in values), float(log["FM"]), rel_tol=1e-10, abs_tol=1e-10):
            problems.append((step, "sum FM contributions differs"))
    audit.check(arm, attempt, "sample_FM_contributions_and_effective_free_scalar_denominator", not problems, problems[:15])


def validate_registration(auth, plan, run, confirmation):
    expected = dict(schema="chair_recipe_confirmation_training_authorization_v1", experiment="CHAIR_CONDITION_RECIPE_FACTORIAL_V1",
        stage="independent_confirmation", selected_pair=list(ARMS), authorized_recipes=list(ARMS),
        hybrid_schedules={a:"all" for a in ARMS}, namespace=NAMESPACE,
        out=str(Path(confirmation).resolve()), experiment_root=str(Path(run).resolve()),
        updates_per_recipe=6000, effective_batch=8, microbatch=1, max_training_recipes=2)
    for key, value in expected.items():
        if auth.get(key) != value: raise ValueError("Confirmation authorization differs: " + key)
    expected_plan = dict(schema="meshflow_recipe_confirmation_plan_v1", namespace=NAMESPACE,
        selected_pair=list(ARMS), hybrid_schedules=expected["hybrid_schedules"], total_updates=6000,
        effective_batch_count=6000, batch_size=8, microbatch_size=1, sample_count=48000,
        independent_training_input_stream=True, independent_model_initialization=False, Geo_seed=1010,
        no_historical_pair_or_OT_reuse=True, coupling_contract=CONTRACT,
        FM_normalization="Unchanged effective-batch valid free scalar count")
    for key, value in expected_plan.items():
        if plan.get(key) != value: raise ValueError("Confirmation plan differs: " + key)
    if Path(confirmation).resolve() != Path(run).resolve()/"confirmation":
        raise ValueError("This final repeat requires the registered confirmation child directory")
    if any(key in plan for key in ("base_stream_root", "base_plan_sha256", "base_paired_inputs_sha256")):
        raise ValueError("Historical mother-stream substitution is forbidden")
    if len(plan["mixed_records"]) != 48000 or len(plan["base_records"]) != 48000 or len(set(plan["eligible_parents"])) != 26:
        raise ValueError("48000-slot / 26-parent confirmation plan required")


def derived_seed(namespace, domain, *identities):
    raw = json.dumps([namespace, "confirmation_base/"+domain, *identities], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def independent_input_checks(plan, historical_plan, pairs, historical_pairs, audit):
    audit.check("", "", "new_namespace_not_historical", plan["namespace"] == NAMESPACE and plan["namespace"] != historical_plan["namespace"])
    domains = ("parent_schedule", "task40", "alpha", "alpha_slot_shuffle", "gaussian", "epsilon", "permutation", "time")
    audit.check("", "", "namespace_derived_base_domains", plan.get("base_domain_seeds") == {d:derived_seed(plan["namespace"], d) for d in domains})
    errors = []
    for row in plan["mixed_records"]:
        for field, domain in (("gaussian_seed", "gaussian"), ("epsilon_seed", "epsilon"), ("permutation_seed", "permutation"), ("t_seed", "time")):
            if row.get(field) != derived_seed(plan["namespace"], domain, row["sample_index"]):
                errors.append((row["sample_index"], field))
    audit.check("", "", "all_origin_slot_seeds_derive_from_new_namespace", not errors, errors[:15])
    audit.check("", "", "same_training_parent_support", set(plan["eligible_parents"]) == set(historical_plan["eligible_parents"]))
    for key in ("shared_input_sha256", "input_bytehash", "ot_cache_key"):
        old = {r[key] for r in historical_pairs["records"]}
        new = {r[key] for r in pairs[BASELINE]["records"]}
        audit.check("", "", "no_historical_actual_" + key + "_reuse", not (old & new), "shared identities="+str(len(old & new)))


def expected_training_sources():
    """Read only the trainer's literal source list without importing torch."""
    path = REPO/"src/meshflow_control/training/recipe_confirmation.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_identity")
    value = next(n.value for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "relatives" for t in n.targets))
    return {relative:file_hash(REPO/"src/meshflow_control"/relative) for relative in ast.literal_eval(value)}


def verify_endpoint_metadata(confirm, arm, plan, auth_path, audit, sources):
    folder = confirm/"training"/arm; completed = read(folder/"completed.json")
    initialization = read(folder/"initialization_audit.json"); audit.record(folder/"initialization_audit.json")
    required = dict(selected_state="ema", official_file_sha256=OFFICIAL_SHA,
        generator_initial_sha256=INITIAL_SHA, role_nonzero=0, geo_exit_nonzero=0, Geo_seed=1010,
        fresh_AdamW=True, new_optimizer_step=0, teacher_loaded=False, no_auxiliary_head_constructed=True)
    audit.check(arm, "", "same_official_Geo1010_fresh_AdamW", all(initialization.get(k) == v for k,v in required.items()))
    audit.check(arm, "", "completed_initialization_lineage", completed.get("lineage") == initialization)
    audit.check(arm, "", "completed_identity_recipe_namespace", completed.get("recipe") == arm and completed.get("segment") == arm
        and completed.get("namespace") == plan["namespace"] and completed.get("hybrid_schedule") == "all"
        and completed.get("teacher_forwards") == 0 and completed.get("head_forwards") == 0 and completed.get("alignment_active") is False)
    ids = completed.get("identities", {})
    audit.check(arm, "", "all_transitive_training_sources", ids.get("training_sources") == sources)
    audit.check(arm, "", "stream_implementation_identity", ids.get("stream_implementation_sha256") == json_hash(plan["implementation_sources"]))
    checkpoint, exported = completed["checkpoint"], completed["inference_export"]
    audit.check(arm, "", "confirmation_checkpoint_schema", checkpoint.get("schema") == "chair_recipe_confirmation_training_v1")
    audit.check(arm, "", "confirmation_generator_schema", exported.get("schema") == "chair_recipe_confirmation_generator_v1")
    audit.check(arm, "", "same_protocol_hash_all_receipts", bool(completed.get("protocol_sha256")) and
        completed["protocol_sha256"] == checkpoint.get("protocol_sha256") == exported.get("protocol_sha256"))
    for key in ("optimizer_state_sha256", "rng_sha256", "stream_state_sha256"):
        audit.check(arm, "", "checkpoint_"+key+"_present", isinstance(checkpoint.get(key), str) and len(checkpoint[key]) == 64)
    attempt = folder/"attempts"/completed["attempt_id"]/"final_status.json"
    audit.record(attempt); audit.check(arm, "", "completed_equals_final_attempt", read(attempt) == completed)
    for p in sorted((folder/"attempts").glob("*/final_status.json")):
        row = read(p)
        audit.check(arm, p.parent.name, "every_attempt_frozen_identity_and_lineage", row.get("identities") == ids and row.get("lineage") == initialization)
    return initialization


def final_pair_checks(plan, tables, audit):
    final = {}
    for arm in ARMS:
        records = confirmation_records(plan, arm)
        positions = {(r["step"], r["sample_index"]): pos for pos,r in enumerate(records)}
        rows = [r for r in tables["exposures"] if r["recipe"] == arm and r["final_logical_trajectory"]]
        rows.sort(key=lambda r:positions[int(r["step"]), int(r["sample_index"])])
        final[arm] = rows
        audit.check(arm, "", "actual_final_exact_registered_origin_order", [int(r["sample_index"]) for r in rows] == [r["sample_index"] for r in records])
        audit.check(arm, "", "actual_final_same_complete_multiset", sorted(int(r["sample_index"]) for r in rows) == list(range(len(records))))
        audit.check(arm, "", "actual_final_total_free_scalars", sum(int(r["free_coordinates"]) for r in rows) == sum(9*(r["N"]-r["K"]) for r in records))
    identities = ("sample_index", "origin_sample_index", "shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "free_coordinates", "lambda_value", "t")
    tokens = lambda rows: [tuple(str(r[k]) for k in identities) for r in rows]
    audit.check("", "", "actual_same_input_byte_multiset", Counter(tokens(final[ARMS[0]])) == Counter(tokens(final[ARMS[1]])))
    start = 2*len(plan["mixed_records"])//3
    audit.check("", "", "actual_same_final_2000_update_suffix", tokens(final[ARMS[0]][start:]) == tokens(final[ARMS[1]][start:]) and len(final[ARMS[0]]) == len(plan["mixed_records"]))


def run_audit(run, confirmation_out=None, authorization=None, destination=None):
    started = time.time(); run = Path(run).resolve()
    confirm = Path(confirmation_out).resolve() if confirmation_out else run/"confirmation"
    auth_path = Path(authorization).resolve() if authorization else run/"registrations/confirmation_training.json"
    target = Path(destination).resolve() if destination else run/"audit/confirmation_training_accounting"
    if target.exists(): raise FileExistsError("Choose a fresh audit destination")
    missing = [str(confirm/"training"/a/"completed.json") for a in ARMS if not (confirm/"training"/a/"completed.json").exists()]
    if missing:
        return dict(schema=VERSION, status="PENDING", missing_completed_files=missing, scientific_calls=0), None
    plan_path = confirm/"streams/confirmation_plan.json"
    auth, plan = read(auth_path), read(plan_path)
    validate_registration(auth, plan, run, confirm)
    audit = Audit(); tables = {k:[] for k in ("updates", "exposures", "groups", "calls", "attempts", "resources", "dispatch")}
    for p in (auth_path, plan_path, Path(__file__), Path(course.__file__), Path(base.__file__), run/"registrations/CONFIRMATION_DECISION.md"):
        audit.record(p)
    frozen = read(run/"registrations/source_hashes.json"); audit.record(run/"registrations/source_hashes.json")
    audit.check("", "", "original97_source_registration", len(frozen) == 97)
    training_sources = expected_training_sources()
    required = {"src/meshflow_control/"+name for name in training_sources} | {"tools/train_recipe_confirmation.py"}
    audit.check("", "", "confirmation_sources_registered", required <= set(auth.get("source_hashes", {})))
    for label, hashes in (("original97", frozen), ("confirmation", auth["source_hashes"])):
        for relative, sha in hashes.items():
            path = (REPO/relative).resolve()
            if not path.is_relative_to(REPO): raise ValueError("Source path escapes repo")
            audit.record(path); audit.check("", "", label+"_source_hash", path.is_file() and file_hash(path) == sha, relative)
    for relative, sha in plan["implementation_sources"].items():
        path = (REPO/"src/meshflow_control"/relative).resolve()
        if not path.is_relative_to(REPO/"src/meshflow_control"): raise ValueError("Stream source path escapes package")
        audit.record(path); audit.check("", "", "confirmation_stream_source_hash", file_hash(path) == sha, relative)
    construction_path = run/"audit/condition_construction_audit.json"; construction = read(construction_path)
    mp = Path(construction["source_files"]["imported_manifest"]["path"])
    for p in (construction_path, mp): audit.record(p)
    audit.check("", "", "task_manifest_frozen", file_hash(mp) == plan["data_manifest_sha256"] == construction["source_files"]["imported_manifest"]["sha256"])
    families = {t["task_id"]:condition_family(t) for t in read(mp)["tasks"] if t["ratio"] in (.2,.4)}
    paired = {a:read(confirm/"streams/paired_inputs"/(a+".json")) for a in ARMS}
    for a, document in paired.items():
        paired_contract(plan, a, document, paired[BASELINE])
        audit.check(a, "", "paired_implementation_data_plan", document.get("plan_sha256") == file_hash(plan_path)
            and document.get("implementation_sha256") == json_hash(plan["implementation_sources"])
            and document.get("data_manifest_sha256") == plan["data_manifest_sha256"])
    historical_plan_path = run/"streams/recipe_plan.json"; historical_pair_path = run/"streams/paired_inputs/R2_MIX_H_ALL.json"
    for p in (historical_plan_path, historical_pair_path): audit.record(p)
    independent_input_checks(plan, read(historical_plan_path), paired, read(historical_pair_path), audit)
    pre_path = confirm/"streams/precompute_receipt.json"; pre = read(pre_path); audit.record(pre_path)
    expected_pre = dict(status="PASS", namespace=NAMESPACE, selected_pair=list(ARMS), plan_sha256=file_hash(plan_path),
        registered_unique_OT_inputs=48000, completed_unique_OT_inputs=48000, actual_OT_attempts=48000, actual_OT_returns=48000,
        OT_cache_hits=0, historical_OT_reused=0, model_calls=0, optimizer_updates=0, paired_input_rows_per_recipe={a:48000 for a in ARMS})
    audit.check("", "", "fresh_selected_pair_OT_precompute_receipt", all(pre.get(k) == v for k,v in expected_pre.items()))
    results = [audit_arm(confirm, a, plan, auth, audit, tables, families, auth_path) for a in ARMS]
    initial = {a:verify_endpoint_metadata(confirm, a, plan, auth_path, audit, training_sources) for a in ARMS}
    main_initial = run/"training/R2_MIX_H_ALL/initialization_audit.json"; audit.record(main_initial)
    audit.check("", "", "same_initialization_as_historical_R2", all(i == read(main_initial) for i in initial.values()))
    final_pair_checks(plan, tables, audit)
    dispatch = resources_and_dispatch(run, audit, tables)
    # Only confirmation waves count here: historical R2/course names are reused.
    dispatch["intervals"] = [r for r in dispatch["intervals"] if "confirm" in str(r.get("wave", ""))]
    tables["dispatch"] = list(dispatch["intervals"])
    state_path = run/"audit/dispatch_state.json"
    if state_path.exists():
        state = read(state_path); audit.record(state_path)
        audit.check("", "", "global_single_OOM_downgrade_limit", state.get("oom_downgrades", 0) <= 1)
    unique = sum(r["unique_successful_updates"] for r in results)
    physical = sum(r["physical_successful_optimizer_updates"] for r in results)
    audit.check("", "", "exactly12000_new_unique_updates", unique == 12000)
    logical_calls = {kind:sum(int(r["succeeded"]) for r in tables["calls"] if r["kind"] == kind) for kind in EXPECTED_CALLS}
    audit.check("", "", "physical_optimizer_total_matches_call_ledger", logical_calls["optimizer_update"] == physical)
    selected = [r for r in tables["exposures"] if r["final_logical_trajectory"]]
    audit.check("", "", "exact96000_unique_successful_forward_backward_samples", len(selected) == 96000
        and all(r["forward_succeeded"] and r["backward_succeeded"] for r in selected))
    prior_arms = ("R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE", "C20_40_MIX", "C40_20_MIX")
    for label in prior_arms:
        path = run/"training"/label/"completed.json"; value = read(path); audit.record(path)
        audit.check(label, "", "historical_authorized6000_endpoint", value.get("status") == "COMPLETE" and value.get("step") == 6000)
    audit.check("", "", "global_unique_update_ceiling42000", 6000*len(prior_arms)+unique == 42000)
    status = "FAIL" if any(c["status"] == "FAIL" for c in audit.checks) else "PENDING" if any(c["status"] == "PENDING" for c in audit.checks) else "PASS"
    result = dict(schema=VERSION, status=status, run=str(run), confirmation_out=str(confirm), arms=results,
        checks=audit.checks, inputs=audit.sources, initializations=initial, dispatch=dispatch,
        new_unique_logical_updates=unique, new_physical_optimizer_updates=physical,
        successful_physical_calls=logical_calls, expected_unique_calls=dict(train_forward=96000, train_backward=96000, optimizer_update=12000),
        replayed_successful_updates=sum(r["successful_replayed_updates"] for r in results),
        authorized_new_updates=12000, authorized_task_unique_update_ceiling=42000,
        authorized_confirmation_rollouts=128, authorized_task_output_cells=640, evaluation_execution_verified_here=False,
        scientific_calls=0, torch_imported="torch" in sys.modules, elapsed_seconds=time.time()-started,
        evidence_limits=["No tensor deserialization. Internal model/AdamW/RNG/stream hashes are receipt claims bound to immutable actual file SHA.",
            "Whole-effective-batch free-scalar normalization is checked against sample diagnostics and frozen implementation; same multiset does not imply same update denominators.",
            "New input namespace and actual paired-input identities are audited; Gaussian arrays and OT optimization are not recomputed.",
            "Historical condition-pool construction policies differ. This repeat does not repair that deviation or change the initialization seed.",
            "Confirmation uses the same observed DEV parents. New evaluation noise is not unseen-parent generalization.",
            "Physical failed/replayed F/B and optimizer calls remain visible separately from unique logical updates.",
            "Evaluation counts are registered ceilings; this training audit does not claim evaluation completion."])
    target.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items(): write_csv(target/(name+".csv"), rows)
    write_csv(target/"checks.csv", audit.checks)
    (target/"accounting.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    lines = ["# Independent confirmation training accounting", "", "Status: "+status, "",
        "Candidate C20_40_MIX and newly trained matched control R2_MIX_H_ALL; H_ALL; 6000 updates each.", "",
        "| Recipe | Unique | Physical | Completed replay |", "|---|---:|---:|---:|"]
    lines += ["| %s | %s | %s | %s |" % (r["recipe"], r["unique_successful_updates"], r["physical_successful_optimizer_updates"], r["successful_replayed_updates"]) for r in results]
    lines += ["", *result["evidence_limits"], "", *["- "+str(c) for c in audit.checks if c["status"] != "PASS"]]
    (target/"ACCOUNTING.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    return result, target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT)
    parser.add_argument("--confirmation-out", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result, target = run_audit(args.run, args.confirmation_out, args.authorization, args.output)
    print(json.dumps(dict(status=result["status"], output=str(target) if target else None, scientific_calls=0,
        missing=result.get("missing_completed_files", []))), flush=True)
    return 0 if result["status"] == "PASS" else 2 if result["status"] == "PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
