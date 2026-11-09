#!/usr/bin/env python3
"""Read-only CPU accounting for the fixed schedule40 experiment.

Reads bounded snapshots of append-only logs; never imports torch, loads weights,
runs matching, generation, or training. Failed/unfinished and replayed calls remain
visible. Completed checkpoint identities are verified from writer JSON receipts,
not by reloading multi-gigabyte weights.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
from meshflow_control.data.io import atomic_json, digest, json_hash, read_json

NAME = "CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1"
ROOT_NAME = "chair_early_alignment_schedule40_v1"
SEGMENTS = {"EARLY": (1, 6000), "COMMON": (1, 5000),
            "FM_ONLY": (5001, 6000), "LATE": (5001, 6000)}
FINAL_MODELS = ("FM_ONLY", "EARLY", "LATE")
LABELS = ("START5000", "HIST_FM6000", "HIST_FREE_ALIGN6000", *FINAL_MODELS)


def boolean(value):
    if isinstance(value, bool): return value
    if value in ("True", "true", "1", 1): return True
    if value in ("False", "false", "0", 0): return False
    raise ValueError("Not a recorded boolean: "+repr(value))


def optional_json(path):
    return read_json(path) if Path(path).exists() else None


def complete_lines(path, warnings):
    """Never parse a partly written trailing line or read beyond initial EOF."""
    path = Path(path)
    if not path.exists(): return
    with path.open("rb") as handle:
        limit = os.fstat(handle.fileno()).st_size
        number = 0
        while handle.tell() < limit:
            line = handle.readline(limit-handle.tell())
            number += 1
            if not line.endswith(b"\n"):
                warnings.append(dict(path=str(path), line=number,
                                     reason="TRAILING_PARTIAL_LINE_IGNORED"))
                break
            yield number, line.decode("utf-8-sig")


def json_rows(path, warnings, errors):
    for number, text in complete_lines(path, warnings):
        if not text.strip(): continue
        try: yield json.loads(text)
        except (json.JSONDecodeError, UnicodeError) as exc:
            errors.append(dict(path=str(path), line=number, reason="MALFORMED_COMPLETE_JSON_LINE", error=str(exc)))


def csv_rows(path, warnings, errors):
    rows = (text for _, text in complete_lines(path, warnings))
    for number, row in enumerate(csv.DictReader(rows), 2):
        if None in row or any(value is None for value in row.values()):
            errors.append(dict(path=str(path), line=number, reason="MALFORMED_COMPLETE_CSV_ROW"))
            continue
        yield row


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def union_seconds(intervals):
    ordered = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    if not ordered: return 0.
    total = 0.; lo, hi = ordered[0]
    for a, b in ordered[1:]:
        if a > hi: total += hi-lo; lo, hi = a, b
        else: hi = max(hi, b)
    return total+hi-lo


def active(segment, step):
    return (segment == "EARLY" and step <= 1000) or (segment == "LATE" and step >= 5001)


def parse_attempt(folder, segment, warnings, errors):
    """Preserve every attempted step, even if no training CSV row was written."""
    attempt_id = folder.name
    updates = {}
    def update(step):
        key = int(step)
        if key not in updates:
            updates[key] = dict(segment=segment, attempt_id=attempt_id, step=key,
                input_begin=0, input_done=0, update_begin=0, update_done=0,
                optimizer_call_begin=0, optimizer_call_done=0, optimizer_call_failed=0)
        return updates[key]
    event_counts = Counter()
    checkpoint_receipts = []
    ancestry = []
    for row in json_rows(folder/"events.jsonl", warnings, errors):
        kind = row["event"]; event_counts[kind] += 1
        if kind in ("INPUT_BEGIN", "INPUT_DONE", "UPDATE_BEGIN", "UPDATE_DONE"):
            record = update(row["step"]); record[kind.lower()] += 1
            if kind == "INPUT_DONE":
                record["event_shared_batch_sha256"] = row["shared_batch_sha256"]
                record["event_input_batch_sha256"] = row["input_batch_sha256"]
                record["event_label_batch_sha256"] = row["label_batch_sha256"]
            if kind == "UPDATE_DONE": record["update_done_time_unix"] = row["time_unix"]
        elif kind == "CHECKPOINT_DONE": checkpoint_receipts.append(row["checkpoint"])
        elif kind in ("FORK", "RESUME"): ancestry.append(row)
    call_counts = defaultdict(Counter)
    call_seconds = Counter()
    open_calls = {}
    failed_calls = []
    for row in json_rows(folder/"calls.jsonl", warnings, errors):
        call_id = row["call_id"]; kind = row["kind"]; event = row["event"]
        call_counts[kind][event] += 1
        if event == "CALL_BEGIN":
            if call_id in open_calls:
                errors.append(dict(segment=segment, attempt_id=attempt_id, reason="DUPLICATE_OPEN_CALL", call_id=call_id))
            open_calls[call_id] = row
        else:
            initial = open_calls.pop(call_id, None)
            if initial is None or initial["kind"] != kind:
                errors.append(dict(segment=segment, attempt_id=attempt_id, reason="CALL_END_WITHOUT_MATCHING_BEGIN", call_id=call_id))
            if event == "CALL_DONE": call_seconds[kind] += float(row["seconds"])
            elif event == "CALL_FAILED": failed_calls.append(row)
        record = update(row["step"])
        if kind == "optimizer_update":
            record[dict(CALL_BEGIN="optimizer_call_begin", CALL_DONE="optimizer_call_done",
                        CALL_FAILED="optimizer_call_failed")[event]] += 1
        else:
            count_key = kind+"_"+event.lower()
            record[count_key] = record.get(count_key, 0)+1
    seen_log = set()
    for row in csv_rows(folder/"training_log.csv", warnings, errors):
        step = int(row["step"]); record = update(step)
        if step in seen_log:
            errors.append(dict(segment=segment, attempt_id=attempt_id, step=step, reason="DUPLICATE_TRAINING_LOG_ROW"))
        seen_log.add(step)
        record.update(row)
        record["step"] = step
        record["has_training_log"] = True
    for record in updates.values():
        record.setdefault("has_training_log", False)
        record["step_status"] = (
            "COMPLETED_AND_LOGGED" if record["update_done"] == 1 and record["has_training_log"] else
            "OPTIMIZER_RETURNED_UPDATE_AUDIT_INCOMPLETE" if record["optimizer_call_done"] else
            "OPTIMIZER_CALL_FAILED_APPLICATION_UNKNOWN" if record["optimizer_call_failed"] else
            "ATTEMPTED_NO_OPTIMIZER_RETURN")
    final_status = optional_json(folder/"final_status.json")
    status = final_status["status"] if final_status else "RUNNING_OR_UNCLOSED"
    return dict(segment=segment, attempt_id=attempt_id, status=status,
        final_status=final_status, events=dict(event_counts),
        call_counts={k: dict(v) for k, v in call_counts.items()},
        synchronized_call_seconds=dict(call_seconds), failed_calls=failed_calls,
        unfinished_calls=list(open_calls.values()), checkpoint_receipts=checkpoint_receipts,
        ancestry=ancestry, updates=updates)


def controller_accounting(out, warnings, errors):
    calls = Counter(); events = []; pending = {}; workers = []; cpu = []; intervals = defaultdict(list)
    gpu = defaultdict(list); ram_min = None; load_max = [0., 0., 0.]
    for row in json_rows(out/"audit/controller.jsonl", warnings, errors):
        kind = row["event"]; calls[kind] += 1
        if kind in ("WORKER_START", "WORKER_END"):
            key = (row["phase"], row["name"], row["attempt"])
            if kind == "WORKER_START": pending[key] = row
            else:
                begin = pending.pop(key, None)
                end = dict(row)
                if begin:
                    end["start_time_unix"] = begin["time_unix"]
                    end["end_time_unix"] = row["time_unix"]
                    end["worker_count_at_start"] = begin["worker_count"]
                    end["observed_single_device_peak_mib_before_dispatch"] = begin["observed_single_device_peak_mib"]
                    intervals[row["phase"]].append((begin["time_unix"], row["time_unix"]))
                else: errors.append(dict(reason="WORKER_END_WITHOUT_START", identity=list(key)))
                workers.append(end)
        elif kind in ("CPU_START", "CPU_END"):
            if kind == "CPU_START": pending[("cpu", row["name"], "")] = row
            else:
                begin = pending.pop(("cpu", row["name"], ""), None)
                cpu.append(dict(row, start_time_unix=None if begin is None else begin["time_unix"]))
        elif kind == "RESOURCES":
            r = row["snapshot"]; names = row.get("active", [])
            phase = "training" if names and all(n in SEGMENTS for n in names) else "sampling" if names and all(n in LABELS for n in names) else "idle_or_cpu"
            if r.get("gpu"): gpu[phase].append(dict(r["gpu"], active_workers=len(names)))
            available = r.get("ram", {}).get("MemAvailable")
            if available is not None: ram_min = available if ram_min is None else min(ram_min, available)
            load_max = [max(a, float(b)) for a, b in zip(load_max, r.get("loadavg", [0, 0, 0]))]
        if kind in ("OOM_DOWNGRADE", "RECOVERY_ACCOUNTING", "REPLAY_BLOCKED", "BLOCKED_WORKER", "CONTROLLER_FAILURE"):
            events.append(row)
    peaks = {}
    for phase, rows in gpu.items():
        peaks[phase] = dict(snapshot_count=len(rows),
            maximum_device_used_mib=max(r["used_mib"] for r in rows),
            minimum_device_free_mib=min(r["free_mib"] for r in rows),
            maximum_gpu_utilization=max(r["utilization"] for r in rows),
            average_sampled_gpu_utilization=sum(r["utilization"] for r in rows)/len(rows),
            maximum_active_workers=max(r["active_workers"] for r in rows),
            scope="Five-second-ish device snapshots include desktop; may miss instantaneous peaks; not process VRAM")
    return dict(event_counts=dict(calls), workers=workers, cpu_phases=cpu,
        unfinished_dispatches=[dict(identity=list(k), **v) for k, v in pending.items()],
        resource_summary=peaks, minimum_available_RAM_bytes=ram_min,
        maximum_loadavg=load_max, recovery_and_failures=events,
        worker_owner_wall_seconds={p: sum(float(r["wall_seconds"]) for r in workers if r["phase"] == p) for p in ("training", "sampling")},
        union_closed_worker_wall_seconds={p: union_seconds(v) for p, v in intervals.items()},
        training_dispatch_receipt=optional_json(out/"audit/training_dispatch_receipt.json"),
        sampling_dispatch_receipt=optional_json(out/"audit/sampling_dispatch_receipt.json"),
        timing_interpretation="Concurrent owner/call time sums overlap; wall times are not isolated throughput benchmarks")


def verify_sources(out):
    baseline = optional_json(out/"audit/source_hashes_before.json")
    if baseline is None: return dict(status="MISSING", checked=0, changed=[])
    changed = []
    for relative, expected in baseline.items():
        for kind, path in (("live", REPO/relative), ("snapshot", out/"source_code"/relative)):
            actual = digest(path) if path.is_file() else None
            if actual != expected: changed.append(dict(kind=kind, path=str(path), expected=expected, actual=actual))
    return dict(status="PASS" if not changed else "FAIL", checked=2*len(baseline),
                changed=changed, frozen_manifest_sha256=digest(out/"audit/source_hashes_before.json"),
                analyzer_source_sha256=digest(__file__),
                analyzer_scope="Postrun CPU analysis added after training freeze; not a training source mutation")


def fork_audit(out):
    common = optional_json(out/"checkpoints/COMMON_step5000.json")
    results = []
    for segment in ("FM_ONLY", "LATE"):
        completed = optional_json(out/"training"/segment/"completed.json")
        status = completed or optional_json(out/"training"/segment/"status.json")
        if common is None or status is None or "lineage" not in status:
            results.append(dict(segment=segment, status="PENDING")); continue
        line = status["lineage"]
        checks = dict(prefix_step_5000=common.get("global_step") == 5000,
            fork_step_5000=line.get("fork_step") == 5000,
            generator_optimizer_not_reset=line.get("generator_optimizer_reset") is False,
            source_checkpoint_sha256=line.get("common_checkpoint_sha256") == common["sha256"],
            generator_sha256=line.get("common_generator_sha256") == common["generator_state_sha256"],
            optimizer_sha256=line.get("common_optimizer_sha256") == common["optimizer_state_sha256"],
            RNG_sha256=line.get("common_rng_sha256") == common["rng_sha256"],
            destination=line.get("fork_destination") == segment)
        results.append(dict(segment=segment, status="PASS" if all(checks.values()) else "FAIL",
                            checks=checks, common_checkpoint_receipt=common,
                            claimed_lineage=line))
    early = optional_json(out/"training/EARLY/initialization_audit.json")
    late = optional_json(out/"training/LATE/completed.json") or optional_json(out/"training/LATE/status.json")
    head_same = None
    if early and late and late.get("lineage", {}).get("head_initial_sha256"):
        head_same = early["head_initial_sha256"] == late["lineage"]["head_initial_sha256"]
    return dict(rows=results, early_late_head_initial_sha256_equal=head_same,
        evidence_scope="JSON writer receipts and fork lineage, not independent large-weight reload",
        generator_optimizer_RNG_stream_restore_implementation="Frozen schedule40.load_checkpoint and train_segment; stream cursor/identity checked by worker")


def analyze(out, final=False):
    out = Path(out).resolve()
    if out.name != ROOT_NAME or out.parent.name != "experiments":
        raise ValueError("Wrong fixed experiment output directory")
    started = time.perf_counter(); warnings = []; errors = []
    plan_path, paired_path = out/"stream/train_plan.json", out/"stream/paired_inputs.json"
    plan, paired = read_json(plan_path), read_json(paired_path)
    if paired["plan_sha256"] != digest(plan_path): raise ValueError("Paired/plan identity differs")
    records = {r["sample_index"]: r for r in plan["records"]}
    expected = {r["sample_index"]: r for r in paired["records"]}
    if set(records) != set(range(48000)) or set(expected) != set(records):
        raise ValueError("Expected complete registered 48,000 inputs")
    expected_batch = {step: dict(
        shared_batch_sha256=json_hash([expected[i]["shared_input_sha256"] for i in range(8*(step-1), 8*step)]),
        input_batch_sha256=json_hash([expected[i]["input_bytehash"] for i in range(8*(step-1), 8*step)]),
        label_batch_sha256=json_hash([expected[i]["label_bytehash"] for i in range(8*(step-1), 8*step)]))
        for step in range(1, 6001)}
    attempts = []
    updates = {}
    exposures = []
    pair_failures = []
    exposure_groups = defaultdict(list)
    observed_samples = defaultdict(set)
    for segment in SEGMENTS:
        folder = out/"training"/segment/"attempts"
        for attempt in sorted(folder.iterdir()) if folder.exists() else []:
            if not attempt.is_dir(): continue
            parsed = parse_attempt(attempt, segment, warnings, errors)
            attempts.append(parsed)
            for step, row in parsed["updates"].items(): updates[(segment, attempt.name, step)] = row
            for row in csv_rows(attempt/"exposure.csv", warnings, errors):
                step, index = int(row["step"]), int(row["sample_index"])
                key = (segment, attempt.name, step)
                e, p = expected.get(index), records.get(index)
                mismatch = []
                if e is None or p is None: mismatch.append("unregistered_sample")
                else:
                    for name in ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key", "task_id"):
                        if row[name] != e[name]: mismatch.append(name)
                    for name in ("uid", "task_id"):
                        if row[name] != p[name]: mismatch.append("plan_"+name)
                    for name in ("N", "K"):
                        if int(row[name]) != p[name]: mismatch.append("plan_"+name)
                    if float(row["alpha"]) != p["alpha"] or float(row["ratio"]) != .4: mismatch.append("alpha_or_ratio")
                    if step != p["step"] or not 8*(step-1) <= index < 8*step: mismatch.append("sample_step")
                    if int(row["free_coordinates"]) != e["free_coordinate_count"]: mismatch.append("free_denominator")
                    if row["ot_cache_status"] != "HIT": mismatch.append("unexpected_OT_execution")
                    if boolean(row["alignment_active"]) != active(segment, step): mismatch.append("active_schedule")
                    if boolean(row["alignment_eligible"]) != (active(segment, step) and p["alpha"] == 1.): mismatch.append("clean_B_eligibility")
                if mismatch: pair_failures.append(dict(segment=segment, attempt=attempt.name, step=step, sample_index=index, fields=mismatch))
                normalized = dict(row, step=step, sample_index=index)
                exposures.append(normalized); exposure_groups[key].append(normalized)
                observed_samples[index].add(segment)
    physical = Counter()
    successful = defaultdict(list)
    curve_rows = []
    for key, row in sorted(updates.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        segment, attempt, step = key
        if not SEGMENTS[segment][0] <= step <= SEGMENTS[segment][1]:
            errors.append(dict(reason="UPDATE_OUTSIDE_SEGMENT", identity=list(key)))
        for name, wanted in expected_batch[step].items():
            for actual_key in (name, "event_"+name):
                if actual_key in row and row[actual_key] != wanted:
                    pair_failures.append(dict(segment=segment, attempt=attempt, step=step, fields=[actual_key]))
        physical[segment] += row["optimizer_call_done"]
        if row["optimizer_call_done"]: successful[(segment, step)].append(key)
        sample_rows = exposure_groups.get(key, [])
        row["exposure_samples"] = len(sample_rows)
        row["clean_exposure_samples"] = sum(float(s["alpha"]) == 1. for s in sample_rows)
        row["eligible_B_exposure_samples"] = sum(boolean(s["alignment_eligible"]) for s in sample_rows)
        if row["update_done"] or row["optimizer_call_done"]:
            if len(sample_rows) != 8 or len({s["sample_index"] for s in sample_rows}) != 8 or len({s["uid"] for s in sample_rows}) != 8:
                errors.append(dict(reason="COMPLETED_UPDATE_EXPOSURE_INCOMPLETE", identity=list(key)))
            if row["clean_exposure_samples"] != 4 or row["eligible_B_exposure_samples"] != (4 if active(segment, step) else 0):
                errors.append(dict(reason="COMPLETED_UPDATE_CLEAN_ELIGIBILITY_MISMATCH", identity=list(key)))
            expected_calls = dict(train_forward_call_done=8, train_backward_call_done=8,
                                  aligner_forward_call_done=4 if active(segment, step) else 0)
            for call_key, count in expected_calls.items():
                if row.get(call_key, 0) != count:
                    errors.append(dict(reason="COMPLETED_UPDATE_MODEL_CALL_COUNT_DIFFERS",
                                       identity=list(key), field=call_key, expected=count,
                                       actual=row.get(call_key, 0)))
        curve_rows.append(row)
    for row in curve_rows:
        keys = sorted(successful.get((row["segment"], row["step"]), []), key=lambda k: (0 if k[1] == "initial" else 1, k[1]))
        row["successful_executions_at_segment_step"] = len(keys)
        key = (row["segment"], row["attempt_id"], row["step"])
        row["successful_execution_ordinal"] = keys.index(key)+1 if key in keys else 0
        row["is_replayed_successful_update"] = key in keys and keys.index(key) > 0
        row["attempt_final_status"] = next(a["status"] for a in attempts if a["segment"] == row["segment"] and a["attempt_id"] == row["attempt_id"])
    unique = {s: {step for seg, step in successful if seg == s} for s in SEGMENTS}
    unique_count = sum(map(len, unique.values()))
    logical = {"EARLY": len(unique["EARLY"]),
               "FM_ONLY": len(unique["COMMON"] | unique["FM_ONLY"]),
               "LATE": len(unique["COMMON"] | unique["LATE"])}
    duplicate_count = sum(physical.values())-unique_count
    exposure_counts = defaultdict(Counter)
    for row in exposures:
        key = (row["segment"], row["attempt_id"], row["step"])
        record = updates.get(key, {})
        count = exposure_counts[(row["segment"], row["attempt_id"], row["uid"])]
        count["written_before_update_samples"] += 1
        clean = float(row["alpha"]) == 1.
        count["written_clean_samples"] += clean
        count["written_augmented_samples"] += not clean
        if record.get("optimizer_call_done"):
            count["optimizer_returned_samples"] += 1
            count["optimizer_returned_clean_samples"] += clean
            count["optimizer_returned_B_eligible_samples"] += boolean(row["alignment_eligible"])
        if record.get("update_done"): count["validated_update_done_samples"] += 1
    perparent = [dict(segment=s, attempt_id=a, uid=u, **dict(v)) for (s,a,u),v in sorted(exposure_counts.items())]
    by_segment = {}
    for segment, (lo, hi) in SEGMENTS.items():
        segment_rows = [r for r in curve_rows if r["segment"] == segment]
        aligned_unique = sum(4 for step in unique[segment] if active(segment, step))
        complete = optional_json(out/"training"/segment/"completed.json")
        by_segment[segment] = dict(expected_physical_unique=hi-lo+1,
            actual_successful_optimizer_calls=physical[segment], unique_optimizer_steps=len(unique[segment]),
            minimum_completed_step=min(unique[segment], default=None), maximum_completed_step=max(unique[segment], default=None),
            unique_update_coverage_complete=unique[segment] == set(range(lo,hi+1)),
            successful_replayed_updates=physical[segment]-len(unique[segment]),
            unique_active_B_updates=aligned_unique//4, unique_B_eligible_samples=aligned_unique,
            completed_receipt_status=None if complete is None else complete.get("status"),
            completed_receipt_global_step=None if complete is None else complete.get("step"),
            logged_stage_seconds={name: sum(float(r.get(name) or 0) for r in segment_rows)
                                  for name in ("input_seconds", "forward_seconds", "aligner_forward_seconds",
                                               "backward_seconds", "optimizer_seconds", "update_seconds")},
            alignment_phase_timing={phase: dict(
                logged_updates=sum(bool(r.get("has_training_log")) and active(segment,r["step"])==enabled for r in segment_rows),
                update_seconds=sum(float(r.get("update_seconds") or 0) for r in segment_rows if active(segment,r["step"])==enabled),
                maximum_reserved_bytes=max((int(r.get("cuda_peak_reserved_bytes") or 0) for r in segment_rows if active(segment,r["step"])==enabled),default=0))
                for phase,enabled in (("B_active",True),("B_inactive",False))},
            maximum_torch_allocated_bytes=max((int(r.get("cuda_peak_allocated_bytes") or 0) for r in segment_rows),default=0),
            maximum_torch_reserved_bytes=max((int(r.get("cuda_peak_reserved_bytes") or 0) for r in segment_rows),default=0),
            torch_peak_scope="Per-effective-update allocator counters; includes resident tensors and optimizer/backward; excludes desktop/context; allocated and reserved are not additive")
    cross_steps = []
    for step in range(1,6001):
        segments = ("EARLY","COMMON") if step <= 5000 else ("EARLY","FM_ONLY","LATE")
        completed_segments = [s for s in segments if step in unique[s]]
        seen = [i for i in range(8*(step-1),8*step)
                if all(s in observed_samples[i] for s in completed_segments)] if completed_segments else []
        cross_steps.append(dict(step=step, expected_segments=list(segments),
            completed_segments=completed_segments,
            all_required_segments_complete=len(completed_segments)==len(segments),
            complete_shared_samples=len(seen),
            expected_shared_batch_sha256=expected_batch[step]["shared_batch_sha256"]))
    pairing = dict(schema=NAME+"/input_pairing/v1", status="FAIL" if pair_failures else "PASS" if unique_count == 13000 else "IN_PROGRESS",
        registered_samples=48000, actual_exposure_rows=len(exposures),
        actual_unique_sample_indices=len(observed_samples), comparisons_to_frozen_inputs=len(exposures),
        plan_sha256=digest(plan_path), paired_inputs_sha256=digest(paired_path),
        mismatch_count=len(pair_failures), mismatches=pair_failures,
        cross_segment_steps_complete=sum(r["all_required_segments_complete"] and r["complete_shared_samples"]==8 for r in cross_steps),
        rows=cross_steps,
        inherited_prefix="FM_ONLY and LATE reuse COMMON steps1..5000; no duplicate physical prefix training")
    sources = verify_sources(out); forks = fork_audit(out)
    checkpoint_rows = []
    for segment, (_, endpoint) in SEGMENTS.items():
        path = out/"checkpoints"/f"{segment}_step{endpoint:04d}.json"
        receipt = optional_json(path)
        completed = optional_json(out/"training"/segment/"completed.json")
        receipt_identity = bool(receipt and receipt.get("segment")==segment and receipt.get("global_step")==endpoint
                                and completed and completed.get("checkpoint",{}).get("sha256")==receipt.get("sha256"))
        checkpoint_rows.append(dict(segment=segment, receipt_path=str(path), receipt=receipt,
            endpoint_receipt_identity_valid=receipt_identity,
            weight_file_exists=bool(receipt and Path(receipt["path"]).is_file()),
            receipt_sha256=digest(path) if receipt else None))
    completed_exports = []
    for segment in FINAL_MODELS:
        receipt = optional_json(out/"training"/segment/"completed.json")
        export = receipt.get("inference_export") if receipt else None
        completed_exports.append(dict(segment=segment, receipt=export,
                                      file_exists=bool(export and Path(export["path"]).is_file())))
    controller = controller_accounting(out, warnings, errors)
    dispatch_state = optional_json(out/"audit/dispatch_state.json")
    recovery_errors = []
    if dispatch_state and dispatch_state.get("oom_downgrades", 0) > 1:
        recovery_errors.append("More than one declared dual-to-serial downgrade")
    for segment in SEGMENTS:
        arms = [a for a in attempts if a["segment"] == segment]
        if len(arms) > 2: recovery_errors.append(segment+": more than one retry")
        replayed = by_segment[segment]["successful_replayed_updates"]
        if replayed > 99: recovery_errors.append(segment+": more than99 successful updates replayed")
        if replayed and not any(r.get("event") == "RECOVERY_ACCOUNTING" and r.get("name") == segment
                                for r in controller["recovery_and_failures"]):
            recovery_errors.append(segment+": replay lacks explicit controller accounting")
        if len(arms) > 1 and not any(a["status"] == "OOM" for a in arms):
            recovery_errors.append(segment+": retry lacks recorded OOM")
    if recovery_errors: errors.extend(dict(reason="RECOVERY_CONTRACT", detail=x) for x in recovery_errors)
    controller["recovery_contract"] = dict(status="PASS" if not recovery_errors else "FAIL",
        errors=recovery_errors, dispatch_state=dispatch_state)
    sample_complete = sum(1 for label in LABELS for p in (out/"outputs"/label).glob("*/receipt.json")
                          if optional_json(p).get("status") == "COMPLETE")
    score = optional_json(out/"evaluation/scoring_receipt.json")
    gallery = optional_json(out/"gallery/manifest.json")
    unique_complete = unique_count == 13000 and all(x["unique_update_coverage_complete"] for x in by_segment.values())
    recorded_complete = all(x["completed_receipt_status"] == "COMPLETE" for x in by_segment.values())
    validated_logs_complete = all(r["step_status"] == "COMPLETED_AND_LOGGED" and r["optimizer_call_done"] == 1 for r in curve_rows if r["optimizer_call_done"])
    checkpoints_complete = all(r["endpoint_receipt_identity_valid"] and r["weight_file_exists"] for r in checkpoint_rows) and all(r["file_exists"] for r in completed_exports)
    forks_complete = all(r["status"] == "PASS" for r in forks["rows"]) and forks["early_late_head_initial_sha256_equal"] is True
    teacher_path = out/"teacher/teacher_manifest.json"
    teacher_registration = optional_json(teacher_path)
    known_counts = (by_segment["EARLY"]["unique_B_eligible_samples"] == 4000
                    and by_segment["LATE"]["unique_B_eligible_samples"] == 4000
                    and teacher_registration is not None
                    and teacher_registration.get("teacher_forwards") == 0)
    expected_identities = dict(data_manifest_sha256=plan["data_manifest_sha256"],
                               plan_sha256=digest(plan_path),paired_inputs_sha256=digest(paired_path),
                               teacher_registration_sha256=digest(teacher_path) if teacher_path.exists() else None)
    for segment in SEGMENTS:
        completed = optional_json(out/"training"/segment/"completed.json")
        if completed and completed.get("identities") != expected_identities:
            errors.append(dict(reason="COMPLETED_SEGMENT_INPUT_TEACHER_IDENTITY_CHANGED",segment=segment))
    training_complete = all((unique_complete, recorded_complete, validated_logs_complete, checkpoints_complete,
                             forks_complete, known_counts, sources["status"] == "PASS", not pair_failures, not errors))
    evaluation_complete = sample_complete == 48 and score is not None and score.get("status") == "COMPLETE" and score.get("scored") == 48 and gallery is not None and gallery.get("status") == "COMPLETE"
    status = "COMPLETE" if training_complete and evaluation_complete and not warnings else "TRAINING_COMPLETE_EVALUATION_PENDING" if training_complete else "IN_PROGRESS"
    if errors or pair_failures or sources["status"] == "FAIL" or any(r["status"]=="FAIL" for r in forks["rows"]):
        status = "FAIL"
    if final and status != "COMPLETE" and status != "FAIL": status = "PARTIAL_BLOCKED"
    attempts_serializable = [{k:v for k,v in a.items() if k != "updates"} for a in attempts]
    accounting = dict(schema=NAME+"/final_accounting/v1", status=status, generated_unix=time.time(),
        analyzer_source_sha256=digest(__file__), final_requested=bool(final),
        model_forwards_by_analyzer=0, optimizer_updates_by_analyzer=0,
        registered_physical_unique_optimizer_updates=13000,
        actual_optimizer_successful_returns=sum(physical.values()),
        actual_unique_segment_optimizer_updates=unique_count,
        actual_successful_replayed_updates=duplicate_count,
        logical_updates=logical, expected_logical_updates_each=6000,
        inherited_prefix_updates=dict(FM_ONLY=5000 if len(unique["COMMON"])==5000 else len(unique["COMMON"]),
                                     LATE=5000 if len(unique["COMMON"])==5000 else len(unique["COMMON"])),
        failed_optimizer_calls=sum(a["call_counts"].get("optimizer_update",{}).get("CALL_FAILED",0) for a in attempts),
        failed_optimizer_call_interpretation="Application UNKNOWN if optimizer raised or synchronization failed; successful returns and failed attempts reported separately",
        segments=by_segment, attempts=attempts_serializable,
        input_pairing_summary={k:v for k,v in pairing.items() if k not in ("rows","mismatches")},
        expected_training_identities=expected_identities,
        teacher_forwards_new=None if teacher_registration is None else teacher_registration.get("teacher_forwards"),
        teacher_cache_registered_clean_samples=None if teacher_registration is None else teacher_registration["inputs"]["alignment_samples"],
        OT_precompute_receipt=optional_json(out/"stream/precompute_receipt.json"),
        checkpoint_receipts=checkpoint_rows, inference_exports=completed_exports,
        checkpoint_verification_scope="JSON writer receipts plus file presence; large checkpoint bytes not rehashed/reloaded by this CPU analyzer",
        fork_audit=forks, source_verification=sources, controller=controller,
        training_complete=training_complete, evaluation_complete=evaluation_complete,
        completed_sample_receipts=sample_complete, scoring_receipt=score,
        gallery_status=None if gallery is None else gallery.get("status"),
        warnings=warnings, errors=errors,
        snapshot_note="Logs are bounded at each file's opening size. Concurrent progress may add new rows after this snapshot.",
        wall_seconds=time.perf_counter()-started)
    atomic_json(out/"audit/input_pairing.json", pairing)
    atomic_json(out/"audit/final_accounting.json", accounting)
    write_csv(out/"audit/training_curves.csv", curve_rows)
    write_csv(out/"audit/perparent_actual_exposure.csv", perparent)
    return accounting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/mnt/e/MeshFlow-Control-artifacts/experiments/"+ROOT_NAME)
    parser.add_argument("--final", action="store_true", help="Require full experiment completion; nonzero on pending/fail")
    args = parser.parse_args(argv)
    result = analyze(args.out, final=args.final)
    print(json.dumps({k:result[k] for k in ("status","actual_optimizer_successful_returns",
        "actual_unique_segment_optimizer_updates","actual_successful_replayed_updates",
        "logical_updates","training_complete","evaluation_complete","wall_seconds")}),flush=True)
    return 0 if not args.final or result["status"]=="COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
