"""Read-only progress or a final, ledger-checked training report; no model imports.

--snapshot never writes. --final requires the completed 1500-update worker and
its final-status/resource receipts, then writes only results/training_* and the
loss-curves image. No checkpoint tensor storage is loaded or rescored.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "experiments/r2_bench_newdata_joint_v1"
ARM = "NEW_DATA_JOINT"
SCHEMA = "BENCH_NEWDATA_JOINT_TRAINING_SUMMARY_V1"
KINDS = ("train_forward", "train_backward", "optimizer_update", "head_forward", "head_backward", "head_optimizer_update")
SUM_FIELDS = ("all_free_velocity_sse", "all_free_scalar_count", "seam_tip_velocity_sse",
              "seam_tip_scalar_count", "bulk_velocity_sse", "bulk_scalar_count", "endpoint_sse",
              "seam_tip_endpoint_sse", "bulk_endpoint_sse")
CONFUSION = ("TP", "FN", "FP", "TN")


def resolve(path):
    value = str(path)
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    elif os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":" + value[6:].replace("/", "\\")
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def read_json(path):
    return json.loads(read(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path, *, strict):
    """Ignore an unfinished last append only in a live snapshot."""
    rows, partial = [], False
    if not Path(path).exists():
        return rows, partial
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                if strict:
                    raise ValueError(f"Incomplete JSONL append: {path}:{number}")
                partial = True
                break
            rows.append(json.loads(line))
    return rows, partial


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def near(actual, expected, label, *, rel_tol=1e-7):
    if not math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=1e-8):
        raise ValueError("Training accounting mismatch: " + label)


def ledger_summary(rows, limits, *, strict):
    pending, seen = {}, set()
    attempted, completed, failed, seconds = Counter(), Counter(), Counter(), Counter()
    sequence = defaultdict(list)
    attempts = set()
    for row in rows:
        kind, ident = row["kind"], row["call_id"]
        if row["arm"] != ARM or kind not in limits:
            raise ValueError("Unexpected ledger owner or call kind")
        attempts.add(row["attempt_id"])
        if row["event"] == "CALL_BEGIN":
            if ident in seen:
                raise ValueError("Duplicate ledger call ID")
            seen.add(ident)
            pending[ident] = row
            attempted[kind] += 1
            sequence[kind].append(row["context"])
        elif row["event"] in ("CALL_DONE", "CALL_FAILED"):
            begun = pending.pop(ident, None)
            if begun is None or begun["kind"] != kind or begun["context"] != row["context"]:
                raise ValueError("Ledger completion without matching reservation")
            if row["event"] == "CALL_DONE":
                completed[kind] += 1
                seconds[kind] += float(row["seconds"])
            else:
                failed[kind] += 1
        else:
            raise ValueError("Unknown ledger event")
    if any(attempted[k] > limits[k] for k in limits):
        raise ValueError("Recorded call budget exceeded")
    if strict and (pending or failed or any(attempted[k] != limits[k] or completed[k] != limits[k] for k in limits)):
        raise ValueError("Final ledger is incomplete, failed, pending or short of the frozen budget")
    return dict(limits=limits, attempted={k: attempted[k] for k in limits},
                successful={k: completed[k] for k in limits}, failed={k: failed[k] for k in limits},
                pending_call_ids=sorted(pending), completed_call_seconds={k: seconds[k] for k in limits},
                attempt_ids=sorted(attempts)), sequence


def step_metrics(row):
    result = {k: v for k, v in row.items() if k not in ("head_confusion", "peak_scope")}
    result.update({"head_" + k: row["head_confusion"][k] for k in CONFUSION})
    for name, sse, denominator in (("free_velocity_MSE", "all_free_velocity_sse", "all_free_scalar_count"),
                                   ("tip_velocity_MSE", "seam_tip_velocity_sse", "seam_tip_scalar_count"),
                                   ("bulk_velocity_MSE", "bulk_velocity_sse", "bulk_scalar_count")):
        result[name] = ratio(row[sse], row[denominator])
        result[name + "_status"] = "OK" if row[denominator] else "NA_NO_FREE_SCALARS_IN_STRATUM"
    c = row["head_confusion"]
    result["head_owner_recall"] = ratio(c["TP"], c["TP"] + c["FN"])
    result["head_inactive_recall"] = ratio(c["TN"], c["TN"] + c["FP"])
    return result


def summarize(out, *, final):
    folder = out / "training" / ARM
    status_path, reg_path = folder / "status.json", out / "training_registration.json"
    state, reg = read_json(status_path), read_json(reg_path)
    if state["registration_sha256"] != sha(reg_path):
        raise ValueError("Training status registration hash differs")
    if final and (state["status"] != "COMPLETE" or state["local_step"] != 1500 or state["head_step"] != 1500):
        raise RuntimeError("--final requires COMPLETE generator/head local1500")
    schedule_path, manifest_path = resolve(reg["input_schedule"]), resolve(reg["manifest"])
    schedule, manifest = read_json(schedule_path), read_json(manifest_path)
    if sha(schedule_path) != reg["input_schedule_sha256"] or sha(manifest_path) != reg["manifest_sha256"]:
        raise ValueError("Frozen schedule/manifest changed")
    logs, partial_log = jsonl(folder / "training_log.jsonl", strict=final)
    exposures, partial_exposure = jsonl(folder / "exposure.jsonl", strict=final)
    calls, partial_calls = jsonl(folder / "calls.jsonl", strict=final)
    limits = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                  head_forward=reg["head_forward_budget"], head_backward=reg["head_forward_budget"],
                  head_optimizer_update=reg["head_optimizer_updates"])
    ledger, sequence = ledger_summary(calls, limits, strict=final)
    if [r["local_step"] for r in logs] != list(range(1, len(logs) + 1)):
        raise ValueError("Missing/duplicate training-log step")
    cutoff = min(int(state["local_step"]), len(logs))
    logs = logs[:cutoff]
    visible = [r for r in exposures if r["local_step"] <= cutoff]
    if final and (cutoff != 1500 or len(visible) != 12000 or len(exposures) != 12000):
        raise ValueError("Final logs must contain exactly1500updates/12000instances")
    if len(visible) != cutoff * 8:
        raise ValueError("Completed-step exposure count differs")
    observed_by_step = defaultdict(list)
    for row in visible:
        observed_by_step[row["local_step"]].append(row)
    for step, batch in enumerate(schedule["steps"][:cutoff], 1):
        actual = sorted(observed_by_step[step], key=lambda r: r["microbatch"])
        if [r["microbatch"] for r in actual] != list(range(8)):
            raise ValueError("Missing/duplicate microbatch exposure")
        for expected, observed in zip(batch["samples"], actual):
            if any(observed[k] != value for k, value in expected.items()):
                raise ValueError("Actual exposure differs from frozen sample identity")
        log = logs[step - 1]
        for key in SUM_FIELDS:
            near(log[key], sum(r[key] for r in actual), key)
        near(log["FM"], sum(r["loss_contribution"] for r in actual), "FM contributions")
        near(log["FM"], log["all_free_velocity_sse"] / log["all_free_scalar_count"], "FM SSE denominator", rel_tol=4e-6)
        near(log["occupancy_BCE"], sum(r["BCE_contribution"] or 0. for r in actual), "head BCE contributions")
        if log["head_edge_denominator"] != sum(r["head_edge_count"] for r in actual):
            raise ValueError("Head edge denominator differs")
        if log["head_condition_count"] != sum(bool(r["head_forward_called"]) for r in actual):
            raise ValueError("Head open-condition count differs")
        for key in CONFUSION:
            if log["head_confusion"][key] != sum(r.get("head_confusion", {}).get(key, 0) for r in actual):
                raise ValueError("Head confusion accounting differs")
        if sum(log["head_confusion"].values()) != log["head_edge_denominator"]:
            raise ValueError("Head confusion denominator differs")
    if final:
        expected_micro = [(r["local_step"], r["microbatch"], r["condition_id"]) for r in visible]
        expected_head = [(r["local_step"], r["microbatch"], r["condition_id"]) for r in visible if r["head_forward_called"]]
        for kind in ("train_forward", "train_backward", "head_forward", "head_backward"):
            actual = [(r["local_step"], r["microbatch"], r["condition_id"]) for r in sequence[kind]]
            if actual != (expected_head if kind.startswith("head") else expected_micro):
                raise ValueError("Ledger call sequence differs from actual frozen exposures: " + kind)
        for kind in ("optimizer_update", "head_optimizer_update"):
            if [r["local_step"] for r in sequence[kind]] != list(range(1, 1501)):
                raise ValueError("Optimizer sequence differs")

    by_condition = {r["condition_id"]: r for r in manifest["train"]}
    planned_conditions = Counter(r["condition_id"] for b in schedule["steps"] for r in b["samples"])
    planned_parents = Counter(r["parent_id"] for b in schedule["steps"] for r in b["samples"])
    aggregates = {"condition": defaultdict(Counter), "parent": defaultdict(Counter)}
    for row in visible:
        for unit, ident in (("condition", row["condition_id"]), ("parent", row["parent_id"])):
            acc = aggregates[unit][ident]
            acc["actual_instances"] += 1
            acc["identity_instances" if row["augmentation"] == "CLEAN" else "anisotropic_instances"] += 1
            acc["closed_instances"] += int(row["mechanism"]["closed_C"])
            acc["head_forward_calls"] += int(row["head_forward_called"])
            acc["head_edge_denominator"] += row["head_edge_count"]
            acc["head_weighted_BCE_sum"] += (row["BCE_contribution"] or 0.) * logs[row["local_step"] - 1]["head_edge_denominator"]
            for key in SUM_FIELDS:
                acc[key] += row[key]
            for key in CONFUSION:
                acc["head_" + key] += row.get("head_confusion", {}).get(key, 0)
    exposure_tables = {}
    for unit, planned in (("condition", planned_conditions), ("parent", planned_parents)):
        table = []
        for ident in sorted(planned):
            acc = aggregates[unit][ident]
            row = dict(unit=unit, identifier=ident, planned_instances=planned[ident],
                       **{key: acc[key] for key in ("actual_instances", "identity_instances", "anisotropic_instances",
                          "closed_instances", "head_forward_calls", "head_edge_denominator", "head_weighted_BCE_sum") + SUM_FIELDS + tuple("head_" + k for k in CONFUSION)})
            if unit == "condition":
                row.update({key: by_condition[ident][key] for key in ("parent_id", "part_type", "N", "K")})
            row["planned_exposure_fulfilled"] = row["actual_instances"] == row["planned_instances"]
            for prefix, sse, denominator in (("free", "all_free_velocity_sse", "all_free_scalar_count"),
                                              ("tip", "seam_tip_velocity_sse", "seam_tip_scalar_count"),
                                              ("bulk", "bulk_velocity_sse", "bulk_scalar_count")):
                row[prefix + "_velocity_MSE_pooled"] = ratio(acc[sse], acc[denominator])
                row[prefix + "_MSE_status"] = "OK" if acc[denominator] else "NA_NO_FREE_SCALARS_IN_STRATUM"
            row["head_weighted_BCE_edge_pooled"] = ratio(acc["head_weighted_BCE_sum"], acc["head_edge_denominator"])
            row["head_BCE_status"] = "OK" if acc["head_edge_denominator"] else "NA_NO_C_BOUNDARY_EDGES"
            row["head_owner_recall"] = ratio(acc["head_TP"], acc["head_TP"] + acc["head_FN"])
            row["head_owner_recall_status"] = "OK" if acc["head_TP"] + acc["head_FN"] else "NA_NO_OWNER_LABELS"
            row["head_inactive_recall"] = ratio(acc["head_TN"], acc["head_TN"] + acc["head_FP"])
            row["head_inactive_recall_status"] = "OK" if acc["head_TN"] + acc["head_FP"] else "NA_NO_INACTIVE_LABELS"
            table.append(row)
        if final and not all(r["planned_exposure_fulfilled"] for r in table):
            raise ValueError("Final parent/condition exposure differs from schedule")
        exposure_tables[unit] = table

    windows = []
    for start in range(0, cutoff, 100):
        block = logs[start:start + 100]
        sums = {key: sum(r[key] for r in block) for key in SUM_FIELDS}
        confusion = {key: sum(r["head_confusion"][key] for r in block) for key in CONFUSION}
        windows.append(dict(start_step=start + 1, end_step=start + len(block), updates=len(block),
            FM_batch_mean=sum(r["FM"] for r in block) / len(block),
            free_velocity_MSE_pooled=ratio(sums["all_free_velocity_sse"], sums["all_free_scalar_count"]),
            tip_velocity_MSE_pooled=ratio(sums["seam_tip_velocity_sse"], sums["seam_tip_scalar_count"]),
            bulk_velocity_MSE_pooled=ratio(sums["bulk_velocity_sse"], sums["bulk_scalar_count"]),
            head_weighted_BCE_batch_mean=sum(r["occupancy_BCE"] for r in block) / len(block),
            head_owner_recall=ratio(confusion["TP"], confusion["TP"] + confusion["FN"]),
            head_inactive_recall=ratio(confusion["TN"], confusion["TN"] + confusion["FP"]),
            **{"head_" + k: v for k, v in confusion.items()}, **sums,
            input_seconds=sum(r["input_seconds"] for r in block),
            update_seconds=sum(r["update_seconds"] for r in block)))

    inputs = [reg_path, status_path, schedule_path, manifest_path,
              folder / "training_log.jsonl", folder / "exposure.jsonl", folder / "calls.jsonl",
              folder / "checkpoint_step0.json", Path(__file__)]
    dispatch_path = out / "training_dispatch_go.json"
    dispatch = read_json(dispatch_path)
    if dispatch["training_registration_sha256"] != state["registration_sha256"] or dispatch["actual_gpu_workers"] != 1:
        raise ValueError("Dispatch identity/authorized worker count differs")
    inputs.append(dispatch_path)
    initial = read_json(folder / "checkpoint_step0.json")
    if (initial["generator_state_sha256"] != reg["source_generator_sha256"]
            or initial["head_state_sha256"] != reg["head_state_sha256"]
            or initial["optimizer_state_sha256"] != reg["source_optimizer_sha256"]
            or initial["rng_sha256"] != reg["source_rng_sha256"]):
        raise ValueError("Initial generator/head/optimizer/RNG identity differs")
    attempts, resources, initialization = [], [], []
    for attempt in ledger["attempt_ids"]:
        adir = folder / "attempts" / attempt
        for name, target in (("final_status.json", attempts), ("resources.json", resources), ("initialization.json", initialization)):
            path = adir / name
            if path.exists():
                target.append(dict(attempt=attempt, data=read_json(path)))
                inputs.append(path)
        if final and not (adir / "final_status.json").exists():
            raise RuntimeError("Wait for finalized attempt metadata before --final")
    if final and (not attempts or attempts[-1]["data"]["status"] != "COMPLETE"):
        raise RuntimeError("Final attempt must be closed COMPLETE")
    checkpoint = export = None
    milestone_receipts = [initial]
    if final:
        middle_path = folder / "checkpoint_step500.json"
        middle = read_json(middle_path)
        if (middle["local_step"], middle["global_step"], middle["head_step"]) != (500, 12500, 500):
            raise ValueError("Immutable step500 receipt cursor differs")
        if resolve(middle["path"]).stat().st_size != middle["bytes"]:
            raise ValueError("Immutable step500 checkpoint size differs")
        milestone_receipts.append(middle)
        inputs.append(middle_path)
        for name, status_key in (("checkpoint_step1500.json", "checkpoint"), ("generator_step1500.json", "inference_export")):
            path = folder / name
            receipt = read_json(path)
            if receipt != state[status_key] or (receipt["local_step"], receipt["global_step"], receipt["head_step"]) != (1500, 13500, 1500):
                raise ValueError("Final status/receipt identity differs")
            weight_path = resolve(receipt["path"])
            if not weight_path.is_file() or weight_path.stat().st_size != receipt["bytes"]:
                raise ValueError("Final checkpoint file absent or size differs")
            inputs.append(path)
            if status_key == "checkpoint":
                checkpoint = receipt
                milestone_receipts.append(receipt)
            else:
                export = receipt
        if checkpoint["generator_state_sha256"] != export["generator_state_sha256"] or checkpoint["head_state_sha256"] != export["head_state_sha256"]:
            raise ValueError("Final generator/head export identity differs")
    own_peaks = [r["data"]["summary"]["process_peak_mib"] for r in resources if r["data"]["summary"]["process_peak_mib"] is not None]
    device_peaks = [r["data"]["summary"]["total_device_used_peak_mib"] for r in resources if r["data"]["summary"]["total_device_used_peak_mib"] is not None]
    owner_seconds = sum(r["data"]["elapsed_seconds"] for r in attempts) if final else state.get("elapsed_seconds")
    input_seconds, update_seconds = sum(r["input_seconds"] for r in logs), sum(r["update_seconds"] for r in logs)
    total_confusion = {key: sum(r["head_confusion"][key] for r in logs) for key in CONFUSION}
    all_sums = {key: sum(r[key] for r in logs) for key in SUM_FIELDS}
    summary = dict(schema=SCHEMA, mode="FINAL" if final else "READ_ONLY_SNAPSHOT", status=state["status"],
        experiment=reg["experiment"], arm=ARM, completed_local_step=cutoff, target_local_step=1500,
        global_step=12000 + cutoff, head_step=state["head_step"], observed_exposures=len(visible),
        snapshot_consistency="Completed status/log prefix; ledger may show the currently executing next step" if not final else "All1500steps/12000instances/full call ledger reconciled",
        ignored_partial_last_line=dict(training_log=partial_log, exposure=partial_exposure, calls=partial_calls),
        pool=dict(parents=27, conditions=92, old_conditions=91, new_conditions=1, new_independent_parents=1),
        calls=ledger, losses=dict(last_step=step_metrics(logs[-1]) if logs else None, windows100=windows,
            full_stream_sums=all_sums, full_stream_head_confusion=total_confusion,
            full_stream_head_owner_recall=ratio(total_confusion["TP"], total_confusion["TP"] + total_confusion["FN"]),
            full_stream_head_inactive_recall=ratio(total_confusion["TN"], total_confusion["TN"] + total_confusion["FP"])),
        per_parent_exposure=exposure_tables["parent"], per_condition_exposure=exposure_tables["condition"],
        identity=dict(initial_checkpoint_receipt=initial, final_checkpoint_receipt=checkpoint,
            final_generator_head_export_receipt=export, initialization=initialization,
            immutable_milestone_receipts=milestone_receipts,
            head_class_weights=reg["head_class_weights"], source_head_historical_updates=2996,
            checkpoint_verification_scope="Receipt identity and filesystem size; full tensor/checkpoint content was not loaded or rehashed"),
        time=dict(owner_elapsed_seconds=owner_seconds, completed_batch_input_seconds=input_seconds,
            completed_batch_update_seconds=update_seconds,
            non_batch_overhead_seconds=(owner_seconds - input_seconds - update_seconds) if final else None,
            completed_call_seconds_by_kind=ledger["completed_call_seconds"],
            updates_per_owner_second=ratio(cutoff, owner_seconds),
            initialization_checkpoint_IO_phase_seconds=None,
            initialization_checkpoint_IO_status="NA_NOT_TIMED_SEPARATELY; included in owner/non-batch overhead",
            timing_caveat="Observed worker wall time under actual background/CPU concurrency; not an isolated compute comparison"),
        memory=dict(actual_gpu_workers=dispatch["actual_gpu_workers"],
            dispatch_plan=dict(reason=dispatch["reason"], retained_headroom_mib=dispatch["retained_headroom_mib"],
                measured_first_update_required=dispatch["measured_first_update_required"],
                resource_record=str(dispatch_path),
                **{k: dispatch["measured_resources"].get(k) for k in ("measured_total_mib", "measured_used_mib", "measured_free_mib",
                    "GPU_utilization", "required_free_mib", "process_peak_planning_mib", "process_peak_is_measured", "allocation_reference")}),
            configured_training_CPU_threads=2, summary_CPU_thread_environment=1,
            first_formal_update_peak_allocated_bytes=logs[0]["peak_allocated_bytes"] if logs else None,
            first_formal_update_peak_reserved_bytes=logs[0]["peak_reserved_bytes"] if logs else None,
            peak_allocator_allocated_bytes=max((r["peak_allocated_bytes"] for r in logs), default=None),
            peak_allocator_reserved_bytes=max((r["peak_reserved_bytes"] for r in logs), default=None),
            allocator_scope="Reset within each formal batch, including generator/head backward and both AdamW updates; not total process VRAM",
            observed_process_peak_mib=max(own_peaks) if own_peaks else None,
            process_memory_status="OBSERVED" if own_peaks else "UNAVAILABLE_NVIDIA_SMI" if final else "PENDING_FINAL_OBSERVER_RECEIPT",
            observed_total_device_used_peak_mib=max(device_peaks) if device_peaks else None,
            device_scope="Entire GPU including unrelated/background processes; not attributed to this worker",
            per_attempt_observer_summaries=[dict(attempt=r["attempt"], **r["data"]["summary"]) for r in resources]),
        limitations=["Online training losses/confusion across changing weights and augmented instances; not final-checkpoint test accuracy or generation quality.",
            "Same COMMON12000 generator/A-step0 fresh AdamW and imported OLD_S15000_HEAD with a fresh independent head optimizer; head source has2996 historical updates.",
            "New parent/pool exposure and head adaptation change jointly; comparison with RELATION is a recipe comparison, not a causal single-parent data ablation.",
            "Inactive labels include true Null and alias. BCE/confusion cannot identify these roles separately from the logged binary labels.",
            "Closed C has no head call or BCE; tip/head missing denominators remain NA. No extra probe, model forward, OT or training was run for this report."],
        new_neural_forwards=0, new_training_updates=0, tensor_checkpoints_loaded=0)
    return summary, [step_metrics(r) for r in logs], windows, exposure_tables, inputs


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Existing summary artifact differs; do not overwrite: " + str(path))
        return
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def rolling_ratio(rows, numerator, denominator=None, window=100):
    queue, a, b, result = deque(), 0., 0., []
    for row in rows:
        pair = (float(numerator(row)), float(denominator(row)) if denominator else 1.)
        queue.append(pair)
        a, b = a + pair[0], b + pair[1]
        if len(queue) > window:
            old = queue.popleft()
            a, b = a - old[0], b - old[1]
        result.append(a / b if b else math.nan)
    return result


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["local_step"] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    series = (("FM: original free-corner objective", "FM", None),
              ("Seam-tip velocity MSE (NA when no tips)", "seam_tip_velocity_sse", "seam_tip_scalar_count"),
              ("Bulk velocity MSE", "bulk_velocity_sse", "bulk_scalar_count"),
              ("Independent head: frozen weighted BCE", "occupancy_BCE", None))
    for ax, (title, num, den) in zip(axes.flat, series):
        raw = [ratio(r[num], r[den]) if den else r[num] for r in rows]
        ax.plot(x, raw, color="#879bb0", alpha=.28, lw=.6, label="Formal batch")
        smooth = rolling_ratio(rows, lambda r, key=num: r[key], (lambda r, key=den: r[key]) if den else None)
        ax.plot(x, smooth, color="#175887", lw=1.8, label="Trailing100 updates" + ("; pooled scalars" if den else "; batch mean"))
        ax.set_title(title)
        ax.legend(fontsize=8)
    ax = axes[2, 0]
    ax.plot(x, rolling_ratio(rows, lambda r: r["head_TP"], lambda r: r["head_TP"] + r["head_FN"]), label="Owner recall", color="#166d8c")
    ax.plot(x, rolling_ratio(rows, lambda r: r["head_TN"], lambda r: r["head_TN"] + r["head_FP"]), label="Inactive recall (Null + alias)", color="#bc6c25")
    ax.set(title="Online head confusion: pooled trailing100 updates", ylim=(-.02, 1.02))
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    for key, label, color in (("generator_gradient_norm", "Generator", "#175887"), ("head_gradient_norm", "Head", "#bc6c25")):
        ax.plot(x, rolling_ratio(rows, lambda r, key=key: r[key] or 0.), label=label, color=color)
    ax.set_title("Gradient norm before clip1: trailing100 mean")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Completed local update")
        ax.grid(alpha=.2)
    fig.suptitle("Bench new-data joint adaptation | 1500 formal updates\nOnline training diagnostics; no fixed-probe or final-test claim", fontsize=14)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, metadata={"Software": SCHEMA})
    plt.close(fig)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--snapshot", action="store_true")
    modes.add_argument("--final", action="store_true")
    parser.add_argument("--root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = resolve(args.root)
    receipt_path = out / "results/training_summary_receipt.json"
    if args.final and receipt_path.exists():
        receipt = read_json(receipt_path)
        for path, expected in {**receipt["inputs"], **receipt["outputs"]}.items():
            if sha(resolve(path)) != expected:
                raise ValueError("Completed report input/output changed: " + path)
        print(json.dumps(dict(status="COMPLETE_VERIFIED", summary=str(out / "results/training_summary.json"), new_neural_forwards=0)))
        return
    summary, steps, windows, tables, inputs = summarize(out, final=args.final)
    if args.snapshot:
        print(json.dumps({key: summary[key] for key in ("status", "completed_local_step", "target_local_step", "head_step", "observed_exposures", "snapshot_consistency", "ignored_partial_last_line", "calls", "time", "memory")}
            | dict(last_step=summary["losses"]["last_step"], mode="READ_ONLY_SNAPSHOT", files_written=0), allow_nan=False))
        return
    outputs = {
        out / "results/training_summary.json": json_bytes(summary),
        out / "results/training_summary.csv": csv_bytes(steps),
        out / "results/training_windows100.csv": csv_bytes(windows),
        out / "results/training_per_parent.csv": csv_bytes(tables["parent"]),
        out / "results/training_per_condition.csv": csv_bytes(tables["condition"]),
        out / "results/loss_curves.png": plot(steps),
    }
    for path, data in outputs.items():
        write_once(path, data)
    receipt = dict(schema=SCHEMA, status="COMPLETE", inputs={str(p): sha(p) for p in inputs},
                   outputs={str(p): sha(p) for p in outputs}, new_neural_forwards=0, new_training_updates=0)
    write_once(receipt_path, json_bytes(receipt))
    print(json.dumps(dict(status="COMPLETE", summary=str(out / "results/training_summary.json"),
                         outputs=len(outputs), observed_exposures=summary["observed_exposures"], calls=summary["calls"])))


if __name__ == "__main__":
    main()
