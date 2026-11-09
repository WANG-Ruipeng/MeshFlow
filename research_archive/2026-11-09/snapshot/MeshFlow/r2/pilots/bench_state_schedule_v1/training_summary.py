"""Read-only training-log audit and CPU summaries; never imports torch/models.

Formal artifacts are written only after BOTH registered arms and the reused A
baseline have completed.  This summarizes changing training rows, not a fixed
neural probe or evidence that one arm's native loss is superior to another's.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

from .common import ARMS, OUT, ROOT, read, resolve, sha

BASE = ROOT / "experiments/r2_bench_adaptation_strength_v1"
BASE_ARM = "A_COMMON_LR1E5"
LABELS = (BASE_ARM,) + ARMS
UPDATES, BATCH, START = 1500, 8, 12000
METRICS = {
    "bulk_local_velocity_mse": ("bulk_velocity_sse", "bulk_scalar_count"),
    "active_tip_local_velocity_mse": ("seam_tip_velocity_sse", "seam_tip_scalar_count"),
    "bulk_endpoint_mse": ("bulk_endpoint_sse", "bulk_scalar_count"),
    "active_tip_endpoint_mse": ("seam_tip_active_endpoint_sse", "seam_tip_scalar_count"),
}
SUM_FIELDS = (
    "all_free_velocity_sse", "all_free_scalar_count", "active_scalar_count",
    "seam_tip_velocity_sse", "seam_tip_scalar_count", "inactive_tip_scalar_count",
    "bulk_velocity_sse", "bulk_scalar_count", "active_endpoint_sse",
    "seam_tip_active_endpoint_sse", "bulk_endpoint_sse",
    "global_equivalent_velocity_sse",
)
WARNING = (
    "Native objectives differ across arms and are not a cross-arm quality ranking. "
    "First/last windows contain changing training examples/times, not a fixed probe. "
    "ASYNC active-tip summaries exclude late teacher-forced tips; local velocity "
    "units and endpoint remaining times follow its own clock. Geometric rollout "
    "evaluation, not these training curves, decides the endpoint result."
)


def directory(arm):
    return (BASE if arm == BASE_ARM else OUT) / "training" / arm


def canonical_hash(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jsonl(path):
    rows = []
    with resolve(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Incomplete/corrupt JSONL {path}:{line_number}") from exc
    return rows


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = sorted(float(v) for v in values if v is not None)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite measured statistic")
    if not values:
        return dict(n=0, status="NA_NO_DEFINED_VALUES")
    def quantile(q):
        position = q * (len(values) - 1)
        lo = math.floor(position)
        hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return dict(n=len(values), min=values[0], p05=quantile(.05),
                median=quantile(.5), mean=math.fsum(values)/len(values),
                p95=quantile(.95), max=values[-1])


def normalized_exposure(row, arm):
    """Only baseline endpoint errors are algebraically derived from saved SSE."""
    out = dict(row)
    if arm == BASE_ARM:
        factor = (1. - float(row["t"])) ** 2
        out.update(
            active_scalar_count=row["all_free_scalar_count"],
            inactive_tip_scalar_count=0,
            bulk_endpoint_sse=row["bulk_velocity_sse"] * factor,
            seam_tip_active_endpoint_sse=row["seam_tip_velocity_sse"] * factor,
            active_endpoint_sse=row["all_free_velocity_sse"] * factor,
            global_equivalent_velocity_sse=row["all_free_velocity_sse"],
        )
    for key in SUM_FIELDS:
        if key not in out or not math.isfinite(float(out[key])) or out[key] < 0:
            raise ValueError(f"Missing/nonfinite/negative exposure field {arm}/{key}")
    return out


def summarize_window(rows):
    result = dict(updates=len(rows), first_local_step=rows[0]["local_step"],
                  last_local_step=rows[-1]["local_step"],
                  first_global_step=rows[0]["global_step"],
                  last_global_step=rows[-1]["global_step"])
    for name, (sse, count) in METRICS.items():
        numerator = math.fsum(r[sse] for r in rows)
        denominator = sum(r[count] for r in rows)
        result[name] = dict(sse=numerator, scalar_count=denominator,
                            mse=fraction(numerator, denominator))
    result["native_objective_mean_not_cross_arm_rank"] = (
        math.fsum(r["native_objective"] for r in rows) / len(rows))
    return result


def audit_calls(rows, arm, steps=UPDATES, allow_live=False):
    """Pair every actual call event; also check exact scientific-call identities."""
    begun, done, failures = {}, {}, []
    for row in rows:
        key = (row.get("attempt_id"), row["call_id"])
        event = row["event"]
        if event == "CALL_BEGIN":
            if key in begun:
                raise ValueError(f"Duplicate CALL_BEGIN {arm}/{key}")
            begun[key] = row
        elif event == "CALL_DONE":
            if key in done:
                raise ValueError(f"Duplicate CALL_DONE {arm}/{key}")
            done[key] = row
        else:
            failures.append(row)
    if failures or (not allow_live and set(begun) != set(done)):
        raise ValueError(f"Failed/unpaired calls in {arm}")
    if not set(done) <= set(begun):
        raise ValueError(f"CALL_DONE without BEGIN in {arm}")
    identities = collections.defaultdict(list)
    seconds = collections.defaultdict(float)
    for key, row in done.items():
        start = begun[key]
        if (start["kind"], start["context"]) != (row["kind"], row["context"]):
            raise ValueError(f"Call identity changed {arm}/{key}")
        context = row["context"]
        kind = row["kind"]
        if kind.startswith("train_") or kind == "optimizer_update":
            if context["local_step"] <= steps:
                if context["global_step"] != START + context["local_step"]:
                    raise ValueError("Call global-step identity differs")
                identities[kind].append(
                    (context["local_step"], context.get("microbatch"))
                )
                seconds[kind] += float(row["seconds"])
    expected_samples = [(step, mi) for step in range(1, steps + 1) for mi in range(BATCH)]
    expected_updates = [(step, None) for step in range(1, steps + 1)]
    for kind, expected in (("train_forward", expected_samples), ("train_backward", expected_samples),
                           ("optimizer_update", expected_updates)):
        if identities[kind] != expected:
            raise ValueError(f"Actual call stream differs {arm}/{kind}")
    all_counts = collections.Counter(row["kind"] for row in done.values())
    if arm != BASE_ARM and set(all_counts) - {"train_forward", "train_backward", "optimizer_update"}:
        raise ValueError("Unexpected new scientific call kind")
    return dict(status="PASS", actual_successful_counts=dict(all_counts),
                committed_training_counts={k: len(v) for k, v in identities.items()},
                committed_call_seconds=dict(seconds), pending=len(set(begun) - set(done)),
                reused_baseline_probe_calls=int(all_counts.get("probe_forward", 0)))


def expected_rows(schedule, steps):
    return [(e["step"], mi, row) for e in schedule[:steps]
            for mi, row in enumerate(e["samples"])]


def audit_arm(arm, schedule, baseline_rows=None, prefix=None):
    root = directory(arm)
    status = read(root / "status.json")
    steps = UPDATES if prefix is None else int(prefix)
    if prefix is None and (status["status"], status["local_step"], status["global_step"]) != (
            "COMPLETE", UPDATES, START + UPDATES):
        raise RuntimeError(f"Formal summary requires COMPLETE: {arm}")
    exposure = jsonl(root / "exposure.jsonl")
    logs = jsonl(root / "training_log.jsonl")
    if prefix is not None:
        exposure = [r for r in exposure if r["local_step"] <= steps]
        logs = [r for r in logs if r["local_step"] <= steps]
    expected = expected_rows(schedule, steps)
    if len(exposure) != len(expected) or [r["local_step"] for r in logs] != list(range(1, steps + 1)):
        raise ValueError(f"Exposure/update count or order differs {arm}")
    projected = []
    for i, (row, (step, mi, original)) in enumerate(zip(exposure, expected)):
        if (row["local_step"], row["microbatch"], row["global_step"]) != (step, mi, START + step):
            raise ValueError(f"Exposure cursor differs {arm}/{i}")
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError(f"Frozen input row differs {arm}/{i}")
        projected.append({key: row[key] for key in original})
        if baseline_rows is not None:
            for key in ("t", "N", "K", "all_free_scalar_count", "bulk_scalar_count"):
                if row[key] != baseline_rows[i][key]:
                    raise ValueError(f"A stream/sample identity differs {arm}/{i}/{key}")
    norm = [normalized_exposure(r, arm) for r in exposure]
    per_update = []
    for offset, log in enumerate(logs):
        group = norm[BATCH * offset:BATCH * (offset + 1)]
        sums = {field: math.fsum(r[field] for r in group) for field in SUM_FIELDS}
        for key, value in sums.items():
            if key in log and not math.isclose(value, log[key], rel_tol=2e-9, abs_tol=2e-8):
                raise ValueError(f"Exposure/log sum differs {arm}/{offset+1}/{key}")
        denominator = log["free_coordinate_denominator"]
        objective = log.get("objective", log.get("FM"))
        if denominator != sums["all_free_scalar_count"]:
            raise ValueError("Original free denominator changed")
        if not math.isclose(sums["all_free_velocity_sse"]/denominator, objective,
                            rel_tol=5e-6, abs_tol=1e-7):
            raise ValueError("Actual objective accounting differs")
        result = dict(arm=arm, local_step=offset+1, global_step=START+offset+1,
                      native_objective=objective, **sums,
                      free_coordinate_denominator=denominator,
                      generator_gradient_norm=log["generator_gradient_norm"],
                      input_seconds=log["input_seconds"], update_seconds=log["update_seconds"],
                      peak_allocated_bytes=log["peak_allocated_bytes"],
                      peak_reserved_bytes=log["peak_reserved_bytes"],
                      endpoint_provenance="ALGEBRA_FROM_SAVED_SSE" if arm == BASE_ARM else "SAVED_EXPOSURE")
        for name, (sse, count) in METRICS.items():
            result[name] = fraction(result[sse], result[count])
        per_update.append(result)
    calls = audit_calls(jsonl(root / "calls.jsonl"), arm, steps, allow_live=prefix is not None)
    return dict(arm=arm, rows=norm, updates=per_update, status=status, calls=calls,
                stream_sha256=canonical_hash(projected), checked_rows=len(projected))


def mechanism_summary(rows, arm):
    if arm == BASE_ARM:
        return dict(status="NA_BASELINE_UNMODIFIED_SYNCHRONOUS")
    if arm == "PERTURBED_STATE":
        records = [r["mechanism"] for r in rows]
        eligible = [r for r in records if r["eligible"]]
        applied = [r for r in records if r["applied"]]
        if any(r["applied"] and not r["eligible"] for r in records):
            raise ValueError("Perturbation outside eligible interval")
        for row in records:
            if not all(row[k] for k in ("fixed_coordinates_unchanged", "fixed_velocity_target_zero",
                                       "free_denominator_unchanged")) or row["global_rng_touched"]:
                raise ValueError("Recorded perturbation preservation violation")
        stats = {k: distribution(r[k] for r in applied) for k in (
            "alpha", "d_norm_model", "state_shift_norm_model",
            "target_velocity_l2_gain", "target_velocity_mse_gain", "endpoint_identity_max_abs")}
        old = math.fsum(r["original_target_velocity_sse"] for r in applied)
        new = math.fsum(r["target_velocity_sse"] for r in applied)
        return dict(status="MEASURED", exposures=len(rows), eligible=len(eligible), applied=len(applied),
                    eligible_rate=fraction(len(eligible), len(rows)),
                    applied_rate_all=fraction(len(applied), len(rows)),
                    applied_rate_given_eligible=fraction(len(applied), len(eligible)),
                    nominal_probability_given_eligible=.25, applied_distributions=stats,
                    applied_aggregate_target_sse_ratio=fraction(new, old),
                    applied_aggregate_target_l2_ratio=math.sqrt(new/old) if old else None,
                    gain_gt_2=sum(r["target_velocity_mse_gain"] is not None and
                                  r["target_velocity_mse_gain"] > 2 for r in applied),
                    gain_gt_10=sum(r["target_velocity_mse_gain"] is not None and
                                   r["target_velocity_mse_gain"] > 10 for r in applied),
                    no_gain_defined=sum(r["target_velocity_mse_gain"] is None for r in applied))
    active = sum(r["seam_tip_scalar_count"] for r in rows)
    inactive = sum(r["inactive_tip_scalar_count"] for r in rows)
    return dict(status="MEASURED", exposures=len(rows),
                rows_with_active_tip=sum(r["seam_tip_scalar_count"] > 0 for r in rows),
                rows_with_frozen_tip=sum(r["inactive_tip_scalar_count"] > 0 for r in rows),
                active_tip_scalars=active, frozen_tip_scalars=inactive,
                active_fraction_of_tip_scalars=fraction(active, active+inactive),
                source_time_distribution=distribution(r["t"] for r in rows),
                tip_clock_distribution=distribution(r["mechanism"]["tip_clock"] for r in rows),
                no_assumption_active_exposure_is_70_percent=True)


def per_parent(arm_data):
    grouped = collections.defaultdict(list)
    for row in arm_data["rows"]:
        grouped[row["parent_id"]].append(row)
    result = []
    for parent, rows in sorted(grouped.items()):
        arm = arm_data["arm"]
        active = sum(r["seam_tip_scalar_count"] for r in rows)
        inactive = sum(r["inactive_tip_scalar_count"] for r in rows)
        mechanisms = [r.get("mechanism", {}) for r in rows]
        result.append(dict(
            arm=arm, parent_id=parent, exposures=len(rows),
            distinct_conditions=len({r["condition_id"] for r in rows}),
            first_local_step=min(r["local_step"] for r in rows),
            last_local_step=max(r["local_step"] for r in rows),
            original_free_scalar_exposures=sum(r["all_free_scalar_count"] for r in rows),
            active_tip_scalar_exposures=active, frozen_tip_scalar_exposures=inactive,
            active_tip_fraction=fraction(active, active+inactive),
            rows_with_active_tip=sum(r["seam_tip_scalar_count"] > 0 for r in rows),
            rows_with_frozen_tip=sum(r["inactive_tip_scalar_count"] > 0 for r in rows),
            perturbation_eligible=sum(bool(r.get("eligible")) for r in mechanisms),
            perturbation_applied=sum(bool(r.get("applied")) for r in mechanisms),
        ))
    return result


def check_initialization(reg, sources):
    base = read(BASE / "training" / BASE_ARM / "checkpoint_step0.json")
    expected = {
        "generator_state_sha256": reg["source_generator_sha256"],
        "optimizer_state_sha256": reg["source_optimizer_sha256"],
        "rng_sha256": reg["source_rng_sha256"],
    }
    identities = {}
    for arm in LABELS:
        path = directory(arm) / "checkpoint_step0.json"
        value = read(path)
        if value["local_step"] != 0 or value["global_step"] != START:
            raise ValueError("Initial cursor mismatch")
        if any(value[key] != target for key, target in expected.items()):
            raise ValueError("A0 generator/optimizer/RNG mismatch: " + arm)
        sources[str(path)] = sha(path)
        identities[arm] = {key: value[key] for key in expected}
    if base["sha256"] != reg["source_checkpoint_sha256"]:
        raise ValueError("Registered source is not exact A0 file")
    # Hash source/initial files as bytes without loading any tensor or model.
    weight_hashes = {}
    for arm in LABELS:
        receipt = read(directory(arm) / "checkpoint_step0.json")
        path = resolve(receipt["path"])
        actual = sha(path)
        if actual != receipt["sha256"]:
            raise ValueError("Initial checkpoint file changed: " + arm)
        weight_hashes[arm] = dict(path=str(path), sha256=actual)
        sources[str(path)] = actual
    return dict(status="PASS", equal_internal_identities=identities,
                verified_checkpoint_file_hashes=weight_hashes,
                different_container_file_hashes_are_expected=True,
                no_tensor_deserialization=True)


def immutable_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("Refusing to overwrite different summary artifact: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def curve_bytes(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True, squeeze=False)
    for column, item in enumerate(data):
        rows = item["updates"]
        x = [r["global_step"] for r in rows]
        for index, (name, (sse, count)) in enumerate(METRICS.items()):
            axis = axes[index, column]
            y = []
            for end in range(len(rows)):
                window = rows[max(0, end-24):end+1]
                y.append(fraction(math.fsum(r[sse] for r in window),
                                  sum(r[count] for r in window)))
            axis.plot(x, y, linewidth=1, color=("#4477AA", "#EE6677", "#228833")[column])
            axis.grid(alpha=.2)
            axis.set_ylabel(name.replace("_", " "), fontsize=8)
            if index == 0:
                axis.set_title(item["arm"], fontsize=10)
            if index == 3:
                axis.set_xlabel("Global update (12000 + local)")
    fig.suptitle("Within-arm training trends: trailing 25 updates, scalar-weighted MSE\n"
                 "Different states/clocks/objectives; panel scales are independent; not a quality ranking",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=150,
                metadata={"Software": "MeshFlow Bench CPU training-summary v1"})
    plt.close(fig)
    return output.getvalue()


def formal_summary():
    # Guard before creating ANY output directory or artifact.
    states = {arm: read(directory(arm) / "status.json") for arm in LABELS}
    if any((s["status"], s["local_step"], s["global_step"]) != (
            "COMPLETE", UPDATES, START+UPDATES) for s in states.values()):
        raise RuntimeError("NOT_READY: both new arms and baseline must be COMPLETE; wrote nothing")
    dest = OUT / "diagnostics"
    summary_path = dest / "training_summary.json"
    if summary_path.exists():
        old = read(summary_path)
        for path, expected in {**old["source_hashes"], **old["artifact_hashes"]}.items():
            if sha(path) != expected:
                raise ValueError("Previously finalized training summary evidence changed: " + path)
        return dict(status="VERIFIED_EXISTING", path=str(summary_path), sha256=sha(summary_path))
    reg = read(OUT / "training_registration.json")
    old_reg = read(BASE / "training_registration.json")
    sources = {str(OUT/"training_registration.json"): sha(OUT/"training_registration.json"),
               str(BASE/"training_registration.json"): sha(BASE/"training_registration.json"),
               str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for key in ("manifest", "input_schedule"):
        path = resolve(reg[key])
        if reg[key+"_sha256"] != old_reg[key+"_sha256"] or sha(path) != reg[key+"_sha256"]:
            raise ValueError("Original A shared asset mismatch: " + key)
        sources[str(path)] = sha(path)
    if reg["model_config"] != old_reg["model_config"]:
        raise ValueError("Original A model configuration differs")
    for path, expected in reg["training_code_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Frozen training source modified: " + path)
        sources[str(resolve(path))] = expected
    init = check_initialization(reg, sources)
    schedule = read(reg["input_schedule"])["steps"]
    if [r["step"] for r in schedule[:UPDATES]] != list(range(1, UPDATES+1)):
        raise ValueError("Schedule update sequence differs")
    data = [audit_arm(BASE_ARM, schedule)]
    for arm in ARMS:
        data.append(audit_arm(arm, schedule, baseline_rows=data[0]["rows"]))
    if len({d["stream_sha256"] for d in data}) != 1:
        raise ValueError("Three actual row streams differ")
    arm_summaries = {}
    windows_csv = []
    mechanism_bins = []
    for item in data:
        arm = item["arm"]
        for name in ("status.json", "calls.jsonl", "exposure.jsonl", "training_log.jsonl"):
            path = directory(arm) / name
            sources[str(path)] = sha(path)
        first, last = (summarize_window(item["updates"][sl])
                       for sl in (slice(0, 100), slice(-100, None)))
        trends = {}
        for name in METRICS:
            a, b = first[name]["mse"], last[name]["mse"]
            trends[name] = dict(first100=a, last100=b,
                                absolute_change=b-a if a is not None and b is not None else None,
                                last_over_first=fraction(b, a) if b is not None else None)
            for label, window in (("first100", first), ("last100", last)):
                windows_csv.append(dict(arm=arm, window=label, metric=name,
                    first_global_step=window["first_global_step"],
                    last_global_step=window["last_global_step"], **window[name]))
        arm_summaries[arm] = dict(
            calls=item["calls"], row_stream_sha256=item["stream_sha256"],
            first100=first, last100=last, within_arm_trends=trends,
            mechanism=mechanism_summary(item["rows"], arm),
            gradient_norm=distribution(r["generator_gradient_norm"] for r in item["updates"]),
            total_logged_update_seconds=math.fsum(r["update_seconds"] for r in item["updates"]),
            total_logged_input_seconds=math.fsum(r["input_seconds"] for r in item["updates"]),
            wall_time_comparison_warning="Concurrent contention differs from reused A timing",
        )
        for lo, hi in ((0., .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0000001)):
            selected = [r for r in item["rows"] if lo <= r["t"] < hi]
            mech = [r.get("mechanism", {}) for r in selected]
            applied = [r for r in mech if r.get("applied")]
            active = sum(r["seam_tip_scalar_count"] for r in selected)
            inactive = sum(r["inactive_tip_scalar_count"] for r in selected)
            mechanism_bins.append(dict(
                arm=arm, time_min=lo, time_max=min(hi, 1.), rows=len(selected),
                time_bin_note="descriptive decimal bins; mechanism eligibility uses its recorded FP32 gate",
                eligible=sum(bool(r.get("eligible")) for r in mech), applied=len(applied),
                applied_target_mse_gain_mean=(
                    distribution(r["target_velocity_mse_gain"] for r in applied).get("mean")),
                active_tip_scalars=active, frozen_tip_scalars=inactive,
                active_tip_scalar_fraction=fraction(active, active+inactive),
            ))
    artifacts = {
        "per_update.csv": csv_bytes([row for d in data for row in d["updates"]]),
        "per_parent_exposure.csv": csv_bytes([row for d in data for row in per_parent(d)]),
        "first_last100.csv": csv_bytes(windows_csv),
        "mechanism_by_time_bin.csv": csv_bytes(mechanism_bins),
        "training_curves.png": curve_bytes(data),
    }
    for name, content in artifacts.items():
        immutable_bytes(dest/name, content)
    summary = dict(
        schema="bench_state_schedule_training_summary_v1", status="COMPLETE",
        training_global_start=START, training_global_end=START+UPDATES,
        updates_per_arm=UPDATES, exposures_per_arm=UPDATES*BATCH,
        no_model_load=True, no_neural_forward=True, no_GPU_use=True,
        interpretation_warning=WARNING, initial_identity=init,
        row_stream_comparison=dict(status="PASS", baseline=BASE_ARM,
            actual_rows_checked_per_arm=UPDATES*BATCH, all_source_row_fields_equal=True,
            actual_t_N_K_original_free_denominator_equal=True),
        endpoint_provenance=dict(
            A="Derived per exposure from saved velocity SSE times (1-t)^2, without forward",
            PERTURBED_STATE="Saved training endpoint SSE against unchanged Y under corrective target",
            ASYNC_LEAD_SEAM="Saved active-tip local remainder (1-s); bulk (1-t); frozen tips excluded"),
        arms=arm_summaries, source_hashes=sources,
        artifact_hashes={str(dest/name): hashlib.sha256(value).hexdigest()
                         for name, value in artifacts.items()},
    )
    immutable_bytes(summary_path, (json.dumps(summary, ensure_ascii=False, indent=2,
                                            allow_nan=False)+"\n").encode("utf-8"))
    return dict(status="COMPLETE", path=str(summary_path), sha256=sha(summary_path),
                outputs=list(artifacts), no_GPU_use=True)


def inspect_prefix():
    """No writes, no checkpoint byte reads; audit only status-committed update prefixes."""
    reg = read(OUT / "training_registration.json")
    schedule = read(reg["input_schedule"])["steps"]
    baseline = audit_arm(BASE_ARM, schedule)
    result = {}
    for arm in ARMS:
        state = read(directory(arm)/"status.json")
        step = int(state["local_step"])
        if not step:
            result[arm] = dict(status=state["status"], committed_step=0, audit="WAIT_FIRST_UPDATE")
            continue
        item = audit_arm(arm, schedule, baseline_rows=baseline["rows"], prefix=step)
        result[arm] = dict(status=state["status"], committed_step=step,
                           actual_rows=item["checked_rows"], audit="PASS",
                           mechanism=mechanism_summary(item["rows"], arm))
    return dict(status="READ_ONLY_PREFIX_AUDIT", arms=result, wrote_files=False,
                model_imports=False, no_GPU_use=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-prefix", action="store_true",
                        help="Read-only audit of currently committed prefixes; writes nothing")
    args = parser.parse_args()
    print(json.dumps(inspect_prefix() if args.inspect_prefix else formal_summary(),
                     ensure_ascii=False, indent=2, allow_nan=False))
