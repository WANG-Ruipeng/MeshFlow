"""CPU-only, preregistered loss-plateau accounting for Bench part training.

No model imports, forward calls, checkpoint selection or scientific retries.
``PLATEAU`` means only this finite fixed-probe rule passed. It is neither a
mathematical convergence proof nor evidence of better generated geometry.

Checkpoint schema::

    {"step": 4000, "probe_manifest_sha256": "...",
     "train": {"ALL_free": {"sse": 1., "count": 10},
               "bulk": {"sse": 1., "count": 10},
               "seam_tip": {"sse": 0., "count": 0},
               "parts": {"SEAT_SURFACE": {"sse": 1., "count": 10}}},
     "dev": { ... same structure, diagnostic only ... }}

Counts are numbers of scalar coordinates, not faces, examples or conditions.
An undefined stratum must be null or have count=0 and SSE=0. A missing stratum
is reported as NA. This module never imputes an undefined MSE with zero.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VERSION = "bench_parts_fixed_probe_plateau_v1"
METRICS = ("ALL_free", "bulk", "seam_tip")
DEFAULT_FIELDS = {
    "ALL_free": (("all_free_velocity_sse", "all_free_scalar_count"),
                 ("all_free_velocity_sse", "free_coordinate_count"),
                 ("all_free_sse", "all_free_count")),
    "bulk": (("bulk_velocity_sse", "bulk_scalar_count"),
             ("bulk_sse", "bulk_count")),
    "seam_tip": (("seam_tip_velocity_sse", "seam_tip_scalar_count"),
                 ("source_interface_third_velocity_sse", "source_interface_third_scalar_count"),
                 ("seam_tip_sse", "seam_tip_count")),
}


def _stat(value, label):
    """Validate a sufficient-statistic pair and derive its weighted MSE."""
    if value is None:
        return {"sse": None, "count": 0, "mse": None, "status": "NA_MISSING"}
    if not isinstance(value, Mapping) or "sse" not in value or "count" not in value:
        raise ValueError(f"{label}: expected explicit SSE and scalar count")
    count = value["count"]
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError(f"{label}: count must be a nonnegative integer")
    count = int(count)
    sse = value["sse"]
    if count == 0:
        if sse not in (None, 0, 0.0):
            raise ValueError(f"{label}: nonzero SSE with zero count")
        return {"sse": 0., "count": 0, "mse": None, "status": "NA_NO_COORDINATES"}
    if sse is None or not math.isfinite(float(sse)) or float(sse) < 0:
        raise ValueError(f"{label}: finite nonnegative SSE required")
    sse = float(sse)
    return {"sse": sse, "count": count, "mse": sse/count, "status": "DEFINED"}


def _relative(before, after):
    # Preserve the exact-zero edge case; never hide a positive error behind an
    # arbitrary epsilon. null accompanied by ABOVE_ZERO has infinite relative
    # worsening and fails both the plateau and deterioration comparisons.
    if before == 0:
        return (0., "ZERO_TO_ZERO") if after == 0 else (None, "ABOVE_ZERO")
    return (after-before)/before, "DEFINED"


def _series_intervals(history, split, key, *, part=None):
    rows = []
    for left, right in zip(history, history[1:]):
        def selected(h):
            domain = h.get(split, {}) or {}
            return (domain.get("parts", {}).get(part) if part is not None
                    else domain.get(key))
        a = _stat(selected(left), f"{left['step']}/{split}/{part or key}")
        b = _stat(selected(right), f"{right['step']}/{split}/{part or key}")
        row = {"from_step": left["step"], "to_step": right["step"],
               "split": split, "metric": key, "part": part,
               "before": a, "after": b}
        if a["mse"] is None or b["mse"] is None:
            row.update(status="NA_INTERVAL_SKIPPED", relative_change=None)
        else:
            if a["count"] != b["count"]:
                raise ValueError(f"Fixed-probe scalar count changed: {split}/{part or key}")
            relative, status = _relative(a["mse"], b["mse"])
            row.update(status=status, relative_change=relative)
        rows.append(row)
    return rows


def _worsens(row, threshold):
    return (row["status"] == "ABOVE_ZERO" or
            (row["relative_change"] is not None and row["relative_change"] > threshold))


def _longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current+1 if value else 0
        longest = max(longest, current)
    return longest


def assess_plateau(history: Sequence[Mapping], *, min_updates: int,
                   max_updates: int, stage_start_step: int = 0,
                   probe_interval: int | None = None,
                   major_parts: Sequence[str] = (), relative_tolerance: float = .02,
                   part_worsening_tolerance: float = .05,
                   part_worsening_intervals: int = 3,
                   dev_warning_tolerance: float = .05):
    """Assess the latest four fixed-probe checkpoints without choosing a best.

    Budget/minimum are supplied by the experiment owner. All three adjacent
    intervals must have abs(relative change)<2% for each defined TRAIN stratum.
    A completely NA optional stratum is explicitly skipped; partially absent
    strata block plateau because they cannot establish three intervals.
    ALL_free must always be defined. A named major part blocks plateau when it
    worsens >5% for three consecutive intervals (configurable only pre-freeze).
    DEV only emits warnings, and cannot select/stop/promote an endpoint.
    """
    if not (0 <= min_updates <= max_updates) or max_updates <= 0:
        raise ValueError("Require 0 <= min_updates <= positive max_updates")
    if not 0 < relative_tolerance < 1 or not 0 < part_worsening_tolerance < 1:
        raise ValueError("Invalid preregistered relative tolerances")
    if part_worsening_intervals not in (1, 2, 3):
        raise ValueError("Part consecutive interval count must be 1..3")
    if probe_interval is not None and (isinstance(probe_interval, bool) or
                                      int(probe_interval) != probe_interval or probe_interval <= 0):
        raise ValueError("probe_interval must be a positive integer when registered")
    if not history:
        raise ValueError("At least one observed fixed-probe checkpoint required")
    steps = [int(h["step"]) for h in history]
    if any(h["step"] != step for h, step in zip(history, steps)) or steps != sorted(set(steps)):
        raise ValueError("Probe checkpoints must have unique, strictly increasing integer steps")
    if steps[0] < stage_start_step:
        raise ValueError("Do not mix earlier-stage probes into the new stage curve")
    identities = [h.get("probe_manifest_sha256") for h in history]
    if not all(isinstance(x, str) and x for x in identities) or len(set(identities)) != 1:
        raise ValueError("Fixed probe identity is missing or changed")
    updates = steps[-1]-stage_start_step
    window = list(history[-4:])
    if probe_interval is not None and any(b["step"]-a["step"] != probe_interval
                                          for a, b in zip(history, history[1:])):
        raise ValueError("Fixed-probe checkpoint interval differs from registration")
    reasons = []
    if updates < min_updates:
        reasons.append("MINIMUM_STAGE_UPDATES_NOT_REACHED")
    if len(window) < 4:
        reasons.append("FOUR_CHECKPOINTS_NOT_AVAILABLE")
    metrics = {}
    for key in METRICS:
        intervals = _series_intervals(window, "train", key)
        defined = [r for r in intervals if r["status"] != "NA_INTERVAL_SKIPPED"]
        stats = [_stat((h.get("train") or {}).get(key), f"{h['step']}/train/{key}") for h in window]
        all_na = all(s["mse"] is None for s in stats)
        skipped = key != "ALL_free" and all_na
        stable = len(intervals) == 3 and len(defined) == 3 and all(
            r["relative_change"] is not None and abs(r["relative_change"]) < relative_tolerance
            for r in defined)
        metrics[key] = {"status": "NA_ALL_CHECKPOINTS_SKIPPED" if skipped else "DEFINED",
                        "plateau_pass": None if skipped else stable, "intervals": intervals}
        if not skipped and not stable:
            reasons.append("TRAIN_"+key+"_NOT_STABLE_OR_INCOMPLETE")
    parts = {}
    for part in major_parts:
        intervals = _series_intervals(window, "train", "ALL_free", part=part)
        run = _longest_true_run([_worsens(r, part_worsening_tolerance) for r in intervals])
        blocked = run >= part_worsening_intervals
        parts[part] = {"intervals": intervals, "consecutive_worsening_intervals": run,
                       "blocks_plateau": blocked,
                       "NA_intervals": sum(r["status"] == "NA_INTERVAL_SKIPPED" for r in intervals)}
        if blocked:
            reasons.append("TRAIN_PART_CONTINUOUS_WORSENING:"+part)
    warnings = []
    for key in METRICS:
        for interval in _series_intervals(window, "dev", key):
            if _worsens(interval, dev_warning_tolerance):
                warnings.append(dict(reason="DEV_LOSS_WORSENED_DIAGNOSTIC_ONLY", **interval))
    for part in major_parts:
        for interval in _series_intervals(window, "dev", "ALL_free", part=part):
            if _worsens(interval, dev_warning_tolerance):
                warnings.append(dict(reason="DEV_PART_LOSS_WORSENED_DIAGNOSTIC_ONLY", **interval))
    plateau = not reasons
    status = ("PLATEAU" if plateau else "BUDGET_EXHAUSTED_NOT_CONVERGED"
              if updates >= max_updates else "KEEP_TRAINING")
    return {"schema": VERSION, "status": status, "current_step": steps[-1],
            "stage_updates": updates, "stage_start_step": stage_start_step,
            "minimum_stage_updates": min_updates, "maximum_stage_updates": max_updates,
            "probe_manifest_sha256": identities[0], "window_steps": [h["step"] for h in window],
            "rule": {"adjacent_intervals": 3, "absolute_relative_change_strictly_below": relative_tolerance,
                     "probe_interval_updates": probe_interval,
                     "major_parts": list(major_parts), "part_worsening_strictly_above": part_worsening_tolerance,
                     "part_worsening_consecutive_intervals": part_worsening_intervals,
                     "DEV_used_for_stopping_or_checkpoint_selection": False},
            "metrics": metrics, "part_guard": parts, "reasons": reasons,
            "dev_warnings": warnings, "selected_checkpoint": None,
            "interpretation": "Finite preregistered loss plateau only; not mathematical convergence or generated-quality promotion",
            "no_automatic_extension_beyond_budget": True}


def summarize_training_records(records: Iterable[Mapping], *, fields=None,
                               group_keys=("split", "part", "time_bin")):
    """Aggregate sufficient statistics, never average per-example MSE values.

    Each flat record carries SSE/count pairs listed in DEFAULT_FIELDS. Optional
    fields maps metric names to one (SSE-key, count-key) pair. Group rows include
    every requested grouping key. Missing dimensions are explicit UNKNOWN.
    """
    definitions = ({name: (tuple(pair),) for name, pair in fields.items()}
                   if fields is not None else DEFAULT_FIELDS)
    totals = defaultdict(lambda: {name: {"sse": [], "count": 0, "defined_rows": 0,
                                          "NA_rows": 0} for name in definitions})
    row_counts = defaultdict(int)
    for row in records:
        key = tuple(row.get(k, "UNKNOWN") for k in group_keys)
        if any(isinstance(value, (dict, list)) for value in key):
            raise ValueError("Grouping values must be scalar")
        row_counts[key] += 1
        for name, alternatives in definitions.items():
            matches = [(a, b) for a, b in alternatives if a in row and b in row]
            if len(matches) > 1:
                # Some adapters may emit aliases; conflicting aliases must not
                # silently select whichever spelling was listed first.
                values = [_stat({"sse": row[a], "count": row[b]}, name) for a, b in matches]
                if any(v != values[0] for v in values[1:]):
                    raise ValueError("Conflicting sufficient-statistic aliases: "+name)
            pair = matches[0] if matches else None
            stat = _stat(None if pair is None else {"sse": row[pair[0]], "count": row[pair[1]]}, name)
            entry = totals[key][name]
            if stat["mse"] is None:
                entry["NA_rows"] += 1
            else:
                entry["sse"].append(stat["sse"])
                entry["count"] += stat["count"]
                entry["defined_rows"] += 1
    output = []
    for key in sorted(totals, key=lambda k: tuple(str(v) for v in k)):
        out = dict(zip(group_keys, key)); out["record_count"] = row_counts[key]
        for name, entry in totals[key].items():
            sse = math.fsum(entry["sse"])
            out[name] = dict(sse=sse, count=entry["count"],
                             mse=sse/entry["count"] if entry["count"] else None,
                             status="DEFINED" if entry["count"] else "NA_NO_VALID_SCALARS",
                             defined_rows=entry["defined_rows"], NA_rows=entry["NA_rows"])
        output.append(out)
    return {"schema": VERSION, "aggregation": "sum(SSE)/sum(valid scalar count); never mean(per-sample MSE)",
            "group_keys": list(group_keys), "rows": output,
            "record_count": sum(row_counts.values())}


def analytic_fixture_checks():
    """Small arithmetic fixtures; no tensor, model, training or GPU code."""
    def cp(step, all_mse=1., bulk=1., tip=1., part=1., dev=1.):
        pair = lambda x: {"sse": x*10, "count": 10} if x is not None else {"sse": 0., "count": 0}
        return {"step": step, "probe_manifest_sha256": "fixed_fixture_identity",
                "train": {"ALL_free": pair(all_mse), "bulk": pair(bulk), "seam_tip": pair(tip),
                          "parts": {"SEAT": pair(part)}},
                "dev": {"ALL_free": pair(dev), "bulk": pair(dev), "seam_tip": pair(dev)}}
    cases = []
    stable = [cp(i*1000, tip=None) for i in range(4)]
    r = assess_plateau(stable, min_updates=3000, max_updates=6000, probe_interval=1000, major_parts=["SEAT"])
    assert r["status"] == "PLATEAU" and r["metrics"]["seam_tip"]["plateau_pass"] is None
    cases.append("stable_four_points_optional_NA")
    assert assess_plateau(stable, min_updates=4000, max_updates=6000)["status"] == "KEEP_TRAINING"
    cases.append("minimum_updates_enforced")
    drifting = [cp(i*1000, all_mse=1.1**i) for i in range(4)]
    assert assess_plateau(drifting, min_updates=0, max_updates=3000)["status"] == "BUDGET_EXHAUSTED_NOT_CONVERGED"
    cases.append("budget_without_plateau_is_not_converged")
    bad_part = [cp(i*1000, part=1.06**i) for i in range(4)]
    assert assess_plateau(bad_part, min_updates=0, max_updates=6000, major_parts=["SEAT"])["status"] == "KEEP_TRAINING"
    cases.append("major_part_three_interval_guard")
    bad_dev = [cp(i*1000, dev=1.1**i) for i in range(4)]
    r = assess_plateau(bad_dev, min_updates=0, max_updates=6000)
    assert r["status"] == "PLATEAU" and len(r["dev_warnings"]) == 9 and r["selected_checkpoint"] is None
    cases.append("DEV_warning_does_not_choose_or_stop")
    incomplete = [cp(i*1000, tip=None if i == 1 else 1.) for i in range(4)]
    assert assess_plateau(incomplete, min_updates=0, max_updates=6000)["status"] == "KEEP_TRAINING"
    cases.append("partial_missing_stratum_cannot_claim_three_intervals")
    zero = [cp(i*1000, all_mse=0.) for i in range(4)]
    assert assess_plateau(zero, min_updates=0, max_updates=6000)["status"] == "PLATEAU"
    zero[-1]["train"]["ALL_free"]["sse"] = .01
    assert assess_plateau(zero, min_updates=0, max_updates=6000)["status"] == "KEEP_TRAINING"
    cases.append("zero_denominator_relative_change_handled")
    rows = [dict(split="TRAIN", part="SEAT", time_bin=2, all_free_velocity_sse=1., all_free_scalar_count=1),
            dict(split="TRAIN", part="SEAT", time_bin=2, all_free_velocity_sse=90., all_free_scalar_count=100)]
    summary = summarize_training_records(rows)
    stat = summary["rows"][0]["ALL_free"]
    assert stat["count"] == 101 and stat["sse"] == 91. and stat["mse"] == 91/101
    assert stat["mse"] != (1.+.9)/2
    cases.append("SSE_count_weighting_not_mean_of_means")
    altered = [cp(i*1000) for i in range(4)]
    altered[-1]["probe_manifest_sha256"] = "changed"
    try:
        assess_plateau(altered, min_updates=0, max_updates=6000)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed probe identity accepted")
    cases.append("fixed_probe_identity_required")
    try:
        assess_plateau(stable, min_updates=0, max_updates=6000, probe_interval=500)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed checkpoint interval accepted")
    cases.append("registered_probe_interval_required")
    return {"status": "PASS", "schema": VERSION, "checks": cases,
            "model_calls": 0, "training_updates": 0, "GPU_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--min-updates", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--stage-start-step", type=int, default=0)
    parser.add_argument("--probe-interval", type=int)
    parser.add_argument("--major-part", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = analytic_fixture_checks()
    elif args.history is not None and args.min_updates is not None and args.max_updates is not None:
        history = json.loads(args.history.read_text(encoding="utf-8-sig"))
        result = assess_plateau(history, min_updates=args.min_updates, max_updates=args.max_updates,
                                stage_start_step=args.stage_start_step, probe_interval=args.probe_interval,
                                major_parts=args.major_part)
    else:
        parser.error("Use --self-test, or --history with --min-updates and --max-updates")
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError("Preserve existing convergence evidence: "+str(args.output))
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
