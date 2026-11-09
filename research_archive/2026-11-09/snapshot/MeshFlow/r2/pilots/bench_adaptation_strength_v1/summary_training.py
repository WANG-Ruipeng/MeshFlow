"""Read saved Bench training evidence and draw curves; no torch/model imports.

Each invocation creates an immutable timestamped snapshot. Active runs are
reported INCOMPLETE. --final requires every arm to be complete; it never waits,
replays, repairs training logs, loads weights, evaluates meshes or calls models.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
ARMS = ("A_COMMON_LR1E5", "B_COMMON_LR2E6", "C_OFFICIAL_LR2E6")
PROBE_STEPS = (0, 500, 1000, 1500)
METRICS = ("all_free", "seam_tip", "bulk")
KINDS = ("train_forward", "train_backward", "optimizer_update", "probe_forward")
COLORS = {"A_COMMON_LR1E5": "#2671a3", "B_COMMON_LR2E6": "#dd8d25", "C_OFFICIAL_LR2E6": "#46865c"}
LABELS = {"A_COMMON_LR1E5": "A: COMMON, backbone LR 1e-5",
          "B_COMMON_LR2E6": "B: COMMON, backbone LR 2e-6",
          "C_OFFICIAL_LR2E6": "C: Official, backbone LR 2e-6"}
BOUNDARIES = [
    "A/B inherit the identical COMMON12000 live generator, but use fresh AdamW. A is a new synchronous comparator, not historical CLEAN1500.",
    "C inherits official category Bench EMA plus the original newly initialized role/Geo. C has no local COMMON history; do not plot its local1500 as global13500.",
    "A versus B isolates the registered backbone learning rate at the same initialization and fresh optimizer; both condition groups use LR1e-5.",
    "B versus C compares initial generators at the same future1500-update budget. Their earlier adaptation histories differ; it is not equal lifetime compute.",
    "All arms have 195 eligible conditions from58 TRAIN parents and the same frozen1500x8 stream. Noise, time, augmentations and OT permutation identities are paired.",
    "The original96 supervised probe inputs contain source-target interpolation and source owner layouts. Lower probe error does not certify free-generation quality or C adoption.",
    "All generation is evaluated with one frozen predicted C-only layout. Shared layout mistakes can limit every arm; source owner labels are never inference inputs.",
    "No occupancy head is trained or called here. SEAM adds no new generator parameters; theta=backbone, phi=role_embedding/context_encoder.",
    "The precommitted primary endpoint is local1500; local0/500 are learning-curve observations, not permission to select a best-looking checkpoint.",
    "A lower backbone learning rate need not produce a fivefold smaller parameter displacement. Report measured drift separately for theta and phi.",
    "Training MSE uses changing samples and times. Compare fixed probes for like-for-like supervised change; assess saved RAW/POST and sections for actual generation.",
    "Parallel training and background applications alter elapsed time. Per-worker process VRAM is distinct from PyTorch allocated/reserved memory; do not call this an isolated speed benchmark.",
    "This summary performs zero model construction, forward, backward, optimizer update, generation or re-scoring. It does not change defaults or experiment budgets.",
]


def resolve(value):
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":" + value[6:].replace("/", "\\")
    elif os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class Reader:
    def __init__(self):
        self.sources = []
        self.incomplete_lines = []

    def bytes(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.sources.append(dict(path=str(path), bytes_read=len(data), sha256_of_read_snapshot=hashlib.sha256(data).hexdigest()))
        return data

    def json(self, path, optional=False):
        if optional and not Path(path).exists():
            return None
        return json.loads(self.bytes(path).decode("utf-8-sig"))

    def jsonl(self, path):
        if not Path(path).exists():
            return []
        data = self.bytes(path)
        lines = data.splitlines(keepends=True)
        rows = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            # An in-progress append is not corruption and cannot be invented.
            if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                self.incomplete_lines.append(dict(path=str(path), trailing_fragment_bytes=len(line)))
                break
            rows.append(json.loads(line.decode("utf-8")))
        return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=()):
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    result = dict(examples=len(rows))
    for metric in METRICS:
        sse = sum(float(r[metric + "_velocity_sse"]) for r in rows)
        count = sum(int(r[metric + "_scalar_count"]) for r in rows)
        result.update({metric + "_SSE": sse, metric + "_scalar_count": count,
                       metric + "_MSE": sse / count if count else None})
    return result


def history(arm):
    return 0 if arm == "C_OFFICIAL_LR2E6" else 12000


def approx(a, b):
    return math.isclose(float(a), float(b), rel_tol=4e-6, abs_tol=1e-7)


def audit_calls(rows, arm):
    attempted, done, failed = collections.Counter(), collections.Counter(), collections.Counter()
    call_seconds = collections.defaultdict(float)
    pending = {}
    seen = set()
    issues = []
    update_steps = []
    for row in rows:
        kind, ident, event = row.get("kind"), row.get("call_id"), row.get("event")
        if row.get("arm") != arm or kind not in KINDS:
            issues.append("Unexpected arm/call kind: " + repr((row.get("arm"), kind)))
            continue
        if event == "CALL_BEGIN":
            if ident in seen:
                issues.append("Duplicate call reservation: " + str(ident))
            seen.add(ident)
            pending[ident] = row
            attempted[kind] += 1
            if kind == "optimizer_update":
                update_steps.append(row["context"]["local_step"])
        elif event in ("CALL_DONE", "CALL_FAILED"):
            begin = pending.pop(ident, None)
            if begin is None or begin["kind"] != kind:
                issues.append("Completion without matching begin: " + str(ident))
            if event == "CALL_DONE":
                done[kind] += 1
                call_seconds[kind] += float(row["seconds"])
            else:
                failed[kind] += 1
        else:
            issues.append("Unknown call event: " + str(event))
    if update_steps != list(range(1, len(update_steps) + 1)):
        issues.append("Optimizer reservations are not consecutive local1..k")
    limits = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                  probe_forward=288 if arm == "B_COMMON_LR2E6" else 384)
    if any(attempted[k] > limits[k] for k in limits):
        issues.append("Call attempt budget exceeded")
    return dict(attempted=dict(attempted), completed=dict(done), failed=dict(failed),
                pending_call_ids=sorted(pending), authorized_limits=limits, issues=issues,
                instrumented_call_seconds=dict(call_seconds))


def make_plots(out, per_step, probe_summary, drifts, complete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = "COMPLETE" if complete else "INCOMPLETE snapshot"
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for rowi, metric in enumerate(METRICS):
        for coli, split in enumerate(("TRAIN", "DEV")):
            ax = axes[rowi, coli]
            for arm in ARMS:
                rows = sorted((r for r in probe_summary if r["arm"] == arm and r["split"] == split and r["part"] == "ALL"),
                              key=lambda r: r["local_step"])
                ax.plot([r["local_step"] for r in rows], [r[metric + "_MSE"] for r in rows],
                        marker="o", color=COLORS[arm], label=LABELS[arm])
            ax.set_title(f"{split}: {metric} fixed supervised MSE")
            ax.grid(alpha=.2)
            ax.set_xlabel("New optimizer updates (local step)")
            ax.set_ylabel("sum SSE / actual scalar count")
            ax.set_xlim(-35, 1535)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bench adaptation strength: " + phase + "\nA/B step0 share one96-input probe; C starts from official EMA (no COMMON history)")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / "fixed_probe_learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, metric in zip(axes, METRICS):
        for arm in ARMS:
            rows = [r for r in per_step if r["arm"] == arm]
            xs, ys = [], []
            for end in range(1, len(rows) + 1):
                window = rows[max(0, end - 50):end]
                sse = sum(r[metric + "_velocity_sse"] for r in window)
                count = sum(r[metric + "_scalar_count"] for r in window)
                xs.append(rows[end - 1]["local_step"])
                ys.append(sse / count if count else float("nan"))
            axis.plot(xs, ys, color=COLORS[arm], label=LABELS[arm])
        axis.set_title(metric + ": trailing50-update pooled MSE")
        axis.set_xlabel("New optimizer updates")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Changing training inputs/times; this is not a fixed-probe curve. " + phase)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "training_loss_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, group in zip(axes, ("theta_backbone", "phi_condition")):
        for arm in ARMS:
            rows = sorted((r for r in drifts if r["arm"] == arm and r["group"] == group), key=lambda r: r["local_step"])
            axis.plot([r["local_step"] for r in rows], [r["delta_rms"] for r in rows],
                      marker="o", color=COLORS[arm], label=LABELS[arm])
        axis.set_title(group + ": parameter RMS displacement")
        axis.set_xlabel("New optimizer updates")
        axis.set_ylabel("RMS(parameter - arm's own initialization)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Measured displacement from each arm's own local0; " + phase)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "parameter_drift_curves.png", dpi=160)
    plt.close(fig)


def summarize(root=DEFAULT_ROOT, *, final=False, plots=True):
    root = resolve(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError("Summary is restricted to this experiment root")
    reader = Reader()
    reg = reader.json(root / "training_registration.json")
    statuses = {arm: reader.json(root / "training" / arm / "status.json", optional=True) for arm in ARMS}
    complete = all(s and s["status"] == "COMPLETE" and s["local_step"] == 1500 for s in statuses.values())
    if final and not complete:
        raise RuntimeError("--final refused: one or more registered arms are INCOMPLETE")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "training_summary" / stamp
    out.mkdir(parents=True, exist_ok=False)
    schedule = reader.json(resolve(reg["input_schedule"]))["steps"]
    manifest = reader.json(resolve(reg["manifest"]))
    by_cid = {row["condition_id"]: row for row in manifest["train"]}
    all_parents = sorted({row["parent_id"] for row in manifest["train"]})
    per_step, windows, condition_rows, parent_rows = [], [], [], []
    probe_rows, probe_summary, drift_rows, call_rows, resource_rows = [], [], [], [], []
    arm_audits = {}
    all_issues = []
    for arm in ARMS:
        folder = root / "training" / arm
        logs = reader.jsonl(folder / "training_log.jsonl")
        exposures_all = reader.jsonl(folder / "exposure.jsonl")
        calls = reader.jsonl(folder / "calls.jsonl")
        n = len(logs)
        issues = []
        # During active append, complete exposure rows of the next update can
        # precede the training_log commit. Preserve their count separately.
        exposures = [r for r in exposures_all if int(r["local_step"]) <= n]
        trailing_exposures = [r for r in exposures_all if int(r["local_step"]) > n]
        if [r["local_step"] for r in logs] != list(range(1, n + 1)):
            issues.append("Training log is not consecutive local1..k")
        if any(r["global_step"] != history(arm) + r["local_step"] or r["optimizer_local_step"] != r["local_step"] for r in logs):
            issues.append("Local/global/fresh optimizer history mismatch")
        grouped = collections.defaultdict(list)
        for row in exposures:
            grouped[row["local_step"]].append(row)
        expected_exposure = collections.Counter()
        actual_exposure = collections.Counter(row["condition_id"] for row in exposures)
        for index, log in enumerate(logs):
            step = index + 1
            rows = grouped[step]
            wanted = schedule[index]["samples"]
            expected_exposure.update(r["condition_id"] for r in wanted)
            if len(rows) != 8 or sorted(r["microbatch"] for r in rows) != list(range(8)):
                issues.append(f"Step{step}: completed update lacks exactly8 exposure rows")
                continue
            rows.sort(key=lambda r: r["microbatch"])
            if any(any(row.get(key) != value for key, value in want.items()) for row, want in zip(rows, wanted)):
                issues.append(f"Step{step}: sample identity / augmentation / seed differs from schedule")
            stats = aggregate(rows)
            if not approx(stats["all_free_MSE"], log["FM"]):
                issues.append(f"Step{step}: pooled SSE differs from FM")
            if stats["all_free_scalar_count"] != log["free_coordinate_denominator"] or (
                stats["all_free_scalar_count"] != stats["bulk_scalar_count"] + stats["seam_tip_scalar_count"]
            ):
                issues.append(f"Step{step}: free/tip/bulk scalar accounting differs")
            if any(not approx(stats[m + "_SSE"], log[m + "_velocity_sse"]) or
                   stats[m + "_scalar_count"] != log[m + "_scalar_count"] for m in METRICS):
                issues.append(f"Step{step}: sample/step SSE or count differs")
            per_step.append(dict(arm=arm, source_global_step=history(arm), **log,
                                 **{m + "_MSE": stats[m + "_MSE"] for m in METRICS}))
        if actual_exposure != expected_exposure:
            issues.append("Per-condition completed exposure differs from schedule")
        by_condition = collections.defaultdict(list)
        by_parent = collections.defaultdict(list)
        for row in exposures:
            by_condition[row["condition_id"]].append(row)
            by_parent[row["parent_id"]].append(row)
        for cid in sorted(by_cid):
            source = by_cid[cid]
            rows = by_condition[cid]
            condition_rows.append(dict(arm=arm, condition_id=cid, parent_id=source["parent_id"],
                                       part=source["part_type"], completed_local_updates=n, exposures=len(rows),
                                       clean=sum(r["augmentation"] == "CLEAN" for r in rows),
                                       augmented=sum(r["augmentation"] == "ANISOTROPIC" for r in rows),
                                       expected_exposures=expected_exposure[cid]))
        for parent in all_parents:
            rows = by_parent[parent]
            parent_rows.append(dict(arm=arm, parent_id=parent, completed_local_updates=n, exposures=len(rows),
                                    clean=sum(r["augmentation"] == "CLEAN" for r in rows),
                                    augmented=sum(r["augmentation"] == "ANISOTROPIC" for r in rows),
                                    distinct_conditions_seen=len({r["condition_id"] for r in rows}),
                                    available_conditions=sum(r["parent_id"] == parent for r in by_cid.values())))
        for start in range(1, n + 1, 100):
            end = min(n, start + 99)
            rows = [r for r in exposures if start <= r["local_step"] <= end]
            for part in ["ALL"] + sorted({r["part_type"] for r in rows}):
                for tb in ("ALL", "0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
                    subset = [r for r in rows if (part == "ALL" or r["part_type"] == part) and (tb == "ALL" or r["time_bin"] == tb)]
                    windows.append(dict(arm=arm, start_step=start, end_step=end, part=part, time_bin=tb, **aggregate(subset)))
        probe_actual = 0
        probe_observations = []
        for step in PROBE_STEPS:
            p = folder / "fixed_loss" / f"step{step}.json"
            value = reader.json(p, optional=True)
            if value is None:
                continue
            if value["local_step"] != step or value["global_step"] != history(arm) + step:
                issues.append(f"Probe{step}: local/global history mismatch")
            if value["arm"] != arm or len(value["rows"]) != 96 or not value["RNG_unchanged"]:
                issues.append(f"Probe{step}: identity/count/RNG mismatch")
            expected_actual = 0 if arm == "B_COMMON_LR2E6" and step == 0 else 96
            if value["actual_generator_forwards"] != expected_actual or value["head_forwards"] != 0 or value["backward_calls"] != 0:
                issues.append(f"Probe{step}: actual call budget differs")
            if value["probe_manifest_sha256"] != reg["probe_manifest_sha256"]:
                issues.append(f"Probe{step}: frozen inputs changed")
            if expected_actual == 0:
                inherited = resolve(value["inherited_probe"])
                blob = reader.bytes(inherited)
                shared = json.loads(blob.decode("utf-8-sig"))
                if hashlib.sha256(blob).hexdigest() != value["inherited_probe_sha256"] or shared["rows"] != value["rows"]:
                    issues.append("B step0 inherited rows/hash mismatch")
                if value["generator_state_sha256"] != reg["source_generator_sha256"]:
                    issues.append("B step0 is not COMMON generator")
            probe_actual += expected_actual
            probe_observations.append(step)
            for row in value["rows"]:
                probe_rows.append(dict(arm=arm, local_step=step, global_step=history(arm) + step,
                                       source_global_step=history(arm), inherited_observation=expected_actual == 0, **row))
            for split in ("TRAIN", "DEV"):
                rows = [r for r in value["rows"] if r["split"] == split]
                for part in ["ALL"] + sorted({r["part"] for r in rows}):
                    subset = [r for r in rows if part == "ALL" or r["part"] == part]
                    probe_summary.append(dict(arm=arm, local_step=step, global_step=history(arm) + step,
                                              split=split, part=part, **aggregate(subset)))
        for path in sorted((folder / "parameter_drift").glob("step*.json")):
            value = reader.json(path)
            for group in ("theta_backbone", "phi_condition"):
                drift_rows.append(dict(arm=arm, local_step=value["local_step"], group=group, **value[group]))
        call_audit = audit_calls(calls, arm)
        issues.extend(call_audit["issues"])
        if complete:
            if n != 1500 or len(exposures) != 12000 or trailing_exposures:
                issues.append("Final training / exposure count differs")
            if call_audit["pending_call_ids"] or call_audit["failed"]:
                issues.append("Final calls contain pending / failed attempts")
            if call_audit["attempted"] != call_audit["authorized_limits"] or call_audit["completed"] != call_audit["authorized_limits"]:
                issues.append("Final attempted/completed counts differ from exact fixed budget")
            if probe_observations != list(PROBE_STEPS) or probe_actual != call_audit["authorized_limits"]["probe_forward"]:
                issues.append("Final probe observation / actual-call count differs")
        for kind in KINDS:
            call_rows.append(dict(arm=arm, kind=kind, attempts=call_audit["attempted"].get(kind, 0),
                                  completed=call_audit["completed"].get(kind, 0),
                                  failed=call_audit["failed"].get(kind, 0),
                                  authorized_limit=call_audit["authorized_limits"][kind],
                                  instrumented_seconds=call_audit["instrumented_call_seconds"].get(kind, 0.)))
        process_peaks, device_peaks = [], []
        attempts = []
        for path in sorted((folder / "attempts").glob("*/resources.json")):
            rr = reader.json(path)
            summary = rr["summary"]
            attempts.append(dict(path=str(path), **summary))
            if summary.get("process_peak_mib") is not None:
                process_peaks.append(summary["process_peak_mib"])
            if summary.get("total_device_used_peak_mib") is not None:
                device_peaks.append(summary["total_device_used_peak_mib"])
        for rr in reader.jsonl(folder / "resource_checks.jsonl"):
            resource_rows.append(dict(arm=arm, local_step_before=rr["local_step_before"],
                                      free_mib=rr.get("free_mib"),
                                      snapshot_json=json.dumps(rr, ensure_ascii=False)))
        timing = dict(sum_input_seconds=sum(r["input_seconds"] for r in logs),
                      sum_formal_batch_seconds=sum(r["update_seconds"] for r in logs),
                      per_attempt_elapsed_seconds=[],
                      measured_process_peak_mib=max(process_peaks) if process_peaks else None,
                      measured_device_used_peak_mib=max(device_peaks) if device_peaks else None,
                      max_formal_batch_allocated_bytes=max((r["peak_allocated_bytes"] for r in logs), default=None),
                      max_formal_batch_reserved_bytes=max((r["peak_reserved_bytes"] for r in logs), default=None),
                      process_peak_status="OBSERVED" if process_peaks else "NA_NOT_REPORTED_OR_ATTEMPT_ACTIVE",
                      observer_attempts=attempts)
        for path in sorted((folder / "attempts").glob("*/final_status.json")):
            attempt = reader.json(path)
            timing["per_attempt_elapsed_seconds"].append(dict(attempt=path.parent.name,
                elapsed_seconds=attempt["elapsed_seconds"], status=attempt["status"]))
        arm_audits[arm] = dict(status="COMPLETE" if statuses[arm] and statuses[arm]["status"] == "COMPLETE" else "INCOMPLETE",
                               local_updates_observed=n, source_global_step=history(arm),
                               cumulative_generator_updates_observed=history(arm) + n,
                               status_snapshot=statuses[arm], completed_exposure_rows=len(exposures),
                               trailing_in_progress_exposure_rows=len(trailing_exposures),
                               probe_observation_steps=probe_observations, actual_saved_probe_forwards=probe_actual,
                               call_accounting=call_audit, timing_and_resources=timing, issues=issues)
        all_issues.extend(arm + ": " + issue for issue in issues)
    # All completed arm prefixes must use the same source/random stream. This is
    # already checked row-by-row against the same immutable schedule above.
    totals = {kind: dict(attempted=sum(a["call_accounting"]["attempted"].get(kind, 0) for a in arm_audits.values()),
                         completed=sum(a["call_accounting"]["completed"].get(kind, 0) for a in arm_audits.values()),
                         authorized={"train_forward": 36000, "train_backward": 36000,
                                     "optimizer_update": 4500, "probe_forward": 1056}[kind]) for kind in KINDS}
    if reader.incomplete_lines and complete:
        all_issues.append("Completed files contain unfinished JSONL fragments")
    accounting = dict(status="FAIL" if all_issues else ("PASS" if complete else "INCOMPLETE"),
                      training_complete=complete, source="saved logs only; no checkpoint tensor load",
                      arms=arm_audits, totals=totals, issues=all_issues,
                      incomplete_jsonl_fragments=reader.incomplete_lines,
                      exact_schedule_verified_for_completed_updates=not all_issues,
                      additional_model_calls=0, additional_scoring_calls=0)
    write_json(out / "accounting.json", accounting)
    write_csv(out / "training_per_step.csv", per_step, ("arm", "local_step", "global_step"))
    write_csv(out / "training_windows100_part_time.csv", windows, ("arm", "start_step", "end_step"))
    write_csv(out / "fixed_probe_per_input.csv", probe_rows, ("arm", "local_step", "probe_id"))
    write_csv(out / "fixed_probe_summary.csv", probe_summary, ("arm", "local_step", "split", "part"))
    write_csv(out / "exposure_by_condition.csv", condition_rows)
    write_csv(out / "exposure_by_parent.csv", parent_rows)
    write_csv(out / "parameter_drift.csv", drift_rows, ("arm", "local_step", "group"))
    write_csv(out / "call_budget.csv", call_rows)
    write_csv(out / "resource_checks.csv", resource_rows, ("arm", "local_step_before", "free_mib"))
    write_json(out / "source_snapshots.json", reader.sources)
    write_json(out / "scientific_boundaries.json", BOUNDARIES)
    text = "# Bench adaptation strength：训练曲线解释边界\n\n"
    text += f"本快照状态：`{accounting['status']}`。本工具不构造模型、不读取大权重、不重新评分。\n\n"
    text += "| 实验臂 | 起点历史 | 本轮主干 LR | 条件参数 LR | 本轮固定预算 |\n|---|---|---:|---:|---:|\n"
    text += "| A | COMMON12000 live | 1e-5 | 1e-5 | 1500 |\n| B | 同一 COMMON12000 live | 2e-6 | 1e-5 | 1500 |\n| C | 官方 Bench EMA＋原始 role/Geo 初始化，未经过 COMMON | 2e-6 | 1e-5 | 1500 |\n\n"
    text += "三臂均 fresh AdamW；局部优化器步从0开始，A/B完整生成器历史为12000＋local，C为local。\n\n"
    text += "- A/B比较主干更新强度；B/C比较相同未来预算下的起点，不能称为相同累计训练计算量。\n"
    text += "- 固定探针使用来源目标插值和训练来源owner，不能把MSE下降直接当作自由生成、C采用或接续成功。\n"
    text += "- A/B的step0只执行一次96输入探针；B继承同一结果。总实际探针1056次，12份观测表中有1份是继承结果。\n"
    text += "- 原始COMMON12000已知DEV退化是本轮动机，不预设重启官方或更小LR一定胜出；C的1500步失败也不能证明官方重适配无效。\n"
    text += "- 主结果固定local1500，step500和step1000用于观察学习趋势，不挑best；所有结论最终应结合固定RAW/POST及截面图审。\n"
    text += "- 冻结C-only布局用于隔离生成器变化；三臂共同失败不能自动排除布局本身的限制。\n"
    text += "- 已完成更新的所有条件身份、父对象、增强、噪声/时间/排列种子逐项核对冻结schedule。增强和重复曝光不等于新增独立对象。\n"
    text += "- 波动训练损失按实际标量数聚合；图中50步平滑使用sum SSE / sum count，不平均不同分母的样本MSE。\n"
    text += "- 并行和用户图形任务会影响墙钟；资源峰值分别保留实际进程、设备总占用与PyTorch指标，缺失记NA。\n"
    text += "\n完整英文机器可读边界在`scientific_boundaries.json`，调用/数据核对结果在`accounting.json`。\n"
    (out / "SCIENTIFIC_BOUNDARIES.md").write_text(text, encoding="utf-8")
    if plots:
        make_plots(out, per_step, probe_summary, drift_rows, complete)
    result = dict(status=accounting["status"], path=str(out), training_complete=complete,
                  local_updates={arm: arm_audits[arm]["local_updates_observed"] for arm in ARMS},
                  totals=totals, issues=len(all_issues), zero_model_calls=True)
    write_json(out / "completion.json", result)
    # Separate mutable index; old timestamped summaries are never overwritten.
    index = root / "training_summary/latest.json"
    temp = index.with_name(index.name + f".partial.{os.getpid()}.{time.time_ns()}")
    write_json(temp, result)
    os.replace(temp, index)
    if final and all_issues:
        raise RuntimeError("Final accounting FAIL; see " + str(out / "accounting.json"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summarize(args.root, final=args.final, plots=not args.no_plots)


if __name__ == "__main__":
    main()
