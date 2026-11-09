"""Read-only CPU accounting of all registered Bench training and evaluations.

Run under WSL; no torch imports, model calls, scoring, or production writes.
Canonical training ledger calls and the explicitly audited discarded recovery
attempts are disjoint. Evaluation receipts are deduplicated by physical output
directory, not by the final-label list or by aggregate completion markers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path, sources):
    data = path.read_bytes()
    sources[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return json.loads(data)


def ledger_snapshot(path, consume, sources):
    """Capture a bounded prefix: a concurrently growing log is not chased."""
    initial_bytes = path.stat().st_size
    digest = hashlib.sha256()
    consumed = lines = trailing_bytes = 0
    with path.open("rb") as f:
        while consumed < initial_bytes:
            line = f.readline(initial_bytes - consumed)
            if not line:
                break
            consumed += len(line)
            digest.update(line)
            if not line.endswith(b"\n"):
                trailing_bytes = len(line)
                break
            item = json.loads(line)
            consume(item)
            lines += 1
    info = dict(snapshot_bytes=consumed, complete_jsonl_rows=lines,
                trailing_uncommitted_bytes=trailing_bytes,
                captured_prefix_sha256=digest.hexdigest(), captured_utc=now(),
                final_observed_file_bytes=path.stat().st_size)
    sources[str(path)] = info
    return info


def training_arm(root, arm, sources):
    path = root / "training" / arm / "ledger.jsonl"
    if not path.exists():
        return dict(arm=arm, status="NOT_STARTED")
    events = Counter()
    updates = Counter()
    probes = []
    probe_started = Counter()
    chunks = []

    def consume(row):
        event = row["event"]
        events[event] += 1
        if event == "UPDATE_COMPLETE":
            updates[int(row["step"])] += 1
        elif event == "PROBE_STARTED":
            probe_started[int(row["step"])] += 1
        elif event == "PROBE_COMPLETE":
            probes.append(dict(step=row["step"],
                               generator_forwards=row["generator_forwards"],
                               head_forwards=row["head_forwards"]))
        elif event == "CHUNK_COMPLETE":
            chunks.append({k: row[k] for k in ("step", "target") if k in row})

    snapshot = ledger_snapshot(path, consume, sources)
    completed_probe_steps = Counter(int(x["step"]) for x in probes)
    mismatches = []
    duplicate_updates = {str(s): n for s, n in updates.items() if n > 1}
    if duplicate_updates:
        mismatches.append("DUPLICATE_UPDATE_COMPLETE_STEP")
    completed_probe_forwards = sum(x["generator_forwards"] for x in probes)
    pending_probes = dict(probe_started - completed_probe_steps)
    if not pending_probes and completed_probe_forwards != events["PROBE_FORWARD_COMPLETE"]:
        mismatches.append("PROBE_SUMMARY_VS_RETURN_COUNT")
    pending = {key: events[key + "_STARTED"] - events[key + "_COMPLETE"]
               for key in ("FORWARD", "BACKWARD", "OPTIMIZER", "UPDATE", "HEAD_FORWARD",
                           "HEAD_BACKWARD", "HEAD_OPTIMIZER", "PROBE_FORWARD")}
    return dict(arm=arm, status="SNAPSHOT", snapshot=snapshot,
                event_counts=dict(events), unique_completed_updates=len(updates),
                minimum_completed_step=min(updates, default=None),
                maximum_completed_step=max(updates, default=None),
                duplicate_update_complete_steps=duplicate_updates,
                completed_probes=probes, pending_probe_steps=pending_probes,
                completed_probe_summary_generator_forwards=completed_probe_forwards,
                completed_probe_head_forwards=sum(x["head_forwards"] for x in probes),
                pending_started_minus_returned=pending, completed_chunks=chunks,
                mismatches=mismatches,
                head_probe_incomplete_policy="Head calls inside an incomplete probe are UNKNOWN, not inferred.")


def count_calls(path, sources):
    events = Counter()
    if path.exists():
        ledger_snapshot(path, lambda row: events.update([row["event"]]), sources)
    return dict(events)


def evaluations(root, sources):
    labels = []
    output_rows = []
    layout_rows = []
    for directory in sorted((root / "evaluation").iterdir()):
        if not directory.is_dir() or not re.fullmatch(r"(?:NATIVE|SEAM)_S\d+", directory.name):
            continue
        label = directory.name
        start_outputs, start_layouts = len(output_rows), len(layout_rows)
        completions = []
        for p in sorted(directory.glob("*_complete_*.json")):
            record = read_json(p, sources)
            completions.append(dict(path=str(p), **{k: record[k] for k in
                ("status", "subset", "outputs", "new_rollouts", "new_forwards",
                 "new_head_forwards", "new_D0_forwards") if k in record}))
        layouts = directory / "layouts"
        if layouts.exists():
            for layout in sorted(layouts.iterdir()):
                if not layout.is_dir():
                    continue
                p = layout / "receipt.json"
                if p.exists():
                    receipt = read_json(p, sources)
                    layout_rows.append(dict(label=label, condition_id=layout.name,
                                            status=receipt["status"],
                                            occupancy_forward=receipt["occupancy_forward"],
                                            closed_C=receipt["closed_C"], receipt=str(p)))
                elif (layout / "attempt.json").exists():
                    layout_rows.append(dict(label=label, condition_id=layout.name,
                                            status="PENDING_OR_UNRESOLVED", occupancy_forward=None))
        for request in sorted(directory.iterdir()):
            if not request.is_dir() or not re.search(r"__n\d+$", request.name):
                continue
            row = dict(label=label, request_id=request.name)
            for stage in ("raw", "post"):
                p = request / stage / "receipt.json"
                receipt = read_json(p, sources) if p.exists() else None
                row[stage + "_complete"] = bool(receipt and receipt["status"] == "COMPLETE")
                if receipt:
                    assert receipt["label"] == label and receipt["request_id"] == request.name
                row[stage + "_calls"] = count_calls(request / stage / "calls.jsonl", sources)
                if stage == "raw":
                    row["receipt_generator_forwards"] = receipt["generator_forwards"] if receipt else None
                else:
                    row["receipt_D0_forwards"] = receipt["D0_forwards"] if receipt else None
                    row["receipt_GEO_calls"] = receipt["GEO_calls"] if receipt else None
            proposal_path = request / "post" / "proposal_receipt.json"
            proposal = read_json(proposal_path, sources) if proposal_path.exists() else None
            row["D0_proposal_complete"] = bool(proposal and proposal["status"] == "COMPLETE")
            row["proposal_D0_forwards"] = proposal["D0_forwards"] if proposal else None
            row["mismatches"] = []
            if row["raw_complete"] and row["receipt_generator_forwards"] != row["raw_calls"].get("FORWARD_COMPLETE", 0):
                row["mismatches"].append("RAW_RECEIPT_VS_CALL_LEDGER")
            if row["D0_proposal_complete"] and row["proposal_D0_forwards"] != row["post_calls"].get("D0_FORWARD_RETURNED", 0):
                row["mismatches"].append("D0_PROPOSAL_VS_CALL_LEDGER")
            if row["post_complete"] and row["receipt_D0_forwards"] != row["proposal_D0_forwards"]:
                row["mismatches"].append("POST_VS_D0_PROPOSAL")
            output_rows.append(row)
        rows = output_rows[start_outputs:]
        ls = layout_rows[start_layouts:]
        labels.append(dict(label=label, completion_markers=completions,
                           requests_started=len(rows),
                           RAW_complete=sum(x["raw_complete"] for x in rows),
                           POST_complete=sum(x["post_complete"] for x in rows),
                           generator_forward_started=sum(x["raw_calls"].get("FORWARD_STARTED", 0) for x in rows),
                           generator_forward_returns=sum(x["raw_calls"].get("FORWARD_COMPLETE", 0) for x in rows),
                           D0_forward_started=sum(x["post_calls"].get("D0_FORWARD_STARTED", 0) for x in rows),
                           D0_forward_returns=sum(x["post_calls"].get("D0_FORWARD_RETURNED", 0) for x in rows),
                           layout_head_forward_returns=sum(x["occupancy_forward"] or 0 for x in ls),
                           layout_unresolved=sum(x["occupancy_forward"] is None for x in ls),
                           GEO_complete_calls=sum(x["receipt_GEO_calls"] or 0 for x in rows)))
    return labels, output_rows, layout_rows


def build(root, final=False):
    started = now()
    sources = {}
    arms = [training_arm(root, arm, sources) for arm in ("COMMON", "PLAIN", "SEAM")]
    labels, outputs, layouts = evaluations(root, sources)
    recovery_path = root / "audits/worker_write_race_20261019/recovery_compute_accounting.json"
    recovery = read_json(recovery_path, sources)
    extra = recovery["total_discarded_attempt_extra"]
    decisions = sorted((root / "dispatch/decisions").glob("step*.json"),
                       key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    decision = read_json(decisions[-1], sources) if decisions else None
    canonical = Counter()
    for arm in arms:
        canonical.update(arm.get("event_counts", {}))
    get = lambda key: sum(x.get(key, 0) for x in labels)
    probe_heads = sum(x.get("completed_probe_head_forwards", 0) for x in arms)
    totals = dict(
        unique_generator_optimizer_updates=sum(x.get("unique_completed_updates", 0) for x in arms),
        canonical_generator_optimizer_returns=canonical["OPTIMIZER_COMPLETE"],
        canonical_generator_training_forward_returns=canonical["FORWARD_COMPLETE"],
        canonical_generator_training_backward_returns=canonical["BACKWARD_COMPLETE"],
        canonical_generator_training_backward_starts=canonical["BACKWARD_STARTED"],
        canonical_head_optimizer_returns=canonical["HEAD_OPTIMIZER_COMPLETE"],
        canonical_head_training_forward_returns=canonical["HEAD_FORWARD_COMPLETE"],
        canonical_head_training_backward_returns=canonical["HEAD_BACKWARD_COMPLETE"],
        fixed_probe_generator_forward_returns=canonical["PROBE_FORWARD_COMPLETE"],
        fixed_probe_generator_forward_starts=canonical["PROBE_FORWARD_STARTED"],
        fixed_probe_completed_invocations=sum(len(x.get("completed_probes", [])) for x in arms),
        fixed_probe_completed_head_forwards=probe_heads,
        RAW_complete=get("RAW_complete"), POST_complete=get("POST_complete"),
        sampling_generator_forward_starts=get("generator_forward_started"),
        sampling_generator_forward_returns=get("generator_forward_returns"),
        D0_forward_starts=get("D0_forward_started"), D0_forward_returns=get("D0_forward_returns"),
        inference_layout_head_forward_returns=get("layout_head_forward_returns"),
        inference_layout_unresolved=get("layout_unresolved"),
        GEO_complete_calls=get("GEO_complete_calls"))
    totals["physical_generator_optimizer_returns_including_discarded"] = canonical["OPTIMIZER_COMPLETE"] + extra["optimizer_updates"]
    totals["physical_generator_training_forward_returns_including_discarded"] = canonical["FORWARD_COMPLETE"] + extra["training_forward_returns"]
    totals["physical_generator_training_backward_returns_including_discarded"] = canonical["BACKWARD_COMPLETE"] + extra["training_backward_returns"]
    totals["physical_generator_training_backward_starts_including_discarded"] = canonical["BACKWARD_STARTED"] + extra["training_backward_starts"]
    totals["recovery_backward_unknown"] = extra["backward_result_unknown"]
    totals["all_generator_forward_returns_training_probe_sampling_including_discarded"] = (
        totals["physical_generator_training_forward_returns_including_discarded"]
        + totals["fixed_probe_generator_forward_returns"]
        + totals["sampling_generator_forward_returns"])
    totals["all_known_head_forward_returns"] = (canonical["HEAD_FORWARD_COMPLETE"] + probe_heads
        + totals["inference_layout_head_forward_returns"])
    issues = [{"arm": x["arm"], "mismatches": x["mismatches"]} for x in arms if x.get("mismatches")]
    issues += [{"label": x["label"], "request_id": x["request_id"], "mismatches": x["mismatches"]}
               for x in outputs if x["mismatches"]]
    pending = dict(
        training_started_minus_returned={x["arm"]: x.get("pending_started_minus_returned", {}) for x in arms},
        probes={x["arm"]: x.get("pending_probe_steps", {}) for x in arms},
        layouts=[x for x in layouts if x["occupancy_forward"] is None],
        outputs=[{"label": x["label"], "request_id": x["request_id"],
                  "raw_complete": x["raw_complete"], "post_complete": x["post_complete"]}
                 for x in outputs if not (x["raw_complete"] and x["post_complete"])])
    if final:
        assert decision and decision["stop_training"], "Final accounting requires an existing stop decision"
        assert not issues and not pending["layouts"] and not pending["outputs"]
        assert not any(pending["probes"].values())
        assert not any(v for item in pending["training_started_minus_returned"].values() for v in item.values())
    return dict(schema="bench_parts_actual_execution_accounting_v1", status="FINAL" if final else "PRELIMINARY",
        snapshot_started_utc=started, snapshot_finished_utc=now(), experiment_root=str(root),
        scope="All production COMMON/PLAIN/SEAM canonical ledgers, all NATIVE_S*/SEAM_S* evaluation labels, and audited discarded recovery attempts",
        counts_are_actual_logged_events_not_budget_products=True,
        source_snapshots_are_sequential_not_atomic=True,
        scientific_updates_exclude_discarded_attempts=True,
        completion_markers_are_cross_reference_only_not_added_to_calls=True,
        completed_PROBE_head_calls_are_recorded_once_per_condition_not_per_time_bin=True,
        limitations=["Incomplete probe head calls are unlogged until PROBE_COMPLETE and remain UNKNOWN.",
                     "Counts certify execution, not metric quality, visual approval, or final artifact completeness.",
                     "CPU data preparation, rendering, source fixtures, and historical other experiments are outside this neural-call accounting."],
        latest_stop_decision={k: decision[k] for k in ("status", "current_global_step", "stage_updates", "stop_training", "next_common_target") if k in decision} if decision else None,
        totals=totals, training=arms, evaluation_labels=labels, evaluation_outputs=outputs,
        inference_layouts=layouts, recovery_extra=extra, recovery_accounting=recovery,
        consistency_issues=issues, pending_or_unresolved=pending, sources=sources)


def report(value):
    totals = value["totals"]
    lines = ["# Bench 执行计算账本（" + value["status"] + "）", "",
             "该表由已有账本与完成收据读取；没有运行模型、重评分或改变训练。",
             "快照窗口：`" + value["snapshot_started_utc"] + "` 至 `" + value["snapshot_finished_utc"] + "`。",
             "各文件顺序取快照，训练继续时不是同一瞬间的原子快照。", "",
             "## 训练与固定探针", "",
             "| 臂 | 已完成唯一更新 | 最后 UPDATE_COMPLETE | 生成器训练前向返回 | 固定探针前向返回 | 完整探针次数 | 小头训练前向返回 | 小头探针前向 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in value["training"]:
        e = a.get("event_counts", {})
        lines.append(f"| {a['arm']} | {a.get('unique_completed_updates', 0)} | {a.get('maximum_completed_step')} | {e.get('FORWARD_COMPLETE', 0)} | {e.get('PROBE_FORWARD_COMPLETE', 0)} | {len(a.get('completed_probes', []))} | {e.get('HEAD_FORWARD_COMPLETE', 0)} | {a.get('completed_probe_head_forwards', 0)} |")
    lines += ["", "## 全过程采样与后处理（不只计算最终模型）", "",
              "| 标签 | RAW 完成 | POST 完成 | 采样前向返回 | D0 返回 | 布局头返回 |", "|---|---:|---:|---:|---:|---:|"]
    for label in value["evaluation_labels"]:
        lines.append(f"| {label['label']} | {label['RAW_complete']} | {label['POST_complete']} | {label['generator_forward_returns']} | {label['D0_forward_returns']} | {label['layout_head_forward_returns']} |")
    lines += ["", "## 总数与恢复开销", "", "```json", json.dumps(totals, ensure_ascii=False, indent=2), "```", "",
              "三次中断的额外已完成更新为 77 + 112 + 136 = 325；额外训练前向返回 2604、反向返回 2603，另有 1 次反向结果未知。这些是物理计算开销，不增加科学训练步数。原始恢复核验完整嵌入 JSON。",
              "固定探针小头仅计入已完成探针记录的实际 head_forwards；若某探针未结束，其内部未单独记账的小头调用仍标 UNKNOWN。闭合 C 的布局头调用为 0，不按请求数量反推调用。",
              "", "停止判定：`" + json.dumps(value["latest_stop_decision"], ensure_ascii=False) + "`。",
              "一致性问题：`" + json.dumps(value["consistency_issues"], ensure_ascii=False) + "`。",
              "尚未完成项与逐收据来源哈希见 JSON。本文件不能替代最终报告或代表模型已晋级。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    value = build(args.root.resolve(), args.final)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = HERE / "snapshots" / stamp
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "execution_counts.json", value)
    (output / "EXECUTION_COUNTS.md").write_text(report(value), encoding="utf-8")
    dump(HERE / "latest.json", dict(snapshot=str(output), status=value["status"],
                                    created_utc=now(), source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(snapshot=str(output), status=value["status"], seconds=time.monotonic()-started,
                          totals=value["totals"], issues=value["consistency_issues"]), ensure_ascii=False, indent=2))
