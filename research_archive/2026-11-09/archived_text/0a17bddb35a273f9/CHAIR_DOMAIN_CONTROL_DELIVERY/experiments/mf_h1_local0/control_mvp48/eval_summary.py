"""CPU-only summaries for the frozen Control MVP48 evaluation.

No model loading, geometry scoring, checkpoint choice, or automatic verdicts.
The caller supplies the frozen 96-request manifest, independently scored rows,
metric directions/units, and evidence-grounded answers to the four report questions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import statistics
from collections import Counter
from pathlib import Path

MODELS = ("T1", "Geo", "Full")
COMPARISONS = (("Geo-T1", "Geo", "T1"), ("Full-Geo", "Full", "Geo"),
               ("Full-T1", "Full", "T1"))
KEY_FIELDS = ("parent_id", "condition_id", "noise_id", "model")
PENDING_STATUSES = frozenset(("NOT_RUN", "NOT_SCORED", "PENDING", "REGISTERED", "RUNNING"))
QUESTIONS = (
    "T1是否跨出了双对象沙盒？",
    "Geo是否有初步的独立父对象增量？",
    "Full的接口/质量收益是否值得保留？",
    "下一步应扩大数据、增加训练预算，还是先修数据/实现？",
)


def finite(value):
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_safe(value):
    """Do not silently serialize NaN/Inf as legal metric observations."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            return None
        return value.item() if hasattr(value, "item") else value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def key_sort(key):
    return str(key[0]), str(key[1]), int(key[2]), MODELS.index(key[3])


def validate_manifest(expected_requests):
    """The formal matrix cannot grow by adding an extra seed/repeat/model."""
    keys = [row_key(r) for r in expected_requests]
    if len(keys) != 96 or len(set(keys)) != 96:
        raise ValueError("The frozen formal manifest must contain 96 unique requests")
    parents = sorted({k[0] for k in keys}, key=str)
    if len(parents) != 8 or {k[3] for k in keys} != set(MODELS):
        raise ValueError("Expected eight parents and precisely T1/Geo/Full")
    for parent in parents:
        pkeys = {k for k in keys if k[0] == parent}
        conditions = {k[1] for k in pkeys}
        expected = {(parent, c, n, m) for c in conditions for n in (0, 1) for m in MODELS}
        if len(conditions) != 2 or pkeys != expected:
            raise ValueError(f"Parent {parent!r} must have 2 C x noise_id 0/1 x 3 models")
    return parents


def materialize_rows(expected_requests, observed_rows):
    validate_manifest(expected_requests)
    expected = {row_key(r): dict(r) for r in expected_requests}
    observed = {}
    for row in observed_rows:
        key = row_key(row)
        if key not in expected or key in observed:
            raise ValueError(f"Unregistered or duplicated formal result: {key!r}")
        if "task_id" in row and "task_id" in expected[key] and row["task_id"] != expected[key]["task_id"]:
            raise ValueError(f"Task identity mismatch: {key!r}")
        observed[key] = dict(row)
    rows = []
    for key in sorted(expected, key=key_sort):
        row = dict(expected[key])
        if key in observed:
            row.update(observed[key])
            row.setdefault("status", "NOT_SCORED")
        else:
            row.update(status="NOT_RUN", failure_reason="No result for this registered request")
        row.setdefault("metrics", {})
        row.setdefault("exact_topology", "NOT_RUN")
        row.setdefault("watertight_certification", "NOT_RUN")
        rows.append(row)
    return rows


def metric_value(row, metric):
    value = row.get("metrics", {}).get(metric)
    return float(value) if row.get("status") == "PASS" and finite(value) else None


def attempted(row):
    """Pending records alone do not establish model execution.

    A raw output awaiting scoring may contain explicit attempted NFE. Terminal
    statuses establish an attempted task, including failure before a forward;
    actual forward counts remain in the separate NFE ledger.
    """
    return (row.get("attempted") is True
            or (finite(row.get("nfe")) and float(row["nfe"]) > 0)
            or row.get("status", "NOT_RUN") not in PENDING_STATUSES
            or row.get("status") == "RUNNING")


def _request_record(row):
    return {**{k: row[k] for k in KEY_FIELDS}, "task_id": row.get("task_id"),
            "status": row["status"], "failure_reason": row.get("failure_reason", row.get("error")),
            "raw_path": row.get("raw_path"), "nfe": row.get("nfe")}


def _stats(values):
    valid = [float(v) for v in values if finite(v)]
    return dict(requested=len(values), valid=len(valid), missing=len(values) - len(valid),
                mean=statistics.mean(valid) if valid and len(valid) == len(values) else None,
                available_mean=statistics.mean(valid) if valid else None)


def _worst(values, direction, key_name):
    valid = [(key, float(value)) for key, value in values.items() if finite(value)]
    if not valid or direction == "descriptive":
        return None
    function = min if direction == "higher" else max
    extreme = function(v for _, v in valid)
    # Preserve all exact ties; choosing only the first could conceal a condition.
    return [{key_name: list(k) if isinstance(k, tuple) else k, "value": v}
            for k, v in valid if v == extreme]


def _aggregate(parent_values, case_values, direction, *, paired=False):
    pstats, cstats = _stats(list(parent_values.values())), _stats(list(case_values.values()))
    result = dict(direction=direction, parents_requested=pstats["requested"],
                  parents_complete=pstats["valid"], outputs_requested=cstats["requested"],
                  outputs_valid=cstats["valid"], equal_parent_mean=pstats["mean"],
                  available_complete_parent_mean=pstats["available_mean"],
                  diagnostic_available_case_mean=cstats["available_mean"],
                  worst_parent=_worst(parent_values, direction, "parent_id"),
                  worst_condition=_worst(case_values, direction, "parent_condition_noise"),
                  per_parent=dict(parent_values))
    if paired:
        valid = [float(v) for v in parent_values.values() if finite(v)]
        negative, positive = sum(v < 0 for v in valid), sum(v > 0 for v in valid)
        result.update(parent_paired_median_delta=statistics.median(valid) if valid else None,
                      improved_parents=(positive if direction == "higher" else negative)
                          if direction != "descriptive" else None,
                      worsened_parents=(negative if direction == "higher" else positive)
                          if direction != "descriptive" else None,
                      tied_parents=sum(v == 0 for v in valid),
                      paired_parent_denominator=len(valid),
                      improvement_rule="Exact signed parent mean difference; no success threshold")
    return result


def build_summary(expected_requests, observed_rows, metric_specs, *, teacher_rows=(),
                  accounting=None, identity=None, protocol=None, control=None,
                  conclusions=None, figures=None):
    """Build deterministic reports; only PASS/finite observations enter metrics.

    metric_specs maps names to {direction: lower|higher|descriptive, unit: str}.
    Each parent requires all four registered outputs for its main mean; the
    main aggregate requires all eight parents. Incomplete diagnostics remain
    available under explicit names and denominators, never replace that mean.
    """
    parents = validate_manifest(expected_requests)
    for name, spec in metric_specs.items():
        if spec.get("direction") not in ("lower", "higher", "descriptive") or "unit" not in spec:
            raise ValueError(f"Metric must have a frozen direction and unit: {name}")
    rows = materialize_rows(expected_requests, observed_rows)
    index = {row_key(r): r for r in rows}
    cases = sorted({row_key(r)[:3] for r in rows}, key=lambda k: (str(k[0]), str(k[1]), k[2]))
    per_parent = []
    models = {model: {} for model in MODELS}
    paired = {name: {} for name, _, _ in COMPARISONS}
    parent_model_metrics = {}
    for parent in parents:
        record = {"parent_id": parent, "formal_requested": 12}
        for model in MODELS:
            selected = [r for r in rows if r["parent_id"] == parent and r["model"] == model]
            record[f"{model}__requested"] = 4
            record[f"{model}__passed"] = sum(r["status"] == "PASS" for r in selected)
            record[f"{model}__attempted"] = sum(attempted(r) for r in selected)
            record[f"{model}__pending"] = sum(r["status"] in PENDING_STATUSES for r in selected)
            record[f"{model}__failed"] = sum(r["status"] != "PASS" and r["status"] not in PENDING_STATUSES
                                             for r in selected)
            for metric in metric_specs:
                stats = _stats([metric_value(r, metric) for r in selected])
                parent_model_metrics[parent, model, metric] = stats["mean"]
                for field in ("mean", "available_mean", "valid", "requested"):
                    record[f"{model}__{metric}__{field}"] = stats[field]
        for name, after, before in COMPARISONS:
            for metric in metric_specs:
                a, b = (parent_model_metrics[parent, m, metric] for m in (after, before))
                record[f"{name}__{metric}__delta"] = a - b if finite(a) and finite(b) else None
        per_parent.append(record)
    for metric, spec in metric_specs.items():
        for model in MODELS:
            pv = {p: parent_model_metrics[p, model, metric] for p in parents}
            cv = {c: metric_value(index[c + (model,)], metric) for c in cases}
            models[model][metric] = _aggregate(pv, cv, spec["direction"])
        for name, after, before in COMPARISONS:
            pv = {p: next(r for r in per_parent if r["parent_id"] == p)[f"{name}__{metric}__delta"]
                  for p in parents}
            cv = {}
            for case in cases:
                a, b = (metric_value(index[case + (m,)], metric) for m in (after, before))
                cv[case] = a - b if finite(a) and finite(b) else None
            paired[name][metric] = _aggregate(pv, cv, spec["direction"], paired=True)
    failures = [_request_record(r) for r in rows
                if r["status"] != "PASS" and r["status"] not in PENDING_STATUSES]
    pending = [_request_record(r) for r in rows if r["status"] in PENDING_STATUSES]
    teacher = list(teacher_rows)
    teacher_seen = set()
    for row in teacher:
        key = row["parent_id"], row["model"]
        if row["model"] not in MODELS or key in teacher_seen:
            raise ValueError(f"Unexpected or duplicate validation teacher row: {key}")
        if row.get("t", 0.5) != 0.5:
            raise ValueError("The small validation teacher check is frozen at t=0.5")
        teacher_seen.add(key)
    if len(teacher) > 24 or len({r["parent_id"] for r in teacher}) > 8:
        raise ValueError("Validation teacher exceeds 8 parents x 3 endpoint models")
    nfe_rows = [r.get("nfe") for r in rows]
    nfe_stats = _stats(nfe_rows)
    formal_passed = sum(r["status"] == "PASS" for r in rows)
    teacher_attempted = sum(attempted(r) for r in teacher)
    teacher_passed = sum(r.get("status") == "PASS" for r in teacher)
    result = dict(schema="control_mvp48_eval_v1", status="COMPLETE_SCORED_MATRIX" if formal_passed == 96 else "INCOMPLETE_OR_FAILED_MATRIX",
                  formal_requested=96, formal_passed=formal_passed,
                  formal_attempted=sum(attempted(r) for r in rows),
                  formal_failed=len(failures), formal_pending=len(pending),
                  formal_not_run=sum(r["status"] == "NOT_RUN" for r in rows),
                  formal_not_scored=sum(r["status"] == "NOT_SCORED" for r in rows),
                  formal_status_counts=dict(Counter(r["status"] for r in rows)),
                  expected_parent_count=8, expected_cases_per_parent_model=4,
                  metric_specs=metric_specs, per_output=rows, per_parent=per_parent,
                  models=models, comparisons=paired, failures=failures, pending_requests=pending,
                  teacher=dict(requested=24, registered=len(teacher_seen), recorded=len(teacher),
                               attempted=teacher_attempted, passed=teacher_passed, rows=teacher,
                               failed=sum(r.get("status", "NOT_RUN") not in PENDING_STATUSES
                                          and r.get("status") != "PASS" for r in teacher),
                               pending=sum(r.get("status", "NOT_RUN") in PENDING_STATUSES for r in teacher),
                               status_counts=dict(Counter(r.get("status", "NOT_RUN") for r in teacher)),
                               complete=teacher_passed == 24 and len({r["parent_id"] for r in teacher}) == 8,
                               use="Learning-state explanation only; not checkpoint selection or free-generation evidence"),
                  protocol=protocol or {}, identity=identity or {}, accounting=accounting or {},
                  formal_nfe=dict(requested=4800, rows_with_explicit_actual_nfe=nfe_stats["valid"],
                                  recorded_actual_sum=sum(float(v) for v in nfe_rows if finite(v)),
                                  complete_actual_sum=sum(float(v) for v in nfe_rows)
                                      if nfe_stats["valid"] == 96 and not pending else None),
                  control=control or {"status": "NOT_RUN", "reason": "No registered same-N/K four-cell comparison was supplied"},
                  exact_topology="NOT_RUN", watertight_certification="NOT_RUN",
                  conclusions=conclusions or {}, figures=figures or {},
                  grouping_rule="Average 2 C x 2 noises within each parent; then equal weight 8 parents. Incomplete main mean is null. Available-only diagnostics have explicit names and denominators.",
                  scope="Conditional-finetuning-unseen parents; not necessarily official-pretraining-unseen. Eight test objects do not establish publication-level generalization.",
                  reporting_model_forwards=0, scoring_calls_from_reporting=0, default_model_changed=False)
    return json_safe(result)


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = json_safe(row)
            writer.writerow({k: "NA" if v is None else json.dumps(v, ensure_ascii=False, sort_keys=True)
                             if isinstance(v, (list, dict)) else v for k, v in clean.items()})


def _flat_output(row):
    base = {k: v for k, v in row.items() if k != "metrics"}
    collision = set(base).intersection(row.get("metrics", {}))
    if collision:
        raise ValueError(f"Metadata and metric names collide: {sorted(collision)}")
    return dict(base, **row.get("metrics", {}))


def fmt(value):
    return f"{value:.6g}" if finite(value) else "NA"


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def report_text(result):
    conclusions = result.get("conclusions", {})
    lines = ["# Control MVP48 独立评价 — Session C", ""]
    for i, question in enumerate(QUESTIONS, 1):
        answer = conclusions.get(f"q{i}", "尚未给出证据判读；本工具不依据单一指标自动宣布成功。")
        lines += [f"{i}. **{question}** {answer}", ""]
    lines += [f"正式请求 **96**，实际尝试 **{result['formal_attempted']}/96**，评分有效输出 **{result['formal_passed']}/96**；已尝试失败 **{result['formal_failed']}**，待运行或待评分 **{result['formal_pending']}**。所有请求均保留，登记占位行不计作失败尝试。",
              "均值先在父对象内平均两块 C × 两噪声，再对 8 个父对象等权汇总。任何父对象少一个有效输出，该父对象完整均值为 NA；完整八对象总均值也为 NA。可用分母诊断另列于 JSON，不能替代主均值。", "",
              "## 完整性、接口与质量", "",
              "数值沿用原评分器的单位与阈值。C 本身排除于剩余几何分数，每块 C 使用自己的 GT 自由补集；不同 N 的相交对数需结合受影响 free 比例。", ""]
    rows = []
    for metric, spec in result["metric_specs"].items():
        values = [result["models"][m][metric] for m in MODELS]
        rows.append([metric, spec["unit"], *[fmt(v["equal_parent_mean"]) for v in values],
                     " / ".join(f"{v['parents_complete']}/8父,{v['outputs_valid']}/32输出" for v in values)])
    lines += [_table(["指标", "单位", "T1", "Geo", "Full", "有效分母 T1 / Geo / Full"], rows), "",
              "## 父对象配对增量", "",
              "差值方向固定为后者−前者：Geo−T1、Full−Geo、Full−T1。误差/非法相交越低越好，coverage 越高越好；描述性字段不判断改善。中位差和改善数的单位是父对象，不把 32 个 case 当作 32 个独立形状。", ""]
    for comparison, _, _ in COMPARISONS:
        lines += [f"### {comparison}", ""]
        table_rows = []
        for metric, value in result["comparisons"][comparison].items():
            worst_parent = json.dumps(value["worst_parent"], ensure_ascii=False) if value["worst_parent"] else "NA"
            worst_case = json.dumps(value["worst_condition"], ensure_ascii=False) if value["worst_condition"] else "NA"
            table_rows.append([metric, fmt(value["equal_parent_mean"]), fmt(value["parent_paired_median_delta"]),
                               f"{value['improved_parents']}/{value['paired_parent_denominator']}" if value["improved_parents"] is not None else "描述性",
                               worst_parent, worst_case])
        lines += [_table(["指标", "均值差", "父对象配对中位差", "改善父对象/有效父对象", "最差父对象(Δ)", "最差对象/C/噪声(Δ)"], table_rows), ""]
    lines += ["## 失败、验证与控制边界", ""]
    if result["failures"]:
        lines += [_table(["父对象", "C", "噪声", "模型", "状态", "原因"],
                         [[r["parent_id"], r["condition_id"], r["noise_id"], r["model"], r["status"], r["failure_reason"]]
                          for r in result["failures"]]), ""]
    elif result["formal_passed"] == 96:
        lines += ["96 条正式请求均有有效评分；无输出因几何表现差被替换或删除。", ""]
    else:
        lines += ["尚无已尝试失败记录；待运行或待评分请求仍未成为有效观测。", ""]
    if result["pending_requests"]:
        lines += [f"待运行/待评分状态：{json.dumps(dict(Counter(r['status'] for r in result['pending_requests'])), ensure_ascii=False)}。逐项身份保留在 per_output.csv 与 results.json 的 pending_requests 中。", ""]
    lines += [f"验证 teacher 请求 **24**；已登记 **{result['teacher']['registered']}**、表内记录 **{result['teacher']['recorded']}**、实际尝试 **{result['teacher']['attempted']}/24**、有效检查 **{result['teacher']['passed']}/24**；登记或待运行记录不是已执行的前向。检查固定 t=.5、第一块 C、冻结 teacher 噪声及配对。teacher 的 GT/OT 状态与自由生成严格分开，仅解释学习状态，不替代完整生成、不挑 checkpoint、不推断完整控制能力。", "",
              f"同 N/K 正确/交换条件比较：**{result['control'].get('status', 'NOT_RUN')}**。{result['control'].get('reason', '')}", "",
              "同一父对象的两块 C 来自同一完整目标，输出不同不等于几何编辑成功，也不要求完整形状不同。若未完成冻结的跨父对象四格比较，本轮尚未全面验证几何编辑控制。", "",
              "精确拓扑与水密认证：**NOT_RUN**。C 保持、实际面数、退化、NaN/Inf、UNRESOLVED、角质量以逐输出原记录为准，未运行的认证不填 0。", "",
              "## 协议、身份与实际账本", "",
              "协议目标终点：T1_step1000、Geo_step1000、Full_step1000；实际统一严格加载及身份核对状态详见 results.json 的 identity，共同官方初始化、数据 hash、任务步数、分支来源按 protocol 核对。无 step500 选择，无 raw/EMA 混选。", "",
              "原 Euler50；BF16 前向、FP32 参数/状态/积分；严格确定性、TF32 关闭、SDPA math、eval/no_grad；每条 51 状态保持 C。raw known 速度不要求恒零。没有训练 loss、自由生成 GT/OT 初始化、Guidance、Pair 或几何修补。", "",
              f"正式 NFE 请求 4800；显式记录实际 NFE 的输出 {result['formal_nfe']['rows_with_explicit_actual_nfe']}/96，已记录实际前向合计 {fmt(result['formal_nfe']['recorded_actual_sum'])}；完整实际合计 {fmt(result['formal_nfe']['complete_actual_sum'])}。teacher、必要重复和失败尝试另见实际账本，不混入 96 条。", "",
              "```json", json.dumps(result["accounting"], ensure_ascii=False, indent=2), "```", "",
              "GPU 阶段要求使用共同锁且只有一个 worker；CPU 评分可与后续推理重叠。实际资源快照、峰值及分阶段 wall 以账本为准，并行争用下的 wall 不作为隔离性能比较。", "",
              "## 交付与限制", "",
              "[results.json](results.json)、[逐请求 96 行](per_output.csv)、[逐父对象 8 行](per_parent.csv)、[24 请求 teacher 表](teacher_validation.csv)、[固定视图](../figures/)。正式视图目标是固定第一份噪声，8 个对象的两块 C 均展示 GT、C、T1、Geo、Full，同视角同范围，不删除失败面板。准备阶段的 GT/C 预览与 NOT_RUN 模型占位面板不代表已生成模型输出；实际图件状态见 results.json 的 figures。", "",
              "本轮证据范围是条件微调未参与的 8 个父对象，不等于官方预训练从未见过；没有论文级泛化或统计定论。1000 步无效果仅能称该预算下未成立，不能否定整个方法；持续学习的终点不能称充分收敛。没有临时 10% 成功门槛或偏向 Full 的总分。", "",
              "正式 96 条完成后停止；不自动扩训、扫 loss、改求解器或切换默认模型。", ""]
    return "\n".join(lines)


def write_outputs(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(result["per_output"]) != 96 or len(result["per_parent"]) != 8:
        raise ValueError("Formal tables must retain all 96 requests and eight parents")
    write_json(output_dir / "results.json", result)
    csv_write(output_dir / "per_output.csv", [_flat_output(r) for r in result["per_output"]])
    csv_write(output_dir / "per_parent.csv", result["per_parent"])
    csv_write(output_dir / "teacher_validation.csv", [_flat_output(r) for r in result["teacher"]["rows"]])
    (output_dir / "MVP48_REPORT.md").write_text(report_text(result), encoding="utf-8")
    return {path.name: file_sha256(path) for path in sorted(output_dir.iterdir())
            if path.name in ("results.json", "per_output.csv", "per_parent.csv", "teacher_validation.csv", "MVP48_REPORT.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON containing build_summary arguments; frozen expected_requests required")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_summary(**read_json(args.input))
    hashes = write_outputs(args.out, result)
    print(json.dumps(dict(status=result["status"], formal_passed=result["formal_passed"], artifacts=hashes)))


if __name__ == "__main__":
    main()
