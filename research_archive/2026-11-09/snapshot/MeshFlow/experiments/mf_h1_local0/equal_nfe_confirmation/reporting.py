"""CPU-only, complete-denominator reporting of the frozen equal-NFE matrix.

This module never loads a model, scores geometry, selects outputs, or changes defaults.
All differences are formed within the registered case before any aggregation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

MODELS = ("B", "J")
SOLVERS = ("Euler50", "Midpoint25", "E_quad50")
EFFECTS = ("Delta_B_M", "Delta_B_Q", "Delta_J_M", "Delta_J_Q",
           "A_E", "A_M", "A_Q", "I_M", "I_Q")
MAIN_METRICS = (
    "remainder_symmetric_RMS_bbox_pct", "free_to_R12_RMS_bbox_pct",
    "R12_to_free_RMS_bbox_pct", "fixed_R12_coverage_within_1pct_fraction",
    "boundary_p95_bbox_pct", "boundary_coverage_fraction",
    "Gamma_to_free_edges_p95_bbox_pct", "Gamma_to_free_edges_coverage_0.001_fraction",
    "CF_proper_pair_count", "CF_proper_affected_free_fraction", "FF_illegal_pair_count",
    "free_minimum_angle_lt5_fraction", "free_q_gt10_fraction")
NEUTRAL_METRICS = {
    "near_C_area_fraction", "near_C_boundary_band_area_fraction",
    "near_C_outside_boundary_band_area_fraction", "C_surface_area_fraction",
    "free_total_surface_area", "free_total_area", "GT_free_total_area",
    "free_area_over_GT_free_area", "shared_edge_component_count",
    "exact_C_free_shared_edge_count", "exact_C_free_shared_vertex_count",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def direction(metric):
    """No invented composite score or monotonic claim for descriptive areas/counts."""
    if metric in NEUTRAL_METRICS:
        return "descriptive"
    if ("coverage" in metric and "RMS" not in metric) or metric == "free_angle_p05_degrees":
        return "higher"
    return "lower"


def case_key(row):
    return int(row["parent"]), int(row["patch"]), int(row["seed"])


def summarize(values, metric, differences=False):
    """values maps every requested (parent,patch,seed) to a scalar or explicit NA.

    Full hierarchical mean is NA if any requested seed is absent. Available-only
    statistics are separately named, with both sample and complete-cell counts.
    The paired median always uses case differences, never marginal medians.
    """
    valid = {k: float(v) for k, v in values.items() if number(v)}
    cells = defaultdict(dict)
    for key, value in values.items():
        cells[key[:2]][key] = value
    cell_means = {}
    for cell, members in cells.items():
        usable = [float(v) for v in members.values() if number(v)]
        cell_means[f"P{cell[0]}_patch{cell[1]}"] = {
            "mean": statistics.mean(usable) if len(usable) == len(members) else None,
            "available_mean": statistics.mean(usable) if usable else None,
            "valid": len(usable), "requested": len(members),
        }
    complete = [r["mean"] for r in cell_means.values() if r["mean"] is not None]
    d = direction(metric)
    result = dict(requested=len(values), valid=len(valid), missing=len(values)-len(valid),
                  cells_requested=len(cells), cells_complete=len(complete),
                  direction=d, cells=cell_means,
                  hierarchical_mean=statistics.mean(complete) if len(complete) == len(cells) and complete else None,
                  available_complete_cell_mean=statistics.mean(complete) if complete else None,
                  available_case_mean=statistics.mean(valid.values()) if valid else None,
                  paired_median_delta=statistics.median(valid.values()) if valid and differences else None)
    if valid:
        worst_key = (min if d == "higher" else max)(valid, key=valid.get)
        result["worst_case"] = dict(parent=worst_key[0], patch=worst_key[1], seed=worst_key[2], value=valid[worst_key])
        result["minimum"] = min(valid.values())
        result["maximum"] = max(valid.values())
    else:
        result["worst_case"] = result["minimum"] = result["maximum"] = None
    if differences:
        negative = sum(v < 0 for v in valid.values())
        positive = sum(v > 0 for v in valid.values())
        result.update(negative=negative, equal=sum(v == 0 for v in valid.values()), positive=positive,
                      improved=(positive if d == "higher" else negative) if d != "descriptive" else None,
                      worsened=(negative if d == "higher" else positive) if d != "descriptive" else None,
                      sign_rule="exact raw difference; no post-hoc epsilon or success threshold")
    return result


def difference(a, b):
    return float(a)-float(b) if number(a) and number(b) else None


def make_effects(rows, metrics):
    by = {(r["model"], r["solver"], case_key(r)): r for r in rows}
    cases = sorted({case_key(r) for r in rows})
    result = []
    for case in cases:
        def value(model, solver, metric):
            return by.get((model, solver, case), {}).get("values", {}).get(metric)
        diffs = {name: {} for name in EFFECTS}
        for metric in metrics:
            for code, solver in (("M", "Midpoint25"), ("Q", "E_quad50")):
                for model in MODELS:
                    diffs[f"Delta_{model}_{code}"][metric] = difference(value(model, solver, metric), value(model, "Euler50", metric))
                diffs[f"I_{code}"][metric] = difference(diffs[f"Delta_J_{code}"][metric], diffs[f"Delta_B_{code}"][metric])
            for code, solver in zip(("E", "M", "Q"), SOLVERS):
                diffs[f"A_{code}"][metric] = difference(value("J", solver, metric), value("B", solver, metric))
        for effect in EFFECTS:
            result.append(dict(parent=case[0], patch=case[1], seed=case[2], effect=effect, differences=diffs[effect]))
    return result


def group_cases(cases):
    cases = sorted(cases)
    groups = {"all": cases}
    for p in (0, 1):
        groups[f"P{p}"] = [c for c in cases if c[0] == p]
    groups["patch0_training_C"] = [c for c in cases if c[1] == 0]
    groups["patch1_4_observed_C"] = [c for c in cases if c[1] in (1, 2, 3, 4)]
    for p in (0, 1):
        for patch in range(5):
            groups[f"P{p}_patch{patch}"] = [c for c in cases if c[:2] == (p, patch)]
    for seed in sorted({c[2] for c in cases}):
        groups[f"seed{seed}"] = [c for c in cases if c[2] == seed]
    return groups


def summarize_all(rows, effects, metrics):
    cases = sorted({case_key(r) for r in rows})
    groups = group_cases(cases)
    configs, paired = {}, {}
    for group, keys in groups.items():
        configs[group], paired[group] = {}, {}
        keyset = set(keys)
        for model in MODELS:
            for solver in SOLVERS:
                selected = {case_key(r): r for r in rows if r["model"] == model and r["solver"] == solver and case_key(r) in keyset}
                name = model + "/" + solver
                configs[group][name] = {
                    "requested": len(keys), "scored": sum(r["status"] == "PASS" for r in selected.values()),
                    "metrics": {m: summarize({k: selected.get(k, {}).get("values", {}).get(m) for k in keys}, m) for m in metrics},
                }
        for effect in EFFECTS:
            selected = {case_key(r): r for r in effects if r["effect"] == effect and case_key(r) in keyset}
            paired[group][effect] = {m: summarize({k: selected.get(k, {}).get("differences", {}).get(m) for k in keys}, m, True) for m in metrics}
    return configs, paired


def make_binding(rows, seeds, tolerance=0.0):
    by = {(r["model"], r["solver"], int(r["parent"]), int(r["patch"]), int(r["seed"])): r for r in rows}
    metrics = sorted({m for r in rows for entry in r.get("cross_binding", {}).values() for m in entry.get("metrics", {})})
    records = []
    for model in MODELS:
        for solver in SOLVERS:
            for patch in range(5):
                for seed in seeds:
                    a = by.get((model, solver, 0, patch, seed), {})
                    b = by.get((model, solver, 1, patch, seed), {})
                    four = {"EaFa": a.get("cross_binding", {}).get("0", {}),
                            "EaFb": b.get("cross_binding", {}).get("0", {}),
                            "EbFb": b.get("cross_binding", {}).get("1", {}),
                            "EbFa": a.get("cross_binding", {}).get("1", {})}
                    values = {}
                    for metric in metrics:
                        v = {key: entry.get("metrics", {}).get(metric) for key, entry in four.items()}
                        da, db = difference(v["EaFb"], v["EaFa"]), difference(v["EbFa"], v["EbFb"])
                        avg = (da + db) / 2 if number(da) and number(db) else None
                        sign = lambda x: None if not number(x) else (1 if x > tolerance else (-1 if x < -tolerance else 0))
                        values[metric] = dict(**v, delta_a=da, delta_b=db, mean=avg,
                                              delta_a_sign=sign(da), delta_b_sign=sign(db), mean_sign=sign(avg),
                                              both_directions_positive=(sign(da) == 1 and sign(db) == 1) if number(avg) else None)
                    records.append(dict(model=model, solver=solver, patch=patch, seed=seed,
                                        source_P0=a.get("task_id"), source_P1=b.get("task_id"),
                                        status="PASS" if values and all(number(v["mean"]) for v in values.values()) else "INCOMPLETE",
                                        metrics=values, sources={k: {x: v.get(x) for x in ("target_parent", "prediction_parent", "source_task_id", "identity")} for k, v in four.items()}))
    return records


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "NA" if v is None else (json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in row.items()})


def fmt(value):
    return "NA" if not number(value) else f"{value:.6g}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def collect(out):
    protocol = read_json(out / "registration/protocol.json")
    tasks = [t for group in protocol["groups"] for t in group["tasks"]]
    formal = [t for t in tasks if t["kind"] == "formal"]
    expected = {(m, s, p, patch, seed) for m in MODELS for s in SOLVERS for p in (0, 1) for patch in range(5) for seed in protocol["generation_seeds"]}
    actual = {(t["model"], t["solver"], int(t["parent"]), int(t["patch"]), int(t["seed"])) for t in formal}
    if len(formal) != 120 or actual != expected or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Protocol does not contain the exact unique formal 120-request matrix")
    worker_rows, score_rows, workers, scores = {}, {}, [], []
    for group in protocol["groups"]:
        gid = group["group_id"]
        for folder, dest, records in (("workers", worker_rows, workers), ("scores", score_rows, scores)):
            path = out / folder / (str(gid) + ".json")
            if not path.exists():
                continue
            data = read_json(path)
            records.append(dict(group_id=gid, path=str(path.relative_to(out)), sha256=sha256(path), record=data))
            for row in data.get("rows", []):
                tid = row["task_id"]
                if tid in dest:
                    raise ValueError(f"Duplicate {folder} task {tid}")
                dest[tid] = row
    known_tasks = {t["task_id"]: t for t in tasks}
    for source in (worker_rows, score_rows):
        if set(source)-set(known_tasks):
            raise ValueError("Unregistered task found")
        for tid, row in source.items():
            for field in ("model", "solver", "parent", "patch", "seed"):
                if field in row and row[field] != known_tasks[tid][field]:
                    raise ValueError(f"Identity mismatch for {tid}: {field}")
    rows = []
    metrics = sorted({key for r in score_rows.values() for key in r.get("scalars54", {})})
    derived = sorted({key for r in score_rows.values() for key in r.get("derived", {}) if number(r["derived"][key]) or r["derived"][key] is None})
    for task in formal:
        tid = task["task_id"]
        worker, score = worker_rows.get(tid, {}), score_rows.get(tid, {})
        row = dict(task, sampling=worker, score=score,
                   status=score.get("status", "NOT_SCORED"), sampling_status=score.get("sampling_status", worker.get("status", "NOT_RUN")),
                   scalars54={m: score.get("scalars54", {}).get(m) for m in metrics},
                   derived={m: score.get("derived", {}).get(m) for m in derived},
                   cross_binding=score.get("cross_binding", {}))
        row["values"] = dict(row["scalars54"], **row["derived"])
        rows.append(row)
    rows.sort(key=lambda r: (r["parent"], r["patch"], r["seed"], MODELS.index(r["model"]), SOLVERS.index(r["solver"])))
    return protocol, rows, metrics, derived, workers, scores


def audit_totals(workers, scores):
    """Keep source audit records: never guess attempts from requested NFE."""
    totals = defaultdict(float)
    fields = ("model_forward_attempts", "model_forward_returns", "complete_rollouts", "failed_model_calls")
    for w in workers:
        r = w["record"]
        counts = r.get("counts", r.get("ledger", {}))
        for key in fields:
            if number(counts.get(key)):
                totals[key] += counts[key]
    return {"worker_totals_if_explicit": dict(totals),
            "worker_records": workers, "scoring_records": scores,
            "timing_note": "Per-phase walls and CUDA events are nested. Concurrent CPU task seconds are not end-to-end user waiting time. Resource contention prevents an isolated speed comparison."}


def report_text(result):
    cfg, eff = result["configuration_summaries"], result["paired_summaries"]
    rms = "remainder_symmetric_RMS_bbox_pct"
    def mean(group, name, metric):
        return fmt(cfg[group][name]["metrics"].get(metric, {}).get("hierarchical_mean"))
    names = list(cfg["all"])
    text = ["# JEdge5 Equal-NFE Sampler Confirmation", "",
        "**是否应把 Midpoint25 作为下一阶段统一工作配置：尚待依据下列完整证据作最终科学判读；本自动报告不设置事后胜出阈值，不修改公共默认采样器。**", "",
        "本轮新结果独立于旧 J/P0 四个开发案例。B 为纯 FM、J 为 JEdge5，均使用 raw 累计 3000 步权重；三个固定方法都为 50 NFE。Midpoint25 是主要确认候选，E_quad50 是次要候选，Euler50 是共同参照。", "",
        f"正式请求 120；完成评分 {result['formal_scored']}；未完成评分 {120-result['formal_scored']}。每配置请求分母固定为 20，失败与缺失仍保留。", "",
        "## 六配置主表", "",
        "均值先在每个 parent/patch 内平均两 seed，再对十个单元等权平均。任何成员缺失时完整均值为 NA；JSON 另列可用配对统计、完整单元统计及有效分母。RMS/p95 为 % bbox，coverage 为比例，计数为每输出均值。", ""]
    panels = [
        (["配置", "剩余RMS", "free→GT", "GT→free", "固定coverage@1%L", "Γsurface p95", "Γsurface cov@0.1%L"],
         [rms, "free_to_R12_RMS_bbox_pct", "R12_to_free_RMS_bbox_pct", "fixed_R12_coverage_within_1pct_fraction", "boundary_p95_bbox_pct", "boundary_coverage_fraction"]),
        (["配置", "Γedge p95", "Γedge cov@0.1%L", "CF", "CF受影响free", "FF", "角<5°", "q>10"],
         ["Gamma_to_free_edges_p95_bbox_pct", "Gamma_to_free_edges_coverage_0.001_fraction", "CF_proper_pair_count", "CF_proper_affected_free_fraction", "FF_illegal_pair_count", "free_minimum_angle_lt5_fraction", "free_q_gt10_fraction"])]
    for headers, metrics in panels:
        text += [table(headers, [[name]+[mean("all", name, m) for m in metrics] for name in names]), ""]
    status_rows = []
    for name in names:
        model, solver = name.split("/")
        rows = [r for r in result["rows"] if r["model"] == model and r["solver"] == solver]
        success = sum(r["status"] == "PASS" for r in rows)
        unresolved = sum((r["score"].get("unresolved_pair_count") or 0) > 0 for r in rows)
        nfe = sum(r["sampling"].get("counts_after", {}).get("model_forward_attempts", 0)-r["sampling"].get("counts_before", {}).get("model_forward_attempts", 0) for r in rows)
        wall = sum(r["sampling"].get("audit", {}).get("sampling_wall_seconds", 0) for r in rows)
        cuda = sum(r["sampling"].get("audit", {}).get("model_forward_cuda_event_seconds", 0) or 0 for r in rows)
        status_rows.append([name, f"{success}/{20-success}/{unresolved}", nfe, fmt(wall), fmt(cuda)])
    text += [table(["配置", "评分成功/失败或缺失/未决输出", "实际前向尝试", "采样wall合计s", "模型CUDA事件合计s"], status_rows), "",
        "成功表示执行与评分完成，不等于几何合法；未决输出数可与成功数重叠。连续指标自身有效分母见 JSON。", "",
        "## 配对归因", "",
        "ΔB/ΔJ 是同模型候选−Euler；A 是同采样器 J−B；I=ΔJ−ΔB。误差/相交负值改善，coverage 正值改善。表中为分层均差 / 逐case配对中位差，无总分。J 的训练配方增量主要比较 A_M；I 改善不自动表示 J 已优于 B。", ""]
    key_metrics = [rms, "fixed_R12_coverage_within_1pct_fraction", "boundary_p95_bbox_pct", "Gamma_to_free_edges_p95_bbox_pct", "CF_proper_pair_count", "FF_illegal_pair_count"]
    def paired_cell(effect, metric):
        stats = eff["all"][effect].get(metric, {})
        return fmt(stats.get("hierarchical_mean"))+" / "+fmt(stats.get("paired_median_delta"))
    text += [table(["效应", "剩余RMS", "coverage", "Γsurface p95", "Γedge p95", "CF", "FF"],
                   [[name]+[paired_cell(name, m) for m in key_metrics] for name in EFFECTS]), ""]
    for metric in (rms, "boundary_p95_bbox_pct", "CF_proper_pair_count", "FF_illegal_pair_count"):
        if metric not in result["metric_names54"]:
            continue
        text += [f"{metric}：有效配对、方向及尾部（其余全部指标同结构保存在 JSON）。", "",
            table(["效应", "有效/请求", "改善/相等/恶化", "最差case与原始差"],
                [[name, f"{eff['all'][name][metric]['valid']}/{eff['all'][name][metric]['requested']}",
                  "/".join(str(eff["all"][name][metric][k]) for k in ("improved", "equal", "worsened")),
                  json.dumps(eff["all"][name][metric]["worst_case"], ensure_ascii=False)] for name in EFFECTS]), ""]
    text += ["## 分层结果", "",
        "P0/P1、patch0 训练 C 与 patch1–4 已观察 C、十个单元、两份 seed 分别保留。下表列剩余 RMS；其余 54 指标、面积解释量、各组效应中位差/方向/最差case 均在 results.json。", "",
        table(["分组"]+names, [[group]+[mean(group, name, rms) for name in names] for group in cfg]), "",
        "## 条件绑定", "",
        "60 组来自已有 P0/P1 输出，不新增生成。每组保存 E_a(F_a)、E_a(F_b)、E_b(F_b)、E_b(F_a)，两方向交换−正确差及均值。目标行固定自己的 R12、Gamma、bbox，F 排除复制 C。平均为正不等于两方向均正；coverage 的交换−正确差解释方向与误差相反。原数值规则为 >0/==0，不添加 tie 容差。绑定不是 60 个独立控制任务或新对象成功率。", "",
        f"完整绑定组 {sum(r['status']=='PASS' for r in result['binding'])}/60；四格、原始差值与来源见 [binding.csv](binding.csv)。", "",
        "## 固定视图", "",
        "第一登记 seed 全部十上下文均展示六配置 full/free 与 GT 未知区域覆盖热图。镜头、GT 导出范围、查询点及色阶预登记；超固定范围/色阶计数保留于 figures/plot_manifest.json。", ""]
    for p in (0, 1):
        text.append("、".join(f"[P{p}/patch{patch}](figures/P{p}_patch{patch}_seed{result['generation_seeds'][0]}.png)" for patch in range(5)))
    totals = result["accounting"]["worker_totals_if_explicit"]
    text += ["", "## 执行、限制与交付", "",
        f"原始worker显式计账：{json.dumps(totals, ensure_ascii=False)}。正式预算6000 NFE，历史回归最多200、重复最多100、接线/预热最多20，总硬上限6320。重复不算独立样本，实际失败/尝试保留。", "",
        "共享flock最多一个GPU worker，CPU按资源预算并行。模型前向、采样、落盘、锁等待、加载、CPU评分和绘图分列于JSON原记录；嵌套或重叠时间不累加为用户等待时间。当前计时不是独占硬件或跨设备性能声明。", "",
        "BF16实际前向、FP32参数与积分、严格确定性、TF32关闭、SDPA math。历史BF16 face-permutation FAIL以原容差继承（edge_factorial_20261001_123000/EDGE_FACTORIAL_REPORT.md:1038）；本轮重复通过也不把旧FAIL改为PASS。", "",
        "patch0是原微调C；patch1–4未作为该微调C但已研究观察。两个父对象都参加过训练，patch3与历史片重合11/12，patch4历史重合较低，不称为新对象或全部未见上下文。两seed不支持可靠尾概率、显著性或泛化估计。更低GT误差不等于更精确ODE解，本轮不作数值真值或收敛结论。", "",
        "保留退化与UNRESOLVED，未运行的水密/精确谓词/全局拓扑认证为NOT_RUN。无训练、OT、投影、焊接、删面、重网格；公共默认与权重不变。", "",
        "交付：[results.json](results.json)、[120行逐输出](per_output.csv)、[逐case效应](paired_effects.csv)、[60组绑定](binding.csv)、[协议](registration/protocol.json)、[code.diff](code.diff)、[最终审计](final_audit.json)；raw/保留轨迹。到停止条件后不追加seed、采样器或训练。", ""]
    return "\n".join(text)

def run(out):
    start = time.perf_counter()
    out = Path(out).resolve()
    if out.parent.name != "runs" or not out.name.startswith("equal_nfe_confirmation_"):
        raise ValueError("Only an isolated equal_nfe_confirmation run is writable")
    protocol, rows, metrics, derived, workers, scores = collect(out)
    all_metrics = metrics + [m for m in derived if m not in metrics]
    effects = make_effects(rows, all_metrics)
    configs, paired = summarize_all(rows, effects, all_metrics)
    tolerance = 0.0  # Existing binding uses exact >0 / ==0, no tie tolerance.
    binding = make_binding(rows, protocol["generation_seeds"], tolerance)
    csv_write(out / "per_output.csv", [dict(task_id=r["task_id"], model=r["model"], solver=r["solver"], parent=r["parent"], patch=r["patch"], seed=r["seed"], status=r["status"], sampling_status=r["sampling_status"], sampling_audit=r["sampling"], scoring_identity={k: v for k, v in r["score"].items() if k not in ("scalars54", "derived", "cross_binding")}, **r["values"]) for r in rows])
    csv_write(out / "paired_effects.csv", [dict(parent=r["parent"], patch=r["patch"], seed=r["seed"], effect=r["effect"], **r["differences"]) for r in effects])
    csv_write(out / "binding.csv", [dict(**{k: v for k, v in r.items() if k != "metrics"}, **{f"{m}__{k}": v for m, fields in r["metrics"].items() for k, v in fields.items()}) for r in binding])
    result = dict(status="COMPLETE_SCORED_MATRIX" if all(r["status"] == "PASS" for r in rows) else "INCOMPLETE_OR_FAILED_MATRIX",
                  formal_requested=120, formal_scored=sum(r["status"] == "PASS" for r in rows),
                  generation_seeds=protocol["generation_seeds"], metric_names54=metrics, derived_names=derived,
                  protocol_sha256=sha256(out / "registration/protocol.json"), rows=rows, paired_effects=effects,
                  configuration_summaries=configs, paired_summaries=paired, binding=binding,
                  binding_numerical_tolerance=tolerance, accounting=audit_totals(workers, scores),
                  formal_requested_NFE=6000, total_model_forward_hard_limit=6320,
                  model_forwards_from_reporting=0, default_changed=False, recommendation="PENDING_SCIENTIFIC_REVIEW",
                  grouping_note="Means: seeds within cell, then equal weight cells. Missing full mean=NA. Available statistics explicitly named and denominators preserved.",
                  reporting_seconds=time.perf_counter()-start)
    write_json(out / "results.json", result)
    (out / "EQUAL_NFE_CONFIRMATION_REPORT.md").write_text(report_text(result), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], formal_scored=result["formal_scored"], per_output_rows=len(rows), paired_rows=len(effects), binding_rows=len(binding), metric_count=len(metrics))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
