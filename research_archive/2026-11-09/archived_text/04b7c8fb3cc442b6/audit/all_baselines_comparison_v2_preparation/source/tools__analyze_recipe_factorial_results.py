#!/usr/bin/env python3
"""Read completed recipe-factorial tables; no scoring, geometry, Torch or RNG.

All numeric metrics retain four-noise/parent denominators. Contrasts describe one
training realization on eight observed DEV parents; they never promote a model.
This new analysis module is independent of the frozen execution-source manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

VERSION = "recipe_factorial_table_analysis_v1"
ROOT_NAME = "chair_condition_recipe_factorial_v1"
DEFAULT_OUT = "/mnt/e/MeshFlow-Control-artifacts/experiments/" + ROOT_NAME
LABELS = ("START5000", "HIST_FM6000", "R1_40_H_ALL", "R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE")
NOISES = (0, 1, 6, 7)
RECIPES = LABELS[2:]
CONTRASTS = {
    "MIX_at_H_ALL": (-1., 1., 0., 0.),
    "MIX_at_H_LATE": (0., 0., -1., 1.),
    "LATE_at_40": (-1., 0., 1., 0.),
    "LATE_at_MIX": (0., -1., 0., 1.),
    "MIX_main_effect": (-.5, .5, -.5, .5),
    "LATE_main_effect": (-.5, -.5, .5, .5),
    "interaction_difference_of_differences": (1., -1., -1., 1.),
}
# Directions come from the frozen metric definitions. They apply to that metric
# only and do not establish semantics, shape completeness or overall usefulness.
DIRECTIONS = {
    "CF_proper_affected_free_fraction": "lower", "CF_proper_pair_count": "lower",
    "CF_coplanar_pair_count": "lower", "CF_unresolved_or_degenerate_pair_count": "lower_uncertainty",
    "FF_illegal_affected_free_fraction": "lower", "FF_illegal_pair_count": "lower",
    "FF_proper_pair_count": "lower", "FF_coplanar_pair_count": "lower",
    "FF_unresolved_or_degenerate_pair_count": "lower_uncertainty",
    "ConflictCov_C": "lower", "ConflictCov_C_upper": "lower", "ConflictCov_C_cross": "lower",
    "ConflictCov_C_dup": "lower", "ConflictCov_C_uncertain_fraction": "lower_uncertainty",
    "free_minimum_angle_lt5_fraction": "lower", "free_angle_p05_degrees": "higher",
    "free_near_degenerate_fraction": "lower", "free_exact_zero_area_fraction": "lower",
    "free_exact_duplicate_excess_fraction": "lower", "free_q_gt10_fraction": "lower",
}
DISPLAY = (
    ("CF_proper_affected_free_fraction", "CF proper %", 100.),
    ("FF_illegal_affected_free_fraction", "FF illegal %", 100.),
    ("ConflictCov_C", "I lower %", 100.),
    ("ConflictCov_C_upper", "I upper %", 100.),
    ("free_minimum_angle_lt5_fraction", "Angle<5° %", 100.),
    ("free_near_degenerate_fraction", "Near-degenerate %", 100.),
)


def resolve(value):
    text = str(value).replace("\\", "/")
    if os.name != "nt" and len(text) > 2 and text[1] == ":":
        text = "/mnt/" + text[0].lower() + text[2:]
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = text[5].upper() + ":/" + text[7:]
    return Path(text)


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with resolve(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf8"); temp.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, keys); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*probability; lo = int(math.floor(x)); hi = int(math.ceil(x))
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def distribution(values, expected):
    valid = [float(v) for v in values if is_number(v)]
    complete = len(valid) == expected
    return dict(expected=expected, available=len(valid), missing=expected-len(valid),
                status="COMPLETE" if complete else "INCOMPLETE_METRIC_DENOMINATOR",
                mean=statistics.fmean(valid) if complete else None,
                available_mean=statistics.fmean(valid) if valid else None,
                median=statistics.median(valid) if complete else None,
                minimum=min(valid) if complete else None, maximum=max(valid) if complete else None,
                p05=quantile(valid, .05) if complete else None, p95=quantile(valid, .95) if complete else None,
                population_sd=statistics.pstdev(valid) if complete else None,
                negative_count=sum(v < 0 for v in valid), exact_zero_count=sum(v == 0 for v in valid),
                positive_count=sum(v > 0 for v in valid), sign_counts_are_descriptive_not_significance=True)


def validate_panel(registration, requests, rows, receipt):
    if registration.get("kind") != "main" or tuple(registration.get("labels", ())) != LABELS:
        raise ValueError("Analysis requires the six-label frozen main phase")
    if len(requests) != 64 or registration.get("expected_outputs") != 384 or len(rows) != 384:
        raise ValueError("Complete64 requests x6 models required")
    if receipt.get("status") != "COMPLETE" or receipt.get("registered") != 384 or receipt.get("scored") != 384:
        raise ValueError("Complete scoring receipt required; no partial-panel decision table")
    qby = {q["request_id"]:q for q in requests}
    if len(qby) != 64:
        raise ValueError("Duplicate registered request")
    parents = {q["parent_id"] for q in requests}; groups = defaultdict(list)
    for q in requests:
        groups[q["parent_id"], q["ratio"]].append(q)
    if len(parents) != 8 or set(groups) != {(p,r) for p in parents for r in (.2,.4)}:
        raise ValueError("Eight paired parents at both condition ratios required")
    for (_,ratio), group in groups.items():
        if len(group) != 4 or {q["candidate"] for q in group} != set(NOISES):
            raise ValueError("Incomplete registered n0/n1/n6/n7")
        for key in ("condition_id", "N", "K", "condition_cohort"):
            if len({q[key] for q in group}) != 1:
                raise ValueError("Condition changes across noise: " + key)
    cohorts = Counter(group[0]["condition_cohort"] for (_,r),group in groups.items() if r == .4)
    if cohorts != {"ORIGINAL6":6, "EXTENDED2":2}:
        raise ValueError("A40 original6/extended2 split differs")
    by = {}
    for row in rows:
        key = row["label"], row["request_id"]
        if key in by or key[0] not in LABELS or key[1] not in qby:
            raise ValueError("Duplicate/unregistered scored cell")
        q = qby[key[1]]
        for field in ("parent_id", "condition_id", "candidate", "ratio", "condition_cohort", "N", "K"):
            if row[field] != q[field]:
                raise ValueError("Scored request identity differs: " + field)
        if row.get("status") != "PASS" or not row.get("contracts") or not all(v is True for v in row["contracts"].values()):
            raise ValueError("Scoring/hard output contract is not complete")
        by[key] = row
    if set(by) != {(label,rid) for label in LABELS for rid in qby}:
        raise ValueError("Incomplete registered output cells")
    return by, groups


def metric_catalog(rows):
    names = sorted({k for r in rows for k in r["metrics"]})
    catalog = []
    for name in names:
        present = [r["metrics"][name] for r in rows if name in r["metrics"]]
        numeric = any(is_number(v) for v in present)
        status = Counter("NUMERIC" if is_number(v) else "NA_NULL" if v is None else json.dumps(v, sort_keys=True) for v in present)
        catalog.append(dict(metric=name, numeric_summary=numeric, registered_outputs=len(rows), present=len(present),
                            missing_field=len(rows)-len(present), value_status_counts=dict(status),
                            direction=DIRECTIONS.get(name,"descriptive_no_universal_direction"), source_table="primary_per_output", source_GT_used_for_promotion=False))
    return catalog


def strata(ratio):
    return ("ALL",) if ratio == .2 else ("ALL", "ORIGINAL6", "EXTENDED2")


def parent_means(rows, requests, metrics):
    by = {(r["label"],r["request_id"]):r for r in rows}; groups = defaultdict(list); result = []
    for q in requests:
        groups[q["parent_id"],q["ratio"]].append(q)
    for label in LABELS:
        for (pid,ratio), qs in sorted(groups.items()):
            rr = [by[label,q["request_id"]] for q in sorted(qs,key=lambda q:q["candidate"])]
            for metric in metrics:
                values = [r["metrics"].get(metric) for r in rr]
                result.append(dict(label=label,parent_id=pid,ratio=ratio,condition_cohort=qs[0]["condition_cohort"],metric=metric,
                                   noises=list(NOISES),noise_values=values,**distribution(values,4)))
    return result


def model_summary(parents):
    groups = defaultdict(list)
    for row in parents:
        for cohort in strata(row["ratio"]):
            if cohort == "ALL" or row["condition_cohort"] == cohort:
                groups[row["label"],row["ratio"],cohort,row["metric"]].append(row)
    return [dict(label=label,ratio=ratio,condition_cohort=cohort,metric=metric,
                 aggregation="four noise means then equal parent mean",**distribution([r["mean"] for r in group],len(group)))
            for (label,ratio,cohort,metric),group in sorted(groups.items())]


def paired_tables(rows, requests, parents, metrics):
    by = {(r["label"],r["request_id"]):r for r in rows}
    parent_by = {(r["label"],r["parent_id"],r["ratio"],r["metric"]):r for r in parents}
    conditions = {(q["parent_id"],q["ratio"]):q for q in requests}; noise_rows=[]; parent_rows=[]
    for first,second in itertools.combinations(LABELS,2):
        for q in requests:
            a,b = by[first,q["request_id"]],by[second,q["request_id"]]
            for metric in metrics:
                av,bv = a["metrics"].get(metric),b["metrics"].get(metric)
                noise_rows.append(dict(first=first,second=second,request_id=q["request_id"],parent_id=q["parent_id"],ratio=q["ratio"],
                                       condition_cohort=q["condition_cohort"],candidate=q["candidate"],metric=metric,
                                       first_value=av,second_value=bv,delta=av-bv if is_number(av) and is_number(bv) else None))
        for (pid,ratio),q in sorted(conditions.items()):
            for metric in metrics:
                a,b = parent_by[first,pid,ratio,metric],parent_by[second,pid,ratio,metric]
                deltas = [av-bv if is_number(av) and is_number(bv) else None for av,bv in zip(a["noise_values"],b["noise_values"])]
                parent_rows.append(dict(first=first,second=second,parent_id=pid,ratio=ratio,condition_cohort=q["condition_cohort"],
                                        metric=metric,first_mean=a["mean"],second_mean=b["mean"],noise_deltas=deltas,
                                        delta=a["mean"]-b["mean"] if is_number(a["mean"]) and is_number(b["mean"]) else None,
                                        **{"noise_"+k:v for k,v in distribution(deltas,4).items()}))
    grouped = defaultdict(list)
    for row in parent_rows:
        for cohort in strata(row["ratio"]):
            if cohort == "ALL" or row["condition_cohort"] == cohort:
                grouped[row["first"],row["second"],row["ratio"],cohort,row["metric"]].append(row)
    summary = [dict(first=a,second=b,ratio=ratio,condition_cohort=cohort,metric=metric,delta_direction="first minus second",
                    parent_ids=[r["parent_id"] for r in group],parent_deltas=[r["delta"] for r in group],
                    **distribution([r["delta"] for r in group],len(group)))
               for (a,b,ratio,cohort,metric),group in sorted(grouped.items())]
    return noise_rows,parent_rows,summary


def factorial_tables(parents, metrics):
    by = {(r["label"],r["parent_id"],r["ratio"],r["metric"]):r for r in parents}
    conditions = {(r["parent_id"],r["ratio"]):r for r in parents}; detail=[]
    for (pid,ratio),condition in sorted(conditions.items()):
        for metric in metrics:
            means = [by[label,pid,ratio,metric]["mean"] for label in RECIPES]
            for effect,weights in CONTRASTS.items():
                active = [(v,w) for v,w in zip(means,weights) if w]
                value = sum(v*w for v,w in active) if all(is_number(v) for v,w in active) else None
                detail.append(dict(parent_id=pid,ratio=ratio,condition_cohort=condition["condition_cohort"],metric=metric,
                                   effect=effect,value=value,recipe_order=list(RECIPES),weights=list(weights),recipe_parent_means=means))
    groups=defaultdict(list)
    for row in detail:
        for cohort in strata(row["ratio"]):
            if cohort == "ALL" or row["condition_cohort"] == cohort:
                groups[row["ratio"],cohort,row["metric"],row["effect"]].append(row)
    summary=[dict(ratio=ratio,condition_cohort=cohort,metric=metric,effect=effect,parent_ids=[r["parent_id"] for r in group],
                  parent_effects=[r["value"] for r in group],**distribution([r["value"] for r in group],len(group)))
             for (ratio,cohort,metric,effect),group in sorted(groups.items())]
    return detail,summary


def tradeoff_table(parent_pairs):
    """Metric-by-metric signed costs. No metric weighting, threshold or winner."""
    result=[]
    for candidate in RECIPES:
        for reference in ("START5000","HIST_FM6000","R1_40_H_ALL"):
            if candidate == reference:
                continue
            for row in parent_pairs:
                if {row["first"],row["second"]} != {candidate,reference} or row["metric"] not in DIRECTIONS:
                    continue
                delta=row["delta"]
                if is_number(delta) and row["first"] != candidate:
                    delta=-delta
                direction=DIRECTIONS[row["metric"]]
                movement="UNKNOWN" if delta is None else "LOWER" if delta<0 else "HIGHER" if delta>0 else "EXACT_EQUAL"
                result.append(dict(candidate=candidate,reference=reference,parent_id=row["parent_id"],ratio=row["ratio"],
                                   condition_cohort=row["condition_cohort"],metric=row["metric"],delta=delta,
                                   nominal_preferred_direction=direction,movement=movement,
                                   statistical_significance="NOT_TESTED",overall_model_judgment="NOT_INFERRED"))
    return result




def tradeoff_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        for cohort in strata(row["ratio"]):
            if cohort=="ALL" or row["condition_cohort"]==cohort:
                groups[row["candidate"],row["reference"],row["ratio"],cohort,row["metric"]].append(row)
    return [dict(candidate=candidate,reference=reference,ratio=ratio,condition_cohort=cohort,metric=metric,
                 nominal_preferred_direction=DIRECTIONS[metric],parent_ids=[r["parent_id"] for r in group],
                 parent_deltas=[r["delta"] for r in group],overall_model_judgment="NOT_INFERRED",
                 **distribution([r["delta"] for r in group],len(group)))
            for (candidate,reference,ratio,cohort,metric),group in sorted(groups.items())]


def validate_diversity(condition_rows, pair_rows, scored_rows):
    expected = {(r["label"],r["condition_id"]):(r["parent_id"],r["ratio"],r["condition_cohort"]) for r in scored_rows}
    seen_conditions = set()
    for row in condition_rows:
        key = row["label"],row["condition_id"]
        if key in seen_conditions or key not in expected:
            raise ValueError("Duplicate or unregistered diversity condition")
        parent,ratio,cohort = expected[key]
        if row["parent_id"] != parent or float(row["ratio"]) != ratio or row["condition_cohort"] != cohort:
            raise ValueError("Diversity condition metadata differs")
        if int(row["expected_pairs"]) != 6:
            raise ValueError("Diversity must retain all six pairs")
        seen_conditions.add(key)
    seen_pairs = defaultdict(set)
    for row in pair_rows:
        key = row["label"],row["condition_id"]
        pair = int(row["candidate_a"]),int(row["candidate_b"])
        if key not in expected or pair in seen_pairs[key]:
            raise ValueError("Duplicate or unregistered diversity pair")
        if row["parent_id"] != expected[key][0] or float(row["ratio"]) != expected[key][1]:
            raise ValueError("Diversity pair metadata differs")
        seen_pairs[key].add(pair)
    required = set(itertools.combinations(NOISES,2))
    if seen_conditions != set(expected) or set(seen_pairs) != set(expected) or any(pairs != required for pairs in seen_pairs.values()):
        raise ValueError("Incomplete registered diversity condition or noise pair coverage")


def diversity_summary(saved_rows):
    groups=defaultdict(list)
    for row in saved_rows:
        ratio=float(row["ratio"])
        value=float(row["D_free_mean"]) if row.get("D_free_mean") not in (None,"") else None
        if int(row["expected_pairs"])!=6 or int(row["available_pairs"])!=6:
            value=None
        for cohort in strata(ratio):
            if cohort=="ALL" or row["condition_cohort"]==cohort:
                groups[row["label"],ratio,cohort].append(value)
    return [dict(label=label,ratio=ratio,condition_cohort=cohort,metric="D_free_mean",
                 direction="descriptive_variation_not_larger_is_better",**distribution(values,len(values)))
            for (label,ratio,cohort),values in sorted(groups.items())]


def format_value(value,scale=1.):
    return "NA" if not is_number(value) else f"{value*scale:.4f}"


def markdown_report(summary,effects,tradeoffs,catalog,provenance,auxiliary_status,diversity,condition_context):
    families=condition_context["original_condition_construction_audit"]["fixed_pool"]["task_counts_by_family"]
    lines=["# 条件配方主矩阵：已有结果分析", "",
           "**解释限制：MIX与40比较的是完整历史条件池配方，不能解释成纯条件比例的因果效应。**", "",
           f"20%训练池含{families['MVP_LEGACY20']}个旧MVP任务和{families['DOMAIN_NEW20']}个Domain新任务；40%池的{families['DOMAIN_NEW40']}个任务全部来自Domain。共同BFS几何规则保持一致，但候选筛选策略及顺序锚点不同，因此MIX对比同时包含比例支持范围与条件区域选择分布的变化。", "",
           "R3−R1及R4−R2各自在相同输入流内保持严格配对；二者的差分只描述日程效应对这两个完整历史任务分布的依赖。全程与后期HYBRID的总曝光量不同，未隔离等曝光下的时机效应。使用同池的独立确认也不能消除条件筛选策略混杂。", "",
           "澄清原文及完整审计保存在[条件构造来源](condition_construction_context.json)，两份原始文件的路径与SHA256记录在[来源清单](provenance.json)。审计中的planned exposure只表示计划曝光，不能冒充已完成训练曝光。", "",
           "本报告只分析已保存的384份评价结果，不执行生成或重新评分。各比例独立汇总：先对同父4份噪声取均值，再对父对象等权。", "",
           "这是单份训练流、8个已观察DEV父对象上的开发证据；父均值差异与正负计数均为描述量，不代表独立训练重复或统计显著性。20%与40%是不同补全任务，不作纯增加C的因果解释。", "",
           "数值表不自动晋级模型；来源GT、S/E与来源距离不参与晋级。整体完整性与C使用仍需完整盲标图集审阅。", ""]
    lookup={(r["label"],r["ratio"],r["condition_cohort"],r["metric"]):r for r in summary}
    for ratio in (.2,.4):
        for cohort in strata(ratio):
            lines += [f"## C={int(ratio*100)}%，{cohort}","",
                      "| 模型 | "+" | ".join(label for _,label,_ in DISPLAY)+" |","|---|"+"---:|"*len(DISPLAY)]
            for label in LABELS:
                cells=[format_value(lookup.get((label,ratio,cohort,key),{}).get("mean"),scale) for key,_,scale in DISPLAY]
                lines.append("| "+label+" | "+" | ".join(cells)+" |")
            lines.append("")
    lines += ["## 完整条件池配方与HYBRID策略的对比", "",
              "MIX主效应=[(R2−R1)+(R4−R3)]/2；后期HYBRID主效应=[(R3−R1)+(R4−R2)]/2；交互=(R4−R3)−(R2−R1)。先逐父计算，再等父汇总。全程与后期HYBRID的曝光总量不同，因此这些是策略效应。", "",
              "上述MIX及交互均以完整历史条件池配方为因素；数值公式保持登记定义，不构成纯比例主效应或纯比例交互。", "",
              "下表均为原指标单位的差值；比例指标乘100显示为百分点。负值仅在该指标定义为越低越好时表示测得成本下降。", "",
              "| C比例 | 指标 | MIX主效应 | 后期H主效应 | 交互 |", "|---|---|---:|---:|---:|"]
    eb={(r["ratio"],r["condition_cohort"],r["metric"],r["effect"]):r for r in effects}
    for ratio in (.2,.4):
        for metric,label,scale in DISPLAY:
            cells=[format_value(eb.get((ratio,"ALL",metric,effect),{}).get("mean"),scale) for effect in
                   ("MIX_main_effect","LATE_main_effect","interaction_difference_of_differences")]
            lines.append(f"| {int(ratio*100)}% | {label} | "+" | ".join(cells)+" |")
    lines += ["", "## 分布与取舍文件", "",
              "- `all_metrics.json`保留所有原始metric、状态和合同字段；`metric_catalog.csv`记录非数值、NA与缺字段。",
              "- `model_summary.csv`和`per_parent_metrics.csv`包含全部数值指标及分母。缺少任一噪声时，不用剩余噪声替代完整父均值。",
              "- `paired_noise_metrics.csv`、`paired_parent_metrics.csv`、`paired_parent_summary.csv`保存全部模型对的噪声与父分布、尾部及精确正负计数。",
              "- `factorial_per_parent.csv`和`factorial_summary.csv`保存全部指标的简单效应、主效应与交互；40%另列原6/扩2。",
              "- `tradeoffs_per_parent.csv`按各指标方向列出候选相对START、历史FM和R1的变化，不把它们合并成总分。",
              "- `diversity_summary.csv`只描述变化，不能据此认定更大更好。", "",
              "来源诊断保存状态：`"+auxiliary_status+"`。该文件独立保存，不用于上面的方向判断或任何自动晋级。", "",
              f"完整metric字段数：{len(catalog)}；分析来源身份：`{canonical_hash(provenance)}`。所有输入与新分析脚本的SHA256见`provenance.json`。", ""]
    return "\n".join(lines)


def analyze(out=DEFAULT_OUT):
    out=resolve(out).resolve()
    if out.name!=ROOT_NAME or out.parent.name!="experiments":
        raise ValueError("Wrong experiment analysis root")
    phase=out/"phases/main"; ev=phase/"evaluation"
    files={"registration":phase/"registration.json","per_output":ev/"per_output.json","scoring_receipt":ev/"scoring_receipt.json",
           "diversity_pairs":ev/"diversity_pairs.csv","diversity_conditions":ev/"diversity_per_condition.csv",
           "protocol_clarifications":out/"PROTOCOL_CLARIFICATIONS.md",
           "condition_construction_audit":out/"audit/condition_construction_audit.json"}
    registration=read_json(files["registration"]); files["requests"]=resolve(registration["requests_path"])
    if digest(files["requests"])!=registration["requests_sha256"]:
        raise ValueError("Frozen requests changed")
    if (ev/"reference_A20_per_output.json").exists():
        files["auxiliary_reference_A20"]=ev/"reference_A20_per_output.json"
    if (out/"registrations/source_hashes.json").exists():
        files["frozen_execution_sources"]=out/"registrations/source_hashes.json"
    provenance=dict(version=VERSION, input_files={name:dict(path=str(path),sha256=digest(path)) for name,path in files.items()},
                    analysis_source_path=str(Path(__file__).resolve()),analysis_source_sha256=digest(__file__),
                    operations="Read existing scalar tables only; no model/geometry/feature calls", new_model_calls=0,new_scoring_calls=0)
    identity=canonical_hash(provenance); dest=out/"analysis/main"/(VERSION+"_"+identity[:16])
    receipt_path=dest/"analysis_receipt.json"
    if receipt_path.exists():
        previous=read_json(receipt_path)
        if previous["input_identity"]!=identity:
            raise ValueError("Existing analysis identity differs")
        for relative,h in previous["output_sha256"].items():
            if digest(dest/relative)!=h:
                raise ValueError("Immutable analysis artifact changed")
        return previous
    requests=read_json(files["requests"])["rows"]; rows=read_json(files["per_output"])["rows"]
    validate_panel(registration,requests,rows,read_json(files["scoring_receipt"]))
    catalog=metric_catalog(rows); metrics=[r["metric"] for r in catalog if r["numeric_summary"]]
    parents=parent_means(rows,requests,metrics); summary=model_summary(parents)
    paired_noise,paired_parents,paired_summary=paired_tables(rows,requests,parents,metrics)
    fact_detail,fact_summary=factorial_tables(parents,metrics); tradeoffs=tradeoff_table(paired_parents)
    trade_summary=tradeoff_summary(tradeoffs)
    div_conditions=read_csv(files["diversity_conditions"]); div_pairs=read_csv(files["diversity_pairs"])
    validate_diversity(div_conditions,div_pairs,rows)
    div_summary=diversity_summary(div_conditions)
    auxiliary=read_json(files["auxiliary_reference_A20"]) if "auxiliary_reference_A20" in files else dict(status="NOT_AVAILABLE",rows=[],promotion_gate=False)
    auxiliary_status="SAVED_SEPARATELY_NOT_A_PROMOTION_GATE" if "auxiliary_reference_A20" in files else "NOT_AVAILABLE"
    condition_context=dict(protocol_clarification_text=files["protocol_clarifications"].read_text(encoding="utf-8-sig"),
                           original_condition_construction_audit=read_json(files["condition_construction_audit"]),
                           interpretation=dict(MIX_factor="complete historical condition-pool recipe/distribution",
                                               pure_ratio_causal_effect_identified=False,
                                               ratio_only_interaction_identified=False,
                                               equal_exposure_timing_effect_identified=False,
                                               paired_schedule_contrasts=["R3 minus R1","R4 minus R2"],
                                               planned_exposure_is_completed_exposure=False,
                                               same_pool_confirmation_removes_policy_confound=False))
    # Detect a file being rewritten while reading; no stale mixed snapshot.
    for item in provenance["input_files"].values():
        if digest(item["path"])!=item["sha256"]:
            raise ValueError("Analysis input changed during read")
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"provenance.json",provenance)
    write_json(dest/"condition_construction_context.json",condition_context)
    write_json(dest/"all_metrics.json",dict(rows=rows,source_per_output_sha256=provenance["input_files"]["per_output"]["sha256"]))
    write_json(dest/"reference_A20_diagnostics.json",dict(original_saved_table=auxiliary,promotion_gate=False))
    tables={"metric_catalog":catalog,"per_parent_metrics":parents,"model_summary":summary,
            "paired_noise_metrics":paired_noise,"paired_parent_metrics":paired_parents,"paired_parent_summary":paired_summary,
            "factorial_per_parent":fact_detail,"factorial_summary":fact_summary,"tradeoffs_per_parent":tradeoffs,"tradeoffs_summary":trade_summary,
            "diversity_per_condition":div_conditions,"diversity_pairs":div_pairs,"diversity_summary":div_summary}
    for name,table in tables.items():
        write_csv(dest/(name+".csv"),table)
    write_json(dest/"decision_evidence.json",dict(version=VERSION,metric_catalog=catalog,model_summary=summary,
               paired_parent_distributions=paired_summary,factorial_effects=fact_summary,tradeoffs=trade_summary,diversity=div_summary,
               source_GT_used_for_promotion=False,automatic_promotion=False,overall_score=None,
               condition_pool_interpretation=condition_context["interpretation"],
               visual_review="REQUIRES_ACTUAL_FULL_GALLERY_INSPECTION",independent_training_repeat="NOT_ESTABLISHED_BY_THIS_TABLE"))
    (dest/"REPORT.md").write_text(markdown_report(summary,fact_summary,tradeoffs,catalog,provenance,auxiliary_status,div_summary,condition_context),encoding="utf8")
    result=dict(status="COMPLETE",version=VERSION,input_identity=identity,analysis_dir=str(dest),registered_outputs=384,
                metrics_preserved=len(catalog),numeric_metrics=len(metrics),new_model_calls=0,new_scoring_calls=0,
                automatic_promotion=False,overall_score=None,source_GT_promotion=False,
                output_sha256={str(path.relative_to(dest)):digest(path) for path in sorted(dest.iterdir()) if path.is_file()})
    write_json(receipt_path,result)
    write_json(out/"analysis/main/latest.json",dict(analysis_dir=str(dest),receipt_path=str(receipt_path),receipt_sha256=digest(receipt_path),input_identity=identity))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default=DEFAULT_OUT)
    args=parser.parse_args(argv);result=analyze(args.out);print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
