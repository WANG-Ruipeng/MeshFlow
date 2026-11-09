"""Read-only analysis of already collected scores; never score or load a model.

Default / --check-ready performs a metadata readiness check without writing.
--final is an explicit, separate action and refuses missing/inconsistent inputs.
All new artifacts go to results/analysis_v1; original scores/tables stay intact.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_full_seat_curriculum_v1"
VERSION = "bench_full_seat_readonly_parent_analysis_v1"
LABELS = ("COMMON_S12000", "NATIVE_S6000", "SEAM_S3000")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
GROUPS = ("CURRENT_TRAIN", "DEV")
TABLES = ("per_output", "per_condition", "per_parent", "parent_macro_summary",
          "paired_per_output", "paired_per_condition", "paired_per_parent")
SEAM_FIELDS = (
    "retained_seam_faces", "retained_bulk_faces", "retained_seam_area",
    "retained_bulk_area", "legal_CF_seed_seam_faces", "legal_CF_seed_seam_area",
    "free_only_reached_seam_faces", "free_only_reached_bulk_faces",
    "free_only_reached_bulk_area", "free_only_reached_bulk_area_fraction",
    "legal_seam_bulk_edge_count", "legal_seam_bulk_edge_length",
    "seed_connected_legal_seam_bulk_edge_count", "seed_connected_legal_seam_bulk_edge_length",
    "bulk_reach_ge_0_30_descriptive", "deleted_seam_faces",
)
PRIMARY = (
    "CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction", "ConflictCov_C",
    "ConflictCov_C_cross", "ConflictCov_C_dup", "ConflictCov_C_cross_and_dup",
    "ConflictCov_C_interior_samples", "ConflictCov_C_interior_weight_fraction",
    "ConflictCov_C_uncertain_fraction", "ConflictCov_C_upper",
    "CF_illegal_pair_count", "FF_illegal_pair_count",
    "CF_unresolved_or_degenerate_pair_count", "FF_unresolved_or_degenerate_pair_count",
    "C_F_valid_edge_count", "C_F_valid_edge_total_length", "C_F_has_valid_edge",
    "C_F_valid_opposite_winding_edge_count", "C_F_has_opposite_winding_edge",
    "free_only_reached_bulk_area_fraction", "free_only_reached_bulk_area",
    "retained_bulk_area", "bulk_reach_ge_0_30_descriptive",
    "legal_seam_bulk_edge_length", "seed_connected_legal_seam_bulk_edge_length",
    "free_minimum_angle_lt5_fraction", "free_angle_p05_degrees", "free_q_gt10_fraction",
    "free_near_degenerate_fraction", "free_exact_zero_area_fraction", "free_total_surface_area",
    "actual_free_faces", "deleted_free_faces", "new_bridge_faces", "bridge_area",
    "original_POST_free_faces", "deleted_original_POST_faces",
    "free_face_denominator", "fixed_parent_free_face_denominator",
    "fixed_parent_CF_illegal_affected_fraction", "fixed_parent_FF_illegal_affected_fraction",
    "fixed_parent_minimum_angle_lt5_fraction", "hinge_diag_isolated_1face_edge_components",
    "hinge_diag_isolated_1face_unfiltered_edge_components", "nonmanifold_edges",
    "C_F_nonmanifold_edge_count", "hinge_diag_fully_C_disconnected_free_area_fraction",
    "hinge_diag_vertex_attached_only_free_area_fraction", "C_query_weight_denominator",
    "CF_tested_pair_denominator", "FF_tested_pair_denominator",
)
# This is a presentation filter, not a change to any detector or saved score.
PASS_ONLY_I = {"ConflictCov_C", "ConflictCov_C_cross", "ConflictCov_C_dup",
               "ConflictCov_C_cross_and_dup"}


def resolve(path):
    s = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value):
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        n = float(value)
    except (ValueError, TypeError):
        return None
    return n if math.isfinite(n) else None


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def readiness(out=OUT):
    """No hashing of large assets, imports of evaluation, or output writes."""
    paths = [out / "eval_manifest.json", out / "registration.json",
             out / "evaluation/score_registration.json", out / "results/completion.json"]
    paths += [out / "results" / (name + ".csv") for name in TABLES]
    done_path = out / "results/completion.json"
    if done_path.is_file():
        done = read(done_path)
        paths += [resolve(p) for group in ("files", "source_rows") for p in done.get(group, {})]
    missing = [str(p) for p in paths if not p.is_file()]
    missing = sorted(set(missing))
    result = dict(schema=VERSION, status="NOT_READY" if missing else "READY_FOR_VALIDATION",
                  expected_score_rows=216, missing_files=missing, output_written=False,
                  model_calls=0, scoring_calls=0)
    if (out / "results/completion.json").is_file():
        done = read(out / "results/completion.json")
        result["collector_status"] = done.get("status")
        if done.get("status") != "COMPLETE":
            result.update(status="NOT_READY", reason="Collector has not completed")
    if missing:
        result["reason"] = "Registered collection inputs are missing; no outcome inferred"
    return result


def score_folder(out, label, rid, stage):
    if stage == "RAW":
        return out / "evaluation/scores" / label / rid / stage
    return out / "cpu" / label / rid / stage / "scores"


def load_complete(out=OUT):
    ready = readiness(out)
    if ready["status"] != "READY_FOR_VALIDATION":
        return ready, None
    manifest = read(out / "eval_manifest.json")
    reg = read(out / "registration.json")
    score_reg = read(out / "evaluation/score_registration.json")
    done = read(out / "results/completion.json")
    if (done.get("rows"), done.get("new_score_rows"), done.get("reused_score_rows")) != (216, 216, 0):
        raise ValueError("Collector row/reuse counts differ from the registered panel")
    if (done.get("model_calls"), done.get("scoring_calls")) != (0, 0):
        raise ValueError("Unexpected collector model/scoring calls")
    if manifest.get("status") != "FROZEN" or tuple(manifest["labels"]) != LABELS:
        raise ValueError("Manifest labels/status differ")
    if tuple(manifest["stages"]) != STAGES or set(reg["evaluation_labels"]) != set(LABELS):
        raise ValueError("Registered labels/stages differ")
    if (score_reg.get("eval_manifest_sha256") != digest(out / "eval_manifest.json")
            or score_reg.get("registration_sha256") != digest(out / "registration.json")):
        raise ValueError("Score registration does not bind these inputs")
    if done.get("collector_sha256") != digest(Path(__file__).with_name("collect.py")):
        raise ValueError("Collector source identity changed")
    sources = {resolve(p): h for p, h in done["source_rows"].items()}
    tables = {resolve(p): h for p, h in done["files"].items()}
    required_tables = {out / "results" / (name + ".csv") for name in TABLES}
    if set(tables) != required_tables:
        raise ValueError("Collector receipt must contain exactly the seven original tables")
    if len(sources) != 432 or any(p.name not in ("score.json", "row.json") for p in sources):
        raise ValueError("Collector source inputs must be exactly 432 score/row JSON files")
    for p, h in {**tables, **sources}.items():
        if digest(p) != h:
            raise ValueError("Collected evidence hash mismatch: " + str(p))
    with (out / "results/per_output.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    qs = manifest["requests"]
    requests = {q["request_id"]: q for q in qs}
    if len(requests) != 24 or len(qs) != 24 or len(rows) != 216:
        raise ValueError("Expected 24 unique requests and 216 scored outputs")
    if collections.Counter(q["eval_group"] for q in qs) != {"CURRENT_TRAIN": 16, "DEV": 8}:
        raise ValueError("Unexpected TRAIN/DEV request counts")
    if any(len({q["parent_id"] for q in qs if q["eval_group"] == g}) != n
           for g, n in (("CURRENT_TRAIN", 8), ("DEV", 4))):
        raise ValueError("Unexpected parent counts")
    expected = {(label, stage, rid) for label in LABELS for stage in STAGES for rid in requests}
    keys = [(r["label"], r["stage"], r["request_id"]) for r in rows]
    if len(set(keys)) != 216 or set(keys) != expected:
        raise ValueError("Missing/duplicate/unregistered score cells")
    numeric_fields = sorted(set(done["metric_fields"]) | set(PRIMARY))
    if not set(numeric_fields).issubset(fields):
        raise ValueError("Expected original metric columns absent: " + str(set(numeric_fields) - set(fields)))
    source_evidence = []
    expected_sources = set()
    for row in rows:
        q = requests[row["request_id"]]
        for key in ("condition_id", "parent_id", "eval_group", "split", "part_type"):
            if row[key] != q[key]:
                raise ValueError("Panel identity differs: " + key)
        for key in ("noise_index", "N", "K"):
            if number(row[key]) != q[key]:
                raise ValueError("Panel count differs: " + key)
        folder = score_folder(out, row["label"], row["request_id"], row["stage"])
        sp, rp = folder / "score.json", folder / "row.json"
        expected_sources.update((sp, rp))
        if sp not in sources or rp not in sources:
            raise ValueError("Missing authenticated original score/row")
        score, original = read(sp), read(rp)
        # Check every original cell, including status strings and denominators.
        for key, value in original.items():
            if key in ("label", "stage"):
                continue
            same = (json.loads(row[key]) == value if isinstance(value, (dict, list))
                    else row.get(key) == csv_cell(value))
            if not same:
                raise ValueError("Original field changed in collection: " + key)
        if score["status"] != "PASS" or score["status_by_metric"]["I"] != row["ConflictCov_C_status"]:
            raise ValueError("Score/I status mismatch")
        seam_status = score["seam_to_bulk"]["status"]
        if row["seam_applicability"] != seam_status:
            raise ValueError("SEAM status changed in collection")
        if row["label"] != LABELS[2]:
            if seam_status != "NA_NOT_SEAM_REPRESENTATION":
                raise ValueError("Native SEAM applicability must be NA")
        elif q["closed_C"]:
            if seam_status != "NA_CLOSED_C_NO_SEAM_INTERFACE":
                raise ValueError("Closed-C SEAM applicability must be NA")
        elif seam_status.startswith("NA") and seam_status != "NA_EMPTY_OR_ZERO_AREA_BULK":
            raise ValueError("Open all-null SEAM layouts must remain applicable")
        if seam_status == "NA_EMPTY_OR_ZERO_AREA_BULK" and (
                number(row.get("retained_bulk_area")) != 0
                or number(row.get("free_only_reached_bulk_area_fraction")) is not None
                or number(row.get("bulk_reach_ge_0_30_descriptive")) is not None):
            raise ValueError("Zero-area bulk NA must preserve its zero denominator and missing ratio")
        if seam_status in ("NA_NOT_SEAM_REPRESENTATION", "NA_CLOSED_C_NO_SEAM_INTERFACE") and any(number(row.get(m)) is not None for m in SEAM_FIELDS):
            raise ValueError("NA SEAM field contains a numerical score")
        evidence = {key: row[key] for key in ("label", "stage", "eval_group", "request_id", "condition_id", "parent_id")}
        evidence.update(source_row=str(rp), source_row_sha256=sources[rp], source_score=str(sp),
                        source_score_sha256=sources[sp], scorer_version=score.get("version"),
                        status_by_metric=score.get("status_by_metric"), seam_status=seam_status,
                        C_query_count=score.get("C_query_count"),
                        C_query_weight_denominator=score.get("C_query_weight_denominator"),
                        free_face_denominator=score.get("free_face_denominator"),
                        CF_tested_pair_denominator=score.get("CF_tested_pair_denominator"),
                        FF_tested_pair_denominator=score.get("FF_tested_pair_denominator"),
                        deletion_semantics=score.get("deletion_semantics"),
                        bridge_contract=score.get("bridge_contract"))
        source_evidence.append(evidence)
    if set(sources) != expected_sources:
        raise ValueError("Collector source set is not exactly 216 original scores and rows")
    return dict(status="VALIDATED", rows=216), dict(rows=rows, fields=fields, metrics=numeric_fields,
                completion=done, source_evidence=source_evidence, manifest=manifest)


def parent_values(items):
    """(row, value): mean over noise, then condition, then equally weighted parent."""
    conditions = collections.defaultdict(list)
    for row, value in items:
        conditions[row["parent_id"], row["condition_id"]].append(value)
    parents = collections.defaultdict(list)
    for (parent, _condition), values in conditions.items():
        parents[parent].append(statistics.fmean(values))
    return {p: statistics.fmean(values) for p, values in sorted(parents.items())}


def eligible(row, metric, policy):
    if policy == "I_PASS_ONLY" and row.get("ConflictCov_C_status") != "PASS":
        return None, "I_STATUS_" + row.get("ConflictCov_C_status", "MISSING")
    value = number(row.get(metric))
    if value is not None:
        return value, None
    if metric in SEAM_FIELDS:
        return None, row.get("seam_applicability", "SEAM_STATUS_MISSING")
    if metric.startswith("bridge_"):
        return None, row.get("bridge_shape_status", "NO_FINITE_VALUE")
    return None, "MISSING_OR_NONFINITE"


def support(all_rows, selected, reasons):
    counts = collections.Counter(r["parent_id"] for r in selected)
    return dict(expected_requests=len(all_rows), valid_requests=len(selected),
                expected_conditions=len({(r["parent_id"], r["condition_id"]) for r in all_rows}),
                valid_conditions=len({(r["parent_id"], r["condition_id"]) for r in selected}),
                expected_parents=len({r["parent_id"] for r in all_rows}), valid_parents=len(counts),
                valid_request_ids=sorted(r["request_id"] for r in selected),
                valid_requests_by_parent=dict(sorted(counts.items())), excluded_reasons=dict(sorted(reasons.items())),
                I_status_counts=dict(sorted(collections.Counter(r.get("ConflictCov_C_status", "MISSING") for r in all_rows).items())),
                seam_status_counts=dict(sorted(collections.Counter(r.get("seam_applicability", "MISSING") for r in all_rows).items())))


def aggregate(rows, metric, policy):
    accepted, reasons = [], collections.Counter()
    for row in rows:
        value, reason = eligible(row, metric, policy)
        if reason:
            reasons[reason] += 1
        else:
            accepted.append((row, value))
    parents = parent_values(accepted)
    values = list(parents.values())
    return dict(metric=metric, eligibility=policy, mean=statistics.fmean(values) if values else None,
                median_parent=statistics.median(values) if values else None,
                min_parent=min(values) if values else None, max_parent=max(values) if values else None,
                parent_values=parents, **support(rows, [r for r, _ in accepted], reasons))


def paired(reference, candidate, metric, policy):
    by = {r["request_id"]: r for r in candidate}
    if set(by) != {r["request_id"] for r in reference} or len(by) != len(candidate):
        raise ValueError("Paired request universe differs")
    av, bv, dv, selected = [], [], [], []
    reasons = collections.Counter()
    for a in reference:
        b = by[a["request_id"]]
        if any(a[k] != b[k] for k in ("condition_id", "parent_id", "eval_group", "noise_index")):
            raise ValueError("Pair identity mismatch")
        x, xr = eligible(a, metric, policy)
        y, yr = eligible(b, metric, policy)
        if xr or yr:
            reasons["reference=" + (xr or "VALID") + ";candidate=" + (yr or "VALID")] += 1
            continue
        selected.append(a)
        av.append((a, x)); bv.append((a, y)); dv.append((a, y - x))
    aa, bb, dd = parent_values(av), parent_values(bv), parent_values(dv)
    ds = list(dd.values())
    result = dict(metric=metric, eligibility=policy, delta_definition="candidate_minus_reference",
                  reference_mean_on_common_support=statistics.fmean(aa.values()) if aa else None,
                  candidate_mean_on_common_support=statistics.fmean(bb.values()) if bb else None,
                  mean_delta=statistics.fmean(ds) if ds else None,
                  median_parent_delta=statistics.median(ds) if ds else None,
                  min_parent_delta=min(ds) if ds else None, max_parent_delta=max(ds) if ds else None,
                  parent_decrease=sum(d < 0 for d in ds), parent_equal=sum(d == 0 for d in ds),
                  parent_increase=sum(d > 0 for d in ds), parent_deltas=dd,
                  reference_parent_values=aa, candidate_parent_values=bb,
                  **support(reference, selected, reasons))
    result["candidate_I_status_counts"] = dict(sorted(collections.Counter(r.get("ConflictCov_C_status", "MISSING") for r in candidate).items()))
    result["candidate_seam_status_counts"] = dict(sorted(collections.Counter(r.get("seam_applicability", "MISSING") for r in candidate).items()))
    return result


def analyze(data):
    rows = data["rows"]
    buckets = {(label, stage, group): [r for r in rows if (r["label"], r["stage"], r["eval_group"]) == (label, stage, group)]
               for label in LABELS for stage in STAGES for group in GROUPS}
    original, primary = [], []
    for (label, stage, group), subset in buckets.items():
        meta = dict(label=label, stage=stage, eval_group=group)
        original.extend(dict(**meta, **aggregate(subset, metric, "ORIGINAL_FINITE_ONLY")) for metric in data["metrics"])
        primary.extend(dict(**meta, **aggregate(subset, metric, "I_PASS_ONLY" if metric in PASS_ONLY_I else "ORIGINAL_FINITE_ONLY")) for metric in PRIMARY)
    contrasts = []
    for stage in STAGES:
        for group in GROUPS:
            for a, b in ((LABELS[0], LABELS[1]), (LABELS[0], LABELS[2]), (LABELS[1], LABELS[2])):
                contrasts.append((dict(contrast_type="MODEL", reference=a, candidate=b, reference_stage=stage, candidate_stage=stage, eval_group=group),
                                  buckets[a, stage, group], buckets[b, stage, group]))
    for label in LABELS:
        for group in GROUPS:
            for a, b in ((STAGES[0], STAGES[1]), (STAGES[1], STAGES[2]), (STAGES[0], STAGES[2])):
                contrasts.append((dict(contrast_type="PROCESS_STAGE", reference=label, candidate=label, reference_stage=a, candidate_stage=b, eval_group=group),
                                  buckets[label, a, group], buckets[label, b, group]))
    paired_all, paired_primary = [], []
    for meta, aa, bb in contrasts:
        paired_all.extend(dict(**meta, **paired(aa, bb, m, "ORIGINAL_FINITE_ONLY")) for m in data["metrics"])
        paired_primary.extend(dict(**meta, **paired(aa, bb, m, "I_PASS_ONLY" if m in PASS_ONLY_I else "ORIGINAL_FINITE_ONLY")) for m in PRIMARY)
    # Face retention uses the scorer's original RAW-parent deletion definition.
    # BRIDGE faces contribute to actual_free_faces, not retained original faces.
    retention = []
    for r in rows:
        wanted = number(r["N"]) - number(r["K"])
        deleted, actual, bridge = (number(r[k]) for k in ("deleted_free_faces", "actual_free_faces", "new_bridge_faces"))
        retained = wanted - deleted if deleted is not None else None
        if retained is not None and actual is not None and bridge is not None and actual != retained + bridge:
            raise ValueError("Original retained + BRIDGE face accounting differs")
        retention.append(dict(**{k: r[k] for k in ("label", "stage", "eval_group", "request_id", "condition_id", "parent_id")},
                              requested_original_free_faces=wanted, retained_original_RAW_free_faces=retained,
                              deleted_original_RAW_free_faces=deleted, new_bridge_faces=bridge, actual_free_faces=actual,
                              original_POST_free_faces=number(r.get("original_POST_free_faces")),
                              deleted_original_POST_faces=number(r.get("deleted_original_POST_faces"))))
    return dict(primary_parent_macro=primary, primary_paired_deltas=paired_primary,
                all_numeric_parent_macro=original, all_numeric_paired_deltas=paired_all,
                per_output_denominators_and_status=data["source_evidence"], per_output_face_retention=retention)


NOTES = """# 只读结果分析口径 v1

此工具只读取本轮已完成 collect 的 216 行原评分，不构造网格、不调用模型、
不运行检测器、不增加固定查询，也不填写任何视觉评价。当前训练完成与否、
controller 是否 READY_REVIEW、训练调用账本和逐图真实审阅需要分别核验；
本工具的 COMPLETE 仅表示数值分析产物完成。

面板为 CURRENT_TRAIN 8 父 / 8 C / 16 请求、DEV 4 父 / 4 C / 8 请求。
对每个模型、处理阶段和分组，先把同 C 的有效噪声数值取均值，再把同父
的 C 均值取均值，最后父等权。全部实际分母、有效请求 ID 和逐父数值保留。
两份噪声不等于两个独立父；DEV 是重复使用的开发证据，不是独立测试。

primary 表的 I/交叉/重叠/交集仅收 ConflictCov_C_status=PASS。
每条配对增量先限制双方共同有效（I 必须双方 PASS）的相同请求，再按
同样层级聚合候选减参考。参考/候选的共同支持均值与各自边际均值可能不同，
不能相减不同支持集的边际均值来替代配对。I 支持量、未决量、上界和状态
计数另列；非 PASS 不是零。all_numeric 表原样聚合所有有限原值，包括
非 PASS I，仅用于完整证据保留，不应当作已认证 I 主结论。

COMMON/NATIVE 没有 SEAM 表示，专属 bulk reach 等字段为
NA_NOT_SEAM_REPRESENTATION，不能按零比较，更不能宣称 SEAM 相对此 NA
获得了数值增益。闭合 C 相应字段为 NA_CLOSED_C_NO_SEAM_INTERFACE。
开放 C 上的全 Null 布局仍适用，真实零可达不能改成 NA。合法 CF 边长
属于实际有效图，Native 仍适用；它、bulk reach>=0.30 都是几何描述量，
不是自动的主体完整、接合需求满足或语义成功判定。
原 scorer 若记 NA_EMPTY_OR_ZERO_AREA_BULK，则是保留 bulk 面积为空/零的
分母边界，和全 Null 布局不同；依原值保留，缺失可达比例不填零。

CF/FF 非法受影响自由面比例分母为当阶段实际 free 面数；固定父版本的
分母是该处理链原始父面数（RAW 的基准为自身，BASE/FINAL 为 BASE_POST），
二者不能混用。I 是固定 C 查询面积权重覆盖，
不是全物体自交率。面数/小角比例同时保留实际 free 分母、测试面对分母、
固定父分母；面积/长度保持原模型单位，未重标定百分数或尺度。
deleted_free_faces 按原始 RAW 父 ID 缺失计数，不能用 N 减输出总面数代替。
retained_original_RAW_free_faces=N-K-deleted_free_faces；桥接新面另列，
actual_free_faces=retained_original_RAW_free_faces+new_bridge_faces。
原 POST 父删面、桥接面积、小角、孤岛与非流形均保留，不能用删面或新增
桥面掩盖分母变化。所有原数值、状态和文字字段完整复制在 source_per_output.csv。

这是 COMMON 历史起点上同池 Native 6000→SEAM 3000 的整体课程比较。
任务池、训练量、布局表示和后处理阶段各有边界，不将结果单归因于数据量、
完整座面、AdamW、表示或某项几何操作；不与 Chair 作类别单变量因果比较。
无来源唯一正确形状的假设，也无自动视觉成功标签。
"""


def csv_bytes(rows):
    buf = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: csv_cell(v) for k, v in r.items()})
    return buf.getvalue().encode("utf-8")


def immutable_bytes(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Refuse to replace existing analysis evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)


def final(out=OUT):
    state, data = load_complete(out)
    if data is None:
        return state
    tables = analyze(data)
    dest = out / "results/analysis_v1"
    artifacts = {dest / (name + ".csv"): csv_bytes(rows) for name, rows in tables.items()}
    artifacts[dest / "source_per_output.csv"] = (out / "results/per_output.csv").read_bytes()
    artifacts[dest / "METHODS.md"] = NOTES.encode("utf-8")
    inputs = {str(p): digest(p) for p in (out / "results/completion.json", out / "eval_manifest.json",
              out / "registration.json", out / "evaluation/score_registration.json", Path(__file__))}
    result = dict(schema=VERSION, status="COMPLETE", meaning="NUMERICAL_ANALYSIS_ONLY",
                  score_rows=216, models=list(LABELS), stages=list(STAGES), eval_groups=list(GROUPS),
                  metric_fields=data["metrics"], original_csv_fields=data["fields"], inputs=inputs,
                  source_tables=data["completion"]["files"], source_rows=data["completion"]["source_rows"],
                  analysis_rows={name: len(rows) for name, rows in tables.items()},
                  I_primary_policy="PASS_ONLY; paired comparisons use common PASS requests",
                  aggregation="noise mean -> condition mean within parent -> equal parent mean",
                  delta="candidate minus reference on exactly common valid requests",
                  visual_review="NOT_PERFORMED_BY_THIS_HELPER", model_calls=0, scoring_calls=0,
                  no_new_queries=True, files={str(p): hashlib.sha256(b).hexdigest() for p, b in artifacts.items()})
    for path, payload in artifacts.items():
        immutable_bytes(path, payload)
    immutable_bytes(dest / "completion.json", (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return {k: v for k, v in result.items() if k not in ("source_tables", "source_rows", "files", "metric_fields", "original_csv_fields")}


def self_test():
    def row(parent, condition, request, value, status="PASS"):
        return dict(parent_id=parent, condition_id=condition, request_id=request, noise_index=request,
                    eval_group="DEV", ConflictCov_C=value, ConflictCov_C_status=status,
                    seam_applicability="PASS")
    # Unequal C/noise counts distinguish the three levels from pooled means.
    aa = [row("a", "a1", "a10", 0), row("a", "a1", "a11", 2),
          row("a", "a2", "a20", 5), row("b", "b1", "b10", 9)]
    result = aggregate(aa, "ConflictCov_C", "I_PASS_ONLY")
    assert result["mean"] == 6 and result["parent_values"] == {"a": 3, "b": 9}
    bb = [dict(r, ConflictCov_C=r["ConflictCov_C"] + 1) for r in aa]
    bb[1]["ConflictCov_C_status"] = "LOW_SUPPORT"
    pair = paired(aa, bb, "ConflictCov_C", "I_PASS_ONLY")
    assert pair["mean_delta"] == 1 and pair["valid_requests"] == 3
    assert pair["reference_mean_on_common_support"] == 5.75
    assert pair["candidate_mean_on_common_support"] == 6.75
    assert aggregate(bb, "ConflictCov_C", "ORIGINAL_FINITE_ONLY")["valid_requests"] == 4
    native = dict(aa[0], seam_applicability="NA_NOT_SEAM_REPRESENTATION", free_only_reached_bulk_area_fraction="")
    seam = dict(aa[0], free_only_reached_bulk_area_fraction=0)
    assert aggregate([native], "free_only_reached_bulk_area_fraction", "ORIGINAL_FINITE_ONLY")["mean"] is None
    assert aggregate([seam], "free_only_reached_bulk_area_fraction", "ORIGINAL_FINITE_ONLY")["mean"] == 0
    assert paired([native], [seam], "free_only_reached_bulk_area_fraction", "ORIGINAL_FINITE_ONLY")["valid_requests"] == 0
    assert number("nan") is None and number(True) is None
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    return dict(status="PASS", checks="hierarchical mean, common PASS support, NA versus actual zero, finite values",
                model_calls=0, scoring_calls=0, output_written=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-ready", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        result = self_test() if args.self_test else final() if args.final else readiness()
    except (ValueError, KeyError, OSError) as error:
        result = dict(schema=VERSION, status="INVALID_INPUT", reason=str(error), model_calls=0, scoring_calls=0)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] in ("NOT_READY", "INVALID_INPUT"):
        sys.exit(2)
