"""CPU-only extraction of existing, completed Bench evaluation tables.

No mesh loading, scoring, model imports, new metric definitions, or selection
of checkpoints. Missing registered checkpoints remain PENDING. Original 124
fields and their validity denominators are retained alongside concise views.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FOCUS = [
    "C_preserved_exact", "CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction",
    "ConflictCov_C", "ConflictCov_C_upper", "ConflictCov_C_uncertain_fraction",
    "CF_illegal_pair_count", "FF_illegal_pair_count", "C_F_has_valid_edge",
    "C_F_valid_edge_count", "C_F_valid_edge_total_length",
    "C_F_valid_opposite_winding_edge_count", "opposite_winding_reached_free_area_fraction",
    "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive",
    "legal_seam_bulk_edge_count", "seed_connected_legal_seam_bulk_edge_count",
    "A_VMR__L_tau_0.005", "PMR__L_tau_0.005", "boundary_distance_mean_model",
    "free_minimum_angle_lt5_fraction", "free_q_gt10_fraction", "free_angle_p05_degrees",
    "free_near_degenerate_count", "free_exact_zero_area_count", "actual_free_faces",
    "deleted_free_faces", "deleted_seam_faces", "nonmanifold_edges", "C_F_nonmanifold_edge_count",
    "EXACT_COORD_face_components", "EXACT_COORD_same_direction_two_face_edges",
    "free_total_surface_area", "retained_seam_faces", "retained_bulk_faces",
]


def numeric(value):
    if value in ("True", "False", True, False):
        return float(value in ("True", True))
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def unit(field):
    if field.startswith(("A_VMR__", "PMR__")) or "_fraction" in field or field in (
        "ConflictCov_C", "ConflictCov_C_upper", "ConflictCov_C_cross", "ConflictCov_C_dup", "ConflictCov_C_cross_and_dup"):
        return "fraction [0,1]; delta in fraction units, x100 only for pp display"
    if field in ("C_preserved_exact", "C_F_has_valid_edge", "C_F_has_opposite_winding_edge",
                 "bulk_reach_ge_0_30_descriptive", "EXACT_COORD_closed_two_faces_per_edge"):
        return "per-output Boolean; parent-equal mean is a fraction"
    if field == "free_angle_p05_degrees":
        return "degrees"
    if field == "free_q_mean":
        return "dimensionless original shape factor"
    if field == "C_query_weight_denominator":
        return "normalized query weight"
    if field.startswith("boundary_distance_") or field.endswith(("_length", "_length_model")):
        return "model-coordinate length (not %L)"
    if field.endswith("_area"):
        return "squared model-coordinate units"
    if (field.endswith(("_count", "_faces", "_edges", "_vertices", "_denominator", "_samples", "_pairs", "_components", "_failures"))
            or field.startswith("illegal_pairs_")):
        return "count; parent-equal mean may be fractional"
    raise ValueError("Unmapped original numeric field unit: " + field)


def read_file(path, sources, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError("Completed source hash differs: " + str(path))
    sources[str(path)] = dict(sha256=digest, bytes=len(data))
    return data


def read_csv(path, sources, expected=None):
    return list(csv.DictReader(io.StringIO(read_file(path, sources, expected).decode("utf-8-sig"))))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tag_rows(rows, table_id):
    return [dict(source_table=table_id, **row, original_unit=unit(row["metric"])) for row in rows]


def paired_deltas(rows, grouping, value_field):
    pairs = defaultdict(dict)
    for row in rows:
        arm = "SEAM" if row["label"].startswith("SEAM_") else "NATIVE"
        pairs[tuple(row[key] for key in grouping)][arm] = row
    result = []
    for key, arms in sorted(pairs.items()):
        if "NATIVE" not in arms or "SEAM" not in arms:
            continue
        a, b = arms["NATIVE"], arms["SEAM"]
        av, bv = numeric(a[value_field]), numeric(b[value_field])
        result.append(dict(zip(grouping, key), baseline=a["label"], candidate=b["label"],
                           baseline_value=a[value_field], candidate_value=b[value_field],
                           delta_SEAM_minus_NATIVE=bv-av if av is not None and bv is not None else "",
                           status="DEFINED" if av is not None and bv is not None else "NA",
                           original_unit=a["original_unit"],
                           baseline_denominators={k: a[k] for k in a if k.startswith("n_")},
                           candidate_denominators={k: b[k] for k in b if k.startswith("n_")}))
    # Keep nested validity information lossless as JSON strings within CSV cells.
    for row in result:
        for key in ("baseline_denominators", "candidate_denominators"):
            row[key] = json.dumps(row[key], sort_keys=True)
    return result


def denominator_rows(outputs):
    groups = defaultdict(list)
    for row in outputs:
        groups[(row["label"], row["stage"], row["split"])].append(row)
    result = []
    for (label, stage, split), rows in sorted(groups.items()):
        open_rows = [r for r in rows if r["closed_C"] == "False"]
        field_counts = {}
        for field in ("A_VMR__L_tau_0.005", "PMR__L_tau_0.005", "C_F_has_valid_edge",
                      "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive"):
            valid = [numeric(r[field]) for r in rows if numeric(r[field]) is not None]
            field_counts[field] = dict(n_registered=len(rows), n_valid=len(valid), n_NA=len(rows)-len(valid),
                n_positive=sum(x > 0 for x in valid), status_counts=dict(Counter(
                    r.get("seam_to_bulk_status", "") for r in rows)) if field.startswith(("free_only", "bulk_reach")) else None)
        result.append(dict(label=label, stage=stage, split=split, requests=len(rows),
            conditions=len({r["condition_id"] for r in rows}), parents=len({r["parent_id"] for r in rows}),
            open_C_requests=len(open_rows), closed_C_requests=len(rows)-len(open_rows),
            C_preserved_exact_counts=dict(Counter(r["C_preserved_exact"] for r in rows)),
            C_F_has_valid_edge_whole_panel_positive=sum(numeric(r["C_F_has_valid_edge"]) == 1 for r in rows),
            C_F_has_valid_edge_open_C_only_positive=sum(numeric(r["C_F_has_valid_edge"]) == 1 for r in open_rows),
            open_C_only_is_descriptive_filter_not_new_score=True, field_applicability=field_counts,
            per_output_denominators=[{k: r[k] for k in ("request_id", "condition_id", "parent_id", "part_type",
                "closed_C", "N", "K", "free_face_denominator", "CF_tested_pair_denominator", "FF_tested_pair_denominator",
                "C_query_weight_denominator", "seam_to_bulk_status")} for r in rows]))
    return result


def build(root, final_step):
    sources = {}
    specs = [("COMMON12000", "NATIVE_S12000_full"),
             ("PAIRED15000", "NATIVE_S15000__SEAM_S15000_full"),
             ("FINAL", f"NATIVE_S{final_step}__SEAM_S{final_step}_full")]
    stages = []
    summary, parents, conditions, output_pairs, outputs, deleted = [], [], [], [], [], []
    field_sets = {}
    seen_tables = set()
    for role, table_id in specs:
        folder = root / "evaluation/tables" / table_id
        completion_path = folder / "completion.json"
        if not completion_path.exists():
            stages.append(dict(role=role, table=table_id, status="PENDING", reason="completed unified table absent"))
            continue
        completion = json.loads(read_file(completion_path, sources))
        if completion.get("status") != "COMPLETE":
            stages.append(dict(role=role, table=table_id, status="PENDING", reason="table completion is not COMPLETE"))
            continue
        stages.append(dict(role=role, table=table_id, status="COMPLETE", rows=completion["rows"],
                           labels=completion["labels"], aggregation=completion["aggregation"]))
        if table_id in seen_tables:
            continue
        seen_tables.add(table_id)
        numeric_fields = completion["numeric_fields"]
        field_sets[table_id] = {field: unit(field) for field in numeric_fields}
        if len(numeric_fields) != 124:
            raise ValueError(f"Original field count is not 124: {table_id}")
        files = completion["files"]
        for name, dest in (("summary_parent_equal.csv", summary), ("per_parent.csv", parents),
                           ("per_condition.csv", conditions), ("paired_per_output.csv", output_pairs)):
            p = folder / name
            if p.exists():
                rows = read_csv(p, sources, files.get(str(p)))
                dest.extend(tag_rows(rows, table_id))
        p = folder / "per_output.csv"
        rows = read_csv(p, sources, files.get(str(p)))
        if len(rows) != completion["rows"]:
            raise ValueError("Completed row count differs: " + table_id)
        key_counts = Counter((r["label"], r["stage"], r["request_id"]) for r in rows)
        if any(x != 1 for x in key_counts.values()):
            raise ValueError("Duplicate per-output identity")
        outputs.extend(dict(source_table=table_id, **r) for r in rows)
        p = folder / "deleted_faces.csv"
        if p.exists():
            deleted.extend(dict(source_table=table_id, **r) for r in read_csv(p, sources, files.get(str(p))))
    main = [r for r in summary if r["part_type"] == "ALL"]
    parts = [r for r in summary if r["part_type"] != "ALL"]
    deletion_counts = Counter((r["source_table"], r["label"], r["split"], r["role"], r["reason"]) for r in deleted)
    return dict(schema="bench_parts_report_numeric_extraction_v1", status="COMPLETE_TABLE_EXTRACTION" if all(s["status"]=="COMPLETE" for s in stages) else "PRELIMINARY_PENDING_FINAL",
        generated_utc=datetime.now(timezone.utc).isoformat(), requested_final_step=final_step,
        no_model_calls=True, no_rescoring=True, no_new_mesh_metrics=True,
        original_metric_values_and_units_preserved=True, source_tables=stages,
        aggregation="Existing noise -> condition -> parent equal means copied verbatim; part strata separate. Delta tables subtract existing means without rescore.",
        output_count_summaries_are_descriptive_not_parent_equal=True,
        final_results_are_pending_not_imputed=any(s["status"]=="PENDING" for s in stages),
        metric_units=field_sets, sources=sources, main=main, parts=parts, parents=parents,
        conditions=conditions, output_pairs=output_pairs, denominators=denominator_rows(outputs),
        deletion_reason_counts=[dict(source_table=k[0], label=k[1], split=k[2], role=k[3], reason=k[4], deleted_faces=v)
                                for k,v in sorted(deletion_counts.items())],
        summary_deltas=paired_deltas(summary, ("source_table", "stage", "split", "part_type", "metric"), "mean_parent_equal"),
        parent_deltas=paired_deltas(parents, ("source_table", "stage", "split", "parent_id", "metric"), "value"))


def markdown(data):
    labels = list(dict.fromkeys(r["label"] for r in data["main"]))
    lines = ["# Bench 已有数值摘要（报告支持，" + data["status"] + "）", "",
        "只读完成表，0模型调用、0重评分。保留全部124字段；以下是便于阅读的质量、接续与代价字段，不是事后挑选的论文主指标。",
        "原始比例保持0–1，未换算成百分比；长度为模型坐标单位，面积为模型坐标平方单位。均值为原表父等权值；括号内为有效输出/登记输出。", "",
        "闭合C不要求接续。原一般接口字段对闭合C可能为0，不能把全组接口有无直接称为任务成功率；seam特有字段对Native及闭合C保留NA。图中薄片/孔洞/主体缺失不能由小角或连通分量数直接认证，仍需独立AI_PROXY图审。", "",
        "## 完成状态", ""]
    lines.extend("- " + s["role"] + "：`" + s["table"] + "`，" + s["status"] for s in data["source_tables"])
    for stage in ("RAW", "POST"):
        for split in ("TRAIN", "DEV"):
            lookup = {(r["label"],r["metric"]):r for r in data["main"] if r["stage"]==stage and r["split"]==split}
            lines += ["", f"## {stage} / {split}：父等权", "", "| 原字段 | " + " | ".join(labels) + " |", "|---|" + "---:|"*len(labels)]
            for metric in FOCUS:
                cells = []
                for label in labels:
                    r = lookup.get((label,metric))
                    if not r:
                        cells.append("PENDING")
                    elif numeric(r["mean_parent_equal"]) is None:
                        cells.append(f"NA ({r['n_valid']}/{r['n_outputs']})")
                    else:
                        cells.append(f"{numeric(r['mean_parent_equal']):.7g} ({r['n_valid']}/{r['n_outputs']})")
                lines.append("| `" + metric + "` | " + " | ".join(cells) + " |")
    lines += ["", "## 使用边界", "",
        "- `parent_equal_all124.csv`、`part_strata_all124.csv`、`per_parent_all124.csv`保留原值及分母；原表哈希见manifest。",
        "- `paired_summary_deltas.csv`、`paired_parent_deltas.csv`分别是同checkpoint的SEAM−Native，不混用不同训练预算。",
        "- `denominators.json`披露闭合C、seam适用状态、C精确保持、CF/FF面对分母；其请求计数只是描述性计数。",
        "- `deletion_reason_counts.json`来自原删面账本，不根据面数差猜删面原因；删面降低不自动代表完整主体保住。",
        "- 本入口不按DEV挑checkpoint、不晋级默认、不将来源唯一款式作为成功标准。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--final-step", type=int, default=21000)
    args = p.parse_args()
    data = build(args.root.resolve(), args.final_step)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = HERE / "numeric_snapshots" / stamp
    out.mkdir(parents=True, exist_ok=False)
    for name, key in (("parent_equal_all124.csv", "main"), ("part_strata_all124.csv", "parts"),
        ("per_parent_all124.csv", "parents"), ("per_condition_all124.csv", "conditions"),
        ("existing_paired_per_output_all124.csv", "output_pairs"),
        ("paired_summary_deltas.csv", "summary_deltas"), ("paired_parent_deltas.csv", "parent_deltas")):
        write_csv(out/name, data[key])
    for key in ("denominators", "deletion_reason_counts", "metric_units"):
        write_json(out/(key+".json"), data[key])
    excluded = {"main", "parts", "parents", "conditions", "output_pairs", "summary_deltas", "parent_deltas", "denominators", "deletion_reason_counts", "metric_units"}
    manifest = {k:v for k,v in data.items() if k not in excluded}
    manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["output_row_counts"] = {k:len(data[k]) for k in excluded}
    write_json(out/"manifest.json", manifest)
    (out/"NUMERIC_SUMMARY.md").write_text(markdown(data), encoding="utf-8")
    write_json(HERE/"numeric_latest.json", dict(snapshot=str(out),status=data["status"],created_utc=data["generated_utc"]))
    print(json.dumps(dict(snapshot=str(out),status=data["status"],source_tables=data["source_tables"],row_counts=manifest["output_row_counts"]),ensure_ascii=False,indent=2))
