"""Read-only report aggregation after the complete fixed-panel score collection.

This independent control utility does not import the pilot, Torch, geometry
scorers, model code, or visual labels. It never modifies original score rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent.parent
DEST = OUT / "results/report_metrics_v1"
LABELS = ("EXPOSURE", "PLAIN1500", "MULTILOD1500")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
CONTRASTS = (("PLAIN1500", "MULTILOD1500"), ("EXPOSURE", "PLAIN1500"), ("EXPOSURE", "MULTILOD1500"))
SCHEMA = "BENCH_MULTILOD_REPORT_METRICS_V1"
META = ("label", "stage", "request_id", "condition_id", "parent_id", "noise_index", "report_group", "eval_group", "N", "K")
# Values are fractions, counts, or fixed model units; percentages are scaled only
# in Markdown. Counts/area retention do not by themselves imply better quality.
METRICS = {
    "CF_illegal_affected_free_fraction": ("fraction", "lower", "Affected free faces / actual free faces"),
    "FF_illegal_affected_free_fraction": ("fraction", "lower", "Affected free faces / actual free faces"),
    "I_common_PASS": ("fraction", "lower", "Original ConflictCov_C, common PASS parent/request support only"),
    "CF_illegal_affected_free_count_derived": ("count", "lower", "Original fraction times actual free denominator, integer checked"),
    "FF_illegal_affected_free_count_derived": ("count", "lower", "Original fraction times actual free denominator, integer checked"),
    "CF_illegal_pair_count": ("count", "lower", "Original illegal C/F face-pair count"),
    "FF_illegal_pair_count": ("count", "lower", "Original illegal F/F face-pair count"),
    "C_F_has_valid_edge": ("fraction", "higher", "Descriptive presence of a legal C/F edge; not semantic success"),
    "C_F_valid_edge_count": ("count", "higher", "Actual legal C/F shared edges"),
    "C_F_valid_edge_total_length": ("model_length", "higher", "Actual legal C/F shared-edge length"),
    "C_F_has_opposite_winding_edge": ("fraction", "higher", "Presence of a legal opposed C/F edge"),
    "hinge_diag_legal_opposed_edge_C_reached_free_area_fraction": ("fraction", "higher", "Direction-compatible legal-edge reachable free area; not sealed boundary coverage"),
    "free_minimum_angle_lt5_fraction": ("fraction", "lower", "Actual free faces with minimum angle under five degrees"),
    "free_q_gt10_fraction": ("fraction", "lower", "Original free q greater than ten fraction"),
    "free_near_degenerate_fraction": ("fraction", "lower", "Original near-degenerate free fraction"),
    "nonmanifold_edges": ("count", "lower", "Original indexed nonmanifold edges"),
    "hinge_diag_isolated_1face_edge_components": ("count", "lower", "Single-free-face components in the original legal-edge graph"),
    "actual_free_faces": ("count", "descriptive", "Includes retained free output and new bridge faces"),
    "free_face_denominator": ("count", "descriptive", "Actual denominator used by original CF/FF fractions"),
    "deleted_free_faces": ("count", "descriptive", "Missing original RAW slots, not requested N minus output faces"),
    "deleted_RAW_slot_fraction": ("fraction", "descriptive", "Missing original RAW slots / original RAW free count"),
    "retained_RAW_slot_fraction": ("fraction", "descriptive", "One minus missing original RAW slot fraction"),
    "free_total_surface_area": ("model_area", "descriptive", "Actual stage free surface area"),
    "free_area_ratio_to_RAW": ("fraction", "descriptive", "Per-request stage area / same-model RAW area; includes deformation and bridge effects"),
    "new_bridge_faces": ("count", "descriptive", "Original newly introduced bridge count"),
}
TAIL_METRICS = ("CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction", "I_common_PASS",
                "free_minimum_angle_lt5_fraction", "nonmanifold_edges", "hinge_diag_isolated_1face_edge_components",
                "C_F_valid_edge_total_length", "hinge_diag_legal_opposed_edge_C_reached_free_area_fraction",
                "deleted_RAW_slot_fraction", "free_area_ratio_to_RAW")


def resolve(path):
    s = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def average(values):
    values = [float(v) for v in values if number(v)]
    return statistics.mean(values) if values else None


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*fraction
    left, right = math.floor(index), math.ceil(index)
    return values[left] + (values[right]-values[left])*(index-left)


def immutable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "Immutable report artifact differs: " + str(path))
    else:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)


def write_json(path, value):
    immutable(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def csv_text(rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return stream.getvalue()


def aggregate(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, children in sorted(groups.items()):
        row = dict(zip(keys, key), child_rows=len(children))
        for field in fields:
            values = [r[field] for r in children if number(r.get(field))]
            row[field] = average(values)
            row[field+"__valid_n"] = len(values)
        result.append(row)
    return result


def i_status(row):
    statuses = [row.get("I_status"), row.get("ConflictCov_C_status"), row.get("status_by_metric", {}).get("I")]
    present = [s for s in statuses if s is not None]
    require(present and len(set(present)) == 1, "I status fields disagree or are absent")
    return present[0]


def support(rows, requests):
    by = {(r["label"], r["stage"], r["request_id"]): r for r in rows}
    expected = {(label, stage, q["request_id"]) for label in LABELS for stage in STAGES for q in requests}
    require(len(by) == len(rows) and set(by) == expected, "Incomplete or duplicated three-model/three-stage panel")
    query_by_parent = defaultdict(list)
    records = []
    for q in requests:
        cells = [by[label, stage, q["request_id"]] for label in LABELS for stage in STAGES]
        for row in cells:
            require(all(row[k] == q[k] for k in ("request_id", "parent_id", "condition_id", "noise_index", "N", "K", "report_group", "eval_group")),
                    "Score metadata changed relative to frozen request")
        ok = all(i_status(r) == "PASS" and number(r.get("ConflictCov_C")) for r in cells)
        query_by_parent[q["parent_id"]].append(ok)
        records.append(dict(request_id=q["request_id"], parent_id=q["parent_id"], report_group=q["report_group"],
                            request_common_PASS=ok, statuses={r["label"]+"/"+r["stage"]: i_status(r) for r in cells},
                            original_values={r["label"]+"/"+r["stage"]: r.get("ConflictCov_C") for r in cells},
                            query_weight_denominators={r["label"]+"/"+r["stage"]: r.get("C_query_weight_denominator") for r in cells}))
    # Keep the entire frozen noise/condition set for every parent in the main I
    # table. A partially supported parent is disclosed separately, never silently
    # represented by its convenient subset of noises.
    parents = {p for p, flags in query_by_parent.items() if all(flags)}
    eligible = {r["request_id"] for r in records if r["request_common_PASS"] and r["parent_id"] in parents}
    for r in records:
        r["parent_all_requests_common_PASS"] = r["parent_id"] in parents
        r["main_I_eligible"] = r["request_id"] in eligible
        r["exclusion_reason"] = None if r["main_I_eligible"] else (
            "PARENT_OTHER_REQUEST_NOT_COMMON_PASS" if r["request_common_PASS"] else "ONE_OR_MORE_MODEL_STAGE_I_NOT_PASS")
    return by, eligible, records, dict(
        included_parents=sorted(parents), included_request_ids=sorted(eligible),
        included_parent_count=len(parents), included_request_count=len(eligible),
        requestwise_common_PASS_count=sum(r["request_common_PASS"] for r in records),
        partial_support_parent_ids=sorted(p for p, flags in query_by_parent.items() if any(flags) and not all(flags)),
        excluded_parent_ids=sorted(set(query_by_parent)-parents),
        historical_reference_only="EXPOSURE report used 19 parents / 38 requests across two models and three stages; this number is not enforced.",
        policy="All three models and all three stages must have PASS and finite original ConflictCov_C for each request; all registered requests of a parent must pass for that parent to enter main I.")


def primary_rows(rows, requests, manifest):
    by, eligible, audit, domain = support(rows, requests)
    train_parents = {p["parent_id"] for p in manifest["parents"]}
    low_parents = {p["parent_id"] for p in manifest["parents"] if any(v["kind"] == "LOW" for v in p["variants"])}
    result = []
    for label in LABELS:
        for stage in STAGES:
            for q in requests:
                r = by[label, stage, q["request_id"]]
                raw = by[label, "RAW", q["request_id"]]
                item = {k: r[k] for k in META}
                item["low_variant_stratum"] = ("LOW_AVAILABLE_TRAIN" if r["parent_id"] in low_parents else
                                               "NATURAL_ONLY_TRAIN" if r["parent_id"] in train_parents else "OUTSIDE_TRAIN_POOL")
                derived = {"I_common_PASS", "CF_illegal_affected_free_count_derived", "FF_illegal_affected_free_count_derived",
                           "deleted_RAW_slot_fraction", "retained_RAW_slot_fraction", "free_area_ratio_to_RAW"}
                for key in set(METRICS)-derived:
                    require(key in r, "Missing original registered metric: " + key)
                    item[key] = r[key]
                denom = r["free_face_denominator"]
                require(number(denom) and denom > 0 and denom == r["actual_free_faces"], "CF/FF denominator is not actual output free faces")
                for kind in ("CF", "FF"):
                    fraction = r[kind+"_illegal_affected_free_fraction"]
                    require(number(fraction) and 0 <= fraction <= 1, "Invalid original fraction")
                    affected = fraction*denom
                    require(abs(affected-round(affected)) < 1e-6, "Fraction does not represent an integer affected-face count")
                    item[kind+"_illegal_affected_free_count_derived"] = int(round(affected))
                item.update(original_ConflictCov_C=r["ConflictCov_C"], original_I_status=i_status(r),
                            main_I_eligible=q["request_id"] in eligible,
                            I_common_PASS=r["ConflictCov_C"] if q["request_id"] in eligible else None,
                            CF_tested_pair_denominator=r["CF_tested_pair_denominator"],
                            FF_tested_pair_denominator=r["FF_tested_pair_denominator"],
                            C_query_weight_denominator=r["C_query_weight_denominator"])
                require(raw["actual_free_faces"] == q["N"]-q["K"], "Original RAW free count differs from frozen request")
                deletion = r["deleted_free_faces"]
                require(number(deletion) and 0 <= deletion <= raw["actual_free_faces"], "Deleted original-slot count outside its original denominator")
                item["raw_original_free_slot_denominator"] = raw["actual_free_faces"]
                item["deleted_RAW_slot_fraction"] = deletion/raw["actual_free_faces"]
                item["retained_RAW_slot_fraction"] = 1.-item["deleted_RAW_slot_fraction"]
                raw_area = raw["free_total_surface_area"]
                item["RAW_free_area_denominator"] = raw_area
                item["free_area_ratio_to_RAW"] = r["free_total_surface_area"]/raw_area if number(raw_area) and raw_area > 0 else None
                item["area_ratio_status"] = "PASS" if item["free_area_ratio_to_RAW"] is not None else "NA_NONPOSITIVE_RAW_AREA"
                result.append(item)
    return result, audit, domain


def macro_tables(rows, fields):
    identity = ("label", "stage", "report_group", "eval_group", "low_variant_stratum")
    conditions = aggregate(rows, (*identity, "parent_id", "condition_id"), fields)
    parents = aggregate(conditions, (*identity, "parent_id"), fields)
    summaries = []
    for label in LABELS:
        for stage in STAGES:
            cell = [r for r in parents if r["label"] == label and r["stage"] == stage]
            scopes = [("ALL", "ALL", cell)]
            scopes += [("report_group", g, [r for r in cell if r["report_group"] == g]) for g in sorted({r["report_group"] for r in cell})]
            scopes += [("low_variant_stratum", g, [r for r in cell if r["low_variant_stratum"] == g]) for g in sorted({r["low_variant_stratum"] for r in cell})]
            for scope, group, selected in scopes:
                row = dict(label=label, stage=stage, scope=scope, group=group, parents=len(selected))
                for metric in fields:
                    values = [r[metric] for r in selected if number(r.get(metric))]
                    row[metric] = average(values)
                    row[metric+"__parents"] = len(values)
                selected_ids = {r["parent_id"] for r in selected}
                outputs = [r for r in rows if r["label"] == label and r["stage"] == stage and r["parent_id"] in selected_ids]
                row.update(requests=len(outputs), I_requests=sum(r["main_I_eligible"] for r in outputs),
                           legal_CF_output_count=sum(r["C_F_has_valid_edge"] for r in outputs),
                           CF_illegal_pairs_total=sum(r["CF_illegal_pair_count"] for r in outputs),
                           FF_illegal_pairs_total=sum(r["FF_illegal_pair_count"] for r in outputs))
                summaries.append(row)
    return conditions, parents, summaries


def paired_tables(rows, parents):
    by = {(r["label"], r["stage"], r["request_id"]): r for r in rows}
    paired = []
    transitions = []
    for ref, cand in CONTRASTS:
        for a in rows:
            if a["label"] != ref:
                continue
            b = by[cand, a["stage"], a["request_id"]]
            row = dict(reference=ref, candidate=cand, **{k:a[k] for k in META if k != "label"},
                       low_variant_stratum=a["low_variant_stratum"], main_I_eligible=a["main_I_eligible"])
            for metric in METRICS:
                av, bv = a[metric], b[metric]
                row[metric+"__reference"], row[metric+"__candidate"] = av, bv
                row[metric+"__delta"] = bv-av if number(av) and number(bv) else None
            row["legal_CF_transition"] = ("RETAINED" if a["C_F_has_valid_edge"] else "GAINED") if b["C_F_has_valid_edge"] else ("LOST" if a["C_F_has_valid_edge"] else "NEITHER")
            paired.append(row)
    delta_fields = [m+"__delta" for m in METRICS]
    keys = ("reference", "candidate", "stage", "report_group", "eval_group", "low_variant_stratum")
    pc = aggregate(paired, (*keys, "parent_id", "condition_id"), delta_fields)
    pp = aggregate(pc, (*keys, "parent_id"), delta_fields)
    summaries = []
    for ref, cand in CONTRASTS:
        for stage in STAGES:
            cell = [r for r in pp if (r["reference"],r["candidate"],r["stage"]) == (ref,cand,stage)]
            scopes = [("ALL","ALL",cell)]
            scopes += [("report_group",g,[r for r in cell if r["report_group"] == g]) for g in sorted({r["report_group"] for r in cell})]
            scopes += [("low_variant_stratum",g,[r for r in cell if r["low_variant_stratum"] == g]) for g in sorted({r["low_variant_stratum"] for r in cell})]
            for scope, group, selected in scopes:
                out = dict(reference=ref,candidate=cand,stage=stage,scope=scope,group=group,parents=len(selected))
                for metric in delta_fields:
                    values=[r[metric] for r in selected if number(r[metric])]
                    out[metric]=average(values);out[metric+"__parents"]=len(values)
                summaries.append(out)
            for parent in sorted({r["parent_id"] for r in cell}):
                requests=[r for r in paired if (r["reference"],r["candidate"],r["stage"],r["parent_id"]) == (ref,cand,stage,parent)]
                change=dict(reference=ref,candidate=cand,stage=stage,parent_id=parent,report_group=requests[0]["report_group"],
                            low_variant_stratum=requests[0]["low_variant_stratum"],requests=len(requests))
                for key, metric in (("legal_CF", "C_F_has_valid_edge"), ("opposed_reach_positive", "hinge_diag_legal_opposed_edge_C_reached_free_area_fraction")):
                    old={r["request_id"] for r in requests if r[metric+"__reference"] > 0}
                    new={r["request_id"] for r in requests if r[metric+"__candidate"] > 0}
                    change.update({key+"_reference_requests":len(old),key+"_candidate_requests":len(new),
                                   key+"_gained":sorted(new-old),key+"_lost":sorted(old-new),
                                   key+"_reference_all_noises":len(old)==len(requests),key+"_candidate_all_noises":len(new)==len(requests)})
                p=next(r for r in cell if r["parent_id"] == parent)
                change.update({k:p[k] for k in delta_fields})
                transitions.append(change)
    return paired, pc, pp, summaries, transitions


def tails(rows, paired):
    ranked, summary, delta_ranked = [], [], []
    for label in LABELS:
        for stage in STAGES:
            selected=[r for r in rows if r["label"]==label and r["stage"]==stage]
            for metric in TAIL_METRICS:
                values=[r for r in selected if number(r[metric])]
                direction=METRICS[metric][1]
                # Descriptive quantities are sorted high-to-low, without
                # declaring a high count/area ratio intrinsically better/worse.
                ordered=sorted(values,key=lambda r:((-r[metric] if direction != "higher" else r[metric]),r["request_id"]))
                for rank,r in enumerate(ordered,1):
                    ranked.append(dict(label=label,stage=stage,metric=metric,rank=rank,ordering="ascending" if direction=="higher" else "descending",
                                       direction=direction,request_id=r["request_id"],parent_id=r["parent_id"],report_group=r["report_group"],value=r[metric]))
                vals=[r[metric] for r in values]
                summary.append(dict(label=label,stage=stage,metric=metric,valid_requests=len(vals),
                                    minimum=min(vals) if vals else None,median=percentile(vals,.5),p90=percentile(vals,.9),p95=percentile(vals,.95),
                                    maximum=max(vals) if vals else None,scope="Descriptive request distribution; main averages remain parent-weighted"))
    for ref,cand in CONTRASTS:
        for stage in STAGES:
            selected=[r for r in paired if (r["reference"],r["candidate"],r["stage"])==(ref,cand,stage)]
            for metric in TAIL_METRICS:
                direction=METRICS[metric][1];field=metric+"__delta"
                values=[r for r in selected if number(r[field])]
                ordered=sorted(values,key=lambda r:((-r[field] if direction != "higher" else r[field]),r["request_id"]))
                for rank,r in enumerate(ordered,1):
                    delta_ranked.append(dict(reference=ref,candidate=cand,stage=stage,metric=metric,rank=rank,direction=direction,
                        request_id=r["request_id"],parent_id=r["parent_id"],report_group=r["report_group"],
                        reference_value=r[metric+"__reference"],candidate_value=r[metric+"__candidate"],delta=r[field]))
    return ranked, summary, delta_ranked


def markdown(summaries, paired, domain):
    pct=lambda x:"NA" if not number(x) else f"{100*x:.3f}"
    num=lambda x:"NA" if not number(x) else f"{x:.3f}"
    lines=["# Multi-LOD：固定评分的报告用汇总", "", "本文件只汇总已完成评分，不读取图审标签，不产生语义采用或主体完整性的判断。", "",
           "先在父内按噪声→条件平均，再对父等权；百分比仅在下表乘100。CF/FF分母为当前阶段实际free面。",
           f"I沿用原ConflictCov_C，仅使用三模型×三阶段共同PASS的{domain['included_parent_count']}父／{domain['included_request_count']}请求；其余原值保留，主I记NA。", "",
           "| 阶段 | 模型 | CF% | FF% | I% | 合法C/F输出 | 合法共边长度 | 方向可达面积% | 小角% | 非流形边 | 单面孤岛 | 删原槽 | free面积/RAW% |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["scope"]!="ALL":continue
        lines.append("| "+" | ".join([r['stage'],r['label'],pct(r['CF_illegal_affected_free_fraction']),pct(r['FF_illegal_affected_free_fraction']),
            pct(r['I_common_PASS']),f"{r['legal_CF_output_count']}/{r['requests']}",num(r['C_F_valid_edge_total_length']),
            pct(r['hinge_diag_legal_opposed_edge_C_reached_free_area_fraction']),pct(r['free_minimum_angle_lt5_fraction']),
            num(r['nonmanifold_edges']),num(r['hinge_diag_isolated_1face_edge_components']),num(r['deleted_free_faces']),pct(r['free_area_ratio_to_RAW'])])+" |")
    lines += ["", "所有组别、LOW可用父的描述性分层、全部逐输出差值和数值尾部均在同目录CSV；LOW可用性不是随机分组，不能据此归因为父类别或增强效果。",
              "", "面积比包含D0变形、焊接、删面和新增桥面；不是删除面RAW参考面积比例，也不是GT完整性。接口出现及图可达也不等同语义采用或完整接缝。",
              "", "三项配对分别为MULTILOD−PLAIN、PLAIN−EXPOSURE、MULTILOD−EXPOSURE。所有NA保留，未补零；未选取有利指标替换原177字段。", ""]
    return "\n".join(lines)


def run(check_only=False):
    paths=dict(rows=OUT/"results/metric_rows.json", collection=OUT/"results/completion.json",
               panel=OUT/"data/eval_manifest.json", data=OUT/"data/manifest.json")
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:
        return dict(status="WAITING_INPUT",missing_files=missing,model_calls=0,scoring_calls=0)
    collection=read(paths['collection'])
    if collection.get('status')!='COMPLETE':
        return dict(status="WAITING_INPUT",collector_status=collection.get('status'),model_calls=0,scoring_calls=0)
    recorded={resolve(p).resolve():h for p,h in collection['files'].items()}
    require(recorded.get(paths['rows'].resolve())==sha(paths['rows']),"Collected metric_rows is not authenticated by completion")
    source_rows=read(paths['rows']); panel=read(paths['panel']); manifest=read(paths['data'])
    require(len(source_rows)==len(LABELS)*len(STAGES)*len(panel['requests'])==collection['rows'],"Collector cardinality differs")
    require(len(collection['metric_fields'])==177 and all(all(m in r for m in collection['metric_fields']) for r in source_rows),"Lossless 177-field evidence missing")
    rows,audit,domain=primary_rows(source_rows,panel['requests'],manifest)
    if check_only:
        return dict(status="READY",rows=len(rows),I_domain=domain,model_calls=0,scoring_calls=0)
    fields=list(METRICS)
    conditions,parents,summaries=macro_tables(rows,fields)
    paired,pc,pp,ps,transitions=paired_tables(rows,parents)
    ranked,tail_summary,delta_ranked=tails(rows,paired)
    tables=dict(primary_per_output=rows,primary_per_condition=conditions,primary_per_parent=parents,
                primary_parent_macro=summaries,paired_primary_per_output=paired,paired_primary_per_condition=pc,
                paired_primary_per_parent=pp,paired_primary_parent_macro=ps,parent_numeric_transitions=transitions,
                I_common_support=audit,per_output_tail_ranking=ranked,tail_distribution=tail_summary,
                paired_per_output_tail_ranking=delta_ranked)
    for name,values in tables.items():immutable(DEST/(name+'.csv'),csv_text(values))
    write_json(DEST/'I_domain.json',domain)
    dictionary={k:dict(unit=v[0],direction=v[1],definition=v[2]) for k,v in METRICS.items()}
    write_json(DEST/'metric_dictionary.json',dictionary)
    immutable(DEST/'METRICS.md',markdown(summaries,ps,domain))
    filenames=[name+'.csv' for name in tables]+['I_domain.json','metric_dictionary.json','METRICS.md']
    done=dict(status="COMPLETE",schema=SCHEMA,rows=len(rows),parents=len({r['parent_id'] for r in rows}),
              I_domain=domain,source_files={str(p):sha(p) for p in paths.values()},code_sha256=sha(__file__),
              model_calls=0,scoring_calls=0,geometry_changes=0,visual_labels_read=False,
              full177_table_preserved=True,aggregation="Noise -> condition -> parent -> group/all; equal parent weight",
              no_statistical_significance_claim=True,no_unique_GT_success_claim=True,
              files={str(DEST/name):sha(DEST/name) for name in filenames})
    write_json(DEST/'completion.json',done)
    return done


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    result=run(args.check)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','source_files')},ensure_ascii=False))
