#!/usr/bin/env python3
"""Compare saved curriculum/R2 scalar tables; no model, scoring or selection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import json
from pathlib import Path
import statistics

_spec = importlib.util.spec_from_file_location("_curriculum_base_analysis", Path(__file__).with_name("analyze_recipe_factorial_results.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
VERSION = "recipe_curriculum_table_analysis_v1"
BASELINE = "R2_MIX_H_ALL"
COURSES = ("C20_40_MIX", "C40_20_MIX")
LABELS = (BASELINE, *COURSES)
PAIRS = ((COURSES[0], BASELINE), (COURSES[1], BASELINE), COURSES)
SCOPES = ("ALL", "ORIGINAL6", "EXTENDED2")
read_json, read_csv, digest = base.read_json, base.read_csv, base.digest
write_json, write_csv = base.write_json, base.write_csv
number, distribution = base.is_number, base.distribution


def validate_panel(main_registration, course_registration, requests, main_rows, course_rows, main_receipt, course_receipt):
    base.validate_panel(main_registration, requests, main_rows, main_receipt)
    if course_registration.get("kind") != "course" or tuple(course_registration.get("labels", ())) != COURSES:
        raise ValueError("Exactly the ordered two curriculum labels required")
    if course_registration.get("requests_sha256") != main_registration.get("requests_sha256"):
        raise ValueError("Course/main request identity differs")
    if course_registration.get("expected_outputs") != 128 or len(course_rows) != 128:
        raise ValueError("Complete 128 curriculum outputs required")
    if course_receipt.get("status") != "COMPLETE" or course_receipt.get("registered") != 128 or course_receipt.get("scored") != 128:
        raise ValueError("Complete course scoring receipt required")
    qby = {q["request_id"]: q for q in requests}
    rows = [r for r in main_rows if r["label"] == BASELINE] + course_rows
    by = {}
    for r in rows:
        key = r["label"], r["request_id"]
        if key in by or key[0] not in LABELS or key[1] not in qby:
            raise ValueError("Duplicate/unregistered comparison cell")
        q = qby[key[1]]
        for field in ("parent_id", "condition_id", "candidate", "ratio", "condition_cohort", "N", "K"):
            if r[field] != q[field]:
                raise ValueError("Scored request identity differs: " + field)
        if r.get("status") != "PASS" or not r.get("contracts") or not all(v is True for v in r["contracts"].values()):
            raise ValueError("Failed output contract or scoring status")
        by[key] = r
    if set(by) != {(label, rid) for label in LABELS for rid in qby}:
        raise ValueError("Incomplete 192-cell comparison")
    for rid in qby:
        reference = by[BASELINE, rid]
        for label in COURSES:
            row = by[label, rid]
            for field in ("query_sha256", "registered_free_faces", "registered_C_queries", "query_weight_denominator", "L0"):
                if row[field] != reference[field]:
                    raise ValueError("C/query/denominator binding differs: " + field)
            if row["identity"]["input_sha256"] != reference["identity"]["input_sha256"]:
                raise ValueError("Actual C/noise input identity differs")
    return rows


def membership(requests):
    result = {q["parent_id"]: q["condition_cohort"] for q in requests if q["ratio"] == .4}
    if Counter(result.values()) != {"ORIGINAL6": 6, "EXTENDED2": 2}:
        raise ValueError("Original6/Extended2 membership differs")
    return result


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        for scope in SCOPES:
            if scope == "ALL" or row["parent_membership"] == scope:
                groups[tuple(row[k] for k in keys) + (scope,)].append(row)
    result = []
    for values, group in sorted(groups.items()):
        record = dict(zip(keys, values[:-1]))
        record.update(parent_stratum=values[-1], parent_ids=[r["parent_id"] for r in group])
        result.append((record, group))
    return result


def tables(rows, requests, cohort, source="primary_per_output"):
    catalog = base.metric_catalog(rows)
    for r in catalog:
        r["source_table"] = source
        r["source_GT_used_for_promotion"] = False
        if source != "primary_per_output":
            r["direction"] = "reference_diagnostic_not_a_promotion_gate"
    metrics = [r["metric"] for r in catalog if r["numeric_summary"]]
    by = {(r["label"], r["request_id"]): r for r in rows}
    groups = defaultdict(list)
    for q in requests:
        groups[q["parent_id"], q["ratio"]].append(q)
    parents = []
    for label in LABELS:
        for (pid, ratio), qs in sorted(groups.items()):
            qs = sorted(qs, key=lambda q: q["candidate"])
            for metric in metrics:
                values = [by[label, q["request_id"]]["metrics"].get(metric) for q in qs]
                parents.append(dict(label=label, parent_id=pid, ratio=ratio,
                    condition_cohort=qs[0]["condition_cohort"], parent_membership=cohort[pid],
                    metric=metric, noises=list(base.NOISES), noise_values=values, **distribution(values, 4)))
    summary = [dict(key, aggregation="four noises then equal parents",
                    **distribution([r["mean"] for r in group], len(group)))
               for key, group in summarize(parents, ("label", "ratio", "metric"))]
    pby = {(r["label"], r["parent_id"], r["ratio"], r["metric"]): r for r in parents}
    noise, paired = [], []
    for first, second in PAIRS:
        for q in requests:
            a, b = by[first, q["request_id"]], by[second, q["request_id"]]
            for metric in metrics:
                av, bv = a["metrics"].get(metric), b["metrics"].get(metric)
                noise.append(dict(first=first, second=second, request_id=q["request_id"],
                    parent_id=q["parent_id"], ratio=q["ratio"], condition_cohort=q["condition_cohort"],
                    parent_membership=cohort[q["parent_id"]], candidate=q["candidate"], metric=metric,
                    first_value=av, second_value=bv, delta=av-bv if number(av) and number(bv) else None))
        for (pid, ratio), qs in sorted(groups.items()):
            for metric in metrics:
                a, b = pby[first,pid,ratio,metric], pby[second,pid,ratio,metric]
                dd = [av-bv if number(av) and number(bv) else None for av,bv in zip(a["noise_values"], b["noise_values"])]
                paired.append(dict(first=first, second=second, parent_id=pid, ratio=ratio,
                    condition_cohort=qs[0]["condition_cohort"], parent_membership=cohort[pid], metric=metric,
                    first_mean=a["mean"], second_mean=b["mean"], noise_deltas=dd,
                    delta=a["mean"]-b["mean"] if number(a["mean"]) and number(b["mean"]) else None,
                    **{"noise_"+k:v for k,v in distribution(dd,4).items()}))
    paired_summary = []
    for key, group in summarize(paired, ("first", "second", "ratio", "metric")):
        dd = [r["delta"] for r in group]
        direction = base.DIRECTIONS.get(key["metric"], "descriptive") if source == "primary_per_output" else "reference_diagnostic"
        item = dict(key, delta_direction="first minus second", nominal_preferred_direction=direction,
                    parent_deltas=dd, **distribution(dd,len(group)))
        if all(number(x) for x in dd):
            item.update(minimum_parent_id=min(group,key=lambda r:r["delta"])["parent_id"],
                        maximum_parent_id=max(group,key=lambda r:r["delta"])["parent_id"])
            leave = [statistics.fmean(dd[:i]+dd[i+1:]) for i in range(len(dd))]
            item.update(leave_one_parent_out_min=min(leave), leave_one_parent_out_max=max(leave))
            if direction in ("lower", "lower_uncertainty", "higher"):
                benefit = sorted((max(0,(-1 if direction.startswith("lower") else 1)*v) for v in dd),reverse=True)
                total = sum(benefit)
                item.update(top1_share_of_gross_improvement=benefit[0]/total if total else None,
                            top2_share_of_gross_improvement=sum(benefit[:2])/total if total else None)
        paired_summary.append(item)
    noise_summary = []
    for key, group in summarize(noise, ("first", "second", "ratio", "metric")):
        values = [r["delta"] for r in group]
        record = dict(key, delta_direction="first minus second",
                      **distribution(values, len(group)))
        if all(number(v) for v in values):
            record.update(minimum_request_id=min(group,key=lambda r:r["delta"])["request_id"],
                          maximum_request_id=max(group,key=lambda r:r["delta"])["request_id"])
        noise_summary.append(record)
    return dict(metric_catalog=catalog, per_parent_metrics=parents, model_summary=summary,
                paired_noise_metrics=noise, paired_noise_summary=noise_summary,
                paired_parent_metrics=paired, paired_parent_summary=paired_summary)


def reference_rows(main, course, requests):
    wanted = [q for q in requests if q["ratio"] == .2]
    qby = {q["request_id"]:q for q in requests}
    selected = [r for r in main["rows"] if r["label"] == BASELINE] + course["rows"]
    expected = {(label,rid) for label in LABELS for rid in qby}
    seen, rows = set(), []
    for r in selected:
        key = r["label"],r["request_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unregistered A20 reference cell")
        for field in ("parent_id","condition_id","candidate","ratio","condition_cohort","N","K"):
            if r[field] != qby[key[1]][field]:
                raise ValueError("Reference request identity differs: "+field)
        seen.add(key)
        if r["ratio"] != .2:
            continue  # Existing A40 reference NA rows are retained in the raw auxiliary output.
        result = r["result"]
        if result.get("label") != r["label"] or result.get("request_id") != r["request_id"]:
            raise ValueError("Reference inner result identity differs")
        rows.append(dict(r, metrics=result.get("metrics",{}), metric_status=result.get("metric_status",{}),
                         source_reference_status=result.get("status"), promotion_gate=False))
    if seen != expected or len(rows) != 96:
        raise ValueError("All 192 auxiliary cells including 96 A20 diagnostics required")
    return rows, wanted


def diversity_tables(conditions, pairs, rows, cohort):
    base.validate_diversity(conditions,pairs,rows)
    grouped = defaultdict(list)
    for r in pairs:
        grouped[r["label"],r["condition_id"]].append(r)
    checked = []
    for r in conditions:
        pp = grouped[r["label"],r["condition_id"]]
        valid = [float(p["D_free_pair"]) for p in pp if p.get("status")=="PASS" and p.get("D_free_pair") not in (None,"") and number(float(p["D_free_pair"]))]
        if int(r["available_pairs"]) != len(valid):
            raise ValueError("Diversity available-pair count differs")
        saved = float(r["D_free_mean"]) if r.get("D_free_mean") not in (None,"") else None
        if len(valid)==6:
            if not number(saved) or abs(statistics.fmean(valid)-saved)>max(1e-12,abs(saved)*1e-10):
                raise ValueError("Diversity six-pair mean differs")
        elif saved is not None:
            raise ValueError("Incomplete diversity must not use a survivor mean")
        checked.append(dict(r, ratio=float(r["ratio"]), parent_membership=cohort[r["parent_id"]], value=saved))
    summary = [dict(key, metric="D_free_mean", direction="variation_not_larger_is_better",
                    **distribution([r["value"] for r in group],len(group)))
               for key,group in summarize(checked,("label","ratio"))]
    by = {(r["label"],r["parent_id"],r["ratio"]):r for r in checked}
    parents = []
    for first,second in PAIRS:
        for (label,pid,ratio),a in by.items():
            if label!=first: continue
            b=by[second,pid,ratio];av,bv=a["value"],b["value"]
            parents.append(dict(first=first,second=second,parent_id=pid,ratio=ratio,
                condition_cohort=a["condition_cohort"],parent_membership=cohort[pid],
                first_D_free=av,second_D_free=bv,delta=av-bv if number(av) and number(bv) else None,
                relative_change=av/bv-1 if number(av) and number(bv) and bv!=0 else None,
                direction="variation_not_larger_is_better"))
    paired = [dict(key, direction="variation_not_larger_is_better",
                   **distribution([r["delta"] for r in group],len(group)))
              for key,group in summarize(parents,("first","second","ratio"))]
    return dict(diversity_per_condition=conditions,diversity_pairs=pairs,diversity_summary=summary,
                diversity_paired_parents=parents,diversity_paired_summary=paired)


def flattened(rows):
    return [dict({k:v for k,v in r.items() if k!="metrics"}, **{"metric."+k:v for k,v in r["metrics"].items()}) for r in rows]


def report(summary,pairs,metric_count):
    lines=["# Curriculum comparison from saved results","",
        "192 outputs: the two curriculum endpoints plus 64 existing R2 cells. No model or scoring calls; no automatic winner.",
        "Four noises are averaged per parent and parents are weighted equally. C20 and C40 remain separate. Original6/Extended2 at C20 is a parent-membership diagnostic, not a new evaluation cohort.",
        "All curricula use the registered H_ALL recipe. Reordering the same multiset changes update grouping and the whole-batch free denominator; this is a complete curriculum comparison, not constant per-scalar optimizer weighting. Historical condition-pool construction differences remain.",
        "Single training realizations on previously observed DEV parents; signs, tails and leave-one-parent-out summaries are descriptive, not statistical significance. Reference GT/S/E diagnostics are kept separately and never used to choose a winner.",""]
    for ratio in (.2,.4):
        for scope in SCOPES:
            lines += [f"## C={ratio:.0%}, {scope}","",
                      "| Model | "+" | ".join(label for _,label,_ in base.DISPLAY)+" |","|---|"+"---:|"*len(base.DISPLAY)]
            for label in LABELS:
                cells=[]
                for metric,_,scale in base.DISPLAY:
                    row=next((r for r in summary if (r["label"],r["ratio"],r["parent_stratum"],r["metric"])==(label,ratio,scope,metric)),{})
                    cells.append(base.format_value(row.get("mean"),scale))
                lines.append("| "+label+" | "+" | ".join(cells)+" |")
            lines.append("")
    lines += ["## Complete paired evidence","",
        f"All {metric_count} metric fields and statuses are retained, including nonnumeric or all-NA fields. Every numeric metric has model, parent, noise-pair, parent-pair, distribution, tail and applicable concentration/leave-one-parent-out tables. Missing values preserve their fixed denominators; no survivor mean replaces the main mean.",
        "paired_noise_metrics.csv and paired_parent_metrics.csv use first minus second, explicitly comparing C20_40_MIX minus R2, C40_20_MIX minus R2, and C20_40_MIX minus C40_20_MIX. No factorial coefficient from the earlier 2x2 study is reused.",
        "Diversity preserves all six noise pairs per condition. More variation is not automatically better. Geometry cost reductions do not certify coherent chairs, attachment or successful semantic use of C. Full actual visual review and completed training accounting remain independent prerequisites for any later decision.",""]
    return "\n".join(lines)


def analyze(out=base.DEFAULT_OUT):
    out=base.resolve(out).resolve()
    if out.name!=base.ROOT_NAME or out.parent.name!="experiments":
        raise ValueError("Wrong experiment analysis root")
    files={}
    for phase in ("main","course"):
        root=out/"phases"/phase
        files[phase+"_registration"]=root/"registration.json"
        for name in ("per_output.json","scoring_receipt.json","diversity_pairs.csv","diversity_per_condition.csv","reference_A20_per_output.json"):
            files[phase+"_"+name]=root/"evaluation"/name
    files.update(curriculum_decision=out/"registrations/CURRICULUM_DECISION.md",
                 curriculum_authorization=out/"registrations/curriculum_training.json",
                 protocol_clarifications=out/"PROTOCOL_CLARIFICATIONS.md",
                 condition_construction_audit=out/"audit/condition_construction_audit.json")
    missing=[str(p) for p in files.values() if not p.is_file()]
    if missing:
        return dict(status="PENDING",missing_files=missing,new_model_calls=0,new_scoring_calls=0,automatic_promotion=False)
    regs={p:read_json(files[p+"_registration"]) for p in ("main","course")}
    files["requests"]=base.resolve(regs["main"]["requests_path"])
    files["course_requests"]=base.resolve(regs["course"]["requests_path"])
    if digest(files["course_requests"])!=regs["course"]["requests_sha256"]:
        raise ValueError("Frozen course requests changed")
    if digest(files["requests"])!=regs["main"]["requests_sha256"]:
        raise ValueError("Frozen requests changed")
    auth=read_json(files["curriculum_authorization"])
    if auth.get("authorized_recipes")!=list(COURSES) or auth.get("hybrid_schedules")!={c:"all" for c in COURSES}:
        raise ValueError("Prospective H_ALL two-course authorization differs")
    provenance=dict(version=VERSION,input_files={k:dict(path=str(v),sha256=digest(v)) for k,v in files.items()},
        analysis_sources={str(Path(__file__).resolve()):digest(__file__),str(Path(base.__file__).resolve()):digest(base.__file__)},
        operations="Saved scalar tables only",new_model_calls=0,new_scoring_calls=0,baseline_new_scores=0)
    identity=base.canonical_hash(provenance);dest=out/"analysis/course"/(VERSION+"_"+identity[:16])
    receipt_path=dest/"analysis_receipt.json"
    if receipt_path.exists():
        old=read_json(receipt_path)
        if old["input_identity"]!=identity:raise ValueError("Analysis identity differs")
        for relative,h in old["output_sha256"].items():
            if digest(dest/relative)!=h:raise ValueError("Immutable analysis artifact changed")
        return old
    requests=read_json(files["requests"])["rows"]
    raw={p:read_json(files[p+"_per_output.json"])["rows"] for p in ("main","course")}
    rows=validate_panel(regs["main"],regs["course"],requests,raw["main"],raw["course"],
                        read_json(files["main_scoring_receipt.json"]),read_json(files["course_scoring_receipt.json"]))
    cohort=membership(requests)
    result=tables(rows,requests,cohort)
    main_auxiliary=read_json(files["main_reference_A20_per_output.json"])
    course_auxiliary=read_json(files["course_reference_A20_per_output.json"])
    refs,reference_requests=reference_rows(main_auxiliary,course_auxiliary,requests)
    raw_auxiliary=[r for r in main_auxiliary["rows"] if r["label"]==BASELINE]+course_auxiliary["rows"]
    reference=tables(refs,reference_requests,cohort,"reference_A20_diagnostic")
    conditions=[r for r in read_csv(files["main_diversity_per_condition.csv"]) if r["label"]==BASELINE]+read_csv(files["course_diversity_per_condition.csv"])
    pairs=[r for r in read_csv(files["main_diversity_pairs.csv"]) if r["label"]==BASELINE]+read_csv(files["course_diversity_pairs.csv"])
    diversity=diversity_tables(conditions,pairs,rows,cohort)
    for item in provenance["input_files"].values():
        if digest(item["path"])!=item["sha256"]:raise ValueError("Analysis input changed during read")
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"provenance.json",provenance)
    write_json(dest/"all_metrics.json",dict(rows=rows,preserved_outputs=192))
    write_csv(dest/"per_output.csv",flattened(rows))
    write_json(dest/"reference_A20_diagnostics.json",dict(rows=refs,raw_auxiliary_rows=raw_auxiliary,promotion_gate=False))
    write_csv(dest/"reference_A20_per_output.csv",flattened(refs))
    for name,rr in {**result,**diversity}.items():write_csv(dest/(name+".csv"),rr)
    for name,rr in reference.items():write_csv(dest/("reference_A20_"+name+".csv"),rr)
    write_json(dest/"condition_context.json",dict(
        curriculum_decision=files["curriculum_decision"].read_text(encoding="utf-8-sig"),
        protocol_clarifications=files["protocol_clarifications"].read_text(encoding="utf-8-sig"),
        condition_construction_audit=read_json(files["condition_construction_audit"]),
        parent_membership=cohort,training_completion_proven_by_analysis=False))
    write_json(dest/"decision_evidence.json",dict(model_summary=result["model_summary"],
        paired_parent_distributions=result["paired_parent_summary"],diversity=diversity["diversity_paired_summary"],
        automatic_promotion=False,overall_score=None,source_GT_promotion=False,
        visual_review="REQUIRES_ACTUAL_COMPARATIVE_GALLERY_INSPECTION",independent_training_repeat=False))
    (dest/"REPORT.md").write_text(report(result["model_summary"],result["paired_parent_summary"],len(result["metric_catalog"])),encoding="utf8")
    receipt=dict(status="COMPLETE",version=VERSION,input_identity=identity,analysis_dir=str(dest),registered_outputs=192,
        baseline_reused_outputs=64,course_outputs=128,reference_A20_outputs=len(refs),metrics_preserved=len(result["metric_catalog"]),
        new_model_calls=0,new_scoring_calls=0,automatic_promotion=False,source_GT_promotion=False,
        output_sha256={str(p.relative_to(dest)):digest(p) for p in sorted(dest.iterdir()) if p.is_file()})
    write_json(receipt_path,receipt)
    write_json(out/"analysis/course/latest.json",dict(analysis_dir=str(dest),receipt_path=str(receipt_path),receipt_sha256=digest(receipt_path),input_identity=identity))
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",default=base.DEFAULT_OUT)
    result=analyze(p.parse_args(argv).out);print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":raise SystemExit(main())




