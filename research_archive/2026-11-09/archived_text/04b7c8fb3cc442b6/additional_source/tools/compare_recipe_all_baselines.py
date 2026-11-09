#!/usr/bin/env python3
"""Combine completed 512 saved primary metric rows; no scoring or model calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_all_baselines_course", Path(__file__).with_name("analyze_recipe_curriculum_results.py"))
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
base = course.base
VERSION = "recipe_all_baselines_table_comparison_v1"
LABELS = (*base.LABELS, *course.COURSES)
BASELINES = ("START5000", "R1_40_H_ALL", "HIST_FM6000")
PAIRS = tuple((c, b) for c in course.COURSES for b in BASELINES)
SCOPES = {"ALL": 8, "ORIGINAL6": 6, "EXTENDED2": 2}


def validate_rows(main_registration, course_registration, requests, main_rows, course_rows, main_receipt, course_receipt):
    # Reuse frozen validation, without changing its fixed three/six-label globals.
    course.validate_panel(main_registration, course_registration, requests, main_rows,
                          course_rows, main_receipt, course_receipt)
    rows = main_rows + course_rows
    by = {(r["label"], r["request_id"]): r for r in rows}
    expected = {(label, q["request_id"]) for label in LABELS for q in requests}
    if len(rows) != 512 or len(by) != 512 or set(by) != expected:
        raise ValueError("Complete unique 512-cell eight-label panel required")
    for q in requests:
        reference = by["R2_MIX_H_ALL", q["request_id"]]
        for label in LABELS:
            row = by[label, q["request_id"]]
            for field in ("query_sha256", "registered_free_faces", "registered_C_queries",
                          "query_weight_denominator", "L0"):
                if row[field] != reference[field]:
                    raise ValueError("Eight-label query/denominator binding differs: " + field)
            if row["identity"]["input_sha256"] != reference["identity"]["input_sha256"]:
                raise ValueError("Eight-label actual C/noise binding differs")
    return sorted(rows, key=lambda r: (LABELS.index(r["label"]), r["request_id"]))


def direction_counts(deltas, direction):
    valid = [v for v in deltas if base.is_number(v)]
    directional = direction in ("lower", "lower_uncertainty", "higher")
    factor = -1 if direction.startswith("lower") else 1
    return dict(
        expected_parents=len(deltas), comparable_parents=len(valid),
        missing_parents=len(deltas)-len(valid),
        improved_parents=sum(factor*v > 0 for v in valid) if directional else None,
        worsened_parents=sum(factor*v < 0 for v in valid) if directional else None,
        tied_parents=sum(v == 0 for v in valid) if directional else None,
        improvement_applicability="metric_direction_only" if directional else "NA_NO_PREFERRED_DIRECTION",
        quality_certificate=False, counts_use_exact_sign_no_new_threshold=True)


def aggregate(rows, requests):
    cohort = course.membership(requests)
    catalog = base.metric_catalog(rows)
    cby = {r["metric"]: r for r in catalog}
    metrics = sorted(cby)
    groups = defaultdict(list)
    for q in requests:
        groups[q["parent_id"], q["ratio"]].append(q)
    by = {(r["label"], r["request_id"]): r for r in rows}
    parent_rows = []
    for label in LABELS:
        for (pid, ratio), qs in sorted(groups.items()):
            qs = sorted(qs, key=lambda q: q["candidate"])
            for metric in metrics:
                values = [by[label, q["request_id"]]["metrics"].get(metric) for q in qs]
                parent_rows.append(dict(
                    label=label, parent_id=pid, ratio=ratio,
                    condition_cohort=qs[0]["condition_cohort"], parent_membership=cohort[pid],
                    metric=metric, direction=cby[metric]["direction"],
                    numeric_summary=cby[metric]["numeric_summary"], noises=list(base.NOISES),
                    noise_values=values, **base.distribution(values, 4)))
    summary = []
    for key, group in course.summarize(parent_rows, ("label", "ratio", "metric")):
        expected = SCOPES[key["parent_stratum"]]
        if len(group) != expected:
            raise ValueError("Parent denominator differs")
        summary.append(dict(key, direction=cby[key["metric"]]["direction"],
            numeric_summary=cby[key["metric"]]["numeric_summary"],
            aggregation="four noises then equal parents; no survivor primary mean",
            **base.distribution([r["mean"] for r in group], expected)))
    pby = {(r["label"], r["parent_id"], r["ratio"], r["metric"]): r for r in parent_rows}
    noise_pairs, parent_pairs = [], []
    for first, second in PAIRS:
        for q in requests:
            a, b = by[first, q["request_id"]], by[second, q["request_id"]]
            for metric in metrics:
                av, bv = a["metrics"].get(metric), b["metrics"].get(metric)
                noise_pairs.append(dict(
                    first=first, second=second, delta_direction="first minus second",
                    request_id=q["request_id"], parent_id=q["parent_id"], ratio=q["ratio"],
                    candidate=q["candidate"], condition_cohort=q["condition_cohort"],
                    parent_membership=cohort[q["parent_id"]], metric=metric,
                    direction=cby[metric]["direction"], numeric_summary=cby[metric]["numeric_summary"],
                    first_value=av, second_value=bv,
                    delta=av-bv if base.is_number(av) and base.is_number(bv) else None))
        for pid, ratio in sorted(groups):
            for metric in metrics:
                a, b = pby[first, pid, ratio, metric], pby[second, pid, ratio, metric]
                values = [av-bv if base.is_number(av) and base.is_number(bv) else None
                          for av, bv in zip(a["noise_values"], b["noise_values"])]
                parent_pairs.append(dict(
                    first=first, second=second, delta_direction="first minus second",
                    parent_id=pid, ratio=ratio, condition_cohort=a["condition_cohort"],
                    parent_membership=cohort[pid], metric=metric,
                    direction=cby[metric]["direction"], numeric_summary=cby[metric]["numeric_summary"],
                    first_mean=a["mean"], second_mean=b["mean"],
                    delta=a["mean"]-b["mean"] if base.is_number(a["mean"]) and base.is_number(b["mean"]) else None,
                    noise_deltas=values, **{"noise_"+k:v for k,v in base.distribution(values,4).items()}))
    pair_summary = []
    for key, group in course.summarize(parent_pairs, ("first", "second", "ratio", "metric")):
        expected = SCOPES[key["parent_stratum"]]
        if len(group) != expected:
            raise ValueError("Paired parent denominator differs")
        values = [r["delta"] for r in group]
        direction = cby[key["metric"]]["direction"]
        pair_summary.append(dict(key, delta_direction="first minus second",
            direction=direction, numeric_summary=cby[key["metric"]]["numeric_summary"],
            parent_deltas=values, **base.distribution(values,expected),
            **direction_counts(values,direction)))
    return dict(metric_catalog=catalog, per_parent_metrics=parent_rows, model_summary=summary,
        paired_noise_metrics=noise_pairs, paired_parent_metrics=parent_pairs,
        paired_parent_summary=pair_summary)


def analyze(out=base.DEFAULT_OUT):
    out = base.resolve(out).resolve()
    if out.name != base.ROOT_NAME or out.parent.name != "experiments":
        raise ValueError("Wrong experiment root")
    files = {}
    for phase in ("main", "course"):
        root = out/"phases"/phase
        files[phase+"_registration"] = root/"registration.json"
        files[phase+"_per_output"] = root/"evaluation/per_output.json"
        files[phase+"_receipt"] = root/"evaluation/scoring_receipt.json"
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        return dict(status="PENDING", missing_files=missing, new_model_calls=0,
                    new_scoring_calls=0, automatic_selection=False)
    regs = {p:base.read_json(files[p+"_registration"]) for p in ("main","course")}
    for phase in regs:
        files[phase+"_requests"] = base.resolve(regs[phase]["requests_path"])
        if not files[phase+"_requests"].is_file():
            return dict(status="PENDING", missing_files=[str(files[phase+"_requests"])],
                        new_model_calls=0, new_scoring_calls=0, automatic_selection=False)
        if base.digest(files[phase+"_requests"]) != regs[phase]["requests_sha256"]:
            raise ValueError("Actual requests file differs: "+phase)
    provenance = dict(version=VERSION, input_files={k:dict(path=str(p),sha256=base.digest(p)) for k,p in files.items()},
        source_files={str(Path(p).resolve()):base.digest(p) for p in (__file__,course.__file__,base.__file__)},
        labels=list(LABELS), pairs=[list(p) for p in PAIRS],
        operations="Read saved primary scalar tables only; no scoring, geometry or checkpoint read")
    identity = base.canonical_hash(provenance)
    dest = out/"analysis/all_baselines"/(VERSION+"_"+identity[:16])
    receipt_path = dest/"analysis_receipt.json"
    if receipt_path.exists():
        old = base.read_json(receipt_path)
        if old["input_identity"] != identity:
            raise ValueError("Comparison identity differs")
        for relative, sha in old["output_sha256"].items():
            if base.digest(dest/relative) != sha:
                raise ValueError("Immutable comparison artifact changed")
        return old
    requests = base.read_json(files["main_requests"])["rows"]
    main_rows = base.read_json(files["main_per_output"])["rows"]
    course_rows = base.read_json(files["course_per_output"])["rows"]
    rows = validate_rows(regs["main"],regs["course"],requests,main_rows,course_rows,
        base.read_json(files["main_receipt"]),base.read_json(files["course_receipt"]))
    result = aggregate(rows,requests)
    for item in provenance["input_files"].values():
        if base.digest(item["path"]) != item["sha256"]:
            raise ValueError("Input changed while reading")
    dest.mkdir(parents=True,exist_ok=True)
    base.write_json(dest/"provenance.json",provenance)
    base.write_json(dest/"all_metrics.json",dict(rows=rows,preserved_outputs=512))
    base.write_csv(dest/"per_output.csv",course.flattened(rows))
    for name, values in result.items():
        base.write_csv(dest/(name+".csv"),values)
    (dest/"README.md").write_text(
        "# Eight-label saved-table comparison\n\n"
        "512 original primary score rows: main384 plus course128 on the identical64 requests. "
        "No scoring, model, geometry or checkpoint reads. All scalar fields, statuses, NA values "
        "and original nested rows remain in all_metrics.json. No source table is changed.\n\n"
        "Each mean uses four noises per parent, followed by equal parent weights. C20 and C40 "
        "each retain ALL8, ORIGINAL6 and EXTENDED2; at C20 these denote parent membership. "
        "Missing numeric values keep their original denominators; the main mean stays blank. "
        "Nonnumeric/all-NA fields retain rows and are explicitly numeric_summary=False.\n\n"
        "The only six contrasts are each curriculum minus START5000, R1_40_H_ALL and HIST_FM6000. "
        "Improved/worsened/tied parent counts use the existing individual metric direction and "
        "exact sign, with no new threshold or composite score. Descriptive directions have NA "
        "improvement counts. lower_uncertainty concerns uncertainty, not certified quality. "
        "Counts and means from single training realizations on observed DEV parents are "
        "descriptive; they establish neither significance nor automatic selection.\n\n"
        "Reference GT/S/E and diversity retain their separate original analysis artifacts; "
        "this supplement compares the full primary metric table only.\n",encoding="utf8")
    receipt = dict(status="COMPLETE",version=VERSION,input_identity=identity,analysis_dir=str(dest),
        preserved_outputs=512,main_reused_outputs=384,course_reused_outputs=128,labels=list(LABELS),
        contrasts=[list(p) for p in PAIRS],metric_count=len(result["metric_catalog"]),
        new_model_calls=0,new_scoring_calls=0,automatic_selection=False,composite_score=None,
        output_sha256={str(p.relative_to(dest)):base.digest(p) for p in sorted(dest.iterdir()) if p.is_file()})
    base.write_json(receipt_path,receipt)
    base.write_json(out/"analysis/all_baselines/latest.json",dict(
        analysis_dir=str(dest),receipt_path=str(receipt_path),receipt_sha256=base.digest(receipt_path)))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default=base.DEFAULT_OUT)
    result = analyze(parser.parse_args(argv).out)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":
    raise SystemExit(main())
