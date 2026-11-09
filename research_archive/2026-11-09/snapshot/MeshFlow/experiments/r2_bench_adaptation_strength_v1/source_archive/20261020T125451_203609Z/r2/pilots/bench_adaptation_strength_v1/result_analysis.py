"""Summarize completed frozen CSV/actual-review evidence; never rescore."""
from __future__ import annotations
import argparse
import collections
import csv
import io
import json
import math
from pathlib import Path
import statistics
from .common import OUT, read, resolve, sha, write_new

DEST = OUT / "result_analysis"
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
FINAL = ("A_S1500", "B_S1500", "C_S1500")
CONTRASTS = (("B_MINUS_A", "A_S1500", "B_S1500"), ("C_MINUS_B", "B_S1500", "C_S1500"))
CURVES = {
    "A": ((0, "COMMON_S0"), (500, "A_S500"), (1500, "A_S1500")),
    "B": ((0, "COMMON_S0"), (500, "B_S500"), (1500, "B_S1500")),
    "C": ((0, "OFFICIAL_S0"), (500, "C_S500"), (1500, "C_S1500")),
}
# Exact existing score fields. Percent conversion is presentation only.
PERCENT = {
    "CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction",
    "ConflictCov_C", "free_minimum_angle_lt5_fraction",
    "free_only_reached_bulk_area_fraction",
    "fixed_parent_CF_illegal_affected_fraction", "fixed_parent_FF_illegal_affected_fraction",
}
METRICS = (
    "CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction",
    "ConflictCov_C", "free_minimum_angle_lt5_fraction",
    "CF_illegal_pair_count", "FF_illegal_pair_count",
    "actual_free_faces", "deleted_free_faces", "free_total_surface_area",
    "nonmanifold_edges", "exact_duplicate_faces",
    "hinge_diag_isolated_1face_edge_components",
    "C_F_has_valid_edge", "C_F_valid_edge_count", "C_F_valid_edge_total_length",
    "retained_seam_faces", "retained_bulk_faces", "retained_bulk_area",
    "free_only_reached_bulk_faces", "free_only_reached_bulk_area",
    "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive",
    "legal_seam_bulk_edge_count", "legal_seam_bulk_edge_length",
    "new_bridge_faces", "deleted_original_POST_faces",
    "fixed_parent_CF_illegal_affected_fraction", "fixed_parent_FF_illegal_affected_fraction",
)
SEAM_FIELDS = {
    "retained_seam_faces", "retained_bulk_faces", "retained_bulk_area",
    "free_only_reached_bulk_faces", "free_only_reached_bulk_area",
    "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive",
    "legal_seam_bulk_edge_count", "legal_seam_bulk_edge_length",
}
AREA_FIELDS = {"free_total_surface_area", "retained_bulk_area", "free_only_reached_bulk_area"}
LENGTH_FIELDS = {"C_F_valid_edge_total_length", "legal_seam_bulk_edge_length"}
GOOD_BAD = {
    "overall_shape": ("REASONABLE", "INCOMPLETE"),
    "C_use": ("PLAUSIBLE", "REPLACED_OR_IGNORED"),
    "support_retention": ("PRESENT", "MISSING_OR_DAMAGED"),
    "visible_fragment_or_gap": ("NO_OBVIOUS", "YES"),
    "layering_or_fold": ("NO_OBVIOUS", "YES"),
}
COSTS = ("CF_illegal_affected_free_fraction", "FF_illegal_affected_free_fraction",
         "ConflictCov_C", "free_minimum_angle_lt5_fraction")


def key(row, visual=False):
    return row["request_id"], row["model" if visual else "label"], row["stage"]


def number(value):
    if value is None or value == "" or value in ("None", "NA", "NaN", "nan"):
        return None
    if isinstance(value, bool) or value in ("True", "False"):
        return float(value is True or value == "True")
    n = float(value)
    return n if math.isfinite(n) else None


def unit(metric):
    return "percent" if metric in PERCENT else "model_units_squared" if metric in AREA_FIELDS else "model_units" if metric in LENGTH_FIELDS else "count"


def value(row, metric):
    result = number(row.get(metric))
    if result is not None and metric in PERCENT:
        result *= 100.
    return result


def load_inputs():
    paths = [OUT / "results_v2/per_output.csv", OUT / "results_v2/completion.json",
             OUT / "visual_review/validated_records.json", OUT / "visual_review/completion.json",
             OUT / "visual_review/protocol.json", OUT / "eval_manifest.json", OUT / "registration.json"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return dict(status="INCOMPLETE", missing=missing)
    scope, manifest = read(paths[-1]), read(paths[-2])
    reviews = read(paths[2])
    score_done, review_done = read(paths[1]), read(paths[3])
    if score_done["status"] != "COMPLETE" or reviews["status"] != "ACTUAL_AI_PROXY_REVIEW_COMPLETE" or review_done["status"] != "ACTUAL_AI_PROXY_REVIEW_COMPLETE":
        return dict(status="INCOMPLETE", scores=score_done["status"], visual=reviews["status"], review_completion=review_done["status"])
    hashes = {str(p): sha(p) for p in paths}
    expected_score_hash = next((h for p, h in score_done["files"].items() if resolve(p) == paths[0]), None)
    if expected_score_hash != hashes[str(paths[0])]:
        raise ValueError("Final CSV is not authenticated by its completion")
    with paths[0].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    absent = sorted(set(METRICS) - set(columns or []))
    if absent:
        return dict(status="INCOMPLETE", missing_existing_metric_fields=absent)
    process_ids = set(scope["process_request_ids"])
    expected = {(q["request_id"], label, stage)
                for label, spec in scope["evaluation_labels"].items()
                for q in manifest["requests"] if spec["subset"] == "full" or q["request_id"] in process_ids
                for stage in STAGES}
    keys, vkeys = [key(r) for r in rows], [key(r, True) for r in reviews["rows"]]
    if len(rows) != 672 or len(keys) != len(set(keys)) or set(keys) != expected:
        return dict(status="INCOMPLETE", scored_rows=len(rows), expected_rows=672, missing_scored_cells=sorted(expected-set(keys)))
    if len(vkeys) != 672 or len(vkeys) != len(set(vkeys)) or set(vkeys) != expected:
        return dict(status="INCOMPLETE", reviewed_rows=len(vkeys), expected_rows=672, missing_reviewed_cells=sorted(expected-set(vkeys)))
    protocol = read(paths[4])
    if set(protocol["axes"]) != set(GOOD_BAD):
        raise ValueError("Five-axis review protocol changed")
    qmap = {q["request_id"]: q for q in manifest["requests"]}
    for row in rows:
        q = qmap[row["request_id"]]
        if (row["parent_id"], row["condition_id"], row["split"], row["part_type"]) != tuple(q[k] for k in ("parent_id", "condition_id", "split", "part_type")):
            raise ValueError("CSV request metadata differs")
        if q["closed_C"] and any(number(row[m]) is not None for m in SEAM_FIELDS):
            raise ValueError("Closed-C seam field was converted to a numeric value")
    for row in reviews["rows"]:
        if row.get("actually_viewed") is not True or row.get("views_and_sections_checked") is not True:
            raise ValueError("Unviewed visual row")
        if any(row[a] not in protocol["axes"][a] for a in GOOD_BAD):
            raise ValueError("Invalid visual category")
    return dict(status="COMPLETE", rows=rows, visual=reviews["rows"], protocol=protocol,
                scope=scope, manifest=manifest, hashes=hashes)


def groups(rows):
    """Split totals plus explicit part and parent slices; no cross-split pool."""
    grouped = collections.defaultdict(list)
    for row in rows:
        for level, name in (("SPLIT", "ALL"), ("PART", row["part_type"]), ("PARENT", row["parent_id"])):
            grouped[row["split"], level, name].append(row)
    return sorted(grouped.items())


def numeric_summary(rows, common):
    output = []
    for (split, level, name), group in groups(rows):
        for metric in METRICS:
            valid = [(r, value(r, metric)) for r in group if value(r, metric) is not None]
            values = [v for _, v in valid]
            parents = collections.defaultdict(list)
            for r, v in valid:
                parents[r["parent_id"]].append(v)
            output.append(dict(**common, split=split, grouping=level, group=name, metric=metric,
                unit=unit(metric), output_n=len(group), valid_n=len(values), NA_n=len(group)-len(values),
                NA_reasons=dict(collections.Counter(r.get("seam_to_bulk_status", "MISSING_OR_NOT_APPLICABLE")
                    if metric in SEAM_FIELDS else r.get("I_status", "MISSING_OR_NOT_APPLICABLE") if metric == "ConflictCov_C"
                    else "MISSING_OR_NOT_APPLICABLE" for r in group if value(r, metric) is None)),
                output_mean=statistics.mean(values) if values else None,
                output_median=statistics.median(values) if values else None,
                output_sum=sum(values) if values and metric not in PERCENT else None,
                parent_macro_mean=statistics.mean(statistics.mean(v) for v in parents.values()) if parents else None,
                valid_parent_n=len(parents), value_status_counts=dict(collections.Counter(
                    r.get("I_status", "") for r, _ in valid)) if metric == "ConflictCov_C" else None))
    return output


def visual_summary(rows, common, axes):
    output = []
    for (split, level, name), group in groups(rows):
        for axis, choices in axes.items():
            counts = collections.Counter(r[axis] for r in group)
            for category in choices:
                output.append(dict(**common, split=split, grouping=level, group=name, axis=axis,
                    category=category, count=counts[category], cell_n=len(group),
                    reviewer_kind="AI_PROXY", blind=False, uncertain_is_not_an_ordinal_score=True))
    return output


def transition(axis, before, after):
    if before == after:
        return "UNCHANGED"
    if "UNCERTAIN" in (before, after):
        return "UNCERTAIN_TRANSITION"
    if "NA" in (before, after):
        return "NA_TRANSITION"
    good, bad = GOOD_BAD[axis]
    if (before, after) == (bad, good):
        return "DEFINITE_GAIN"
    if (before, after) == (good, bad):
        return "DEFINITE_LOSS"
    return "OTHER_CATEGORY_TRANSITION"


def paired(rows, visual):
    by, vb = {key(r): r for r in rows}, {key(r, True): r for r in visual}
    numbers, transitions = [], []
    ids = sorted({r["request_id"] for r in rows if r["label"] == FINAL[0]})
    for contrast, ref, candidate in CONTRASTS:
        for rid in ids:
            for stage in STAGES:
                a, b = by[rid, ref, stage], by[rid, candidate, stage]
                base = dict(contrast=contrast, reference=ref, candidate=candidate, stage=stage,
                    **{k: a[k] for k in ("request_id", "condition_id", "parent_id", "noise_index", "split", "part_type")})
                for metric in METRICS:
                    x, y = value(a, metric), value(b, metric)
                    delta = y-x if x is not None and y is not None else None
                    numbers.append(dict(**base, metric=metric, unit="percentage_points" if metric in PERCENT else unit(metric),
                        reference_value=x, candidate_value=y, delta=delta,
                        status="PAIRED" if delta is not None else "NA_PAIRED_VALUE_UNAVAILABLE"))
                va, vv = vb[rid, ref, stage], vb[rid, candidate, stage]
                for axis in GOOD_BAD:
                    transitions.append(dict(**base, axis=axis, reference_category=va[axis],
                        candidate_category=vv[axis], transition=transition(axis, va[axis], vv[axis]),
                        reference_notes=va["notes"], candidate_notes=vv["notes"]))
    return numbers, transitions


def delta_summaries(pairs):
    result = []
    for contrast, _, _ in CONTRASTS:
        for stage in STAGES:
            subset = [r for r in pairs if r["contrast"] == contrast and r["stage"] == stage]
            for (split, level, name), group in groups(subset):
                for metric in METRICS:
                    rr = [r for r in group if r["metric"] == metric]
                    valid = [r for r in rr if r["delta"] is not None]
                    byparent = collections.defaultdict(list)
                    for r in valid:
                        byparent[r["parent_id"]].append(r["delta"])
                    result.append(dict(contrast=contrast, stage=stage, split=split, grouping=level,
                        group=name, metric=metric, unit=rr[0]["unit"], paired_n=len(valid), NA_n=len(rr)-len(valid),
                        mean_delta=statistics.mean(r["delta"] for r in valid) if valid else None,
                        parent_macro_delta=statistics.mean(statistics.mean(v) for v in byparent.values()) if byparent else None,
                        min_delta=min((r["delta"] for r in valid), default=None),
                        max_delta=max((r["delta"] for r in valid), default=None),
                        negative_delta_n=sum(r["delta"] < 0 for r in valid), zero_delta_n=sum(r["delta"] == 0 for r in valid),
                        positive_delta_n=sum(r["delta"] > 0 for r in valid)))
    return result


def development_evidence(transitions, pairs, reference_pp):
    conditions, summary, costs = [], [], []
    for contrast, _, _ in CONTRASTS:
        for stage in STAGES:
            vv = [r for r in transitions if r["contrast"] == contrast and r["stage"] == stage and r["split"] == "DEV"]
            bycondition = collections.defaultdict(list)
            for r in vv:
                bycondition[r["condition_id"]].append(r)
            for cid, rr in sorted(bycondition.items()):
                peraxis = {axis: [r for r in rr if r["axis"] == axis] for axis in GOOD_BAD}
                if any(len(v) != 2 for v in peraxis.values()):
                    raise ValueError("DEV condition must have both frozen noises for every visual axis")
                adopts = peraxis["C_use"]
                shape = peraxis["overall_shape"]
                flags = dict(
                    C_use_definite_gain_both_noise=all(r["transition"] == "DEFINITE_GAIN" for r in adopts),
                    overall_shape_definite_gain_both_noise=all(r["transition"] == "DEFINITE_GAIN" for r in shape),
                    either_axis_definite_gain_each_noise=all(any(r["transition"] == "DEFINITE_GAIN"
                        for r in adopts+shape if r["noise_index"] == n) for n in {r["noise_index"] for r in adopts}),
                    any_C_use_or_shape_definite_loss=any(r["transition"] == "DEFINITE_LOSS" for r in adopts+shape),
                    any_C_use_or_shape_uncertain_transition=any(r["transition"] == "UNCERTAIN_TRANSITION" for r in adopts+shape),
                    support_definite_loss_any_noise=any(r["transition"] == "DEFINITE_LOSS" for r in peraxis["support_retention"]))
                conditions.append(dict(contrast=contrast, stage=stage, condition_id=cid, parent_id=rr[0]["parent_id"],
                    part_type=rr[0]["part_type"], noise_n=2, **flags, interpretation="Evidence only; no automatic promotion"))
            current = [r for r in conditions if r["contrast"] == contrast and r["stage"] == stage]
            summary.append(dict(contrast=contrast, stage=stage, registered_parent_reference=2,
                parents_with_C_use_gain_both_noise=len({r["parent_id"] for r in current if r["C_use_definite_gain_both_noise"]}),
                parents_with_shape_gain_both_noise=len({r["parent_id"] for r in current if r["overall_shape_definite_gain_both_noise"]}),
                parents_with_either_gain_each_noise_and_no_definite_loss=len({r["parent_id"] for r in current
                    if r["either_axis_definite_gain_each_noise"] and not r["any_C_use_or_shape_definite_loss"]}),
                axis_transition_counts={axis: dict(collections.Counter(r["transition"] for r in vv if r["axis"] == axis)) for axis in GOOD_BAD},
                automatic_winner_selection=False, uncertain_not_treated_as_intermediate_score=True))
            for metric in COSTS:
                nn = [r for r in pairs if r["contrast"] == contrast and r["stage"] == stage and r["split"] == "DEV"
                      and r["metric"] == metric and r["delta"] is not None]
                costs.append(dict(contrast=contrast, stage=stage, metric=metric, unit="percentage_points",
                    paired_n=len(nn), mean_delta=statistics.mean(r["delta"] for r in nn) if nn else None,
                    reference_pp=reference_pp, requests_above_reference=[r["request_id"] for r in nn if r["delta"] > reference_pp],
                    worst_delta=max((r["delta"] for r in nn), default=None),
                    interpretation="Registered engineering cost reference only; severe visual losses separately reviewed"))
    return conditions, summary, costs


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    text = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text.replace("\r\n", "\n"):
            raise ValueError("Refusing to replace completed analysis: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(text)


def run(check_only=False):
    data = load_inputs()
    if data["status"] != "COMPLETE" or check_only:
        return {k: v for k, v in data.items() if k not in ("rows", "visual", "scope", "manifest", "protocol")}
    done = DEST / "completion.json"
    if done.exists():
        result = read(done)
        if result["inputs"] != data["hashes"] or result["analysis_source_sha256"] != sha(__file__):
            raise ValueError("Completed analysis inputs/code changed")
        for path, digest in result["files"].items():
            if sha(path) != digest:
                raise ValueError("Completed analysis artifact changed")
        return result
    rows, visual, axes = data["rows"], data["visual"], data["protocol"]["axes"]
    numeric, visuals, curve, vcurve = [], [], [], []
    for label in FINAL:
        for stage in STAGES:
            common = dict(label=label, stage=stage, panel="FULL48")
            numeric += numeric_summary([r for r in rows if r["label"] == label and r["stage"] == stage], common)
            visuals += visual_summary([r for r in visual if r["model"] == label and r["stage"] == stage], common, axes)
    process = set(data["scope"]["process_request_ids"])
    for arm, points in CURVES.items():
        for step, label in points:
            for stage in STAGES:
                common = dict(arm=arm, local_step=step, source_label=label, stage=stage, panel="SAME_PROCESS16",
                              shared_COMMON_S0=step == 0 and arm in ("A", "B"))
                selected = [r for r in rows if r["label"] == label and r["stage"] == stage and r["request_id"] in process]
                assert len(selected) == 16
                curve += numeric_summary(selected, common)
                vcurve += visual_summary([r for r in visual if r["model"] == label and r["stage"] == stage and r["request_id"] in process], common, axes)
    pairs, transitions = paired(rows, visual)
    conditions, evidence, costs = development_evidence(transitions, pairs, data["scope"]["progression"]["cost_reference_pp"])
    tables = dict(key_metrics_final=numeric, key_metrics_paired_per_output=pairs,
                  key_metrics_paired_summary=delta_summaries(pairs), generation_curve_same16=curve,
                  visual_five_axes_final=visuals, visual_five_axes_curve_same16=vcurve,
                  visual_transitions=transitions, dev_two_noise_condition_evidence=conditions,
                  dev_parent_evidence=evidence, dev_cost_reference=costs)
    for name, table in tables.items():
        write_csv(DEST / (name + ".csv"), table)
    policy = dict(status="COMPLETE", input_rows=672, actual_review_rows=672, metrics=list(METRICS),
        primary_contrasts=["B_MINUS_A", "C_MINUS_B"], collector="results_v2",
        panels=dict(terminal="FULL48", curve="SAME_PROCESS16_ONLY"),
        units="Original fractions displayed as percent; paired changes in percentage points. Length/area remain model units.",
        aggregation="Output means plus parent macro means with metric-specific valid denominators. No NA converted to zero.",
        visual_rule="Five independent nominal axes. Only explicit bad-to-good or good-to-bad transitions are definite. UNCERTAIN/NA separately counted.",
        threshold_rule="Two-DEV-parent and one-pp references show evidence, never choose a winner automatically.",
        inputs=data["hashes"], analysis_source_sha256=sha(__file__), model_calls=0, rescoring_calls=0,
        files={str(DEST / (name + ".csv")): sha(DEST / (name + ".csv")) for name in tables})
    write_new(done, policy)
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    result = run(args.check_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
