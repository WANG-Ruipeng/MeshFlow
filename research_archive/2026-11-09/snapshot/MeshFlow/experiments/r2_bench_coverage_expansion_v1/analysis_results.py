"""Read authenticated completed CSVs only; no geometry, scoring, torch or model work.

Run after results/completion.json exists. Missing/incomplete collection returns
WAIT without writing analysis files. Completed outputs are immutable and can be
revalidated. Open-C interface flags receive explicitly named analysis scopes;
all original fields and their unscoped aggregates remain in the source CSVs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RESULTS = OUT / "results"
DEST = OUT / "analysis"
LABELS = ("A_S1500", "COVERAGE_S1500")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
SPLITS = ("TRAIN", "DEV")
VERSION = "BENCH_COVERAGE_CSV_ANALYSIS_V1"
EPSILON = 1e-12
CSV_NAMES = (
    "per_output", "per_condition", "per_parent", "parent_macro_summary",
    "paired_per_output", "paired_per_condition", "paired_per_parent",
)

# Every field comes from a frozen row. No new detector or geometric proxy is added.
FIELDS = (
    "CF_illegal_affected_free_fraction", "CF_illegal_pair_count",
    "CF_proper_affected_free_fraction", "CF_proper_pair_count", "CF_coplanar_pair_count",
    "FF_illegal_affected_free_fraction", "FF_illegal_pair_count",
    "FF_proper_affected_free_fraction", "FF_proper_pair_count", "FF_coplanar_pair_count",
    "ConflictCov_C", "ConflictCov_C_upper", "ConflictCov_C_cross", "ConflictCov_C_dup",
    "ConflictCov_C_uncertain_fraction",
    "actual_free_faces", "deleted_free_faces", "deleted_original_POST_faces",
    "free_total_surface_area", "retained_bulk_area", "retained_bulk_faces",
    "free_minimum_angle_lt5_fraction", "free_angle_p05_degrees", "free_q_gt10_fraction",
    "free_near_degenerate_fraction", "free_exact_zero_area_fraction",
    "free_exact_duplicate_excess_fraction", "nonmanifold_edges", "boundary_edges",
    "C_F_nonmanifold_edge_count", "C_F_valid_edge_count", "C_F_valid_edge_total_length",
    "C_F_valid_opposite_winding_edge_count", "C_F_has_valid_edge",
    "C_F_has_opposite_winding_edge", "free_only_reached_bulk_area_fraction",
    "free_only_reached_bulk_area", "bulk_reach_ge_0_30_descriptive",
    "legal_seam_bulk_edge_count", "legal_seam_bulk_edge_length",
    "seed_connected_legal_seam_bulk_edge_count", "seed_connected_legal_seam_bulk_edge_length",
    "opposite_winding_reached_free_area_fraction",
    "hinge_diag_isolated_1face_edge_components",
    "hinge_diag_isolated_1face_unfiltered_edge_components",
    "hinge_diag_fully_C_disconnected_free_area_fraction",
    "hinge_diag_vertex_attached_only_free_area_fraction",
    "hinge_diag_vertex_attached_only_free_area_numerator",
    "hinge_diag_free_area_denominator",
    "illegal_pairs_bulk-bulk", "illegal_pairs_seam-bulk", "illegal_pairs_seam-seam",
    "illegal_pairs_C-bulk", "illegal_pairs_C-seam",
    "fixed_parent_CF_illegal_affected_fraction", "fixed_parent_FF_illegal_affected_fraction",
    "fixed_parent_CF_illegal_affected_count", "fixed_parent_FF_illegal_affected_count",
    "fixed_parent_CF_illegal_unique_pair_count", "fixed_parent_FF_illegal_unique_pair_count",
    "fixed_parent_free_face_denominator", "fixed_parent_minimum_angle_lt5_fraction",
    "new_bridge_faces", "bridge_CF_illegal_pair_count", "bridge_FF_illegal_pair_count",
    "A_VMR__raw_tau_1e-6", "PMR__raw_tau_1e-6", "A_VMR__L_tau_0.001", "PMR__L_tau_0.001",
    "free_face_denominator", "CF_tested_pair_denominator", "FF_tested_pair_denominator",
    "C_preserved_exact",
)
OPEN_FLAGS = frozenset(("C_F_has_valid_edge", "C_F_has_opposite_winding_edge",
                        "bulk_reach_ge_0_30_descriptive"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    text = str(path).replace("\\", "/")
    for prefix in ("/mnt/e/MeshFlow/", "E:/MeshFlow/"):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def csvread(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def boolean(value):
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError("Expected an explicit boolean, got " + repr(value))


def scalar(value):
    if value is None or str(value).strip() == "":
        return None
    if str(value).lower() == "true":
        return 1.
    if str(value).lower() == "false":
        return 0.
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def metric_value(row, field):
    if field in OPEN_FLAGS and boolean(row["closed_C"]):
        return None
    return scalar(row.get(field))


def avg(values):
    valid = [v for v in values if v is not None]
    return statistics.mean(valid) if valid else None


def grouping(rows, fields):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return groups


def parent_values(rows, getter):
    # Keep exactly the original noise -> condition -> parent hierarchy.
    cond = {identity: avg([getter(r) for r in rr])
            for identity, rr in grouping(rows, ("parent_id", "condition_id")).items()}
    parents = collections.defaultdict(list)
    for (pid, _), value in cond.items():
        parents[pid].append(value)
    return {pid: avg(vv) for pid, vv in parents.items()}, cond


def definitions():
    inherited = read(ROOT / "r2/tools/hinge_opposite_v1/metric_policy.json")["directions"]
    result = {}
    for field in FIELDS:
        if field in inherited:
            direction = inherited[field]
        elif field.startswith("fixed_parent_") and field != "fixed_parent_free_face_denominator":
            direction = "lower_descriptive"
        elif field.startswith("bridge_") and "illegal" in field:
            direction = "lower_descriptive"
        else:
            direction = "context_only"
        unit = ("ratio_0_1" if "fraction" in field or field.startswith(("ConflictCov_C", "A_VMR__", "PMR__"))
                else "binary" if field in OPEN_FLAGS or field == "C_preserved_exact"
                else "degrees" if "degrees" in field
                else "model_length" if "length" in field
                else "model_area" if "area" in field
                else "count")
        result[field] = dict(source_field=field,
            analysis_id=field + ("__OPEN_C_SCOPE" if field in OPEN_FLAGS else ""),
            scope="OPEN_C_ONLY_CLOSED_NA" if field in OPEN_FLAGS else "ORIGINAL_STORED_APPLICABILITY",
            direction=direction, unit=unit,
            percent_display_multiplier=100 if unit in ("ratio_0_1", "binary") else 1)
    return result


def load_inputs():
    completion_path = RESULTS / "completion.json"
    required = [completion_path, OUT / "registration.json", OUT / "eval_manifest.json",
                *[RESULTS / (name + ".csv") for name in CSV_NAMES]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return dict(status="WAIT", reason="Completed collection not yet available", missing=missing,
                    model_calls=0, scoring_calls=0, written_outputs=0)
    completion = read(completion_path)
    if completion.get("status") != "COMPLETE":
        return dict(status="WAIT", reason="Collector has not marked completion",
                    collection_status=completion.get("status"), model_calls=0, scoring_calls=0,
                    written_outputs=0)
    if completion["rows"] != 288 or completion["new_outputs"] != 48 or completion["reused_outputs"] != 48:
        raise ValueError("Unexpected collection scope")
    expected = {resolve(p).resolve(): h for p, h in completion["files"].items()}
    sources, tables = {}, {}
    for path in required:
        digest = sha(path)
        if path.suffix == ".csv":
            if expected.get(path.resolve()) != digest:
                raise ValueError("Unauthenticated source CSV: " + str(path))
            fields, rows = csvread(path)
            tables[path.stem] = rows
            sources[str(path)] = dict(sha256=digest, rows=len(rows), all_original_columns=fields)
        else:
            sources[str(path)] = dict(sha256=digest)
    for key, filename in (("registration_sha256", "registration.json"),
                          ("eval_manifest_sha256", "eval_manifest.json")):
        if completion[key] != sha(OUT / filename):
            raise ValueError("Frozen experiment identity differs: " + key)
    rows = tables["per_output"]
    identities = {(r["label"], r["stage"], r["request_id"]) for r in rows}
    if len(rows) != 288 or len(identities) != 288:
        raise ValueError("Duplicated/missing model-stage-request rows")
    qq = read(OUT / "eval_manifest.json")["requests"]
    if len(qq) != 48:
        raise ValueError("48 original requests required")
    expected_ids = {(label, stage, q["request_id"]) for label in LABELS for stage in STAGES for q in qq}
    if identities != expected_ids:
        raise ValueError("Analysis would use unregistered identities")
    qby = {q["request_id"]: q for q in qq}
    for r in rows:
        q = qby[r["request_id"]]
        for key in ("condition_id", "parent_id", "split"):
            if r[key] != q[key]:
                raise ValueError("Source identity changed: " + key)
        if boolean(r["closed_C"]) != q["closed_C"]:
            raise ValueError("Closed-C applicability differs")
        if set(FIELDS) - set(r):
            raise ValueError("Frozen key fields missing: " + repr(sorted(set(FIELDS) - set(r))))
    return dict(status="READY", tables=tables, sources=sources, completion=completion)


def make_analysis(data):
    rows = data["tables"]["per_output"]
    specs = definitions()
    key_rows = []
    for (label, stage, split), group in sorted(grouping(rows, ("label", "stage", "split")).items()):
        for field, spec in specs.items():
            pv, cv = parent_values(group, lambda r: metric_value(r, field))
            observations = [(r["request_id"], metric_value(r, field)) for r in group]
            valid = [(rid, value) for rid, value in observations if value is not None]
            missing = [r for r in group if metric_value(r, field) is None]
            statuses = collections.Counter(r.get("seam_to_bulk_status", "") for r in missing)
            key_rows.append(dict(label=label, stage=stage, split=split, **spec,
                parent_macro_mean=avg(list(pv.values())),
                per_output_mean_descriptive=avg([v for _, v in valid]),
                valid_output_n=len(valid), planned_output_n=len(group),
                valid_condition_n=sum(v is not None for v in cv.values()),
                planned_condition_n=len(cv), valid_parent_n=sum(v is not None for v in pv.values()),
                planned_parent_n=len(pv), excluded_closed_output_n=sum(
                    field in OPEN_FLAGS and boolean(r["closed_C"]) for r in group),
                missing_or_NA_output_n=len(missing), missing_seam_status_counts=dict(statuses),
                output_min=min((v for _, v in valid), default=None),
                output_max=max((v for _, v in valid), default=None),
                output_min_request_ids=[rid for rid, v in valid if v == min(x for _, x in valid)],
                output_max_request_ids=[rid for rid, v in valid if v == max(x for _, x in valid)]))
    old_parent_pairs = {(r["stage"], r["split"], r["parent_id"]): r
                        for r in data["tables"]["paired_per_parent"]}
    paired_parents, paired_summaries = [], []
    lookup = {(r["label"], r["stage"], r["request_id"]): r for r in rows}
    for stage in STAGES:
        for split in SPLITS:
            refs = [r for r in rows if r["label"] == LABELS[0] and r["stage"] == stage and r["split"] == split]
            for field, spec in specs.items():
                def values(r):
                    candidate = lookup[LABELS[1], stage, r["request_id"]]
                    return metric_value(r, field), metric_value(candidate, field)
                def difference(r):
                    a, b = values(r)
                    return b-a if a is not None and b is not None else None
                deltas, _ = parent_values(refs, difference)
                ref_p, _ = parent_values(refs, lambda r: values(r)[0] if difference(r) is not None else None)
                new_p, _ = parent_values(refs, lambda r: values(r)[1] if difference(r) is not None else None)
                local = []
                for pid, delta in sorted(deltas.items()):
                    parent_rows = [r for r in refs if r["parent_id"] == pid]
                    source = old_parent_pairs[stage, split, pid]
                    stored = scalar(source.get(field + "__delta"))
                    if field not in OPEN_FLAGS and field != "C_preserved_exact":
                        if (delta is None) != (stored is None) or (delta is not None and abs(delta-stored) > EPSILON):
                            raise ValueError("Independent CSV recomputation differs from frozen parent pair: " + field)
                    sign = None if delta is None else ("positive" if delta > EPSILON else "negative" if delta < -EPSILON else "tie")
                    direction = spec["direction"]
                    assessment = ("NA" if sign is None else "CONTEXT_ONLY" if direction == "context_only"
                                  else "TIE" if sign == "tie" else "IMPROVED" if
                                  ((direction == "lower_descriptive" and sign == "negative") or
                                   (direction == "higher_descriptive" and sign == "positive")) else "WORSENED")
                    item = dict(reference=LABELS[0], candidate=LABELS[1], stage=stage, split=split,
                        parent_id=pid, **spec, reference_parent_mean=ref_p[pid], candidate_parent_mean=new_p[pid],
                        delta_candidate_minus_reference=delta, delta_sign=sign, descriptive_assessment=assessment,
                        valid_paired_outputs=sum(difference(r) is not None for r in parent_rows),
                        planned_paired_outputs=len(parent_rows),
                        stored_unscoped_parent_delta=stored,
                        original_parent_pair_row_retained_in="results/paired_per_parent.csv")
                    paired_parents.append(item)
                    local.append(item)
                valid_delta = [r["delta_candidate_minus_reference"] for r in local if r["delta_candidate_minus_reference"] is not None]
                paired_summaries.append(dict(reference=LABELS[0], candidate=LABELS[1], stage=stage, split=split,
                    **spec, delta_parent_macro_mean=avg(valid_delta),
                    delta_parent_median=statistics.median(valid_delta) if valid_delta else None,
                    delta_parent_min=min(valid_delta, default=None), delta_parent_max=max(valid_delta, default=None),
                    valid_paired_parent_n=len(valid_delta), planned_paired_parent_n=len(local),
                    improved_parent_n=sum(r["descriptive_assessment"] == "IMPROVED" for r in local) if spec["direction"] != "context_only" else None,
                    worsened_parent_n=sum(r["descriptive_assessment"] == "WORSENED" for r in local) if spec["direction"] != "context_only" else None,
                    tie_parent_n=sum(r["delta_sign"] == "tie" for r in local),
                    positive_delta_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] == "positive"],
                    negative_delta_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] == "negative"],
                    NA_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] is None]))
    # Part strata are additional descriptive slices of the same fixed requests.
    # No unobserved part or extra candidate is generated to balance these cells.
    part_parent_rows, part_summaries = [], []
    references = [r for r in rows if r["label"] == LABELS[0]]
    for (stage, split, part_type), refs in sorted(grouping(references, ("stage", "split", "part_type")).items()):
        for field, spec in specs.items():
            def part_values(r):
                candidate = lookup[LABELS[1], stage, r["request_id"]]
                return metric_value(r, field), metric_value(candidate, field)
            def part_delta(r):
                a, b = part_values(r)
                return b-a if a is not None and b is not None else None
            deltas, cond = parent_values(refs, part_delta)
            ref_p, _ = parent_values(refs, lambda r: part_values(r)[0] if part_delta(r) is not None else None)
            new_p, _ = parent_values(refs, lambda r: part_values(r)[1] if part_delta(r) is not None else None)
            local = []
            for pid, delta in sorted(deltas.items()):
                pr = [r for r in refs if r["parent_id"] == pid]
                sign = None if delta is None else ("positive" if delta > EPSILON else "negative" if delta < -EPSILON else "tie")
                direction = spec["direction"]
                assessment = ("NA" if sign is None else "CONTEXT_ONLY" if direction == "context_only"
                              else "TIE" if sign == "tie" else "IMPROVED" if
                              ((direction == "lower_descriptive" and sign == "negative") or
                               (direction == "higher_descriptive" and sign == "positive")) else "WORSENED")
                item = dict(reference=LABELS[0], candidate=LABELS[1], stage=stage, split=split,
                    part_type=part_type, parent_id=pid, **spec,
                    reference_parent_mean=ref_p[pid], candidate_parent_mean=new_p[pid],
                    delta_candidate_minus_reference=delta, delta_sign=sign, descriptive_assessment=assessment,
                    valid_paired_output_n=sum(part_delta(r) is not None for r in pr),
                    planned_paired_output_n=len(pr),
                    valid_paired_condition_n=sum(v is not None for (parent, _), v in cond.items() if parent == pid),
                    planned_condition_n=len({r["condition_id"] for r in pr}))
                part_parent_rows.append(item)
                local.append(item)
            valid = [r["delta_candidate_minus_reference"] for r in local if r["delta_candidate_minus_reference"] is not None]
            part_summaries.append(dict(reference=LABELS[0], candidate=LABELS[1], stage=stage, split=split,
                part_type=part_type, **spec, reference_parent_macro_mean=avg(list(ref_p.values())),
                candidate_parent_macro_mean=avg(list(new_p.values())), delta_parent_macro_mean=avg(valid),
                delta_parent_median=statistics.median(valid) if valid else None,
                delta_parent_min=min(valid, default=None), delta_parent_max=max(valid, default=None),
                valid_paired_output_n=sum(part_delta(r) is not None for r in refs), planned_paired_output_n=len(refs),
                valid_paired_condition_n=sum(v is not None for v in cond.values()), planned_condition_n=len(cond),
                valid_paired_parent_n=len(valid), planned_paired_parent_n=len(local),
                improved_parent_n=sum(r["descriptive_assessment"] == "IMPROVED" for r in local) if spec["direction"] != "context_only" else None,
                worsened_parent_n=sum(r["descriptive_assessment"] == "WORSENED" for r in local) if spec["direction"] != "context_only" else None,
                tie_parent_n=sum(r["delta_sign"] == "tie" for r in local),
                positive_delta_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] == "positive"],
                negative_delta_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] == "negative"],
                NA_parent_ids=[r["parent_id"] for r in local if r["delta_sign"] is None]))
    status_rows = []
    for identity, group in sorted(grouping(rows, ("label", "stage", "split")).items()):
        status_rows.append(dict(zip(("label", "stage", "split"), identity),
            rows=len(group), closed_C_outputs=sum(boolean(r["closed_C"]) for r in group),
            C_preserved_true=sum(boolean(r["C_preserved_exact"]) for r in group),
            C_preserved_false_request_ids=[r["request_id"] for r in group if not boolean(r["C_preserved_exact"])],
            I_status_counts=dict(collections.Counter(r.get("I_status", "MISSING") for r in group)),
            conflict_status_counts=dict(collections.Counter(r.get("ConflictCov_C_status", "MISSING") for r in group)),
            seam_status_counts=dict(collections.Counter(r.get("seam_to_bulk_status", "MISSING") for r in group))))
    return dict(status="COMPLETE", schema=VERSION, metrics=list(specs.values()),
        aggregation="Noise -> condition -> parent, then split parent macro; paired changes computed per fixed request first",
        comparison_epsilon_native_units=EPSILON,
        directions_are_descriptive_not_significance=True, source_csvs_unchanged=True,
        all_original_source_fields_preserved=True, source_artifacts=data["sources"],
        original_numeric_fields=data["completion"]["metric_fields"],
        scoped_flags=list(sorted(OPEN_FLAGS)),
        unavailable_metrics=[dict(metric="fragment_area_fraction", status="NA_NOT_IN_FROZEN_ROWS",
            reason="No frozen geometric fragment-area score. C-disconnected/vertex-only area is reported under its own name and may include a large detached body.")],
        key_metrics=key_rows, paired_parent_metrics=paired_parents,
        paired_parent_summary=paired_summaries, status_counts=status_rows,
        paired_part_parent_metrics=part_parent_rows, paired_part_summary=part_summaries,
        part_summary_scope="Original exact part_type strata; noise then condition then parent equal means; no new samples or balancing",
        image_review="NOT_PERFORMED_BY_THIS_SCRIPT", conclusions="NOT_GENERATED",
        model_calls=0, scoring_calls=0, new_geometry=0, new_queries=0,
        analysis_code_sha256=sha(__file__))


def jsontext(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def csvtext(rows):
    buf = io.StringIO(newline="")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return buf.getvalue()


def immutable_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != content.replace("\r\n", "\n"):
            raise ValueError("Refusing to change completed analysis: " + str(path))
        return
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def run(check_inputs=False):
    data = load_inputs()
    if data["status"] != "READY":
        return data
    if check_inputs:
        return dict(status="READY", rows=len(data["tables"]["per_output"]),
                    source_tables={k: len(v) for k, v in data["tables"].items()},
                    model_calls=0, scoring_calls=0, written_outputs=0)
    result = make_analysis(data)
    for path, info in data["sources"].items():
        if sha(path) != info["sha256"]:
            raise ValueError("Source CSV changed during analysis")
    files = {"key_metrics.json": jsontext(result),
             "key_metrics.csv": csvtext(result["key_metrics"]),
             "paired_parent_metrics.csv": csvtext(result["paired_parent_metrics"]),
             "paired_parent_summary.csv": csvtext(result["paired_parent_summary"]),
             "paired_part_parent_metrics.csv": csvtext(result["paired_part_parent_metrics"]),
             "paired_part_summary.csv": csvtext(result["paired_part_summary"]),
             "status_counts.csv": csvtext(result["status_counts"])}
    for name, text in files.items():
        immutable_text(DEST / name, text)
    completion = dict(status="COMPLETE", schema=VERSION,
        analysis_code_sha256=sha(__file__), source_artifacts=data["sources"],
        files={str(DEST / name): sha(DEST / name) for name in files},
        key_metric_rows=len(result["key_metrics"]), paired_parent_rows=len(result["paired_parent_metrics"]),
        paired_part_summary_rows=len(result["paired_part_summary"]),
        model_calls=0, scoring_calls=0, new_geometry=0, conclusions="NOT_GENERATED")
    immutable_text(DEST / "completion.json", jsontext(completion))
    return {k: v for k, v in completion.items() if k not in ("source_artifacts", "files")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    print(jsontext(run(args.check_inputs)), end="")
