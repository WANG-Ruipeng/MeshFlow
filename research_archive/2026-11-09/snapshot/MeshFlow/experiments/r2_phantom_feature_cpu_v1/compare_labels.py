"""Join frozen CPU features to existing stage-specific AI_PROXY reviews.

No geometry is rescored, no images are newly reviewed, and no labels are carried
between stages. This is descriptive development evidence, not a classifier test.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = HERE / "per_output.csv"
SOURCES = {
    "BENCH": ROOT / "experiments/r2_bench_clean_multipart_seam_v1/visual_review/per_cell.csv",
    "CHAIR_BACK": ROOT / "experiments/r2_chair_back_transfer_v1/visual_review/per_stage.csv",
    "CHAIR_SEAT": ROOT / "experiments/r2_seam_quality_wave1_wave2_v1/visual_review.csv",
}
METRICS = [
    "Phantom_Layer_Ratio", "bulk_parallel_area_fraction",
    "no_shared_C_vertex_layer_ratio", "centroid_full_face_ratio",
    "Seam_Extension_mean_over_L", "Seam_Extension_median_over_L",
    "Seam_Extension_min_over_L", "Seam_Extension_max_over_L",
    "seam_tip_within_005L_fraction", "C_area", "bulk_area",
    "matched_parallel_area", "retained_seam", "active_raw_seam",
    "seam_deleted", "bulk_faces", "matched_bulk_faces",
]
AXES = [
    "review_C_use", "review_body_complete", "review_overall_shape",
    "review_support_retention", "review_fragment_or_gap",
    "review_layering_or_fold",
]
POSITIVE = {"YES", "PLAUSIBLE"}
NEGATIVE = {"NO", "REPLACED_OR_IGNORED"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def write_csv(path, rows, fieldnames=None):
    fields = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_reviews():
    indexed = {}
    source_rows = {}
    for domain, path in SOURCES.items():
        records = read_csv(path)
        source_rows[domain] = len(records)
        for line_number, record in enumerate(records, 2):
            if domain == "BENCH":
                model = record["model"]
                stage = {"CLEAN_FINAL_POST": "FINAL_POST"}.get(
                    record["stage"], record["stage"]
                )
                label = {
                    "review_C_use": record["C_use"],
                    "review_body_complete": "NA_NOT_IN_SOURCE_SCHEMA",
                    "review_overall_shape": record["overall_shape"],
                    "review_support_retention": record["support_retention"],
                    "review_fragment_or_gap": record["visible_fragment_or_gap"],
                    "review_layering_or_fold": record["layering_or_fold"],
                }
                note, viewed = record["notes"], record["actually_viewed"]
                reviewer = record.get("reviewer_kind", record.get("reviewer", ""))
            elif domain == "CHAIR_BACK":
                model, stage = "PLAIN_S2000", record["stage"]
                label = {
                    "review_C_use": record["back_role_adoption"],
                    "review_body_complete": record["body_complete"],
                    "review_overall_shape": record["overall_reasonable"],
                    "review_support_retention": "NA_NOT_IN_SOURCE_SCHEMA",
                    "review_fragment_or_gap": record["visible_gap_or_split"],
                    "review_layering_or_fold": record["local_gray_overlay_on_C"],
                }
                note, viewed = record["observations"], record["actual_viewed"]
                reviewer = record["reviewer"]
            else:
                # Only the unbound PLAIN_S2000 POST review describes BASE_POST.
                # BOUND015, other arms and every unreviewed stage remain unused.
                if record["label"] != "PLAIN_S2000" or record["stage"] != "POST":
                    continue
                model, stage = record["label"], "BASE_POST"
                label = {
                    "review_C_use": record["C_adopted_as_seat"],
                    "review_body_complete": record["body_complete"],
                    "review_overall_shape": "NA_NOT_IN_SOURCE_SCHEMA",
                    "review_support_retention": "NA_NOT_IN_SOURCE_SCHEMA",
                    "review_fragment_or_gap": record["visible_cracks"],
                    "review_layering_or_fold": record["obvious_fragments_or_folded_sheets"],
                }
                note, viewed = record["note"], record["actually_viewed"]
                reviewer = record.get("reviewer_identity", record["reviewer"])
            key = (domain, model, record["request_id"], stage)
            if key in indexed:
                raise ValueError(f"Duplicate source review identity: {key}")
            indexed[key] = {
                **label,
                "review_status": "MATCHED_EXISTING_STAGE_REVIEW",
                "review_source_file": path.relative_to(ROOT).as_posix(),
                "review_source_row": line_number,
                "review_original_stage": record["stage"],
                "review_original_split": record.get("split", ""),
                "reviewer_kind": reviewer,
                "review_source_actually_viewed": viewed,
                "review_notes": note,
                "review_original_record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
            }
    return indexed, source_rows


def main():
    rows = read_csv(FEATURES)
    reviews, source_counts = get_reviews()
    joined = []
    for row in rows:
        key = tuple(row[k] for k in ("domain", "model", "request_id", "stage"))
        review = reviews.get(key)
        if review is None:
            review = {
                **{axis: "NA_MISSING_STAGE_REVIEW" for axis in AXES},
                "review_status": "NA_MISSING_STAGE_REVIEW",
                "review_source_file": "", "review_source_row": "",
                "review_original_stage": "", "review_original_split": "",
                "reviewer_kind": "", "review_source_actually_viewed": "",
                "review_notes": "", "review_original_record_json": "",
            }
        joined.append({**row, **review})
    write_csv(HERE / "per_output_with_review.csv", joined)

    coverage = []
    coverage_groups = defaultdict(list)
    for row in joined:
        coverage_groups[tuple(row[k] for k in ("domain", "model", "stage", "split"))].append(row)
    for key, values in sorted(coverage_groups.items()):
        coverage.append({
            **dict(zip(("domain", "model", "stage", "split"), key)),
            "outputs": len(values),
            "matched_reviews": sum(v["review_status"].startswith("MATCHED") for v in values),
            "missing_reviews": sum(v["review_status"].startswith("NA") for v in values),
            "C_use_counts": dict(Counter(v["review_C_use"] for v in values)),
        })

    grouped = defaultdict(list)
    for row in joined:
        for axis in AXES:
            for split in (row["split"], "ALL"):
                key = (row["domain"], row["model"], row["stage"], split, axis, row[axis])
                grouped[key].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        for metric in METRICS:
            measured = [number(v[metric]) for v in values]
            measured = [v for v in measured if v is not None]
            summary.append({
                **dict(zip(("domain", "model", "stage", "split", "label_axis", "label_value"), key)),
                "outputs": len(values),
                "conditions": len({v["condition_id"] for v in values}),
                "parents": len({v["parent_id"] for v in values}),
                "metric": metric,
                "n_finite": len(measured),
                "n_NA": len(values) - len(measured),
                "mean": statistics.mean(measured) if measured else "",
                "median": statistics.median(measured) if measured else "",
                "min": min(measured) if measured else "",
                "max": max(measured) if measured else "",
            })
    write_csv(HERE / "label_group_summary.csv", summary)

    noise_groups = defaultdict(list)
    for row in joined:
        noise_groups[tuple(row[k] for k in ("domain", "panel", "model", "condition_id", "stage", "split"))].append(row)
    pairs = []
    counterexamples = []
    for key, values in sorted(noise_groups.items()):
        values = sorted(values, key=lambda x: (int(x["noise_index"]), x["request_id"]))
        if len({v["noise_index"] for v in values}) != len(values):
            raise ValueError(f"Duplicate noise identities for: {key}")
        for a, b in itertools.combinations(values, 2):
            pair = {
                **dict(zip(("domain", "panel", "model", "condition_id", "stage", "split"), key)),
                "request_a": a["request_id"], "request_b": b["request_id"],
                "noise_a": a["noise_index"], "noise_b": b["noise_index"],
                "C_use_a": a["review_C_use"], "C_use_b": b["review_C_use"],
                "review_status_a": a["review_status"], "review_status_b": b["review_status"],
                "C_use_label_differs": a["review_C_use"] != b["review_C_use"],
                "body_complete_a": a["review_body_complete"],
                "body_complete_b": b["review_body_complete"],
                "overall_shape_a": a["review_overall_shape"],
                "overall_shape_b": b["review_overall_shape"],
                "notes_a": a["review_notes"], "notes_b": b["review_notes"],
            }
            for metric in METRICS:
                av, bv = number(a[metric]), number(b[metric])
                pair[f"{metric}_a"] = "" if av is None else av
                pair[f"{metric}_b"] = "" if bv is None else bv
                pair[f"{metric}_delta_b_minus_a"] = "" if av is None or bv is None else bv - av
            pairs.append(pair)
            positive, negative = None, None
            if a["review_C_use"] in POSITIVE and b["review_C_use"] in NEGATIVE:
                positive, negative = a, b
            elif b["review_C_use"] in POSITIVE and a["review_C_use"] in NEGATIVE:
                positive, negative = b, a
            if positive:
                pv = number(positive["Phantom_Layer_Ratio"])
                nv = number(negative["Phantom_Layer_Ratio"])
                counterexamples.append({
                    **dict(zip(("domain", "panel", "model", "condition_id", "stage", "split"), key)),
                    "positive_request": positive["request_id"],
                    "negative_request": negative["request_id"],
                    "positive_label": positive["review_C_use"],
                    "negative_label": negative["review_C_use"],
                    "positive_Phantom_Layer_Ratio": pv,
                    "negative_Phantom_Layer_Ratio": nv,
                    "positive_tip_mean_over_L": number(positive["Seam_Extension_mean_over_L"]),
                    "negative_tip_mean_over_L": number(negative["Seam_Extension_mean_over_L"]),
                    "counterexample_to_larger_phantom_means_worse_adoption":
                        None if pv is None or nv is None else pv > nv,
                })
    write_csv(HERE / "same_C_noise_pairs.csv", pairs)
    findings = {
        "schema": "FROZEN_STAGE_REVIEW_FEATURE_COMPARISON_V1",
        "status": "COMPLETE",
        "new_image_reviews": 0,
        "new_geometry_scores": 0,
        "training_or_models_called": 0,
        "new_thresholds_selected": 0,
        "classifier_or_AUROC": "NOT_RUN",
        "limitations": [
            "Existing nonblind AI_PROXY ratings are not newly independently verified.",
            "No label is copied between RAW, BASE_POST and FINAL_POST.",
            "CHAIR_SEAT PLAIN_S2000 POST maps only to BASE_POST; RAW and FINAL are NA.",
            "Bench CLEAN_FINAL_POST maps to FINAL_POST; Chair_BACK stages match exactly.",
            "Original labels and full review records are preserved; label schemas differ by domain.",
            "Review_C_use PLAUSIBLE or YES means visual adoption, not legal connectivity.",
            "Same-C noise contrasts are descriptive, dependent observations; no causal claim or reliable AUROC.",
            "ALL split summaries coexist with split-specific summaries; no formal train/test selection.",
            "Phantom_Layer_Ratio is not certified as a success or failure metric.",
            "Missing seam tips remain NA rather than zero in metric summaries.",
        ],
        "input_hashes": {
            str(p.relative_to(ROOT)): sha(p) for p in [FEATURES, *SOURCES.values()]
        },
        "source_review_rows": source_counts,
        "output_rows": len(joined),
        "matched_reviews": sum(r["review_status"].startswith("MATCHED") for r in joined),
        "missing_stage_reviews": sum(r["review_status"].startswith("NA") for r in joined),
        "coverage_by_domain_model_stage_split": coverage,
        "summary_rows": len(summary),
        "same_C_noise_pair_rows": len(pairs),
        "same_C_positive_negative_pairs": counterexamples,
        "positive_negative_pairs_count": len(counterexamples),
        "positive_negative_distinct_conditions_count": len({
            (c["domain"], c["model"], c["condition_id"]) for c in counterexamples
        }),
        "phantom_reverse_order_pairs_count": sum(
            c["counterexample_to_larger_phantom_means_worse_adoption"] is True
            for c in counterexamples
        ),
        "pair_independence_note":
            "The same condition contrast at multiple stages is not an independent observation.",
    }
    findings["output_hashes"] = {
        name: sha(HERE / name) for name in (
            "per_output_with_review.csv", "label_group_summary.csv", "same_C_noise_pairs.csv"
        )
    }
    findings["script_sha256"] = sha(Path(__file__).resolve())
    (HERE / "label_findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        k: findings[k] for k in (
            "status", "output_rows", "matched_reviews", "missing_stage_reviews",
            "summary_rows", "same_C_noise_pair_rows", "positive_negative_pairs_count",
            "phantom_reverse_order_pairs_count"
        )
    }, indent=2))


if __name__ == "__main__":
    main()
