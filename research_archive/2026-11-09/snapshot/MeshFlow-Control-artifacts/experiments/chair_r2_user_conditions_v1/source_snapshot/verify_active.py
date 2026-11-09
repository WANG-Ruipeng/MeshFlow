"""Verify active subset references without rerunning or modifying export validation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import time

from verify_export import ACCEPTED, Verifier, quality_summary, read, sha


FROZEN_EXPORT_VALIDATION_SHA256 = "0eae505428b082f6474e11ee664ea3e3975ce64c916854a1ef694371c9da9649"
EXPECTED_HELD = {
    "21f2927b04e2f22830ddb6ead95f49cc__seat__small",
    "21f2927b04e2f22830ddb6ead95f49cc__back__small",
    "47da08d9c7cd7e104b3c42e318f3affc__seat__small",
    "47da08d9c7cd7e104b3c42e318f3affc__back__small",
    "73b369b9808bc2d0f6ba522682be8978__seat__small",
    "c1b8c2f0c4f3cc43aff7c08b060f5ed6__seat__small",
}


def verify_active(out):
    started = time.perf_counter()
    out = Path(out)
    validator = Verifier()
    filenames = ("validation_export.json", "conditions.json", "regions.json",
                 "quality/summary.json", "usability_review.json", "active_conditions.json")
    hashes_before = {str(out / name): sha(out / name) for name in filenames}
    baseline = read(out / "validation_export.json")
    original = read(out / "conditions.json")
    active = read(out / "active_conditions.json")
    review = read(out / "usability_review.json")
    regions = read(out / "regions.json")
    quality = read(out / "quality/summary.json")
    originals = {r["condition_id"]: r for r in original["rows"]}
    rows = active["rows"]
    active_ids = [r["condition_id"] for r in rows]
    held_ids = [r["condition_id"] for r in review["rows"]]
    baseline_quality = {r["condition_id"]: r for r in baseline["per_condition_source_geometry"]}
    baseline_results = {r["condition_id"]: r for r in baseline["condition_results"]}
    quality_parents = {r["uid"]: r for r in quality["rows"]}
    validator.hash(out / "validation_export.json", FROZEN_EXPORT_VALIDATION_SHA256,
                   "ORIGINAL_6154_CHECK_REPORT_UNCHANGED")
    validator.check(baseline["status"] == "PASS" and baseline["check_count"] == 6154 and baseline["failure_count"] == 0,
                    "FROZEN_EXPORT_VALIDATION_PASS", "validation_export")
    for name in ("conditions.json", "regions.json", "quality/summary.json"):
        validator.hash(out / name, baseline["input_manifests"][str(out / name)], "baseline_binding/" + name)
    validator.hash(active["source_all_exported_manifest"], active["source_all_exported_sha256"], "active_source_binding")
    validator.check(Path(active["source_all_exported_manifest"]) == out / "conditions.json",
                    "ACTIVE_REFERENCES_ORIGINAL_CONDITIONS", "active_conditions")
    validator.hash(out / "usability_review.json", active["usability_review_sha256"], "active_review_binding")
    validator.hash(out / "conditions.json", review["source_manifest_sha256"], "review_source_binding")
    validator.check(len(originals) == len(original["rows"]) == 129, "ORIGINAL_129_UNIQUE_EXPORTS_PRESERVED", "conditions")
    validator.check(len(active_ids) == len(set(active_ids)) == 123, "ACTIVE_123_UNIQUE_REFERENCES", "active_conditions")
    validator.check(len(held_ids) == len(set(held_ids)) == 6 and set(held_ids) == EXPECTED_HELD,
                    "EXACT_SIX_REGISTERED_HOLDS", "usability_review")
    validator.check(set(active_ids).isdisjoint(held_ids) and set(active_ids) | set(held_ids) == set(originals),
                    "ACTIVE_AND_HOLD_DISJOINT_COMPLETE_ORIGINAL_PARTITION", "active_conditions")
    validator.check(set(active["held_condition_ids"]) == EXPECTED_HELD and len(active["held_condition_ids"]) == 6,
                    "ACTIVE_HELD_LIST_MATCHES_REVIEW", "active_conditions")
    counts = Counter(r["split"] for r in rows)
    hold_counts = Counter(r["split"] for r in review["rows"])
    validator.check(counts == {"train": 117, "dev": 6}, "ACTIVE_SPLIT_COUNTS_117_6", "active_conditions")
    validator.check(hold_counts == {"train": 5, "dev": 1}, "HELD_SPLIT_COUNTS_5_1", "usability_review")
    validator.check(active["train_condition_count"] == 117 and active["dev_condition_count"] == 6,
                    "ACTIVE_REPORTED_COUNTS_ACCURATE", "active_conditions")
    train_parents = {r["uid"] for r in rows if r["split"] == "train"}
    original_train_parents = {r["uid"] for r in original["rows"] if r["split"] == "train"}
    validator.check(train_parents == original_train_parents and len(train_parents) == 25,
                    "ALL_25_TRAIN_PARENTS_RETAINED", "active_conditions")
    validator.check(set(active["train_parents"]) == train_parents and len(active["train_parents"]) == 25,
                    "ACTIVE_PARENT_LIST_ACCURATE", "active_conditions")
    validator.check(train_parents.isdisjoint({r["uid"] for r in rows if r["split"] == "dev"}),
                    "ACTIVE_TRAIN_DEV_DISJOINT", "active_conditions")
    by_region = defaultdict(list)
    for row in rows:
        cid = row["condition_id"]
        validator.check(cid in originals and row == originals[cid], "ACTIVE_ROW_EXACT_ORIGINAL_REFERENCE", cid)
        validator.hash(row["npz"], row["sha256"], "active_npz/" + cid)
        validator.check(cid in baseline_results and baseline_results[cid]["status"] == "PASS",
                        "ACTIVE_REFERENCES_PREVIOUSLY_VERIFIED_EXPORT", cid)
        validator.check(cid in baseline_quality, "ACTIVE_HAS_FROZEN_SOURCE_QUALITY", cid)
        by_region[row["region_id"]].append(row)
    held_evidence = []
    for row in review["rows"]:
        cid = row["condition_id"]
        source = originals[cid]
        validator.check(row["uid"] == source["uid"] and row["split"] == source["split"], "HELD_SOURCE_IDENTITY", cid)
        validator.check(row["include_in_active_pool"] is False and row["status"] == "HOLD_TINY_EDGE_STRIP_AI_PROXY",
                        "HELD_EXPLICIT_NEGATIVE_DISPOSITION", cid)
        validator.check(isinstance(row["reason"], str) and len(row["reason"].strip()) > 20,
                        "HELD_HAS_NEGATIVE_REASON", cid)
        validator.check(row["image_actually_viewed"] is True and "AI_PROXY" in row["reviewer"],
                        "HELD_HAS_ACTUAL_AI_PROXY_REVIEW_RECORD", cid)
        validator.check(row["original_npz"] == source["npz"] and row["original_npz_sha256"] == source["sha256"],
                        "HELD_ORIGINAL_DATA_REFERENCE", cid)
        validator.hash(row["original_npz"], row["original_npz_sha256"], "held_npz/" + cid)
        validator.hash(row["image"], row["image_sha256"], "held_image/" + cid)
        with Path(row["image"]).open("rb") as stream:
            validator.check(stream.read(8) == b"\x89PNG\r\n\x1a\n", "HELD_IMAGE_PNG_RETAINED", cid)
        validator.check(row["region_area_fraction"] == source["region_area_fraction"], "HELD_AREA_REFERENCE_UNCHANGED", cid)
        held_evidence.append(dict(condition_id=cid, split=row["split"], status=row["status"], reason=row["reason"],
            npz=row["original_npz"], npz_sha256=row["original_npz_sha256"], image=row["image"],
            image_sha256=row["image_sha256"], image_actually_viewed_recorded=True,
            independent_validator_visual_review=False))
    region_coverage = []
    accepted_train_regions = [r for r in regions["rows"] if r["split"] == "train" and r["status"] in ACCEPTED]
    validator.check(len(accepted_train_regions) == 47, "47_ACCEPTED_TRAIN_REGIONS", "regions")
    for region in accepted_train_regions:
        rid = region["region_id"]
        selections = by_region[rid]
        tiers = defaultdict(list)
        for row in selections:
            for tier in row["requested_tiers"]:
                tiers[tier].append(row["condition_id"])
        validator.check(len(tiers["medium"]) == 1 and len(tiers["full"]) == 1,
                        "TRAIN_REGION_MEDIUM_AND_FULL_RETAINED", rid)
        if tiers["full"]:
            full = originals[tiers["full"][0]]
            validator.check(set(full["source_face_ids"]) == set(region["source_face_ids"]), "FULL_REMAINS_COMPLETE_R", rid)
        region_coverage.append(dict(region_id=rid, uid=region["uid"], available_requested_tiers=sorted(tiers),
            medium_condition_ids=tiers["medium"], full_condition_ids=tiers["full"],
            held_small_condition_ids=sorted(cid for cid in EXPECTED_HELD if originals[cid]["region_id"] == rid)))
    validator.check(review["model_output_selection"] is False and review["training_authorization"] is False,
                    "REVIEW_PRE_MODEL_AND_NOT_TRAINING_AUTHORIZATION", "usability_review")
    validator.check(active["new_generated_arrays"] == 0, "ACTIVE_SUBSET_CREATES_NO_ARRAYS", "active_conditions")
    selected_quality = [baseline_quality[cid] for cid in active_ids]
    summary = quality_summary(selected_quality, quality_parents)
    for path, expected in hashes_before.items():
        validator.check(sha(path) == expected, "INPUT_UNCHANGED_DURING_ACTIVE_VALIDATION", path)
    result = dict(schema="r2_user_conditions_active_subset_independent_validation_v1",
        status="PASS" if not validator.failures else "FAIL", check_count=validator.check_count,
        failure_count=len(validator.failures), failures=validator.failures,
        implementation_sha256=sha(__file__), input_hashes=hashes_before,
        original_validation=dict(path=str(out / "validation_export.json"), sha256=FROZEN_EXPORT_VALIDATION_SHA256,
            checks=6154, unchanged=True, rerun=False),
        counts=dict(original_exports=129, active_train=counts["train"], active_dev=counts["dev"],
            held_train=hold_counts["train"], held_dev=hold_counts["dev"], train_parents=len(train_parents),
            accepted_train_regions_with_medium_and_full=len(region_coverage)),
        checked_file_hashes=validator.hashes, held_evidence=held_evidence,
        accepted_train_region_coverage=region_coverage,
        active_condition_source_geometry=summary,
        active_condition_quality_reference=dict(path=str(out / "validation_export.json"),
            key="per_condition_source_geometry", selected_condition_ids=active_ids),
        limitations=[
            "Independent check verifies preserved references and recorded visual evidence; it does not claim a second visual review of the six images.",
            "Medium/full coverage applies to the 47 accepted TRAIN regions; DEV retains only its originally registered single tier.",
            "The original 129 exports and their 6154-check validation remain unchanged; six holds are preparation evidence, not deleted data.",
            "Subset quality is re-aggregation of frozen source-pair classifications, without new geometry scoring.",
            "Zero confirmed CC does not mean source geometry is clean; CC UNKNOWN, CF and FF remain separately reported.",
            "Condition pair occurrences repeat source pairs; unique parent totals count each source parent once."],
        execution=dict(workers=1, numeric_threads=2, wall_seconds=time.perf_counter() - started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            model_calls=0, OT_calls=0, CUDA_calls=0, training_updates=0, new_geometry_pair_tests=0,
            source_mutations=0, original_validation_rewrites=0))
    target = out / "validation_active.json"
    temporary = out / "validation_active.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(dict(status=result["status"], check_count=result["check_count"], failure_count=result["failure_count"],
        failures=result["failures"], counts=result["counts"], source_quality=summary,
        execution=result["execution"], output=str(target)), ensure_ascii=False))
    return not validator.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(0 if verify_active(arguments.out) else 1)
