"""Deterministic CPU-only merge of three completed original-image AI reviews.

This helper preserves judgment strings and categories exactly. It imports no
model, scorer, experiment evaluator or audit runner.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import os

VERSION = "r2_user_mix_ai_proxy_review_merge_v1"
ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "review"
SOURCES = ("visual_33ec_534b.json", "visual_866e_86ec.json", "visual_bdd_f596.json")
CATEGORIES = ("improved", "similar", "degraded", "uncertain")
DIMENSIONS = ("overall", "C_use")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def resolve(path):
    value = str(path)
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return Path(value).resolve()


def frozen_text(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf8") != text:
            raise ValueError("Refuse overwrite of frozen merged review: " + str(path))
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf8")
    temp.replace(path)


def write_json(path, value):
    frozen_text(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, prefix):
    fields = list(dict.fromkeys(prefix + [key for row in rows for key in row]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    frozen_text(path, buffer.getvalue())


def counts(rows, dimension):
    values = collections.Counter(row[dimension] for row in rows)
    if set(values) - set(CATEGORIES):
        raise ValueError("Unknown judgment category")
    return {category: values[category] for category in CATEGORIES}


def main():
    reg_path = ROOT / "evaluation/registration.json"
    reg = read(reg_path)
    labels = reg["labels"]
    if labels != ["R2_FROZEN", "R2_CONT_OLD", "R2_CONT_USER", "R2_USER_MIX"]:
        raise ValueError("Model identities differ from the reviewed panel")
    baselines, candidate = labels[:3], labels[3]
    requests = {q["request_id"]: q for q in reg["requests"]}
    if len(requests) != 12 or len({q["parent_id"] for q in requests.values()}) != 6:
        raise ValueError("Not the registered six-parent/two-noise panel")
    manifest_path = ROOT / "gallery/manifest.json"
    manifest = read(manifest_path)
    manifest_images = {resolve(page["image"]): page for page in manifest["pages"]}
    expected_images = {resolve(ROOT / "gallery" / mode / (rid + ".png")) for rid in requests for mode in ("colored", "neutral")}
    if set(manifest_images) != expected_images or len(expected_images) != 24:
        raise ValueError("Gallery does not cover exactly all registered original images")
    images = []
    for path in sorted(expected_images):
        page = manifest_images[path]
        actual_sha = sha(path)
        if actual_sha != page["image_sha256"] or page["identity"]["columns"] != labels:
            raise ValueError("Image bytes or column identities changed")
        images.append({"path": str(path), "sha256": actual_sha, "request_id": page["identity"]["request_id"],
                       "mode": page["identity"]["mode"], "registered_views": len(page["identity"]["cameras"]),
                       "actual_viewing_claim_sources": []})
    image_map = {resolve(row["path"]): row for row in images}
    outputs, paired, provenance = [], [], []
    seen_outputs, seen_pairs, seen_images = set(), set(), set()
    original_outputs, original_pairs = {}, {}
    for filename in SOURCES:
        source = REVIEW / filename
        document = read(source)
        if document.get("reviewer") != "AI_PROXY" or document.get("human_rating") is not False:
            raise ValueError("Every source must explicitly be AI_PROXY and not human ratings")
        viewed = [resolve(path) for path in document["images_viewed"]]
        if len(viewed) != len(set(viewed)) or any(path not in expected_images for path in viewed):
            raise ValueError("Duplicate or unregistered image in a source review")
        if seen_images.intersection(viewed):
            raise ValueError("Overlapping source assignments need independent adjudication, not duplicate counting")
        declared_hashes = {resolve(path): value for path, value in document.get("image_sha256", {}).items()}
        if set(declared_hashes) - set(viewed):
            raise ValueError("Source claims an image hash outside its viewed set")
        for path in viewed:
            if path in declared_hashes and sha(path) != declared_hashes[path]:
                raise ValueError("Image changed since source visual review")
            image_map[path]["actual_viewing_claim_sources"].append(filename)
        seen_images.update(viewed)
        # Keep all non-row context as source provenance, including explicitly
        # non-blinded root review. Do not present the merged set as blinded.
        provenance.append({"path": str(source), "sha256": sha(source), "original_metadata": {
            key: value for key, value in document.items() if key not in ("outputs", "comparisons", "images_viewed", "image_sha256")},
            "images_viewed": [str(path) for path in viewed], "source_declared_image_hash_count": len(declared_hashes)})
        source_requests = {row["request_id"] for row in document["outputs"]}
        required_source_images = {resolve(ROOT / "gallery" / mode / (rid + ".png")) for rid in source_requests for mode in ("colored", "neutral")}
        if set(viewed) != required_source_images:
            raise ValueError("Every reviewed request must include both colored and neutral original images")
        for row in document["outputs"]:
            rid, label = row["request_id"], row["label"]
            key = (rid, label)
            if rid not in requests or label not in labels or key in seen_outputs or row.get("actually_viewed") is not True:
                raise ValueError("Invalid, duplicate or not-actually-viewed output")
            if row.get("human_rating", False) is not False:
                raise ValueError("Unexpected human-rating row")
            seen_outputs.add(key)
            original_outputs[key] = row
            q = requests[rid]
            enriched = {"reviewer": "AI_PROXY", "human_rating": False, "source_review": filename, "source_review_sha256": sha(source),
                        "parent_id": q["parent_id"], "condition_id": q["condition_id"], "panel": q["panel"],
                        "region_type": q["region_type"], "tier": q["tier"], "noise_candidate": q["candidate"], "N": q["N"], "K": q["K"]}
            for mode in ("colored", "neutral"):
                path = resolve(ROOT / "gallery" / mode / (rid + ".png"))
                enriched[mode + "_image"] = str(path)
                enriched[mode + "_image_sha256"] = image_map[path]["sha256"]
            if set(row).intersection(enriched):
                raise ValueError("Enrichment would replace an original judgment field")
            outputs.append({**enriched, **row})
        for row in document["comparisons"]:
            rid, baseline = row["request_id"], row["baseline"]
            key = (rid, baseline)
            if rid not in source_requests or baseline not in baselines or row["candidate"] != candidate or key in seen_pairs:
                raise ValueError("Invalid or duplicate paired judgment")
            if any(row[dimension] not in CATEGORIES for dimension in DIMENSIONS):
                raise ValueError("Unknown paired category")
            if (rid, baseline) not in original_outputs or (rid, candidate) not in original_outputs:
                raise ValueError("Paired judgment lacks an actually viewed output")
            seen_pairs.add(key)
            original_pairs[key] = row
            q = requests[rid]
            enriched = {"reviewer": "AI_PROXY", "human_rating": False, "source_review": filename, "source_review_sha256": sha(source),
                        "parent_id": q["parent_id"], "condition_id": q["condition_id"], "panel": q["panel"],
                        "region_type": q["region_type"], "tier": q["tier"], "noise_candidate": q["candidate"], "N": q["N"], "K": q["K"]}
            for mode in ("colored", "neutral"):
                path = resolve(ROOT / "gallery" / mode / (rid + ".png"))
                enriched[mode + "_image"] = str(path)
                enriched[mode + "_image_sha256"] = image_map[path]["sha256"]
            if set(row).intersection(enriched):
                raise ValueError("Enrichment would replace a paired judgment field")
            paired.append({**enriched, **row})
    if seen_outputs != {(rid, label) for rid in requests for label in labels} or len(outputs) != 48:
        raise ValueError("Missing output judgments")
    if seen_pairs != {(rid, baseline) for rid in requests for baseline in baselines} or len(paired) != 36:
        raise ValueError("Missing paired judgments")
    if seen_images != expected_images:
        raise ValueError("Missing original-image review coverage")
    outputs.sort(key=lambda row: (row["request_id"], labels.index(row["label"])))
    paired.sort(key=lambda row: (row["request_id"], baselines.index(row["baseline"])))
    for row in outputs:
        original = original_outputs[row["request_id"], row["label"]]
        if any(row[key] != value for key, value in original.items()):
            raise ValueError("Original output judgment was modified")
    for row in paired:
        original = original_pairs[row["request_id"], row["baseline"]]
        if any(row[key] != value for key, value in original.items()):
            raise ValueError("Original paired judgment was modified")
    per_baseline, regional, consistency = [], [], []
    parents = sorted({q["parent_id"] for q in requests.values()})
    for baseline in baselines:
        selected = [row for row in paired if row["baseline"] == baseline]
        per_baseline.append({"baseline": baseline, "candidate": candidate, "requests": len(selected),
                             **{dimension: counts(selected, dimension) for dimension in DIMENSIONS}})
        for region in ("seat", "back"):
            rows = [row for row in selected if row["region_type"] == region]
            regional.append({"baseline": baseline, "candidate": candidate, "region_type": region, "requests": len(rows),
                             "parents": len({row["parent_id"] for row in rows}),
                             **{dimension: counts(rows, dimension) for dimension in DIMENSIONS}})
        for parent in parents:
            rows = sorted((row for row in selected if row["parent_id"] == parent), key=lambda row: row["noise_candidate"])
            if len(rows) != 2 or [row["noise_candidate"] for row in rows] != [0, 1] or len({row["condition_id"] for row in rows}) != 1:
                raise ValueError("Parent two-noise pairing changed")
            record = {"baseline": baseline, "candidate": candidate, "parent_id": parent, "condition_id": rows[0]["condition_id"],
                      "region_type": rows[0]["region_type"], "tier": rows[0]["tier"], "request_ids": [row["request_id"] for row in rows]}
            for dimension in DIMENSIONS:
                statuses = [row[dimension] for row in rows]
                record[dimension] = {"noise_0": statuses[0], "noise_1": statuses[1],
                                     "same_label_both_noises": statuses[0] == statuses[1],
                                     "consistency": "both_" + statuses[0] if statuses[0] == statuses[1] else "mixed",
                                     "both_improved": statuses == ["improved", "improved"],
                                     "any_improved": "improved" in statuses, "any_degraded": "degraded" in statuses,
                                     "any_uncertain": "uncertain" in statuses}
            consistency.append(record)
    parent_counts = []
    for baseline in baselines:
        records = [record for record in consistency if record["baseline"] == baseline]
        entry = {"baseline": baseline, "parents": len(records)}
        for dimension in DIMENSIONS:
            counter = collections.Counter(record[dimension]["consistency"] for record in records)
            entry[dimension] = {key: counter[key] for key in (*("both_" + category for category in CATEGORIES), "mixed")}
        parent_counts.append(entry)
    summary = {"schema": VERSION, "status": "COMPLETE", "reviewer": "AI_PROXY", "human_rating": False,
               "coverage": {"registered_requests": 12, "parents": 6, "conditions": 6, "noise_candidates_per_condition": 2,
                            "registered_models": labels, "output_judgments": 48, "paired_judgments": 36, "original_pngs_viewed": 24,
                            "all_outputs_actually_viewed": True, "all_original_judgment_fields_preserved": True,
                            "duplicate_outputs": 0, "duplicate_pairs": 0, "missing_outputs": 0, "missing_pairs": 0, "missing_images": 0},
               "per_baseline": per_baseline, "by_region": regional, "parent_two_noise_consistency": consistency,
               "parent_consistency_counts": parent_counts, "image_coverage": images, "source_reviews": provenance,
               "interpretation": ["AI_PROXY visual observations, not human ratings or a calibrated success probability.",
                                  "Similar means no clear relative change; it does not mean either output succeeds or is geometrically clean.",
                                  "Overall shape and C use are separate judgments; a change of chair style alone is not a quality improvement.",
                                  "Two noises are paired outcomes of one condition, not independent parents or training runs.",
                                  "The root review explicitly had prior numerical aggregate context; the merged review is not wholly blinded.",
                                  "Image inspection cannot certify hidden structure, vertex connectivity, watertightness, mechanical validity or illegal intersections.",
                                  "Repeated development evidence and historical checkpoints do not establish broad generalization or single-factor causal attribution."],
               "model_calls": 0, "scorer_calls": 0, "audit_runner_calls": 0}
    output_csv = REVIEW / "AI_PROXY_per_output.csv"
    paired_csv = REVIEW / "AI_PROXY_paired.csv"
    summary_json = REVIEW / "AI_PROXY_summary.json"
    summary_md = REVIEW / "AI_PROXY_summary.md"
    write_csv(output_csv, outputs, ["request_id", "label", "parent_id", "condition_id", "region_type", "tier", "noise_candidate"])
    write_csv(paired_csv, paired, ["request_id", "baseline", "candidate", "parent_id", "region_type", "tier", "noise_candidate", "overall", "C_use"])
    # Round-trip every original text/category through CSV, without coercing it.
    with output_csv.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        original = original_outputs[row["request_id"], row["label"]]
        for key, value in original.items():
            if row[key] != str(value):
                raise ValueError("Output CSV altered an original text or flag")
    with paired_csv.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        original = original_pairs[row["request_id"], row["baseline"]]
        if any(row[key] != str(value) for key, value in original.items()):
            raise ValueError("Paired CSV altered an original text/category")
    write_json(summary_json, summary)
    lines = ["# R2 USER mix: complete AI_PROXY visual-review summary", "", "All 24 original colored/neutral PNGs were reviewed: 12 fixed requests, 6 parents, 4 models, 48 output judgments and 36 new-versus-historical paired judgments. The two camera views are not independent samples.", "",
             "These are AI_PROXY observations, not human ratings. **Similar does not mean successful or geometrically clean.** Whole shape and C use remain separate. The root reviewer had prior numerical aggregate context; this is not a wholly blinded review.", "",
             "## Per-baseline paired counts", "", "| Baseline | Dimension | Improved | Similar | Degraded | Uncertain | Requests |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in per_baseline:
        for dimension in DIMENSIONS:
            c = item[dimension]
            lines.append(f"| {item['baseline']} | {dimension} | {c['improved']} | {c['similar']} | {c['degraded']} | {c['uncertain']} | {item['requests']} |")
    lines += ["", "## Region counts", "", "| Baseline | Region | Dimension | Improved | Similar | Degraded | Uncertain | Requests / parents |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for item in regional:
        for dimension in DIMENSIONS:
            c = item[dimension]
            lines.append(f"| {item['baseline']} | {item['region_type']} | {dimension} | {c['improved']} | {c['similar']} | {c['degraded']} | {c['uncertain']} | {item['requests']} / {item['parents']} |")
    lines += ["", "## Parent consistency across the two fixed noises", "", "The table preserves both noise judgments. Only `both_improved` supports a consistent improvement for that one parent/condition; it remains development evidence.", "", "| Baseline | Parent | Region/tier | Overall n0 / n1 | C use n0 / n1 |", "|---|---|---|---|---|"]
    for item in consistency:
        o, c = item["overall"], item["C_use"]
        lines.append(f"| {item['baseline']} | {item['parent_id']} | {item['region_type']}/{item['tier']} | {o['noise_0']} / {o['noise_1']} | {c['noise_0']} / {c['noise_1']} |")
    lines += ["", "## Coverage and interpretation", "", "- All 48 outputs are explicitly marked actually viewed; all 36 pairs and all 24 original images are present once, with no missing or duplicate records.",
              "- Original source text and categories are preserved exactly in `AI_PROXY_per_output.csv` and `AI_PROXY_paired.csv`; their source file/hash and input metadata are attached.",
              "- `AI_PROXY_summary.json` retains per-image hashes, source review context, region counts and all six-parent/two-noise comparisons.",
              "- Different chair style does not itself establish better quality. Pixels cannot certify hidden geometry, connectivity or illegal intersections.",
              "- These historical checkpoints differ in training history. This small, repeatedly viewed development panel does not establish generalization or sole-factor causality.",
              "- This merge used no model, scorer or experiment audit runner.", ""]
    frozen_text(summary_md, "\n".join(lines))
    artifacts = (output_csv, paired_csv, summary_json, summary_md)
    identity = {"schema": VERSION, "helper_path": str(Path(__file__).resolve()), "helper_sha256": sha(__file__),
                "evaluation_registration_sha256": sha(reg_path), "gallery_manifest_sha256": sha(manifest_path),
                "source_review_sha256": {name: sha(REVIEW / name) for name in SOURCES},
                "outputs_sha256": {path.name: sha(path) for path in artifacts}, "original_judgment_text_preserved": True,
                "model_calls": 0, "scorer_calls": 0, "audit_runner_calls": 0}
    write_json(REVIEW / "AI_PROXY_review_identity.json", identity)
    print(json.dumps({"status": "COMPLETE", "outputs": len(outputs), "pairs": len(paired), "images": len(images),
                      "parents": len(parents), "helper_sha256": identity["helper_sha256"], "per_baseline": per_baseline}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
