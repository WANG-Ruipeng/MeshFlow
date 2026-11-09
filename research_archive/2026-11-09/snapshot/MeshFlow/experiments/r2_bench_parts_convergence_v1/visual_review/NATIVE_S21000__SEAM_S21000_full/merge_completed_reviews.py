"""Merge actual completed reviews; retain original labels and wording."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LABELS = ("NATIVE_S21000", "SEAM_S21000")

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def write(p, value):
    with p.open("x", encoding="utf-8") as f:
        f.write(value)

manifest_path = ROOT / "gallery_manifests" / (OUT.name + ".json")
completion_path = ROOT / "gallery" / OUT.name / "completion.json"
input_path = ROOT / "evaluation_inputs" / "manifest.json"
manifest, completion, inputs = map(read, (manifest_path, completion_path, input_path))
pages = {r["request_id"]: r for r in manifest["pages"]}
images = {r["request_id"]: r for r in completion["images"]}
expected = set(pages)
assert len(expected) == len(manifest["pages"]) == len(images) == 48
assert expected == set(images) == {r["request_id"] for r in inputs["requests"]}
for rid, im in images.items():
    assert sha(ROOT / "gallery" / OUT.name / (rid + ".png")) == im["image_sha256"]

merged, comparisons, shards = [], [], []
for i in range(1, 4):
    folder = OUT / f"part_{i:02d}"
    rp, cp = folder / "review.jsonl", folder / "raw_post_comparison.jsonl"
    rr, cc = rows(rp), rows(cp)
    assigned = set(sorted(expected)[(i - 1) * 16:i * 16])
    assert len(rr) == 64 and len(cc) == 32
    assert {(r["request_id"], r["label"], r["stage"]) for r in rr} == {
        (rid, label, stage) for rid in assigned for label in LABELS for stage in ("RAW", "POST")}
    assert {(r["request_id"], r["label"]) for r in cc} == {
        (rid, label) for rid in assigned for label in LABELS}
    for r in rr:
        assert r["actual_image_viewed"] and r["reviewer"] == "AI_PROXY" and not r["human_reviewed"]
        assert not r["metric_values_used_for_labels"]
        assert r["exact_shared_connectivity"] == "NOT_ASSESSED_FROM_IMAGES"
        assert r["topologically_disconnected_fragment"] == "UNKNOWN_IMAGE_ONLY"
        assert r["image_sha256"] == images[r["request_id"]]["image_sha256"]
        assert r["gallery_definition_sha256"] == completion["definition_sha256"]
        for field in ("condition_id", "parent_id", "split", "K", "N"):
            assert r[field] == pages[r["request_id"]][field]
        merged.append(dict(r, review_source_shard=folder.name, review_source_file_sha256=sha(rp)))
    for r in cc:
        assert r["actual_image_viewed"] and r["reviewer"] == "AI_PROXY"
        assert r["image_sha256"] == images[r["request_id"]]["image_sha256"]
        comparisons.append(dict(r, review_source_shard=folder.name, review_source_file_sha256=sha(cp)))
    files = [rp, cp, folder / "receipt.json", folder / "review_schema.json", folder / "REVIEW.md"]
    shards.append(dict(shard=folder.name, requests=16, stage_rows=64, comparison_rows=32,
        files={str(p.relative_to(ROOT)): sha(p) for p in files}))

merged.sort(key=lambda r: (r["request_id"], r["label"], r["stage"]))
comparisons.sort(key=lambda r: (r["request_id"], r["label"]))
assert len(merged) == len({(r["request_id"], r["label"], r["stage"]) for r in merged}) == 192
assert len(comparisons) == len({(r["request_id"], r["label"]) for r in comparisons}) == 96
write(OUT / "review.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
write(OUT / "raw_post_comparison.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comparisons))
counts = Counter((r["label"], r["stage"], r["split"], key, r[key]) for r in merged for key in (
    "C_visual_coherent_use", "construction_intended_role_visible", "body_completeness", "thin_or_dangling_sheet_appearance"))
with (OUT / "observed_label_counts.csv").open("x", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("label", "stage", "split", "observation", "value", "count"))
    writer.writerows((*k, v) for k, v in sorted(counts.items()))
receipt = dict(status="COMPLETE_48_REQUESTS_192_STAGE_ROWS", created_utc=datetime.now(timezone.utc).isoformat(),
    labels=LABELS, requests=48, stage_rows=192, comparison_rows=96,
    original_records_changed=0, added_provenance_fields_only=True, missing_requests=[], duplicate_requests=[],
    all_actual_image_hashes_match=True, shards=shards,
    gallery_manifest_sha256=sha(manifest_path), gallery_completion_sha256=sha(completion_path),
    evaluation_input_manifest_sha256=sha(input_path), model_calls=0, rescoring=0, new_rendering=0,
    review_scope="Nonblind AI_PROXY split assignment; not human review or repeated independent raters. Image-only topology UNKNOWN. Counts describe observations, not validated task-success metrics.",
    output_hashes={p.name: sha(p) for p in (OUT / "review.jsonl", OUT / "raw_post_comparison.jsonl", OUT / "observed_label_counts.csv")})
write(OUT / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ("status", "requests", "stage_rows", "comparison_rows", "original_records_changed")}))
