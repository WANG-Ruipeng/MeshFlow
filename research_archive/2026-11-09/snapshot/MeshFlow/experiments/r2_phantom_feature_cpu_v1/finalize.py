"""Verify frozen inputs and archive this CPU-only result; no scoring or models."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                       encoding="utf-8")


manifest = read(OUT / "manifest.json")
expected = dict(manifest["input_manifests"])
for j in manifest["jobs"]:
    for path_key, hash_key in (("path", "sha256"), ("raw_path", "raw_sha256")):
        path, digest = j[path_key], j[hash_key]
        assert digest and expected.get(path, digest) == digest, path
        expected[path] = digest
labels = read(OUT / "label_findings.json")
root = Path("/mnt/e/MeshFlow")
for rel, digest in labels["input_hashes"].items():
    expected[str(root / rel)] = digest

checks = []
for path, digest in sorted(expected.items()):
    p = Path(path)
    actual = sha(p) if p.is_file() else None
    checks.append({"path": path, "expected": digest, "actual": actual,
                   "status": "PASS" if actual == digest else "FAIL"})
assert all(c["status"] == "PASS" for c in checks), "Frozen source hash changed"

cells = [read(OUT / "cells" / (j["job_id"] + ".json")) for j in manifest["jobs"]]
assert len(cells) == 282
assert all(c["status"] == "COMPLETE" and c["job"] == j
           for c, j in zip(cells, manifest["jobs"]))
assert all(c["model_calls"] == 0 and not c["torch_imported"] for c in cells)
assert len(read(OUT / "analytic_checks.json")) == 12
assert read(OUT / "independent_audit.json")["status"] == "PASS"
assert read(OUT / "independent_audit.json")["source_sha256"] == sha(OUT / "run.py")

with (OUT / "per_output_with_review.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 282
selected = []
for r in rows:
    rid = r["request_id"]
    tags = []
    if r["model"] == "NEW_HEAD" and rid.startswith("1dd740") and "__SEAT_" in rid:
        tags.append("SAME_C_ADOPTION_CONTRAST")
    if r["panel"] == "USER_PICTURE_APPENDIX":
        tags.append("USER_PICTURE_1" if r["model"] == "SEAM_S21000" else "USER_PICTURE_3")
    if r["model"] == "NEW_HEAD" and rid == "2d1e5eeb255e79dcab53bf7fa22f3636__LEG_SUPPORT_1__n0":
        tags.append("USER_PICTURE_2")
    if rid.startswith("122a480") and "__LEG_" in rid:
        tags.append("SECOND_SAME_C_CONTRAST")
    if tags:
        selected.append(dict(example=";".join(tags), **r))
with (OUT / "key_examples.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(selected[0]))
    w.writeheader()
    w.writerows(selected)

execution = read(OUT / "execution.json")
save(OUT / "completion.json", {
    "status": "COMPLETE", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "main_requests": 92, "main_cached_stage_cells": 276, "appendix_stage_cells": 6,
    "complete_cells": 282, "failed_cells": 0,
    "training_updates": 0, "model_forwards": 0, "model_backwards": 0,
    "GPU_jobs": 0, "new_generated_meshes": 0, "new_noise": 0,
    "new_surface_queries": 0, "original_assets_modified": False,
    "geometry_scan_seconds": execution["seconds"],
    "geometry_scan_workers": execution["dispatch"]["workers"],
    "analytic_checks_passed": 12, "independent_checks_passed": 13,
    "matched_old_stage_reviews": 225, "missing_old_stage_reviews": 57,
    "review_note": "Existing AI_PROXY labels reused; only highlighted galleries visually rechecked.",
    "input_checks": checks,
    "rows_by_domain_stage": dict(Counter(r["domain"] + "/" + r["stage"] for r in rows)),
    "decision": "Descriptors not validated as quality/adoption gate. No automatic training."
})
hashes = {str(p.relative_to(OUT)).replace("\\", "/"): sha(p)
          for p in sorted(OUT.rglob("*"))
          if p.is_file() and p.name != "artifact_hashes.json" and "__pycache__" not in p.parts}
save(OUT / "artifact_hashes.json", hashes)
print(json.dumps({"status": "COMPLETE", "cells": len(cells),
                  "verified_input_files": len(checks), "hashed_artifacts": len(hashes),
                  "key_example_rows": len(selected)}, ensure_ascii=False))
