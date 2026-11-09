"""Encode manually supplied observations after actual image inspection."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
axes = [
    ("overall_shape", {"R": "REASONABLE", "I": "INCOMPLETE", "U": "UNCERTAIN"}),
    ("C_use", {"P": "PLAUSIBLE", "R": "REPLACED_OR_IGNORED", "U": "UNCERTAIN", "N": "NA"}),
    ("support_retention", {"P": "PRESENT", "M": "MISSING_OR_DAMAGED", "U": "UNCERTAIN", "N": "NA"}),
    ("visible_fragment_or_gap", {"Y": "YES", "N": "NO_OBVIOUS", "U": "UNCERTAIN"}),
    ("layering_or_fold", {"Y": "YES", "N": "NO_OBVIOUS", "U": "UNCERTAIN"}),
]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def save(part, observations):
    assignments = read(ROOT / "visual_review/assignments.json")
    allowed = set(assignments["assignments"]["eval"])
    rows = []
    for page, ratings in observations:
        meta = read(ROOT / "gallery" / (page + ".json"))
        assert meta["request_id"] in allowed
        image = ROOT / "gallery" / (page + ".png")
        assert sha(image) == meta["image_sha256"]
        assert len(ratings) == len(meta["cells"])
        for cell, (codes, note) in zip(meta["cells"], ratings):
            assert len(codes) == 5 and note
            row = dict(request_id=meta["request_id"], model=cell["model"],
                       stage=cell["stage"], image_sha256=meta["image_sha256"],
                       actually_viewed=True, views_and_sections_checked=True,
                       notes=note)
            row.update({key: choices[code] for (key, choices), code in zip(axes, codes)})
            rows.append(row)
    result = dict(schema="BENCH_ADAPTATION_ACTUAL_AI_PROXY_REVIEW_V1",
                  reviewer="eval", reviewer_kind="AI_PROXY", blind=False,
                  protocol_sha256=sha(ROOT / "visual_review/protocol.json"),
                  assignments_sha256=sha(ROOT / "visual_review/assignments.json"),
                  rows=rows)
    path = ROOT / "visual_review" / ("review_eval_" + part + ".json")
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(str(path), len(rows))
if __name__ == "__main__":
    payload = json.load(sys.stdin)
    save(payload["part"], payload["observations"])
