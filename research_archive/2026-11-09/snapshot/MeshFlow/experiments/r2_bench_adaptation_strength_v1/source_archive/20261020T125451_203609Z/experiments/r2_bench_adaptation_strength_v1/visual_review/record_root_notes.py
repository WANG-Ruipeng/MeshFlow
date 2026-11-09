"""Serialize explicit human-readable Codex observations; never infer ratings."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).parent
G = R.parent / "gallery"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
axes = (
    ("overall_shape", dict(R="REASONABLE", I="INCOMPLETE", U="UNCERTAIN")),
    ("C_use", dict(P="PLAUSIBLE", R="REPLACED_OR_IGNORED", U="UNCERTAIN", N="NA")),
    ("support_retention", dict(P="PRESENT", M="MISSING_OR_DAMAGED", U="UNCERTAIN", N="NA")),
    ("visible_fragment_or_gap", dict(Y="YES", N="NO_OBVIOUS", U="UNCERTAIN")),
    ("layering_or_fold", dict(Y="YES", N="NO_OBVIOUS", U="UNCERTAIN")),
)
rows = []
seen = set()
for line in (R / "root_manual_notes.jsonl").read_text(encoding="utf-8-sig").splitlines():
    note = json.loads(line)
    page = note["page"]
    receipt = json.loads((G / (page + ".json")).read_text())
    assert sha(G / (page + ".png")) == receipt["image_sha256"]
    assert len(note["cells"]) == len(receipt["cells"])
    for cell, (code, text) in zip(receipt["cells"], note["cells"]):
        assert len(code) == 5 and text.strip()
        key = (receipt["request_id"], cell["model"], cell["stage"])
        assert key not in seen
        seen.add(key)
        rows.append(dict(
            request_id=key[0], model=key[1], stage=key[2],
            image_sha256=receipt["image_sha256"],
            actually_viewed=True, views_and_sections_checked=True,
            **{axis: choices[c] for c, (axis, choices) in zip(code, axes)},
            notes=text,
        ))
document = dict(
    schema="BENCH_ADAPTATION_ACTUAL_AI_PROXY_REVIEW_V1",
    reviewer="root", reviewer_kind="AI_PROXY", blind=False,
    protocol_sha256=sha(R / "protocol.json"),
    assignments_sha256=sha(R / "assignments.json"), rows=rows,
)
(R / "review_root.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
print("Explicit actual root observations:", len(rows))
