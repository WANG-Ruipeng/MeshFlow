"""Mechanically combine two sealed, disjoint anonymous reviews; no relabelling."""
from datetime import datetime, timezone
from . import run as A

C = A.core


def run():
    base = A.DEFAULT / "blind_review/review_records"
    paths = []
    for part in (1, 2):
        found = list(base.glob(f"*/review_part{part}.json"))
        if len(found) != 1:
            return dict(status="WAITING_FOR_TWO_SEALED_PARTS",part=part,files=len(found))
        paths.append(found[0])
    docs = [C.read(p) for p in paths]
    protocol = A.DEFAULT / "blind_review/reviewer/review_protocol.md"
    records = []
    for i, (p, doc) in enumerate(zip(paths, docs)):
        assert doc["reviewer_kind"] == "AI_PROXY" and doc["blind"] is True
        assert doc["protocol_sha256"] == C.sha(protocol)
        rows = doc["records"]
        expected = {(f"page_{j:03d}", col) for j in range(i*16+1, i*16+17) for col in ("A", "B")}
        assert len(rows) == 32 and {(r["page_id"],r["column"]) for r in rows} == expected
        records.extend(rows)
    target = base / "review.json"
    source_hashes = {str(p): C.sha(p) for p in paths}
    merged = dict(schema="TWO_DISJOINT_ANONYMOUS_REVIEWERS_V1",reviewer_kind="AI_PROXY",
        blind=True,identity_blind=True,prior_exposure="NONE_REPORTED_BY_BOTH_REVIEWERS",
        protocol_sha256=C.sha(protocol),source_review_hashes=source_hashes,
        reviewer_metadata=[{k:v for k,v in d.items() if k != "records"} for d in docs],
        merge_only=True,labels_changed=0,records=sorted(records,key=lambda r:(r["page_id"],r["column"])))
    C.write_json(target,merged)
    A.verify_hashes(source_hashes)
    return dict(status="COMPLETE",records=64,pages=32,review_sha256=C.sha(target),labels_changed=0)


if __name__ == "__main__":
    import json
    print(json.dumps(run(),ensure_ascii=False))
