"""Read-only CPU verification of published score collection and source hashes."""
from __future__ import annotations
import collections
import csv
import json
from . import visual_review as v


def main():
    completion = v.read(v.OUT / "results/completion.json")
    registration = v.read(v.OUT / "evaluation/score_registration.json")
    hashes = {**completion["files"], **completion["source_rows"], **registration["source_hashes"]}
    for path, expected in hashes.items():
        assert v.sha(path) == expected, path
    assert v.sha(v.ROOT / "r2/pilots/bench_full_seat_curriculum_v1/collect.py") == completion["collector_sha256"]
    assert v.sha(v.OUT / "eval_manifest.json") == registration["eval_manifest_sha256"]
    assert v.sha(v.OUT / "registration.json") == registration["registration_sha256"]
    with (v.OUT / "results/per_output.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == completion["rows"] == completion["new_score_rows"] == 216
    assert completion["reused_score_rows"] == 0
    keys = {(r["request_id"], r["label"], r["stage"]) for r in rows}
    qs = v.read(v.OUT / "eval_manifest.json")["requests"]
    expected = {(q["request_id"], label, stage) for q in qs for label in v.LABELS for stage in v.STAGES}
    assert len(keys) == 216 and keys == expected
    assert all(r["C_preserved_exact"] == "True" for r in rows)
    assert all(r["seam_applicability"] == "NA_NOT_SEAM_REPRESENTATION" for r in rows if r["label"] != v.LABELS[2])
    counts = collections.Counter((r["label"], r["stage"]) for r in rows)
    assert set(counts.values()) == {24}
    result = dict(status="PASS", collected_rows=216, request_model_stage_unique=216,
        C_preserved_exact_true=216, native_seam_NA_rows=144,
        scoring_source_hashes_verified=len(registration["source_hashes"]),
        score_and_row_hashes_verified=len(completion["source_rows"]),
        collected_table_hashes_verified=len(completion["files"]),
        seam_applicability_counts=dict(collections.Counter(r["seam_applicability"] for r in rows)),
        counts={a+":"+b:n for (a,b),n in sorted(counts.items())},
        model_calls=0, scoring_calls=0, modified_existing_results=0,
        source_sha256=v.sha(__file__), completion_sha256=v.sha(v.OUT / "results/completion.json"),
        scoring_registration_sha256=v.sha(v.OUT / "evaluation/score_registration.json"))
    v.immutable(v.OUT / "engineering/collected_evidence_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
