"""Freeze and aggregate actual AI_PROXY review; never invent visual ratings."""
from __future__ import annotations
import argparse
import csv
import json
from .common import OUT, read, sha, atomic_write, write_new, resolve

PROTOCOL = dict(
    schema="BENCH_CLEAN_ACTUAL_AI_PROXY_REVIEW_V1",
    reviewer_kind="AI_PROXY", blind=False, request_count=48, sheets_per_request=2,
    cells_per_request=12, expected_cell_records=576,
    unit="One model/stage cell, inspected in both inherited views and both central sections",
    axes={
        "overall_shape": ["REASONABLE", "INCOMPLETE", "UNCERTAIN"],
        "C_use": ["PLAUSIBLE", "REPLACED_OR_IGNORED", "UNCERTAIN", "NA"],
        "support_retention": ["PRESENT", "MISSING_OR_DAMAGED", "UNCERTAIN", "NA"],
        "visible_fragment_or_gap": ["YES", "NO_OBVIOUS", "UNCERTAIN"],
        "layering_or_fold": ["YES", "NO_OBVIOUS", "UNCERTAIN"],
    },
    required_text="notes: concrete visual evidence; do not infer legal topology or intersection counts from image",
    limits=[
        "No unique source style or exact source shape is used as the completion target",
        "Section endpoints alone do not prove physical cracks",
        "Small projected triangles do not automatically count as detached fragments",
        "Uncertain or occluded appearance remains UNCERTAIN",
        "Numerical connectivity and illegal-intersection evidence are reported separately",
        "Rendering and numeric metric extraction never count as actual image review",
    ],
)


def prepare():
    p = OUT / "visual_review/protocol.json"
    if p.exists():
        assert read(p) == PROTOCOL
    else:
        write_new(p, PROTOCOL)
    assignment_path = OUT / "visual_review/assignments.json"
    qq = read(OUT / "eval_manifest.json")["requests"]
    reviewers = ("root", "bench_clean_training", "bench_clean_evaluation")
    assignment = {
        "policy": "Frozen manifest order, contiguous16 requests each; both sheets and all12 cells per request",
        "assignments": {
            reviewer: [r["request_id"] for r in qq[16*i:16*(i+1)]]
            for i, reviewer in enumerate(reviewers)
        },
    }
    if assignment_path.exists():
        assert read(assignment_path) == assignment
    else:
        write_new(assignment_path, assignment)
    return PROTOCOL


def aggregate():
    prepare()
    manifest = read(OUT / "gallery_manifests/BENCH_CLEAN_FIXED48_MAIN_AUX.json")
    image_root = OUT / "gallery/BENCH_CLEAN_FIXED48_MAIN_AUX"
    expected = {}
    for page in manifest["pages"]:
        for cell in page["cells"]:
            key = (page["request_id"], cell["model"], cell["stage"])
            expected[key] = page
    assert len(expected) == 576
    rows = []
    for p in sorted((OUT / "visual_review").glob("review_*.json")):
        r = read(p)
        assert r["reviewer_kind"] == "AI_PROXY" and r["blind"] is False
        for row in r["rows"]:
            row = dict(row, reviewer=r["reviewer"], reviewer_kind="AI_PROXY", blind=False)
            key = (row["request_id"], row["model"], row["stage"])
            page = expected[key]
            image = image_root / (page["page_id"] + ".png")
            assert row["image_sha256"] == sha(image)
            assert row["actually_viewed"] is True and row["views_and_sections_checked"] is True
            for axis, values in PROTOCOL["axes"].items():
                assert row[axis] in values
            assert row["notes"].strip()
            row.update(sheet=page["sheet"], split=page["split"], part_type=page["part_type"],
                       parent_id=page["parent_id"], condition_id=page["condition_id"],
                       noise_index=page["noise_index"], image=str(image))
            rows.append(row)
    keys = [(r["request_id"], r["model"], r["stage"]) for r in rows]
    assert len(keys) == 576 and len(set(keys)) == 576 and set(keys) == set(expected)
    rows.sort(key=lambda r: (r["request_id"], r["sheet"], r["model"], r["stage"]))
    out = OUT / "visual_review/per_cell.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="ACTUAL_AI_PROXY_REVIEW_COMPLETE", blind=False,
                  requests=48, image_sheets=96, cells=576, formal_human_ratings=0,
                  files={str(p): sha(p) for p in sorted((OUT / "visual_review").glob("review_*.json"))},
                  per_cell_csv=str(out), per_cell_sha256=sha(out),
                  warning="Nonblind AI visual observations, not topology certification or human evaluation.")
    atomic_write(OUT / "visual_review/completion.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "aggregate"))
    a = p.parse_args()
    print(json.dumps(prepare() if a.action == "prepare" else aggregate(), ensure_ascii=False, indent=2))
