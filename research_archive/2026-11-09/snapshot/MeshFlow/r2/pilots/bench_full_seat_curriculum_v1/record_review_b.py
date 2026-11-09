"""Serialize this reviewer's actual image observations; no inference or rendering."""
from __future__ import annotations
import collections
import csv
import json
from . import visual_review as v


def main():
    src = v.REVIEW / "observations_b_actual.json"
    doc = v.read(src)
    expected_doc = v.read(v.REVIEW / "expected_cells.json")
    cells = [r for r in expected_doc["rows"] if r["review_group"] == "b_multiface_and_dev"]
    observations = {(r["cid"], r["noise"]): r for r in doc["cases"]}
    assert len(observations) == 8 and len(cells) == 72
    names = ("overall", "C_use", "support", "visible_fragment_or_gap", "layering_or_fold")
    axis_names = tuple(v.AXES)
    rows = []
    for c in cells:
        item = observations[(c["condition_id"], c["noise_index"])]
        code, note = item["stages"][c["stage"]][v.LABELS.index(c["label"])]
        assert len(code) == 5
        row = {k:c.get(k, "") for k in v.FIELDS}
        row.update(schema=v.SCHEMA,
            protocol_sha256=v.sha(v.REVIEW / "protocol.md"),
            assignments_sha256=expected_doc["assignments_sha256"],
            reviewer="newdata_sampling", reviewer_kind="AI_PROXY", blind="false",
            review_round="1", view_event_id="B_ACTUAL_VIEW_" + c["page_id"],
            viewed_at_utc=item["viewed"], image_sha256=v.sha(c["image_path"]),
            image_receipt_sha256=v.sha(c["image_receipt"]),
            gallery_definition_sha256=v.sha(v.OUT / "gallery/definition.json"),
            actual_view_source=doc["actual_view_source"], actually_viewed="true",
            views_and_sections_checked="true", note=note)
        row.update({axis:doc["legend"][key][value] for axis,key,value in zip(axis_names,names,code)})
        rows.append(row)
    path = v.REVIEW / "review_b_multiface_and_dev.csv"
    output = v.csv_text(rows, v.FIELDS)
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if existing and existing not in (v.csv_text([], v.FIELDS), output):
        raise ValueError("Refusing to replace prior actual review evidence")
    if existing != output:
        path.write_text(output, encoding="utf-8", newline="")
    counts = collections.Counter(r["page_id"] for r in rows)
    assert len(counts) == 24 and set(counts.values()) == {3}
    receipt = dict(status="COMPLETE_SHARD_B_ACTUAL_AI_PROXY_REVIEW", pages=24, cells=72,
        manual_observations_sha256=v.sha(src), csv_sha256=v.sha(path),
        source_sha256=v.sha(__file__), protocol_sha256=v.sha(v.REVIEW / "protocol.md"),
        source="Explicit reviewer labels after tools.view_image; no numerical metric inputs",
        model_calls=0, scoring_calls=0, rendered_images=0, generated_ratings=0)
    v.immutable(v.REVIEW / "review_b_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
