"""Append explicit, actually viewed B-shard observations; never infer labels."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from r2.pilots.bench_newdata_joint_v1 import visual_review as v


def main():
    batch = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    task = v.read(v.REVIEW / "assignments.json")["tasks"]["b_train_and_backdev"]
    pages = {p["page_id"]: p for p in task["expected_pages"]}
    path = v.resolve(task["records_csv"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        prior = list(csv.DictReader(handle))
    keys = {v.cell_key(r) for r in prior}
    rows = []
    for observation in batch:
        page = pages[observation["page_id"]]
        assert len(observation["cells"]) == 2
        for label, observed in zip(v.LABELS, observation["cells"]):
            assert set(observed) == set(v.AXES) | {"note"}
            for axis, values in v.AXES.items():
                assert observed[axis] in values
            assert observed["note"].strip()
            row = {k: page[k] for k in ("review_group", "request_id", "condition_id", "parent_id", "noise_index", "eval_group", "split", "part_type", "stage", "page_id", "image_path")}
            row.update(schema=v.SCHEMA, protocol_sha256=v.sha(v.REVIEW / "protocol.md"),
                       assignments_sha256=v.sha(v.REVIEW / "assignments.json"),
                       reviewer="newdata_eval_panel", reviewer_kind="AI_PROXY", blind="false",
                       review_round="B_ACTUAL_VIEW_ROUND_1", view_event_id=observation["view_event_id"],
                       viewed_at_utc=observation["viewed_at_utc"], label=label,
                       image_sha256=v.sha(page["image_path"]),
                       image_receipt_sha256=v.sha(page["image_receipt"]),
                       gallery_definition_sha256=v.sha(v.OUT / "gallery/definition.json"),
                       actual_view_source="functions.exec tools.view_image original full comparison PNG",
                       actually_viewed="true", views_and_sections_checked="true", **observed)
            assert v.cell_key(row) not in keys, "Refusing duplicate or replacement observation"
            keys.add(v.cell_key(row))
            rows.append(row)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=v.FIELDS, lineterminator="\n").writerows(rows)
    print(json.dumps({"appended_cells": len(rows), "B_total_cells": len(prior) + len(rows)}))


if __name__ == "__main__":
    main()
