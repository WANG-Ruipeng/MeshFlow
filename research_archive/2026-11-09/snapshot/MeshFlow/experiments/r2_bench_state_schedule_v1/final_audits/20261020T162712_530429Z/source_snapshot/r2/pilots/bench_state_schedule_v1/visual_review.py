"""Register and validate actual AI_PROXY observations; never create ratings.

This module uses only the Python standard library. Preparing the review does
not require generated meshes or images, import a model, score, or render.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import io
import json
from pathlib import Path

from .common import OLD, OUT, atomic_write, read, sha, write_new

REVIEW = OUT / "visual_review"
REVIEWERS = ("root", "training", "eval")
FINAL = ("A_S1500", "PERTURBED_STATE_S1500", "ASYNC_LEAD_SEAM_S1500")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
SCHEMA = "BENCH_STATE_SCHEDULE_ACTUAL_AI_PROXY_REVIEW_V1"


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("Frozen visual-review definition differs: " + str(path))
    else:
        write_new(path, value)


def cell_key(row):
    return row["request_id"], row["model"], row["stage"]


def prepare():
    inherited_path = OLD / "visual_review/protocol.json"
    inherited = read(inherited_path)
    scope_path, manifest_path = OUT / "registration.json", OUT / "eval_manifest.json"
    scope, manifest = read(scope_path), read(manifest_path)
    qq = manifest["requests"]
    ids = [q["request_id"] for q in qq]
    if len(ids) != 48 or len(set(ids)) != 48:
        raise ValueError("Expected original 48 unique requests")
    if set(scope["evaluation_labels"]) != set(FINAL[1:]):
        raise ValueError("Registered model labels differ")
    if scope["expected_new_raw"] != 96 or scope["reused_baseline"]["label"] != FINAL[0]:
        raise ValueError("Expected 96 new outputs and read-only historical A")
    protected = {str(p): sha(p) for p in (inherited_path, scope_path, manifest_path)}
    protocol = dict(
        schema=SCHEMA, reviewer_kind="AI_PROXY", blind=False,
        request_count=48, output_count=144, new_output_count=96,
        historical_output_count=48, expected_cell_records=432,
        expected_image_sheets=144, final_cells_per_request=9,
        process_request_count=0,
        unit=inherited["unit"], axes=copy.deepcopy(inherited["axes"]),
        required_text=inherited["required_text"], limits=copy.deepcopy(inherited["limits"]),
        inherited_protocol_path=str(inherited_path),
        inherited_protocol_sha256=sha(inherited_path),
        assignment_policy="Original frozen eval_manifest order; contiguous 16 requests for root, training, eval",
        record_envelope=dict(
            required_fields=["schema", "reviewer", "reviewer_kind", "blind",
                             "protocol_sha256", "assignments_sha256", "rows"],
            reviewer_values=list(REVIEWERS),
            row_required_fields=["request_id", "model", "stage", "image_sha256",
                                 "actually_viewed", "views_and_sections_checked",
                                 *inherited["axes"], "notes"],
            assertions="actually_viewed=true and views_and_sections_checked=true only after inspecting the actual image",
            files="Save actual observations as visual_review/review_<reviewer>[_part].json; no filled rating templates are created",
        ),
        completion_policy="Missing valid rows => INCOMPLETE; malformed, duplicate, unassigned or unauthenticated rows => validation error",
        rendering_is_review=False, formal_human_ratings=0,
        historical_A_policy="Original meshes/scores reused; all three displayed columns receive fresh actual image observations. No old ratings are copied.",
        protected=protected,
    )
    assignments = {
        reviewer: ids[16 * i:16 * (i + 1)]
        for i, reviewer in enumerate(REVIEWERS)
    }
    owner = {rid: reviewer for reviewer, assigned in assignments.items() for rid in assigned}
    cells = []
    for q in qq:
        for stage in STAGES:
            page_id = stage + "__" + q["request_id"]
            for label in FINAL:
                cells.append(dict(
                    **{k: q[k] for k in ("request_id", "condition_id", "parent_id",
                                         "noise_index", "split", "part_type")},
                    reviewer=owner[q["request_id"]], model=label, stage=stage,
                    phase="FINAL", sheet=stage, page_id=page_id,
                    image=str(OUT / "gallery" / (page_id + ".png")),
                    image_receipt=str(OUT / "gallery" / (page_id + ".json")),
                ))
    keys = [cell_key(c) for c in cells]
    if len(cells) != 432 or len(set(keys)) != 432 or len({c["page_id"] for c in cells}) != 144:
        raise ValueError("Visual-review cell/sheet accounting differs")
    assignment_document = dict(
        schema=SCHEMA, policy=protocol["assignment_policy"],
        assignments=assignments, expected_total_cells=432,
        expected_cells_by_reviewer=dict(collections.Counter(c["reviewer"] for c in cells)),
        protected=protected,
    )
    immutable(REVIEW / "protocol.json", protocol)
    immutable(REVIEW / "assignments.json", assignment_document)
    immutable(REVIEW / "expected_cells.json", dict(
        schema=SCHEMA, status="AWAITING_ACTUAL_OBSERVATIONS", rows=cells,
        protocol_sha256=sha(REVIEW / "protocol.json"),
        assignments_sha256=sha(REVIEW / "assignments.json"),
        expected_cells=432, expected_sheets=144, generated_ratings=0,
    ))
    return dict(status="PREPARED_NO_RATINGS", requests=48, cells=432, sheets=144,
                assignments=assignment_document["expected_cells_by_reviewer"],
                protocol=str(REVIEW / "protocol.json"),
                protocol_sha256=sha(REVIEW / "protocol.json"),
                assignments_sha256=sha(REVIEW / "assignments.json"),
                model_calls=0, scoring_calls=0, rendering_calls=0)


def authenticate_image(expected, row):
    image, receipt_path = Path(expected["image"]), Path(expected["image_receipt"])
    actual = sha(image)
    if not isinstance(row.get("image_sha256"), str) or row["image_sha256"] != actual:
        raise ValueError("Reviewed image SHA differs: " + expected["page_id"])
    receipt = read(receipt_path)
    if receipt.get("image_sha256") != actual or receipt.get("page_id") != expected["page_id"]:
        raise ValueError("Renderer receipt does not authenticate reviewed image")
    if receipt.get("request_id") != expected["request_id"] or receipt.get("condition_id") != expected["condition_id"]:
        raise ValueError("Renderer receipt request identity differs")
    if not any(c.get("model") == expected["model"] and c.get("stage") == expected["stage"]
               for c in receipt.get("cells", [])):
        raise ValueError("Reviewed model/stage absent from image receipt")
    definition = OUT / "gallery/definition.json"
    if receipt.get("definition_sha256") != sha(definition):
        raise ValueError("Image belongs to another gallery definition")
    return {str(image): actual, str(receipt_path): sha(receipt_path)}


def csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def collect():
    prepared = prepare()
    protocol = read(REVIEW / "protocol.json")
    expected = {cell_key(r): r for r in read(REVIEW / "expected_cells.json")["rows"]}
    rows, observed, input_hashes, image_hashes = [], set(), {}, {}
    for path in sorted(REVIEW.glob("review_*.json")):
        document = read(path)
        reviewer = document.get("reviewer")
        if (document.get("schema") != SCHEMA or reviewer not in REVIEWERS
                or document.get("reviewer_kind") != "AI_PROXY" or document.get("blind") is not False):
            raise ValueError("Unknown review envelope/reviewer: " + str(path))
        if (document.get("protocol_sha256") != prepared["protocol_sha256"]
                or document.get("assignments_sha256") != prepared["assignments_sha256"]):
            raise ValueError("Review belongs to another protocol or assignment")
        if not isinstance(document.get("rows"), list):
            raise ValueError("Actual observation rows must be a JSON list")
        input_hashes[str(path)] = sha(path)
        for index, record in enumerate(document["rows"]):
            if not isinstance(record, dict):
                raise ValueError("Observation must be an object")
            key = cell_key(record)
            if key not in expected or key in observed:
                raise ValueError("Unknown or duplicate review cell: " + repr(key))
            target = expected[key]
            if target["reviewer"] != reviewer:
                raise ValueError("Reviewer does not own this request: " + repr(key))
            if record.get("actually_viewed") is not True or record.get("views_and_sections_checked") is not True:
                raise ValueError("Actual inspection of views and sections must be declared")
            for axis, values in protocol["axes"].items():
                if record.get(axis) not in values:
                    raise ValueError("Missing/invalid observation axis " + axis + " for " + repr(key))
            notes = record.get("notes")
            if not isinstance(notes, str) or not notes.strip():
                raise ValueError("Concrete visual notes are required: " + repr(key))
            image_hashes.update(authenticate_image(target, record))
            row = dict(target, reviewer_kind="AI_PROXY", blind=False, actually_viewed=True,
                       views_and_sections_checked=True, image_sha256=record["image_sha256"],
                       **{axis: record[axis] for axis in protocol["axes"]}, notes=notes.strip(),
                       observation_file=str(path), observation_index=index)
            rows.append(row)
            observed.add(key)
    rows.sort(key=lambda r: cell_key(r))
    missing = [expected[key] for key in sorted(set(expected) - observed)]
    complete = len(rows) == 432 and not missing
    result = dict(
        schema=SCHEMA, status="ACTUAL_AI_PROXY_REVIEW_COMPLETE" if complete else "INCOMPLETE",
        expected_cells=432, reviewed_cells=len(rows), missing_cells=len(missing),
        expected_requests=48, reviewed_requests=len({r["request_id"] for r in rows}),
        reviewed_by_reviewer=dict(collections.Counter(r["reviewer"] for r in rows)),
        protocol_sha256=prepared["protocol_sha256"],
        assignments_sha256=prepared["assignments_sha256"],
        blind=False, reviewer_kind="AI_PROXY", formal_human_ratings=0,
        source_review_files=input_hashes, authenticated_image_files=image_hashes,
        model_calls=0, scoring_calls=0, rendering_calls=0,
        warning="Actual nonblind AI observations; not human ratings, topology certification or unique-source reconstruction scores.",
    )
    atomic_write(REVIEW / "missing_cells.json", dict(status=result["status"], rows=missing))
    atomic_write(REVIEW / "validated_records.json", dict(status=result["status"], rows=rows))
    # The partial table contains actual submitted observations only, never NA
    # placeholders for unseen images. A final table exists only at completion.
    if rows:
        csv_path = REVIEW / ("per_cell.csv" if complete else "per_cell.partial.csv")
        fields = list(rows[0])
        text = csv_text(rows, fields)
        temp = csv_path.with_name(csv_path.name + ".partial_write")
        with temp.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
        temp.replace(csv_path)
        result.update(per_cell_csv=str(csv_path), per_cell_sha256=sha(csv_path))
    atomic_write(REVIEW / "completion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "aggregate"))
    args = parser.parse_args()
    value = prepare() if args.action == "prepare" else collect()
    print(json.dumps({k: v for k, v in value.items()
                      if k not in ("source_review_files", "authenticated_image_files")},
                     ensure_ascii=False, indent=2))
