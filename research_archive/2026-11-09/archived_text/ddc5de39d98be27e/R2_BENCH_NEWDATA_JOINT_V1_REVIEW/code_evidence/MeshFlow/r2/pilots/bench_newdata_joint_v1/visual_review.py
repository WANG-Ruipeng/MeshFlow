"""Prepare empty actual-view CSVs; validate and summarize submitted observations.

Standard library only. No model, geometry scoring, rendering or inferred labels.
Review inputs are read-only; aggregate snapshots use a content-derived directory.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_newdata_joint_v1"
REVIEW = OUT / "visual_review"
SCHEMA = "BENCH_NEWDATA_ACTUAL_AI_PROXY_REVIEW_V1"
LABELS = ("RELATION_V1_S1500", "NEWDATA_JOINT_S1500")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
GROUPS = ("CURRENT_TRAIN", "SAME_PARENT_HELDOUT", "DEV")
AXES = {
    "overall_shape": ("REASONABLE", "INCOMPLETE", "UNCERTAIN"),
    "C_use": ("PLAUSIBLE", "REPLACED_OR_IGNORED", "UNCERTAIN"),
    "support_retention": ("PRESENT", "MISSING_OR_DAMAGED", "UNCERTAIN"),
    "visible_fragment_or_gap": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
    "layering_or_fold": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
}
SHARDS = {
    "a_new_and_flat": ("27eb79dd", "35948e4d"),
    "b_train_and_backdev": ("4cd23aae", "5f0e28d1", "2d1e5eeb"),
    "c_curved_and_dev": ("943dde27", "459f90aa", "9fd5b67a", "fd355d06"),
}
FIELDS = (
    "schema", "protocol_sha256", "assignments_sha256", "review_group", "reviewer", "reviewer_kind", "blind",
    "review_round", "view_event_id", "viewed_at_utc", "request_id", "condition_id", "parent_id", "noise_index",
    "eval_group", "split", "part_type", "label", "stage", "page_id", "image_path", "image_sha256",
    "image_receipt_sha256", "gallery_definition_sha256", "actual_view_source", "actually_viewed",
    "views_and_sections_checked", *AXES, "note",
)
PROTOCOL = """# Bench new-data actual image review

This protocol retains the exact five-axis vocabulary of the preceding RELATION
and SHIFT reviews. It is a nonblind AI_PROXY review, not an independent human
study. The panel is 28 requests x 3 stages = 84 paired pages and 168 model-stage
cells. Columns are RELATION_V1_S1500 then NEWDATA_JOINT_S1500.

Inspect each full comparison PNG, both model columns, both azimuth -60/120 views
at elevation23 and both fixed front/side sections. Record both columns in the
same review round/view event. Contact sheets can aid navigation but cannot
replace full-page inspection. Blue is C, orange SEAM, gray BULK, green BRIDGE.
The source completion is not the unique correct style. Do not grade resemblance
to it, or use numerical metrics, head diagnostics, training curves, historical
ratings or column order to supply visual labels. Reused old DEV meshes still
require actual viewing in these new paired pages; old ratings are not copied.

- overall_shape: REASONABLE / INCOMPLETE / UNCERTAIN
- C_use: PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN
- support_retention: PRESENT / MISSING_OR_DAMAGED / UNCERTAIN
- visible_fragment_or_gap: YES / NO_OBVIOUS / UNCERTAIN
- layering_or_fold: YES / NO_OBVIOUS / UNCERTAIN

Judge axes separately. A complete-looking Bench can replace or ignore C.
UNCERTAIN covers occlusion, projection, finite-section ambiguity or ambiguous
input semantics; explain it in the note. NO_OBVIOUS is not absence certification.
Section endpoints cannot establish cracks, floating surfaces, intersections or
indexed connectivity. Visible folds/fragments are observations, not detector
results. Do not invent a joint-success score or trade C adoption against loss of
the main body or supports. Numeric topology/CF/FF/I and deleted faces remain
separate evidence.

CSV files are header-only at preparation. Append only genuinely inspected cells;
never prefill axes, image hashes, actually_viewed or observation notes. Every row
records the image hash, renderer receipt hash and gallery-definition hash, actual
view source, reviewer, UTC view timestamp, review round, view-event ID and a
concise image-grounded note. actually_viewed and views_and_sections_checked must
be true only after inspection. These are reviewer declarations tied to real
image files; hash validation alone does not prove that an image was inspected.
Save both columns of each viewed page in the same batch. Leave unviewed pages
absent; missing rows are pending, not failure, UNCERTAIN or NA. Never silently
replace original observations. Adjudication requires a separately registered
round and retained original rows; this collector does not merge duplicate cells.

CURRENT_TRAIN contains the nine selected training C (18 requests). The new
parent's R2 is SAME_PARENT_HELDOUT (2 requests), not an unseen-parent test. DEV
contains four held-out adaptation parents (8 requests), repeatedly used as a
development set; official pretraining novelty is not certified. Keep all three
groups and RAW / BASE_POST / CLEAN_FINAL_POST separate. Within a group, average
two noises within C, then conditions within parent, then weight parents equally.
Also retain raw counts and full category denominators including UNCERTAIN.

Assignments retain whole parents, both noises and all three stages: a_new_and_flat
has 24 pages/48 cells; b_train_and_backdev and c_curved_and_dev each have 30
pages/60 cells. R1 and R2 are viewed by the same reviewer. Shards can run in
parallel when reviewers are available; they do not create independent replicates.

The helper reads submitted review CSVs without editing them. It validates exact
cell identity, actual-view declarations, all five labels and real image receipts.
Incomplete reviews produce coverage/missing-cell records only. Numerical visual
summaries and paired/cross-stage transitions are emitted only after all 168
valid cells are present. No generation, rendering, model calls, rescoring, best
sample selection, extra samples or frozen-code changes are part of this review.
"""


def resolve(path):
    s = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def immutable_text(path, text):
    path = resolve(path)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise ValueError("Refusing to replace review evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def immutable_json(path, value):
    immutable_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_text(rows, fields=None):
    rows = list(rows)
    fields = list(fields or dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def cell_key(row):
    return row["request_id"], row["label"], row["stage"]


def prepare():
    source = OUT / "eval_manifest.json"
    manifest = read(source)
    requests = manifest["requests"]
    assert len(requests) == 28 and len({q["request_id"] for q in requests}) == 28
    protected = {str(source): sha(source)}
    inherited = [ROOT / "experiments/r2_bench_relation_training_v1/visual_review/REVIEW_PROTOCOL.md",
                 ROOT / "experiments/r2_bench_joint_shift_v1/visual_review/protocol.md"]
    protected.update({str(p): sha(p) for p in inherited})
    expected, tasks = [], {}
    for name, prefixes in SHARDS.items():
        qq = [q for q in requests if any(q["parent_id"].startswith(p) for p in prefixes)]
        pages = []
        for q in qq:
            for stage in STAGES:
                page_id = stage + "__" + q["request_id"]
                base = {k:q[k] for k in ("request_id", "condition_id", "parent_id", "noise_index", "eval_group", "split", "part_type")}
                page = dict(**base, review_group=name, stage=stage, page_id=page_id,
                    image_path=str(OUT / "gallery" / (page_id + ".png")),
                    image_receipt=str(OUT / "gallery" / (page_id + ".json")), status="PENDING_ACTUAL_VIEW")
                pages.append(page)
                expected.extend(dict(page, label=label) for label in LABELS)
        tasks[name] = dict(parent_ids=sorted({q["parent_id"] for q in qq}), request_ids=[q["request_id"] for q in qq],
                          request_count=len(qq), pages=len(pages), cells=2*len(pages),
                          records_csv=str(REVIEW / ("review_" + name + ".csv")), expected_pages=pages)
    assert len(expected) == 168 and len({cell_key(r) for r in expected}) == 168
    assert [tasks[n]["pages"] for n in SHARDS] == [24, 30, 30]
    immutable_text(REVIEW / "protocol.md", PROTOCOL)
    schema = dict(schema=SCHEMA, axes=AXES, fields=list(FIELDS), reviewer_kind="AI_PROXY", blind=False,
        labels=list(LABELS), stages=list(STAGES), eval_groups=list(GROUPS), expected_pages=84, expected_cells=168,
        protocol_sha256=sha(REVIEW / "protocol.md"), protected=protected,
        actual_view_bool_encoding="true/false, case-insensitive", image_hash_policy="Actual PNG, renderer receipt and gallery definition SHA256",
        pending_policy="Unviewed cells absent; no rating placeholders", duplicate_policy="ERROR_NO_SILENT_ADJUDICATION",
        aggregate_policy="Only complete 168-cell reviews produce category summaries; separate groups/stages and condition-then-parent macro rates",
        generated_ratings=0, model_calls=0, scoring_calls=0, rendered_images=0)
    immutable_json(REVIEW / "schema.json", schema)
    assignments = dict(schema=SCHEMA, status="PREPARED_NO_RATINGS", policy="Frozen input metadata, complete parent blocks, all stages and paired columns",
                       protocol_sha256=sha(REVIEW / "protocol.md"), eval_manifest_sha256=sha(source), tasks=tasks)
    immutable_json(REVIEW / "assignments.json", assignments)
    immutable_json(REVIEW / "expected_cells.json", dict(schema=SCHEMA, status="AWAITING_ACTUAL_OBSERVATIONS", rows=expected,
        schema_sha256=sha(REVIEW / "schema.json"), assignments_sha256=sha(REVIEW / "assignments.json"), generated_ratings=0))
    for task in tasks.values():
        path = resolve(task["records_csv"])
        if not path.exists():
            immutable_text(path, csv_text([], FIELDS))
    return dict(status="PREPARED_NO_RATINGS", schema=SCHEMA, cells=168, pages=84,
                shard_pages={k:v["pages"] for k,v in tasks.items()}, generated_ratings=0,
                protocol_sha256=sha(REVIEW / "protocol.md"), assignments_sha256=sha(REVIEW / "assignments.json"))


def require_bool(value, expected, field):
    if str(value).strip().lower() != ("true" if expected else "false"):
        raise ValueError("Invalid actual-view declaration: " + field)


def load_review():
    schema = read(REVIEW / "schema.json")
    assignments = read(REVIEW / "assignments.json")
    expected_doc = read(REVIEW / "expected_cells.json")
    for path, digest in schema["protected"].items():
        if sha(path) != digest:
            raise ValueError("Review protocol source changed: " + path)
    if sha(REVIEW / "schema.json") != expected_doc["schema_sha256"] or sha(REVIEW / "assignments.json") != expected_doc["assignments_sha256"]:
        raise ValueError("Review assignments changed")
    expected = {cell_key(r):r for r in expected_doc["rows"]}
    observed, rows, inputs, images = set(), [], {}, {}
    for shard, task in assignments["tasks"].items():
        path = resolve(task["records_csv"])
        inputs[str(path)] = sha(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Review CSV columns differ: " + str(path))
            for line, row in enumerate(reader, start=2):
                if not any(str(v or "").strip() for v in row.values()):
                    continue
                key = cell_key(row)
                if key not in expected or key in observed:
                    raise ValueError("Unknown/duplicate visual cell: " + repr(key))
                target = expected[key]
                if row["schema"] != SCHEMA or row["protocol_sha256"] != schema["protocol_sha256"] or row["assignments_sha256"] != expected_doc["assignments_sha256"]:
                    raise ValueError("Review CSV belongs to another protocol")
                for field in ("request_id", "condition_id", "parent_id", "eval_group", "split", "part_type", "label", "stage", "page_id"):
                    if row[field] != str(target[field]):
                        raise ValueError("Review identity differs: " + field)
                if int(row["noise_index"]) != target["noise_index"] or row["review_group"] != shard or target["review_group"] != shard:
                    raise ValueError("Review assignment/noise differs")
                for field in ("reviewer", "review_round", "view_event_id", "actual_view_source", "note"):
                    if not row[field].strip():
                        raise ValueError("Missing actual-view evidence: " + field)
                if row["reviewer_kind"] != "AI_PROXY":
                    raise ValueError("This review protocol requires separately identified AI_PROXY records")
                require_bool(row["blind"], False, "blind")
                require_bool(row["actually_viewed"], True, "actually_viewed")
                require_bool(row["views_and_sections_checked"], True, "views_and_sections_checked")
                timestamp = datetime.datetime.fromisoformat(row["viewed_at_utc"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None or timestamp.utcoffset() != datetime.timedelta(0):
                    raise ValueError("viewed_at_utc must be an explicit UTC timestamp")
                for axis, values in AXES.items():
                    if row[axis] not in values:
                        raise ValueError("Missing/invalid observed axis: " + axis)
                image, receipt_path = resolve(target["image_path"]), resolve(target["image_receipt"])
                if resolve(row["image_path"]).resolve() != image.resolve() or row["image_sha256"] != sha(image):
                    raise ValueError("Actual reviewed PNG identity differs")
                receipt = read(receipt_path)
                definition = OUT / "gallery/definition.json"
                if row["image_receipt_sha256"] != sha(receipt_path) or row["gallery_definition_sha256"] != sha(definition):
                    raise ValueError("Image receipt/definition hash differs")
                if receipt["image_sha256"] != row["image_sha256"] or receipt["definition_sha256"] != row["gallery_definition_sha256"]:
                    raise ValueError("Renderer does not authenticate this PNG")
                if receipt["page_id"] != row["page_id"] or receipt["request_id"] != row["request_id"] or receipt["condition_id"] != row["condition_id"]:
                    raise ValueError("Renderer page identity differs")
                if not any(c["model"] == row["label"] and c["stage"] == row["stage"] for c in receipt["cells"]):
                    raise ValueError("Model/stage is absent from reviewed image")
                for p in (image, receipt_path, definition):
                    images[str(p)] = sha(p)
                rows.append(dict(row, noise_index=int(row["noise_index"]), record_file=str(path), record_line=line))
                observed.add(key)
    page_rows = collections.defaultdict(list)
    parent_reviewers = collections.defaultdict(set)
    for r in rows:
        page_rows[r["page_id"]].append(r)
        parent_reviewers[r["parent_id"]].add(r["reviewer"])
    if any(len(reviewers) != 1 for reviewers in parent_reviewers.values()):
        raise ValueError("A complete parent block must stay with one reviewer")
    for page, pair in page_rows.items():
        if len(pair) != 2 or {r["label"] for r in pair} != set(LABELS):
            raise ValueError("A viewed page must have both model cells: " + page)
        for field in ("reviewer", "review_round", "view_event_id", "viewed_at_utc", "image_sha256"):
            if pair[0][field] != pair[1][field]:
                raise ValueError("Paired cells were not recorded in the same view event: " + field)
    missing = [r for key,r in expected.items() if key not in observed]
    rows.sort(key=cell_key)
    return rows, missing, inputs, images


def summaries(rows):
    summary, parent_rows = [], []
    for label in LABELS:
        for stage in STAGES:
            for group in GROUPS:
                subset = [r for r in rows if (r["label"],r["stage"],r["eval_group"]) == (label,stage,group)]
                for axis, values in AXES.items():
                    for value in values:
                        by_condition = collections.defaultdict(list)
                        for r in subset:
                            by_condition[r["parent_id"],r["condition_id"]].append(int(r[axis] == value))
                        by_parent = collections.defaultdict(list)
                        for (parent,cid), flags in by_condition.items():
                            if len(flags) != 2:
                                raise ValueError("Complete two-noise visual condition required")
                            by_parent[parent].append(statistics.mean(flags))
                        means = []
                        for parent, fractions in by_parent.items():
                            fraction = statistics.mean(fractions)
                            means.append(fraction)
                            parent_rows.append(dict(label=label,stage=stage,eval_group=group,parent_id=parent,
                                axis=axis,value=value,condition_count=len(fractions),fraction=fraction))
                        count = sum(r[axis] == value for r in subset)
                        summary.append(dict(label=label,stage=stage,eval_group=group,axis=axis,value=value,
                            count=count,output_denominator=len(subset),output_fraction=count/len(subset),
                            conditions=len(by_condition),parents=len(by_parent),parent_macro_fraction=statistics.mean(means)))
    paired, stage_rows = [], []
    indexed = {cell_key(r):r for r in rows}
    request_ids = sorted({r["request_id"] for r in rows})
    for rid in request_ids:
        for stage in STAGES:
            a,b = (indexed[rid,label,stage] for label in LABELS)
            r = {k:a[k] for k in ("request_id","condition_id","parent_id","noise_index","eval_group","part_type","stage","page_id","image_sha256")}
            r.update(reference=LABELS[0],candidate=LABELS[1],reference_note=a["note"],candidate_note=b["note"])
            for axis in AXES:
                r.update({axis+"__reference":a[axis],axis+"__candidate":b[axis],axis+"__transition":a[axis]+" -> "+b[axis]})
            paired.append(r)
        for label in LABELS:
            for first,last in zip(STAGES,STAGES[1:]):
                a,b = indexed[rid,label,first],indexed[rid,label,last]
                r = {k:a[k] for k in ("request_id","condition_id","parent_id","noise_index","eval_group","part_type")}
                r.update(label=label,from_stage=first,to_stage=last,from_image_sha256=a["image_sha256"],to_image_sha256=b["image_sha256"],from_note=a["note"],to_note=b["note"])
                for axis in AXES:
                    r.update({axis+"__from":a[axis],axis+"__to":b[axis],axis+"__transition":a[axis]+" -> "+b[axis]})
                stage_rows.append(r)
    return dict(summary=summary,per_parent=parent_rows,paired_per_output=paired,cross_stage_per_output=stage_rows)


def aggregate(write=False):
    rows, missing, inputs, images = load_review()
    complete = len(rows) == 168 and not missing
    value = dict(schema=SCHEMA,status="COMPLETE_ACTUAL_AI_PROXY_REVIEW" if complete else "INCOMPLETE",
        expected_cells=168,reviewed_cells=len(rows),missing_cells=len(missing),expected_pages=84,
        reviewed_pages=len({r["page_id"] for r in rows}),reviewer_kind="AI_PROXY",blind=False,
        human_ratings=0,model_calls=0,scoring_calls=0,rendering_calls=0,generated_ratings=0,
        source_review_files=inputs,authenticated_image_files=images,
        source_code_sha256=sha(__file__),schema_sha256=sha(REVIEW/"schema.json"),
        warning="Actual nonblind AI observations, not independent human ratings or topology certification. UNCERTAIN remains a full category. No combined success score.")
    if write:
        identity = hashlib.sha256(json.dumps({"inputs":inputs,"code":sha(__file__),"schema":value["schema_sha256"]},sort_keys=True).encode()).hexdigest()[:16]
        dest = REVIEW / "aggregates" / identity
        tables = dict(validated_per_cell=rows,missing_cells=missing)
        if complete:
            tables.update(summaries(rows))
        outputs = {}
        for name,table in tables.items():
            if table:
                p = dest / (name+".csv")
                immutable_text(p,csv_text(table))
                outputs[str(p)] = sha(p)
        value.update(output_files=outputs,output_directory=str(dest))
        immutable_json(dest/"summary.json",value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","validate","aggregate"))
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else aggregate(write=args.action == "aggregate")
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_review_files","authenticated_image_files","output_files")},ensure_ascii=False))
