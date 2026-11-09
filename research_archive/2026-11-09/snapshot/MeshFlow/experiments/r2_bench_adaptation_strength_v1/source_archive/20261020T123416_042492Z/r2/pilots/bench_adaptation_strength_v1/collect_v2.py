"""Lossless final-table collection from original scores; no scoring/model work."""
from __future__ import annotations
import argparse
import collections
import csv
import io
import json
import math
import statistics
from pathlib import Path
from .common import OUT, read, resolve, sha, write_new

DEST = OUT / "results_v2"
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
CONTRASTS = (
    ("A_S1500", "B_S1500", "PRIMARY_LR_B_MINUS_A"),
    ("B_S1500", "C_S1500", "PRIMARY_INITIALIZATION_C_MINUS_B"),
    ("A_S1500", "C_S1500", "ADDITIONAL_C_MINUS_A"),
)
SUPERSEDED = """# Legacy collector superseded

The original frozen `evaluation.py` collector is retained unchanged. Its
`per_output.csv` overwrites the original whole-mesh `boundary_edges` score with
the input C boundary-edge count; its aggregate/pair metric list also excludes
that score. The original per-cell `row.json` files retain the correct score.
No model output or per-cell scoring is affected.

The independent `collect_v2.py` reads all 672 original rows. It preserves the
whole-mesh `boundary_edges` value and adds the input count as `C_boundary_edges`.
The original whole-mesh count participates in V2 aggregate and paired tables.

The authoritative corrected tables are `../results_v2/` once that directory's
`completion.json` reports `COMPLETE`. Until then V2 is incomplete; this notice
does not certify completed collection. Legacy files here remain unchanged.
See `../diagnostics/collector_v2_revision.json` for the engineering record.
"""


def immutable(path, obj):
    if path.exists():
        if read(path) != obj:
            raise ValueError("Refusing to replace immutable evidence: " + str(path))
    else:
        write_new(path, obj)


def register_revision():
    frozen = Path(__file__).with_name("evaluation.py")
    value = dict(
        schema="bench_adaptation_collector_v2_engineering_revision",
        reason="Legacy collector overwrites whole-mesh boundary_edges with input-C boundary count and excludes it from aggregation.",
        discovery_before_legacy_collection=True,
        legacy_collector=str(frozen), legacy_collector_sha256=sha(frozen),
        replacement_collector=str(Path(__file__)), replacement_sha256=sha(__file__),
        retained_legacy_results=str(OUT / "results"), corrected_results=str(DEST),
        correction="Keep original boundary_edges exactly; add C_boundary_edges metadata; aggregate/pair original boundary_edges.",
        original_672_score_rows_read_only=True, scientific_definitions_changed=False,
        generator_calls=0, D0_calls=0, scoring_calls=0, geometry_changes=0,
        old_source_modified=False,
    )
    immutable(OUT / "diagnostics/collector_v2_revision.json", value)
    note = OUT / "results/SUPERSEDED.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    if note.exists():
        if note.read_text(encoding="utf-8") != SUPERSEDED:
            raise ValueError("Existing supersession notice differs")
    else:
        with note.open("x", encoding="utf-8", newline="\n") as f:
            f.write(SUPERSEDED)
    return value


def decorate(source_row, label, q, stage):
    row = dict(source_row)
    for field in ("request_id", "condition_id", "parent_id", "split"):
        if row[field] != q[field]:
            raise ValueError("Source score identity differs: " + field)
    for field in ("noise_index", "N", "K"):
        if int(row[field]) != int(q[field]):
            raise ValueError("Source score count differs: " + field)
    # Label/stage describe the model and pipeline stage in a joint table. All
    # actual metric values remain copied exactly, including boundary_edges.
    row.update(label=label, stage=stage, part_type=q["part_type"],
               closed_C=q["closed_C"], C_boundary_edges=q["boundary_edges"],
               seam_applicability="NA_CLOSED_C_NO_SEAM_INTERFACE" if q["closed_C"] else "APPLICABLE")
    if row["boundary_edges"] != source_row["boundary_edges"]:
        raise AssertionError("Whole-mesh boundary score was changed")
    return row


def score_folder(label, rid, stage):
    if stage == "RAW":
        return OUT / "evaluation/scores" / label / rid / "RAW"
    return OUT / "cpu" / label / rid / stage / "scores"


def load():
    scope, manifest = read(OUT / "registration.json"), read(OUT / "eval_manifest.json")
    process = set(scope["process_request_ids"])
    planned = [(label, q, stage)
               for label, spec in scope["evaluation_labels"].items()
               for q in manifest["requests"] if spec["subset"] == "full" or q["request_id"] in process
               for stage in STAGES]
    if len(planned) != 672:
        raise ValueError("Expected exactly 672 registered score rows")
    required = [(score_folder(label, q["request_id"], stage), label, q, stage) for label, q, stage in planned]
    missing = [str(folder / name) for folder, _, _, _ in required
               for name in ("row.json", "complete.json") if not (folder / name).exists()]
    if missing:
        return dict(status="INCOMPLETE", expected_rows=672, missing_files=missing)
    rows, source_hashes, receipt_hashes = [], {}, {}
    for folder, label, q, stage in required:
        completion = read(folder / "complete.json")
        if completion["status"] != "PASS" or completion["model_calls"] != 0:
            return dict(status="INCOMPLETE", incomplete_score=str(folder / "complete.json"))
        path = folder / "row.json"
        expected = next((digest for p, digest in completion["files"].items() if resolve(p) == path), None)
        actual = sha(path)
        if expected != actual:
            raise ValueError("Original row not authenticated: " + str(path))
        source = read(path)
        row = decorate(source, label, q, stage)
        metadata = {"label", "stage", "part_type", "closed_C", "seam_applicability"}
        if any(row[k] != v for k, v in source.items() if k not in metadata):
            raise AssertionError("Source field was not copied exactly")
        rows.append(row)
        source_hashes[str(path)] = actual
        receipt_hashes[str(folder / "complete.json")] = sha(folder / "complete.json")
    return dict(status="COMPLETE", rows=rows, source_rows=source_hashes, source_receipts=receipt_hashes,
                requests=manifest["requests"], registration_sha256=sha(OUT / "registration.json"),
                eval_manifest_sha256=sha(OUT / "eval_manifest.json"))


def numeric_fields(rows):
    return sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
                  - {"noise_index", "N", "K", "C_boundary_edges"})


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(rows, keys, metrics):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    output = []
    for identity, group in sorted(grouped.items()):
        row = dict(zip(keys, identity))
        row["rows"] = len(group)
        for metric in metrics:
            values = [float(r[metric]) for r in group if numeric(r.get(metric))]
            row[metric] = statistics.mean(values) if values else None
            row[metric + "__valid_n"] = len(values)
        output.append(row)
    return output


def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    text = buf.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text.replace("\r\n", "\n"):
            raise ValueError("Refusing to overwrite V2 table: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(text)


def run(check_inputs=False):
    data = load()
    if data["status"] != "COMPLETE" or check_inputs:
        return {k: v for k, v in data.items() if k not in ("rows", "source_rows", "source_receipts", "requests")}
    revision = register_revision()
    rows = data["rows"]
    metrics = numeric_fields(rows)
    if "boundary_edges" not in metrics or "C_boundary_edges" in metrics:
        raise AssertionError("V2 topology metric/metadata separation failed")
    done = DEST / "completion.json"
    if done.exists():
        old = read(done)
        if old["source_rows"] != data["source_rows"] or old["collector_sha256"] != sha(__file__):
            raise ValueError("Completed V2 source identity differs")
        for path, digest in old["files"].items():
            if sha(path) != digest:
                raise ValueError("Completed V2 table changed")
        return old
    conditions = aggregate(rows, ("label", "stage", "split", "parent_id", "condition_id"), metrics)
    parents = aggregate(conditions, ("label", "stage", "split", "parent_id"), metrics)
    summary = aggregate(parents, ("label", "stage", "split"), metrics)
    pairs = []
    for stage in STAGES:
        by = {(r["label"], r["request_id"]): r for r in rows if r["stage"] == stage}
        for reference, candidate, contrast in CONTRASTS:
            for q in data["requests"]:
                a, b = by[reference, q["request_id"]], by[candidate, q["request_id"]]
                r = dict(reference=reference, candidate=candidate, contrast=contrast, stage=stage,
                         **{k: q[k] for k in ("request_id", "condition_id", "parent_id", "split", "part_type")})
                for metric in metrics:
                    av, bv = a.get(metric), b.get(metric)
                    r[metric + "__reference"] = av
                    r[metric + "__candidate"] = bv
                    r[metric + "__delta"] = float(bv-av) if numeric(av) and numeric(bv) else None
                pairs.append(r)
    tables = dict(per_output=rows, per_condition=conditions, per_parent=parents,
                  parent_macro_summary=summary, paired_final=pairs)
    for name, content in tables.items():
        csvwrite(DEST / (name + ".csv"), content)
    result = dict(status="COMPLETE", schema="bench_adaptation_score_collection_v2",
                  rows=672, outputs=224, metric_fields=metrics, all_fields_preserved=True,
                  whole_mesh_boundary_edges_preserved=True, input_C_boundary_field="C_boundary_edges",
                  source_rows=data["source_rows"], source_receipts=data["source_receipts"],
                  collector_sha256=sha(__file__), revision=revision,
                  registration_sha256=data["registration_sha256"], eval_manifest_sha256=data["eval_manifest_sha256"],
                  model_calls=0, scoring_calls=0, new_queries=0,
                  files={str(DEST / (name + ".csv")): sha(DEST / (name + ".csv")) for name in tables})
    write_new(done, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    result = register_revision() if args.register_only else run(args.check_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
