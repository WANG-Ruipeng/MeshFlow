"""Lossless collection: 48 new outputs plus 96 authenticated USER v2/A outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import OUT, read, resolve, sha, write_new
from .eval_plan import BASELINE_LABEL, BASELINE_ROOT, REFERENCE_LABELS, ALL_LABELS, LABELS, requests, root_for, verify
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import (
    decorate, numeric_fields, numeric, aggregate, csvwrite,
)

DEST = OUT / "results"
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")


def score_folder(label, rid, stage):
    root = root_for(label)
    if stage == "RAW":
        return root / "evaluation/scores" / label / rid / "RAW"
    return root / "cpu" / label / rid / stage / "scores"


def scorer_identity_one(label):
    """Entry wrappers differ; every shared scientific scoring source must agree."""
    reference_root = root_for(label)
    a = read(reference_root / "evaluation/score_registration.json")
    b = read(OUT / "evaluation/score_registration.json")
    old = {str(resolve(p).resolve()): h for p, h in a["source_hashes"].items()}
    new = {str(resolve(p).resolve()): h for p, h in b["source_hashes"].items()}
    shared = sorted(set(old) & set(new))
    if not shared:
        raise ValueError("No common scoring sources authenticated")
    if a["stages"] != b["stages"] or a["chain"] != b["chain"]:
        raise ValueError("Stage or postprocess definitions differ from historical A")
    for p in shared:
        if old[p] != new[p] or sha(p) != old[p]:
            raise ValueError("Historical/current scientific scoring source differs: " + p)
    return dict(common_source_count=len(shared), common_sources={p: old[p] for p in shared},
                historical_registration_sha256=sha(reference_root / "evaluation/score_registration.json"),
                current_registration_sha256=sha(OUT / "evaluation/score_registration.json"))


def scorer_identity():
    return {label:scorer_identity_one(label) for label in REFERENCE_LABELS}


def load():
    qs = requests()
    planned = [(label, q, stage) for label in ALL_LABELS for q in qs for stage in STAGES]
    if len(planned) != 432:
        raise ValueError("Expected 3 models x 48 requests x 3 stages")
    missing = [str(score_folder(label, q["request_id"], stage) / name)
               for label, q, stage in planned for name in ("row.json", "complete.json")
               if not (score_folder(label, q["request_id"], stage) / name).exists()]
    if missing:
        return dict(status="INCOMPLETE", expected_rows=432, missing_files=missing)
    scores = scorer_identity()
    rows, source_hashes, receipt_hashes = [], {}, {}
    for label, q, stage in planned:
        folder = score_folder(label, q["request_id"], stage)
        completion = read(folder / "complete.json")
        if completion["status"] != "PASS" or completion["model_calls"] != 0:
            raise ValueError("Scoring receipt is not PASS: " + str(folder))
        path = folder / "row.json"
        expected = next((h for p, h in completion["files"].items()
                         if resolve(p).resolve() == path.resolve()), None)
        actual = sha(path)
        if expected != actual:
            raise ValueError("Unauthenticated score row: " + str(path))
        source = read(path)
        row = decorate(source, label, q, stage)
        metadata = {"label", "stage", "part_type", "closed_C", "seam_applicability"}
        if any(row[k] != v for k, v in source.items() if k not in metadata):
            raise AssertionError("A source metric was changed")
        row["result_origin"] = "REUSED_HISTORICAL_"+label if label in REFERENCE_LABELS else "NEW_REGISTERED_OUTPUT"
        rows.append(row)
        source_hashes[str(path)] = actual
        receipt_hashes[str(folder / "complete.json")] = sha(folder / "complete.json")
    return dict(status="COMPLETE", rows=rows, requests=qs, scorer_identity=scores,
                source_rows=source_hashes, source_receipts=receipt_hashes)


def run(check_inputs=False):
    data = load()
    if data["status"] != "COMPLETE" or check_inputs:
        return {k: v for k, v in data.items() if k not in ("rows", "source_rows", "source_receipts", "requests")}
    rows = data["rows"]
    metrics = numeric_fields(rows)
    if "boundary_edges" not in metrics or "C_boundary_edges" in metrics:
        raise AssertionError("Metric/metadata boundary counts were conflated")
    done = DEST / "completion.json"
    if done.exists():
        old = read(done)
        if old["source_rows"] != data["source_rows"] or old["collector_sha256"] != sha(__file__):
            raise ValueError("Completed comparison source identity changed")
        verify(old["files"])
        return old
    conditions = aggregate(rows, ("label", "stage", "split", "parent_id", "condition_id"), metrics)
    parents = aggregate(conditions, ("label", "stage", "split", "parent_id"), metrics)
    summary = aggregate(parents, ("label", "stage", "split"), metrics)
    pairs = []
    for stage in STAGES:
        by = {(r["label"], r["request_id"]): r for r in rows if r["stage"] == stage}
        for reference in REFERENCE_LABELS:
            for candidate in LABELS:
                for q in data["requests"]:
                    a, b = by[reference, q["request_id"]], by[candidate, q["request_id"]]
                    r = dict(reference=reference, candidate=candidate,
                             contrast=candidate + "_MINUS_"+reference, stage=stage,
                             **{k: q[k] for k in ("request_id", "condition_id", "parent_id", "noise_index", "split", "part_type")})
                    for metric in metrics:
                        av, bv = a.get(metric), b.get(metric)
                        r[metric + "__reference"] = av
                        r[metric + "__candidate"] = bv
                        r[metric + "__delta"] = float(bv-av) if numeric(av) and numeric(bv) else None
                    pairs.append(r)
    pair_metrics = [m + "__delta" for m in metrics]
    paired_conditions = aggregate(pairs, ("reference", "candidate", "stage", "split", "parent_id", "condition_id"), pair_metrics)
    paired_parents = aggregate(paired_conditions, ("reference", "candidate", "stage", "split", "parent_id"), pair_metrics)
    tables = dict(per_output=rows, per_condition=conditions, per_parent=parents,
                  parent_macro_summary=summary, paired_per_output=pairs,
                  paired_per_condition=paired_conditions, paired_per_parent=paired_parents)
    for name, content in tables.items():
        csvwrite(DEST / (name + ".csv"), content)
    result = dict(status="COMPLETE", schema="BENCH_RELATION_TRAINING_COLLECTION_V1",
                  rows=432, new_outputs=48, reused_outputs=96, new_score_rows=144, reused_score_rows=288,
                  metric_fields=metrics, all_fields_preserved=True,
                  whole_mesh_boundary_edges_preserved=True, input_C_boundary_field="C_boundary_edges",
                  source_rows=data["source_rows"], source_receipts=data["source_receipts"],
                  scorer_identity=data["scorer_identity"], collector_sha256=sha(__file__),
                  registration_sha256=sha(OUT / "registration.json"),
                  eval_manifest_sha256=sha(OUT / "eval_manifest.json"),
                  model_calls=0, scoring_calls=0, new_queries=0,
                  files={str(DEST / (name + ".csv")): sha(DEST / (name + ".csv")) for name in tables})
    write_new(done, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-inputs", action="store_true")
    a = p.parse_args()
    value = run(a.check_inputs)
    print(json.dumps({k: v for k, v in value.items() if k not in ("files", "source_rows", "source_receipts")},
                     ensure_ascii=False, indent=2))
