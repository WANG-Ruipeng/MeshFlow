"""Saved-layout source-label diagnostic; no forward, threshold search or edit.

Run only after both 28-request RAW completions exist. This compares the actual
saved logit>=0 decisions against the source-owner training convention. Alias
and true-Null edges remain separate. The source completion is one supervised
target, not the unique valid open-ended completion; source labels do not impose
a new required-seam task or certify generated C use, topology or shape quality.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import eval_runtime as R
from .common import OUT, ROOT, read, resolve, sha, write_new

VERSION = "BENCH_NEWDATA_SAVED_HEAD_SOURCE_DIAGNOSTIC_V1"
DEST = OUT / "results"
DONE = DEST / "head_summary.json"
CASE_FIELDS = ("C_vertices", "C_faces_original", "C_faces_model", "N", "K")
COUNTS = ("source_owner_edges", "source_alias_edges", "source_true_null_edges", "predicted_active_edges",
          "owner_TP", "owner_FN", "owner_FP", "owner_TN", "active_owner_edges", "active_alias_edges", "active_true_null_edges")


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def freeze_file(files, path, expected=None):
    path = resolve(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Changed diagnostic input: " + str(path))
    if str(path) in files and files[str(path)] != digest:
        raise ValueError("Input changed while reading: " + str(path))
    files[str(path)] = digest
    return digest


def layout_paths(label, q):
    if label == R.BASELINE_LABEL and q["baseline_reuse"]:
        return resolve(q["old_layout_npz"]), resolve(q["old_layout_receipt"]), "ORIGINAL_OLD_DEV_LAYOUT"
    base = OUT / "evaluation" / label / "layouts" / q["condition_id"]
    return base / "layout.npz", base / "receipt.json", "NEW_REGISTERED_SAVED_LAYOUT"


def ready():
    """Only check files/JSON status. Missing results are pending, not failures."""
    manifest = read(OUT / "eval_manifest.json")
    conditions = {q["condition_id"]: q for q in manifest["requests"]}
    missing = []
    for label in R.LABELS:
        completion = OUT / "evaluation" / label / "raw_complete_full.json"
        if not completion.exists() or read(completion).get("status") != "COMPLETE":
            missing.append(str(completion))
        for q in conditions.values():
            paths = layout_paths(label, q)
            missing.extend(str(p) for p in paths[:2] if not p.exists())
    return dict(status="READY" if not missing else "PENDING_INPUTS", missing=missing,
                required_model_condition_cells=28, model_calls=0, diagnostic_edges_processed=0)


def source_case(q, lookup, files):
    row = lookup[q["condition_id"]]
    path = resolve(row["case_path"])
    freeze_file(files, path, row["case_sha256"])
    if "case_path" in q and resolve(q["case_path"]).resolve() != path.resolve():
        raise ValueError("Panel/source-supervision case differs")
    case = R.arrays(path)
    inp = R.arrays(q["inference_npz"])
    for key in CASE_FIELDS:
        if not np.array_equal(case[key], inp[key]):
            raise ValueError("Source labels belong to a different input: " + key)
    edges = np.asarray(case["edge_records"])
    owner, alias, supported = (np.asarray(case[key]) for key in
                              ("occupied_labels", "source_alias_labels", "source_supported_labels"))
    count = q["boundary_edges"]
    if edges.shape != (count, 3) or any(a.dtype != np.bool_ or a.shape != (count,) for a in (owner, alias, supported)):
        raise ValueError("Invalid source-label edge domain")
    if np.any(owner & alias) or not np.array_equal(owner | alias, supported):
        raise ValueError("Source owner/alias/supported partition differs")
    if not np.array_equal(supported, np.asarray(case["edge_owner_face"]) >= 0):
        raise ValueError("Source owner-face support does not match labels")
    return case, owner, alias, ~supported, path


def saved_layout(label, q, case, files, protocol, registration):
    path, receipt_path, origin = layout_paths(label, q)
    receipt = read(receipt_path)
    freeze_file(files, receipt_path)
    if receipt.get("status") != "COMPLETE" or receipt.get("condition_id") != q["condition_id"]:
        raise ValueError("Incomplete or wrong-condition saved layout")
    if receipt.get("source_labels_read") is not False:
        raise ValueError("Layout producer does not certify C-only prediction")
    if origin == "ORIGINAL_OLD_DEV_LAYOUT":
        freeze_file(files, path, q["old_layout_sha256"])
        if receipt["layout_sha256"] != q["old_layout_sha256"] or receipt["threshold"] != 0:
            raise ValueError("Old layout hash/threshold differs")
        if receipt["checkpoint_sha256"] != protocol["source_checkpoint_sha256"]:
            raise ValueError("Old DEV head checkpoint differs")
        calls, active, count = receipt["occupancy_forward"], receipt["active"], receipt["edges"]
        identity = dict(kind="FROZEN_OLD_S15000_HEAD", state_sha256=registration["head_state_sha256"],
                        source_checkpoint_sha256=receipt["checkpoint_sha256"],
                        identity_basis="Original checkpoint hash linked by retained head-only extraction protocol")
    else:
        if receipt["label"] != label or receipt["input_sha256"] != q["inference_npz_sha256"]:
            raise ValueError("New layout label/input identity differs")
        for p, digest in receipt["files"].items():
            freeze_file(files, p, digest)
        if not any(resolve(p).resolve() == path.resolve() for p in receipt["files"]):
            raise ValueError("New receipt does not bind layout")
        head = receipt["head"]
        if receipt["threshold_logit"] != 0 or head["threshold_logit"] != 0 or head["parameters"] != 833:
            raise ValueError("Head architecture/threshold changed")
        if head["source_labels_read"] is not False or head["input"] != "C_ONLY_24_FEATURES":
            raise ValueError("Unexpected head input contract")
        if label == R.BASELINE_LABEL and head["state_sha256"] != registration["head_state_sha256"]:
            raise ValueError("Baseline head differs from registered old head")
        if label == R.NEW_LABEL:
            sidecar = resolve(R.label_spec(label)["generator_path"]).with_suffix(".json")
            export = read(sidecar)
            freeze_file(files, sidecar)
            if head["state_sha256"] != export["head_state_sha256"]:
                raise ValueError("Saved candidate head differs from the final endpoint sidecar")
        identity = dict(head)
        calls, active, count = receipt["occupancy_forward_calls"], receipt["active"], receipt["boundary_edges"]
    layout = R.arrays(path)
    logits, pred = np.asarray(layout["occupancy_logits"]), np.asarray(layout["occupied"])
    b = int(q["boundary_edges"])
    if logits.dtype != np.float32 or logits.shape != (b,) or not np.isfinite(logits).all():
        raise ValueError("Invalid saved FP32 logit domain")
    if pred.dtype != np.bool_ or pred.shape != (b,) or not np.array_equal(pred, logits >= 0):
        raise ValueError("Saved binary decision differs from fixed logit>=0")
    if (int(calls), int(active), int(count)) != (int(b > 0), int(pred.sum()), b):
        raise ValueError("Saved head-call or boundary count differs")
    if bool(q["closed_C"]) != (b == 0):
        raise ValueError("Closed-C applicability differs")
    for key in (*CASE_FIELDS, "edge_records"):
        if not np.array_equal(layout[key], case[key]):
            raise ValueError("Saved/source C edge order differs: " + key)
    # Confirm the recorded addresses without computing features or invoking a head.
    n, k = q["N"], q["K"]
    anchors = np.full((n, 3), -1, np.int64)
    anchors[:k] = case["C_faces_model"]
    for edge_id, (_, u, v) in enumerate(case["edge_records"]):
        if pred[edge_id]:
            anchors[k + edge_id, :2] = [v, u]
    known = np.arange(n) < k
    expected = dict(anchor_ids=anchors, known_mask=known, valid_mask=np.ones(n, bool),
        free_corner_mask=anchors < 0, seam_slots=k + np.flatnonzero(pred),
        bulk_slots=np.flatnonzero(~known & (anchors < 0).all(1)))
    for key, value in expected.items():
        if not np.array_equal(layout[key], value):
            raise ValueError("Saved layout addressing differs: " + key)
    return layout, identity, origin, path, receipt_path


def verify_noise_consumption(label, qs, layout, files):
    """Two existing RAWs must actually carry this condition's saved decisions."""
    for q in qs:
        root = OUT / "evaluation" / label / q["request_id"] / "raw"
        receipt = read(root / "receipt.json")
        freeze_file(files, root / "receipt.json")
        freeze_file(files, root / "raw.npz", receipt["raw_sha256"])
        if receipt["status"] != "COMPLETE" or receipt["request_id"] != q["request_id"] or receipt["source_labels_read"] is not False:
            raise ValueError("RAW producer identity differs")
        a = R.arrays(root / "raw.npz")
        for key in ("occupied", "occupancy_logits", "edge_records"):
            if a[key].dtype != layout[key].dtype or a[key].tobytes() != layout[key].tobytes():
                raise ValueError("Actual RAW used different decisions: " + key)
        if not np.array_equal(a["raw_anchor_ids"], layout["anchor_ids"]):
            raise ValueError("Actual RAW anchors differ")
        seam = np.zeros(q["N"], bool)
        seam[layout["seam_slots"]] = True
        if not np.array_equal(a["raw_seam_slot_mask"], seam):
            raise ValueError("Actual RAW seam role differs")


def condition_rows(label, q, case, layout, identity, origin, case_path, layout_path, receipt_path):
    owner, alias = case["occupied_labels"], case["source_alias_labels"]
    null = ~case["source_supported_labels"]
    logits, pred = layout["occupancy_logits"], layout["occupied"]
    b = len(pred)
    common = dict(label=label, condition_id=q["condition_id"], parent_id=q["parent_id"],
                  eval_group=q["eval_group"], part_type=q["part_type"], N=q["N"], K=q["K"])
    edges = []
    source_ids = case["C_source_vertex_ids"]
    for i, (face, u, v) in enumerate(case["edge_records"]):
        role = "OWNER" if owner[i] else "ALIAS" if alias[i] else "TRUE_NULL"
        edges.append(dict(**common, edge_index=i, C_face_id=int(face), C_vertex_u=int(u), C_vertex_v=int(v),
            source_C_face_id=int(case["known_ids"][face]), source_vertex_u=int(source_ids[u]), source_vertex_v=int(source_ids[v]),
            source_owner_face_id=int(case["edge_owner_face"][i]), source_role=role,
            source_owner=int(owner[i]), source_alias=int(alias[i]), source_true_null=int(null[i]),
            source_supported=int(not null[i]), predicted_logit=float(logits[i]), threshold_logit=0.0,
            predicted_active=int(pred[i]), owner_target_agreement=int(pred[i] == owner[i]),
            source_label_interpretation="ORIGINAL_TARGET_SUPERVISION_ONLY_NOT_REQUIRED_SEAM", layout_origin=origin))
    counts = dict(source_owner_edges=int(owner.sum()), source_alias_edges=int(alias.sum()), source_true_null_edges=int(null.sum()),
        predicted_active_edges=int(pred.sum()), owner_TP=int((pred & owner).sum()), owner_FN=int((~pred & owner).sum()),
        owner_FP=int((pred & ~owner).sum()), owner_TN=int((~pred & ~owner).sum()),
        active_owner_edges=int((pred & owner).sum()), active_alias_edges=int((pred & alias).sum()),
        active_true_null_edges=int((pred & null).sum()))
    row = dict(**common, status="PASS" if b else "NA_CLOSED_C_NO_HEAD_EDGES", C_boundary_edges=b, closed_C=b == 0,
        **counts, owner_edge_accuracy=ratio((pred == owner).sum(), b),
        owner_recall=ratio(counts["owner_TP"], counts["source_owner_edges"]),
        owner_precision=ratio(counts["owner_TP"], counts["predicted_active_edges"]),
        true_null_inactive_fraction=ratio((~pred & null).sum(), null.sum()),
        alias_inactive_fraction=ratio((~pred & alias).sum(), alias.sum()),
        owner_layout_exact=bool(np.array_equal(pred, owner)) if b else None,
        source_all_true_null=bool(null.all()) if b else None,
        saved_logit_min=float(logits.min()) if b else None, saved_logit_max=float(logits.max()) if b else None,
        saved_head_forward_calls=int(b > 0), diagnostic_forward_calls=0, source_edge_order="EXACT",
        two_noise_RAW_layout_identity="PASS", layout_origin=origin, case_path=str(case_path),
        layout_path=str(layout_path), layout_receipt=str(receipt_path), head_state_sha256=identity["state_sha256"])
    if not b:
        for key in ("owner_TP", "owner_FN", "owner_FP", "owner_TN"):
            row[key] = None
    return row, edges


def summarize(rows):
    result = []
    for label in R.LABELS:
        for group in ("ALL", "CURRENT_TRAIN", "SAME_PARENT_HELDOUT", "DEV"):
            cells = [r for r in rows if r["label"] == label and (group == "ALL" or r["eval_group"] == group)]
            applicable = [r for r in cells if r["status"] == "PASS"]
            counts = {key: sum(r[key] or 0 for r in cells) for key in COUNTS}
            edges = sum(r["C_boundary_edges"] for r in cells)
            accuracy = [r["owner_edge_accuracy"] for r in applicable]
            by_parent = collections.defaultdict(list)
            for r in applicable:
                by_parent[r["parent_id"]].append(r["owner_edge_accuracy"])
            parent_values = [float(np.mean(v)) for v in by_parent.values()]
            result.append(dict(label=label, eval_group=group, conditions=len(cells),
                applicable_open_conditions=len(applicable), closed_NA_conditions=len(cells)-len(applicable),
                source_edges=edges, **counts,
                edge_micro_owner_accuracy=ratio(counts["owner_TP"] + counts["owner_TN"], edges),
                condition_macro_owner_accuracy=float(np.mean(accuracy)) if accuracy else None,
                parent_macro_owner_accuracy=float(np.mean(parent_values)) if parent_values else None,
                owner_exact_layout_conditions=sum(r["owner_layout_exact"] for r in applicable),
                true_null_inactive_fraction=ratio(counts["source_true_null_edges"]-counts["active_true_null_edges"], counts["source_true_null_edges"]),
                alias_inactive_fraction=ratio(counts["source_alias_edges"]-counts["active_alias_edges"], counts["source_alias_edges"])))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(stream.getvalue())


def run():
    if DONE.exists():
        value = read(DONE)
        R.verify(value["source_files"])
        R.verify(value["output_files"])
        return dict(status="VERIFIED_COMPLETE", model_calls=0, new_diagnostic_rows=0)
    availability = ready()
    if availability["status"] != "READY":
        return availability
    for name in ("head_per_edge.csv", "head_per_condition.csv"):
        if (DEST / name).exists():
            raise RuntimeError("Partial head analysis retained; inspect before explicit recovery")
    tick = time.perf_counter()
    files = {}
    registration = read(OUT / "registration.json")
    freeze_file(files, OUT / "registration.json")
    freeze_file(files, registration["manifest"], registration["manifest_sha256"])
    freeze_file(files, registration["eval_manifest"], registration["eval_manifest_sha256"])
    freeze_file(files, __file__)
    data = read(registration["manifest"])
    lookup = {r["condition_id"]: r for r in data["train"] + data["dev"] + data["same_parent_heldout"]}
    qs = R.requests()
    conditions = {q["condition_id"]: q for q in qs}
    if len(conditions) != 14:
        raise ValueError("Exactly 14 unique conditions required")
    protocol_path = R.HEAD_ROOT / "HEAD_PROTOCOL.json"
    protocol = read(protocol_path)
    freeze_file(files, protocol_path)
    rows, edges = [], []
    head_states = collections.defaultdict(set)
    for label in R.LABELS:
        R.verify_raw(label)
        freeze_file(files, OUT / "evaluation" / label / "raw_complete_full.json")
        for cid, q in conditions.items():
            case, _, _, _, case_path = source_case(q, lookup, files)
            layout, identity, origin, layout_path, receipt_path = saved_layout(label, q, case, files, protocol, registration)
            verify_noise_consumption(label, [r for r in qs if r["condition_id"] == cid], layout, files)
            row, per_edge = condition_rows(label, q, case, layout, identity, origin, case_path, layout_path, receipt_path)
            rows.append(row)
            edges.extend(per_edge)
            head_states[label].add(identity["state_sha256"])
    if len(rows) != 28 or any(len(v) != 1 for v in head_states.values()):
        raise ValueError("Condition or head identity changed within a model label")
    if "torch" in sys.modules:
        raise RuntimeError("Saved-head analysis imported neural framework")
    R.verify(files)
    write_csv(DEST / "head_per_edge.csv", edges)
    write_csv(DEST / "head_per_condition.csv", rows)
    summary = dict(schema=VERSION, status="COMPLETE", labels=list(R.LABELS), source_conditions=14,
        model_condition_rows=28, edge_rows=len(edges), closed_NA_rows=sum(r["closed_C"] for r in rows),
        threshold_logit=0.0, threshold_equality_is_active=True, threshold_scan=False,
        aggregates=summarize(rows), head_state_sha256={k:next(iter(v)) for k,v in head_states.items()},
        source_role_definitions=dict(OWNER="occupied_labels=True, one selected boundary owner per source free face",
            ALIAS="source_alias_labels=True, supported source adjacency represented by another owner; inactive training target",
            TRUE_NULL="source_supported_labels=False, neither OWNER nor ALIAS in the retained source supervision"),
        interpretation="Source-owner matching is a training-target diagnostic, not open-ended generation success. Source all-Null does not mean C must acquire a seam; no new interface requirement is imposed. Active alias and active true-Null edges are reported separately. These results do not certify final C use, legal shared topology, source-target representability or shape quality.",
        aggregation="One saved layout per model/condition; the two noise RAWs are checked for identical use, not double-counted. Closed C has NA head accuracy/layout match. Keep edge-micro, condition-macro and parent-macro distinct.",
        source_labels_used="DIAGNOSTIC_ONLY_AFTER_SAVED_INFERENCE; no layout/output modifications",
        model_calls=0, head_forwards=0, generator_forwards=0, geometry_score_calls=0,
        new_noise_arrays=0, new_queries=0, outputs_modified=0,
        seconds=time.perf_counter()-tick, source_files=files,
        output_files={str(DEST/name):sha(DEST/name) for name in ("head_per_edge.csv", "head_per_condition.csv")})
    write_new(DONE, summary)
    return {k:v for k,v in summary.items() if k not in ("source_files", "output_files", "aggregates")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"), nargs="?", default="check")
    args = parser.parse_args()
    print(json.dumps(ready() if args.action == "check" else run(), ensure_ascii=False), flush=True)
