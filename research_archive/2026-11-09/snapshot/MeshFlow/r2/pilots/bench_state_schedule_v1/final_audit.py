"""Independent CPU closeout: read existing ledgers, arrays, scores and reviews.

No model, renderer, scorer, training helper, or scientific producer is imported.
``--check-ready`` reads only completion/status JSON and never writes anything.
The real audit refuses to start until both training arms, collection, gallery
and actual visual review have completed. Weights are hashed/indexed, not loaded
or copied. A failure receipt never turns missing work into completed work.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import difflib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from .common import ROOT, OUT, ARMS, EXPERIMENT, read, resolve, sha, write_new

BASE = ROOT / "experiments/r2_bench_adaptation_strength_v1"
LABELS = ("PERTURBED_STATE_S1500", "ASYNC_LEAD_SEAM_S1500")
ALL_LABELS = ("A_S1500", *LABELS)
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
SCHEMA = "BENCH_STATE_SCHEDULE_CPU_FINAL_AUDIT_V1"


def readiness():
    requirements = [(OUT / "training" / arm / "status.json", "COMPLETE") for arm in ARMS]
    requirements += [
        (OUT / "results/completion.json", "COMPLETE"),
        (OUT / "gallery/completion.json", "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW"),
        (OUT / "visual_review/completion.json", "ACTUAL_AI_PROXY_REVIEW_COMPLETE"),
    ]
    for label in LABELS:
        requirements += [
            (OUT / "evaluation" / label / "raw_complete_full.json", "COMPLETE"),
            (OUT / "evaluation" / label / "post_complete_full.json", "COMPLETE"),
        ]
    blockers = []
    for path, status in requirements:
        if not path.is_file():
            blockers.append(dict(path=str(path), reason="MISSING_COMPLETION"))
        elif read(path).get("status") != status:
            blockers.append(dict(path=str(path), reason="NOT_COMPLETE",
                                 actual=read(path).get("status"), expected=status))
    return dict(status="READY" if not blockers else "NOT_READY", blockers=blockers,
                model_calls=0, scoring_calls=0, wrote_files=False)


def lines(path):
    with resolve(path).open(encoding="utf-8-sig") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError("Blank line in durable ledger: " + str(path) + ":" + str(i))
            yield json.loads(line)


def csv_rows(path):
    with resolve(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Audit:
    def __init__(self, dest):
        self.dest = dest
        self.checks = []
        self.hashed = {}
        self.weights = {}
        self.source_paths = set()

    def check(self, ok, name, detail=None):
        row = dict(check=name, status="PASS" if bool(ok) else "FAIL")
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        if not bool(ok):
            raise AssertionError(name + (": " + str(detail) if detail is not None else ""))

    def hash(self, path, expected=None, *, source=False):
        p = resolve(path).resolve(strict=True)
        key = str(p)
        stat = p.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if key not in self.hashed:
            digest = sha(p)
            after = p.stat()
            self.check(stamp == (after.st_size, after.st_mtime_ns), "stable_while_hashing", key)
            self.hashed[key] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            prior = self.hashed[key]
            self.check(stamp == (prior["bytes"], prior["mtime_ns"]), "stable_since_hash", key)
        digest = self.hashed[key]["sha256"]
        if expected is not None:
            self.check(digest == expected, "registered_content_hash", key)
        if source:
            self.source_paths.add(key)
        if p.suffix.lower() in (".pt", ".pth", ".safetensors", ".ckpt"):
            self.weights[key] = dict(path=key, bytes=stat.st_size, sha256=digest,
                                     verified_by="FILE_SHA256_ONLY_NO_DESERIALIZATION")
        return digest

    def files(self, mapping, *, source=False):
        for path, expected in mapping.items():
            self.hash(path, expected, source=source)

    def document(self, path):
        self.hash(path)
        return read(path)


def training_ledger(path, arm):
    limits = dict(train_forward=12000, train_backward=12000, optimizer_update=1500)
    attempted, done, failed = collections.Counter(), collections.Counter(), collections.Counter()
    pending, ids = {}, set()
    contexts = {key: [] for key in limits}
    for r in lines(path):
        if r.get("arm") != arm or r.get("kind") not in limits:
            raise ValueError("Training ledger ownership or unauthorized call kind")
        ident, kind, event = r["call_id"], r["kind"], r["event"]
        if event == "CALL_BEGIN":
            if ident in ids or pending:
                raise ValueError("Repeated or concurrent/unresolved training call")
            ids.add(ident)
            pending[ident] = r
            attempted[kind] += 1
        elif event in ("CALL_DONE", "CALL_FAILED"):
            start = pending.pop(ident, None)
            if start is None or any(start[k] != r[k] for k in ("kind", "arm", "context", "attempt_id")):
                raise ValueError("Training call completion identity differs")
            (done if event == "CALL_DONE" else failed)[kind] += 1
            contexts[kind].append(r["context"])
        else:
            raise ValueError("Unknown training event: " + str(event))
    if pending or failed or dict(attempted) != limits or dict(done) != limits:
        raise ValueError("Training attempts/failures/pending/counts differ")
    if sorted(ids) != list(range(1, 25501)):
        raise ValueError("Training call IDs are not complete and unique")
    for kind in ("train_forward", "train_backward"):
        actual = [(r["local_step"], r["microbatch"]) for r in contexts[kind]]
        expected = [(s, m) for s in range(1, 1501) for m in range(8)]
        if actual != expected:
            raise ValueError("Training forward/backward cursor differs: " + kind)
        if any(r["global_step"] != 12000 + r["local_step"] for r in contexts[kind]):
            raise ValueError("Training global cursor differs")
    if [r["local_step"] for r in contexts["optimizer_update"]] != list(range(1, 1501)):
        raise ValueError("Optimizer cursors differ")
    return dict(attempted=dict(attempted), successful=dict(done), failed={},
                pending_call_ids=[], limits=limits)


def sampling_ledger(path, asynchronous):
    rows = list(lines(path))
    expected = [(name, step) for step in range(50)
                for name in ("FORWARD_STARTED", "FORWARD_COMPLETE")]
    if [(r.get("event"), r.get("step")) for r in rows] != expected:
        raise ValueError("Euler50 ledger not exactly 50 completed calls")
    if asynchronous and any(r.get("schema") != "ASYNC_LEAD_SEAM_LOCAL_VELOCITY_V1" for r in rows):
        raise ValueError("Async sampler schema differs")
    return dict(forwards=50, first_unix=float(rows[0]["unix"]), last_unix=float(rows[-1]["unix"]))


def denoiser_ledger(path):
    rows = list(lines(path))
    if [r.get("event") for r in rows] != ["D0_FORWARD_STARTED", "D0_FORWARD_RETURNED"]:
        raise ValueError("D0 must have exactly one completed call")
    if rows[0].get("budget") != 1:
        raise ValueError("D0 call budget differs")
    return dt.datetime.fromisoformat(rows[0]["utc"].replace("Z", "+00:00")).timestamp()


def audit_registration(a):
    reg = a.document(OUT / "training_registration.json")
    a.check(a.document(OUT / "registration.json") == reg, "training_and_evaluation_scope_identical")
    a.check(reg["status"] == "FROZEN" and reg["experiment"] == EXPERIMENT, "frozen_experiment_identity")
    a.check(reg["arms"] == list(ARMS) and reg["updates"] == 1500 and
            reg["batch"] == 8 and reg["microbatch"] == 1 and reg["new_parameters"] == 0,
            "two_fixed_1500_update_arms")
    a.files(reg["training_code_hashes"], source=True)
    for key in ("manifest", "input_schedule", "source_checkpoint", "evaluation_manifest"):
        a.hash(reg[key], reg[key + "_sha256"])
    a.hash(OUT / "PROTOCOL.md", reg["protocol_sha256"], source=True)
    a.hash(OUT / "cpu_gate.json", reg["cpu_gate_sha256"])
    manifest = a.document(reg["manifest"])
    a.check((len(manifest["rows"]), len(manifest["train"]), len(manifest["dev"])) == (316, 195, 39),
            "original_data_counts")
    a.check(len({r["parent_id"] for r in manifest["train"]}) == 58, "58_training_parents")
    for row in manifest["rows"]:
        a.hash(row["case_path"], row["case_sha256"])
    steps = a.document(reg["input_schedule"])["steps"]
    a.check([s["step"] for s in steps] == list(range(1, 1501)) and
            all(len(s["samples"]) == 8 for s in steps), "frozen_1500_by_8_stream")
    evaluation = a.document(OUT / "eval_manifest.json")
    a.check(evaluation == a.document(BASE / "eval_manifest.json"), "historical_A_panel_exact_identity")
    a.check(a.hash(OUT / "eval_manifest.json") == a.hash(BASE / "eval_manifest.json"),
            "historical_A_panel_byte_identity")
    a.files(evaluation.get("protected", {}))
    qs = evaluation["requests"]
    a.check(len(qs) == 48 and len({q["request_id"] for q in qs}) == 48, "48_unique_frozen_requests")
    for q in qs:
        for key in ("inference_npz", "metadata_npz", "source_npz", "noise_npz"):
            expected = q.get(key + "_sha256", q.get("noise_sha256") if key == "noise_npz" else None)
            a.check(expected is not None, "registered_input_digest_present", key)
            a.hash(q[key], expected)
        a.hash(q["old_layout_npz"], q["old_layout_sha256"])
        a.hash(q["query_descriptor"]["path"], q["query_descriptor"]["sha256"])
    return reg, steps, qs


def audit_training(a, reg, steps):
    results = {}
    for arm in ARMS:
        folder = OUT / "training" / arm
        status = a.document(folder / "status.json")
        a.check((status["status"], status["local_step"], status["global_step"]) == ("COMPLETE", 1500, 13500),
                "training_final_cursor", arm)
        a.check(status.get("head_calls") == 0 and status.get("probe_calls") == 0,
                "no_head_or_neural_probe", arm)
        a.check(status["registration_sha256"] == a.hash(OUT / "training_registration.json"),
                "training_registration_identity", arm)
        ledger = training_ledger(folder / "calls.jsonl", arm)
        a.hash(folder / "calls.jsonl")
        a.check(status["calls"] == ledger, "training_status_matches_reparsed_call_ledger", arm)
        exposures = list(lines(folder / "exposure.jsonl"))
        logs = list(lines(folder / "training_log.jsonl"))
        a.hash(folder / "exposure.jsonl")
        a.hash(folder / "training_log.jsonl")
        a.check(len(exposures) == 12000 and len(logs) == 1500, "training_log_counts", arm)
        for index, exposure in enumerate(exposures):
            expected = steps[index // 8]["samples"][index % 8]
            a.check(all(exposure.get(k) == v for k, v in expected.items()), "cloned_frozen_input_stream",
                    [arm, index // 8 + 1, index % 8])
            a.check((exposure["local_step"], exposure["microbatch"], exposure["global_step"]) ==
                    (index // 8 + 1, index % 8, 12001 + index // 8), "exposure_cursor")
            a.check(math.isfinite(exposure["loss_contribution"]), "finite_training_loss")
            if arm == "ASYNC_LEAD_SEAM":
                a.check(exposure["active_scalar_count"] + exposure["inactive_tip_scalar_count"] ==
                        exposure["all_free_scalar_count"], "async_active_original_denominator_accounting")
        for i, log in enumerate(logs):
            a.check(log["local_step"] == i+1 and log["global_step"] == i+12001, "update_log_cursor")
            a.check(log["free_coordinate_denominator"] == log["all_free_scalar_count"],
                    "original_effective_batch_denominator")
            quotient = log["all_free_velocity_sse"] / log["free_coordinate_denominator"]
            a.check(math.isclose(quotient, log["objective"], rel_tol=4e-6, abs_tol=1e-7),
                    "objective_equals_logged_active_SSE_over_original_count")
        failures = list((folder / "attempts").glob("*/failure.json"))
        a.check(not failures, "no_training_failure_receipts", [str(p) for p in failures])
        for path in sorted((folder / "attempts").glob("*/initialization.json")):
            ini = a.document(path)
            a.check(ini["generator_initial_sha256"] == reg["source_generator_sha256"] and
                    ini["imported_optimizer_sha256"] == reg["source_optimizer_sha256"] and
                    ini["imported_rng_sha256"] == reg["source_rng_sha256"] and ini["head_calls"] == 0,
                    "exact_shared_A_initialization", arm)
        for key in ("checkpoint", "inference_export"):
            r = status[key]
            a.hash(r["path"], r["sha256"])
            a.check((r["local_step"], r["global_step"]) == (1500, 13500), "prespecified_endpoint", key)
        a.check(status["checkpoint"]["generator_state_sha256"] ==
                status["inference_export"]["generator_state_sha256"], "export_matches_final_live_generator", arm)
        # Small receipts authenticate every retained new weight; do not unpickle.
        for weight in sorted(folder.glob("*.pt")):
            receipt = a.document(weight.with_suffix(".json"))
            a.hash(weight, receipt["sha256"])
            a.check(weight.stat().st_size == receipt["bytes"], "retained_weight_size", str(weight))
        results[arm] = dict(updates=1500, train_forwards=12000, backwards=12000,
                            head_forwards=0, fixed_probe_forwards=0,
                            checkpoint=status["checkpoint"], export=status["inference_export"],
                            exposure_rows=12000, call_ledger=ledger)
    return results


def arrays(path):
    import numpy as np
    with np.load(resolve(path), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def byte_equal(x, y):
    import numpy as np
    return x.dtype == y.dtype and x.shape == y.shape and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()


def check_source_C(a, mesh, source, stage, label, rid):
    import numpy as np
    k = len(source["C_original"])
    output = mesh["output"]
    faces, vertices = mesh["faces"], mesh["vertices"]
    a.check(byte_equal(output[:k], source["C_original"]), "source_C_ordered_coordinates_exact", [label, rid, stage])
    a.check(byte_equal(vertices[faces], output), "indexed_export_matches_triangle_array", [label, rid, stage])
    a.check(np.array_equal(mesh["C_original_source_vertex_ids"], source["C_original_source_vertex_ids"]),
            "source_C_vertex_identity_metadata", [label, rid, stage])
    ids = source["C_original_source_vertex_ids"].reshape(-1)
    mapped = faces[:k].reshape(-1)
    table = {}
    for source_id, vertex_id in zip(ids.tolist(), mapped.tolist()):
        table.setdefault(source_id, set()).add(vertex_id)
    a.check(all(len(v) == 1 for v in table.values()) and
            len({next(iter(v)) for v in table.values()}) == len(table),
            "C_source_to_output_vertex_bijection", [label, rid, stage])
    # Exact three-edge hole filling is append-only. Added BRIDGE faces have no
    # original RAW corner lineage, so its producer deliberately retires the
    # legacy arrays to base_* instead of inventing fake anchor rows.
    if "output_anchor_ids" in mesh:
        anchor = mesh["output_anchor_ids"]
        anchor_output = output
        a.check(anchor.shape == faces.shape, "export_anchor_shape", [label, rid, stage])
    else:
        count = int(mesh["base_output_face_count"])
        anchor = mesh["base_output_anchor_ids"]
        anchor_output = output[:count]
        a.check(stage == "CLEAN_FINAL_POST" and
                str(mesh["index_domain"].item()) == "HINGE_V2_APPEND_ONLY_BRIDGE_INDEX_V1" and
                k <= count < len(faces) and anchor.shape == (count, 3),
                "bridge_append_only_anchor_domain", [label, rid, stage])
        a.check(np.array_equal(mesh["parent_post_face_id"][:count], np.arange(count)) and
                np.all(mesh["parent_post_face_id"][count:] == -1) and
                np.all(mesh["parent_source_raw_slot"][count:] == -1) and
                not np.any(mesh["bridge_mask"][:count]) and
                np.all(mesh["bridge_mask"][count:]) and
                np.all(mesh["face_role"][count:] == "BRIDGE"),
                "bridge_explicit_absent_original_lineage", [label, rid, stage])
    mask = anchor >= 0
    a.check(byte_equal(anchor_output[mask], mesh["C_vertices"][anchor[mask]]),
            "retained_shared_C_anchor_coordinates", [label, rid, stage])
    if stage != "RAW":
        a.check(str(mesh["source_C_vertex_ID_topology_status"].item()) == "PASS",
                "source_orientation_topology_receipt", [label, rid, stage])


def audit_outputs(a, qs):
    import numpy as np
    output_rows, last_raw, first_d0 = [], [], []
    for label in LABELS:
        folder = OUT / "evaluation" / label
        inference = a.document(OUT / "evaluation/inference_registrations" / (label+".json"))
        a.files(inference["protected"])
        a.files(inference["code"], source=True)
        a.check(inference["outputs"] == 48 and inference["generator_forwards"] == 2400 and
                inference["new_noise"] == 0 and inference["head_forwards"] == 0 and
                inference["source_labels_read"] is False, "frozen_sampling_budget", label)
        for kind in ("raw", "post"):
            completion = a.document(folder / (kind+"_complete_full.json"))
            a.check(completion["status"] == "COMPLETE" and completion["outputs"] == 48,
                    "all_outputs_complete", [label, kind])
            a.files(completion["files"])
        a.check(not list(folder.glob("*attempt_*/failure.json")), "no_failed_inference_attempts", label)
        for barrier_path in folder.glob("postprocess_attempt_*/all_RAW_barrier.json"):
            barrier = a.document(barrier_path)
            a.check(barrier["status"] == "PASS" and barrier["outputs"] == 96 and
                    barrier["labels"] == list(LABELS), "96_RAW_cross_label_barrier")
            a.files(barrier["files"])
        for identity_path in folder.glob("postprocess_attempt_*/model_identity.json"):
            identity = a.document(identity_path)
            a.check(identity["category"] == "Bench" and identity["state_branch"] == "ema", "official_Bench_D0")
            a.hash(identity["checkpoint_path"], identity["checkpoint_sha256"])
        for q in qs:
            rid, k, n = q["request_id"], int(q["K"]), int(q["N"])
            rawdir, postdir = folder/rid/"raw", folder/rid/"post"
            r, p = a.document(rawdir/"receipt.json"), a.document(postdir/"receipt.json")
            a.files(r["files"])
            a.files(p["files"])
            a.check((r["status"], r["label"], r["request_id"], r["generator_forwards"], r["D0_forwards"]) ==
                    ("COMPLETE", label, rid, 50, 0), "RAW_producer_budget_identity")
            a.check(r["head_forwards"] == 0 and r["source_labels_read"] is False and
                    r["noise_sha256"] == q["noise_sha256"], "no_new_head_noise_or_source_Y", [label, rid])
            a.hash(rawdir/"raw.npz", r["raw_sha256"])
            a.check((p["status"], p["label"], p["request_id"], p["D0_forwards"], p["GEO_calls"]) ==
                    ("COMPLETE", label, rid, 1, 1), "POST_producer_budget_identity")
            a.hash(postdir/"post.npz", p["post_sha256"])
            a.check(p["raw_sha256"] == r["raw_sha256"], "D0_uses_current_RAW")
            proposal = a.document(postdir/"proposal_receipt.json")
            a.hash(postdir/"proposal.npz", proposal["proposal_sha256"])
            a.check(proposal["D0_forwards"] == 1 and proposal["raw_sha256"] == r["raw_sha256"] and
                    proposal["official_checkpoint_sha256"] == p["official_D0_sha256"], "single_D0_proposal_identity")
            ledger = sampling_ledger(rawdir/"calls.jsonl", label.startswith("ASYNC"))
            last_raw.append(ledger["last_unix"])
            first_d0.append(denoiser_ledger(postdir/"calls.jsonl"))
            layout_dir = folder/"layouts"/q["condition_id"]
            layout_receipt = a.document(layout_dir/"receipt.json")
            a.files(layout_receipt["files"])
            a.check(layout_receipt["occupancy_forward_calls"] == 0 and
                    layout_receipt["old_layout_sha256"] == q["old_layout_sha256"], "cached_frozen_layout")
            layout, old_layout = arrays(layout_dir/"layout.npz"), arrays(q["old_layout_npz"])
            for key in ("anchor_ids", "edge_records", "seam_slots", "bulk_slots", "known_mask", "occupied"):
                a.check(byte_equal(layout[key], old_layout[key]), "identical_C_only_layout", key)
            raw, trajectory = arrays(rawdir/"raw.npz"), arrays(rawdir/"trajectory.npz")["state"]
            source = arrays(q["source_npz"])
            supplied_source = arrays(rawdir/"source_condition.npz")
            a.check(set(source) == set(supplied_source) and
                    all(byte_equal(source[key], supplied_source[key]) for key in source), "source_metadata_exact")
            a.check(trajectory.dtype == np.float32 and trajectory.shape == (51, n, 3, 3),
                    "51_FP32_trajectory_states")
            a.check(np.isfinite(trajectory).all(), "finite_trajectory")
            fixed = layout["anchor_ids"] >= 0
            initial = np.zeros((n, 3, 3), np.float32)
            initial[k:] = arrays(q["noise_npz"])["z"]
            initial[fixed] = layout["C_vertices"][layout["anchor_ids"][fixed]]
            a.check(byte_equal(trajectory[0], initial), "initial_state_uses_original_saved_noise")
            a.check(all(byte_equal(state[fixed], initial[fixed]) for state in trajectory), "all51_original_anchors_exact")
            a.check(byte_equal(trajectory[-1], raw["raw"]), "RAW_is_final_trajectory_state")
            tips = (fixed.sum(1) == 2)[:, None] & ~fixed
            if label.startswith("ASYNC"):
                a.check(all(byte_equal(state[tips], trajectory[35][tips]) for state in trajectory[35:]),
                        "async_self_generated_tips_frozen_step35_through50", rid)
                a.check(r["audit"]["tips_frozen_after_global_step"] == 35 and
                        r["audit"]["tip_freeze_value"] == "ACTUAL_GENERATED_STATE; NO_SOURCE_Y",
                        "async_freeze_receipt")
            for stage, mesh in (
                ("RAW", raw), ("BASE_POST", arrays(postdir/"post.npz")),
                ("CLEAN_FINAL_POST", arrays(OUT/"cpu"/label/rid/"CLEAN_FINAL_POST/post.npz")),
            ):
                check_source_C(a, mesh, source, stage, label, rid)
            anchors = a.document(postdir/"anchor_identity.json")
            restoration = a.document(postdir/"source_restoration.json")
            a.check(anchors["status"] == "PASS" and anchors["final_anchor_vertex_identity_verified"] is True,
                    "D0_GEO_anchor_identity_receipt")
            a.check(restoration["status"] == "PASS" and restoration["source_C_order_fixed_after_restoration"] is True,
                    "source_winding_restoration_receipt")
            cpu = a.document(OUT/"cpu"/label/rid/"complete.json")
            a.check(cpu["status"] == "COMPLETE" and all(cpu[key] == 0 for key in
                    ("model_calls", "D0_calls", "GPU_calls", "new_noise", "new_queries")), "CPU_postprocess_budget")
            a.files(cpu["files"])
            output_rows.append(dict(label=label, request_id=rid, split=q["split"], part_type=q["part_type"],
                                    generator_forwards=50, D0_forwards=1, fixed_anchor_corners=int(fixed.sum()),
                                    seam_tip_count=int(tips.sum()), source_C="PASS", trajectory="PASS",
                                    late_tip_freeze="PASS" if label.startswith("ASYNC") else "NA_SYNCHRONOUS"))
    a.check(len(output_rows) == 96 and len(first_d0) == 96, "96_new_outputs_and_D0_calls")
    a.check(min(first_d0) >= max(last_raw), "all_RAW_forwards_finish_before_first_D0",
            dict(last_RAW_unix=max(last_raw), first_D0_unix=min(first_d0)))
    return output_rows


def score_folder(root, label, rid, stage):
    if stage == "RAW":
        return root/"evaluation/scores"/label/rid/"RAW"
    return root/"cpu"/label/rid/stage/"scores"


def audit_scores(a, qs):
    old, new = a.document(BASE/"evaluation/score_registration.json"), a.document(OUT/"evaluation/score_registration.json")
    old_sources = {str(resolve(p).resolve()): h for p, h in old["source_hashes"].items()}
    new_sources = {str(resolve(p).resolve()): h for p, h in new["source_hashes"].items()}
    shared = sorted(old_sources.keys() & new_sources.keys())
    a.check(len(shared) >= 20, "shared_original_scientific_scorers_present", len(shared))
    a.check(old["stages"] == new["stages"] == list(STAGES) and old["chain"] == new["chain"],
            "same_original_scoring_and_postprocess_chain")
    for path in shared:
        a.check(old_sources[path] == new_sources[path], "baseline_and_new_scorer_hash_identical", path)
    a.files(new_sources, source=True)
    a.files({p: old_sources[p] for p in shared}, source=True)
    collection = a.document(OUT/"results/completion.json")
    a.check(collection["status"] == "COMPLETE" and collection["rows"] == 432 and
            collection["new_score_rows"] == 288 and collection["reused_score_rows"] == 144,
            "full_lossless_score_collection")
    a.files(collection["files"])
    a.files(collection["source_rows"])
    a.files(collection["source_receipts"])
    collected = csv_rows(OUT/"results/per_output.csv")
    by = {(r["label"], r["request_id"], r["stage"]): r for r in collected}
    a.check(len(by) == len(collected) == 432, "unique_per_output_CSV_rows")
    originals = {}
    for q in qs:
        rid = q["request_id"]
        for stage in STAGES:
            baseline_keys = None
            for label in ALL_LABELS:
                root = BASE if label == "A_S1500" else OUT
                folder = score_folder(root, label, rid, stage)
                r = a.document(folder/"complete.json")
                a.check(r["status"] == "PASS" and r["model_calls"] == 0, "existing_score_complete")
                a.files(r["files"])
                row = a.document(folder/"row.json")
                if baseline_keys is None:
                    baseline_keys = set(row)
                a.check(set(row) == baseline_keys, "same_original_metric_field_set", [label, rid, stage])
                out = by[label, rid, stage]
                for key, value in row.items():
                    if key in {"label", "stage", "part_type", "closed_C", "seam_applicability"}:
                        continue
                    expected = "" if value is None else str(value)
                    a.check(out.get(key) == expected, "CSV_preserves_original_metric_value", [label, rid, stage, key])
                origin = "REUSED_HISTORICAL_A" if label == "A_S1500" else "NEW_REGISTERED_OUTPUT"
                a.check(out["result_origin"] == origin, "comparison_origin_explicit")
                originals[label, rid, stage] = row
    a.check(len(originals) == 432, "all144_historical_and288_new_score_rows_authenticated")
    return dict(score_rows=432, new_rows=288, historical_A_rows=144,
                shared_scientific_source_count=len(shared), metrics=collection["metric_fields"])


def audit_review(a, qs):
    gallery = a.document(OUT/"gallery/completion.json")
    definition = a.document(OUT/"gallery/definition.json")
    manifest = a.document(OUT/"gallery/manifest.json")
    a.check(definition["manifest_sha256"] == a.hash(OUT/"gallery/manifest.json"),
            "gallery_manifest_definition_identity")
    a.files(definition["sources"])
    a.files(definition["protected"], source=False)
    a.check(gallery["sheets"] == len(gallery["images"]) == 144, "144_rendered_three_model_sheets")
    for image in gallery["images"]:
        a.hash(image["path"], image["image_sha256"])
        receipt = a.document(resolve(image["path"]).with_suffix(".json"))
        a.check(receipt["definition_sha256"] == a.hash(OUT/"gallery/definition.json") and
                receipt["image_sha256"] == image["image_sha256"], "gallery_definition_identity")
    a.check(manifest["sheets"] == 144, "inherited_gallery_scope")
    review = a.document(OUT/"visual_review/completion.json")
    a.check(review["status"] == "ACTUAL_AI_PROXY_REVIEW_COMPLETE" and
            (review["reviewed_cells"], review["missing_cells"], review["reviewed_requests"]) == (432, 0, 48),
            "actual_visual_review_complete")
    a.check(review["reviewer_kind"] == "AI_PROXY" and review["blind"] is False and
            review["formal_human_ratings"] == 0, "visual_evidence_scope_explicit")
    a.files(review["source_review_files"])
    a.files(review["authenticated_image_files"])
    a.hash(review["per_cell_csv"], review["per_cell_sha256"])
    protocol = a.document(OUT/"visual_review/protocol.json")
    assignments = a.document(OUT/"visual_review/assignments.json")
    a.check(a.hash(OUT/"visual_review/protocol.json") == review["protocol_sha256"] and
            a.hash(OUT/"visual_review/assignments.json") == review["assignments_sha256"], "review_protocol_identity")
    cells, seen = [], set()
    for path in review["source_review_files"]:
        document = a.document(path)
        reviewer = document["reviewer"]
        a.check(document.get("schema") == protocol["schema"] and
                document.get("reviewer_kind") == "AI_PROXY" and document.get("blind") is False and
                document.get("protocol_sha256") == review["protocol_sha256"] and
                document.get("assignments_sha256") == review["assignments_sha256"],
                "actual_review_envelope_identity")
        for row in document["rows"]:
            key = (row["request_id"], row["model"], row["stage"])
            a.check(key not in seen and row["request_id"] in assignments["assignments"][reviewer],
                    "unique_assigned_actual_review_cell")
            seen.add(key)
            a.check(row["actually_viewed"] is True and row["views_and_sections_checked"] is True and
                    isinstance(row["notes"], str) and bool(row["notes"].strip()), "actual_observation_attestation")
            a.check(all(row.get(axis) in choices for axis, choices in protocol["axes"].items()),
                    "valid_review_axis_values")
            image = OUT/"gallery"/(row["stage"]+"__"+row["request_id"]+".png")
            a.hash(image, row["image_sha256"])
            cells.append(row)
    expected = {(q["request_id"], label, stage) for q in qs for label in ALL_LABELS for stage in STAGES}
    a.check(seen == expected and len(cells) == 432, "actual_review_exact_complete_panel")
    return dict(images=144, actual_AI_PROXY_cells=432, human_ratings=0, blind=False)


def snapshot_sources(a):
    own = Path(__file__).resolve().parent
    previous = ROOT/"r2/pilots/bench_adaptation_strength_v1"
    wrapper_names = ("training.py", "sample.py", "post_d0.py", "evaluation.py", "eval_plan.py")
    for name in wrapper_names:
        if (previous/name).is_file():
            a.hash(previous/name, source=True)
    for path in sorted(own.glob("*.py")):
        a.hash(path, source=True)
    for name in ("RUN_AND_RESUME.md", "CODE_REVIEW.md", "perturbed_contract.md"):
        path = OUT/name
        if path.exists():
            a.hash(path, source=True)
    rows = []
    for key in sorted(a.source_paths):
        p = Path(key)
        try:
            relative = p.relative_to(ROOT.resolve())
        except ValueError:
            # Historical scorer registration intentionally includes this
            # adjacent read-only repository. Copy only registered small source
            # files, under a separate archive prefix; never write there.
            scoring_root = (ROOT.parent/"MeshFlow-Control").resolve()
            try:
                relative = Path("external_MeshFlow-Control")/p.relative_to(scoring_root)
            except ValueError as exc:
                raise ValueError("Unexpected registered source root: " + key) from exc
        a.check(p.stat().st_size <= 2 << 20 and p.suffix.lower() not in (".pt", ".pth", ".npz"),
                "small_source_only_snapshot", key)
        dest = a.dest/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        copied = sha(dest)
        a.check(copied == a.hashed[key]["sha256"], "snapshot_byte_identity", key)
        rows.append(dict(original=key, snapshot=str(dest.relative_to(a.dest)),
                         bytes=dest.stat().st_size, sha256=copied))
    changes = []
    for name in wrapper_names:
        old, new = previous/name, own/name
        if not old.exists() or not new.exists():
            continue
        changes.extend(difflib.unified_diff(old.read_text(encoding="utf-8").splitlines(True),
                       new.read_text(encoding="utf-8").splitlines(True),
                       fromfile="baseline_A/"+name, tofile="state_schedule/"+name))
    for name in ("async_core.py", "perturbed_core.py"):
        changes.extend(difflib.unified_diff([], (own/name).read_text(encoding="utf-8").splitlines(True),
                       fromfile="/dev/null", tofile="state_schedule/"+name))
    (a.dest/"code.diff").write_text("".join(changes), encoding="utf-8", newline="\n")
    write_new(a.dest/"source_manifest.json", dict(schema=SCHEMA, rows=rows,
                diff_scope="Historical A shared orchestration wrappers versus new wrappers; complete added mechanism modules",
                weights_included=False, source_bytes=sum(r["bytes"] for r in rows)))
    with (a.dest/"weights_index.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256", "verified_by"])
        writer.writeheader()
        writer.writerows(a.weights[p] for p in sorted(a.weights))
    return dict(source_files=len(rows), source_bytes=sum(r["bytes"] for r in rows),
                indexed_weights=len(a.weights), copied_weights=0, deserialized_weights=0)


def run():
    ready = readiness()
    if ready["status"] != "READY":
        return ready  # no audit folder, no mutation, no false final receipt
    if "torch" in sys.modules:
        raise RuntimeError("Closeout must be an independent pure-CPU process without torch")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = OUT/"final_audits"/timestamp
    dest.mkdir(parents=True, exist_ok=False)
    write_new(dest/"started.json", dict(schema=SCHEMA, started_utc=timestamp,
                model_calls=0, scoring_calls=0, rendering_calls=0, new_noise=0))
    a, started = Audit(dest), time.perf_counter()
    try:
        reg, steps, qs = audit_registration(a)
        training = audit_training(a, reg, steps)
        outputs = audit_outputs(a, qs)
        scores = audit_scores(a, qs)
        review = audit_review(a, qs)
        archive = snapshot_sources(a)
        a.check("torch" not in sys.modules, "no_neural_framework_import")
        # Detect writes racing this read-only audit without repeatedly reading
        # multi-GB weights; each file was content-hashed once above.
        for path, facts in a.hashed.items():
            stat = Path(path).stat()
            a.check((stat.st_size, stat.st_mtime_ns) == (facts["bytes"], facts["mtime_ns"]),
                    "authenticated_file_unchanged_during_audit", path)
        write_new(dest/"checks.json", dict(status="PASS", rows=a.checks))
        write_new(dest/"verified_hashes.json", a.hashed)
        write_new(dest/"per_output_contracts.json", dict(status="PASS", rows=outputs))
        result = dict(status="PASS", schema=SCHEMA, experiment=EXPERIMENT,
                      audit_directory=str(dest), checks=len(a.checks),
                      training=training, outputs=96, new_generator_forwards=4800,
                      new_D0_forwards=96, new_training_updates=3000,
                      new_training_forwards=24000, new_training_backwards=24000,
                      new_head_forwards=0, fixed_neural_probes=0,
                      scores=scores, review=review, archive=archive,
                      audit_model_calls=0, audit_scoring_calls=0, audit_rendering_calls=0,
                      audit_new_noise=0, audit_weight_deserializations=0,
                      seconds=time.perf_counter()-started,
                      limitation="Checks saved evidence and actual-review attestations; no new geometry quality claim or independent re-scoring.",
                      files={str(p): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()})
        write_new(dest/"completion.json", result)
        return result
    except BaseException as exc:
        write_new(dest/"failure.json", dict(status="FAILED", schema=SCHEMA, error=repr(exc),
                    traceback=traceback.format_exc(), checks=a.checks, model_calls=0,
                    scoring_calls=0, new_noise=0, no_scientific_replay=True))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-ready", action="store_true")
    args = p.parse_args()
    result = readiness() if args.check_ready else run()
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "training")},
                     ensure_ascii=False, indent=2))
