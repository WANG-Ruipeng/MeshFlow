"""Independent, read-only final evidence audit. No torch, models, scoring or rendering.

Run after training/evaluation/review finish. Missing completion evidence produces WAIT;
contradictory completed evidence produces FAIL. Every invocation has its own receipt.
This is outside the frozen scientific package and never edits original artifacts.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / "experiments/r2_bench_adaptation_strength_v1"
ARM = "COVERAGE_SEAM"
LABEL = "COVERAGE_S1500"
BASE = "A_S1500"
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/"):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and len(text) > 2 and text[1:3] == ":/":
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def read(p):
    return json.loads(path(p).read_text(encoding="utf-8-sig"))


def lines(p):
    return [json.loads(s) for s in path(p).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def arrays(p):
    with np.load(path(p), allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


class Audit:
    def __init__(self):
        self.checks = collections.Counter()
        self.hashes = {}
        self.missing = []
        self.facts = {}

    def require(self, ok, name, detail=None):
        if not ok:
            raise AssertionError(name + (": " + str(detail) if detail is not None else ""))
        self.checks[name] += 1

    def sha(self, p):
        p = path(p).resolve()
        if str(p) not in self.hashes:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(block)
            self.hashes[str(p)] = h.hexdigest()
        return self.hashes[str(p)]

    def verify(self, mapping):
        for p, expected in mapping.items():
            self.require(self.sha(p) == expected, "artifact_hash", p)

    def complete(self, p, accepted=("COMPLETE",)):
        p = path(p)
        if not p.exists():
            self.missing.append({"path": str(p), "reason": "NOT_YET_PRESENT"})
            return None
        obj = read(p)
        if obj.get("status") not in accepted:
            self.missing.append({"path": str(p), "reason": "NOT_COMPLETE", "status": obj.get("status")})
            return None
        return obj


def prerequisites(a):
    """No evidence-completeness claim before every scientific stage and review exist."""
    qpath = OUT / "eval_manifest.json"
    if not qpath.exists():
        a.missing.append({"path": str(qpath), "reason": "NOT_YET_PRESENT"})
        return None
    qs = read(qpath)["requests"]
    a.complete(OUT / f"training/{ARM}/status.json")
    a.complete(OUT / f"evaluation/{LABEL}/raw_complete_full.json")
    a.complete(OUT / f"evaluation/{LABEL}/post_complete_full.json")
    a.complete(OUT / "results/completion.json")
    a.complete(OUT / "gallery/completion.json", ("RENDER_COMPLETE_PENDING_ACTUAL_REVIEW", "COMPLETE"))
    for q in qs:
        rid = q["request_id"]
        for kind in ("raw", "post"):
            a.complete(OUT / f"evaluation/{LABEL}/{rid}/{kind}/receipt.json")
        a.complete(OUT / f"cpu/{LABEL}/{rid}/complete.json")
        for stage in STAGES:
            folder = score_folder(LABEL, rid, stage)
            a.complete(folder / "complete.json", ("PASS",))
    review_files = sorted((OUT / "visual_review").glob("review_*.json"))
    keys = []
    for f in review_files:
        keys.extend((r["request_id"], r["label"], r["stage"]) for r in read(f).get("rows", []))
    expected = {(q["request_id"], label, stage) for q in qs for label in (BASE, LABEL) for stage in STAGES}
    if set(keys) != expected:
        a.missing.append({"path": str(OUT / "visual_review"), "reason": "ACTUAL_REVIEW_INCOMPLETE",
                          "expected": len(expected), "present_unique": len(set(keys)),
                          "missing_cells": sorted(expected - set(keys))})
    return qs


def frozen_inputs(a, qs):
    reg = read(OUT / "registration.json")
    a.require(reg == read(OUT / "training_registration.json"), "registrations_identical")
    a.require(reg["status"] == "FROZEN" and reg["updates"] == 1500 and reg["batch"] == 8
              and reg["microbatch"] == 1 and reg["arms"] == [ARM], "fixed_single_arm_budget")
    a.verify(reg["training_code_hashes"])
    for key in ("manifest", "input_schedule", "source_checkpoint", "eval_manifest"):
        a.verify({reg[key]: reg[key + "_sha256"]})
    gate = read(OUT / "cpu_gate.json")
    a.require(gate["status"] == "PASS", "prior_independent_cpu_gate_PASS")
    a.verify(read(gate["input_hashes"]))
    a.require(read(OUT / "eval_manifest.json") == read(OLD / "eval_manifest.json"), "old48_manifest_exact")
    a.require(len(qs) == 48 and len({q["request_id"] for q in qs}) == 48, "48_unique_requests")
    for q in qs:
        for name in ("noise_npz", "inference_npz", "source_npz", "metadata_npz"):
            a.verify({q[name]: q[name + "_sha256"]})
        a.verify({q["old_layout_npz"]: q["old_layout_sha256"],
                  q["query_descriptor"]["path"]: q["query_descriptor"]["sha256"]})
    manifest = read(reg["manifest"])
    train = manifest["train"]
    a.require((len(train), len({r["parent_id"] for r in train}), len(manifest["dev"])) == (321, 79, 39),
              "realized321_79_39")
    a.require(not ({r["parent_id"] for r in train} & {q["parent_id"] for q in qs if q["split"] == "DEV"}),
              "DEV_never_train")
    a.facts["protected_CPU_gate_inputs"] = len(read(gate["input_hashes"]))
    return reg


def training(a, reg):
    folder = OUT / "training" / ARM
    status = read(folder / "status.json")
    a.require((status["local_step"], status["global_step"]) == (1500, 13500), "final_update_identity")
    start = read(folder / "checkpoint_step0.json")
    for actual_key, registered_key in (("generator_state_sha256", "source_generator_sha256"),
                                       ("optimizer_state_sha256", "source_optimizer_sha256"),
                                       ("rng_sha256", "source_rng_sha256")):
        a.require(start[actual_key] == reg[registered_key], "A0_tensor_state_identity_in_receipts", actual_key)
    a.verify({folder / "checkpoint_step0.pt": start["sha256"]})
    for name in ("checkpoint_step1500", "generator_step1500"):
        r = read(folder / (name + ".json"))
        a.require((r["local_step"], r["global_step"]) == (1500, 13500), "weight_receipt_endpoint")
        a.verify({folder / (name + ".pt"): r["sha256"]})
    events = lines(folder / "calls.jsonl")
    begins, dones = {}, {}
    scientific = collections.defaultdict(set)
    for e in events:
        a.require(e["event"] in ("CALL_BEGIN", "CALL_DONE"), "ledger_event_type", e["event"])
        key = (e["attempt_id"], e["call_id"])
        bag = begins if e["event"] == "CALL_BEGIN" else dones
        a.require(key not in bag, "no_duplicate_call_id")
        bag[key] = e
        if e["event"] == "CALL_BEGIN":
            c = e["context"]
            slot = (c["local_step"], c.get("microbatch"))
            a.require(slot not in scientific[e["kind"]], "no_replayed_scientific_call", [e["kind"], slot])
            scientific[e["kind"]].add(slot)
    a.require(set(begins) == set(dones), "every_call_returned")
    for key, start in begins.items():
        end = dones[key]
        a.require(start["kind"] == end["kind"] and start["context"] == end["context"], "call_pair_identity")
    expected_micro = {(s, m) for s in range(1, 1501) for m in range(8)}
    a.require(scientific["train_forward"] == expected_micro, "exact12000_training_forwards")
    a.require(scientific["train_backward"] == expected_micro, "exact12000_training_backwards")
    a.require(scientific["optimizer_update"] == {(s, None) for s in range(1, 1501)}, "exact1500_optimizer_updates")
    a.require(set(scientific) == {"train_forward", "train_backward", "optimizer_update"}, "no_other_training_calls")
    steps = read(reg["input_schedule"])["steps"]
    exposure = lines(folder / "exposure.jsonl")
    a.require(len(exposure) == 12000, "exposure_rows12000")
    by = {(x["local_step"], x["microbatch"]): x for x in exposure}
    a.require(set(by) == expected_micro and len(by) == len(exposure), "exposure_unique_coverage")
    for s in steps:
        for m, planned in enumerate(s["samples"]):
            actual = by[s["step"], m]
            a.require(all(actual.get(k) == v for k, v in planned.items()), "actual_exposure_matches_schedule")
            a.require(actual["global_step"] == 12000 + s["step"], "exposure_global_step")
    logs = lines(folder / "training_log.jsonl")
    a.require([x["local_step"] for x in logs] == list(range(1, 1501)), "1500_unique_loss_rows")
    for row in logs:
        a.require(all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int))), "finite_training_log")
        a.require(row["free_coordinate_denominator"] == row["all_free_scalar_count"] > 0, "FM_denominator_identity")
        samples = [by[row["local_step"], m] for m in range(8)]
        a.require(sum(x["all_free_scalar_count"] for x in samples) == row["all_free_scalar_count"], "exposure_FM_denominator")
    a.facts["training"] = dict(updates=1500, forwards=12000, backwards=12000, replayed_calls=0,
                                checkpoint_deserializations_by_audit=0, event_rows=len(events),
                                attempt_ids=sorted({e["attempt_id"] for e in events}))


def event_time(e):
    for key in ("time_unix", "unix", "time", "timestamp"):
        if key in e and isinstance(e[key], (int, float)):
            return float(e[key])
    if "utc" in e:
        return datetime.datetime.fromisoformat(e["utc"].replace("Z", "+00:00")).timestamp()
    raise ValueError("Missing numeric event time: " + str(e))


def meshes(a, qs):
    raw_ends, d0_starts = [], []
    preservation = []
    for q in qs:
        rid, k, n = q["request_id"], q["K"], q["N"]
        root = OUT / "evaluation" / LABEL / rid
        rr = read(root / "raw/receipt.json")
        pr = read(root / "post/receipt.json")
        a.verify(rr["files"])
        a.verify(pr["files"])
        a.require(rr["generator_forwards"] == 50 and rr["D0_forwards"] == 0
                  and rr["head_forwards"] == 0 and not rr["source_labels_read"], "RAW_model_budget")
        a.require(pr["D0_forwards"] == 1 and pr["GEO_calls"] == 1 and not pr["source_labels_read"], "single_D0_contract")
        for kind, start_name, end_name, count in (("raw", "FORWARD_STARTED", "FORWARD_COMPLETE", 50),
                                                ("post", "D0_FORWARD_STARTED", "D0_FORWARD_RETURNED", 1)):
            ev = lines(root / kind / "calls.jsonl")
            starts = [e for e in ev if e["event"] == start_name]
            ends = [e for e in ev if e["event"] == end_name]
            a.require(len(ev) == 2 * count and len(starts) == len(ends) == count, "exact_inference_call_pairs")
            if kind == "raw":
                a.require([e["step"] for e in starts] == list(range(50)) == [e["step"] for e in ends], "Euler50_steps")
                raw_ends.extend(map(event_time, ends))
            else:
                d0_starts.extend(map(event_time, starts))
        cond = arrays(q["inference_npz"])
        original = cond["C_vertices"][cond["C_faces_original"]]
        canonical = cond["C_vertices"][cond["C_faces_model"]]
        layout = arrays(OUT / f"evaluation/{LABEL}/layouts/{q['condition_id']}/layout.npz")
        old_layout = arrays(q["old_layout_npz"])
        for name in ("anchor_ids", "edge_records", "seam_slots", "bulk_slots", "known_mask", "occupied"):
            a.require(np.array_equal(layout[name], old_layout[name]), "layout_array_exact_old", name)
        traj = arrays(root / "raw/trajectory.npz")["state"]
        raw = arrays(root / "raw/raw.npz")
        pre = arrays(root / "post/pre_geo.npz")
        a.require(traj.shape == (51, n, 3, 3) and np.isfinite(traj).all(), "51_finite_states")
        anchors = layout["anchor_ids"]
        fixed = anchors >= 0
        expected = np.zeros((n, 3, 3), np.float32)
        expected[k:] = arrays(q["noise_npz"])["z"]
        expected[fixed] = layout["C_vertices"][anchors[fixed]]
        a.require(np.array_equal(traj[0], expected), "same_frozen_noise_initial_state")
        a.require(np.array_equal(traj[:, fixed], np.broadcast_to(expected[fixed], traj[:, fixed].shape)), "all51_anchor_positions_exact")
        a.require(np.array_equal(raw["raw"], traj[-1]), "RAW_is_Euler50_terminal")
        a.require(np.array_equal(raw["raw"][:k], canonical), "RAW_C_exact_canonical")
        a.require(np.array_equal(pre["output"][:k], canonical), "PRE_GEO_C_exact_canonical")
        for stage, mp in (("BASE_POST", root / "post/post.npz"),
                          ("CLEAN_FINAL_POST", OUT / f"cpu/{LABEL}/{rid}/CLEAN_FINAL_POST/post.npz")):
            mesh = arrays(mp)
            tri = mesh["vertices"][mesh["faces"]]
            a.require(np.array_equal(tri[:k], original), "POST_C_exact_source_corner_order", [rid, stage])
            a.require(np.isfinite(tri).all(), "finite_POST_geometry")
        preservation.append(dict(request_id=rid, RAW_C=True, PRE_GEO_C=True, BASE_POST_C=True, FINAL_C=True,
                                 coordinate_contract="EXACT; NOT A GLOBAL VERTEX-ID CONNECTIVITY CERTIFICATE"))
        cpu = read(OUT / f"cpu/{LABEL}/{rid}/complete.json")
        a.verify(cpu["files"])
        a.require(cpu["expanded_base_scores"] == 1 and cpu["final_scores"] == 1, "one_CPU_score_per_POST_stage")
    a.require(max(raw_ends) < min(d0_starts), "all48_RAW_before_first_D0")
    for filename in ("raw_complete_full.json", "post_complete_full.json"):
        receipt = read(OUT / "evaluation" / LABEL / filename)
        a.verify(receipt["files"])
        a.require(receipt["outputs"] == 48, "48_completed_outputs")
    a.facts["inference"] = dict(raw=48, Euler_forwards=2400, D0_forwards=48, head_forwards=0,
                                 all_RAW_before_D0=True, C_preservation=preservation)


def score_folder(label, rid, stage):
    root = OLD if label == BASE else OUT
    if stage == "RAW":
        return root / f"evaluation/scores/{label}/{rid}/RAW"
    return root / f"cpu/{label}/{rid}/{stage}/scores"


def csv_equal(text, value):
    if value is None:
        return text == ""
    if isinstance(value, bool):
        return text == str(value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value)) and float(text) == value
    return text == str(value)


def scores(a, qs):
    completion = read(OUT / "results/completion.json")
    a.require((completion["rows"], completion["new_score_rows"], completion["reused_score_rows"]) == (288, 144, 144),
              "all288_score_rows")
    for key in ("files", "source_rows", "source_receipts"):
        a.verify(completion[key])
    for root in (OLD, OUT):
        a.verify(read(root / "evaluation/score_registration.json")["source_hashes"])
    a.verify(completion["scorer_identity"]["common_sources"])
    with (OUT / "results/per_output.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {(r["request_id"], r["label"], r["stage"]): r for r in rows}
    expected = {(q["request_id"], label, stage) for q in qs for label in (BASE, LABEL) for stage in STAGES}
    a.require(len(rows) == len(by) == 288 and set(by) == expected, "CSV_full_unique_matrix")
    na = []
    for q in qs:
        for label in (BASE, LABEL):
            for stage in STAGES:
                rid = q["request_id"]
                folder = score_folder(label, rid, stage)
                rec = read(folder / "complete.json")
                a.require(rec["status"] == "PASS" and rec["model_calls"] == 0, "scoring_receipt_PASS_zero_model")
                a.verify(rec["files"])
                source, flat = read(folder / "row.json"), by[rid, label, stage]
                for key, value in source.items():
                    if key in ("label", "stage", "part_type", "closed_C", "seam_applicability"):
                        continue
                    a.require(key in flat and csv_equal(flat[key], value), "lossless_source_score_field", [rid, label, stage, key])
                a.require(source["C_preserved_exact"] is True, "scored_C_exact")
                nulls = [key for key, value in source.items() if value is None]
                if q["closed_C"]:
                    a.require(source["seam_to_bulk_status"] == "NA_CLOSED_C_NO_SEAM_INTERFACE", "closed_interface_NA_reason")
                    for key in ("retained_seam_faces", "free_only_reached_bulk_area_fraction", "bulk_reach_ge_0_30_descriptive"):
                        a.require(source[key] is None, "closed_interface_not_zero_imputed", key)
                na.append(dict(request_id=rid, label=label, stage=stage, null_fields=nulls,
                               seam_status=source.get("seam_to_bulk_status")))
    a.facts["scores"] = dict(new=144, historical_reused=144, merged=288,
                             all_source_fields_preserved=True, NA_records=na)


def reviews(a, qs):
    manifest = read(OUT / "gallery/manifest.json")
    done = read(OUT / "gallery/completion.json")
    a.require(manifest["sheets"] == done["sheets"] == 144, "144_rendered_comparison_pages")
    pages = {p["page_id"]: p for p in manifest["pages"]}
    images = {p["page_id"]: p for p in done["images"]}
    a.require(set(pages) == set(images) and len(pages) == 144, "gallery_complete_identity")
    a.verify(manifest["protected"])
    definition = read(OUT / "gallery/definition.json")
    a.verify(definition["sources"])
    rubric = a.sha(OUT / "REVIEW_RUBRIC.md")
    by = {}
    page_preference = collections.defaultdict(set)
    for review_file in sorted((OUT / "visual_review").glob("review_*.json")):
        wrapper = read(review_file)
        a.require(wrapper["rubric_sha256"] == rubric, "review_frozen_rubric")
        a.require(wrapper["reviewer"].startswith("AI_PROXY"), "reviewer_identity")
        for r in wrapper["rows"]:
            key = (r["request_id"], r["label"], r["stage"])
            a.require(key not in by, "one_actual_review_per_cell", key)
            by[key] = r
            a.require(r["reviewer"] == wrapper["reviewer"], "reviewer_wrapper_identity")
            pg, im = pages[r["page_id"]], images[r["page_id"]]
            for field in ("request_id", "condition_id", "parent_id", "split", "part_type"):
                a.require(r[field] == pg[field], "review_metadata_identity", field)
            a.require(r["stage"] == pg["sheet"] and r["label"] in (BASE, LABEL), "review_stage_label")
            a.require(path(r["image_path"]).resolve() == path(im["path"]).resolve(), "review_actual_image_path")
            a.require(r["image_sha256"] == im["image_sha256"] == a.sha(im["path"]), "review_actual_image_hash")
            for field in ("C_use", "layered_or_conflicting", "visible_fragment"):
                a.require(r[field] in ("YES", "NO", "UNCERTAIN"), "review_label_schema", field)
            a.require(r["whole_shape"] in ("COMPLETE", "PARTIAL", "UNCERTAIN"), "review_shape_schema")
            a.require(isinstance(r["note"], str) and len(r["note"].strip()) > 5, "specific_image_note_present")
            a.require(r["pair_preference"] in ("COVERAGE", "A", "TIE", "UNCERTAIN"), "pair_preference_schema")
            page_preference[r["page_id"]].add(r["pair_preference"])
        a.sha(review_file)
    expected = {(q["request_id"], label, stage) for q in qs for label in (BASE, LABEL) for stage in STAGES}
    a.require(set(by) == expected and len(by) == 288, "288_complete_actual_review_records")
    a.require(all(len(v) == 1 for v in page_preference.values()) and len(page_preference) == 144,
              "consistent_pair_preference")
    a.facts["visual_review"] = dict(pages=144, cells=288, rubric_sha256=rubric,
                                    reviewers=sorted({r["reviewer"] for r in by.values()}),
                                    limitation="Nonblind AI_PROXY records; geometry scores separate; audit checks provenance/coverage, not image judgment correctness")


def main():
    started = time.perf_counter()
    audit = Audit()
    destination = OUT / "final_audits" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(status="WAIT", schema="BENCH_COVERAGE_INDEPENDENT_FINAL_AUDIT_V1",
                  neural_forwards=0, GPU_calls=0, scoring_calls=0, checkpoint_deserializations=0,
                  scientific_inputs_modified=0, script_sha256=audit.sha(__file__))
    try:
        qs = prerequisites(audit)
        if not audit.missing:
            reg = frozen_inputs(audit, qs)
            training(audit, reg)
            meshes(audit, qs)
            scores(audit, qs)
            reviews(audit, qs)
            audit.require("torch" not in sys.modules, "no_torch_import")
            result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
    result.update(checks=dict(audit.checks), missing=audit.missing, facts=audit.facts,
                  hashed_files=len(audit.hashes), seconds=time.perf_counter() - started,
                  input_hashes=str(destination / "input_hashes.json"))
    (destination / "input_hashes.json").write_text(json.dumps(audit.hashes, indent=2), encoding="utf-8")
    receipt = destination / "receipt.json"
    receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "final_verify_latest.json").write_text(json.dumps(dict(status=result["status"], receipt=str(receipt)), indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "seconds", "hashed_files", "neural_forwards")}, indent=2))
    print(str(receipt))
    return 0 if result["status"] == "PASS" else (3 if result["status"] == "WAIT" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
