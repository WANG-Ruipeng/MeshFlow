"""Read-only final accounting audit; never imports a model or repairs an output.

Run only after training, paired RAW/D0, scores and gallery have completed. The
default availability check performs no incomplete-stage audit. This validates
execution and artifact identities, not the scientific superiority of a model
and not whether a human/AI reviewer has actually inspected the rendered pages.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
from .common import ROOT, OUT, ARM, LABEL, REFERENCE_LABEL, REFERENCE_ROOT, read, resolve, sha, write_new

LABELS = (REFERENCE_LABEL, LABEL)
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
DONE = OUT / "final_audit.json"
EXPECTED_TRAIN = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                      head_forward=11334, head_backward=11334, head_optimizer_update=1500)


def ready():
    """Availability only: running stages remain PENDING, never FAIL."""
    required = {OUT / "training" / ARM / "status.json": "COMPLETE",
                OUT / "results/completion.json": "COMPLETE",
                OUT / "gallery/completion.json": "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW"}
    for label in LABELS:
        required[OUT / "evaluation" / label / "raw_complete_full.json"] = "COMPLETE"
        required[OUT / "evaluation" / label / "post_complete_full.json"] = "COMPLETE"
    pending = [dict(path=str(path), expected_status=status,
                    actual_status=read(path).get("status") if path.exists() else "MISSING")
               for path, status in required.items()
               if not path.exists() or read(path).get("status") != status]
    return dict(status="PENDING_INPUTS" if pending else "READY", pending=pending,
                model_calls=0, artifact_audit_executed=False)


class Reader:
    def __init__(self):
        self.files = {}

    def freeze(self, path, expected=None):
        path = resolve(path)
        key = path.as_posix()
        actual = self.files.get(key)
        if actual is None:
            actual = sha(path)
            self.files[key] = actual
        if expected is not None and expected != actual:
            raise ValueError("Artifact SHA mismatch: " + str(path))
        return actual

    def json(self, path, expected=None):
        self.freeze(path, expected)
        return read(path)

    def jsonl(self, path):
        self.freeze(path)
        with resolve(path).open(encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def csv(self, path):
        self.freeze(path)
        with resolve(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def verify(self, files):
        for path, expected in files.items():
            self.freeze(path, expected)


def require(value, message):
    if not value:
        raise ValueError(message)


def check_training(a, reg, trainreg):
    folder = OUT / "training" / ARM
    state = a.json(folder / "status.json")
    require((state["status"], state["local_step"], state["global_step"], state["head_step"], state["head_calls"])
            == ("COMPLETE", 1500, 13500, 1500, 11334), "Training endpoint/cursor differs")
    require(state["probe_calls"] == 0, "Unexpected training probe calls")
    require(state["registration_sha256"] == a.freeze(OUT / "training_registration.json"), "Training registration differs")
    ledger = a.jsonl(folder / "calls.jsonl")
    beginnings, endings = {}, {}
    for row in ledger:
        require(row["arm"] == ARM and row["event"] in ("CALL_BEGIN", "CALL_DONE"), "Unexpected training ledger event")
        require(row["kind"] in EXPECTED_TRAIN, "Unregistered training call kind")
        dest = beginnings if row["event"] == "CALL_BEGIN" else endings
        require(row["call_id"] not in dest, "Duplicate training call ID")
        dest[row["call_id"]] = row
    require(set(beginnings) == set(endings) == set(range(1, sum(EXPECTED_TRAIN.values()) + 1)),
            "Unfinished, missing or excess training calls")
    counts = Counter(row["kind"] for row in endings.values())
    require(dict(counts) == EXPECTED_TRAIN, "Actual training call counts differ")
    for call_id, begin in beginnings.items():
        end = endings[call_id]
        require((begin["kind"], begin["context"], begin["attempt_id"]) ==
                (end["kind"], end["context"], end["attempt_id"]), "Training call identity mismatch")
        require(end["time_unix"] >= begin["time_unix"], "Reversed training call timestamps")
    for kind in ("optimizer_update", "head_optimizer_update"):
        require([r["context"]["local_step"] for r in endings.values() if r["kind"] == kind] == list(range(1, 1501)),
                "Optimizer update sequence has a replay/gap")
    schedule = a.json(trainreg["input_schedule"], trainreg["input_schedule_sha256"])["steps"]
    expected = {(entry["step"], mi): sample for entry in schedule for mi, sample in enumerate(entry["samples"])}
    exposure = a.jsonl(folder / "exposure.jsonl")
    actual = {(r["local_step"], r["microbatch"]): r for r in exposure}
    require(len(actual) == len(exposure) == 12000 and set(actual) == set(expected), "Exposure identities incomplete/duplicated")
    for key, row in actual.items():
        planned = expected[key]
        require(all(row[k] == planned[k] for k in ("condition_id", "parent_id", "noise_seed", "epsilon_seed", "time_seed", "permutation_seed", "scale_xyz")),
                "Actual training stream differs from frozen schedule")
        require(row["global_step"] == 12000 + key[0], "Exposure global cursor differs")
    wanted_head = {key for key, row in actual.items() if row["head_forward_called"]}
    require(len(wanted_head) == 11334, "Exposure head count differs")
    for kind in ("train_forward", "train_backward", "head_forward", "head_backward"):
        rows = [r for r in endings.values() if r["kind"] == kind]
        keys = [(r["context"]["local_step"], r["context"]["microbatch"]) for r in rows]
        wanted = wanted_head if kind.startswith("head") else set(expected)
        require(len(keys) == len(set(keys)) and set(keys) == wanted, "Generator/head exposure coverage differs")
        require(all(r["context"]["condition_id"] == expected[k]["condition_id"] for r, k in zip(rows, keys)),
                "Call condition differs from exposure")
    losses = a.jsonl(folder / "training_log.jsonl")
    require([r["local_step"] for r in losses] == list(range(1, 1501)), "Learning curve is incomplete")
    require(sum(r["head_condition_count"] for r in losses) == 11334, "Loss-log head budget differs")
    for key, filename in (("checkpoint", "checkpoint_step1500.pt"), ("inference_export", "generator_step1500.pt")):
        receipt = state[key]
        require(resolve(receipt["path"]).as_posix() == (folder / filename).as_posix(), "Wrong retained final weight")
        a.freeze(receipt["path"], receipt["sha256"])
        sidecar = a.json((folder / filename).with_suffix(".json"))
        require(sidecar == receipt and sidecar["local_step"] == 1500 and sidecar["head_step"] == 1500,
                "Final weight sidecar differs from completed state")
    require(state["checkpoint"]["generator_state_sha256"] == state["inference_export"]["generator_state_sha256"] and
            state["checkpoint"]["head_state_sha256"] == state["inference_export"]["head_state_sha256"],
            "Checkpoint/export tensor identity records disagree")
    return dict(actual_successful_calls=dict(counts), optimizer_updates=1500,
        exposures=12000, learning_curve_rows=1500, head_calls=11334, probe_calls=0,
        checkpoint=state["checkpoint"], inference_export=state["inference_export"],
        weight_validation="File SHA and producer tensor-identity receipts; no torch deserialize/forward in this audit")


def check_engineering_failure(a):
    recovery = a.json(OUT / "engineering/pre_forward_import_recovery.json")
    require((recovery["attempted_context_entries"], recovery["actual_head_forward_calls"],
             recovery["generator_forwards"], recovery["D0_forwards"], recovery["extra_generator_loads"])
            == (1, 0, 0, 0, 1), "Recorded pre-forward failure accounting differs")
    a.freeze(OUT / "workflow/raw-baseline/output.log", recovery["failure_log_sha256"])
    saved = resolve(recovery["preserved_failed_layout"])
    for former, expected in recovery["original_files"].items():
        a.freeze(saved / Path(former).name, expected)
    rows = a.jsonl(saved / "calls.jsonl")
    require(len(rows) == 1 and rows[0]["event"] == "HEAD_FORWARD_STARTED", "Preserved failed-head ledger differs")
    failures = list((OUT / "evaluation" / REFERENCE_LABEL).glob("sampling_attempt_*/failure.json"))
    require(len(failures) == 1, "Unexpected sampling failures require explicit accounting")
    failure = a.json(failures[0])
    require(failure["completed_rollouts"] == failure["returned_head_calls"] == 0 and
            'torch.autocast("cuda", dtype=torch.bfloat16)' in failure["traceback"] and
            "Invalid device id" in failure["traceback"], "Failure was not the documented pre-forward context error")
    a.freeze(ROOT / "r2/pilots/bench_newdata_joint_v1/sample.py", recovery["v1_sha256"])
    a.freeze(ROOT / "r2/pilots/bench_newdata_joint_v1/sample_v2.py", recovery["v2_sha256"])
    return dict(actual_neural_calls=0, context_entry_start_markers=1, extra_generator_loads=1,
        evidence="Preserved traceback fails constructing autocast before executing head; the START marker alone is not counted as an executed forward",
        preserved_failure=str(failures[0]), recovery_receipt=str(OUT / "engineering/pre_forward_import_recovery.json"))


def check_sampling(a, reg, qs):
    totals = Counter()
    last_raw_time = 0.
    first_d0_time = None
    per_label = {}
    expected_head = {REFERENCE_LABEL: 8, LABEL: 11}
    for label in LABELS:
        root = OUT / "evaluation" / label
        for kind in ("raw", "post"):
            expected_receipts = {(root / q["request_id"] / kind / "receipt.json").as_posix() for q in qs}
            actual_receipts = {p.as_posix() for p in root.glob("*/" + kind + "/receipt.json")}
            require(actual_receipts == expected_receipts, "Unregistered or missing RAW/POST producer receipt")
        raw_done = a.json(root / "raw_complete_full.json")
        post_done = a.json(root / "post_complete_full.json")
        require(raw_done["status"] == post_done["status"] == "COMPLETE" and
                raw_done["outputs"] == post_done["outputs"] == 28, "Incomplete RAW/POST products")
        a.verify(raw_done["files"])
        a.verify(post_done["files"])
        registration = a.json(OUT / "evaluation/inference_registrations" / (label + "__v2.json"))
        a.verify(registration["protected"])
        a.verify(registration["code"])
        require(registration["maximum_head_forwards"] == expected_head[label], "Frozen inference head budget differs")
        d0reg = a.json(OUT / "evaluation/inference_registrations" / (label + "__D0.json"))
        a.verify(d0reg["code"])
        a.verify(d0reg["RAW_barrier"])
        require(len(d0reg["RAW_barrier"]) == 56, "D0 registration lacks all paired RAW receipts")
        count = Counter()
        for q in qs:
            reused = label == REFERENCE_LABEL and q["baseline_reuse"]
            rawdir, postdir = root / q["request_id"] / "raw", root / q["request_id"] / "post"
            raw, post = a.json(rawdir / "receipt.json"), a.json(postdir / "receipt.json")
            for receipt in (raw, post):
                require(receipt["status"] == "COMPLETE" and receipt["label"] == label and
                        receipt["request_id"] == q["request_id"] and receipt["condition_id"] == q["condition_id"],
                        "Producer/request identity mismatch")
                require(receipt["source_labels_read"] is False, "Inference accessed source-owner labels")
                a.verify(receipt["files"])
            a.freeze(rawdir / "raw.npz", raw["raw_sha256"])
            a.freeze(postdir / "post.npz", post["post_sha256"])
            require(raw["noise_sha256"] == q["noise_sha256"] and post["raw_sha256"] == raw["raw_sha256"],
                    "Noise or RAW-to-POST pairing differs")
            require(raw["generator_forwards"] == 50 and raw["D0_forwards"] == 0 and
                    post["D0_forwards"] == post["GEO_calls"] == 1,
                    "Historical or new producer definition differs")
            if reused:
                require(raw["execution"] == post["execution"] == "REUSED" and
                        raw["new_generator_forwards"] == post["new_D0_forwards"] == post["new_GEO_calls"] == 0,
                        "Reference reuse incorrectly counted as new execution")
                for receipt in (raw, post):
                    a.freeze(receipt["origin_receipt"], receipt["origin_receipt_sha256"])
                count["reused_RAW"] += 1
                count["reused_POST"] += 1
            else:
                require(raw["execution"] == "NEW", "New rollout mislabeled")
                calls = a.jsonl(rawdir / "calls.jsonl")
                begins = [r for r in calls if r["event"] == "FORWARD_STARTED"]
                ends = [r for r in calls if r["event"] == "FORWARD_COMPLETE"]
                require(len(calls) == 100 and [r["step"] for r in begins] == list(range(50)) and
                        [r["step"] for r in ends] == list(range(50)), "Euler50 call ledger differs")
                require(all(b["unix"] <= e["unix"] for b, e in zip(begins, ends)), "Reversed rollout call times")
                last_raw_time = max(last_raw_time, max(r["unix"] for r in ends))
                dcalls = a.jsonl(postdir / "calls.jsonl")
                require([r["event"] for r in dcalls] == ["D0_FORWARD_STARTED", "D0_FORWARD_RETURNED"],
                        "D0 call count differs")
                # The inherited D0 ledger uses an ISO UTC clock, unlike RAW.
                from datetime import datetime
                dtime = datetime.fromisoformat(dcalls[0]["utc"]).timestamp()
                first_d0_time = dtime if first_d0_time is None else min(first_d0_time, dtime)
                proposal = a.json(postdir / "proposal_receipt.json")
                require(proposal["D0_forwards"] == 1 and proposal["raw_sha256"] == raw["raw_sha256"],
                        "D0 proposal producer differs")
                a.freeze(postdir / "proposal.npz", proposal["proposal_sha256"])
                count["new_RAW"] += 1
                count["new_D0"] += 1
                count["new_generator_forwards"] += 50
        unique = {q["condition_id"]: q for q in qs if not (label == REFERENCE_LABEL and q["baseline_reuse"])}
        require({p.parent.name for p in (root / "layouts").glob("*/receipt.json")} == set(unique),
                "Unregistered or missing predicted-C layout receipt")
        for cid, q in unique.items():
            lp = root / "layouts" / cid
            receipt = a.json(lp / "receipt.json")
            a.verify(receipt["files"])
            require(receipt["label"] == label and receipt["condition_id"] == cid and
                    receipt["input_sha256"] == q["inference_npz_sha256"] and
                    receipt["source_labels_read"] is False and receipt["threshold_logit"] == 0.,
                    "C-only layout producer differs")
            expected = int(not q["closed_C"])
            require(receipt["occupancy_forward_calls"] == expected, "Closed/open head budget differs")
            if expected:
                events = a.jsonl(lp / "calls.jsonl")
                require([r["event"] for r in events] == ["HEAD_FORWARD_STARTED", "HEAD_FORWARD_RETURNED"],
                        "Head forward ledger differs")
            else:
                require(not (lp / "calls.jsonl").exists(), "Closed C has unexpected head-call ledger")
            count["new_head_forwards"] += expected
        want = 20 if label == REFERENCE_LABEL else 28
        require(count["new_RAW"] == count["new_D0"] == want and count["new_head_forwards"] == expected_head[label],
                "Per-label inference budget differs")
        require(raw_done["total_new_rollouts"] == want and raw_done["total_new_head_forwards"] == expected_head[label],
                "RAW completion accounting differs")
        totals.update(count)
        per_label[label] = dict(count)
    require(totals["new_RAW"] == totals["new_D0"] == 48 and totals["reused_RAW"] == totals["reused_POST"] == 8 and
            totals["new_generator_forwards"] == 2400 and totals["new_head_forwards"] == 19,
            "Whole inference budget differs")
    require(first_d0_time is not None and first_d0_time >= last_raw_time, "A D0 started before all new RAWs completed")
    return dict(counts=dict(totals), per_label=per_label, last_new_RAW_forward_unix=last_raw_time,
                first_new_D0_forward_unix=first_d0_time, all_RAW_before_D0=True)


def check_metrics_gallery(a, qs):
    completion = a.json(OUT / "results/completion.json")
    require((completion["rows"], completion["new_score_rows"], completion["reused_score_rows"], completion["model_calls"])
            == (168, 144, 24, 0), "Metric completion counts differ")
    a.verify(completion["files"])
    a.verify(completion["source_rows"])
    rows = a.csv(OUT / "results/per_output.csv")
    wanted = {(label, q["request_id"], stage) for label in LABELS for q in qs for stage in STAGES}
    require(len(rows) == 168 and {(r["label"], r["request_id"], r["stage"]) for r in rows} == wanted,
            "Missing/duplicated paired metric cells")
    expected_group = {q["request_id"]: q["eval_group"] for q in qs}
    require(all(r["eval_group"] == expected_group[r["request_id"]] for r in rows), "Evaluation group changed in metrics")
    require(Counter(r["result_origin"] for r in rows) == Counter(NEW_REGISTERED=144, REUSED_HISTORICAL=24),
            "Metric reuse provenance differs")
    pairs = a.csv(OUT / "results/paired_per_output.csv")
    require(len(pairs) == 84 and {(r["request_id"], r["stage"]) for r in pairs} ==
            {(q["request_id"], stage) for q in qs for stage in STAGES}, "Incomplete metric pair table")
    for label in LABELS:
        for q in qs:
            root = REFERENCE_ROOT if label == REFERENCE_LABEL and q["baseline_reuse"] else OUT
            for stage in STAGES:
                folder = root / "evaluation/scores" / label / q["request_id"] / "RAW" if stage == "RAW" else root / "cpu" / label / q["request_id"] / stage / "scores"
                receipt = a.json(folder / "complete.json")
                require(receipt["status"] == "PASS" and receipt["model_calls"] == 0, "Missing completed original score")
                a.verify(receipt["files"])
    gallery = a.json(OUT / "gallery/completion.json")
    definition = a.json(OUT / "gallery/definition.json")
    a.verify(definition["sources"])
    a.verify(definition["protected"])
    plan = a.json(OUT / "gallery/manifest.json", definition["manifest_sha256"])
    require(gallery["sheets"] == plan["sheets"] == len(gallery["images"]) == 84, "Expected 84 paired gallery pages")
    wanted_pages = {stage + "__" + q["request_id"] for stage in STAGES for q in qs}
    require({r["page_id"] for r in gallery["images"]} == wanted_pages, "Gallery page identity coverage differs")
    for row in gallery["images"]:
        a.freeze(row["path"], row["image_sha256"])
        receipt = a.json(resolve(row["path"]).with_suffix(".json"))
        require(receipt == row and row["definition_sha256"] == a.freeze(OUT / "gallery/definition.json"), "Gallery receipt differs")
        require(len(row["cells"]) == 2 and {c["model"] for c in row["cells"]} == set(LABELS), "Gallery omits a paired model")
        require(row["model_calls"] == row["scoring_calls"] == 0, "Gallery caused unexpected model/scoring calls")
    return dict(metric_rows=168, new_metric_rows=144, reused_metric_rows=24,
        metric_fields=len(completion["metric_fields"]), paired_request_stage_rows=84,
        paired_gallery_pages=84, rendered_model_stage_cells=168,
        actual_visual_review="NOT_CERTIFIED_BY_RENDER_AUDIT; inspect the separate actual AI_PROXY review records")


def run():
    if DONE.exists():
        prior = read(DONE)
        for path, expected in prior["verified_files"].items():
            require(sha(path) == expected, "Previously audited artifact changed: " + path)
        return dict(status="VERIFIED_COMPLETE", model_calls=0, originally_audited_status=prior["status"])
    available = ready()
    if available["status"] != "READY":
        return available
    a = Reader()
    started = time.perf_counter()
    a.freeze(__file__)
    reg = a.json(OUT / "registration.json")
    trainreg = a.json(OUT / "training_registration.json")
    a.verify(trainreg["training_code_hashes"])
    for key in ("manifest", "input_schedule", "source_checkpoint", "eval_manifest"):
        a.freeze(trainreg[key], trainreg[key + "_sha256"])
    panel = a.json(reg["eval_manifest"], reg["eval_manifest_sha256"])
    a.verify(panel["protected"])
    qs = panel["requests"]
    require(len(qs) == 28 and len({q["request_id"] for q in qs}) == 28 and len({q["condition_id"] for q in qs}) == 14,
            "Frozen evaluation size differs")
    require(Counter(q["eval_group"] for q in qs) == Counter(CURRENT_TRAIN=18, SAME_PARENT_HELDOUT=2, DEV=8),
            "Frozen group coverage differs")
    require(sum(q["baseline_reuse"] for q in qs) == 8, "Frozen historical reuse count differs")
    training = check_training(a, reg, trainreg)
    engineering = check_engineering_failure(a)
    sampling = check_sampling(a, reg, qs)
    artifacts = check_metrics_gallery(a, qs)
    require("torch" not in sys.modules, "CPU final audit imported neural framework")
    result = dict(schema="BENCH_NEWDATA_JOINT_FINAL_ACCOUNTING_AUDIT_V1", status="PASS",
        training=training, sampling=sampling, engineering_pre_forward_failure=engineering,
        artifacts=artifacts, model_calls=0, new_queries=0, new_noise=0, rescoring_calls=0,
        automatic_repairs=0, files_modified_except_audit=0, seconds=time.perf_counter()-started,
        scientific_interpretation="Integrity/accounting PASS only; no claim that new data or the adapted head improves generation",
        verified_files=a.files)
    write_new(DONE, result)
    return {k: v for k, v in result.items() if k != "verified_files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"), nargs="?", default="check")
    args = parser.parse_args()
    print(json.dumps(ready() if args.action == "check" else run(), ensure_ascii=False), flush=True)
