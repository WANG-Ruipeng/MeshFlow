"""V2 final file/call audit using corrected tables; no model imports.

Independent successor to retained closeout_audit.py. Reads results_v2 and
requires a separate value-by-value V2 audit PASS before large hashes/archive.

Default is a lightweight completeness inventory. --final performs SHA256 reads
only after all required producers and actual-review collection are complete.
No model is loaded, geometry rescored, rating fabricated, old file changed,
weight copied, checkpoint removed, or final scientific report written.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_adaptation_strength_v1"
CODE = Path(__file__).resolve().parent
ARMS = ("A_COMMON_LR1E5", "B_COMMON_LR2E6", "C_OFFICIAL_LR2E6")
LABELS = ("COMMON_S0", "OFFICIAL_S0", "A_S500", "B_S500", "C_S500",
          "A_S1500", "B_S1500", "C_S1500")
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")


def resolve(value):
    value = str(value).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def csvwrite(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


class Audit:
    def __init__(self):
        self.missing, self.failures, self.checks = [], [], []
        self.hash_cache = {}
        self.verified_hashes = {}
        self.archive_sources = set()
        self.registration_paths = set()

    def need(self, path):
        path = resolve(path)
        if not path.is_file():
            self.missing.append(str(path))
            return None
        return path

    def document(self, path):
        path = self.need(path)
        if path is None:
            return None
        try:
            return read(path)
        except (OSError, ValueError) as exc:
            self.failures.append(dict(check="JSON_READ", path=str(path), error=repr(exc)))
            return None

    def check(self, condition, name, **facts):
        self.checks.append(dict(check=name, passed=bool(condition), **facts))
        if not condition:
            self.failures.append(dict(check=name, **facts))
        return bool(condition)

    def sha(self, path):
        path = resolve(path).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        if identity not in self.hash_cache:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    h.update(block)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("File changed during final hash: " + str(path))
            self.hash_cache[identity] = h.hexdigest()
        return self.hash_cache[identity]

    def hash_equals(self, path, expected, *, context):
        path = self.need(path)
        if path is None:
            return False
        actual = self.sha(path)
        valid = actual == expected
        if not valid:
            self.failures.append(dict(check="SHA256", context=context, path=str(path),
                                      expected=expected, actual=actual))
        self.verified_hashes[str(path.resolve())] = dict(sha256=actual, bytes=path.stat().st_size,
                                                      matches_registered=valid)
        return valid

    def files(self, mapping, *, context, code=False):
        for path, expected in mapping.items():
            self.hash_equals(path, expected, context=context)
            if code and resolve(path).suffix.lower() in (".py", ".md", ".toml", ".yaml", ".yml"):
                self.archive_sources.add(resolve(path).resolve())


def expected_outputs(scope, manifest):
    rows = manifest["requests"]
    byid = {q["request_id"]: q for q in rows}
    process = scope["process_request_ids"]
    if len(rows) != 48 or len(byid) != 48 or len(process) != 16 or len(set(process)) != 16:
        raise ValueError("Registered 48/16 request identities differ")
    outputs = []
    for label in LABELS:
        spec = scope["evaluation_labels"][label]
        selected = rows if spec["subset"] == "full" else [byid[rid] for rid in process]
        if spec["requests"] != len(selected):
            raise ValueError("Label panel count differs")
        outputs.extend((label, q) for q in selected)
    if len(outputs) != 224:
        raise ValueError("Registered224 outputs required")
    return outputs


def readiness(audit, scope, outputs):
    training = {}
    ready = True
    for arm in ARMS:
        s = audit.document(OUT / "training" / arm / "status.json")
        training[arm] = None if s is None else {k: s.get(k) for k in ("status", "local_step", "global_step")}
        if s is None or s.get("status") != "COMPLETE" or s.get("local_step") != 1500:
            ready = False
    summary = audit.document(OUT / "training_summary/latest.json")
    if summary is None or summary.get("status") != "PASS" or summary.get("training_complete") is not True:
        ready = False
    counts = dict(raw_receipts=0, post_receipts=0, scored_stages=0)
    for label, q in outputs:
        base = OUT / "evaluation" / label / q["request_id"]
        for kind in ("raw", "post"):
            p = audit.need(base / kind / "receipt.json")
            counts[kind + "_receipts"] += int(p is not None)
            if p is None:
                ready = False
        for stage in STAGES:
            folder = (OUT / "evaluation/scores" / label / q["request_id"] / "RAW" if stage == "RAW"
                      else OUT / "cpu" / label / q["request_id"] / stage / "scores")
            p = audit.need(folder / "complete.json")
            counts["scored_stages"] += int(p is not None)
            if p is None:
                ready = False
    for label in LABELS:
        subset = scope["evaluation_labels"][label]["subset"]
        for kind in ("raw", "post"):
            if audit.need(OUT / "evaluation" / label / f"{kind}_complete_{subset}.json") is None:
                ready = False
    results = audit.document(OUT / "results_v2/completion.json")
    if results is None or results.get("status") != "COMPLETE" or results.get("rows") != 672:
        ready = False
    independent = audit.document(OUT / "diagnostics/results_v2_independent_audit.json")
    if independent is None or independent.get("status") != "PASS":
        ready = False
        if independent is not None and independent.get("status") == "FAIL":
            audit.check(False, "INDEPENDENT_V2_VALUE_AUDIT_FAILED")
    if independent is not None and independent.get("status") == "PASS":
        authenticated = {str(resolve(p).resolve()) for p in independent.get("files", {})}
        required = {str((OUT / "results_v2" / n).resolve()) for n in ("completion.json", "per_output.csv")}
        if not audit.check(required <= authenticated, "INDEPENDENT_V2_AUDIT_BINDINGS_PRESENT"):
            ready = False
    gallery = audit.document(OUT / "gallery/completion.json")
    if gallery is None or gallery.get("status") != "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW" or gallery.get("sheets") != 144:
        ready = False
    review = audit.document(OUT / "visual_review/completion.json")
    if review is None or review.get("status") != "ACTUAL_AI_PROXY_REVIEW_COMPLETE" or review.get("reviewed_cells") != 672:
        ready = False
    return ready, dict(training=training, counts=counts,
                       training_summary_status=None if summary is None else summary.get("status"),
                       scores_status=None if results is None else results.get("status"),
                       score_table_root=str(OUT / "results_v2"),
                       independent_value_audit_status=None if independent is None else independent.get("status"),
                       gallery_sheets=None if gallery is None else gallery.get("sheets"),
                       actual_review_cells=None if review is None else review.get("reviewed_cells"))


def check_registration(audit, path, mappings):
    value = audit.document(path)
    if value is None:
        return None
    audit.registration_paths.add(resolve(path).resolve())
    for key, code in mappings:
        if key in value:
            audit.files(value[key], context=str(path) + ":" + key, code=code)
    return value


def training_audit(audit, reg):
    from .summary_training import audit_calls
    latest = read(OUT / "training_summary/latest.json")
    summary = resolve(latest["path"])
    value = read(summary / "accounting.json")
    audit.check(value["status"] == "PASS" and value["training_complete"], "FINAL_TRAINING_SUMMARY_PASS")
    for source in read(summary / "source_snapshots.json"):
        audit.hash_equals(source["path"], source["sha256_of_read_snapshot"], context="final training summary source")
    totals = collections.Counter()
    for arm in ARMS:
        folder = OUT / "training" / arm
        status = read(folder / "status.json")
        expected_global = 1500 if arm == ARMS[2] else 13500
        audit.check((status["status"], status["local_step"], status["global_step"]) == ("COMPLETE", 1500, expected_global),
                    "TRAINING_FINAL_IDENTITY", arm=arm)
        ledger = audit_calls(jsonl(folder / "calls.jsonl"), arm)
        valid = not ledger["issues"] and not ledger["failed"] and not ledger["pending_call_ids"]
        valid = valid and ledger["attempted"] == ledger["completed"] == ledger["authorized_limits"]
        audit.check(valid, "TRAINING_EXACT_CALL_LEDGER", arm=arm, counts=ledger)
        totals.update(ledger["completed"])
        rows = jsonl(folder / "training_log.jsonl")
        audit.check([r["local_step"] for r in rows] == list(range(1, 1501)), "TRAINING_LOG_1500_CONSECUTIVE", arm=arm)
        receipt = status["checkpoint"]
        audit.check(receipt["local_step"] == 1500 and Path(receipt["path"]).name == "checkpoint_step1500.pt",
                    "FINAL_WEIGHT_RECEIPT", arm=arm)
        audit.hash_equals(receipt["path"], receipt["sha256"], context="final full checkpoint " + arm)
        audit.check(read(resolve(receipt["path"]).with_suffix(".json")) == receipt,
                    "FINAL_WEIGHT_SIDECAR_MATCHES_STATUS", arm=arm)
        for step in (0, 500, 1500):
            path = OUT / "inference" / f"{arm}_S{step}_generator.pt"
            rec = audit.document(path.with_suffix(".json"))
            if rec is not None:
                audit.check((rec["arm"], rec["local_step"], rec["global_step"]) ==
                            (arm, step, (0 if arm == ARMS[2] else 12000) + step),
                            "GENERATOR_EXPORT_IDENTITY", arm=arm, step=step)
                audit.hash_equals(path, rec["sha256"], context="generator export")
    want = dict(train_forward=36000, train_backward=36000, optimizer_update=4500, probe_forward=1056)
    audit.check(dict(totals) == want, "TOTAL_TRAINING_CALL_BUDGET", actual=dict(totals), expected=want)
    return dict(totals)


def output_audit(audit, scope, outputs):
    raw_calls = d0_calls = 0
    expected_raw_paths, expected_post_paths = set(), set()
    raw_receipt_hashes = {}
    for label in LABELS:
        check_registration(audit, OUT / "evaluation/inference_registrations" / (label + ".json"),
                           (("protected", False), ("code", True)))
        subset = scope["evaluation_labels"][label]["subset"]
        for kind in ("raw", "post"):
            p = OUT / "evaluation" / label / f"{kind}_complete_{subset}.json"
            r = read(p)
            audit.check(r["status"] == "COMPLETE" and r["outputs"] == scope["evaluation_labels"][label]["requests"],
                        "LABEL_PRODUCER_COMPLETE", label=label, kind=kind)
            audit.files(r["files"], context=str(p))
    for label, q in outputs:
        base = OUT / "evaluation" / label / q["request_id"]
        raw_path, post_path = base / "raw/receipt.json", base / "post/receipt.json"
        raw, post = read(raw_path), read(post_path)
        raw_receipt_hashes[str(raw_path.resolve())] = audit.sha(raw_path)
        for kind, receipt, path in (("raw", raw, raw_path), ("post", post, post_path)):
            audit.check(receipt["status"] == "COMPLETE" and receipt["label"] == label and
                        receipt["request_id"] == q["request_id"] and receipt["condition_id"] == q["condition_id"],
                        "OUTPUT_IDENTITY", label=label, request_id=q["request_id"], kind=kind)
            audit.files(receipt["files"], context=str(path))
            audit.hash_equals(path.parent / (kind + ".npz"), receipt[kind + "_sha256"], context=str(path))
        audit.check(raw["generator_forwards"] == 50 and raw["D0_forwards"] == 0 and raw["head_forwards"] == 0 and
                    raw["source_labels_read"] is False and raw["audit"]["anchored_all51_exact"] is True,
                    "RAW_CONTRACT", label=label, request_id=q["request_id"])
        audit.check(raw["noise_sha256"] == q["noise_sha256"], "FROZEN_NOISE_IDENTITY", label=label, request_id=q["request_id"])
        rledger = jsonl(base / "raw/calls.jsonl")
        started = [r["step"] for r in rledger if r["event"] == "FORWARD_STARTED"]
        done = [r["step"] for r in rledger if r["event"] == "FORWARD_COMPLETE"]
        audit.check(len(rledger) == 100 and started == done == list(range(50)), "EULER50_CALL_LEDGER",
                    label=label, request_id=q["request_id"])
        raw_calls += len(done)
        expected_raw_paths.add((base / "raw/calls.jsonl").resolve())
        dledger = jsonl(base / "post/calls.jsonl")
        events = [r["event"] for r in dledger]
        audit.check(events == ["D0_FORWARD_STARTED", "D0_FORWARD_RETURNED"] and
                    post["D0_forwards"] == 1 and post["GEO_calls"] == 1 and post["source_labels_read"] is False,
                    "SINGLE_D0_CALL_LEDGER", label=label, request_id=q["request_id"])
        d0_calls += events.count("D0_FORWARD_RETURNED")
        expected_post_paths.add((base / "post/calls.jsonl").resolve())
        proposal = read(base / "post/proposal_receipt.json")
        audit.check(proposal["D0_forwards"] == 1 and proposal["raw_sha256"] == raw["raw_sha256"] and
                    proposal["official_checkpoint_sha256"] == post["official_D0_sha256"],
                    "D0_PROPOSAL_IDENTITY", label=label, request_id=q["request_id"])
        audit.hash_equals(base / "post/proposal.npz", proposal["proposal_sha256"], context="saved one-call D0 proposal")
    actual_raw = {p.resolve() for p in (OUT / "evaluation").glob("*/*/raw/calls.jsonl")}
    actual_post = {p.resolve() for p in (OUT / "evaluation").glob("*/*/post/calls.jsonl")}
    audit.check(actual_raw == expected_raw_paths and actual_post == expected_post_paths,
                "NO_UNREGISTERED_RAW_POST_CALL_LEDGERS", raw=len(actual_raw), post=len(actual_post))
    barriers = list((OUT / "evaluation").glob("*/postprocess_attempt_*/all_RAW_barrier.json"))
    audit.check(bool(barriers), "D0_RAW_BARRIER_RECEIPTS_EXIST")
    for path in barriers:
        value = read(path)
        actual = {str(resolve(p).resolve()): h for p, h in value["files"].items()}
        audit.check(value["status"] == "PASS" and value["outputs"] == 224 and actual == raw_receipt_hashes,
                    "ALL224_RAW_BEFORE_D0_BARRIER", path=str(path))
    audit.check(raw_calls == 11200 and d0_calls == 224, "SAMPLING_TOTAL_CALLS", generator_forwards=raw_calls, D0_forwards=d0_calls)
    return dict(outputs=224, generator_forwards=raw_calls, D0_forwards=d0_calls, head_forwards=0, RAW_barriers=len(barriers))


def score_audit(audit, outputs):
    reg = check_registration(audit, OUT / "evaluation/score_registration.json", (("source_hashes", True),))
    independent_path = OUT / "diagnostics/results_v2_independent_audit.json"
    independent = read(independent_path)
    audit.check(independent.get("status") == "PASS", "INDEPENDENT_V2_VALUE_AUDIT_PASS")
    authenticated = {str(resolve(p).resolve()) for p in independent.get("files", {})}
    required = {str((OUT / "results_v2" / n).resolve()) for n in ("completion.json", "per_output.csv")}
    audit.check(required <= authenticated, "INDEPENDENT_V2_VALUE_AUDIT_BINDS_CURRENT_TABLES")
    audit.files(independent.get("files", {}), context="independent V2 value-by-value audit")
    audit.registration_paths.add(independent_path.resolve())
    initial_independent_path = independent_path.with_name("results_v2_independent_audit.initial.json")
    if initial_independent_path.is_file():
        audit.registration_paths.add(initial_independent_path.resolve())
    revision_path = OUT / "diagnostics/collector_v2_revision.json"
    revision = read(revision_path)
    audit.hash_equals(revision["replacement_collector"], revision["replacement_sha256"],
                      context="independent V2 collector revision")
    audit.hash_equals(revision["legacy_collector"], revision["legacy_collector_sha256"],
                      context="retained original collector unchanged")
    audit.archive_sources.add(resolve(revision["replacement_collector"]).resolve())
    audit.registration_paths.add(revision_path.resolve())
    complete = read(OUT / "results_v2/completion.json")
    audit.check(complete["status"] == "COMPLETE" and complete["rows"] == 672 and complete["outputs"] == 224 and
                complete["all_fields_preserved"] is True and complete["model_calls"] == 0,
                "ALL672_SCORES_COMPLETE")
    audit.files(complete["files"], context="collected full result tables")
    audit.files(complete["source_rows"], context="collected original score rows")
    with (OUT / "results_v2/per_output.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames)
        rows = list(reader)
    expected = {(label, q["request_id"], stage) for label, q in outputs for stage in STAGES}
    actual = [(r["label"], r["request_id"], r["stage"]) for r in rows]
    audit.check(len(rows) == len(set(actual)) == 672 and set(actual) == expected, "SCORE_TABLE_EXACT_IDENTITIES")
    source_rows = set()
    for label, q in outputs:
        cascade = OUT / "cpu" / label / q["request_id"] / "complete.json"
        cpu = read(cascade)
        audit.check(cpu["status"] == "COMPLETE" and cpu["model_calls"] == cpu["D0_calls"] == cpu["GPU_calls"] == 0,
                    "CPU_CASCADE_COMPLETED_WITHOUT_MODEL_CALLS", label=label, request_id=q["request_id"])
        audit.files(cpu["files"], context=str(cascade))
        for stage in STAGES:
            folder = (OUT / "evaluation/scores" / label / q["request_id"] / "RAW" if stage == "RAW"
                      else OUT / "cpu" / label / q["request_id"] / stage / "scores")
            receipt = read(folder / "complete.json")
            audit.files(receipt["files"], context=str(folder / "complete.json"))
            path = folder / "row.json"
            row = read(path)
            audit.check(set(row) <= fields, "FULL_SCORE_FIELDS_PRESERVED", label=label, request_id=q["request_id"],
                        stage=stage, row_fields=len(row), omitted=sorted(set(row) - fields))
            source_rows.add(str(path.resolve()))
            if stage == "RAW":
                raw_audit = read(folder / "audit_receipt.json")
                identity = raw_audit["identity"]
                audit.check(raw_audit["status"] == "PASS" and identity["label"] == label and
                            identity["request_id"] == q["request_id"] and identity["stage"] == stage and
                            identity["registration_sha256"] == audit.sha(OUT / "evaluation/score_registration.json"),
                            "RAW_SCORE_REGISTRATION_IDENTITY", label=label, request_id=q["request_id"])
                audit.files(raw_audit["files"], context="RAW audit")
    audit.check({str(resolve(p).resolve()) for p in complete["source_rows"]} == source_rows,
                "ALL672_SOURCE_ROWS_AUTHENTICATED")
    return dict(scored_cells=len(rows), table_fields=len(fields), source_rows=len(source_rows),
                result_tables="results_v2", independent_value_audit="PASS")


def gallery_and_review_audit(audit):
    gm = check_registration(audit, OUT / "gallery/manifest.json", (("protected", True),))
    definition = check_registration(audit, OUT / "gallery/definition.json", (("sources", False), ("protected", True)))
    gc = read(OUT / "gallery/completion.json")
    audit.check(gc["sheets"] == 144 and len(gc["images"]) == 144, "GALLERY144_SHEETS")
    pages = {p["page_id"]: p for p in gm["pages"]}
    audit.check(len(pages) == 144 and {r["page_id"] for r in gc["images"]} == set(pages),
                "GALLERY_EXACT_REGISTERED_PAGES")
    definition_sha = audit.sha(OUT / "gallery/definition.json")
    image_hashes = {}
    for item in gc["images"]:
        path = resolve(item["path"])
        audit.hash_equals(path, item["image_sha256"], context="rendered comparison page")
        receipt = read(path.with_suffix(".json"))
        audit.check(receipt["definition_sha256"] == definition_sha and
                    receipt["image_sha256"] == item["image_sha256"] and receipt["page_id"] == item["page_id"],
                    "GALLERY_IMAGE_RECEIPT", page_id=item["page_id"])
        image_hashes[item["page_id"]] = item["image_sha256"]
    protocol = check_registration(audit, OUT / "visual_review/protocol.json", (("protected", False),))
    assignment = check_registration(audit, OUT / "visual_review/assignments.json", (("protected", False),))
    expected_file = read(OUT / "visual_review/expected_cells.json")
    expected = {(r["request_id"], r["model"], r["stage"]): r for r in expected_file["rows"]}
    vc = read(OUT / "visual_review/completion.json")
    validated = read(OUT / "visual_review/validated_records.json")
    audit.check(vc["status"] == validated["status"] == "ACTUAL_AI_PROXY_REVIEW_COMPLETE" and
                vc["reviewed_cells"] == 672 and vc["missing_cells"] == 0 and len(validated["rows"]) == 672 and
                vc["reviewer_kind"] == "AI_PROXY" and vc["blind"] is False and vc["formal_human_ratings"] == 0,
                "ACTUAL_AI_PROXY_REVIEW672")
    audit.files(vc["source_review_files"], context="actual reviewer submissions")
    audit.files(vc["authenticated_image_files"], context="reviewed images/receipts")
    audit.hash_equals(vc["per_cell_csv"], vc["per_cell_sha256"], context="actual review final CSV")
    raw_records, observed = {}, set()
    for path in vc["source_review_files"]:
        document = read(path)
        audit.check(document["protocol_sha256"] == audit.sha(OUT / "visual_review/protocol.json") and
                    document["assignments_sha256"] == audit.sha(OUT / "visual_review/assignments.json") and
                    document["reviewer_kind"] == "AI_PROXY" and document["blind"] is False,
                    "REVIEW_ENVELOPE", path=path)
        for row in document["rows"]:
            key = row["request_id"], row["model"], row["stage"]
            audit.check(key in expected and key not in raw_records, "REVIEW_UNIQUE_EXPECTED_CELL", key=list(key))
            if key not in expected:
                continue
            target = expected[key]
            audit.check(document["reviewer"] == target["reviewer"], "REVIEW_ASSIGNMENT", key=list(key))
            audit.check(row["actually_viewed"] is True and row["views_and_sections_checked"] is True and
                        row["image_sha256"] == image_hashes[target["page_id"]] and
                        isinstance(row["notes"], str) and bool(row["notes"].strip()) and
                        all(row.get(axis) in allowed for axis, allowed in protocol["axes"].items()),
                        "ACTUAL_REVIEW_ATTESTATION_AXES_NOTES", key=list(key))
            raw_records[key] = row
    for row in validated["rows"]:
        key = row["request_id"], row["model"], row["stage"]
        audit.check(key in expected and key not in observed and key in raw_records,
                    "VALIDATED_REVIEW_IDENTITIES", key=list(key))
        observed.add(key)
        if key in raw_records:
            source = raw_records[key]
            audit.check(all(row[field] == source[field] for field in
                            (*protocol["axes"], "image_sha256", "actually_viewed", "views_and_sections_checked")) and
                        row["notes"] == source["notes"].strip(), "VALIDATED_REVIEW_MATCHES_ACTUAL_SUBMISSION", key=list(key))
    audit.check(set(raw_records) == observed == set(expected) and len(observed) == 672,
                "ALL_REGISTERED_REVIEW_CELLS_COVERED")
    return dict(sheets=144, actual_AI_PROXY_cells=len(observed), formal_human_ratings=0, blind=False,
                attestation_limit="File identities and submitted actual-view declarations verified; rendering alone never counted as review.")


def inventory_tree(path):
    rows = []
    for folder, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not Path(folder, d).is_symlink()]
        for name in files:
            file = Path(folder) / name
            if file.is_symlink():
                continue
            stat = file.stat()
            rows.append(dict(path=str(file), bytes=stat.st_size,
                             OS_reported_allocated_bytes=stat.st_blocks * 512 if hasattr(stat, "st_blocks") else None))
    return rows


def archive_and_inventory(audit, out):
    archive = OUT / "source_archive" / out.name
    archive.mkdir(parents=True, exist_ok=False)
    for path in CODE.iterdir():
        if path.is_file() and path.suffix.lower() in (".py", ".md", ".toml", ".yaml", ".yml"):
            audit.archive_sources.add(path.resolve())
    for path in (list((OUT / "diagnostics").glob("*.py")) +
                 list((OUT / "visual_review").glob("*.ps1")) +
                 list((OUT / "visual_review").glob("*.py")) + [OUT / "RUNBOOK.md"]):
        if path.is_file():
            audit.archive_sources.add(path.resolve())
    rows = []
    for source in sorted(audit.archive_sources | audit.registration_paths):
        try:
            relative = source.relative_to(ROOT)
        except ValueError:
            raise ValueError("Refuse source archive outside workspace: " + str(source))
        if source.suffix.lower() in (".pt", ".pth", ".ckpt", ".npz", ".npy", ".zip") or source.stat().st_size > (4 << 20):
            raise ValueError("Not a necessary small source/registration artifact: " + str(source))
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_sha, copy_sha = audit.sha(source), audit.sha(destination)
        if source_sha != copy_sha:
            raise ValueError("Copied source hash differs: " + str(source))
        rows.append(dict(source=str(source), archive_path=str(destination), bytes=source.stat().st_size,
                         sha256=source_sha, kind="registration" if source in audit.registration_paths else "source"))
    write(archive / "manifest.json", dict(status="PASS", files=rows, copied_weights=0,
                                        copied_old_results=0, archive_root=str(archive)))
    weights = []
    for path in sorted(list((OUT / "training").glob("**/*.pt")) + list((OUT / "inference").glob("*.pt"))):
        receipt_path = path.with_suffix(".json")
        receipt = audit.document(receipt_path)
        if receipt:
            audit.hash_equals(path, receipt["sha256"], context="retained local weight inventory")
        stat = path.stat()
        weights.append(dict(path=str(path), bytes=stat.st_size,
                            OS_reported_allocated_bytes=stat.st_blocks * 512 if hasattr(stat, "st_blocks") else None,
                            sha256=audit.sha(path), receipt_path=str(receipt_path),
                            receipt_present=receipt is not None, arm=receipt.get("arm") if receipt else None,
                            local_step=receipt.get("local_step") if receipt else None))
    csvwrite(out / "weights_inventory.csv", weights)
    tree = inventory_tree(OUT)
    csvwrite(out / "experiment_disk_files.csv", tree)
    usage = shutil.disk_usage(OUT)
    total_allocated = (sum(r["OS_reported_allocated_bytes"] for r in tree)
                       if all(r["OS_reported_allocated_bytes"] is not None for r in tree) else None)
    disk = dict(scope=str(OUT), files=len(tree), logical_bytes=sum(r["bytes"] for r in tree),
                OS_reported_allocated_bytes=total_allocated,
                allocation_limit="st_blocks when provided by the OS; no claim of exclusive physical blocks, compression or hardlink deduplication.",
                weight_files=len(weights), weight_logical_bytes=sum(r["bytes"] for r in weights),
                source_archive_files=len(rows), source_archive_logical_bytes=sum(r["bytes"] for r in rows),
                filesystem_total_bytes=usage.total, filesystem_used_bytes=usage.used, filesystem_free_bytes=usage.free,
                measurement_time_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                excludes="Audit JSON written after this inventory; no scan outside the one experiment root.")
    write(out / "disk_usage.json", disk)
    return dict(source_archive=str(archive), source_files=len(rows), weights=weights, disk=disk)


def run(*, final=False):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = OUT / "closeout_audits_v2" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    started = time.perf_counter()
    scope = audit.document(OUT / "registration.json")
    manifest = audit.document(OUT / "eval_manifest.json")
    if scope is None or manifest is None:
        value = dict(status="INCOMPLETE", missing=audit.missing, failures=audit.failures)
        write(folder / "closeout_audit.json", value)
        return value
    outputs = expected_outputs(scope, manifest)
    ready, inventory = readiness(audit, scope, outputs)
    value = dict(schema="BENCH_ADAPTATION_FILE_CALL_CLOSEOUT_V2", status="INCOMPLETE",
                 final_requested=final, readiness=inventory, scientific_conclusion="ROOT_REVIEW_PENDING",
                 model_loads=0, model_calls=0, scoring_calls=0, generated_ratings=0,
                 copied_weights=0, deleted_files=0, git_actions=0)
    if not ready or not final:
        value.update(reason="Required evidence incomplete" if not ready else "Use --final for final hash and source archive",
                     missing=sorted(set(audit.missing)), failures=audit.failures, large_hashes_started=False,
                     audit_path=str(folder / "closeout_audit.json"))
        # Malformed completed evidence is never silently called complete.
        if audit.failures:
            value["status"] = "FAIL"
        write(folder / "closeout_audit.json", value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return value
    try:
        reg = check_registration(audit, OUT / "training_registration.json", (("training_code_hashes", True),))
        check_registration(audit, OUT / "registration.json", (("protected_inputs", False),))
        remaining = check_registration(audit, OUT / "remaining_runtime_code_registration.json",
                                       (("code", True), ("source_hashes", True), ("protected", False)))
        audit.registration_paths.add((OUT / "eval_manifest.json").resolve())
        audit.hash_equals(OUT / "registration.json", reg["scope_sha256"], context="training frozen scope")
        for key in ("manifest", "input_schedule", "probe_manifest"):
            audit.hash_equals(reg[key], reg[key + "_sha256"], context="training input " + key)
        audit.hash_equals(reg["official"], reg["official_sha256"], context="protected official Bench generator")
        audit.hash_equals(reg["source_checkpoint"], reg["source_checkpoint_sha256"], context="protected COMMON12000")
        value["training"] = training_audit(audit, reg)
        value["sampling"] = output_audit(audit, scope, outputs)
        value["scoring"] = score_audit(audit, outputs)
        value["visual"] = gallery_and_review_audit(audit)
        if not audit.failures and not audit.missing:
            value["archive"] = archive_and_inventory(audit, folder)
        value["status"] = "FAIL" if audit.failures else ("INCOMPLETE" if audit.missing else "PASS")
    except Exception as exc:
        import traceback
        audit.failures.append(dict(check="AUDIT_EXECUTION", error=repr(exc), traceback=traceback.format_exc()))
        value["status"] = "FAIL"
    value.update(failures=audit.failures, missing=sorted(set(audit.missing)), checks=audit.checks,
                 verified_file_count=len(audit.verified_hashes), large_hashes_started=True,
                 elapsed_seconds=time.perf_counter() - started, audit_path=str(folder / "closeout_audit.json"),
                 no_final_REPORT_or_completion_written=True)
    write(folder / "verified_hashes.json", audit.verified_hashes)
    write(folder / "closeout_audit.json", value)
    print(json.dumps({k: v for k, v in value.items() if k not in ("checks", "archive", "failures", "missing")},
                     ensure_ascii=False, indent=2))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = run(final=args.final)
    sys.exit(1 if result["status"] == "FAIL" else 0)
