"""Complete <10 MB review package, only after final actual review.

CPU-only packaging. Never train, sample, score, or assign visual judgments.
All selected latest CSVs and actual source files are mandatory. Only a shared
preview size/quality may change to meet the strict byte limit.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import time
import zipfile
from . import common as C

VERSION = "full_contour_readout_small_review_archive_v1_1"
MAX_BYTES = 10_000_000
PREVIEW_TARGET_BYTES = 6_000_000
MAIN = ("CONTOUR_S0", "CONTOUR_S1000", "CONTOUR_S3000", "CONTOUR_S6000")
BASELINE = ("FULL_S0000", "FULL_S1000", "FULL_S3000", "FULL_S6000")
ALL_LABELS = MAIN + BASELINE
STAGES = ("RAW", "POST")
PREVIEW_SETTINGS = ((1800, 78), (1600, 72), (1400, 66), (1200, 60),
                    (1000, 54), (800, 48), (640, 42))
REVIEW_FIELDS = ("C_adoption", "C_floating", "C_replaced_by_other_surface",
                 "C_wrong_layer", "whole_body_complete", "support_missing",
                 "fragments_or_folded_layers", "visible_join")
ARCHIVE = C.OUT / "archive"
OLD = C.ROOT / "experiments/r2_full_seat_learning_curve_v1"


def encoded(value):
    return (json.dumps(C.clean(value), indent=2, sort_keys=True,
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def bytes_sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with C.resolve(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def checked_file(path, expected=None):
    p = C.resolve(path)
    if not p.is_file():
        raise RuntimeError("Required archive input missing: " + str(p))
    actual = C.sha(p)
    if expected is not None and actual != expected:
        raise ValueError("Archive source hash differs: " + str(p))
    return p, actual


def latest_snapshot(pointer):
    pointer, _ = checked_file(pointer)
    record = C.read(pointer)
    path, _ = checked_file(record["path"], record["sha256"])
    data = C.read(path)
    if record.get("status") != "COMPLETE" or data.get("status") != "COMPLETE":
        raise RuntimeError("Latest snapshot must be COMPLETE: " + str(pointer))
    for p, digest in data.get("output_hashes", {}).items():
        checked_file(p, digest)
    return path, data


def reviewed_rows(path, labels, requests):
    p, digest = checked_file(path)
    rows = read_csv(p)
    expected = {(label, rid, stage) for label in labels
                for rid in requests for stage in STAGES}
    keys = [(r.get("label"), r.get("request_id"), r.get("stage")) for r in rows]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise RuntimeError("Actual review identity coverage differs: " + str(p))
    for row in rows:
        q = requests[row["request_id"]]
        if row.get("review_status") not in ("REVIEWED", "COMPLETE"):
            raise ValueError("Unreviewed row may not enter complete archive")
        if row.get("reviewer_kind") != "AI_PROXY" or not row.get("reviewer", "").strip():
            raise ValueError("Actual reviewer identity missing")
        if not row.get("notes", "").strip():
            raise ValueError("Actual observation notes missing")
        if any(row.get(field) not in ("YES", "NO", "UNCERTAIN") for field in REVIEW_FIELDS):
            raise ValueError("Incomplete observation label")
        if row.get("parent_id") != q["parent_id"] or int(row["candidate"]) != int(q["candidate"]):
            raise ValueError("Review parent/noise identity differs")
        for optional in ("N", "K"):
            if row.get(optional) and int(row[optional]) != int(q[optional]):
                raise ValueError("Review optional numerical field differs: " + optional)
    return p, digest, rows


def final_gate(final_audit):
    """Read-only gate: no archive writes until every requirement succeeds."""
    reg_path, reg_sha = checked_file(C.OUT / "registration.json")
    reg = C.read(reg_path)
    if reg.get("status") != "FROZEN":
        raise RuntimeError("Registration is not frozen")
    requests = {q["request_id"]: q for q in reg["requests"]}
    if len(requests) != 24 or len({q["parent_id"] for q in requests.values()}) != 6:
        raise ValueError("Not the fixed 24-request/six-parent panel")
    status_path, _ = checked_file(C.OUT / "training/CONTOUR/status.json")
    status = C.read(status_path)
    if status.get("status") != "COMPLETE" or status.get("local_step") != 6000 or status.get("global_step") != 12000:
        raise RuntimeError("Candidate fixed-budget training incomplete")
    report_path, report_sha = checked_file(C.OUT / "REPORT.md")
    if not report_path.read_text(encoding="utf-8-sig").strip():
        raise RuntimeError("Final REPORT is empty")
    audit_path, audit_sha = checked_file(final_audit)
    audit = C.read(audit_path)
    if not str(audit.get("status", "")).startswith("PASS"):
        raise RuntimeError("Explicit final audit has not passed")
    for name in ("blocking", "blocking_findings", "blocking_errors"):
        if audit.get(name):
            raise RuntimeError("Final audit has unresolved " + name)
    for name, expected in (("registration_sha256", reg_sha), ("report_sha256", report_sha)):
        if audit.get(name) is not None and audit[name] != expected:
            raise ValueError("Final audit is bound to another " + name)
    comparison_path, comparison = latest_snapshot(C.OUT / "evaluation/comparison/latest.json")
    if not comparison.get("numerical_complete") or not comparison.get("joint_and_review_complete"):
        raise RuntimeError("Paired numerical and actual-review comparison incomplete")
    training_path, training = latest_snapshot(C.OUT / "training_analysis/latest.json")
    if training.get("logged_updates") != 6000 or training.get("usable_complete_diagnostic_rows") != 48000:
        raise RuntimeError("Full learning curve missing")
    new_review = reviewed_rows(C.OUT / "evaluation/review/AI_PROXY_reviews.csv", MAIN[1:], requests)
    old_review = reviewed_rows(OLD / "evaluation/review/AI_PROXY_reviews.csv", BASELINE, requests)
    images = []
    for kind, directory in (("depth", "gallery"), ("sections", "sections_gallery")):
        path, _ = checked_file(C.OUT / "evaluation" / directory / "manifest.json")
        manifest = C.read(path)
        if manifest.get("status") not in ("COMPLETE", "COMPLETE_RENDERING_NOT_REVIEW"):
            raise RuntimeError("Final gallery rendering incomplete")
        if tuple(manifest.get("labels", ())) != ALL_LABELS:
            raise ValueError("Preview must contain every registered model column")
        pages = manifest["images"]
        if len(pages) != 24 or {p["request_id"] for p in pages} != set(requests):
            raise ValueError("Preview must cover every request, including failures")
        for page in pages:
            checked_file(page["path"], page["sha256"])
            images.append(dict(page, kind=kind, gallery_manifest_path=str(path)))
    for label in ALL_LABELS:
        for stage in STAGES:
            path = C.OUT / "evaluation/indices" / label / (stage.lower() + "_index.json")
            index = C.read(checked_file(path)[0])
            if index.get("status") != "COMPLETE" or len(index.get("rows", [])) != 24:
                raise RuntimeError("Incomplete output index: " + label + " " + stage)
        for directory in ("score_completion", "boundary_completion"):
            record = C.read(checked_file(C.OUT / "evaluation" / directory / (label + ".json"))[0])
            if record.get("status") != "COMPLETE" or len(record.get("receipts", [])) != 48:
                raise RuntimeError("Incomplete scoring receipt: " + directory + "/" + label)
    return dict(reg=reg, reg_sha=reg_sha, requests=requests, report_path=report_path,
                report_sha=report_sha, audit_path=audit_path, audit_sha=audit_sha,
                audit_report_binding_present=bool(audit.get("report_sha256")),
                comparison_path=comparison_path, comparison=comparison,
                training_path=training_path, training=training, status_path=status_path,
                new_review=new_review, old_review=old_review, images=images)


class Entries:
    def __init__(self):
        self.data, self.sources, self.source_keys = {}, [], set()

    def add(self, name, data, *, path=None, expected=None, role="evidence"):
        name = str(PurePosixPath(name))
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Unsafe ZIP member")
        data = bytes(data)
        digest = bytes_sha(data)
        if expected is not None and digest != expected:
            raise ValueError("Archived content hash differs: " + str(path))
        if name in self.data and self.data[name] != data:
            raise ValueError("Conflicting archive member: " + name)
        self.data[name] = data
        if path is not None:
            key = (str(path), digest, name)
            if key not in self.source_keys:
                self.source_keys.add(key)
                self.sources.append(dict(role=role, source_path=str(path),
                                         sha256=digest, bytes=len(data), archive_member=name))

    def file(self, path, name, *, expected=None, role="evidence"):
        p = C.resolve(path)
        self.add(name, p.read_bytes(), path=str(p), expected=expected, role=role)

    def json(self, name, value):
        self.add(name, encoded(value))


def relative_source(path):
    p = C.resolve(path)
    try:
        return p.relative_to(C.ROOT).as_posix()
    except ValueError:
        return "external/" + bytes_sha(str(p).encode())[:12] + "/" + p.name


def find_source(path, expected):
    p = C.resolve(path)
    candidates = [p]
    try:
        relative = p.relative_to(C.ROOT)
        candidates += [C.OUT / "source_snapshot" / relative, OLD / "source_snapshot" / relative]
    except ValueError:
        pass
    for candidate in candidates:
        if candidate.is_file() and C.sha(candidate) == expected:
            return candidate
    for candidate in (C.OUT / "audits").rglob("*.py"):
        if C.sha(candidate) == expected:
            return candidate
    raise RuntimeError("Actual executed source unavailable: " + str(path) + " " + expected)


def local_asset(rows, path, sha256, kind, *, receipt=None):
    p = C.resolve(path)
    if not p.is_file():
        raise RuntimeError("Manifest asset missing locally: " + str(p))
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("Local-only asset needs recorded SHA256")
    rows.append(dict(kind=kind, path=str(p), sha256=sha256, bytes=p.stat().st_size,
                     verification="RECORDED_SHA_FROM_FROZEN_REGISTRATION_OR_COMPLETED_RECEIPT; NOT_REHASHED_BY_ARCHIVER",
                     receipt_path=str(receipt) if receipt else ""))


def collect_review_gallery_manifests(entries, reviews):
    """Bind actual CSV review pages to their original small gallery manifests.

    This copies manifest bytes only. Images are hash-checked and stay local;
    no rendering, preview creation, visual rating, or completion claim occurs.
    """
    manifests, image_hashes, references = {}, {}, []
    for cohort, (csv_path, csv_sha, rows) in reviews:
        for row_number, row in enumerate(rows, 2):
            for kind, path_key, hash_key in (
                    ("depth", "image_path", "image_sha256"),
                    ("sections", "section_image_path", "section_image_sha256")):
                source = row.get(path_key, "")
                expected = row.get(hash_key, "")
                if not source or not expected:
                    raise ValueError("Actual review page path/hash missing: " + path_key)
                image_path = C.resolve(source)
                if str(image_path) not in image_hashes:
                    image_hashes[str(image_path)] = checked_file(image_path)[1]
                if image_hashes[str(image_path)] != expected:
                    raise ValueError("Actual reviewed page hash differs: " + source)
                # Depth pages sit beside manifest; section pages use images/.
                candidates = (image_path.parent / "manifest.json",
                              image_path.parent.parent / "manifest.json")
                manifest_path = next((v for v in candidates if v.is_file()), None)
                if manifest_path is None:
                    raise RuntimeError("Actual reviewed gallery manifest missing: " + source)
                key = str(manifest_path)
                if key not in manifests:
                    checked, digest = checked_file(manifest_path)
                    data = C.read(checked)
                    if data.get("status") not in ("COMPLETE", "COMPLETE_RENDERING_NOT_REVIEW"):
                        raise ValueError("Actual reviewed gallery is incomplete")
                    member = "reviews/source_gallery_manifests/" + relative_source(checked)
                    entries.file(checked, member, expected=digest,
                                 role="actual_review_source_gallery_manifest")
                    manifests[key] = (data, digest, member)
                data, digest, member = manifests[key]
                pages = [v for v in data.get("images", ())
                         if C.resolve(v["path"]) == image_path]
                if len(pages) != 1:
                    raise ValueError("Actual review image not unique in source gallery")
                page = pages[0]
                if (page.get("sha256") != expected
                        or page.get("request_id") != row["request_id"]
                        or page.get("parent_id") != row["parent_id"]
                        or int(page["candidate"]) != int(row["candidate"])
                        or row["label"] not in data.get("labels", ())):
                    raise ValueError("Actual review identity differs from source gallery")
                references.append(dict(
                    cohort=cohort, review_csv_path=str(csv_path), review_csv_sha256=csv_sha,
                    review_csv_line=row_number, label=row["label"],
                    request_id=row["request_id"], stage=row["stage"], kind=kind,
                    source_image_path=source, source_image_sha256=expected,
                    manifest_image_path=page["path"], source_manifest_path=key,
                    source_manifest_sha256=digest, manifest_archive_member=member,
                    source_image_included=False))
    entries.add("reviews/ACTUAL_REVIEW_SOURCE_PAGES.csv", csv_bytes(references))
    identity = dict(
        status="SOURCE_IDENTITY_BOUND", review_page_references=len(references),
        distinct_source_images=len(image_hashes), source_gallery_manifests=len(manifests),
        original_manifest_bytes_copied=True, original_review_CSV_unchanged=True,
        source_images="LOCAL_ONLY; original paths and verified SHA256 are in ACTUAL_REVIEW_SOURCE_PAGES.csv",
        final_48_previews="SEPARATE_ALL_MODEL_DISPLAY_SET; not the actual subset pages used by reviewers",
        new_renders=0, new_visual_reviews=0, new_model_calls=0,
        limitations=["The final all-model previews may differ in layout and scale from actual reviewed subset pages.",
                     "Manifest copying and image hashing do not constitute visual review."])
    entries.json("reviews/SOURCE_PAGE_IDENTITY.json", identity)
    return identity


def collect(gate):
    entries = Entries()
    reg = gate["reg"]
    entries.file(C.OUT / "REPORT.md", "REPORT.md", expected=gate["report_sha"])
    entries.file(C.OUT / "registration.json", "contracts/registration.json", expected=gate["reg_sha"])
    entries.file(gate["audit_path"], "contracts/final_audit.json", expected=gate["audit_sha"])
    entries.file(gate["status_path"], "contracts/final_training_status.json")
    for relative in ("plans/replay_binding.json", "inputs/topology_manifest.json"):
        entries.file(C.OUT / relative, "contracts/" + relative)
    entries.file(C.ROOT / "native_t1/docs/control_metrics.md", "contracts/control_metrics.md")
    entries.file(C.CODE / "README.md", "IMPLEMENTATION.md")
    entries.file(gate["new_review"][0], "reviews/new_AI_PROXY_144.csv", expected=gate["new_review"][1])
    entries.file(gate["old_review"][0], "reviews/baseline_AI_PROXY_192.csv", expected=gate["old_review"][1])
    collect_review_gallery_manifests(entries, (("new_candidate", gate["new_review"]),
                                              ("historical_baseline", gate["old_review"])))
    for pointer in ("evaluation/comparison/latest.json", "training_analysis/latest.json"):
        entries.file(C.OUT / pointer, "contracts/" + pointer)
    for name, summary_path in (("comparison", gate["comparison_path"]), ("training", gate["training_path"])):
        for path in sorted(summary_path.parent.iterdir()):
            if path.is_file() and path.suffix.lower() in (".csv", ".json", ".png", ".svg"):
                entries.file(path, "results/" + name + "/" + path.name, role="complete_latest_snapshot")
    plots_path, plots = latest_snapshot(C.OUT / "evaluation/result_plots/latest.json")
    if plots.get("figures_created") != 4 or plots.get("rows") != 148:
        raise RuntimeError("Final four paired figures and complete 148-row plot data required")
    entries.file(C.OUT / "evaluation/result_plots/latest.json", "contracts/evaluation/result_plots/latest.json")
    entries.file(plots_path, "results/figures/completion.json")
    for source, digest in sorted(plots["files"].items()):
        path = C.resolve(source)
        if path.parent.resolve() != plots_path.parent.resolve():
            raise ValueError("Plot payload outside completed snapshot")
        entries.file(path, "results/figures/" + path.name, expected=digest, role="complete_final_paired_figures")
    for path in sorted((C.OUT / "evaluation/tables").rglob("*.csv")):
        entries.file(path, "results/tables/" + path.relative_to(C.OUT / "evaluation/tables").as_posix())
    for path in sorted((C.OUT / "evaluation/review").rglob("*.csv")):
        if not path.name.startswith("BLANK_"):
            entries.file(path, "reviews/final_files/" + path.relative_to(C.OUT / "evaluation/review").as_posix())
    for path in sorted((C.OUT / "inputs").rglob("*.csv")):
        entries.file(path, "contracts/inputs/" + path.relative_to(C.OUT / "inputs").as_posix())
    for path in sorted((C.OUT / "audits").rglob("*")):
        if path.is_file() and path.suffix.lower() in (".json", ".csv", ".py", ".diff", ".md"):
            entries.file(path, "audits/" + path.relative_to(C.OUT / "audits").as_posix())
    added = []
    for path in sorted(C.CODE.iterdir()):
        if path.is_file() and path.suffix.lower() in (".py", ".ps1", ".md"):
            rel = path.relative_to(C.ROOT).as_posix()
            entries.file(path, "sources/current/" + rel, role="current_pilot_source")
            added.extend(difflib.unified_diff([], path.read_text(encoding="utf-8-sig").splitlines(True),
                                              fromfile="/dev/null", tofile="b/" + rel))
    entries.add("code.diff", "".join(added).encode("utf-8"))
    for path in sorted((C.OUT / "source_snapshot").rglob("*")):
        if path.is_file():
            entries.file(path, "sources/frozen/" + path.relative_to(C.OUT / "source_snapshot").as_posix(),
                         role="frozen_executed_source")
    definitions, local, source_payloads, indices = {}, [], [], {}
    for label in ALL_LABELS:
        for stage in STAGES:
            path = C.OUT / "evaluation/indices" / label / (stage.lower() + "_index.json")
            index = C.read(path)
            indices[label, stage] = {r["request_id"]: r for r in index["rows"]}
            entries.file(path, "contracts/indices/" + label + "/" + path.name)
            for row in index["rows"]:
                key = stage.lower()
                local_asset(local, row[key + "_path"], row[key + "_sha256"],
                            label + "_" + stage + "_GEOMETRY_WITH_SAVED_TRAJECTORY_WHERE_PRESENT", receipt=path)
        score_path = C.OUT / "evaluation/score_completion" / (label + ".json")
        score = C.read(score_path)
        entries.file(score_path, "contracts/score_completion/" + score_path.name)
        for path, digest in score.get("definition_hashes", {}).items():
            definitions[path, digest] = "score_completion"
        for ref in score["receipts"]:
            path, digest = checked_file(ref["path"], ref["sha256"])
            payload = C.read(path)
            source_payloads.append(dict(label=label, request_id=ref["request_id"], stage=ref["stage"],
                                        source_path=str(path), source_sha256=digest, payload=payload))
            for source, sha in payload.get("identity", {}).get("definition_hashes", {}).items():
                definitions[source, sha] = "actual_score_identity"
            for source, sha in payload.get("files", {}).items():
                local_asset(local, source, sha, "LOW_LEVEL_SCORE_PAYLOAD", receipt=path)
        boundary_path = C.OUT / "evaluation/boundary_completion" / (label + ".json")
        boundary = C.read(boundary_path)
        entries.file(boundary_path, "contracts/boundary_completion/" + boundary_path.name)
        for ref in boundary["receipts"]:
            for source, sha in ref["identity"].get("definition_hashes", {}).items():
                definitions[source, sha] = "actual_boundary_identity"
            for source, sha in ref["files"].items():
                local_asset(local, source, sha, "LOW_LEVEL_BOUNDARY_QUERY_TRACE", receipt=boundary_path)

    entries.json("contracts/actual_per_output_score_receipts.json", source_payloads)
    if not definitions:
        for path, digest in reg["evaluation_code_hashes"].items():
            definitions[path, digest] = "registered_evaluation_definition_fallback"
    definition_rows = []
    for (path, digest), origin in sorted(definitions.items()):
        source = find_source(path, digest)
        member = "sources/actual_scoring/" + digest[:12] + "/" + relative_source(path)
        entries.file(source, member, expected=digest, role="actual_scoring_definition")
        definition_rows.append(dict(definition_path=path, sha256=digest, evidence=origin,
                                    resolved_exact_source=str(source), archive_member=member))
    entries.add("ACTUAL_SCORING_SOURCE_MANIFEST.csv", csv_bytes(definition_rows))
    inherited = []
    for row in gate["old_review"][2]:
        if row["label"] != "FULL_S0000":
            continue
        rid, stage = row["request_id"], row["stage"]
        a, b = indices["CONTOUR_S0", stage][rid], indices["FULL_S0000", stage][rid]
        key = stage.lower() + "_sha256"
        if a[key] != b[key]:
            raise ValueError("Cannot inherit S0 labels from different geometry")
        inherited.append(dict(row, label="CONTOUR_S0", historical_label="FULL_S0000",
                              historical_review_csv=str(gate["old_review"][0]),
                              historical_review_csv_sha256=gate["old_review"][1],
                              inherited_geometry_sha256=a[key], new_visual_review=False,
                              review_provenance="INHERITED_IDENTICAL_FULL_S0000_GEOMETRY"))
    if len(inherited) != 48:
        raise ValueError("Expected 48 explicit inherited S0 rows")
    entries.add("reviews/inherited_CONTOUR_S0_48.csv", csv_bytes(inherited))
    # Existing validated receipt hashes avoid rereading multi-GB checkpoints.
    local_asset(local, reg["source_checkpoint"], reg["source_checkpoint_sha256"],
                "SOURCE_GENERATOR_MATCHED_ADAMW_RNG", receipt=C.OUT / "registration.json")
    for folder in ("checkpoints", "inference"):
        for path in sorted((C.OUT / folder).glob("*.json")):
            rec = C.read(path)
            if "path" in rec and "sha256" in rec:
                local_asset(local, rec["path"], rec["sha256"], folder.upper(), receipt=path)
                entries.file(path, "contracts/" + folder + "/" + path.name)
    for request in reg["requests"]:
        for path_key, hash_key in (("input_path", "input_sha256"), ("full_noise_path", "full_noise_file_sha256"),
                                   ("condition_source_path", "condition_source_sha256")):
            if request.get(path_key) and request.get(hash_key):
                local_asset(local, request[path_key], request[hash_key], "FROZEN_INPUT_" + path_key,
                            receipt=C.OUT / "registration.json")
    for path_key, hash_key in (("replay_plan_path", "replay_plan_sha256"),
                               ("replay_exposure_path", "replay_exposure_sha256"),
                               ("source_data", "source_data_sha256"),
                               ("interface_manifest_path", "interface_manifest_sha256")):
        local_asset(local, reg[path_key], reg[hash_key], "FROZEN_" + path_key,
                    receipt=C.OUT / "registration.json")
    ot = []
    for path in sorted((OLD / "ot/FULL_SEAT").glob("[0-9][0-9][0-9][0-9][0-9].json")):
        raw = path.read_bytes()
        rec = json.loads(raw)
        data_path = path.with_suffix(".npz")
        if not data_path.is_file():
            raise RuntimeError("Original OT cache is missing")
        ot.append(dict(path=str(data_path), sha256=rec["sha256"], bytes=data_path.stat().st_size,
                       receipt_path=str(path), receipt_sha256=bytes_sha(raw),
                       batch_index=rec["identity"]["batch_index"],
                       verification="ORIGINAL_FROZEN_RECEIPT_SHA; CACHE_BYTES_NOT_REHASHED"))
    if len(ot) != 6000 or sorted(r["batch_index"] for r in ot) != list(range(6000)):
        raise RuntimeError("Original 6000-batch OT inventory incomplete")
    entries.add("local_assets/OT_CACHE_6000.csv", csv_bytes(ot))
    dedup = {(r["kind"], r["path"], r["sha256"]): r for r in local}
    entries.add("local_assets/LOCAL_ONLY_ASSETS.csv", csv_bytes([dedup[k] for k in sorted(dedup)]))
    entries.json("local_assets/POLICY.json", dict(
        large_weights_RAW_trajectories_OT="LOCAL_ONLY_BY_RECORDED_PATH_SHA256",
        per_query_boundary_and_score_arrays="LOCAL_ONLY_BY_COMPLETED_RECEIPT_PATH_SHA256",
        latest_full_per_output_per_parent_curves="ALL_SELECTED_CSV_INCLUDED_UNCHANGED",
        original_assets_modified=False, new_model_calls=0, new_scores=0,
        note="The package supports saved-score re-aggregation; meshes for rerunning geometric scores remain local."))
    entries.add("SOURCE_FILE_HASHES.csv", csv_bytes(entries.sources))
    return entries


def previews(images, size, quality):
    from PIL import Image
    result, manifest = {}, []
    for item in sorted(images, key=lambda r: (r["kind"], r["request_id"])):
        path = C.resolve(item["path"])
        with Image.open(path) as original:
            image = original.convert("RGB")
            original_size = list(image.size)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, subsampling=0,
                       optimize=True, progressive=False)
            data = buffer.getvalue()
        name = "previews/" + item["kind"] + "/" + item["parent_id"][:8] + "_n" + str(item["candidate"]) + ".jpg"
        if name in result:
            raise ValueError("Preview name collision")
        result[name] = data
        manifest.append(dict(kind=item["kind"], request_id=item["request_id"],
                             source_path=str(path), source_sha256=item["sha256"],
                             archive_member=name, sha256=bytes_sha(data), bytes=len(data),
                             original_pixels=original_size, preview_pixels=list(image.size),
                             uniform_long_edge=size, uniform_JPEG_quality=quality))
    if len(result) != 48:
        raise RuntimeError("Exactly 24 depth and 24 section previews required")
    page = '<!doctype html><meta charset="utf-8"><title>All 24 requests: compact previews</title>'
    page += '<p>All requests retained. Compressed previews of saved evidence; not new renders/reviews. Original paths and hashes: PREVIEWS.json.</p>'
    page += '<p>These 48 all-model preview pages are a separate display set, not the actual subset pages used by reviewers. Actual reviewed page paths, SHA256, and original gallery manifests are under reviews/.</p>'
    for kind in ("depth", "sections"):
        page += "<h1>" + html.escape(kind) + "</h1>"
        for item in manifest:
            if item["kind"] == kind:
                link = item["archive_member"].removeprefix("previews/")
                page += '<h2>' + html.escape(item["request_id"]) + '</h2><a href="' + link + '"><img loading="lazy" style="width:100%" src="' + link + '"></a>'
    result["previews/index.html"] = page.encode("utf-8")
    return result, manifest


def write_zip(path, members):
    """Immutable member order/timestamps; preserve failed partial files."""
    temp = path.with_suffix(".partial")
    if temp.exists():
        raise RuntimeError("Inspect partial archive before explicit recovery: " + str(temp))
    with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_LZMA, allowZip64=False) as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.lower().endswith((".jpg", ".png")) else zipfile.ZIP_LZMA
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    os.replace(temp, path)


def verify_zip(path):
    path = C.resolve(path)
    if path.stat().st_size >= MAX_BYTES:
        raise ValueError("ZIP must be strictly smaller than 10,000,000 bytes")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError("Unsafe or duplicate member names")
        manifest = json.loads(z.read("MANIFEST.json"))
        expected = {r["archive_member"]: r for r in manifest["members"]}
        if set(names) != set(expected) | {"MANIFEST.json"}:
            raise ValueError("Internal manifest must cover every member")
        for name, row in expected.items():
            data = z.read(name)
            if len(data) != row["bytes"] or bytes_sha(data) != row["sha256"]:
                raise ValueError("Internal hash/size mismatch: " + name)
        counts = {}
        for name, expected_rows in (("reviews/new_AI_PROXY_144.csv", 144),
                                    ("reviews/baseline_AI_PROXY_192.csv", 192),
                                    ("reviews/inherited_CONTOUR_S0_48.csv", 48),
                                    ("results/comparison/main_per_output.csv", 192),
                                    ("results/comparison/reference_per_output.csv", 192),
                                    ("results/training/per_update.csv", 6000),
                                    ("local_assets/OT_CACHE_6000.csv", 6000)):
            rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
            if len(rows) != expected_rows:
                raise ValueError("Packed full evidence row count differs: " + name)
            counts[name] = len(rows)
        source_refs = list(csv.DictReader(io.StringIO(
            z.read("reviews/ACTUAL_REVIEW_SOURCE_PAGES.csv").decode("utf-8-sig"))))
        wanted_refs = set()
        for cohort, csv_name in (("new_candidate", "reviews/new_AI_PROXY_144.csv"),
                                 ("historical_baseline", "reviews/baseline_AI_PROXY_192.csv")):
            for row in csv.DictReader(io.StringIO(z.read(csv_name).decode("utf-8-sig"))):
                for kind, pk, hk in (("depth", "image_path", "image_sha256"),
                                     ("sections", "section_image_path", "section_image_sha256")):
                    wanted_refs.add((cohort, row["label"], row["request_id"], row["stage"],
                                     kind, row[pk], row[hk]))
        actual_refs = [(r["cohort"], r["label"], r["request_id"], r["stage"],
                        r["kind"], r["source_image_path"], r["source_image_sha256"])
                       for r in source_refs]
        if len(actual_refs) != 672 or len(set(actual_refs)) != 672 or set(actual_refs) != wanted_refs:
            raise ValueError("Actual reviewed page references differ from complete original CSVs")
        manifest_cache = {}
        for r in source_refs:
            member = r["manifest_archive_member"]
            if member not in manifest_cache:
                payload = z.read(member)
                manifest_cache[member] = (bytes_sha(payload), json.loads(payload))
            digest, data = manifest_cache[member]
            pages = [v for v in data.get("images", ()) if v["path"] == r["manifest_image_path"]]
            if (digest != r["source_manifest_sha256"] or len(pages) != 1
                    or pages[0].get("sha256") != r["source_image_sha256"]
                    or pages[0].get("request_id") != r["request_id"]
                    or r["label"] not in data.get("labels", ())):
                raise ValueError("Packed actual review source manifest binding differs")
        counts["reviews/ACTUAL_REVIEW_SOURCE_PAGES.csv"] = len(source_refs)
        preview = json.loads(z.read("PREVIEWS.json"))
        if len(preview) != 48 or any(sum(r["kind"] == kind for r in preview) != 24 for kind in ("depth", "sections")):
            raise ValueError("Incomplete preview coverage")
        if not list(csv.DictReader(io.StringIO(z.read("ACTUAL_SCORING_SOURCE_MANIFEST.csv").decode("utf-8-sig")))):
            raise ValueError("Actual scoring source missing")
    return dict(status="PASS", path=str(path), sha256=C.sha(path),
                bytes=path.stat().st_size, member_count=len(names),
                verified_member_sha256=len(expected), row_counts=counts, previews=48)


def build(final_audit):
    if "torch" in __import__("sys").modules:
        raise RuntimeError("Archive process must not import models")
    started = time.perf_counter()
    gate = final_gate(final_audit)
    entries = collect(gate)
    source_signature = bytes_sha(encoded(dict(
        version=VERSION, members={k: bytes_sha(v) for k, v in entries.data.items()},
        images=[(r["kind"], r["request_id"], r["sha256"]) for r in gate["images"]],
        final_audit_sha256=gate["audit_sha"], report_sha256=gate["report_sha"])))
    dest = C.checked(ARCHIVE / ("snapshot_" + source_signature[:16]))
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / "completion.json"
    if done.exists():
        saved = C.read(done)
        if saved.get("source_signature") != source_signature:
            raise ValueError("Completed archive identity differs")
        result = verify_zip(saved["archive_path"])
        if result["sha256"] != saved["sha256"]:
            raise ValueError("Completed ZIP bytes changed")
        C.write(ARCHIVE / "latest.json", saved, replace=True)
        return saved
    attempts = []
    for size, quality in PREVIEW_SETTINGS:
        pictures, preview_manifest = previews(gate["images"], size, quality)
        preview_bytes = sum(len(v) for k, v in pictures.items() if k.endswith(".jpg"))
        if preview_bytes > PREVIEW_TARGET_BYTES:
            attempts.append(dict(uniform_long_edge=size, uniform_quality=quality,
                                 preview_bytes=preview_bytes, status="PREVIEW_TARGET_EXCEEDED"))
            continue
        policy = dict(
            version=VERSION, byte_limit_exclusive=MAX_BYTES, preview_target_bytes=PREVIEW_TARGET_BYTES,
            uniform_preview_long_edge=size, uniform_preview_quality=quality,
            preview_count=48, all_requests_retained=True, selected_examples=False,
            all_latest_snapshot_CSV_and_actual_code_included=True,
            CSV_drop_to_fit=False, source_drop_to_fit=False, images_only_adjustment=True,
            text_compression="ZIP_LZMA", images_compression="ZIP_STORED",
            prior_partial_snapshots_included=False, actual_new_AI_PROXY_rows=144,
            historical_baseline_AI_PROXY_rows=192, inherited_candidate_zero_rows=48,
            actual_review_source_manifests_included=True, actual_source_images_duplicated=False,
            final_preview_set_differs_from_actual_review_source_pages=True,
            final_audit_sha256=gate["audit_sha"], report_sha256=gate["report_sha"],
            registration_sha256=gate["reg_sha"],
            final_audit_itself_declared_report_binding=gate["audit_report_binding_present"],
            source_signature=source_signature,
            limitations=["Compressed previews do not substitute for original image review.",
                         "The 48 all-model previews are distinct from actual reviewer subset pages; original paths, hashes and source gallery manifests are preserved under reviews/.",
                         "Score CSVs can be re-aggregated; meshes, full OT arrays and weights remain local.",
                         "Historical/inherited labels are not counted as new visual reviews."])
        members = dict(entries.data, **pictures)
        members["PREVIEWS.json"] = encoded(preview_manifest)
        members["PACKAGING_POLICY.json"] = encoded(policy)
        members["MANIFEST.json"] = encoded(dict(
            version=VERSION, source_signature=source_signature,
            self_hash_note="External ZIP SHA256 covers MANIFEST.json; all other members appear below.",
            members=[dict(archive_member=k, bytes=len(v), sha256=bytes_sha(v))
                     for k, v in sorted(members.items())]))
        path = dest / ("preview_" + str(size) + "_q" + str(quality) + ".zip")
        if not path.exists():
            write_zip(path, members)
        else:
            with zipfile.ZipFile(path) as z:
                if z.read("MANIFEST.json") != members["MANIFEST.json"]:
                    raise ValueError("Existing attempt contains different evidence")
        size_bytes = path.stat().st_size
        attempt = dict(uniform_long_edge=size, uniform_quality=quality,
                       preview_bytes=preview_bytes, zip_bytes=size_bytes,
                       path=str(path), status="PASS_SIZE" if size_bytes < MAX_BYTES else "SIZE_EXCEEDED")
        attempts.append(attempt)
        C.write(dest / ("attempt_" + str(size) + "_q" + str(quality) + ".json"), attempt)
        if size_bytes >= MAX_BYTES:
            continue
        verification = verify_zip(path)
        final = dest / "R2_FULL_CONTOUR_READOUT_V1_REVIEW.zip"
        if final.exists():
            if C.sha(final) != verification["sha256"]:
                raise ValueError("Existing final archive differs")
        else:
            os.replace(path, final)
        verification["path"] = str(final)
        C.write(dest / "verification.json", verification)
        receipt = dict(status="COMPLETE", version=VERSION, source_signature=source_signature,
                       archive_path=str(final), sha256=verification["sha256"], bytes=verification["bytes"],
                       final_audit_path=str(gate["audit_path"]), final_audit_sha256=gate["audit_sha"],
                       report_sha256=gate["report_sha"], registration_sha256=gate["reg_sha"],
                       verification_path=str(dest / "verification.json"),
                       actual_new_AI_PROXY_rows=144, inherited_rows=240,
                       preview_pages=48, source_file_manifest="SOURCE_FILE_HASHES.csv",
                       attempts=attempts, wall_seconds=time.perf_counter() - started,
                       new_model_calls=0, new_scores=0, new_visual_reviews=0)
        C.write(done, receipt)
        C.write(ARCHIVE / "latest.json", receipt, replace=True)
        return receipt
    blocked = dict(status="BLOCKED_SIZE_BUDGET", source_signature=source_signature,
                   byte_limit_exclusive=MAX_BYTES, attempts=attempts,
                   mandatory_CSV_or_source_removed=False,
                   explanation="Uniform preview reductions exhausted; retain evidence and report size block.")
    C.write(dest / "size_blocked.json", blocked)
    raise RuntimeError("Archive exceeds 10 MB; no CSV or source was removed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("status", "build", "verify"), default="status")
    parser.add_argument("--final-audit", help="Explicit PASS final audit, required for build")
    parser.add_argument("--zip", dest="zip_path", help="Existing archive path for verify")
    args = parser.parse_args()
    if args.stage == "build":
        if not args.final_audit:
            parser.error("--final-audit required; no guessed final review")
        result = build(args.final_audit)
    elif args.stage == "verify":
        if not args.zip_path:
            parser.error("--zip required")
        result = verify_zip(args.zip_path)
    else:
        pointer = ARCHIVE / "latest.json"
        result = C.read(pointer) if pointer.exists() else dict(status="NOT_PACKAGED", read_only=True)
    print(json.dumps(C.clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

