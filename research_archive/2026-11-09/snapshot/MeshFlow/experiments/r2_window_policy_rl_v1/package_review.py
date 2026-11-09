"""CPU-only immutable review ZIP builder; default is a read-only dry run.

Run with WSL Python from the repository. No project modules are imported.
Missing terminal evidence is PENDING; contradictions are FAIL. A ZIP is only
written with --package after complete checks, and must be strictly below 10 MB.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

VERSION = "window_policy_rl_review_package_v1"
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LIMIT = 10_000_000
LABELS = ("FULL_DET", "FULL_STOCH", "RL_STOCH")
REVIEW_FIELDS = ("RAW_adoption", "POST_adoption", "POST_body_complete")
FORBIDDEN = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".ply", ".obj", ".pkl", ".safetensors"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def js(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def resolve(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and text.startswith("/mnt/"):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and len(text) > 2 and text[1:3] == ":/":
        text = "/mnt/" + text[0].lower() + "/" + text[3:]
    p = Path(text)
    return (p if p.is_absolute() else OUT / p).resolve()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class Inventory:
    def __init__(self):
        self.members = {}
        self.sources = {}
        self.missing = []
        self.checks = []

    def add(self, value, arc=None, expected=None):
        p = resolve(value)
        if not p.is_file():
            self.missing.append(str(p))
            return None
        if arc is None:
            arc = p.relative_to(OUT).as_posix()
        require(not PurePosixPath(arc).is_absolute() and ".." not in PurePosixPath(arc).parts,
                "Unsafe archive member: " + arc)
        require(p.suffix.lower() not in FORBIDDEN, "Large raw/weight asset excluded from ZIP: " + str(p))
        data = p.read_bytes()
        digest = sha(data)
        require(expected is None or digest == expected, "SHA256 differs: " + str(p))
        require(arc not in self.members or self.members[arc] == data, "Conflicting archive member: " + arc)
        self.members[arc] = data
        self.sources[arc] = dict(original_path=str(p), sha256=digest, bytes=len(data))
        return data

    def read(self, value, arc=None, expected=None):
        data = self.add(value, arc, expected)
        return None if data is None else json.loads(data.decode("utf-8-sig"))

    def csv(self, value, count=None):
        data = self.add(value)
        if data is None:
            return None
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        require(count is None or len(rows) == count, "CSV row count differs: " + str(value))
        return rows

    def generated(self, arc, data):
        require(arc not in self.members, "Duplicate generated member: " + arc)
        self.members[arc] = data
        self.sources[arc] = dict(original_path=None, sha256=sha(data), bytes=len(data), generated_by=VERSION)


def complete_reviews(inv, path, gallery_rel, requests, aliases, expected_hash):
    reviews = inv.read(path, expected=expected_hash)
    gallery = inv.read(gallery_rel)
    if reviews is None or gallery is None:
        return None
    require(reviews.get("status") == "COMPLETE", "Actual review bundle is not COMPLETE: " + str(path))
    require(gallery.get("status") == "COMPLETE" and gallery["identity"].get("model_blinded") is True,
            "Gallery incomplete or nonblind: " + gallery_rel)
    qids = {q["request_id"] for q in requests}
    images = {r["request_id"]: r for r in gallery["images"]}
    require(len(images) == len(gallery["images"]) == len(qids) and set(images) == qids, "Gallery request coverage differs")
    expected = {(r, a) for r in qids for a in aliases}
    rows = reviews.get("rows", [])
    seen = set()
    for row in rows:
        key = (row["request_id"], row["label"])
        require(key in expected and key not in seen, "Unexpected/duplicate actual review row")
        seen.add(key)
        require(row.get("review_status") == "REVIEWED" and row.get("reviewer_kind") == "AI_PROXY"
                and bool(row.get("reviewer")) and row.get("blinded") in (True, "True", "true", "1"),
                "Actual blind reviewer identity missing")
        require(all(row.get(f) in ("YES", "NO", "UNKNOWN") for f in REVIEW_FIELDS), "Unknown review labels")
        im = images[key[0]]
        require(row.get("image_sha256") == im["image_sha256"], "Review/gallery image identity differs")
        require(resolve(row["image_path"]) == resolve(im["image_path"]), "Review/gallery image path differs")
        require(bool(row.get("notes", "").strip()), "Actual review notes missing")
    require(seen == expected, "Incomplete actual image review coverage")
    for im in images.values():
        p = resolve(im["image_path"])
        require(p.is_file(), "Reviewed image missing: " + str(p))
        require(sha(p.read_bytes()) == im["image_sha256"], "Reviewed image changed: " + str(p))
    inv.checks.append(dict(check="actual_reviews", path=str(resolve(path)), rows=len(rows), pages=len(images),
                           image_bytes_verified=True, automatic_semantic_labels=0))
    return images


def source_files(inv):
    code_dir = ROOT / "r2/pilots/window_policy_rl_v1"
    lock = inv.read("training_source_lock.json")
    if lock:
        require(lock.get("status") == "FROZEN", "Training source lock not frozen")
    locked = {str(resolve(k)): v for k, v in (lock or {}).get("source_files", {}).items()}
    rows = []
    diff = []
    for p in sorted(code_dir.iterdir()):
        if not p.is_file() or p.suffix not in (".py", ".md"):
            continue
        rel = p.relative_to(ROOT).as_posix()
        arc = "source_snapshot/implementation/" + rel
        data = inv.add(p, arc=arc, expected=locked.get(str(p.resolve())))
        if data is None:
            continue
        digest = sha(data)
        rows.append(dict(path=str(p), repository_relative_path=rel, package_path=arc,
                         sha256=digest, frozen_sha256=locked.get(str(p.resolve())),
                         role="FROZEN_EXPERIMENT_SOURCE" if str(p.resolve()) in locked else "POSTRUN_SUPPORT_SOURCE"))
        diff.extend(["diff --git a/" + rel + " b/" + rel + "\n", "new file mode 100644\n", "# SHA256 " + digest + "\n"])
        # This is explicitly a full new-file diff, never a fabricated git-base comparison.
        lines = data.decode("utf-8-sig").splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        diff.extend(difflib.unified_diff([], lines, fromfile="/dev/null", tofile="b/" + rel))
    require(set(locked).issubset({r["path"] for r in rows}), "Frozen experiment source omitted from snapshot")
    inv.add(__file__, arc="source_snapshot/package_review.py")
    inv.generated("code.diff", "".join(diff).encode("utf-8"))
    inv.generated("source_snapshot/implementation_manifest.json", js(dict(status="COMPLETE", files=rows,
        diff_path="code.diff", diff_sha256=sha(inv.members["code.diff"]),
        diff_contract="Hash-bound full new-file diff from /dev/null; not a git historical/base diff; original bytes in snapshot are authoritative.")))
    snapshot = inv.read("source_snapshot/scorers_manifest.json")
    if snapshot:
        require(snapshot.get("status") == "COMPLETE" and snapshot.get("definition_count") == 19
                and len(snapshot.get("files", [])) == 19, "Actual scorer snapshot does not contain19 files")
        for row in snapshot["files"]:
            require(row["copied_sha256"] == row["score_definition_sha256"] == row["original_sha256"], "Scorer identity chain differs")
            inv.add(row["package_relative_path"], expected=row["copied_sha256"])
    for rel in ("source_snapshot/scorers_manifest.csv", "source_snapshot/protected_assets_manifest.json",
                "source_snapshot/frozen_request_noise_hashes.json", "source_snapshot/packaging_preparation_completion.json",
                "source_snapshot/README.md", "source_snapshot/prepare_snapshot.py"):
        inv.add(rel)


def scientific_figures(inv):
    # Include both explicitly versioned calibration figures, not an outcome-based choice.
    for rel in sorted((OUT / "figures").glob("calibration_quantitative*.json")):
        obj = inv.read(rel)
        require(obj.get("status") == "COMPLETE", "Calibration scientific figure incomplete")
        for p, digest in obj["files"].items():
            inv.add(p, expected=digest)
    kinds = set()
    for p in sorted((OUT / "figures/scientific").glob("*/plot_data.json")):
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if data.get("status") != "COMPLETE":
            continue  # Predeclared policy: partial scientific drafts stay local, indexed below.
        kinds.add(data.get("kind"))
        inv.add(p)
        for name, receipt in data["artifacts"].items():
            inv.add(name, expected=receipt["sha256"])
    for kind in ("training_ledger", "final_comparison"):
        if kind not in kinds:
            inv.missing.append("COMPLETE scientific figure: " + kind)
    inv.generated("figures/scientific_draft_index.json", js(dict(policy="Only COMPLETE scientific versions are bundled; all local versions are indexed.",
        files=[dict(path=str(p), sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
               for p in sorted((OUT / "figures/scientific").glob("*/plot_data.json"))])))


def inventory():
    inv = Inventory()
    required = ("REPORT.md", "README_RESUME.md", "analysis/CALIBRATION_REPORT_DRAFT.md", "registration.json", "gate_contract.json", "training_config.json", "USER_PROMPT.txt",
                "user_budget_decision.json", "EXECUTION_PLAN.md", "resume.ps1", "calibration_source_lock.json",
                "engineering/final_artifact_plan.json", "inputs/manifest.json", "inputs/training_manifest.json",
                "training/final.json", "analysis/training_latest.json", "evaluation/analysis/readiness.json",
                "evaluation/analysis/summary.json", "analysis/decode_gate.json")
    objects = {}
    for rel in required:
        if rel.endswith(".json"):
            objects[rel] = inv.read(rel)
        else:
            inv.add(rel)
    manifest = objects.get("inputs/manifest.json")
    if manifest:
        qs = manifest["evaluation_requests"]
        require(len(qs) == len({q["request_id"] for q in qs}) == 44, "Final panel must contain44 unique requests")
        require(Counter(q["split"] for q in qs) == Counter(TRAIN=32, DEV=12), "Final TRAIN/DEV coverage differs")
        require(sorted(Counter(q["condition_id"] for q in qs).values()) == [4] * 11, "Final condition/noise coverage differs")
        registration = objects["registration.json"]
        if registration:
            require(sha(inv.members["inputs/manifest.json"]) == registration["manifest_sha256"], "Registration manifest hash differs")
    final = objects.get("training/final.json")
    if final:
        require(final.get("status") == "COMPLETE" and final.get("batches") == 64 and final.get("rollouts") == 512,
                "Training final receipt is not the fixed64/512 endpoint")
        require(len(final.get("records", [])) == 64, "Training final records incomplete")
        # Weight bytes stay local; reuse the final checkpoint hash verified by the controller and analysis.
        p = resolve(final["checkpoint_path"])
        require(p.is_file() and len(final.get("checkpoint_sha256", "")) == 64, "Retained endpoint checkpoint missing")
        inv.generated("retained_checkpoint.json", js(dict(path=str(p), sha256=final["checkpoint_sha256"], bytes=p.stat().st_size,
            hash_verification="Reused complete controller/final-analysis binding; packaging does not reload or rehash model weights",
            packaged=False, selection=final.get("selection"))))
        for row in final["records"]:
            inv.add(row["reward_path"])
            inv.add(row["checkpoint_receipt"])
    pointer = objects.get("analysis/training_latest.json")
    if pointer:
        if pointer.get("status") != "COMPLETE" or pointer.get("completed_logical_opportunities") != 64:
            inv.missing.append("COMPLETE analysis/training_latest.json for all64 opportunities")
        else:
            summary = inv.read(pointer["summary_path"], expected=pointer["summary_sha256"])
            require(summary["status"] == "COMPLETE" and not summary["violations"] and not summary["input_read_notes"], "Training summary violations")
            parent = resolve(pointer["summary_path"]).parent
            inv.csv(parent / "training_per_batch.csv", 64)
            inv.csv(parent / "training_per_condition.csv", 8)
            inv.csv(parent / "training_resources.csv")
            inv.csv(parent / "training_resident_services.csv")
            inv.add(parent / "training_input_sources.json")
    readiness = objects.get("evaluation/analysis/readiness.json")
    summary = objects.get("evaluation/analysis/summary.json")
    if readiness:
        if readiness.get("status") != "COMPLETE" or readiness.get("complete_summary_written") is not True:
            inv.missing.append("COMPLETE evaluation/analysis/readiness.json with actual final conclusions")
        elif summary:
            require(sha(inv.members["evaluation/analysis/summary.json"]) == readiness["summary_sha256"], "Final analysis summary hash differs")
    if summary:
        require(summary.get("status") == "COMPLETE" and summary.get("outputs") == 132 and summary.get("stage_scores") == 264
                and summary.get("review_rows") == 132, "Final analysis coverage incomplete")
    for rel in sorted((OUT / "analysis").glob("calibration*.csv")):
        inv.csv(rel)
    for name, count in (("calibration_per_output_stage.csv", 512),
                        ("calibration_per_condition_stage.csv", 64),
                        ("calibration_per_output_review_metrics.csv", 256)):
        inv.csv("analysis/" + name, count)
    for name in ("calibration_original_metrics.json", "calibration_quantitative.json", "calibration_readiness.json"):
        inv.add("analysis/" + name)
    for name, count in (("per_output_stage.csv", 264), ("per_condition_stage.csv", 66), ("per_model_split_stage.csv", 18),
                        ("paired_request_stage.csv", 264), ("paired_condition_and_split_stage.csv", 84),
                        ("all_geometry_tails.csv", None), ("per_output_review_metrics.csv", 132),
                        ("review_condition_and_split.csv", 42), ("paired_request_reviews.csv", 132),
                        ("paired_condition_and_split_reviews.csv", 42)):
        inv.csv("evaluation/analysis/" + name, count)
    for name in ("quantitative.json", "original_metrics.json"):
        inv.add("evaluation/analysis/" + name)
    gate = objects.get("analysis/decode_gate.json")
    if gate:
        require(gate.get("status") == "PASS" and gate.get("review_rows") == 256, "Calibration gate incomplete or notPASS")
        if manifest:
            complete_reviews(inv, gate["actual_reviews_path"], "calibration/blind_gallery/manifest.json",
                             manifest["calibration_requests"], "ABCD", gate["actual_reviews_sha256"])
        inv.add("calibration/private/label_mapping.json")
    if summary and manifest:
        images = complete_reviews(inv, summary["actual_reviews_path"], "evaluation/blind_gallery/manifest.json",
                                  manifest["evaluation_requests"], "ABC", summary["actual_reviews_sha256"])
        inv.add("evaluation/private/label_mapping.json")
        if images:
            selected = [q for q in manifest["evaluation_requests"] if int(q["candidate"]) == 0]
            require(len(selected) == len({q["condition_id"] for q in selected}) == 11, "Expected11 predeclared n0 pages")
            image_index = []
            for q in sorted(manifest["evaluation_requests"], key=lambda q:q["request_id"]):
                row = images[q["request_id"]]
                packaged = int(q["candidate"]) == 0
                arc = "previews/" + q["request_id"] + ".jpg" if packaged else None
                if packaged:
                    inv.add(row["image_path"], arc=arc, expected=row["image_sha256"])
                image_index.append(dict(request_id=q["request_id"], condition_id=q["condition_id"], candidate=q["candidate"],
                    local_path=row["image_path"], sha256=row["image_sha256"], package_path=arc))
            inv.generated("previews/image_index.json", js(dict(selection="All11 conditions, fixed candidate0/n0, independent of result",
                                                               preview_count=11, all44_pages=image_index)))
    # Actual independent reviewer rows and engineering decisions remain in full.
    for p in sorted((OUT / "review").glob("*.json")):
        inv.add(p)
    for p in sorted((OUT / "engineering").glob("*.json")):
        inv.add(p)
    for p in sorted((OUT / "engineering").glob("*_AUDIT_SOURCE.py")):
        inv.add(p)
    for p in sorted((OUT / "engineering/plot_sources").glob("*.py")):
        inv.add(p)
    source_files(inv)
    if summary and "source_snapshot/scorers_manifest.json" in inv.members:
        scored_sources = summary.get("input_sources", {})
        scorers = json.loads(inv.members["source_snapshot/scorers_manifest.json"])
        for row in scorers["files"]:
            require(scored_sources.get(row["original_path"]) == row["score_definition_sha256"],
                    "Final evaluation scorer differs from packaged actual source: " + row["original_path"])
    scientific_figures(inv)
    inv.generated("PACKAGE_README.md", ("# R2 window-policy RL review evidence\n\n"
        "All per-output, per-condition, paired, and actual blind-review rows are included. UNKNOWN remains explicit. "
        "The11 preview pages are each condition's predeclared n0, not selected best examples. Full local image paths and hashes are indexed.\n\n"
        "Weights, trajectories, mesh arrays, and buffers remain local and are not in this review ZIP. Protected asset and endpoint hashes identify them. "
        "The original19 scorer source files and actual pilot implementation bytes are included. code.diff is a full new-file diff, not a historical git diff.\n\n"
        "MANIFEST.sha256.json hashes every other ZIP member. The external receipt hashes the ZIP and manifest; self-hashing a manifest is intentionally excluded. "
        "No model calls, geometry re-scoring, training, git operations, or source modifications occur during packaging.\n").encode())
    return inv


def encoded_zip(inv):
    # Fixed timestamp/order make the archive reproducible for unchanged source bytes.
    members = dict(inv.members)
    members["MANIFEST.sha256.json"] = js(dict(version=VERSION, coverage="All ZIP members except this manifest",
        members=[dict(path=p, sha256=sha(data), bytes=len(data), **{k:v for k,v in inv.sources[p].items() if k not in ("sha256", "bytes")})
                 for p, data in sorted(members.items())], checks=inv.checks))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compresslevel=9)
    blob = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        require(archive.testzip() is None, "ZIP CRC verification failed")
        actual = {info.filename:sha(archive.read(info.filename)) for info in archive.infolist()}
        require(actual == {k:sha(v) for k,v in members.items()}, "ZIP member hash verification failed")
        sizes = [dict(path=i.filename, compressed_bytes=i.compress_size, bytes=i.file_size) for i in archive.infolist()]
    return blob, members["MANIFEST.sha256.json"], sizes


def run(package=False):
    inv = inventory()
    inventory_rows = [dict(path=k, bytes=len(v), sha256=sha(v)) for k,v in sorted(inv.members.items())]
    base = dict(version=VERSION, mode="PACKAGE" if package else "DRY_RUN", limit_bytes=LIMIT,
                strict_less_than=True, members=inventory_rows, checks=inv.checks, model_calls=0,
                geometry_rescores=0, weight_rehashes=0, deleted_files=0, frozen_code_modified=False)
    if inv.missing:
        return dict(base, status="PENDING", missing=sorted(set(inv.missing)), zip_written=False)
    blob, manifest, sizes = encoded_zip(inv)
    base.update(zip_bytes=len(blob), zip_sha256=sha(blob), manifest_sha256=sha(manifest),
                member_count=len(sizes), compressed_members=sizes)
    if len(blob) >= LIMIT:
        return dict(base, status="FAIL_SIZE_LIMIT", zip_written=False,
                    reason="All registered evidence retained; nothing silently omitted. Explicit package-plan change required.")
    if not package:
        return dict(base, status="READY_DRY_RUN", zip_written=False)
    dest = OUT / "R2_WINDOW_POLICY_RL_V1_REVIEW.zip"
    receipt = OUT / "R2_WINDOW_POLICY_RL_V1_REVIEW.receipt.json"
    require(not dest.exists() and not receipt.exists(), "Output exists; packaging refuses overwrite. Inspect prior receipt explicitly.")
    with dest.open("xb") as f:
        f.write(blob)
    require(sha(dest.read_bytes()) == sha(blob), "Saved ZIP hash differs")
    result = dict(base, status="COMPLETE", zip_written=True, zip_path=str(dest), receipt_path=str(receipt))
    with receipt.open("xb") as f:
        f.write(js(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Explicitly write the verified ZIP; default is read-only dry-run")
    parser.add_argument("--plan-json", help="Optional NEW planning/receipt JSON beneath the experiment; never overwrites")
    args = parser.parse_args()
    try:
        result = run(args.package)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        result = dict(version=VERSION, status="FAIL", reason=str(error), zip_written=False)
    if args.plan_json:
        p = resolve(args.plan_json)
        require(p.is_relative_to(OUT), "Plan path must stay under this experiment")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write(js(result))
    # Keep normal terminal output short; full inventory is available with --plan-json.
    print(json.dumps({k:v for k,v in result.items() if k not in ("members", "compressed_members")}, ensure_ascii=False, sort_keys=True))
    sys.exit(0 if result["status"] in ("READY_DRY_RUN", "COMPLETE") else 2)
