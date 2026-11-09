"""Read-only scientific acceptance, immutable source archive and small review ZIP.

This entrypoint never imports model/scorer modules, loads weights, draws noise,
runs geometry, scores, or invents reviews.  It requires the final report and
all60 actual RAW/FINAL reviews.  On COMPLETE it only verifies existing hashes
and returns; no source/manifest/receipt is recreated or overwritten.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_c_edgeflip_v1"
EVAL = OUT / "candidate_evaluation"
ARCHIVE = OUT / "R2_BENCH_C_EDGEFLIP_REVIEW.zip"
MAX_ARCHIVE_BYTES = 10 * (1 << 20)
COMPLETE = OUT / "COMPLETE.json"
SEALED = {}
HASH_CACHE = {}


def resolve(value):
    s = str(value).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    if os.name == "nt" and s.startswith("/mnt/e/"):
        s = "E:/" + s[len("/mnt/e/"):]
    elif os.name != "nt" and s.startswith("E:/"):
        s = "/mnt/e/" + s[len("E:/"):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(value):
    return json.loads(resolve(value).read_text("utf-8-sig"))


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(value):
    p = resolve(value)
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def require(value, message):
    if not value:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def put(path, data):
    path = resolve(path)
    if path.exists():
        require(path.read_bytes() == data, "Immutable artifact differs; no overwrite: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def verify(book):
    for p, expected in book.items():
        require(isinstance(expected, str) and len(expected) == 64, "Malformed hash: " + str(p))
        resolved = str(resolve(p))
        require(sha(resolved) == expected, "Sealed file changed: " + resolved)
        if resolved in SEALED:
            require(SEALED[resolved] == expected, "Conflicting sealed identity: " + resolved)
        SEALED[resolved] = expected


def seal(path):
    p = resolve(path)
    verify({str(p): sha(p)})


def receipt(path, status=("PASS", "COMPLETE")):
    r = read(path)
    require(r.get("status") in status, "Incomplete receipt: " + str(path))
    verify(r.get("files", {}))
    verify(r.get("protected", {}))
    seal(path)
    return r


def events(path):
    seal(path)
    return [json.loads(s) for s in resolve(path).read_text("utf-8-sig").splitlines()]


def event_time(row):
    if "unix" in row:
        return float(row["unix"])
    if "time_unix" in row:
        return float(row["time_unix"])
    return datetime.fromisoformat(row["utc"]).timestamp()


def csv_rows(path):
    seal(path)
    with resolve(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_completed():
    """No writes, no model imports; verification of the already sealed package."""
    done = read(COMPLETE)
    require(done["status"] == "COMPLETE", "Invalid completion marker")
    manifest_path = OUT / "artifact_manifest.json"
    require(sha(manifest_path) == done["artifact_manifest_sha256"], "Artifact manifest changed")
    verify(read(manifest_path)["files"])
    archive_receipt = read(OUT / "archive_receipt.json")
    require(archive_receipt["status"] == "COMPLETE" and resolve(archive_receipt["path"]) == ARCHIVE, "Archive identity differs")
    require(sha(ARCHIVE) == archive_receipt["sha256"] and ARCHIVE.stat().st_size == archive_receipt["bytes"], "Archive bytes/hash changed")
    require(ARCHIVE.stat().st_size < MAX_ARCHIVE_BYTES, "Review archive exceeds10MiB")
    with zipfile.ZipFile(ARCHIVE) as archive:
        require(archive.testzip() is None, "Corrupt review ZIP")
        require(set(archive.namelist()) == set(archive_receipt["entry_sha256"]), "ZIP entry inventory changed")
        for name, expected in archive_receipt["entry_sha256"].items():
            require(digest_bytes(archive.read(name)) == expected, "ZIP entry changed: " + name)
        require(archive.read("COMPLETE.json") == COMPLETE.read_bytes(), "Local/archive completion records differ")
    return dict(status="COMPLETE_VERIFIED_NO_REWRITE", files_verified=len(SEALED),
                archive_bytes=ARCHIVE.stat().st_size, model_calls=0, scoring_calls=0, files_written=0)


def scientific_acceptance():
    for name in ("REPORT.md", "RUNBOOK.md", "training_audit_final.json", "training_audit_scope.json", "report_numeric_audit.json", "report_numeric_audit_v2.json",
                 "visual_review/candidates_actual.json"):
        require((OUT / name).is_file(), "Await final report/audit/review: " + name)
        seal(OUT / name)
    frozen, trained, evaluation = read(OUT / "registration.json"), read(OUT / "training_registration.json"), read(EVAL / "registration.json")
    numeric = read(OUT / "report_numeric_audit_v2.json")
    require(numeric.get("status") == "PASS", "Independent report numeric audit has not passed")
    require(numeric["report_sha256"] == sha(OUT / "REPORT.md"), "Final report changed after its numeric audit")
    metric_manifest = ROOT / "experiments/r2_bench_four_routes_v1/exposure/data/eval_manifest.json"
    metric_fields = read(metric_manifest)["metric_fields"]
    require(len(metric_fields) == len(set(metric_fields)) == 177, "Original metric definition differs")
    seal(metric_manifest)
    verify(frozen["protected"])
    verify(frozen.get("inputs", {}))
    verify(evaluation["protected"])
    for name in ("registration.json", "training_registration.json", "candidate_evaluation/registration.json"):
        seal(OUT / name)
    for key in ("training_code_hashes", "gate_hashes", "input_case_hashes"):
        verify(trained[key])
    audit = receipt(OUT / "training_audit_final.json")
    verify(audit["evidence_hashes"])
    require(audit["training_registration_sha256"] == sha(OUT / "training_registration.json"), "Training audit registration differs")
    require((audit["total_train_forward"], audit["total_train_backward"], audit["total_optimizer_updates"]) == (3200, 3200, 400), "Training budget differs")
    require(audit["comparability_issues"] == [] and audit["scientific_failures"] == audit["scientific_retries"] == 0, "Unresolved scientific training issue")
    require(trained["updates"] == 200 and trained["arms"] == ["PLAIN200", "FLIP200"], "Training branch contract differs")
    for key in ("source_checkpoint", "manifest", "eval_manifest", "input_schedule", "coefficient_schedule", "flip_plan"):
        verify({trained[key]: trained[key+"_sha256"]})
    baseline, tasks = read(EVAL / "baseline.json"), read(EVAL / "tasks.json")
    cells = baseline + tasks
    require(len(baseline) == 6 and len(tasks) == 24 and len({c["cell_id"] for c in cells}) == 30, "Actual evaluation panel differs")
    require(Counter(c["label"] for c in cells) == {"EXPOSURE":10,"PLAIN200":10,"FLIP200":10}, "Per-model evaluation count differs")
    require(Counter(c["representation"] for c in cells) == {"ORIGINAL":18,"EDGE_FLIP":12}, "Geometry evaluation panel differs")
    seal(EVAL / "tasks.json"); seal(EVAL / "baseline.json")
    raw_ends, d0_starts, raw_times, score_count, common_count = [], [], [], 0, 0
    raw_receipts = []
    for task in tasks:
        cid = task["cell_id"]
        folder = resolve(task["raw_path"]).parent
        r = receipt(folder / "complete.json")
        require(r["generator_forwards"] == 50 and r["training_updates"] == r["D0_forwards"] == r["new_noise"] == 0, "RAW budget differs: " + cid)
        require(r["audit"]["anchored_all51_exact"], "C anchor invariant failed: " + cid)
        log = events(folder / "calls.jsonl")
        require(Counter(x["event"] for x in log) == {"FORWARD_STARTED":50,"FORWARD_COMPLETE":50}, "RAW call ledger incomplete/replayed: " + cid)
        raw_ends.append(max(event_time(x) for x in log))
        raw_times.append((folder / "complete.json").stat().st_mtime)
        raw_receipts.append(str(folder / "complete.json"))
        if task["label"] == "EXPOSURE":
            score_folder = resolve(task["raw_score_path"])
        else:
            score_folder = EVAL / "raw_scores" / cid
        score = receipt(score_folder / "audit_receipt.json")
        require(score["full_quality_scans"] == 1 and score["model_calls"] == score["D0_calls"] == score["new_queries"] == score["extra_common_FF_scans"] == 0, "RAW scoring scope differs")
        require(score["extra_common_CF_scans"] == int(task["representation"] == "EDGE_FLIP"), "COMMON_C RAW applicability differs")
        require(set(metric_fields).issubset(read(score_folder / "row.json")), "Original RAW177 fields missing")
        score_count += score["full_quality_scans"]
        common_count += score["extra_common_CF_scans"]
        post = EVAL / "post" / cid
        pr = receipt(post / "receipt.json")
        require(pr["D0_forwards"] == pr["GEO_calls"] == 1 and pr["new_noise"] == pr["new_queries"] == 0, "D0/GEO budget differs")
        dlog = events(post / "calls.jsonl")
        require(Counter(x["event"] for x in dlog) == {"D0_FORWARD_STARTED":1,"D0_FORWARD_RETURNED":1}, "D0 replay/pending call")
        d0_starts.append(next(event_time(x) for x in dlog if x["event"] == "D0_FORWARD_STARTED"))
        cp = EVAL / "cpu" / cid
        cr = receipt(cp / "complete.json")
        require(cr["expanded_base_scores"] == cr["final_scores"] == 1 and cr["model_calls"] == cr["D0_calls"] == cr["new_noise"] == cr["new_queries"] == 0, "CPU cascade budget differs")
        score_count += cr["expanded_base_scores"] + cr["final_scores"]
        for stage in ("BASE_POST", "CLEAN_FINAL_POST"):
            receipt(cp / stage / "scores/complete.json")
            require(set(metric_fields).issubset(read(cp / stage / "scores/row.json")), "Original POST177 fields missing")
        bp = receipt(EVAL / "base_pair_caches" / cid / "complete.json")
        require(bp["full_pair_scans"] == 1, "CPU baseline pair-cache scope differs")
        if task["representation"] == "EDGE_FLIP":
            common = receipt(EVAL / "final_common_C" / cid / "complete.json")
            require(common["new_CF_scans"] == 1 and common["new_FF_scans"] == 0, "COMMON_C FINAL scope differs")
            common_count += 1
    require(max(raw_ends) < min(d0_starts), "D0 began before all24 Euler rollouts ended")
    require(max(raw_times) < min(d0_starts), "D0 began before all24 RAW completion receipts were saved")
    barriers = sorted((EVAL / "d0_workers").glob("*/all_RAW_barrier.json"))
    require(bool(barriers), "No all-RAW barrier evidence")
    for path in barriers:
        barrier = receipt(path)
        require(barrier["outputs"] == 24 and all(str(resolve(r)) in {str(resolve(p)) for p in barrier["files"]} for r in raw_receipts), "D0 worker did not authenticate all24 RAW receipts")
    require(score_count == 72 and common_count == 24, "Actual scoring totals differ")
    for cell in baseline:
        receipt(resolve(cell["score_path"]) / "complete.json")
        receipt(resolve(cell["final_score_path"]) / "complete.json")
    metrics = receipt(EVAL / "analysis/metrics_complete.json")
    require((metrics["outputs"],metrics["stage_rows"],metrics["native_fields"]) == (30,60,177), "Final metric panel incomplete")
    receipt(EVAL / "analysis/training_curves_complete.json")
    render = receipt(EVAL / "gallery/render_complete.json")
    require(render["pages"] == 20 and render["columns_per_page"] == 3, "Final gallery incomplete")
    rows = csv_rows(EVAL / "analysis/native_full177_per_output.csv")
    expected_pairs = {(c["cell_id"],stage) for c in cells for stage in ("RAW","CLEAN_FINAL_POST")}
    require({(r["cell_id"],r["stage"]) for r in rows} == expected_pairs and len(rows) == 60, "Final per-output CSV identities differ")
    review_bundle = read(OUT / "visual_review/candidates_actual.json")
    verify(review_bundle["source_reviews"])
    reviews = review_bundle["records"]
    require(len(reviews) == 60 and {(r["cell_id"],r["stage"]) for r in reviews} == expected_pairs, "Actual review identities/denominator differ")
    gallery_images = {str(p.resolve()) for p in (EVAL / "gallery").glob("*/comparison.png")}
    for r in reviews:
        require(r["actually_viewed"] is True and r["blind"] is False and bool(r["reviewer"]), "Review is not identified actual non-blind evidence")
        image = resolve(r["source_image"])
        require(str(image.resolve()) in gallery_images, "Review image is outside frozen candidate gallery")
        verify({str(image):r["source_image_sha256"]})
    failure_files = [p for root in (OUT/"training",OUT/"workers",OUT/"outputs",EVAL/"sample_workers",EVAL/"d0_workers",EVAL/"raw",EVAL/"post")
                     for p in root.rglob("failure.json")]
    require(not failure_files, "Scientific execution failure record requires explicit review: " + repr(failure_files))
    return dict(status="PASS", training_updates=400, training_forwards=3200, training_backwards=3200,
                new_RAW=24, frozen_flip_RAW=4, trained_RAW=20, Euler_generator_forwards=1200,
                new_D0_forwards=24, new_GEO_calls=24, new_noise_arrays=0, reused_original_RAW=6, reused_original_FINAL=6,
                new_full_scores=72, extra_common_C_CF_I_scans=24, extra_common_FF_scans=0,
                backend_own_BASE_pair_scans=24, comparison_outputs=30, RAW_FINAL_rows=60, actual_AI_PROXY_records=60,
                gallery_pages=20, original_parents=3, legal_flip_parents=2, f495_flip="BLOCKED_INPUT_GEOMETRY",
                all24_RAW_saved_before_D0=True, scientific_failures=0, scientific_retries=0,
                actual_training_flip_exposures=728, actual_training_CLEAN_flips=364, actual_training_X_WIDTH_flips=364,
                read_only_acceptance_model_loads=0, read_only_acceptance_forwards=0, read_only_acceptance_scores=0,
                timing_note="Training partially overlapped; training_audit_scope.json defines call-window overlap. Process/load/save windows differ; no isolated speed claim.",
                final_report_numeric_gate="report_numeric_audit_v2.json; first FAIL audit retained with the display-rounding correction evidence")


def source_archive():
    training = read(OUT / "training_registration.json")
    sources = {resolve(p) for p in training["training_code_hashes"]}
    books = list((OUT / "workers").glob("*/code_hashes.json")) + list((EVAL / "d0_workers").glob("*/code_hashes.json"))
    for p in books:
        book = read(p); verify(book); seal(p)
        sources.update(resolve(s) for s in book)
    for identity in (EVAL / "sample_workers").glob("*/identity.json"):
        row = read(identity)
        verify({str(Path(__file__).with_name("candidate_eval.py")):row["code_sha256"],row["weight"]:row["weight_sha256"]})
        seal(identity)
    sources.update(Path(__file__).parent.glob("*.py"))
    sources.add(OUT / "training_preparation/independent_training_audit.py")
    required = [
        "r2/pilots/bench_c_tessellation_v1/evaluate.py", "r2/pilots/bench_c_tessellation_v1/analyze.py",
        "r2/pilots/bench_clean_multipart_seam_v1/post_cpu.py", "r2/pilots/bench_control_transfer_v1/postprocess.py",
        "r2/pilots/seam_slot_c200_v1/postprocess.py", "r2/pilots/seam_slot_c200_v1/generate.py",
        "r2/pilots/seam_abc_v1/evaluate.py", "r2/pilots/seam_abc_v1/eval_common.py",
        "r2/pilots/seam_followup_v1/binding.py", "r2/pilots/seam_hinge_quality_v1/topology.py",
        "r2/pilots/user_condition_finetune_v1/evaluation.py", "r2/pilots/anchor_aware_denoiser_v1/evaluation.py",
        "r2/pilots/user_mix_from_official_v1/evaluation.py", "r2/tools/avmr_pmr_v1/run.py",
        "r2/tools/c_restore_coupling_v1/evaluation.py", "r2/tools/postprocess_avmr_pmr_v1/quality.py",
        "r2/tools/partial_edge_binding_v1/metrics.py", "r2/data_tools/user_conditions_v1/render.py",
        "r2/pilots/chair_multipart_seam_v1/gallery.py", "r2/pilots/regular_patch_control_v1/depth_gallery.py",
        "r2/tools/cref_layer_audit_v1/render.py", "r2/src/meshflow_control/source_winding.py",
        "r2/src/meshflow_control/denoise_model.py"]
    sources.update(ROOT / p for p in required)
    for relative in ("r2/tools/hinge_opposite_v1", "r2/tools/hinge_v2_cpu_v1"):
        sources.update((ROOT / relative).glob("*.py"))
    external = ROOT.parent / "MeshFlow-Control"
    sources.update((external / "src/meshflow_control/evaluation").glob("*.py"))
    sources.add(external / "tools/score_recipe_factorial_conflict.py")
    records = []
    for source in sorted(sources):
        data = source.read_bytes()
        ast.parse(data.decode("utf-8-sig"), filename=str(source))
        if source.is_relative_to(ROOT):
            relative = source.relative_to(ROOT)
        elif source.is_relative_to(external):
            relative = Path("external/MeshFlow-Control") / source.relative_to(external)
        else:
            raise ValueError("Unrecognized source archive root: " + str(source))
        target = OUT / "source_archive" / relative
        put(target, data)
        seal(source); seal(target)
        records.append(dict(source=str(source), archive=target.relative_to(OUT).as_posix(), sha256=sha(source), bytes=len(data)))
    value = dict(scope="Actual training/runtime plus scoring/postprocessing/render mathematics and wrappers; source snapshot, not a self-contained dependency installation",
                 execution_evidence="Registered/worker code hashes verified where recorded; remaining metric dependencies archived read-only at closeout", files=records)
    put(OUT / "source_hashes.json", json_bytes(value)); seal(OUT / "source_hashes.json")
    diff = []
    for source in sorted(Path(__file__).parent.glob("*.py")):
        diff.extend(difflib.unified_diff([], source.read_text("utf-8-sig").splitlines(True), fromfile="/dev/null", tofile=source.relative_to(ROOT).as_posix()))
    put(OUT / "code.diff", "".join(diff).encode("utf-8")); seal(OUT / "code.diff")
    return records


def weight_index():
    reg = read(OUT / "training_registration.json")
    records = [dict(role="frozen_source_full_recovery", path=reg["source_checkpoint"], sha256=reg["source_checkpoint_sha256"], global_step=25500)]
    frozen = read(OUT / "registration.json")
    records.append(dict(role="frozen_source_inference",path=frozen["weight"],sha256=frozen["weight_sha256"],global_step=25500))
    for arm in ("PLAIN200", "FLIP200"):
        state = read(OUT / "training" / arm / "status.json")
        for field in ("checkpoint","inference_export"):
            records.append(dict(role=field,**state[field]))
        rolling = read(OUT / "training" / arm / "checkpoint_latest.json")
        records.append(dict(role="rolling_intermediate_superseded_not_for_evaluation",**rolling))
    for r in records:
        verify({r["path"]:r["sha256"]})
    index = dict(weights_in_archive=False, endpoint_selection="Fixed local200/global25700; no best selection",
                 files=records, note="All weights remain local. Superseded rolling intermediates are listed honestly; this closeout performs no cleanup.")
    put(OUT / "LOCAL_WEIGHTS.json", json_bytes(index)); seal(OUT / "LOCAL_WEIGHTS.json")
    return index


def archive_entries():
    entries = {}
    def add(p):
        if p.is_file():
            require(p.suffix.lower() not in (".pt",".pth",".npz",".npy",".ply",".zip",".pyc"), "Large/binary model asset in review archive")
            entries[p.relative_to(OUT).as_posix()] = p.read_bytes()
            seal(p)
    for p in OUT.iterdir():
        if p.suffix.lower() in (".md",".txt",".html",".diff",".json") and p.name not in ("COMPLETE.json","archive_receipt.json","artifact_manifest.json"):
            add(p)
    for folder in (OUT/"source_archive",OUT/"visual_review",OUT/"geometry_checks",OUT/"training_preparation",OUT/"inputs",OUT/"gallery",EVAL/"gallery"):
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in (".py",".json",".csv",".png",".html",".md",".txt"):
                add(p)
    for folder in (OUT/"analysis",EVAL/"analysis"):
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in (".csv",".png") or p.name in ("metrics_complete.json","training_curves_complete.json","report_summary.json","metric_status_per_output.json","training_first_last50_and_resources.json"):
                add(p)
    for name in ("registration.json","tasks.json","baseline.json"):
        add(EVAL/name)
    for arm in ("PLAIN200","FLIP200"):
        folder=OUT/"training"/arm
        for name in ("status.json","branch_initialization.json","checkpoint_history.jsonl","training_log.jsonl","checkpoint_step200.json","generator_step200.json","rolling_checkpoint_retirement.json"):
            add(folder/name)
    for folder in (OUT/"outputs",EVAL/"raw",EVAL/"post",OUT/"scores",EVAL/"raw_scores",EVAL/"final_common_C",EVAL/"base_pair_caches",EVAL/"cpu"):
        for p in sorted(folder.rglob("*")):
            if p.name in ("complete.json","receipt.json","audit_receipt.json","row.json","score.json","common_C.json","metric_scope.json","calls.jsonl"):
                add(p)
    for base in read(EVAL/"baseline.json"):
        for key in ("score_path","final_score_path"):
            for name in ("row.json","score.json","complete.json"):
                p=resolve(base[key])/name
                entries["reference/"+p.relative_to(ROOT).as_posix()]=p.read_bytes()
                seal(p)
    return entries


def main(verify_only=False):
    if COMPLETE.exists():
        return verify_completed()
    require(not verify_only, "No final completion marker exists yet")
    acceptance=scientific_acceptance()
    sources=source_archive()
    weights=weight_index()
    put(OUT/"closeout_validation.json",json_bytes(acceptance));seal(OUT/"closeout_validation.json")
    entries=archive_entries()
    # The manifest covers local scientific originals including omitted large
    # arrays, and snapshot/packaged documents. It has no self/archive cycle.
    manifest=dict(status="SEALED",files=dict(sorted(SEALED.items())),
        omitted_from_small_archive="Weights, mesh/trajectory NPZ, per-query vectors, large caches and full environment remain local at these verified paths.")
    manifest_data=json_bytes(manifest)
    put(OUT/"artifact_manifest.json",manifest_data)
    entries["artifact_manifest.json"]=manifest_data
    complete=dict(status="COMPLETE",schema="bench_C_edgeflip_full_closeout_v1",**{k:v for k,v in acceptance.items() if k!="status"},
        artifact_manifest_sha256=digest_bytes(manifest_data),source_files=len(sources),
        local_weights_index="LOCAL_WEIGHTS.json",archive_path=str(ARCHIVE),
        archive_scope="Report/RUNBOOK, all per-output CSV, true177 metric records, actual reviews, fixed-view galleries, source/dependency snapshots, code.diff, training step logs/checkpoint receipts and generation ledgers; training per-instance/call logs stay locally indexed; no weights or trajectory/query arrays",
        completed_entrypoint="Re-run this module only verifies sealed hashes and ZIP; it does not overwrite any receipt or regenerate any result")
    complete_data=json_bytes(complete)
    entries["COMPLETE.json"]=complete_data
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    payload=buffer.getvalue()
    require(len(payload)<MAX_ARCHIVE_BYTES,"Prepared review ZIP exceeds10MiB; no oversized ZIP/COMPLETE written: "+str(len(payload)))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(archive.testzip() is None,"Prepared ZIP integrity failure")
    put(ARCHIVE,payload)
    archive_receipt=dict(status="COMPLETE",path=str(ARCHIVE),sha256=digest_bytes(payload),bytes=len(payload),
        entries=len(entries),entry_sha256={k:digest_bytes(v) for k,v in sorted(entries.items())},
        contains_weights=False,contains_full_trajectories=False,contains_query_vectors=False,
        model_calls=0,scoring_calls=0,scientific_recomputation=0)
    put(OUT/"archive_receipt.json",json_bytes(archive_receipt))
    # Completion is last. An interrupted earlier bookkeeping run may only
    # resume when every already-written artifact is byte-identical.
    put(COMPLETE,complete_data)
    result=verify_completed()
    result["status"]="COMPLETE"
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    result=main(args.verify_only)
    require("torch" not in sys.modules,"Closeout unexpectedly imported Torch")
    print(json.dumps(result,ensure_ascii=False))
