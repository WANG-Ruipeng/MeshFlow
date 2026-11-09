"""Read-only final evidence closure. Default only prints preparation status.

No experiment modules, NumPy, torch, model, scorer or geometry code are imported.
Run after every registered stage and actual image review is finished:
  python -B -m r2.pilots.seam_pool47_couple_v1.final_audit --run --verify-large
Large assets are hashed once per unique path. Alternatively supply a fresh hash
receipt: {"files": [{"path": ..., "sha256": ..., "bytes": ..., "mtime_ns": ...,
"verified_unix": ...}]}. Registered old hashes alone are not final verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_seam_pool47_couple_v1"
ARMS = ("PLAIN3000", "COUPLE3000")
LABELS = ("B_LONG_LAYOUT", *ARMS)
STAGES = ("RAW", "POST", "BOUND015")
REQUIRED = ["protocol.json", "training_registration.json", "shared_schedule.json",
    "shared_data_manifest.json", "training_dispatch_go.json", "generation_registration.json",
    "generation_dispatch_go.json", "evaluation/score_registration.json",
    "evaluation/score_completion.json", "evaluation/summary_completion.json",
    "endpoint_binding/registration.json", "endpoint_binding/completion.json",
    "results/completion.json", "results/per_output.csv", "visual_review_endpoint.csv",
    "gallery/training/completion.json", "gallery/endpoint_binding/completion.json",
    *[f"training/{a}/status.json" for a in ARMS]]


def resolve(value):
    text = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    p = Path(text)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")


class Audit:
    def __init__(self, verify_large=False, large_receipt=None):
        self.expected = {}
        self.scopes = {}
        self.checks = {}
        self.verify_large = verify_large
        self.large = {}
        self.latest_work = 0.
        if large_receipt:
            for r in read(large_receipt)["files"]:
                self.large[str(resolve(r["path"]))] = r

    def hashes(self, mapping, scope):
        for path, sha in mapping.items():
            p = str(resolve(path))
            require(re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "Invalid hash " + p)
            require(p not in self.expected or self.expected[p] == sha, "Conflicting evidence hashes " + p)
            self.expected[p] = sha
            self.scopes.setdefault(p, set()).add(scope)

    def receipt(self, path, scope):
        p = Path(path)
        value = read(p)
        self.hashes(value.get("files", {}), scope)
        return value

    def training(self):
        reg = read(OUT / "training_registration.json")
        schedule = read(OUT / "shared_schedule.json")
        manifest = read(OUT / "shared_data_manifest.json")
        require(reg["updates_per_arm"] == 3000 and reg["source_global_step"] == 13000, "Training contract")
        require(reg["arms"] == list(ARMS) and reg["batch"] == 8 and reg["microbatch"] == 1, "Arms/batch")
        require(len(reg["records"]) == 47 and all(r["split"] == "TRAIN" for r in reg["records"]), "TRAIN47 isolation")
        require(manifest["status"] == "COMPLETE" and manifest["logical_OT_calls"] == 24000, "Shared cache incomplete")
        require(len(manifest["steps"]) == len(schedule["steps"]) == 3000, "Shared stream steps")
        go = read(OUT / "training_dispatch_go.json")
        require(go["authorized"] is True, "Training release")
        self.hashes({str(OUT / "training_registration.json"): go["training_registration_sha256"],
            str(OUT / "shared_data_manifest.json"): go["shared_data_manifest_sha256"]}, "training release")
        initial, signatures, budgets = {}, {}, {}
        for arm in ARMS:
            folder = OUT / "training" / arm
            status = read(folder / "status.json")
            require((status["status"], status["local_step"], status["global_step"]) == ("COMPLETE", 3000, 16000), "Unfinished " + arm)
            began, done, pending, ids, updates = Counter(), Counter(), {}, set(), []
            for r in jsonl(folder / "calls.jsonl"):
                k, ident = r["kind"], r["call_id"]
                require(r["arm"] == arm, "Mixed arm ledger")
                if r["event"] == "CALL_BEGIN":
                    require(ident not in ids, "Reused call ID")
                    ids.add(ident); pending[ident] = k; began[k] += 1
                    if k == "optimizer_update": updates.append(r["context"]["global_step"])
                else:
                    require(pending.pop(ident, None) == k and r["event"] == "CALL_DONE", "Failed/unpaired call")
                    done[k] += 1
                self.latest_work = max(self.latest_work, float(r["time_unix"]))
            expected = {"train_forward": 24000, "train_backward": 24000, "optimizer_update": 3000}
            require(dict(began) == dict(done) == expected and not pending, "Exact training call budget " + arm)
            require(updates == list(range(13001, 16001)), "Update gaps/replay")
            starts = [read(p) for p in (folder / "attempts").glob("*/initialization.json") if read(p)["global_step"] == 13000]
            require(len(starts) == 1, "Unique source initialization")
            initial[arm] = starts[0]
            for field, source in (("generator_state_sha256", "source_generator_sha256"),
                ("optimizer_state_sha256", "source_optimizer_sha256"), ("rng_sha256", "source_rng_sha256")):
                require(initial[arm][field] == reg[source], "Restoration differs: " + field)
            signature = hashlib.sha256(); counts = Counter(); n = 0
            for n, r in enumerate(jsonl(folder / "exposure.jsonl"), 1):
                step, micro = (n - 1) // 8 + 1, (n - 1) % 8
                require(step <= 3000, "Exposure overflow")
                frozen = manifest["steps"][step - 1]["samples"][micro]
                scheduled = schedule["steps"][step - 1]["rows"][micro]
                require(r["local_step"] == step and r["microbatch"] == micro, "Exposure order")
                for key in ("uid", "alpha", "path"):
                    require(r[key] == frozen[key], "Exposure/cache mismatch: " + key)
                require(r["uid"] == scheduled["uid"] and r["alpha"] == scheduled["alpha"], "Exposure/schedule mismatch")
                stable = {k: v for k, v in r.items() if k not in ("time_unix", "pid")}
                signature.update(json.dumps(stable, sort_keys=True).encode()); signature.update(b"\n")
                counts[r["uid"]] += 1
            require(n == 24000 and dict(counts) == schedule["parent_exposure"], "Parent exposure budget")
            signatures[arm] = signature.hexdigest(); budgets[arm] = dict(done)
            for key in ("checkpoint", "inference_export"):
                rp = status[key]
                self.hashes({rp["path"]: rp["sha256"]}, arm + " final weight")
            logs = list(jsonl(folder / "training_log.jsonl"))
            require([r["local_step"] for r in logs] == list(range(1, 3001)), "Training update logs")
            require(all(r.get("occupancy_BCE") is None and r["head_status"] == "FROZEN_NOT_COMPUTED" for r in logs), "Unexpected head loss")
        for field in ("generator_state_sha256", "optimizer_state_sha256", "rng_sha256", "generator_hyperparameters", "EMA"):
            require(initial[ARMS[0]][field] == initial[ARMS[1]][field], "Different initial " + field)
        require(len(set(signatures.values())) == 1, "Different actual input streams")
        self.checks["training"] = dict(status="PASS", budgets=budgets, shared_exposure_signature=signatures,
            source_identity="Registered B global13000 live generator/AdamW/RNG", cache_limit="Loader checks each sample hash during training; this closure compares all recorded exposures without re-running OT or re-reading every cache array")

    def generation(self):
        reg = read(OUT / "generation_registration.json")
        requests = reg["requests"]
        require(reg["arms"] == list(ARMS) and len(requests) == 28, "Generation panel")
        require(len({q["request_id"] for q in requests}) == 28, "Duplicate requests")
        self.requests = {q["request_id"]: q for q in requests}
        expected_raw = {}; raw_saved = []; d0_start = []; forward_count = 0; barrier_maps = []
        expected_call_paths = {str((OUT / 'evaluation' / a / q / s / 'calls.jsonl').resolve())
            for a in ARMS for q in self.requests for s in ('raw', 'post')}
        actual_call_paths = {str(p.resolve()) for s in ('raw', 'post')
            for p in (OUT / 'evaluation').glob('*/*/' + s + '/calls.jsonl')}
        require(actual_call_paths == expected_call_paths, 'Unregistered or missing generation call logs')
        for arm in ARMS:
            folder = OUT / "evaluation" / arm
            raw_done = self.receipt(folder / "raw_complete.json", "RAW aggregate")
            post_done = self.receipt(folder / "post_complete.json", "POST aggregate")
            require(raw_done["outputs"] == post_done["outputs"] == 28, "Output count")
            require(raw_done["generator_forwards"] == 1400 and post_done["D0_forwards"] == 28, "Aggregate call count")
            saved = [r for r in jsonl(folder / "ledger.jsonl") if r["event"] == "RAW_SAVED"]
            require(Counter(r["request_id"] for r in saved) == Counter(self.requests.keys()), "RAW save identities")
            raw_saved.extend(float(r["unix"]) for r in saved)
            for q in requests:
                base = folder / q["request_id"]
                rr = self.receipt(base / "raw/receipt.json", "RAW files")
                pr = self.receipt(base / "post/receipt.json", "POST files")
                for r in (rr, pr):
                    require(r["arm"] == arm and r["request_id"] == q["request_id"], "Generated identity")
                require(rr["generator_forwards"] == 50 and rr["head_forwards"] == 0, "RAW call declaration")
                require(pr["D0_forwards"] == 1 and pr["GEO_calls"] == 1, "POST call declaration")
                self.hashes({q["noise_npz"]: rr["noise_sha256"]}, "Fixed generation noise")
                require(rr["checkpoint"]["sha256"] == reg["models"][arm]["sha256"], "Final generator identity")
                require(rr["layout_path"] == reg["layouts"][q["condition_id"]]["path"], "C-only layout identity")
                calls = list(jsonl(base / "raw/calls.jsonl"))
                require(len(calls) == 100, "RAW exact 50 started/returned")
                for step in range(50):
                    a, b = calls[2 * step:2 * step + 2]
                    require(a["event"] == "FORWARD_STARTED" and b["event"] == "FORWARD_RETURNED" and a["euler_step"] == b["euler_step"] == step, "Euler call order")
                forward_count += 50
                dc = list(jsonl(base / "post/calls.jsonl"))
                require([r["event"] for r in dc] == ["D0_FORWARD_STARTED", "D0_FORWARD_RETURNED"], "D0 exact single-call ledger")
                d0_start.append(float(dc[0]["unix"])); self.latest_work = max(self.latest_work, float(dc[1]["unix"]))
                expected_raw[str((base / "raw/receipt.json").resolve())] = digest(base / "raw/receipt.json")
            barriers = list(folder.glob("post_attempt_*/all_RAW_barrier.json"))
            require(barriers, "No actual all-RAW barrier receipt")
            for path in barriers:
                barrier = read(path)
                require(barrier["status"] == "PASS" and barrier["RAW"] == 56 and len(barrier["files"]) == 56, "Barrier count")
                self.hashes(barrier["files"], "D0 all56RAW barrier")
                barrier_maps.append({str(resolve(p)): h for p, h in barrier["files"].items()})
        require(len(expected_raw) == len(d0_start) == 56 and forward_count == 2800, "Generation total")
        require(all(m == expected_raw for m in barrier_maps), "Barrier is not the exact56 final RAW receipts")
        require(max(raw_saved) < min(d0_start), "D0 began before every RAW was saved")
        self.checks["generation"] = dict(status="PASS", RAW=56, generator_forwards=2800, D0_forwards=56,
            last_RAW_saved=max(raw_saved), first_D0_started=min(d0_start), all_RAW_before_any_D0=True)

    def scores_and_binding(self):
        score = read(OUT / "evaluation/score_completion.json")
        summary = self.receipt(OUT / "evaluation/summary_completion.json", "112 score tables")
        require(score["status"] == "COMPLETE" and score["cells"] == 112 and len(score["receipts"]) == 112, "112 score completion")
        require(summary["rows"] == 112 and summary["numeric_field_count"] == 124, "124 fields")
        self.numeric = summary["numeric_fields"]
        require(len(self.numeric) == len(set(self.numeric)) == 124, "Metric dictionary uniqueness")
        identities = set()
        for rr in score["receipts"]:
            self.hashes({rr["path"]: rr["sha256"]}, "Score receipt")
            r = self.receipt(resolve(rr["path"]), "Score evidence")
            identities.add((r["label"], r["request_id"], r["stage"]))
        require(identities == {(a, q, s) for a in ARMS for q in self.requests for s in ("RAW", "POST")}, "Score identity matrix")
        require(len(list((OUT / 'evaluation/scores').glob('*/*/*/score.json'))) == 112, 'Extra or missing score artifacts')
        bound = self.receipt(OUT / "endpoint_binding/completion.json", "Binding tables")
        require(bound["status"] == "COMPLETE" and bound["cells"] == 56 and bound["model_calls"] == 0, "Binding completion")
        bind_reg = read(OUT / "endpoint_binding/registration.json")
        require(bind_reg["deltas"] == [.015] and bind_reg["arms"] == list(ARMS), "Binding threshold/arms")
        cells = list((OUT / "endpoint_binding/cells").glob("*/*/complete.json"))
        require(len(cells) == 56, "Exactly56 endpoint binding cells")
        seen = set()
        for p in cells:
            r = self.receipt(p, "Binding cell")
            require(r["delta"] == .015, "Extra binding cap")
            seen.add((r["arm"], r["request_id"]))
        require(seen == {(a, q) for a in ARMS for q in self.requests}, "Binding identity matrix")
        result = self.receipt(OUT / "results/completion.json", "Final result tables")
        require(result["rows"] == 252 and result["numeric_fields"] == 124, "Final result dimensions")
        self.hashes(result["source_scores"], "Reused baseline/new evidence")
        rows = table(OUT / "results/per_output.csv")
        require(len(rows) == 252, "252 rows")
        expected = {(q, a, s) for q in self.requests for a in LABELS for s in STAGES}
        require(Counter((r["request_id"], r["label"], r["stage"]) for r in rows) == Counter({k: 1 for k in expected}), "Unique final metric matrix")
        for r in rows:
            q = self.requests[r["request_id"]]
            require(r["split"] == q["split"] and r["parent_id"] == q["parent_id"] and int(r["noise_index"]) == q["noise_index"], "Final metadata identity")
            require(set(self.numeric).issubset(r), "Missing metric columns")
            for key in self.numeric:
                value = r[key]
                if value not in ("", "None", "null", "True", "False", "true", "false"):
                    require(math.isfinite(float(value)), "Nonfinite metric without explicit NA: " + key)
        self.checks["scoring"] = dict(status="PASS", newly_scored_RAW_POST=112, endpoint_binding_cells=56,
            final_rows=252, numeric_fields=124, note="Baseline RAW/POST and baseline1.5% binding reused; no rescoring by this audit")

    def visuals(self):
        pages = {}
        for kind, count in (("training", 56), ("endpoint_binding", 28)):
            completion = read(OUT / "gallery" / kind / "completion.json")
            require(len(completion["images"]) == count, "Gallery image count")
            for image in completion["images"]:
                p = str(resolve(image["path"]))
                require(p not in pages, "Duplicate gallery path")
                pages[p] = image
                self.hashes({p: image["image_sha256"]}, "Rendered actual review page")
        rows = table(OUT / "visual_review_endpoint.csv")
        required = {"request_id", "label", "stage", "page_path", "page_sha256", "reviewer", "actually_viewed",
            "body_complete", "C_adopted_as_seat", "obvious_fragments_or_folded_sheets", "visible_cracks", "note"}
        require(len(rows) == 252 and all(required.issubset(r) for r in rows), "Canonical252 visual records")
        expected = {(q, a, s) for q in self.requests for a in LABELS for s in STAGES}
        require(Counter((r["request_id"], r["label"], r["stage"]) for r in rows) == Counter({k: 1 for k in expected}), "Unique252 visual identity matrix")
        used = Counter()
        for r in rows:
            p = str(resolve(r["page_path"])); require(p in pages, "Unregistered reviewed image")
            image = pages[p]
            require(image["request_id"] == r["request_id"] and image["stage"] == r["stage"], "Review/page identity")
            require(r["page_sha256"] == image["image_sha256"], "Reviewed image hash")
            require(truth(r["actually_viewed"]) and r["reviewer"].strip() and r["note"].strip(), "Actual reviewer and note required")
            require(all(r[k].strip() for k in ("body_complete", "C_adopted_as_seat", "obvious_fragments_or_folded_sheets", "visible_cracks")), "Empty visual observation")
            used[p] += 1
        require(len(pages) == len(used) == 84 and set(used.values()) == {3}, "84 pages with3 actual model reviews each")
        self.checks["visuals"] = dict(status="PASS", pages=84, rows=252,
            limit="Checks recorded actual review declarations and PNG identity; cannot prove subjective ratings correct. No metric values substituted for visual review.")

    def collect_protected(self):
        specs = {"protocol.json": ("protected_files",), "training_registration.json": ("protected_hashes", "cache_code_hashes"),
            "training_dispatch_go.json": ("code_hashes",), "generation_registration.json": ("protected_files",),
            "generation_dispatch_go.json": ("code_hashes",), "evaluation/score_registration.json": ("protected", "source_hashes"),
            "cpu_binding/registration.json": ("protected", "code"), "endpoint_binding/registration.json": ("protected", "code")}
        for rel, fields in specs.items():
            value = read(OUT / rel)
            for field in fields:
                self.hashes(value[field], rel + ":" + field)

    def verify_hashes(self):
        verified, deferred, failed = [], [], []; bytes_read = 0
        for path, expected in sorted(self.expected.items()):
            p = Path(path)
            if not p.exists(): failed.append(dict(path=path, reason="MISSING")); continue
            stat = p.stat(); mode = "DIRECT_SHA256"
            if stat.st_size > 64 << 20 and not self.verify_large:
                prior = self.large.get(path)
                if not prior or any(prior.get(k) != v for k, v in (("sha256", expected), ("bytes", stat.st_size), ("mtime_ns", stat.st_mtime_ns))) or float(prior.get("verified_unix", 0)) < self.latest_work:
                    deferred.append(dict(path=path, expected_sha256=expected, bytes=stat.st_size, reason="Requires --verify-large or matching post-work large-hash receipt")); continue
                actual = prior["sha256"]; mode = "FRESH_HASH_RECEIPT_PLUS_EXACT_STAT"
            else:
                actual = digest(p); bytes_read += stat.st_size
            if actual != expected: failed.append(dict(path=path, expected=expected, actual=actual)); continue
            verified.append(dict(path=path, sha256=actual, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                verified_unix=time.time(), verification=mode, scopes=sorted(self.scopes[path])))
        self.checks["hashes"] = dict(status="FAIL" if failed else "INCOMPLETE" if deferred else "PASS",
            unique_paths=len(self.expected), verified=len(verified), failed=failed, deferred=deferred,
            bytes_read=bytes_read, duplicate_hash_scans=0, files=verified)
        return not failed and not deferred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Verify completed evidence; otherwise only report readiness")
    p.add_argument("--verify-large", action="store_true", help="Hash each unique >64MiB asset once")
    p.add_argument("--large-hash-receipt")
    p.add_argument("--output", default=str(OUT / "final_audit.json"))
    args = p.parse_args()
    missing = [x for x in REQUIRED if not (OUT / x).exists()]
    if not args.run:
        print(json.dumps(dict(status="PREPARED_NOT_EXECUTED", missing=missing,
            intended=dict(updates_per_arm=3000, exposures_per_arm=24000, RAW=56, D0=56, new_RAW_POST_scores=112,
                endpoint_BIND015=56, result_rows=252, numeric_fields=124, reviewed_pages=84, reviewed_rows=252),
            model_calls=0, scorer_calls=0), indent=2)); return
    destination = resolve(args.output)
    require(destination.is_relative_to(OUT.resolve()), "Audit output must stay within the new experiment")
    require(not destination.exists(), "Refuse audit overwrite; select a new --output")
    result = dict(status="INCOMPLETE", model_calls=0, scorer_calls=0, geometry_calls=0,
        new_OT_calls=0, source_sha256=digest(__file__), missing=missing, checks={})
    if not missing:
        audit = Audit(args.verify_large, args.large_hash_receipt)
        try:
            audit.training(); audit.generation(); audit.scores_and_binding(); audit.visuals()
            audit.collect_protected()
            ok = audit.verify_hashes()
            result["status"] = "PASS" if ok else "ISSUES" if audit.checks["hashes"]["failed"] else "INCOMPLETE"
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            result.update(status="ISSUES", error=repr(error))
        result["checks"] = audit.checks
    result["interpretation"] = "Evidence/budget closure only. No scientific superiority, default promotion or new action is inferred."
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__":
    main()
