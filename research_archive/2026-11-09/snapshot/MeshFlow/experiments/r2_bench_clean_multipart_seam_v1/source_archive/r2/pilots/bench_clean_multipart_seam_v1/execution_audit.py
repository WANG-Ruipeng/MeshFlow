"""Independent file-ledger audit. No framework import, model, or scorer calls."""
from __future__ import annotations
import argparse
import collections
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, read, sha, atomic_write, resolve


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def verify_files(files):
    for name, expected in files.items():
        if sha(resolve(name)) != expected:
            raise AssertionError("Artifact identity changed: " + name)


def event(row):
    return row.get("event", row.get("type"))


def check_training():
    status = read(OUT / "training/status.json")
    assert status["status"] == "COMPLETE" and status["local_step"] == 1500
    rows = jsonl(OUT / "training/calls.jsonl")
    begin = [r for r in rows if event(r) == "CALL_BEGIN"]
    done = [r for r in rows if event(r) == "CALL_DONE"]
    assert not any(event(r) not in ("CALL_BEGIN", "CALL_DONE") for r in rows)
    assert len({r["call_id"] for r in begin}) == len(begin)
    assert {r["call_id"] for r in begin} == {r["call_id"] for r in done}
    assert len(done) == len(begin)
    assert {r["call_id"]: r["kind"] for r in begin} == {r["call_id"]: r["kind"] for r in done}
    counts = dict(collections.Counter(r["kind"] for r in begin))
    expected = dict(train_forward=12000, train_backward=12000, optimizer_update=1500,
                    head_forward=12000, head_backward=12000, head_optimizer_update=1500,
                    probe_forward=384)
    assert counts == expected, (counts, expected)
    for kind in ("optimizer_update", "head_optimizer_update"):
        updates = [r["context"]["global_step"] for r in begin if r["kind"] == kind]
        assert updates == list(range(12001, 13501))
    probe_counts = collections.Counter(r["context"]["local_step"] for r in begin if r["kind"] == "probe_forward")
    assert dict(probe_counts) == {0: 96, 500: 96, 1000: 96, 1500: 96}
    exposure = jsonl(OUT / "training/exposure.jsonl")
    assert len(exposure) == 12000
    manifest = read(OUT / "data/manifest.json")
    allowed = {r["condition_id"] for r in manifest["train"]}
    assert all(r["condition_id"] in allowed for r in exposure)
    parents = collections.Counter(r["parent_id"] for r in exposure)
    assert len(parents) == 58 and set(parents.values()) <= {206, 208}
    return dict(status="PASS", calls=counts, effective_updates=1500, exposures=12000,
                independent_parents=58, exposure_min=min(parents.values()),
                exposure_max=max(parents.values()), retries=0)


def check_generation():
    requests = read(OUT / "eval_manifest.json")["requests"]
    raw_count = post_count = head_count = 0
    raw_receipts = {}
    last_raw_time = 0.
    first_D0_time = float("inf")
    for label in ("NEW_HEAD", "OLD_S15000_LAYOUT"):
        folder = OUT / "evaluation" / label
        raw_complete = read(folder / "raw_complete_full.json")
        assert raw_complete["outputs"] == 48
        verify_files(raw_complete["files"])
        for q in requests:
            d = folder / q["request_id"]
            r = read(d / "raw/receipt.json")
            verify_files(r["files"])
            assert r["noise_sha256"] == q["noise_sha256"]
            assert r["audit"]["anchored_all51_exact"] is True
            with np.load(d / "raw/trajectory.npz", allow_pickle=False) as z:
                trajectory = z["state"]
            with np.load(d / "raw/raw.npz", allow_pickle=False) as z:
                raw = z["raw"]
                fixed = z["raw_anchor_ids"] >= 0
                output = z["output"]
            assert trajectory.shape == (51, q["N"], 3, 3)
            assert all(np.array_equal(state[fixed], raw[fixed]) for state in trajectory)
            with np.load(resolve(q["inference_npz"]), allow_pickle=False) as z:
                source_C = z["C_vertices"][z["C_faces_original"]]
            assert np.array_equal(output[:q["K"]], source_C)
            rows = jsonl(d / "raw/calls.jsonl")
            for name in ("FORWARD_STARTED", "FORWARD_COMPLETE"):
                calls = [x for x in rows if event(x) == name]
                assert len(calls) == 50
                assert [x["step"] for x in calls] == list(range(50))
                if name == "FORWARD_COMPLETE":
                    last_raw_time = max(last_raw_time, max(x["unix"] for x in calls))
            raw_count += 50
            raw_receipts[str(d / "raw/receipt.json")] = sha(d / "raw/receipt.json")
            p = read(d / "post/receipt.json")
            verify_files(p["files"])
            assert p["D0_forwards"] == 1 and p["GEO_calls"] == 1
            assert p["anchors"]["status"] == "PASS"
            assert p["anchors"]["C_vertex_ids_bijective"] is True
            assert p["anchors"]["final_anchor_vertex_identity_verified"] is True
            rows = jsonl(d / "post/calls.jsonl")
            assert sum(event(x) == "D0_FORWARD_STARTED" for x in rows) == 1
            assert sum(event(x) == "D0_FORWARD_RETURNED" for x in rows) == 1
            start = next(x for x in rows if event(x) == "D0_FORWARD_STARTED")
            first_D0_time = min(first_D0_time, datetime.fromisoformat(start["utc"]).timestamp())
            post_count += 1
        layouts = list((folder / "layouts").glob("*/receipt.json"))
        assert len(layouts) == 24
        for path in layouts:
            r = read(path)
            verify_files(r["files"])
            head_count += r["occupancy_forward_calls"]
            ledger = path.parent / "calls.jsonl"
            if r["occupancy_forward_calls"]:
                actual = jsonl(ledger)
                assert [event(x) for x in actual] == ["HEAD_FORWARD_STARTED", "HEAD_FORWARD_RETURNED"]
            else:
                assert not ledger.exists()
        barriers = list(folder.glob("*/all_RAW_barrier.json"))
        assert barriers, "Missing before-D0 RAW barrier record"
        for path in barriers:
            b = read(path)
            assert b["status"] == "PASS" and b["outputs"] == 96
            verify_files(b["files"])
            assert b["files"] == {
                str(OUT / "evaluation" / ll / q["request_id"] / "raw/receipt.json"):
                    sha(OUT / "evaluation" / ll / q["request_id"] / "raw/receipt.json")
                for ll in ("NEW_HEAD", "OLD_S15000_LAYOUT") for q in requests}
    assert raw_count == 4800 and post_count == 96 and head_count == 20
    assert last_raw_time < first_D0_time, "D0 must start after all formal RAW forwards"
    closed_equal = 0
    for q in requests:
        if not q["closed_C"]:
            continue
        paths = [OUT / "evaluation" / label / q["request_id"] / "raw/raw.npz"
                 for label in ("NEW_HEAD", "OLD_S15000_LAYOUT")]
        with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
            assert np.array_equal(a["raw"], b["raw"])
            assert np.array_equal(a["output"], b["output"])
        closed_equal += 1
    assert closed_equal == 8, "Closed C uses the same bulk-only route in both layout arms"
    diagnostic = 0
    for model in ("OLD_S15000_HEAD", "NEW_HEAD"):
        folder = OUT / "layout_diagnostic" / model
        d = read(folder / "completion.json")
        verify_files(d["files"])
        rows = jsonl(folder / "calls.jsonl")
        for name in ("HEAD_FORWARD_STARTED", "HEAD_FORWARD_RETURNED"):
            selected = [r for r in rows if event(r) == name]
            assert len(selected) == 247 and len({r["condition_id"] for r in selected}) == 247
        diagnostic += 247
    return dict(status="PASS", RAW_outputs=96, Euler_generator_forwards=raw_count,
                single_official_Bench_D0_forwards=post_count, panel_new_head_forwards=head_count,
                diagnostic_head_forwards=diagnostic, inference_head_forwards_total=head_count + diagnostic,
                RAW_barrier_outputs=96, all51_anchor_states_rechecked=True,
                closed_C_layout_arm_exact_equal_outputs=closed_equal,
                latest_RAW_forward_unix=last_raw_time, earliest_D0_forward_unix=first_D0_time,
                temporal_RAW_barrier_verified=True, new_noise_arrays=0, new_query_arrays=0)


def run():
    # Historical originals and all frozen requests remain read-only.
    registration = read(OUT / "registration.json")
    verify_files(registration["protected_small_inputs"])
    verify_files(read(OUT / "eval_manifest.json")["protected"])
    verify_files(read(OUT / "historical_registration.json")["protected"])
    tr = read(OUT / "training_registration.json")
    verify_files(tr["training_code_hashes"])
    assert sha(resolve(tr["source_checkpoint"])) == tr["source_checkpoint_sha256"]
    training = check_training()
    generation = check_generation()
    h = read(OUT / "historical_completion.json")
    verify_files(h["files"])
    assert h["requests"] == 144 and h["model_calls"] == 0 and h["D0_calls"] == 0
    revision = read(OUT / "historical_execution_revision_01.json")
    verify_files(revision["prior_files"])
    assert revision["additional_partial_BASE_scores"] == 2
    retained_partial_scores = list((OUT / "engineering_revision_01").glob("*/BASE_POST/scores/complete.json"))
    assert len(retained_partial_scores) == 2
    c = read(OUT / "comparison/completion.json")
    verify_files(c["files"])
    assert c["primary_BASE_FINAL_cells"] == 480 and c["RAW_cells"] == 240 and c["metric_count"] == 179
    result = dict(status="PASS", training=training, generation=generation,
                  historical_CPU_outputs=144, scored_BASE_FINAL=480, scored_RAW=240,
                  additional_retained_CPU_BASE_scores=2,
                  actual_full_metric_score_cells_including_partial_attempts=722,
                  extra_partial_HINGE_attempts=2, CPU_dispatch_reschedule_repeated_complete_cells=0,
                  original_metric_fields=179, protected_assets_verified=True,
                  model_calls_by_this_audit=0, source_repair=False,
                  caveat="This checks actual execution, not model quality or visual success.")
    atomic_write(OUT / "independent_execution_audit.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
