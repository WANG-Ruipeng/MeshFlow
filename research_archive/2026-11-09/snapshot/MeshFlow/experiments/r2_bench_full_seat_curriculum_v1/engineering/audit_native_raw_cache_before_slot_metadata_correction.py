"""Read-only array/receipt audit of the two saved Native RAW waves.

Writes a new audit receipt only. No framework/model imports, geometry scoring,
generation, image rendering, or mutation of cached assets.
"""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_full_seat_curriculum_v1"
LABELS = ("COMMON_S12000", "NATIVE_S6000")


def resolve(p):
    s = str(p).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(p):
    return json.loads(resolve(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with resolve(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()


def arrays(p):
    with np.load(resolve(p), allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def run():
    manifest = read(OUT / "eval_manifest.json")
    reg = read(OUT / "registration.json")
    assert sha(OUT / "eval_manifest.json") == reg["eval_manifest_sha256"]
    qs = manifest["requests"]
    assert len(qs) == 24 and len({q["request_id"] for q in qs}) == 24
    checked = {}

    def verify(items):
        for p,h in items.items():
            if p not in checked:
                checked[p] = sha(p)
            assert checked[p] == h, p

    verify(manifest["protected"])
    groups, rows = {}, []
    for label in LABELS:
        folder = OUT / "evaluation" / label
        done = read(folder / "raw_complete_full.json")
        registration = read(OUT / "evaluation/inference_registrations" / (label + "__RAW.json"))
        assert done["status"] == "COMPLETE" and done["outputs"] == done["total_new_rollouts"] == 24
        assert done["total_new_generator_forwards"] == 1200 and done["total_new_head_forwards"] == 0
        assert done["new_forwards_this_attempt"] == 1200 and done["new_rollouts_this_attempt"] == 24
        assert done["reused_rollouts"] == 0 and registration["solver"] == "Euler50"
        assert registration["maximum_head_forwards"] == 0 and registration["representation"] == "PLAIN"
        verify(done["files"]); verify(registration["protected"]); verify(registration["code"])
        attempts = sorted(folder.glob("sampling_attempt_*"))
        assert len(attempts) == 1 and not (attempts[0] / "failure.json").exists()
        events_total = collections.Counter()
        seen_layouts = set()
        max_alloc, max_res = 0, 0
        for q in qs:
            dest = folder / q["request_id"] / "raw"
            receipt = read(dest / "receipt.json")
            assert receipt["status"] == "COMPLETE" and receipt["execution"] == "NEW"
            assert receipt["label"] == label and receipt["request_id"] == q["request_id"] and receipt["condition_id"] == q["condition_id"]
            assert receipt["representation"] == "PLAIN" and receipt["source_labels_read"] is False
            assert receipt["generator_forwards"] == receipt["new_generator_forwards"] == 50
            assert receipt["D0_forwards"] == receipt["new_D0_forwards"] == receipt["head_forwards"] == 0
            assert receipt["noise_sha256"] == q["noise_sha256"] == sha(q["noise_npz"])
            assert receipt["raw_sha256"] == sha(dest / "raw.npz")
            verify(receipt["files"])
            attempt = read(dest / "attempt.json")
            assert attempt["request"] == q and attempt["forward_budget"] == 50
            calls = [json.loads(s) for s in (dest / "calls.jsonl").read_text().splitlines() if s.strip()]
            assert len(calls) == 100
            assert [(c["event"],c["step"]) for c in calls] == [(event,step) for step in range(50) for event in ("FORWARD_STARTED","FORWARD_COMPLETE")]
            events_total.update(c["event"] for c in calls)
            core, raw, traj, noise = arrays(q["inference_npz"]), arrays(dest / "raw.npz"), arrays(dest / "trajectory.npz")["state"], arrays(q["noise_npz"])["z"]
            K,N = q["K"],q["N"]
            original = core["C_vertices"][core["C_faces_original"]]
            canonical = core["C_vertices"][core["C_faces_model"]]
            assert int(raw["known_faces"]) == K and int(raw["input_faces"]) == N
            assert raw["raw"].shape == raw["output"].shape == (N,3,3)
            assert traj.shape == (51,N,3,3) and traj.dtype == np.float32 and np.isfinite(traj).all()
            assert same(traj[0,K:], noise) and same(traj[-1],raw["raw"])
            assert all(same(state[:K],canonical) for state in traj)
            assert same(raw["raw"][:K],canonical) and same(raw["output"][:K],original)
            assert same(raw["vertices"][raw["faces"]],raw["output"])
            assert same(raw["C_vertices"],core["C_vertices"])
            assert same(raw["C_faces_original"],core["C_faces_original"])
            assert same(raw["C_faces_model"],core["C_faces_model"])
            assert not raw["raw_seam_slot_mask"].any() and not raw["retained_seam_mask"].any() and not raw["occupied"].any()
            assert (raw["raw_anchor_ids"][K:] == -1).all() and (raw["output_anchor_ids"][K:] == -1).all()
            assert (raw["raw_slot_edge_id"] == -1).all()
            for a in raw.values():
                if a.dtype.kind == "f":
                    assert np.isfinite(a).all()
            assert sha(dest / "source_condition.npz") == q["source_npz_sha256"]
            layout_path = resolve(receipt["layout_receipt"])
            layout_receipt = read(layout_path)
            verify(layout_receipt["files"])
            assert layout_receipt["input_sha256"] == q["inference_npz_sha256"] == sha(q["inference_npz"])
            assert layout_receipt["occupancy_forward_calls"] == layout_receipt["active"] == 0
            assert layout_receipt["head"]["kind"] == "NONE_NATIVE_ALL_FREE"
            lay = arrays(layout_path.parent / "layout.npz")
            assert len(lay["seam_slots"]) == 0 and same(lay["bulk_slots"],np.arange(K,N,dtype=np.int64))
            assert (lay["anchor_ids"][K:] == -1).all() and lay["free_corner_mask"][K:].all()
            assert not lay["free_corner_mask"][:K].any() and not lay["occupied"].any()
            seen_layouts.add(q["condition_id"])
            max_alloc = max(max_alloc, receipt["audit"]["peak_allocated_bytes"])
            max_res = max(max_res, receipt["audit"]["peak_reserved_bytes"])
            rows.append(dict(label=label,request_id=q["request_id"],condition_id=q["condition_id"],parent_id=q["parent_id"],N=N,K=K,
                noise_sha256=receipt["noise_sha256"],input_sha256=layout_receipt["input_sha256"],source_npz_sha256=q["source_npz_sha256"],
                receipt_sha256=sha(dest / "receipt.json"),raw_sha256=receipt["raw_sha256"],trajectory_sha256=sha(dest / "trajectory.npz"),
                generator_forwards=50,head_forwards=0,D0_forwards=0,finite=True,C_all51_bitwise=True,C_source_winding_bitwise=True,
                free_anchor_count=0,seam_slots=0,initial_noise_bitwise=True))
        assert len(seen_layouts) == 12 and events_total == {"FORWARD_STARTED":1200,"FORWARD_COMPLETE":1200}
        resource = read(attempts[0] / "resource_observation.json")
        groups[label] = dict(outputs=24,unique_layouts=12,event_counts=dict(events_total),new_head_forwards=0,new_D0_forwards=0,
            completion_sha256=sha(folder / "raw_complete_full.json"),inference_registration_sha256=sha(OUT / "evaluation/inference_registrations" / (label + "__RAW.json")),
            checkpoint_sha256=done["checkpoint_sha256"],seconds_concurrent=done["seconds"],max_rollout_allocated_bytes=max_alloc,
            max_rollout_reserved_bytes=max_res,device_observer_summary=resource["summary"],attempts=1,failure_records=0)
    by = {(r["label"],r["request_id"]):r for r in rows}
    for q in qs:
        a,b = (by[label,q["request_id"]] for label in LABELS)
        for key in ("request_id","condition_id","parent_id","N","K","noise_sha256","input_sha256","source_npz_sha256"):
            assert a[key] == b[key]
    assert "torch" not in sys.modules
    return dict(schema="NATIVE_COMMON_PAIRED_RAW_CACHE_AUDIT_V1",status="PASS",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        paired_requests=24,observed_outputs=48,labels=groups,rows=rows,source_code_sha256=sha(__file__),
        model_calls=0,head_calls=0,D0_calls=0,geometry_scoring_calls=0,rendered_images=0,
        note="Read-only cache audit. Actual recorded calls are in labels; this audit makes zero model calls. C/source winding preserved, no Native free anchors. Quality/semantic success is not inferred.")


if __name__ == "__main__":
    result = run()
    dest = OUT / "engineering/native_common_raw_cache_audit.json"
    with dest.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows",)},ensure_ascii=False))
