"""CPU-only final authentication of all 72 saved RAW products and call ledgers."""
from __future__ import annotations
import collections
import datetime
import json
import sys
from pathlib import Path
import numpy as np
from .audit_native_raw_cache import ROOT, OUT, read, sha, arrays, same, resolve

LABELS = ("COMMON_S12000", "NATIVE_S6000", "SEAM_S3000")


def run():
    manifest = read(OUT / "eval_manifest.json")
    reg = read(OUT / "registration.json")
    assert sha(OUT / "eval_manifest.json") == reg["eval_manifest_sha256"]
    qs = manifest["requests"]
    assert len(qs) == len({q["request_id"] for q in qs}) == 24
    checked, groups, rows = {}, {}, []

    def verify(items):
        for p,h in items.items():
            if p not in checked:
                checked[p] = sha(p)
            assert checked[p] == h, p

    verify(manifest["protected"])
    for label in LABELS:
        is_seam = label == LABELS[-1]
        folder = OUT / "evaluation" / label
        done = read(folder / "raw_complete_full.json")
        registration = read(OUT / "evaluation/inference_registrations" / (label + "__RAW.json"))
        mode = "SEAM" if is_seam else "PLAIN"
        assert done["status"] == "COMPLETE" and done["outputs"] == done["total_new_rollouts"] == 24
        assert done["total_new_generator_forwards"] == done["new_forwards_this_attempt"] == 1200
        assert done["total_new_head_forwards"] == done["new_head_forwards_this_attempt"] == (12 if is_seam else 0)
        assert done["new_rollouts_this_attempt"] == 24 and done["reused_rollouts"] == 0
        assert registration["solver"] == "Euler50" and registration["representation"] == mode
        verify(done["files"]); verify(registration["protected"]); verify(registration["code"])
        attempts = sorted(folder.glob("sampling_attempt_*"))
        assert len(attempts) == 1 and not (attempts[0] / "failure.json").exists()
        events_total, head_events = collections.Counter(), collections.Counter()
        layouts, layout_rows = {}, []
        for q in qs:
            dest = folder / q["request_id"] / "raw"
            receipt = read(dest / "receipt.json")
            verify(receipt["files"])
            assert receipt["status"] == "COMPLETE" and receipt["execution"] == "NEW"
            assert receipt["label"] == label and receipt["request_id"] == q["request_id"] and receipt["condition_id"] == q["condition_id"]
            assert receipt["representation"] == mode and receipt["source_labels_read"] is False
            assert receipt["generator_forwards"] == receipt["new_generator_forwards"] == 50
            assert receipt["D0_forwards"] == receipt["new_D0_forwards"] == receipt["head_forwards"] == 0
            assert receipt["noise_sha256"] == q["noise_sha256"] == sha(q["noise_npz"])
            assert receipt["raw_sha256"] == sha(dest / "raw.npz")
            assert read(dest / "attempt.json")["request"] == q
            calls = [json.loads(s) for s in (dest / "calls.jsonl").read_text().splitlines() if s.strip()]
            assert [(c["event"],c["step"]) for c in calls] == [(event,step) for step in range(50) for event in ("FORWARD_STARTED","FORWARD_COMPLETE")]
            events_total.update(c["event"] for c in calls)
            cid = q["condition_id"]
            lp = resolve(receipt["layout_receipt"])
            if cid not in layouts:
                lr, lay = read(lp), arrays(lp.parent / "layout.npz")
                verify(lr["files"])
                assert lr["input_sha256"] == q["inference_npz_sha256"] == sha(q["inference_npz"])
                assert lr["source_labels_read"] is False and lr["occupancy_forward_calls"] == int(is_seam)
                if is_seam:
                    assert lr["head"]["kind"] == "FROZEN_OLD_S15000_HEAD" and lr["head"]["parameters"] == 833
                    assert lr["head"]["state_sha256"] == reg["head_state_sha256"]
                    assert lr["threshold_logit"] == 0 and lr["head"]["input"] == "C_ONLY_24_FEATURES"
                    calls_head = [json.loads(s) for s in (lp.parent / "calls.jsonl").read_text().splitlines() if s.strip()]
                    assert [c["event"] for c in calls_head] == ["HEAD_FORWARD_STARTED","HEAD_FORWARD_RETURNED"]
                    head_events.update(c["event"] for c in calls_head)
                    assert np.array_equal(lay["occupied"],lay["occupancy_logits"] >= 0)
                else:
                    assert lr["head"]["kind"] == "NONE_NATIVE_ALL_FREE" and not (lp.parent / "calls.jsonl").exists()
                    assert not lay["occupied"].any()
                assert np.isfinite(lay["occupancy_logits"]).all()
                assert lr["active"] == int(lay["occupied"].sum()) == len(lay["seam_slots"])
                layouts[cid] = lay
                layout_rows.append(dict(condition_id=cid,active=int(lr["active"]),head_calls=lr["occupancy_forward_calls"],receipt_sha256=sha(lp)))
            lay = layouts[cid]
            core, raw = arrays(q["inference_npz"]), arrays(dest / "raw.npz")
            traj, noise = arrays(dest / "trajectory.npz")["state"], arrays(q["noise_npz"])["z"]
            K,N = q["K"],q["N"]
            original = core["C_vertices"][core["C_faces_original"]]
            canonical = core["C_vertices"][core["C_faces_model"]]
            assert int(raw["known_faces"]) == K and int(raw["input_faces"]) == N
            assert int(lay["N"]) == N and int(lay["K"]) == K
            assert raw["raw"].shape == raw["output"].shape == (N,3,3)
            assert traj.shape == (51,N,3,3) and traj.dtype == np.float32 and np.isfinite(traj).all()
            assert same(lay["anchor_ids"],raw["raw_anchor_ids"])
            fixed = lay["anchor_ids"] >= 0
            initial = np.zeros((N,3,3),np.float32)
            initial[K:] = noise
            initial[fixed] = lay["C_vertices"][lay["anchor_ids"][fixed]]
            assert same(initial,traj[0]) and same(traj[-1],raw["raw"])
            assert all(same(state[fixed],initial[fixed]) for state in traj)
            assert all(same(state[:K],canonical) for state in traj)
            assert same(raw["output"][:K],original) and same(raw["vertices"][raw["faces"]],raw["output"])
            expected_mask = np.zeros(N,bool);expected_mask[lay["seam_slots"]] = True
            assert np.array_equal(raw["raw_seam_slot_mask"],expected_mask) and np.array_equal(raw["retained_seam_mask"],expected_mask)
            assert np.array_equal(lay["seam_slots"],K+np.flatnonzero(lay["occupied"]))
            assert np.array_equal(lay["bulk_slots"],np.flatnonzero((np.arange(N)>=K)&~expected_mask))
            assert (fixed[K:].sum(axis=1) == (2*expected_mask[K:])).all()
            assert not fixed[lay["bulk_slots"]].any() and fixed[:K].all()
            assert np.array_equal(lay["free_corner_mask"],~fixed)
            for key in ("C_vertices","C_faces_original","C_faces_model"):
                assert same(raw[key],core[key]) and same(lay[key],core[key])
            assert same(raw["occupied"],lay["occupied"]) and same(raw["occupancy_logits"],lay["occupancy_logits"])
            edge_ids=np.full(N,-1,np.int64);edge_ids[K:K+len(lay["occupied"])]=np.arange(len(lay["occupied"]))
            assert same(raw["raw_slot_edge_id"],edge_ids)
            for a in raw.values():
                if a.dtype.kind == "f":assert np.isfinite(a).all()
            assert sha(dest / "source_condition.npz") == q["source_npz_sha256"]
            source=arrays(dest / "source_condition.npz")
            for key in ("C_original_source_vertex_ids","C_source_corner_permutations"):
                assert same(raw[key],source[key])
            assert np.array_equal(raw["source_corner_ids"][:K],np.argsort(source["C_source_corner_permutations"],axis=1))
            rows.append(dict(label=label,request_id=q["request_id"],condition_id=cid,parent_id=q["parent_id"],N=N,K=K,
                noise_sha256=q["noise_sha256"],input_sha256=q["inference_npz_sha256"],source_npz_sha256=q["source_npz_sha256"],
                raw_receipt_sha256=sha(dest / "receipt.json"),raw_sha256=receipt["raw_sha256"],trajectory_sha256=sha(dest / "trajectory.npz"),
                forward_started=50,forward_complete=50,D0_calls=0,finite_FP32=True,C_all51_bitwise=True,source_C_winding_restored=True,
                seam_slots=int(expected_mask.sum()),fixed_free_corners=int(fixed[K:].sum()),initial_shared_noise_before_anchor_writeback=True))
        assert len(layouts)==12 and events_total=={"FORWARD_STARTED":1200,"FORWARD_COMPLETE":1200}
        assert head_events==({"HEAD_FORWARD_STARTED":12,"HEAD_FORWARD_RETURNED":12} if is_seam else {})
        groups[label]=dict(outputs=24,generator_events=dict(events_total),head_events=dict(head_events),layouts=layout_rows,
            attempts=1,failures=0,raw_completion_sha256=sha(folder/"raw_complete_full.json"),seconds_concurrent=done["seconds"],
            actual_rollout_peak_allocated_bytes=max(read(folder/q["request_id"]/"raw/audit.json")["peak_allocated_bytes"] for q in qs),
            actual_rollout_peak_reserved_bytes=max(read(folder/q["request_id"]/"raw/audit.json")["peak_reserved_bytes"] for q in qs),
            resource_summary=read(attempts[0]/"resource_observation.json")["summary"])
    by={(r["label"],r["request_id"]):r for r in rows}
    for q in qs:
        trio=[by[label,q["request_id"]] for label in LABELS]
        for key in ("request_id","condition_id","parent_id","N","K","noise_sha256","input_sha256","source_npz_sha256"):
            assert len({r[key] for r in trio})==1
    assert "torch" not in sys.modules
    return dict(schema="ALL72_RAW_CACHE_AUDIT_V1",status="PASS",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        paired_requests=24,outputs=72,total_generator_forwards_recorded=3600,total_head_forwards_recorded=12,total_RAW_D0_forwards_recorded=0,
        labels=groups,rows=rows,source_code_sha256=sha(__file__),source_helper_sha256=sha(Path(__file__).with_name("audit_native_raw_cache.py")),
        model_calls=0,scoring_calls=0,rendered_images=0,notes="Read-only saved-product audit. SEAM fixes two corners of active slots; shared standard Gaussian precedes registered anchor writeback. Quality not inferred.")


if __name__=="__main__":
    result=run();p=OUT/"engineering/all72_raw_cache_audit.json"
    with p.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","labels")},ensure_ascii=False))
