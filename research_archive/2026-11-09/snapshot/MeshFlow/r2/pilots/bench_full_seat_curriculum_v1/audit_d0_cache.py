"""Read-only CPU audit of the registered single-D0/source-oriented products."""
from __future__ import annotations
import collections
import datetime
import json
import sys
import numpy as np
from .audit_native_raw_cache import OUT, read, sha, arrays, same
from .audit_all_raw_cache import LABELS


def run():
    qs = read(OUT / "eval_manifest.json")["requests"]
    checked, rows, groups = {}, [], {}

    def verify(items):
        for p,h in items.items():
            if p not in checked:checked[p]=sha(p)
            assert checked[p]==h,p

    raw_receipts={}
    latest_raw_return=0
    for label in LABELS:
        folder=OUT/"evaluation"/label
        r=read(folder/"raw_complete_full.json")
        assert r["status"]=="COMPLETE" and r["outputs"]==24
        verify(r["files"])
        for q in qs:
            rawdir=folder/q["request_id"]/"raw"
            raw_receipts[str(rawdir/"receipt.json")]=sha(rawdir/"receipt.json")
            calls=[json.loads(s) for s in (rawdir/"calls.jsonl").read_text().splitlines() if s.strip()]
            latest_raw_return=max(latest_raw_return,max(c["unix"] for c in calls))
    assert len(raw_receipts)==72
    first_D0_start=None
    for label in LABELS:
        folder=OUT/"evaluation"/label
        done=read(folder/"post_complete_full.json")
        assert done["status"]=="COMPLETE" and done["outputs"]==done["new_D0_forwards"]==done["new_GEO_calls"]==24
        verify(done["files"])
        registration=read(OUT/"evaluation/inference_registrations"/(label+"__D0.json"))
        assert registration["new_D0_budget"]==registration["new_GEO_budget"]==24
        assert registration["RAW_barrier"]==raw_receipts and registration["reused_POST"]==0
        verify(registration["RAW_barrier"]);verify(registration["code"])
        attempts=list(folder.glob("postprocess_attempt_*"));observers=list(folder.glob("D0_resource_observer_*"))
        assert len(attempts)==len(observers)==1 and not (attempts[0]/"failure.json").exists()
        barrier=read(attempts[0]/"all_RAW_barrier.json")
        assert barrier["status"]=="PASS" and barrier["outputs"]==72 and barrier["files"]==raw_receipts
        events=collections.Counter();peak_alloc=peak_reserved=0
        for q in qs:
            root=folder/q["request_id"];dest=root/"post"
            receipt=read(dest/"receipt.json");pr=read(dest/"proposal_receipt.json")
            verify(receipt["files"])
            assert receipt["status"]=="COMPLETE" and receipt["label"]==label and receipt["request_id"]==q["request_id"]
            assert receipt["condition_id"]==q["condition_id"] and receipt["source_labels_read"] is False
            assert receipt["D0_forwards"]==receipt["GEO_calls"]==pr["D0_forwards"]==1
            assert receipt["official_D0_sha256"]==pr["official_checkpoint_sha256"]==registration["D0_checkpoint_sha256"]
            assert receipt["raw_sha256"]==pr["raw_sha256"]==sha(root/"raw/raw.npz")
            assert receipt["post_sha256"]==sha(dest/"post.npz") and pr["proposal_sha256"]==sha(dest/"proposal.npz")
            calls=[json.loads(s) for s in (dest/"calls.jsonl").read_text().splitlines() if s.strip()]
            assert [c["event"] for c in calls]==["D0_FORWARD_STARTED","D0_FORWARD_RETURNED"]
            events.update(c["event"] for c in calls)
            started_unix=datetime.datetime.fromisoformat(calls[0]["utc"]).timestamp()
            assert started_unix>latest_raw_return
            first_D0_start=started_unix if first_D0_start is None else min(first_D0_start,started_unix)
            raw=arrays(root/"raw/raw.npz");post=arrays(dest/"post.npz");canonical=arrays(dest/"post_canonical.npz")
            pre=arrays(dest/"pre_geo.npz");proposal=arrays(dest/"proposal.npz")["official_unconstrained_proposal"]
            core=arrays(q["inference_npz"]);K,N=q["K"],q["N"]
            original=core["C_vertices"][core["C_faces_original"]]
            assert proposal.shape==(N,3,3) and proposal.dtype==np.float32 and np.isfinite(proposal).all()
            expected=proposal.copy();fixed=raw["raw_anchor_ids"]>=0
            expected[fixed]=raw["C_vertices"][raw["raw_anchor_ids"][fixed]]
            assert same(pre["output"],expected)
            assert int(post["known_faces"])==K and int(post["input_faces"])==N
            assert post["output"].dtype==np.float32 and np.isfinite(post["output"]).all() and np.isfinite(post["vertices"]).all()
            assert same(post["vertices"][post["faces"]],post["output"]) and same(post["output"][:K],original)
            assert same(post["vertices"],canonical["vertices"]) and np.array_equal(np.sort(post["faces"],axis=1),np.sort(canonical["faces"],axis=1))
            assert same(post["raw_anchor_ids"],raw["raw_anchor_ids"]) and same(post["raw_seam_slot_mask"],raw["raw_seam_slot_mask"])
            assert np.array_equal(post["retained_seam_mask"],raw["raw_seam_slot_mask"][post["raw_slot_ids"]])
            aid=post["output_anchor_ids"];mask=aid>=0;cmap=post["compact_C_vertex_to_post"]
            assert len(set(cmap.tolist()))==len(cmap)==len(core["C_vertices"])
            assert np.array_equal(post["faces"][mask],cmap[aid[mask]])
            assert same(post["output"][mask],core["C_vertices"][aid[mask]])
            anchors=read(dest/"anchor_identity.json");rest=read(dest/"source_restoration.json")
            assert anchors==receipt["anchors"] and anchors["status"]==rest["status"]=="PASS"
            assert anchors["C_vertex_ids_bijective"] and anchors["final_anchor_vertex_identity_verified"]
            assert rest["source_C_vertex_ID_topology"]["status"]=="PASS" and rest["source_C_order_fixed_after_restoration"]
            assert anchors["input_active_seam_faces"]==int(raw["raw_seam_slot_mask"].sum())
            assert anchors["retained_seam_faces"]==int(post["retained_seam_mask"].sum())
            assert anchors["deleted_seam_faces"]==anchors["input_active_seam_faces"]-anchors["retained_seam_faces"]
            if label!=LABELS[-1]:assert anchors["retained_anchor_corners"]==anchors["input_active_seam_faces"]==0
            peak_alloc=max(peak_alloc,pr["peak_allocated_bytes"]);peak_reserved=max(peak_reserved,pr["peak_reserved_bytes"])
            rows.append(dict(label=label,request_id=q["request_id"],condition_id=q["condition_id"],parent_id=q["parent_id"],
                N=N,K=K,D0_calls=1,GEO_calls=1,C_original_bitwise=True,C_vertex_ID_bijection=True,retained_anchors_exact=True,
                pre_geo_expected_anchor_writeback_exact=True,finite_FP32=True,output_faces=len(post["faces"]),
                input_seam_faces=anchors["input_active_seam_faces"],retained_seam_faces=anchors["retained_seam_faces"],deleted_seam_faces=anchors["deleted_seam_faces"],
                post_sha256=receipt["post_sha256"],receipt_sha256=sha(dest/"receipt.json"),proposal_receipt_sha256=sha(dest/"proposal_receipt.json")))
        assert events=={"D0_FORWARD_STARTED":24,"D0_FORWARD_RETURNED":24}
        observation=read(observers[0]/"resource_observation.json")
        groups[label]=dict(outputs=24,event_counts=dict(events),GEO_calls=24,postprocess_attempts=1,observer_instances=1,failure_records=0,
            post_completion_sha256=sha(folder/"post_complete_full.json"),peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,
            resource_observer_summary=observation["summary"],seconds=done["seconds"])
    assert len(rows)==72 and "torch" not in sys.modules
    return dict(schema="ALL72_SINGLE_D0_CACHE_AUDIT_V1",status="PASS",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        outputs=72,total_D0_forwards_recorded=72,total_GEO_calls_recorded=72,labels=groups,rows=rows,
        latest_RAW_forward_return_unix=latest_raw_return,first_D0_forward_start_unix=first_D0_start,RAW_barrier_precedes_D0=True,
        model_calls=0,geometry_scoring_calls=0,geometry_processing_calls=0,rendered_images=0,source_code_sha256=sha(__file__),
        note="Single official proposal and exact registered anchor/C restoration verified from cached arrays. Topology identity is not geometry quality or C-use certification.")


if __name__=="__main__":
    result=run();p=OUT/"engineering/all72_d0_cache_audit.json"
    with p.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","labels")},ensure_ascii=False))
