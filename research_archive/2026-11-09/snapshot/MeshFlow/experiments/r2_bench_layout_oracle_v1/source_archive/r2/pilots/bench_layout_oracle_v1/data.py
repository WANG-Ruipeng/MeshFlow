"""CPU-only source-region construction for the Bench layout oracle pilot.

Source geometry is used only for the explicitly privileged layout diagnostic.
No model, CUDA, target repair, source overwrite or generated noise is involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_layout_oracle_v1/data"
PRIOR = ROOT / "experiments/r2_bench_three_mechanisms_v1/data"
L = float(np.sqrt(3) * 2 / .3762)
SLOTS = ("BACK_REGION", "SUPPORT_REGION")
REGION_SOURCES = (
    "experiments/r2_bench_user_data_v2/manifest.json",
    "experiments/r2_bench_user_data_v2_expansion_v1/manifest.json",
    "experiments/r2_bench_parts_v1/data/manifest.json",
    "experiments/r2_bench_coverage_expansion_v1/data/manifest.json",
    "experiments/r2_bench_user_relation_rebuild_v1/manifest.json",
    "experiments/r2_bench_three_family_v1/tasks/final_manifest.json",
)


def resolve(p):
    s = str(p).replace("\\", "/")
    if s.startswith("/mnt/e/"):
        return ROOT.parent / s[len("/mnt/e/"):]
    if s.lower().startswith("e:/"):
        return ROOT.parent / s[3:]
    return ROOT / s if not Path(s).is_absolute() else Path(s)


def read(p):
    return json.loads(resolve(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with resolve(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = resolve(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_rows(doc):
    if isinstance(doc, list):
        return doc
    for k in ("conditions", "rows", "train", "candidates", "accepted"):
        if isinstance(doc.get(k), list):
            return doc[k]
    return []


def inventory():
    manifest = read(PRIOR / "manifest.json")
    ep = read(PRIOR / "eval_manifest.json")
    cases = manifest["train"] + list({q["condition_id"]: q for q in ep["requests"]}.values())
    result = []
    for r in cases:
        z = np.load(resolve(r["case_path"]), allow_pickle=False)
        result.append(dict(parent=r["parent_id"], condition=r["condition_id"],
                           group=r.get("replay_group", r.get("eval_group")),
                           case_path=r["case_path"], arrays={k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype)) for k in z.files}))
    write(OUT / "source_inventory.json", result)
    print(json.dumps(result[:2] + result[-6:], indent=2))


def type_slot(r):
    part = r.get("part_type", "")
    if part == "BACK_SURFACE":
        return 0
    if part in ("LEG_SUPPORT", "SEAT_SUPPORT"):
        return 1
    cid = r.get("condition_id", "")
    # These explicit reviewed condition names identify a complete end support.
    if any(x in cid for x in ("END_FRAME_FULL", "END_WALL_FULL", "COMPLETE_LEFT_END_OUTER_SURFACE")):
        return 1
    return None


def library():
    result = []
    for path in REGION_SOURCES:
        p = resolve(path)
        if not p.exists():
            continue
        for r in get_rows(read(p)):
            slot = type_slot(r)
            if slot is None or not r.get("case_path"):
                continue
            review = r.get("review") or {}
            if isinstance(review, str):
                review = json.loads(review)
            decisions = [str(r.get(k, "")) for k in ("review_status", "eligibility", "decision")]
            decisions.append(str(review.get("decision", "")))
            if not any("ACCEPT" in s for s in decisions):
                continue
            result.append(dict(row=r, slot=slot, manifest=str(p), manifest_sha256=sha(p)))
    return result


def region_inventory():
    manifest = read(PRIOR / "manifest.json")
    ep = read(PRIOR / "eval_manifest.json")
    cases = list({r["condition_id"]: r for r in manifest["train"] + ep["requests"]}.values())
    lib = library()
    records = []
    for r in cases:
        z = np.load(resolve(r["case_path"]), allow_pickle=False)
        full = z["full_target"]
        parts = []
        failures = []
        for item in lib:
            rr = item["row"]
            if rr["parent_id"] != r["parent_id"]:
                continue
            pp = resolve(rr["case_path"])
            if not pp.exists():
                failures.append(dict(condition=rr["condition_id"],reason="MISSING_CASE"))
                continue
            zz = np.load(pp, allow_pickle=False)
            if "full_target" not in zz or not np.array_equal(full, zz["full_target"]):
                failures.append(dict(condition=rr["condition_id"],reason="TARGET_ARRAY_NOT_IDENTICAL",keys=list(zz.files)))
                continue
            ids = sorted(set(zz["known_ids"].tolist()) - set(z["known_ids"].tolist()))
            if not ids:
                continue
            parts.append(dict(slot=item["slot"], source_condition=rr["condition_id"], face_ids=ids,
                extent=rr.get("extent_description", ""), manifest=item["manifest"], case_path=str(pp)))
        records.append(dict(parent_id=r["parent_id"],condition_id=r["condition_id"],
            group=r.get("replay_group",r.get("eval_group")),case_path=r["case_path"],
            N=int(z["N"]),K=int(z["K"]),regions=parts,failures=failures))
    write(OUT / "region_inventory.json", records)
    print(json.dumps([dict(parent=x["parent_id"][:6],group=x["group"],regions=[(p["slot"],len(p["face_ids"])) for p in x["regions"]],failures=len(x["failures"])) for x in records],indent=2))


def area_center(C):
    C = np.asarray(C, np.float64).reshape(-1,3,3)
    a = np.linalg.norm(np.cross(C[:,1]-C[:,0], C[:,2]-C[:,0]), axis=1) / 2
    if not np.isfinite(a).all() or a.sum() <= 0:
        raise ValueError("C has no finite positive area")
    return np.sum(C.mean(1) * a[:, None], axis=0) / a.sum()


def propose_and_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    records = read(OUT / "region_inventory.json")
    proposed = []
    for r in records:
        z = np.load(resolve(r["case_path"]), allow_pickle=False)
        full, known = z["full_target"], z["known_ids"]
        item = dict(r)
        item["proposal"] = None
        if not r["regions"]:
            # A geometric proposal only. Admission requires actual source-image review.
            cutoff = float(full[known, :, 1].min()) - .005 * L
            ids = [int(i) for i in z["free_ids"] if float(full[i,:,1].mean()) < cutoff]
            item["proposal"] = dict(slot=1, face_ids=ids, cutoff_y=cutoff,
                recipe="triangle_centroid_y < min(C.y) - 0.005L; source lower-region proposal, not automatic semantic acceptance",
                status="PENDING_ACTUAL_VISUAL_REVIEW")
        proposed.append(item)
        fig = plt.figure(figsize=(14,4.8), dpi=130)
        for j, (elev, azim) in enumerate(((20,-65),(15,65),(10,0))):
            ax = fig.add_subplot(1,3,j+1,projection="3d")
            ax.add_collection3d(Poly3DCollection(full,facecolors="#b6bec4",edgecolors="#6f777d",linewidths=.12,alpha=.2))
            ax.add_collection3d(Poly3DCollection(full[known],facecolors="#078fc5",edgecolors="#005577",linewidths=.25,alpha=.9))
            for slot in range(2):
                ids = sorted({i for p in item["regions"] if p["slot"]==slot for i in p["face_ids"]})
                if item["proposal"] and item["proposal"]["slot"]==slot:
                    ids = item["proposal"]["face_ids"]
                if ids:
                    color = "#9c4dcc" if slot==0 else "#ed8c20"
                    ax.add_collection3d(Poly3DCollection(full[ids],facecolors=color,edgecolors="#464646",linewidths=.22,alpha=.78))
            points=full.reshape(-1,3);lo=points.min(0);hi=points.max(0);mid=(lo+hi)/2; span=max(hi-lo)*.53
            ax.set_xlim(mid[0]-span,mid[0]+span);ax.set_ylim(mid[1]-span,mid[1]+span);ax.set_zlim(mid[2]-span,mid[2]+span)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim,vertical_axis="y")
            ax.set_xlabel("x");ax.set_ylabel("y (up)");ax.set_zlabel("z")
        proposal_text="PROPOSAL" if item["proposal"] else "EXACT-SAME-TARGET REVIEWED REGIONS"
        fig.suptitle(f'{r["parent_id"][:6]} | N={r["N"]}, K={r["K"]} | {proposal_text}\nC cyan / back purple / support orange. Regions need not cover every component.')
        path=OUT / "region_gallery" / (r["parent_id"]+".png")
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.tight_layout();fig.savefig(path);plt.close(fig)
        item["review_image"] = str(path);item["review_image_sha256"] = sha(path)
    write(OUT / "region_proposals.json", proposed)
    print(json.dumps([dict(parent=r["parent_id"][:6],count=len(r["proposal"]["face_ids"]),image=r["review_image"]) for r in proposed if r["proposal"]],indent=2))


MANUAL_REVIEW = {
    "27eb79": "Three actual source views show the selected lower vertical leg portions below the complete curved seat. Their union bounds are a coarse lower-support region; upper arm/frame and back remain UNKNOWN, not absent.",
    "28b9f2": "Three source views show two rectangular end supports and the undersides/sides of the slatted seat between them. Admit the broad seat-support assembly region, not a pure individual-leg label.",
    "abf250": "Three views show the selected two thick rectangular end supports below the slatted seat. The selected region does not include the blue top faces.",
    "945214": "Three views show lower end frames, legs, longitudinal brace and lower slat surfaces. Admit a broad lower support-and-frame region; raised end rails are not relabelled as a back.",
    "af264c": "Three views show lower legs and the seat-surrounding lower frame. Tall rear bars and arm tops are unselected and remain UNKNOWN in the back slot.",
    "bcb4f5": "Three views show the curved continuous ring base/shell under the flat top. Admit as broad support shell, explicitly not four legs and not independent disconnected parts.",
    "77aa33": "Three views show the two ornamented pedestal supports under the blue seat. Admit their combined coarse bounds, with internal shape and count not transmitted to the model.",
}


def encode_boxes(bboxes, valid, C):
    """Fixed 2x8 schema: relative centre, FULL extent, BACK/SUPPORT onehot."""
    bbox = np.asarray(bboxes, np.float32)
    valid = np.asarray(valid, bool)
    if bbox.shape != (2,2,3) or valid.shape != (2,):
        raise ValueError("Expected two coarse boxes and separate known-region mask")
    centre = area_center(C)
    out = np.zeros((2,8), np.float32)
    out[:,:3] = ((bbox[:,0].astype(np.float64)+bbox[:,1].astype(np.float64))/2-centre)/L
    out[:,3:6] = (bbox[:,1].astype(np.float64)-bbox[:,0].astype(np.float64))/L
    out[0,6],out[1,7] = 1,1
    out[~valid] = 0
    if not np.isfinite(out).all() or (out[valid,3:6]<0).any():
        raise ValueError("Nonfinite or negative coarse extent")
    return out, valid.copy(), centre.astype(np.float32)


def build():
    proposals = read(OUT / "region_proposals.json")
    original = read(PRIOR / "manifest.json")
    evaluation = read(PRIOR / "eval_manifest.json")
    by_cond = {}
    protected = {}
    manual = []
    all_rows = []
    now = datetime.now(timezone.utc).isoformat()
    for r in proposals:
        path=resolve(r["case_path"]);protected[str(path)]=sha(path)
        z=np.load(path,allow_pickle=False);full=z["full_target"];known=set(z["known_ids"].tolist())
        regions=[set(),set()]
        provenance=[]
        for p in r["regions"]:
            regions[p["slot"]].update(p["face_ids"])
            provenance.append(dict(kind="INHERITED_REVIEW_EXACT_SAME_TARGET",**p))
            for source_path in (p["manifest"],p["case_path"]):
                protected[str(resolve(source_path))]=sha(source_path)
        if r["proposal"]:
            note=MANUAL_REVIEW[r["parent_id"][:6]]
            review=dict(parent_id=r["parent_id"],condition_id=r["condition_id"],
                reviewer="AI_PROXY_layout_data_build_NONBLIND",actually_viewed=True,
                decision="ACCEPT_COARSE_SUPPORT_REGION",reviewed_utc=now,
                image=r["review_image"],image_sha256=r["review_image_sha256"],observations=note,
                scope="Source-target region curation only; not generated-output or human review")
            if sha(r["review_image"]) != r["review_image_sha256"]:
                raise ValueError("Reviewed image changed")
            manual.append(review)
            p=r["proposal"];regions[p["slot"]].update(p["face_ids"])
            provenance.append(dict(kind="NEW_ACTUALLY_REVIEWED_SOURCE_REGION",**p,review=review))
        regions=[sorted(s-known) for s in regions]
        bbox=np.zeros((2,2,3),np.float32);valid=np.zeros(2,bool)
        for slot,ids in enumerate(regions):
            if ids:
                triangles=full[ids]
                area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2
                if area <= 1e-12*L*L:
                    raise ValueError("Region has no positive geometric area")
                pts=triangles.reshape(-1,3);bbox[slot]=np.stack((pts.min(0),pts.max(0)));valid[slot]=True
        tokens,valid,centre=encode_boxes(bbox,valid,full[z["known_ids"]])
        dest=OUT/"layouts"/(r["condition_id"]+".npz");dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez(dest,tokens=tokens,valid=valid,center_C=centre,absolute_bbox=bbox,
            back_face_ids=np.asarray(regions[0],np.int64),support_face_ids=np.asarray(regions[1],np.int64),
            L=np.asarray(L,np.float64),schema=np.asarray("BACK_SUPPORT_BBOX_CENTRE_EXTENT_ONEHOT8_V1"))
        record=dict(parent_id=r["parent_id"],condition_id=r["condition_id"],case_path=str(path),case_sha256=sha(path),
            layout_npz=str(dest),layout_npz_sha256=sha(dest),valid=valid.tolist(),
            region_status=["OBSERVED_REGION" if v else "UNKNOWN_NOT_ABSENT" for v in valid],
            bbox_scope="Union bounds of reviewed selected non-C surfaces; not a complete semantic decomposition",
            input_scope="ORACLE_PRIVILEGED_SOURCE_LAYOUT",source_face_ids=regions,provenance=provenance,
            geometry_modified=False,N=r["N"],K=r["K"],image=r["review_image"],image_sha256=r["review_image_sha256"])
        by_cond[r["condition_id"]]=record;all_rows.append(record)
    train=[];excluded=[]
    for r in original["train"]:
        rec=by_cond[r["condition_id"]]
        if any(rec["valid"]):
            row=dict(r);row.update(layout_npz=rec["layout_npz"],layout_npz_sha256=rec["layout_npz_sha256"],layout_valid=rec["valid"])
            train.append(row)
        else:
            excluded.append(dict(parent_id=r["parent_id"],reason="NO_VERIFIED_NON_C_LAYOUT"))
    eval_rows=[]
    for q in evaluation["requests"]:
        rec=by_cond[q["condition_id"]]
        if not any(rec["valid"]):
            raise ValueError("Registered evaluation request has no privileged layout")
        eval_rows.append(dict(request_id=q["request_id"],parent_id=q["parent_id"],condition_id=q["condition_id"],
            layout_npz=rec["layout_npz"],layout_npz_sha256=rec["layout_npz_sha256"],
            valid=rec["valid"],input_scope="ORACLE_PRIVILEGED_SOURCE_LAYOUT"))
    write(OUT/"manifest.json",dict(schema="R2_BENCH_LAYOUT_ORACLE_V1_data",train=train,excluded=excluded,
        source_manifest=str(PRIOR/"manifest.json"),source_manifest_sha256=sha(PRIOR/"manifest.json"),
        slots=list(SLOTS),layout_dim=8,independent_new_parents=0,new_C=0,source_geometry_changed=False))
    write(OUT/"eval_manifest.json",evaluation)
    write(OUT/"layout_eval_manifest.json",dict(requests=eval_rows,input_scope="ORACLE_PRIVILEGED_SOURCE_LAYOUT"))
    write(OUT/"layout_records.json",dict(records=all_rows,schema="BACK_SUPPORT_BBOX_CENTRE_EXTENT_ONEHOT8_V1"))
    write(OUT/"manual_source_review.json",dict(records=manual,source_review_only=True))
    write(OUT/"protected_sources.json",protected)
    write(OUT/"data_ready.json",dict(status="COMPLETE",train_parents=len(train),eval_requests=len(eval_rows),
        eval_parents=len({r["parent_id"] for r in eval_rows}),old_review_target_matches=sum(not r["proposal"] for r in proposals),
        new_source_regions_reviewed=len(manual),training_model_calls=0,model_forwards=0,new_scientific_noise=0,
        new_layout_bboxes=sum(sum(r["valid"]) for r in all_rows),
        engineering_events=["Historical nullable review field handled before region extraction; zero model calls"],
        caveat="Known masks identify available reviewed regions, not semantic absence. Correct-source P is extra information, not C-only inference or a mathematical upper bound."))
    print(json.dumps(dict(status="COMPLETE",parents=len(train),requests=len(eval_rows),manual_reviews=len(manual))))


@lru_cache(maxsize=128)
def _load_layout(path):
    with np.load(resolve(path),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


@lru_cache(maxsize=1)
def _records_by_condition():
    return {r["condition_id"]:r for r in read(OUT/"layout_records.json")["records"]}


def training_layout(case, row, sample):
    record=_records_by_condition()[row["condition_id"]]
    z=_load_layout(row.get("layout_npz",record["layout_npz"]))
    bbox=z["absolute_bbox"].copy();bbox[...,0]*=np.float32(row["alpha"])
    C=np.asarray(sample["xt"])[np.asarray(sample["known_mask"],bool)]
    tokens,valid,_=encode_boxes(bbox,z["valid"],C)
    if row.get("layout_drop",False):
        tokens[:]=0;valid[:]=False
    return tokens,valid


def eval_layout(q):
    record=_records_by_condition()[q["condition_id"]]
    z=_load_layout(record["layout_npz"])
    return z["tokens"].copy(),z["valid"].copy()


def verify():
    manifest=read(OUT/"manifest.json");records=_records_by_condition();checks=[]
    for cid,r in records.items():
        assert sha(r["case_path"])==r["case_sha256"]
        assert sha(r["layout_npz"])==r["layout_npz_sha256"]
        case=np.load(resolve(r["case_path"]),allow_pickle=False);layout=_load_layout(r["layout_npz"])
        for a in (.9,1.,1.1):
            full=case["full_target"].copy();full[...,0]*=np.float32(a)
            sample=dict(xt=full[case["known_ids"]],known_mask=np.ones(len(case["known_ids"]),bool))
            tokens,valid=training_layout(case,dict(condition_id=cid,alpha=a),sample)
            bbox=np.zeros((2,2,3),np.float32)
            for slot,ids in enumerate(r["source_face_ids"]):
                assert not set(ids)&set(case["known_ids"].tolist())
                if ids:
                    pts=full[ids].reshape(-1,3);bbox[slot]=np.stack((pts.min(0),pts.max(0)))
            truth,mask,_=encode_boxes(bbox,layout["valid"],sample["xt"])
            np.testing.assert_array_equal(tokens,truth);np.testing.assert_array_equal(valid,mask)
            dropped,vm=training_layout(case,dict(condition_id=cid,alpha=a,layout_drop=True),sample)
            assert not dropped.any() and not vm.any()
        checks.append(dict(condition_id=cid,three_width_scales_exact=True,unknown_not_absent=True,
            non_C_region_identity=True,dropout_all_null=True))
    for p,h in read(OUT/"protected_sources.json").items():
        assert sha(p)==h
    result=dict(status="PASS",conditions=len(checks),checks=checks,model_calls=0,OT_calls=0,
        source_target_modified=False,layout_truth_scope="Source diagnostic only")
    write(OUT/"verification.json",result)
    print(json.dumps({k:result[k] for k in ("status","conditions","model_calls","OT_calls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--region-inventory", action="store_true")
    parser.add_argument("--propose-render", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.inventory:
        inventory()
    if args.region_inventory:
        region_inventory()
    if args.propose_render:
        propose_and_render()
    if args.build:
        build()
    if args.verify:
        verify()
