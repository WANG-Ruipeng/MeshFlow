#!/usr/bin/env python3
"""Compare saved R2/two-curriculum outputs; CPU rendering only, no new scores.

The primary 128 pages use unchanged main-phase bounds and five-view renderer.
If any comparison RAW vertex exceeds those original bounds, all three methods
and all four noises for that condition receive separate complete-view pages.
Their joint bounds use the same renderer's 1.08 margin. No extra samples/queries.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"src"));sys.path.insert(0,str(REPO/"tools"))
import numpy as np
from meshflow_control.evaluation.io import arrays,digest,read_json,resolve,write_json
import evaluate_recipe_factorial as ev
from evaluate_schedule40 import render_page,VIEWS

VERSION="recipe_curriculum_comparison_v1"
BASELINE="R2_MIX_H_ALL"
COURSES=("C20_40_MIX","C40_20_MIX")
LABELS=(BASELINE,*COURSES)
CAMERA_POLICY={
    "primary":"All 128 pages retain main cameras.json bounds exactly, with frozen render_page five directions and 1.08 margin.",
    "overflow_trigger":"Any comparison RAW vertex outside original main bounds, even if still inside the padded camera cube.",
    "supplement":"Affected condition only: R2 plus both courses and all four noises determine joint bounds; all three methods and all four noises are shown in separate colored/neutral pages.",
    "supplement_margin":"Unchanged render_page 1.08 margin; same five view directions.",
    "no_overflow":"Zero supplementary pages.",
    "interpretation":"Primary framing limitation is explicitly marked; supplement is visualization, not a new sample, query, score or model call."
}
def aliases():
    order=sorted(LABELS,key=lambda x:hashlib.sha256((VERSION+"/"+x).encode()).hexdigest())
    return {label:"Method "+str(i+1) for i,label in enumerate(order)}
def check_hash(path,expected):
    if digest(path)!=expected:raise ValueError("Frozen file changed: "+str(path))
def source_hashes(out):
    frozen=read_json(out/"registrations/source_hashes.json")
    if len(frozen)!=97:raise ValueError("Expected original 97-source freeze")
    for rel,h in frozen.items():check_hash(REPO/rel,h)
    return {"original_frozen_sources":frozen,"comparison_source_sha256":digest(__file__),
            "renderer_sha256":digest(REPO/"tools/evaluate_schedule40.py")}
def validate_saved_panel(main_reg,course_reg,requests,main_scores,course_scores,main_done,course_done):
    if tuple(main_reg.get("labels",()))!=ev.LABELS or main_reg.get("expected_outputs")!=384:
        raise ValueError("Full six-label main registration required")
    if course_reg.get("kind")!="course" or tuple(course_reg.get("labels",()))!=COURSES or course_reg.get("expected_outputs")!=128:
        raise ValueError("Exactly two ordered curriculum labels required")
    if main_reg.get("requests_sha256")!=course_reg.get("requests_sha256"):
        raise ValueError("Main/course request identity differs")
    for receipt,expected in ((main_done,384),(course_done,128)):
        if receipt.get("status")!="COMPLETE" or receipt.get("registered")!=expected or receipt.get("scored")!=expected:
            raise ValueError("All saved scoring must be complete")
    if len(main_scores)!=384 or len(course_scores)!=128:raise ValueError("Complete saved score tables required")
    qby={q["request_id"]:q for q in requests}
    if len(qby)!=64 or len(requests)!=64:raise ValueError("Exactly 64 unique fixed requests required")
    for rows,labels in ((main_scores,ev.LABELS),(course_scores,COURSES)):
        actual=[(r["label"],r["request_id"]) for r in rows]
        if len(set(actual))!=len(actual) or set(actual)!={(l,q) for l in labels for q in qby}:
            raise ValueError("Duplicate or incomplete saved score cells")
    selected=[r for r in main_scores if r["label"]==BASELINE]+course_scores
    by={}
    for r in selected:
        q=qby[r["request_id"]]
        if any(r.get(k)!=q[k] for k in ("parent_id","condition_id","candidate","ratio","condition_cohort","N","K")):
            raise ValueError("Scored request metadata differs")
        if r.get("status")!="PASS" or not r.get("contracts") or not all(v is True for v in r["contracts"].values()):
            raise ValueError("Failed score/output contract")
        if r["identity"]["input_sha256"]!=q["input_sha256"]:raise ValueError("Saved input identity differs")
        by[r["label"],r["request_id"]]=r
    for rid in qby:
        for label in COURSES:
            a,b=by[BASELINE,rid],by[label,rid]
            for k in ("query_sha256","registered_free_faces","registered_C_queries","query_weight_denominator","L0"):
                if a[k]!=b[k]:raise ValueError("Saved query/denominator differs: "+k)
    return by


def verify_saved_binding(q,receipt,score,binding,query_sha):
    if receipt["label"]!=score["label"] or any(receipt.get(k)!=q[k] for k in ("request_id","input_sha256","N","K","candidate","condition_id")):
        raise ValueError("Existing output receipt differs")
    if receipt["checkpoint_sha256"]!=binding["checkpoint_sha256"]:
        raise ValueError("Existing output checkpoint binding differs")
    for field in ("raw","ply","result","checkpoint"):
        if score["identity"][field+"_sha256"]!=receipt[field+"_sha256"]:
            raise ValueError("Score is not bound to current RAW/receipt: "+field)
    if score["query_sha256"]!=query_sha or score["identity"]["query_sha256"]!=query_sha:
        raise ValueError("Score is not bound to original query")


def compare_bounds(raw,bounds):
    lo,hi=np.asarray(bounds,dtype=float)
    points=np.asarray(raw).reshape(-1,3)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(hi<lo):
        raise ValueError("Invalid frozen main camera bounds")
    if not np.isfinite(points).all():raise ValueError("Nonfinite RAW geometry")
    center=(lo+hi)/2;half=max(float((hi-lo).max()),.01)*1.08/2
    outside=((points<lo)|(points>hi)).any(axis=1)
    padded=((points<center-half)|(points>center+half)).any(axis=1)
    return dict(outside_original_bounds=bool(outside.any()),outside_padded_axis_cube=bool(padded.any()),
                outside_original_vertex_instances=int(outside.sum()),outside_padded_vertex_instances=int(padded.sum()),
                raw_min=points.min(0).tolist(),raw_max=points.max(0).tolist())

def collect(out):
    """Read existing complete receipts and score tables. Never call completed_rows/score/sample."""
    out=ev.root(out);sources=source_hashes(out)
    mp,mreg,qs=ev.phase_registration(out,"main")
    cp,creg,cqs=ev.phase_registration(out,"course")
    if qs!=cqs:raise ValueError("Course requests differ from main")
    paths=[out/"evaluation/requests.json",mp/"registration.json",cp/"registration.json",
           mp/"evaluation/per_output.json",cp/"evaluation/per_output.json",
           mp/"evaluation/scoring_receipt.json",cp/"evaluation/scoring_receipt.json",
           mp/"outputs/all_completed.json",cp/"outputs/all_completed.json",
           mp/"gallery/cameras.json",mp/"gallery/gallery.json"]
    ms=read_json(paths[3])["rows"];cs=read_json(paths[4])["rows"]
    scored=validate_saved_panel(mreg,creg,qs,ms,cs,read_json(paths[5]),read_json(paths[6]))
    saved_main=read_json(paths[7]);saved_course=read_json(paths[8])
    for d,regpath,labels in ((saved_main,paths[1],ev.LABELS),(saved_course,paths[2],COURSES)):
        if d.get("status")!="COMPLETE" or d.get("registration_sha256")!=digest(regpath):
            raise ValueError("Saved complete RAW manifest differs")
        cells=[(r["label"],r["request_id"]) for r in d["rows"]]
        if len(cells)!=len(set(cells)) or set(cells)!={(l,q["request_id"]) for l in labels for q in qs}:
            raise ValueError("All registered RAW outputs required before comparison")
    receipt_by={(r["label"],r["request_id"]):r for r in saved_main["rows"]+saved_course["rows"]}
    preparation_path=out/"registrations/curriculum_preparation.json"
    preparation=read_json(preparation_path)
    pinned={str(resolve(k)):v for k,v in preparation["source_and_input_hashes"].items()}
    for original in (paths[0],paths[9]):
        expected=pinned.get(str(resolve(original)))
        if expected is None:raise ValueError("Curriculum preparation did not pin original input/camera")
        check_hash(original,expected)
    paths.append(preparation_path)
    cameras=read_json(paths[9]);gallery=read_json(paths[10])
    if gallery.get("status")!="COMPLETE" or gallery.get("page_count")!=128:
        raise ValueError("Complete main camera/gallery provenance required")
    if set(cameras)!={q["condition_id"] for q in qs}:raise ValueError("Main camera condition set differs")
    from score_recipe_factorial_conflict import load_registration,load_queries,query_registration_path
    query_path=query_registration_path(out,out/"evaluation/requests.json")
    qr=load_registration(out,out/"evaluation/requests.json");paths.append(query_path)
    qb={r["condition_id"]:r for r in qr["rows"]}
    if len(qb)!=16:raise ValueError("Exactly 16 existing query banks required")
    for q in qb.values():load_queries(q);paths.append(resolve(q["query_path"]))
    loaded={};inputs={};records=[];bindings={}
    for label in LABELS:
        phase=mp if label==BASELINE else cp
        bp=phase/"checkpoints"/(label+".json");binding=read_json(bp);paths.append(bp);bindings[label]=binding
        if binding["registration_sha256"]!=digest(phase/"registration.json"):
            raise ValueError("Existing checkpoint binding differs")
        for q in qs:
            rid=q["request_id"];r=receipt_by[label,rid];s=scored[label,rid]
            verify_saved_binding(q,r,s,binding,qb[q["condition_id"]]["query_sha256"])
            for p,h in s.get("files",{}).items():
                check_hash(p,h);paths.append(resolve(p))
            raw,facts=ev.verify_output(q,r,binding["checkpoint_sha256"])
            loaded[label,rid]=raw["raw"].copy()
            inputs[rid]=ev.read_input(q)["C"]
            paths.extend(resolve(r[field+"_path"]) for field in ("raw","ply","result"))
            paths.append(resolve(q["input_path"]))
            rp=phase/"outputs"/label/rid/"receipt.json";paths.append(rp)
            if read_json(rp)!=r:raise ValueError("Individual and complete RAW receipts differ")
            records.append(dict(r,source_phase="main" if label==BASELINE else "course",
                                original_receipt_path=str(rp),original_receipt_sha256=digest(rp),
                                source_score_table=str(phase/"evaluation/per_output.json"),
                                source_score_table_sha256=digest(phase/"evaluation/per_output.json"),
                                saved_score_identity=s["identity"],existing_output_contracts=facts))
    unique_paths={str(resolve(p)):digest(p) for p in paths}
    return dict(out=out,requests=qs,rows=records,loaded=loaded,inputs=inputs,cameras=cameras,
                input_file_hashes=unique_paths,source_hashes=sources,saved_scores=[scored[l,q["request_id"]] for l in LABELS for q in qs])

def render_registered_page(path,columns,bounds,title,neutral,identity,renderer=render_page):
    receipt=path.with_suffix(".json")
    if path.exists() or receipt.exists():
        if not (path.exists() and receipt.exists()):raise RuntimeError("Unregistered partial render; no silent overwrite: "+str(path))
        old=read_json(receipt)
        if old["identity"]!=identity or digest(path)!=old["sha256"]:raise ValueError("Existing comparison render changed")
        return old
    renderer(path,columns,bounds,title,neutral)
    row=dict(path=str(path),sha256=digest(path),identity=identity,AI_VISUAL_REVIEW="PENDING")
    write_json(receipt,row,freeze=True)
    return row

def render_bundle(bundle,dest,renderer=render_page):
    """CPU-only renderer adapter; bundle is already fully verified before the first page."""
    dest=Path(dest);qs=bundle["requests"];loaded=bundle["loaded"];amap=aliases()
    qids=[q["request_id"] for q in qs]
    if len(qids)!=64 or len(set(qids))!=64:
        raise ValueError("All four noises of the 16 fixed conditions required")
    conditions={q["condition_id"] for q in qs}
    if len(conditions)!=16 or set(loaded)!={(l,r) for l in LABELS for r in qids}:
        raise ValueError("All 192 verified comparison geometries required")
    for cid in conditions:
        group=[q for q in qs if q["condition_id"]==cid]
        if len(group)!=4 or {q["candidate"] for q in group}!={0,1,6,7}:
            raise ValueError("All four noises required before rendering")
    # Completeness is established before any output is created.
    ordered=sorted(LABELS,key=lambda l:amap[l]);pages=[];supp=[];audit=[];camera_records={}
    write_json(dest/"camera_policy.json",CAMERA_POLICY,freeze=True)
    write_json(dest/"private_method_mapping.json",amap,freeze=True)
    for cid in sorted({q["condition_id"] for q in qs}):
        group=sorted([q for q in qs if q["condition_id"]==cid],key=lambda q:q["candidate"])
        if len(group)!=4 or {q["candidate"] for q in group}!={0,1,6,7}:raise ValueError("All four noises required")
        bounds=bundle["cameras"][cid]["bounds"]
        overflow=False
        for q in group:
            for label in LABELS:
                a=compare_bounds(loaded[label,q["request_id"]],bounds);overflow|=a["outside_original_bounds"]
                audit.append(dict(condition_id=cid,request_id=q["request_id"],label=label,**a))
        points=np.concatenate([loaded[l,q["request_id"]].reshape(-1,3) for q in group for l in LABELS])
        joint=[points.min(0).tolist(),points.max(0).tolist()]
        camera_records[cid]=dict(primary_bounds=bounds,primary_views=VIEWS,primary_margin=1.08,
                                supplement_required=overflow,supplement_joint_bounds=joint if overflow else None,
                                scope="Same bounds across all three methods and all four noises within condition",
                                original_main_camera=bundle["cameras"][cid])
        for q in group:
            rid=q["request_id"]
            columns=[("Input C",bundle["inputs"][rid],q["K"])]+[(amap[l],loaded[l,rid],q["K"]) for l in ordered]
            for supplemental,bb in ((False,bounds),(True,joint)):
                if supplemental and not overflow:continue
                for mode in ("colored","neutral"):
                    folder=dest/("supplement_complete" if supplemental else "fixed_main")
                    path=folder/mode/(rid+".png")
                    frame="complete joint bounds" if supplemental else "fixed main bounds"
                    if overflow and not supplemental:frame+=" | FRAMING LIMIT: see complete supplement"
                    title=q["parent_id"]+" | C="+str(q["ratio"])+" | n"+str(q["candidate"])+" | "+frame+" | "+mode
                    identity=dict(version=VERSION,request_id=rid,mode=mode,supplemental=supplemental,bounds=bb,
                                  camera_policy=CAMERA_POLICY,renderer_sha256=digest(REPO/"tools/evaluate_schedule40.py"),
                                  method_mapping=amap,inputs_sha256=q["input_sha256"],
                                  triangle_hashes={l:ev.array_sha(loaded[l,rid]) for l in LABELS})
                    receipt=render_registered_page(path,columns,bb,title,mode=="neutral",identity,renderer)
                    item=dict(receipt,request_id=rid,condition_id=cid,candidate=q["candidate"],mode=mode,
                              primary_framing_limit=overflow and not supplemental,supplemental=supplemental)
                    (supp if supplemental else pages).append(item)
    if len(pages)!=2*len(qs):raise AssertionError("Incomplete primary gallery")
    affected=sum(c["supplement_required"] for c in camera_records.values())
    if len(supp)!=8*affected:raise AssertionError("Incomplete supplemental conditions")
    write_json(dest/"cameras.json",camera_records,freeze=True)
    write_json(dest/"camera_visibility_audit.json",dict(scope="Visualization framing only; not a geometric-quality metric",rows=audit,
                                                      affected_conditions=affected,policy=CAMERA_POLICY),freeze=True)
    manifest=dict(status="COMPLETE",version=VERSION,labels=list(LABELS),pages=pages,supplemental_pages=supp,
                  page_count=len(pages),expected_pages=2*len(qs),supplemental_page_count=len(supp),
                  affected_conditions=affected,all_registered_outputs_shown=True,five_views_per_model=True,
                  model_labels_blinded=True,AI_VISUAL_REVIEW="PENDING",human_usability="NOT_RUN")
    write_json(dest/"gallery.json",manifest,freeze=True)
    links=[]
    for title,items in (("Fixed main camera / primary",pages),("Complete views / affected conditions only",supp)):
        links.append("<h2>"+title+"</h2><ul>")
        for r in items:
            rel=Path(r["path"]).relative_to(dest).as_posix()
            links.append('<li><a href="'+html.escape(rel)+'">'+html.escape(r["request_id"]+" / "+r["mode"])+"</a></li>")
        links.append("</ul>")
    body='<!doctype html><meta charset="utf-8"><title>Complete curriculum comparison</title><h1>All three methods, all fixed requests</h1><p>Primary pages retain original main cameras. Flagged conditions have complete supplemental views with shared joint bounds. No new samples or scores.</p>'+"\n".join(links)
    index=dest/"INDEX.html"
    if index.exists() and index.read_text(encoding="utf-8")!=body:raise ValueError("Existing comparison HTML differs")
    if not index.exists():index.write_text(body,encoding="utf-8")
    return manifest

def compare(out):
    bundle=collect(out);dest=bundle["out"]/"comparisons/curriculum";dest.mkdir(parents=True,exist_ok=True)
    registration=dict(version=VERSION,labels=list(LABELS),requests=64,comparison_outputs=192,
                      baseline_reused_outputs=64,new_curriculum_outputs=128,camera_policy=CAMERA_POLICY,
                      input_file_hashes=bundle["input_file_hashes"],source_hashes=bundle["source_hashes"],
                      rows=bundle["rows"],model_calls=0,new_scoring_calls=0,new_queries=0,new_noise=0)
    write_json(dest/"comparison_manifest.json",registration,freeze=True)
    write_json(dest/"saved_scores.json",dict(rows=bundle["saved_scores"],new_scoring_calls=0),freeze=True)
    manifest=render_bundle(bundle,dest/"gallery")
    for p,h in bundle["input_file_hashes"].items():check_hash(p,h)
    if source_hashes(bundle["out"])!=bundle["source_hashes"]:raise ValueError("Source changed during comparison")
    receipt=dict(status="COMPLETE",version=VERSION,comparison_outputs=192,primary_pngs=128,
                 supplemental_pngs=manifest["supplemental_page_count"],affected_conditions=manifest["affected_conditions"],
                 baseline_model_calls=0,baseline_scoring_calls=0,total_model_calls=0,new_scoring_calls=0,new_queries=0,
                 original_inputs_and_sources_unchanged=True,manifest_sha256=digest(dest/"comparison_manifest.json"),
                 gallery_sha256=digest(dest/"gallery/gallery.json"),source_sha256=digest(__file__))
    write_json(dest/"completion.json",receipt,freeze=True)
    return receipt
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True);a=p.parse_args(argv)
    import json
    print(json.dumps(compare(a.out)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
