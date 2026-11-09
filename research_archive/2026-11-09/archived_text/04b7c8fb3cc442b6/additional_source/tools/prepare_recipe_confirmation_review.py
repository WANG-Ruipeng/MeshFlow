#!/usr/bin/env python3
"""Create an allowlisted anonymous image review manifest; no render or scoring.

Do not open private_method_mapping.json or copy model labels from metadata.
The reviewer must freeze original alias-only observations before unblinding.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

VERSION="recipe_confirmation_anonymous_review_manifest_v1"
DEFAULT_OUT="/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1"
PREFIXES={"root":("33ec","86ec"),"schedule40_data":("341b","bdd","866e"),"schedule40_train":("534b","c1b","f596")}
NOISES={8,9,10,11}

def require(value,message):
    if not value:raise ValueError(message)

def resolve(value):
    value=str(value).replace("\\","/")
    if os.name!="nt"and len(value)>2 and value[1]==":":value="/mnt/"+value[0].lower()+value[2:]
    if os.name=="nt"and value.startswith("/mnt/"):value=value[5].upper()+":/"+value[7:]
    return Path(value)

def digest(path):
    h=hashlib.sha256()
    with resolve(path).open("rb")as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))

def reviewer(parent):
    matches=[r for r,prefixes in PREFIXES.items()if any(parent.startswith(p)for p in prefixes)]
    require(len(matches)==1,"Parent lacks unique frozen reviewer assignment")
    return matches[0]

def build_manifest(requests,gallery,gallery_root,requests_sha,gallery_sha):
    qby={q["request_id"]:q for q in requests}
    require(len(requests)==len(qby)==64,"Exactly64 registered requests")
    groups={}
    for q in requests:groups.setdefault((q["parent_id"],q["ratio"]),[]).append(q)
    parents={q["parent_id"]for q in requests}
    require(len(parents)==8 and set(groups)=={(p,r)for p in parents for r in(.2,.4)},"Eight parents at two ratios")
    require(all(len(qs)==4 and {q["candidate"]for q in qs}==NOISES for qs in groups.values()),"Confirmation n8 through n11 required")
    require(gallery.get("status")=="COMPLETE"and gallery.get("page_count")==128 and gallery.get("expected_pages")==128,"Complete128 primary pages required")
    require(gallery.get("all_registered_outputs_shown")is True and gallery.get("five_views_per_model")is True and gallery.get("model_labels_blinded")is True,"Rendered gallery contract differs")
    pages=gallery["pages"];seen=set();rows=[]
    gallery_root=resolve(gallery_root).resolve()
    for page in pages:
        key=page["request_id"],page["mode"]
        require(key[0]in qby and key[1]in("colored","neutral")and key not in seen,"Unknown/duplicate gallery page")
        seen.add(key);q=qby[key[0]]
        require(page["condition_id"]==q["condition_id"]and page["candidate"]==q["candidate"],"Image condition/noise mismatch")
        path=resolve(page["path"]).resolve()
        require(path.is_relative_to(gallery_root)and path.suffix.lower()==".png","Image must be original registered PNG inside gallery")
        require(path.is_file()and digest(path)==page["sha256"],"Actual original PNG hash mismatch")
        # Explicit allowlist: never spread page/request dictionaries into output.
        rows.append(dict(request_id=q["request_id"],parent_id=q["parent_id"],condition_id=q["condition_id"],
            ratio=q["ratio"],condition_cohort=q["condition_cohort"],candidate=q["candidate"],
            mode=page["mode"],path=str(path),sha256=page["sha256"],reviewer=reviewer(q["parent_id"]),
            display_aliases=["Method 1","Method 2"],supplemental=False,
            framing="Same union bounds across both methods and all four noises of this condition",
            actually_viewed=False))
    require(len(pages)==128 and seen=={(rid,mode)for rid in qby for mode in("colored","neutral")},"Missing registered image")
    counts=Counter(r["reviewer"]for r in rows)
    require(dict(counts)=={"root":32,"schedule40_data":48,"schedule40_train":48},"Reviewer page coverage differs")
    return dict(schema=VERSION,status="READY_FOR_ACTUAL_IMAGE_REVIEW",source_requests_sha256=requests_sha,
        source_gallery_sha256=gallery_sha,registered_requests=64,registered_unique_outputs=128,
        primary_pages=128,supplemental_pages=0,reviewer_primary_pages=dict(counts),
        reviewer_output_counts=dict(counts),display_aliases=["Method 1","Method 2"],rows=sorted(rows,key=lambda r:(r["request_id"],r["mode"])),
        private_mapping_opened=False,model_labels_copied=False,quality_tables_read=False,
        prior_main_and_course_knowledge=True,AI_reviewer=True,human_independent_blind_study=False,
        actual_review_complete=False,instructions=[
            "Actually inspect every assigned colored and neutral original PNG before claiming coverage.",
            "Record each Method 1/2 observation, source image path/hash, reviewer and request identity.",
            "Describe visible whole-chair organization, missing/collapsed/isolated/repeated surfaces, C incorporation and uncertainty; no synthetic score or ranking.",
            "Freeze initial alias-only notes before reading private mapping or confirmation quality tables.",
            "Disclose prior main/course knowledge and AI review; this is not an independent blinded human study.",
            "Do not infer preserved original vertex IDs or semantic seat usability from blue C preservation.",
            "Original images are unmodified; this helper does not render, crop, regenerate, score or run a model."])

def prepare(out=DEFAULT_OUT):
    out=resolve(out).resolve();require(out.name=="chair_condition_recipe_factorial_v1"and out.parent.name=="experiments","Wrong run")
    phase=out/"phases/confirm"
    paths={"registration":phase/"registration.json","gallery":phase/"gallery/gallery.json"}
    missing=[str(p)for p in paths.values()if not p.is_file()]
    if missing:return dict(status="PENDING",missing_files=missing,new_model_calls=0,new_render_calls=0)
    reg=read(paths["registration"])
    require(reg.get("kind")=="confirm"and reg.get("expected_outputs")==128 and len(reg.get("labels",[]))==2,"Wrong confirmation panel")
    paths["requests"]=resolve(reg["requests_path"])
    require(paths["requests"].is_file()and digest(paths["requests"])==reg["requests_sha256"],"Frozen request file changed")
    before={k:digest(p)for k,p in paths.items()}
    manifest=build_manifest(read(paths["requests"])["rows"],read(paths["gallery"]),phase/"gallery",before["requests"],before["gallery"])
    require(all(digest(paths[k])==h for k,h in before.items()),"Manifest inputs changed during read")
    dest=out/"audit/visual_review/confirm/review_manifest.json"
    if dest.exists():require(read(dest)==manifest,"Immutable anonymous manifest changed")
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x",encoding="utf8")as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write("\n")
    return dict(status="COMPLETE",manifest_path=str(dest),manifest_sha256=digest(dest),primary_pages=128,
        supplemental_pages=0,registered_unique_outputs=128,actually_viewed=0,private_mapping_opened=False,
        quality_tables_read=False,new_model_calls=0,new_render_calls=0,new_scoring_calls=0)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default=DEFAULT_OUT)
    result=prepare(parser.parse_args(argv).out);print(json.dumps(result),flush=True)
    return 0 if result["status"]=="COMPLETE"else 2

if __name__=="__main__":raise SystemExit(main())
