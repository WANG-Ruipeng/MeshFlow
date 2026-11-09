#!/usr/bin/env python3
"""Freeze and merge original alias-only confirmation AI review notes.

Metadata/hashes only. No model, rendering, scoring, image rewriting or generated
ratings. The private mapping is opened only after all three frozen reviews pass.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path

VERSION="recipe_confirmation_visual_merge_v1"
REQUEST_SCHEMA="recipe_confirmation_initial_visual_request_v1"
CROSS_SCHEMA="recipe_confirmation_cross_noise_initial_v1"
FREEZE_SCHEMA="recipe_confirmation_initial_visual_freeze_v1"
MANIFEST_SCHEMA="recipe_confirmation_anonymous_review_manifest_v1"
DEFAULT_OUT="/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1"
ALIASES=("Method 1","Method 2")
LABELS={"C20_40_MIX","R2_MIX_H_ALL"}
REVIEWERS=("root","schedule40_data","schedule40_train")
COUNTS={"root":dict(requests=16,outputs=32,images=32),
        "schedule40_data":dict(requests=24,outputs=48,images=48),
        "schedule40_train":dict(requests=24,outputs=48,images=48)}
PREFIXES={"root":("33ec","86ec"),"schedule40_data":("341b","bdd","866e"),"schedule40_train":("534b","c1b","f596")}

def need(ok,message):
    if not ok:raise ValueError(message)

def resolve(value):
    value=str(value).replace("\\","/")
    if os.name!="nt"and len(value)>2 and value[1]==":":value="/mnt/"+value[0].lower()+value[2:]
    if os.name=="nt"and value.startswith("/mnt/"):value=value[5].upper()+":/"+value[7:]
    return Path(value)

def digest(path):
    h=hashlib.sha256()
    with resolve(path).open("rb")as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))

def dump(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n"

def immutable(path,value):
    path=resolve(path);data=value.encode("utf8")if isinstance(value,str)else value
    if path.exists():need(path.read_bytes()==data,"Refuse changed immutable output: "+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb")as f:f.write(data)

def utc(value):
    need(isinstance(value,str),"UTC timestamp required")
    try:stamp=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:raise ValueError("Invalid UTC timestamp")from exc
    need(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0,"Timezone must explicitly be UTC")
    return stamp

def nonempty(value,field):
    need(isinstance(value,str)and bool(value.strip()),"Missing original text/evidence: "+field)

def claims(doc,reviewer):
    need(doc.get("reviewer")==reviewer,"Wrong reviewer")
    need(doc.get("prior_main_and_course_knowledge")is True,"Prior evidence knowledge must be disclosed")
    need(doc.get("private_mapping_read")is False and doc.get("confirmation_quality_read")is False,
         "Initial review must precede mapping and quality reads")

def validate_manifest(safe):
    need(safe.get("schema")==MANIFEST_SCHEMA,"Wrong safe manifest schema")
    need(safe.get("registered_requests")==64 and safe.get("registered_unique_outputs")==128
         and safe.get("primary_pages")==128 and safe.get("supplemental_pages")==0,"Wrong fixed review size")
    need(safe.get("display_aliases")==list(ALIASES),"Exactly two displayed aliases")
    rows=safe["rows"];by=defaultdict(list);paths=set()
    need(len(rows)==128,"Exactly128 original PNG entries")
    for row in rows:
        by[row["request_id"]].append(row)
        path=str(resolve(row["path"]).resolve())
        need(path not in paths,"Duplicate registered PNG");paths.add(path)
        owner=row.get("reviewer")
        need(owner in REVIEWERS and any(row["parent_id"].startswith(p)for p in PREFIXES[owner]),
             "Registered reviewer assignment differs")
        need(row.get("supplemental")is False and row.get("display_aliases")==list(ALIASES),"Unexpected supplement or alias")
    need(len(by)==64,"Exactly64 registered requests")
    conditions=defaultdict(set);parents=set()
    for rid,pages in by.items():
        need(len(pages)==2 and {p["mode"]for p in pages}=={"colored","neutral"},"Both modes exactly once")
        for field in("parent_id","condition_id","ratio","condition_cohort","candidate","reviewer"):
            need(len({p[field]for p in pages})==1,"Image request metadata differs: "+field)
        q=pages[0];parents.add(q["parent_id"]);conditions[q["parent_id"],q["ratio"]].add(q["candidate"])
    need(len(parents)==8 and set(conditions)=={(p,r)for p in parents for r in(.2,.4)}
         and all(v=={8,9,10,11}for v in conditions.values()),"Complete8parent x2ratio x4noise required")
    need(Counter(p["reviewer"]for p in rows)=={r:COUNTS[r]["images"]for r in REVIEWERS},"Reviewer coverage differs")
    return dict(by)

def validate_request(doc,expected,reviewer,cutoff):
    need(doc.get("schema")==REQUEST_SCHEMA,"Wrong initial request-note schema")
    claims(doc,reviewer);need(doc.get("AI_reviewer")is True,"AI review declaration required")
    q=expected[0]
    for key in("request_id","parent_id","condition_id","ratio","candidate"):
        need(doc.get(key)==q[key],"Original note request differs: "+key)
    images=doc.get("images",[]);expected_by={p["mode"]:p for p in expected}
    need(len(images)==2 and {i["mode"]for i in images}==set(expected_by),"Both actual viewed images required")
    for image in images:
        registered=expected_by[image["mode"]]
        need(resolve(image["path"]).resolve()==resolve(registered["path"]).resolve()
             and image["sha256"]==registered["sha256"],"Recorded image path/hash differs")
        need(image.get("actually_viewed")is True and image.get("all_five_views_examined")is True,
             "Actual image/all five views claim missing")
        for key in("view_tool","view_evidence"):nonempty(image.get(key),key)
        need(utc(image.get("viewed_at_utc"))<=cutoff,"Image viewing occurs after initial freeze")
        need(digest(image["path"])==image["sha256"],"Actual original PNG bytes changed")
    notes=doc.get("notes",[])
    need(len(notes)==2 and {n.get("display_alias")for n in notes}==set(ALIASES),"Each alias exactly once")
    for note in notes:
        nonempty(note.get("observation"),"observation");nonempty(note.get("uncertainty"),"uncertainty")
        for key in("model_label","true_label","recipe"):
            need(key not in note,"Initial observations must use displayed aliases only")
        if "request_id"in note:need(note["request_id"]==q["request_id"],"Note inner request mismatch")
    return notes

def validated_review(run,safe,manifest_sha,reviewer,cutoff):
    by=validate_manifest(safe)
    wanted={rid:pages for rid,pages in by.items()if pages[0]["reviewer"]==reviewer}
    folder=run/"audit/visual_review/confirm"/reviewer
    paths=sorted((folder/"requests").glob("*.json"))
    need({p.stem for p in paths}==set(wanted)and len(paths)==len(wanted),"Reviewer missing/extra request files")
    normalized=[];originals=[];images=[];sources={}
    for path in paths:
        before=digest(path);doc=read(path)
        notes=validate_request(doc,wanted[path.stem],reviewer,cutoff)
        need(digest(path)==before,"Initial note changed during validation")
        sources[str(path.relative_to(folder))]=before
        originals.append(dict(path=str(path),sha256=before,original_document=doc))
        images.extend(dict(i,request_id=doc["request_id"],reviewer=reviewer,source_note_path=str(path),
                           source_note_sha256=before)for i in doc["images"])
        for note in notes:
            normalized.append(dict(request_id=doc["request_id"],parent_id=doc["parent_id"],
                condition_id=doc["condition_id"],ratio=doc["ratio"],candidate=doc["candidate"],
                display_alias=note["display_alias"],reviewer=reviewer,observation=note["observation"],
                uncertainty=note["uncertainty"],original_note=note,original_request_document=doc,
                source_note_path=str(path),source_note_sha256=before,original_images=doc["images"],
                prior_main_and_course_knowledge=True,AI_reviewer=True,human_independent_blind_study=False))
    cross_path=folder/"cross_noise_initial.json";before=digest(cross_path);cross=read(cross_path)
    need(cross.get("schema")==CROSS_SCHEMA,"Wrong cross-noise original schema");claims(cross,reviewer)
    entries=cross.get("observations",[]);parents={p[0]["parent_id"]for p in wanted.values()}
    need(len(entries)==len(parents)and {e["parent_id"]for e in entries}==parents,"Cross-noise parent coverage differs")
    for entry in entries:
        nonempty(entry.get("observation"),"cross-noise observation")
        nonempty(entry.get("uncertainty"),"cross-noise uncertainty")
    need(digest(cross_path)==before,"Cross-noise note changed during validation")
    sources["cross_noise_initial.json"]=before
    need(len(normalized)==COUNTS[reviewer]["outputs"]and len(images)==COUNTS[reviewer]["images"],"Reviewer totals differ")
    return dict(reviewer=reviewer,counts=COUNTS[reviewer],source_review_manifest_sha256=manifest_sha,
        request_file_sha256=sources,rows=normalized,original_request_documents=originals,
        cross_noise_source=dict(path=str(cross_path),sha256=before,original_document=cross,
                               original_text=cross_path.read_text(encoding="utf-8-sig")),images=images)

def root(value):
    p=resolve(value).resolve()
    need(p.name=="chair_condition_recipe_factorial_v1"and p.parent.name=="experiments","Wrong run root")
    return p

def safe_manifest(run):
    path=run/"audit/visual_review/confirm/review_manifest.json"
    sha=digest(path);safe=read(path);validate_manifest(safe)
    need(digest(path)==sha,"Safe manifest changed during read")
    # Hash gallery bytes only. Never parse mapping/page model identities here.
    gallery=run/"phases/confirm/gallery/gallery.json"
    need(digest(gallery)==safe["source_gallery_sha256"],"Original gallery registration changed")
    registration=read(run/"phases/confirm/registration.json")
    need(digest(registration["requests_path"])==safe["source_requests_sha256"]==registration["requests_sha256"],
         "Frozen review requests changed")
    return path,sha,safe

def freeze_review(out,reviewer):
    run=root(out);need(reviewer in REVIEWERS,"Unknown reviewer")
    path,sha,safe=safe_manifest(run)
    receipt_path=run/"audit/visual_review/confirm"/reviewer/"initial_freeze_receipt.json"
    old=read(receipt_path)if receipt_path.exists()else None
    stamp=old["initial_frozen_at_utc"]if old else datetime.now(timezone.utc).isoformat()
    pack=validated_review(run,safe,sha,reviewer,utc(stamp))
    receipt=dict(schema=FREEZE_SCHEMA,status="FROZEN_INITIAL_REVIEW",reviewer=reviewer,
        initial_frozen_at_utc=stamp,source_review_manifest_sha256=sha,counts=pack["counts"],
        request_file_sha256=pack["request_file_sha256"],actual_image_evidence_entries=len(pack["images"]),
        original_notes_saved_before_unblinding=True,private_mapping_read=False,confirmation_quality_read=False,
        prior_main_and_course_knowledge=True,AI_reviewer=True,human_independent_blind_study=False,
        source_tool_sha256=digest(__file__),
        viewing_evidence_basis="Reviewer recorded actual tool viewing, timestamps and original notes; not independent proof of human viewing")
    immutable(receipt_path,dump(receipt))
    return dict(status="COMPLETE",stage="freeze-review",reviewer=reviewer,receipt_path=str(receipt_path),
        receipt_sha256=digest(receipt_path),**pack["counts"],private_mapping_opened=False)

def validate_frozen_reviews(run,safe,manifest_sha):
    packs=[];receipts={}
    for reviewer in REVIEWERS:
        path=run/"audit/visual_review/confirm"/reviewer/"initial_freeze_receipt.json"
        receipt=read(path)
        need(receipt.get("schema")==FREEZE_SCHEMA and receipt.get("status")=="FROZEN_INITIAL_REVIEW","Initial freeze receipt missing/invalid")
        claims(receipt,reviewer)
        need(receipt.get("original_notes_saved_before_unblinding")is True and receipt.get("AI_reviewer")is True,"Missing pre-unblind freeze declaration")
        need(receipt.get("source_review_manifest_sha256")==manifest_sha,"Reviewer froze a different safe manifest")
        need(receipt.get("source_tool_sha256")==digest(__file__),"Initial freeze adapter source changed")
        pack=validated_review(run,safe,manifest_sha,reviewer,utc(receipt["initial_frozen_at_utc"]))
        need(receipt.get("counts")==pack["counts"]and receipt.get("request_file_sha256")==pack["request_file_sha256"],
             "Original notes changed after initial freeze")
        need(receipt.get("actual_image_evidence_entries")==len(pack["images"]),"Frozen viewing evidence count differs")
        packs.append(pack);receipts[str(path)]=digest(path)
    rows=[r for p in packs for r in p["rows"]];images=[i for p in packs for i in p["images"]]
    expected={(r["request_id"],alias)for r in safe["rows"]for alias in ALIASES}
    need(len(rows)==128 and {(r["request_id"],r["display_alias"])for r in rows}==expected,"Complete128 unique outputs required")
    need(len(images)==128 and len({str(resolve(i["path"]).resolve())for i in images})==128,"Complete128 unique PNGs required")
    return packs,receipts

def csv_text(rows):
    stream=io.StringIO(newline="")
    keys=list(dict.fromkeys(k for row in rows for k in row))
    w=csv.DictWriter(stream,fieldnames=keys,lineterminator="\n");w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True)if isinstance(v,(dict,list))else v for k,v in row.items()})
    return stream.getvalue()

def merge(out):
    run=root(out);path,manifest_sha,safe=safe_manifest(run)
    # This sequence is intentional: do not open mapping until all three initial
    # receipts, original source hashes, image evidence and panel counts pass.
    packs,receipt_hashes=validate_frozen_reviews(run,safe,manifest_sha)
    mapping_path=run/"phases/confirm/gallery/private_method_mapping.json"
    mapping=read(mapping_path)
    need(set(mapping)==LABELS and set(mapping.values())==set(ALIASES)and len(mapping)==2,"Exact two-alias mapping required")
    alias_to_label={alias:label for label,alias in mapping.items()}
    rows=sorted([dict(r,model_label=alias_to_label[r["display_alias"]])for p in packs for r in p["rows"]],
                key=lambda r:(r["parent_id"],r["ratio"],r["candidate"],r["display_alias"]))
    images=sorted([i for p in packs for i in p["images"]],key=lambda i:(i["request_id"],i["mode"]))
    original=[d for p in packs for d in p["original_request_documents"]]
    cross=[p["cross_noise_source"]for p in packs]
    # Recheck frozen sources after constructing the output; never edit originals.
    for source in original+cross:need(digest(source["path"])==source["sha256"],"Original source changed during merge")
    for receipt,sha in receipt_hashes.items():need(digest(receipt)==sha,"Initial freeze receipt changed during merge")
    need(digest(path)==manifest_sha,"Safe manifest changed during merge")
    sources={str(path):manifest_sha,str(mapping_path):digest(mapping_path),**receipt_hashes}
    sources.update({s["path"]:s["sha256"]for s in original+cross})
    coverage=dict(schema=VERSION,status="PASS",requests=64,unique_outputs=128,primary_pngs=128,supplemental_pngs=0,
        aliases=list(ALIASES),alias_to_label=alias_to_label,per_reviewer={p["reviewer"]:p["counts"]for p in packs},
        all_image_bytes_sha256_verified=True,all_recorded_viewing_evidence_present=True,
        viewing_evidence_basis="AI reviewer records and tool-viewing declarations; not independently verified human viewing",
        original_notes_frozen_before_mapping_opened_by_merge=True,all_original_text_and_fields_preserved=True,
        source_file_sha256=sources,prior_main_and_course_knowledge=True,AI_review=True,
        human_independent_blind_study=False,numeric_subjective_scores_created=False,automatic_ranking=False,
        quality_tables_read_by_merge=False,new_model_calls=0,new_render_calls=0,new_scoring_calls=0)
    dest=run/"audit/visual_review/confirm_complete"
    payload=dict(schema=VERSION,rows=rows,original_request_documents=original,
        original_cross_noise_sources=cross,coverage=coverage)
    outputs={"per_output.json":dump(payload),"per_output.csv":csv_text(rows),
        "coverage.json":dump(coverage),"image_coverage.json":dump(dict(rows=images,primary_count=128)),
        "cross_noise_initial_sources.json":dump(dict(rows=cross,original_wording_preserved=True))}
    outputs["README.md"]="""# Complete confirmation visual review / ??????????

64 requests ? two displayed methods =128 unique outputs; both colored and neutral modes =128 originalPNG images. Root covers32 outputs/32PNG; each other reviewer covers48/48. Each output is counted once, not once per image.
64??????=128???????????????128???????????????????????

All three initial reviews were hash-frozen before this adapter opened the private mapping. Original per-request fields, uncertainty and cross-noise wording are preserved without rewriting. Image bytes, note hashes and recorded tool-viewing evidence are checked; these records are AI-review evidence, not independent proof of human viewing.
??????????????????????????????????????????????????????????????

All reviewers had prior main/course knowledge. No subjective numeric score, ranking or quality-table lookup was created. Images do not certify exact intersections, original vertex connectivity or semantic seat usability.
?????????????????????????????????????????????????????????
"""
    for name,value in outputs.items():immutable(dest/name,value)
    return dict(status="COMPLETE",stage="merge",output_dir=str(dest),requests=64,unique_outputs=128,images=128,
        output_sha256={name:digest(dest/name)for name in outputs},quality_tables_read=False,
        numeric_subjective_scores_created=False,new_model_calls=0,new_render_calls=0,new_scoring_calls=0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",default=DEFAULT_OUT)
    p.add_argument("--stage",required=True,choices=("freeze-review","merge"));p.add_argument("--reviewer",choices=REVIEWERS)
    a=p.parse_args(argv)
    if a.stage=="freeze-review":need(a.reviewer is not None,"--reviewer required for freeze-review")
    result=freeze_review(a.out,a.reviewer)if a.stage=="freeze-review"else merge(a.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
