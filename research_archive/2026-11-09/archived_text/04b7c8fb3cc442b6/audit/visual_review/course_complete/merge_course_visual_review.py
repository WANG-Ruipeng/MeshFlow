#!/usr/bin/env python3
"""Merge frozen course AI visual notes; no scoring, rendering, model calls or edits."""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, re
from pathlib import Path

SAFE_SHA = "292711f60899a223c3ad1b7689824c76518b67711a852a067ff1a5e432addc97"
ALIASES = {"Method 1", "Method 2", "Method 3"}
REVIEWERS = ("root", "schedule40_data", "schedule40_train")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def need(ok, message):
    if not ok:
        raise ValueError(message)

def dump(x):
    return json.dumps(x, ensure_ascii=False, indent=2) + "\n"

def immutable(p, data):
    p = Path(p)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if p.exists():
        need(p.read_bytes() == data, "Refuse overwriting changed delivery: " + str(p))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write(data)

def validate_unique(rows, expected):
    pairs = [(r["request_id"], r["display_alias"]) for r in rows]
    need(len(pairs) == len(set(pairs)), "Duplicate request/alias")
    need(set(pairs) == {(r,a) for r in expected for a in ALIASES}, "Incomplete or unexpected output set")

def validate_pages(actual, expected):
    need(len(actual) == len({i["path"] for i in actual}), "Duplicate image in request")
    need({i["path"] for i in actual} == {i["path"] for i in expected}, "Missing or unexpected image")
    e = {i["path"]:i for i in expected}
    for i in actual:
        need(all(i[k] == e[i["path"]][k] for k in ("sha256","request_id","condition_id","candidate","mode","supplemental")), "Image identity mismatch")

def validate_claims(reviewer, document, row):
    if reviewer == "root":
        for key in ("all_five_views_reviewed","colored_reviewed","neutral_reviewed","all_supplements_reviewed"):
            need(row.get(key) is True, "Root actual-view claim absent: " + key)
        need(row.get("model_mapping_accessed") is False and row.get("course_quality_tables_accessed") is False, "Root pre-unblind claims")
    elif reviewer == "schedule40_train":
        need(row.get("all_assigned_views_actually_displayed") is True, "Train actual-view claim absent")
        need(set(row.get("reviewed_views",[])) == {"front_3q","back_3q","front","side","wire"}, "Train view set")
        need(row.get("course_method_mapping_read") is False and row.get("course_quality_metrics_read") is False, "Train pre-unblind claims")
    else:
        need(document.get("status") == "ALL_REGISTERED_ORIGINAL_PAGES_VIEWED", "Data incomplete request")
        need(row.get("course_mapping_accessed") is False and row.get("course_quality_accessed") is False, "Data pre-unblind claims")
        for i in document["images"]:
            need(i.get("original_png_actually_viewed") is True and i.get("all_five_views_examined") is True, "Data actual-view claim absent")

def self_test():
    rs=[{"request_id":"q","display_alias":a}for a in sorted(ALIASES)]
    validate_unique(rs,{"q"})
    failures=0
    for invalid in [rs[:2],rs+[rs[0]],[dict(r,request_id="x")for r in rs]]:
        try:validate_unique(invalid,{"q"})
        except ValueError:failures+=1
    need(failures==3,"coverage rejection tests")
    image={"path":"p","sha256":"h","request_id":"q","condition_id":"C","candidate":0,"mode":"colored","supplemental":False}
    validate_pages([image],[image])
    failures=0
    for invalid in [[],[dict(image,sha256="bad")],[image,image]]:
        try:validate_pages(invalid,[image])
        except ValueError:failures+=1
    need(failures==3,"page rejection tests")
    return {"status":"PASS","positive_cases":2,"negative_cases":6}

def build(run):
    run=Path(run).resolve()
    out=run/"audit/visual_review/course_complete"
    out.mkdir(parents=True,exist_ok=True)
    safe_path=run/"audit/visual_review/course/review_manifest.json"
    need(sha(safe_path)==SAFE_SHA,"Safe manifest changed")
    safe=read(safe_path)
    page_by_request=collections.defaultdict(list)
    for p in safe["rows"]:page_by_request[p["request_id"]].append(p)
    need(len(safe["rows"])==184 and len(page_by_request)==64,"Registered panel size")
    need(sum(not p["supplemental"]for p in safe["rows"])==128,"Registered primary size")
    assignment_path=run/"registrations/curriculum_visual_review_assignment.json"
    assignment=read(assignment_path)
    source_hashes={str(safe_path):sha(safe_path),str(assignment_path):sha(assignment_path)}
    request_docs=[]
    normalized=[]
    seen_requests=set()
    review_counts={}
    for reviewer in REVIEWERS:
        base=run/"audit/visual_review/course"/reviewer
        if reviewer=="schedule40_train":
            paths=sorted((base/"requests").glob("*.json"))
        else:
            paths=sorted(base.glob("*.json"))
        counts=collections.Counter()
        for p in paths:
            d=read(p)
            if reviewer=="root":
                original_rows=d.get("notes")
            elif reviewer=="schedule40_data":
                original_rows=d.get("rows") if d.get("schema")=="course_anonymous_initial_review_v1" else None
            else:
                original_rows=d.get("rows")
            if not original_rows:continue
            need(len(original_rows)==3,"Exactly three aliases per request")
            rid=d.get("request_id",original_rows[0]["request_id"])
            need(rid in page_by_request and rid not in seen_requests,"Unknown or duplicate request")
            seen_requests.add(rid)
            images=d["images"]
            validate_pages(images,page_by_request[rid])
            cid=images[0]["condition_id"]
            # Parent IDs have historical variable hex lengths; use frozen assignment as authority.
            matching=[parent for parent in assignment["reviewers"][reviewer] if cid.startswith(parent+"_")]
            need(len(matching)==1,"Parent outside reviewer assignment")
            parent=matching[0]
            ratio=0.4 if cid.endswith("_C40") else 0.2
            need(cid in {parent+"_C40",parent+"_C1"},"Unknown condition convention")
            candidate=images[0]["candidate"]
            need(candidate in [0,1,6,7],"Unexpected noise")
            need(rid.endswith("__n"+str(candidate)),"Noise/request mismatch")
            aliases=[]
            for r in original_rows:
                alias=r.get("display_alias",r["method_alias"])
                aliases.append(alias)
                need(r["request_id"]==rid and alias in ALIASES,"Row identity")
                if "parent_id"in r:need(r["parent_id"]==parent,"Row parent mismatch")
                if "ratio"in r:need(r["ratio"]==ratio,"Row ratio mismatch")
                validate_claims(reviewer,d,r)
                normalized.append({
                    "request_id":rid,"parent_id":parent,"condition_id":cid,"ratio":ratio,
                    "noise_index":candidate,"display_alias":alias,"reviewer":reviewer,
                    "observation":r.get("observation",""),
                    "organization":r.get("organization",""),
                    "missing_or_collapsed_regions":r.get("missing_or_collapsed_regions",""),
                    "visible_defects":r.get("visible_defects",""),
                    "isolated_or_duplicate_surfaces":r.get("isolated_or_duplicate_surfaces",""),
                    "C_relation":r.get("C_relation",r.get("C_incorporation","")),
                    "uncertainty":r.get("uncertainty",r.get("limits","")),
                    "common_observation":r.get("common_observation",""),
                    "condition_note":d.get("condition_note",""),
                    "within_request_comparison":d.get("within_request_comparison",""),
                    "supplement_observation":d.get("supplement_observation",""),
                    "primary_png_count":sum(not i["supplemental"]for i in images),
                    "supplemental_png_count":sum(i["supplemental"]for i in images),
                    "original_images":images,"original_row":r,
                    "source_note_path":str(p),"source_note_sha256":sha(p),
                    "initial_observation_saved_before_course_unblind":True,
                    "prior_main_evidence_seen":True,"human_study":False,
                    "subjective_numeric_score":None})
            need(set(aliases)==ALIASES and len(set(aliases))==3,"Alias coverage")
            source_hashes[str(p)]=sha(p)
            request_docs.append({"source_note_path":str(p),"source_note_sha256":sha(p),"original_document":d})
            counts.update({"requests":1,"outputs":3,"primary_pngs":sum(not i["supplemental"]for i in images),"supplemental_pngs":sum(i["supplemental"]for i in images)})
        expected_requests=len(assignment["reviewers"][reviewer])*8
        need(counts["requests"]==expected_requests,"Reviewer incomplete: "+reviewer)
        review_counts[reviewer]=dict(counts)
    validate_unique(normalized,set(page_by_request))
    need(len(normalized)==192,"192 unique outputs")
    all_images={i["path"]:i for d in request_docs for i in d["original_document"]["images"]}
    need(set(all_images)=={i["path"]for i in safe["rows"]},"Global image coverage")
    for image in safe["rows"]:
        need(sha(image["path"])==image["sha256"],"PNG bytes changed: "+image["path"])
    # Completed initial-note coverage and all anonymous claims are checked BEFORE mapping is loaded.
    mapping_path=run/"comparisons/curriculum/gallery/private_method_mapping.json"
    mapping=read(mapping_path)
    need(set(mapping)=={"R2_MIX_H_ALL","C20_40_MIX","C40_20_MIX"} and set(mapping.values())==ALIASES,"Mapping contract")
    alias_to_label={v:k for k,v in mapping.items()}
    for row in normalized:row["model_label"]=alias_to_label[row["display_alias"]]
    normalized.sort(key=lambda r:(r["parent_id"],r["ratio"],r["noise_index"],r["display_alias"]))
    source_hashes[str(mapping_path)]=sha(mapping_path)
    cross_sources=[]
    for rel in ("audit/visual_review/course/root/cross_noise_initial.json",
                "audit/visual_review/course/schedule40_data/cross_noise_initial.json",
                "audit/visual_review/course/schedule40_train/VISUAL_REVIEW.md",
                "audit/visual_review/course/schedule40_train/initial_freeze_receipt.json"):
        p=run/rel
        source_hashes[str(p)]=sha(p)
        cross_sources.append({"path":str(p),"sha256":sha(p),"original_text":p.read_text(encoding="utf-8-sig")})
    payload={"schema":"recipe_course_visual_review_merged_v1","initial_notes_preserved":True,
        "mapping_read_only_after_initial_notes_completed":True,"alias_to_label":alias_to_label,
        "human_study":False,"numeric_subjective_scores_created":False,
        "prior_main_study_evidence_seen_by_reviewers":True,"rows":normalized,
        "original_request_documents":request_docs,"original_cross_noise_sources":cross_sources}
    immutable(out/"per_output.json",dump(payload))
    f=io.StringIO(newline="")
    fields=list(normalized[0])
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
    writer.writeheader()
    writer.writerows({k:json.dumps(v,ensure_ascii=False,separators=(",",":")) if isinstance(v,(dict,list))else v for k,v in r.items()}for r in normalized)
    immutable(out/"per_output.csv",f.getvalue())
    image_rows=[]
    for i in safe["rows"]:
        image_rows.append(dict(i,bytes_sha256_verified=True,reviewer=next(r["reviewer"]for r in normalized if r["request_id"]==i["request_id"]),actual_view_claim_source=next(r["source_note_path"]for r in normalized if r["request_id"]==i["request_id"])))
    immutable(out/"image_coverage.json",dump({"schema":"course_png_coverage_v1","rows":image_rows,"primary_count":128,"supplemental_count":56}))
    coverage={"schema":"recipe_course_visual_review_coverage_v1","status":"PASS","requests":64,
        "unique_outputs":192,"reused_R2_outputs":64,"new_course_outputs":128,
        "parent_count":8,"ratios":[0.2,0.4],"noise_indices":[0,1,6,7],
        "primary_pngs":128,"supplemental_pngs":56,"supplemented_conditions":len({i["condition_id"]for i in safe["rows"]if i["supplemental"]}),
        "actual_view_claims_verified":True,"all_image_bytes_sha256_verified":True,
        "per_reviewer":review_counts,"alias_to_label":alias_to_label,
        "source_file_sha256":source_hashes,"all_original_fields_and_text_preserved":True,
        "no_quality_tables_read_by_merge":True,"no_numeric_subjective_scores":True,
        "limits":["AI review, not an independent blinded human study; all reviewers had prior main-stage evidence.",
                  "Image observations do not certify exact intersections, duplication, topology, welding or C-coordinate preservation.",
                  "Input artifacts and open/slatted backs are not automatically free-generation defects.",
                  "Supplements repeat the same outputs at wider condition-wide joint bounds and are not extra samples.",
                  "No counts of subjective wins or pooled subjective quality ranking were created."]}
    immutable(out/"coverage.json",dump(coverage))
    lines=["# Cross-noise visual observations / ?????","",
        "Original initial-review wording is retained below. Aliases were assigned before unblinding. / ???????????????????","",
        "Course mapping: "+json.dumps(alias_to_label,ensure_ascii=False),""]
    for s in cross_sources[:3]:
        lines.extend(["## "+Path(s["path"]).parent.name,"","Source: "+s["path"],"","SHA256: "+s["sha256"],"",s["original_text"],""])
    immutable(out/"cross_noise_observations.md","\n".join(lines))
    readme="""# Complete course visual review / ????????

This merge preserves 192 unique outputs (3 labels ? 8 parents ? 2 ratios ? 4 fixed noises), 64 requests, 128 primary PNGs and all 56 supplementary PNGs. It adds model labels only after all initial alias notes were complete. The original per-request documents and all observation fields are embedded in per_output.json; per_output.csv retains original_row JSON plus readable rubric columns. Original reviewer directories remain untouched.

?192??????64????128????56????????????????????????????????????????CSV/JSON?????????????????SHA?????????????????????

This is an AI assistant review with prior main-study exposure, not an independent blinded human study. Main-gallery aliases differ from course aliases. Projection alone cannot establish welded attachment, topology, exact intersection or duplication. C's inherited defects and open/slatted styles are separately noted.

??????????AI??????????????????????????????????????C?????????/???????????????

Files: per_output.csv/json, image_coverage.json (184 actual PNG hashes), coverage.json, cross_noise_observations.md, merge_course_visual_review.py, merge_receipt.json. No models, scoring, inference, rendering or OT is performed by this merger.
"""
    immutable(out/"README.md",readme)
    # Reverify original sources after writing the independent merge.
    for p,h in source_hashes.items():need(sha(p)==h,"Source modified during merge")
    delivered=["per_output.json","per_output.csv","image_coverage.json","coverage.json","cross_noise_observations.md","README.md"]
    receipt={"schema":"recipe_course_visual_merge_receipt_v1","status":"PASS",
             "self_test":self_test(),"source_unchanged_after_merge":True,
             "script_sha256":sha(__file__),"files":{name:sha(out/name)for name in delivered},
             "requests":64,"unique_outputs":192,"primary_pngs":128,"supplemental_pngs":56,
             "model_calls":0,"score_calls":0,"render_calls":0,"OT_calls":0}
    immutable(out/"merge_receipt.json",dump(receipt))
    return receipt

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:print(dump(self_test()),end="")
    else:
        if args.run is None:ap.error("--run is required")
        print(dump(build(args.run)),end="")
