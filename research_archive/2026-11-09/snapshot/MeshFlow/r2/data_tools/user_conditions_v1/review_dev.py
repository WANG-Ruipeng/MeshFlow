"""Root AI_PROXY visual decisions after viewing all eight DEV candidate sheets."""
from pathlib import Path
from prepare import read,write,sha,DEFAULT_OUT
from quality import partition_condition
from construct import nested_extents,geometry
out=DEFAULT_OUT
parents={p["uid"]:p for p in read(out/"parent_inventory.json")["rows"]}
decisions={
"33ec":("seat","p04","medium","Flat upper circular cushion surface, distinct from the cylindrical side and surrounding seat frame."),
"341b":("seat","p05","full","Identifiable sitting surface, but 15 confirmed C/free penetration pairs occur at/near its junction; not a clean user-input demonstration."),
"534b":("back","p01","full","Seat-facing rectangular backrest plane consists of two original triangles; excludes rear plane, thickness rims and legs."),
"866e":("seat","p01","full","Two original triangles form the complete horizontal sitting upper surface, distinct from underside/apron."),
"86ec":("back","p01","full","Seat-facing continuous openwork backrest surface; includes the arched upper section and its connected front members, excludes side walls/rear legs."),
"bdd2":("back","p03","medium","Continuous seat-facing gently curved back panel; rear face and top thickness cap excluded."),
"c1b8":("seat","p00","small","Six original coplanar faces cover the sitting top; underside and narrow frame/leg faces excluded."),
"f596":("back","p01","small","Continuous tall seat-facing slightly curved back panel; rear face and side posts excluded."),
}
rows=[]
for uid,p in parents.items():
 if p["split"]!="dev":continue
 kind,tag,tier,note=decisions[uid[:4]]
 candidate=next(c for c in read(out/"candidates"/(uid+".json"))["candidates"] if c["candidate_id"]==tag)
 image=out/"gallery/proposals"/(uid+"__candidates.png")
 q=partition_condition(uid,candidate["source_face_ids"],out/"quality")
 rejected=uid.startswith("341b")
 ext,growth=nested_extents(geometry(p)["full_target"],geometry(p)["source_faces"],candidate["source_face_ids"],kind)
 rows.append(dict(uid=uid,split="dev",region_id=uid+"__"+kind,region_type=kind,
     source_face_ids=candidate["source_face_ids"],candidate_ids=[tag],dev_requested_tier=tier,
     status="NA_LOCAL_TARGET_CONFLICT" if rejected else "ACCEPTED_AI_PROXY",reason=note,
     reviewer="root AI_PROXY",human_semantic_annotation=False,
     visual_evidence=[dict(path=str(image),sha256=sha(image))],
     quality_at_R={k:dict(confirmed_pairs=v["confirmed_illegal_pair_count"],unknown_pairs=v["unknown_pair_count"]) for k,v in q["partitions"].items()},
     extent_preview=[dict(K=len(e["source_face_ids"]),tier=e["tier"],requested_tiers=e["requested_tiers"],area_fraction=e["region_area_fraction"]) for e in ext],
     notes=(["Full source has 88 confirmed source-pair conflicts and 1 unresolved pair.",
             "Alternative front-back p10 was examined; region is partial and has 1 C/free penetration. Not substituted merely to fill the quota.",
             "Old regression condition and both frozen noises remain in the regression panel; no old bad case removed."] if rejected else [])))
write(out/"reviews/dev_root.json",dict(schema="region_visual_review_v1",reviewer="root AI_PROXY",rows=rows,
     accepted_new_dev_parents=7,NA_new_dev_parents=1,
     semantics_note="AI geometry-informed visual review, not human ground truth; no model output consulted.",
     noise_created=0,training_updates=0))
print("DEV review saved: 7 accepted, 1 NA (341b local target conflict); old regression retains all eight.")
