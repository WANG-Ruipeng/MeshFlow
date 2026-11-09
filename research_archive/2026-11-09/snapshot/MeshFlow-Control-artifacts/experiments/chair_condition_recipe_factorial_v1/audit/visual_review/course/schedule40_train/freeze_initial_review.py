import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
run=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
out=run/"audit/visual_review/course/schedule40_train"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=run/"audit/visual_review/course/review_manifest.json"
assert sha(manifest)=="292711f60899a223c3ad1b7689824c76518b67711a852a067ff1a5e432addc97"
parents=["534b0e6e2c53a8e1a7cdfd3874ed3723","c1b8c2f0c4f3cc43aff7c08b060f5ed6","f59627a077dcc2566c63b8b48495261a"]
m=json.loads(manifest.read_text())
assigned=[r for r in m["rows"] if any(p in r["request_id"] for p in parents)]
docs=[json.loads(p.read_text(encoding="utf-8-sig")) for p in sorted((out/"requests").glob("*.json"))]
rows=[r for d in docs for r in d["rows"]]
assert len(docs)==24 and len(rows)==72 and len(assigned)==72
assert len({(r["request_id"],r["method_alias"]) for r in rows})==72
assert all(r["all_assigned_views_actually_displayed"] and not r["course_method_mapping_read"] and not r["course_quality_metrics_read"] for r in rows)
expected={r["path"]:r for r in assigned}
observed={r["path"]:r for d in docs for r in d["images"]}
assert set(expected)==set(observed)
image_checks=[]
for name,r in expected.items():
    p=Path(name.replace("E:\\","/mnt/e/").replace("\\","/"))
    actual=sha(p)
    assert actual==r["sha256"]==observed[name]["sha256"]
    image_checks.append(dict(path=name,sha256=actual,supplemental=r["supplemental"],mode=r["mode"],request_id=r["request_id"],status="PASS"))
def write_json(name,data):
    p=out/name
    assert not p.exists(),p
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
write_json("observations.json",dict(schema="recipe_curriculum_initial_visual_notes_v1",reviewer="AI assistant / schedule40_train",prior_main_stage_images_and_analysis_seen=True,course_method_mapping_read_before_initial_freeze=False,course_quality_metrics_read_before_initial_freeze=False,rows=rows))
fields=list(rows[0])
with (out/"observations.csv").open("x",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
write_json("image_coverage.json",dict(status="PASS",requests=24,anonymous_output_notes=72,primary_png=48,supplemental_png=24,all_assigned_png_actually_viewed=72,manifest_sha256=sha(manifest),checks=image_checks))
summary="""# Course anonymous visual review

Reviewer: AI assistant / schedule40_train. This is an image observation record, not a human rating study.
All 24 requests, 72 anonymous outputs, and 72 original PNGs (48 primary and 24 complete-bounds supplements) were actually viewed in colored/neutral modes with front/back three-quarter, front, side and wire views. All image bytes match the safe registered manifest. Earlier main-stage images and numerical analysis had been seen; course method mapping and course quality metrics had not been read before this initial freeze. Method 1/2/3 here are strictly the course-page aliases.

The dominant result in these three parents is a recognizable full chair silhouette; that does not certify clean or connected surfaces. The notes retain occasional long thin planes, close parallel layers, crest cracks, and isolated-looking scraps. Open back panels and armchair variants are not automatically defects, and input-C gaps/thin legs are not attributed to generated free geometry. Rendering does not establish exact intersections, welding or vertex connectivity.

- 534b: The rounded high-back chair is mostly preserved across noise. At 20%, Method 3 n0 has a conspicuous side wing; n7 has thin hovering seat-level strips in Methods 1/3 and crest irregularities. At 40%, Method 3 n0 has a long downward spike; n1 has narrow seat-corner tips/folds. Most n6/n7 silhouettes remain intact. These are noise-dependent tails, not a universal failure of an alias.
- c1b: Most outputs remain coherent chairs with a top-back patch and open lower back. At 20%, Method 1 n7 has a long downward seat spike; Method 2 n1 generates an arm/frame variant with broad sloping panels, whose function is ambiguous. At 40%, Method 2 n0 has a broad under-seat triangle; Method 3 n1 has a conspicuous knife-like triangle projecting above and below the seat; Methods 1/3 n7 have long sloping under-seat sheets while Method 2's is shorter. These visible tails must accompany any favorable aggregate.
- f596: The 20% inputs sit at the seat. All aliases retain tall-back chairs; Method 2 n0 has a small isolated-looking fragment near a foot, and Method 3 n1 has a broader under-seat triangle. At n7 the blue seat patch can appear slightly above the orange seat band in side view for all aliases; attachment is not established. At 40%, all aliases produce a broad planar free back next to the input back, often giving a close parallel-layer silhouette. Methods 2/3 n1 have long thin orange strips among the legs; n6 Methods 2/3 leave a triangular/open free back in front of C, which may be stylistic rather than collapse. No large remote sheet was newly revealed by complete-bounds supplements.

Per-request immutable JSON/CSV contains the full alias-specific observations, uncertainty, original image paths and hashes. No winner or numerical score is assigned in this initial review.
"""
(out/"VISUAL_REVIEW.md").write_text(summary,encoding="utf-8")
files=[p for p in sorted(out.rglob("*")) if p.is_file()]
write_json("initial_freeze_receipt.json",dict(status="COMPLETE",created_utc=datetime.now(timezone.utc).isoformat(),requests=24,notes=72,images_actually_viewed=72,course_mapping_read=False,course_quality_metrics_read=False,prior_main_stage_images_and_analysis_seen=True,files=[dict(path=str(p),sha256=sha(p)) for p in files]))
print(json.dumps({"status":"COMPLETE","notes":len(rows),"images":len(image_checks),"observations_sha256":sha(out/"observations.json"),"freeze_sha256":sha(out/"initial_freeze_receipt.json")}))
