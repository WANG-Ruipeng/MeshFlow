"""Expose only image paths/hashes for initial alias-only visual inspection."""
import hashlib,json
from pathlib import Path
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
p=r/"comparisons/curriculum/gallery/gallery.json"
v=json.loads(p.read_text())
assert v["status"]=="COMPLETE" and v["page_count"]==128
fields=("path","sha256","request_id","condition_id","candidate","mode","primary_framing_limit","supplemental")
rows=[{k:x[k] for k in fields if k in x} for x in v["pages"]+v["supplemental_pages"]]
assert len(rows)==128+v["supplemental_page_count"] and len({x["path"] for x in rows})==len(rows)
assert all(hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in rows)
out=r/"audit/visual_review/course"
out.mkdir(parents=True,exist_ok=True)
target=out/"review_manifest.json"
value=dict(schema="alias_only_visual_review_manifest_v1",source_gallery_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
primary_pages=128,supplemental_pages=v["supplemental_page_count"],rows=rows,
excluded_fields="All identity/method_mapping/triangle-label fields omitted; original complete provenance preserved.",
assistant_review_not_human_study=True)
if target.exists():assert json.loads(target.read_text())==value
else:target.write_text(json.dumps(value,indent=2),encoding="utf8")
print(json.dumps(dict(status="PASS",primary=128,supplemental=v["supplemental_page_count"],sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
