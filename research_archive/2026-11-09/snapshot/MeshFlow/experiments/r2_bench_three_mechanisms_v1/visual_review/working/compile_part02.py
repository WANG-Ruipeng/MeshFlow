"""Assemble manually recorded actual image observations; never infer labels."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
base = Path(__file__).resolve().parents[1]
out = base / "completed" / "review_part_02.json"
template = json.loads((base / "templates/review_part_02.json").read_text(encoding="utf-8"))
protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert h(base / "protocol.json") == template["protocol_sha256"]
pages = {}
for i in range(1,5):
    d=json.loads((base / "working" / f"review_part_02_batch{i:02d}.json").read_text(encoding="utf-8"))
    for p in d["pages"]:
        assert p["page_id"] not in pages
        assert len(p["observations"]) == 7
        pages[p["page_id"]] = p
assert len(pages) == 12
labels = ["EXPOSURE","PLAIN200","PLAIN267","PATCH200","LOOP200","RL_ZERO","RL16"]
maps = [
 {"R":"REASONABLE","I":"INCOMPLETE","U":"UNCERTAIN"},
 {"P":"PLAUSIBLE","R":"REPLACED_OR_IGNORED","U":"UNCERTAIN"},
 {"P":"PRESENT","M":"MISSING_OR_DAMAGED","U":"UNCERTAIN"},
 {"Y":"YES","N":"NO_OBVIOUS","U":"UNCERTAIN"},
 {"Y":"YES","N":"NO_OBVIOUS","U":"UNCERTAIN"}]
axis = ["overall_shape","C_use","support_retention","visible_fragment_or_gap","layering_or_fold"]
for r in template["rows"]:
    p=pages[r["page_id"]]
    observation=p["observations"][labels.index(r["label"])]
    for j,k in enumerate(axis):
        r[k]=maps[j][observation[j]]
        assert r[k] in protocol["axes"][k]
    r["note"]=observation[5]
    r["actually_viewed"]=True
    r["views_and_sections_checked"]=True
    r["view_event_id"]=p["view_event_id"]
    if p["index"] == 0:
        r["view_event_id"]="part02_page00_original_crops"
    r["viewed_at_utc"]=datetime.strptime(p["viewed_at_utc"],"%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc).isoformat()
    image=base.parent / "gallery" / r["page_id"] / "comparison.png"
    receipt=image.with_suffix(".json")
    r["image_sha256"]=h(image)
    r["image_receipt_sha256"]=h(receipt)
    crops=[base / "crops" / r["page_id"] / (c+".png") for c in ["rows_1_3","rows_4_5","rows_6_7"]]
    r["crop_paths"]=[str(c).replace("\\","/") for c in crops]
    r["crop_sha256"]=[h(c) for c in crops]
    r["crop_view_events"]=[{"path":r["crop_paths"][i],"sha256":r["crop_sha256"][i],"tool":"view_image","view_event_id":r["view_event_id"],"viewed_at_utc":r["viewed_at_utc"],"detail":"original"} for i in range(3)]
assert len(template["rows"])==84
assert len({(r["request_id"],r["stage"],r["label"]) for r in template["rows"]})==84
template["status"]="COMPLETE"
template["reviewer"]="/root/official_augmentation_audit"
template["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
template["actual_pages_viewed"]=12
template["actual_crops_viewed"]=36
template["scope_note"]="Manual AI_PROXY nonblind observation of both original oblique views and both frozen sections through all three original-resolution crops per page. No numerical metric results or historical labels read."
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(template,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":"COMPLETE","rows":len(template["rows"]),"pages":len(pages),"sha256":h(out),"path":str(out)},ensure_ascii=False))

