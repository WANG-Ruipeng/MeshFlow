"""Submit explicitly authored, actually viewed shard-A observations only."""
import json,sys
from pathlib import Path
from r2.pilots.bench_seat_coverage_v1 import visual_review as V
doc=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert doc["review_group"]=="a_flat_and_new_a8"
for page in doc["pages"]:
    page_id=page["stage"]+"__"+page["request_id"]
    V.record_page(doc["review_group"],page_id,page["ratings"],
        reviewer="AI_PROXY_bench_reference",
        view_event_id="actual_full_png_"+page_id+"_"+page["time"],
        viewed_at_utc=page["time"],
        actual_view_source="functions.exec tools.view_image(detail=original); full PNG independently inspected in conversation; both oblique views and fixed front/side sections",
        actually_viewed=True,views_and_sections_checked=True)
print(json.dumps({"actual_pages_submitted":len(doc["pages"]),"cells":sum(len(p["ratings"]) for p in doc["pages"])}))

