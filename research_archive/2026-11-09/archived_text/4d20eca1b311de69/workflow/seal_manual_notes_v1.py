"""Attach image identities to explicitly authored actual-view observations."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
p=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('notes');a.add_argument('output');args=a.parse_args()
notes=json.loads((p/'visual_review'/args.notes).read_text())
rows=[]
for i,x in enumerate(notes):
    image=p/'gallery/final_single'/('CLEAN_FINAL_POST__'+x['rid']+'__'+x['arm'])/'comparison.png'
    if not image.is_file() or not image.with_suffix('.json').is_file():raise FileNotFoundError(image)
    r=dict(request_id=x['rid'],cell_id=x['arm']+'__'+x['rid'],display_label=x['arm'],stage='CLEAN_FINAL_POST',
           actually_viewed=True,reviewer='AI_PROXY',method_labels_hidden=False,source_image=str(image),
           source_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),view_event_id=Path(args.notes).stem+'_'+str(i+1),
           observation=x['note'],recorded_at_utc=datetime.now(timezone.utc).isoformat())
    r.update(zip(('body','adoption','support','fragment_evidence','folding'),x['ratings']))
    rows.append(r)
f=p/'visual_review'/args.output
with f.open('x',encoding='utf-8') as h:
    json.dump(dict(schema='actual_visual_review_v1',records=rows),h,indent=2,ensure_ascii=False);h.write('\n')
print(len(rows),hashlib.sha256(f.read_bytes()).hexdigest())
