"""Transcribe actual human-readable AI_PROXY viewing notes; never read metrics."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_three_mechanisms_v1'
V=OUT/'visual_review'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=read(V/'templates/review_part_03.json')
manual=read(V/'completed/review03_manual.json')
labels=['EXPOSURE','PLAIN200','PLAIN267','PATCH200','LOOP200','RL_ZERO','RL16']
maps=[{'R':'REASONABLE','I':'INCOMPLETE','U':'UNCERTAIN'},
      {'P':'PLAUSIBLE','R':'REPLACED_OR_IGNORED','U':'UNCERTAIN'},
      {'P':'PRESENT','M':'MISSING_OR_DAMAGED','U':'UNCERTAIN'},
      {'Y':'YES','N':'NO_OBVIOUS','U':'UNCERTAIN'}, {'Y':'YES','N':'NO_OBVIOUS','U':'UNCERTAIN'}]
axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
by={x['page_id']:x for x in manual}
for row in data['rows']:
    if row['page_id'] not in by:continue
    entry=by[row['page_id']];code,note=entry['ratings'][labels.index(row['label'])]
    for axis,m,ch in zip(axes,maps,code):row[axis]=m[ch]
    p=OUT/'gallery'/row['page_id'];cp=V/'crops'/row['page_id']
    crops=[cp/name for name in ('rows_1_3.png','rows_4_5.png','rows_6_7.png')]
    event=entry['view']['event'];utc=entry['view']['utc'].replace(' UTC','+00:00').replace(' ','T')
    row.update(actually_viewed=True,views_and_sections_checked=True,note=note,
        image_sha256=sha(p/'comparison.png'),image_receipt_sha256=sha(p/'comparison.json'),
        view_event_id=event+'_original',viewed_at_utc=utc,
        crop_paths=[str(x) for x in crops],crop_sha256=[sha(x) for x in crops],
        crop_view_events=[dict(event_id=event+'_'+x.stem,tool='view_image',detail='original',viewed_at_utc=utc) for x in crops])
count=sum(r['actually_viewed'] for r in data['rows'])
data.update(status='COMPLETE' if count==84 else 'IN_PROGRESS',reviewer='/root/structural_local_history',
    reviewer_kind='AI_PROXY',blind=False,actual_viewed_cells=count,actual_viewed_pages=len(manual),
    copied_historical_ratings=0,numeric_metrics_read_for_ratings=False)
(V/'completed/review_part_03.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=data['status'],viewed_cells=count,pages=len(manual))))
