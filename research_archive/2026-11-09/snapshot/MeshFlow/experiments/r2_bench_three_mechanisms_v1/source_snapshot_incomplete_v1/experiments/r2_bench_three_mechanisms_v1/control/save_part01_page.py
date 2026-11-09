"""Persist manually supplied actual observations; no inferred ratings."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C
v=C.OUT/'visual_review';obs=C.read(sys.argv[1]);dest=v/'completed/review_part_01.json'
d=C.read(dest if dest.exists() else v/'templates/review_part_01.json')
d['reviewer']='structural_math_review';page=obs['page_id']
assert len(obs['ratings'])==7
axes=('overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold')
maps=({'R':'REASONABLE','I':'INCOMPLETE','U':'UNCERTAIN'}, {'P':'PLAUSIBLE','R':'REPLACED_OR_IGNORED','U':'UNCERTAIN'},
      {'P':'PRESENT','M':'MISSING_OR_DAMAGED','U':'UNCERTAIN'}, {'Y':'YES','N':'NO_OBVIOUS','U':'UNCERTAIN'}, {'Y':'YES','N':'NO_OBVIOUS','U':'UNCERTAIN'})
for index,(label,rating) in enumerate(zip(C.LABELS,obs['ratings'])):
    target=[r for r in d['rows'] if r['page_id']==page and r['label']==label];assert len(target)==1
    row=target[0];assert not row['actually_viewed']
    code,note=rating;assert len(code)==5
    row.update({axis:m[c] for axis,m,c in zip(axes,maps,code)})
    image=C.OUT/'gallery'/page/'comparison.png';receipt=image.with_suffix('.json')
    crop='rows_1_3.png' if index<3 else 'rows_4_5.png' if index<5 else 'rows_6_7.png'
    cp=v/'crops'/page/crop
    row.update(actually_viewed=True,views_and_sections_checked=True,image_sha256=C.sha(image),
        image_receipt_sha256=C.sha(receipt),view_event_id=obs['view_event_id'],viewed_at_utc=obs['viewed_at_utc'],
        crop_paths=[str(cp)],crop_sha256=[C.sha(cp)],crop_view_events=[obs['view_event_id']],note=note)
d['status']='COMPLETE' if all(r['actually_viewed'] for r in d['rows']) else 'IN_PROGRESS'
C.write(dest,d,replace=True)
print(json.dumps(dict(status=d['status'],viewed_cells=sum(r['actually_viewed'] for r in d['rows']),page=page)))
