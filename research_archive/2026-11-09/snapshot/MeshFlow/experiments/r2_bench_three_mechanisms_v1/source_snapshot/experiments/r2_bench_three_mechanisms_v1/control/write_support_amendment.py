"""Persist two actual visual re-review observations; keep v1 untouched."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C
v=C.OUT/'visual_review';source=v/'completed/review_part_01.json';source_sha=C.sha(source)
assert source_sha=='8dad58d4d8ede1247cf41891beeabee6992fdcece1338469bb8a0b459401b772'
doc=C.read(source);changes=[]
request='f495bd67af8d9f0cadd1cf65e8dc4de__SEAT_SURFACE_1__n1'
for stage in ('BASE_POST','CLEAN_FINAL_POST'):
    row=next(r for r in doc['rows'] if r['request_id']==request and r['label']=='LOOP200' and r['stage']==stage)
    assert row['support_retention']=='PRESENT'
    crop=v/'crops'/row['page_id']/'rows_4_5.png';digest=C.sha(crop)
    assert digest==row['crop_sha256'][0]
    changes.append(dict(request_id=request,label='LOOP200',stage=stage,axis='support_retention',
        old_value='PRESENT',new_value='UNCERTAIN',old_note=row['note'],
        new_note='重新查看同一双视角与固定截面后，端部仍有多条细长竖杆，至少部分与座框在投影上连续；但az=120左下方一段竖条与座框之间存在可见间隔，远侧也有遮挡/重叠。固定中截面不经过角部支撑，不能确认四条杆均连贯。不能仅因变细判坏，也不能把剩余细线直接认证为完整支撑，故改记UNCERTAIN。蓝C用途及其他轴保持原记录。',
        actually_viewed=True,tool='view_image',viewed_at_utc='2026-10-29T16:51:21Z',
        view_event_id='functions.exec:view_image:part01_support_amendment:'+stage+':20261029T165121Z',
        crop_path=str(crop),crop_sha256=digest,image_sha256=row['image_sha256']))
out=dict(schema='bench_three_mechanisms_visual_amendment_v2',status='COMPLETE',reviewer='structural_math_review',
         reviewer_kind='AI_PROXY',blind=False,original_review_path=str(source),original_review_sha256=source_sha,
         reason='root独立观察与原评阅对细杆支撑完整性存在分歧，因此仅对两张既有图进行遮挡/断杆复核；非数值驱动。',
         metrics_read=0,historical_labels_read=0,original_completed_review_modified=False,
         original_summary_modified=False,scope='仅新增独立修订记录；原v1标签、计数与图审哈希保留，报告另列保守解释。',changes=changes)
dest=v/'amendments/f495_n1_loop_support_v2.json';C.write(dest,out)
assert C.sha(source)==source_sha
print(json.dumps(dict(status='COMPLETE',path=str(dest),sha256=C.sha(dest),units=len(changes),original_preserved=True)))
