"""Versioned actual-view uncertainty correction; preserve original review v1."""
from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C
import summarize_visual as S

vr=C.OUT/'visual_review';source=vr/'summary_v1';dest=vr/'summary_v2'
original=C.read(source/'completion.json');assert original['status']=='COMPLETE'
C.verify(original['files'])
path=vr/'amendments/f495_n1_loop_support_v2.json';am=C.read(path)
assert am['status']=='COMPLETE' and am['metrics_read']==0 and len(am['changes'])==2
C.verify({am['original_review_path']:am['original_review_sha256']})
with (source/'actual_observations.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
for r in rows:
    r['amendment_path']='';r['original_view_note']=''
idx={(r['request_id'],r['label'],r['stage']):r for r in rows};h=S.helpers()
protected={str(path):C.sha(path),str(source/'completion.json'):C.sha(source/'completion.json'),
    str(source/'actual_observations.csv'):C.sha(source/'actual_observations.csv')}
for change in am['changes']:
    r=idx[change['request_id'],change['label'],change['stage']]
    assert change['actually_viewed'] is True and change['tool']=='view_image'
    assert r[change['axis']]==change['old_value'] and r['note']==change['old_note']
    assert r['image_sha256']==change['image_sha256']
    C.verify({change['crop_path']:change['crop_sha256']})
    protected[change['crop_path']]=change['crop_sha256']
    r[change['axis']]=change['new_value'];r['note']=change['new_note'];r[h.JOINT]=h.joint(r)
    r['view_event_id']=change['view_event_id'];r['viewed_at_utc']=change['viewed_at_utc']
    r['amendment_path']=str(path);r['original_view_note']=change['old_note']
tables=h.build_tables(rows,C.read(vr/'protocol.json'))
dest.mkdir(parents=True,exist_ok=True)
for name,table in tables.items():
    payload=h.csv_bytes(table);p=dest/name
    if p.exists():assert p.read_bytes()==payload
    else:p.write_bytes(payload)
p=dest/'baseline_review_differences.csv';payload=(source/p.name).read_bytes()
if p.exists():assert p.read_bytes()==payload
else:p.write_bytes(payload)
C.verify(protected)
C.write(dest/'completion.json',dict(status='COMPLETE',cells=252,pages=36,
    reviewer_kind='AI_PROXY',blind=False,original_v1_preserved=True,amended_cells=2,
    reason='Actual reinspection of ambiguous support; no metric-based label change',
    labels_generated_by_program=0,additional_model_calls=0,protected=protected,
    files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='completion.json'}))
print(json.dumps(dict(status='COMPLETE',cells=252,amended_cells=2,original_v1_preserved=True)))
