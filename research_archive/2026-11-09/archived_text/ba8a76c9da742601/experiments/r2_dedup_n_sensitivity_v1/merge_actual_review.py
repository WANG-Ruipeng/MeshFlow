"""Merge completed independent visual records without relabeling outcomes."""
import collections,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
G=ROOT/'evaluation/gallery'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
sources=[G/'actual_review_train.csv',G/'actual_review_dev.csv']
delta_sources=[G/'visual_delta_train.csv',G/'visual_delta_dev.csv']
rows=[r for p in sources for r in read(p)]
ds=[r for p in delta_sources for r in read(p)]
expected={(r['request_id'],r['label'],r['stage']) for r in read(G/'review_template.csv')}
assert len(rows)==192 and len({(r['request_id'],r['label'],r['stage']) for r in rows})==192
assert {(r['request_id'],r['label'],r['stage']) for r in rows}==expected
assert len(ds)==48 and len({(r['request_id'],r['source_arm']) for r in ds})==48
scope='Fresh non-blind AI_PROXY visible seat-role assessment, not historical strict adoption/joint. All six fathers were excluded from the 16-parent cleanup continuation pool. Original-N outputs are reviewed anew; earlier review records remain unchanged. UNKNOWN and UNCERTAIN are distinct. No figure alone certifies indexed connectivity.'
for r in rows:
    assert r['reviewer']=='AI_PROXY' and r['review_status']!='UNREVIEWED'
    assert r['C_usage'] in ('YES','NO','UNKNOWN','UNCERTAIN')
    assert digest(G/Path(r['image_path']).name)==r['image_sha256']
    r['scope_note']=scope
for r in ds:
    assert r['stage']=='POST' and r['judgment'] in ('IMPROVED','WORSE','MIXED','UNCHANGED','UNKNOWN')
    assert digest(G/Path(r['image_path']).name)==r['image_sha256']
    r.setdefault('reviewer','AI_PROXY');r.setdefault('review_mode','NON_BLIND_AI_PROXY')
rows.sort(key=lambda r:(r['request_id'],r['label'],r['stage']))
ds.sort(key=lambda r:(r['request_id'],r['source_arm']))
write(G/'actual_visual_review.csv',rows);write(G/'per_request_visual_delta.csv',ds)
summary=[]
for source in ('ORIGINAL_C200','DEDUP_C200'):
    for split in ('ALL','TRAIN','DEV'):
        group=[r for r in ds if r['source_arm']==source and (split=='ALL' or r['split']==split)]
        summary.append(dict(source_arm=source,historical_split=split,stage='POST',outputs=len(group),
            **{key:sum(r['judgment']==key for r in group) for key in ('IMPROVED','WORSE','MIXED','UNCHANGED','UNKNOWN')},
            C_role_retained_YES=sum(r['C_role_retained']=='YES' for r in group),
            new_major_body_loss_YES=sum(r['new_major_body_loss']=='YES' for r in group)))
write(G/'visual_delta_summary.csv',summary)
receipt=dict(status='COMPLETE',actual_contact_sheets=24,review_cells=192,post_pairs=48,
    scope_note=scope,sources={str(p):digest(p) for p in sources+delta_sources},
    outputs={p.name:digest(p) for p in [G/'actual_visual_review.csv',G/'per_request_visual_delta.csv',G/'visual_delta_summary.csv']},
    summary=summary)
(G/'actual_visual_review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(rows=len(rows),delta_rows=len(ds),summary=summary),ensure_ascii=False))
