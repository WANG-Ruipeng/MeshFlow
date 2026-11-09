"""Join two actual AI_PROXY review ledgers; never infer labels from metrics."""
from r2.pilots.target_cleanup_c200_v1 import common as C
import json
dest=C.OUT/'evaluation/gallery'
sources=[dest/'actual_review_historical_train.csv',dest/'actual_review_dev.csv']
rows=[]
for source in sources:
    for item in C.csvread(source):
        row=dict(item)
        assert row['reviewer']=='AI_PROXY'
        assert row['review_status'] in ('REVIEWED','ACTUALLY_VIEWED')
        assert row['C_usage'] in ('YES','NO','UNKNOWN','UNCERTAIN')
        assert C.sha(row['image_path'])==row['image_sha256']
        row['review_source_path']=str(source);row['review_source_sha256']=C.sha(source)
        row['current_continuation_exposure']='NOT_IN_16_PARENT_POOL'
        row['scope_note']='Fresh non-blind visual seat-role assessment, not the historical strict adoption/joint protocol. A visual-role-and-edge conjunction must be named fresh_AI_PROXY_visual_role_and_edge and cannot establish an old 6-to-9 strict-joint gain.'
        row.setdefault('observation_scope','Two opposite saved views; NO is not a certification of absence. Hidden topology is UNKNOWN.')
        row.setdefault('reviewed_images_json',json.dumps([dict(path=row['image_path'],sha256=row['image_sha256'])]))
        rows.append(row)
key=lambda r:(r['request_id'],r['label'],r['stage'])
expected={key(r) for r in C.csvread(dest/'review_template.csv')}
assert len(rows)==144 and len({key(r) for r in rows})==144 and {key(r) for r in rows}==expected
rows.sort(key=key)
for name in ('actual_visual_review.csv','actual_visual_review_receipt.json'):
    p=dest/name
    backup=dest/(p.stem+'_v1_pre_scope'+p.suffix)
    if p.exists() and not backup.exists():C._write_bytes(backup,p.read_bytes())
C.csvwrite(dest/'actual_visual_review.csv',rows,replace=True)
C.write(dest/'actual_visual_review_receipt.json',dict(status='COMPLETE',review_rows=144,actually_viewed_images=24,
    reviewer='AI_PROXY',review_mode='NON_BLIND_AI_PROXY',source_hashes={str(p):C.sha(p) for p in sources},
    output_sha256=C.sha(dest/'actual_visual_review.csv'),code_sha256=C.sha(__file__),score_derived_labels=False,
    c_usage_vocabulary='YES/NO/UNKNOWN/UNCERTAIN preserved from each observer. DEV UNCERTAIN denotes ambiguous use; UNKNOWN can include visibility or use uncertainty in the other observer. Neither counts as affirmative success.',
    scope_note='The fresh visible seat-role judgement does not replace historical strict adoption/joint labels. No historical 6-to-9 strict success claim.',
    source_notes_preserved=True,source_rows_edited=False,new_model_calls=0),replace=True)
print(json.dumps(dict(status='COMPLETE',rows=144,images=24,sha256=C.sha(dest/'actual_visual_review.csv'))))
