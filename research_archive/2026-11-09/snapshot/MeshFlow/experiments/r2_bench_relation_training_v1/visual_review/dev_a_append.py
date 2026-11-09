"""Persist manually assigned actual-image review batches; no model/scorer calls."""
from pathlib import Path
import hashlib
import json
import sys

root = Path('/mnt/e/MeshFlow/experiments/r2_bench_relation_training_v1')
out = root / 'visual_review/dev_a_records.json'
batch = json.loads(sys.stdin.read())
record = json.loads(out.read_text()) if out.exists() else {'reviewer_kind': 'AI_PROXY', 'blind': False, 'records': []}
axes = ['overall_shape', 'C_use', 'support_retention', 'visible_fragment_or_gap', 'layering_or_fold']
codes = [dict(R='REASONABLE', I='INCOMPLETE', U='UNCERTAIN'), dict(P='PLAUSIBLE', R='REPLACED_OR_IGNORED', U='UNCERTAIN'), dict(P='PRESENT', M='MISSING_OR_DAMAGED', U='UNCERTAIN'), dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN'), dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN')]
for page in batch:
    p = root / 'gallery' / (page['page_id']+'.json')
    meta = json.loads(p.read_text())
    image = Path(meta['path'])
    assert hashlib.sha256(image.read_bytes()).hexdigest() == meta['image_sha256']
    assert len(page['cells']) == 3
    for cell, manual in zip(meta['cells'], page['cells']):
        assert len(manual['axes']) == 5
        r = dict(request_id=meta['request_id'], parent_id=meta['request_id'].split('__')[0], condition_id=meta['condition_id'], label=cell['model'], stage=cell['stage'], page_id=meta['page_id'], image_path=meta['path'], image_sha256=meta['image_sha256'], actually_viewed=True, actual_view_source='functions.view_image: complete comparison PNG, both views and both fixed sections', reviewer_kind='AI_PROXY', blind=False, note=manual['note'])
        r.update({a:d[c] for a,d,c in zip(axes,codes,manual['axes'])})
        assert not any(x['page_id']==r['page_id'] and x['label']==r['label'] for x in record['records']), 'duplicate review'
        record['records'].append(r)
out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'saved_rows':len(record['records']), 'saved_pages':len({x['page_id'] for x in record['records']})}))
