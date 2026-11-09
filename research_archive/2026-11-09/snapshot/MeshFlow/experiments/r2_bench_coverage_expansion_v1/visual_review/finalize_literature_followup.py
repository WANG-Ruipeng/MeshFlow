"""Join actual, manually entered visual observations to immutable page identities."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REVIEWER = 'AI_PROXY_literature_followup'
notes = json.loads((OUT/'visual_review/manual_notes_literature_followup.json').read_text(encoding='utf-8'))
rows = []
pages = set()
for item in notes:
    rid = item['rid']
    for stage, preference, a, b in item['stages']:
        page_id = stage+'__'+rid
        receipt = json.loads((OUT/'gallery'/(page_id+'.json')).read_text(encoding='utf-8'))
        image_path = OUT/'gallery'/(page_id+'.png')
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert image_hash == receipt['image_sha256'], page_id
        assert receipt['request_id'] == rid and receipt['sheet'] == stage
        assert page_id not in pages
        pages.add(page_id)
        for label, values in [('A_S1500',a),('COVERAGE_S1500',b)]:
            row = dict(page_id=page_id,request_id=rid,
                condition_id=receipt['condition_id'],parent_id=rid.split('__')[0],
                split=receipt['split'],part_type=receipt['part_type'],stage=stage,
                label=label,reviewer=REVIEWER,image_path=receipt['path'],
                image_sha256=image_hash,pair_preference=preference)
            row.update(zip(['C_use','whole_shape','layered_or_conflicting','visible_fragment','note'],values))
            rows.append(row)
assert len(rows) == 96 and len(pages) == 48
assert len({(r['request_id'],r['label'],r['stage']) for r in rows}) == 96
assert {r['parent_id'][:4] for r in rows} == {'9fd5','f495','f5d3','fd35'}
assert {r['split'] for r in rows} == {'DEV'}
enums = dict(C_use={'YES','NO','UNCERTAIN'},whole_shape={'COMPLETE','PARTIAL','UNCERTAIN'},
    layered_or_conflicting={'YES','NO','UNCERTAIN'},visible_fragment={'YES','NO','UNCERTAIN'},
    pair_preference={'COVERAGE','A','TIE','UNCERTAIN'})
for row in rows:
    for field, valid in enums.items():
        assert row[field] in valid, (row['page_id'],field)
result = dict(reviewer=REVIEWER,
    rubric_sha256=hashlib.sha256((OUT/'REVIEW_RUBRIC.md').read_bytes()).hexdigest(),
    review_basis='Actual inspection of all 48 PNG pages, both views and fixed sections; no scoring or model calls.',
    rows=rows)
target = OUT/'visual_review/review_literature_followup.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE',reviewed_pages=len(pages),reviewed_rows=len(rows),
    output=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
