"""Package manual judgments made after actual viewing of all 48 assigned pages."""
import hashlib
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1]
reviewer = 'AI_PROXY_current_geometry'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
notes = read(out / 'visual_review/current_geometry_notes.json')
pages = {p['page_id']: p for p in read(out / 'gallery/manifest.json')['pages']}
images = {p['page_id']: p for p in read(out / 'gallery/completion.json')['images']}
rows = []
for item in notes:
    for stage, pref, left, right in item['stages']:
        pid = stage + '__' + item['rid']
        page, im = pages[pid], images[pid]
        actual = out / 'gallery' / (pid + '.png')
        assert sha(actual) == im['image_sha256']
        for label, judgment in [('A_S1500', left), ('COVERAGE_S1500', right)]:
            row = {k: page[k] for k in ('page_id','request_id','condition_id','parent_id','split','part_type')}
            row.update(stage=stage, label=label, reviewer=reviewer,
                       image_path=im['path'], image_sha256=im['image_sha256'], pair_preference=pref)
            row.update(dict(zip(('C_use','whole_shape','layered_or_conflicting','visible_fragment','note'), judgment)))
            rows.append(row)
assert len(rows) == len({(r['request_id'],r['label'],r['stage']) for r in rows}) == 96
assert {r['parent_id'][:4] for r in rows} == {'2d1e','459f','5dda','9d56'}
result = dict(reviewer=reviewer, rubric_sha256=sha(out / 'REVIEW_RUBRIC.md'),
              actual_viewed_pages=48, nonblind=True, human_ratings=False,
              scope='DEV four assigned parents; two views and both inherited sections on every page', rows=rows)
target = out / 'visual_review/review_current_geometry.json'
with target.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(str(target),len(rows),'rows')
