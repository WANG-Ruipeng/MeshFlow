"""Expand read-only reviewer's actual-view message batches; never infer labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AXES = (
    ('overall_shape', dict(R='REASONABLE', I='INCOMPLETE', U='UNCERTAIN')),
    ('C_use', dict(P='PLAUSIBLE', R='REPLACED_OR_IGNORED', U='UNCERTAIN')),
    ('support_retention', dict(P='PRESENT', M='MISSING_OR_DAMAGED', U='UNCERTAIN')),
    ('visible_fragment_or_gap', dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN')),
    ('layering_or_fold', dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN')),
)
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def local(p):
    return Path(p.replace('E:/', '/mnt/e/')) if str(ROOT).startswith('/mnt/e/') else Path(p)

assignment = read(ROOT/'assignments.json')
task = assignment['tasks']['dev_b']
expected = {p['page_id']: p for p in task['pages']}
gallery = {p['page_id']: p for p in read(ROOT.parent/'gallery/completion.json')['images']}
records, seen, batches = [], set(), []
for batch in sorted(ROOT.glob('dev_b_batch_*.json')):
    doc = read(batch)
    assert doc['actually_viewed_full_png'] is True
    assert doc['reviewer'] == '/root/cleanup_cache_audit'
    batches.append(dict(path=batch.name, sha256=hashlib.sha256(batch.read_bytes()).hexdigest()))
    for item in doc['pages']:
        pid = item['page_id']
        assert pid in expected and pid not in seen
        seen.add(pid)
        e, g = expected[pid], gallery[pid]
        image_path = local(e['image_path'])
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert digest == g['image_sha256']
        assert len(item['models']) == 3
        assert [x['model'] for x in g['cells']] == assignment['labels']
        for label, (codes, note) in zip(assignment['labels'], item['models']):
            assert len(codes) == 5 and note.strip()
            row = {k:e[k] for k in ('request_id','parent_id','condition_id','stage','page_id')}
            row.update(label=label, image_path=e['image_path'], image_sha256=digest,
                       actually_viewed=True, reviewer_kind='AI_PROXY', blind=False,
                       actual_view_source='tools.view_image by /root/cleanup_cache_audit; transcribed from message',
                       reviewer='/root/cleanup_cache_audit', persisted_by='/root', note=note)
            for (axis, values), code in zip(AXES, codes):
                row[axis] = values[code]
            records.append(row)
out = dict(reviewer_kind='AI_PROXY', blind=False, reviewer='/root/cleanup_cache_audit',
           provenance='Literal expansion of actual-view messages from read-only reviewer; no new judgement by transcriber.',
           expected_pages=48, completed_pages=len(seen), complete=len(seen)==48,
           source_message_batches=batches, records=records)
dest = ROOT/'dev_b_records.json'
tmp = ROOT/'dev_b_records.tmp'
tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
tmp.replace(dest)
print(json.dumps(dict(pages=len(seen), rows=len(records), complete=out['complete'])))
