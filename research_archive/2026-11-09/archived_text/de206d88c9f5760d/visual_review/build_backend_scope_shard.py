"""Attach provenance to actual manually recorded DEV image observations. No scoring."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

manual_path = OUT / 'visual_review/backend_scope_manual_observations.json'
manual = read(manual_path)
protocol = read(OUT / 'visual_review/protocol.json')
expected = {(u['parent_id'], u['noise_index'], u['stage'], u['label']): u
            for u in protocol['expected_units'] if u['report_group'] == 'DEV'}
assert len(manual) == 24 and len(expected) == 72
rows = []
seen = set()
for page in manual:
    page_keys = []
    for obs in page['observations']:
        label, shape, use, support, frag, layering, note = obs
        key = (page['parent'], page['noise'], page['stage'], label)
        assert key not in seen
        seen.add(key)
        u = expected[key]
        image = OUT / 'gallery' / (u['page_id'] + '.png')
        receipt_path = image.with_suffix('.json')
        receipt = read(receipt_path)
        assert receipt['image_sha256'] == sha(image)
        r = dict(u)
        r.update(reviewer='/root/backend_scope_audit', reviewer_kind='AI_PROXY', blind=False,
                 view_event_id='backend_scope_audit:' + u['page_id'],
                 viewed_at_utc=page['viewed_at_utc'], actually_viewed=True,
                 views_and_sections_checked=True, image_sha256=sha(image),
                 image_receipt_sha256=sha(receipt_path),
                 gallery_definition_sha256=receipt['definition_sha256'],
                 overall_shape=shape, C_use=use, support_retention=support,
                 visible_fragment_or_gap=frag, layering_or_fold=layering, note=note,
                 view_tool='functions.view_image')
        assert all(r[a] in values for a, values in protocol['axes'].items())
        assert all(k in r for k in protocol['required_fields'])
        rows.append(r)
assert seen == set(expected)
payload = dict(schema='bench_exposure_allocation_actual_review_shard_v1',
               reviewer='/root/backend_scope_audit', reviewer_kind='AI_PROXY', blind=False,
               actual_pages=24, actual_cells=72, report_group='DEV',
               manual_observations_sha256=sha(manual_path),
               protocol_sha256=sha(OUT / 'visual_review/protocol.json'), rows=rows)
dest = OUT / 'visual_review/shards/backend_scope_audit.json'
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    assert read(dest) == payload, 'Existing shard differs; refusing overwrite'
else:
    with dest.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
print(json.dumps(dict(path=str(dest), sha256=sha(dest), pages=24, cells=72)))
