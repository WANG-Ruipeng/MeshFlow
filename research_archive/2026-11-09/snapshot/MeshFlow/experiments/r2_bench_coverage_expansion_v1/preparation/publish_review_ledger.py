"""Combine the two actually viewed review ledgers; never infer review labels."""
import json, hashlib
from pathlib import Path
root=Path('/mnt/e/MeshFlow/experiments/r2_bench_coverage_expansion_v1')
rows=[];sources={}
for name in ('visual_reviews_root.json','visual_reviews_agent.json'):
    path=root/'preparation'/name
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows+=json.loads(path.read_text())['rows']
assert len(rows)==47 and len({r['candidate_id'] for r in rows})==47
for r in rows:
    assert r['actually_viewed'] and r['reviewer'].startswith('AI_PROXY')
    p=Path(r['image_path'].replace('E:/','/mnt/e/').replace('E:\\','/mnt/e/').replace('\\','/'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']
out=root/'preparation/visual_reviews.json'
assert not out.exists()
out.write_text(json.dumps(dict(schema='ACTUAL_AI_PROXY_REVIEW_COMBINED_V1',source_ledgers=sources,
    rows=sorted(rows,key=lambda r:r['candidate_id'])),indent=2,ensure_ascii=False)+'\n')
print('Combined actual review rows:',len(rows))
