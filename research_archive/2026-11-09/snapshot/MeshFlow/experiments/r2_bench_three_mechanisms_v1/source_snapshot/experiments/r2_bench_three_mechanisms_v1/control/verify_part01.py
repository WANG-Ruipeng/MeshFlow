"""Validate this reviewer's actual shard, without metric/history reads."""
from pathlib import Path
import json,sys,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C
v=C.OUT/'visual_review';p=v/'completed/review_part_01.json';d=C.read(p);protocol=C.read(v/'protocol.json')
assert d['status']=='COMPLETE' and len(d['rows'])==84
assert d['protocol_sha256']==C.sha(v/'protocol.json')
keys=set();pages=set();joint=collections.Counter();uncertain=[]
for r in d['rows']:
    key=(r['request_id'],r['label'],r['stage']);assert key not in keys;keys.add(key);pages.add(r['page_id'])
    assert r['actually_viewed'] and r['views_and_sections_checked'] and r['tool']=='view_image'
    assert r['note'] and r['view_event_id'] and r['viewed_at_utc']
    assert all(r[a] in values for a,values in protocol['axes'].items())
    ip=C.OUT/'gallery'/r['page_id']/'comparison.png'
    assert C.sha(ip)==r['image_sha256'] and C.sha(ip.with_suffix('.json'))==r['image_receipt_sha256']
    for cp,ch in zip(r['crop_paths'],r['crop_sha256']):assert C.sha(cp)==ch
    if (r['overall_shape'],r['C_use'],r['support_retention'])==('REASONABLE','PLAUSIBLE','PRESENT'):joint[(r['label'],r['stage'])]+=1
    if any(r[a]=='UNCERTAIN' for a in protocol['axes']):uncertain.append(dict(request=r['request_id'],label=r['label'],stage=r['stage']))
assert len(pages)==12
receipt=dict(status='PASS',cells=84,pages=12,actual_review_sha256=C.sha(p),score_files_read=0,historical_labels_read=0,
             joint_counts=[dict(label=k[0],stage=k[1],count=n) for k,n in sorted(joint.items())],uncertain=uncertain)
C.write(v/'completed/review_part_01_verification.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
