"""Assemble exclusively hand-reviewed backend_runtime observations; no metric/model imports."""
import hashlib,json
from pathlib import Path
from collections import Counter
O=Path('/mnt/e/MeshFlow/experiments/r2_bench_exposure_allocation_v1')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
notes=read(O/'visual_review/backend_runtime_page_notes.json')
protocol=read(O/'visual_review/protocol.json')
rows=[];events=[]
for page in notes['pages']:
    units=[x for x in protocol['expected_units'] if x['parent_id'].startswith(page['parent']) and x['noise_index']==page['noise'] and x['stage']==page['stage']]
    assert len(units)==3
    pageids={x['page_id'] for x in units};assert len(pageids)==1
    pid=next(iter(pageids));png=O/'gallery'/(pid+'.png');rp=png.with_suffix('.json');receipt=read(rp)
    assert sha(png)==receipt['image_sha256']
    for label,rating in zip(('P0','A','B'),page['ratings']):
        unit=next(x for x in units if x['label']==label)
        row=dict(unit,reviewer='backend_runtime',reviewer_kind='AI_PROXY',blind=False,
                 view_event_id=page['event']+':'+label,viewed_at_utc=page['utc'],
                 actually_viewed=True,views_and_sections_checked=True,
                 image_sha256=sha(png),image_receipt_sha256=sha(rp),
                 gallery_definition_sha256=receipt['definition_sha256'],note=rating[5])
        for key,value in zip(protocol['axes'],rating[:5]):
            assert value in protocol['axes'][key];row[key]=value
        assert set(protocol['required_fields'])<=set(row)
        rows.append(row)
    events.append(dict(view_event_id=page['event'],viewed_at_utc=page['utc'],
                       page_id=pid,image_sha256=sha(png),tool='view_image',
                       actual_columns=('P0','A','B'),views=(-60,120),
                       sections=('front_central','side_central')))
assert len(rows)==72 and len(events)==24
assert len({(r['request_id'],r['label'],r['stage']) for r in rows})==72
result=dict(schema=protocol['schema'],status='COMPLETE_ASSIGNED_SHARD',reviewer='backend_runtime',
            reviewer_kind='AI_PROXY',blind=False,actual_pages=24,actual_observations=72,
            numerical_metrics_used_for_labels=False,protocol_sha256=sha(O/'visual_review/protocol.json'),
            manual_page_notes_sha256=sha(O/'visual_review/backend_runtime_page_notes.json'),
            view_events=events,rows=rows)
dest=O/'visual_review/shards/backend_runtime.json'
assert not dest.exists(),'Never overwrite completed actual review'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
strict=lambda r:r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT'
summary={}
for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
    summary[stage]={label:sum(strict(r) for r in rows if r['stage']==stage and r['label']==label) for label in ('P0','A','B')}
print(json.dumps(dict(shard=str(dest),sha256=sha(dest),pages=24,cells=72,strict_visual_intersection_of_assigned_8_requests=summary),indent=2))
