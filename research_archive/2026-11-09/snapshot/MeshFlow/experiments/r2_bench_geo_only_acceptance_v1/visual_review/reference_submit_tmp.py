import json,hashlib,pathlib,datetime,sys
root=pathlib.Path('/mnt/e/MeshFlow');out=root/'experiments/r2_bench_geo_only_acceptance_v1'
obs=json.loads(sys.stdin.read())
manifest=json.loads((out/'gallery/review_manifest.json').read_text())
axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
pages=[]
for ob in obs:
    src=sorted([x for x in manifest['cells_to_review'] if x['parent_id'].startswith(ob['prefix']) and x['noise_index']==ob['noise']],key=lambda x:x['column'])
    assert len(src)==5
    p=pathlib.Path(src[0]['image_path']); h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==src[0]['image_sha256']
    cells=[]
    for row,rating in zip(src,ob['ratings']):
        assert all(rating[i] in manifest['axes'][axis] for i,axis in enumerate(axes))
        cells.append(dict(label=row['label'],stage=row['stage'],**dict(zip(axes,rating[:5])),note=rating[5]))
    pages.append(dict(request_id=src[0]['request_id'],image_path=str(p),image_sha256=h,viewed_at_utc=ob['viewed_at_utc'],actually_viewed=True,actual_view_source='tools.view_image full comparison.png, original detail; rendered at 2048px width by tool',views_and_sections_checked=True,cells=cells))
assert len(pages)==10 and len({p['request_id'] for p in pages})==10
result=dict(reviewer='/root/bench_reference',reviewer_kind='AI_PROXY',blind=False,protocol='Original five axes; fresh simultaneous five-column images, no metric/history-based label filling',pages=pages)
path=out/'visual_review/review_reference.json';path.parent.mkdir(exist_ok=True)
with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(path=str(path),pages=len(pages),cells=sum(len(p['cells']) for p in pages),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
