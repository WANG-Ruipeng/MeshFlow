import json,pathlib,hashlib,sys
out=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_bench_geo_only_acceptance_v1/identity');gallery=json.loads((out/'gallery/COMPLETE.json').read_text());data=json.loads(sys.stdin.read());pages=[]
for o in data:
    r=next(x for x in gallery['images'] if x['job_id'].startswith(o['parent_prefix']))
    p=pathlib.Path(r['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==r['image_sha256'];assert [x['stage'] for x in o['observations']]==[x['stage'] for x in r['cells']]
    pages.append(dict(job_id=r['job_id'],image_path=str(p),image_sha256=h,viewed_at_utc=o['viewed_at_utc'],actually_viewed=True,views_and_sections_checked=True,actual_view_source='tools.view_image full page original detail',observations=o['observations']))
result=dict(reviewer='/root/bench_reference',reviewer_kind='AI_PROXY',blind=False,scope='Two source identity references, no generator/style ranking or exact topology certification from images',pages=pages)
p=out/'gallery/actual_review.json'
with p.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(str(p));reg=json.loads((out/'registration.json').read_text())
for row in reg['rows']:
    d=pathlib.Path(row['output_dir']);print(row['parent_id'][:6]);print((d/'stage_summary.json').read_text() if (d/'stage_summary.json').exists() else [p.name for p in d.iterdir()])
