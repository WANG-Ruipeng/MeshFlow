from pathlib import Path
import json,hashlib,datetime
root=Path('/mnt/e/MeshFlow');run=root/'native_t1/runs/geometry_context_revised_20260930_153000'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(run/'results.json')
expected={0:'c96e7ecb17b43577032a945c24938e9902bd81ca33e6a274e1eee2cae37cc8ed',1:'9385507cac20effc0bd6c49709f39cc50ed3c37e461c1dd90d2f5a379571cbbd',2:'bcece972e826575909a854b5077ed8ce1037586f45ef92f69a4c85a384676e09'}
figures=[]
for f in r['figures']:
 p=Path(f['path']);j=f['patch_id']
 assert sha(p)==expected[j]==f['sha256'], 'Final image changed after visual inspection'
 f['assistant_visual_review']='PASS'
 f['human_visual_review']='NOT_PERFORMED'
 figures.append(dict(patch_id=j,path=str(p),sha256=sha(p),status='PASS',
  inspected='Titles/model labels, full/free rows, known/free colors, shared parent bounds, clipping and legibility',
  scope='Layout/faithful comparison review only, not a geometric validity certificate'))
review=dict(status='PASS',reviewer='assistant_visual_inspection',images=figures,model_calls=0,
 io_notes=['view_image could not read due to Windows sandbox logon1385; reviewed read-only JPEG previews of all three unchanged PNGs.',
 'One PowerShell default-codepage JSON inspection failed; explicit UTF-8 Python JSON parsing passed. Saved JSON was not corrupted.'])
(run/'visual_review.json').write_text(json.dumps(review,indent=2)+'\n')
integrity=read(run/'delivery_integrity.json');manifest=read(run/'code_manifest.json')
integrity['code_manifest_matches']={f['path']:sha(root/f['path'])==f['sha256'] for f in manifest['files']}
assert all(integrity['code_manifest_matches'].values())
assert integrity['status']=='PASS' and r['counts']['total']['optimizer_updates']==1500
assert r['generation']['formal_counts']['complete_rollouts']==64
assert r['counts']['total']['complete_rollouts']==68
(run/'delivery_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
r['delivery_integrity']=dict(status='PASS',path=str(run/'delivery_integrity.json'),sha256=sha(run/'delivery_integrity.json'))
r['visual_review']=dict(status='PASS',reviewer='assistant_visual_inspection',path=str(run/'visual_review.json'),sha256=sha(run/'visual_review.json'))
r['reporter']['post_render_verification']=dict(model_calls=0,training_updates=0,geometry_rescoring=0,scope='Recorded visual review and refreshed reporter code manifest only')
report=run/'GEOMETRY_CONTEXT_REPORT.md'
s=report.read_text()
s=s.replace('PNG完整性已核验，视觉验收由调用方执行。','PNG完整性已核验；助手已查看全部三张图，核对标签、同视角范围、颜色与裁切，记录见visual_review.json。这不替代几何审计。')
s+='\n交付完整性复核：64份原始输出的文件hash、全部51状态C、终点与轨迹对应、同seed跨模型/上下文Gaussian一致性均通过；20份执行源码、初始权重、输入和历史诊断身份保持。详见delivery_integrity.json。\n'
report.write_text(s)
(run/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='FINAL_DELIVERY_VERIFIED',figures=3,outputs=integrity['raw_count'],counts=r['counts']['total'],code_diff_sha256=sha(run/'code.diff'))))
