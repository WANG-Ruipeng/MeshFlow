import csv,json,hashlib,re,datetime
from pathlib import Path
R=Path('/mnt/e/MeshFlow-Control-artifacts'); S=Path('/mnt/e/MeshFlow-Control'); OLD=Path('/mnt/e/MeshFlow')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def path(s):
    s=s.replace('\\','/')
    if len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def first(r,*keys):
    return next((r[k] for k in keys if r.get(k)), '')
gallery=read(R/'gallery/gallery_manifest.json')
pages={(p['condition_id'],int(p['candidate'])):p for p in gallery['pages']}
expected={(label,'P1__'+p['condition_id']+'__n'+str(p['candidate']) if p['panel']=='P1' else 'P2__'+p['condition_id']+'__n'+str(p['candidate'])) for p in gallery['pages'] for label in p['labels']}
assert len(expected)==288
rr=[];receipts=[]
for who,n in [('root',64),('diagnostics',84),('coupling',84),('scoring',56)]:
    p=R/('analysis/visual_review_'+who+'.csv');src=readcsv(p);assert len(src)==n,(who,len(src))
    receipts.append(dict(reviewer=who,path=str(p),sha256=sha(p),rows=n))
    for r in src:
        candidate=int(first(r,'candidate','noise'))
        condition=r.get('condition_id') or r['request_id'].split('__')[1]
        g=pages[(condition,candidate)]
        assert r['label'] in g['labels']
        blind=first(r,'blind_page','blind_path','blind_view_path','view_path')
        bh=first(r,'blind_sha256','blind_view_sha256','view_sha256')
        assert bh==g['blind_sha256'] and sha(path(blind))==bh,(who,r['request_id'],'blind hash')
        reference=first(r,'reference_page','reference_path','reference_view_path');rh=first(r,'reference_sha256','reference_view_sha256')
        if g['panel']=='P1':
            assert reference and rh==g['reference_sha256'] and sha(path(reference))==rh,(who,r['request_id'],'reference hash')
        else:assert not reference
        rr.append(dict(request_id=r['request_id'],panel=g['panel'],condition_id=condition,parent_id=r['parent_id'],
            candidate=candidate,label=r['label'],review_type='AI_VISUAL_REVIEW',reviewer=who,
            C_compatibility=first(r,'C_compatibility','c_compatibility'),
            overall_recognizability_coherence=first(r,'overall_recognizability_coherence','overall_recognition_coordination','overall_recognition','shape_and_local_defects','overall_coherence'),
            local_defects=first(r,'local_defects','shape_and_local_defects'),
            style_vs_fragmentation=first(r,'style_vs_fragmentation'),certainty=first(r,'certainty'),
            reference_observation=first(r,'reference_observation','reference_followup','reference_context'),
            blind_page=blind,blind_sha256=bh,reference_page=reference,reference_sha256=rh,
            view_scope='front_3q,back_3q,front,side,wire',human_usability='NOT_RUN',source_review_file=str(p),source_row_json=json.dumps(r,ensure_ascii=False)))
actual={(r['label'],r['request_id']) for r in rr}
assert len(rr)==len(actual)==288 and actual==expected
assert all(r['C_compatibility'] and r['local_defects'] and r['certainty'] for r in rr)
rr.sort(key=lambda r:(r['panel'],r['condition_id'],r['candidate'],r['label']))
table(R/'analysis/visual_review_all.csv',rr)
completion=dict(status='COMPLETE',review_type='AI_VISUAL_REVIEW',output_count=288,P1=224,P2=64,
    blind_pages=48,reference_pages=32,total_image_pages=80,human_usability='NOT_RUN',
    method_blinded=False,source_reference_order='P1 each assigned reviewer first viewed all assigned no-reference pages and saved notes, then reference pages; P2 no source target.',
    interpretation='Actual visual notes, not human labels or automatic usability rate. Small defects and exact connectivity remain uncertain.',
    sources=receipts,combined_csv_sha256=sha(R/'analysis/visual_review_all.csv'))
dump(R/'analysis/visual_review_completion.json',completion)
# Preserve original render manifest as-is; actual review is an overlay with immutable source hashes.
overlay=dict(render_manifest_sha256=sha(R/'gallery/gallery_manifest.json'),review_receipt='../analysis/visual_review_completion.json',
             review_csv='../analysis/visual_review_all.csv',AI_VISUAL_REVIEW='COMPLETE',human_usability='NOT_RUN',
             pages=[dict(condition_id=p['condition_id'],candidate=p['candidate'],AI_VISUAL_REVIEW='COMPLETE',labels=p['labels'],blind_sha256=p['blind_sha256'],reference_sha256=p.get('reference_sha256')) for p in gallery['pages']])
dump(R/'gallery/visual_review_status.json',overlay)
visual='P1 224 条输出已由三份分工记录覆盖：先完成各自全部无参考页并保存初次观察，再查看对应参考页。多数仍可辨椅形，500→1000 的款式/比例变化比同终点 A/B/C 的差别更显著。33ec 的侧板封闭变化、86ec 的实心/条栅差异不能仅因不似参考判错；bdd n4 的长三角片和 C1000 外伸片、f596 n1 C1000 更紊乱的折片则是独立局部问题。f596 n4 C1000 又较简洁，不能将单个坏例或好例推广。c1b n4 的高脚凳式完成属于款式变化。全部 288 条（P1 224 + P2 64）的实际审阅已按请求/模型身份合并，图像 hash 与原图集核对一致。'
report=(R/'analysis/EXPERIMENT_REPORT.template.md').read_text(encoding='utf-8')
tables=read(R/'analysis/report_tables.json');tables['visual_summary']=visual
for k,v in tables.items():report=report.replace('{{'+k+'}}',v)
assert '{{' not in report
report=report.replace('C−B：FF 降 0.711 个百分点，7/8 父下降，是一个实际、较一致的局部改善；同时 CF 升 0.137 个百分点、I 升 0.093 个百分点，coverage 降 0.293 个百分点，p95 升 0.0445%L。这构成取舍，不支持整体晋级。','C−B：FF 降 0.711 个百分点，7/8 父下降，是一个实际、较一致的局部改善；同时 CF 升 0.137 个百分点、I 升 0.093 个百分点，coverage 降 0.293 个百分点，p95 升 0.0445%L。I 的 crossing 增 0.081 个百分点、near-duplicate 反降 0.021 个百分点，不能统称重复增加。这构成取舍，不支持整体晋级。')
report=report.replace('预检完成 47F/11B','评分迁移也用旧 RAW 的 129 个共同数值标量核验，与历史缓存完全一致，未静默修改旧定义。预检完成 47F/11B')
(R/'EXPERIMENT_REPORT.md').write_text(report,encoding='utf-8')
# Add an explicit review receipt to gallery HTML without rerendering or changing images.
index=R/'gallery/INDEX.html';html=index.read_text()
tag='<p id="actual-review"><strong>AI_VISUAL_REVIEW: COMPLETE (288 outputs / 80 image pages)</strong>; human usability: NOT_RUN. Model labels remain visible. <a href="../analysis/visual_review_all.csv">Per-output actual review</a> | <a href="../analysis/visual_review_completion.json">Review receipt</a> | <a href="../EXPERIMENT_REPORT.md">Experiment report</a></p>'
if 'id="actual-review"' not in html:
    pos=html.find('>',html.find('<body'))
    assert pos>=0;html=html[:pos+1]+tag+html[pos+1:];index.write_text(html,encoding='utf-8')
# Protected original source/data: read-only hashes, no changes in old repository.
source_map=read(S/'docs/source_map.json');checks=[]
for e in source_map['files']:
    checks.append(dict(path=str(OLD/e['source']),sha256=sha(OLD/e['source']),expected=e['source_sha256']))
for p,h in read(R/'data/source_identity.json')['files'].items():
    checks.append(dict(path=p,sha256=sha(path(p)),expected=h))
assert all(x['sha256']==x['expected'] for x in checks)
# Verify final source remains equal to the actually successful scientific preflight.
identity=read(R/'preflight/real/preflight.json')['identity']
for rel,h in identity['scientific_sources'].items():assert sha(S/'src/meshflow_control'/rel)==h,rel
for rel,field in [('data/train_manifest.json','data_manifest_sha256'),('training_stream/train_plan.json','plan_sha256')]:
    assert sha(R/rel)==identity[field]
inventory=[]
for p in sorted(S.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(S)
    if any(x in ('.git','__pycache__','.pytest_cache') or x.endswith('.egg-info') for x in rel.parts):continue
    assert p.stat().st_size<10*1024*1024,rel
    assert p.suffix.lower() not in {'.pt','.pth','.npz','.npy','.ply','.obj','.png','.zip'},rel
    inventory.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
table(R/'audit/final_source_hashes.csv',inventory)
# Link checks are limited to explicit relative Markdown links, not shell placeholders.
broken=[]
for p in list(S.rglob('*.md'))+[R/'EXPERIMENT_REPORT.md',R/'METHOD.md']:
    if '.git' in p.parts:continue
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        target=target.split('#',1)[0].strip('<>')
        if not target or '://' in target:continue
        if (p.parent/target).resolve()==(R/'audit/final_acceptance.json').resolve():continue
        if not (p.parent/target).exists():broken.append(dict(file=str(p),target=target))
assert not broken,broken
git=read(R/'audit/final_git_state.json')
assert git['original_head']=='ae91eaa6f882a7d311233fa32053b7cd5de7ebe0'
assert git['original_status']==['?? AGENTS.md','?? deliverables/','?? native_t1/docs/control_metrics.md']
assert not git['new_remotes'] and git['new_commit_count']==0
delivery=read(R/'audit/delivery_integrity.json')
assert delivery['status']=='PASS' and delivery['gpu_owners_closed']
assert read(R/'orchestration/status.json')['status']=='COMPLETE'
receipt=dict(status='COMPLETE_WITH_DOCUMENTED_LICENSE_LIMITATION',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    code_and_training='COMPLETE',evaluation='288/288',visual_review='288/288 outputs; 80 pages actually viewed',human_usability='NOT_RUN',
    scientific_sources_unchanged_since_preflight=True,protected_source_data_hash_checks=checks,
    source_file_count=len(inventory),source_total_bytes=sum(x['bytes'] for x in inventory),
    original_git=git,broken_markdown_links=broken,
    actual_training_updates={'A':1000,'B':1000,'C':1000},actual_new_formal_rollouts=272,verified_cached_outputs=16,
    gpu_owners_closed=True,license_exception='NVIDIA express-license notice retained in vendored backbone/utils.py; separate redistribution basis unverified',
    no_commit_push=True,no_new_experiment=True,report_sha256=sha(R/'EXPERIMENT_REPORT.md'),
    review_receipt_sha256=sha(R/'analysis/visual_review_completion.json'),source_inventory_sha256=sha(R/'audit/final_source_hashes.csv'))
dump(R/'audit/final_acceptance.json',receipt)
print(json.dumps({k:receipt[k] for k in ('status','source_file_count','source_total_bytes','evaluation','visual_review','report_sha256')},ensure_ascii=False))
