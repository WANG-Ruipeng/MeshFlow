from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'r2/pilots/bench_natural_n_recipe_v1/delivery.py'
dst=ROOT/'r2/pilots/bench_high_joint_coverage_v1/delivery.py'
text=src.read_text(encoding='utf-8-sig')
changes=[
('bench_natural_n_recipe_v1','bench_high_joint_coverage_v1'),
("GROUPS = ('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT')","GROUPS = ('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')"),
("LABELS = ('P0','A','B')","LABELS = ('B','W')"),
('P0=原曝光B，A=普通续训，B=自然面数新配方','B=旧窄HIGH，W=新宽HIGH，同起点同1500步'),
('|指标|P0|A|B|B−A|','|指标|B窄池|W宽池|W−B|'),
('|---|---:|---:|---:|---:|','|---|---:|---:|---:|'),
('vals[1] is None or vals[2] is None else vals[2]-vals[1]','vals[0] is None or vals[1] is None else vals[1]-vals[0]'),
('len(data)==342','len(data)==276'),
("if r['label']!='B':continue","if r['label']!='W':continue"),
("for ref in ('A','P0'):","for ref in ('B',):"),
("candidate='B'","candidate='W'"),
("'actual_observations':342","'actual_observations':276"),
]
for a,b in changes:
    if a not in text:raise ValueError('Missing '+a)
    text=text.replace(a,b)
# Package contains all tables/code, and deterministic complete new-parent figures.
start=text.index('def package():');end=text.index("if __name__=='__main__':",start)
text=text[:start]+'''def package():
    assert read(OUT/'audits/evaluation_closeout.json')['status']=='PASS'
    assert (OUT/'REPORT.md').exists()
    files=set()
    for name in ('REPORT.md','PROTOCOL.md','RUNBOOK.md','MODELS.md','AUTHORIZATION.json','training_registration.json','code.diff'):
        p=OUT/name
        if p.is_file():files.add(p)
    for folder in ('data','results','training_closeout','visual_review','audits','dispatch','backend_geo_only'):
        for p in (OUT/folder).rglob('*'):
            if p.suffix.lower() in ('.json','.csv','.md') and p.stat().st_size<12_000_000:files.add(p)
    files.update((OUT/'source_archive/final').rglob('*.py'))
    files.add(OUT/'source_archive/final_source_hashes.json')
    for name in ('generator_step1500.json','checkpoint_step1500.json','status.json'):
        files.add(OUT/'training/WIDE_HIGH'/name)
    panel=read(OUT/'data/eval_manifest.json')['requests']
    new=[q for q in panel if q['report_group']=='NEW_WIDE_TRAIN']
    assert len(new)==8
    for q in new:
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            p=OUT/'gallery'/(stage+'__'+q['request_id']+'.png');assert p.is_file();files.add(p)
    write(OUT/'PACKAGE_CONTENTS.md','包含完整逐输出177字段、逐父配对、实际五轴图审、数据与预算、实际源码hash/diff、W权重身份、四RAW CPU对照、全部新四父八请求三阶段24页图。完整138页图集、大权重、固定数组及原始调用账本保留本地；不是独立可重训发行包。没有按结果选图。\\n')
    files.add(OUT/'PACKAGE_CONTENTS.md')
    target=OUT/'R2_BENCH_HIGH_JOINT_COVERAGE_REVIEW.zip'
    assert not target.exists(),'Do not overwrite delivery'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    result=dict(status='COMPLETE',path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files),
        gallery_pages_included=24,model_weights_included=False,
        files_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(files)})
    write(OUT/'package_receipt.json',result)
    return {k:v for k,v in result.items() if k!='files_sha256'}


'''+text[end:]
compile(text,str(dst),'exec')
if dst.exists():
    assert dst.read_text(encoding='utf-8')==text
else:dst.write_text(text,encoding='utf-8')
print('PREPARED delivery.py; no scientific calls')
