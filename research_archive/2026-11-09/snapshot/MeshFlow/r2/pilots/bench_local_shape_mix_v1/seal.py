"""Verify and archive completed evidence, with no scientific recomputation."""
from __future__ import annotations
import difflib
import html
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from .closeout import ROOT, OUT, DEST, read, sha, write

def main():
    if (OUT/'DELIVERY_COMPLETE.json').exists():
        old=read(OUT/'DELIVERY_COMPLETE.json')
        for p,h in old['files'].items():
            assert sha(OUT/p)==h, p
        assert sha(OUT/old['review_zip'])==old['zip_sha256']
        print(json.dumps(dict(status='DELIVERY_VERIFIED',zip=old['review_zip'],bytes=old['zip_bytes'])))
        return
    assert (OUT/'REPORT.txt').is_file()
    reg=read(OUT/'jobs.json')
    checked={}
    for group in ('protected','source_hashes'):
        for p,h in reg[group].items():
            assert sha(p)==h, p
            checked[p]=h
    for cp in (OUT/'analysis/COMPLETE.json',):
        c=read(cp)
        for p,h in c['files'].items(): assert sha(p)==h, p
    receipts=[]
    for job in reg['jobs']:
        cp=OUT/'runs'/job['job_id']/'complete.json'; c=read(cp)
        assert c['status']=='COMPLETE'
        assert c['registration_sha256']==sha(OUT/'jobs.json')
        for p,h in c['files'].items(): assert sha(p)==h, p
        receipts.append(c)
    assert len(receipts)==12
    assert sum(r['new_score_rows'] for r in receipts)==24
    assert sum(r['fit_calls'] for r in receipts)==6
    assert sum(r['full_cascade_calls'] for r in receipts)==12
    for key in ('model_calls','D0_calls','training_updates','new_noise','new_queries','GEO_calls'):
        assert sum(r[key] for r in receipts)==0
    vr=read(DEST/'visual_review_complete.json')
    for p,h in vr['source_records'].items(): assert sha(p)==h
    for cp in sorted((OUT/'gallery').glob('R*.json')):
        c=read(cp); assert sha(c['image_path'])==c['image_sha256']
        for p,h in c['source_meshes'].items(): assert sha(p)==h
    report=html.escape((OUT/'REPORT.txt').read_text(encoding='utf-8'))
    links=''.join(f'<li><a href="gallery/{p.name}">{p.stem}</a></li>' for p in sorted((OUT/'gallery').glob('R*.png')))
    (OUT/'REVIEW.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>局部保形融合：实际完成</title>'
        '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#20252c}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#185ba0}</style>'
        '<h1>局部保形融合：完成，候选不晋级</h1><p>12页均已实际审阅；原图的 pending 脚注为渲染时状态，最终图审记录见下列文件。非盲 AI_PROXY，不能替代独立人类审阅。</p>'
        '<p><a href="gallery/index.html">完整图集</a> · <a href="analysis/per_output_full177.csv">60条完整指标</a> · '
        '<a href="closure/visual_unique54.csv">54唯一图审记录</a> · <a href="closure/connectivity_check.json">连接及外部硬锁核查</a> · '
        '<a href="RUNBOOK.txt">验证与恢复</a></p><details><summary>逐页截面与双视图</summary><ul>'+links+'</ul></details><pre>'+report+'</pre></html>',encoding='utf-8')
    archive=OUT/'source_archive'; archive.mkdir(exist_ok=True)
    code=dict(reg['source_hashes'])
    for p in Path(__file__).parent.glob('*.py'): code[str(p)]=sha(p)
    manifest=[]
    for p,h in sorted(code.items()):
        original=Path(p); assert sha(original)==h
        rel=Path(p).relative_to('/mnt/e')
        target=archive/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert sha(target)==h
        else: shutil.copyfile(original,target)
        manifest.append(dict(original=p,archive=str(target.relative_to(OUT)),sha256=h,bytes=target.stat().st_size))
    write(OUT/'source_hashes.json',dict(files=manifest,source_archive_includes_actual_original_scorers=True))
    diff=[]
    for p in sorted(Path(__file__).parent.glob('*.py')):
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),
                    fromfile='/dev/null',tofile='b/'+str(p.relative_to(ROOT))))
    (OUT/'code.diff').write_text(''.join(diff),encoding='utf-8')
    write(DEST/'execution_accounting_final.json',dict(status='COMPLETE',
          science_jobs=12,indexed_assemblies=12,local_linear_fits=6,new_full_metric_rows=24,
          reused_metric_rows=36,original_metric_field_count=177,render_pages=12,
          training_updates=0,model_calls=0,D0_calls=0,new_noise=0,new_GEO_calls=0,
          scientific_failures=0,scientific_replays=0,model_or_backend_promotions=0,
          actual_review_unique=54,actual_review_displays=60,
          protected_and_source_files_verified=len(checked),
          engineering_notes=[
              'Original analysis aggregation stopped on logical RAW_ALL / physical HYBRID producer-label alias; frozen source retained, v2 aggregation validates both identities. See engineering_recovery.json.',
              'Closeout accepted equivalent reviewer status ACTUAL_VIEW_COMPLETE / COMPLETE_ACTUAL_VIEW after one assertion stop. Only root review transcription existed at that stop and was reused; no ratings overwritten and no scientific work repeated.',
              'RUNBOOK recovery receipt path corrected from audits/engineering_recovery.json to engineering_recovery.json.'],
          dispatch=read(OUT/'dispatch/complete.json'),render=read(OUT/'gallery/dispatch_complete.json')))
    # Preserve all experiment output, original tables, images and actual scoring code.
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.suffix!='.zip' and p.name!='DELIVERY_COMPLETE.json'}
    # No self-referential hash: manifest covers payload, zip excludes final receipt.
    write(OUT/'payload_hashes.json',dict(sha256=files,scope='Payload before this manifest and final delivery receipt'))
    files['payload_hashes.json']=sha(OUT/'payload_hashes.json')
    zp=OUT/'R2_BENCH_LOCAL_SHAPE_MIX_REVIEW.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in sorted(files): z.write(OUT/rel,rel)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
    receipt=dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
                 decision='LOCAL_SHAPE_NOT_PROMOTED',review_zip=zp.name,zip_bytes=zp.stat().st_size,
                 zip_sha256=sha(zp),payload_files=len(files),source_files=len(manifest),files=files,
                 no_new_experiment_or_default_change=True)
    write(OUT/'DELIVERY_COMPLETE.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
