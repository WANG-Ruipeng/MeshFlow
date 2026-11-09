"""Package complete tabular evidence and actual code; no weights or mesh arrays."""
from pathlib import Path
import hashlib, io, json, sys, zipfile

OUT=Path(__file__).resolve().parent
DEST=OUT.parents[1]/'deliverables/R2_BENCH_EXPOSURE_ALLOCATION_V1_REVIEW_SMALL.zip'
def main(dry_run=False):
    if not dry_run: assert (OUT/'COMPLETE.json').is_file()
    if DEST.exists(): raise RuntimeError('Do not overwrite an existing review package')
    files=set()
    for folder in ('results','visual_review','final_source_archive'):
        files.update(p for p in (OUT/folder).rglob('*') if p.is_file())
    for folder in ('audits','data_expansion'):
        files.update(p for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv','.md'))
    for name in ('REPORT.md','COMPLETE.json','PROTOCOL.md','USER_REQUEST.txt','RUNBOOK.md','RECOVERY_README.md',
                 'registration.json','training_registration.json','closeout.py','audit_result_tables.py','derive_visual_intersection.py','package_review.py',
                 'data/manifest.json','data/eval_manifest.json','training_closeout/complete.json',
                 'training_closeout/FM_50step_bins.csv','training_closeout/actual_exposure_per_parent.csv',
                 'training_closeout/resource_and_attempt_wallclock.json','training_closeout/training_audit.json',
                 'training_closeout/learning_curves.png','gallery/manifest.json','gallery/definition.json'):
        files.add(OUT/name)
    for arm in ('A_50_50','B_75_25'):
        for name in ('checkpoint_step1500.json','generator_step1500.json','status.json'):
            files.add(OUT/'training'/arm/name)
    plan=json.loads((OUT/'gallery/manifest.json').read_text('utf-8-sig'))
    # Stage/group rule, never outcome selection: every FINAL plus all FOCUS/HOLDOUT RAW.
    # The 42-page version including BASE exceeded 10 MB in the documented dry run.
    selected=[p for p in plan['pages'] if p['sheet']=='CLEAN_FINAL_POST' or (p['report_group'] in ('FOCUS','HOLDOUT') and p['sheet']=='RAW')]
    for page in selected:
        files.add(OUT/'gallery'/(page['page_id']+'.png')); files.add(OUT/'gallery'/(page['page_id']+'.json'))
    assert len(selected)==36
    manifest=[]
    for p in sorted(files):
        if dry_run and p.name=='COMPLETE.json' and not p.exists(): continue
        if not p.is_file(): raise FileNotFoundError(p)
        assert p.suffix not in ('.pt','.pth','.npz','.npy')
        manifest.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    index='<meta charset="utf-8"><h1>All 30 FINAL comparisons + FOCUS/HOLDOUT RAW</h1>'
    index+='<p>36 of the local 90 pages. All270 observations and complete177 metrics, including BASE, are included. No favorable-case selection; full local gallery remains unchanged.</p>'
    for p in selected:
        name=p['page_id']+'.png';index+='<h2>'+p['page_id']+'</h2><img loading="lazy" style="width:100%" src="'+name+'">'
    readme='Read REPORT.md first. Complete per-output/per-parent paired CSVs, all270 actual AI_PROXY observations, code and hashes are included. Image subset is fixed by stage/group: every FINAL request, plus every FOCUS/HOLDOUT RAW. BASE numerical results and actual ratings are complete, but BASE images remain local. All90 pages and actual weights/meshes remain in the local experiment. This package contains no model weights or raw arrays.\n'
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for item in manifest:z.write(OUT/item['path'],item['path'])
        z.writestr('gallery/review_index.html',index)
        z.writestr('README_REVIEW.txt',readme)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    data=buf.getvalue()
    if dry_run:
        print(json.dumps(dict(status='SIZE_ESTIMATE_ONLY',bytes_without_missing_closeout=len(data),files=len(manifest),images=36,files_written=0))); return
    assert len(data)<10_000_000, f'Package over10MB: {len(data)}'
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with DEST.open('xb') as f:f.write(data)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    receipt=dict(status='PASS',path=str(DEST),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),files_checked=len(manifest),images=36,
                 source_local_images=90,actual_observations=270,metric_fields=177,raw_arrays=False,weights=False)
    (OUT/'PACKAGE_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main('--dry-run' in sys.argv)
