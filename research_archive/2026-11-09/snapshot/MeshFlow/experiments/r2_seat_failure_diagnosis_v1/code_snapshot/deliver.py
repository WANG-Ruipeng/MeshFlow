"""Snapshot exact small sources and package completed evidence; no model loading."""
import difflib, json, zipfile
from pathlib import Path
from . import common as E

def run():
    definition=E.read(E.OUT/'scoringlogs/definition.json');sources=[]
    for name,h in definition['score_hashes'].items():
        p=E.resolve(name)
        if E.sha(p)!=h:raise ValueError('Executed scoring source changed '+name)
        rel=Path('workspace')/p.relative_to(E.ROOT) if p.is_relative_to(E.ROOT) else Path('MeshFlow-Control')/p.relative_to(E.ROOT.parent/'MeshFlow-Control')
        target=E.OUT/'scoring_source'/rel;E._write_bytes(target,p.read_bytes())
        sources.append(dict(original_path=str(p),copied_path=str(target),sha256=h,bytes=p.stat().st_size,origin='ACTUAL_EXECUTED_SCORING_DEFINITION'))
    new=[];diff=[]
    for p in sorted(E.CODE.glob('*.py')):
        target=E.OUT/'code_snapshot'/p.name;E._write_bytes(target,p.read_bytes())
        new.append(dict(original_path=str(p),copied_path=str(target),sha256=E.sha(p),bytes=p.stat().st_size,origin='NEW_ISOLATED_CODE'))
        lines=p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        diff.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile=str(p.relative_to(E.ROOT))))
    E.write(E.OUT/'source_file_hashes.json',dict(scoring=sources,new_code=new,old_code_modified=False))
    E._write_bytes(E.OUT/'code.diff',''.join(diff).encode('utf8'))
    # This intentionally packages analysis evidence, not bulky trajectories or weights.
    roots=['code_snapshot','scoring_source','statistics','coverage_audit','scoringlogs','independent_audit','resources','execution','evaluation/gallery']
    include=set()
    for sub in roots:
        for p in (E.OUT/sub).rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.jsonl','.csv','.md','.txt','.py','.diff','.jpg','.html'):
                include.add(p)
    for p in (E.OUT/'source_audit').iterdir():
        if p.is_file() and p.suffix.lower() in ('.json','.csv','.md','.py'):include.add(p)
    for p in (E.OUT/'source_audit/engineering').rglob('*'):
        if p.is_file():include.add(p)
    for name in ['REPORT.md','registration.json','requests.csv','conditions.csv','subdivision_audit.json','dispatch_plan.json','resources_windows_before_dispatch.json','resources_before_generator_worker1.json','resources_before_D0_worker1.json','source_file_hashes.json','code.diff','ledger.jsonl','evaluation/raw_index.json','evaluation/post_index.json']:
        include.add(E.OUT/name)
    rows=[dict(path=str(p.relative_to(E.OUT)),sha256=E.sha(p),bytes=p.stat().st_size) for p in sorted(include)]
    manifest=dict(status='COMPLETE_EVIDENCE_PACKAGE',files=rows,omissions=['large weights','RAW/POST mesh arrays and full Euler trajectories','per-face pair JSON/vectors available locally under evaluation/scores','source parent NPZ assets'],scope='Recomputable aggregation and evidence review; full geometry rescoring uses retained local arrays bound by hashes',size_limit_bytes=10_000_000)
    E.write(E.OUT/'package_manifest.json',manifest);include.add(E.OUT/'package_manifest.json')
    zip_path=E.OUT/'R2_SEAT_FAILURE_DIAGNOSIS_REVIEW.zip'
    if zip_path.exists():raise ValueError('Do not overwrite completed package')
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(include):z.write(p,str(p.relative_to(E.OUT)))
    if zip_path.stat().st_size>=10_000_000:raise ValueError('Evidence package exceeds10MB; preserve and explicitly revise manifest, not silent delete')
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise ValueError('Bad archive CRC')
        assert len(z.namelist())==len(include)
        for row in rows:
            import hashlib
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise ValueError('Archive payload mismatch')
    r=E.load_registration();reviews=E.read(E.OUT/'evaluation/gallery/review_completion.json');assert reviews['unique_output_stage_rows']==80
    complete=dict(status='COMPLETE',utc=E.utc(),new_rollouts=24,generator_forwards=1200,D0_forwards=24,training_updates=0,backwards=0,logical_RAW=40,logical_POST=40,scored_stages=80,reviewed_stages=80,actual_review_pages=24,resume_through_render_executed=True,resume_new_forwards=0,code_or_model_default_changed=False,git_commit_or_push=False,package_path=str(zip_path),package_bytes=zip_path.stat().st_size,package_sha256=E.sha(zip_path),report_sha256=E.sha(E.OUT/'REPORT.md'),package_files=len(include),registration_sha256=E.sha(E.OUT/'registration.json'))
    E.write(E.OUT/'COMPLETE.json',complete);print(json.dumps(complete,ensure_ascii=False))
if __name__=='__main__':run()