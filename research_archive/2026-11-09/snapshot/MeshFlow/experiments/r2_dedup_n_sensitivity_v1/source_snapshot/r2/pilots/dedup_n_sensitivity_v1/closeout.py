"""Verify and package completed A; never starts B or calls a model."""
import difflib,json,shutil,zipfile
from pathlib import Path
from . import common as C
from . import evaluate as E
from .decide import calculate

def main():
    E.check_code();r=C.load_registration();gate=calculate()
    assert gate['actual_visual'] is not None
    assert gate['status']=='FAIL','A positive gate requires completing the authorized B before final closeout'
    audit=C.read(C.OUT/'audit/final.json');assert audit['status']=='PASS'
    vis=C.csvread(C.OUT/'evaluation/gallery/actual_visual_review.csv')
    delta=C.csvread(C.OUT/'evaluation/gallery/per_request_visual_delta.csv')
    assert len(vis)==192 and len(delta)==48
    assert len({(x['label'],x['request_id'],x['stage']) for x in vis})==192
    assert (C.OUT/'REPORT.md').is_file()
    for path,digest in r['protected'].items():assert C.sha(path)==digest,path
    C.write(C.OUT/'protection_audit.json',dict(status='PASS',protected_files=len(r['protected']),all_unchanged=True,
        old_raw_scores_weights_readonly=True,model_calls=0))
    C.write(C.OUT/'WEIGHTS.json',dict(new_weights=0,training_updates=0,bindings=r['models'],
        note='Reference existing immutable step200 exports; do not copy or delete old weights. B was not run.'))
    freeze=C.read(C.OUT/'evaluation_freeze.json')
    sources=set(freeze['evaluation_code_hashes'])
    for folder in (C.CODE,C.ROOT/'r2/tools/dedup_n_sensitivity_audit_v1'):
        sources.update(str(p) for p in folder.rglob('*.py'))
    records=[];diff=[]
    for raw in sorted(sources):
        p=C.resolve(raw)
        rel=p.relative_to(C.ROOT) if p.is_relative_to(C.ROOT) else Path('external')/p.relative_to('/mnt/e')
        out=C.checked(C.OUT/'source_snapshot'/rel);out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():shutil.copyfile(p,out)
        assert C.sha(out)==C.sha(p)
        records.append(dict(path=str(p),sha256=C.sha(p),snapshot=str(out)))
        if p.is_relative_to(C.CODE):
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+str(rel)))
    C.csvwrite(C.OUT/'source_hashes.csv',records)
    C._write_bytes(C.OUT/'code.diff',''.join(diff).encode('utf-8'))
    C.write(C.OUT/'resources/final.json',C.resources())
    files=[]
    for p in sorted(C.OUT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.md','.csv','.py','.diff','.html'):
            files.append(p)
    files.extend(sorted((C.OUT/'evaluation/gallery').glob('*.jpg')))
    for rel in ('registration.json','B_GATE.json','evaluation_freeze.json','WEIGHTS.json','protection_audit.json',
        'analysis/gate_decision.json','audit/final.json','audit/preflight.json','audit/legacy_deletion_audit.json',
        'resources/generator_dispatch_decision.json','resources/D0_dispatch_decision.json'):
        files.append(C.OUT/rel)
    archive=C.OUT/'R2_DEDUP_N_SENSITIVITY_V1_REVIEW.zip'
    if archive.exists():raise FileExistsError('Do not overwrite evidence archive')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(C.ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        filecount=len(z.namelist())
    C.write(C.OUT/'archive_manifest.json',dict(path=str(archive),sha256=C.sha(archive),bytes=archive.stat().st_size,
        file_count=filecount,files=[dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p),bytes=p.stat().st_size) for p in sorted(set(files))],
        exclusions='Large weights and raw NPZ stay at recorded local immutable paths. All per-output/per-face CSV, scoring code and 24 figures are included.'))
    final=[C.OUT/p for p in ('REPORT.md','B_GATE.json','WEIGHTS.json','analysis/gate_decision.json','analysis/main_summary.csv',
        'evaluation/tables/per_output.csv','evaluation/gallery/actual_visual_review.csv','evaluation/gallery/per_request_visual_delta.csv',
        'audit/final.json','source_hashes.csv','code.diff')]+[archive]
    C.write(C.OUT/'completion.json',dict(status='COMPLETE',A_status='EXECUTED_GATE_FAILED',B_status='NOT_RUN_GATE_NOT_MET',
        deliverable_hashes={str(p):C.sha(p) for p in final},new_rollouts=48,new_sampling_forwards=2400,new_D0_forwards=48,
        new_quality_scans=96,cached_quality_scores=96,actual_visual_rows=192,actual_pair_reviews=48,
        training_updates=0,backwards=0,new_weights=0,default_changed=False,commit_or_push=False))
    C.event('EXPERIMENT_COMPLETE',A='EXECUTED_GATE_FAILED',B='NOT_RUN_GATE_NOT_MET',archive_sha256=C.sha(archive))
    return dict(status='COMPLETE',B='NOT_RUN_GATE_NOT_MET',archive=str(archive),bytes=archive.stat().st_size,files=filecount)

if __name__=='__main__':print(json.dumps(main(),indent=2))
