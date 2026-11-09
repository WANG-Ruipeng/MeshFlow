"""Seal this completed bounded experiment and archive the actual source files."""
from __future__ import annotations
import ast
import csv
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
CODE=ROOT/'r2/pilots/bench_geo_only_acceptance_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def resolve(p):
    s=str(p).replace('\\','/')
    if str(ROOT).startswith('/mnt/e/') and s.startswith('E:/'):s='/mnt/e/'+s[3:]
    elif str(ROOT).startswith('E:') and s.startswith('/mnt/e/'):s='E:/'+s[7:]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p
def write(p,x):
    if p.exists():assert read(p)==x,str(p);return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify_book(book):
    for p,h in book.items():assert sha(resolve(p))==h,str(p)

def run():
    if (OUT/'COMPLETE.json').exists():
        from .run import verify
        return verify()
    m=read(OUT/'manifest.json');core=read(OUT/'core_complete.json');scores=read(OUT/'scores/COMPLETE.json')
    vis=read(OUT/'visual_review/COMPLETE.json');ident=read(OUT/'identity/COMPLETE.json')
    gate=read(OUT/'acceptance_gate.json');reg=read(OUT/'identity/registration.json')
    review=read(OUT/'identity/gallery/actual_review.json')
    audit=read(OUT/'audits/reference_review.json')
    assert core['new_GEO']==26 and core['reused_GEO']==4 and scores['new_full_scores']==30
    assert len(scores['metric_fields'])==177 and scores['old_full_scores_reused']==180
    assert vis['cells']==150 and vis['pages']==30 and gate['status']=='FAIL'
    assert ident['D0_forwards_total']==2 and ident['GEO_calls_total']==2 and ident['training']==0
    assert len(review['pages'])==2 and all(p['actually_viewed'] for p in review['pages'])
    for p in review['pages']:assert sha(resolve(p['image_path']))==p['image_sha256']
    for name in ['REPORT.md','RUNBOOK.md','backend_preparation/inventory.json','backend_preparation/PREPARATION.md','backend_preparation/IDENTITY_UPDATE.md']:
        assert (OUT/name).is_file()
    verify_book(m['protected']);verify_book(m['source_hashes'])
    verify_book(scores['original_score_hashes']);verify_book(scores['source_hashes'])
    verify_book(reg['protected_inputs']);verify_book(reg['code_hashes'])
    for rec in [core,scores,vis,ident]:verify_book(rec['files'])
    events=[json.loads(x) for x in (OUT/'identity/calls.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(r['event']=='D0_STARTED' for r in events)==2
    assert sum(r['event']=='D0_RETURNED' for r in events)==2
    assert sum(r['event']=='GEO_RETURNED' for r in events)==2
    # Archive the implementation actually used, plus all recorded scorer/geometry
    # dependencies. This is an independent source archive, not a Git mutation.
    paths=set(CODE.glob('*.py'))
    for book in [m['source_hashes'],scores['source_hashes'],reg['code_hashes']]:
        for p,h in book.items():
            path=resolve(p)
            if path.suffix in ['.py','.json','.yaml','.yml']:paths.add(path)
    paths.add(ROOT/'r2/pilots/bench_backend_counterfactual_v1/render.py')
    sources={}
    for p in sorted(paths):
        try:rel=p.relative_to(ROOT)
        except ValueError:
            # The frozen production scorer also imports the sibling maintained
            # MeshFlow-Control checkout. Read/copy it, never modify that source.
            rel=Path('external_dependencies/MeshFlow-Control')/p.relative_to(ROOT.parent/'MeshFlow-Control')
        target=OUT/'source_archive'/rel
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(rel))
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(p) and p==Path(__file__):
            # Only this unfinished packaging utility changed after the recorded
            # path correction. Preserve its prior archive before refreshing it.
            prior=OUT/'source_archive_interrupted'/('closeout_'+sha(target)+'.py')
            prior.parent.mkdir(parents=True,exist_ok=True)
            if not prior.exists():shutil.copyfile(target,prior)
            shutil.copyfile(p,target)
        if not target.exists():shutil.copyfile(p,target)
        assert sha(target)==sha(p)
        sources[str(rel.as_posix())]=dict(original_path=str(p),sha256=sha(p),bytes=p.stat().st_size,archive=str(target.relative_to(OUT).as_posix()))
    write(OUT/'source_hashes.json',dict(scope='Actual current implementation and recorded dependencies; official checkpoint identity lives in identity registration/receipts',files=sources))
    diff=[]
    for p in sorted(CODE.glob('*.py')):
        rel=p.relative_to(ROOT).as_posix()
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile=rel))
    difftext=''.join(diff)
    diffpath=OUT/'code.diff'
    if diffpath.exists():assert diffpath.read_text(encoding='utf-8')==difftext
    else:diffpath.write_text(difftext,encoding='utf-8')
    ledger=dict(schema='BENCH_GEO_ONLY_ACCEPTANCE_CLOSEOUT_V1',status='COMPUTE_AND_REVIEW_COMPLETE',
        new_training_updates=0,backward_calls=0,generator_forward_calls=0,new_rollouts=0,new_noise_arrays=0,new_OT=0,
        D0_forward_calls=2,new_generated_GEO_calls=26,reused_generated_GEO=4,new_source_identity_GEO_calls=2,
        full_score_calls=30,readonly_existing_score_rows=180,unique_new_generator_samples=0,
        full_metric_fields=177,main_visual_pages=30,main_visual_cells=150,source_identity_pages=2,source_identity_cells=10,
        automatic_model_retries=0,scientific_stage_failures=0,packaging_stopped_attempts=2,packaging_path_fixes=1,CPU_score_workers=3,CPU_render_workers=1,GPU_workers=1,
        code_and_source_archive_files=len(sources),default_changed=False,commit=False,push=False,cleanup=False,
        interpretation='Local structural positives retained; uniform backend rejected on preregistered geometry cost; prepare identity/thin-structure backend supervision, no unbudgeted training.')
    write(OUT/'execution_summary.json',ledger)
    files={str(p.relative_to(ROOT).as_posix()):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','COMPLETE.json']}
    # Include nested COMPLETE receipts, excluding only root seal itself.
    for p in OUT.rglob('COMPLETE.json'):
        if p.parent!=OUT:files[str(p.relative_to(ROOT).as_posix())]=sha(p)
    for p in CODE.glob('*.py'):files[str(p.relative_to(ROOT).as_posix())]=sha(p)
    write(OUT/'artifact_hashes.json',dict(files=files,scope='All completed task artifacts and task source; original assets additionally protected by nested manifests'))
    complete=dict(status='COMPLETE',experiment_status='UNIFORM_GEO_ONLY_NOT_ACCEPTED_WITH_BOUNDED_POSITIVES',
        artifact_manifest_sha256=sha(OUT/'artifact_hashes.json'),artifacts=len(files),
        execution_summary_sha256=sha(OUT/'execution_summary.json'),report_sha256=sha(OUT/'REPORT.md'),
        next_stage='Backend data/contract preparation complete; no training until reliable targets and fixed budget exist',
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(OUT/'COMPLETE.json',complete)
    from .run import verify
    verify()

if __name__=='__main__':run()
