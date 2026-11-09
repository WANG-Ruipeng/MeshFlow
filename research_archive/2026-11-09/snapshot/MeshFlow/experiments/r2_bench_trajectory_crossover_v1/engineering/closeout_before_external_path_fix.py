"""Read-only asset verification and bounded evidence packaging."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import shutil, difflib, zipfile, argparse

def current_sources():
    reg=registration();sources=dict(reg['source_hashes']);fix=read(OUT/'engineering/score_wrapper_fix2.json')
    assert sources[fix['path']]==fix['frozen_sha256'];sources[fix['path']]=fix['corrected_sha256']
    verify(sources)
    sources.update({str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    return sources

def verify_complete():
    done=read(OUT/'COMPLETE.json');verify(done['files']);verify(read(OUT/'delivery_hashes.json')['files'])
    verify(registration()['files']);verify(read(OUT/'source_hashes.json')['files'])
    for combo in ('CU','UC'):
        rec=read(OUT/'execution'/combo/'complete.json');verify(rec['files'])
        for rp in rec['files']: verify(read(rp)['files'])
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='COMPLETE_VERIFIED',new_generator_calls=800,D0=0,updates=0,new_model_calls=0,
        complete_RAW_comparisons=64,reviewed_pages=16,zip_bytes=done['zip_bytes']),ensure_ascii=False))

def main():
    if (OUT/'COMPLETE.json').exists():verify_complete();return
    reg=registration();verify(reg['files']);sources=current_sources()
    result=read(OUT/'analysis/complete.json');verify(result['files'])
    assert result['outputs']==64 and result['visual_review_pages']==16
    for combo in ('CU','UC'):
        assert read(OUT/'execution'/combo/'complete.json')['generator_calls']==400
    assert read(OUT/'scores_complete.json')['rows']==32
    assert read(OUT/'gallery/complete.json')['pages']==16
    snapshot=OUT/'source_snapshot';diff=[]
    for path,digest in sorted(sources.items()):
        p=resolve(path);rel=p.relative_to(ROOT);dest=snapshot/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==digest
        else:shutil.copy2(p,dest)
        if p.is_relative_to(Path(__file__).parent):
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(rel)))
    save(OUT/'source_hashes.json',dict(files=sources,original_scorers_modified=False,score_wrapper_exception='engineering/score_wrapper_fix2.json'))
    text=''.join(diff);p=OUT/'code.diff'
    if p.exists():assert p.read_text(encoding='utf-8')==text
    else:p.write_text(text,encoding='utf-8')
    save(OUT/'resources/final.json',resources())
    small=[]
    for directory in ('analysis','gallery','engineering','resources','execution','source_snapshot'):
        for p in (OUT/directory).rglob('*'):
            if p.is_file() and p.suffix in ('.csv','.json','.jsonl','.md','.py','.png','.html'):small.append(p)
    small.extend(OUT/n for n in ('REPORT.md','registration.json','cpu_contract_checks.json','visual_review_notes.json',
        'source_hashes.json','code.diff','DISPATCH_PLAN.md','DISPATCH_SECOND.md','scores_complete.json','scores_first.json'))
    # Export exact original small scorer products for every cached/new output.
    for q in reg['requests']:
        for combo in COMBOS:
            source=score_path(combo,q['request_id']);dest=OUT/'score_evidence'/combo/q['request_id']
            for name in ('row.json','score.json','valid_edges.csv','seam_bulk_faces.csv','illegal_pairs_by_role.csv'):
                p=source/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert sha(target)==sha(p)
                else:shutil.copy2(p,target)
                small.append(target)
    small=sorted(set(small))
    assert all(p.exists() for p in small)
    save(OUT/'delivery_hashes.json',dict(files={str(p):sha(p) for p in small}))
    archive=OUT/'BENCH_TRAJECTORY_CROSSOVER_REVIEW.zip'
    if archive.exists():raise RuntimeError('Archive already exists without completion; inspect before packaging again')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in small+[OUT/'delivery_hashes.json']:z.write(p,p.relative_to(OUT).as_posix())
    assert archive.stat().st_size<10_000_000,archive.stat().st_size
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save(OUT/'COMPLETE.json',dict(status='COMPLETE',experiment='R2_BENCH_TRAJECTORY_CROSSOVER_V1',generator_started=800,
        generator_returned=800,GPU_replay=0,D0=0,head=0,training_updates=0,new_noise=0,new_OT=0,
        new_raw=32,reused_raw=32,metric_rows=64,reviewed_pages=16,reviewed_cells=64,
        CPU_wrapper_failed_invocations=2,CPU_requests_recomputed=1,zip_bytes=archive.stat().st_size,
        files={str(p):sha(p) for p in (archive,OUT/'delivery_hashes.json',OUT/'analysis/complete.json',OUT/'source_hashes.json')}))
    verify_complete()

if __name__=='__main__':main()
