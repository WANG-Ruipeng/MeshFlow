"""Archive actual executed source and seal evidence without Git mutation."""
from __future__ import annotations
import difflib, json
from pathlib import Path
from . import common as C


def run():
    complete=C.OUT/'COMPLETE.json'
    if complete.exists():r=C.read(complete);C.verify(r['files']);C.verify(r['protected']);C.verify(r['source_hashes']);return r
    required=['data/assets_COMPLETE.json','targets/COMPLETE.json','scores/COMPLETE.json',
              'analysis/COMPLETE.json','visual_review/COMPLETE.json','audits/independent_result_audit.json']
    for rel in required:
        r=C.read(C.OUT/rel);C.verify(r.get('files',{}))
        if rel.startswith('audits/') and r['status']!='PASS':raise ValueError('Independent audit did not pass')
    decision=C.read(C.OUT/'decision.json')
    if decision['status']!='NO_TRAINING_DATA_GATE_FAILED':
        raise RuntimeError('This closeout entry is only for the completed failed data gate; other outcomes require their actual training record')
    assert (C.OUT/'REPORT.md').is_file()
    m=C.read(C.OUT/'data/manifest.json');C.verify(m['protected_assets']);C.verify(m['derived_files'])
    scorer=C.read(C.OUT/'scores/COMPLETE.json')['source_hashes'];C.verify(scorer)
    source={str(p):C.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    source.update(scorer)
    archive={};skipped=[]
    for path,h in sorted(source.items()):
        p=C.resolve(path)
        try:rel=p.relative_to(C.ROOT)
        except ValueError:
            skipped.append(dict(path=path,sha256=h,reason='External dependency hash retained, not copied as repository code'));continue
        if p.suffix not in ('.py','.json','.md'):continue
        dest=C.OUT/'source_archive'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(p.read_bytes())
        assert C.sha(dest)==h;archive[str(dest)]=h
    diff=[]
    for p in sorted(Path(__file__).parent.glob('*.py')):
        rel=p.relative_to(C.ROOT).as_posix()
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=rel))
    with (C.OUT/'code.diff').open('x',encoding='utf-8') as f:f.writelines(diff)
    C.write(C.OUT/'source_hashes.json',dict(current_executed_and_audit_sources=source,archived_files=archive,external_not_copied=skipped,
        note='Actual original scoring source is included where repository-local. Old source files are unchanged; code.diff contains only the new pilot code.'))
    C.verify(m['protected_assets']);C.verify(source)
    files={str(p):C.sha(p) for p in sorted(C.OUT.rglob('*')) if p.is_file()}
    result=dict(status='COMPLETE_VERIFIED',experiment='R2_BENCH_RAW_LOCAL_REPAIR_V1',requests=12,parents=6,
        decision=decision['status'],model_calls=0,training_updates=0,new_weights=0,GPU_calls=0,new_noise=0,
        target_GEO_calls=12,reused_identity_GEO=12,CPU_cascade_jobs=24,full_score_rows=48,
        actual_visual_review='Nonblind AI_PROXY; actual image records retained',
        files_count=len(files),bytes=sum(p.stat().st_size for p in C.OUT.rglob('*') if p.is_file()),
        protected=m['protected_assets'],source_hashes=source,files=files,
        defaults_changed=False,cleanup_performed=False,git_commit_or_push=False)
    C.write(complete,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','source_hashes')},ensure_ascii=False));return result

if __name__=='__main__':run()
