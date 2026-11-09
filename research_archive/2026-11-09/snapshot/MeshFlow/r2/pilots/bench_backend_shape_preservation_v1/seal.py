"""Seal the completed bounded experiment; no models, generation or scoring."""
from __future__ import annotations
import datetime,difflib,shutil
from pathlib import Path
from .common import *

def run():
    final=OUT/'COMPLETE.json'
    if final.exists():
        r=read(final);verify(r['files']);return r
    required=['training/complete.json','evaluation/CANDIDATE_RAW/complete.json',
        'evaluation/OFFICIAL_SOURCE/complete.json','evaluation/CANDIDATE_SOURCE/complete.json',
        'scores/COMPLETE.json','source_analysis/COMPLETE.json','analysis/COMPLETE.json',
        'visual_review/COMPLETE.json','audits/training_exact_budget.json',
        'audits/results_review.json','audits/independent_audit_final.json',
        'audits/source_visual_review.json','REPORT.md']
    for name in required:
        if not (OUT/name).is_file():raise RuntimeError('Required deliverable missing: '+name)
    train=read(OUT/required[0]);assert train['counts']==dict(forwards=2000,backwards=2000,updates=500)
    for arm in ('CANDIDATE_RAW','OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
        r=read(OUT/'evaluation'/arm/'complete.json');assert r['outputs']==30 and r['new_forwards']==30
        ledger=[json.loads(s) for s in (OUT/'evaluation'/arm/'ledger.jsonl').read_text().splitlines()]
        assert sum(r['event']=='FORWARD_STARTED' for r in ledger)==30
        assert sum(r['event']=='FORWARD_RETURNED' for r in ledger)==30
    assert read(OUT/'audits/training_exact_budget.json')['status']=='PASS'
    assert read(OUT/'audits/results_review.json')['status']=='PASS_NO_ARITHMETIC_OR_IDENTITY_MISMATCH'
    assert read(OUT/'audits/independent_audit_final.json')['status']=='PASS'
    assert read(OUT/'visual_review/COMPLETE.json')['cells']==150
    # Exact old inputs and old scorers remain protected; no old outputs overwritten.
    registration=read(OUT/'evaluation/CPU_registration.json')
    verify(registration['protected']);verify(registration['source_hashes'])
    verify(read(OUT/'scores/COMPLETE.json')['reference_hashes'])
    verify(read(OUT/'data/manifest.json')['protected_inputs'])
    sources=dict(registration['source_hashes'])
    for p in Path(__file__).parent.glob('*.py'):sources[str(p)]=sha(p)
    archived={}
    for path,digest in sources.items():
        p=resolve(path)
        try:rel=p.relative_to(ROOT)
        except ValueError:rel=Path('external')/p.name
        target=OUT/'source_archive'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==digest
        else:shutil.copyfile(p,target)
        archived[str(target)]=digest
    write(OUT/'source_hashes.json',dict(sources=sources,archived=archived))
    diff=''
    for p in sorted(Path(__file__).parent.glob('*.py')):
        rel=p.relative_to(ROOT).as_posix()
        diff+=''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
    with (OUT/'code.diff').open('x',encoding='utf-8') as f:f.write(diff)
    analysis=read(OUT/'analysis/COMPLETE.json');visual=read(OUT/'visual_review/COMPLETE.json')
    result=dict(status='COMPLETE',decision='NOT_PROMOTED',default_changed=False,
        reason='Preservation signal retained; predeclared real-output geometry gate failed.',
        generator_forwards=0,training_logical_forwards=2000,backwards=2000,optimizer_updates=500,
        evaluation_D0_forwards=90,new_free_generation_noise=0,real_RAW_outputs=30,
        new_original_metric_rows=60,readonly_metric_rows=210,original_metric_fields=177,
        actual_visual_pages=30,actual_visual_cells=150,source_identity_visual_pages=15,
        engineering_failure_count=1,scientific_replay_updates=0,
        geometry_alert_old_parents=analysis['geometry_alert_old_parents'],
        visual_all_groups_nondecrease=visual['all_groups_visual_nondecrease'],
        no_commit_or_push=True,no_old_asset_cleanup=True,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # One full hash pass, including actual weights. The verifier is read-only.
    result['files']={str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(final,result)
    return result
if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k!='files'}))
