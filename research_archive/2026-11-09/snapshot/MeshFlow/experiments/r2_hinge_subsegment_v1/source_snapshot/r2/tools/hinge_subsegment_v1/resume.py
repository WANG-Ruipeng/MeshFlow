"""Finite CPU experiment entry; completed scientific stages verify and skip."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import difflib
import json
import shutil
from pathlib import Path
from . import census as S

C, OUT = S.C, S.OUT


def event(name, **extra):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=C.utc(),event=name,**extra),ensure_ascii=False)+'\n')


def main():
    done=OUT/'RUN_COMPLETE.json'
    if done.exists():
        rec=C.read(done)
        for name in ('files','sources','protected'):
            S.O.B.verify(rec[name])
        print('COMPLETE_VERIFIED_NO_NEW_EXECUTION',flush=True)
        return rec
    event('RECOVERABLE_ENTRY',scope='Existing finite 28x2 CPU protocol; no new configuration')
    from . import audit_census, engine, audit_split, collect_scores, summarize, gallery, report
    stages=[('CENSUS',S.run),('CENSUS_AUDIT',audit_census.main),('SPLIT',engine.run),
            ('SPLIT_AUDIT',audit_split.main),('SCORES',collect_scores.run)]
    for name,call in stages:
        event('VERIFY_OR_RESUME',stage=name)
        call()
    # Derived tables can be reproduced, without any geometry or model execution.
    summarize.run()
    from . import audit_metrics
    audit_metrics.main()
    gallery.run()
    report.run()
    sources={str(p):C.sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.md')}
    for p in (OUT/'census_protocol.json',OUT/'split_protocol.json',OUT/'scores/completion.json'):
        sources.update(C.read(p).get('code',{}))
    sources.update(S.O.Q.definition_hashes())
    S.O.B.verify(sources)
    snapshot=OUT/'source_snapshot'
    index=[]
    for path,digest in sorted(sources.items()):
        source=C.resolve(path)
        if source.suffix not in ('.py','.json','.md'):
            continue
        try: relative=source.relative_to(C.ROOT)
        except ValueError: relative=Path('external')/digest[:12]/source.name
        dest=snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert C.sha(dest)==digest
        else:shutil.copyfile(source,dest)
        index.append(dict(source=str(source),snapshot=str(dest),sha256=digest))
    C.write(OUT/'source_hashes.json',dict(status='VERIFIED',files=index))
    diff=''
    for source in sorted(Path(__file__).parent.glob('*.py')):
        diff+=''.join(difflib.unified_diff([],source.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(source.relative_to(C.ROOT))))
    (OUT/'code.diff').write_text(diff,encoding='utf-8')
    protected=C.read(OUT/'census_protocol.json')['protected']
    S.O.B.verify(protected)
    C.resources(OUT/'resources/after_closeout.json')
    event('COMPLETE',decision='No DEV gain; one existing-positive TRAIN micro-repair, no default promotion',GPU_calls=0)
    files={str(p):C.sha(p) for p in OUT.rglob('*') if p.is_file() and p!=done}
    rec=dict(status='COMPLETE',GPU_calls=0,model_calls=0,training_updates=0,new_noise=0,new_D0=0,
             cells=56,candidate_attempts=8,accepted_unique_requests=1,DEV_bulk_ge30='0/12',
             default_changed=False,commit_push=False,protected=protected,sources=sources,files=files)
    C.write(done,rec)
    print(json.dumps({k:v for k,v in rec.items() if k not in ('protected','sources','files')}),flush=True)
    return rec


if __name__=='__main__':
    main()
