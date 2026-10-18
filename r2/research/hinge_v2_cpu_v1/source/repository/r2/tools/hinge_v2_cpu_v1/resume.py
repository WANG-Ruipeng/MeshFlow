"""Recover the bounded experiment; completed runs only verify and exit."""
from __future__ import annotations
import difflib
import json
import shutil
from pathlib import Path
from . import run as S
C,B,OUT=S.C,S.B,S.OUT

def main():
    done=OUT/'RUN_COMPLETE.json'
    if done.exists():
        r=C.read(done)
        for name in ('files','sources','protected'):B.verify(r[name])
        print('COMPLETE_VERIFIED_NO_NEW_EXECUTION',flush=True);return r
    S.event('RECOVERABLE_ENTRY',scope='Fixed 28 requests, four CPU arms; completed geometry/scoring stages verify and skip')
    S.run()
    from . import collect_scores,audit,audit_score_pairs,audit_key_metrics,gallery,summarize,report
    collect_scores.run()
    # Independent audits are read-only; their own completion records prevent mutation.
    if not (OUT/'independent_audit.json').exists():audit.main()
    if not (OUT/'independent_score_pair_audit.json').exists():audit_score_pairs.main()
    if not (OUT/'independent_key_metrics_audit.json').exists():audit_key_metrics.main()
    summarize.main();gallery.run()
    for p in (OUT/'review_dev.csv',OUT/'gallery/review_train.csv',OUT/'gallery/review_train_complete.json'):
        if not p.exists():raise RuntimeError('Actual image review required: '+str(p))
    report.run()
    sources={str(p):C.sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.md')}
    for p in (OUT/'protocol.json',OUT/'scores/protocol.json'):sources.update(C.read(p)['code'])
    sources.update(S.Q.definition_hashes())
    sources[str(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')]=C.sha(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')
    B.verify(sources);index=[]
    for source,digest in sorted(sources.items()):
        p=C.resolve(source)
        try:relative=p.relative_to(C.ROOT)
        except ValueError:relative=Path('external')/digest[:12]/p.name
        dest=OUT/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert C.sha(dest)==digest
        else:shutil.copyfile(p,dest)
        index.append(dict(source=source,snapshot=str(dest),sha256=digest))
    C.write(OUT/'source_hashes.json',dict(status='VERIFIED',files=index))
    diff=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT)))) for p in sorted(Path(__file__).parent.glob('*.py')))
    dp=OUT/'code.diff'
    if dp.exists():assert dp.read_text(encoding='utf-8')==diff
    else:dp.write_text(diff,encoding='utf-8')
    reg=C.read(OUT/'protocol.json');B.verify(reg['protected']);C.resources(OUT/'resources/closeout.json')
    S.event('COMPLETE',decision='Bounded local positive; no default switch, training or publication',GPU_calls=0)
    files={str(p):C.sha(p) for p in OUT.rglob('*') if p.is_file() and p!=done}
    rec=dict(status='COMPLETE',geometry_cells=112,score_rows=140,model_calls=0,GPU_calls=0,training_updates=0,new_D0=0,new_noise=0,
        default_changed=False,commit_push=False,protected=reg['protected'],sources=sources,files=files)
    C.write(done,rec);print(json.dumps({k:v for k,v in rec.items() if k not in ('protected','sources','files')}),flush=True);return rec

if __name__=='__main__':main()
