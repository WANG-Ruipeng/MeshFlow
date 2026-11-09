"""Archive this experiment's source additions without touching Git or weights."""
from pathlib import Path
import difflib, shutil
from .common import ROOT,OUT,sha,read,write_new

def run():
    dest=OUT/'source_archive';receipt=dest/'receipt.json'
    if receipt.exists():
        r=read(receipt)
        assert all(sha(p)==h for p,h in r['archived_files'].items())
        return r
    roots=[ROOT/'r2/pilots/bench_seat_coverage_v1',ROOT/'r2/data_tools/bench_seat_coverage_v1']
    files=sorted(p for root in roots for p in root.rglob('*.py'))
    files += [ROOT/'r2/docs/bench_current_reference.md',ROOT/'r2/maintenance/bench_reference_20261023/retained_assets.json']
    files += sorted((OUT/'engineering').glob('*.py'))
    parts=[];hashes={};copies={}
    for p in files:
        relative=p.relative_to(ROOT).as_posix();q=dest/relative
        q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():raise RuntimeError('Partial archive retained; inspect before recovery')
        shutil.copyfile(p,q);hashes[relative]=sha(p);copies[str(q)]=sha(q)
        content=p.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        parts.append(''.join(difflib.unified_diff([],content,fromfile='/dev/null',tofile='b/'+relative)))
    diff=dest/'code.diff'
    with diff.open('x',encoding='utf-8',newline='') as f:f.write('\n'.join(parts))
    copies[str(diff)]=sha(diff)
    shared=read(OUT/'training_registration.json')['training_code_hashes']
    result=dict(status='COMPLETE',kind='Explicit new source additions, not whole dirty-worktree Git diff',
        sources=hashes,archived_files=copies,inherited_training_dependencies=shared,
        no_commit_push=True,no_weights_copied=True,model_calls=0)
    write_new(receipt,result);return result

if __name__=='__main__':print({k:v for k,v in run().items() if k not in ('sources','archived_files','inherited_training_dependencies')})
