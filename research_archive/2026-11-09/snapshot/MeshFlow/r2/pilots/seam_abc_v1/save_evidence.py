"""Archive small actual scoring/pilot sources, their hashes, diff and retained weights."""
from pathlib import Path
import difflib
import json
import shutil
from . import eval_common as C

def main():
    out=C.OUT/'evidence'
    if out.exists():raise RuntimeError('Evidence already exists; inspect without overwriting')
    out.mkdir()
    registration=C.read(C.OUT/'evaluation/score_registration.json')
    source_rows=[]
    for filename,expected in registration['source_hashes'].items():
        p=Path(filename)
        if C.sha(p)!=expected:raise ValueError('Original scoring source changed: '+filename)
        rel=Path(*p.parts[3:]) if str(p).startswith('/mnt/e/') else Path(p.name)
        dest=out/'scoring_sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
        source_rows.append(dict(kind='ACTUAL_SCORING_SOURCE',source=str(p),saved=str(dest),bytes=p.stat().st_size,sha256=expected))
    package=Path(__file__).parent
    diff=[]
    for p in sorted(package.rglob('*')):
        if not p.is_file() or p.suffix not in ('.py','.md') or '__pycache__' in p.parts:continue
        rel=p.relative_to(package);dest=out/'pilot_sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
        source_rows.append(dict(kind='NEW_PILOT_SOURCE',source=str(p),saved=str(dest),bytes=p.stat().st_size,sha256=C.sha(p)))
        text=p.read_text(encoding='utf-8-sig')
        diff.extend(difflib.unified_diff([],text.splitlines(keepends=True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT))))
    (out/'code.diff').write_text(''.join(diff),encoding='utf-8')
    C.csvwrite(out/'source_hashes.csv',source_rows)
    weights=[]
    for folder in ('checkpoints','inference'):
        for p in sorted((C.OUT/folder).glob('*.pt')):
            weights.append(dict(path=str(p),bytes=p.stat().st_size,sha256=C.sha(p),
                purpose='MATCHED_RESUME_CHECKPOINT' if folder=='checkpoints' else 'FINAL_INFERENCE_EXPORT',
                checkpoint_selection='Fixed local step800 endpoint; rolling is recovery, not selected best'))
    C.csvwrite(out/'retained_weights.csv',weights)
    C.write(out/'receipt.json',dict(status='COMPLETE',sources=len(source_rows),weights=len(weights),
        original_sources_unchanged=True,weights_copied=False,model_calls=0,
        files={str(p):C.sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='COMPLETE',sources=len(source_rows),weights=len(weights))))

if __name__=='__main__':main()
