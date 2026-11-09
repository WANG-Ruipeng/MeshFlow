"""Seal actual implementation and scorer sources after completed evaluation."""
from pathlib import Path
import difflib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from r2.pilots.bench_three_mechanisms_v1 import common as C

def main():
    done=C.read(C.OUT/'results/completion.json')
    assert done['status']=='COMPLETE' and done['rows']==252
    C.verify(done['files']);C.verify(done['source_rows'])
    files=set((ROOT/'r2/pilots/bench_three_mechanisms_v1').glob('*.py'))
    files.update((C.OUT/'control').glob('*.py'))
    score=C.read(C.OUT/'evaluation/score_registration_v2.json')
    for p in score['source_hashes']:
        path=C.resolve(p)
        if path.suffix.lower() in ('.py','.md','.json'):files.add(path)
    dest=C.OUT/'source_snapshot';manifest=[];diff=[]
    for path in sorted(files):
        try:
            rel=path.relative_to(ROOT)
        except ValueError:
            # The actual scorer also imports the retained sibling source tree.
            # Copy its small source files into an explicit external namespace.
            rel=Path('_external/E')/path.relative_to(Path('/mnt/e'))
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        digest=C.sha(path)
        if target.exists():assert C.sha(target)==digest
        else:shutil.copyfile(path,target)
        manifest.append(dict(path=str(path),sha256=digest,snapshot=str(target),bytes=path.stat().st_size))
        if path.parent==ROOT/'r2/pilots/bench_three_mechanisms_v1':
            diff.extend(difflib.unified_diff([],path.read_text('utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel.as_posix()))
    patch=C.OUT/'code.diff';content=''.join(diff)
    if patch.exists():assert patch.read_text('utf-8')==content
    else:patch.write_text(content,encoding='utf-8',newline='\n')
    result=dict(status='COMPLETE',meaning='New isolated pilot files; original scorer files copied read-only for inspection, not redefined',
        files=manifest,code_diff_sha256=C.sha(patch),scoring_registration_sha256=C.sha(C.OUT/'evaluation/score_registration_v2.json'),
        original_assets_changed=False,commit=False,push=False)
    C.write(C.OUT/'source_hashes.json',result)
    print(json.dumps(dict(status='COMPLETE',source_files=len(manifest),code_diff_bytes=patch.stat().st_size)))

if __name__=='__main__':main()
