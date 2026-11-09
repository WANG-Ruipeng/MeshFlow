"""CPU-only final delivery additions; never imports or calls a real model."""
from pathlib import Path
import hashlib
import json
import shutil

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf8')

def main():
    identity=json.loads((ROOT/'native_t1/runs/chair_domain_control_v1/protocol.json').read_text(encoding='utf8'))['source_scorer_identity']
    rows=[]
    for rel,sha in identity.items():
        p=ROOT/rel;assert digest(p)==sha
        target=OUT/'source_code/scorer_dependencies'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        rows.append(dict(original=rel,path=target.relative_to(OUT).as_posix(),sha256=digest(target),bytes=target.stat().st_size))
    write(OUT/'source_code/scorer_dependencies_manifest.json',rows)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.pt',) and p.name!='delivery_hashes.json']
    write(OUT/'delivery_hashes.json',{p.relative_to(OUT).as_posix():digest(p) for p in files})
    print('DELIVERY_EVIDENCE',len(rows),'actual scorer dependencies;',len(files),'small evidence files')

if __name__=='__main__':main()
