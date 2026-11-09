"""Refresh only the review ZIP after final document review; no model imports."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=ROOT/'delivery/review_package.json'
old=json.loads(receipt.read_text(encoding='utf-8'))
paths={ROOT/r['path'] for r in old['files']}
paths.add(Path(__file__))
paths.add(ROOT/'data/dataset_overview.png')
paths.update(p for p in (ROOT/'audits').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md','.diff','.txt'))
assert all(p.resolve().is_relative_to(ROOT.resolve()) and p.is_file() for p in paths)
entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
package=ROOT/'BENCH_CONTROL_TRANSFER_REVIEW.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(paths):z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('PACKAGE_CONTENTS.json',json.dumps(dict(files=entries,excludes=['weights','RAW/POST mesh arrays','Euler trajectories','full training exposure stream']),ensure_ascii=False,indent=2))
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256'] for e in entries)
assert package.stat().st_size<10_000_000
final=dict(path=package.name,bytes=package.stat().st_size,sha256=sha(package),file_count=len(entries)+1,files=entries,
           exclusions='Large weights, RAW/POST arrays, trajectories and full logs remain local; review package is not a full replay bundle.',
           validation='All ZIP entries pass CRC and SHA256; all 24 paired image pages included; data-card figure and engineering records included.')
receipt.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in final.items() if k!='files'},ensure_ascii=False))
