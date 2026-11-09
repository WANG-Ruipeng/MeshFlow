"""Make a <10 MB review copy without omitting per-output metrics or actual code."""
import hashlib, json, zipfile
from pathlib import Path
p=Path(__file__).resolve().parent
source=p/'R2_BENCH_NATURAL_N_RECIPE_REVIEW.zip'
dest=p/'R2_BENCH_NATURAL_N_RECIPE_REVIEW_SMALL.zip'
assert not dest.exists()
omit={'data/schedule_CONTROL_OLD.json','data/schedule_NATURAL_HIGH.json','training_closeout/exposure_identity_checks.csv'}
hashes={}
with zipfile.ZipFile(source) as zin, zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zout:
    for member in zin.infolist():
        data=zin.read(member.filename)
        if member.filename in omit:
            hashes[member.filename]=hashlib.sha256(data).hexdigest()
        else:zout.writestr(member,data)
    assert set(hashes)==omit
    note={'purpose':'Upload-friendly <10,000,000-byte copy; scientific result rows, actual visual labels, 24 high-N pages and actual scorer code retained.',
          'omitted_redundant_expanded_training_records':hashes,
          'retained':'Original per-parent exposure/loss summaries, deterministic stream construction code, registration and all per-output result/visual/face-fate CSVs.',
          'full_local_archive':source.name,'full_archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'scientific_calls':0,'package_engineering':'An earlier inline shell zip-inspection command had a quoting SyntaxError; no output or scientific computation was affected.'}
    zout.writestr('SMALL_PACKAGE_SCOPE.json',json.dumps(note,ensure_ascii=False,indent=2))
    zout.writestr('make_small_review.py',Path(__file__).read_bytes())
assert dest.stat().st_size<10_000_000
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
receipt=dict(status='COMPLETE',bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),file=dest.name,
             full_archive=source.name,omitted=hashes,model_calls=0,scoring_calls=0)
(p/'small_package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
