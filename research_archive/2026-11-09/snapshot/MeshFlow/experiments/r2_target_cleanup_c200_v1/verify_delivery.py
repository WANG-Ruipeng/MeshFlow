"""Read-only post-packaging verification; no scientific stage is invoked."""
import json
import zipfile
from pathlib import Path
from r2.pilots.target_cleanup_c200_v1.resume import resume

root = Path(__file__).resolve().parent
archive = root / 'R2_QUALITY_TARGET_CLEANUP_C200_REVIEW.zip'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert z.testzip() is None
    assert any(x.endswith('evaluation/tables/per_output.csv') for x in names)
    assert any('source_snapshot' in x and x.endswith('.py') for x in names)
    archive_result = dict(
        zip_integrity='PASS', bytes=archive.stat().st_size, files=len(names),
        paired_figures=sum(x.endswith('.jpg') and 'target_cleanup_c200_v1/evaluation/gallery/' in x for x in names),
        source_files=sum('source_snapshot' in x and x.endswith('.py') for x in names),
    )
result = dict(archive=archive_result, recovery=resume())
print(json.dumps(result, indent=2))
