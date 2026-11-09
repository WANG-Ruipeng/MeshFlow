import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE/'publish/research_archive/2026-11-09'

scope = json.loads((OUT/'SCOPE.json').read_text(encoding='utf-8'))
if scope['failures']:
    raise SystemExit('Resolve all archive preparation failures first.')
tools_dir = OUT/'archive_tools'
tools_dir.mkdir(exist_ok=True)
for name in ('inventory.py', 'build_archive.py', 'refine_archive.py', 'audit_coverage.py', 'finalize_archive.py'):
    shutil.copyfile(HERE/name, tools_dir/name)

# Check the new handoff links, not intentionally unchanged historical links.
errors = []
for name in ('README.md', 'RESUME_RESEARCH.md'):
    text = (OUT/name).read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if '://' not in target and not (OUT/target.split('#')[0]).exists():
            errors.append({'file': name, 'missing_link': target})
if errors:
    print(json.dumps(errors))
    raise SystemExit('Handoff links must resolve before upload.')

rows = []
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name != 'CHECKSUMS.json' and '__pycache__' not in p.parts:
        data = p.read_bytes()
        rows.append({'path': p.relative_to(OUT).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(OUT/'CHECKSUMS.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'files': len(rows)+1, 'payload_bytes': sum(x['bytes'] for x in rows), 'largest_bytes': max(x['bytes'] for x in rows), 'handoff_links': 'PASS'}))
