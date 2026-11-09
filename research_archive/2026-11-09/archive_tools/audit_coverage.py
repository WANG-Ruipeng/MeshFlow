import csv
import hashlib
import json
import subprocess
from pathlib import Path
from build_archive import ROOTS, SKIP, OUT, category, write_csv

rows = [json.loads(x) for x in (OUT/'MANIFEST.jsonl').read_text(encoding='utf-8').splitlines()]
sources = {x['source'].casefold() for x in rows if not x.get('container')}
documents = []
added = []
expected = 0
for root in ROOTS:
    cmd = ['rg', '--files', '--hidden', '--no-ignore', '-g', '*.md', '-g', '*.rst']
    for name in sorted(SKIP):
        cmd += ['-g', '!'+name]
    cmd.append(str(root))
    result = subprocess.run(cmd, capture_output=True, check=True)
    for raw in result.stdout.decode().splitlines():
        p = Path(raw)
        if category(p) != 'document':
            continue
        expected += 1
        if str(p).casefold() not in sources:
            rel = Path('snapshot')/root.name/p.relative_to(root)
            data = p.read_bytes()
            target = OUT/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row = {'path': rel.as_posix(), 'source': str(p), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'category': 'document'}
            rows.append(row)
            added.append(rel.as_posix())
with (OUT/'MANIFEST.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
    for row in sorted(rows, key=lambda x:x['path']):
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
for row in rows:
    if row['category'] == 'document' and row['path'].lower().endswith('.md'):
        text = (OUT/row['path']).read_text(encoding='utf-8-sig', errors='replace')
        title = next((x.lstrip('# ').strip() for x in text.splitlines() if x.startswith('#')), Path(row['path']).name)
        documents.append({'path': row['path'], 'title': title, 'source': row['source'], 'sha256': row['sha256']})
write_csv(OUT/'DOCUMENT_INDEX.csv', sorted(documents, key=lambda x:x['path']), ['path', 'title', 'source', 'sha256'])
scope = json.loads((OUT/'SCOPE.json').read_text(encoding='utf-8'))
scope.update(files=len(rows), bytes=sum(x['bytes'] for x in rows), document_index_rows=len(documents))
scope['categories']['document'] += len(added)
scope['primary_document_coverage'] = {'status':'PASS','markdown_rst_files_in_scope':expected,'recovered_after_filter_review':added,'missing':0}
(OUT/'SCOPE.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(scope['primary_document_coverage']))
