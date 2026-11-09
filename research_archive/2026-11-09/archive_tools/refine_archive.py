"""Remove only generated archive copies outside the user's refined scope."""
import collections
import csv
import json
from pathlib import Path
from build_archive import OUT, category, write_csv

scope = json.loads((OUT/'SCOPE.json').read_text(encoding='utf-8'))
rows = [json.loads(x) for x in (OUT/'MANIFEST.jsonl').read_text(encoding='utf-8').splitlines()]
keep = []
removed = 0
removed_bytes = 0
for row in rows:
    kind = category(row['source'])
    if kind is None:
        target = (OUT/row['path']).resolve()
        if not target.is_relative_to(OUT.resolve()) or target.is_symlink():
            raise RuntimeError('Deletion escaped generated archive directory')
        target.unlink()
        removed += 1
        removed_bytes += row['bytes']
    else:
        row['category'] = kind
        keep.append(row)
with (OUT/'MANIFEST.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
    for row in keep:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
scope['files'] = len(keep)
scope['bytes'] = sum(x['bytes'] for x in keep)
scope['categories'] = dict(collections.Counter(x['category'] for x in keep))
scope['omitted_counts']['per_sample_transient_or_environment_records_refined_out'] = scope['omitted_counts'].pop('per_sample_or_transient_structured_records_refined_out', 0) + scope['omitted_counts'].get('per_sample_transient_or_environment_records_refined_out', 0) + removed
scope['refinement_note'] = 'User explicitly omitted detailed geometry diagnostics and machine ledgers; per-sample and transient structured records were also removed from the prepared copies. Original sources were untouched.'
(OUT/'SCOPE.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
valid = {x['path'] for x in keep}
with (OUT/'DOCUMENT_INDEX.csv').open(encoding='utf-8', newline='') as stream:
    documents = [x for x in csv.DictReader(stream) if x['path'] in valid]
write_csv(OUT/'DOCUMENT_INDEX.csv', documents, ['path', 'title', 'source', 'sha256'])
scope['document_index_rows'] = len(documents)
(OUT/'SCOPE.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
container_counts = collections.Counter(x.get('container') for x in keep if x.get('container'))
with (OUT/'ARCHIVE_CONTAINERS.csv').open(encoding='utf-8', newline='') as stream:
    containers = list(csv.DictReader(stream))
for row in containers:
    row['text_members'] = container_counts[row['source_container']]
write_csv(OUT/'ARCHIVE_CONTAINERS.csv', containers, ['archive_id', 'source_container', 'text_members'])
print(json.dumps({'removed_generated_copies': removed, 'removed_bytes': removed_bytes, 'kept_files': len(keep), 'kept_bytes': scope['bytes'], 'categories': scope['categories']}))
