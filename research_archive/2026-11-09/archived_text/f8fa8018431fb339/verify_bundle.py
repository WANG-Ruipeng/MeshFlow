from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
errors=[]
for row in manifest['files']:
    p=(root/row['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        errors.append(row['path']);continue
    data=p.read_bytes()
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
        errors.append(row['path'])
print(json.dumps(dict(status='PASS' if not errors else 'FAIL',files=len(manifest['files']),errors=errors)))
raise SystemExit(bool(errors))
