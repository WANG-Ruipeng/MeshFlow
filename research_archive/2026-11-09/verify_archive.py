"""Read-only SHA256 verification. Uses only the Python standard library."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parent
    rows = json.loads((root/'CHECKSUMS.json').read_text(encoding='utf-8'))
    errors = []
    total = 0
    for row in rows:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or any(':' in x for x in rel.parts):
            raise ValueError('Unsafe manifest path')
        path = root.joinpath(*rel.parts)
        if not path.is_file():
            errors.append({'path': row['path'], 'reason': 'missing'})
            continue
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                sha.update(chunk)
        size = path.stat().st_size
        total += size
        if sha.hexdigest() != row['sha256'] or size != row['bytes']:
            errors.append({'path': row['path'], 'reason': 'content_mismatch'})
    expected = {x['path'] for x in rows} | {'CHECKSUMS.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for path in sorted(actual - expected):
        errors.append({'path': path, 'reason': 'unexpected_file'})
    result = {'status': 'PASS' if not errors else 'FAIL', 'checked_files': len(rows), 'bytes': total, 'errors': errors}
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
