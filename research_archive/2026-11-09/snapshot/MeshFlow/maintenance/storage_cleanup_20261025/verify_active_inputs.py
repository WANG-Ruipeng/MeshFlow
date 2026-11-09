"""CPU-only check of retained data references and current-round archive evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'maintenance/storage_cleanup_20261025'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def local(path):
    path = str(path).replace('\\', '/')
    if path.lower().startswith('e:/'):
        return Path('/mnt/e') / path[3:]
    if path.startswith('/mnt/e/'):
        return Path(path)
    return ROOT / path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    plan = read(OUT / 'prepared_manifest.json')
    removed = {local(row['Path']) for row in plan['Remove']}
    deps = read(OUT / 'dataset_active_dependencies.json')
    failures = []
    refs = deps['nonweight_references']
    for row in refs:
        path = local(row['path'])
        if path in removed:
            failures.append({'path': str(path), 'reason': 'DELETE_OVERLAP'})
        elif not path.exists():
            failures.append({'path': str(path), 'reason': 'MISSING'})
        elif row.get('kind') == 'file' and row.get('bytes') is not None and path.stat().st_size != row['bytes']:
            failures.append({'path': str(path), 'reason': 'SIZE_CHANGED'})
    archive = read(OUT / 'local_repair_archive_receipt.json')
    source = ROOT / 'experiments/r2_bench_raw_local_repair_v1'
    for name, identity in archive['member_hashes'].items():
        path = source / name
        if not path.is_file() or sha(path) != identity['sha256']:
            failures.append({'path': str(path), 'reason': 'CURRENT_ROUND_HASH_CHANGED'})
    zip_path = local(archive['path'])
    if sha(zip_path) != archive['sha256']:
        failures.append({'path': str(zip_path), 'reason': 'ARCHIVE_HASH_CHANGED'})
    result = {'status': 'PASS' if not failures else 'FAIL',
              'active_reference_count': len(refs),
              'current_round_source_hash_checks': len(archive['member_hashes']),
              'archive_hash_check': True,
              'failures': failures,
              'scope': 'Selected active manifest direct references; archived current-round original evidence. No model loading or scoring.'}
    (OUT / 'active_inputs_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
