"""Copy actual small source files; no models, scoring, or scientific mutation."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_bench_full_prior_motif_guidance_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def immutable(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != data:
            raise ValueError(f'Archive mismatch: {p}')
    else:
        with p.open('xb') as f:
            f.write(data)

def main():
    registered = json.loads((OUT / 'evaluation_code.json').read_text())['source_hashes']
    entries = []
    for name, expected in sorted(registered.items()):
        p = Path(name)
        if sha(p) != expected:
            raise ValueError(f'Registered source changed: {p}')
        rel = p.relative_to('/mnt/e')
        target = OUT / 'archive/sources' / rel
        immutable(target, p.read_bytes())
        entries.append(dict(source=str(p), archive=str(target.relative_to(OUT)),
                            sha256=expected, bytes=p.stat().st_size, role='REGISTERED_EVALUATOR_OR_DEPENDENCY'))
    diffs = []
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix not in ('.py', '.md'):
            continue
        target = OUT / 'archive/sources/MeshFlow' / p.relative_to(ROOT)
        immutable(target, p.read_bytes())
        entries.append(dict(source=str(p), archive=str(target.relative_to(OUT)),
                            sha256=sha(p), bytes=p.stat().st_size, role='ACTUAL_PILOT_SOURCE'))
        diffs.extend(difflib.unified_diff([], p.read_text().splitlines(keepends=True),
                     fromfile='/dev/null', tofile='b/' + str(p.relative_to(ROOT))))
    immutable(OUT / 'code.diff', ''.join(diffs).encode())
    manifest = dict(schema='actual_source_archive_v1', records=entries,
        code_diff_basis='Pilot directory is new/untracked; unified additions against /dev/null. Registered original dependencies copied byte-identically, not modified.',
        official_weight_files_not_copied=True, old_generation_revisions='engineering_incidents/01_initial_journal and 02_closed_C_export retain previous generation source identities')
    immutable(OUT / 'SOURCE_HASHES.json', (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode())
    reg = json.loads((OUT / 'registration.json').read_text())
    models = []
    for m in reg['models']:
        p=Path(m['path']); s=p.stat()
        if s.st_size != m['bytes'] or s.st_mtime_ns != m['mtime_ns']:
            raise ValueError(f'Protected model stat changed: {p}')
        models.append(dict(role=m['role'],path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                           registered_sha256=m['sha256'],check='SIZE_MTIME_UNCHANGED; original SHA authenticated in registered model loading; no redundant final multi-GB hash'))
    receipt = dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
                   source_rows=len(entries),unique_archived_files=len({e['archive'] for e in entries}),
                   models=models,source_manifest_sha256=sha(OUT/'SOURCE_HASHES.json'),
                   code_diff_sha256=sha(OUT/'code.diff'),new_model_calls=0,new_scores=0)
    f=OUT/'archive/source_archive_receipt.json'
    if not f.exists(): immutable(f,(json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(dict(status='COMPLETE',source_rows=len(entries),unique_files=receipt['unique_archived_files'])))

if __name__ == '__main__':
    main()
