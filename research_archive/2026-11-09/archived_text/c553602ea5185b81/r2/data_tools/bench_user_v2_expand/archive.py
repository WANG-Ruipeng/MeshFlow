"""Read-only final verification and a small code/evidence archive. No model or OT."""
from pathlib import Path
import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/r2_bench_user_data_v2_expansion_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sealed(path, data):
    data = data.encode('utf-8') if isinstance(data, str) else data
    if path.exists():
        assert path.read_bytes() == data, f'Preserved artifact differs: {path}'
    else:
        path.write_bytes(data)


def json_sealed(path, value):
    sealed(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def main():
    checked = subprocess.run(
        [sys.executable, '-B', '-m', 'r2.data_tools.bench_user_v2_expand.finalize', 'verify'],
        cwd=ROOT, text=True, capture_output=True, check=True)
    links = sorted(set(html.unescape(p) for p in re.findall(
        r'(?:href|src)="([^"]+)"', (OUT / 'index.html').read_text(encoding='utf-8'))))
    for link in links:
        assert (OUT / link).resolve().is_file(), link
    manifest = read(OUT / 'manifest.json')
    summary = read(OUT / 'final_summary.json')
    assert len(manifest['train']) == summary['accepted_conditions'] == 81
    assert len({r['parent_id'] for r in manifest['train']}) == summary['accepted_parents'] == 25
    assert len(manifest['dev']) == 39
    latest = read(OUT / 'validation/latest.json')
    receipt = read(Path(latest['receipt']))
    assert receipt['torch_imported'] is False
    assert receipt['new_CPU_OT_completed'] == receipt['new_CPU_OT_attempts'] == 63
    source = sorted(p for p in CODE.iterdir() if p.suffix in ('.py', '.ps1'))
    diff = ''.join(''.join(difflib.unified_diff(
        [], p.read_text(encoding='utf-8').splitlines(keepends=True),
        fromfile='/dev/null', tofile='b/' + p.relative_to(ROOT).as_posix())) for p in source)
    sealed(OUT / 'code.diff', diff)
    archive_inputs = source + [OUT / 'REPORT.md', OUT / 'code.diff']
    archive = OUT / 'implementation.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in archive_inputs:
                info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), (2026, 10, 21, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sorted(z.namelist()) == sorted(p.relative_to(ROOT).as_posix() for p in archive_inputs)
        for p in archive_inputs:
            assert z.read(p.relative_to(ROOT).as_posix()) == p.read_bytes()
    deps = [ROOT / 'r2/data_tools/bench_user_v2' / name for name in ('build.py', 'render.py', 'validate.py')]
    delivery = [OUT / n for n in (
        'REPORT.md', 'index.html', 'DATASET_READY.json', 'manifest.json',
        'new_validation_manifest.json', 'final_summary.json', 'visual_review.json',
        'accepted_conditions.csv', 'held_conditions.csv', 'parents_admission.csv',
        'sampling_policy.json', 'code.diff', 'implementation.zip')]
    json_sealed(OUT / 'source_hashes.json', {
        'schema': 'BENCH_V2_EXPANSION_DELIVERY_HASHES_V1',
        'implementation': {p.relative_to(ROOT).as_posix(): sha(p) for p in source},
        'reused_data_tools': {p.relative_to(ROOT).as_posix(): sha(p) for p in deps},
        'frozen_protected_dependencies': manifest['protected_source_hashes'],
        'delivery': {p.relative_to(ROOT).as_posix(): sha(p) for p in delivery},
        'code_diff_scope': 'New isolated implementation only; old production/scoring source unchanged.',
    })
    result = {
        'status': 'PASS', 'scientific_status': summary['status'],
        'verification_stdout': checked.stdout.strip(),
        'gallery_local_links_checked': len(links), 'parents': 25, 'conditions': 81,
        'original_DEV_conditions': 39, 'protected_source_files': len(manifest['protected_source_hashes']),
        'new_CPU_OT_attempts': 63, 'reused_CPU_receipts': 18,
        'torch_imported_by_validation': False,
        'new_model_calls': 0, 'new_training_updates': 0,
        'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
        'source_hashes_sha256': sha(OUT / 'source_hashes.json'),
        'scale_goal': [47, 67], 'scale_goal_status': summary['scale_goal_status'],
    }
    json_sealed(OUT / 'closeout_verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
