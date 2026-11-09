"""Read-only post-deletion checks; writes receipts only in this maintenance folder."""
import hashlib
import json
import os
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.perf_counter()
    plan = read('delete_plan.json')
    before = read('protected_evidence_before.json')
    cleanup = read('cleanup_result.json')
    assert cleanup['status'] == 'DONE'
    assert cleanup['deleted_files'] == len(plan['files']) == 8061
    assert cleanup['deleted_bytes'] == plan['bytes'] == 15677075950
    assert all(not (ROOT / r['relative_path']).exists() for r in plan['files'])
    assert all(not Path(p).exists() for p in plan['source_directories'])
    log = [json.loads(line) for line in (OUT / 'deletions.jsonl').read_text().splitlines()]
    assert len(log) == len(plan['files'])
    assert {r['path'] for r in log} == {r['path'] for r in plan['files']}
    assert all(r['status'] == 'DONE' for r in log)
    differences = []
    for i, (rel, expected) in enumerate(before.items(), 1):
        path = ROOT / rel
        if not path.is_file():
            differences.append({'path': rel, 'error': 'MISSING'})
        elif path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            differences.append({'path': rel, 'error': 'CHANGED'})
        if i % 2000 == 0:
            print('PROTECTED_CHECKED', i, flush=True)
    archive_sha = sha(OUT / 'retired_activity_source.zip')
    assert archive_sha == plan['source_archive_sha256']
    original = ROOT / 'native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt'
    canonical = ROOT / 'native_t1/checkpoints/chair_hybrid_start_total5000.pt'
    assert os.path.samefile(original, canonical)
    run = ROOT / 'native_t1/runs/chair_moment_channel_v2'
    counts = {
        'formal_v2_checkpoint_pt': len(list((run / 'checkpoints').glob('*.pt'))),
        'formal_v2_checkpoint_sidecars': len(list((run / 'checkpoints').glob('*.json'))),
        'derived_ot_npz': len(list((run / 'ot_cache').rglob('*.npz'))),
        'ot_attempt_receipts': len(list((run / 'ot_cache').rglob('*.attempt.json'))),
    }
    assert counts == {'formal_v2_checkpoint_pt': 0, 'formal_v2_checkpoint_sidecars': 10,
                      'derived_ot_npz': 0, 'ot_attempt_receipts': 8000}, counts
    result = {
        'status': 'PASS' if not differences else 'FAIL',
        'protected_files_checked': len(before), 'differences': differences,
        'deleted_files_confirmed_absent': len(log), 'counts_after': counts,
        'source_archive_sha256': archive_sha, 'start_samefile': True,
        'start_sha256': before[canonical.relative_to(ROOT).as_posix()]['sha256'],
        'real_model_forwards': 0, 'training_updates': 0, 'generations': 0,
        'seconds': time.perf_counter() - started,
    }
    (OUT / 'protection_verification_after.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    assert not differences


if __name__ == '__main__':
    main()
