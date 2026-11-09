"""Bounded asset cleanup audit; stdlib only, no model loading or execution."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import os
import shutil
import stat
import sys

ROOT = Path('E:/MeshFlow').resolve()
OUT = Path(__file__).resolve().parent
ASSESSMENT = ROOT / 'native_t1/maintenance/weight_cleanup_review_20261004/cleanup_assessment.json'
TIER = 'A_STALE_RECOVERY'
EXPECTED = {
    'native_t1/runs/chair_domain_control_v1/checkpoints/Full_latest.pt',
    'native_t1/runs/chair_domain_control_v1/checkpoints/Geo_latest.pt',
    'native_t1/runs/chair_domain_control_v1/checkpoints/T1_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/Full_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/Geo_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/T1_latest.pt',
    'native_t1/runs/control_mvp48_budget_single_v1/checkpoints/T1_fit2_probe_latest.pt',
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def walk():
    rows = []
    links = []
    for current, dirs, names in os.walk(ROOT, followlinks=False):
        kept = []
        for name in dirs:
            p = Path(current) / name
            if p.is_symlink() or p.is_junction():
                links.append(str(p.relative_to(ROOT)))
            elif p != OUT:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            p = Path(current) / name
            s = p.stat(follow_symlinks=False)
            rows.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=s.st_size,
                             mtime_ns=s.st_mtime_ns, mode=s.st_mode))
    return rows, links


def prepare():
    if (OUT / 'prepared.json').exists() or (OUT / 'deletion_receipts.jsonl').exists():
        raise RuntimeError('Existing preparation or deletion ledger; do not overwrite evidence')
    assessment = read(ASSESSMENT)
    candidates = [r for r in assessment['candidates'] if r['tier'] == TIER]
    assert {r['path'] for r in candidates} == EXPECTED and len(candidates) == 7
    assert sum(r['bytes'] for r in candidates) == 10950114685
    for run in ('chair_domain_control_v1', 'control_mvp48_budget_single_v1'):
        state = read(ROOT / 'native_t1/runs' / run / 'run_state.json')
        assert state['status'] in ('DONE', 'COMPLETE')
        assert state['active_stage'] is None and state['worker_pid'] is None
    verified = []
    for r in candidates:
        p = ROOT / r['path']
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        s = p.stat(follow_symlinks=False)
        assert stat.S_ISREG(s.st_mode) and not s.st_file_attributes & 1024
        assert s.st_size == r['bytes'] and sha(p) == r['sha256']
        original = read(ROOT / r['original_manifest'])
        match = next(x for x in original['checkpoints'] if x['path'].endswith('/' + p.name)
                     and x.get('origin') == 'current_run')
        assert match['checkpoint_sha256'] == r['sha256']
        verified.append(dict(r, absolute_path=str(p.resolve())))
    write('deletion_manifest.json', verified)
    rows, links = walk()
    with gzip.open(OUT / 'inventory_before.json.gz', 'wt', encoding='utf8') as f:
        json.dump(rows, f, separators=(',', ':'))
    dependencies = assessment['retained_directory_dependencies']
    protected = []
    for r in rows:
        rel = r['path']
        if rel in EXPECTED or rel.startswith(('.git/', '.worktrees/')):
            continue
        p = ROOT / rel
        relevant = (p.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors')
                    or any(rel == d or rel.startswith(d + '/') for d in dependencies))
        if relevant and not (p.stat(follow_symlinks=False).st_file_attributes & 1024):
            protected.append(dict(r, sha256=sha(p)))
    with gzip.open(OUT / 'protected_before.json.gz', 'wt', encoding='utf8') as f:
        json.dump(protected, f, separators=(',', ':'))
    available = {r['path']: r for r in protected}
    for r in assessment['protected_current_chair_fixed_checkpoints'] + [r for r in assessment['candidates'] if r['tier'] != TIER]:
        assert available[r['path']]['sha256'] == r['sha256']
    write('prepared.json', dict(status='PREPARED', time=datetime.datetime.now().isoformat(),
          assessment_sha256=sha(ASSESSMENT), candidate_count=len(verified),
          candidate_bytes=sum(r['bytes'] for r in verified), inventory_files=len(rows),
          project_logical_bytes=sum(r['bytes'] for r in rows),
          protected_hash_files=len(protected), protected_hash_bytes=sum(r['bytes'] for r in protected),
          disk_free_before=shutil.disk_usage(ROOT).free, skipped_directory_links=links,
          authorization='User approved the recommended first tier (7 old latest checkpoints) after reviewing assessment.',
          model_loads=0, model_forwards=0, training_updates=0, generations=0))
    print(json.dumps(read(OUT / 'prepared.json'), ensure_ascii=False), flush=True)


def verify():
    prepared = read(OUT / 'prepared.json')
    candidates = read(OUT / 'deletion_manifest.json')
    receipts = [json.loads(s) for s in (OUT / 'deletion_receipts.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert len(receipts) == 7 and {r['relative_path'] for r in receipts} == EXPECTED
    for r in candidates:
        assert not os.path.lexists(ROOT / r['path']), r['path']
        receipt = next(x for x in receipts if x['relative_path'] == r['path'])
        assert receipt['sha256'] == r['sha256'] and receipt['bytes'] == r['bytes']
    with gzip.open(OUT / 'inventory_before.json.gz', 'rt', encoding='utf8') as f:
        rows = json.load(f)
    missing = []
    changed = []
    for r in rows:
        if r['path'] in EXPECTED:
            continue
        p = ROOT / r['path']
        if not os.path.lexists(p):
            missing.append(r['path'])
            continue
        s = p.stat(follow_symlinks=False)
        if (s.st_size, s.st_mtime_ns, s.st_mode) != (r['bytes'], r['mtime_ns'], r['mode']):
            changed.append(r['path'])
    with gzip.open(OUT / 'protected_before.json.gz', 'rt', encoding='utf8') as f:
        protected = json.load(f)
    mismatches = [r['path'] for r in protected if not (ROOT / r['path']).is_file() or sha(ROOT / r['path']) != r['sha256']]
    result = dict(status='PASS' if not (missing or changed or mismatches) else 'FAILED',
                  deleted_files=len(receipts), deleted_bytes=sum(r['bytes'] for r in receipts),
                  missing_outside_authorized_list=missing, metadata_changes_outside_list=changed,
                  protected_sha256_mismatches=mismatches, protected_hash_files=len(protected),
                  unchanged_metadata_files=len(rows)-7, disk_free_before=prepared['disk_free_before'],
                  disk_free_after=shutil.disk_usage(ROOT).free,
                  model_loads=0, model_forwards=0, training_updates=0, generations=0,
                  old_manifests_and_sidecars='Retained unchanged as historical evidence; use this separate deletion ledger for retired paths.')
    result['observed_disk_free_increase'] = result['disk_free_after'] - result['disk_free_before']
    write('verification.json', result)
    if result['status'] != 'PASS':
        raise RuntimeError(json.dumps(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'verify': verify}[sys.argv[1]]()
