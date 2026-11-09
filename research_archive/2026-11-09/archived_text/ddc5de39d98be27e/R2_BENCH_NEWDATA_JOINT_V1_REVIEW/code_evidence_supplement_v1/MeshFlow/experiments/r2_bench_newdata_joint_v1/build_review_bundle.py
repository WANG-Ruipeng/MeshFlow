"""Build the <=10 MB local review bundle from saved evidence, never model work.

Use plan while REPORT/METRIC_TABLES are being written; build only after their
author explicitly confirms completion. Original evidence is never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ARCHIVE = OUT / 'R2_BENCH_NEWDATA_JOINT_V1_REVIEW_MINI.zip'
RECEIPT = OUT / 'review_bundle_receipt.json'
SUPPLEMENT = OUT / 'code_evidence_supplement_v1'
PREFIX = 'R2_BENCH_NEWDATA_JOINT_V1_REVIEW'
LIMIT = 10_000_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def files():
    required = [OUT / 'REPORT.md', OUT / 'results/METRIC_TABLES.md',
                OUT / 'final_audit.json', OUT / 'results/key_metrics_independent_audit.json',
                OUT / 'results/key_metrics_summary.json', OUT / 'code_evidence/source_hashes.json',
                OUT / 'results/loss_curves.png']
    selected = set()
    # Core task/report/evidence metadata only; no asset array or mesh files.
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.json', '.md') and p.name != RECEIPT.name:
            selected.add(p)
    for p in (OUT / 'results').rglob('*'):
        if p.is_file() and p.suffix in ('.csv', '.json', '.md'):
            selected.add(p)
    selected.add(OUT / 'results/loss_curves.png')
    for p in (OUT / 'visual_review').rglob('*'):
        if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.py'):
            selected.add(p)
    for p in (OUT / 'code_evidence').rglob('*'):
        if p.is_file():
            if p.suffix not in ('.py', '.json', '.diff', '.md'):
                raise ValueError('Unexpected large/non-source code-evidence asset: ' + str(p))
            selected.add(p)
    # Small data manifests and frozen exposure schedule are useful to recompute
    # allocation. No NPZ/raw/weight data are packaged.
    for p in (OUT / 'data').glob('*.json'):
        selected.add(p)
    training = OUT / 'training/NEW_DATA_JOINT'
    selected.update(training.glob('*.json'))
    for name in ('training_log.jsonl', 'calls.jsonl', 'exposure.jsonl',
                 'checkpoint_history.jsonl', 'resource_checks.jsonl'):
        selected.add(training / name)
    for p in (OUT / 'engineering').rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.jsonl'):
            selected.add(p)
    # Stage receipts and original failed attempt, not all generated geometry.
    selected.add(OUT / 'evaluation/score_registration.json')
    selected.update((OUT / 'evaluation/inference_registrations').glob('*.json'))
    for label in ('RELATION_V1_S1500', 'NEWDATA_JOINT_S1500'):
        base = OUT / 'evaluation' / label
        selected.update(base.glob('*_complete_full.json'))
        selected.update(base.glob('sampling_attempt_*/failure.json'))
    selected.update((OUT / 'evaluation/completions').glob('*.json'))
    selected.update((OUT / 'gallery').glob('*.json'))
    selected.update((OUT / 'dispatch').glob('*.json'))
    selected.update((OUT / 'evaluation/dispatch').glob('*.json'))
    rid = '27eb79ddf26fd7ac7898580cb9d16864__R1__THREE_FAMILY_V1__n0'
    for stage in ('RAW', 'CLEAN_FINAL_POST'):
        p = OUT / 'gallery' / (stage + '__' + rid + '.png')
        selected.add(p)
        required.append(p)
    for p in required:
        selected.add(p)
    missing = [str(p.relative_to(OUT)) for p in sorted(selected) if not p.is_file()]
    return sorted(selected), missing


def extra_sources():
    sources = [ROOT / 'r2/pilots/bench_newdata_joint_v1/record_root_review.py',
               OUT / 'audit_key_summary.py', OUT / 'build_metric_tables.py',
               Path(__file__).resolve()]
    sources += [p for p in (OUT / 'finalize.py', ROOT / 'r2/pilots/bench_newdata_joint_v1/finalize.py') if p.is_file()]
    return sources


def check_completed():
    assert read(OUT / 'final_audit.json')['status'] == 'PASS'
    assert read(OUT / 'results/key_metrics_independent_audit.json')['status'] == 'PASS'
    summaries = sorted((OUT / 'visual_review/aggregates').glob('*/summary.json'))
    if not summaries:
        raise RuntimeError('Actual visual-review aggregate missing')
    for path in summaries:
        d = read(path)
        if d['status'] == 'COMPLETE_ACTUAL_AI_PROXY_REVIEW':
            assert (d['reviewed_cells'], d['reviewed_pages'], d['missing_cells']) == (168, 84, 0)
            csvpath = path.parent / 'validated_per_cell.csv'
            rows = list(csv.DictReader(csvpath.open(encoding='utf-8-sig', newline='')))
            assert len(rows) == 168
            return path
    raise RuntimeError('No complete actual 168-cell visual-review aggregate')


def supplement():
    sources = extra_sources()
    records = []
    for p in sources:
        payload = p.read_bytes()
        rel = Path('MeshFlow') / p.relative_to(ROOT)
        records.append(dict(source=str(p), archive_path=rel.as_posix(), sha256=sha(payload), bytes=len(payload)))
    manifest = dict(schema='BENCH_NEWDATA_CODE_SUPPLEMENT_V1',
        original_code_archive_unchanged=True,
        original_manifest_sha256=sha((OUT / 'code_evidence/source_hashes.json').read_bytes()),
        description='Actual visual-label serializer, independent aggregate auditor, metric-table builder, finalizer when present, and review-bundle builder added after the original source snapshot.',
        model_calls=0, source_files=records)
    if SUPPLEMENT.exists():
        assert read(SUPPLEMENT / 'source_hashes.json') == manifest, 'Existing supplement differs; preserve it and use another version'
        for row in records:
            assert sha((SUPPLEMENT / row['archive_path']).read_bytes()) == row['sha256']
    else:
        SUPPLEMENT.mkdir()
        for row in records:
            dest = SUPPLEMENT / row['archive_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(row['source']).read_bytes())
        (SUPPLEMENT / 'source_hashes.json').write_bytes(jbytes(manifest))
    return [p for p in sorted(SUPPLEMENT.rglob('*')) if p.is_file()]


def plan():
    selected, missing = files()
    return dict(status='WAIT_AUTHOR_FINALIZATION', missing=missing,
        selected_files_now=len(selected)-len(missing),
        uncompressed_bytes_now=sum(p.stat().st_size for p in selected if p.is_file()),
        supplementary_scripts=[str(p.relative_to(ROOT)) for p in extra_sources()],
        compressed_limit_bytes=LIMIT, archive=str(ARCHIVE), model_calls=0,
        note='No files written by plan; build waits for root confirmation that REPORT.md and METRIC_TABLES.md are final.')


def build():
    if ARCHIVE.exists() or RECEIPT.exists():
        raise RuntimeError('Final bundle exists; use verify, never overwrite')
    selected, missing = files()
    if missing:
        raise RuntimeError('Required evidence missing: ' + json.dumps(missing))
    actual_review = check_completed()
    selected += supplement()
    # Snapshot once. Original files are never modified, moved or removed.
    payloads = {p.relative_to(OUT).as_posix(): p.read_bytes() for p in selected}
    if any(Path(name).suffix in ('.pt', '.pth', '.npz', '.npy', '.ply', '.obj') for name in payloads):
        raise RuntimeError('Forbidden weight/raw asset selected')
    pngs = sorted(name for name in payloads if name.endswith('.png'))
    assert len(pngs) == 3, pngs  # one existing loss curve plus two existing mesh pages
    note = '''# Bench new-data joint 1500: compact review evidence

Start with REPORT.md, then results/METRIC_TABLES.md and METHODS_NOTE.md.
results/per_output.csv contains all 168 actual score rows; 24 rows reuse the
unchanged old baseline DEV scores, and 144 were newly scored. Complete paired,
per-condition, per-parent and head tables are included. There are 177 numeric
fields under the same original collector definition, with statuses and NA
reasons retained separately. No metrics were selectively dropped.

visual_review/ retains all 168 actual, nonblind AI_PROXY cell observations,
84 paired-page evidence identities, five independent rating axes, uncertain
labels, and complete aggregate/transition tables. These are not independent
human ratings. The numeric and image-review evidence remain distinct.

The only included mesh PNGs are new parent 27 R1 / noise 0 at RAW and FINAL.
They are explanatory examples, not representative success-rate evidence. The
full 84-page fixed-view/cut-section gallery remains in the local experiment
gallery/ directory, with its complete manifest and image receipts included.
The third PNG is the already-generated training loss curve. No images were
generated by this packager. Model weights, RAW/POST geometry arrays and input
NPZ meshes remain local and are identified by the saved hashes and receipts.

Training logs, the frozen schedule and full exposure/model-call ledgers are
included for recomputation. final_audit.json certifies recorded execution and
file identity only, not model superiority. The original failed sampler did
zero neural forwards and incurred one extra generator load; its exception and
recovery evidence are retained. No training or generation was retried here.

code_evidence/ is the unchanged original 219-source snapshot, including actual
scoring/geometry code, code.diff and hashes. code_evidence_supplement_v1/ adds
later review/audit/bundle helpers without replacing that snapshot.

All original local assets are preserved. This ZIP is an evidence package, not
a standalone model installation. Paths inside original receipts intentionally
refer to the actual Windows/WSL workspace; BUNDLE_MANIFEST.json authenticates
the included relative paths and distinguishes omitted large local assets.
'''
    payloads['BUNDLE_README.md'] = note.encode('utf-8')
    manifest = dict(schema='BENCH_NEWDATA_MINI_REVIEW_V1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        experiment=str(OUT), review_aggregate=str(actual_review.relative_to(OUT)),
        bytes_limit=LIMIT, included_pngs=pngs, full_gallery_local=str(OUT / 'gallery'),
        omitted='Weights, optimizers, RAW/POST mesh arrays, input NPZs, 82 other mesh-gallery PNGs; originals retained locally',
        model_calls=0, scoring_calls=0, rendering_calls=0, original_file_mutations=0,
        files={name: dict(sha256=sha(data), bytes=len(data)) for name,data in sorted(payloads.items())})
    payloads['BUNDLE_MANIFEST.json'] = jbytes(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(PREFIX + '/' + name, date_time=(2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    blob = buffer.getvalue()
    if len(blob) >= LIMIT:
        raise RuntimeError('Compression exceeds fixed 10,000,000-byte ceiling; no ZIP written: ' + str(len(blob)))
    for p in selected:
        if p.read_bytes() != payloads[p.relative_to(OUT).as_posix()]:
            raise RuntimeError('Evidence changed during packaging: ' + str(p))
    with ARCHIVE.open('xb') as f:
        f.write(blob)
    receipt = dict(status='COMPLETE', archive=str(ARCHIVE), archive_bytes=len(blob),
        archive_sha256=sha(blob), limit_bytes=LIMIT, included_files=len(payloads),
        manifest_sha256=sha(payloads['BUNDLE_MANIFEST.json']),
        source_manifest_sha256=sha((OUT / 'code_evidence/source_hashes.json').read_bytes()),
        model_calls=0, scoring_calls=0, rendering_calls=0, original_file_mutations=0)
    with RECEIPT.open('xb') as f:
        f.write(jbytes(receipt))
    return verify()


def verify():
    receipt = read(RECEIPT)
    blob = ARCHIVE.read_bytes()
    assert len(blob) == receipt['archive_bytes'] < LIMIT
    assert sha(blob) == receipt['archive_sha256']
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert z.testzip() is None
        mb = z.read(PREFIX + '/BUNDLE_MANIFEST.json')
        assert sha(mb) == receipt['manifest_sha256']
        manifest = json.loads(mb)
        assert len(z.namelist()) == len(manifest['files']) + 1 == receipt['included_files']
        for name, expected in manifest['files'].items():
            data = z.read(PREFIX + '/' + name)
            assert sha(data) == expected['sha256'] and len(data) == expected['bytes'], name
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plan', 'build', 'verify'))
    args = parser.parse_args()
    print(json.dumps({'plan': plan, 'build': build, 'verify': verify}[args.mode](), indent=2))
