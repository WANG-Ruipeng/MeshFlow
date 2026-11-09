"""Prepare or seal the four-route evidence package; stdlib only.

plan is read-only and permits unfinished reports. seal requires the root's exact
final REPORT SHA and every branch's actual visual closeout. No models, metrics,
rendering or scientific files are changed. Full training call/exposure logs are
indexed with hashes, while their execution closeouts and final CSVs are included.
"""
from __future__ import annotations
import argparse
import collections
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/r2_bench_four_routes_v1'
CODE = ROOT / 'r2/pilots/bench_four_routes_v1'
BRANCHES = ('hybrid', 'exposure', 'bestof', 'seam')
LIMIT = 10_000_000
SMALL_JSON = 2_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT.resolve()):
        return path.relative_to(ROOT.resolve()).as_posix()
    scorer_root = ROOT.parent / 'MeshFlow-Control'
    if path.is_relative_to(scorer_root.resolve()) and path.suffix == '.py':
        return '_external_sources/MeshFlow-Control/' + path.relative_to(scorer_root.resolve()).as_posix()
    raise ValueError('Source outside the workspace and registered scorer root: ' + str(path))


def resolve(value):
    value = str(value).replace('\\', '/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    for prefix in ('E:/MeshFlow-Control/', '/mnt/e/MeshFlow-Control/'):
        if value.startswith(prefix):
            return ROOT.parent / 'MeshFlow-Control' / value[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def frozen_code_claims():
    """Never replace a registered hash with a current hash before validation."""
    registrations = sorted(set(OUT.glob('*registration*.json')) | {
        p for branch in BRANCHES for directory in (OUT / branch, OUT / branch / 'data', OUT / branch / 'evaluation')
        for p in directory.glob('*registration*.json')})
    claims = {}

    def visit(value, registration):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and re.fullmatch(r'[0-9a-f]{64}', item) and str(key).endswith('.py'):
                    path = resolve(key).resolve()
                    relative(path)  # Only workspace or exact registered scorer sources.
                    row = claims.setdefault(path, {'sha256': item, 'registrations': []})
                    if row['sha256'] != item:
                        raise ValueError('Contradictory frozen code claims: ' + str(path))
                    row['registrations'].append(relative(registration))
                else:
                    visit(item, registration)
        elif isinstance(value, list):
            for item in value:
                visit(item, registration)

    for registration in registrations:
        visit(read(registration), registration)
    if not claims:
        raise ValueError('No original frozen source claims found')
    for path, row in claims.items():
        if sha(path) != row['sha256']:
            raise ValueError('Frozen source changed: ' + str(path))
    return claims, registrations


def select_evidence():
    """All final per-output/parent/step CSVs; internal query scans stay local."""
    chosen = set(p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.md', '.json'))
    for audit_directory in (OUT / 'audits/baseline_consistency/final_v1', OUT / 'audits/base46_readonly_v1'):
        if audit_directory.exists():
            chosen.update(p for p in audit_directory.rglob('*')
                          if p.is_file() and p.suffix.lower() in ('.csv', '.json', '.md', '.txt'))
    for name in ('four_routes_execution_ledger_final.json', 'backend_runtime_final_report_review.json',
                 'owned_cpu_process_exit_check.json'):
        audit = OUT / 'audits' / name
        if audit.exists():
            chosen.add(audit)
    indexed = {}
    evidence_dirs = {'results', 'visual_review', 'audits', 'closeout', 'training_closeout',
                     'execution_closeout', 'coefficient_audit', 'data', 'selection',
                     'training', 'evaluation', 'workflow', 'dispatch', 'layout_diagnostics',
                     'compact_deletion_v2', 'completions'}
    internal_tables = {'selection_per_face.csv', 'mixed_boundary_clusters.csv',
                       'original_edge_fates.csv', 'original_illegal_pair_fates.csv',
                       'deletion_fate_per_slot.csv'}
    for branch in BRANCHES:
        folder = OUT / branch
        for name in ('complete.json', 'started.json', 'failure.json'):
            chosen.update((folder / 'outputs').glob('*/' + name))
        chosen.update((folder / 'cpu').glob('*/*/complete.json'))
        candidates = [p for p in folder.iterdir() if p.is_file()]
        for name in evidence_dirs:
            directory = folder / name
            if not directory.exists():
                continue
            for current, dirs, names in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in ('scores', '__pycache__', 'raw', 'post', 'inputs', 'layouts')]
                candidates.extend(Path(current) / n for n in names)
        gallery = folder / 'gallery'
        if gallery.exists():
            candidates.extend(p for p in gallery.iterdir() if p.is_file() and not p.name.startswith(('RAW__', 'BASE_POST__', 'CLEAN_FINAL_POST__')))
        for path in candidates:
            parts = path.relative_to(folder).parts
            if any(part in ('archive', 'review_package', '__pycache__') for part in parts):
                continue
            suffix = path.suffix.lower()
            if suffix == '.log' and 'preserved_worker_recovery_v1' in parts:
                chosen.add(path)
                continue
            if suffix == '.png' and parts[0] in ('closeout', 'training_closeout') and 'loss' in path.name:
                chosen.add(path)
                continue
            if path.name in internal_tables:
                indexed[path] = ('Original deletion CSV fully encoded in bestof/compact_deletion_v2; exact byte reconstruction verified'
                                 if path.name == 'deletion_fate_per_slot.csv' else
                                 'Detailed internal face/cluster/pair scan; final per-output/parent summaries and exact hash included')
                continue
            if suffix == '.csv':
                chosen.add(path)
                continue
            if suffix not in ('.md', '.json', '.jsonl', '.txt'):
                continue
            if 'gallery' in parts:
                # Preserve definitions and render receipt, not 400 near-identical
                # image receipts or HTML pointing at deliberately excluded PNGs.
                if path.name in ('manifest.json', 'raw_manifest.json', 'post_manifest.json',
                                 'render_complete.json', 'raw_render_complete.json',
                                 'post_render_complete.json', 'definition.json'):
                    chosen.add(path)
                continue
            if path.name in ('calls.jsonl', 'exposure.jsonl') and 'training' in parts:
                indexed[path] = 'Full training event log; exact hash/index plus execution closeout and all CSV summaries included'
                continue
            if suffix == '.json' and path.stat().st_size > SMALL_JSON:
                indexed[path] = 'Large schedule/duplicate JSON; exact hash/index and all CSV summaries included'
                continue
            if suffix == '.jsonl' and path.name != 'training_log.jsonl' and path.stat().st_size > SMALL_JSON:
                indexed[path] = 'Large event log; exact hash/index and phase closeout included'
                continue
            if parts[0] in evidence_dirs or len(parts) == 1:
                chosen.add(path)
    return chosen, indexed


def check_ready(expected_report_sha):
    from .normalize_shared_tables import verify as verify_compact_deletion
    if verify_compact_deletion()['status'] != 'PASS':
        raise ValueError('Lossless deletion-ledger delivery is not verified')
    report = OUT / 'REPORT.md'
    if not report.exists() or sha(report) != expected_report_sha:
        raise ValueError('Final root REPORT missing or different from explicitly supplied SHA')
    for name in ('PROTOCOL.md', 'METHODS.md', 'RUNBOOK.md', 'WEIGHTS.md'):
        if not (OUT / name).is_file():
            raise ValueError('Missing root document: ' + name)
    for path, report_sha in (
            (OUT / 'audits/backend_runtime_final_report_review.json', expected_report_sha),
            (OUT / 'hybrid/audits/backend_runtime_report_review.json', sha(OUT / 'hybrid/REPORT.md'))):
        review = read(path)
        if review.get('status') != 'PASS' or review.get('report_sha256') != report_sha:
            raise ValueError('Final report independent review missing or for different bytes: ' + str(path))
        for source, digest in review['inputs'].items():
            if sha(resolve(source)) != digest:
                raise ValueError('Reviewed report evidence changed: ' + source)
    process_check = read(OUT / 'audits/owned_cpu_process_exit_check.json')
    if process_check.get('status') != 'KNOWN_CPU_PROCESSES_ABSENT':
        raise ValueError('Owned CPU process exit receipt is incomplete')
    ledger = read(OUT / 'audits/four_routes_execution_ledger_final.json')
    if (ledger.get('status') != 'FINAL' or not ledger.get('final_checks')
            or not all(value is True for value in ledger['final_checks'].values())):
        raise ValueError('Final four-route execution accounting is incomplete')
    consistency = read(OUT / 'audits/baseline_consistency/final_v1/complete.json')
    if (consistency.get('status') != 'COMPLETE'
            or not all(consistency.get(key) is True for key in (
                'objective_assets_and_metrics_pass', 'hybrid_collected_table_available',
                'hybrid_collected_reference_matches_registered_rows'))):
        raise ValueError('Final baseline consistency audit is incomplete')
    hybrid_results = read(OUT / 'hybrid/results/complete.json')
    if (any(hybrid_results.get(key) != value for key, value in {
            'status': 'COMPLETE', 'rows': 230, 'new_scores': 92, 'reference_rows': 138}.items())
            or len(hybrid_results.get('metric_fields', [])) != 177
            or not hybrid_results.get('files')):
        raise ValueError('Final HYBRID result tables are incomplete')
    for path, digest in hybrid_results['files'].items():
        if sha(resolve(path)) != digest:
            raise ValueError('Final HYBRID table changed: ' + path)
    chosen, _ = select_evidence()
    required = [OUT / 'audits/four_routes_execution_ledger_final.json',
                OUT / 'audits/baseline_consistency/final_v1/complete.json',
                OUT / 'audits/backend_runtime_final_report_review.json',
                OUT / 'audits/owned_cpu_process_exit_check.json',
                OUT / 'hybrid/audits/backend_runtime_report_review.json']
    required.extend(OUT / 'audits/base46_readonly_v1' / name for name in (
        'receipt.json', 'summary.json', 'source_identities.json', 'per_request.csv',
        'per_parent.csv', 'group_parent_macro.csv', 'cf_counts_per_request.csv',
        'cf_counts_per_parent.csv', 'cf_counts_group_summary.csv',
        'cf_counts_support.json', 'cf_counts_receipt.json'))
    required.extend(OUT / 'hybrid/results' / name for name in (
        'complete.json', 'per_output.csv', 'per_condition.csv', 'per_parent.csv',
        'parent_macro.csv', 'paired_per_output.csv', 'paired_per_parent.csv',
        'core_parent_macro.csv', 'selection_per_output.csv',
        'area_deletion_per_output.csv', 'connectivity_per_output.csv', 'phase_runtime.csv'))
    required.append(OUT / 'hybrid/dispatch/preserved_worker_recovery_v1/complete.json')
    if any(not path.is_file() or path not in chosen for path in required):
        raise ValueError('Final tables, BASE46 supplement, or recovery evidence missing from package selection')
    for branch in BRANCHES:
        if not (OUT / branch / 'REPORT.md').is_file():
            raise ValueError('Branch report incomplete: ' + branch)
        review = OUT / branch / 'visual_review'
        completions = [review / 'summary.json', review / 'summary_v1/completion.json']
        completions += list(review.rglob('*complete*.json')) + list(review.rglob('completion.json'))
        completions += list((OUT / branch / 'results').rglob('*visual*complete*.json'))
        if not any(p.is_file() and read(p).get('status') == 'COMPLETE' for p in completions):
            # Different branches use their own delivery completion layout.
            raise ValueError('No completed visual aggregate receipt: ' + branch)
        assignments = review / 'assignments.json'
        if assignments.exists():
            assignment_payload = read(assignments)
            assigned = assignment_payload.get('assignments', assignment_payload)
            for owner in assigned:
                shard = review / 'shards' / (owner + '.json')
                payload = read(shard)
                if branch == 'hybrid':
                    records = payload.get('records', [])
                    expected_pages = assigned[owner]
                    if (payload.get('status') != 'COMPLETE_ACTUAL_VIEW'
                            or payload.get('pages') != len(expected_pages)
                            or {r['page_id'] for r in records} != set(expected_pages)
                            or len(records) != 3 * len(expected_pages)
                            or not all(r.get('reviewer_kind') == 'AI_PROXY' and r.get('blind') is False
                                       and r.get('actually_viewed') is True
                                       and r.get('views_and_sections_checked') is True for r in records)):
                        raise ValueError('Hybrid actual review coverage/identity differs: ' + str(shard))
                    continue
                if payload['status'] != 'COMPLETE_ASSIGNED_SHARD':
                    raise ValueError('Actual review shard incomplete: ' + str(shard))
                if payload.get('reviewer_kind') != 'AI_PROXY' or payload.get('blind') is not False:
                    raise ValueError('Visual evidence identity differs: ' + str(shard))
                expected_pages = assigned[owner].get('pages', len(assigned[owner].get('page_ids', [])))
                if payload.get('actual_pages') != expected_pages:
                    raise ValueError('Visual coverage differs: ' + str(shard))


def source_diff(paths):
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    head = result.stdout.strip()
    diff = ['# Actual packaged source vs repository HEAD ' + head + '\n',
            '# Untracked files are shown as additions; this is a provenance diff,\n',
            '# not a causal method comparison or a full-workspace patch.\n']
    for path in sorted(paths):
        rel = relative(path)
        previous = None if rel.startswith('_external_sources/') else subprocess.run(
            ['git', 'show', head + ':' + rel], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        present = previous is not None and previous.returncode == 0
        before = previous.stdout.decode('utf-8', errors='replace').splitlines(keepends=True) if present else []
        after = path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        diff.extend(difflib.unified_diff(before, after, fromfile=('a/' + rel if present else '/dev/null'),
                                       tofile='b/' + rel))
    return ''.join(diff).encode('utf-8'), head


def plan():
    files, indexed = select_evidence()
    missing = [str(p.relative_to(OUT)) for p in [OUT / 'REPORT.md'] + [OUT / b / 'REPORT.md' for b in BRANCHES]
               if not p.exists()]
    return dict(status='READ_ONLY_PLAN',reports_pending=missing,selected_text_CSV_files=len(files),
                selected_bytes=sum(p.stat().st_size for p in files),
                counts_by_suffix=dict(collections.Counter(p.suffix for p in files)),
                indexed_large_file_count=len(indexed),
                indexed_large_bytes=sum(p.stat().st_size for p in indexed),
                weights_arrays_meshes_gallery_PNG_included=False,archive_written=False)


def assemble(compression='deflate'):
    """Build and verify in memory; never writes or implies branch completion."""
    from .normalize_shared_tables import verify as verify_compact_deletion
    verify_compact_deletion()
    claims, registrations = frozen_code_claims()
    sources = set(claims) | set(CODE.rglob('*.py'))
    sources.update((OUT / 'audits/base46_readonly_v1').glob('*.py'))
    files, indexed = select_evidence()
    files.update(registrations)
    source_book = {relative(p): {'actual_sha256': sha(p),
                                'actual_source_path': str(p),
                                'frozen_sha256': claims[p]['sha256'] if p in claims else None,
                                'registration_paths': claims[p]['registrations'] if p in claims else [],
                                'scope': 'REGISTERED_SOURCE' if p in claims else 'NEW_HELPER_ACTUAL_SNAPSHOT'}
                   for p in sorted(sources)}
    source_blobs = {}
    for path in sorted(sources):
        identity = source_book[relative(path)]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != identity['actual_sha256']:
            raise ValueError('Source changed before content deduplication')
        blob = 'review_package/source_blobs/' + identity['actual_sha256'] + '.py'
        identity['archive_blob'] = 'experiments/r2_bench_four_routes_v1/' + blob
        if blob in source_blobs and source_blobs[blob] != content:
            raise ValueError('Source SHA collision')
        source_blobs[blob] = content
    diff, head = source_diff(sources)
    file_paths = {relative(p): p for p in files}
    if len(file_paths) != len(files):
        raise ValueError('Archive source path collision')
    file_book = {relative(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)}
    index = {relative(p): {'bytes': p.stat().st_size, 'sha256': sha(p), 'reason': why} for p, why in sorted(indexed.items())}
    note = ('Four-route evidence package. Reports and full per-output/per-parent CSVs retain all scientific statuses.\n'
            + ('ZIP compression: standard DEFLATE level9.\n' if compression == 'deflate' else
               'ZIP compression: ZIP_LZMA. Extract with Python standard library: python -m zipfile -e FOUR_ROUTES_REVIEW_SMALL.zip extracted\n') +
            'No model weights, meshes, RAW/noise arrays or full gallery PNGs are included.\n'
            'Full training calls/exposure event logs and large schedules remain local; indexed_logs.json identifies exact bytes.\n'
            'Execution closeouts, all final per-output/parent/step CSVs, training loss logs/plots and actual visual records are included.\n'
            'Large internal face/pair scans remain local with hashes; final full-field output and parent tables are retained.\n'
            'The three deletion_fate_per_slot ledgers are fully retained in compact_deletion_v2 as reversible CSVs.\n'
            'normalize_shared_tables.py reconstructs original field strings, row order, and exact CSV bytes; verification is included.\n'
            'Registered external scorer sources are copied under _external_sources, with original paths in source_hashes.json.\n'
            'Source bytes are stored once per SHA in source_blobs; every original path is mapped in source_hashes.json.\n'
            'From the extracted workspace, restore exact source paths using:\n'
            '  python experiments/r2_bench_four_routes_v1/review_package/restore_sources.py --package-root .\n'
            'Then reconstruct each original deletion CSV to a new path, for example:\n'
            '  python -B -m r2.pilots.bench_four_routes_v1.bestof.normalize_shared_tables reconstruct --source seam --output reconstructed/seam_deletion.csv\n'
            'Use --source exposure or bestof for the other original ledgers. No models or scores are run.\n'
            'code.diff is against repository HEAD, with untracked source as additions; it is not a method ablation diff.\n'
            'WEIGHTS.md is an index only. Source/report hashes are preserved; no experiments were run by this packager.\n')
    virtual = {'review_package/source_hashes.json': serialized(source_book),
               'review_package/included_files.json': serialized(file_book),
               'review_package/indexed_logs.json': serialized(index),
               'review_package/code.diff': diff,
               'review_package/README.txt': note.encode('utf-8'),
               'review_package/restore_sources.py': (CODE / 'bestof/restore_sources.py').read_bytes(),
               **source_blobs}
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compression == 'deflate' else zipfile.ZIP_LZMA
    with zipfile.ZipFile(buffer, 'w', method, compresslevel=9 if compression == 'deflate' else None) as archive:
        for path in sorted(files):
            archive.write(path, relative(path))
        for name, value in virtual.items():
            archive.writestr('experiments/r2_bench_four_routes_v1/' + name, value)
    payload = buffer.getvalue()
    # Detect a concurrent report/source change before publication.
    for name, identity in file_book.items():
        if sha(file_paths[name]) != identity['sha256']:
            raise ValueError('Input changed while packaging: ' + name)
    for path in sources:
        if sha(path) != source_book[relative(path)]['actual_sha256']:
            raise ValueError('Source changed while packaging: ' + str(path))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
    metadata = dict(bytes=len(payload), limit_bytes=LIMIT,
                    compression='ZIP_DEFLATED_LEVEL9' if compression == 'deflate' else 'ZIP_LZMA',
                    included_source_files=len(sources), included_real_files=len(files), virtual_files=len(virtual),
                    unique_source_blobs=len(source_blobs), source_paths_restorable=True,
                    indexed_large_logs=len(index), git_HEAD=head,
                    frozen_source_claims_verified_before_current_snapshot=len(claims),
                    models_loaded=0, scoring_calls=0, render_calls=0,
                    weights_arrays_meshes_gallery_PNG_included=False, helper_sha256=sha(__file__))
    return payload, virtual, metadata


def preflight(compare_compression=False):
    payload, _, metadata = assemble()
    comparison = {}
    if compare_compression:
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(buffer, 'w', zipfile.ZIP_LZMA) as alternate:
            for item in original.infolist():
                alternate.writestr(item.filename, original.read(item.filename))
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP_LZMA preview CRC validation failed')
        comparison = dict(deflate9_bytes=len(payload), lzma_bytes=len(buffer.getvalue()),
                          lzma_standard_library_roundtrip='PASS',
                          note='Same content except final compression README sentence will identify chosen format')
    return dict(status='READ_ONLY_PREFLIGHT', **metadata, fits_under_10MB=len(payload) < LIMIT,
                archive_written=False, reports_pending=plan()['reports_pending'], compression_comparison=comparison)


def verify_archive(path, expected_report_sha):
    """Read the actual ZIP and reconstruct all three ledgers from its contents."""
    from . import normalize_shared_tables as shared
    from . import normalize_tables as v1
    prefix = 'experiments/r2_bench_four_routes_v1/'
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('Final ZIP duplicate member or CRC failure')
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in names):
            raise ValueError('Unsafe final ZIP member path')
        book = json.loads(archive.read(prefix + 'review_package/included_files.json'))
        sources = json.loads(archive.read(prefix + 'review_package/source_hashes.json'))
        indexed = json.loads(archive.read(prefix + 'review_package/indexed_logs.json'))
        for name, identity in book.items():
            value = archive.read(name)
            if len(value) != identity['bytes'] or v1.sha_bytes(value) != identity['sha256']:
                raise ValueError('Final ZIP evidence hash differs: ' + name)
        for name, identity in sources.items():
            value = archive.read(identity['archive_blob'])
            if v1.sha_bytes(value) != identity['actual_sha256']:
                raise ValueError('Final ZIP source blob differs: ' + name)
            if identity['frozen_sha256'] not in (None, identity['actual_sha256']):
                raise ValueError('Final ZIP frozen source differs: ' + name)
        for helper in (Path(v1.__file__), Path(shared.__file__)):
            if sources[relative(helper)]['actual_sha256'] != sha(helper):
                raise ValueError('Reconstruction helper differs from final packaged source')
        if v1.sha_bytes(archive.read(prefix + 'REPORT.md')) != expected_report_sha:
            raise ValueError('Final ZIP root REPORT differs')
        compact = prefix + 'bestof/compact_deletion_v2/'
        manifest = json.loads(archive.read(compact + 'manifest.json'))
        for name, identity in manifest['files'].items():
            value = archive.read(compact + name)
            if len(value) != identity['bytes'] or v1.sha_bytes(value) != identity['sha256']:
                raise ValueError('Final ZIP compact table differs: ' + name)
        _, values = v1.load_csv(archive.read(compact + 'slot_values.csv'))
        _, references = v1.load_csv(archive.read(compact + 'slot_references.csv'))
        if set(manifest['sources']) != set(v1.SOURCES):
            raise ValueError('Final ZIP compact source identities differ')
        reconstructed = {}
        for name, metadata in manifest['sources'].items():
            _, requests = v1.load_csv(archive.read(compact + metadata['request_index_file']))
            rows = v1.restored_rows(metadata, requests, shared.expanded(name, references, values))
            value = v1.csv_bytes(metadata['original_columns'], rows,
                                 line_ending=metadata['line_ending'], bom=metadata['bom'])
            digest = v1.sha_bytes(value)
            if (digest != metadata['original_sha256'] or len(value) != metadata['original_bytes']
                    or indexed[metadata['original_relative_path']]['sha256'] != digest):
                raise ValueError('Final ZIP reconstructed ledger differs from original: ' + name)
            reconstructed[name] = dict(status='BYTE_IDENTICAL', rows=len(rows),
                                       fields=len(metadata['original_columns']), bytes=len(value),
                                       original_relative_path=metadata['original_relative_path'],
                                       reconstructed_sha256=digest)
    return dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
                archive_sha256=sha(path), archive_bytes=Path(path).stat().st_size,
                CRC='PASS', evidence_files_verified=len(book), source_paths_verified=len(sources),
                root_report_sha256=expected_report_sha, deletion_ledgers=reconstructed,
                reconstruction_inputs='Actual final ZIP compact CSVs and manifest only',
                original_CSV_read_for_reconstruction=False, scientific_calls=0, models_loaded=0)


def seal(expected_report_sha, compression='deflate'):
    check_ready(expected_report_sha)
    dest = OUT / 'FOUR_ROUTES_REVIEW_SMALL.zip'
    receipts = OUT / 'review_package'
    if dest.exists() or receipts.exists():
        raise ValueError('Existing package/receipt retained; no overwrite or automatic reseal')
    payload, virtual, metadata = assemble(compression)
    if len(payload) >= LIMIT:
        raise ValueError('Complete selected evidence exceeds10MB: ' + str(len(payload)) + '; nothing written or silently omitted')
    if sha(OUT / 'REPORT.md') != expected_report_sha:
        raise ValueError('Root REPORT changed after readiness check')
    receipts.mkdir()
    for name, value in virtual.items():
        (OUT / name).parent.mkdir(parents=True, exist_ok=True)
        with (OUT / name).open('xb') as handle:
            handle.write(value)
    with dest.open('xb') as handle:
        handle.write(payload)
    readback = verify_archive(dest, expected_report_sha)
    readback_path = receipts / 'verified_readback.json'
    with readback_path.open('xb') as handle:
        handle.write(serialized(readback))
    result = dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
                  archive_path=relative(dest),archive_sha256=sha(dest),
                  root_report_sha256=expected_report_sha,
                  final_ZIP_readback_path=relative(readback_path),
                  final_ZIP_readback_sha256=sha(readback_path), **metadata)
    with (receipts / 'complete.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'seal'))
    parser.add_argument('--expected-report-sha')
    parser.add_argument('--compression', choices=('deflate', 'lzma'), default='deflate')
    parser.add_argument('--compare-compression', action='store_true')
    args = parser.parse_args()
    if args.action == 'seal' and not re.fullmatch(r'[0-9a-f]{64}', args.expected_report_sha or ''):
        parser.error('seal requires the root final REPORT SHA256')
    result = plan() if args.action == 'plan' else preflight(args.compare_compression) if args.action == 'preflight' else seal(args.expected_report_sha, args.compression)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
