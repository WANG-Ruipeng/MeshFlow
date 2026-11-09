"""Verify completed evidence and create a portable review archive on CPU only.

No model import, inference, training, rescoring, or scientific report generation.
REPORT/METHODS/README_RESUME are supplied by the experiment investigator and
are never overwritten. Large-file hashes come from existing audited receipts.
"""
from __future__ import annotations
import argparse
import ast
import csv
import difflib
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import zipfile

try:
    from . import common as C
except ImportError:
    import common as C

OUT, CODE = C.OUT, C.CODE
PACKAGE = OUT / 'review_package'
FINAL = OUT / 'finalization'
ZIP = OUT / 'R2_ANCHOR_AWARE_DENOISER_REVIEW.zip'
VERSION = 'r2_anchor_denoiser_finalization_v1'
ARMS = ('D_CONT_NO_ROLE', 'D_ANCHOR')
DENOISERS = ('D0_OFFICIAL',) + ARMS
BINARY = {'.pt', '.pth', '.npz', '.npy', '.ply', '.safetensors', '.bin'}
HEX = re.compile(r'^[0-9a-f]{64}$')


class Incomplete(RuntimeError):
    pass


def need(test, message):
    if not test:
        raise Incomplete(message)


def existing(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size > 0, 'Required artifact missing/empty: ' + str(path))
    return path


def read_required(path, status=None):
    d = C.read(existing(path))
    if status is not None:
        need(d.get('status') == status, f'{path}: expected status {status}, got {d.get("status")}')
    return d


def rows(path):
    with existing(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def same_destination(path, data):
    path = C.checked(path)
    if path.exists():
        need(path.read_bytes() == data, 'Existing finalization artifact differs; inspect before replacing: ' + str(path))
        return
    C.atomic(path, lambda f: f.write(data), True)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def json_new(path, value):
    same_destination(path, json_bytes(value))


def csv_new(path, data):
    need(bool(data), 'Cannot write empty CSV without schema')
    buf = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in data for k in row))
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader(); writer.writerows(data)
    same_destination(path, buf.getvalue().encode('utf-8'))


def prerequisites():
    checks = {}
    registration = read_required(OUT / 'registration.json')
    need(registration.get('status') == 'REGISTERED', 'Main registration not frozen')
    for name in ('REPORT.md', 'METHODS.md', 'README_RESUME.md'):
        path = existing(OUT / name)
        checks[str(path)] = C.sha(path)
    for arm in ARMS:
        path = OUT / 'training' / arm / 'completion.json'
        d = read_required(path, 'COMPLETE')
        for key, expected in (('completed_updates', 1000), ('optimizer_updates', 1000),
                              ('exposures', 8000), ('forwards', 8000), ('backwards', 8000)):
            need(d.get(key) == expected, f'{arm}: incomplete/incorrect {key}')
        checks[str(path)] = C.sha(path)
    phases = [('synthetic', 'D0_OFFICIAL', 18), ('real', ARMS[0], 48), ('synthetic', ARMS[0], 18),
              ('real', ARMS[1], 48), ('synthetic', ARMS[1], 18)]
    for stage, arm, count in phases:
        path = OUT / 'eval_completion' / f'{stage}_{arm}.json'
        d = read_required(path, 'COMPLETE')
        need(d.get('outputs') == d.get('expected_outputs') == d.get('total_attempts') == count,
             f'{stage}/{arm}: evaluation output/call count does not equal {count}')
        need(d.get('all_C_byte_exact') is True and d.get('all_no_rng_consumed') is True,
             f'{stage}/{arm}: C preservation or RNG contract not verified')
        checks[str(path)] = C.sha(path)
    e = read_required(OUT / 'evaluation/completion.json', 'PASS')
    for key, expected in (('primary_POST', 144), ('new_GEO_C', 96), ('new_quality', 96),
                          ('D0_quality_reused', 48), ('primary_AVMR_POST_and_PRE_GEO_rows', 2304),
                          ('boundary_vertex_rows', 2976)):
        need(e.get(key) == expected, 'Incomplete evaluation field ' + key)
    for name, count in (('post_index.csv', 144), ('quality_per_output.csv', 144),
                         ('topology_per_output.csv', 144), ('per_output.csv', 2304),
                         ('per_boundary_vertex.csv', 2976)):
        need(len(rows(OUT / 'evaluation' / name)) == count, 'Wrong final CSV row count: ' + name)
    need(len(rows(OUT / 'synthetic_results/per_input.csv')) == 54, 'Synthetic table must contain 54 outputs')
    need(read_required(OUT / 'synthetic_results/aggregation.json', 'COMPLETE')['outputs'] == 54,
         'Synthetic aggregate incomplete')
    audit_path = OUT / 'independent_audit/final_REPORT.json'
    audit = read_required(audit_path, 'PASS')
    need(audit.get('stage') == 'final', 'Independent final-stage audit required')
    for section in ('protected', 'preflight', 'frozen_stream', 'training', 'checkpoints', 'inference', 'evaluation'):
        need(audit.get(section, {}).get('status') == 'PASS', 'Independent audit incomplete: ' + section)
    need(audit.get('source_sha256') == C.sha(CODE / 'audit_final.py'), 'Independent audit source changed after audit')
    need(audit['checkpoints'].get('CUDA_initialized') is False, 'Checkpoint inspection was not CPU-only')
    checks[str(audit_path)] = C.sha(audit_path)
    for p, expected in registration['training_source_hashes'].items():
        need(C.sha(p) == expected, 'Frozen training source changed: ' + p)
    gallery_path = OUT / 'gallery/gallery_manifest.json'
    gallery = read_required(gallery_path, 'COMPLETE')
    need(gallery.get('POST_outputs') == 144 and gallery.get('images') == 56,
         'Require all 144 POST outputs and 56 actual rendered sheets')
    need(gallery.get('code_sha256') == C.sha(CODE / 'gallery.py'), 'Gallery code changed after rendering')
    index = rows(OUT / 'evaluation/post_index.csv')
    expected_keys = {(r['denoiser'], r['label'], r['request_id']) for r in index}
    need(len(expected_keys) == 144, 'POST identity duplicates')
    review_path = OUT / 'gallery/AI_PROXY_per_output_review.csv'
    review = rows(review_path)
    actual_keys = {(r['denoiser'], r['generator'], r['request_id']) for r in review}
    need(len(review) == 144 and actual_keys == expected_keys, 'Actual visual review must cover each of 144 POST identities once')
    placeholders = {'', 'PENDING', 'NOT_REVIEWED', 'TODO'}
    for r in review:
        need(r.get('review_kind') == 'AI_PROXY' and r.get('review_status') == 'COMPLETE',
             'Review contains pending/non-AI_PROXY rows')
        for key in ('C_use', 'boundary_continuation', 'overall_quality'):
            need(r.get(key, '').strip().upper() not in placeholders, 'Review field unfilled: ' + key)
        need(r.get('blind', '').lower() in ('false', '0', 'no'), 'Do not claim blind review')
    receipt_path = OUT / 'gallery/AI_PROXY_review_receipt.json'
    reviewed = read_required(receipt_path, 'COMPLETE')
    need(reviewed.get('reviewed_POST_outputs') == 144 and reviewed.get('reviewed_images') == 56,
         'Actual image review receipt incomplete')
    image_receipts = reviewed.get('images', [])
    need(len(image_receipts) == 56, 'Expected 56 actually viewed image hashes')
    expected_pngs = {str(Path(r['png']).resolve()): r['png_sha256'] for r in gallery['receipts']}
    seen = set()
    for item in image_receipts:
        image = Path(item['path']).resolve()
        need(item.get('actually_viewed') is True, 'Review receipt does not attest actual image view')
        need(str(image) in expected_pngs and str(image) not in seen, 'Unexpected/duplicate viewed image')
        need(item['sha256'] == expected_pngs[str(image)] == C.sha(image), 'Viewed image hash differs')
        seen.add(str(image))
    need(seen == set(expected_pngs), 'Some rendered images have not actually been reviewed')
    for r in gallery['receipts']:
        need(C.sha(r['jpg']) == r['jpg_sha256'], 'Compact JPEG changed')
    for path in (gallery_path, review_path, receipt_path, OUT / 'evaluation/completion.json'):
        checks[str(path)] = C.sha(path)
    existing(OUT / 'gallery/AI_PROXY_REVIEW.md')
    phases = [C.read(p) for p in (OUT / 'gpu_phases').glob('*.json')]
    need(phases and all(p['status'] in ('COMPLETE', 'PAUSED_BOUNDARY') for p in phases),
         'Unfinished or failed GPU phase remains')
    need(sum(p.get('gpu_occupancy_seconds', 0) for p in phases) <= 10800, 'Recorded GPU budget exceeded')
    return registration, audit, gallery, checks


def snapshot_sources(registration):
    """Copy real executed Python files, including the original scorer definitions."""
    entries = {}
    for path in CODE.glob('*.py'):
        entries[path.resolve()] = ('pilot', None)
    eval_registration = read_required(OUT / 'evaluation/registration.json')
    for origin in (registration.get('protected', {}), eval_registration.get('protected', {}),
                   eval_registration.get('quality_source_hashes', {})):
        for name, expected in origin.items():
            p = Path(name)
            if p.suffix == '.py':
                entries[p.resolve()] = ('inherited_readonly', expected)
    official = C.OLD / 'source_snapshot/official_denoiser_db86bc3'
    official_manifest = C.read(C.OLD / 'asset_audit/official_denoiser_source_manifest.json')
    need(official_manifest['commit'] == 'db86bc32423c316c00bcc2f645a26243e8010f20', 'Official code revision differs')
    for r in official_manifest['files']:
        p = official / r['path']
        if p.suffix == '.py':
            entries[p.resolve()] = ('official_readonly', r['sha256'])
    for base in (OUT / 'evaluation/compatibility', OUT / 'stream_independent_audit'):
        for p in base.glob('*.py'):
            entries[p.resolve()] = ('audit_or_provenance', None)
    result = []
    for path, (kind, expected) in sorted(entries.items(), key=lambda item: str(item[0])):
        # Official Python package markers can legitimately be empty.
        need(path.is_file(), 'Required source missing: ' + str(path))
        value = C.sha(path)
        if expected is not None:
            need(value == expected, 'Source does not match registered hash: ' + str(path))
        need(path.stat().st_size < 4 * (1 << 20), 'Unexpectedly large source file')
        if kind == 'pilot':
            dest = OUT / 'source_snapshot/pilot' / path.name
        elif kind == 'official_readonly':
            dest = OUT / 'source_snapshot/official' / path.relative_to(official)
        else:
            dest = OUT / 'source_snapshot' / kind / (value[:12] + '_' + path.name)
        same_destination(dest, path.read_bytes())
        result.append(dict(original_path=str(path), snapshot_path=str(dest.relative_to(OUT)),
            role=kind, bytes=path.stat().st_size, sha256=value))
    # Preserve the exact in-memory source fragments actually extracted by model.py.
    rms_path = official / 'models/rmsnorm.py'; src = rms_path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'RMSNorm')
    fragments = {'RMSNorm.py': 'import torch\nfrom torch import nn\n\n' + ast.get_source_segment(src, node) + '\n'}
    utils_path = official / 'models/utils.py'; src = utils_path.read_text(encoding='utf-8')
    wanted = {'get_2d_sincos_pos_embed', 'get_2d_sincos_pos_embed_from_grid', 'get_1d_sincos_pos_embed_from_grid'}
    parts = [ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    need(len(parts) == 3, 'Official runtime positional helper fragment mismatch')
    fragments['positional_helpers.py'] = 'import numpy as np\n\n' + '\n\n'.join(parts) + '\n'
    for name, text in fragments.items():
        dest = OUT / 'source_snapshot/actual_runtime_fragments' / name
        same_destination(dest, text.encode('utf-8'))
        result.append(dict(original_path=str(rms_path if name.startswith('RMS') else utils_path) + '#AST_extracted_runtime_fragment',
            snapshot_path=str(dest.relative_to(OUT)), role='actual_runtime_fragment_no_model_executed',
            bytes=dest.stat().st_size, sha256=C.sha(dest)))
    need(any(Path(r['original_path']).name == 'intersections.py' for r in result), 'Actual intersections scorer source missing')
    need(any(Path(r['original_path']).name == 'pair_audit.py' for r in result), 'Actual pair audit source missing')
    need(any(Path(r['original_path']).name == 'shape.py' or 'shape' in Path(r['original_path']).stem for r in result), 'Actual shape scorer source missing')
    csv_new(OUT / 'source_hashes.csv', result)
    diff = ['# All pilot files below are additions relative to the unchanged original code.\n',
            '# Readonly inherited/official sources are included byte-for-byte with hashes.\n']
    for p in sorted(CODE.glob('*.py')):
        diff.extend(difflib.unified_diff([], p.read_text(encoding='utf-8-sig').splitlines(keepends=True),
            fromfile='/dev/null', tofile='b/r2/pilots/anchor_aware_denoiser_v1/' + p.name))
    same_destination(OUT / 'code.diff', ''.join(diff).encode('utf-8'))
    json_new(OUT / 'source_snapshot/official_source_manifest.json', official_manifest)
    return result


def receipt_hashes():
    """Recover saved path/hash links without loading binary arrays or weights."""
    saved = {}
    def add(path, digest, origin):
        if not isinstance(path, str) or not path.startswith('/mnt/') or not isinstance(digest, str) or not HEX.fullmatch(digest):
            return
        p = Path(path)
        if p.suffix.lower() not in BINARY:
            return
        key = str(p.resolve())
        if key in saved:
            need(saved[key]['sha256'] == digest, 'Conflicting saved binary hashes: ' + key)
        else:
            saved[key] = dict(sha256=digest, receipt=origin)
    def walk(value, origin, directory):
        if isinstance(value, dict):
            for k, v in value.items():
                add(k, v, origin)
            for pkey, hkey in (('path', 'sha256'), ('output_path', 'output_sha256'), ('npz', 'sha256'),
                               ('source_npz', 'source_npz_sha256'), ('raw_path', 'raw_sha256')):
                add(value.get(pkey), value.get(hkey), origin)
            files = value.get('files')
            if isinstance(files, dict):
                for name, item in files.items():
                    if isinstance(item, dict) and 'sha256' in item:
                        add(str(Path(name) if Path(name).is_absolute() else directory / name), item['sha256'], origin)
            for v in value.values():
                walk(v, origin, directory)
        elif isinstance(value, list):
            for v in value:
                walk(v, origin, directory)
    for p in sorted(OUT.rglob('*.json')):
        if any(x in p.relative_to(OUT).parts for x in ('review_package', 'finalization', 'source_snapshot')):
            continue
        walk(C.read(p), str(p.relative_to(OUT)), p.parent)
    return saved


def local_binary_manifest(audit):
    saved = receipt_hashes()
    files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() in BINARY and 'review_package' not in p.parts]
    files.extend(Path(p) for p in saved if Path(p).is_file())
    files.append(C.ASSET)
    audited = {r['arm']: r for r in audit['checkpoints']['arms']}
    result = []
    for p in sorted(set(files), key=str):
        key = str(p.resolve())
        if key in saved:
            record = saved[key]
            h = record['sha256']
            origin = record['receipt']
            basis = 'IMPORTED_EXISTING_RECEIPT_HASH_AFTER_FINAL_AUDIT_PASS; NOT_REHASHED_BY_FINALIZER'
        else:
            need(p.stat().st_size <= 64*(1 << 20), 'Large binary has no hash receipt: ' + str(p))
            h, origin, basis = C.sha(p), 'finalizer_direct_small_binary_hash', 'DIRECT_SMALL_BINARY_SHA256'
        if p == C.ASSET:
            need(h == C.EXPECTED, 'Official checkpoint receipt hash differs')
            basis = 'OFFICIAL_HASH_VERIFIED_BY_FINAL_PROTECTED_AUDIT; RECEIPT_IMPORTED_NOT_REHASHED'
        for arm in ARMS:
            if p == OUT / 'checkpoints' / arm / 'step1000.pt':
                need(h == audited[arm]['full_checkpoint_sha256'], 'Full checkpoint final-audit hash differs')
                basis = 'FINAL_CPU_CHECKPOINT_AUDIT_HASH_IMPORTED; NOT_REHASHED'
            elif p == OUT / 'checkpoints' / arm / 'inference.pt':
                need(h == audited[arm]['inference_checkpoint_sha256'], 'Inference checkpoint final-audit hash differs')
                basis = 'FINAL_CPU_CHECKPOINT_AUDIT_HASH_IMPORTED; NOT_REHASHED'
            elif p.suffix == '.pt' and p.is_relative_to(OUT / 'checkpoints' / arm):
                basis = 'CHECKPOINT_WRITER_RECEIPT_HASH_IMPORTED; NOT_REHASHED_BY_FINAL_AUDIT_OR_FINALIZER'
        result.append(dict(path=str(p), bytes=p.stat().st_size, sha256=h, hash_origin_receipt=origin,
            verification_basis=basis, artifact_location='CURRENT_PILOT' if p.is_relative_to(OUT) else 'REFERENCED_READONLY_SOURCE', embedded_in_review_zip=False))
    csv_new(OUT / 'local_binary_artifacts.csv', result)
    return result


def omitted_json_reason(path):
    rel = path.relative_to(OUT)
    name = rel.as_posix()
    exact = {
        'stream/manifest.json': 'Full recoverable batch manifest; stream plan_rows.csv, audit and all batch hash references remain available locally.',
        'stream/plan.json': 'Full recoverable stream plan duplicated by the complete embedded plan_rows.csv.',
        'stream/pre_materialization_plan_v0.json': 'Superseded pre-materialization plan; correction ledger retained.',
        'pre_gpu_registration_v0.json': 'Superseded registration; current formal registration and correction evidence retained.',
        'evaluation/compatibility/registration_initial.json': 'Initial registration duplicate; current formal evaluation registration and compatibility audit retained.',
        'stream_independent_audit/protected.json': 'Detailed repeated protected-hash inventory; independent audit summary and exact local artifact hash retained.',
    }
    if name in exact:
        return exact[name]
    if rel.parts[:2] == ('evaluation', 'outputs') and path.name == 'postprocess.json':
        return 'Detailed per-face geometry-processing map remains local; all per-output CSVs, output hashes and actual processing code embedded.'
    if rel.parts[0] == 'inference' and path.suffix == '.json':
        return 'Individual saved-array receipt remains local; formal evaluation completions, call ledger, index and binary hash manifest embedded.'
    if rel.parts[0] == 'synthetic_results' and path.name not in ('aggregation.json',) and path.suffix == '.json':
        return 'Individual synthetic result receipt remains local; complete per_input.csv and aggregation embedded.'
    if rel.parts[0] == 'gallery' and len(rel.parts) == 2 and '__' in path.name and path.suffix == '.json':
        return 'Individual render sidecar duplicated in full by the embedded gallery_manifest receipts.'
    return None


def selected_for_package(path):
    rel = path.relative_to(OUT)
    if path.suffix == '.json' and omitted_json_reason(path):
        return False
    if any(part in ('review_package', 'finalization', '__pycache__') for part in rel.parts):
        return False
    if path.suffix.lower() in BINARY or path.suffix.lower() in ('.png', '.zip', '.tmp'):
        return False
    if path.name in ('review_pending.csv', 'index.html'):
        return False
    if path.suffix.lower() in ('.csv', '.md', '.jsonl', '.diff'):
        return True
    if path.suffix == '.py':
        return rel.parts[0] in ('source_snapshot', 'stream_independent_audit', 'independent_audit')
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        return rel.parts[:2] == ('gallery', 'review_jpg')
    if path.suffix == '.json':
        if rel.parts[:2] in (('stream', 'batches'), ('evaluation', 'scores'), ('evaluation', 'diagnostics')):
            return False
        return True
    return False


def portable_gallery(gallery):
    content = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Anchor-aware denoiser review</title>',
        '<style>body{font:15px Arial;margin:24px;max-width:1600px}img{width:100%;height:auto}section{margin:32px 0}a{color:#176299}</style>',
        '<h1>RAW / D0_OFFICIAL / D_CONT_NO_ROLE / D_ANCHOR</h1>',
        '<p>All 48 fixed requests / 144 POST outputs. Three solid whole-object views plus opposite wireframe. Eight additional USER_MIX local views. Blue C and gray free; identical camera, bounds and materials within each sheet.</p>',
        '<p>AI_PROXY review, not blind human evaluation. Appearance and C preservation do not certify actual attachment. Full-resolution PNGs, meshes, arrays and checkpoints remain local.</p>',
        '<p><a href="../REPORT.md">Scientific report</a> | <a href="AI_PROXY_REVIEW.md">Actual visual review</a> | <a href="AI_PROXY_per_output_review.csv">All 144 review rows</a></p>']
    for r in gallery['receipts']:
        name = Path(r['jpg']).name
        title = r['identity']['generator'] + ' / ' + r['identity']['request_id']
        if r['identity']['bounds']['local_plot_crop']:
            title += ' / additional local view'
        content.append('<section><h2>' + html.escape(title) + '</h2><a href="review_jpg/' + html.escape(name, quote=True) + '"><img loading="lazy" src="review_jpg/' + html.escape(name, quote=True) + '"></a></section>')
    content.append('</html>')
    return '\n'.join(content).encode('utf-8')


def omitted_json_manifest():
    rows = []
    for path in sorted(OUT.rglob('*.json')):
        if any(part in ('review_package', 'finalization', 'source_snapshot') for part in path.relative_to(OUT).parts):
            continue
        reason = omitted_json_reason(path)
        if reason:
            rows.append(dict(path=str(path.relative_to(OUT)), absolute_local_path=str(path),
                bytes=path.stat().st_size, sha256=C.sha(path), reason=reason,
                local_original_preserved=True, embedded_in_review_zip=False))
    csv_new(OUT / 'OMITTED_JSON.csv', rows)
    return rows


def small_manifest():
    result = []
    excluded_names = {'small_file_hashes.csv', 'finalization_completion.json'}
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or any(v in ('review_package', 'finalization', '__pycache__') for v in p.relative_to(OUT).parts):
            continue
        if p.name in excluded_names or p.suffix.lower() in BINARY or p.suffix.lower() in ('.zip', '.tmp'):
            continue
        need(p.stat().st_size < 128*(1 << 20), 'Unexpected very large text/image artifact: ' + str(p))
        result.append(dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size,
            sha256=C.sha(p), hash_basis='DIRECT_FINALIZER_SHA256', embedded_in_review_zip=selected_for_package(p)))
    csv_new(OUT / 'small_file_hashes.csv', result)
    return result


def build_package(gallery, checks, source_rows, binaries):
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and selected_for_package(p):
            dest = PACKAGE / p.relative_to(OUT)
            need(p.resolve() != dest.resolve(), 'Source and package destination identical')
            same_destination(dest, p.read_bytes())
    same_destination(PACKAGE / 'gallery/index.html', portable_gallery(gallery))
    same_destination(PACKAGE / 'index.html', b'<!doctype html><meta charset="utf-8"><title>R2 anchor denoiser review</title><h1>R2_ANCHOR_AWARE_DENOISER_PILOT_V1</h1><ul><li><a href="REPORT.md">Report</a></li><li><a href="METHODS.md">Methods</a></li><li><a href="gallery/index.html">All image comparisons</a></li><li><a href="evaluation/quality_per_output.csv">Per-output quality</a></li><li><a href="evaluation/topology_per_output.csv">Actual connectivity</a></li><li><a href="README_PACKAGE.md">Package scope and omitted large artifacts</a></li></ul>')
    scope = ('# Review package\n\nThis package contains the supplied scientific REPORT, METHODS and README_RESUME, all existing per-output/per-parent/noise CSV tables, all 56 compact JPEG comparison sheets, actual AI_PROXY review records, the data whitelist and exclusions/source interfaces, actual source snapshots/code.diff/hashes, and resource/call/update ledgers.\n\n'
        'Large checkpoints, NPZ/NPY arrays, PLY meshes and full-resolution PNGs remain local and are deliberately not embedded. `local_binary_artifacts.csv` lists paths, sizes, saved SHA256 and the precise receipt/audit provenance. The finalizer does not reload or re-hash large weights; final full/inference checkpoint hashes come from the completed independent CPU checkpoint audit. Recovery-checkpoint hashes come from their writer receipts and are labelled accordingly.\n\n'
        '`small_file_hashes.csv` covers the local small text/image artifacts, including PNGs not embedded; its embedded flag distinguishes availability. `PACKAGE_MANIFEST.csv` hashes the actual portable package members. Source data/interface identifiers are evidence only, not model inference inputs. Absolute paths in provenance retain their original local identities and are not download links.\n\n'
        'The archive first uses standard DEFLATE. If that exceeds 10,000,000 bytes, CSV members use lossless ZIP-LZMA (method 14), leaving their exact bytes and all JPEG bytes unchanged. ZIP-LZMA requires a compatible extractor such as Python zipfile or 7-Zip; older Windows Explorer versions may not support it. The archive compression methods are recorded in the local finalization completion.\n\n'
        'The portable gallery uses JPEG copies only; no scientific metric was recomputed to build this archive. Main numerical tables preserve all generators, parents, noises and thresholds. Reports are copied unchanged; finalization does not invent success, select images, change defaults, run models, or commit/push.\n\n'
        '`OMITTED_JSON.csv` records the exact local path, SHA256, bytes and reason for recoverable stream plans, superseded registration copies and duplicate per-output/render detail JSONs excluded from this compact archive. Every original remains unchanged locally. All CSVs, official resource/call/update JSONL ledgers, current formal registrations, source interfaces/whitelist, final audit and all 56 unchanged JPEGs remain embedded.\n')
    same_destination(PACKAGE / 'README_PACKAGE.md', scope.encode('utf-8'))
    provenance = dict(schema=VERSION, source_files=len(source_rows), large_or_array_artifacts_local=len(binaries),
        prerequisite_hashes=checks, required_AI_PROXY_rows=144, JPEG_sheets=56, model_calls=0,
        weights_loaded=0, scientific_reports_modified=False, large_weight_hashes_reused=True,
        excluded_artifact_extensions=sorted(BINARY | {'.png'}))
    json_new(PACKAGE / 'PACKAGE_PROVENANCE.json', provenance)
    members = []
    for p in sorted(PACKAGE.rglob('*')):
        if p.is_file() and p.name != 'PACKAGE_MANIFEST.csv':
            members.append(dict(path=str(p.relative_to(PACKAGE)), bytes=p.stat().st_size, sha256=C.sha(p)))
    csv_new(PACKAGE / 'PACKAGE_MANIFEST.csv', members)
    package_files = sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    need(len([p for p in package_files if p.suffix.lower() == '.jpg']) == 56, 'Package missing JPEG comparisons')
    need(not any(p.suffix.lower() in BINARY | {'.png'} for p in package_files), 'Large array/weight/PNG entered compact package')
    # Deterministic ZIP metadata permits exact resume verification without replacing artifacts.
    def archive_bytes(csv_lzma=False):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for p in package_files:
                method = zipfile.ZIP_LZMA if csv_lzma and p.suffix.lower() == '.csv' else zipfile.ZIP_DEFLATED
                info = zipfile.ZipInfo(p.relative_to(PACKAGE).as_posix(), date_time=(2026, 10, 10, 0, 0, 0))
                info.compress_type = method; info.external_attr = 0o644 << 16
                archive.writestr(info, p.read_bytes(), compress_type=method, compresslevel=9)
        return buf.getvalue()
    if ZIP.exists():
        with zipfile.ZipFile(ZIP) as archive:
            need(set(archive.namelist()) == {p.relative_to(PACKAGE).as_posix() for p in package_files}, 'Existing ZIP members differ')
            for p in package_files:
                need(hashlib.sha256(archive.read(p.relative_to(PACKAGE).as_posix())).hexdigest() == C.sha(p), 'Existing ZIP member differs')
    else:
        payload = archive_bytes(False)
        if len(payload) >= 10_000_000:
            payload = archive_bytes(True)
        need(len(payload) < 10_000_000, 'Complete lossless archive still exceeds 10MB; preserve all package files for inspection')
        C.atomic(ZIP, lambda f: f.write(payload), True)
    with zipfile.ZipFile(ZIP) as archive:
        need(archive.testzip() is None, 'ZIP CRC verification failed')
        need(len(archive.namelist()) == len(package_files), 'ZIP duplicate/lost member')
        csv_methods = sorted({i.compress_type for i in archive.infolist() if i.filename.endswith('.csv')})
    # Verify portable links from the actual generated page, not the original local PNG gallery.
    portable = (PACKAGE / 'gallery/index.html').read_text(encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"]+)"', portable):
        path = (PACKAGE / 'gallery' / html.unescape(target)).resolve()
        need(path.is_relative_to(PACKAGE.resolve()) and path.is_file(), 'Portable gallery link missing: ' + target)
    return dict(path=str(ZIP), bytes=ZIP.stat().st_size, sha256=C.sha(ZIP), members=len(package_files),
        target_under_10MB=10_000_000, under_10MB=ZIP.stat().st_size < 10_000_000,
        readability_policy='Existing 56 JPEG render copies included unchanged; never reduce image readability merely to meet a byte target',
        CRC='PASS', csv_ZIP_compression_methods=csv_methods, package_manifest_sha256=C.sha(PACKAGE / 'PACKAGE_MANIFEST.csv'), portable_gallery_links='PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Import/AST validation only; no experiment artifacts written')
    args = parser.parse_args()
    if args.check_only:
        ast.parse(Path(__file__).read_text(encoding='utf-8-sig'))
        print(json.dumps(dict(status='PASS_STATIC_ONLY', model_calls=0, experiment_writes=0)))
        return
    completion = OUT / 'finalization_completion.json'
    if completion.exists():
        done = read_required(completion, 'COMPLETE')
        need(C.sha(ZIP) == done['archive']['sha256'], 'Completed archive changed')
        for p, expected in done['prerequisite_hashes'].items():
            need(C.sha(p) == expected, 'Finalized prerequisite changed: ' + p)
        print(json.dumps(dict(status='SKIP_VERIFIED_COMPLETE', archive=done['archive'])))
        return
    registration, audit, gallery, checks = prerequisites()
    source_rows = snapshot_sources(registration)
    binaries = local_binary_manifest(audit)
    omitted_json_manifest()
    small = small_manifest()
    archive = build_package(gallery, checks, source_rows, binaries)
    for p, expected in checks.items():
        need(C.sha(p) == expected, 'Prerequisite/report changed during finalization: ' + p)
    result = dict(schema=VERSION, status='COMPLETE', archive=archive, prerequisite_hashes=checks,
        source_files=len(source_rows), small_artifacts_directly_hashed=len(small),
        binary_artifacts_listed_local=len(binaries), actual_AI_PROXY_POST_rows=144,
        actual_reviewed_images=56, large_weight_rehashes=0, model_calls=0, training_updates=0,
        source_code_sha256=C.sha(__file__), scientific_reports_modified=False,
        default_models_changed=False, git_commit_push=False)
    json_new(completion, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Incomplete as exc:
        print(json.dumps(dict(status='FINALIZATION_BLOCKED_OR_INCOMPLETE', reason=str(exc),
            model_calls=0, retries_or_replays=0), ensure_ascii=False))
        raise SystemExit(2)
