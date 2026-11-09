"""Complete-only small review archive, from original pilot evidence.

Default/--check writes readiness only. --build requires the final analyses,
independent audit, root-authored REPORT.md and every registered case. No model
imports/forwards, scoring, image edits or scientific conclusions. Large model
files, mesh arrays and full galleries remain local with paths and hashes.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

from . import data as D

VERSION = 'terminal_reward_nft_evidence_pack_v2'
DEST = D.OUT / 'evidence_pack'
ARMS = ('NFT', 'SELF_FM')
MODELS = ('BASE', 'NFT', 'SELF_FM')
REVIEW_KEYS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
ALLOWED = {'YES', 'NO', 'UNKNOWN', 'UNCERTAIN', 'NA'}
TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.md', '.txt', '.log', '.diff', '.before', '.py'}
LIMIT = 10_000_000  # Decimal MB target, never a reason to discard evidence.
FINAL_TABLES = ('per_output.csv', 'per_output_stage.csv', 'per_parent.csv', 'paired_per_request.csv', 'diversity_pairs.csv')
LEARNING_TABLES = {'training_updates': 512, 'physical_workers': 33, 'adapter_changes': 32,
    'reward_outputs': 512, 'reward_conditions': 64, 'bestof8_frozen_FULL_round00': 4, 'bestof8_online_NFT': 60}


def blob_sha(b):
    return hashlib.sha256(b).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def rows(path):
    if str(path).lower().endswith('.csv'):
        with D.resolve(path).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    v = D.read(path)
    return v['rows'] if isinstance(v, dict) else v


def local_paths(path):
    p = D.resolve(path).resolve()
    s = str(p).replace('\\', '/')
    win = s
    if s.startswith('/mnt/') and len(s) > 7:
        win = s[5].upper() + ':/' + s[7:]
    return dict(local_path=str(p), windows_path=win)


def pick_learning(explicit=None):
    if explicit:
        return D.resolve(explicit)
    candidates = []
    for p in (D.OUT / 'learning_analysis/snapshots').glob('*/summary.json'):
        try:
            v = D.read(p)
            candidates.append((v.get('status') == 'COMPLETE', p.stat().st_mtime_ns, str(p), p))
        except (ValueError, OSError):
            continue
    return max(candidates)[-1] if candidates else D.OUT / 'learning_analysis/COMPLETE_SUMMARY_MISSING.json'


def audit_summary_path():
    pointer = D.OUT / 'audit/latest.json'
    if not pointer.is_file():
        return D.OUT / 'audit/LATEST_SUMMARY_MISSING.json'
    v = D.read(pointer)
    for key in ('summary_path', 'summary', 'path'):
        if isinstance(v.get(key), str):
            p = D.resolve(v[key])
            return p / 'summary.json' if p.is_dir() else p
    for key in ('output_dir', 'snapshot_dir'):
        if isinstance(v.get(key), str):
            return D.resolve(v[key]) / 'summary.json'
    if v.get('input_digest'):
        return D.OUT / 'audit/snapshots' / str(v['input_digest'])[:20] / 'summary.json'
    raise ValueError('audit/latest.json has no recognized summary pointer')


PACKAGING_CONTRACT = dict(
    version=VERSION, change='V2 packaging projection only; no scientific definition, label, score, stream or original file changed',
    prior_v1='Attempted to include complete score.json and duplicated detailed arrays; no formal V1 archive was generated',
    rationale='Eight-round metadata: score.json 941036894 bytes, pairs.json 696875631 bytes; sampled level9 DEFLATE estimates about48.2MB and32.6MB respectively before other evidence',
    required='All per-output CSVs, actual reviews including all NO/UNKNOWN, complete reward components, all original score metrics, topology scalar summaries/statuses, denominators, registration/streams, analyses and actual sources',
    score_projection='Every available RAW/POST score gets a score_compact.json and a row in score_compact.csv. Full metrics/identity/files and all scalar summaries retained; omitted arrays/nested detail keys explicitly listed',
    external='Full original score.json, repeated score detail files, per-face/per-pair/per-query arrays, original resource polling and non-recovery forward-event ledgers remain local with exact path/hash/bytes. Round0 recovery event ledgers retained',
    resources='Include complete latest resource_summary workers/groups/dispatch/admission/concurrency and CSVs; observations.json is external with hash; WDDM NA and coverage limitations remain in summaries',
    images='All actual review rows retained and all full image paths/hashes indexed; no quality-based example filtering',
    size_target_bytes=LIMIT, size_policy='Report actual compressed size and target_exceeded; do not silently remove valid evidence to meet target',
    models_loaded=0, metric_recomputations=0, original_files_modified=False,
)


def resource_summary_path():
    pointer = D.OUT / 'resource_summary/latest.json'
    if not pointer.is_file():
        return D.OUT / 'resource_summary/LATEST_SUMMARY_MISSING.json'
    v = D.read(pointer)
    if isinstance(v.get('summary_path'), str):
        return D.resolve(v['summary_path'])
    return D.resolve(v.get('output_dir', v.get('snapshot_path', 'resource_summary/MISSING'))) / 'summary.json'


def readiness(learning_summary=None, report=None, reviews=None):
    paths = dict(final_summary=D.OUT / 'evaluation/analysis/summary.json',
        learning_summary=pick_learning(learning_summary), audit_summary=audit_summary_path(),
        resource_summary=resource_summary_path(),
        report=D.resolve(report or D.OUT / 'REPORT.md'),
        final_reviews=D.resolve(reviews or D.OUT / 'evaluation/blind_gallery/reviews_actual.json'))
    missing, problems, observed = [], [], {}

    def get(path):
        p = D.resolve(path)
        if not p.is_file():
            missing.append(str(p)); return None
        observed[str(p)] = dict(sha256=D.sha(p), bytes=p.stat().st_size)
        if p.suffix == '.json':
            try:
                return D.read(p)
            except (ValueError, OSError) as exc:
                problems.append(str(p) + ': ' + str(exc))
        return None

    manifest = get(D.OUT / 'inputs/manifest.json')
    for path in paths.values():
        get(path)
    fs = D.read(paths['final_summary']) if paths['final_summary'].is_file() else {}
    ls = D.read(paths['learning_summary']) if paths['learning_summary'].is_file() else {}
    au = D.read(paths['audit_summary']) if paths['audit_summary'].is_file() else {}
    if fs and (fs.get('status') != 'COMPLETE' or any(fs.get(k) != n for k, n in
            (('outputs', 108), ('RAW_scores', 108), ('POST_scores', 108), ('actual_reviews', 108), ('distinct_requests', 36)))):
        problems.append('Final analysis is not COMPLETE with all108 outputs/216 scores/108 reviews')
    if ls and (ls.get('status') != 'COMPLETE' or ls.get('completed_rounds') != {a: list(range(16)) for a in ARMS}
            or ls.get('reward_rounds') != list(range(16)) or ls.get('totals', {}).get('logical_committed_updates') != 512):
        problems.append('Learning analysis is not COMPLETE with 16 rounds/256 logical updates per arm')
    if au and (au.get('status') != 'COMPLETE' or au.get('budget_checks', {}).get('status') != 'PASS'
            or au.get('protected_inputs', {}).get('status') != 'PASS' or au.get('violations') or au.get('partial_reasons')):
        problems.append('Latest independent execution audit is not COMPLETE/PASS without violations')
    resource = D.read(paths['resource_summary']) if paths['resource_summary'].is_file() else {}
    if resource and resource.get('status') != 'COMPLETE':
        problems.append('Latest resource summary is not COMPLETE; documented historical coverage gaps are retained, not hidden')
    for folder, key in (('audit', 'audit_summary'), ('resource_summary', 'resource_summary')):
        pointer = get(D.OUT / folder / 'latest.json')
        if pointer and paths[key].is_file() and pointer.get('summary_sha256') != D.sha(paths[key]):
            problems.append(folder + ' latest pointer summary hash differs')
    if paths['report'].is_file() and paths['report'].stat().st_size < 64:
        problems.append('Root REPORT.md is empty or a placeholder')
    for name in FINAL_TABLES:
        get(paths['final_summary'].parent / name)
    for name in LEARNING_TABLES:
        get(paths['learning_summary'].parent / (name + '.csv'))
        get(paths['learning_summary'].parent / (name + '.json'))
    if manifest:
        for name in ('registration.json', 'execution_source_lock.json', 'inputs/training_stream_schedule.json',
                'inputs/training_queries.json', 'inputs/selection.json', 'inputs/bestof8_scope_addendum.json',
                'evaluation/private/label_mapping.json', 'evaluation/blind_gallery/manifest.json', 'audit/latest.json'):
            get(D.OUT / name)
        for ri in range(16):
            stage = D.OUT / f'training/round_{ri:02d}'
            for name in ('reviews_actual.json', 'rewards.json', 'rewards.csv', 'training_stream.json',
                    'learning_input_seal.json', 'gallery/manifest.json', 'raw_index.json', 'proposal_index.json'):
                get(stage / name)
            for arm in ARMS:
                receipt = get(D.OUT / f'checkpoints/{arm}/round{ri:02d}.json')
                if receipt and (receipt.get('step') != (ri + 1) * 16 or receipt.get('round_index') != ri):
                    problems.append(f'{arm} round{ri:02d} checkpoint receipt clock differs')
            if (stage / 'reviews_actual.json').is_file():
                rr = rows(stage / 'reviews_actual.json')
                expected = {q['request_id'] for q in manifest['training_requests'] if q['round_index'] == ri}
                if len(rr) != 32 or {r['request_id'] for r in rr} != expected:
                    problems.append(f'round{ri:02d} actual review coverage differs')
                if any(r.get('review_status') != 'REVIEWED' or any(r.get(k) not in ALLOWED for k in REVIEW_KEYS) for r in rr):
                    problems.append(f'round{ri:02d} has invalid/unreviewed actual review rows')
        for arm in ARMS:
            p = D.OUT / f'checkpoints/{arm}/round15.pt'
            if not p.is_file():
                missing.append(str(p))
        for q in manifest['training_requests']:
            p = D.OUT / f"training/round_{q['round_index']:02d}/scores/{q['request_id']}/POST/score.json"
            if not p.is_file():
                missing.append(str(p))
        for model in MODELS:
            for q in manifest['evaluation_requests']:
                for stage in ('RAW', 'POST'):
                    p = D.OUT / 'evaluation' / model / 'scores' / q['request_id'] / stage / 'score.json'
                    if not p.is_file():
                        missing.append(str(p))
        if paths['final_reviews'].is_file():
            rr = rows(paths['final_reviews'])
            expect = {(q['request_id'], label) for q in manifest['evaluation_requests'] for label in ('A', 'B', 'C')}
            if len(rr) != 108 or {(r['request_id'], r['label']) for r in rr} != expect:
                problems.append('Final108 actual anonymous review coverage differs')
            if any(r.get('review_status') != 'REVIEWED' or any(r.get(k) not in ALLOWED for k in REVIEW_KEYS) for r in rr):
                problems.append('Final actual review contains invalid/unreviewed rows')
    missing = sorted(set(missing))
    return dict(version=VERSION, status='READY_FOR_VALIDATION' if not missing and not problems else 'INCOMPLETE',
        missing=missing, missing_count=len(missing), problems=problems, paths={k: str(p) for k, p in paths.items()},
        observed_gates=observed, archive_created=False, report_authored_by_builder=False,
        models_loaded=0, model_forwards=0, reason='Only COMPLETE inputs permit archive creation; UNKNOWN review labels remain valid retained evidence')


def _scalar_summary(value, pointer, omitted):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(item, list):
                omitted.append(pointer + '/' + key)
            else:
                result[key] = _scalar_summary(item, pointer + '/' + key, omitted)
        return result
    if isinstance(value, list):
        omitted.append(pointer)
        return None
    return value


def _csv_bytes(records):
    fields = sorted({k for row in records for k in row})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in records:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


class Pack:
    def __init__(self):
        self.members = {}
        self.origins = {}
        self.external = {}
        self.compact_rows = {}

    def add_bytes(self, name, payload, origin=None):
        name = PurePosixPath(name).as_posix()
        if name.startswith('/') or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe archive member name')
        if name in self.members and self.members[name] != payload:
            raise ValueError('Archive member collision: ' + name)
        self.members[name] = payload
        self.origins[name] = origin or dict(kind='DERIVED_CONTAINER_METADATA')
        return name

    def member_name(self, path):
        p = D.resolve(path).resolve()
        if p.is_relative_to(D.OUT.resolve()):
            return 'experiment/' + p.relative_to(D.OUT.resolve()).as_posix()
        if p.is_relative_to(D.ROOT.resolve()):
            return 'sources/repository/' + p.relative_to(D.ROOT.resolve()).as_posix()
        return 'sources/external/' + D.sha(p)[:12] + '/' + p.name

    def add(self, path, name=None, expected=None):
        p = D.resolve(path).resolve()
        if p.suffix.lower() not in TEXT_SUFFIXES | {'.png'}:
            raise ValueError('Large/raw/binary asset not eligible for archive: ' + str(p))
        if p.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('Small evidence file exceeds 64 MiB: ' + str(p))
        payload = p.read_bytes(); digest = blob_sha(payload)
        if expected is not None and digest != expected:
            raise ValueError('Source/input hash differs: ' + str(p))
        return self.add_bytes(name or self.member_name(p), payload,
            dict(kind='ORIGINAL_BYTES', sha256=digest, **local_paths(p)))

    def externalize(self, path, reason, digest=None, size=None):
        p = D.resolve(path).resolve()
        row = dict(**local_paths(p), sha256=digest or D.sha(p),
            bytes=p.stat().st_size if size is None else size,
            externalization_reason=reason, included_in_archive=False,
            original_file_modified=False)
        old = self.external.get(str(p))
        if old and (old['sha256'], old['bytes']) != (row['sha256'], row['bytes']):
            raise ValueError('External evidence changed while packaging: ' + str(p))
        self.external[str(p)] = row
        return row

    def add_score(self, path):
        p = D.resolve(path).resolve()
        if str(p) in self.compact_rows:
            return
        payload = p.read_bytes(); digest = blob_sha(payload)
        original = json.loads(payload.decode('utf-8-sig'))
        if not isinstance(original.get('metrics'), dict) or not isinstance(original.get('topology'), dict):
            raise ValueError('Score lacks full metrics/topology dictionaries: ' + str(p))
        source = self.externalize(p, 'FULL_SCORE_WITH_REPEATED_DETAILS; compact metrics and topology summaries included', digest, len(payload))
        full_keys = [k for k in ('metrics', 'identity', 'files', 'boundary_audit', 'distance') if k in original]
        retained = {k: original[k] for k in full_keys}
        scalar_keys, summary_keys, omitted = [], [], []
        for key, value in original.items():
            if key in retained:
                continue
            if isinstance(value, (dict, list)):
                if key in ('topology', 'source_C_connectivity_audit'):
                    retained[key] = _scalar_summary(value, '/' + key, omitted)
                    summary_keys.append(key)
                else:
                    omitted.append('/' + key)
            else:
                retained[key] = value
                scalar_keys.append(key)
        projection = dict(packaging_version=VERSION, projection_only=True, metric_recomputed=False,
            original_score=source, retained_full_top_level_keys=full_keys,
            retained_scalar_top_level_keys=sorted(scalar_keys), retained_summary_top_level_keys=summary_keys,
            externalized_json_pointers=sorted(omitted),
            externalized_key_policy='Original complete objects remain in the exact hash-addressed local score; no score/output quality filtering',
            score=retained)
        if projection['score']['metrics'] != original['metrics']:
            raise AssertionError('Compact score lost a metric')
        member = self.member_name(p.with_name('score_compact.json'))
        self.add_bytes(member, json_bytes(projection), dict(kind='LOSSLESS_METRIC_PROJECTION', original_path=str(p), original_sha256=digest))
        row = dict(original_score_path=str(p), original_score_sha256=digest, original_score_bytes=len(payload),
            compact_member=member, retained_full_keys=full_keys, retained_summary_keys=summary_keys,
            externalized_json_pointers=sorted(omitted))
        for key in ('request_id', 'label', 'stage', 'status', 'N', 'K', 'C_preserved_exact', 'free_face_denominator',
                'CF_tested_pair_denominator', 'FF_tested_pair_denominator', 'C_query_count', 'C_query_weight_denominator',
                'source_C_connectivity_status'):
            row[key] = original.get(key)
        for key, value in original['identity'].items():
            if not isinstance(value, (dict, list)):
                row['identity__' + key] = value
        for key, value in original['metrics'].items():
            row['metrics__' + key] = value
        for key, value in retained['topology'].items():
            row['topology__' + key] = value
        self.compact_rows[str(p)] = row

    def add_tree(self, path):
        for p in sorted(D.resolve(path).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            # CSVs and actual reviews are always kept, regardless of size/label.
            if p.suffix.lower() == '.csv':
                self.add(p)
            elif p.name == 'score.json' and 'scores' in p.parts:
                self.add_score(p)
            elif 'scores' in p.parts:
                self.externalize(p, 'PER_FACE_PAIR_QUERY_OR_OTHER_ORIGINAL_SCORER_DETAIL')
            elif p.name in ('geometry.json', 'pairs.json', 'shape_per_face.json', 'orientation.json', 'connectivity.json'):
                self.externalize(p, 'PER_FACE_PAIR_OR_POSTPROCESS_DETAIL')
            elif p.name in ('resources.json', 'observations.json'):
                self.externalize(p, 'FULL_POLLING_TELEMETRY; complete resource summaries remain included')
            elif p.suffix.lower() == '.jsonl' and ('event' in p.name.lower()) and not ('round_00' in p.parts and p.parent.name.startswith('train_')):
                self.externalize(p, 'DETAILED_FORWARD_EVENT_LEDGER; complete update/accounting tables and recovery evidence remain included')
            elif p.suffix.lower() in TEXT_SUFFIXES:
                self.add(p)

    def finalize_indexes(self):
        records = [self.compact_rows[k] for k in sorted(self.compact_rows)]
        external = [self.external[k] for k in sorted(self.external)]
        self.add_bytes('score_compact.csv', _csv_bytes(records))
        self.add_bytes('score_compact_index.json', json_bytes(dict(version=VERSION, scores=len(records),
            all_available_original_scores_included=True, no_quality_filter=True,
            metric_values_recomputed=False, rows=records)))
        self.add_bytes('external_detail_inventory.json', json_bytes(dict(version=VERSION, files=len(external),
            bytes=sum(r['bytes'] for r in external), original_files_unmodified=True, rows=external)))
        self.add_bytes('external_detail_inventory.csv', _csv_bytes(external))
        self.add_bytes('PACKAGING_CONTRACT_V2.json', json_bytes(PACKAGING_CONTRACT))


def verify_declared_inputs(final_summary, learning_summary, audit_summary):
    # Check fresh local bytes; never accept stale successful analysis summaries.
    hashes = dict(final_summary.get('protected_hashes', {}))
    for path, value in learning_summary.get('input_sources', {}).items():
        hashes[path] = value['sha256']
    for path, expected in hashes.items():
        if D.sha(path) != expected:
            raise ValueError('Completed analysis is stale: ' + path)
    if set(final_summary['tables']) != set(FINAL_TABLES):
        raise ValueError('Final per-output table inventory differs')
    for name, item in final_summary['tables'].items():
        if D.sha(item['path']) != item['sha256']:
            raise ValueError('Final analysis table changed: ' + name)
        if len(rows(item['path'])) != item['rows']:
            raise ValueError('Final CSV row count differs: ' + name)
    sources = D.resolve(audit_summary['output_dir']) / 'input_sources.json'
    if not sources.is_file():
        raise ValueError('Independent audit lacks input_sources.json')
    value = D.read(sources)
    if isinstance(value, dict) and 'sources' in value:
        value = value['sources']
    if not isinstance(value, dict):
        raise ValueError('Unrecognized audit input_sources format')
    for path, recorded in value.items():
        expected = recorded if isinstance(recorded, str) else recorded.get('sha256')
        if expected is None:
            raise ValueError('Audit input source lacks a hash: ' + path)
        if D.sha(path) != expected:
            raise ValueError('Independent audit is stale: ' + path)
    return sources


def source_diff(pack, source_paths):
    head = subprocess.run(['git', '-C', str(D.ROOT), 'rev-parse', 'HEAD'], capture_output=True, check=True).stdout.decode().strip()
    diffs, index = [], []
    for p in sorted(set(source_paths)):
        p = D.resolve(p).resolve()
        if not p.is_relative_to(D.ROOT.resolve()):
            index.append(dict(**local_paths(p), working_sha256=D.sha(p), comparison='EXTERNAL_SOURCE_NO_REPOSITORY_BASELINE'))
            continue
        rel = p.relative_to(D.ROOT.resolve()).as_posix()
        before = subprocess.run(['git', '-C', str(D.ROOT), 'show', head + ':' + rel], capture_output=True)
        if before.returncode and not (b'does not exist' in before.stderr or b'exists on disk' in before.stderr):
            raise RuntimeError('Unable to read Git HEAD source: ' + rel)
        old = before.stdout if before.returncode == 0 else b''
        current = p.read_bytes()
        diffs.extend(difflib.unified_diff(old.decode('utf-8-sig').splitlines(True), current.decode('utf-8-sig').splitlines(True),
            fromfile='a/' + rel if before.returncode == 0 else '/dev/null', tofile='b/' + rel))
        index.append(dict(repository_path=rel, working_sha256=blob_sha(current),
            HEAD_sha256=blob_sha(old) if before.returncode == 0 else None, tracked_at_HEAD=before.returncode == 0))
    pack.add_bytes('code.diff', ''.join(diffs).encode('utf-8'), dict(kind='UNIFIED_DIFF', baseline_git_HEAD=head))
    pack.add_bytes('code_inventory.json', json_bytes(dict(baseline_git_HEAD=head, files=index,
        note='Exact current sources included. New/untracked files appear as additions; original engineering before snapshots/diffs are included separately.')))


def preview_payload(image_path):
    from PIL import Image, __version__ as pillow_version
    with Image.open(image_path) as image:
        if image.mode != 'RGB' or image.getexif().get(274, 1) != 1:
            raise ValueError('Preview requires RGB with unchanged orientation; no color/orientation conversion permitted')
        source_size = list(image.size)
        size = (1440, max(1, round(image.height * 1440 / image.width)))
        profile = image.info.get('icc_profile')
        resized = image.resize(size, resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        kwargs = dict(icc_profile=profile) if profile else {}
        resized.save(buffer, format='JPEG', quality=80, subsampling=0, optimize=True, progressive=False, **kwargs)
        payload = buffer.getvalue()
    return payload, dict(source_pixels=source_size, preview_pixels=list(size),
        resize=dict(width=1440, aspect_ratio='PRESERVED_HEIGHT_ROUNDED_TO_NEAREST_PIXEL',
            resampling='LANCZOS', JPEG_quality=80, JPEG_subsampling=0, optimize=True,
            progressive=False, color_conversion=False, crop=False, view_changed=False,
            ICC_profile_preserved=True, ICC_profile_sha256=blob_sha(profile) if profile else None,
            pillow_version=pillow_version))


def gallery_catalog(pack, manifest, reviews_path):
    catalog, previews = [], []
    galleries = [(f'TRAIN_ROUND_{ri:02d}', D.OUT / f'training/round_{ri:02d}/gallery/manifest.json',
                  D.OUT / f'training/round_{ri:02d}/reviews_actual.json') for ri in range(16)]
    galleries.append(('FINAL_ANONYMOUS', D.OUT / 'evaluation/blind_gallery/manifest.json', reviews_path))
    for scope, path, review_path in galleries:
        gallery = D.read(path)
        actual = rows(review_path)
        byid = {}
        for row in actual:
            byid.setdefault(row['request_id'], []).append(row)
        for im in gallery['images']:
            image_path = D.resolve(im['image_path'])
            if D.sha(image_path) != im['image_sha256']:
                raise ValueError('Actual reviewed image changed: ' + str(image_path))
            rr = byid[im['request_id']]
            preview = None
            if scope == 'FINAL_ANONYMOUS':
                payload, preview_definition = preview_payload(image_path)
                member = 'previews/final/' + blob_sha(im['request_id'].encode())[:20] + '.jpg'
                preview = dict(request_id=im['request_id'], member=member, source_sha256=im['image_sha256'],
                    **local_paths(image_path), **preview_definition,
                    preview_sha256=blob_sha(payload), preview_bytes=len(payload),
                    used_for_original_review=False, model_calls=0)
                pack.add_bytes(member, payload, dict(kind='UNIFORM_RESIZE_OF_EXISTING_SCIENTIFIC_PANEL', **preview))
                previews.append(preview)
            catalog.append(dict(scope=scope, request_id=im['request_id'], condition_id=im['condition_id'],
                **local_paths(image_path), sha256=im['image_sha256'], bytes=image_path.stat().st_size,
                reviews=[{k: r.get(k) for k in ('label', *REVIEW_KEYS, 'reviewer_kind', 'reviewer', 'notes')} for r in rr],
                contains_NO_or_UNKNOWN=any(r.get(k) != 'YES' for r in rr for k in REVIEW_KEYS),
                archive_full_image_included=False, preview=preview))
    if len(catalog) != 512 + 36 or sum(len(r['reviews']) for r in catalog) != 512 + 108:
        raise ValueError('Full image/review catalog coverage differs')
    if len(previews) != 36:
        raise ValueError('Uniform final preview set must include all36 panels')
    pack.add_bytes('final_previews.json', json_bytes(dict(version=VERSION, count=36, all_final_requests=True,
        no_quality_selection=True, original_images_unmodified=True, rows=previews)))
    preview_lines = ['# All36 final anonymous comparison previews', '',
        'Every final request is included. Uniform resizing only; use local full originals for close inspection. Labels A/B/C are per-request anonymous aliases.', '']
    for item in previews:
        preview_lines.extend(['## ' + item['request_id'], '', '![' + item['request_id'] + '](' + item['member'] + ')', ''])
    pack.add_bytes('FINAL_PREVIEWS.md', ('\n'.join(preview_lines) + '\n').encode('utf-8'))
    pack.add_bytes('local_full_image_catalog.json', json_bytes(dict(images=catalog, image_count=len(catalog),
        all_outputs_included=True, all_unknown_and_negative_review_rows_retained=True,
        selection='No gallery image quality filtering; all original full images are externally linked by hash and all36 final panels have uniform previews')))
    lines = ['# Local full image index', '', 'All 512 training images and 36 anonymous final comparison panels are listed. Each final panel contains all three models. All NO/UNKNOWN cases remain in the review tables.', '']
    for r in catalog:
        mark = ' — contains NO/UNKNOWN/UNCERTAIN/NA' if r['contains_NO_or_UNKNOWN'] else ''
        lines.append(f"- {r['scope']} [{r['request_id']}]({r['windows_path']}){mark}")
    pack.add_bytes('LOCAL_FULL_IMAGES.md', ('\n'.join(lines) + '\n').encode('utf-8'))


def collect(ready):
    if ready['status'] != 'READY_FOR_VALIDATION':
        raise ValueError('Incomplete pilot; archive construction prohibited')
    paths = {k: D.resolve(v) for k, v in ready['paths'].items()}
    manifest = D.read(D.OUT / 'inputs/manifest.json')
    final = D.read(paths['final_summary']); learning = D.read(paths['learning_summary']); audit = D.read(paths['audit_summary'])
    audit_sources = verify_declared_inputs(final, learning, audit)
    for name, expected in LEARNING_TABLES.items():
        csv_path = paths['learning_summary'].parent / (name + '.csv')
        json_path = paths['learning_summary'].parent / (name + '.json')
        # Additional audited recovery workers must remain in the table.
        n = len(rows(csv_path)); nj = len(rows(json_path))
        if n != nj or (n != expected if name != 'physical_workers' else n < expected):
            raise ValueError('Complete learning table coverage differs: ' + name)
    pack = Pack()
    pack.add(paths['report'], 'REPORT.md')
    pack.add_tree(paths['final_summary'].parent)
    pack.add_tree(paths['learning_summary'].parent)
    pack.add_tree(paths['audit_summary'].parent)
    resource = D.read(paths['resource_summary'])
    for filename, digest in resource['files'].items():
        if D.sha(filename) != digest:
            raise ValueError('Resource summary detail changed: ' + filename)
    pack.add_tree(paths['resource_summary'].parent)
    pack.add(D.OUT / 'resource_summary/latest.json')
    pack.add(audit_sources)
    pack.add(D.OUT / 'audit/latest.json')
    for p in sorted(D.OUT.iterdir()):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            pack.add(p)
        elif p.is_dir() and p.name.startswith('engineering'):
            pack.add_tree(p)
    # JSON/CSV/JSONL metadata only: NPZ, PLY, PT and gallery images cannot enter.
    pack.add_tree(D.OUT / 'inputs')
    pack.add_tree(D.OUT / 'training')
    pack.add_tree(D.OUT / 'evaluation')
    for p in (D.OUT / 'checkpoints').glob('*/*.json'):
        pack.add(p)
    pack.add(paths['final_reviews'])
    plot = paths['learning_summary'].parent / 'learning_curves.png'
    if plot.is_file():
        pack.add(plot)
    sources = set(Path(__file__).parent.glob('*.py'))
    sources.update((D.ROOT / 'r2/src/meshflow_control').rglob('*.py'))
    definitions = {}
    score_paths = [D.OUT / f"training/round_{q['round_index']:02d}/scores/{q['request_id']}/POST/score.json" for q in manifest['training_requests']]
    score_paths += [D.OUT / 'evaluation' / model / 'scores' / q['request_id'] / stage / 'score.json'
                   for model in MODELS for q in manifest['evaluation_requests'] for stage in ('RAW', 'POST')]
    for path in score_paths:
        score = D.read(path)
        if score.get('status') != 'PASS':
            raise ValueError('Original score does not pass its computation contract: ' + str(path))
        for filename, digest in score['identity']['definition_hashes'].items():
            p = D.resolve(filename).resolve()
            if p in definitions and definitions[p] != digest:
                raise ValueError('Scorer source changed across registered outputs: ' + str(p))
            definitions[p] = digest
    sources.update(definitions)
    source_records = []
    locked = D.read(D.OUT / 'execution_source_lock.json')['hashes']
    for filename, digest in locked.items():
        pack.add(filename, expected=digest)
        sources.add(D.resolve(filename).resolve())
    for path in sorted(sources):
        member = pack.add(path, expected=definitions.get(path))
        source_records.append(dict(**local_paths(path), member=member, sha256=D.sha(path),
            role='ACTUAL_ORIGINAL_SCORER_DEFINITION' if path in definitions else 'EXECUTION_OR_ANALYSIS_SOURCE'))
    pack.add_bytes('source_hashes.json', json_bytes(source_records))
    source_diff(pack, sources)
    external = []
    for arm in ARMS:
        receipt = D.read(D.OUT / f'checkpoints/{arm}/round15.json')
        path = D.OUT / f'checkpoints/{arm}/round15.pt'
        if D.sha(path) != receipt['sha256']:
            raise ValueError('Final adapter external hash differs')
        external.append(dict(kind='FINAL_ADAPTER_AND_OPTIMIZER', arm=arm, **local_paths(path),
            bytes=path.stat().st_size, sha256=receipt['sha256'], adapters_sha256=receipt['adapters_sha256'],
            step=receipt['step'], included_in_archive=False, model_deserialized=False))
    external.append(dict(kind='PROTECTED_BASE_REFERENCE', **manifest['source_binding'], included_in_archive=False,
        digest_verification='Independent final execution audit; no base model deserialization by this builder'))
    d0_registration = D.read(D.OUT / 'training/round_00/denoise_worker/batch/registration.json')
    external.append(dict(kind='PROTECTED_OFFICIAL_D0_REFERENCE',
        path=d0_registration['checkpoint_path'], sha256=d0_registration['checkpoint_sha256'],
        included_in_archive=False, model_deserialized=False,
        digest_verification='Independent final execution audit; original resident-D0 registration'))
    pack.add_bytes('external_model_assets.json', json_bytes(external))
    gallery_catalog(pack, manifest, paths['final_reviews'])
    pack.finalize_indexes()
    pack.add_bytes('PACKAGE_README.md', ('# Terminal reward NFT pilot evidence\n\n'
        'REPORT.md is the root-authored report. This builder adds no scientific conclusions.\n\n'
        'All final per-output CSVs, original actual reviews (including NO and UNKNOWN), training rewards/streams, '
        'complete original metric dictionaries/topology summaries and accounting tables, engineering failure/recovery records, complete analyses and '
        'the independent audit are included. Current code and the actual original scorer sources are included '
        'with their hashes; code.diff compares repository files to recorded Git HEAD. Engineering before files '
        'and their original diffs retain execution history.\n\n'
        'Full score.json and repeated per-face/pair/query details, detailed event/polling telemetry, large base/D0 weights, adapters, mesh arrays, and full gallery images are external. Each external score/detail has an exact path/hash/byte inventory; compact per-score files enumerate retained and externalized JSON keys. Final adapters '
        'have separate paths and hashes. LOCAL_FULL_IMAGES.md and local_full_image_catalog.json cover every '
        'reviewed image, including negative and unknown cases. These links require the original local workspace. '
        'All36 final anonymous panels also receive uniform1440px previews, without outcome selection. Previews never substitute for full per-output evidence. PACKAGING_CONTRACT_V2.json records this packaging-only change.\n\n'
        'CHECKSUMS.json hashes every other ZIP member. The external build receipt hashes the ZIP and '
        'CHECKSUMS.json. The ZIP is reopened, every member is hashed, and CRC/member coverage is checked. '
        'Actual ZIP size and whether the10MB target was exceeded are reported; evidence is never silently dropped to meet that target. This is a compact review package, not a standalone weight/data distribution.\n').encode('utf-8'))
    return pack


def archive_bytes(pack):
    inventory = {name: dict(bytes=len(data), sha256=blob_sha(data), origin=pack.origins[name]) for name, data in sorted(pack.members.items())}
    checksum = json_bytes(dict(version=VERSION, status='COMPLETE_INPUTS_VALIDATED', members=inventory,
        self_hash_policy='CHECKSUMS.json is hashed in the external build receipt; all other members are hashed here'))
    members = dict(pack.members, **{'CHECKSUMS.json': checksum})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(members) or z.testzip() is not None:
            raise ValueError('ZIP member coverage/CRC failure')
        for name, expected in members.items():
            if blob_sha(z.read(name)) != blob_sha(expected):
                raise ValueError('ZIP member hash failure: ' + name)
    return payload, dict(zip_sha256=blob_sha(payload), zip_bytes=len(payload), members=len(members),
        target_bytes=LIMIT, target_exceeded=len(payload) >= LIMIT, evidence_omitted_to_hit_target=False,
        checksums_sha256=blob_sha(checksum), every_member_hash_verified=True, CRC_verified=True)


def run(*, build=False, learning_summary=None, report=None, reviews=None):
    if 'torch' in sys.modules:
        raise RuntimeError('Evidence pack must run in a fresh torch-free CPU process')
    D.write(DEST / 'packaging_contract_v2.json', PACKAGING_CONTRACT)
    ready = readiness(learning_summary, report, reviews)
    digest = blob_sha(json_bytes(ready))[:20]
    D.write(DEST / 'readiness' / (digest + '.json'), ready)
    if not build or ready['status'] != 'READY_FOR_VALIDATION':
        return dict(status=ready['status'], missing_count=ready['missing_count'], problems=ready['problems'],
            readiness_path=str(DEST / 'readiness' / (digest + '.json')), archive_created=False)
    pack = collect(ready)
    payload, verification = archive_bytes(pack)
    path = DEST / ('terminal_reward_nft_v1_' + verification['zip_sha256'][:16] + '.zip')
    if path.exists():
        if D.sha(path) != verification['zip_sha256']:
            raise ValueError('Existing package bytes differ')
    else:
        with path.open('xb') as f:
            f.write(payload)
    # Reopen disk artifact as well; verify against the in-memory verified ZIP.
    if D.sha(path) != verification['zip_sha256']:
        raise ValueError('Written ZIP differs from verified payload')
    with zipfile.ZipFile(path) as z:
        checksums = json.loads(z.read('CHECKSUMS.json'))
        for name, item in checksums['members'].items():
            if blob_sha(z.read(name)) != item['sha256']:
                raise ValueError('On-disk ZIP member hash differs')
    receipt = dict(version=VERSION, status='COMPLETE', archive_created=True, path=str(path),
        **verification, readiness_path=str(DEST / 'readiness' / (digest + '.json')),
        source_sha256=D.sha(__file__), models_loaded=0, model_forwards=0, torch_imported=False,
        scientific_conclusions_authored=False, per_output_csv_or_negative_cases_omitted=False)
    D.write(path.with_suffix('.receipt.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--build', action='store_true'); mode.add_argument('--check', action='store_true')
    parser.add_argument('--learning-summary'); parser.add_argument('--report'); parser.add_argument('--reviews')
    args = parser.parse_args()
    result = run(build=args.build, learning_summary=args.learning_summary, report=args.report, reviews=args.reviews)
    print(json.dumps(result, ensure_ascii=False))
    if args.build and result['status'] != 'COMPLETE':
        raise SystemExit(2)
