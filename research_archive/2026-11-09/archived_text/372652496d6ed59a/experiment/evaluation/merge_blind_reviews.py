"""Validate and merge two actual anonymous review files, without unblinding.

Each frozen assignment contains 18 request pages and all three local A/B/C
aliases. Missing inputs return WAIT and write nothing. No model, image review,
score, relabeling, private mapping or execution-source modification occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

VERSION = 'terminal_nft_public_blind_review_merge_v1'
EVALUATION = Path(__file__).resolve().parent
OUT = EVALUATION.parent
ROOT = OUT.parent.parent
ALIASES = ('A', 'B', 'C')
REVIEW_FIELDS = ('RAW_adoption', 'POST_adoption', 'POST_body_complete')
ALLOWED = {'YES', 'NO', 'UNKNOWN'}
REQUIRED = {'request_id', 'label', *REVIEW_FIELDS, 'review_status', 'reviewer_kind',
            'reviewer', 'blinded', 'image_sha256', 'notes'}


def resolve(path):
    value = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', value):
        value = '/mnt/' + value[0].lower() + value[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    result = Path(value)
    return (result if result.is_absolute() else ROOT / result).resolve()


def public_path(path):
    path = resolve(path)
    if path.is_relative_to((EVALUATION / 'private').resolve()):
        raise ValueError('Private mapping/provenance inputs are forbidden')
    return path


def sha(path):
    path = public_path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = public_path(path)
    if path.suffix != '.json':
        raise ValueError('Use JSON inputs with native boolean true, not CSV strings')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(manifest_path, assignments_path, review_paths, input_manifest_path, rubric_path):
    paths = [manifest_path, assignments_path, input_manifest_path, rubric_path, *review_paths.values()]
    paths = [public_path(p) for p in paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return dict(status='WAIT', missing=missing, files_written=0, review_labels_created=0,
                    private_mapping_accessed=False, model_calls=0), None, None
    require(set(review_paths) == {'review_a', 'review_b'}, 'Exactly review_a and review_b are required')
    require(len({public_path(p) for p in review_paths.values()}) == 2, 'Two review files must be distinct')
    inputs = {str(p): sha(p) for p in paths}
    gallery, assignments, rubric = read(manifest_path), read(assignments_path), read(rubric_path)
    require(gallery.get('status') == 'COMPLETE', 'Public gallery is not complete')
    identity = gallery['identity']
    require(identity.get('blinded') is True and identity.get('cases') == 108 and identity.get('requests') == 36
            and set(identity.get('labels', [])) == set(ALIASES), 'Public gallery must be 36 complete anonymous A/B/C pages')
    require(assignments.get('status') == 'FROZEN_BEFORE_FINAL_SAMPLING'
            and assignments.get('outputs_per_request') == 3 and assignments.get('expected_total_review_rows') == 108,
            'Frozen review assignment contract differs')
    require(assignments['manifest_sha256'] == inputs[str(public_path(input_manifest_path))], 'Assignment source manifest hash differs')
    require(set(rubric['labels']) == ALLOWED and set(rubric['required_fields']) >= REQUIRED,
            'Frozen public review rubric differs')
    assigned = assignments['assignments']
    require(set(assigned) == {'review_a', 'review_b'}, 'Unexpected assignment keys')
    for key, ids in assigned.items():
        require(len(ids) == len(set(ids)) == 18, key + ' must own exactly 18 unique requests')
    require(set(assigned['review_a']).isdisjoint(assigned['review_b']), 'Review assignments overlap')
    expected_ids = set(assigned['review_a']) | set(assigned['review_b'])
    ordered = sorted(expected_ids)
    require(assigned['review_a'] == ordered[::2] and assigned['review_b'] == ordered[1::2],
            'Assignments differ from frozen even/odd sorted-request allocation')
    images = gallery['images']
    require(len(images) == 36 and len({im['request_id'] for im in images}) == 36,
            'Public gallery request pages duplicate or are missing')
    by_id = {im['request_id']: im for im in images}
    require(set(by_id) == expected_ids, 'Gallery and assignment request sets differ')
    frozen_requests = read(input_manifest_path)['evaluation_requests']
    require(len(frozen_requests) == 36 and {q['request_id'] for q in frozen_requests} == expected_ids,
            'Public requests do not cover the registered final 36')
    image_root = public_path(manifest_path).parent / 'images'
    for rid, im in by_id.items():
        path = public_path(im['image_path'])
        require(path.is_relative_to(image_root.resolve()), 'Reviewed image is outside public blind-gallery images')
        require(set(im.get('labels', [])) == set(ALIASES), 'Page is missing local A/B/C cells: ' + rid)
        require(sha(path) == im['image_sha256'], 'Public image hash changed: ' + rid)
        inputs[str(path)] = im['image_sha256']
    rows, parts = [], []
    gallery_sha = inputs[str(public_path(manifest_path))]
    for name in ('review_a', 'review_b'):
        path = public_path(review_paths[name]); document = read(path)
        local = document['rows'] if isinstance(document, dict) else document
        require(isinstance(local, list) and len(local) == 54, name + ' must contain 18 x 3 = 54 actual review rows')
        if isinstance(document, dict) and 'gallery_manifest_sha256' in document:
            require(document['gallery_manifest_sha256'] == gallery_sha, name + ' refers to a different gallery')
        if isinstance(document, dict) and 'assignment' in document:
            require(document['assignment'] == name, 'Review file assignment identity differs')
        expected = {(rid, alias) for rid in assigned[name] for alias in ALIASES}
        seen = set()
        for i, row in enumerate(local):
            prefix = name + ' row ' + str(i) + ': '
            require(isinstance(row, dict) and REQUIRED <= set(row), prefix + 'required review fields missing')
            key = row['request_id'], row['label']
            require(key in expected and key not in seen, prefix + 'duplicate/unassigned request or alias')
            seen.add(key)
            require(row['review_status'] == 'REVIEWED', prefix + 'actual review is not complete')
            require(row['reviewer_kind'] == rubric.get('reviewer_kind', 'AI_PROXY'), prefix + 'reviewer kind differs from frozen rubric')
            require(isinstance(row['reviewer'], str) and bool(row['reviewer'].strip()), prefix + 'reviewer identity missing')
            require(row['blinded'] is True, prefix + 'blinded must be native JSON true')
            require(all(row[field] in ALLOWED for field in REVIEW_FIELDS), prefix + 'labels must be YES, NO or UNKNOWN')
            require(isinstance(row['notes'], str) and bool(row['notes'].strip()), prefix + 'concrete visual notes missing')
            im = by_id[row['request_id']]
            require(row['image_sha256'] == im['image_sha256'], prefix + 'reviewed image SHA differs')
            if 'condition_id' in row:
                require(row['condition_id'] == im['condition_id'], prefix + 'condition ID differs')
            if row.get('image_path'):
                require(public_path(row['image_path']) == public_path(im['image_path']), prefix + 'review image path differs')
            if row.get('viewed_contact_sheet_path'):
                sheet = public_path(row['viewed_contact_sheet_path'])
                require(sheet.is_relative_to(public_path(manifest_path).parent / 'contact_sheets'), prefix + 'contact sheet outside public gallery')
                require(row.get('viewed_contact_sheet_sha256') == sha(sheet), prefix + 'viewed contact sheet hash differs')
                inputs[str(sheet)] = row['viewed_contact_sheet_sha256']
            rows.append(dict(row))
        require(seen == expected, name + ' omits an assigned request/alias')
        parts.append(dict(assignment=name, path=str(path), sha256=inputs[str(path)], requests=18, review_rows=len(local),
                          reviewers=sorted({r['reviewer'] for r in local})))
    require(len(rows) == 108 and len({(r['request_id'], r['label']) for r in rows}) == 108,
            'Merged final review coverage is not exactly 108 unique outputs')
    rows.sort(key=lambda r: (r['request_id'], r['label']))
    # Preserve every submitted row, field, label and note. Only ordering and
    # the aggregate envelope are new; no anonymous alias is decoded.
    merged = dict(status='COMPLETE', version=VERSION, blinded=True, actual_review_rows=108,
        request_pages=36, gallery_manifest_sha256=gallery_sha,
        assignments_sha256=inputs[str(public_path(assignments_path))],
        rubric_sha256=inputs[str(public_path(rubric_path))], source_review_parts=parts, rows=rows,
        review_labels_created=0, review_labels_changed=0,
        observation_limit='Validates submitted actual-review records and image hashes; does not independently reconstruct image-tool observations')
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Input changed during merge validation: ' + path)
    validation = dict(status='VALIDATED', version=VERSION, assigned_requests_per_part=18, review_rows_per_part=54,
        total_review_rows=108, request_pages=36, unknown_labels=sum(r[k] == 'UNKNOWN' for r in rows for k in REVIEW_FIELDS),
        label_counts={k: dict(Counter(r[k] for r in rows)) for k in REVIEW_FIELDS}, source_hashes=inputs, source_parts=parts,
        private_mapping_accessed=False, model_calls=0, scores_read=0, review_labels_created=0, review_labels_changed=0)
    return validation, merged, inputs


def immutable_json(path, value):
    path = public_path(path)
    if path.exists():
        require(read(path) == value, 'Refusing to overwrite a different completed review or receipt: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    with path.open('xb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=str(EVALUATION / 'blind_gallery/manifest.json'))
    p.add_argument('--assignments', default=str(EVALUATION / 'review_assignments.json'))
    p.add_argument('--review-a', default=str(EVALUATION / 'blind_gallery/review_a.json'))
    p.add_argument('--review-b', default=str(EVALUATION / 'blind_gallery/review_b.json'))
    p.add_argument('--input-manifest', default=str(OUT / 'inputs/manifest.json'))
    p.add_argument('--rubric', default=str(EVALUATION / 'final_review_rubric.json'))
    p.add_argument('--check', action='store_true', help='Validate existing actual reviews without writing the merged JSON')
    args = p.parse_args()
    try:
        validation, merged, _ = validate(args.manifest, args.assignments,
            {'review_a': args.review_a, 'review_b': args.review_b}, args.input_manifest, args.rubric)
        if validation['status'] == 'WAIT' or args.check:
            print(json.dumps(validation, ensure_ascii=False, sort_keys=True)); return
        dest = EVALUATION / 'blind_gallery/reviews_actual.json'
        receipt_path = EVALUATION / 'blind_gallery/reviews_merge_receipt.json'
        immutable_json(dest, merged)
        receipt = dict(validation, status='COMPLETE', merged_path=str(dest), merged_sha256=sha(dest),
                       merger_source_path=str(Path(__file__).resolve()), merger_source_sha256=sha(__file__))
        immutable_json(receipt_path, receipt)
        print(json.dumps(dict(status='COMPLETE', output=str(dest), output_sha256=receipt['merged_sha256'],
                              receipt=str(receipt_path), rows=108, labels_created=0, private_mapping_accessed=False), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='REJECTED', error=str(exc), private_mapping_accessed=False, model_calls=0), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
