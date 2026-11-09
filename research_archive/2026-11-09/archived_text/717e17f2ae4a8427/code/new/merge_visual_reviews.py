"""Validate actual image-backed reviews; freeze only on explicit --freeze.

This CPU-only reader never opens scores, losses, checkpoints, or model code.
It preserves observations exactly and does not claim hashes prove perception:
actual viewing is the reviewers' explicit, retained attestation.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import sys

OUT = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
HIST_RAW = Path('/mnt/e/MeshFlow/experiments/r2_user_mix_from_official_v1/evaluation/outputs/R2_USER_MIX')
CODE = Path(__file__).resolve().parent
FIELDS = ('C_adoption', 'C_floating', 'C_replaced_by_other_surface', 'C_wrong_layer',
          'whole_body_complete', 'support_missing', 'fragments_or_folded_layers', 'visible_join')
MUTABLE = set(FIELDS) | {'notes', 'review_status', 'reviewed_image_sha256'}
LABELS = ['USER_SMALL', 'USER_EXPANDED_67', 'HIST_USER_MIX_FROZEN']
MAPPING = dict(zip(LABELS, 'ABC'))
EXPECTED_ROWS, EXPECTED_PAGES, EXPECTED_REQUESTS = 840, 280, 140


class InvalidReview(ValueError):
    pass


def require(value, message):
    if not value:
        raise InvalidReview(message)


def key(row):
    return row['request_id'], row['stage'], row['display_label']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8-sig')


def parse_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    rows = list(reader)
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            'Missing or duplicate CSV header')
    require(all(None not in r and None not in r.values() for r in rows), 'Malformed CSV row')
    return reader.fieldnames, rows


def unique(rows, name):
    keys = [key(r) for r in rows]
    require(len(keys) == len(set(keys)), f'Duplicate identity in {name}')
    return dict(zip(keys, rows))


def review_row(row, template):
    require(set(row) == set(template), 'Review column set differs from template')
    require(all(row[c] == template[c] for c in template if c not in MUTABLE),
            f'Immutable identity/default changed: {key(row)}')
    require(row['review_status'] == 'REVIEWED_AI_PROXY', 'Unreviewed row')
    require(row['reviewer'] == 'AI_PROXY' and row['blind'] == 'False', 'Reviewer/blinding mismatch')
    require(all(row[f] in ('YES', 'NO', 'UNCERTAIN') for f in FIELDS), 'Missing explicit observation')
    require(len(row['notes'].strip()) >= 8, 'Concrete visual note missing')


class Snapshot:
    def __init__(self):
        self.hashes = {}

    def checked(self, path):
        path = Path(path)
        require(path.is_absolute(), f'Absolute path required: {path}')
        require(path == OUT or OUT in path.parents or (HIST_RAW in path.parents and path.name == 'raw.npz') or path == CODE / 'rendering.py'
                or path == Path(__file__).resolve(), f'Unexpected input/output path: {path}')
        require(not any(p.is_symlink() for p in (path, *path.parents)), f'Symlink forbidden: {path}')
        require(path.resolve() == path, f'Noncanonical path: {path}')
        return path

    def read(self, path):
        path = self.checked(path)
        require(path.is_file(), f'Missing file: {path}')
        data = path.read_bytes()
        sha = digest(data)
        old = self.hashes.setdefault(str(path), sha)
        require(old == sha, f'Input changed during validation: {path}')
        return data

    def json(self, path):
        return json.loads(self.read(path).decode('utf-8-sig'))

    def verify_unchanged(self):
        for name, sha in self.hashes.items():
            require(digest(self.checked(name).read_bytes()) == sha, f'Input changed: {name}')


def validate():
    snap = Snapshot()
    work, gallery = OUT / 'review_work', OUT / 'evaluation/gallery'
    early_path, official_path = work / 'early_review_template.csv', gallery / 'AI_PROXY_review_template.csv'
    fields, template = parse_csv(snap.read(official_path))
    ef, early = parse_csv(snap.read(early_path))
    require(ef == fields and early == template, 'Early and official templates differ in identity/default/order')
    table = unique(template, 'official template')
    require(len(template) == EXPECTED_ROWS, 'Expected exactly 840 template rows')
    require(all(r['review_status'] == 'NOT_REVIEWED' and not r['notes'] and
                not r['reviewed_image_sha256'] and all(r[f] == 'UNKNOWN' for f in FIELDS)
                and r['reviewer'] == 'AI_PROXY' and r['blind'] == 'False' for r in template),
            'Unexpected initial template defaults')
    registration = snap.json(OUT / 'evaluation/registration.json')
    require(registration['status'] == 'FROZEN' and registration['labels'] == LABELS, 'Registration mismatch')
    requests = registration['requests']
    request_map = {r['request_id']: r for r in requests}
    require(len(requests) == len(request_map) == EXPECTED_REQUESTS, 'Expected 140 unique registered requests')
    expected = {(rid, s, d) for rid in request_map for s in ('RAW', 'POST') for d in 'ABC'}
    require(set(table) == expected, 'Template does not match all registered requests/stages/methods')
    for row in template:
        req = request_map[row['request_id']]
        for name in ('parent_id', 'condition_id', 'candidate', 'panel', 'region_type', 'tier'):
            require(row[name] == str(req[name]), f'Template/registration mismatch: {name}, {key(row)}')
        require(MAPPING[row['label']] == row['display_label'], 'Method mapping mismatch')
    mapping = snap.json(gallery / 'method_mapping.json')
    require(mapping['mapping'] == MAPPING and mapping['blind_review'] is False, 'Gallery mapping/blinding mismatch')
    assignment = snap.json(work / 'assignments.json')
    require(assignment['numeric_results_read'] is False and set(assignment['chunks']) == set('0123'),
            'Assignment provenance mismatch')
    owners = {}
    for chunk, assigned in assignment['chunks'].items():
        for req in assigned:
            rid = req['request_id']
            require(rid not in owners and rid in request_map, 'Duplicate or unregistered assigned request')
            require(all(req[k] == request_map[rid][k] for k in request_map[rid]), 'Assignment/registration differs')
            owners[rid] = chunk
            for stage in ('RAW', 'POST'):
                require(req['expected_images'][stage] == str(gallery / stage / (rid + '.jpg')),
                        'Assignment image path mismatch')
    require(set(owners) == set(request_map), 'Assignments do not cover registration')
    manifest = snap.json(gallery / 'manifest.json')
    require(manifest['status'] == 'RENDERED_NOT_REVIEWED' and manifest['no_metric_rank_read'] is True,
            'Unexpected renderer provenance')
    require(manifest['images'] == EXPECTED_PAGES and manifest['requests'] == EXPECTED_REQUESTS,
            'Incomplete render manifest')
    pages = {}
    renderer_hash = digest(snap.read(CODE / 'rendering.py'))
    registered_geometry = {}
    for stage in ('raw', 'post'):
        index = snap.json(OUT / 'evaluation' / (stage + '_index.json'))
        require(index['status'] == 'COMPLETE' and not index.get('missing', []), 'Geometry index incomplete')
        seen = set()
        for row in index['rows']:
            ik = (row['request_id'], row['label'], stage)
            require(ik not in seen and ik[0] in request_map and ik[1] in LABELS, 'Geometry index identity differs')
            seen.add(ik)
            registered_geometry[ik] = (row[stage + '_path'], row[stage + '_sha256'])
        require(len(seen) == EXPECTED_REQUESTS * 3, 'Geometry index coverage differs')
    geometry_hashes = {}
    for page in manifest['receipts']:
        ident = page['identity']
        rid, stage = ident['request_id'], ident['stage']
        pk = (rid, stage)
        require(pk not in pages and rid in request_map and stage in ('RAW', 'POST'), 'Invalid render page')
        image_path = gallery / stage / (rid + '.jpg')
        require(page['image_path'] == str(image_path), 'Render image path mismatch')
        require(snap.json(image_path.with_suffix('.json')) == page, 'Page receipt differs from manifest')
        require(digest(snap.read(image_path)) == page['image_sha256'], 'JPEG hash mismatch')
        require(ident['labels'] == LABELS and ident['renderer_sha256'] == renderer_hash,
                'Page method/renderer identity mismatch')
        require(ident['all_geometry_visible'] is True and len(ident['cameras']) == 4,
                'Page lacks registered full geometry/four views')
        expected_geometry = dict(registered_geometry[(rid, label, st)] for label in LABELS for st in ('raw', 'post'))
        require(ident['hashes'] == expected_geometry, 'Rendered source set/hash differs from registered geometry')
        for name, sha in ident['hashes'].items():
            if name not in geometry_hashes:
                geometry_hashes[name] = digest(snap.read(name))
            require(geometry_hashes[name] == sha, 'Rendered geometry source hash differs')
        pages[pk] = page
    require(len(pages) == EXPECTED_PAGES and set(pages) == {(rid, s) for rid in request_map for s in ('RAW', 'POST')},
            'Render page coverage differs')
    reviews, chunk_audit, pending = {}, [], []
    for chunk in '0123':
        receipt_path = work / f'chunk_{chunk}_receipt.json'
        receipt = snap.json(receipt_path)
        path = work / f'chunk_{chunk}.csv'
        data = snap.read(path)
        cf, rows = parse_csv(data)
        require(cf == fields and digest(data) == receipt['review_file_sha256'], 'Chunk CSV schema/hash mismatch')
        require(receipt['review_file'] == str(path) and receipt['chunk'] == int(chunk), 'Chunk identity mismatch')
        require(receipt['review_kind'] == 'AI_PROXY' and receipt['blind'] is False
                and receipt['numeric_ranking_seen_before_review'] is False, 'Chunk review provenance mismatch')
        eligible = [r for r in template if owners[r['request_id']] == chunk]
        require(receipt['expected_rows'] == len(eligible) and receipt['rows_reviewed'] == len(rows), 'Chunk row count mismatch')
        chunk_table = unique(rows, f'chunk {chunk}')
        require(all(owners.get(r['request_id']) == chunk for r in rows), 'Unassigned chunk review')
        require(rows == [chunk_table[key(r)] for r in eligible if key(r) in chunk_table], 'Chunk order differs from template')
        observed_pages = {(r['request_id'], r['stage']) for r in rows}
        require(receipt['images_actually_viewed'] == len(observed_pages), 'Chunk image attestation count mismatch')
        for row in rows:
            rk = key(row)
            require(rk in table and rk not in reviews, 'Duplicate/unregistered reviewed row')
            review_row(row, table[rk])
            require(row['reviewed_image_sha256'] == pages[(row['request_id'], row['stage'])]['image_sha256'],
                    'Observation is not tied to the actual rendered image')
            reviews[rk] = row
        if receipt['status'] == 'COMPLETE':
            require(len(rows) == len(eligible), 'Falsely complete chunk')
        else:
            require(receipt['status'] == 'IN_PROGRESS' and len(rows) < len(eligible), 'Invalid partial status')
            pending.append(chunk)
        chunk_audit.append(dict(chunk=int(chunk), receipt=receipt, receipt_sha256=snap.hashes[str(receipt_path)]))
    handoff_path = work / 'chunk_0_handoff.json'
    handoff = snap.json(handoff_path)
    require(handoff['chunk'] == 0 and handoff['review_kind'] == 'AI_PROXY' and handoff['blind'] is False
            and handoff['numeric_ranking_seen_before_review'] is False, 'Handoff provenance mismatch')
    root_ids = {x['request_id'] for x in assignment['chunks']['0'][:20]}
    root_rows = [reviews[key(r)] for r in template if r['request_id'] in root_ids and key(r) in reviews]
    require(len(root_rows) == handoff['root_rows_preserved'] == 120, 'Root pre-handoff rows absent')
    require(digest(csv_bytes(fields, root_rows)) == handoff['root_csv_sha256_at_handoff'],
            'Pre-handoff root observations changed')
    if handoff['status'] != 'COMPLETE' or not handoff.get('assignee_finished_utc'):
        pending.append('chunk_0_handoff')
    snap.read(work / 'REVIEW_RULES.md')
    for name in ('parallel_review_preparation.json', 'stdin_bom_fix.json'):
        if (work / name).exists():
            snap.read(work / name)
    snap.read(Path(__file__).resolve())
    snap.verify_unchanged()
    ordered = [reviews[key(r)] for r in template if key(r) in reviews]
    audit = dict(schema='visual_review_identity_audit_v1',
                 status='READY_TO_FREEZE' if not pending else 'PREPARED_AWAITING_COMPLETE',
                 template_all_identity_default_fields_equal=True, template_order_preserved=True,
                 observations_unchanged=True, expected_rows=EXPECTED_ROWS, reviewed_rows=len(ordered),
                 requests=EXPECTED_REQUESTS, image_pages_hash_verified=len(pages),
                 rendered_geometry_files_hash_verified=len(geometry_hashes), pending=pending,
                 review_kind='AI_PROXY', blind=False, numeric_ranking_seen_before_review=False,
                 actual_viewing_evidence='Explicit chunk reviewer attestations; hashes bind observations to images but do not prove perception',
                 chunks=chunk_audit, serial_handoff=handoff, input_sha256=snap.hashes,
                 numerical_results_read=False, model_calls=0)
    return fields, ordered, audit, snap


def freeze(fields, rows, audit, snap):
    require(not audit['pending'] and len(rows) == EXPECTED_ROWS, 'All four actual reviews and handoff must be COMPLETE')
    destinations = (OUT / 'evaluation/AI_PROXY_visual_review.csv',
                    OUT / 'evaluation/visual_review_identity_audit.json',
                    OUT / 'evaluation/gallery/output_review_receipt.json')
    for path in destinations:
        snap.checked(path)
        require(not path.exists(), f'Refusing to overwrite frozen artifact: {path}')
    snap.verify_unchanged()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data = csv_bytes(fields, rows)
    audit = dict(audit, status='PASS_FROZEN', frozen_utc=now)
    audit_data = (json.dumps(audit, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    receipt = dict(schema='actual_visual_review_freeze_v1', status='COMPLETE', review_kind='AI_PROXY',
                   blind=False, numeric_ranking_seen_before_review=False, actually_reviewed=len(rows),
                   images_actually_viewed=EXPECTED_PAGES, requests=EXPECTED_REQUESTS, expected_rows=EXPECTED_ROWS,
                   review_file=str(destinations[0]), review_file_sha256=digest(data),
                   identity_audit=str(destinations[1]), identity_audit_sha256=digest(audit_data),
                   chunk_receipts=[a['receipt'] for a in audit['chunks']], serial_handoff=audit['serial_handoff'],
                   frozen_utc=now, observations_changed=False, numerical_results_read=False, model_calls=0,
                   actual_viewing_evidence=audit['actual_viewing_evidence'])
    payloads = (data, audit_data, (json.dumps(receipt, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    # The gate receipt is created last; exclusive creation forbids silent replacement.
    for path, payload in zip(destinations, payloads):
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
    return receipt


def self_test():
    template = dict(request_id='r', stage='RAW', display_label='A', parent_id='p', reviewer='AI_PROXY',
                    blind='False', review_status='NOT_REVIEWED', reviewed_image_sha256='', notes='',
                    **dict.fromkeys(FIELDS, 'UNKNOWN'))
    valid = dict(template, review_status='REVIEWED_AI_PROXY', reviewed_image_sha256='x',
                 notes='Actually observed specific visible geometry.', **dict.fromkeys(FIELDS, 'UNCERTAIN'))
    review_row(valid, template)
    rejected = []
    for name, changed in [('identity', dict(valid, parent_id='other')), ('unknown', dict(valid, visible_join='UNKNOWN')),
                          ('blinding', dict(valid, blind='True')), ('empty_note', dict(valid, notes='')),
                          ('unreviewed', dict(valid, review_status='NOT_REVIEWED'))]:
        try:
            review_row(changed, template)
        except InvalidReview:
            rejected.append(name)
        else:
            raise InvalidReview('Self-test accepted invalid ' + name)
    try:
        unique([valid, valid], 'synthetic duplicate')
    except InvalidReview:
        rejected.append('duplicate')
    else:
        raise InvalidReview('Self-test accepted duplicate')
    cf, rows = parse_csv(csv_bytes(list(valid), [valid]))
    require(rows == [valid] and cf == list(valid), 'CSV exact observation roundtrip failed')
    return dict(status='CPU_SELF_TEST_PASS', rejected=rejected, exact_csv_roundtrip=True,
                actual_reviews_frozen=False, model_calls=0, numerical_results_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only validation; never freezes even if complete')
    mode.add_argument('--self-test', action='store_true', help='Synthetic CPU validator checks only')
    mode.add_argument('--freeze', action='store_true', help='Root-authorized final freeze after all actual reviews complete')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        fields, rows, audit, snap = validate()
        if args.freeze:
            result = freeze(fields, rows, audit, snap)
        else:
            result = {k: audit[k] for k in ('status', 'reviewed_rows', 'expected_rows', 'image_pages_hash_verified',
                                           'rendered_geometry_files_hash_verified', 'pending', 'numerical_results_read', 'model_calls')}
            result['actual_reviews_frozen'] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (InvalidReview, OSError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps(dict(status='REVIEW_VALIDATION_BLOCKED', reason=str(exc), actual_reviews_frozen=False)), file=sys.stderr)
        sys.exit(2)
