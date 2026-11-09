"""Summarize frozen image observations without inventing ratings or winners.

The real entrypoint first reuses the controller's actual-review receipt gate.
It never opens numerical scores, training logs, meshes, or model weights.
--self-check exercises synthetic metadata only and does not access run assets.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'experiments/r2_user_data_scaling_32_67_v1'
GALLERY = OUT / 'evaluation/gallery'
DEST = OUT / 'review_tables'
VERSION = 'r2_scaling_32_67_image_observation_counts_v1'
LABELS = ('USER_SMALL', 'USER_EXPANDED_67', 'HIST_USER_MIX_FROZEN')
STAGES = ('RAW', 'POST')
VALUES = ('YES', 'NO', 'UNCERTAIN', 'UNKNOWN', 'NOT_REVIEWED')
VIEWED = frozenset(('REVIEWED', 'REVIEWED_AI_PROXY', 'REVIEWED_HUMAN'))
IDENTITY = ('request_id', 'parent_id', 'condition_id', 'candidate', 'panel',
            'region_type', 'tier', 'label', 'display_label', 'stage')
AUDIT = ('review_status', 'reviewer', 'blind', 'reviewed_image_sha256', 'notes')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def csvwrite(path, rows):
    if not rows: raise ValueError('Refusing an empty summary product')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def review_gate():
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from r2.pilots.user_data_scaling_32_67_v1.resume_all import reviewed_record
    return reviewed_record()


def key(row):
    return row['request_id'], row['label'], row['stage']


def reviewed(row):
    return row['review_status'] in VIEWED


def value(row, field, state_aware):
    # Unseen UNKNOWN is not confused with an UNKNOWN observation after viewing.
    # The untouched source value is also counted in a separate complete table.
    return 'NOT_REVIEWED' if state_aware and not reviewed(row) else row[field]


def slices(row):
    region, tier = row['region_type'], row['tier']
    return (('ALL', 'ALL'), ('region_type', region), ('tier', tier),
            ('condition_type', region + '_' + tier))


def validate(reg, template_header, template, header, rows):
    if reg['labels'] != list(LABELS): raise ValueError('Model registration differs')
    if len(header) != len(set(header)) or header != template_header:
        raise ValueError('Review columns must retain the complete original template')
    if not set(IDENTITY + AUDIT).issubset(header):
        raise ValueError('Original identity/audit columns are missing')
    fields = [x for x in template_header if x not in IDENTITY + AUDIT]
    if not fields: raise ValueError('Template contains no observation dimensions')
    expected = {(q['request_id'], label, stage) for q in reg['requests']
                for label in LABELS for stage in STAGES}
    if len(expected) != 840 or len(reg['requests']) != 140:
        raise ValueError('Expected the unchanged 140 requests / 840 stage identities')
    for name, source in (('template', template), ('review', rows)):
        keys = [key(x) for x in source]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(name + ' must include every identity exactly once')
        if any(None in x or any(v is None for v in x.values()) for x in source):
            raise ValueError(name + ' has malformed CSV rows')
    registered = {q['request_id']: q for q in reg['requests']}
    originals = {key(x): x for x in template}
    for row in rows:
        source = registered[row['request_id']]
        for field in ('parent_id', 'condition_id', 'candidate', 'panel', 'region_type', 'tier'):
            if str(row[field]) != str(source[field]):
                raise ValueError('Registration metadata drift: ' + field)
        if any(row[field] != originals[key(row)][field] for field in IDENTITY):
            raise ValueError('Template identity drift')
        if row['review_status'] not in VIEWED | {'NOT_REVIEWED'}:
            raise ValueError('Unexpected review status: ' + row['review_status'])
        if any(row[field] not in VALUES for field in fields):
            raise ValueError('Observation must explicitly be YES/NO/UNCERTAIN/UNKNOWN/NOT_REVIEWED')
        if not reviewed(row) and any(row[field] not in ('UNKNOWN', 'NOT_REVIEWED') for field in fields):
            raise ValueError('Unreviewed rows may not contain asserted observations')
        if reviewed(row) and (not row['notes'].strip() or not row['reviewed_image_sha256']):
            raise ValueError('Viewed rows require specific notes and an image identity')
    if {r['panel'] for r in rows} != {'NEW32', 'OLD6'}:
        raise ValueError('Both panels must remain present')
    return fields


def count_table(rows, fields, per_parent=False, state_aware=True):
    groups = collections.defaultdict(list)
    for row in rows:
        for stratum, stratum_value in slices(row):
            group = (row['panel'], row['stage'], row['label'], stratum, stratum_value)
            if per_parent: group += (row['parent_id'],)
            groups[group].append(row)
    output = []
    for group, members in sorted(groups.items()):
        columns = ('panel', 'stage', 'label', 'stratum', 'stratum_value')
        if per_parent: columns += ('parent_id',)
        base = dict(zip(columns, group))
        for field in fields:
            counts = collections.Counter(value(x, field, state_aware) for x in members)
            for category in VALUES:
                output.append(dict(base, field=field, category=category, count=counts[category],
                    registered_outputs=len(members), actually_viewed_outputs=sum(reviewed(x) for x in members),
                    unique_parents=len({x['parent_id'] for x in members}),
                    unique_conditions=len({(x['parent_id'], x['condition_id']) for x in members}),
                    value_policy='REVIEW_STATE_AWARE' if state_aware else 'ORIGINAL_TEMPLATE_COLUMN'))
    return output


def products(rows, fields):
    index = []
    indexed = {}
    for row in sorted(rows, key=key):
        ident = row['request_id'] + '|' + row['label'] + '|' + row['stage']
        entry = dict(observation_id=ident, **row)
        entry['actually_viewed'] = reviewed(row)
        entry['interpretation'] = 'AI_PROXY_IMAGE_OBSERVATION' if row['reviewer'] == 'AI_PROXY' else row['reviewer']
        index.append(entry); indexed[key(row)] = entry
    pairs = []
    for row in sorted((x for x in rows if x['label'] == LABELS[0]), key=key):
        other = indexed[(row['request_id'], LABELS[1], row['stage'])]
        small = indexed[key(row)]
        for field in fields:
            base = {x: row[x] for x in ('panel', 'stage', 'request_id', 'parent_id',
                    'condition_id', 'candidate', 'region_type', 'tier')}
            pairs.append(dict(base, field=field, small_original_value=row[field],
                expanded_original_value=other[field], small_review_state_value=value(row, field, True),
                expanded_review_state_value=value(other, field, True), small_review_status=row['review_status'],
                expanded_review_status=other['review_status'], small_notes_index=small['observation_id'],
                expanded_notes_index=other['observation_id'], comparison_policy='SIDE_BY_SIDE_NO_WINNER'))
    return {
        'classification_counts.csv': count_table(rows, fields),
        'original_value_counts.csv': count_table(rows, fields, state_aware=False),
        'classification_counts_per_parent.csv': count_table(rows, fields, per_parent=True),
        'original_value_counts_per_parent.csv': count_table(rows, fields, per_parent=True, state_aware=False),
        'expanded_vs_small_per_request.csv': pairs,
        'observations_and_notes_index.csv': index,
    }


def image_identity_check(rows, manifest):
    images = {(r['identity']['request_id'], r['identity']['stage']): r for r in manifest['receipts']}
    expected = {(r['request_id'], r['stage']) for r in rows}
    if len(images) != len(manifest['receipts']) or set(images) != expected:
        raise ValueError('Rendered image manifest does not cover the closed panel')
    for row in rows:
        image = images[(row['request_id'], row['stage'])]
        if reviewed(row) and row['reviewed_image_sha256'] != image['image_sha256']:
            raise ValueError('Observation is not bound to its rendered request/stage image')


def run():
    review = review_gate()  # First action; no actual review/template data read before this gate.
    review_path = Path(review['review_file'])
    sources = [GALLERY / 'output_review_receipt.json', review_path,
               GALLERY / 'AI_PROXY_review_template.csv', GALLERY / 'manifest.json',
               OUT / 'evaluation/registration.json']
    identity = dict(version=VERSION, code_sha256=digest(__file__), sources={str(p): digest(p) for p in sources})
    receipt = DEST / 'receipt.json'
    if receipt.exists():
        prior = read(receipt)
        if prior['identity'] != identity: raise ValueError('Frozen summary identity differs; no overwrite')
        for path, sha in prior['outputs'].items():
            if digest(path) != sha: raise ValueError('Summary output identity changed')
        return dict(status='VERIFIED_REUSE', receipt=str(receipt))
    if DEST.exists(): raise RuntimeError('Incomplete summary output exists; no silent retry/overwrite')
    header, rows = csvread(review_path)
    template_header, template = csvread(GALLERY / 'AI_PROXY_review_template.csv')
    fields = validate(read(OUT / 'evaluation/registration.json'), template_header, template, header, rows)
    image_identity_check(rows, read(GALLERY / 'manifest.json'))
    result_tables = products(rows, fields)
    DEST.mkdir(parents=True, exist_ok=False)
    for name, table in result_tables.items(): csvwrite(DEST / name, table)
    text = '\n'.join([
        '# Frozen image-observation counts', '',
        'These are descriptive counts of the saved observations, not a new metric or automatic model ranking.',
        'AI_PROXY observations are not human ratings. Reviewer identity, source notes and image hashes remain attached to every row.',
        'Actual receipt status: ' + review['status'] + '; viewed rows: ' + str(review['actually_reviewed']) + ' / 840.',
        'Numerical ranking seen before review: ' + str(review['numeric_ranking_seen_before_review']) + '.', '',
        'All template judgment dimensions are included: ' + ', '.join(fields) + '.',
        'Each count table separates NEW32/OLD6, RAW/POST, method, region, original tier and region+tier. Historical MEDIUM is not renamed HALF.',
        'Each category includes explicit zero counts. Counts are output observations, not parent-equal rates; parent-level counts are exported separately.',
        'UNCERTAIN, UNKNOWN and NOT_REVIEWED remain distinct. Review-state-aware tables mark an unviewed row NOT_REVIEWED for every field; original-value tables preserve its untouched UNKNOWN or NOT_REVIEWED values.',
        'No uncertainty is mapped to failure, no missing row is dropped, no binary success index or automatic winner is created.',
        'observations_and_notes_index.csv retains all 840 original rows and exact notes. expanded_vs_small_per_request.csv lists both values and note keys for every paired request/stage/dimension.',
        'The receipt gate and image manifest binding verify recorded identities; this utility does not itself view images or establish blinding.', '',
    ])
    (DEST / 'README.md').write_text(text, encoding='utf8')
    result = dict(status='COMPLETE_DESCRIPTIVE_OBSERVATION_COUNTS', identity=identity,
        registered_stage_rows=len(rows), actual_review_status=review['status'], actually_viewed=review['actually_reviewed'],
        observation_dimensions=fields, no_new_metric=True, no_automatic_winner=True,
        numeric_results_read=0, training_logs_read=0, model_calls=0,
        outputs={str(p): digest(p) for p in DEST.iterdir() if p.is_file()})
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8')
    return dict(status=result['status'], receipt=str(receipt))


def self_check():
    """In-memory, invented identities only; no real template or results are read."""
    fields = ['C_adoption', 'C_floating', 'C_replaced_by_other_surface', 'C_wrong_layer',
              'whole_body_complete', 'support_missing', 'fragments_or_folded_layers', 'visible_join']
    header = list(IDENTITY) + list(AUDIT[:-1]) + fields + ['notes']
    requests = []
    for panel, parents, conditions in (('NEW32', 32, 2), ('OLD6', 6, 1)):
        for parent in range(parents):
            for condition in range(conditions):
                for noise in range(2):
                    requests.append(dict(request_id=f'SYNTHETIC_{panel}_{parent}_{condition}_{noise}',
                        parent_id=f'SYNTHETIC_{panel}_{parent}', condition_id=f'SYNTHETIC_C{condition}',
                        candidate=noise, panel=panel, region_type='SEAT' if parent % 2 == 0 else 'BACK',
                        tier='MEDIUM' if panel == 'OLD6' else ('FULL' if condition == 0 else 'HALF')))
    reg = dict(labels=list(LABELS), requests=requests)
    template, rows = [], []
    for q in requests:
        for li, label in enumerate(LABELS):
            for stage in STAGES:
                row = {k: str(v) for k, v in q.items()}
                row.update(label=label, stage=stage, display_label=chr(65 + li), review_status='NOT_REVIEWED',
                    reviewer='AI_PROXY', blind='False', reviewed_image_sha256='', notes='')
                row.update({f: 'UNKNOWN' for f in fields})
                template.append(dict(row))
                if len(rows) % 4:
                    row.update(review_status='REVIEWED_AI_PROXY', reviewed_image_sha256='synthetic_only',
                               notes='Invented synthetic note; no mesh was viewed.')
                    row.update({f: VALUES[(len(rows) + j) % len(VALUES)] for j, f in enumerate(fields)})
                rows.append(row)
    assert validate(reg, header, template, header, rows) == fields
    tables = products(rows, fields)
    assert len(tables['observations_and_notes_index.csv']) == 840
    assert len(tables['expanded_vs_small_per_request.csv']) == 140 * 2 * len(fields)
    for name in ('classification_counts.csv', 'original_value_counts.csv'):
        all_rows = [r for r in tables[name] if r['stratum'] == 'ALL']
        assert sum(r['count'] for r in all_rows) == 840 * len(fields)
        assert all(sum(r['count'] for r in all_rows if r['field'] == f) == 840 for f in fields)
    aware = tables['classification_counts.csv']; original = tables['original_value_counts.csv']
    for f in fields:
        select = lambda xs, v: sum(r['count'] for r in xs if r['stratum'] == 'ALL' and r['field'] == f and r['category'] == v)
        assert select(aware, 'UNCERTAIN') == select(original, 'UNCERTAIN')
        assert select(aware, 'NOT_REVIEWED') == 210 + select(original, 'NOT_REVIEWED')
        assert select(aware, 'UNKNOWN') + 210 == select(original, 'UNKNOWN')
    assert any(r['stratum_value'] == 'MEDIUM' for r in aware)
    bad = [dict(r) for r in rows]; bad[0]['C_adoption'] = 'YES'
    try: validate(reg, header, template, header, bad)
    except ValueError: pass
    else: raise AssertionError('Unreviewed asserted judgment was accepted')
    try: validate(reg, header, template, header, rows[:-1])
    except ValueError: pass
    else: raise AssertionError('Missing identity was accepted')
    return dict(status='PASS_SYNTHETIC_METADATA_ONLY', rows=840, dimensions=len(fields),
        paired_dimension_rows=140 * 2 * len(fields), real_review_files_read=0,
        numeric_results_read=0, model_calls=0, files_written=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', help='Only synthetic in-memory metadata; no run assets')
    args = parser.parse_args()
    print(json.dumps(self_check() if args.self_check else run(), ensure_ascii=False, indent=2))
