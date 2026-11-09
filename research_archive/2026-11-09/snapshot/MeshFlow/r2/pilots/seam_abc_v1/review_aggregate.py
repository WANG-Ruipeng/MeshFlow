"""Validate frozen AI_PROXY records and aggregate their unchanged labels.

No model, rendering, geometric score, or new visual judgment. Review files are
populated only by reviewers who actually opened the registered pages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/r2_seam_abc_v1'
OUT = RUN / 'evaluation'
FIELDS = ('C_adoption', 'whole_body_complete', 'support_missing',
          'fragments_or_folded_layers', 'visible_cracks', 'visible_join')
SOURCE_NAMES = ('3f', '73', '85', '8c', '2d', '610', '608')
MODELS = ['LONG', 'A_COUPLE', 'B_POOL47', 'C_SHARED_TIP']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_once_or_verify(path, contents):
    if path.exists():
        check(path.read_bytes() == contents, f'Refuse changed existing output: {path}')
        return 'VERIFIED_EXISTING'
    with path.open('xb') as handle:
        handle.write(contents)
    return 'WRITTEN_NEW'


def configuration():
    schema_path, manifest_path = OUT / 'review_schema.json', RUN / 'eval_manifest.json'
    schema, manifest = read(schema_path), read(manifest_path)
    original_schema_path = ROOT / schema['derived_from']
    check(sha(original_schema_path) == schema['derived_from_sha256'], 'Original review schema changed')
    original_schema = read(original_schema_path)
    check(all(schema['field_definitions'][f] == original_schema['field_definitions'][f] for f in FIELDS),
          'Original visual label definition changed')
    columns = schema['csv_columns']
    check(len(columns) == len(set(columns)) == 21, 'Expected frozen 21-field schema')
    check(schema['labels'] == ['YES', 'NO', 'UNCERTAIN'], 'Label contract changed')
    check(schema['model_order'] == MODELS, 'Model order changed')
    check(schema['stages'] == ['RAW', 'POST'], 'Stage contract changed')
    check('split' in columns and 'change_vs_LONG' in columns and 'change_vs_SEAM200' not in columns,
          'Expected split and LONG comparison fields')
    requests = {r['request_id']: r for r in manifest['requests']}
    check(len(requests) == len(manifest['requests']) == 28, 'Expected 28 unique frozen requests')
    check(Counter(r['split'] for r in requests.values()) == {'TRAIN': 16, 'DEV': 12}, 'Split counts changed')
    groups = {split: {r['parent_id'] for r in requests.values() if r['split'] == split} for split in ('TRAIN', 'DEV')}
    check(len(groups['TRAIN']) == 4 and len(groups['DEV']) == 3 and groups['TRAIN'].isdisjoint(groups['DEV']), 'Parent split changed')
    for parent in groups['TRAIN'] | groups['DEV']:
        q = [r for r in requests.values() if r['parent_id'] == parent]
        check(len(q) == 4 and {r['noise_index'] for r in q} == set(range(4)), 'Parent noise coverage changed')
    check(schema['source_review_files'] == list(SOURCE_NAMES), 'Source review file contract changed')
    check(schema['expected_totals'] == {'conditions': 7, 'requests': 28, 'pages': 56, 'rows': 224}, 'Review budget changed')
    return schema, requests, schema_path, manifest_path


def aggregate():
    schema, requests, schema_path, manifest_path = configuration()
    columns, labels, stages = schema['csv_columns'], schema['labels'], schema['stages']
    expected = {(r, s, m) for r in requests for s in stages for m in MODELS}
    protected = {relative(p): sha(p) for p in (schema_path, manifest_path, Path(__file__))}
    protected[schema['derived_from']] = schema['derived_from_sha256']
    definition_path = OUT / 'gallery/definition.json'
    render_definition = read(definition_path)
    check(render_definition['labels'] == MODELS, 'Rendered column order differs')
    protected[relative(definition_path)] = sha(definition_path)
    original_rows, provenance, image_cache = {}, [], {}
    for name in SOURCE_NAMES:
        path, note_path = OUT / 'reviews' / (name + '.csv'), OUT / 'reviews' / (name + '.md')
        protected[relative(path)], protected[relative(note_path)] = sha(path), sha(note_path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            check(reader.fieldnames == columns, f'Schema/order mismatch: {path}')
            local = list(reader)
        check(len(local) == 32, f'Expected 32 actual-review records: {path}')
        check(len({r['parent_id'] for r in local}) == 1, 'Mixed parents in one source review')
        check(all(r['parent_id'].startswith(name) for r in local), f'Wrong parent source file: {name}')
        for row in local:
            key = (row['request_id'], row['stage'], row['model'])
            check(key in expected and key not in original_rows, f'Unregistered/repeated identity: {key}')
            req = requests[row['request_id']]
            check(row['parent_id'] == req['parent_id'] and row['noise_index'] == str(req['noise_index']) and row['split'] == req['split'], f'Wrong parent/noise/split: {key}')
            check(row['review_status'] == 'ACTUALLY_VIEWED', f'Unviewed or inherited record: {key}')
            check(row['review_kind'] == 'AI_PROXY' and row['blinding'] == 'NONBLIND_MODEL_LABELS_VISIBLE', f'Review kind changed: {key}')
            check(row['reviewer'].startswith('AI_PROXY_'), f'Missing reviewer: {key}')
            check(all(row[f] in labels for f in FIELDS), f'Unexpected label: {key}')
            check(all(row[f].strip() for f in ('visible_observations', 'change_vs_LONG', 'uncertainty')), f'Missing written evidence: {key}')
            check(row['model'] != 'LONG' or row['change_vs_LONG'] == 'REFERENCE', f'LONG must be REFERENCE: {key}')
            image_path = OUT / 'gallery' / f"{row['request_id']}_{row['stage']}.png"
            check(row['image_path'] == relative(image_path), f'Wrong image identity: {key}')
            if image_path not in image_cache:
                image_hash = sha(image_path)
                render_path = image_path.with_suffix('.json')
                render = read(render_path)
                check(render['request_id'] == row['request_id'] and render['stage'] == row['stage'], 'Render receipt identity differs')
                check(render['image_sha256'] == image_hash and render['definition_sha256'] == protected[relative(definition_path)], 'Render receipt hash differs')
                protected[relative(image_path)] = image_hash
                protected[relative(render_path)] = sha(render_path)
                image_cache[image_path] = image_hash
            check(row['image_sha256'] == image_cache[image_path], f'Reviewed image hash differs: {key}')
            original_rows[key] = dict(row)
        provenance.append(dict(source=relative(path), sha256=sha(path), rows=len(local),
            reviewers=sorted({r['reviewer'] for r in local}), parent_id=local[0]['parent_id'], split=local[0]['split'],
            notes=relative(note_path), notes_sha256=sha(note_path)))
    check(set(original_rows) == expected, 'Missing frozen identities')
    order = {q: i for i, q in enumerate(requests)}
    rows = sorted(original_rows.values(), key=lambda r: (order[r['request_id']], stages.index(r['stage']), MODELS.index(r['model'])))
    page_groups = defaultdict(list)
    for row in rows:
        page_groups[row['image_path']].append(row)
    check(len(page_groups) == 56, 'Expected 56 actual-view page attestations')
    pages = []
    for image_path, page in page_groups.items():
        check(len(page) == 4 and {r['model'] for r in page} == set(MODELS), 'Incomplete model columns')
        pages.append(dict(request_id=page[0]['request_id'], split=page[0]['split'], stage=page[0]['stage'],
            image_path=image_path, image_sha256=page[0]['image_sha256'], models_viewed=MODELS, row_count=4,
            reviewers=sorted({r['reviewer'] for r in page}), status='ALL_FOUR_ROWS_ATTEST_ACTUALLY_VIEWED', inherited_rows=0))
    counts = []
    for split, denominator in (('ALL', 28), ('TRAIN', 16), ('DEV', 12)):
        for stage in stages:
            for model in MODELS:
                group = [r for r in rows if r['stage'] == stage and r['model'] == model and (split == 'ALL' or r['split'] == split)]
                check(len(group) == denominator, 'Model/stage/split denominator changed')
                counts.append(dict(split=split, stage=stage, model=model, denominator=len(group),
                    label_counts={f: {v: Counter(r[f] for r in group)[v] for v in labels} for f in FIELDS}))
    flags, uncertainties = [], []
    for row in rows:
        bad = [f for f in FIELDS if row[f] == ('YES' if f in ('support_missing', 'fragments_or_folded_layers', 'visible_cracks') else 'NO')]
        unsure = [f for f in FIELDS if row[f] == 'UNCERTAIN']
        if bad: flags.append(dict(flagged_fields=bad, record=row))
        if unsure: uncertainties.append(dict(uncertain_fields=unsure, record=row))
    summary = dict(schema='seam_abc_visual_review_aggregate_v1', status='COMPLETE_VALIDATED_EXISTING_REVIEWS',
        scope='Four seen TRAIN C and three held-out DEV C; four fixed noises each; four models; RAW and POST separate',
        review_kind='NONBLIND_AI_PROXY_MULTIPLE_REVIEWERS_BY_PARENT', rows=224, requests=28, parents=7, pages=56,
        actual_view_attestation='All 56 pages have four ACTUALLY_VIEWED records from their original named reviewer. Aggregation validates records and hashes, not an independent second viewing.',
        new_visual_judgments=0, new_geometry_evaluations=0, model_calls=0, inherited_records=0, UNCERTAIN_relabelled=0,
        counts=counts, page_ledger=pages, sources=provenance, explicit_unfavorable_labels=flags, uncertainty_records=uncertainties,
        flag_selection_only='Index of original positive-property NO or defect-property YES, not a new failure score or combined success criterion.',
        excluded_evidence=['previous quality_four reviews', 'antiflatten/redirection CPU reviews', 'historical SEAM_SLOT_C200 reviews', 'numeric CF/FF/I/connectivity scores'],
        limits=['Not a blind human study or an inter-rater agreement estimate.',
            'Reviewer calibration differences can contribute to descriptive counts; TRAIN and DEV are shown separately.',
            'Visible join does not certify indexed/legal connectivity. Blue C may conceal near-coplanar overlap.',
            'Three DEV conditions provide limited held-out evidence; four TRAIN conditions were seen during training.',
            'No unique-source style ranking and no new composite visual quality metric.'])
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    merged = buffer.getvalue().encode('utf-8')
    reread = list(csv.DictReader(io.StringIO(merged.decode('utf-8'))))
    check(reread == rows and all(original_rows[(r['request_id'], r['stage'], r['model'])] == r for r in reread), 'CSV roundtrip changed source cells')
    md = ['# SEAM A/B/C actual visual-review aggregation', '',
        '28 fixed requests × 4 models × RAW/POST = 224 unchanged review records from 56 pages. Each page has four ACTUALLY_VIEWED column records. This program verifies identities and hashes; it does not perform a second image review.', '',
        '[Full CSV](visual_review.csv) · [JSON and page ledger](visual_review_summary.json) · [Validation receipt](visual_review_receipt.json)', '',
        'Every table cell is **YES / NO / UNCERTAIN**. Defect-field YES and positive-property YES have different meanings. Uncertain labels remain uncertain; columns are not added into a quality score.', '']
    for split in ('ALL', 'TRAIN', 'DEV'):
        md += [f'## {split}', '']
        for fields in (FIELDS[:3], FIELDS[3:]):
            md += ['| Stage | Model | n | ' + ' | '.join(fields) + ' |', '| --- | --- | ---: | ' + ' | '.join(['---'] * len(fields)) + ' |']
            for item in counts:
                if item['split'] != split: continue
                md.append('| ' + item['stage'] + ' | ' + item['model'] + ' | ' + str(item['denominator']) + ' | ' + ' | '.join(' / '.join(str(item['label_counts'][f][v]) for v in labels) for f in fields) + ' |')
            md.append('')
    md += ['## Index of explicit adverse observations', '',
        'The following preserves original defect YES or positive-property NO labels. All uncertain records remain in the full CSV and JSON. This is an index, not an additional evaluation.', '',
        '| Request / split | Stage / model | Original labels | Original visible observations |', '| --- | --- | --- | --- |']
    for item in flags:
        r = item['record']; observation = r['visible_observations'].replace('|', '\\|').replace('\n', ' ')
        md.append(f"| {r['request_id']} / {r['split']} | {r['stage']} / {r['model']} | " + '; '.join(f'{f}={r[f]}' for f in item['flagged_fields']) + f' | {observation} |')
    if not flags: md.append('| No explicit adverse label | — | — | Uncertainty and hidden defects remain possible |')
    md += ['', '## Original reviewer notes', '']
    for source in provenance:
        name = Path(source['source']).stem
        md.append(f"- [{source['parent_id']} / {source['split']}](reviews/{name}.md): {', '.join(source['reviewers'])}; 32 original rows.")
    md += ['', '## All 56 viewed pages', '', '| Request / split | Stage | Original reviewer | Model columns | Page |', '| --- | --- | --- | ---: | --- |']
    for p in pages:
        filename = Path(p['image_path']).name
        md.append(f"| {p['request_id']} / {p['split']} | {p['stage']} | {', '.join(p['reviewers'])} | 4 | [{filename}](gallery/{filename}) |")
    md += ['', 'These are nonblind AI_PROXY observations. Source-chair style is not a unique target. Visually continuous surfaces do not establish legal topology or absence of FF/I. Prior CPU/historical reviews are excluded.', '']
    contents = {'visual_review.csv': merged, 'visual_review_summary.json': jbytes(summary),
                'visual_review_summary.md': '\n'.join(md).encode('utf-8')}
    check(all(sha(ROOT / p) == digest for p, digest in protected.items()), 'Protected input changed during aggregation')
    for name, data in contents.items(): write_once_or_verify(OUT / name, data)
    receipt = dict(schema='seam_abc_visual_review_validation_receipt_v1', status='PASS',
        script=relative(Path(__file__)), script_sha256=sha(__file__), row_count=224, page_count=56, field_count=21,
        unique_registered_identities=224, actual_view_record_count=224, actual_view_attested_pages=56,
        independent_second_viewing_by_aggregator=False, all_page_four_model_columns_complete=True,
        all_image_hashes_match_source_and_render_receipt=True, source_cell_round_trip_identical=True,
        protected_originals_unchanged=True, labels_reassigned=0, new_visual_judgments=0, new_geometry_scores=0,
        model_calls=0, excluded_prior_review_rows=True, protected_hashes=protected,
        output_sha256={name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        runtime='stdlib Python only; no numpy, torch, model, network or GPU calls',
        recovery='Rerun validates source hashes and exact deterministic output bytes; changed existing outputs are refused.')
    action = write_once_or_verify(OUT / 'visual_review_receipt.json', jbytes(receipt))
    print(json.dumps(dict(status='PASS', rows=224, pages=56, action=action, explicit_flag_rows=len(flags), uncertainty_rows=len(uncertainties))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate-config', action='store_true', help='Validate schema and request identities only; do not populate any review/output')
    args = p.parse_args()
    if args.validate_config:
        schema, requests, *_ = configuration()
        print(json.dumps(dict(status='READY_NOT_REVIEWED', fields=len(schema['csv_columns']), requests=len(requests), pages=56, rows=224, labels_populated=0)))
    else:
        aggregate()


if __name__ == '__main__':
    main()
