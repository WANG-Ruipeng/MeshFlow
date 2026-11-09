"""Lossless string-level normalization of three deletion ledgers for delivery.

No geometry, model, score, or numeric parsing is used. Source CSVs stay intact.
prepare writes a new compact delivery directory only after every source table
round-trips in original row/column order. verify also reads back actual files.
reconstruct writes a requested new CSV and refuses an existing output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/r2_bench_four_routes_v1'
DEST = OUT / 'bestof/compact_deletion_v1'
SOURCES = {
    'exposure': OUT / 'exposure/results/analysis_v1/deletion_fate_per_slot.csv',
    'seam': OUT / 'seam/results/analysis_v1/deletion_fate_per_slot.csv',
    'bestof': OUT / 'bestof/results/delivery_v1/deletion_fate_per_slot.csv',
}
STAGES = ('BASE_POST', 'CLEAN_FINAL_POST')
SHARED = ('raw_area', 'pre_geo_area', 'GEO_kept', 'GEO_reason')
META = ('label', 'request_id', 'parent_id', 'report_group', 'condition_id')
INDEX_FIELDS = ('request_ref',) + META
SLOT_FIELDS = ('request_ref', 'raw_slot') + SHARED + tuple(
    field for stage in STAGES for field in (stage + '_retained', stage + '_row'))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def load_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames
    rows = list(reader)
    if not fields or len(fields) != len(set(fields)) or any(None in row or None in row.values() for row in rows):
        raise ValueError('Malformed CSV')
    return fields, rows


def csv_bytes(fields, rows, *, line_ending='\r\n', bom=False):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator=line_ending)
    writer.writeheader()
    writer.writerows(rows)
    value = buffer.getvalue().encode('utf-8')
    return (b'\xef\xbb\xbf' if bom else b'') + value


def restored_rows(metadata, requests, slots):
    refs = {r['request_ref']: r for r in requests}
    if len(refs) != len(requests):
        raise ValueError('Duplicate request reference')
    ordered = {}
    for slot in slots:
        request = refs[slot['request_ref']]
        for stage in STAGES:
            row_number = int(slot[stage + '_row'])
            if row_number in ordered:
                raise ValueError('Duplicate original row index')
            value = {k: request[k] for k in META if k in metadata['original_columns']}
            value.update(stage=stage, raw_slot=slot['raw_slot'], retained=slot[stage + '_retained'])
            value.update({k: slot[k] for k in SHARED})
            if set(value) != set(metadata['original_columns']):
                raise ValueError('Original fields not exactly preserved')
            ordered[row_number] = value
    if set(ordered) != set(range(metadata['original_rows'])):
        raise ValueError('Original row order/count incomplete')
    return [ordered[i] for i in range(len(ordered))]


def equality_check(metadata, requests, slots, source_bytes):
    original_fields, original_rows = load_csv(source_bytes)
    restored = restored_rows(metadata, requests, slots)
    if original_fields != metadata['original_columns'] or len(restored) != len(original_rows):
        raise ValueError('Original row/column count differs')
    for position, (original, rebuilt) in enumerate(zip(original_rows, restored)):
        for field in original_fields:
            if original[field] != rebuilt[field]:
                raise ValueError('String mismatch at row %s field %s' % (position, field))
    rebuilt_bytes = csv_bytes(original_fields, restored, line_ending=metadata['line_ending'], bom=metadata['bom'])
    return dict(status='PASS', rows=len(restored), fields=len(original_fields),
                compared_field_strings=len(restored) * len(original_fields),
                original_order_preserved=True, all_field_strings_equal=True,
                byte_identical=rebuilt_bytes == source_bytes,
                reconstructed_sha256=sha_bytes(rebuilt_bytes)), rebuilt_bytes


def normalize(name, source):
    source_bytes = source.read_bytes()
    fields, rows = load_csv(source_bytes)
    expected = {'label', 'stage', 'request_id', 'parent_id', 'report_group', 'raw_slot', 'retained', *SHARED}
    if set(fields) not in (expected, expected | {'condition_id'}):
        raise ValueError('Unexpected original columns: ' + name)
    metadata = dict(branch=name, original_path=str(source),
                    original_relative_path=source.relative_to(ROOT).as_posix(),
                    original_sha256=sha_bytes(source_bytes), original_bytes=len(source_bytes),
                    original_columns=fields, original_rows=len(rows),
                    line_ending='\r\n' if b'\r\n' in source_bytes else '\n',
                    bom=source_bytes.startswith(b'\xef\xbb\xbf'))
    requests = []
    request_refs = {}
    slots = {}
    seen = set()
    for position, row in enumerate(rows):
        stage = row['stage']
        if stage not in STAGES:
            raise ValueError('Unexpected stage')
        request_key = (row['label'], row['request_id'])
        request_meta = {k: row.get(k, '') for k in META}
        if request_key not in request_refs:
            reference = str(len(requests))
            request_refs[request_key] = reference
            requests.append(dict(request_ref=reference, **request_meta))
        reference = request_refs[request_key]
        if {k: requests[int(reference)][k] for k in META} != request_meta:
            raise ValueError('Request metadata varies within source')
        key = (reference, row['raw_slot'])
        if (key, stage) in seen:
            raise ValueError('Duplicate label/request/raw_slot/stage')
        seen.add((key, stage))
        common = {k: row[k] for k in SHARED}
        if key not in slots:
            slots[key] = dict(request_ref=reference, raw_slot=row['raw_slot'], **common)
        if {k: slots[key][k] for k in SHARED} != common:
            raise ValueError('Area/GEO strings differ between BASE/FINAL; cannot normalize')
        slots[key][stage + '_retained'] = row['retained']
        slots[key][stage + '_row'] = str(position)
    slot_rows = list(slots.values())
    if any(set(row) != set(SLOT_FIELDS) for row in slot_rows):
        raise ValueError('Every slot must have both registered stage records')
    check, _ = equality_check(metadata, requests, slot_rows, source_bytes)
    metadata.update(requests=len(requests), compact_slot_rows=len(slot_rows),
                    labels=list(dict.fromkeys(r['label'] for r in rows)),
                    request_index_file=name + '_request_index.csv', slot_file=name + '_slots.csv',
                    roundtrip=check)
    return metadata, {name + '_request_index.csv': csv_bytes(INDEX_FIELDS, requests),
                      name + '_slots.csv': csv_bytes(SLOT_FIELDS, slot_rows)}


def verify(directory=DEST):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    checks = {}
    for name, metadata in manifest['sources'].items():
        source = SOURCES[name]
        source_bytes = source.read_bytes()
        if sha_bytes(source_bytes) != metadata['original_sha256']:
            raise ValueError('Original source changed')
        for key in ('request_index_file', 'slot_file'):
            path = directory / metadata[key]
            if sha(path) != manifest['files'][metadata[key]]['sha256']:
                raise ValueError('Compact CSV changed')
        _, requests = load_csv((directory / metadata['request_index_file']).read_bytes())
        _, slots = load_csv((directory / metadata['slot_file']).read_bytes())
        checks[name], _ = equality_check(metadata, requests, slots, source_bytes)
    return dict(status='PASS', sources=checks, scientific_calls=0, models_loaded=0,
                source_tables_unchanged=True, manifest_sha256=sha(directory / 'manifest.json'))


def prepare():
    if DEST.exists():
        return verify()
    payloads = {}
    metadata = {}
    for name, source in SOURCES.items():
        metadata[name], files = normalize(name, source)
        payloads.update(files)
    for name, source in SOURCES.items():
        if sha(source) != metadata[name]['original_sha256']:
            raise ValueError('Source changed during normalization')
    manifest = dict(schema='four_routes_lossless_deletion_delivery_v1', status='LOSSLESS_VALIDATED',
                    sources=metadata, original_values='All cells remain literal CSV strings; no numeric conversion',
                    row_order='Explicit original zero-based row index in each stage column',
                    label_scope='Each branch and original reference/candidate label remain separate',
                    normalization_script=str(Path(__file__).resolve()), helper_sha256=sha(__file__),
                    files={name:dict(bytes=len(data), sha256=sha_bytes(data)) for name, data in payloads.items()},
                    scientific_calls=0, models_loaded=0)
    payloads['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    DEST.mkdir(parents=True)
    for name, value in payloads.items():
        with (DEST / name).open('xb') as handle:
            handle.write(value)
    result = verify()
    with (DEST / 'verification.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    return result


def reconstruct(name, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refuse to overwrite a source or existing output')
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Reconstruction output must stay in workspace')
    manifest = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    metadata = manifest['sources'][name]
    _, requests = load_csv((DEST / metadata['request_index_file']).read_bytes())
    _, slots = load_csv((DEST / metadata['slot_file']).read_bytes())
    rows = restored_rows(metadata, requests, slots)
    value = csv_bytes(metadata['original_columns'], rows, line_ending=metadata['line_ending'], bom=metadata['bom'])
    if sha_bytes(value) != metadata['roundtrip']['reconstructed_sha256']:
        raise ValueError('Reconstructed CSV differs from validated original field strings')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(value)
    return dict(status='RECONSTRUCTED', path=str(destination), sha256=sha_bytes(value), rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'reconstruct'))
    parser.add_argument('--source', choices=tuple(SOURCES))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'reconstruct':
        if not args.source or not args.output:
            parser.error('reconstruct requires --source and --output')
        result = reconstruct(args.source, args.output)
    else:
        result = prepare() if args.action == 'prepare' else verify()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
