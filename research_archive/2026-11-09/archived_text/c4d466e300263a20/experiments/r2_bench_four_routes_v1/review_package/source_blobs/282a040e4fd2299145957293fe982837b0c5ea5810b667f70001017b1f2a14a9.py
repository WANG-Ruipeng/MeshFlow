"""V2 lossless delivery: shared slot value strings, separate scientific identities.

This is storage deduplication only. Identical strings do not assert identical
geometries or source models. Every original branch/label/request/slot/stage and
row order is reconstructed and checked against each unchanged source CSV.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import normalize_tables as v1

DEST = v1.OUT / 'bestof/compact_deletion_v2'
VALUE_FIELDS = ('raw_slot',) + v1.SHARED + tuple(s + '_retained' for s in v1.STAGES)
REF_FIELDS = ('branch', 'request_ref', 'value_ref') + tuple(s + '_row' for s in v1.STAGES)


def expanded(name, references, values):
    lookup = {r['value_ref']: r for r in values}
    if len(lookup) != len(values):
        raise ValueError('Duplicate value reference')
    rows = []
    for reference in references:
        if reference['branch'] != name:
            continue
        value = lookup[reference['value_ref']]
        row = {k: value[k] for k in VALUE_FIELDS}
        row.update(request_ref=reference['request_ref'])
        row.update({s + '_row': reference[s + '_row'] for s in v1.STAGES})
        rows.append(row)
    return rows


def read_compact(directory=DEST):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name, identity in manifest['files'].items():
        if v1.sha(directory / name) != identity['sha256']:
            raise ValueError('Shared compact file changed: ' + name)
    _, values = v1.load_csv((directory / 'slot_values.csv').read_bytes())
    _, references = v1.load_csv((directory / 'slot_references.csv').read_bytes())
    if {r['branch'] for r in references} != set(v1.SOURCES):
        raise ValueError('Branch reference identities differ')
    return manifest, values, references


def verify(directory=DEST):
    directory = Path(directory)
    manifest, values, references = read_compact(directory)
    checks = {}
    for name, source in v1.SOURCES.items():
        metadata = manifest['sources'][name]
        source_bytes = source.read_bytes()
        if v1.sha_bytes(source_bytes) != metadata['original_sha256']:
            raise ValueError('Original deletion table changed')
        _, requests = v1.load_csv((directory / metadata['request_index_file']).read_bytes())
        checks[name], _ = v1.equality_check(metadata, requests, expanded(name, references, values), source_bytes)
    return dict(status='PASS', sources=checks, total_slot_references=len(references),
                unique_slot_value_records=len(values), shared_record_savings=len(references)-len(values),
                reference_identities_preserved=True, scientific_calls=0, models_loaded=0,
                manifest_sha256=v1.sha(directory / 'manifest.json'))


def prepare():
    if DEST.exists():
        return verify()
    original = json.loads((v1.DEST / 'manifest.json').read_text(encoding='utf-8'))
    v1.verify()
    payloads = {}
    values = []
    value_refs = {}
    references = []
    for name, metadata in original['sources'].items():
        payloads[metadata['request_index_file']] = (v1.DEST / metadata['request_index_file']).read_bytes()
        _, rows = v1.load_csv((v1.DEST / metadata['slot_file']).read_bytes())
        for row in rows:
            key = tuple(row[k] for k in VALUE_FIELDS)
            if key not in value_refs:
                value_refs[key] = str(len(values))
                values.append(dict(value_ref=value_refs[key], **dict(zip(VALUE_FIELDS, key))))
            references.append(dict(branch=name, request_ref=row['request_ref'], value_ref=value_refs[key],
                                   **{s + '_row': row[s + '_row'] for s in v1.STAGES}))
    checks = {}
    for name, metadata in original['sources'].items():
        _, requests = v1.load_csv(payloads[metadata['request_index_file']])
        checks[name], _ = v1.equality_check(metadata, requests, expanded(name, references, values), v1.SOURCES[name].read_bytes())
    payloads['slot_values.csv'] = v1.csv_bytes(('value_ref',) + VALUE_FIELDS, values)
    payloads['slot_references.csv'] = v1.csv_bytes(REF_FIELDS, references)
    sources = {name: {**metadata, 'slot_file': 'slot_values.csv', 'slot_reference_file': 'slot_references.csv'}
               for name, metadata in original['sources'].items()}
    manifest = dict(schema='four_routes_lossless_shared_deletion_delivery_v2', status='LOSSLESS_VALIDATED',
                    sources=sources, original_v1_manifest_sha256=v1.sha(v1.DEST / 'manifest.json'),
                    method='Deduplicate equal slot value strings only; branch/label/request identities and stage row order are separate references',
                    total_slot_references=len(references), unique_slot_value_records=len(values),
                    original_values='No numeric conversion; original CSV strings, column order, row order, line ending and BOM retained',
                    normalization_script=str(Path(__file__).resolve()), helper_sha256=v1.sha(__file__),
                    files={name:dict(bytes=len(data), sha256=v1.sha_bytes(data)) for name, data in payloads.items()},
                    roundtrip=checks, scientific_calls=0, models_loaded=0)
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
    if destination.exists() or not destination.resolve().is_relative_to(v1.ROOT.resolve()):
        raise ValueError('Reconstruction must target a new path in the workspace')
    manifest, values, references = read_compact()
    metadata = manifest['sources'][name]
    _, requests = v1.load_csv((DEST / metadata['request_index_file']).read_bytes())
    rows = v1.restored_rows(metadata, requests, expanded(name, references, values))
    value = v1.csv_bytes(metadata['original_columns'], rows, line_ending=metadata['line_ending'], bom=metadata['bom'])
    if v1.sha_bytes(value) != metadata['original_sha256']:
        raise ValueError('Reconstructed CSV not byte-identical to original')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(value)
    return dict(status='RECONSTRUCTED_BYTE_IDENTICAL', path=str(destination), rows=len(rows), sha256=v1.sha_bytes(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'reconstruct'))
    parser.add_argument('--source', choices=tuple(v1.SOURCES))
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
