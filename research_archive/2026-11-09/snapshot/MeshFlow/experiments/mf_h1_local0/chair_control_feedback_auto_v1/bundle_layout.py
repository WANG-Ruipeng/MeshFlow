"""Lossless review-package layout. Unique evidence and complete CSVs stay intact.

This operates on in-memory packaging copies only. It never changes source files,
loads a model, runs a scorer, or builds a ZIP. Exact duplicate payloads may share
one physical archive member with an explicit, hash-checked reconstruction map.
"""
from __future__ import annotations
import hashlib
import json
import lzma
import zlib


RESTORE_SOURCE = '''"""Restore exact duplicate paths after extracting the review ZIP."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path


def resolve_inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Archive path escapes extraction directory: ' + name)
    return path


def restore(root):
    root = Path(root).resolve()
    index = json.loads((root / 'ALIAS_AND_OMISSION_INDEX.json').read_text(encoding='utf-8'))
    restored = reused = 0
    jobs = [(row, True) for row in index.get('lossless_payloads', [])] + [(row, False) for row in index['exact_duplicate_aliases']]
    for row, compressed in jobs:
        source = resolve_inside(root, row['physical_archive_path'])
        target = resolve_inside(root, row['logical_archive_path'])
        data = source.read_bytes()
        if compressed:
            if hashlib.sha256(data).hexdigest() != row['compressed_sha256']:
                raise ValueError('Compressed manifest is corrupt: ' + str(source))
            data = lzma.decompress(data)
        if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Canonical member is corrupt: ' + str(source))
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise FileExistsError('Refuse to replace a different file: ' + str(target))
            reused += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        restored += 1
    return dict(restored=restored, already_identical=reused)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(restore(args.root), sort_keys=True))
'''


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _csv_must_be_physical(name):
    # Every original evaluation, parent/noise, diagnostic and attribution CSV
    # remains a physical member. Only the explicit identical proof copy may alias.
    proof_copy = name.startswith(('review/cap_control_equivalence/final/',
                                  'review/decisions/H3_control_equivalence_freeze/'))
    return name.lower().endswith('.csv') and not proof_copy


def _priority(name):
    prefixes = ('source_scoring/', 'source_code/current/',
                'review/decisions/H3_control_equivalence_freeze/',
                'review/decisions/H3_evidence_freeze/',
                'review/manifests/', 'review/comparisons/')
    for i, prefix in enumerate(prefixes):
        if name.startswith(prefix):
            return i, name
    return (20 if name.startswith('source_code/executed/') else 10), name


def size_rows(entries):
    rows = [dict(archive_path=name, source_path=value.get('source_path'), category=value['category'],
                 bytes=len(value['data']), deflate_bytes=len(zlib.compress(value['data'], 9)),
                 sha256=value['sha256']) for name, value in entries.items()]
    return sorted(rows, key=lambda row: (-row['deflate_bytes'], row['archive_path']))


def compact(entries):
    """Return layout + receipt; no logical file bytes are discarded."""
    physical = dict(entries)
    groups = {}
    for name, value in entries.items():
        assert _sha(value['data']) == value['sha256'], 'Entry hash does not match payload'
        groups.setdefault(value['sha256'], []).append(name)
    aliases = []
    for digest, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        names = sorted(names, key=_priority)
        protected = [name for name in names if _csv_must_be_physical(name) or name == 'code.diff'
                     or name.startswith(('source_code/current/', 'source_scoring/'))]
        canonical = sorted(protected or names, key=_priority)[0]
        for name in names:
            if name == canonical or name in protected:
                continue
            value = entries[name]
            assert value['data'] == entries[canonical]['data'], 'SHA collision or mixed bytes'
            aliases.append(dict(logical_archive_path=name, physical_archive_path=canonical,
                                source_path=value.get('source_path'), sha256=digest,
                                bytes=len(value['data']), category=value['category'],
                                reason='Exact byte-identical duplicate; all unique bytes are retained',
                                reconstruction='Copy the physical member bytes to the logical path'))
            del physical[name]
    aliases.sort(key=lambda row: row['logical_archive_path'])
    payloads = []
    for name, value in list(physical.items()):
        if value['category'] != 'full_gallery_manifest' or len(value['data']) < 1_000_000:
            continue
        # Full render receipts are large and repetitive. Preserve every byte in
        # a standard-library XZ payload; all actual reviews/selection JSON,
        # scientific CSVs and current/original scoring code remain plain files.
        data = lzma.compress(value['data'], preset=6)
        assert lzma.decompress(data) == value['data']
        packed_name = name + '.xz'
        assert packed_name not in physical
        payloads.append(dict(logical_archive_path=name, physical_archive_path=packed_name,
                             source_path=value.get('source_path'), sha256=value['sha256'],
                             bytes=len(value['data']), compressed_sha256=_sha(data),
                             compressed_bytes=len(data), encoding='XZ/LZMA2',
                             category=value['category'], reason='Lossless byte-preserving encoding of the complete large rendering manifest'))
        physical[packed_name] = dict(data=data, sha256=_sha(data), source_path=None,
                                     category='lossless_complete_gallery_manifest')
        del physical[name]
    payloads.sort(key=lambda row: row['logical_archive_path'])
    csv_names = sorted(name for name in entries if name.lower().endswith('.csv'))
    must_keep = [name for name in csv_names if _csv_must_be_physical(name)]
    assert all(name in physical and physical[name]['data'] == entries[name]['data'] for name in must_keep)
    index = dict(version='chair_feedback_lossless_bundle_layout_v1',
                 scope='Selected small review evidence; declared large local-only assets remain outside the ZIP.',
                 policy='All unique selected evidence is retained. Exact duplicate payload copies share a physical ZIP member; complete large rendering manifests use lossless XZ encoding with byte-hash reconstruction. No CSV rows, fields, scientific values or unique JSON evidence are omitted.',
                 logical_input_files=len(entries), physical_input_files=len(physical),
                 exact_duplicate_aliases=aliases, lossless_payloads=payloads, unique_evidence_omitted=0,
                 original_CSV_files=len(csv_names), protected_physical_CSV_files=len(must_keep),
                 proof_CSV_aliases=[row['logical_archive_path'] for row in aliases if row['logical_archive_path'].lower().endswith('.csv')],
                 complete_CSV_rows_preserved=True,
                 reconstruction_command='python RESTORE_ALIASES.py --root <extracted-directory>',
                 reconstruction_note='The optional helper decompresses complete manifest bytes and copies verified duplicate bytes inside the extracted directory. It performs no experiment, model call, scoring, network operation or overwrite of different content.',
                 excluded_large_materials='RAW, model/optimizer weights, full call/training logs and full-resolution galleries remain local; see LOCAL_MATERIALS_MANIFEST.json. These are not alias payloads.')
    for name, data, category in (
            ('ALIAS_AND_OMISSION_INDEX.json', _json(index), 'lossless_alias_and_omission_index'),
            ('RESTORE_ALIASES.py', RESTORE_SOURCE.encode('utf-8'), 'lossless_alias_restore_helper')):
        assert name not in physical
        physical[name] = dict(data=data, sha256=_sha(data), source_path=None, category=category)
    receipt = dict(index=index, before_largest=size_rows(entries)[:25], after_largest=size_rows(physical)[:25],
                   before_deflate_bytes=sum(len(zlib.compress(v['data'], 9)) for v in entries.values()),
                   after_deflate_bytes=sum(len(zlib.compress(v['data'], 9)) for v in physical.values()))
    receipt['deflate_bytes_saved'] = receipt['before_deflate_bytes'] - receipt['after_deflate_bytes']
    return physical, receipt
