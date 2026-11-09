"""Restore exact duplicate paths after extracting the review ZIP."""
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
