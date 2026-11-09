"""Restore every archived source path from SHA-shared source blobs.

Run from an extracted evidence workspace:
  python experiments/r2_bench_four_routes_v1/review_package/restore_sources.py --package-root .
Existing identical files are accepted; any differing file aborts before writes.
External scorer snapshots stay under _external_sources, never outside the chosen
workspace. This restores source bytes only and runs no scientific task.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def inside(root, name):
    candidate = Path(name)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('Unsafe archive path: ' + name)
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Archive path leaves chosen workspace')
    return path


def restore(package_root, verify_only=False):
    root = Path(package_root).resolve()
    book_path = root / 'experiments/r2_bench_four_routes_v1/review_package/source_hashes.json'
    book = json.loads(book_path.read_text(encoding='utf-8'))
    payloads = {}
    existing = 0
    for name, identity in book.items():
        blob = inside(root, identity['archive_blob'])
        value = blob.read_bytes()
        if hashlib.sha256(value).hexdigest() != identity['actual_sha256']:
            raise ValueError('Source blob hash differs: ' + name)
        destination = inside(root, name)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != value:
                raise ValueError('Refuse differing existing source: ' + str(destination))
            existing += 1
        else:
            payloads[destination] = value
    if not verify_only:
        for destination, value in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(value)
    return dict(status='VERIFIED_NO_WRITE' if verify_only else 'RESTORED_BYTE_IDENTICAL',
                source_paths=len(book), existing_identical=existing,
                written_files=0 if verify_only else len(payloads), scientific_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore(args.package_root, args.verify_only), indent=2))


if __name__ == '__main__':
    main()
