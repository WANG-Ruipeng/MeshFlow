"""Read-only SHA256 verification. Uses only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(stream):
    h = hashlib.sha256()
    n = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return h.hexdigest(), n
        h.update(block)
        n += len(block)


def verify(target):
    target = Path(target)
    count = total = 0
    if target.is_file():
        with zipfile.ZipFile(target) as z:
            names = z.namelist()
            assert len(names) == len(set(names)), 'Duplicate archive members'
            for name in names:
                parts = PurePosixPath(name)
                assert not parts.is_absolute() and '..' not in parts.parts, name
            manifests = [n for n in names if n.endswith('/PACKAGE_MANIFEST.json')]
            assert len(manifests) == 1, 'Expected one manifest'
            mp = manifests[0]
            prefix = mp[:-len('PACKAGE_MANIFEST.json')]
            manifest = json.loads(z.read(mp))
            expected_names = {prefix + f['path'] for f in manifest['files']} | {mp}
            assert set(names) == expected_names, 'Missing or unlisted ZIP members'
            for record in manifest['files']:
                with z.open(prefix + record['path']) as stream:
                    sha, size = digest(stream)  # Also validates ZIP CRC while reading.
                assert sha == record['sha256'], record['path'] + ': SHA256 mismatch'
                assert size == record['bytes'], record['path'] + ': size mismatch'
                count += 1
                total += size
    else:
        manifest = json.loads((target / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
        base = target.resolve()
        for record in manifest['files']:
            f = (base / record['path']).resolve()
            assert f.is_relative_to(base), record['path']
            with f.open('rb') as stream:
                sha, size = digest(stream)
            assert sha == record['sha256'], record['path'] + ': SHA256 mismatch'
            assert size == record['bytes'], record['path'] + ': size mismatch'
            count += 1
            total += size
    return {'status': 'PASS', 'verified_payload_files': count, 'verified_payload_bytes': total,
            'scope': 'All manifest-listed payload files. Manifest is covered by external ZIP SHA256.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', default=str(Path(__file__).parent))
    args = parser.parse_args()
    print(json.dumps(verify(args.target), indent=2))
