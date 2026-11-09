import collections
import json
import os
import subprocess
from pathlib import Path

BASE = Path('E:/') if os.name == 'nt' else Path('/mnt/e')
ROOTS = [BASE/name for name in ('MeshFlow', 'MeshFlow-Control', 'MeshFlow-Control-artifacts')]
SKIP = {'.git', 'venv', '.venv', 'node_modules', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.aws', '.ssh', '.codex', '.agents', 'cloud_text_archive_20261109'}
TEXT = {'.md', '.txt', '.rst', '.json', '.jsonl', '.csv', '.tsv', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log', '.out', '.err', '.py', '.ps1', '.sh', '.bat', '.cmd', '.html', '.htm', '.js', '.css', '.cpp', '.h', '.cu', '.cuh', '.c', '.hpp', '.diff', '.patch', '.lock', '.xml', '.ipynb'}
NAMES = {'Dockerfile', 'Makefile', '.gitignore', '.gitattributes', 'LICENSE', 'NOTICE', 'CMakeLists.txt', '.gitmodules'}

def files(root):
    cmd = ['rg', '--files', '--hidden', '--no-ignore']
    for name in sorted(SKIP):
        cmd += ['-g', '!' + name]
    cmd.append(str(root))
    result = subprocess.run(cmd, capture_output=True, check=True)
    for name in sorted(result.stdout.decode('utf-8').splitlines()):
        yield Path(name)

def is_text(p):
    return p.suffix.lower() in TEXT or p.name in NAMES

def main():
    rows = []
    for root in ROOTS:
        ext = collections.defaultdict(lambda: [0, 0])
        top = collections.defaultdict(lambda: [0, 0])
        large = []
        archives = []
        for p in files(root):
            if not is_text(p) and p.suffix.lower() not in {'.zip', '.gz', '.tar', '.tgz', '.7z'}:
                continue
            size = p.stat().st_size
            rel = p.relative_to(root)
            if p.suffix.lower() in {'.zip', '.gz', '.tar', '.tgz', '.7z'}:
                archives.append({'path': str(rel), 'bytes': size})
            if not is_text(p):
                continue
            ext[p.suffix.lower()][0] += 1
            ext[p.suffix.lower()][1] += size
            top[rel.parts[0]][0] += 1
            top[rel.parts[0]][1] += size
            if size > 20*1024*1024:
                large.append({'path': str(rel), 'MiB': round(size/1048576, 2)})
        result = {'root': str(root), 'files': sum(x[0] for x in ext.values()), 'MiB': round(sum(x[1] for x in ext.values())/1048576, 2), 'extensions': dict(sorted(ext.items(), key=lambda x: -x[1][1])), 'top_dirs': dict(sorted(top.items(), key=lambda x: -x[1][1])), 'large': sorted(large, key=lambda x: -x['MiB']), 'archives': archives}
        rows.append(result)
        print(json.dumps({**result, 'large': result['large'][:10], 'archives': {'count': len(archives), 'MiB': round(sum(x['bytes'] for x in archives)/1048576, 2)}}, ensure_ascii=False), flush=True)
    Path(__file__).with_name('inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
