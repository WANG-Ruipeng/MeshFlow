"""Read-only check of the explicitly staged publication; writes one local receipt."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_denoise_publish_v1'

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

def sha(data):
    return hashlib.sha256(data).hexdigest()

paths = git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')
assert paths and all(p == 'README.md' or p.startswith('r2/') for p in paths)
forbidden = {'.pt', '.pth', '.ckpt', '.safetensors', '.npz', '.npy', '.ply', '.obj', '.zip', '.whl', '.png', '.jpg', '.jpeg'}
files = []
for name in paths:
    path = ROOT / name
    data = path.read_bytes()
    assert path.suffix.lower() not in forbidden, name
    assert len(data) < 2_000_000, name
    data.decode('utf-8')
    staged = git('show', ':' + name)
    assert staged == (data.replace(b'\r\n', b'\n') if name == 'README.md' else data), name
    files.append(dict(path=name, bytes=len(data), sha256=sha(staged)))

published = set(git('ls-files', '-z').decode().strip('\0').split('\0'))
links = []
for name in paths:
    if not name.endswith('.md'):
        continue
    for link in re.findall(r'\]\(([^)]+)\)', (ROOT/name).read_text(encoding='utf-8')):
        if link.startswith(('https://', 'http://', '#', 'mailto:')):
            continue
        target = link.split('#')[0].split(' "')[0].strip('<>')
        if not target:
            continue
        dest = ((ROOT/name).parent/target).resolve()
        assert dest.is_relative_to(ROOT), (name, link)
        relative = dest.relative_to(ROOT).as_posix()
        assert relative in published, (name, link, 'not staged/tracked')
        links.append(dict(source=name, target=relative))

verification = json.loads((OUT/'gpu_smoke_validation/verification.json').read_text())
assert verification['status'] == 'PASS'
for name, expected in verification['implementation_after'].items():
    assert sha((ROOT/'r2/src/meshflow_control'/name).read_bytes()) == expected, name

suite = ET.parse(OUT/'cpu_tests_pass.xml').getroot().find('testsuite')
assert int(suite.attrib['tests']) == 115
assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0

result = dict(status='PASS', file_count=len(files), bytes=sum(f['bytes'] for f in files),
              files=files, public_links_checked=len(links), links=links,
              no_weights_raw_meshes_images=True, implementation_hashes_match_real_forward=True,
              tests=115, existing_untracked_research_not_staged=True)
(OUT/'publication_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('files','links')}, indent=2))
