"""Publication checks only. No model creation, checkpoint read or GPU calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/seam_publication_20261017'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'r2/src'))
checks = []
for module in ('train', 'generate', 'postprocess'):
    cmd = [sys.executable, '-B', '-m', 'r2.pilots.seam_slot_c200_v1.' + module, '--help']
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    checks.append(dict(check='historical_help_' + module, returncode=result.returncode,
                       stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        raise RuntimeError(checks[-1])
result = subprocess.run([sys.executable, '-B', '-m', 'r2.prototypes.seam_slot_c200_v1.check_corner_contract'],
                        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
checks.append(dict(check='historical_numpy_contract', stdout=result.stdout))
installed = OUT / 'installed'
sys.path.insert(0, str(installed))
import meshflow_control.seam as seam
assert Path(seam.__file__).is_relative_to(installed)
package = Path(seam.__file__).parent
assert (package / 'README.md').is_file()
manifest = json.loads((package / 'source_manifest.json').read_text())
for item in manifest['files']:
    original = ROOT / item['source']
    packaged = package / item['package_file']
    assert original.read_bytes() == packaged.read_bytes()
    assert hashlib.sha256(packaged.read_bytes()).hexdigest() == item['source_sha256']
checks.append(dict(check='isolated_wheel_five_core_files_match', status='PASS', path=str(package)))
receipt = dict(status='PASS', checks=checks, GPU_calls=0, model_forwards=0,
               optimizer_updates=0, weights_loaded=0)
(OUT / 'code_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
