"""Check the exact staged source tree, without the untracked local workspace."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

OUT = Path(__file__).resolve().parent
dest = OUT / 'staged_checkout'
dest.mkdir(exist_ok=False)
with tarfile.open(OUT / 'staged_tree.tar') as archive:
    for member in archive.getmembers():
        assert (dest / member.name).resolve().is_relative_to(dest.resolve())
        assert not member.issym() and not member.islnk()
    archive.extractall(dest)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(dest / 'r2/src'))
commands = [
    ['-m', 'pytest', 'r2/tests/test_seam_contract.py', '-q'],
    ['r2/docs/evidence/seam_slot_c200_v1/verify_evidence.py'],
    ['-m', 'r2.pilots.seam_slot_c200_v1.train', '--help'],
    ['-m', 'r2.pilots.seam_slot_c200_v1.generate', '--help'],
    ['-m', 'r2.pilots.seam_slot_c200_v1.postprocess', '--help'],
]
results = []
for args in commands:
    result = subprocess.run([sys.executable, '-B', *args], cwd=dest, env=env,
                            capture_output=True, text=True)
    results.append(dict(arguments=args, returncode=result.returncode,
                        stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        (OUT / 'staged_tree_check.json').write_text(json.dumps(results, indent=2))
        raise RuntimeError(results[-1])
receipt = dict(status='PASS', independent_staged_checkout=True, checks=results,
               new_model_calls=0, GPU_calls=0)
(OUT / 'staged_tree_check.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(dict(status='PASS', checks=len(results), independent_staged_checkout=True,
                     tests=results[0]['stdout'].strip(), new_model_calls=0)))
