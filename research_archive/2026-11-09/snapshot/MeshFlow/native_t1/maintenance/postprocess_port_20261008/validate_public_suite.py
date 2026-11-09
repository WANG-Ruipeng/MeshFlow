from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
env.pop('PYTHONPATH', None)
files = ['native_t1/__init__.py', 'native_t1/_runtime_environment.py',
         'native_t1/__main__.py', 'native_t1/runtime.py',
         'native_t1/postprocessing.py', 'native_t1/postprocess_cli.py',
         'native_t1/tests/test_postprocessing.py', 'native_t1/tests/test_postprocess_cli.py']
hashes = lambda: {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files}
before = hashes()
start = time.monotonic()
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'native_t1/tests', '-v']
with (out / 'public_cpu_tests.log').open('wb') as stream:
    result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
after = hashes()
record = dict(status='PASS' if result.returncode == 0 and before == after else 'FAIL',
              command=command, returncode=result.returncode, seconds=time.monotonic()-start,
              source_before=before, source_after=after, source_unchanged=before==after,
              cuda_visible_devices='', scope='Public CPU synthetic unit suite; no retained model weights or new generator samples')
(out / 'public_cpu_tests.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if not k.startswith('source_')},indent=2))
print('\n'.join((out / 'public_cpu_tests.log').read_text().splitlines()[-7:]))
sys.exit(0 if record['status'] == 'PASS' else 1)
