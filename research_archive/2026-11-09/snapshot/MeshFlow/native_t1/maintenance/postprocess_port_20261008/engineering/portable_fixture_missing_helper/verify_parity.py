"""CPU-only port verification against existing results, never a model/scorer.

Only this maintenance directory receives files. Historical inputs and saved
outputs are read and hashed before/after, without importing historical code.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def query(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return dict(command=args, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=args, status='NA', reason=repr(error))


def resource_snapshot():
    memory = {}
    source = Path('/proc/meminfo')
    if source.exists():
        for line in source.read_text().splitlines():
            name, value = line.split(':', 1)
            if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                memory[name + '_bytes'] = int(value.strip().split()[0]) * 1024
    smi = shutil.which('nvidia-smi')
    if not smi and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        smi = '/usr/lib/wsl/lib/nvidia-smi'
    return dict(
        utc=datetime.now(timezone.utc).isoformat(), cpu_count=os.cpu_count(),
        affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
        memory=memory, threads_per_process=2, GPU_workers=0,
        gpu=query([smi, '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits']) if smi else dict(status='NA'),
        gpu_processes=query([smi, '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv,noheader']) if smi else dict(status='NA'),
        cpu_processes=query(['ps', '-eo', 'pid,comm,pcpu,pmem,args', '--sort=-pcpu']),
        scope='Read-only resource queries; GPU process memory may be unavailable under WDDM.')


def local_path(value):
    text = str(value)
    if len(text) > 2 and text[1] == ':' and os.name != 'nt':
        return Path('/mnt/' + text[0].lower() + '/' + text[2:].lstrip('\\/').replace('\\', '/'))
    return Path(text)


def portable_tests(repo, out):
    """Run the public tests with no original repo on the subprocess import path."""
    stage = out / 'portable_fixture'
    package = stage / 'native_t1'
    (package / 'tests').mkdir(parents=True, exist_ok=True)
    # Actual package init is included: the port must work without torch too.
    for relative in ('__init__.py', 'postprocessing.py', 'tests/test_postprocessing.py'):
        source = repo / 'native_t1' / relative
        shutil.copyfile(source, package / relative)
    (package / 'tests/__init__.py').write_text('', encoding='utf-8')
    harness = (
        'import importlib.abc,sys,unittest\n'
        'class Deny(importlib.abc.MetaPathFinder):\n'
        ' def find_spec(self,fullname,path=None,target=None):\n'
        '  if fullname.split(".")[0] in {"experiments","torch","trimesh","pymeshlab"}:\n'
        '   raise ImportError("Portable postprocess test forbids "+fullname)\n'
        'sys.meta_path.insert(0,Deny())\n'
        'suite=unittest.defaultTestLoader.loadTestsFromName("native_t1.tests.test_postprocessing")\n'
        'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'
        'raise SystemExit(0 if result.wasSuccessful() else 1)\n')
    env = os.environ.copy(); env['PYTHONPATH'] = str(stage)
    tick = time.perf_counter()
    result = subprocess.run([sys.executable, '-B', '-c', harness], cwd=stage,
                            env=env, capture_output=True, text=True)
    (out / 'portable_tests.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    receipt = dict(status='PASS' if result.returncode == 0 else 'FAIL',
                   returncode=result.returncode, seconds=time.perf_counter() - tick,
                   stage=str(stage), imports_forbidden=['experiments', 'torch', 'trimesh', 'pymeshlab'],
                   original_repo_not_on_subprocess_path=True,
                   no_weights_or_raw_assets_in_fixture=True,
                   source_sha256={relative: sha(package / relative) for relative in
                                  ('__init__.py', 'postprocessing.py', 'tests/test_postprocessing.py')})
    write(out / 'portable_tests.json', receipt)
    return receipt


def run(repo, legacy, out):
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() != Path(__file__).resolve().parent:
        raise ValueError('Only this dedicated maintenance directory may be written')
    write(out / 'resources_before.json', resource_snapshot())
    sources = [repo / 'native_t1/__init__.py', repo / 'native_t1/postprocessing.py',
               repo / 'native_t1/tests/test_postprocessing.py', Path(__file__).resolve()]
    source_before = {str(path): sha(path) for path in sources}
    portability = portable_tests(repo, out)
    sys.path.insert(0, str(repo))
    import numpy as np
    from native_t1.postprocessing import FIXED_CONFIG, process

    cases_path = legacy / 'cases.json'
    cases = read(cases_path)
    errors = []
    if len(cases) != 32 or len({row['id'] for row in cases}) != 32:
        raise ValueError('Expected exactly 32 distinct existing cases')
    protected = {str(cases_path): sha(cases_path)}
    rows = []
    tick = time.perf_counter()
    array_comparisons = 0
    for case in cases:
        rid = case['id']
        raw_path = local_path(case['raw_path'])
        saved_path = local_path(case['geo_path'])
        metadata_path = legacy / 'geometry' / (rid + '.json')
        for path in (raw_path, saved_path, metadata_path):
            protected[str(path)] = sha(path)
        metadata = read(metadata_path)
        with np.load(raw_path, allow_pickle=False) as data:
            raw = data['output']
        input_bytes = raw.tobytes()
        result = process(raw, int(case['K']))
        actual_arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        array_checks = {}
        with np.load(saved_path, allow_pickle=False) as old:
            keys_equal = set(old.files) == set(actual_arrays)
            for key in sorted(set(old.files) | set(actual_arrays)):
                if key not in old.files or key not in actual_arrays:
                    array_checks[key] = False
                    continue
                before, after = old[key], actual_arrays[key]
                array_checks[key] = (before.dtype == after.dtype and before.shape == after.shape
                                     and before.tobytes() == after.tobytes())
                array_comparisons += 1
        old_stats = {key: value for key, value in metadata['stats'].items() if key != 'seconds'}
        new_stats = {key: value for key, value in result['stats'].items() if key != 'seconds'}
        stats_equal = old_stats == new_stats
        records_equal = metadata['face_records'] == result['face_records']
        K = int(case['K'])
        checks = dict(array_keys_equal=keys_equal, all_arrays_bitwise_equal=all(array_checks.values()),
                      face_records_equal=records_equal, statistics_equal_ignoring_seconds=stats_equal,
                      input_unchanged=raw.tobytes() == input_bytes,
                      known_bits_equal=result['output'][:K].tobytes() == raw[:K].tobytes(),
                      indexed_roundtrip=result['vertices'][result['faces']].tobytes() == result['output'].tobytes())
        if not all(checks.values()):
            errors.append(dict(id=rid, checks=checks, array_checks=array_checks,
                               unequal_stat_keys=sorted(key for key in set(old_stats) | set(new_stats)
                                                        if old_stats.get(key) != new_stats.get(key))))
        rows.append(dict(id=rid, status='PASS' if all(checks.values()) else 'FAIL',
                         raw_path=str(raw_path), raw_sha256=protected[str(raw_path)],
                         old_GEO_C_path=str(saved_path), old_GEO_C_sha256=protected[str(saved_path)],
                         arrays_compared=len(array_checks), input_faces=len(raw),
                         output_faces=len(result['output']), deleted_free_faces=result['stats']['deleted_free_faces'],
                         **checks))
    changed = [path for path, before in protected.items() if sha(path) != before]
    source_changed = [path for path, before in source_before.items() if sha(path) != before]
    if changed: errors.append(dict(historical_files_changed=changed))
    if source_changed: errors.append(dict(verification_sources_changed=source_changed))
    if portability['status'] != 'PASS': errors.append(dict(portability='FAIL'))
    if 'torch' in sys.modules: errors.append(dict(unexpected_torch_import=True))
    with (out / 'parity_per_output.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status='PASS' if not errors else 'FAIL', cases=len(rows),
                   array_comparisons=array_comparisons, all_case_checks_pass=all(row['status'] == 'PASS' for row in rows),
                   portable_tests=portability, ignored_stat_keys=['seconds'],
                   original_files_checked=len(protected), original_files_unchanged=not changed,
                   source_sha256=source_before, source_unchanged=not source_changed,
                   historical_sha256=protected, fixed_config=FIXED_CONFIG,
                   seconds=time.perf_counter() - tick, errors=errors,
                   outputs_recomputed='CPU geometry only, for exact parity; outputs kept in memory',
                   historical_algorithm_executed=False, model_calls=0, model_loads=0,
                   training_updates=0, rescored_outputs=0, new_generator_samples=0,
                   torch_imported='torch' in sys.modules)
    write(out / 'parity_summary.json', summary)
    write(out / 'resources_after.json', resource_snapshot())
    print(json.dumps({key: summary[key] for key in ('status', 'cases', 'array_comparisons',
          'original_files_checked', 'original_files_unchanged', 'seconds', 'errors', 'model_calls')}))
    return not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('/mnt/e/MeshFlow'))
    parser.add_argument('--legacy', type=Path, default=Path('/mnt/e/MeshFlow/native_t1/runs/chair_start_postprocess_v1'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.repo, args.legacy, args.out) else 1)
