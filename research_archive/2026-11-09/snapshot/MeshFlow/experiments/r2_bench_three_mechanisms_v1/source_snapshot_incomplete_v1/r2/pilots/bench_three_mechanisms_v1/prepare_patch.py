"""Actual 30-NATURAL-parent CPU patch qualification; no model/GPU calls."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import patch as P


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text: raise FileExistsError('Do not overwrite patch gate: '+str(path))
    else: path.write_text(text, encoding='utf-8')


def main():
    start = time.time()
    source = P.ROOT/'experiments/r2_bench_multilod_training_v1/data/manifest.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    rows = [r for r in manifest['train'] if r['variant'] == 'NATURAL']
    if len(rows) != 30 or len({r['parent_id'] for r in rows}) != 30:
        raise ValueError('Must gate the full fixed 30-parent natural pool')
    records = []
    for i, row in enumerate(rows):
        path = P.local_path(row['case_path'])
        if sha(path) != row['case_sha256']: raise ValueError('Original case hash changed')
        with np.load(path, allow_pickle=False) as a: case = {k:a[k].copy() for k in a.files}
        destination = P.OUT/'data/patch_sources'/f"{row['parent_id']}.json"
        plan = P.build_source_plan(case)
        plan.update(parent_id=row['parent_id'], condition_id=row['condition_id'],
                    case_path=str(path), case_sha256=row['case_sha256'])
        write(destination, plan)
        checks = []
        if plan['status'] == 'PASS':
            for j, alpha in enumerate((1., .937)):
                example = dict(row, alpha=alpha, noise_seed=2026102900+i*19+j,
                               epsilon_seed=2026103900+i*19+j, time_seed=2026104900+i*19+j,
                               permutation_seed=2026105900+i*19+j)
                sample = P.make_sample(case, example, plan=plan)
                for field in ('x0','x1','xt','u'): P.assert_shared(sample[field],sample)
                original = case['full_target'].copy(); original[...,0] *= np.float32(alpha)
                if not np.array_equal(original[sample['source_face_ids'][:,None],sample['source_corner_permutations']],sample['x1']):
                    raise AssertionError('Target face-corner roundtrip')
                checks.append(dict(alpha=alpha, N=int(sample['N']), K=int(sample['K']),
                                   M=sample['patch_count'], independent_D=int(sample['free_coordinate_count']),
                                   full_target_exact=True, state_velocity_shared=True,
                                   patch_and_singleton_OT_calls=2))
        records.append(dict(parent_id=row['parent_id'], condition_id=row['condition_id'],
                            status=plan['status'], N=plan['N'], K=plan['K'],
                            M=plan['required_patch_count'], maximum_pairs=plan['maximum_disjoint_legal_pairs'],
                            candidate_pairs=plan['legal_candidate_pairs'], plan_path=str(destination),
                            plan_sha256=sha(destination), sample_checks=checks))
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    result = dict(schema=P.VERSION, status='PASS' if all(r['status']=='PASS' for r in records) else 'BLOCKED',
                  source_manifest=str(source), source_manifest_sha256=sha(source), parents=records,
                  expected_parents=30, passed_parents=sum(r['status']=='PASS' for r in records),
                  no_parent_dropped=True, no_patch_count_reduced=True,
                  CPU_OT_samples=sum(len(r['sample_checks']) for r in records),
                  generator_forwards=0, D0_forwards=0, GPU_calls=0,
                  wall_seconds=time.time()-start)
    destination=P.OUT/'data/patch_gate.json'
    # Final receipt contains wall time and is emitted once; preserved source
    # plans are deterministic and are not overwritten by a rerun.
    if destination.exists(): raise FileExistsError(str(destination))
    write(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='parents'},ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
