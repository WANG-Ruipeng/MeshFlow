"""Frozen Bench valid-interface pool and parent-balanced anisotropic stream.

CPU only. Existing source arrays and source owner labels remain read-only.
No source cleanup/retriangulation: "clean" names the eligibility filter only.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import data as old

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_clean_multipart_seam_v1/data'
SOURCE = ROOT / 'experiments/r2_bench_parts_convergence_v1/data/manifest.json'
SEED = 2026102001
STEPS = 1500
resolve, read, sha, arrays, load_case, write_new = old.resolve, old.read, old.sha, old.arrays, old.load_case, old.write_new


def scaled_case(case, scale_xyz):
    scale = np.asarray(scale_xyz, np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Expected positive finite xyz scale')
    copied = {key: value.copy() for key, value in case.items()}
    for key in ('full_target', 'pre_ot_full', 'C_vertices'):
        copied[key] *= scale
    return copied


def sample_for_training(case, row):
    from .head import features
    copied = scaled_case(case, row['scale_xyz'])
    n, k = int(copied['N']), int(copied['K'])
    z = np.random.RandomState(int(row['noise_seed'])).standard_normal((n-k, 3, 3)).astype(np.float32)
    eps = np.random.RandomState(int(row['epsilon_seed'])).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(int(row['time_seed'])).standard_normal())
    t = float(1/(1+np.exp(-normal)))
    # Keep the historical order: scale geometry BEFORE role-constrained OT;
    # independent standard Gaussian noise is never scaled.
    sample = old.supervised_sample(copied, z, eps, t, int(row['permutation_seed']), 'SEAM')
    sample['head_features'] = features(copied['C_vertices'], copied['C_faces_original'], n)
    return sample


def sample_for_probe(case, z, eps, t, permutation_seed):
    return old.supervised_sample(case, z, eps, t, permutation_seed, 'SEAM')


def make_schedule(train):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for index, row in enumerate(train):
        groups[row['parent_id']][row['part_type']].append(index)
    parents = sorted(groups)
    if len(parents) != 58 or len(train) != 195:
        raise ValueError('Frozen actual pool must be 58 parents / 195 conditions')
    rng = np.random.RandomState(SEED)
    aug = np.random.RandomState(SEED ^ 0xABCDEF)
    parts = {p: sorted(groups[p]) for p in parents}
    cursors, variants, exposure = collections.Counter(), collections.Counter(), collections.Counter()
    queue, steps = [], []
    for step in range(1, STEPS+1):
        chosen = []
        while len(chosen) < 4:
            if not queue:
                queue.extend(map(int, rng.permutation(len(parents))))
            at = next((i for i, p in enumerate(queue) if p not in chosen), None)
            if at is None:
                queue.extend(map(int, rng.permutation(len(parents))))
                continue
            chosen.append(queue.pop(at))
        selected = []
        for pi in chosen:
            parent = parents[pi]
            part = parts[parent][cursors[parent] % len(parts[parent])]
            cursors[parent] += 1
            options = groups[parent][part]
            ci = options[variants[(parent, part)] % len(options)]
            variants[(parent, part)] += 1
            selected.append(ci)
        batch = []
        for augmentation in range(2):
            for ci in selected:
                src = train[ci]
                mi = len(batch)
                seed = 10000000 + step*1000 + mi
                scale = [1., 1., 1.] if augmentation == 0 else list(map(float, aug.uniform(.92, 1.08, 3)))
                row = dict(condition_index=ci, condition_id=src['condition_id'], parent_id=src['parent_id'],
                           part_type=src['part_type'], sample_index=mi, scale_xyz=scale,
                           augmentation='CLEAN' if augmentation == 0 else 'ANISOTROPIC',
                           noise_seed=seed, epsilon_seed=seed+100000000, time_seed=seed+200000000,
                           permutation_seed=seed+300000000)
                batch.append(row)
                exposure[src['condition_id']] += 1
        steps.append(dict(step=step, samples=batch))
    return dict(schema='BENCH_CLEAN_PARENT_PART_XYZ_STREAM_V1', seed=SEED, steps=steps, batch=8,
                distinct_parents_per_update=4, record_index_domain='manifest.train',
                condition_exposure=dict(sorted(exposure.items())),
                policy='Parent shuffled cycles; parent part/variant round-robin; four distinct parents each clean and xyz U[.92,1.08]; origin unchanged; scale C/Y then OT; Gaussian untouched')


def prepare():
    from .head import features, FEATURE_SCHEMA
    OUT.mkdir(parents=True, exist_ok=True)
    source = read(SOURCE)
    if len(source['rows']) != 316 or len(source['train']) != 277 or len(source['dev']) != 39:
        raise ValueError('Original Bench complete split differs')
    train, excluded, features_all = [], [], []
    for row in source['train']:
        case = load_case(row)
        if sha(row['case_path']) != row['case_sha256']:
            raise ValueError('Source case changed')
        b = len(case['occupied_labels'])
        owners = int(np.count_nonzero(case['occupied_labels']))
        if b and owners:
            train.append(row)
            features_all.append(features(case['C_vertices'], case['C_faces_original'], int(case['N'])))
        else:
            excluded.append(dict(condition_id=row['condition_id'], parent_id=row['parent_id'],
                                 boundary_edges=b, owner_edges=owners,
                                 reason='CLOSED_C' if not b else 'OPEN_NO_SOURCE_OWNER'))
    if (len(train), len({r['parent_id'] for r in train})) != (195, 58):
        raise ValueError('Actual eligible count differs')
    # Population statistics from original (unaugmented) eligible TRAIN edges;
    # no DEV, no probes, no evaluation results and no augmented repeats.
    xx = np.concatenate(features_all).astype(np.float64)
    mean = xx.mean(0)
    std = xx.std(0, ddof=0)
    scale = np.where(std > 1e-6, std, 1.)
    scaler = dict(schema='BENCH_C_ONLY_TRAIN_STANDARDIZATION_V1', feature_schema=FEATURE_SCHEMA,
                  population='eligible195 original unaugmented TRAIN C open edges', conditions=len(train), edges=len(xx),
                  mean=mean.astype(np.float32).tolist(), population_std=std.tolist(), scale=scale.astype(np.float32).tolist(),
                  zero_scale_policy='population std<=1e-6 -> divisor1', DEV_used=False, class_labels_used=False,
                  source_manifest_sha256=sha(SOURCE))
    write_new(OUT/'head_scaler.json', scaler)
    schedule = make_schedule(train)
    write_new(OUT/'schedule.json', schedule)
    result = dict(schema='BENCH_CLEAN_MULTIPART_DATA_V1', rows=source['rows'], train=train,
                  train_original=source['train'], dev=source['dev'], excluded_train=excluded,
                  source_manifest=str(SOURCE), source_manifest_sha256=sha(SOURCE),
                  summary=dict(rows=316, original_train=277, dev=39, train=195, train_parents=58,
                               closed_train=sum(r['reason']=='CLOSED_C' for r in excluded),
                               open_no_owner_train=sum(r['reason']=='OPEN_NO_SOURCE_OWNER' for r in excluded)),
                  head_scaler=str(OUT/'head_scaler.json'), head_scaler_sha256=sha(OUT/'head_scaler.json'),
                  schedule=str(OUT/'schedule.json'), schedule_sha256=sha(OUT/'schedule.json'),
                  filtering='B_c>0 and at least one verified source owner; no repairs, no multicomponent removal',
                  no_geometry_mutation=True, source_labels_for_training_diagnostics_only=True)
    write_new(OUT/'manifest.json', result)
    write_new(OUT/'preparation_receipt.json', dict(status='COMPLETE', summary=result['summary'],
              model_forwards=0, GPU_calls=0, training_updates=0, OT_calls=0,
              feature_scaler_edges=len(xx), feature_scaler_has_no_DEV=True))
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare()['summary'], indent=2))
