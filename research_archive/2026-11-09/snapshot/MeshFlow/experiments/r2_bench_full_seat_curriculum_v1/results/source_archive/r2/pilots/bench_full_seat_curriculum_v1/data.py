"""CPU-only full-seat curation and the unchanged Native/owner-only data adapter.

The curation command does not import torch, generate noise, solve OT, construct a
model, or choose a training budget. Existing geometry and parent splits remain
immutable. Actual image review is required before building the training manifest.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np
from .common import ROOT, OUT, read, resolve, sha, write_new

DATA = OUT / 'data'
SOURCE = ROOT / 'experiments/r2_bench_newdata_joint_v1/data/manifest.json'
SEED = 2026102301
SCHEMA = 'BENCH_FULL_SEAT_CURRICULUM_DATA_V1'
PREPARATION_VERSION = 2
CANDIDATES = DATA / 'candidates_v2.json'
VIEWS = (('front_oblique', 22., -60.), ('rear_oblique', 22., 120.), ('top', 90., -90.))
EXCLUSIONS = {
    'd9de3b770e53a41d5b7bc62e486d578c__SEAT_SURFACE_1__V2':
        'The source extent is only the front rectangular strip of the seat, not the complete seating surface.',
    'a8e987a6a6486b962d6c8e51365a5a87__SINGLE_SLAT_TOP_FULL__REL_V1':
        'One complete central slat is coherent C, but the remaining seating slats are Y; not a complete seat.',
    '7b58a3e5d7f4dddcad1d789f3b2120d0__SINGLE_SLAT_TOP_FULL__REL_V1':
        'One central slat is not the complete seat; the same parent has the complete multi-slat upper surface candidate.',
}


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_case(row_or_path):
    return arrays(row_or_path['case_path'] if isinstance(row_or_path, dict) else row_or_path)


def _relative(path):
    return resolve(path).relative_to(ROOT).as_posix()


def _write_once(path, value):
    path = resolve(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable preparation output differs: ' + str(path))
    else:
        write_new(path, value)


def check_case(row):
    """Identity/partition checks only, not a new geometry quality scorer."""
    case, inp, paired = (arrays(row[key]) for key in ('case_path', 'input_path', 'paired_path'))
    n, k = int(case['N']), int(case['K'])
    if (n, k) != (int(row['N']), int(row['K'])) or not (128 <= n <= 256 and 2 <= k < n):
        raise ValueError('Unexpected N/K: ' + row['condition_id'])
    for name in ('C', 'Y', 'full_target', 'pre_ot_full', 'C_vertices'):
        if not np.isfinite(case[name]).all():
            raise ValueError('Nonfinite: ' + name)
    if not np.array_equal(np.sort(np.r_[case['known_ids'], case['free_ids']]), np.arange(n)):
        raise ValueError('Source face partition differs')
    if not np.array_equal(case['full_target'][case['known_ids']], case['C']):
        raise ValueError('C source identity differs')
    if not np.array_equal(case['full_target'][case['free_ids']], case['Y']):
        raise ValueError('Y source identity differs')
    if not np.array_equal(case['pre_ot_full'] * np.float32(2), case['full_target']):
        raise ValueError('Pre-OT transform differs')
    for label, other in (('input', inp), ('paired', paired)):
        for key in set(case) & set(other):
            if not np.array_equal(case[key], other[key]):
                raise ValueError(label + ' differs: ' + key)
    b = len(case['edge_records'])
    owner = np.asarray(case['occupied_labels'], bool)
    alias = np.asarray(case['source_alias_labels'], bool)
    if owner.shape != (b,) or alias.shape != (b,) or (owner & alias).any():
        raise ValueError('Source label shape/disjointness differs')
    return dict(status='PASS', N=n, K=k, boundary_edges=b, source_owner_count=int(owner.sum()),
                source_alias_count=int(alias.sum()), source_true_null_count=int((~(owner | alias)).sum()),
                exact_C_Y_partition=True, paired_input_identity=True,
                full_target_bbox=np.asarray([case['full_target'].min((0, 1)), case['full_target'].max((0, 1))]).tolist()), case


def _render(row, case, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    full, known, free = case['full_target'], case['known_ids'], case['free_ids']
    # Render world x/z/y so vertical y is visibly up; no geometric modification.
    faces = full[..., [0, 2, 1]]
    low, high = faces.min((0, 1)), faces.max((0, 1))
    center = (low + high) / 2
    radius = max(float((high - low).max()) / 2 * 1.08, .01)
    limits = np.stack([center - radius, center + radius]).tolist()
    figure = plt.figure(figsize=(13.5, 12.8), dpi=125, facecolor='white')
    for vi, (name, elev, azim) in enumerate(VIEWS):
        for col, label in enumerate(('Source M: blue C + gray Y', 'Preserved C only', 'Completion target Y only')):
            ax = figure.add_subplot(3, 3, vi * 3 + col + 1, projection='3d')
            if col == 0:
                colors = np.array(['#c9cdd3'] * len(faces)); colors[known] = '#238ad2'
                ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors='#424a54', linewidths=.32, alpha=1))
            if col == 2:
                ax.add_collection3d(Poly3DCollection(faces[free], facecolors='#c9cdd3', edgecolors='#424a54', linewidths=.32, alpha=1))
            if col == 1:
                ax.add_collection3d(Poly3DCollection(faces[known], facecolors='#238ad2', edgecolors='#083f68', linewidths=.6, alpha=1))
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
            ax.set_box_aspect((1, 1, 1), zoom=1.25); ax.set_proj_type('ortho'); ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off(); ax.set_title(label + '\n' + name, fontsize=10, pad=-10)
    figure.suptitle(row['condition_id'] + '\nN=' + str(row['N']) + '  K=' + str(row['K']) +
                   ' | same source frame and orthographic camera in every column\nBlue = unchanged observed triangles; gray = exact source complement', fontsize=11, y=.985)
    figure.subplots_adjust(left=.005, right=.995, bottom=.01, top=.92, wspace=0, hspace=.015)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Refuse overwriting inspection PNG: ' + str(target))
    figure.savefig(target, facecolor='white'); plt.close(figure)
    return dict(image_path=_relative(target), image_sha256=sha(target),
                views=[dict(name=name, elevation=elev, azimuth=azim) for name, elev, azim in VIEWS],
                projection='ORTHOGRAPHIC', plotted_axes=['source_x', 'source_z', 'source_y'],
                limits=limits, columns=['SOURCE_M_C_BLUE_Y_GRAY', 'C_BLUE', 'Y_GRAY'],
                same_C_Y_frame=True, source_case_sha256=row['case_sha256'], model_forwards=0,
                renderer_version=PREPARATION_VERSION, source_overlay_depth_sort='one collection / per-triangle')


def prepare(source=SOURCE):
    started = time.perf_counter()
    original = read(source)
    candidates = [r for r in original['train'] if r['part_type'] == 'SEAT_SURFACE']
    if len(candidates) != 25 or len({r['parent_id'] for r in candidates}) != 24:
        raise ValueError('Expected the reviewed 25 seat candidates / 24 TRAIN parents')
    candidates.sort(key=lambda row: (row['parent_id'], row['condition_id']))
    protected = {_relative(source): sha(source)}
    records = []
    for i, row in enumerate(candidates, 1):
        for key, hash_key in (('case_path', 'case_sha256'), ('input_path', 'input_sha256'),
                              ('paired_path', 'paired_sha256'), ('quality_file', 'quality_sha256'),
                              ('supervision_file', 'supervision_sha256')):
            actual = sha(row[key])
            if actual != row[hash_key]:
                raise ValueError('Source hash changed: ' + row[key])
            protected[_relative(row[key])] = actual
        checked, case = check_case(row)
        target = DATA / 'inspection_v2' / (f'{i:02d}_' + row['condition_id'] + '.png')
        receipt_path = target.with_suffix('.json')
        if receipt_path.exists():
            image_receipt = read(receipt_path)
            if sha(target) != image_receipt['image_sha256'] or row['case_sha256'] != image_receipt['source_case_sha256']:
                raise ValueError('Existing inspection evidence changed')
        else:
            image_receipt = _render(row, case, target)
            write_new(receipt_path, image_receipt)
        reason = EXCLUSIONS.get(row['condition_id'])
        records.append(dict(index=i, condition_id=row['condition_id'], parent_id=row['parent_id'],
            N=row['N'], K=row['K'], original_extent_description=row.get('extent_description'),
            original_review=row.get('review'), original_source_semantics=row.get('source_candidate_semantics'),
            original_row=row, input_validation=checked, inspection=image_receipt,
            metadata_recommendation='REJECT_PARTIAL_SEAT' if reason else 'ELIGIBLE_PENDING_ACTUAL_VIEW',
            metadata_reason=reason or 'Source description identifies the complete real seating upper surface; curved, L-shaped and multi-piece seats remain eligible.',
            actual_review_status='PENDING'))
        print(f'[{i:02d}/25] {row["condition_id"]}', flush=True)
    ledger = dict(schema=SCHEMA + '_CANDIDATES', source_manifest=_relative(source),
        source_manifest_sha256=sha(source), scope='All 25 SEAT_SURFACE candidates in the trusted NEWDATA train pool; no candidate created or re-sampled.',
        selection_policy='One most complete real seating surface per existing TRAIN parent; K>=2. Complete curved/L-shaped/multi-piece seats eligible; single slat/front strips excluded. No model-output-based selection.',
        original_status_note='Inherited geometry/production statuses describe prior frozen stages; this task creates its own curation decision without changing old records.',
        protected_inputs=protected, candidates=records, required_actual_review=True)
    _write_once(CANDIDATES, ledger)
    csvpath = DATA / 'candidates_v2.csv'
    if not csvpath.exists():
        with csvpath.open('x', encoding='utf-8-sig', newline='') as f:
            columns = ['index', 'condition_id', 'parent_id', 'N', 'K', 'metadata_recommendation', 'metadata_reason', 'original_extent_description', 'image_path', 'image_sha256']
            w = csv.DictWriter(f, fieldnames=columns); w.writeheader()
            for r in records:
                w.writerow({key: r.get(key, r['inspection'].get(key)) for key in columns})
    indexpath = DATA / 'inspection_v2' / 'index.html'
    if not indexpath.exists():
        blocks = ['<!doctype html><meta charset="utf-8"><title>Bench full-seat candidates</title><style>body{font-family:sans-serif;max-width:1600px;margin:auto}img{width:100%}article{border-top:2px solid #555;margin:2em 0}p{line-height:1.5}</style><h1>All 25 original seat candidates: C / Y inspection</h1><p>Blue C, gray Y. Every page uses the same source frame and cameras across columns. Candidate selection precedes training and does not use model outputs.</p>']
        for r in records:
            image_name = Path(r['inspection']['image_path']).name
            blocks.append('<article><h2>' + str(r['index']) + '. ' + html.escape(r['condition_id']) + '</h2><p>' + html.escape(r['original_extent_description'] or '') + '</p><p>' + html.escape(r['metadata_recommendation'] + ': ' + r['metadata_reason']) + '</p><a href="' + image_name + '"><img loading="lazy" src="' + image_name + '"></a></article>')
        indexpath.write_text('\n'.join(blocks), encoding='utf-8')
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError('Protected source changed during CPU preparation')
    if not (DATA / 'preparation_receipt_v2.json').exists():
        write_new(DATA / 'preparation_receipt_v2.json', dict(status='COMPLETE', candidates=25, candidate_parents=24,
            inspection_pngs=25, manifest_built=False, actual_review='PENDING',
            code_sha256=sha(__file__), candidates_sha256=sha(CANDIDATES),
            wall_seconds=time.perf_counter()-started, numeric_threads=2, training_updates=0,
            model_forwards=0, head_forwards=0, GPU_jobs=0, OT_calls=0, noise_arrays_generated=0,
            source_geometry_modified=False, protected_input_hashes_unchanged=True))
    return ledger


def build(source=SOURCE, review_path=None):
    """Freeze a one-C-per-parent manifest only after recorded actual image review."""
    review_path = resolve(review_path or DATA / 'actual_review.json')
    ledger, reviewed, original = read(CANDIDATES), read(review_path), read(source)
    if sha(source) != ledger['source_manifest_sha256']:
        raise ValueError('Candidate source manifest changed')
    review_rows = reviewed['reviews']
    reviews = {r['condition_id']: r for r in review_rows}
    if len(reviews) != len(review_rows) or set(reviews) != {r['condition_id'] for r in ledger['candidates']}:
        raise ValueError('Exactly one actual review per candidate is required')
    train, rejected = [], []
    for item in ledger['candidates']:
        r = reviews[item['condition_id']]
        image = item['inspection']
        if r.get('actually_viewed') is not True or r['image_sha256'] != image['image_sha256'] or sha(image['image_path']) != image['image_sha256']:
            raise ValueError('Review lacks verified actual image identity')
        if not r.get('observations') or r['decision'] not in ('ELIGIBLE_FULL_SEAT', 'REJECT_PARTIAL_SEAT', 'REJECT_UNCERTAIN'):
            raise ValueError('Review decision/reason missing')
        row = dict(item['original_row'], curriculum_task='COMPLETE_REAL_SEATING_SURFACE',
                   curriculum_decision=r['decision'], curriculum_review=r,
                   curriculum_inspection=image)
        (train if r['decision'] == 'ELIGIBLE_FULL_SEAT' else rejected).append(row)
    train.sort(key=lambda r: r['parent_id'])
    parents = [r['parent_id'] for r in train]
    dev = original['dev']
    if len(train) < 4 or len(set(parents)) != len(parents) or set(parents) & {r['parent_id'] for r in dev}:
        raise ValueError('One C per original TRAIN parent and no DEV parent overlap required')
    protected = dict(ledger['protected_inputs'])
    protected[_relative(review_path)] = sha(review_path)
    protected[_relative(CANDIDATES)] = sha(CANDIDATES)
    for p, h in protected.items():
        if sha(p) != h:
            raise ValueError('Protected input changed: ' + p)
    manifest = dict(schema=SCHEMA, train=train, dev=dev, rejected_candidates=rejected,
        rows=train + dev, train_parents=parents, actual_data_ready=True,
        source_manifest=_relative(source), source_manifest_sha256=sha(source),
        candidate_ledger=_relative(CANDIDATES), actual_review=_relative(review_path),
        protected_inputs=protected, protected_hashes=protected,
        summary=dict(train_conditions=len(train), train_parents=len(parents),
                     rejected_candidates=len(rejected), inherited_dev_conditions=len(dev)),
        split_policy='Retain all prior parent assignments. No re-split, DEV admission, mesh repair or re-sampling.',
        model_input_policy='C/N only. Part names, source targets and owner/alias labels are curation/supervision metadata only.',
        source_target_role='Training supervision and source-reconstruction diagnostic; not the only valid open-ended completion.',
        augmentation_policy='Four distinct parents per batch, each identity and x-width U[.9,1.1], batch8; no shift or yz scaling.',
        training_budget_defined=False, training_updates=0, model_forwards=0)
    _write_once(DATA / 'manifest.json', manifest)
    _write_once(DATA / 'cpu_gate.json', dict(schema=SCHEMA + '_CPU_GATE', status='PASS',
        candidate_array_checks=len(ledger['candidates']), source_identity_and_partition='PASS',
        admitted_conditions=len(train), admitted_parents=len(parents), rejected_conditions=len(rejected),
        one_C_per_parent=True, min_K=min(r['K'] for r in train), prior_parent_split_preserved=True,
        actual_review_sha256=sha(review_path), manifest_sha256=sha(DATA / 'manifest.json'),
        source_manifest_sha256=sha(source), protected_hashes_unchanged=True,
        geometry_rebuilt=False, geometry_repaired=False, geometry_resampled=False,
        training_updates=0, model_forwards=0, head_forwards=0, GPU_jobs=0, OT_calls=0,
        note='CPU identity and semantic-scope preparation only. No new geometry scoring or model-based selection.'))
    return manifest


def load_manifest(path=None):
    manifest = read(path or DATA / 'manifest.json')
    if manifest['schema'] != SCHEMA or not manifest['actual_data_ready']:
        raise ValueError('Full-seat manifest not ready')
    for p, h in manifest['protected_inputs'].items():
        if sha(p) != h:
            raise ValueError('Frozen data identity changed: ' + p)
    return manifest


def make_schedule(train, native_updates, seam_updates, seed=SEED):
    """Return deterministic parent-balanced scalar metadata; does not authorize work."""
    if not isinstance(native_updates, int) or not isinstance(seam_updates, int) or min(native_updates, seam_updates) < 1:
        raise ValueError('Positive integer stage budgets required')
    parents = sorted({r['parent_id'] for r in train})
    if len(parents) != len(train) or len(parents) < 4:
        raise ValueError('One full-seat C per parent; at least four parents required')
    lookup = {r['parent_id']: i for i, r in enumerate(train)}
    rng, aug = np.random.RandomState(seed), np.random.RandomState(seed ^ 0xABCDEF)
    queue, steps = [], []
    exposure = collections.Counter()
    stage_exposure = {'NATIVE': collections.Counter(), 'SEAM': collections.Counter()}
    for step in range(1, native_updates + seam_updates + 1):
        chosen = []
        while len(chosen) < 4:
            if not queue:
                queue.extend(map(int, rng.permutation(len(parents))))
            position = next((i for i, p in enumerate(queue) if p not in chosen), None)
            if position is None:
                queue.extend(map(int, rng.permutation(len(parents)))); continue
            chosen.append(queue.pop(position))
        stage = 'NATIVE' if step <= native_updates else 'SEAM'
        samples = []
        for j, pi in enumerate(chosen):
            ci = lookup[parents[pi]]; original = train[ci]
            for a in range(2):
                number = 10000000 + step * 1000 + j * 10 + a
                alpha = 1. if a == 0 else float(aug.uniform(.9, 1.1))
                samples.append(dict(index=ci, condition_index=ci, condition_id=original['condition_id'],
                    parent_id=original['parent_id'], part_type=original['part_type'],
                    sample_index=len(samples), alpha=alpha, scale_xyz=[alpha, 1., 1.],
                    augmentation='CLEAN' if a == 0 else 'X_WIDTH', noise_seed=number,
                    epsilon_seed=number+100000000, time_seed=number+200000000,
                    permutation_seed=number+300000000))
                exposure[original['condition_id']] += 1; stage_exposure[stage][original['condition_id']] += 1
        steps.append(dict(step=step, stage=stage, stage_step=step if stage == 'NATIVE' else step-native_updates, samples=samples))
    return dict(schema=SCHEMA + '_PARENT_WIDTH_STREAM', seed=seed, batch=8,
        distinct_parents_per_update=4, native_updates=native_updates, seam_updates=seam_updates,
        max_prepared_steps=native_updates+seam_updates, training_budget_authorized_by_this_file=False,
        record_index_domain='manifest.train', steps=steps,
        condition_exposure=dict(sorted(exposure.items())),
        stage_condition_exposure={k: dict(sorted(v.items())) for k, v in stage_exposure.items()},
        policy='Parent-shuffled cycles with no repeated parent in a batch; paired identity+x-width U[.9,1.1]; unchanged Gaussian/logit-normal/OT definitions.')


def sample_for_training(case, row, arm):
    from r2.pilots.bench_parts_convergence_v1.data import sample_for_training as original
    if arm not in ('PLAIN', 'SEAM') or not .9 <= float(row['alpha']) <= 1.1:
        raise ValueError('Invalid original adapter arguments')
    return original(case, row, arm)


def sample_for_probe(case, z, epsilon, t, permutation_seed, arm):
    from r2.pilots.bench_parts_convergence_v1.data import supervised_sample
    return supervised_sample(case, z, epsilon, t, permutation_seed, arm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'build', 'check'])
    args = p.parse_args()
    result = prepare() if args.command == 'prepare' else build() if args.command == 'build' else load_manifest()
    print(json.dumps(result.get('summary', dict(candidates=len(result['candidates'])) if 'candidates' in result else {}), ensure_ascii=False))


if __name__ == '__main__':
    main()
