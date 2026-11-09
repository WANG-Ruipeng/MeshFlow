"""Prepare source identities and an existing-output regression panel; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARTIFACTS = Path('/mnt/e/MeshFlow-Control-artifacts')
REPO = Path('/mnt/e/MeshFlow')
HIST = ARTIFACTS / 'experiments/chair_condition_recipe_factorial_v1'
DEFAULT_OUT = ARTIFACTS / 'experiments/chair_r2_user_conditions_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(str(a.shape).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def frozen_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing prepared identity differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def load_parent(path, uid):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    m = a['full_target']
    n = len(m)
    assert str(a['uid'].item()) == uid
    assert 128 <= n <= 256 and m.shape == (n, 9) and m.dtype == np.float32
    assert np.isfinite(m).all() and a['pre_ot_full'].dtype == np.float32
    assert bits(a['pre_ot_full'] * np.float32(2), m.reshape(n, 3, 3))
    ids = a['source_face_vertex_ids']
    assert ids.shape == (n, 3) and np.issubdtype(ids.dtype, np.integer)
    assert (ids >= 0).all() and (ids < len(a['model_vertices'])).all()
    assert bits(a['model_vertices'][ids], m.reshape(n, 3, 3))
    kept = a['retained_to_original_face']
    assert len(np.unique(kept)) == n
    assert (kept >= 0).all() and (kept < len(a['original_source_faces'])).all()
    inverse = np.full(len(a['original_source_faces']), -1, np.int64)
    inverse[kept] = np.arange(n)
    assert np.array_equal(inverse, a['original_to_retained_face'])
    assert np.array_equal(a['original_source_faces'][kept], a['source_faces'])
    cp = a['source_corner_permutations']
    assert np.array_equal(np.sort(cp, axis=1), np.tile(np.arange(3), (n, 1)))
    assert np.array_equal(np.take_along_axis(a['source_faces'], cp, axis=1), ids)
    return a


def prepare_inventory(out):
    train_path = ARTIFACTS / 'data/train_manifest.json'
    requests_path = HIST / 'evaluation/requests.json'
    split_path = REPO / 'native_t1/runs/control_mvp48_v1/data/splits.json'
    a40_path = ARTIFACTS / 'experiments/chair_early_alignment_schedule40_v1/evaluation/condition_construction.json'
    sources = {str(p): digest(p) for p in (train_path, requests_path, split_path, a40_path)}
    train, requests, splits, a40 = map(read, (train_path, requests_path, split_path, a40_path))
    train_ids = sorted({r['uid'] for r in train['tasks'] if r['ratio'] == .4})
    dev_ids = sorted({r['parent_id'] for r in requests['rows']})
    assert len(train_ids) == 26 and len(dev_ids) == 8 and not set(train_ids).intersection(dev_ids)
    assert set(dev_ids) == set(splits['validation'])
    parent_meta = {r['uid']: r for r in train['parents']}
    rows = []
    for split, uids in (('train', train_ids), ('dev', dev_ids)):
        for uid in uids:
            path = ((ARTIFACTS / 'data' / parent_meta[uid]['npz']) if split == 'train' else
                    REPO / 'native_t1/runs/control_mvp48_v1/data/meshes' / (uid + '.npz'))
            h = digest(path)
            if split == 'train':
                assert h == parent_meta[uid]['sha256']
            a = load_parent(path, uid)
            n = len(a['full_target'])
            checks = []
            if split == 'dev':
                for ratio in (.2, .4):
                    request = next(r for r in requests['rows'] if r['parent_id'] == uid and r['ratio'] == ratio and r['candidate'] == 0)
                    input_path = Path(request['input_path'])
                    assert digest(input_path) == request['input_sha256']
                    sources[str(input_path)] = request['input_sha256']
                    with np.load(input_path, allow_pickle=False) as z:
                        c = z['C'].copy()
                        assert int(z['N']) == n and int(z['K']) == len(c)
                    if ratio == .2:
                        reference_path = Path(request['reference_path'])
                        sources[str(reference_path)] = digest(reference_path)
                        assert sources[str(reference_path)] == request['reference_sha256']
                        reference = read(reference_path)
                        condition_path = Path(reference['condition_path'])
                        sources[str(condition_path)] = digest(condition_path)
                        assert sources[str(condition_path)] == reference['condition_sha256']
                        with np.load(condition_path, allow_pickle=False) as z:
                            face_ids = z['source_face_ids'].copy()
                            assert bits(c, z['C']) and bits(a['full_target'].reshape(n, 3, 3), z['GT_full'])
                            assert np.array_equal(z['retained_to_original_face'], a['retained_to_original_face'])
                            assert np.array_equal(z['source_face_vertex_ids'], a['source_face_vertex_ids'])
                        mapping_path = condition_path
                    else:
                        construction = next(r for r in a40['rows'] if r['parent_id'] == uid)
                        origin = Path(construction['source_path'])
                        sources[str(origin)] = digest(origin)
                        assert sources[str(origin)] == construction['source_sha256']
                        if 'source_face_ids' in construction:
                            face_ids = np.asarray(construction['source_face_ids'], np.int64)
                        else:
                            assert construction['construction'] == 'OLD_C_PLUS_NEAREST_FREE_CENTROIDS_EXACT_COORDINATES'
                            with np.load(origin, allow_pickle=False) as z:
                                appended = np.asarray(construction['appended_free_indices'], np.int64)
                                face_ids = np.r_[z['source_face_ids'], z['free_source_ids'][appended]]
                        mapping_path = a40_path
                    assert len(face_ids) == len(c) and len(set(face_ids.tolist())) == len(c)
                    assert (face_ids >= 0).all() and (face_ids < n).all()
                    assert bits(a['full_target'][face_ids].reshape(-1, 3, 3), c)
                    assert array_hash(c) == request['C_sha256']
                    checks.append({'ratio': ratio, 'condition_id': request['condition_id'],
                                   'input_path': str(input_path), 'input_sha256': request['input_sha256'],
                                   'source_face_ids': face_ids.tolist(), 'mapping_path': str(mapping_path),
                                   'mapping_sha256': sources[str(mapping_path)], 'C_source_bitwise': True})
            sources[str(path)] = h
            rows.append({'uid': uid, 'split': split, 'npz': str(path), 'sha256': h, 'N': n,
                         'official_split': splits['official_splits'][uid],
                         'group_id': splits['group_ids'][uid], 'full_target_sha256': array_hash(a['full_target']),
                         'identity_status': 'PASS', 'existing_condition_checks': checks})
    for p, h in sources.items():
        assert digest(p) == h, p
    result = {'schema': 'chair_r2_user_conditions_parent_inventory_v1', 'rows': rows,
              'counts': {'train': 26, 'dev': 8, 'total': 34}, 'train_dev_uid_overlap': [],
              'source_files': sources, 'source_before_after_equal': True,
              'parent_selection': 'Exact historical R2 READY40-compatible train26 plus observed DEV8; no mesh quality filtering.',
              'dev_interpretation': 'Repeatedly observed development parents, all official train; not independent test.',
              'coordinate_contract': 'Unchanged repaired full_target FP32; no copying, normalization, repair, or model.',
              'model_calls': 0, 'OT_calls': 0, 'training_updates': 0, 'sampling_calls': 0}
    frozen_write(out / 'parent_inventory.json', result)
    return result



def prepare_regression(out):
    inventory_path = out / 'parent_inventory.json'
    requests_path = HIST / 'evaluation/requests.json'
    completed_path = HIST / 'phases/main/sampling/R2_MIX_H_ALL/completed.json'
    report_path = HIST / 'EXPERIMENT_REPORT.md'
    inventory, requests, completed = map(read, (inventory_path, requests_path, completed_path))
    dev = sorted(r['uid'] for r in inventory['rows'] if r['split'] == 'dev')
    mandatory40 = {uid for uid in dev if uid.startswith(('341b', 'c1b', 'f596'))}
    assert len(mandatory40) == 3
    c40 = mandatory40 | {next(uid for uid in dev if uid not in mandatory40)}
    c20 = set(dev) - c40
    assert len(c40) == len(c20) == 4
    sources = {str(p): digest(p) for p in (inventory_path, requests_path, completed_path, report_path)}
    old_by_id = {r['request_id']: r for r in completed['rows']}
    parent_by_id = {r['uid']: r for r in inventory['rows']}
    rows = []
    checkpoint_ids = set()
    for uid in dev:
        ratio = .4 if uid in c40 else .2
        condition_check = next(c for c in parent_by_id[uid]['existing_condition_checks'] if c['ratio'] == ratio)
        if uid.startswith('341b'):
            reason = 'Existing report line81: C40 has enclosing walls/base; outputs often box/tub-like. Old appended-free C40, no trusted interface.'
        elif uid.startswith('c1b'):
            reason = 'Existing report line81: main R2 C40 n1 has a penetrating blade. Old appended-free C40, no trusted interface.'
        elif uid.startswith('f596'):
            reason = 'Existing report line81: several methods show free back panel close to C back panel; image alone cannot establish weld or legal thickness.'
        elif uid in c40:
            reason = 'Lexicographically first remaining DEV parent fills the fourth C40 slot; no new-output selection.'
        else:
            reason = 'Remaining DEV parents assigned C20 to balance four conditions per ratio; no new-output selection.'
        for noise in (0, 1):
            request = next(r for r in requests['rows'] if r['parent_id'] == uid and r['ratio'] == ratio and r['candidate'] == noise)
            input_path = Path(request['input_path'])
            sources[str(input_path)] = digest(input_path)
            assert sources[str(input_path)] == request['input_sha256']
            with np.load(input_path, allow_pickle=False) as z:
                c, gaussian = z['C'].copy(), z['z'].copy()
                assert array_hash(c) == request['C_sha256']
                assert array_hash(gaussian) == request['Gaussian_sha256']
                assert int(z['N']) == request['N'] and int(z['K']) == request['K']
            original = old_by_id[request['request_id']]
            assert original['status'] == 'COMPLETE' and original['label'] == 'R2_MIX_H_ALL'
            assert original['input_sha256'] == request['input_sha256']
            raw_path, run_path = Path(original['raw_path']), Path(original['result_path'])
            sources[str(raw_path)], sources[str(run_path)] = digest(raw_path), digest(run_path)
            assert sources[str(raw_path)] == original['raw_sha256']
            assert sources[str(run_path)] == original['result_sha256']
            run = read(run_path)
            checkpoint_ids.add(original['checkpoint_sha256'])
            assert run['request']['checkpoint_sha256'] == original['checkpoint_sha256']
            assert run['request']['input_sha256'] == request['input_sha256']
            assert run['nfe'] == 50 and run['output_stage'] == 'RAW' and run['postprocessing'] is None
            with np.load(raw_path, allow_pickle=False) as z:
                raw = z['raw']
                assert raw.shape == (request['N'], 9) and raw.dtype == np.float32 and np.isfinite(raw).all()
                assert bits(z['C'], c) and bits(z['z'], gaussian)
                assert bits(raw[:len(c)].reshape(-1, 3, 3), c)
                assert bits(z['free'], raw[len(c):].reshape(-1, 3, 3))
                trajectory = z['trajectory']
                assert trajectory.shape == (51, request['N'], 9) and bits(trajectory[-1], raw)
                assert all(bits(s[:len(c)].reshape(-1, 3, 3), c) for s in trajectory)
            rows.append({'uid': uid, 'split': 'dev', 'ratio': ratio, 'candidate': noise,
                         'condition_id': request['condition_id'], 'request_id': request['request_id'],
                         'selection_reason': reason, 'original_request': request,
                         'input_path': str(input_path), 'input_sha256': sources[str(input_path)],
                         'source_face_ids': condition_check['source_face_ids'],
                         'source_mapping': condition_check,
                         'existing_R2_RAW': {'status': 'VERIFIED_REUSABLE_FOR_RECORDED_MAIN_ENDPOINT',
                            'checkpoint_sha256': original['checkpoint_sha256'],
                            'reuse_condition': 'Frozen R2 endpoint must equal this recorded SHA256; a different R2 endpoint requires separate authorization and generation.',
                            'raw_path': str(raw_path), 'raw_sha256': sources[str(raw_path)],
                            'result_path': str(run_path), 'result_sha256': sources[str(run_path)],
                            'original_completed_row': original,
                            'current_identity_checks': {'input_hash': True, 'RAW_hash': True,
                                'C_bitwise': True, 'Gaussian_bitwise': True, 'C_all51_bitwise': True,
                                'N_FP32_finite': True, 'Euler50_recorded': True},
                            'checkpoint_file_rehashed': False, 'scientific_scores_recomputed': 0}})
    assert len(rows) == 16 and len(checkpoint_ids) == 1
    for p, h in sources.items():
        assert digest(p) == h, p
    result = {'schema': 'chair_r2_user_conditions_old_regression_selection_v1', 'rows': rows,
              'counts': {'parents': 8, 'conditions': 8, 'C20_conditions': 4, 'C40_conditions': 4,
                         'existing_noises_per_condition': 2, 'registered_requests': 16, 'verified_old_R2_RAW': 16},
              'selected_C40_parents': sorted(c40), 'selected_C20_parents': sorted(c20),
              'selection_rule': 'Freeze reported c1b40/f59640/341b40; fourth C40 is lexicographically first remaining parent; others C20; use original n0/n1 arrays without redraw.',
              'representation': 'Purposeful historical regression coverage, not representative prevalence, a random test, or selection on new outputs.',
              'historical_evidence': {'path': str(report_path), 'sha256': sources[str(report_path)], 'line': 81},
              'endpoint_binding': {'label': 'historical main R2_MIX_H_ALL', 'checkpoint_sha256': next(iter(checkpoint_ids)),
                                   'checkpoint_identity_source': str(completed_path), 'automatic_default_selection': False},
              'source_files': sources, 'source_before_after_equal': True,
              'model_calls': 0, 'OT_calls': 0, 'training_updates': 0, 'sampling_calls': 0,
              'new_Gaussian_draws': 0, 'scientific_scores_recomputed': 0}
    frozen_write(out / 'old_regression_selection.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--stage', choices=('inventory', 'regression', 'all'), default='all')
    args = p.parse_args()
    for stage in (('inventory', 'regression') if args.stage == 'all' else (args.stage,)):
        result = (prepare_inventory if stage == 'inventory' else prepare_regression)(args.out)
        print(json.dumps({'status': 'PASS', 'stage': stage, 'counts': result['counts']}))


if __name__ == '__main__':
    main()
