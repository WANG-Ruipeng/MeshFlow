"""CPU-only frozen inputs for the bounded terminal-reward NFT pilot.

Select eight existing coherent TRAIN FULL patches using C geometry alone.
No torch import, model deserialization, generation, scoring, or GT target use.
Every new Gaussian is registered here before a rollout can start.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_terminal_reward_nft_v1'
FULL = ROOT / 'experiments/r2_full_seat_learning_curve_v1'
VERSION = 'terminal_reward_nft_inputs_v1'
L = float(np.sqrt(3) * 2 / .3762)
FEATURES = ('K', 'C_area_over_L2', 'C_boundary_length_over_L', 'C_perimeter_squared_over_area')


def resolve(path):
    s = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', s):
        s = '/mnt/' + s[0].lower() + s[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s):
        s = s[5].upper() + ':/' + s[7:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def checked(path):
    p = resolve(path).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise ValueError('Write outside independent NFT output: ' + str(p))
    return p


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, Path):
        return str(x)
    return x


def write(path, value):
    p = checked(path)
    value = clean(value)
    if p.exists():
        if read(p) != value:
            raise ValueError('Refusing to change frozen JSON: ' + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with p.open('xb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return p


def save_npz(path, **values):
    p = checked(path)
    values = {k: np.asarray(v) for k, v in values.items()}
    if p.exists():
        old = arrays(p)
        if set(old) != set(values) or any(not same(old[k], v) for k, v in values.items()):
            raise ValueError('Refusing to change frozen NPZ: ' + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    buff = io.BytesIO()
    np.savez_compressed(buff, **values)
    with p.open('xb') as f:
        f.write(buff.getvalue())
        f.flush()
        os.fsync(f.fileno())
    return p


def noise_seed(namespace, identity):
    return int.from_bytes(hashlib.sha256((namespace + '/' + identity).encode()).digest()[:4], 'little')


def make_noise(seed, n, k):
    """Same MT19937/free-only draws and FP32 conversion as maintained sampler."""
    z = np.zeros((1, n, 9), np.float32)
    z[:, k:] = np.random.RandomState(seed).randn(n-k, 3, 3).astype(np.float32).reshape(1, n-k, 9)
    return z


def load_request(q):
    if sha(q['input_path']) != q['input_sha256']:
        raise ValueError('Frozen input changed: ' + q['request_id'])
    b = arrays(q['input_path'])
    n, k = int(q['N']), int(q['K'])
    if not 128 <= n <= 256 or not 2 <= k < n or (int(b['N']), int(b['K'])) != (n, k):
        raise ValueError('N/K contract')
    if b['C'].shape != (k, 3, 3) or b['z'].shape != (1, n, 9):
        raise ValueError('Input shape contract')
    if b['C'].dtype != np.float32 or b['z'].dtype != np.float32:
        raise ValueError('Input FP32 contract')
    if not np.isfinite(b['C']).all() or not np.isfinite(b['z']).all() or np.any(b['z'][:, :k]):
        raise ValueError('Input finite/known noise contract')
    for key, field in (('C', 'C_sha256'), ('z', 'Gaussian_sha256')):
        if array_sha(b[key]) != q[field]:
            raise ValueError('Input array changed: ' + field)
    return b


def _features(task):
    c = arrays(task['condition_source'])['C'].reshape(task['K'], 3, 3)
    p = arrays(task['parent_npz'])
    face_ids = np.asarray(task['source_task']['source_face_ids'], np.int64)
    faces = p['source_faces'][face_ids]
    vertices = p['model_vertices'].astype(np.float64)
    canonical = p['source_face_vertex_ids'][face_ids]
    if not same(p['model_vertices'][canonical], c):
        raise ValueError('C source topology/coordinate mismatch')
    edges = Counter(tuple(sorted((int(f[j]), int(f[(j+1) % 3])))) for f in faces for j in range(3))
    boundary = sorted(e for e, count in edges.items() if count == 1)
    if not boundary or any(count > 2 for count in edges.values()):
        raise ValueError('Unexpected C topology in qualified FULL task')
    a = c.astype(np.float64)
    area = .5 * np.linalg.norm(np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]), axis=1).sum()
    perimeter = sum(np.linalg.norm(vertices[u]-vertices[v]) for u, v in boundary)
    if not area > 0 or not perimeter > 0:
        raise ValueError('Nonpositive C area/perimeter')
    return dict(task_id=task['task_id'], parent_id=task['parent_id'], N=task['N'], K=task['K'],
                C_area_model=float(area), C_boundary_length_model=float(perimeter),
                C_area_over_L2=float(area/L**2), C_boundary_length_over_L=float(perimeter/L),
                C_perimeter_squared_over_area=float(perimeter**2/area),
                C_boundary_edge_count=len(boundary), structure_tags=task['structure_tags'])


def select_eight(tasks):
    """Average empirical ranks preserve exact ties; deterministic farthest point."""
    rows = [_features(tasks[tid]) for tid in sorted(tasks)]
    values = np.asarray([[r[k] for k in FEATURES] for r in rows], np.float64)
    ranks = np.empty_like(values)
    for j in range(values.shape[1]):
        v = values[:, j]
        ranks[:, j] = [(np.sum(v < x) + .5*(np.sum(v == x)-1)) / (len(v)-1) for x in v]
    for row, rank in zip(rows, ranks):
        row['selection_rank_features'] = dict(zip(FEATURES, rank.tolist()))
    center = np.mean(ranks, axis=0)
    first = min(range(len(rows)), key=lambda i: (float(np.square(ranks[i]-center).sum()), rows[i]['task_id']))
    chosen = [first]
    trace = [dict(selection_rank=0, task_id=rows[first]['task_id'], criterion='MIN_DISTANCE_TO_RANK_CENTROID',
                  squared_distance=float(np.square(ranks[first]-center).sum()))]
    while len(chosen) < 8:
        distance = np.min(np.square(ranks[:, None, :]-ranks[chosen][None, :, :]).sum(2), axis=1)
        available = [i for i in range(len(rows)) if i not in chosen]
        best = min(available, key=lambda i: (-float(distance[i]), rows[i]['task_id']))
        trace.append(dict(selection_rank=len(chosen), task_id=rows[best]['task_id'],
                          criterion='MAX_MIN_SQUARED_RANK_DISTANCE', squared_distance=float(distance[best])))
        chosen.append(best)
    return rows, [rows[i]['task_id'] for i in chosen], trace


def _freeze_request(q, c, namespace, out):
    q = dict(q)
    seed = noise_seed(namespace, q['request_id'])
    z = make_noise(seed, q['N'], q['K'])
    p = save_npz(out/'inputs/requests'/(q['request_id']+'.npz'), C=c, z=z, N=q['N'], K=q['K'])
    q.update(input_path=str(p), input_sha256=sha(p), C_sha256=array_sha(c), Gaussian_sha256=array_sha(z),
             free_Gaussian_sha256=array_sha(z[:, q['K']:]), noise_seed=seed, noise_namespace=namespace,
             noise_provenance='PREREGISTERED_MT19937_FREE_ONLY_FP32; no outcome-based resampling',
             full_noise_reused=False, action='NEW')
    load_request(q)
    return q


def prepare(out=OUT):
    out = checked(out)
    target = out/'inputs/manifest.json'
    if target.exists():
        return verify_manifest(target)
    if any((out/'training').glob('**/calls.jsonl')) or any((out/'evaluation').glob('**/calls.jsonl')):
        raise RuntimeError('Cannot prepare fresh inputs after model calls')
    protected = {}
    def protect(path, expected=None):
        path = resolve(path)
        h = sha(path)
        if expected is not None and h != expected:
            raise ValueError('Source hash mismatch: ' + str(path))
        protected[str(path)] = h
        return h
    source = FULL/'inputs/training_data.json'
    protect(source)
    old = read(source)
    tasks = old['tasks']
    if old['status'] != 'FROZEN' or len(tasks) != 47 or len({t['parent_id'] for t in tasks.values()}) != 47:
        raise ValueError('Require exact 47 coherent TRAIN FULL source pool')
    for t in tasks.values():
        for path, digest in ((t['condition_source'], t['condition_source_sha256']), (t['parent_npz'], t['parent_npz_sha256'])):
            protect(path, digest)
        if t['status'] != 'ELIGIBLE' or t['K'] < 2 or t['C_components'] != 1:
            raise ValueError('Changed FULL eligibility')
    feature_rows, selected, trace = select_eight(tasks)
    write(out/'inputs/selection.json', dict(version=VERSION, features=list(FEATURES), L=L, all47=feature_rows,
          selected_task_ids=selected, trace=trace, source_manifest_sha256=sha(source),
          selection_uses='C geometry and immutable task ID only; no generated outcomes, source-Y quality, or reward',
          ranking='Per feature average empirical rank / 46, ties retained; squared Euclidean rank distance; task_id tie break'))
    conditions = []
    for i, tid in enumerate(selected):
        t = tasks[tid]
        conditions.append(dict(task_id=tid, condition_id=tid+'__NFT_TRAIN', parent_id=t['parent_id'], N=t['N'], K=t['K'],
                               selection_rank=i, group=i % 2, condition_source=t['condition_source'],
                               condition_source_sha256=t['condition_source_sha256'], source_task=t['source_task']))
    training, rounds = [], []
    for round_index in range(16):
        group = round_index % 2
        active = [c for c in conditions if c['group'] == group]
        round_rows = []
        for c in active:
            condition = arrays(c['condition_source'])['C'].reshape(c['K'], 3, 3)
            for candidate in range(8):
                rid = 'NFTTRAIN__r'+str(round_index).zfill(2)+'__'+c['task_id']+'__n'+str(candidate)
                q = dict(request_id=rid, condition_id=c['condition_id'], parent_id=c['parent_id'], task_id=c['task_id'],
                         N=c['N'], K=c['K'], round_index=round_index, candidate=candidate, group=group,
                         condition_source=c['condition_source'], condition_source_sha256=c['condition_source_sha256'],
                         split='TRAIN', panel='NFT_TRAIN_INPUT_SELECTED', condition_type='SEAT_FULL',
                         region_type='SEAT', tier='FULL', variant='base', alpha=1.0,
                         terminal_pool_owner='NFT', self_fm_endpoint_reuse='SAME_NFT_ENDPOINTS; OFF_POLICY_FOR_SELF_FM',
                         selection_order=c['selection_rank'])
                q = _freeze_request(q, condition, VERSION+'/TRAIN', out)
                training.append(q)
                round_rows.append(rid)
        rounds.append(dict(round_index=round_index, group=group, task_ids=[c['task_id'] for c in active],
                           request_ids=round_rows, requests=32, rollout_policy='NFT_CURRENT_SNAPSHOT_BEFORE_ROUND_UPDATES',
                           self_fm_rollouts=0, self_fm_reuse='Same registered NFT endpoint pool; explicitly off-policy'))
    regpath = FULL/'registration.json'
    protect(regpath)
    reg = read(regpath)
    eval_old = [dict(q, evaluation_noise_partition='OLD', action='NEW') for q in reg['requests']]
    if len(eval_old) != 24 or len({q['condition_id'] for q in eval_old}) != 6:
        raise ValueError('Expected old six-parent / 24-request FULL panel')
    for q in eval_old:
        q.pop('cache', None)
        protect(q['input_path'], q['input_sha256'])
        protect(q['condition_source'], q['condition_source_sha256'])
        load_request(q)
    fresh = []
    for cid in dict.fromkeys(q['condition_id'] for q in eval_old):
        template = next(q for q in eval_old if q['condition_id'] == cid)
        c = load_request(template)['C']
        for candidate in (4, 5):
            q = dict(template)
            rid = 'NFTFRESH__'+cid+'__n'+str(candidate)
            q.update(request_id=rid, base_request_id=rid, candidate=candidate, evaluation_noise_partition='FRESH')
            for key in ('full_Gaussian_sha256', 'full_noise_path', 'full_noise_file_sha256', 'original_partial_request_id'):
                q.pop(key, None)
            fresh.append(_freeze_request(q, c, VERSION+'/EVAL_FRESH', out))
    queries = reg['queries']
    for q in queries:
        protect(q['path'], q['sha256'])
    bindpath = FULL/'evaluation/bindings/FULL_S6000.json'
    protect(bindpath)
    binding = read(bindpath)
    if binding['global_step'] != 12000 or binding['local_step'] != 6000 or binding['EMA']:
        raise ValueError('Wrong FULL_S6000 starting identity')
    protect(binding['path'], binding['sha256'])
    idxpath = FULL/'evaluation/indices/FULL_S6000/raw_index.json'
    protect(idxpath)
    idx = {r['request_id']: r for r in read(idxpath)['rows']}
    cache = {}
    for q in eval_old:
        r = idx[q['request_id']]
        if r['status'] != 'COMPLETE' or r['checkpoint_sha256'] != binding['sha256'] or r['input_sha256'] != q['input_sha256']:
            raise ValueError('Old FULL RAW identity mismatch')
        for key in ('raw', 'ply', 'result'):
            protect(r[key+'_path'], r[key+'_sha256'])
        cache[q['request_id']] = r
    seeds = [q['noise_seed'] for q in training+fresh]
    if len(set(seeds)) != 524 or set(seeds) & {q['noise_seed'] for q in eval_old}:
        raise ValueError('Registered seed collision; no silent retry allowed')
    exposure = Counter(q['parent_id'] for q in training)
    if len(training) != 512 or set(exposure.values()) != {64} or len(exposure) != 8:
        raise ValueError('Unbalanced training exposure')
    manifest = dict(version=VERSION, status='FROZEN', source_manifest=str(source), source_manifest_sha256=sha(source),
        selection_path=str(out/'inputs/selection.json'), selection_sha256=sha(out/'inputs/selection.json'),
        train_conditions=conditions, training_requests=training, rounds=rounds, parent_rollout_exposure=dict(exposure),
        evaluation_requests=eval_old+fresh, evaluation_queries=queries, queries=queries,
        base_raw_cache=cache, source_binding=binding, protected_hashes=protected,
        seed_rule='uint32 little-endian first4 SHA256(namespace/request_id); MT19937 free-only Gaussian; no worker identity',
        budgets=dict(training_rollouts=512, self_fm_rollouts=0, final_new_rollouts=84, base_cache_rollouts=24,
                     final_BASE_new=12, final_NFT_new=36, final_SELF_FM_new=36, total_new_rollouts=596,
                     total_new_sampling_forwards=29800, new_noise_arrays=524),
        scientific_limits=['Eight input-selected TRAIN parents; DEV remains untouched by selection and training.',
                           'SELF_FM is an off-policy control sharing NFT-generated endpoints, not independently sampled self-training.',
                           'Old evaluation requests are repeatedly observed development evidence; fresh noises are new, parents are not new.',
                           'C geometry selection does not select reward outcomes or require reconstructing the source parent.'],
        preparation=dict(model_deserializations=0, model_forwards=0, quality_rescoring=0, new_training=0,
                         checkpoint_file_hashed=True, source_geometry_changed=False),
        data_code_sha256=sha(__file__))
    write(target, manifest)
    return verify_manifest(target)


def verify_manifest(path=OUT/'inputs/manifest.json'):
    m = read(path)
    if m.get('version') != VERSION or m.get('status') != 'FROZEN':
        raise ValueError('Unrecognized NFT inputs')
    for p, h in m['protected_hashes'].items():
        if sha(p) != h:
            raise ValueError('Protected source changed: ' + p)
    if sha(m['selection_path']) != m['selection_sha256']:
        raise ValueError('Input selection changed')
    for q in m['training_requests'] + m['evaluation_requests']:
        load_request(q)
    if len(m['training_requests']) != 512 or len(m['evaluation_requests']) != 36 or len(m['base_raw_cache']) != 24:
        raise ValueError('Input budget changed')
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify_manifest() if args.verify else prepare()
    print(json.dumps(dict(status=result['status'], manifest=str(OUT/'inputs/manifest.json'),
                          train_parents=[c['parent_id'] for c in result['train_conditions']],
                          budgets=result['budgets']), sort_keys=True), flush=True)
