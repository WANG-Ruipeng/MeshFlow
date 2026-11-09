"""CPU-only immutable inputs for the authorized frozen-prior motif experiment.

No torch, model, scorer, GPU probe, training, or new free Gaussian is called.
Only the 46 registered shadow Gaussians are created by prepare(). Generation
and evaluation code identities are frozen separately at their stage boundaries.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_full_prior_motif_guidance_v1'
REFERENCE = ROOT / 'experiments/r2_bench_four_routes_v1/exposure'
PROTOCOL = ROOT / 'maintenance/motif_guidance_design_20261029/PROTOCOL.md'
MANIFEST = REFERENCE / 'data/eval_manifest.json'
REQUESTS = OUT / 'inputs/requests.json'
TASKS = OUT / 'tasks.json'
VERSION = 'bench_full_prior_motif_guidance_cpu_inputs_v1'
EXPERIMENT = 'R2_BENCH_FULL_PRIOR_MOTIF_GUIDANCE_V1'
L = 9.208138264587333
LAMBDAS = (0.1, 0.3, 1.0)
CALIBRATION_PARENTS = (
    '6b894c1c57ba921c98d10ca8df5ecbe7',
    'a8e987a6a6486b962d6c8e51365a5a87',
    '28b9f2908f0cbb179c21c139c0613648',
    'bcb4f576747e10bdd7f6e86d7d335698',
)
MANIFEST_SHA = 'd261dbd7a3246613f994411f60fa082be933abe2b8c08fc167948dcc5fab8f18'
OFFICIAL = ROOT / 'output/120m-ot-v-bench/checkpoints/01000000.pt'
OFFICIAL_SHA = '00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505'
OFFICIAL_EMA_SHA = '2cc63f56d79b88418840fc44ad4c954d1961ad898a7b81d198a35bf49f0deb25'
D0 = ROOT / 'checkpoints/official_bench_denoiser/0021000.pt'
D0_SHA = '67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14'
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')


def resolve(path):
    s = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', s):
        s = '/mnt/' + s[0].lower() + s[2:]
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s):
        s = s[5].upper() + ':/' + s[7:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _owned(path):
    p = resolve(path).resolve()
    if not p.is_relative_to(OUT.resolve()):
        raise ValueError('Only this experiment output directory is writable: ' + str(p))
    return p


def write_immutable(path, value):
    p = _owned(path); value = clean(value)
    if p.exists():
        if read(p) != value: raise ValueError('Immutable JSON differs: ' + str(p))
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    return p


def save_npz(path, data):
    p = _owned(path)
    if p.exists(): raise FileExistsError(p)
    if any(np.asarray(v).dtype.hasobject for v in data.values()):
        raise ValueError('Object NPZ arrays forbidden')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
    return p


def copy_immutable(source, dest):
    source, dest = resolve(source), _owned(dest)
    content = source.read_bytes()
    if dest.exists():
        if dest.read_bytes() != content: raise ValueError('Immutable copy differs: ' + str(dest))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
    return dest


def verify(book):
    for path, digest in book.items():
        if sha(path) != digest: raise ValueError('Frozen file changed: ' + str(path))


def _safe_id(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:'):
        raise ValueError('Unsafe identity')
    return value


def shadow_seed(request_id):
    key = EXPERIMENT + '|SHADOW|' + _safe_id(request_id)
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4], 'little')


def native_layout(q):
    """The existing all-free/C-only corner contract, without neural imports."""
    inp = arrays(q['inference_npz']); n, k = int(q['N']), int(q['K'])
    if set(inp) != {'C_vertices', 'C_faces_original', 'C_faces_model', 'N', 'K'}:
        raise ValueError('Inference input must contain C/N only')
    v = inp['C_vertices']; original = inp['C_faces_original']; canonical = inp['C_faces_model']
    if v.dtype != np.float32 or original.shape != (k, 3) or canonical.shape != (k, 3):
        raise ValueError('C dtype/shape mismatch')
    if int(inp['N']) != n or int(inp['K']) != k or not 0 < k < n <= 799 or n < 128:
        raise ValueError('Invalid N/K')
    if not np.isfinite(v).all() or not np.array_equal(np.sort(original, 1), np.sort(canonical, 1)):
        raise ValueError('Canonical C changed source face identity')
    book = {}
    for fi, face in enumerate(original):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, w = int(face[a]), int(face[b])
            book.setdefault(tuple(sorted((u, w))), []).append((fi, u, w))
    if any(len(rows) > 2 for rows in book.values()): raise ValueError('Nonmanifold C')
    if any(len(rows) == 2 and rows[0][1:] != rows[1][1:][::-1] for rows in book.values()):
        raise ValueError('Source C winding mismatch')
    edges = np.asarray([rows[0] for _, rows in sorted(book.items()) if len(rows) == 1], np.int64).reshape(-1, 3)
    anchors = np.full((n, 3), -1, np.int64); anchors[:k] = canonical
    return dict(N=n, K=k, C_vertices=v, C_faces_original=original, C_faces_model=canonical,
                known_mask=np.arange(n) < k, valid_mask=np.ones(n, bool), anchor_ids=anchors,
                occupied=np.zeros(len(edges), bool), edge_records=edges,
                seam_slots=np.empty(0, np.int64), bulk_slots=np.arange(k, n, dtype=np.int64),
                free_corner_mask=anchors < 0, free_coordinate_count=9 * (n-k))


def raw_arrays(q, canonical_full):
    """Decode via the unchanged NumPy RAW helper; source metadata is export-only."""
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays as original
    layout = native_layout(q); x = np.asarray(canonical_full)
    if x.dtype != np.float32 or x.shape != (layout['N'], 3, 3) or not np.isfinite(x).all():
        raise ValueError('Canonical terminal must be finite FP32 [N,3,3]')
    if not np.array_equal(x[:layout['K']], layout['C_vertices'][layout['C_faces_model']]):
        raise ValueError('Final canonical C overwrite missing')
    source = arrays(q['source_npz'])
    result = original(x, layout, np.full(len(layout['occupied']), -1, np.float32), source)
    # Generic helper numbers inactive candidate edge slots. They are ordinary
    # Native free slots here; explicit -1 avoids implying any SEAM role.
    result['raw_slot_edge_id'] = np.full(layout['N'], -1, np.int64)
    return result


def _base_task(q, arm, lambda_value=None, suffix=None):
    label = suffix or arm; rid = q['request_id']; cell = _safe_id(label + '__' + rid)
    folder = OUT / 'raw' / cell
    return dict(cell_id=cell, job_id=cell, arm=arm, label=label, lambda_value=lambda_value,
                request_id=rid, panel_role=q['panel_role'], q=q,
                shadow_npz=q['shadow_noise_npz'], source_npz=q['source_npz'],
                output_dir=str(folder), raw_path=str(folder/'raw.npz'),
                raw_receipt_path=str(folder/'receipt.json'), baseline_origin='NEW_OFFICIAL_FULL_PRIOR',
                main_after_lock=not (arm == 'G3' and q['panel_role'] == 'calibration'))


def lambda_label(value):
    return {0.1: 'G3_L0p1', 0.3: 'G3_L0p3', 1.0: 'G3_L1p0'}[float(value)]


def job(task):
    return {k: task[k] for k in ('job_id', 'arm', 'lambda_value', 'request_id', 'output_dir')}


def load_requests():
    data = read(REQUESTS)
    rows = data['requests'] if isinstance(data, dict) else data
    if len(rows) != 46 or len({q['request_id'] for q in rows}) != 46:
        raise ValueError('Exactly 46 distinct frozen requests required')
    return rows


def load_tasks(resolved=False):
    tasks = read(TASKS)
    if len(tasks) != 200 or len({t['cell_id'] for t in tasks}) != 200:
        raise ValueError('Exactly 200 immutable task identities required')
    if resolved:
        locked = read(OUT/'lock.json'); value = float(locked['lambda_value'])
        if value not in LAMBDAS: raise ValueError('Unsupported locked lambda')
        for t in tasks:
            if t['arm'] in ('G2', 'G3') and t['lambda_value'] is None: t['lambda_value'] = value
    return tasks


def selected_tasks():
    tasks = load_tasks(resolved=True); value = float(read(OUT/'lock.json')['lambda_value'])
    selected = [t for t in tasks if t['main_after_lock'] or float(t['lambda_value']) == value]
    if len(selected) != 184: raise ValueError('Expected four methods times 46 selected outputs')
    return selected


def calibration_tasks():
    return [t for t in load_tasks() if t['panel_role'] == 'calibration' and t['arm'] != 'G2']


def raw_receipt(task):
    return resolve(task['raw_receipt_path'])


def verify_raw(task):
    r = read(raw_receipt(task)); p = resolve(task['raw_path']); q = task['q']
    if r.get('status') != 'COMPLETE' or r.get('generator_forwards') != 50 or r.get('D0_forwards', 0) != 0:
        raise ValueError('Incomplete or wrong-budget RAW: ' + task['cell_id'])
    if r.get('request_id') != q['request_id'] or r.get('arm') != task['arm']:
        raise ValueError('RAW producer task differs')
    if task['lambda_value'] is not None and r.get('lambda_value') != task['lambda_value']:
        raise ValueError('RAW lambda differs')
    if r.get('noise_sha256') != q['noise_sha256'] or r.get('shadow_noise_sha256') != q['shadow_noise_sha256']:
        raise ValueError('RAW frozen noise identity differs')
    verify(r['files'])
    if not any(resolve(k) == p and h == sha(p) for k, h in r['files'].items()):
        raise ValueError('RAW not sealed in producer receipt')
    a, inp = arrays(p), arrays(q['inference_npz']); n, k = int(q['N']), int(q['K'])
    for name in ('raw', 'output', 'raw_anchor_ids', 'raw_seam_slot_mask', 'raw_slot_edge_id', 'occupied',
                 'C_vertices', 'C_faces_original', 'C_faces_model', 'known_faces', 'input_faces'):
        if name not in a: raise ValueError('Missing original RAW contract: ' + name)
    if a['raw'].dtype != np.float32 or a['raw'].shape != (n, 3, 3) or not np.isfinite(a['raw']).all():
        raise ValueError('Nonfinite/invalid RAW')
    if int(a['known_faces']) != k or int(a['input_faces']) != n:
        raise ValueError('RAW N/K differs')
    for name in ('C_vertices', 'C_faces_original', 'C_faces_model'):
        if not np.array_equal(a[name], inp[name]): raise ValueError('RAW C metadata changed')
    if not np.array_equal(a['raw'][:k], inp['C_vertices'][inp['C_faces_model']]):
        raise ValueError('Canonical C preservation failed')
    if not np.array_equal(a['output'][:k], inp['C_vertices'][inp['C_faces_original']]):
        raise ValueError('Source-winding C preservation failed')
    if not np.array_equal(a['raw'][k:], a['output'][k:]): raise ValueError('Export changed free corners')
    if np.any(a['raw_anchor_ids'][k:] >= 0) or np.any(a['raw_seam_slot_mask']) or np.any(a['occupied']):
        raise ValueError('Unexpected SEAM/free anchors')
    if np.any(a['raw_slot_edge_id'] != -1): raise ValueError('Native output must not assign seam slot identities')
    return r


def verify_raw_barrier():
    tasks = load_tasks(resolved=True); files = {}
    for task in tasks:
        r = verify_raw(task)
        for p in (task['raw_path'], task['raw_receipt_path']): files[str(resolve(p))] = sha(p)
    for p in (TASKS, OUT/'lock.json', REQUESTS): files[str(p)] = sha(p)
    return tasks, dict(status='PASS', outputs=200, selected_outputs=184,
                       generator_forwards=10000, D0_before_barrier=0, files=files)


def _baseline(q):
    rid = q['request_id']; rawdir = REFERENCE/'evaluation/EXPOSURE'/rid/'raw'
    stage_paths = dict(RAW=str(rawdir/'raw.npz'),
                       BASE_POST=str(REFERENCE/'evaluation/EXPOSURE'/rid/'post/post.npz'),
                       CLEAN_FINAL_POST=str(REFERENCE/'cpu/EXPOSURE'/rid/'CLEAN_FINAL_POST/post.npz'))
    scores = {s: str(REFERENCE/'evaluation/scores/EXPOSURE'/rid/'RAW') if s == 'RAW'
              else str(REFERENCE/'cpu/EXPOSURE'/rid/s/'scores') for s in STAGES}
    return dict(cell_id='R__'+rid, arm='R', label='R', request_id=rid, panel_role=q['panel_role'], q=q,
                raw_path=stage_paths['RAW'], raw_receipt_path=str(rawdir/'receipt.json'),
                source_npz=q['source_npz'], stage_paths=stage_paths, score_paths=scores,
                baseline_origin='EXPOSURE_GLOBAL25500_FROZEN_CACHE')


def prepare():
    started = time.perf_counter()
    if (OUT/'registration.json').exists():
        return verify_prepared()
    if 'torch' in sys.modules: raise RuntimeError('CPU preparation must not import torch')
    if sha(MANIFEST) != MANIFEST_SHA: raise ValueError('Original frozen panel hash mismatch')
    old = read(MANIFEST)
    if len(old['requests']) != 46 or len(old['metric_fields']) != 177:
        raise ValueError('Unexpected panel/schema')
    OUT.mkdir(parents=True, exist_ok=True)
    copy_immutable(PROTOCOL, OUT/'PROTOCOL.md')
    copy_immutable(MANIFEST, OUT/'inputs/original_eval_manifest.json')
    hashes = {}; memo = {}
    def freeze(path, expected=None):
        p = resolve(path); key = str(p)
        digest = memo.setdefault(key, None)
        if digest is None: digest = memo[key] = sha(p)
        if expected is not None and digest != expected: raise ValueError('Original asset mismatch: ' + key)
        hashes[key] = digest
        return digest
    for p in (PROTOCOL, MANIFEST, REFERENCE/'data/evaluation_plan.json', REFERENCE/'evaluation/score_registration.json',
              REFERENCE/'results/per_output.csv', REFERENCE/'visual_review/summary_v1/actual_observations.csv'):
        freeze(p)
    models = []
    for role, path, expected in (('OFFICIAL_EMA', OFFICIAL, OFFICIAL_SHA), ('OFFICIAL_D0', D0, D0_SHA)):
        print(json.dumps(dict(stage='CPU_HASH_MODEL_FILE_NO_LOAD', role=role, path=str(path))), flush=True)
        digest = sha(path)
        if digest != expected: raise ValueError('Official weight file hash mismatch: ' + role)
        stat = path.stat()
        models.append(dict(role=role, path=str(path), sha256=digest, bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns, loaded=False))
    requests = []; input_audit = []
    for old_q in old['requests']:
        q = copy.deepcopy(old_q); rid = _safe_id(q['request_id']); n, k = int(q['N']), int(q['K'])
        q['panel_role'] = 'calibration' if q['parent_id'] in CALIBRATION_PARENTS else 'confirmation'
        for key, digest_key in (('inference_npz', 'inference_npz_sha256'), ('metadata_npz', 'metadata_sha256'),
                                ('source_npz', 'source_npz_sha256'), ('noise_npz', 'noise_sha256')):
            freeze(q[key], q[digest_key])
        freeze(q['query_descriptor']['path'], q['query_descriptor']['sha256'])
        inp = arrays(q['inference_npz']); z = arrays(q['noise_npz'])['z']; query = arrays(q['query_descriptor']['path'])
        if z.dtype != np.float32 or z.shape != (n-k, 3, 3) or not np.isfinite(z).all():
            raise ValueError('Original free Gaussian invalid')
        if len(query['C_points']) != 2048 or not np.isclose(query['C_weights'].sum(), 1):
            raise ValueError('Original query contract differs')
        seed = shadow_seed(rid); folder = OUT/'inputs/shadow'/rid
        path, receipt = folder/'shadow_noise.npz', folder/'receipt.json'
        if path.exists():
            if not receipt.exists(): raise RuntimeError('Unsealed partial shadow file; no silent overwrite')
            prior = read(receipt)
            if prior['seed'] != seed or prior['request_id'] != rid or sha(path) != prior['sha256']:
                raise ValueError('Existing shadow noise differs')
            shadow = arrays(path)['z']
        else:
            shadow = np.random.RandomState(seed).standard_normal((k, 3, 3)).astype(np.float32)
            save_npz(path, dict(z=shadow))
            write_immutable(receipt, dict(status='COMPLETE', request_id=rid, seed=seed, shape=[k, 3, 3], dtype='float32',
                seed_rule='SHA256(experiment+|SHADOW|+request_id) first4 bytes little endian',
                RNG='numpy.random.RandomState.standard_normal', numpy_version=np.__version__,
                sha256=sha(path), array_sha256=array_sha(shadow), new_shadow_arrays=1, new_free_arrays=0))
        if shadow.dtype != np.float32 or shadow.shape != (k, 3, 3) or not np.isfinite(shadow).all():
            raise ValueError('Shadow dtype/shape/nonfinite failure')
        q.update(shadow_noise_npz=str(path), shadow_noise_sha256=sha(path), shadow_seed=seed,
                 shadow_noise_array_sha256=array_sha(shadow), original_free_array_sha256=array_sha(z))
        freeze(path); freeze(receipt)
        # Round-trip the actual immutable C corner metadata, without generating F.
        x = np.empty((n, 3, 3), np.float32); x[:k] = inp['C_vertices'][inp['C_faces_model']]; x[k:] = z
        raw = raw_arrays(q, x)
        if not np.array_equal(raw['output'][:k], inp['C_vertices'][inp['C_faces_original']]) or not np.array_equal(raw['output'][k:], z):
            raise ValueError('C source corner roundtrip/free identity failure')
        input_audit.append(dict(request_id=rid, N=n, K=k, panel_role=q['panel_role'], shadow_seed=seed,
            shadow_sha256=q['shadow_noise_sha256'], free_array_sha256=array_sha(z), C_roundtrip=True,
            original_query_count=2048, no_new_free_noise=True, no_source_target_for_generation=True))
        requests.append(q)
    if sum(q['panel_role'] == 'calibration' for q in requests) != 8:
        raise ValueError('Four registered calibration parents required')
    write_immutable(REQUESTS, dict(schema=VERSION+'_requests', source_manifest=str(MANIFEST),
                                  source_manifest_sha256=MANIFEST_SHA, requests=requests))
    tasks = [_base_task(q, arm) for arm in ('G0', 'G1', 'G2') for q in requests]
    for value in LAMBDAS:
        tasks.extend(_base_task(q, 'G3', value, lambda_label(value)) for q in requests if q['panel_role'] == 'calibration')
    tasks.extend(_base_task(q, 'G3') for q in requests if q['panel_role'] == 'confirmation')
    if len(tasks) != 200: raise AssertionError('200 fixed outputs')
    write_immutable(TASKS, tasks)
    cal = [t for t in tasks if t['panel_role'] == 'calibration' and t['arm'] != 'G2']
    first = next(t for t in cal if t['arm'] == 'G3' and t['lambda_value'] == 0.1
                 and t['q']['parent_id'].startswith('bcb4f5') and t['q']['noise_index'] == 0)
    cal = [first] + [t for t in cal if t is not first]
    write_immutable(OUT/'jobs_calibration.json', dict(schema=VERSION+'_jobs', phase='calibrate_raw', jobs=[job(t) for t in cal]))
    baseline = [_baseline(q) for q in requests]
    for b in baseline:
        freeze(b['raw_receipt_path'])
        for path in b['stage_paths'].values(): freeze(path)
        for stage, folder in b['score_paths'].items():
            complete = read(resolve(folder)/'complete.json')
            if complete.get('status') != 'PASS': raise ValueError('Baseline score not complete')
            for name in ('row.json', 'score.json', 'complete.json'): freeze(resolve(folder)/name)
        # Sealed original files remain at their registered paths; no expanded copies.
        for stage in ('RAW', 'CLEAN_FINAL_POST'):
            freeze(REFERENCE/'gallery'/(stage+'__'+b['request_id']+'.png'))
    write_immutable(OUT/'baseline.json', baseline)
    write_immutable(OUT/'inputs/metric_fields.json', old['metric_fields'])
    write_immutable(OUT/'inputs/audit.json', dict(status='PASS', rows=input_audit, requests=46,
        parents=23, calibration_parents=4, confirmation_parents=19,
        new_shadow_arrays=46, new_free_arrays=0, model_forwards=0, scoring_calls=0, OT_calls=0))
    generated = [OUT/'PROTOCOL.md', OUT/'inputs/original_eval_manifest.json', REQUESTS, TASKS,
                 OUT/'jobs_calibration.json', OUT/'baseline.json', OUT/'inputs/metric_fields.json', OUT/'inputs/audit.json']
    for p in generated: freeze(p)
    reg = dict(schema=VERSION+'_registration', status='FROZEN', experiment=EXPERIMENT,
        execution_authorization='User authorized execution after the design-only source protocol; stages dispatched separately',
        protocol_path=str(OUT/'PROTOCOL.md'), protocol_sha256=sha(PROTOCOL),
        requests_path=str(REQUESTS), requests_sha256=sha(REQUESTS), tasks_path=str(TASKS), tasks_sha256=sha(TASKS),
        baseline_path=str(OUT/'baseline.json'), metric_fields_path=str(OUT/'inputs/metric_fields.json'),
        official_path=str(OFFICIAL), official_sha256=OFFICIAL_SHA, official_ema_tensor_sha256=OFFICIAL_EMA_SHA,
        d0_path=str(D0), d0_sha256=D0_SHA, models=models, inputs=hashes,
        calibration_parent_ids=CALIBRATION_PARENTS, lambda_candidates=LAMBDAS,
        L=L, count_cfg=2.0, steps=50, states=51, guidance_start_k=5,
        maximum_corner_guidance_displacement=0.005*L,
        budget=dict(new_RAW=200, main_outputs=184, auxiliary_RAW=16, cached_R_outputs=46,
                    new_shadow_arrays=46, new_free_arrays=0, logical_vector_fields=10000,
                    cfg_network_sample_forwards=20000, input_VJPs=4860, D0=184,
                    new_score_rows=568, reused_score_rows=138, all_score_rows=706),
        preparation_code_hashes={str(Path(__file__)): sha(__file__)},
        generation_code_freeze='generation_code.json created only after generate/guidance/run are complete',
        evaluation_code_freeze='created immediately before first new scoring invocation',
        utc=utc(), wall_seconds=time.perf_counter()-started, model_loads=0, GPU_calls=0, scoring_calls=0)
    write_immutable(OUT/'registration.json', reg)
    return dict(status='PASS', registration=str(OUT/'registration.json'), registration_sha256=sha(OUT/'registration.json'),
                requests=46, calibration_jobs=40, all_tasks=200, new_shadow_arrays=46, new_free_arrays=0,
                wall_seconds=time.perf_counter()-started, model_loads=0, scoring_calls=0)


def verify_prepared(full_weight_hash=False):
    reg = read(OUT/'registration.json'); verify(reg['inputs'])
    for model in reg['models']:
        p = resolve(model['path']); stat = p.stat()
        if stat.st_size != model['bytes'] or stat.st_mtime_ns != model['mtime_ns']:
            raise ValueError('Frozen model file stat changed: ' + str(p))
        if full_weight_hash and sha(p) != model['sha256']: raise ValueError('Frozen model file hash differs')
    load_requests(); load_tasks()
    return dict(status='PASS', completed_state_read_only=True, registration_sha256=sha(OUT/'registration.json'),
                requests=46, tasks=200, weight_check='SHA256' if full_weight_hash else 'STAT_AGAINST_INITIAL_HASH_RECEIPT',
                new_shadow_arrays=0, new_free_arrays=0, model_loads=0, scoring_calls=0)


def freeze_generation_code():
    paths = [Path(__file__), Path(__file__).with_name('generate.py'),
             Path(__file__).with_name('guidance.py'), Path(__file__).with_name('run.py')]
    result = dict(schema=VERSION+'_generation_code', status='FROZEN',
                  registration_sha256=sha(OUT/'registration.json'), files={str(p): sha(p) for p in paths},
                  excluded='evaluate.py is frozen separately before first scoring; no scientific runtime change implied')
    write_immutable(OUT/'generation_code.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'freeze-generation'))
    parser.add_argument('--full-weight-hash', action='store_true')
    args = parser.parse_args()
    result = prepare() if args.action == 'prepare' else verify_prepared(args.full_weight_hash) if args.action == 'verify' else freeze_generation_code()
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
