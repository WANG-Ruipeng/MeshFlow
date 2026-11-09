"""Local paths and immutable evidence for the authorized best-of-four route."""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'experiments/r2_bench_four_routes_v1/bestof'
CODE = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/r2_bench_high_joint_coverage_v1'
NARROW = ROOT / 'experiments/r2_bench_natural_n_recipe_v1'
MANIFEST = OUT / 'data/manifest.json'
LABEL = 'B4'
VERSION = 'bench_four_routes_bestof_v1'
GENERATOR = NARROW / 'training/NATURAL_HIGH/generator_step1500.pt'
GENERATOR_SHA = 'b9a3979c84bb851489c49f8b1539618c6908d9f64b4b579cad4e51043264f2f8'
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(ROOT / 'r2/src'))


def resolve(path):
    s = str(path).replace('\\', '/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def immutable(path, obj):
    path = resolve(path)
    obj = json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read(path) != obj:
            raise ValueError('Immutable file differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def verify(book):
    for path, digest in book.items():
        if sha(path) != digest:
            raise ValueError('Frozen identity differs: ' + str(path))


def arrays(path):
    import numpy as np
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def array_file(path, values):
    import numpy as np
    path = resolve(path)
    if path.exists():
        previous = arrays(path)
        if set(previous) != set(values) or any(not np.array_equal(previous[k], v) for k, v in values.items()):
            raise ValueError('Existing array differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **values)


def utc():
    return datetime.now(timezone.utc).isoformat()


def candidates():
    qs = read(MANIFEST)['candidates']
    if len(qs) != 92 or len({q['request_id'] for q in qs}) != 92:
        raise ValueError('Expected exactly 92 candidate identities')
    return qs


def new_requests(label=LABEL):
    if label != LABEL:
        raise ValueError(label)
    return [q for q in candidates() if q['noise_index'] >= 2]


def cached_root(q):
    return SOURCE if q['report_group'] == 'NEW_WIDE_TRAIN' else NARROW


def raw_dir(q):
    if q['noise_index'] < 2:
        return cached_root(q) / 'evaluation/B' / q['request_id'] / 'raw'
    return OUT / 'evaluation' / LABEL / q['request_id'] / 'raw'


def post_dir(q):
    if q['noise_index'] < 2:
        return cached_root(q) / 'evaluation/B' / q['request_id'] / 'post'
    return OUT / 'evaluation' / LABEL / q['request_id'] / 'post'


def score_dir(q, stage):
    root = cached_root(q) if q['noise_index'] < 2 else OUT
    label = 'B' if q['noise_index'] < 2 else LABEL
    if stage == 'RAW':
        return root / 'evaluation/scores' / label / q['request_id'] / stage
    if stage not in STAGES:
        raise ValueError(stage)
    return root / 'cpu' / label / q['request_id'] / stage / 'scores'


def checked_raw(q):
    path = raw_dir(q) / 'receipt.json'
    r = read(path)
    if (r['status'] != 'COMPLETE' or r['request_id'] != q['request_id'] or r['generator_forwards'] != 50
            or r['D0_forwards'] != 0 or r['noise_sha256'] != q['noise_sha256']
            or r['raw_sha256'] != sha(path.parent / 'raw.npz')
            or r['generator']['sha256'] != GENERATOR_SHA):
        raise ValueError('RAW identity mismatch: ' + str(path))
    verify(r['files'])
    return path


def checked_score(q, stage):
    folder = score_dir(q, stage)
    done = read(folder / 'complete.json')
    if done['status'] != 'PASS' or done['model_calls'] != 0:
        raise ValueError('Incomplete score')
    verify(done['files'])
    row, score = read(folder / 'row.json'), read(folder / 'score.json')
    for key in ('request_id', 'condition_id', 'parent_id', 'noise_index', 'N', 'K'):
        if row[key] != q[key]:
            raise ValueError('Score identity differs: ' + key)
    if score['seam_to_bulk']['status'] != 'NA_NOT_SEAM_REPRESENTATION':
        raise ValueError('Native applicability changed')
    return row, score


def csvwrite(path, rows):
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import csvwrite as writer
    writer(resolve(path), rows)


def registered():
    r = read(OUT / 'registration.json')
    if r['status'] != 'FROZEN' or r['version'] != VERSION:
        raise ValueError('Best-of-four registration absent')
    verify(r['code']); verify(r['inputs'])
    return r
