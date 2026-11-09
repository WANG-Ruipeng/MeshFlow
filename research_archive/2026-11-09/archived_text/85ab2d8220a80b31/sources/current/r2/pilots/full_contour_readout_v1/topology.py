"""C-only source topology for the fixed FULL contour readout experiment.

No torch/model imports, no coordinate welding, no source-Y interface labels.
The only network input returned here is bool[N,3], edges (01,12,20), aligned
with the actual C corners in the current model input. Source IDs stay here.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib, json, os, re
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_full_contour_readout_v1'
BASE = ROOT / 'experiments/r2_full_seat_learning_curve_v1'
MANIFEST = OUT / 'inputs/topology_manifest.json'
VERSION = 'r2_full_contour_readout_c_topology_v1'
EDGES = np.asarray([[0, 1], [1, 2], [2, 0]], np.int64)
EDGE_LOOKUP = np.asarray([[-1, 0, 2], [0, -1, 1], [2, 1, -1]], np.int64)


def resolve(path):
    s = str(path).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', s): s = '/mnt/' + s[0].lower() + s[2:]
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s): s = s[5].upper() + ':/' + s[7:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path): return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def freeze(path, value):
    p = resolve(path).resolve()
    if not p.is_relative_to(OUT.resolve()) or not p.name.startswith('topology'):
        raise ValueError('Topology writes are restricted to new topology artifacts')
    if p.exists():
        if read(p) != value: raise ValueError('Frozen topology artifact differs: ' + str(p))
        return sha(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    return sha(p)


def boundary_mask(vertex_ids):
    """Count only input-C undirected indexed edges; IDs need not be contiguous."""
    ids = np.asarray(vertex_ids)
    if ids.dtype.kind not in 'iu' or ids.ndim != 2 or ids.shape[1] != 3:
        raise ValueError('Expected integer C vertex identities [K,3]')
    if np.any(ids < 0) or any(len(set(x.tolist())) != 3 for x in ids):
        raise ValueError('Missing vertex identities or repeated vertex in a face')
    pairs = np.sort(ids[:, EDGES], axis=-1)
    counts = Counter(tuple(int(v) for v in p) for p in pairs.reshape(-1, 2))
    incidence = np.asarray([[counts[tuple(int(v) for v in p)] for p in face] for face in pairs], np.int64).reshape(len(ids), 3)
    mask = incidence == 1
    return mask, dict(boundary_edges=int(mask.sum()), internal_edges=sum(v == 2 for v in counts.values()),
                      nonmanifold_edges=sum(v > 2 for v in counts.values()), unique_edges=len(counts),
                      incidence_histogram={str(v): list(counts.values()).count(v) for v in sorted(set(counts.values()))})


def permute_edge_mask(mask, corners):
    """Transport edge attributes, not corner attributes, through any permutation."""
    mask = np.asarray(mask); corners = np.asarray(corners)
    if mask.dtype != np.bool_ or mask.shape != corners.shape or corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError('Edge-mask/corner permutation dimensions differ')
    if not np.array_equal(np.sort(corners, axis=1), np.tile(np.arange(3), (len(corners), 1))):
        raise ValueError('Corner permutation is not bijective')
    edge_order = EDGE_LOOKUP[corners[:, EDGES[:, 0]], corners[:, EDGES[:, 1]]]
    return mask[np.arange(len(mask))[:, None], edge_order]


def _condition(path, expected, uid, n, k, region_ids=None):
    p = resolve(path)
    if sha(p) != expected: raise ValueError('C source hash differs: ' + str(p))
    # Explicit C-only allowlist: no free_source_ids, Y, source interfaces or quality.
    keys = ('C', 'C_original', 'C_source_vertex_ids', 'C_original_source_vertex_ids',
            'C_source_corner_permutations', 'C_original_source_face_ids', 'source_face_ids',
            'region_source_face_ids', 'N', 'K', 'parent_uid', 'condition_type')
    with np.load(p, allow_pickle=False) as z: d = {key: z[key].copy() for key in keys}
    if (int(d['N']), int(d['K']), str(d['parent_uid'].item()), str(d['condition_type'].item())) != (n, k, uid, 'SEAT_FULL'):
        raise ValueError('Actual condition identity is not the registered FULL request')
    c, original = d['C'], d['C_original']; ids = d['C_source_vertex_ids']; order = d['C_source_corner_permutations']
    if c.dtype != np.float32 or c.shape != (k, 3, 3) or not np.isfinite(c).all():
        raise ValueError('C must be finite FP32[K,3,3]')
    if not np.array_equal(np.sort(order, axis=1), np.tile(np.arange(3), (k, 1))):
        raise ValueError('Saved original-to-canonical corner map is invalid')
    if not bits(original[np.arange(k)[:, None], order], c):
        raise ValueError('Saved corner mapping does not recover actual C')
    if not np.array_equal(d['C_original_source_vertex_ids'][np.arange(k)[:, None], order], ids):
        raise ValueError('Saved corner mapping does not transport vertex identities')
    faces = d['source_face_ids']
    if faces.shape != (k,) or len(set(faces.tolist())) != k or not np.array_equal(faces, d['region_source_face_ids']):
        raise ValueError('FULL condition must equal its complete region source face set')
    if region_ids is not None and not np.array_equal(faces, np.asarray(region_ids, np.int64)):
        raise ValueError('Actual FULL region differs from recorded region identity')
    mask, stats = boundary_mask(ids)
    original_mask, _ = boundary_mask(d['C_original_source_vertex_ids'])
    if not np.array_equal(mask, permute_edge_mask(original_mask, order)):
        raise ValueError('Canonical boundary mask does not match original source topology')
    # Coordinate validation uses identities; coincident different IDs stay distinct.
    seen = {}
    for vertex, coord in zip(ids.reshape(-1), c.reshape(-1, 3)):
        key = int(vertex)
        if key in seen and not bits(seen[key], coord): raise ValueError('One C vertex ID has inconsistent coordinates')
        seen[key] = coord.copy()
    result = dict(parent_id=uid, N=n, K=k, condition_type='SEAT_FULL', condition_path=str(p), condition_sha256=expected,
                  source_face_ids=faces.tolist(), region_source_face_ids=d['region_source_face_ids'].tolist(),
                  C_source_vertex_ids=ids.tolist(), C_source_corner_permutations=order.tolist(), C=c.tolist(),
                  C_sha256=array_sha(c), boundary_edge_mask=mask.tolist(), boundary_edge_mask_sha256=array_sha(mask),
                  topology=stats, topology_scope='C_ONLY_UNDIRECTED_SOURCE_VERTEX_ID_INCIDENCE',
                  original_mapping_verified=True, coordinate_welding=False, source_Y_used=False)
    return result


def prepare():
    """Verify 47 training C and 24 actual FULL requests; freeze C-only sidecar."""
    regpath = BASE / 'registration.json'; reg = read(regpath)
    datapath = resolve(reg['source_data'])
    if sha(datapath) != reg['source_data_sha256']: raise ValueError('Baseline source data hash differs')
    data = read(datapath); sidepath = BASE / 'audits/metadata_identity_sidecar.json'; side = read(sidepath)
    if side['source_registration_sha256'] != sha(regpath) or side['affected_requests'] != 8:
        raise ValueError('Baseline FULL metadata identity sidecar does not match')
    dependencies = {str(regpath): sha(regpath), str(datapath): sha(datapath), str(sidepath): sha(sidepath)}
    training = {}; requests = {}
    for tid in reg['train_task_ids']:
        t = data['tasks'][tid]; p = data['parents'][t['uid']]
        row = _condition(t['npz'], t['sha256'], t['uid'], t['N'], t['K'], t['region_source_face_ids'])
        if not np.array_equal(row['source_face_ids'], t['source_face_ids']): raise ValueError('Training source C face IDs differ')
        parentpath = resolve(p['npz'])
        if sha(parentpath) != p['sha256']: raise ValueError('Training source parent hash differs')
        indices = np.asarray(row['source_face_ids'], np.int64)
        # These arrays are indexed to C immediately; no Y geometry or labels enters topology.
        with np.load(parentpath, allow_pickle=False) as z:
            c = z['full_target'].reshape(t['N'], 3, 3)[indices]
            vids = z['source_face_vertex_ids'][indices]
            mapped = z['model_vertices'][vids]
        if not bits(c, np.asarray(row['C'], np.float32)) or not bits(mapped, c): raise ValueError('Training C does not match parent indexed geometry')
        if not np.array_equal(vids, row['C_source_vertex_ids']): raise ValueError('Training source C vertex map differs')
        row.update(task_id=tid, parent_path=str(parentpath), parent_sha256=p['sha256'])
        training[tid] = row; dependencies[str(parentpath)] = p['sha256']; dependencies[row['condition_path']] = row['condition_sha256']
    for q in reg['requests']:
        # condition_source, actual N/K and actual FULL region are authoritative.
        # Inherited task_id/source_face_ids/npz/free_source_ids are intentionally ignored.
        row = _condition(q['condition_source'], q['condition_source_sha256'], q['parent_id'], q['N'], q['K'], q.get('region_source_face_ids'))
        path = resolve(q['input_path'])
        if sha(path) != q['input_sha256']: raise ValueError('Fixed evaluation input file changed')
        with np.load(path, allow_pickle=False) as z:
            c = z['C'].copy(); n = int(z['N']); k = int(z['K'])
        if (n, k) != (q['N'], q['K']) or not bits(c, np.asarray(row['C'], np.float32)) or array_sha(c) != q['C_sha256']:
            raise ValueError('Actual evaluation C differs from FULL source geometry')
        full_mask = np.zeros((n, 3), bool); full_mask[:k] = np.asarray(row['boundary_edge_mask'], bool)
        row.update(request_id=q['request_id'], condition_id=q['condition_id'], split=q['split'],
                   input_path=str(path), input_sha256=q['input_sha256'], model_mask_sha256=array_sha(full_mask),
                   ignored_stale_fields=[name for name in ('task_id','source_face_ids','npz','free_source_ids') if name in q],
                   inherited_HALF_metadata_ignored=q['parent_id'].startswith(('2d08a6', '610b16')))
        requests[q['request_id']] = row
        dependencies[str(path)] = q['input_sha256']; dependencies[row['condition_path']] = row['condition_sha256']
    if len(training) != 47 or len(requests) != 24: raise ValueError('Frozen baseline scope changed')
    value = dict(schema=VERSION, status='FROZEN', training=training, requests=requests, source_files=dependencies,
                 edge_order=EDGES.tolist(), network_input='bool[B,N,3] only; numerical source identities are not features',
                 boundary_rule='incidence_C(undirected source vertex-ID pair) == 1',
                 source_Y_used=False, source_interface_labels_used=False, coordinate_welding=False,
                 metadata_policy='Actual FULL condition_source NPZ and request C; legacy HALF descriptors never used')
    digest = freeze(MANIFEST, value)
    audit = dict(status='PASS', topology_manifest_path=str(MANIFEST), topology_manifest_sha256=digest,
                 training_conditions=len(training), evaluation_requests=len(requests), evaluation_conditions=len({r['condition_id'] for r in requests.values()}),
                 inherited_HALF_requests_verified=sum(r['inherited_HALF_metadata_ignored'] for r in requests.values()),
                 training_nonmanifold_C=sum(r['topology']['nonmanifold_edges'] > 0 for r in training.values()),
                 evaluation_nonmanifold_C=sum(r['topology']['nonmanifold_edges'] > 0 for r in requests.values()),
                 source_files=len(dependencies), model_calls=0, updates=0, new_OT=0, noise_draws=0,
                 implementation_sha256=sha(__file__))
    freeze(OUT / 'audits/topology_prepare.json', audit)
    return audit


class TopologyInputs:
    def __init__(self, path=MANIFEST, verify_sources=True):
        self.path = resolve(path); self.manifest = read(self.path); self.manifest_sha256 = sha(self.path)
        if self.manifest.get('schema') != VERSION or self.manifest.get('status') != 'FROZEN': raise ValueError('Unfrozen topology manifest')
        if verify_sources:
            for p, h in self.manifest['source_files'].items():
                if sha(p) != h: raise ValueError('Topology dependency changed: ' + p)
        self.training = self.manifest['training']; self.requests = self.manifest['requests']

    @lru_cache(maxsize=128)
    def _training(self, task_id):
        r = self.training[task_id]
        return r, np.asarray(r['C'], np.float32), np.asarray(r['source_face_ids'], np.int64), np.asarray(r['boundary_edge_mask'], bool)

    def for_training(self, sample):
        """Return bool[N,3] aligned through final face and corner permutations."""
        row, c, face_ids, source_mask = self._training(sample['task_id'])
        n, k = int(sample['y']), int(sample['K']); known = np.asarray(sample['known_mask'])
        valid = np.asarray(sample['valid_mask'])
        if (n, k, sample['object_id']) != (row['N'], row['K'], row['parent_id']): raise ValueError('Training topology identity differs')
        if known.dtype != np.bool_ or known.shape != (n,) or known.sum() != k or np.any(known & ~valid): raise ValueError('Invalid known mask')
        ids = np.asarray(sample['source_face_ids'])[known]; corners = np.asarray(sample['corner_permutations'])[known]
        if not np.array_equal(np.sort(ids), np.sort(face_ids)): raise ValueError('Actual known source IDs differ from C-only topology')
        lookup = {int(v): i for i, v in enumerate(face_ids)}; order = np.asarray([lookup[int(v)] for v in ids], np.int64)
        expected = c.copy(); expected[:, :, 0] *= np.float32(sample['alpha'])
        expected = expected[order][np.arange(k)[:, None], corners]
        actual = np.asarray(sample['context']).reshape(n, 3, 3)[known]
        if not bits(expected, actual) or not bits(np.asarray(sample['xt']).reshape(n, 3, 3)[known], actual):
            raise ValueError('C topology/geometry transport differs after width and corner permutation')
        mask = np.zeros((n, 3), bool); mask[known] = permute_edge_mask(source_mask[order], corners)
        if mask[~known].any() or int(mask.sum()) != int(source_mask.sum()): raise ValueError('Boundary-mask transport changed edge count')
        return mask

    def attach(self, sample):
        mask = self.for_training(sample)
        sample['boundary_edge_mask'] = mask
        sample['boundary_edge_mask_sha256'] = array_sha(mask)
        sample['boundary_topology_manifest_sha256'] = self.manifest_sha256
        return sample

    def for_request(self, request_id):
        """Canonical C occupies first K slots in the frozen inference input."""
        row = self.requests[request_id]; mask = np.zeros((row['N'], 3), bool)
        mask[:row['K']] = np.asarray(row['boundary_edge_mask'], bool)
        if array_sha(mask) != row['model_mask_sha256']: raise ValueError('Evaluation boundary mask changed')
        return mask


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
