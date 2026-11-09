"""Read-only CPU qualification of source-index Bench part conditions.

No model imports, source repair, welding, simplification, or semantic inference.
This module separates usable C geometry from source-only SEAM supervision.
Geometric PASS is never by itself an ACTIVE semantic part annotation.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import subprocess
import time
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'bench_parts_source_quality_v1'
L = float(np.sqrt(3) * 2 / .3762)
FROZEN_HASHES = {
    'intersections.py': '23a2e8c8deefcceb51ace60ad8d8a58d352d7c38bdcba2e8f85d6ceabd447d86',
    'seams.py': '9ce6927710d80eedec19aa10e2100ade582cf00e6942e9de4ef987618a005e4b',
}


def local_path(value):
    value = str(value)
    if os.name == 'nt' and value.startswith('/mnt/'):
        value = value[5].upper() + ':/' + value[7:]
    elif os.name != 'nt' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')
    return Path(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text('utf-8') != text:
            raise FileExistsError('Refuse overwriting changed evidence: ' + str(path))
        return
    path.write_text(text, encoding='utf-8')


def detector():
    """Load only hash-verified NumPy geometry code, no model package imports."""
    folder = local_path('E:/MeshFlow-Control-artifacts/experiments/chair_user_condition_data_audit_v1/geometry/source_snapshot/src/meshflow_control/evaluation')
    for name, expected in FROZEN_HASHES.items():
        if sha(folder / name) != expected:
            raise ValueError('Frozen detector changed: ' + str(folder / name))
    package = '_bench_parts_frozen_geometry_v1'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
    return importlib.import_module(package + '.intersections')


def edgebook(faces):
    book = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            u, v = int(face[j]), int(face[(j + 1) % 3])
            book[tuple(sorted((u, v)))].append((fi, u, v))
    return book


def components(nodes, edges):
    adj = {int(i): set() for i in nodes}
    for a, b in edges:
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    unseen, groups = set(adj), []
    while unseen:
        stack, found = [min(unseen)], []
        while stack:
            i = stack.pop()
            if i not in unseen:
                continue
            unseen.remove(i); found.append(i)
            stack.extend(sorted(adj[i] & unseen, reverse=True))
        groups.append(sorted(found))
    return groups


def topology(faces):
    book = edgebook(faces)
    open_edges, nonmanifold, same_winding, adjacency = [], [], [], []
    for edge, rows in sorted(book.items()):
        if len(rows) == 1:
            open_edges.append(rows[0])
        elif len(rows) > 2:
            nonmanifold.append(dict(vertex_ids=list(edge), face_ids=[r[0] for r in rows]))
        elif rows[0][1:] != rows[1][1:][::-1]:
            same_winding.append(dict(vertex_ids=list(edge), face_ids=[r[0] for r in rows]))
        for row in rows[1:]:
            adjacency.append((rows[0][0], row[0]))
    vertex_bad = []
    for vid in np.unique(faces):
        incident = np.flatnonzero(np.any(faces == vid, axis=1))
        links = [tuple(int(v) for v in faces[fi] if v != vid) for fi in incident]
        if any(len(pair) != 2 for pair in links):
            vertex_bad.append(dict(vertex_id=int(vid), reason='DEGENERATE_INCIDENT_FACE'))
            continue
        nodes = sorted({v for pair in links for v in pair})
        groups = components(nodes, links)
        degree = Counter(v for pair in links for v in pair)
        if len(groups) != 1 or any(d > 2 for d in degree.values()) or sum(d == 1 for d in degree.values()) not in (0, 2):
            vertex_bad.append(dict(vertex_id=int(vid), link_components=len(groups), link_degrees=dict(degree)))
    return dict(open_edges=open_edges, nonmanifold_edges=nonmanifold,
                same_winding_double_edges=same_winding, nonmanifold_vertices=vertex_bad,
                edge_connected_components=components(range(len(faces)), adjacency))


def duplicates(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(sorted(row))].append(i)
    return [group for group in groups.values() if len(group) > 1]


class SourceAuditor:
    def __init__(self, arrays, source_sha256=None):
        self.p = arrays
        self.source_sha256 = source_sha256
        self.f = np.asarray(arrays['source_faces'], np.int64)
        self.v = np.asarray(arrays['model_vertices'])
        self.full = np.asarray(arrays['full_target'])
        self.canonical = np.asarray(arrays['source_face_vertex_ids'], np.int64)
        self.perm = np.asarray(arrays['source_corner_permutations'], np.int64)
        if self.v.ndim != 2 or self.v.shape[1] != 3 or not np.isfinite(self.v).all():
            raise ValueError('Invalid source vertices')
        if self.f.ndim != 2 or self.f.shape[1] != 3 or self.f.min() < 0 or self.f.max() >= len(self.v):
            raise ValueError('Invalid source IDs')
        if not np.array_equal(self.v[self.canonical], self.full):
            raise ValueError('Source model coordinates do not reproduce full_target')
        if not np.array_equal(np.take_along_axis(self.f, self.perm, axis=1), self.canonical):
            raise ValueError('Source corner permutation does not reproduce canonical IDs')
        if not np.array_equal(np.sort(self.perm, axis=1), np.tile(np.arange(3), (len(self.f), 1))):
            raise ValueError('Invalid reversible corner permutation')
        if not np.array_equal(np.asarray(arrays['pre_ot_full']) * np.float32(2), self.full):
            raise ValueError('Original OT/model x2 normalization changed')
        self.tri = self.v[self.f].astype(np.float64)
        self.lo, self.hi = self.tri.min(1), self.tri.max(1)
        self.book = edgebook(self.f)
        self.pair_cache = {}
        self.calls = Counter()
        self.D = detector()

    def pair(self, i, j):
        key = tuple(sorted((int(i), int(j))))
        if key in self.pair_cache:
            self.calls['reused_pair_cache'] += 1
            return self.pair_cache[key]
        i, j = key
        self.calls['unique_pair_requests'] += 1
        if np.any(self.lo[i] > self.hi[j] + L * 1e-9) or np.any(self.lo[j] > self.hi[i] + L * 1e-9):
            result = dict(classification='disjoint', illegal=False, verified=True, aabb_disjoint=True)
            self.calls['aabb_disjoint'] += 1
        else:
            result = self.D.verified_pair(self.tri[i], self.tri[j], L)
            self.calls['finite_detector_calls'] += 1
        self.pair_cache[key] = result
        return result

    def candidate(self, known, metadata=None):
        metadata = metadata or {}
        raw_known = np.asarray(known)
        if raw_known.ndim != 1 or raw_known.dtype.kind not in 'iu':
            raise ValueError('source_face_ids must be a one-dimensional integer array')
        known = raw_known.astype(np.int64)
        n, k = len(self.f), len(known)
        if k == 0 or len(set(map(int, known))) != k or known.min() < 0 or known.max() >= n:
            raise ValueError('C source face IDs must be unique and in range')
        free = np.setdiff1d(np.arange(n), known)
        reconstructed = np.empty_like(self.full)
        reconstructed[known] = self.full[known]; reconstructed[free] = self.full[free]
        assert np.array_equal(reconstructed, self.full)
        assert not np.intersect1d(known, free).size
        vids = np.unique(self.f[known])
        lut = {int(v): i for i, v in enumerate(vids)}
        cf = np.asarray([[lut[int(v)] for v in face] for face in self.f[known]], np.int64)
        cm = np.asarray([[lut[int(v)] for v in face] for face in self.canonical[known]], np.int64)
        assert np.array_equal(self.v[vids][cm], self.full[known])
        restored = np.take_along_axis(self.full[known], np.argsort(self.perm[known])[:, :, None], axis=1)
        assert np.array_equal(restored, self.v[self.f[known]])
        topo = topology(cf)
        ctri = self.tri[known]
        areas = np.linalg.norm(np.cross(ctri[:, 1] - ctri[:, 0], ctri[:, 2] - ctri[:, 0]), axis=1) / 2
        degenerate = np.flatnonzero(areas <= 1e-12 * L * L).tolist()
        duplicate_ids = duplicates([tuple(int(v) for v in face) for face in self.f[known]])
        duplicate_coordinates = duplicates([tuple(tuple(float(x) for x in vertex) for vertex in face) for face in ctri])
        counts, bad_pairs, unresolved_pairs = Counter(), [], []
        for i in range(k):
            for j in range(i + 1, k):
                q = self.pair(known[i], known[j]); counts[q['classification']] += 1
                row = dict(C_pair=[i, j], source_face_ids=[int(known[i]), int(known[j])], detector=q)
                if q.get('illegal') is True:
                    bad_pairs.append(row)
                if not q.get('verified', False):
                    unresolved_pairs.append(row)

        # The following labels are source-completion diagnostics, never inference inputs.
        labels, owner, alias, triples, interface_rows = [], [], [], [], []
        by_free_face = defaultdict(list)
        ks = set(map(int, known))
        for ei, (ci, u, w) in enumerate(topo['open_edges']):
            su, sw = int(vids[u]), int(vids[w])
            entries = self.book[tuple(sorted((su, sw)))]
            outside = [row for row in entries if row[0] not in ks]
            reason, cls, fi, triple = 'NOT_SINGLE_C_SINGLE_FREE_INCIDENCE', 'NA', -1, [-1] * 3
            if len(entries) == 2 and len(outside) == 1:
                jf, x, y = outside[0]
                if (x, y) != (sw, su):
                    reason = 'SOURCE_WINDING_NOT_OPPOSITE'
                else:
                    q = self.pair(known[ci], jf); cls = q['classification']
                    reason = 'NOT_VERIFIED_LEGAL_CONTACT'
                    if cls == 'shared_edge_contact' and q.get('verified'):
                        third = [int(v) for v in self.f[jf] if v not in (su, sw)]
                        if len(third) == 1:
                            reason, fi, triple = 'PASS', int(jf), [sw, su, third[0]]
                            by_free_face[fi].append(ei)
            owner.append(fi); triples.append(triple); labels.append(False); alias.append(False)
            interface_rows.append(dict(edge_id=ei, C_face_id=int(ci), source_C_face_id=int(known[ci]),
                source_u=su, source_v=sw, reason=reason, classification=cls,
                source_incident_face_ids=[row[0] for row in entries], role='NULL_UNSUPPORTED'))
        for fi, edges in by_free_face.items():
            owner_edge = min(edges); labels[owner_edge] = True
            for ei in edges:
                alias[ei] = ei != owner_edge
                interface_rows[ei].update(owner_edge=owner_edge, owner_source_face_id=fi,
                    third_vertex_in_C=triples[ei][2] in lut, role='ALIAS' if alias[ei] else 'OWNER')
        b = len(labels)
        reasons = []
        if k < 2: reasons.append('K_LT_2')
        if not len(free): reasons.append('NO_FREE_COMPLETION')
        if degenerate: reasons.append('C_DEGENERATE_FACES')
        if duplicate_ids or duplicate_coordinates: reasons.append('C_EXACT_DUPLICATE_FACES')
        if topo['nonmanifold_edges']: reasons.append('C_NONMANIFOLD_EDGES')
        if topo['nonmanifold_vertices']: reasons.append('C_NONMANIFOLD_VERTICES')
        if topo['same_winding_double_edges']: reasons.append('C_SOURCE_WINDING_CONFLICT')
        if bad_pairs: reasons.append('C_ILLEGAL_FINITE_TRIANGLE_PAIRS')
        if unresolved_pairs: reasons.append('C_GEOMETRY_UNRESOLVED')
        seam_reasons = []
        if not b: seam_reasons.append('NO_OPEN_BOUNDARY')
        if b > len(free): seam_reasons.append('BOUNDARY_SLOTS_EXCEED_FREE_BUDGET')
        if any(alias): seam_reasons.append('SOURCE_ALIAS_REQUIRES_EXPLICIT_DECODER')
        if any(ok and triples[i][2] in lut for i, ok in enumerate(labels)):
            seam_reasons.append('OWNER_THIRD_VERTEX_REFERENCES_C')
        if reasons: seam_reasons.append('C_GEOMETRY_NOT_QUALIFIED')
        caveats = []
        if len(topo['edge_connected_components']) > 1:
            caveats.append('MULTICOMPONENT_INPUT_REQUIRES_COMPOSITION_REVIEW')
        if not b:
            caveats.append('CLOSED_PART_IS_VALID_NATIVE_INPUT_NO_SEAM_BOUNDARY')
        source_duplicate_count = sum(len(g) - 1 for g in duplicates([tuple(int(v) for v in face) for face in self.f]))
        full_area = np.linalg.norm(np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0]), axis=1).sum() / 2
        report = dict(schema=VERSION, candidate_id=metadata.get('candidate_id'), parent_id=metadata.get('parent_id'),
            group=metadata.get('group'), proposed_part_type=metadata.get('proposed_part_type'), source_sha256=self.source_sha256,
            N=n, K=k, free_count=len(free), source_face_ids=known.tolist(), source_vertex_ids=vids.tolist(),
            mapping=dict(C_plus_Y_source_ID_partition='PASS_EXACT', model_coordinates='PASS_BITWISE',
                corner_permutation_inverse='PASS_BITWISE', source_geometry_modified=False),
            geometry_status='QUARANTINE' if reasons else 'PASS_PENDING_SEMANTIC_REVIEW', reasons=reasons,
            native_eligible_after_semantic_review=not reasons, caveats=caveats,
            current_bench_entry_consumable=not reasons and b > 0 and b <= len(free),
            current_bench_entry_limitations=(['Existing bench edge_features rejects closed C'] if not b else []) +
                (['Existing bench edge_features reserves one slot per boundary edge'] if b > len(free) else []),
            seam_applicability='NA_OR_UNSUPPORTED' if seam_reasons else 'SUPPORTED_CURRENT_SINGLE_EDGE_SLOT',
            seam_reasons=seam_reasons, topology=topo, area_model=float(areas.sum()),
            area_fraction_source_face_area=float(areas.sum() / full_area),
            degenerate_C_face_ids=degenerate, duplicate_C_ID_face_groups=duplicate_ids,
            duplicate_C_coordinate_face_groups=duplicate_coordinates,
            C_pair_classifications=dict(counts), illegal_C_pairs=bad_pairs, unresolved_C_pairs=unresolved_pairs,
            source_owner_count=sum(labels), source_alias_count=sum(alias), source_null_unsupported_count=sum(fi < 0 for fi in owner),
            source_boundary_edge_count=b, source_interface_rows=interface_rows,
            source_target_risks=dict(source_exact_duplicate_ID_faces=source_duplicate_count,
                full_Y_finite_intersection_scan='NOT_RUN', all_CF_intersection_scan='NOT_RUN',
                scope='Only C-C pairs and source-index adjacent C/Y boundary pairs tested; this is not certification of clean Y'),
            policy=dict(nonplanar_allowed=True, closed_parts_allowed=True, no_repair=True,
                no_source_outward_normal_certification=True, min_angle='DESCRIPTIVE_NOT_A_REJECTION_RULE',
                activation='Requires independent actual part review; quality alone does not assign semantics',
                source_labels='TRAINING_OR_DIAGNOSTIC_ONLY_NOT_INFERENCE'))
        supervision = dict(known_ids=known, free_ids=free, C_vertices=self.v[vids], C_faces_original=cf, C_faces_model=cm,
            C_source_vertex_ids=vids, C_source_corner_permutations=self.perm[known],
            occupied_labels=np.asarray(labels, bool), edge_owner_face=np.asarray(owner, np.int64),
            edge_source_triple=np.asarray(triples, np.int64).reshape(b, 3),
            source_alias_labels=np.asarray(alias, bool), source_supported_labels=np.asarray(owner, np.int64) >= 0,
            edge_records=np.asarray(topo['open_edges'], np.int64).reshape(b, 3), N=np.asarray(n), K=np.asarray(k))
        return report, supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest_path, out = local_path(args.manifest), local_path(args.out)
    manifest = json.loads(manifest_path.read_text('utf-8-sig'))
    rows = manifest if isinstance(manifest, list) else manifest['rows']
    manifest_sha = sha(manifest_path)
    if (out / 'summary.json').exists():
        old = json.loads((out / 'summary.json').read_text('utf-8'))
        if old['manifest_sha256'] != manifest_sha or old['implementation_sha256'] != sha(__file__):
            raise ValueError('Completed audit input or implementation changed')
        for row in old['rows']:
            assert sha(out / row['quality_file']) == row['quality_sha256']
            assert sha(out / row['supervision_file']) == row['supervision_sha256']
        print(json.dumps(dict(status='VERIFIED_COMPLETED_CACHE', candidates=len(old['rows'])))); return
    started = time.perf_counter()
    from psutil import virtual_memory, cpu_percent
    resource = dict(cpu_count=os.cpu_count(), available_RAM_bytes=virtual_memory().available,
        CPU_utilization_percent=cpu_percent(interval=.2), process_id=os.getpid(), workers=1, numeric_threads=1,
        CUDA_calls=0, model_calls=0, before_scan=True)
    for name, query in (('gpu', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu'),
                        ('gpu_processes', '--query-compute-apps=pid,process_name,used_memory')):
        p = subprocess.run(['nvidia-smi', query, '--format=csv'], capture_output=True, text=True, check=False)
        resource[name] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    write_new(out / 'resources_before.json', resource)
    by_source = defaultdict(list)
    for row in rows: by_source[str(row['source_case_path'])].append(row)
    results, ledger, protected = [], Counter(), {}
    for source_path, source_rows in sorted(by_source.items()):
        path = local_path(source_path); h = sha(path)
        if any(row['source_case_sha256'] != h for row in source_rows):
            raise ValueError('Source cache identity differs: ' + str(path))
        protected[str(path)] = h
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        auditor = SourceAuditor(arrays, h)
        for row in source_rows:
            cid = row['candidate_id']
            report, supervision = auditor.candidate(row['source_face_ids'], row)
            name = 'conditions/' + cid + '.json'; supname = 'supervision_only/' + cid + '.npz'
            write_new(out / name, report)
            target = out / supname; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with np.load(target, allow_pickle=False) as old:
                    assert set(old.files) == set(supervision)
                    assert all(np.array_equal(old[key], value) for key, value in supervision.items())
            else:
                with target.open('xb') as handle: np.savez_compressed(handle, **supervision)
            results.append(dict(candidate_id=cid, parent_id=row['parent_id'], group=row['group'],
                proposed_part_type=row['proposed_part_type'], N=report['N'], K=report['K'],
                geometry_status=report['geometry_status'], reasons=report['reasons'], caveats=report['caveats'],
                seam_applicability=report['seam_applicability'], seam_reasons=report['seam_reasons'],
                quality_file=name, quality_sha256=sha(out / name), supervision_file=supname, supervision_sha256=sha(target)))
        ledger.update(auditor.calls)
        print(json.dumps(dict(parent=source_rows[0]['parent_id'], completed=len(results), calls=dict(ledger))), flush=True)
    for path, expected in protected.items():
        if sha(path) != expected: raise ValueError('Original source changed during audit')
    counts = dict(Counter(row['geometry_status'] for row in results))
    summary = dict(schema=VERSION, manifest_sha256=manifest_sha, implementation_sha256=sha(__file__),
        candidates=len(results), parents=len(by_source), counts=counts, rows=results,
        ledger=dict(ledger, GPU_calls=0, model_calls=0, training_updates=0, source_modifications=0, workers=1, numeric_threads=1),
        frozen_detector_hashes=FROZEN_HASHES, protected_source_hashes=protected,
        wall_seconds=time.perf_counter() - started)
    write_new(out / 'summary.json', summary)
    print(json.dumps(dict(status='COMPLETE', counts=counts, wall_seconds=summary['wall_seconds'])))


if __name__ == '__main__':
    main()
