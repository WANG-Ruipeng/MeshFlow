"""CPU-only frozen quality mathematics and source-corner connectivity accounts.

This adapter does not construct candidates, bind vertices, orient faces, run
GEO_C or call models. Original score functions are imported unchanged. The
caller owns immutable inputs, dispatch, destinations and attempt accounting.

Public API:
  helpers() -> original geometry/control/shape modules and A/G/H helpers.
  pair_scan(first, second, L, **kwargs) -> original verified pair scan.
  score_arrays(vertices, faces, K, canonical_C, source_C_vertex_ids,
               frozen_query_record, original_N, source_face_ids=None)
      -> JSON-safe scores plus NumPy ``vectors``; no filesystem writes.
  connectivity_ledger(vertices, faces, K, pairs, source_face_ids,
                      source_corner_ids=None) -> all edge records.
  compare_ledgers(before, after, before_pairs, after_pairs,
                  affected_source_face_ids=None) -> detailed fates + totals.

source_corner_ids is [N,3], giving each current corner's ORIGINAL local corner
identity (0,1,2), or its original globally unique identity. Carry its permutation
through winding changes. Never replace it with current indices after a flip.
"""
from __future__ import annotations

import collections
import functools
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
L = float(np.sqrt(3.) * 2 / .3762)
VERSION = 'r2_partial_edge_binding_frozen_metrics_source_corner_ledger_v1'
ILLEGAL = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def resolve(path):
    raw = str(path).replace('\\', '/')
    if os.name != 'nt' and len(raw) > 2 and raw[1] == ':':
        raw = '/mnt/' + raw[0].lower() + raw[2:]
    elif os.name == 'nt' and raw.startswith('/mnt/'):
        raw = raw[5].upper() + ':/' + raw[7:]
    return Path(raw)


def _module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, resolve(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@functools.lru_cache(None)
def helpers():
    if 'torch' in sys.modules:
        raise RuntimeError('Partial binding scoring requires a model-free CPU process')
    saved = {key: os.environ.get(key) for key in THREAD_KEYS}
    try:
        old = _module('_partial_binding_original_math_adapter',
                      ROOT / 'r2/pilots/user_condition_finetune_v1/evaluation.py')
        old.LEGACY = resolve(old.LEGACY)
        result = dict(old.cpu_math(), old=old)
        for key, path in (
            ('A', 'r2/tools/avmr_pmr_v1/run.py'),
            ('G', 'r2/pilots/anchor_aware_denoiser_v1/evaluation.py'),
            ('H', 'r2/tools/c_restore_coupling_v1/evaluation.py'),
            ('Q', 'r2/tools/postprocess_avmr_pmr_v1/quality.py')):
            result[key] = _module('_partial_binding_original_' + key, ROOT / path)
        if 'torch' in sys.modules:
            raise RuntimeError('CPU metric dependency imported model framework')
        return result
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def definition_hashes():
    h = helpers()
    paths = [Path(__file__), Path(h['old'].__file__)]
    paths.extend(Path(h[key].__file__) for key in ('A', 'G', 'H', 'Q'))
    paths.extend(h['old'].LEGACY / name for name in
                 ('__init__.py', 'control.py', 'geometry.py', 'shape.py', 'surface.py',
                  'quality.py', 'seams.py', 'pair_audit.py', 'intersections.py', 'hashing.py'))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def pair_scan(first, second, length=L, **kwargs):
    """Original detector; independent verification is mandatory by default."""
    kwargs.setdefault('independent', True)
    return helpers()['geometry'].pair_scan(first, second, length, **kwargs)


def shape_quality(triangles, length=L):
    return helpers()['shape'].shape_quality(triangles, length)


def illegal_pair_sets(pairs, source_face_ids, K):
    """Illegal CF and FF pairs keyed by original RAW face IDs, not row numbers."""
    ids = np.asarray(source_face_ids, dtype=np.int64)
    result = {}
    for kind in ('CF', 'FF'):
        found = set()
        for row in pairs[kind]['contacts']:
            if row['classification'] not in ILLEGAL:
                continue
            i, j = map(int, row['pair'])
            if kind == 'CF':
                found.add((int(ids[i]), int(ids[K+j])))
            else:
                found.add(tuple(sorted((int(ids[K+i]), int(ids[K+j])))))
        result[kind] = found
    return result


def _same_unordered_faces(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return all(sorted(tuple(map(int, x)) for x in aa.view(np.uint32)) ==
               sorted(tuple(map(int, x)) for x in bb.view(np.uint32))
               for aa, bb in zip(a, b))


def score_arrays(vertices, faces, K, canonical_C, source_C_vertex_ids,
                 frozen_query_record, original_N, source_face_ids=None):
    """Apply original FULL metrics to an independently registered bound POST.

    C in ``vertices[faces[:K]]`` may be source-oriented; ``canonical_C`` and its
    aligned source_C_vertex_ids are the frozen inference/source metadata arrays.
    frozen_query_record is the full original registration query object, with
    path, sha256, array_sha256 and identity. No query is sampled or transformed.
    The returned ``vectors`` remain NumPy arrays for NPZ persistence.
    """
    tick = time.perf_counter()
    h = helpers()
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    C = np.ascontiguousarray(canonical_C)
    K = int(K)
    if (vertices.dtype != np.float32 or faces.dtype.kind not in 'iu'
            or faces.ndim != 2 or faces.shape[1:] != (3,)
            or vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('Expected finite FP32 vertices and indexed triangle faces')
    tri = np.ascontiguousarray(vertices[faces])
    if not np.isfinite(tri).all() or C.dtype != np.float32 or C.shape != (K, 3, 3):
        raise ValueError('Invalid mesh/C coordinates')
    if not _same_unordered_faces(tri[:K], C):
        raise ValueError('Binding altered C coordinates or per-face identity')
    F = tri[K:]
    reference = h['old'].load_queries(frozen_query_record)
    if len(F):
        metrics, cf, ff = h['geometry'].scan_geometry(C, F, L)
    else:
        cf, ff = h['Q']._empty_scans(K)
        metrics = {name + '_' + field: value for name in ('CF', 'FF') for field, value in (
            ('proper_pair_count', 0), ('coplanar_pair_count', 0),
            ('proper_affected_free_fraction', None), ('unresolved_or_degenerate_pair_count', 0))}
        metrics.update(FF_illegal_pair_count=0, FF_illegal_affected_free_fraction=None)
    trusted = h['control'].trusted_cf(C, F, L, cf_scan=cf)
    im, vectors = h['control'].conflict_metrics(F, {'C': C, 'L0': L}, reference, trusted)
    if abs(im['ConflictCov_C'] - im['ConflictCov_C_cross'] - im['ConflictCov_C_dup']
           + im['ConflictCov_C_cross_and_dup']) > 1e-12:
        raise AssertionError('Original I union identity failed')
    metrics.update(im)
    cf_affected = {int(row['pair'][1]) for row in cf['contacts'] if row['classification'] in ILLEGAL}
    metrics.update(CF_illegal_pair_count=sum(row['classification'] in ILLEGAL for row in cf['contacts']),
                   CF_illegal_affected_free_fraction=len(cf_affected)/len(F) if len(F) else None)
    shape = {name: h['shape'].shape_quality(part, L) for name, part in
             (('C', C), ('free', F), ('whole', tri))}
    fq = shape['free']
    metrics.update(free_minimum_angle_lt5_fraction=fq['min_angle_lt5_fraction'],
                   free_q_gt10_fraction=fq['q_actual_gt10_fraction'],
                   free_q_mean=fq['q_actual']['mean'], free_angle_p05_degrees=fq['angle_p05_degrees'],
                   free_near_degenerate_count=fq['near_degenerate_area_count'],
                   free_exact_zero_area_count=fq['exact_zero_area_count'],
                   free_total_surface_area=fq['total_area'], actual_free_faces=len(F),
                   deleted_free_faces=int(original_N)-len(tri),
                   C_total_surface_area=shape['C']['total_area'])
    metrics.update(h['geometry'].topology(tri, K))
    for label, prefix in (('C', 'C'), ('whole', 'whole')):
        metrics[prefix+'_near_degenerate_count'] = shape[label]['near_degenerate_area_count']
        metrics[prefix+'_exact_zero_area_count'] = shape[label]['exact_zero_area_count']
    metrics.update(whole_minimum_angle_lt5_fraction=shape['whole']['min_angle_lt5_fraction'],
                   whole_total_surface_area=shape['whole']['total_area'])
    pairs = {'CF': cf, 'FF': ff}
    graph, edges, reached = h['G'].actual_valid_graph(vertices, faces, K, pairs)
    graph.update(h['H'].opposite_reach(tri, K, edges))
    graph['C_F_has_opposite_winding_edge'] = int(graph['C_F_valid_opposite_winding_edge_count'] > 0)
    graph['domain'] = 'ACTUAL_BOUND_POST_INDEX_WITH_SOURCE_C_RESTORED'
    ids, points, boundary_audit = h['A'].topology(C, source_C_vertex_ids)
    dist, nearest = h['A'].nearest(points, F)
    boundary_vertices = [dict(source_vertex_id=int(i), x=float(p[0]), y=float(p[1]), z=float(p[2]),
        distance_model=float(d) if np.isfinite(d) else None,
        distance_status='FINITE' if np.isfinite(d) else 'POSITIVE_INFINITY_EMPTY_FREE',
        distance_L_fraction=float(d/L) if np.isfinite(d) else None,
        nearest_free_face=int(n)//3 if n >= 0 else None,
        nearest_free_corner=int(n)%3 if n >= 0 else None)
        for i, p, d, n in zip(ids, points, dist, nearest)]
    boundary = [dict(setting=name, threshold_model=threshold, scale=scale, tau=tau,
                     domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',
                     **h['A'].metric(dist, threshold))
                for name, threshold, scale, tau in h['A'].SETTINGS]
    distance = {name: float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None
                for name, fn in (('boundary_distance_mean_model', np.mean),
                                 ('boundary_distance_median_model', np.median),
                                 ('boundary_distance_p95_model', lambda x: np.quantile(x, .95)),
                                 ('boundary_distance_max_model', np.max))}
    if 'torch' in sys.modules:
        raise RuntimeError('CPU score imported model framework')
    result = dict(version=VERSION, status='PASS', metrics=metrics, topology=graph,
                  pairs=pairs, shape=shape, boundary=boundary, boundary_vertices=boundary_vertices,
                  boundary_audit=boundary_audit, distance=distance, vectors=vectors, edges=edges,
                  C_preserved_exact=True, C_query_count=2048,
                  C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),
                  free_face_denominator=len(F), CF_tested_pair_denominator=K*len(F),
                  FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
                  original_definition_unchanged=True, full_quality_scans=1,
                  new_queries=0, model_calls=0, seconds=time.perf_counter()-tick)
    if source_face_ids is not None:
        result['illegal_source_pairs'] = {key: sorted(value) for key, value in
                                         illegal_pair_sets(pairs, source_face_ids, K).items()}
    return result


def connectivity_ledger(vertices, faces, K, pairs, source_face_ids, source_corner_ids=None):
    """Track every indexed edge using original face+corner identities.

    Legal means exactly the existing actual_valid_graph definition, including
    both incident faces nonzero/nonduplicate and resolved nonillegal contact.
    Winding is reported separately. Only geometry, never semantics, is certified.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    ids = np.asarray(source_face_ids, dtype=np.int64)
    if ids.shape != (len(faces),) or len(set(map(int, ids))) != len(ids):
        raise ValueError('source_face_ids must uniquely identify every retained source face')
    if source_corner_ids is None:
        corners = np.tile(np.arange(3, dtype=np.int64), (len(faces), 1))
        corner_scope = 'CURRENT_CORNER_IDENTITY_CALLER_CERTIFIES_NO_PRIOR_PERMUTATION'
    else:
        corners = np.asarray(source_corner_ids, dtype=np.int64)
        corner_scope = 'EXPLICIT_ORIGINAL_CORNER_IDENTITY'
    if corners.shape != faces.shape or any(len(set(map(int, row))) != 3 for row in corners):
        raise ValueError('Original source corner identities must be distinct within each face')
    graph, rows, _ = helpers()['G'].actual_valid_graph(vertices, faces, K, pairs)
    incidents = collections.defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            vertex_key = tuple(sorted((int(face[a]), int(face[b]))))
            incidents[vertex_key].append(dict(source_face_id=int(ids[face_id]),
                source_corners=sorted((int(corners[face_id, a]), int(corners[face_id, b]))),
                output_face_id=face_id, is_C=face_id < K))
    by_index = {(int(row['vertex_u']), int(row['vertex_v'])): row for row in rows}
    records = []
    for vertex_key, inc in sorted(incidents.items()):
        if all(row['is_C'] for row in inc):
            continue
        row = dict(by_index[vertex_key])
        ordered = sorted(inc, key=lambda r: (r['source_face_id'], r['source_corners']))
        row.update(source_incidents=ordered,
            source_edge_key=tuple((r['source_face_id'], *r['source_corners']) for r in ordered),
            kind='CF' if any(r['is_C'] for r in inc) else 'FF')
        records.append(row)
    tri = vertices[faces].astype(np.float64)
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    return dict(version=VERSION, K=int(K), summary=graph, edges=records,
                source_face_ids=ids.tolist(), source_face_area={int(i): float(a) for i, a in zip(ids, area)},
                original_corner_identity=corner_scope)


def _edge_key(row):
    return tuple(tuple(map(int, part)) for part in row['source_edge_key'])


def compare_ledgers(before, after, before_pairs, after_pairs, affected_source_face_ids=None):
    """Connectivity fates use original edges, not mutable output vertex numbers."""
    old = {_edge_key(row): row for row in before['edges']}
    new = {_edge_key(row): row for row in after['edges']}
    old_faces, new_faces = set(before['source_face_ids']), set(after['source_face_ids'])
    deleted = old_faces-new_faces
    affected = None if affected_source_face_ids is None else set(map(int, affected_source_face_ids))
    new_half_edges = collections.defaultdict(set)
    for edge_key in new:
        for half_edge in edge_key:
            new_half_edges[half_edge].add(edge_key)
    before_area = {int(key): value for key, value in before['source_face_area'].items()}
    fates, additions = [], []
    summary = {}
    for kind in ('CF', 'FF'):
        base = {key: row for key, row in old.items() if row['kind'] == kind and row['valid_edge']}
        current = {key: row for key, row in new.items() if row['kind'] == kind and row['valid_edge']}
        retained = set(base) & set(current)
        lost = set(base)-set(current)
        gained = set(current)-set(base)
        summary.update({kind+'_baseline_legal_edges': len(base),
                        kind+'_retained_legal_edges': len(retained), kind+'_lost_legal_edges': len(lost),
                        kind+'_new_legal_edges': len(gained),
                        kind+'_baseline_legal_length': sum(r['length_model'] for r in base.values()),
                        kind+'_retained_baseline_length': sum(base[k]['length_model'] for k in retained),
                        kind+'_retained_current_length': sum(current[k]['length_model'] for k in retained),
                        kind+'_lost_baseline_length': sum(base[k]['length_model'] for k in lost),
                        kind+'_new_current_length': sum(current[k]['length_model'] for k in gained)})
        for key, row in sorted(base.items()):
            now = new.get(key)
            if now is None:
                # A retained edge with added incidences is invalid, not a split.
                matches = set.intersection(*(new_half_edges.get(part, set()) for part in key))
                if matches:
                    now = new[sorted(matches)[0]]
            missing_faces = sorted({x[0] for x in key} & deleted)
            fate = ('RETAINED_LEGAL' if key in retained else 'INCIDENT_FACE_DELETED' if missing_faces else
                    'NO_LONGER_SHARED' if now is None else 'SHARED_BECAME_INVALID')
            fates.append(dict(kind=kind, source_edge_key=key, fate=fate,
                baseline_length_model=row['length_model'],
                current_length_model=now['length_model'] if now is not None else None,
                deleted_source_face_ids=missing_faces,
                current_incidence=now['incidence'] if now else None,
                current_pair_classification=now['pair_classification'] if now else None,
                current_rejected_reasons=now['rejected_reasons'] if now else [],
                baseline_winding_opposite=row['winding_opposite'],
                current_winding_opposite=now['winding_opposite'] if now else None))
        for key in sorted(gained):
            additions.append(dict(kind=kind, source_edge_key=key, **{name: current[key][name] for name in
                ('length_model', 'pair_classification', 'winding_opposite')}))
    removed = [dict(source_face_id=i, baseline_area=before_area[i],
                    affected=None if affected is None else i in affected) for i in sorted(deleted)]
    summary.update(deleted_source_faces=len(deleted),
                   deleted_source_area=sum(row['baseline_area'] for row in removed),
                   affected_deleted_source_faces=None if affected is None else sum(row['affected'] for row in removed),
                   affected_deleted_source_area=None if affected is None else sum(row['baseline_area'] for row in removed if row['affected']))
    old_illegal = illegal_pair_sets(before_pairs, before['source_face_ids'], before['K'])
    new_illegal = illegal_pair_sets(after_pairs, after['source_face_ids'], after['K'])
    pair_fates = []
    for kind in ('CF', 'FF'):
        persisted = old_illegal[kind] & new_illegal[kind]
        resolved = old_illegal[kind]-new_illegal[kind]
        new_bad = new_illegal[kind]-old_illegal[kind]
        deleted_bad = {pair for pair in resolved if set(pair) & deleted}
        summary.update({kind+'_illegal_pairs_persisted': len(persisted),
                        kind+'_illegal_pairs_removed': len(resolved),
                        kind+'_illegal_pairs_removed_by_deletion': len(deleted_bad),
                        kind+'_new_illegal_pairs': len(new_bad)})
        for pair in sorted(old_illegal[kind] | new_illegal[kind]):
            fate = ('PERSISTENT_ILLEGAL' if pair in persisted else 'NEW_ILLEGAL' if pair in new_bad else
                    'INCIDENT_FACE_DELETED' if pair in deleted_bad else 'RETAINED_PAIR_NO_LONGER_ILLEGAL')
            pair_fates.append(dict(kind=kind, source_face_a=pair[0], source_face_b=pair[1], fate=fate))
    return dict(summary=summary, baseline_edge_fates=fates, new_edges=additions,
                deleted_faces=removed, illegal_pair_fates=pair_fates)
def classify_cf(C, F_affected, length=L):
    """Illegal (C row, affected-F row) pairs using the original verified scanner."""
    if len(C) == 0 or len(F_affected) == 0:
        return set()
    scan = pair_scan(C, F_affected, length, independent=True)
    return {tuple(map(int, row['pair'])) for row in scan['contacts']
            if row['classification'] in ILLEGAL}
