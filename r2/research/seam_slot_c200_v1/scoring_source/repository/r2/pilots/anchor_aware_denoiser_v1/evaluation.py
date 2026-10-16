"""Frozen-math CPU evaluation for the anchor-aware denoiser pilot.

96 new GEO_C and quality jobs; 48 D0 quality jobs are verified cache reads.
All mesh/scoring writers receive guarded fresh staging directories, atomically
promoted under this pilot. No old module globals, source, data or outputs change.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import concurrent.futures
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
try:
    from . import common as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from r2.pilots.anchor_aware_denoiser_v1 import common as C

ROOT, OLD = C.ROOT, C.OLD
OUT = C.OUT / 'evaluation'
CODE = Path(__file__).resolve()
DENOISERS = ('D0_OFFICIAL', 'D_CONT_NO_ROLE', 'D_ANCHOR')
META = ('label', 'request_id', 'condition_id', 'parent_id', 'region_type', 'tier', 'candidate', 'N', 'K')
ILLEGAL = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))
LEGAL = frozenset(('shared_edge_contact', 'shared_vertex_contact', 'coplanar_boundary_or_tolerance_contact',
                   'noncoplanar_edge_or_tangent_contact', 'point_or_tangent_contact'))
VERSION = 'r2_anchor_denoiser_eval_frozen_math_v1'


def module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def helpers():
    A = module('_anchor_frozen_avmr', ROOT / 'r2/tools/avmr_pmr_v1/run.py')
    Q = module('_anchor_frozen_quality', ROOT / 'r2/tools/postprocess_avmr_pmr_v1/quality.py')
    from native_t1 import postprocess_cli as PC
    from native_t1.postprocessing import FIXED_CONFIG
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    return A, Q, PC, FIXED_CONFIG


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write(path, obj):
    C.write(C.checked(path), clean(obj))


def event(kind, **kw):
    C.event(kind, ledger='evaluation/ledger.jsonl', **kw)


def payload(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def meta(r):
    return {k: r[k] for k in META}


def paths(r, denoiser):
    if denoiser == 'D0_OFFICIAL':
        den = OLD / 'denoised' / r['label'] / r['request_id'] / 'denoised.npz'
        post = OLD / 'outputs' / r['label'] / r['request_id'] / 'post.npz'
        score = OLD / 'scores' / r['label'] / r['request_id'] / 'score.json'
    else:
        den = C.OUT / 'inference' / denoiser / r['label'] / r['request_id'] / 'denoised.npz'
        post = OUT / 'outputs' / denoiser / r['label'] / r['request_id'] / 'post.npz'
        score = OUT / 'scores' / denoiser / r['label'] / r['request_id'] / 'score.json'
    return den, post, score


def verify_files(mapping):
    for path, h in mapping.items():
        if C.sha(path) != h:
            raise ValueError('Artifact identity changed: ' + path)


def prepare():
    A, Q, PC, config = helpers()
    target = OUT / 'registration.json'
    if target.exists():
        reg = C.read(target)
        verify_files(reg['protected'])
        if C.sha(CODE) != reg['evaluation_source_sha256']:
            correction = C.read(OUT / 'compatibility/evaluation_source_revision.json')
            if (correction['original_source_sha256'] != reg['evaluation_source_sha256']
                    or correction['revised_source_sha256'] != C.sha(CODE)
                    or correction['core_registration_sha256'] != C.sha(target)
                    or C.sha(correction['original_source_snapshot']) != correction['original_source_sha256']
                    or C.sha(correction['revised_source_snapshot']) != correction['revised_source_sha256']
                    or correction['scientific_functions_AST_unchanged'] is not True
                    or correction['existing_D0_cache_identity_unchanged'] is not True):
                raise ValueError('Evaluation source correction provenance does not match')
        return reg
    old = C.read(OLD / 'registration.json')
    assert len(old['records']) == 48 and old['postprocess_config'] == config
    assert old['settings'] == [dict(name=n, threshold_model=t, scale=s, tau=v) for n, t, s, v in A.SETTINGS]
    Q.source_hashes()
    protected = {}
    source_paths = [ROOT / 'r2/tools/avmr_pmr_v1/run.py', ROOT / 'r2/tools/postprocess_avmr_pmr_v1/quality.py',
        ROOT / 'native_t1/postprocessing.py', ROOT / 'native_t1/postprocess_cli.py', OLD / 'registration.json',
        OLD / 'denoise_registration.json', OLD / 'quality_per_output.csv', OLD / 'per_output.csv',
        ROOT / 'experiments/r2_avmr_pmr_offline_v1/per_output.csv',
        ROOT / 'experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv',
        C.OUT / 'data_audit/dev_conditions.json'] + [Path(p) for p in Q.protected_paths()]
    for r in old['records']:
        den, post, score = paths(r, 'D0_OFFICIAL')
        s = C.read(score)
        assert s['status'] == 'PASS' and s['identity']['raw_sha256'] == r['raw_sha256']
        verify_files(s['files'])
        assert C.sha(post) == s['identity']['post_sha256']
        d = C.read(den.with_name('receipt.json'))
        assert d['status'] == 'PASS' and d['output_sha256'] == C.sha(den)
        source_paths += [den, den.with_name('receipt.json'), post, post.with_name('post.ply'),
                         post.with_name('postprocess.json'), score, Path(r['raw_path']), Path(r['condition_source'])]
        source_paths += [Path(p) for p in s['files']]
    for r in C.read(C.OUT / 'data_audit/dev_conditions.json')['rows']:
        source_paths.append(Path(r['source_interface_json']))
    for p in set(source_paths):
        protected[str(p)] = C.sha(p)
    reg = dict(version=VERSION, status='REGISTERED', records=old['records'], denoisers=DENOISERS,
        expected_primary_POST=144, expected_RAW_reference=48, new_GEO_C_jobs=96, new_quality_jobs=96,
        reused_D0_quality_jobs=48, settings=old['settings'], postprocess_config=config,
        quality_definition=Q.definition(), quality_source_hashes=Q.source_hashes(),
        primary_AVMR_domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY', L=A.L,
        supported_interface='Extra source-supported boundary stratum only; never replaces main denominator or declares user-required interface',
        valid_edge_definition='Exactly two incident nonzero-area nonduplicate faces; at least one free; frozen pair class is resolved legal, not positive-area overlap. Winding reported separately.',
        reachable_area_definition='BFS in the valid C/F and F/F edge graph seeded with every C face; reached free area / all free area. C/C edges unnecessary because all C faces are seeds. Not semantic body identification.',
        intermediate='pre-GEO restored-C boundary vertex metrics and actual per-face angle/area only; no extra whole pair scans',
        source_geometry='Open completion; no GT reconstruction of generated RAW; original frozen C queries reused',
        FF_fates='D0 and new POST pairs mapped to unchanged original RAW face IDs; separate deleted-face events, retained resolved nonillegal, retained UNKNOWN, persistent illegal and new illegal',
        dispatch=dict(CPU_workers=2, per_process_threads=2, GPU_workers=0),
        evaluation_source_sha256=C.sha(CODE), protected=protected)
    write(target, reg)
    write(OUT / 'resources_before.json', C.resources())
    event('EVALUATION_REGISTERED', new_quality_jobs=96, new_GEO_C_jobs=96, reused_quality_jobs=48)
    return reg


def rewrite_paths(obj, before, after):
    if isinstance(obj, dict):
        return {rewrite_paths(k, before, after): rewrite_paths(v, before, after) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v, before, after) for v in obj]
    if isinstance(obj, str):
        return obj.replace(str(before), str(after))
    return obj


def new_inference_check(r, denoiser):
    den, post, score = paths(r, denoiser)
    rec = C.read(den.with_name('receipt.json'))
    identity = rec.get('identity', rec)
    for k, v in (('raw_sha256', r['raw_sha256']), ('label', r['label']), ('request_id', r['request_id'])):
        if identity.get(k) != v:
            raise ValueError('New inference identity mismatch ' + k)
    output_hash = rec.get('output_sha256', identity.get('output_sha256'))
    checkpoint_hash = identity.get('model_checkpoint_sha256', identity.get('checkpoint_sha256'))
    if not checkpoint_hash:
        raise ValueError('New inference lacks checkpoint identity')
    if rec.get('status') != 'PASS' or output_hash != C.sha(den):
        raise ValueError('New inference receipt not complete')
    d = payload(den)
    raw = payload(r['raw_path'])['raw'].reshape(-1, 3, 3)
    assert C.sha(r['raw_path']) == r['raw_sha256']
    K = int(r['K'])
    assert same(d['C'], raw[:K]) and same(d['output'][:K], raw[:K])
    assert d['output'].shape == raw.shape and d['output'].dtype == np.float32
    assert np.isfinite(d['output']).all()
    proposal = d['proposal'] if 'proposal' in d else d['official_unconstrained_proposal']
    assert same(proposal[K:], d['output'][K:])
    return den, checkpoint_hash


def postprocess():
    reg = prepare()
    _, _, PC, _ = helpers()
    index = []
    for denoiser in DENOISERS:
        for r in reg['records']:
            den, post, score = paths(r, denoiser)
            if denoiser != 'D0_OFFICIAL':
                den, checkpoint_hash = new_inference_check(r, denoiser)
                dest = C.checked(post.parent)
                if (dest / 'postprocess.json').exists():
                    pp = C.read(dest / 'postprocess.json')
                    assert pp['status'] == 'PASS' and pp['input']['file_sha256'] == C.sha(den)
                    for name, info in pp['files'].items():
                        assert C.sha(dest / name) == info['sha256']
                else:
                    if dest.exists():
                        raise RuntimeError('Incomplete POST directory; no implicit retry')
                    staging = C.checked(dest.with_name(dest.name + '.staging'))
                    if staging.exists():
                        raise RuntimeError('Interrupted GEO_C staging; inspect before recovery')
                    event('GEO_C_STARTED', denoiser=denoiser, label=r['label'], request_id=r['request_id'])
                    pp = PC.run(den, int(r['K']), staging, input_key='output')
                    pp = rewrite_paths(pp, staging, dest)
                    write(staging / 'postprocess.json', pp)
                    os.replace(staging, dest)
                    event('GEO_C_COMPLETE', denoiser=denoiser, label=r['label'], request_id=r['request_id'], seconds=pp['seconds'])
            else:
                checkpoint_hash = C.EXPECTED
                pp = C.read(post.with_name('postprocess.json'))
            a = payload(post)
            raw = payload(r['raw_path'])['raw'].reshape(-1, 3, 3)
            assert same(a['output'][:int(r['K'])], raw[:int(r['K'])])
            assert same(a['vertices'][a['faces']], a['output'])
            vertices, faces = PC.read_indexed_ply(post.with_name('post.ply'))
            assert same(vertices, a['vertices']) and same(faces, a['faces'])
            index.append(dict(**r, denoiser=denoiser, denoised_path=str(den), post_path=str(post),
                post_ply=str(post.with_name('post.ply')), post_sha256=C.sha(post), output_N=len(a['output']),
                denoiser_checkpoint_sha256=checkpoint_hash, stage='POST'))
    C.csvwrite(OUT / 'post_index.csv', index)
    write(OUT / 'post_completion.json', dict(status='PASS', primary_POST_rows=len(index), new_GEO_C_jobs=96,
                                          reused_D0_POST=48, indexed_PLY_and_C_bitwise=True))
    return index


def actual_valid_graph(vertices, faces, K, pairs):
    triangles = vertices[faces].astype(np.float64)
    areas = .5 * np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    duplicate_groups = collections.defaultdict(list)
    for i, face in enumerate(faces):
        duplicate_groups[tuple(sorted(map(int, face)))].append(i)
    duplicate_faces = {i for group in duplicate_groups.values() if len(group) > 1 for i in group}
    edges = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[tuple(sorted((int(u), int(v))))].append((i, int(u), int(v)))
    cf = {tuple(map(int, p['pair'])): p['classification'] for p in pairs['CF']['contacts']}
    ff = {tuple(sorted(map(int, p['pair']))): p['classification'] for p in pairs['FF']['contacts']}
    adjacent = collections.defaultdict(set)
    rows = []
    for (u, v), incidents in sorted(edges.items()):
        if not any(i >= K for i, _, _ in incidents):
            continue
        has_C = any(i < K for i, _, _ in incidents)
        is_cf = has_C and any(i >= K for i, _, _ in incidents)
        reasons, classification, opposed = [], 'NA_NONMANIFOLD_OR_BOUNDARY', None
        length = float(np.linalg.norm(vertices[u].astype(float)-vertices[v]))
        if len(incidents) != 2:
            reasons.append('NOT_TWO_INCIDENT_FACES')
        else:
            (i, a, b), (j, c, d) = incidents
            opposed = a == d and b == c
            if i == j or u == v or length <= 0 or areas[i] <= 0 or areas[j] <= 0:
                reasons.append('DEGENERATE')
            if i in duplicate_faces or j in duplicate_faces:
                reasons.append('DUPLICATE_FACE')
            if is_cf:
                known, free = (i, j) if i < K else (j, i)
                classification = cf.get((known, free-K), 'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
            else:
                classification = ff.get(tuple(sorted((i-K, j-K))), 'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
            if classification not in LEGAL:
                reasons.append('ILLEGAL' if classification in ILLEGAL else 'UNKNOWN_OR_NOT_CONTACT')
            if not reasons:
                adjacent[i].add(j)
                adjacent[j].add(i)
        rows.append(dict(vertex_u=u, vertex_v=v, length_model=length, incident_face_ids=[i for i, _, _ in incidents],
            incidence=len(incidents), is_C_F=is_cf, valid_edge=not reasons, winding_opposite=opposed,
            pair_classification=classification, rejected_reasons=reasons))
    seen = set(range(K))
    todo = list(range(K))
    while todo:
        face = todo.pop()
        for other in adjacent[face] - seen:
            seen.add(other)
            todo.append(other)
    reached = sorted(i for i in seen if i >= K)
    cf_rows = [r for r in rows if r['is_C_F']]
    valid_cf = [r for r in cf_rows if r['valid_edge']]
    summary = dict(C_F_shared_edge_count=len(cf_rows), C_F_shared_edge_total_length=sum(r['length_model'] for r in cf_rows),
        C_F_valid_edge_count=len(valid_cf), C_F_has_valid_edge=int(bool(valid_cf)),
        C_F_valid_edge_total_length=sum(r['length_model'] for r in valid_cf),
        C_F_valid_opposite_winding_edge_count=sum(bool(r['winding_opposite']) for r in valid_cf),
        C_F_nonmanifold_edge_count=sum(r['incidence'] > 2 for r in cf_rows),
        C_F_rejected_duplicate_edge_count=sum('DUPLICATE_FACE' in r['rejected_reasons'] for r in cf_rows),
        C_F_rejected_illegal_edge_count=sum('ILLEGAL' in r['rejected_reasons'] for r in cf_rows),
        C_F_rejected_unknown_edge_count=sum('UNKNOWN_OR_NOT_CONTACT' in r['rejected_reasons'] for r in cf_rows),
        graph_valid_edges=sum(r['valid_edge'] for r in rows), reached_free_faces=len(reached),
        reached_free_area=float(areas[reached].sum()), free_area=float(areas[K:].sum()),
        reached_free_area_fraction=float(areas[reached].sum()/areas[K:].sum()) if areas[K:].sum() > 0 else None,
        exact_duplicate_faces=len(duplicate_faces), nonmanifold_edges=sum(len(v) > 2 for v in edges.values()),
        boundary_edges=sum(len(v) == 1 for v in edges.values()),
        status='PASS' if not any('UNKNOWN_OR_NOT_CONTACT' in r['rejected_reasons'] for r in rows if r['incidence'] == 2) else 'PARTIAL_UNKNOWN_EDGE_REJECTED')
    return summary, rows, reached


def boundary_values(r, tri, A):
    K = int(r['K'])
    source = payload(r['condition_source'])
    assert same(source['C'], tri[:K])
    ids, pts, audit = A.topology(source['C'], source['C_source_vertex_ids'])
    distances, nearest = A.nearest(pts, tri[K:])
    dev = next(q for q in C.read(C.OUT / 'data_audit/dev_conditions.json')['rows'] if q['condition_id'] == r['condition_id'])
    interface = C.read(dev['source_interface_json'])
    supported = set(interface['supported_C_source_vertex_ids'])
    vertices = []
    for vid, p, dist, near in zip(ids, pts, distances, nearest):
        vertices.append(dict(source_vertex_id=vid, x=float(p[0]), y=float(p[1]), z=float(p[2]),
            distance_model=float(dist) if np.isfinite(dist) else None, distance_L_fraction=float(dist/A.L) if np.isfinite(dist) else None,
            nearest_free_face=int(near)//3 if near >= 0 else None, nearest_free_corner=int(near)%3 if near >= 0 else None,
            source_supported_reference=int(vid in supported)))
    metrics = []
    for domain, ds in (('ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY', distances),
                       ('SOURCE_SUPPORTED_INTERFACE_REFERENCE_ONLY', distances[[vid in supported for vid in ids]])):
        for name, threshold, scale, tau in A.SETTINGS:
            metrics.append(dict(domain=domain, setting=name, threshold_model=threshold, scale=scale, tau=tau, **A.metric(ds, threshold)))
    return vertices, metrics


def canonical_identity(identity):
    """Normalize only integer metadata after CSV roundtrip; preserve cache bytes."""
    return {k: int(v) if k in ('candidate', 'N', 'K') else v for k, v in identity.items()}


def score_one(row):
    start = time.perf_counter()
    A, Q, _, _ = helpers()
    denoiser = row['denoiser']
    den, post, score = paths(row, denoiser)
    out = C.checked(OUT / 'diagnostics' / denoiser / row['label'] / row['request_id'])
    receipt_path = out / 'diagnostic.json'
    identity = dict(denoiser=denoiser, **meta(row), post_sha256=C.sha(post),
                    denoised_sha256=C.sha(den), registration_sha256=C.sha(OUT / 'registration.json'))
    if receipt_path.exists():
        done = C.read(receipt_path)
        assert done['status'] == 'PASS' and canonical_identity(done['identity']) == canonical_identity(identity)
        verify_files(done['files'])
        return str(receipt_path)
    if denoiser == 'D0_OFFICIAL':
        frozen = C.read(score)
        quality = frozen['quality']
        assert frozen['status'] == quality['status'] == 'PASS'
        verify_files(quality['files'])
        assert quality['identity']['processed_sha256'] == C.sha(post)
    elif score.exists():
        quality = C.read(score)
        assert quality['status'] == 'PASS' and quality['identity']['processed_sha256'] == C.sha(post)
        verify_files(quality['files'])
    else:
        dest = C.checked(score.parent)
        staging = C.checked(dest.with_name(dest.name + '.staging'))
        if dest.exists() or staging.exists():
            raise RuntimeError('Unfinished quality output: no implicit rerun')
        event('QUALITY_STARTED', denoiser=denoiser, label=row['label'], request_id=row['request_id'])
        quality = Q.score_quality(row, post, C.checked(staging / 'quality'))
        quality = rewrite_paths(quality, staging, dest)
        write(staging / 'score.json', quality)
        os.replace(staging, dest)
        verify_files(quality['files'])
        event('QUALITY_COMPLETE', denoiser=denoiser, label=row['label'], request_id=row['request_id'], seconds=quality['seconds'])
    a = payload(post)
    pairs = C.read(quality['pair_audit_path'])
    topology, edge_rows, reached = actual_valid_graph(a['vertices'], a['faces'], int(row['K']), pairs)
    vertices, metrics = boundary_values(row, a['output'], A)
    before = payload(den)['output']
    before_vertices, before_metrics = boundary_values(row, before, A)
    shape = Q._old().cpu_math()['shape'].shape_quality(before[int(row['K']):], A.L)
    write(out / 'pre_GEO_shape_per_face.json', shape)
    C.csvwrite(out / 'valid_edge_details.csv', edge_rows)
    C.csvwrite(out / 'boundary_vertices_POST.csv', vertices)
    C.csvwrite(out / 'boundary_vertices_PRE_GEO.csv', before_vertices)
    diagnostic = dict(status='PASS', identity=identity, quality=quality, topology=topology,
        vertices=vertices, metrics=metrics, pre_GEO_vertices=before_vertices, pre_GEO_metrics=before_metrics,
        pre_GEO_shape=dict(free_faces=len(before)-int(row['K']), free_total_surface_area=shape['total_area'],
            free_minimum_angle_lt5_fraction=shape['min_angle_lt5_fraction'], free_q_gt10_fraction=shape['q_actual_gt10_fraction'],
            free_exact_zero_area_count=shape['exact_zero_area_count'], free_near_degenerate_count=shape['near_degenerate_area_count']),
        reached_free_output_face_ids=reached, source_face_ids=a['source_face_ids'].tolist(),
        postprocess=C.read(post.with_name('postprocess.json')), seconds=time.perf_counter()-start,
        files={str(p): C.sha(p) for p in out.iterdir() if p.is_file()})
    write(receipt_path, diagnostic)
    return str(receipt_path)


def scores():
    import csv
    reg = prepare()
    with (OUT / 'post_index.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 144
    # Resource snapshot accompanies every CPU scoring dispatch. This is wholly
    # independent of generator/denoiser ownership and consumes no GPU memory.
    write(OUT / 'resources_before_score.json', C.resources())
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts = list(pool.map(score_one, rows, chunksize=1))
    verify_files(reg['protected'])
    write(OUT / 'score_completion.json', dict(status='PASS', receipts=receipts, new_quality_jobs=96,
        reused_D0_quality_jobs=48, total_primary_POST=144, CPU_workers=2, CPU_threads_per_worker=2,
        model_forwards=0, new_C_queries=0, new_F_queries=0, seconds=time.perf_counter()-start))
    return receipts


def pair_source_map(pairs, source_ids, K):
    result = {}
    for r in pairs['FF']['contacts']:
        i, j = map(int, r['pair'])
        result[tuple(sorted((int(source_ids[K+i]), int(source_ids[K+j]))))] = r['classification']
    return result


def ff_fates(row, diagnostic, baseline):
    K = int(row['K'])
    old_map = pair_source_map(C.read(baseline['quality']['pair_audit_path']), baseline['source_face_ids'], K)
    new_map = pair_source_map(C.read(diagnostic['quality']['pair_audit_path']), diagnostic['source_face_ids'], K)
    old_illegal = {p for p, label in old_map.items() if label in ILLEGAL}
    new_illegal = {p for p, label in new_map.items() if label in ILLEGAL}
    new_ids = set(diagnostic['source_face_ids'])
    events = []
    for pair in sorted(old_illegal | new_illegal):
        old_class = old_map.get(pair, 'disjoint')
        new_class = new_map.get(pair, 'disjoint')
        both_kept = all(i in new_ids for i in pair)
        if pair in old_illegal:
            if not both_kept:
                fate = 'D0_ILLEGAL_FACE_REMOVED'
            elif pair in new_illegal:
                fate = 'PERSISTENT_ILLEGAL'
            elif new_class == 'disjoint' or new_class in LEGAL:
                fate = 'BOTH_RETAINED_RESOLVED_NONILLEGAL'
            else:
                fate = 'BOTH_RETAINED_UNKNOWN'
        else:
            fate = 'NEW_ILLEGAL'
        events.append(dict(**meta(row), denoiser=row['denoiser'], raw_source_face_a=pair[0], raw_source_face_b=pair[1],
            D0_classification=old_class, new_classification=new_class if both_kept else 'FACE_REMOVED',
            both_D0_faces_retained=both_kept, fate=fate))
    counts = collections.Counter(e['fate'] for e in events)
    assert sum(counts[k] for k in ('D0_ILLEGAL_FACE_REMOVED', 'PERSISTENT_ILLEGAL',
                                  'BOTH_RETAINED_RESOLVED_NONILLEGAL', 'BOTH_RETAINED_UNKNOWN')) == len(old_illegal)
    assert counts['PERSISTENT_ILLEGAL'] + counts['NEW_ILLEGAL'] == len(new_illegal)
    summary = dict(**meta(row), denoiser=row['denoiser'], D0_illegal_pairs=len(old_illegal), new_illegal_pairs=len(new_illegal),
        **{k: counts[k] for k in ('D0_ILLEGAL_FACE_REMOVED', 'PERSISTENT_ILLEGAL', 'BOTH_RETAINED_RESOLVED_NONILLEGAL',
                                'BOTH_RETAINED_UNKNOWN', 'NEW_ILLEGAL')})
    return summary, events


def summaries(rows, keys, numeric):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    result = []
    for values, group in sorted(groups.items(), key=lambda p: str(p[0])):
        row = dict(zip(keys, values))
        row['outputs'] = len(group)
        for field in numeric:
            vals = [r[field] for r in group if r.get(field) is not None and r.get(field) != '']
            vals = [float(x) for x in vals]
            row[field + '_valid_n'] = len(vals)
            row[field + '_mean'] = float(np.mean(vals)) if vals else None
            row[field + '_median'] = float(np.median(vals)) if vals else None
            row[field + '_p95'] = float(np.quantile(vals, .95)) if vals else None
        result.append(row)
    return result


def aggregate():
    import csv
    reg = prepare()
    with (OUT / 'post_index.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    results = {}
    for row in rows:
        path = OUT / 'diagnostics' / row['denoiser'] / row['label'] / row['request_id'] / 'diagnostic.json'
        results[(row['denoiser'], row['label'], row['request_id'])] = C.read(path)
    quality_rows, topology_rows, vm_rows, supported_rows, vertices, pre_shapes, modifications = [], [], [], [], [], [], []
    fates, fate_events = [], []
    for row in rows:
        d = results[(row['denoiser'], row['label'], row['request_id'])]
        m = dict(**meta(row), denoiser=row['denoiser'])
        quality_rows.append(dict(**m, stage='POST', **d['quality']['metrics'], free_face_denominator=d['quality']['free_face_denominator']))
        topology_rows.append(dict(**m, stage='POST', **d['topology']))
        pre_shapes.append(dict(**m, stage='PRE_GEO_C_RESTORED', **d['pre_GEO_shape']))
        for stage, key in (('POST', 'metrics'), ('PRE_GEO_C_RESTORED', 'pre_GEO_metrics')):
            for v in d[key]:
                (vm_rows if v['domain'] == 'ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY' else supported_rows).append(dict(**m, stage=stage, **v))
        for stage, key in (('POST', 'vertices'), ('PRE_GEO_C_RESTORED', 'pre_GEO_vertices')):
            vertices += [dict(**m, stage=stage, **v) for v in d[key]]
        pp = d['postprocess']
        before = payload(row['denoised_path'])['output']
        after = payload(row['post_path'])
        raw = payload(row['raw_path'])['raw'].reshape(-1, 3, 3)
        K = int(row['K'])
        drift = np.linalg.norm(before[K:].astype(float)-raw[K:], axis=-1)
        modifications.append(dict(**m, raw_free_faces=len(raw)-K, POST_free_faces=len(after['output'])-K,
            deleted_free_faces=len(raw)-len(after['output']), deleted_free_fraction=(len(raw)-len(after['output']))/(len(raw)-K),
            denoise_free_corner_displacement_mean=float(drift.mean()), denoise_free_corner_displacement_p95=float(np.quantile(drift,.95)),
            denoise_free_corner_displacement_max=float(drift.max()),
            GEO_free_corner_displacement_mean=float(after['corner_displacement'][K:].mean()),
            GEO_free_corner_displacement_max=float(after['corner_displacement'][K:].max()),
            POST_free_surface_area=d['quality']['metrics']['free_total_surface_area']))
        if row['denoiser'] != 'D0_OFFICIAL':
            baseline = results[('D0_OFFICIAL', row['label'], row['request_id'])]
            fr, ev = ff_fates(row, d, baseline)
            fates.append(fr); fate_events += ev
    C.csvwrite(OUT/'quality_per_output.csv', quality_rows)
    C.csvwrite(OUT/'topology_per_output.csv', topology_rows)
    C.csvwrite(OUT/'per_output.csv', vm_rows)
    C.csvwrite(OUT/'source_supported_reference_per_output.csv', supported_rows)
    C.csvwrite(OUT/'per_boundary_vertex.csv', vertices)
    C.csvwrite(OUT/'pre_GEO_shape_per_output.csv', pre_shapes)
    C.csvwrite(OUT/'modifications_per_output.csv', modifications)
    C.csvwrite(OUT/'FF_fates_per_output.csv', fates)
    if fate_events:
        C.csvwrite(OUT/'FF_pair_fates.csv', fate_events)
    # Original RAW evidence is copied as tables with its own stage/identity;
    # no source-parent reconstruction and no additional RAW quality jobs.
    def csvread(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    raw_quality = csvread(ROOT/'experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv')
    assert len(raw_quality) == 48
    C.csvwrite(OUT/'RAW_quality_reference.csv', raw_quality)
    raw_vm = csvread(ROOT/'experiments/r2_avmr_pmr_offline_v1/per_output.csv')
    assert len(raw_vm) == 384
    C.csvwrite(OUT/'RAW_AVMR_reference.csv', raw_vm)
    # Distance distribution summaries use the unchanged unique source boundary IDs.
    distance_rows=[]
    for row in rows:
        d=results[(row['denoiser'],row['label'],row['request_id'])]
        for stage,key in (('POST','vertices'),('PRE_GEO_C_RESTORED','pre_GEO_vertices')):
            ds=[v['distance_model'] for v in d[key]]
            finite=[x for x in ds if x is not None]
            distance_rows.append(dict(**meta(row),denoiser=row['denoiser'],stage=stage,
                boundary_vertex_count=len(ds),finite_distance_count=len(finite),
                distance_mean_model=float(np.mean(finite)) if len(finite)==len(ds) and finite else None,
                distance_median_model=float(np.median(finite)) if len(finite)==len(ds) and finite else None,
                distance_p95_model=float(np.quantile(finite,.95)) if len(finite)==len(ds) and finite else None,
                status='PASS' if len(finite)==len(ds) and finite else 'NA_EMPTY_BOUNDARY_OR_FREE'))
    C.csvwrite(OUT/'boundary_distance_per_output.csv',distance_rows)
    avmr_paired=[]
    for new,base in (('D_CONT_NO_ROLE','D0_OFFICIAL'),('D_ANCHOR','D0_OFFICIAL'),('D_ANCHOR','D_CONT_NO_ROLE')):
        reference={(r['label'],r['request_id'],r['stage'],r['setting']):r for r in vm_rows if r['denoiser']==base}
        for row in vm_rows:
            if row['denoiser']!=new:continue
            ref=reference[(row['label'],row['request_id'],row['stage'],row['setting'])]
            avmr_paired.append(dict(**meta(row),stage=row['stage'],setting=row['setting'],comparison=new+'_minus_'+base,
                VMR_delta=row['VMR']-ref['VMR'] if row['VMR'] is not None and ref['VMR'] is not None else None,
                perfect_delta=row['perfect']-ref['perfect'] if row['perfect'] is not None and ref['perfect'] is not None else None))
    C.csvwrite(OUT/'AVMR_paired_per_output.csv',avmr_paired)
    C.csvwrite(OUT/'AVMR_paired_summary.csv',summaries(avmr_paired,('comparison','label','stage','setting'),('VMR_delta','perfect_delta')))
    q_numeric = sorted(k for k,v in quality_rows[0].items() if k not in META and isinstance(v,(int,float)) or k in ('free_face_denominator',))
    t_numeric = sorted(k for k,v in topology_rows[0].items() if k not in META and isinstance(v,(int,float)))
    quality_summaries, topo_summaries, vm_summaries = [], [], []
    for group, extra in (('ALL',()), ('REGION',('region_type',)), ('PARENT',('parent_id',)), ('NOISE',('candidate',)),
                         ('PARENT_NOISE',('parent_id','candidate'))):
        quality_summaries += [dict(grouping=group,**r) for r in summaries(quality_rows,('denoiser','label')+extra,q_numeric)]
        topo_summaries += [dict(grouping=group,**r) for r in summaries(topology_rows,('denoiser','label')+extra,t_numeric)]
        vm_summaries += [dict(grouping=group,**r) for r in summaries(vm_rows,('denoiser','label','stage','setting')+extra,('VMR','perfect'))]
    C.csvwrite(OUT/'quality_summary.csv', quality_summaries)
    C.csvwrite(OUT/'topology_summary.csv', topo_summaries)
    C.csvwrite(OUT/'AVMR_summary.csv', vm_summaries)
    # Both trained arms vs D0 and ANCHOR vs CONT, always same generator/parent/noise.
    paired = []
    for new, base in (('D_CONT_NO_ROLE','D0_OFFICIAL'),('D_ANCHOR','D0_OFFICIAL'),('D_ANCHOR','D_CONT_NO_ROLE')):
        for row in quality_rows:
            if row['denoiser'] != new: continue
            ref = next(r for r in quality_rows if r['denoiser']==base and r['label']==row['label'] and r['request_id']==row['request_id'])
            tr = next(r for r in topology_rows if r['denoiser']==new and r['label']==row['label'] and r['request_id']==row['request_id'])
            tb = next(r for r in topology_rows if r['denoiser']==base and r['label']==row['label'] and r['request_id']==row['request_id'])
            delta = dict(**meta(row), comparison=new+'_minus_'+base)
            for k in q_numeric:
                delta[k+'_delta'] = float(row[k])-float(ref[k]) if row[k] is not None and ref[k] is not None else None
            for k in t_numeric:
                delta[k+'_delta'] = float(tr[k])-float(tb[k]) if tr[k] is not None and tb[k] is not None else None
            paired.append(delta)
    C.csvwrite(OUT/'paired_per_output.csv', paired)
    delta_keys = [k for k in paired[0] if k.endswith('_delta')]
    C.csvwrite(OUT/'paired_summary.csv', summaries(paired,('comparison','label'),delta_keys))
    verify_files(reg['protected'])
    assert len(quality_rows)==144 and len(topology_rows)==144 and len(vm_rows)==2304 and len(vertices)==2976
    write(OUT/'completion.json',dict(status='PASS',primary_POST=144,RAW_reference=48,new_GEO_C=96,new_quality=96,
        D0_quality_reused=48,primary_AVMR_POST_and_PRE_GEO_rows=2304,boundary_vertex_rows=2976,
        supported_reference_rows=len(supported_rows),FF_fate_output_rows=len(fates),FF_pair_fate_rows=len(fate_events),
        protected_files_unchanged=len(reg['protected']),model_calls=0,new_queries=0,source_reconstruction_scores=0))
    event('EVALUATION_COMPLETE',primary_POST=144,new_quality=96,old_quality_reused=48)
    print(json.dumps(C.read(OUT/'completion.json'),indent=2))


def self_check():
    # Real structural regression checks: shared vertex != shared edge; duplicate
    # faces and illegal overlap cannot create a valid edge; reachability extends.
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,-1,0]],dtype=np.float32)
    f=np.array([[0,1,2],[1,0,3],[1,3,4]],dtype=np.int64)
    pair={'CF':{'contacts':[{'pair':[0,0],'classification':'shared_edge_contact'}]},
          'FF':{'contacts':[{'pair':[0,1],'classification':'shared_edge_contact'}]}}
    s,_,reach=actual_valid_graph(v,f,1,pair)
    assert s['C_F_valid_edge_count']==1 and reach==[1,2] and s['reached_free_area_fraction']==1
    bad=json.loads(json.dumps(pair));bad['CF']['contacts'][0]['classification']='coplanar_positive_area_overlap'
    s,_,reach=actual_valid_graph(v,f,1,bad);assert s['C_F_valid_edge_count']==0 and not reach
    dup=np.array([[0,1,2],[2,1,0]],dtype=np.int64)
    s,_,reach=actual_valid_graph(v,dup,1,pair);assert s['C_F_valid_edge_count']==0 and not reach
    point=np.array([[0,1,2],[0,3,4]],dtype=np.int64)
    s,_,reach=actual_valid_graph(v,point,1,{'CF':{'contacts':[]},'FF':{'contacts':[]}})
    assert s['C_F_shared_edge_count']==0 and not reach
    write(OUT/'valid_graph_self_check.json',dict(status='PASS',checks=4,official_model_loads=0,geometry_detector_calls=0))
    print('PASS 4 valid-graph structural checks; zero model/scorer calls')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','post','score','aggregate','all','self-check'),default='all')
    args=parser.parse_args()
    if args.stage=='self-check':self_check();return
    if args.stage=='prepare':prepare();return
    if args.stage in ('post','all'):postprocess()
    if args.stage in ('score','all'):scores()
    if args.stage in ('aggregate','all'):aggregate()


if __name__=='__main__':main()
