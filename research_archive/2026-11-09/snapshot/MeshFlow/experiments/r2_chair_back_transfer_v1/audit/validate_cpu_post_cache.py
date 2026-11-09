"""Independent cache acceptance only: no scorer, decoder, model or renderer imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow/experiments/r2_chair_back_transfer_v1')
STAGES = ('BASE_POST', 'HINGE_OPP015', 'ALPHA', 'FINAL_POST')
ILLEGAL = {'proper_transverse_penetration', 'coplanar_positive_area_overlap'}
HASHES = {}


def path(p):
    p = str(p).replace('\\', '/')
    if p.startswith('E:/'):
        p = '/mnt/e/' + p[3:]
    return Path(p)


def read(p):
    return json.loads(path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    p = path(p)
    if str(p) not in HASHES:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        HASHES[str(p)] = h.hexdigest()
    return HASHES[str(p)]


def verify(files):
    for p, digest in files.items():
        assert sha(p) == digest, ('hash mismatch', p)


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def meshpath(q, stage):
    p = ROOT / 'evaluation/S2000' / q['request_id']
    return p / ('post/post.npz' if stage == 'BASE_POST' else 'cpu/' + stage + '/post.npz')


def scoredir(q, stage):
    return ROOT / 'evaluation/scores/S2000' / q['request_id'] / stage


def pair_keys(pairs):
    return {kind: {(tuple(r['pair']), r['classification']) for r in scan['contacts']}
            for kind, scan in pairs.items()}


def bad_sets(pairs, K):
    return {kind: {(int(r['pair'][0]) + (K if kind == 'FF' else 0), int(r['pair'][1]) + K)
                   for r in scan['contacts'] if r['classification'] in ILLEGAL}
            for kind, scan in pairs.items()}


def source_key(mesh, edge):
    """Corner identity reconstructed from cached edge endpoints, not a new geometry scan."""
    u, v = int(edge['vertex_u']), int(edge['vertex_v'])
    sf = mesh.get('source_face_ids', mesh.get('base_source_face_ids'))
    sc = mesh.get('source_corner_ids', mesh.get('base_source_corner_ids'))
    result = []
    faces = edge['incident_face_ids']
    if isinstance(faces, str):
        faces = json.loads(faces)
    for i in map(int, faces):
        corners = [j for j in range(3) if int(mesh['faces'][i, j]) in (u, v)]
        assert len(corners) == 2
        if i < len(sf):
            result.append((int(sf[i]), *sorted(int(sc[i, j]) for j in corners)))
        else:
            result.append(('BRIDGE', i, *sorted(corners)))
    return tuple(sorted(result, key=str))


def cached_edges(q, stage, mesh):
    with (scoredir(q, stage) / 'valid_edges.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return {kind: {source_key(mesh, e): e for e in rows if e['valid_edge'] == 'True'
                   and (e['is_C_F'] == 'True') == (kind == 'CF')}
            for kind in ('CF', 'FF')}


def ledger_edges(data):
    return {kind: {tuple(sorted((tuple(x) for x in e['source_edge_key']), key=str))
                   for e in data['edges'] if e['valid_edge'] and e['kind'] == kind}
            for kind in ('CF', 'FF')}


def main():
    tick = time.perf_counter()
    result = dict(schema='chair_back_CPU_cache_acceptance_v1', status='RUNNING',
        model_calls=0, GPU_calls=0, new_pair_scans=0, rescoring_calls=0, decoder_calls=0,
        rendering_calls=0, scope='Existing scorer definitions and saved cache identities only; no semantic or global geometric certification')
    rows = []
    try:
        manifest = read(ROOT / 'eval_manifest.json'); qq = manifest['requests']
        assert len(qq) == 16 and len({q['request_id'] for q in qq}) == 16
        for file in ('cpu_post_completion.json', 'evaluation/completions/RAW.json',
                     'evaluation/completions/BASE_POST.json',
                     'evaluation/completions/HINGE_OPP015__ALPHA__FINAL_POST.json'):
            done = read(ROOT / file)
            assert done['status'] == 'COMPLETE', file
            verify(done['files'])
        protocol = read(ROOT / 'cpu_protocol.json')
        reg = read(ROOT / 'evaluation/score_registration.json')
        verify(protocol['protected']); verify(protocol['code'])
        verify(reg['protected']); verify(reg['source_hashes'])
        totals = collections.defaultdict(collections.Counter)
        rejections = {s: collections.Counter() for s in STAGES[1:]}
        costs = []
        receipt_hashes = {}
        for q in qq:
            base = arrays(meshpath(q, 'BASE_POST')); K = int(q['K']); count = len(base['faces'])
            inp = arrays(path(q['inference_npz']))
            original_C = inp['C_vertices'][inp['C_faces_original']]
            assert equal(base['C'], original_C)
            previous = None; baseline = None
            for stage in STAGES:
                mp = meshpath(q, stage); mesh = arrays(mp); sd = scoredir(q, stage)
                auditp = sd / 'audit_receipt.json'; audit = read(auditp)
                assert audit['status'] == 'PASS' and audit['model_calls'] == 0
                assert audit['identity']['mesh_sha256'] == sha(mp)
                assert audit['identity']['registration_sha256'] == sha(ROOT / 'evaluation/score_registration.json')
                assert audit['identity']['request_id'] == q['request_id'] and audit['identity']['stage'] == stage
                verify(audit['files']); receipt_hashes[str(auditp)] = sha(auditp)
                producerp = mp.parent / ('receipt.json' if stage == 'BASE_POST' else 'complete.json')
                producer = read(producerp)
                assert audit['identity']['producer_sha256'] == sha(producerp)
                assert producer.get('post_sha256') == sha(mp)
                verify(producer['files']); receipt_hashes[str(producerp)] = sha(producerp)
                assert equal(mesh['C'], original_C) and equal(mesh['output'][:K], original_C)
                assert equal(mesh['vertices'][mesh['faces']], mesh['output'])
                # C must retain the same indexed topology, not just coordinates.
                assert equal(mesh['faces'][:K], base['faces'][:K])
                cids = np.unique(base['faces'][:K])
                assert equal(mesh['vertices'][cids], base['vertices'][cids])
                for key in ('source_face_ids', 'source_corner_ids', 'raw_slot_ids'):
                    current = mesh.get(key, mesh.get('base_' + key))
                    assert current is not None and equal(current, base[key]), (q['request_id'], stage, key)
                pairs = read(sd / 'pairs.json'); bad = bad_sets(pairs, K)
                edges = cached_edges(q, stage, mesh); metric = read(sd / 'row.json')
                for kind in ('CF', 'FF'):
                    assert metric[kind + '_illegal_pair_count'] == len(bad[kind])
                assert metric['deleted_original_POST_faces'] == 0
                bridge = mesh.get('bridge_mask', np.zeros(len(mesh['faces']), bool))
                nbridge = int(bridge.sum())
                assert len(mesh['faces']) == count + nbridge
                if stage == 'FINAL_POST':
                    assert np.array_equal(mesh['parent_post_face_id'][:count], np.arange(count))
                    assert equal(mesh['parent_source_raw_slot'][:count], base['raw_slot_ids'])
                    assert (mesh['parent_post_face_id'][bridge] == -1).all()
                    assert (mesh['parent_source_raw_slot'][bridge] == -1).all()
                    assert (mesh['face_role'][bridge] == 'BRIDGE').all()
                    assert np.array_equal(mesh['output_face_id'], np.arange(len(mesh['faces'])))
                    assert equal(mesh['faces'][:count], previous['mesh']['faces'])
                    assert equal(mesh['output'][:count], previous['mesh']['output'])
                    assert equal(mesh['vertices'], previous['mesh']['vertices'])
                if stage != 'BASE_POST':
                    identity = producer['identity']
                    assert identity['protocol_sha256'] == sha(ROOT / 'cpu_protocol.json')
                    assert identity['source_mesh_sha256'] == sha(previous['meshpath'])
                    oldpairpath = (scoredir(q, 'BASE_POST') if stage == 'HINGE_OPP015' else previous['meshpath'].parent) / 'pairs.json'
                    assert identity['source_pairs_sha256'] == sha(oldpairpath)
                    assert identity['pre_geo_sha256'] == sha(meshpath(q, 'BASE_POST').parent / 'pre_geo.npz')
                    assert identity['raw_sha256'] == sha(ROOT / 'evaluation/S2000' / q['request_id'] / 'raw/raw.npz')
                    assert all(producer[k] == 0 for k in ('model_calls', 'GPU_calls', 'D0_calls'))
                    assert pair_keys(pairs) == pair_keys(read(mp.parent / 'pairs.json'))
                    summary = read(mp.parent / 'summary.json')
                    records = read(mp.parent / 'candidate_records.json')
                    decisions = collections.Counter(r['decision'] for r in records)
                    assert dict(decisions) == summary['decisions']
                    assert decisions['ACCEPTED'] == summary['accepted']
                    rejections[stage].update(decisions)
                    ledger = read(mp.parent / 'paired_ledger.json') if stage != 'FINAL_POST' else None
                    if ledger:
                        before = ledger_edges(read(mp.parent / 'baseline_connectivity.json'))
                        after = ledger_edges(read(mp.parent / 'connectivity.json'))
                        assert before == {k: set(v) for k, v in previous['edges'].items()}
                        assert after == {k: set(v) for k, v in edges.items()}
                    row = dict(request_id=q['request_id'], split=q['split'], stage=stage,
                        original_faces=count, original_faces_deleted=0, bridge_faces=nbridge,
                        accepted=summary['accepted'], decisions=dict(decisions), stage_receipt_sha256=sha(producerp),
                        score_receipt_sha256=sha(auditp), all_original_C_coordinates_and_indices_exact=True)
                    for kind in ('CF', 'FF'):
                        old_edges = set(previous['edges'][kind]); new_edges = set(edges[kind])
                        stats = {kind + '_baseline_legal_edges': len(old_edges),
                            kind + '_retained_legal_edges': len(old_edges & new_edges),
                            kind + '_lost_legal_edges': len(old_edges - new_edges),
                            kind + '_new_legal_edges': len(new_edges - old_edges),
                            kind + '_illegal_pairs_persisted': len(previous['bad'][kind] & bad[kind]),
                            kind + '_illegal_pairs_removed': len(previous['bad'][kind] - bad[kind]),
                            kind + '_new_illegal_pairs': len(bad[kind] - previous['bad'][kind])}
                        if ledger:
                            for key, value in stats.items():
                                assert ledger['summary'][key] == value, (q['request_id'], stage, key)
                        row.update(stats)
                        assert stats[kind + '_lost_legal_edges'] == 0
                        assert stats[kind + '_new_illegal_pairs'] == 0
                    totals[stage].update({k: v for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool)})
                    rows.append(row)
                state = dict(mesh=mesh, meshpath=mp, edges=edges, bad=bad, metric=metric)
                if baseline is None:
                    baseline = state
                if stage == 'FINAL_POST':
                    for field in ('CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
                                  'ConflictCov_C', 'free_minimum_angle_lt5_fraction', 'free_q_gt10_fraction', 'free_q_mean'):
                        a, b = baseline['metric'].get(field), metric.get(field)
                        if a is not None and b is not None and b > a + 1e-12:
                            costs.append(dict(request_id=q['request_id'], split=q['split'], metric=field,
                                baseline=a, final=b, delta=b-a, status='PRESERVED_OBSERVED_INCREASE'))
                previous = state
        assert 'torch' not in sys.modules
        result.update(status='PASS', requests=len(qq), CPU_cells=len(rows), total_scored_stages_verified=64,
            stage_totals={k: dict(v) for k, v in totals.items()}, candidate_decisions={k: dict(v) for k, v in rejections.items()},
            existing_metric_increases_BASE_to_FINAL=costs, rows=rows, receipt_hashes=receipt_hashes,
            source_protocol_sha256=sha(ROOT / 'cpu_protocol.json'), score_registration_sha256=sha(ROOT / 'evaluation/score_registration.json'),
            protected_hashes_verified=len(HASHES), illegal_definition=sorted(ILLEGAL),
            bridge_identity='Unique nonnegative output_face_id; absent original parent_post_face_id and parent_source_raw_slot are -1',
            caveat='PASS authenticates cached contracts; existing illegal pairs, rejection cases and metric increases remain. No unique source-GT ranking or visual adoption inference.')
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc), completed_rows=rows)
        raise
    finally:
        result['wall_seconds'] = time.perf_counter() - tick
        result['validator_sha256'] = sha(__file__)
        dest = ROOT / 'cpu_post_acceptance.json'
        with dest.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'receipt_hashes', 'completed_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
