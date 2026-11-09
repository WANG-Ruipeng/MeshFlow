"""Independent, model-free endpoint/operation audit of the 56 saved HINGE cells.

The incidence and direction checks below do not call the new engine.  Cached
full pair classifications are reused, never recomputed.  Intermediate collision
scans were not persisted, so this verifies those classifications at endpoints
and verifies geometry/topology/anchors at every saved accepted operation.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from r2.pilots.seam_hinge_quality_v1 import hinge as OLD
from r2.pilots.seam_hinge_quality_v1.hinge_audit import geometry

C = OLD.C
OUT = C.ROOT / 'experiments/r2_hinge_opposite_v1'
ILLEGAL = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))
LEGAL = frozenset(('shared_edge_contact', 'shared_vertex_contact', 'coplanar_boundary_or_tolerance_contact',
                   'noncoplanar_edge_or_tangent_contact', 'point_or_tangent_contact'))
IDENTITIES = ('source_face_ids', 'source_corner_ids', 'raw_slot_ids', 'source_face_kept',
              'source_face_reason', 'duplicate_of_source_face', 'raw_anchor_ids',
              'output_anchor_ids', 'raw_seam_slot_mask', 'known_faces')


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def table(path):
    with C.resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def key_of(occ, sf, sc):
    return tuple(sorted((int(sf[i]), *sorted((int(sc[i, a]), int(sc[i, b])))) for i, a, b in occ))


def incidence(faces):
    result = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            result[tuple(sorted((int(face[a]), int(face[b]))))].append((i, a, b))
    return result


def topo(mesh, pairs=None):
    """Reconstruct indexed topology and the frozen legal-edge definition."""
    v, f, sf, sc = (mesh[k] for k in ('vertices', 'faces', 'source_face_ids', 'source_corner_ids'))
    K = int(mesh['known_faces'])
    assert len(set(map(int, sf))) == len(sf) == len(f)
    assert sc.shape == f.shape and all(len(set(map(int, z))) == 3 for z in sc)
    assert same(v[f], mesh['output'])
    groups = collections.defaultdict(list)
    for i, face in enumerate(f):
        groups[tuple(sorted(map(int, face)))].append(i)
    dupfaces = {i for g in groups.values() if len(g) > 1 for i in g}
    area = geometry(v[f])['area']
    cached = {} if pairs is None else {
        kind: {tuple(map(int, z['pair'])): z['classification'] for z in pairs[kind]['contacts']}
        for kind in ('CF', 'FF')}
    all_edges, two, legal, nm = {}, {}, {}, set()
    for (u, w), occ in incidence(f).items():
        key = key_of(occ, sf, sc)
        rows = [i for i, a, b in occ]
        kind = 'CC' if max(rows) < K else 'CF' if min(rows) < K else 'FF'
        length = float(np.linalg.norm(v[u].astype(float) - v[w]))
        record = dict(kind=kind, source_edge_key=key, vertex_u=u, vertex_v=w,
                      incidence=len(occ), incident_face_ids=rows, winding_opposite=None,
                      length_model=length, valid_edge=False)
        if len(occ) > 2:
            nm.add(key)
        if len(occ) == 2:
            (i, a, b), (j, c, d) = occ
            opposite = bool(i != j and u != w and f[i, a] == f[j, d] and f[i, b] == f[j, c])
            record['winding_opposite'] = opposite
            two[key] = record
            if pairs is not None and kind != 'CC':
                pi, pj = ((i, j-K) if i < K else (j, i-K)) if kind == 'CF' else tuple(sorted((i-K, j-K)))
                classification = cached[kind].get((pi, pj), 'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
                record['pair_classification'] = classification
                valid = (i != j and u != w and length > 0 and area[i] > 0 and area[j] > 0
                         and i not in dupfaces and j not in dupfaces and classification in LEGAL)
                record['valid_edge'] = valid
                if valid:
                    legal[key] = record
        all_edges[key] = record
    return dict(all=all_edges, two=two, legal=legal, nm=nm)


def source_pairs(pairs, mesh, unknown=False):
    sf, K = mesh['source_face_ids'], int(mesh['known_faces'])
    answer = {kind: set() for kind in ('CF', 'FF')}
    for kind in answer:
        scan = pairs[kind]
        assert scan['tested_pairs'] == scan['expected_full_pairs']
        assert scan['affected_subset'] is None and scan['independent_verification'] is True
        expected = K*(len(sf)-K) if kind == 'CF' else (len(sf)-K)*(len(sf)-K-1)//2
        assert expected == scan['expected_full_pairs']
        seen = set()
        for row in scan['contacts']:
            ii, jj = map(int, row['pair'])
            assert (ii, jj) not in seen
            seen.add((ii, jj))
            label = row['classification']
            select = (any(s in label.lower() for s in ('unknown', 'unresolved', 'degenerate'))
                      or row.get('detector', {}).get('resolved') is False) if unknown else label in ILLEGAL
            if select:
                i, j = int(sf[ii if kind == 'CF' else K+ii]), int(sf[K+jj])
                answer[kind].add((i, j) if kind == 'CF' else tuple(sorted((i, j))))
    return answer


def verify_saved_ledger(mesh, topology, path):
    saved = C.read(path)
    assert saved['source_face_ids'] == mesh['source_face_ids'].tolist()
    expected = {k: r for k, r in topology['all'].items() if r['kind'] != 'CC'}
    records = {tuple(tuple(x) for x in r['source_edge_key']): r for r in saved['edges']}
    assert set(records) == set(expected)
    for key, row in records.items():
        got = expected[key]
        for field in ('kind', 'vertex_u', 'vertex_v', 'incidence', 'incident_face_ids', 'winding_opposite', 'valid_edge'):
            assert row[field] == got[field], (path, key, field)
        assert abs(row['length_model']-got['length_model']) < 1e-11


def replay(base, pre, candidate, records, delta):
    """Independent original-cluster replay; inspect all new edges after each edit."""
    v, f = base['vertices'].copy(), base['faces']
    roots = np.arange(len(v), dtype=np.int64)
    sf, sc, K = base['source_face_ids'], base['source_corner_ids'], int(base['known_faces'])
    fixed = set(map(int, f[:K].reshape(-1)))
    hard = base['output_anchor_ids'] >= 0
    assert set(map(int, f[hard])) <= fixed
    cv = base['corner_vertex_ids'].reshape(-1)
    initial = geometry(v[f]); baseline = topo(base)
    old_dups = OLD.B.numeric_duplicate_pairs(v, f)
    accepted, maximum, details = 0, 0., []
    for row in records:
        if row['decision'] != 'ACCEPTED':
            continue
        previous_tri = v[roots[f]]
        previous = geometry(previous_tri)
        current_mesh = dict(base, vertices=v, faces=roots[f], output=previous_tri)
        current_topo = topo(current_mesh)
        u, w = row['endpoint_original_vertices']; a, b = int(roots[u]), int(roots[w])
        assert a != b and np.linalg.norm(v[a].astype(float)-v[b]) <= delta*C.L+1e-12
        expected_group = set(np.flatnonzero(np.isin(roots, [a, b])).tolist())
        assert len(row['merged_clusters']) == 1
        op = row['merged_clusters'][0]
        group = set(op['original_cluster_ids']); assert group == expected_group
        pinned = group & fixed; assert len(pinned) <= 1
        target = np.asarray(op['target'], np.float32)
        expected = v[next(iter(pinned))] if pinned else ((v[a].astype(float)+v[b].astype(float))/2).astype(np.float32)
        assert same(target, expected)
        expected_root = a if a in fixed else b if b in fixed else min(a, b)
        assert op['new_root'] == expected_root
        corners = np.flatnonzero(np.isin(cv, list(group)))
        displacement = np.linalg.norm(pre.reshape(-1, 3)[corners].astype(float)-target, axis=1)
        assert len(corners) == op['original_corner_count']
        assert displacement.max() <= delta*C.L+1e-12
        assert abs(displacement.max()-op['maximum_PRE_GEO_displacement']) < 1e-12
        v[list(group)] = target; roots[list(group)] = expected_root
        ff = roots[f]; tri = v[ff]
        assert same(tri[:K], base['output'][:K]) and same(tri[hard], base['output'][hard])
        affected = np.flatnonzero(np.any(np.isin(f, list(group)), axis=1)); affected = affected[affected >= K]
        assert sorted(sf[affected].tolist()) == sorted(row['affected_source_raw_slots'])
        assert all(len(set(map(int, ff[i]))) == 3 for i in affected)
        shape = geometry(tri)
        assert np.all(shape['area'][affected] > 1e-12*C.L*C.L)
        for ref in (initial, previous):
            assert not np.any((shape['angle'][affected] < 5) & (shape['angle'][affected] < ref['angle'][affected]-1e-10))
            assert not np.any((shape['q'][affected] > 10) & (shape['q'][affected] > ref['q'][affected]+1e-10))
            assert np.all((shape['normal'][affected]*ref['normal'][affected]).sum(1) >= .5-1e-12)
        after = topo(dict(base, vertices=v, faces=ff, output=tri))
        added = set(after['two'])-set(current_topo['two'])
        assert all(after['two'][k]['winding_opposite'] for k in added), (row['rank'], added)
        assert not after['nm']-baseline['nm']
        assert not OLD.B.numeric_duplicate_pairs(v, ff)-old_dups
        maximum = max(maximum, float(displacement.max())); accepted += 1
        details.append(dict(rank=row['rank'],new_two_face_edges=len(added),
                            maximum_PRE_GEO_displacement=float(displacement.max())))
    assert same(v, candidate['vertices']) and same(roots[f], candidate['faces'])
    assert same(roots, candidate['original_POST_vertex_to_hinge_vertex'])
    assert same(v[roots[f]], candidate['output'])
    assert same(candidate['C'], candidate['output'][:K]) and same(candidate['free'], candidate['output'][K:])
    for field in IDENTITIES:
        assert same(base[field], candidate[field]), field
    mapped = base['corner_vertex_ids'].copy(); valid = mapped >= 0; mapped[valid] = roots[mapped[valid]]
    assert same(mapped, candidate['corner_vertex_ids'])
    changed = [i for i in range(len(v)) if roots[i] != i or not same(v[i], base['vertices'][i])]
    corners = np.flatnonzero(np.isin(cv, changed))
    final_displacement = np.linalg.norm(pre.reshape(-1, 3)[corners].astype(float)-v[roots[cv[corners]]], axis=1)
    assert not len(corners) or final_displacement.max() <= delta*C.L+1e-12
    return dict(accepted=accepted, maximum_PRE_GEO_displacement=maximum,
                final_maximum_PRE_GEO_displacement=float(final_displacement.max()) if len(corners) else 0.,
                checked_accepted_operations=details, moved_original_corners=len(corners))


def main():
    start = time.perf_counter()
    destination = OUT/'independent_audit'
    done = OUT/'independent_audit.json'
    if done.exists():
        result = C.read(done); OLD.B.verify(result['files'])
        print(json.dumps(dict(status=result['status'],cells=result['cells'],already_complete=True))); return
    protocol = C.read(OUT/'protocol.json')
    assert protocol['cells'] == 56 and protocol['require_opposite_winding'] is True
    assert C.read(OUT/'execution_complete.json')['status'] == 'COMPLETE'
    OLD.B.verify(protocol['protected']); OLD.B.verify(protocol['code'])
    input_hashes = {str(OUT/'protocol.json'): C.sha(OUT/'protocol.json')}
    rows, details, replay_details = [], [], []
    for q in protocol['requests']:
        paths = OLD.paths('PLAIN_S2000', q)
        base = C.arrays(paths['post']); pre = C.arrays(paths['pre'])['output']; bp = C.read(paths['pairs'])
        baseline = topo(base, bp); original_bad = source_pairs(bp, base); original_unknown = source_pairs(bp, base, True)
        for delta in protocol['deltas']:
            label = 'HINGE_OPP'+str(round(delta*1000)).zfill(3)
            folder = OUT/'cells'/label/q['request_id']
            receipt = C.read(folder/'complete.json'); assert receipt['status'] == 'COMPLETE'
            OLD.B.verify(receipt['files'])
            input_hashes[str(folder/'complete.json')] = C.sha(folder/'complete.json')
            mesh = C.arrays(folder/'post.npz'); pairs = C.read(folder/'pairs.json')
            nt = topo(mesh, pairs); nb = source_pairs(pairs, mesh); nu = source_pairs(pairs, mesh, True)
            verify_saved_ledger(mesh, nt, folder/'connectivity.json')
            old_folder = OLD.OUT/'cells'/OLD.label('PLAIN_S2000', delta)/q['request_id']
            oldmesh = C.arrays(old_folder/'post.npz'); oldpairs = C.read(old_folder/'pairs.json')
            ot = topo(oldmesh, oldpairs); ob = source_pairs(oldpairs, oldmesh)
            verify_saved_ledger(oldmesh, ot, old_folder/'connectivity.json')
            lost = set(baseline['legal'])-set(nt['legal'])
            added = set(nt['legal'])-set(baseline['legal'])
            new_two = set(nt['two'])-set(baseline['two'])
            old_new_two = set(ot['two'])-set(baseline['two'])
            assert not lost
            assert not nt['nm']-baseline['nm']
            assert all(nt['two'][k]['winding_opposite'] for k in new_two)
            for key in set(baseline['two']) & set(nt['two']):
                assert baseline['two'][key]['winding_opposite'] == nt['two'][key]['winding_opposite']
            actual = replay(base, pre, mesh, C.read(folder/'candidate_records.json'), delta)
            assert actual['accepted'] == receipt['summary']['accepted']
            assert abs(actual['final_maximum_PRE_GEO_displacement']-receipt['summary']['max_PRE_GEO_displacement']) < 1e-12
            fates = table(folder/'baseline_edge_fates.csv'); newrows = table(folder/'new_edges.csv')
            rowkey = lambda r: tuple(tuple(z) for z in json.loads(r['source_edge_key']))
            assert {rowkey(r) for r in fates} == set(baseline['legal'])
            assert all(r['fate'] == 'RETAINED_LEGAL' for r in fates)
            assert {rowkey(r) for r in newrows} == added
            pairing = table(folder/'illegal_pair_fates.csv')
            info = {}
            for kind in ('CF', 'FF'):
                assert not nb[kind]-original_bad[kind]
                assert not nu[kind]-original_unknown[kind]
                selected = {(int(r['source_face_a']),int(r['source_face_b'])):r['fate'] for r in pairing if r['kind'] == kind}
                assert set(selected) == original_bad[kind] | nb[kind]
                for pair in selected:
                    expected = 'PERSISTENT_ILLEGAL' if pair in nb[kind] else 'RETAINED_PAIR_NO_LONGER_ILLEGAL'
                    assert selected[pair] == expected
                newlegal = sum(nt['legal'][k]['kind'] == kind for k in added)
                info.update({kind+'_baseline_legal':sum(r['kind']==kind for r in baseline['legal'].values()),
                    kind+'_old_hinge_new_legal':sum(ot['legal'][k]['kind']==kind for k in set(ot['legal'])-set(baseline['legal'])),
                    kind+'_new_hinge_new_legal':newlegal,kind+'_lost_legal':0,
                    kind+'_baseline_illegal':len(original_bad[kind]),kind+'_old_hinge_illegal':len(ob[kind]),
                    kind+'_new_hinge_illegal':len(nb[kind]),kind+'_new_illegal':0,
                    kind+'_removed_illegal':len(original_bad[kind]-nb[kind]),kind+'_new_unknown':0})
                assert receipt['pair_summary'][kind+'_new_legal_edges'] == newlegal
                assert receipt['pair_summary'][kind+'_illegal_pairs_removed'] == len(original_bad[kind]-nb[kind])
            rows.append(dict(request_id=q['request_id'],split=q['split'],variant=label,delta=delta,
                accepted=actual['accepted'],faces_before=len(base['faces']),faces_after=len(mesh['faces']),deleted_faces=0,
                original_all_two_face_edges=len(baseline['two']),original_same_direction_two_face_edges=sum(not r['winding_opposite'] for r in baseline['two'].values()),
                old_hinge_new_same_direction_two_face_edges=sum(not ot['two'][k]['winding_opposite'] for k in old_new_two),
                new_hinge_new_same_direction_two_face_edges=0,new_hinge_all_new_two_face_edges=len(new_two),
                C_and_seam_anchors_bitwise=True,source_face_corner_identity_exact=True,
                maximum_PRE_GEO_displacement=actual['maximum_PRE_GEO_displacement'],
                old_vs_new_arrays_identical=same(mesh['output'],oldmesh['output']) and same(mesh['faces'],oldmesh['faces']),**info))
            details.append(dict(request_id=q['request_id'],variant=label,
                original_legal_source_edges=sorted(baseline['legal']),new_legal_source_edges=sorted(added),
                new_all_two_face_source_edges=sorted(new_two),
                old_new_same_direction_source_edges=sorted(k for k in old_new_two if not ot['two'][k]['winding_opposite']),
                old_hinge_legal_edges_not_in_new=sorted(set(ot['legal'])-set(nt['legal'])),
                new_hinge_legal_edges_not_in_old=sorted(set(nt['legal'])-set(ot['legal']))))
            replay_details.append(dict(request_id=q['request_id'],variant=label,**actual))
        print(json.dumps(dict(audited_request=q['request_id'],completed_cells=len(rows))),flush=True)
    assert len(rows) == 56 and 'torch' not in sys.modules
    OLD.B.verify(protocol['protected']); OLD.B.verify(protocol['code']); OLD.B.verify(input_hashes)
    C.csvwrite(destination/'per_cell.csv',rows)
    C.write(destination/'source_identity_sets.json',details)
    C.write(destination/'accepted_operation_replay.json',replay_details)
    result = dict(status='PASS_INDEPENDENT_ARRAY_AND_SOURCE_IDENTITY_AUDIT',cells=len(rows),
        accepted_operations=sum(r['accepted'] for r in rows),
        old_new_same_direction_edges=sum(r['old_hinge_new_same_direction_two_face_edges'] for r in rows),
        new_new_same_direction_edges=0,original_legal_edges_lost=0,new_illegal_pairs=0,deleted_faces=0,
        changed_cells_vs_old=sum(not r['old_vs_new_arrays_identical'] for r in rows),
        checks=['Independent actual indexed incidence/directions: all new incidence-two CC/CF/FF edges, including illegal pairs',
                'Every accepted original-cluster edit replayed exactly with PRE_GEO member displacement cap',
                'Every accepted edit C/seam bitwise anchors, original and previous shape/normal guards, no new duplicates/nonmanifold',
                'Full source-face/corner identities and original GEO member map retained; no face deletion/reorientation',
                'Saved full endpoint CF/FF classifications independently paired by source IDs; no new illegal/unknown pairs',
                'Original legal edges independently reconstructed and all retained; CSV ledgers checked against actual arrays',
                'Baseline, historical HINGE and corrected HINGE compared without changing old assets'],
        limits=['Uses already executed full endpoint detectors; no new detector or model calls.',
                'Intermediate collision scans were not persisted; not independently recertified here.',
                'Discrete endpoint safety is not continuous collision detection or a guarantee of semantic quality.',
                'Old same-direction edges may remain. Preserving edges/faces does not preserve area or guarantee lower I/q.',
                'Historical HINGE gains rejected by the new guard may be lost relative to old HINGE, while baseline legal edges are protected.'],
        seconds=time.perf_counter()-start,CPU_threads=1,model_calls=0,GPU_calls=0,new_quality_scans=0,
        protected_input_count=len(protocol['protected']),audit_source_sha256=C.sha(__file__),
        input_receipt_hashes=input_hashes,files={str(p):C.sha(p) for p in destination.iterdir() if p.is_file()})
    C.write(done,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','limits','input_receipt_hashes','files')}),flush=True)


if __name__ == '__main__':
    main()
