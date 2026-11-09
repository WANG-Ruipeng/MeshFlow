"""Three preregistered 2D cursor queries on original section planes, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import json
import hashlib
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'experiments/r2_seam_s2000_section_audit_v1'
QUERIES = [('3f741759_n2', 'side', [-.80, .81]),
           ('608b46b0_n3', 'front', [-.60, -.10]),
           ('608b46b0_n3', 'front', [.80, -.10])]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segments(triangle, axis, position, eps):
    triangle = np.asarray(triangle, np.float64)
    d = triangle[:, axis]-position
    if np.all(np.abs(d) <= eps):
        return [(triangle[i].copy(), triangle[(i+1) % 3].copy(), 'COPLANAR_TRIANGLE_EDGE') for i in range(3)]
    points = []
    for i in range(3):
        if abs(d[i]) <= eps:
            p = triangle[i].copy(); p[axis] = position; points.append(p)
        j = (i+1) % 3
        if d[i]*d[j] < 0:
            p = triangle[i] + d[i]/(d[i]-d[j])*(triangle[j]-triangle[i])
            p[axis] = position; points.append(p)
    unique = []
    for p in points:
        if not any(np.linalg.norm(p-q) <= eps for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return []
    pairs = [(np.linalg.norm(a-b), a, b) for i, a in enumerate(unique) for b in unique[i+1:]]
    length, a, b = max(pairs, key=lambda z: z[0])
    return [(a, b, 'FINITE_TRIANGLE_PLANE_INTERSECTION')] if length > eps else []


def distance(point, a, b):
    delta = b-a
    t = float(np.clip(np.dot(point-a, delta)/np.dot(delta, delta), 0, 1))
    nearest = a+t*delta
    return float(np.linalg.norm(point-nearest)), t, nearest.tolist()


def main():
    started = time.perf_counter()
    protocol_path = OUT/'protocol.json'
    protocol = read(protocol_path); sources = {str(protocol_path): sha(protocol_path), str(Path(__file__)): sha(__file__)}
    requests = {q['request_id']: q for q in protocol['requests']}
    cases = []
    eps = 1e-10*float(protocol['L'])
    for rid, plane, xy in QUERIES:
        q = requests[rid]; axis = 0 if plane == 'side' else 2; projected = [2, 1] if plane == 'side' else [0, 1]
        position = float(protocol['bounds'][q['condition_id']][plane+'_positions'][1])
        topo_path = OUT/'topology/requests'/(rid+'.json')
        topology = read(topo_path); sources[str(topo_path)] = sha(topo_path)
        ledger = {int(r['source_raw_slot']): r for r in topology['faces']}
        aa = {}
        for stage in ('RAW', 'D0', 'RESTORED', 'POST'):
            path = Path(q['paths'][stage]); sources[str(path)] = sha(path)
            assert sources[str(path)] == protocol['source_hashes'][str(path)]
            with np.load(path, allow_pickle=False) as z:
                key = 'raw' if stage == 'RAW' else 'official_unconstrained_proposal' if stage == 'D0' else 'output'
                tri = z[key].astype(np.float64)
                ids = z['source_face_ids'].copy() if stage == 'POST' else np.arange(q['N'])
                aa[stage] = {int(slot): t for slot, t in zip(ids, tri)}
        candidates = []
        for slot, triangle in aa['POST'].items():
            for si, (a, b, kind) in enumerate(segments(triangle, axis, position, eps)):
                dist, along, nearest = distance(np.asarray(xy), a[projected], b[projected])
                candidates.append(dict(source_raw_slot=slot, segment_index=si, section_kind=kind,
                    cursor_distance_model=dist, fraction_along_segment=along, closest_projected_point=nearest,
                    endpoints_xyz=[a.tolist(), b.tolist()], endpoints_plot=[a[projected].tolist(), b[projected].tolist()]))
        candidates.sort(key=lambda r: (r['cursor_distance_model'], r['source_raw_slot'], r['segment_index']))
        selected = []
        for rank, candidate in enumerate(candidates[:3], 1):
            slot = candidate['source_raw_slot']; face = ledger[slot]
            candidate['rank'] = rank
            candidate['full_POST_topology'] = {k: face[k] for k in ('post_face_index','role','post_vertex_ids','source_corner_ids',
                'edge_component_id','edge_face_count','edge_has_C','edge_neighbor_raw_slots',
                'vertex_component_id','vertex_face_count','vertex_has_C','vertex_neighbor_raw_slots',
                'minimum_angle_degrees','area_model2','small_angle','tiny_1e5')}
            candidate['connectivity_interpretation'] = dict(
                edge_path_to_C=face['edge_has_C'],
                vertex_path_to_C_without_edge_path=face['vertex_has_C'] and not face['edge_has_C'],
                edge_component_joins_larger_component_only_via_vertices=face['vertex_face_count'] > face['edge_face_count'],
                face_has_no_edge_neighbors_but_has_vertex_neighbors=not face['edge_neighbor_raw_slots'] and bool(face['vertex_neighbor_raw_slots']),
                completely_isolated_in_full_vertex_graph=face['vertex_face_count'] == 1)
            trace = []
            for stage in ('RAW', 'D0', 'RESTORED', 'POST'):
                triangle = aa[stage][slot]
                segs = segments(triangle, axis, position, eps)
                trace.append(dict(stage=stage, raw_slot=slot, triangle_xyz=triangle.tolist(),
                    plane_intersection_status='SEGMENT' if segs else 'NO_NONZERO_SEGMENT',
                    plane_signed_vertex_distances=(triangle[:, axis]-position).tolist(),
                    segments=[dict(endpoints_xyz=[a.tolist(), b.tolist()], endpoints_plot=[a[projected].tolist(), b[projected].tolist()],
                                   kind=kind, cursor_distance_model=distance(np.asarray(xy), a[projected], b[projected])[0]) for a,b,kind in segs]))
            candidate['same_raw_slot_four_stage_trace'] = trace
            selected.append(candidate)
        cases.append(dict(request_id=rid, plane=plane, axis=axis, plane_position=position,
            plot_axes=['z','y'] if plane == 'side' else ['x','y'], approximate_cursor=xy,
            identification_status='THREE_NEAREST_SEGMENTS_TO_APPROXIMATE_CURSOR_NOT_CERTIFIED_VISUAL_IDENTITY',
            tested_POST_segments=len(candidates), nearest=selected))
    for path, h in sources.items():
        assert sha(path) == h
    result = dict(status='COMPLETE', cpu_seconds=time.perf_counter()-started, cursor_queries=3,
        cases=cases, source_hashes_before_after=sources, epsilon_model=eps,
        limits='Finite-plane segment query only; full POST graph defines connectivity, never clipped geometry. No illegal-contact or inside/outside claim.',
        model_calls=0, GEO_calls=0, scoring_calls=0)
    (OUT/'case_trace.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','seconds':result['cpu_seconds'],'cases':[dict(request_id=c['request_id'],cursor=c['approximate_cursor'],
        nearest=[{k:r[k] for k in ('rank','source_raw_slot','cursor_distance_model','endpoints_plot','full_POST_topology','connectivity_interpretation')} for r in c['nearest']]) for c in cases]},indent=2))


if __name__ == '__main__':
    main()
