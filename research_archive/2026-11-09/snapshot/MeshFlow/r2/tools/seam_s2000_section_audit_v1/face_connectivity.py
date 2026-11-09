"""Queryable saved POST connectivity. No clipping, geometry repair, or rescoring."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
PREVIOUS = ROOT / 'experiments/r2_seam_s2000_fragment_audit_v1'
OUT = ROOT / 'experiments/r2_seam_s2000_section_audit_v1/topology'
VERSION = 's2000_full_saved_POST_connectivity_query_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def parse_bool(value):
    if value not in ('True', 'False'):
        raise ValueError(value)
    return value == 'True'


def build(resource_snapshot):
    snapshot = Path(resource_snapshot)
    if not snapshot.is_file():
        raise ValueError('Root resource snapshot must exist before the authorized CPU dispatch')
    if (OUT/'completion.json').exists():
        old = read(OUT/'completion.json')
        for p, h in {**old['source_hashes'], **old['output_hashes']}.items():
            assert sha(p) == h, p
        return dict(status='ALREADY_COMPLETE_VERIFIED', completion=str(OUT/'completion.json'))
    started = time.perf_counter()
    protocol_path = PREVIOUS/'protocol.json'
    protocol = read(protocol_path)
    section_protocol_path = OUT.parent/'protocol.json'
    section_protocol = read(section_protocol_path)
    assert section_protocol['status'] == 'FROZEN'
    assert section_protocol['requests'] == protocol['requests']
    previous = read(PREVIOUS/'numeric/completion.json')
    assert previous['status'] == 'COMPLETE' and protocol['model'] == 'PLAIN_S2000'
    paths = [protocol_path, PREVIOUS/'numeric/completion.json', PREVIOUS/'numeric/per_face.csv',
             PREVIOUS/'numeric/components.csv', PREVIOUS/'numeric/deletion.csv', snapshot, Path(__file__), section_protocol_path]
    source_hashes = {str(p): sha(p) for p in paths}
    for p in paths[2:5]:
        assert source_hashes[str(p)] == previous['output_hashes'][str(p)]
    shapes = {(r['request_id'], int(r['source_raw_slot'])): r
              for r in table(paths[2]) if r['stage'] == 'POST'}
    component_rows = table(paths[3])
    deletes = {(r['request_id'], int(r['source_raw_slot'])): r for r in table(paths[4])}
    components = {}
    for row in component_rows:
        rid, kind = row['request_id'], row['connectivity']
        slots = json.loads(row['source_raw_slot_ids']); faces = json.loads(row['post_face_ids'])
        assert len(slots) == len(faces) == int(row['face_count'])
        for slot, face in zip(slots, faces):
            key = rid, kind, slot
            assert key not in components
            components[key] = dict(component_id=int(row['component_id']), face_count=int(row['face_count']),
                has_C=parse_bool(row['has_C']), C_faces=int(row['C_faces']), seam_faces=int(row['seam_faces']),
                bulk_faces=int(row['bulk_faces']), post_face_index=face,
                component_area_model2=float(row['area_model2']),
                component_free_area_fraction=float(row['area_fraction_total_free']),
                small_free_only_detached=parse_bool(row['small_free_only_detached']))
    OUT.mkdir(parents=True, exist_ok=True)
    all_faces, request_records, files = [], [], []
    for q in protocol['requests']:
        rid = q['request_id']; mesh_path = Path(q['paths']['POST'])
        source_hashes[str(mesh_path)] = sha(mesh_path)
        assert source_hashes[str(mesh_path)] == protocol['source_hashes'][str(mesh_path)]
        with np.load(mesh_path, allow_pickle=False) as z:
            faces, vertices = z['faces'].copy(), z['vertices'].copy()
            ids, raw_ids = z['source_face_ids'].copy(), z['raw_slot_ids'].copy()
            corners, seam = z['source_corner_ids'].copy(), z['raw_seam_slot_mask'].copy()
            assert np.array_equal(ids, raw_ids)
            assert np.array_equal(vertices[faces], z['output'])
        assert len(set(ids.tolist())) == len(ids)
        assert np.all(np.sort(corners, axis=1) == np.arange(3))
        edge_faces = collections.defaultdict(set); vertex_faces = collections.defaultdict(set)
        for face_index, f in enumerate(faces):
            for v in f:
                vertex_faces[int(v)].add(face_index)
            for j in range(3):
                edge_faces[tuple(sorted((int(f[j]), int(f[(j+1) % 3]))))].add(face_index)
        records = []
        for face_index, (slot, f) in enumerate(zip(ids, faces)):
            slot = int(slot); old = shapes[rid, slot]
            role = 'C' if slot < q['K'] else 'seam' if seam[slot] else 'bulk'
            assert old['role'] == role and int(old['post_face_index']) == face_index
            neighbors = dict(vertex=set().union(*(vertex_faces[int(v)] for v in f)) - {face_index},
                             edge=set().union(*(edge_faces[tuple(sorted((int(f[j]), int(f[(j+1) % 3]))))] for j in range(3))) - {face_index})
            row = dict(request_id=rid, parent_id=q['parent_id'], split=q['split'], noise_index=q['noise_index'],
                stage='POST_FULL_UNCLIPPED', source_raw_slot=slot, post_face_index=face_index, role=role,
                post_vertex_ids=f.tolist(), source_corner_ids=corners[face_index].tolist(),
                post_corner_coordinates=vertices[f].astype(float).tolist(),
                minimum_angle_degrees=float(old['minimum_angle_degrees']), area_model2=float(old['area_model2']),
                area_over_L2=float(old['area_over_L2']), longest_edge_model=float(old['longest_edge_model']),
                small_angle=parse_bool(old['small_angle']), tiny_1e5=parse_bool(old['tiny_1e5']),
                geometry_metrics='READ_ONLY_REUSE_FROM_PREVIOUS_PER_FACE_CSV')
            for kind in ('edge', 'vertex'):
                comp = components[rid, kind, slot]
                assert comp['post_face_index'] == face_index
                assert int(old[kind+'_component_id']) == comp['component_id']
                row.update({kind+'_'+k: v for k, v in comp.items() if k != 'post_face_index'})
                row[kind+'_neighbor_post_faces'] = sorted(neighbors[kind])
                row[kind+'_neighbor_raw_slots'] = sorted(int(ids[i]) for i in neighbors[kind])
                for i in neighbors[kind]:
                    assert components[rid, kind, int(ids[i])]['component_id'] == comp['component_id']
            row['edge_incidence_records'] = [dict(vertices=list(edge), incident_raw_slots=sorted(int(ids[i]) for i in edge_faces[edge]))
                for edge in [tuple(sorted((int(f[j]), int(f[(j+1) % 3])))) for j in range(3)]]
            records.append(row); all_faces.append(row)
        missing = []
        for slot in range(q['N']):
            if slot not in set(ids.tolist()):
                old = deletes[rid, slot]
                missing.append(dict(source_raw_slot=slot, status='DELETED_BY_SAVED_GEO_NOT_A_POST_FACE',
                    role=old['role'], GEO_reason=old['GEO_reason'], duplicate_of_source_face=int(old['duplicate_of_source_face'])))
        info = dict(version=VERSION, request_id=rid, input_post_path=str(mesh_path), input_post_sha256=source_hashes[str(mesh_path)],
            N=q['N'], K=q['K'], retained_faces=len(ids), deleted_faces=len(missing),
            identity='Original generated RAW slot; not the unknown full-parent source face ID',
            connectivity='Full saved indexed POST including C; no section clipping or legal-contact filter',
            faces=records, deleted_source_slots=missing)
        output = OUT/'requests'/(rid+'.json'); write(output, info); files.append(output)
        request_records.append(dict(request_id=rid, path=str(output), retained_faces=len(ids), deleted_faces=len(missing)))
    assert len(request_records) == 28 and len(all_faces) == len(shapes)
    csvwrite(OUT/'per_face_connectivity.csv', all_faces); files.append(OUT/'per_face_connectivity.csv')
    write(OUT/'index.json', dict(version=VERSION, requests=request_records,
        lookup='Use --query REQUEST_ID --slot ORIGINAL_RAW_SLOT; deleted slots return saved deletion reason',
        limits='Vertex connection is weaker than a shared edge; graph connectivity does not certify legal contact or a desirable part'))
    files.append(OUT/'index.json')
    for p, h in source_hashes.items():
        assert sha(p) == h, p
    result = dict(status='COMPLETE', version=VERSION, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seconds=time.perf_counter()-started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        requests=28, retained_face_records=len(all_faces), source_hashes=source_hashes,
        output_hashes={str(p): sha(p) for p in files}, worker_count=1, threads=1,
        original_assets_unchanged=True, model_calls=0, GEO_calls=0, rescoring_calls=0, clipping_calls=0)
    write(OUT/'completion.json', result)
    return {k: v for k, v in result.items() if k not in ('source_hashes', 'output_hashes')}


def query(request_id, slot):
    if not request_id or Path(request_id).name != request_id:
        raise ValueError('Invalid request ID')
    doc = read(OUT/'requests'/(request_id+'.json'))
    for r in doc['faces']:
        if r['source_raw_slot'] == slot:
            return r
    for r in doc['deleted_source_slots']:
        if r['source_raw_slot'] == slot:
            return r
    raise ValueError('Unknown slot')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true')
    ap.add_argument('--resource-snapshot'); ap.add_argument('--query'); ap.add_argument('--slot', type=int)
    args = ap.parse_args()
    if args.build:
        if not args.resource_snapshot:
            ap.error('--build requires --resource-snapshot')
        result = build(args.resource_snapshot)
    elif args.query is not None and args.slot is not None:
        result = query(args.query, args.slot)
    else:
        ap.error('Choose --build or --query REQUEST --slot RAW_SLOT')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
