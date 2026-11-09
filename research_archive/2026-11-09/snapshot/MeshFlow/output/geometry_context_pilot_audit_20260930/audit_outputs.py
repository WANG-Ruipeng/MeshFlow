"""Independent, CPU-only use of the unchanged local geometry auditors.

This file intentionally lives outside native_t1 and Git. It imports historical
auditors only inside this standalone process. No model, optimizer, OT, guidance,
distance-based replacement for intersections, or mesh repair is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'geometry_context_external_audit_v1'
MODELS = ('T0', 'T_base', 'T_geo', 'T_graph')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def patch_key(parent, patch_id):
    return f'P{int(parent)}:{str(patch_id)}'


def resolve_path(value, base):
    text = str(value).replace('\\', '/')
    if len(text) > 2 and text[1] == ':' and os.name != 'nt':
        text = '/mnt/' + text[0].lower() + text[2:]
    if os.name == 'nt' and text.startswith('/mnt/') and len(text) > 7:
        text = text[5].upper() + ':/' + text[7:]
    path = Path(text)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _auditors():
    # The offline process alone has access to these local historical modules.
    # The production native_t1 package never imports this script or experiments.
    sys.path.insert(0, str(ROOT))
    from experiments.mf_h1_local0.guided_native_geometry_audit import pair_scan
    from experiments.mf_h1_local0.guided_native_geometry_geometry import DETECTOR_POLICY
    from experiments.mf_h1_local0.gpu_guidance_evaluation import audit_context, _detector_view
    from experiments.mf_h1_local0 import t1_native_confirmation_topology as topology
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Offline geometry process unexpectedly initialized CUDA')
    return dict(pair_scan=pair_scan, policy=DETECTOR_POLICY, audit_context=audit_context,
                detector_view=_detector_view, topology=topology, torch=torch)


def _float_triangles(value, faces, name):
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape not in ((faces, 9), (faces, 3, 3)):
        raise ValueError(f'{name} requires original FP32[{faces},9] or [{faces},3,3]')
    value = np.ascontiguousarray(value.reshape(faces, 3, 3))
    if not np.isfinite(value).all():
        raise ValueError(f'{name} has nonfinite coordinates; no replacement or repair')
    return value


def _source_patch_audit(patch, arrays, api, ledger):
    started = time.perf_counter()
    C = _float_triangles(arrays[patch['constraints_key']], 12, 'C')
    target = _float_triangles(arrays[patch['target_key']], 112, 'target')
    known = np.asarray(arrays[patch['known_face_ids_key']])
    if (known.shape != (12,) or not np.issubdtype(known.dtype, np.integer)
            or len(set(known.tolist())) != 12 or np.any((known < 0) | (known >= 112))):
        raise ValueError('Need twelve distinct canonical source face IDs')
    if target[known].tobytes() != C.tobytes():
        raise ValueError('C differs from the untouched registered target subset')
    L = float(patch['bbox_diagonal'])
    actual_L = float(np.linalg.norm(np.ptp(target.reshape(-1, 3), axis=0)))
    if not np.isfinite(L) or L <= 0 or L != actual_L:
        raise ValueError('Length scale must be the original complete target FP32 bbox diagonal')
    vertices = np.asarray(arrays[patch['source_vertices_key']])
    faces = np.asarray(arrays[patch['source_face_vertex_ids_key']])
    if (vertices.dtype != np.float32 or vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not np.isfinite(vertices).all() or faces.shape != (112, 3)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any((faces < 0) | (faces >= len(vertices)))):
        raise ValueError('Invalid trusted source-index arrays')
    if vertices[faces].tobytes() != target.tobytes():
        raise ValueError('Trusted source indexing does not exactly reconstruct target coordinates')
    topology = api['topology']
    C_vertices, C_faces = topology.exact_index(C)
    full_source = topology.audit_mesh(vertices, faces)
    C_source = topology.audit_mesh(vertices, faces[known])
    C_exact = topology.audit_mesh(C_vertices, C_faces)
    ledger['indexed_mesh_audits'] = ledger.get('indexed_mesh_audits', 0) + 3
    ledger['exact_coordinate_indexings'] = ledger.get('exact_coordinate_indexings', 0) + 1
    full_links = {row['vertex_id']: row for row in full_source['vertex_links']['all']}
    C_links = {row['vertex_id']: row for row in C_source['vertex_links']['all']}
    anomalies = [dict(source_vertex_id=vertex, coordinate=vertices[vertex].tolist(),
        C_link=C_links[vertex], full_GT_link=full_links[vertex],
        full_GT_repairs_subpatch_link=full_links[vertex]['classification'] != 'NONMANIFOLD_LINK')
        for vertex in C_source['vertex_links']['nonmanifold_vertex_ids']]
    source_bad = {topology._key(vertices[i]) for i in C_source['vertex_links']['nonmanifold_vertex_ids']}
    exact_bad = {topology._key(C_vertices[i]) for i in C_exact['vertex_links']['nonmanifold_vertex_ids']}
    context = api['audit_context'](C, L, ledger)
    if context['pair_scan']['tested_pairs'] != 66:
        raise RuntimeError('C/C complete pair denominator changed')
    identity = dict(parent=int(patch['parent']), patch_id=str(patch['patch_id']),
        C_exact_archive_winding=C_exact, C_source_archive_winding=C_source,
        full_archive_source_indices=full_source, C_link_anomalies_vs_full_GT=anomalies,
        C_link_anomalies_added_by_exact_indexing_count=len(exact_bad - source_bad),
        C_link_anomalies_removed_by_exact_indexing_count=len(source_bad - exact_bad),
        coincident_distinct_source_vertex_ids=topology._coincidences(vertices),
        source_identity_scope='Trusted manifest/array hashes plus exact indexed coordinate reconstruction; no new raw-dataset provenance verification',
        source_face_ids=known.tolist(), exact_index_policy=topology.EXACT_POLICY,
        array_hashes={field:array_sha256(arrays[patch[field]]) for field in (
            'constraints_key','target_key','known_face_ids_key','source_vertices_key','source_face_vertex_ids_key')},
        full_GT_role='Offline source-link attribution and length scale only; never passed to model',
        C_modifications=0, seconds=time.perf_counter()-started)
    return dict(C=C, L=L, input_audit=identity, context=context)


def _topology_scalars(topology):
    attr = topology['attribution']; topo = topology['topology']
    return dict(exact_C_free_shared_edge_count=attr['exact_C_free_shared_edge_count'],
        exact_C_free_shared_vertex_count=attr['exact_C_free_shared_vertex_count'],
        C_only_nonmanifold_vertex_count=attr['C_only_nonmanifold_vertex_count'],
        inherited_C_only_unmodified_link_count=len(attr['inherited_C_only_unmodified_link_vertex_ids']),
        C_inherited_link_anomalies_with_new_free_incidence_count=len(attr['C_inherited_link_anomalies_with_new_free_incidence']),
        new_free_related_nonmanifold_vertex_count=len(attr['new_free_related_nonmanifold_vertex_ids']),
        C_current_winding_conflict_count=len(attr['orientation_conflicts_C_only']),
        free_related_current_winding_conflict_count=len(attr['orientation_conflicts_free_related']),
        exact_index_boundary_edge_count=topo['edge_incidence']['one'],
        exact_index_overincident_edge_count=topo['edge_incidence']['more_than_two'],
        shared_edge_component_count=topo['shared_edge_components']['count'])


def _audit_output(row, case, input_dir, api, ledger):
    started = time.perf_counter(); path = resolve_path(row['raw_path'], input_dir)
    digest = file_sha256(path)
    if digest != row['raw_sha256']:
        raise ValueError('Raw endpoint file SHA256 differs: ' + str(row['output_id']))
    with np.load(path, allow_pickle=False) as saved:
        full = _float_triangles(saved['output'], 112, 'output')
        trajectory = np.asarray(saved['path'])
    if trajectory.dtype != np.float32 or trajectory.shape not in ((51,112,9),(51,112,3,3)):
        raise ValueError('Require all51 original FP32 Euler states')
    trajectory = trajectory.reshape(51,112,3,3)
    if not np.isfinite(trajectory).all() or trajectory[-1].tobytes() != full.tobytes():
        raise ValueError('Trajectory invalid or its endpoint differs from output')
    C = case['C']; L = case['L']
    if full[:12].tobytes() != C.tobytes() or any(v[:12].tobytes() != C.tobytes() for v in trajectory):
        raise ValueError('Registered C was not preserved in every stored state')
    before = full.tobytes(); free = full[12:]
    cf = api['pair_scan'](C, free, L, ledger=ledger, independent=True)
    ff = api['pair_scan'](free, free, L, ledger=ledger, self_pairs=True, independent=True)
    if cf['tested_pairs'] != 1200 or ff['tested_pairs'] != 4950:
        raise RuntimeError('Original complete CF/FF pair denominators changed')
    views = {scope:{which:api['detector_view'](scan,which) for which in ('primary','independent','verified')}
             for scope,scan in (('CF',cf),('FF',ff))}
    views['CC'] = case['context']['detector_views']
    topo = api['topology'].generated_attribution(full, C, case['input_audit'])
    for key,value in topo['ledger'].items():ledger[key] = ledger.get(key,0) + value
    ledger['static_context_audit_reuses'] = ledger.get('static_context_audit_reuses',0) + 1
    ledger['endpoint_audits'] = ledger.get('endpoint_audits',0) + 1
    scalars = dict(CF_proper_affected_free_fraction=cf['proper_affected_face_fraction'],
        CF_proper_affected_free_count=cf['proper_affected_face_count'],
        CF_proper_pair_count=cf['proper_pair_count'], CF_coplanar_pair_count=cf['coplanar_positive_area_pairs'],
        CF_unresolved_or_degenerate_pair_count=cf['unresolved_or_degenerate_pairs'],
        FF_proper_pair_count=ff['proper_pair_count'], FF_coplanar_pair_count=ff['coplanar_positive_area_pairs'],
        FF_illegal_pair_count=ff['proper_pair_count']+ff['coplanar_positive_area_pairs'],
        FF_unresolved_or_degenerate_pair_count=ff['unresolved_or_degenerate_pairs'], **_topology_scalars(topo))
    any_illegal = any(view[which]['illegal_pair_count'] for view in views.values() for which in ('primary','independent'))
    unresolved = sum(view['verified']['unresolved_count'] for view in views.values())
    if full.tobytes() != before:
        raise RuntimeError('Read-only audit changed coordinates')
    return dict(**{key:row[key] for key in ('output_id','model','parent','patch_id','seed')},
        patch_group=row.get('patch_group'), raw_path=str(path), raw_sha256=digest,
        output_array_sha256=array_sha256(full), status='PASS_EXECUTION', scalars=scalars,
        scans=dict(C_free=cf,free_free=ff), detector_views=views, exact_coordinate_topology=topo,
        C_all51_bitwise_preserved=True,total_faces=112,free_faces=100,
        embedded_geometry_status='FAIL_DETECTED_ILLEGAL_INTERSECTIONS' if any_illegal else 'UNRESOLVED' if unresolved else 'NO_ILLEGAL_TRIANGLE_INTERSECTIONS_DETECTED',
        exact_predicate_certification='NOT_RUN',continuous_collision_safety='NOT_RUN',geometry_modified=False,
        seconds=time.perf_counter()-started,model_forwards=0,OT_calls=0,autograd_calls=0,optimizer_updates=0)


def run(manifest_path, out, input_dir=None):
    manifest_path = Path(manifest_path).resolve(); out = Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != SCHEMA:raise ValueError('Unexpected audit manifest schema')
    base = Path(input_dir).resolve() if input_dir else manifest_path.parent
    array_path = resolve_path(manifest['arrays_path'], base)
    if file_sha256(array_path) != manifest['arrays_sha256']:raise ValueError('Patch/source array archive SHA256 mismatch')
    patches = manifest['patches']; outputs = manifest['outputs']
    if not outputs or len(outputs)>64:raise ValueError('Require 1..64 registered formal outputs, never extra replays')
    patch_ids = [patch_key(p['parent'],p['patch_id']) for p in patches]
    if len(set(patch_ids)) != len(patch_ids):raise ValueError('Duplicate patch identity')
    output_ids = [str(row['output_id']) for row in outputs]
    if len(set(output_ids)) != len(output_ids):raise ValueError('Duplicate formal output identity')
    patch_map = {patch_key(p['parent'],p['patch_id']):p for p in patches}
    matrix_keys = [(row['model'],int(row['parent']),str(row['patch_id']),int(row['seed'])) for row in outputs]
    if len(set(matrix_keys)) != len(matrix_keys):raise ValueError('Duplicate model/parent/patch/seed output')
    for row in outputs:
        patch = patch_map.get(patch_key(row['parent'],row['patch_id']))
        if patch is None or int(row['parent'])!=int(patch['parent']):raise ValueError('Output/patch parent mismatch')
        if row['model'] not in MODELS:raise ValueError('Unregistered model identity')
    out.mkdir(parents=True)
    ledger = dict(model_forwards=0,backward_calls=0,optimizer_updates=0,OT_calls=0,autograd_calls=0,geometry_repairs=0)
    result = dict(schema=SCHEMA,status='RUNNING',manifest_path=str(manifest_path),manifest_sha256=file_sha256(manifest_path),
        arrays_path=str(array_path),arrays_sha256=manifest['arrays_sha256'],patch_audits={},rows=[],ledger=ledger,
        expected_outputs=len(outputs),source_identity='Trusted pre-registered source mappings, verified exact array decomposition',
        scope='Independent offline unchanged static CF/FF/CC and exact-index topology only; no surface scoring or guidance')
    started = time.perf_counter(); destination=out/'audit.json'
    atomic_json(destination,result)
    try:
        api = _auditors(); result['detector_policy']=dict(api['policy'])
        old = ROOT/'experiments/mf_h1_local0'
        result['auditor_sources']={name:file_sha256(old/name) for name in (
            'guided_native_geometry_audit.py','guided_native_geometry_geometry.py','native_patch_seams.py',
            'gpu_guidance_evaluation.py','t1_native_confirmation_topology.py')}
        result['wrapper_sha256']=file_sha256(__file__)
        with np.load(array_path,allow_pickle=False) as archive:
            required={patch[field] for patch in patches for field in ('constraints_key','target_key','known_face_ids_key','source_vertices_key','source_face_vertex_ids_key')}
            arrays={key:archive[key].copy() for key in required}
        cases={}
        for patch in patches:
            case=_source_patch_audit(patch,arrays,api,ledger); key=patch_key(patch['parent'],patch['patch_id']); cases[key]=case
            result['patch_audits'][key]=dict(input_topology=case['input_audit'],context=case['context'])
        for row in outputs:
            result['current_output_id']=row['output_id']
            result['rows'].append(_audit_output(row,cases[patch_key(row['parent'],row['patch_id'])],base,api,ledger))
            atomic_json(destination,result)
            print(f"audited {len(result['rows'])}/{len(outputs)}: {row['output_id']}",flush=True)
        if api['torch'].cuda.is_initialized():raise RuntimeError('Offline audit unexpectedly initialized CUDA')
        result.update(status='PASS_EXECUTION',current_output_id=None,completed_outputs=len(result['rows']),CUDA_initialized=False)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),completed_outputs=len(result['rows']))
        raise
    finally:
        result['total_seconds']=time.perf_counter()-started
        result['endpoint_seconds']=sum(row['seconds'] for row in result['rows'])
        result['patch_input_seconds']=sum(row['input_topology']['seconds'] for row in result['patch_audits'].values())
        atomic_json(destination,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--input-dir',type=Path,help='Optional base for relative manifest paths; default manifest parent')
    parser.add_argument('--out',type=Path,required=True,help='New audit-only output directory; never overwrite')
    args=parser.parse_args(argv)
    run(args.manifest,args.out,args.input_dir)


if __name__=='__main__':main()
