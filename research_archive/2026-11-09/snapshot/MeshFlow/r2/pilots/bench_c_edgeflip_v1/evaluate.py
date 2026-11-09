"""CPU-only fixed-query adapters for the registered Bench C edge-flip test.

The original scorer and all cached scores remain unchanged.  Retriangulation
may cross old face boundaries, so fixed physical queries are assigned across
all new C triangles rather than restricted to children of their old owner.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C

OUT = C.ROOT / 'experiments/r2_bench_c_edgeflip_v1'
VERSION = 'bench_C_edgeflip_fixed_physical_query_remap_v1'


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode('ascii') + str(value.shape).encode('ascii') + value.tobytes()).hexdigest()


def _barycentric(points, triangle):
    """FP64 coordinates and projection residual, without surface sampling."""
    points, tri = np.asarray(points, np.float64), np.asarray(triangle, np.float64)
    e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
    a, b, d = float(e0 @ e0), float(e0 @ e1), float(e1 @ e1)
    determinant = a*d - b*b
    if determinant <= np.finfo(float).eps * max(a*d, 1e-30):
        raise ValueError('Degenerate condition triangle in query remapping')
    rel = points - tri[0]
    r0, r1 = rel @ e0, rel @ e1
    u, v = (d*r0-b*r1)/determinant, (a*r1-b*r0)/determinant
    bary = np.stack((1-u-v, u, v), axis=-1)
    return bary, np.linalg.norm(bary @ tri-points, axis=-1)


def _save_npz(path, values):
    path = C.resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = C.arrays(path)
        if set(previous) != set(values) or any(array_sha(previous[k]) != array_sha(values[k]) for k in values):
            raise ValueError('Frozen array differs: ' + str(path))
        return
    with path.open('xb') as handle:
        np.savez_compressed(handle, **values)


def prepare_query_any(old_q, new_vertices, new_faces_original, dest, new_C_faces_model=None):
    """Preserve every physical query; remap only owner and model-order barycentrics.

    All eligible new triangles are considered, including triangles crossing
    original face boundaries.  Multiple containment matches choose the lowest
    new face ID.  No centroid, random or nearest-surface query is generated.
    The geometry preparation stage separately certifies equal C surfaces.
    """
    if 'torch' in sys.modules:
        raise RuntimeError('Query preparation must not import torch')
    inp = C.arrays(old_q['inference_npz'])
    old_tri = inp['C_vertices'][inp['C_faces_model']].astype(np.float64)
    vertices = np.asarray(new_vertices, np.float32)
    original_faces = np.asarray(new_faces_original, np.int64)
    faces = np.asarray(new_faces_original if new_C_faces_model is None else new_C_faces_model, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Malformed new C vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape != original_faces.shape or not len(faces):
        raise ValueError('Malformed new C face tables')
    if not np.array_equal(np.sort(faces, axis=1), np.sort(original_faces, axis=1)):
        raise ValueError('Model corner order must be a per-face permutation of original corners')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('C face index is outside the vertex array')
    if len(faces) != len(old_tri):
        raise ValueError('The registered edge-flip experiment preserves K')
    new_tri = vertices[faces]
    descriptor = copy.deepcopy(old_q['query_descriptor'])
    if C.sha(descriptor['path']) != descriptor['sha256']:
        raise ValueError('Original query bank hash differs')
    old_arrays = C.arrays(descriptor['path'])
    if {k:array_sha(a) for k,a in old_arrays.items()} != descriptor['array_sha256']:
        raise ValueError('Original query array hashes differ')
    points = np.asarray(old_arrays['C_points'], np.float64)
    if len(points) != 2048 or points.shape != (2048, 3) or not np.isfinite(points).all():
        raise ValueError('Frozen physical C query bank differs')
    plane_tol, bary_tol = 2e-6*C.L, 2e-5
    owners = np.full(len(points), -1, np.int64)
    bary_out = np.empty_like(old_arrays['C_barycentric'])
    residual_out = np.full(len(points), np.inf)
    match_counts = np.zeros(len(points), np.int64)
    # Ascending face order implements the registered lowest-ID edge tie rule.
    for face_id, triangle in enumerate(new_tri):
        bary, residual = _barycentric(points, triangle)
        eligible = (residual <= plane_tol) & (bary.min(axis=1) >= -bary_tol) & (bary.max(axis=1) <= 1+bary_tol)
        match_counts += eligible
        choose = eligible & (owners < 0)
        owners[choose] = face_id
        bary_out[choose] = bary[choose]
        residual_out[choose] = residual[choose]
    if np.any(owners < 0):
        raise ValueError('Frozen physical queries outside every new triangle: '+repr(np.flatnonzero(owners < 0).tolist()))
    reconstructed = (new_tri[owners].astype(np.float64)*np.asarray(bary_out, np.float64)[...,None]).sum(axis=1)
    saved_residual = np.linalg.norm(reconstructed-points, axis=1)
    if saved_residual.max() > plane_tol or not np.allclose(np.asarray(bary_out, np.float64).sum(axis=1), 1, rtol=0, atol=bary_tol):
        raise ValueError('Stored model-corner barycentrics fail reconstruction')
    changed = ['C_point_faces', 'C_barycentric']
    derived = {k:a.copy() for k,a in old_arrays.items()}
    derived['C_point_faces'] = owners.astype(old_arrays['C_point_faces'].dtype)
    derived['C_barycentric'] = bary_out
    preserved = {k:array_sha(a) for k,a in old_arrays.items() if k not in changed}
    if any(array_sha(derived[k]) != h for k,h in preserved.items()):
        raise AssertionError('Physical query coordinates/weights/normals/borders changed')
    dest = C.resolve(dest)
    path = dest/'queries.npz'
    _save_npz(path, derived)
    descriptor.update(path=str(path), sha256=C.sha(path), array_sha256={k:array_sha(a) for k,a in derived.items()})
    descriptor['identity'] = dict(descriptor['identity'], C_sha256=array_sha(new_tri))
    audit = dict(status='PASS', version=VERSION, source_query_path=old_q['query_descriptor']['path'],
        source_query_sha256=old_q['query_descriptor']['sha256'], query_count=2048,
        original_K=len(old_tri), derived_K=len(new_tri), new_queries=0, new_random_draws=0,
        match_domain='ALL_NEW_C_TRIANGLES', boundary_tie_policy='LOWEST_NEW_FACE_ID',
        queries_with_multiple_matches=int((match_counts>1).sum()), unmatched_queries=0,
        query_owner_id_changed_count=int((owners != old_arrays['C_point_faces']).sum()),
        query_match_count_min=int(match_counts.min()), query_match_count_max=int(match_counts.max()),
        max_query_projection_residual_model=float(residual_out.max()),
        max_stored_barycentric_reconstruction_residual_model=float(saved_residual.max()),
        plane_tolerance_model=plane_tol, barycentric_tolerance=bary_tol,
        changed_array_names=changed, preserved_array_sha256=preserved,
        new_C_model_sha256=array_sha(new_tri), model_corner_permutation_verified=True,
        physical_query_preservation='Every array except C_point_faces/C_barycentric is byte-identical, including coordinates, weights, normals, borders and visibility.',
        geometry_scope='Containment of frozen queries is checked here; same-surface certification belongs to the independent C geometry preparation audit.',
        I_scope='Native I may depend on triangle clipping even at equal K; every new output requires an additional original-C common diagnostic.')
    C.write(dest/'descriptor.json', descriptor)
    C.write(dest/'audit.json', audit)
    return dict(descriptor=descriptor, audit=audit)


def require_all_raw():
    """No CPU quality phase until every registered feasible new RAW is sealed."""
    cells = C.read(OUT/'inputs/cells.json')
    registration = C.read(OUT/'registration.json')
    new = [c for c in cells if c['label'] == 'EDGE_FLIP']
    if len([c for c in cells if c['label']=='ORIGINAL']) != 6 or not 0 < len(new) <= 6:
        raise ValueError('Expected six original plus the registered feasible EDGE_FLIP cells')
    if len(cells) != registration.get('expected_cells',len(cells)) or len(new) != registration.get('expected_new_raw',len(new)):
        raise ValueError('Registered feasible cell budget differs')
    for cell in new:
        raw = C.resolve(cell['raw_path'])
        receipt = C.read(raw.parent/'complete.json')
        if receipt.get('status') not in ('COMPLETE', 'PASS'):
            raise ValueError('RAW not complete: '+cell['cell_id'])
        for path,digest in receipt.get('files',{}).items():
            if C.sha(path) != digest:
                raise ValueError('Sealed RAW asset differs: '+str(path))
        if not raw.is_file():
            raise FileNotFoundError(raw)
    return cells


def score_one(q, mesh_path, dest, label, original_q=None):
    require_all_raw()
    if label != 'EDGE_FLIP' or original_q is None:
        raise ValueError('Only registered new EDGE_FLIP scores are authorized; original_q is mandatory at equal K')
    if (int(q['K']),int(q['N'])) != (int(original_q['K']),int(original_q['N'])):
        raise ValueError('Edge-flip score must preserve K/N')
    from r2.pilots.bench_c_tessellation_v1.evaluate import score_one as original_score
    return original_score(q, mesh_path, dest, label, original_q=original_q)


def render_page(reference_q, cells, dest):
    require_all_raw()
    from r2.pilots.bench_c_tessellation_v1.evaluate import render_page as original_render
    return original_render(reference_q, cells, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('score','render'))
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = C.read(args.job)
    result = score_one(**job) if args.action == 'score' else render_page(**job)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','row','identity','display')},ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
