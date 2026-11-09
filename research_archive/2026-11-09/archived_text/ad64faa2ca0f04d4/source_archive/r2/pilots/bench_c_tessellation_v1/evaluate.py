"""CPU-only, read-only scorer and fixed-view adapter for C tessellation.

The historic 177-field scorer is not modified.  Derived query banks retain all
physical queries and only remap their child-face ownership/barycentric metadata.
COMMON_PHYSICAL_C is an additional CF/I diagnostic, never a topology certificate
for the subdivided output.  This module imports no torch and launches no models.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C

VERSION = 'bench_C_tessellation_raw_quality_fixed_physical_queries_v1'


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode('ascii') + str(value.shape).encode('ascii') + value.tobytes()).hexdigest()


def save_npz(path, values):
    path = C.resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = C.arrays(path)
        if set(previous) != set(values) or any(array_sha(previous[k]) != array_sha(values[k]) for k in values):
            raise ValueError('Frozen array differs: ' + str(path))
        return
    with path.open('xb') as handle:
        np.savez_compressed(handle, **values)


def _barycentric(points, triangle):
    """FP64 projection and reconstruction residual; no nearest-surface sampling."""
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
    reconstructed = bary @ tri
    return bary, np.linalg.norm(reconstructed-points, axis=-1)


def prepare_query(old_q, new_C_vertices, new_C_faces_original, dest, new_C_faces_model=None):
    """Derive a fixed-query child-face map; no geometry or random draws changed.

    Return {descriptor, audit}.  new_C_faces_model must be supplied whenever
    generator canonical corner order differs from restored/source corner order.
    Each new triangle is matched to a containing original C face using all
    three vertices.  Every old point remains in a child of its own old face.
    """
    dest = C.resolve(dest)
    inp = C.arrays(old_q['inference_npz'])
    old_tri = inp['C_vertices'][inp['C_faces_model']].astype(np.float64)
    vertices = np.asarray(new_C_vertices, np.float32)
    faces = np.asarray(new_C_faces_original if new_C_faces_model is None else new_C_faces_model, np.int64)
    new_tri = vertices[faces]
    descriptor = copy.deepcopy(old_q['query_descriptor'])
    if C.sha(descriptor['path']) != descriptor['sha256']:
        raise ValueError('Original physical query bank hash differs')
    old_arrays = C.arrays(descriptor['path'])
    if {k:array_sha(a) for k,a in old_arrays.items()} != descriptor['array_sha256']:
        raise ValueError('Original physical query array hashes differ')
    if len(old_arrays['C_points']) != 2048:
        raise ValueError('Frozen physical C query count differs')
    # FP32 midpoint rounding is bounded explicitly; no approximate repair.
    plane_tol, bary_tol = 2e-6*C.L, 2e-5
    parents = []
    child_residuals = []
    for child in new_tri:
        choices = []
        for parent_id, parent in enumerate(old_tri):
            bary, residual = _barycentric(child, parent)
            if residual.max() <= plane_tol and bary.min() >= -bary_tol and bary.max() <= 1+bary_tol:
                choices.append((float(residual.max()), -float(bary.min()), parent_id))
        if not choices:
            raise ValueError('A new C face is not contained in any original face')
        best = min(choices)
        parents.append(best[2]); child_residuals.append(best[0])
    parents = np.asarray(parents, np.int64)
    points = np.asarray(old_arrays['C_points'], np.float64)
    old_owner = np.asarray(old_arrays['C_point_faces'], np.int64)
    new_owner = np.full_like(old_owner, -1)
    new_bary = np.empty_like(old_arrays['C_barycentric'])
    residuals = np.full(len(points), np.inf)
    tie_counts = 0
    for old_id in range(len(old_tri)):
        point_ids = np.flatnonzero(old_owner == old_id)
        children = np.flatnonzero(parents == old_id)
        if len(point_ids) and not len(children):
            raise ValueError('A queried original face has no child coverage')
        if not len(point_ids):
            continue
        bary_list, residual_list = zip(*[_barycentric(points[point_ids], new_tri[c]) for c in children])
        bary_values = np.stack(bary_list, axis=1)
        errors = np.stack(residual_list, axis=1)
        minima = bary_values.min(axis=2)
        eligible = (errors <= plane_tol) & (minima >= -bary_tol) & (bary_values.max(axis=2) <= 1+bary_tol)
        if not eligible.any(axis=1).all():
            raise ValueError('A frozen physical query falls outside all children of its original face')
        # Interior-most child, then fixed child order, gives deterministic edge ties.
        selected = np.argmax(np.where(eligible, minima, -np.inf), axis=1)
        tie_counts += int((eligible.sum(axis=1)>1).sum())
        new_owner[point_ids] = children[selected]
        new_bary[point_ids] = bary_values[np.arange(len(point_ids)), selected]
        residuals[point_ids] = errors[np.arange(len(point_ids)), selected]
    if (new_owner < 0).any():
        raise ValueError('Unmapped query owner')
    derived = {k:a.copy() for k,a in old_arrays.items()}
    derived['C_point_faces'] = new_owner.astype(old_arrays['C_point_faces'].dtype)
    derived['C_barycentric'] = new_bary
    changed = ['C_point_faces', 'C_barycentric']
    preserved = {k:array_sha(a) for k,a in old_arrays.items() if k not in changed}
    if any(array_sha(derived[k]) != h for k,h in preserved.items()):
        raise AssertionError('Physical query data mutated')
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'queries.npz'
    save_npz(path, derived)
    descriptor.update(path=str(path), sha256=C.sha(path), array_sha256={k:array_sha(a) for k,a in derived.items()})
    descriptor['identity'] = dict(descriptor['identity'], C_sha256=array_sha(new_tri))
    audit = dict(status='PASS', version=VERSION, source_query_path=old_q['query_descriptor']['path'],
        source_query_sha256=old_q['query_descriptor']['sha256'], query_count=len(points),
        original_K=len(old_tri), derived_K=len(new_tri), new_queries=0,
        new_face_parent_ids=parents.tolist(), query_child_ties=tie_counts,
        max_query_projection_residual_model=float(residuals.max()),
        max_child_parent_plane_residual_model=max(child_residuals, default=0.),
        plane_tolerance_model=plane_tol, barycentric_tolerance=bary_tol,
        changed_array_names=changed, preserved_array_sha256=preserved,
        note='Original physical points, weights, normals, boundary-distance and visibility arrays are unchanged. Child ownership/barycentric metadata only.',
        I_scope='Native-C I remains potentially tessellation-sensitive because original detector clips cross-segments to query-owning triangles; compare COMMON_PHYSICAL_C separately.')
    C.write(dest/'descriptor.json', descriptor)
    C.write(dest/'audit.json', audit)
    return dict(descriptor=descriptor, audit=audit)


def _common_physical_C(original_q, mesh, K, destination):
    from r2.pilots.seam_abc_v1.evaluate import helpers
    old, A, _, _, _ = helpers()
    mod = old.cpu_math()
    inp, metadata = C.arrays(original_q['inference_npz']), C.arrays(original_q['metadata_npz'])
    cc = inp['C_vertices'][inp['C_faces_model']]
    original = inp['C_vertices'][inp['C_faces_original']]
    free = np.asarray(mesh['output'])[int(K):]
    cf = mod['geometry'].pair_scan(cc, free, C.L, independent=True)
    trusted = mod['control'].trusted_cf(cc, free, C.L, cf_scan=cf)
    metrics, vectors = mod['control'].conflict_metrics(free, dict(C=cc, L0=C.L), old.load_queries(original_q['query_descriptor']), trusted)
    illegal = [r for r in cf['contacts'] if r['classification'] in {'proper_transverse_penetration','coplanar_positive_area_overlap'}]
    affected = sorted({int(r['pair'][1]) for r in illegal})
    metrics.update(CF_illegal_pair_count=len(illegal), CF_illegal_affected_free_fraction=len(affected)/len(free),
        CF_proper_pair_count=cf['proper_pair_count'], CF_coplanar_pair_count=cf['coplanar_positive_area_pairs'])
    source_ids = metadata['C_source_vertex_ids'][inp['C_faces_original']]
    vertex_ids, points, boundary_audit = A.topology(original, source_ids)
    distances, nearest = A.nearest(points, free)
    boundary = [dict(setting=n, threshold_model=t, scale=s, tau=value, domain='COMMON_ORIGINAL_C_BOUNDARY_VERTICES',
                     **A.metric(distances,t)) for n,t,s,value in A.SETTINGS]
    result = dict(status='PASS', version=VERSION+'_common_physical_C', original_K=int(original_q['K']),
        actual_input_K=int(K), fixed_free_face_denominator=len(free), fixed_CF_tested_pairs=len(cc)*len(free),
        fixed_C_query_count=2048, metrics=metrics, boundary=boundary, boundary_audit=boundary_audit,
        geometry_changed=False, original_C_query_sha256=original_q['query_descriptor']['sha256'],
        original_C_input_sha256=C.sha(original_q['inference_npz']), model_calls=0, full_CF_scans=1, full_FF_scans=0,
        scope='Common physical reference C for geometric CF/I/boundary distances only; not actual subdivided C topology, connectivity, or generated-C preservation certification.')
    destination = C.resolve(destination)
    C.write(destination/'common_C.json',result)
    C.write(destination/'common_C_pairs.json',cf)
    save_npz(destination/'common_C_vectors.npz',dict(vectors, boundary_vertex_ids=vertex_ids,
        boundary_points=points,boundary_distances=distances,boundary_nearest=nearest))
    return result


def score_one(q, mesh_path, dest, label, original_q=None):
    """One unchanged RAW 177-field score; optional extra original-C-only scan."""
    if 'torch' in sys.modules:
        raise RuntimeError('Scoring must run in an isolated CPU process without torch')
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    dest, mesh_path = C.resolve(dest), C.resolve(mesh_path)
    identity = dict(version=VERSION, label=label, request_id=q['request_id'], mesh_sha256=C.sha(mesh_path),
        inference_sha256=C.sha(q['inference_npz']), metadata_sha256=C.sha(q['metadata_npz']),
        query_sha256=q['query_descriptor']['sha256'], original_q=original_q,
        adapter_sha256=C.sha(__file__), scorer_sha256=C.sha(CPU.M.__file__))
    receipt = dest/'audit_receipt.json'
    if receipt.exists():
        saved=C.read(receipt)
        if saved['identity'] != identity:
            raise ValueError('Completed score identity changed')
        for path,digest in saved['files'].items():
            if C.sha(path) != digest:raise ValueError('Completed score changed')
        return saved
    if dest.exists():
        raise RuntimeError('Partial score requires explicit engineering recovery: '+str(dest))
    start=time.perf_counter()
    mesh=C.arrays(mesh_path)
    payload=CPU.score_current(q,mesh,dest,label,seam_model=False)
    common=None
    if original_q is not None:
        common=_common_physical_C(original_q,mesh,int(q['K']),dest)
    scope=dict(status='PASS', version=VERSION, stage='RAW', source_GT_success=False,
        native_C_discrete_metrics='Original 177 fields retained, including changed C boundary vertex / CF-pair denominators.',
        primary_geometric_comparison='FF/free quality on equal free count; COMMON_PHYSICAL_C CF/I and original boundary vertices when tessellated.',
        no_POST='NOT_RUN: This experiment isolates generator response before D0/GEO.',
        no_new_queries=True, common_C_extra_CF_scans=int(common is not None), extra_FF_scans=0,
        semantics='C adoption, completeness and support remain separate actual AI_PROXY review; no GT-style ranking.')
    C.write(dest/'metric_scope.json',scope)
    result=dict(status='COMPLETE', identity=identity, row=payload['row'], model_calls=0,D0_calls=0,new_queries=0,
        full_quality_scans=1, extra_common_CF_scans=int(common is not None), extra_common_FF_scans=0,
        seconds=time.perf_counter()-start, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(receipt,result)
    return result


def render_page(reference_q, cells, dest):
    """Cells {label,path,q}; different K/N, identical old view/section coordinates."""
    if 'torch' in sys.modules:
        raise RuntimeError('Rendering must run without torch')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    dest=C.resolve(dest); dest.mkdir(parents=True,exist_ok=True)
    png,receipt=dest/'comparison.png',dest/'comparison.json'
    identity=dict(version=VERSION, reference_request_id=reference_q['request_id'],
        bounds=reference_q['gallery_bounds'],depth_bounds=reference_q['gallery_depth_bounds'],
        cells=[dict(label=c['label'],path=c['path'],mesh_sha256=C.sha(c['path']),K=int(c['q']['K']),N=int(c['q']['N']),
                    input_sha256=C.sha(c['q']['inference_npz'])) for c in cells],adapter_sha256=C.sha(__file__))
    if receipt.exists():
        old=C.read(receipt)
        if old['identity']!=identity or C.sha(png)!=old['image_sha256']:
            raise ValueError('Rendered comparison changed')
        return old
    if png.exists():raise RuntimeError('Unsealed rendering image; no silent overwrite')
    start=time.perf_counter();plt=G.H._mpl();nc=len(cells)
    fig=plt.figure(figsize=(5.3*nc,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,nc,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    bounds,depth=reference_q['gallery_bounds'],reference_q['gallery_depth_bounds'];display=[]
    try:
        for col,cell in enumerate(cells):
            q=cell['q'];k=int(q['K']);arr=C.arrays(cell['path']);tri,roles=G.roles(arr,k)
            inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
            if not np.array_equal(tri[:k],original):raise ValueError('Rendered C differs from declared variant')
            geometry=dict(C=tri[:k],F=tri[k:],C_original=original,tri=tri)
            facts=dict(label=cell['label'],K=k,N=int(q['N']),views=[])
            for row,azimuth in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,roles,k,G.D.basis(azimuth,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(cell['label']+f' | K={k} N={q["N"]}\nRAW | az={azimuth}',fontsize=9)
                facts['views'].append(dict(azimuth=azimuth,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited-frame clipping; see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);G.H.draw_section(ax,geometry,np.empty(0,np.int64),bounds,C.L,view,1)
                position=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ax.set_title(cell['label']+' | '+view+f'={position:.5f}',fontsize=8)
            display.append(facts)
        fig.suptitle(reference_q['request_id']+' | Frozen EXPOSURE, same free noise and physical C surface\n'
            'Blue: condition C | Gray: generated free. Same original frame and fixed sections. RAW only.',fontsize=12)
        fig.text(.5,.018,'Actual AI_PROXY ratings recorded separately. No unique-source-style grading; section endpoints alone do not certify gaps.',ha='center',fontsize=9)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as handle:handle.write(buffer.getvalue())
    finally:plt.close(fig)
    result=dict(status='PENDING_ACTUAL_VIEW',identity=identity,image_path=str(png),image_sha256=C.sha(png),
        display=display,seconds=time.perf_counter()-start,model_calls=0,scoring_calls=0,ratings_generated=0,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    C.write(receipt,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('score','render'));p.add_argument('--job',required=True)
    a=p.parse_args();job=C.read(a.job)
    result=score_one(**job) if a.action=='score' else render_page(**job)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','row','identity','display')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
