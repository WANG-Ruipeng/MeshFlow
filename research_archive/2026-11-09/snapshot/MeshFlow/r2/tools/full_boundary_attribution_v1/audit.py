"""Independent CPU verification of fixed cached boundary attribution.

No torch/scorer/generation imports. Distance verification uses an independent
plane projection with dominant-axis 2D barycentrics plus segment fallback, not production geometry helpers.
"""
from collections import Counter
import argparse, csv, hashlib, json, math, sys, time
import numpy as np
from . import common as C


def csv_rows(path):
    with C.resolve(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def independent_segments(points, segments):
    """All finite segment distances, including exact zero-length segments."""
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    u = segments[:, 1] - segments[:, 0]
    den = (u * u).sum(axis=1)
    rel = points[:, None] - segments[None, :, 0]
    t = np.zeros((len(points), len(segments)), dtype=np.float64)
    np.divide(np.einsum('qsd,sd->qs', rel, u), den[None], out=t, where=den[None] > 0)
    t = np.maximum(0., np.minimum(1., t))
    closest = segments[None, :, 0] + t[..., None] * u[None]
    distance = np.linalg.norm(points[:, None] - closest, axis=2)
    return distance, closest, t


def independent_triangles(points, triangles):
    """Exact finite triangles by plane projection and three-edge fallback.

    This independently returns every query/face distance and closest point.
    Rank-deficient triangles are treated as their finite edge union.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    nq, nf = len(points), len(triangles)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    den = (normal * normal).sum(axis=1)
    ap = points[:, None] - a[None]
    signed_scale = np.zeros((nq, nf), dtype=np.float64)
    np.divide(np.einsum('qfd,fd->qf', ap, normal), den[None], out=signed_scale, where=den[None]>0)
    projected = points[:, None] - signed_scale[..., None] * normal[None]
    pa = projected - a[None]
    # Independent 2D projected-area solve on each face's dominant normal axis.
    # The production kernel uses 3D cross-product barycentric numerators.
    bary_v = np.zeros((nq, nf), dtype=np.float64)
    bary_w = np.zeros((nq, nf), dtype=np.float64)
    dominant = np.argmax(np.abs(normal), axis=1)
    for axis in range(3):
        selected = np.flatnonzero(dominant == axis)
        i, j = [k for k in range(3) if k != axis]
        determinant = ab[selected,i]*ac[selected,j]-ab[selected,j]*ac[selected,i]
        dv = pa[:,selected,i]*ac[selected,j]-pa[:,selected,j]*ac[selected,i]
        dw = ab[selected,i]*pa[:,selected,j]-ab[selected,j]*pa[:,selected,i]
        vv = np.zeros_like(dv); ww = np.zeros_like(dw)
        np.divide(dv, determinant[None], out=vv, where=determinant[None]!=0)
        np.divide(dw, determinant[None], out=ww, where=determinant[None]!=0)
        bary_v[:,selected] = vv; bary_w[:,selected] = ww
    inside = (den[None]>0)&(bary_v>=0)&(bary_w>=0)&((bary_v+bary_w)<=1.)
    seg = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2, 3)
    ed, ep, _ = independent_segments(points, seg)
    ed = ed.reshape(nq, nf, 3)
    ep = ep.reshape(nq, nf, 3, 3)
    idx = np.argmin(ed, axis=2)
    closest = np.take_along_axis(ep, idx[:, :, None, None], axis=2)[:, :, 0]
    closest = np.where(inside[..., None], projected, closest)
    return np.linalg.norm(points[:, None]-closest, axis=2), closest


def independent_source_boundary(source):
    coords = source['C_original'].astype(np.float64)
    ids = source['C_original_source_vertex_ids']
    lookup, edge_incidence = {}, {}
    for fi, (face_ids, face) in enumerate(zip(ids, coords)):
        for sid, point in zip(face_ids, face):
            sid = int(sid)
            if sid in lookup and not np.array_equal(lookup[sid], point):
                raise AssertionError('One source vertex ID has unequal coordinates')
            lookup[sid] = point
        for corner in range(3):
            u, v = int(face_ids[corner]), int(face_ids[(corner+1)%3])
            edge_incidence.setdefault(tuple(sorted((u,v))), []).append((fi,u,v))
    edges = sorted(key for key, owners in edge_incidence.items() if len(owners)==1)
    return lookup, edge_incidence, edges


def check_source_mapping(source, input_C):
    original = source['C_original']
    perm = source['C_source_corner_permutations']
    forward = np.take_along_axis(original, perm[..., None], axis=1)
    assert C.same(forward, input_C), 'Source forward corner mapping mismatch'
    inverse = np.argsort(perm, axis=1)
    recovered = np.take_along_axis(input_C, inverse[..., None], axis=1)
    assert C.same(recovered, original), 'Source inverse corner mapping mismatch'
    sid = source['C_original_source_vertex_ids']
    assert np.array_equal(np.take_along_axis(sid, perm, axis=1),source['C_source_vertex_ids'])


def check_numeric_contracts():
    triangle=np.asarray([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]]])
    points=np.asarray([[.2,.3,1.],[2.,0.,0.],[.5,0.,0.]])
    d,p=independent_triangles(points,triangle)
    np.testing.assert_allclose(d[:,0],[1.,1.,0.],atol=1e-13,rtol=0)
    np.testing.assert_allclose(p[:,0],[[.2,.3,0.],[1.,0.,0.],[.5,0.,0.]],atol=1e-13,rtol=0)
    np.testing.assert_allclose(d[:,1],[math.sqrt(1.09),1.,0.],atol=1e-13,rtol=0)
    return 3

def verify_inputs(reg=None):
    tick=time.perf_counter()
    reg=C.read(C.OUT/'registration.json') if reg is None else reg
    cells=reg['cells'];assert len(cells)==44
    assert Counter(x['stage'] for x in cells)=={'RAW':22,'POST':22}
    assert Counter(x['label'] for x in cells)=={'SMALL_C200':22,'CREF_C200':22}
    seen=set();geometry_checks=0;source_checks=0
    for cell in cells:
        key=(cell['request_id'],cell['label'],cell['stage'])
        assert key not in seen;seen.add(key)
        for stem in ('geometry','inherited','condition_source'):
            if stem+'_sha256' in cell:
                assert C.sha(cell[stem+'_path'])==cell[stem+'_sha256'],(cell['cell_id'],stem)
        bundle=C.arrays(cell['geometry_path']);source=C.arrays(cell['condition_source_path']);inp=C.arrays(cell['input_path'])
        check_source_mapping(source,inp['C']);source_checks+=1
        assert C.same(bundle['C'],inp['C'])
        assert C.same(bundle['C_original'],source['C_original'])
        assert np.array_equal(bundle['C_source_vertex_ids'],source['C_original_source_vertex_ids'])
        k=len(bundle['C']);v=bundle['mesh_vertices'];f=bundle['mesh_faces'];tri=v[f]
        assert C.same(tri[k:],bundle['F'])
        assert C.same(tri[:k],bundle['C_original'])
        assert np.array_equal(bundle['known_mask'],np.arange(len(f))<k)
        prior=C.arrays(cell['source_geometry_path'])
        assert C.same(prior['vertices'],v) and C.same(prior['faces'],f)
        assert C.same(prior['output'],tri)
        slots=np.arange(k,len(f),dtype=np.int64) if cell['stage']=='RAW' else prior['source_face_ids'][k:]
        assert np.array_equal(bundle['original_free_slot_ids'],slots)
        score=C.read(cell['score_path']);inherited=C.read(cell['inherited_path'])
        assert inherited['score']==score
        assert score['request_id']==cell['request_id'] and score['label']==cell['label'] and score['stage']==cell['stage']
        assert score['K']==k and score['free_face_denominator']==len(bundle['F'])
        assert score['identity']['C_sha256']==C.array_sha(inp['C'])
        pairs=inherited['pairs'];nf=len(bundle['F'])
        assert pairs['CF']['tested_pairs']==k*nf
        assert pairs['FF']['tested_pairs']==nf*(nf-1)//2
        for row in pairs['CF']['contacts']:assert 0<=row['pair'][0]<k and 0<=row['pair'][1]<nf
        for row in pairs['FF']['contacts']:assert 0<=row['pair'][0]<row['pair'][1]<nf
        geometry_checks+=1
    for path,expected in reg['protected_hashes'].items():assert C.sha(path)==expected,path
    receipt=dict(status='PASS',version='independent_boundary_audit_v1',cells=len(cells),
        source_inverse_mapping_checks=source_checks,exact_saved_geometry_checks=geometry_checks,
        protected_files=len(reg['protected_hashes']),registration_sha256=C.sha(C.OUT/'registration.json'),
        source_pair_indexing='CF=[C-local,F-local]; FF=[F-local,F-local]; geometry global rows=K+F-local',
        seconds=time.perf_counter()-tick,model_imports=0,model_calls=0,quality_scans=0)
    dest=C.OUT/'independent_audit/inputs.json'
    if dest.exists():
        previous=C.read(dest);assert previous['registration_sha256']==receipt['registration_sha256'];return previous
    C.write(dest,receipt);return receipt


def _close(actual,expected,name,tol=1e-9):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=expected.shape:raise AssertionError((name,'shape',actual.shape,expected.shape))
    np.testing.assert_allclose(actual,expected,atol=tol,rtol=1e-8,err_msg=name)
    finite=np.isfinite(actual)&np.isfinite(expected)
    return float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.


def _points(rows,prefix):
    return np.asarray([[float(row[prefix+'_'+axis]) for axis in 'xyz'] for row in rows],np.float64)


def verify_cell(cell):
    tick=time.perf_counter();dest=C.OUT/'results'/cell['cell_id']
    a=C.arrays(dest/'vectors.npz');summary=C.read(dest/'summary.json')
    bundle=C.arrays(cell['geometry_path']);src=C.arrays(cell['condition_source_path']);old=C.read(cell['inherited_path'])
    b_edges=csv_rows(dest/'boundary_edges.csv');f_edges=csv_rows(dest/'free_edges.csv');f_vertices=csv_rows(dest/'free_vertices.csv')
    samples=csv_rows(dest/'samples.csv')
    points=a['points'].astype(np.float64);F=bundle['F'].astype(np.float64);K=len(bundle['C'])
    coords,inc,boundary=independent_source_boundary(src)
    assert len(b_edges)==len(boundary)
    max_error=0.;checks=0;count_midpoints=0
    for row,key in zip(b_edges,boundary):
        u,v=int(row['source_vertex0']),int(row['source_vertex1']);assert tuple(sorted((u,v)))==key
        owner=inc[key][0];assert (int(row['C_face_row']),u,v)==owner
        length=np.linalg.norm(coords[v]-coords[u]);n=max(1,int(np.ceil(length/(summary['config']['boundary_step_over_L']*C.L))))
        assert int(row['sample_count'])==n
        selected=np.flatnonzero(a['boundary_edge_id']==int(row['boundary_edge_id']))
        assert len(selected)==n and np.all(a['sample_kind'][selected]==0)
        t=(np.arange(n)+.5)/n;expected=(1-t[:,None])*coords[u]+t[:,None]*coords[v]
        max_error=max(max_error,_close(points[selected],expected,'boundary query positions'))
        max_error=max(max_error,_close(a['length_weights'][selected],np.full(n,length/n),'boundary weights'))
        count_midpoints+=n;checks+=n*4
    vertex_idx=np.flatnonzero(a['sample_kind']==1)
    assert all(float(x)==0. for x in a['length_weights'][vertex_idx])
    boundary_vids=sorted({v for edge in boundary for v in edge})
    assert a['source_vertex_id'][vertex_idx].tolist()==boundary_vids
    _close(points[vertex_idx],np.asarray([coords[v] for v in boundary_vids]),'source boundary vertices')
    perimeter=sum(np.linalg.norm(coords[v]-coords[u]) for u,v in boundary)
    max_error=max(max_error,_close([a['length_weights'].sum()],[perimeter],'perimeter'))
    assert len(points)==len(samples)==count_midpoints+len(vertex_idx)
    # Independent edge incidence and F-open status from saved indexed geometry.
    verts=bundle['mesh_vertices'].astype(np.float64);faces=bundle['mesh_faces'];edge_occ={}
    for fi,ids in enumerate(faces):
        for c0,c1 in ((0,1),(1,2),(2,0)):
            u,v=map(int,[ids[c0],ids[c1]])
            if u!=v:edge_occ.setdefault(tuple(sorted((u,v))),[]).append(fi)
    fkeys=sorted(key for key,owners in edge_occ.items() if any(i>=K for i in owners))
    assert len(f_edges)==len(fkeys)
    for row,key in zip(f_edges,fkeys):
        assert (int(row['vertex0']),int(row['vertex1']))==key
        nf=sum(i>=K for i in edge_occ[key]);assert int(row['F_incidence'])==nf
        assert (row['F_open']=='True')==(nf==1)
    segments=np.stack((_points(f_edges,'start'),_points(f_edges,'end')),axis=1)
    open_ids=np.asarray([i for i,key in enumerate(fkeys) if sum(j>=K for j in edge_occ[key])==1],np.int64)
    assert np.array_equal(a['F_open_edge_ids'],open_ids)
    fv_ids=np.asarray(sorted(set(faces[K:].reshape(-1).tolist())),np.int64)
    assert np.array_equal(a['F_vertex_actual_ids'],fv_ids)
    _close(_points(f_vertices,'point'),verts[fv_ids],'free vertices')
    fn_raw=np.cross(F[:,1]-F[:,0],F[:,2]-F[:,0]);twice=np.linalg.norm(fn_raw,axis=1)
    fn=np.divide(fn_raw,twice[:,None],out=np.zeros_like(fn_raw),where=twice[:,None]>0)
    good=twice*.5>summary['config']['area_floor_L2']*C.L*C.L
    distance_blocks=[];triangle_pairs=0;segment_pairs=0
    for start in range(0,len(points),64):
        stop=min(len(points),start+64);q=points[start:stop];n=len(q);ii=np.arange(n)
        td,tp=independent_triangles(q,F);triangle_pairs+=td.size
        sd,sp,st=independent_segments(q,segments);segment_pairs+=sd.size
        vd=np.linalg.norm(q[:,None]-verts[fv_ids][None],axis=2)
        aligned=(np.abs(a['source_normals'][start:stop]@fn.T)>=np.cos(np.deg2rad(summary['config']['normal_angle_degrees'])))&good[None]
        pd=np.where(aligned,td,np.inf)
        targets={'surface':td,'parallel_surface':pd,'edge':sd,'open_edge':sd[:,open_ids],'vertex':vd}
        for name,distances in targets.items():
            expected=distances.min(axis=1) if distances.shape[1] else np.full(n,np.inf)
            max_error=max(max_error,_close(a[name+'_distance'][start:stop],expected,name+' minimum',tol=2e-9))
            selected=a[name+'_id'][start:stop];valid=selected>=0
            witness=a[name+'_point'][start:stop]
            if valid.any():
                max_error=max(max_error,_close(np.linalg.norm(witness[valid]-q[valid],axis=1),expected[valid],name+' witness distance',tol=2e-9))
                if name in ('surface','parallel_surface'):
                    bc=a[name+'_barycentric'][start:stop][valid]
                    assert np.min(bc)>=-1e-9 and np.max(bc)<=1.+1e-9
                    max_error=max(max_error,_close(bc.sum(1),np.ones(valid.sum()),name+' barysum'))
                    max_error=max(max_error,_close(np.einsum('qi,qij->qj',bc,F[selected[valid]]),witness[valid],name+' finite triangle witness'))
                elif name in ('edge','open_edge'):
                    t=a[name+'_parameter'][start:stop][valid]
                    assert np.all((t>=0)&(t<=1))
                    seg=segments[selected[valid]]
                    max_error=max(max_error,_close(seg[:,0]*(1-t[:,None])+seg[:,1]*t[:,None],witness[valid],name+' finite segment witness'))
                    if name=='open_edge':assert set(selected[valid]).issubset(set(open_ids))
                else:max_error=max(max_error,_close(witness[valid],verts[fv_ids[selected[valid]]],'finite vertex witness'))
            checks+=n*4
        distance_blocks.append(np.stack((td.min(1),sd.min(1),vd.min(1)),axis=1))
    independently=np.concatenate(distance_blocks)
    # Previously saved boundary-vertex metrics must be preserved exactly up to
    # independent FP64 arithmetic, although free corner ordering can differ.
    prior_vertices={int(row['source_vertex_id']):row for row in old['old_boundary_vertices']}
    prior_dist=old['old_boundary_distances']
    assert len(prior_vertices)==len(vertex_idx)==len(prior_dist)
    for local,(query_id,sid) in enumerate(zip(vertex_idx,boundary_vids)):
        before=prior_vertices[sid];surface=prior_dist[local]
        _close(points[query_id],np.asarray([float(before[axis]) for axis in 'xyz']),'old vertex coordinates')
        old_values=[float(surface['C_boundary_to_F_surface_model']),float(surface['C_boundary_to_F_all_finite_edges_model']),float(before['distance_model'])]
        max_error=max(max_error,_close(independently[query_id],old_values,'inherited distances',tol=3e-9));checks+=3
    # Recompute saved classification annotations without invoking a detector.
    cf_bad=set();ff_bad=set();cf_book={}
    for row in old['pairs']['CF']['contacts']:
        ci,fi=row['pair'];cf_book[(ci,fi)]=row
        if row['detector'].get('illegal') is True:cf_bad.add(fi)
    for row in old['pairs']['FF']['contacts']:
        if row['detector'].get('illegal') is True:ff_bad.update(row['pair'])
    for qi,row in enumerate(samples):
        for name in ('surface','parallel_surface'):
            fi=int(a[name+'_id'][qi])
            if fi<0:continue
            assert (row[name+'_saved_CF_any_illegal']=='True')==(fi in cf_bad)
            assert (row[name+'_saved_FF_any_illegal']=='True')==(fi in ff_bad)
            relationships=json.loads(row[name+'_saved_CF_relationship'])
            for rel in relationships:
                prior=cf_book.get((rel['C_face_row'],fi))
                assert rel['classification']==(prior['classification'] if prior else 'disjoint')
                checks+=1
    assert 'torch' not in sys.modules
    return dict(cell_id=cell['cell_id'],status='PASS',queries=len(points),edge_midpoints=count_midpoints,
        source_boundary_vertices=len(vertex_idx),source_boundary_edges=len(boundary),
        independent_triangle_pairs=triangle_pairs,independent_segment_pairs=segment_pairs,
        checks=checks,max_distance_or_reconstruction_error=max_error,
        original_vertex_distance_reuse='PASS',CF_FF_annotation_mapping='PASS',
        seconds=time.perf_counter()-tick,model_calls=0,quality_scans=0)


def verify_results(reg=None):
    tick=time.perf_counter();reg=C.read(C.OUT/'registration.json') if reg is None else reg
    previous_path=C.OUT/'independent_audit/completion.json'
    if previous_path.exists():
        previous=C.read(previous_path);assert previous['registration_sha256']==C.sha(C.OUT/'registration.json')
        for path,expected in reg['protected_hashes'].items():assert C.sha(path)==expected,path
        for receipt in C.read(C.OUT/'analysis_completion.json')['receipts']:
            for path,expected in receipt['files'].items():assert C.sha(path)==expected,path
        return previous
    rows=[verify_cell(cell) for cell in reg['cells']]
    for path,expected in reg['protected_hashes'].items():assert C.sha(path)==expected,path
    C.csvwrite(C.OUT/'independent_audit/per_cell.csv',rows)
    result=dict(status='PASS',version='independent_boundary_audit_v1',cells=len(rows),
        queries=sum(x['queries'] for x in rows),checks=sum(x['checks'] for x in rows),
        independent_triangle_pairs=sum(x['independent_triangle_pairs'] for x in rows),
        independent_segment_pairs=sum(x['independent_segment_pairs'] for x in rows),
        max_distance_or_reconstruction_error=max(x['max_distance_or_reconstruction_error'] for x in rows),
        protected_files_unchanged=len(reg['protected_hashes']),
        registration_sha256=C.sha(C.OUT/'registration.json'),
        seconds=time.perf_counter()-tick,numeric_contracts=check_numeric_contracts(),
        method='Independent dominant-axis finite-triangle projection + clamped segment enumeration + saved indexed edge incidence; no production numeric helper import',
        model_calls=0,quality_scans=0)
    C.write(C.OUT/'independent_audit/completion.json',result);return result



def verify_aggregation():
    tick=time.perf_counter();reg=C.read(C.OUT/'registration.json');tables=C.OUT/'tables'
    outputs=csv_rows(tables/'per_output.csv');mapped_rows=csv_rows(tables/'source_boundary_legal_edges.csv')
    bounds=csv_rows(tables/'sampling_bounds.csv');partitions=csv_rows(tables/'distance_partitions.csv')
    arms=csv_rows(tables/'paired_models.csv');stages=csv_rows(tables/'paired_stages.csv');parents=csv_rows(tables/'per_parent.csv')
    previous_path=C.OUT/'independent_audit/aggregation_completion.json'
    if previous_path.exists():
        previous=C.read(previous_path);assert previous['registration_sha256']==C.sha(C.OUT/'registration.json')
        for path,expected in previous['source_tables_hashes'].items():assert C.sha(path)==expected,path
        for path,expected in reg['protected_hashes'].items():assert C.sha(path)==expected,path
        return previous
    assert len(outputs)==44 and len(arms)==22 and len(stages)==22
    out={x['cell_id']:x for x in outputs};assert len(out)==44
    case_rows=[];checks=0;maxerr=0.;na_values=0
    def value_assert(actual,expected,context):
        nonlocal checks,maxerr,na_values
        checks+=1
        if expected is None or (isinstance(expected,(float,np.floating)) and not np.isfinite(expected)):
            assert actual=='',(context,'expected preserved NA',actual);na_values+=1;return
        maxerr=max(maxerr,_close([float(actual)],[float(expected)],context,tol=2e-12))
    for cell in reg['cells']:
        beginchecks=checks;cid=cell['cell_id'];row=out[cid];folder=C.OUT/'results'/cid
        a=C.arrays(folder/'vectors.npz');g=C.arrays(cell['geometry_path']);old=C.read(cell['inherited_path'])
        src=C.arrays(cell['condition_source_path']);summary=C.read(folder/'summary.json');samples=csv_rows(folder/'samples.csv')
        w=a['length_weights'];mask=w>0;h=w[mask];den=float(h.sum());k=len(g['C']);nf=len(g['F'])
        for key in ('cell_id','request_id','parent_id','label','stage'):assert row[key]==str(cell[key])
        for key,expected in [('K',k),('N',cell['N']),('source_boundary_length',den),('edge_queries',mask.sum()),('vertex_queries',(~mask).sum()),('F_faces',nf),('F_components',len(set(a['F_component_ids'].tolist())))]:value_assert(row[key],expected,cid+key)
        assert row['required_interface_status']=='NA_NO_REQUIRED_INTERFACE'
        assert row['inherited_adoption']==old['review']['C_adoption']
        inherited_map={'inherited_legal_edges':'C_F_valid_edge_count','inherited_CF':'CF_illegal_affected_free_fraction','inherited_FF':'FF_illegal_affected_free_fraction','inherited_I':'ConflictCov_C','inherited_A_VMR_001L':'A_VMR__L_tau_0.001','inherited_PMR_001L':'PMR__L_tau_0.001'}
        for key,original in inherited_map.items():value_assert(row[key],float(old['numeric'][original]),cid+key)
        for name in ('surface','parallel_surface','edge','open_edge','vertex'):
            d=a[name+'_distance'][mask]/C.L
            expected={'length_weighted_mean_over_L':float(np.sum(d*h)/den),'max_over_L':float(d.max())}
            order=np.argsort(d,kind='stable');cum=np.cumsum(h[order])
            for suffix,quantile in [('length_weighted_median_over_L',.5),('length_weighted_p95_over_L',.95)]:
                j=min(int(np.searchsorted(cum,quantile*cum[-1],side='left')),len(order)-1);expected[suffix]=float(d[order[j]])
            for tau in reg['geometry_config']['distance_thresholds_over_L']:expected['length_fraction_lt_'+format(float(tau),'.9g')+'L']=float(h@(d<tau)/den)
            for key,val in expected.items():value_assert(row[name+'__'+key],val,cid+name+key)
        for field in ('surface_saved_CF_any_illegal','surface_saved_FF_any_illegal','surface_parallel_15deg'):
            expected=sum(float(ww) for ww,sample in zip(w,samples) if sample[field]=='True')/den
            value_assert(row[field+'__length_fraction'],expected,cid+field)
        value_assert(row['multiple_normal_groups__length_fraction'],float(h@(a['corridor_layer_count'][mask]>=2)/den),cid+'multilayer')
        signed=np.einsum('ij,ij->i',a['surface_point']-a['points'],a['source_normals'])/C.L
        value_assert(row['surface_signed_offset__weighted_mean_over_L'],float(h@signed[mask]/den),cid+'signed')
        # Original valid edges contain actual vertex IDs. Match them to true
        # source-ID boundary edges and require unique endpoint identities.
        coords,inc,source_edges=independent_source_boundary(src)
        coord_key=lambda a,b:tuple(sorted((tuple(map(float,a)),tuple(map(float,b)))))
        boundary_by_coordinates={coord_key(coords[u],coords[v]):(u,v) for u,v in source_edges}
        assert len(boundary_by_coordinates)==len(source_edges),'Ambiguous source coordinates must not collapse source identities'
        legal=[z for z in old['valid_edges'] if z['is_C_F']=='True' and z['valid_edge']=='True']
        legal_keys={}
        for edge in legal:
            u,v=int(edge['vertex_u']),int(edge['vertex_v']);key=coord_key(g['mesh_vertices'][u],g['mesh_vertices'][v])
            assert key not in legal_keys and key in boundary_by_coordinates
            legal_keys[key]=edge
            incident=json.loads(edge['incident_face_ids']) if isinstance(edge['incident_face_ids'],str) else edge['incident_face_ids']
            assert len(incident)==2 and any(fi<k for fi in incident) and any(fi>=k for fi in incident)
            assert all(u in g['mesh_faces'][fi] and v in g['mesh_faces'][fi] for fi in incident)
        legal_length=0.;selected=[z for z in mapped_rows if z['cell_id']==cid];assert len(selected)==len(source_edges)
        for z in selected:
            source_key=tuple(sorted((int(z['source_vertex0']),int(z['source_vertex1']))));assert source_key in source_edges
            u,v=source_key;key=coord_key(coords[u],coords[v]);truth=key in legal_keys
            assert (z['saved_legal_CF_shared_edge']=='True')==truth
            if truth:
                assert json.loads(z['saved_valid_edge_record'])==legal_keys[key]
                legal_length+=float(np.linalg.norm(coords[v]-coords[u]))
            else:assert z['saved_valid_edge_record']==''
            checks+=4
        value_assert(row['exact_legal_source_boundary_edges'],len(legal_keys),cid+'legalcount')
        value_assert(row['exact_legal_source_boundary_length'],legal_length,cid+'legallength')
        value_assert(row['exact_legal_source_boundary_length_fraction'],legal_length/den,cid+'legalfraction')
        # Per-bin midpoint integration and rigorous 1-Lipschitz uncertainty.
        selected_bounds=[z for z in bounds if z['cell_id']==cid]
        assert len(selected_bounds)==len(reg['geometry_config']['distance_thresholds_over_L'])*4
        expected_error=float(sum(float(hi)**2 for hi in h)/(4*den*C.L))
        for z in selected_bounds:
            tau=float(z['tau_over_L']);d=a[z['domain']+'_distance'][mask]/C.L;radius=h/(2*C.L)
            expected={'midpoint_length_fraction':float(h@(d<tau)/den),'guaranteed_length_fraction_lower':float(h@(d+radius<tau)/den),'possible_length_fraction_upper':float(h@(d-radius<tau)/den),'mean_distance_quadrature_error_bound_over_L':expected_error}
            for key,val in expected.items():value_assert(z[key],val,cid+z['domain']+key)
            lo,mid,hi=(float(z[key]) for key in ('guaranteed_length_fraction_lower','midpoint_length_fraction','possible_length_fraction_upper'))
            assert lo>=-1e-12 and lo<=mid+1e-12 and mid<=hi+1e-12 and hi<=1.+1e-12
        selected_parts=[z for z in partitions if z['cell_id']==cid];assert len(selected_parts)==len(reg['geometry_config']['distance_thresholds_over_L'])
        ds=a['surface_distance'][mask]/C.L;de=a['edge_distance'][mask]/C.L;dv=a['vertex_distance'][mask]/C.L
        for z in selected_parts:
            tau=float(z['tau_over_L']);groups={'surface_far':ds>=tau,'surface_near_all_edge_far':(ds<tau)&(de>=tau),'all_edge_near_vertex_far':(de<tau)&(dv>=tau),'vertex_near':dv<tau}
            assert np.all(sum(groups.values())==1)
            for key,m in groups.items():value_assert(z[key],float(h@m/den),cid+'partition'+key)
        case_rows.append(dict(cell_id=cid,status='PASS',checks=checks-beginchecks,exact_legal_source_edges=len(legal_keys),exact_legal_source_length=legal_length,sampling_error_bound_over_L=expected_error))
    # Verify every published delta column, including missing candidate states.
    ix={(x['request_id'],x['label'],x['stage']):x for x in outputs}
    for pairing,suffix in [(arms,'__CREF_minus_SMALL'),(stages,'__POST_minus_RAW')]:
        for p in pairing:
            if suffix=='__CREF_minus_SMALL':left=ix[p['request_id'],'SMALL_C200',p['stage']];right=ix[p['request_id'],'CREF_C200',p['stage']]
            else:left=ix[p['request_id'],p['label'],'RAW'];right=ix[p['request_id'],p['label'],'POST']
            for column,val in p.items():
                if not column.endswith(suffix):continue
                key=column[:-len(suffix)];a,b=left[key],right[key]
                expected=float(b)-float(a) if a!='' and b!='' else None
                value_assert(val,expected,'paired'+p['request_id']+column)
            if suffix=='__CREF_minus_SMALL':assert p['SMALL_adoption']==left['inherited_adoption'] and p['CREF_adoption']==right['inherited_adoption']
    for p in parents:
        group=[x for x in outputs if x['parent_id']==p['parent_id'] and x['label']==p['label'] and x['stage']==p['stage']]
        value_assert(p['outputs'],len(group),'parentoutputcount')
        value_assert(p['adoption_yes'],sum(x['inherited_adoption']=='YES' for x in group),'parentadoption')
        value_assert(p['legal_positive_outputs'],sum(int(x['inherited_legal_edges'])>0 for x in group),'parentlegal')
        counts=json.loads(p['valid_counts'])
        for key,count in counts.items():
            observed=[float(x[key]) for x in group if x[key]!=''];assert count==len(observed)
            value_assert(p[key+'__macro_mean'],float(np.mean(observed)) if observed else None,'parentmean'+key)
    completion=C.read(C.OUT/'aggregation_completion.json')
    for path,sha in completion['table_hashes'].items():assert C.sha(path)==sha,path
    for path,sha in reg['protected_hashes'].items():assert C.sha(path)==sha,path
    C.csvwrite(C.OUT/'independent_audit/aggregation_per_cell.csv',case_rows)
    result=dict(status='PASS',version='independent_boundary_aggregation_audit_v1',cells=44,checks=checks,
        exact_boundary_mapping_rows=len(mapped_rows),sampling_bounds_rows=len(bounds),distance_partition_rows=len(partitions),
        paired_model_rows=len(arms),paired_stage_rows=len(stages),parent_rows=len(parents),
        NA_values_and_deltas_preserved=na_values,max_absolute_arithmetic_discrepancy=maxerr,
        protected_files_unchanged=len(reg['protected_hashes']),source_tables_hashes=completion['table_hashes'],
        registration_sha256=C.sha(C.OUT/'registration.json'),seconds=time.perf_counter()-tick,
        new_geometry_analyses=0,new_quality_scans=0,model_calls=0,
        bounds_interpretation='Fixed finite-set distance is 1-Lipschitz; threshold bounds are conservative per-segment-midpoint bins, not semantic or required-interface success certificates.')
    C.write(C.OUT/'independent_audit/aggregation_completion.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('inputs','results','aggregation','all'),default='all')
    args=parser.parse_args();check_numeric_contracts()
    if args.stage in ('inputs','all'):print(json.dumps(verify_inputs(),sort_keys=True))
    if args.stage in ('results','all'):print(json.dumps(verify_results(),sort_keys=True))
    if args.stage in ('aggregation','all'):print(json.dumps(verify_aggregation(),sort_keys=True))
    assert 'torch' not in sys.modules


if __name__=='__main__':main()