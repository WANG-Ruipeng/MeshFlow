"""Independent saved-artifact audit; never extracts meshes or fits to a reference.

Regression is independently recomputed with numpy.linalg.lstsq/solve. Analytic
Jacobians are read only in a final diagnostic pass, never for acceptance, tuning,
fallback, or quadric construction. Existing audit files are never overwritten.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def close(a, b, rtol=2e-10, atol=2e-12):
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def summary(x):
    x = np.asarray(x); good = x[np.isfinite(x)]
    return dict(finite_count=int(good.size), nonfinite_count=int(x.size-good.size),
                minimum=float(good.min()) if good.size else None,
                median=float(np.median(good)) if good.size else None,
                maximum=float(good.max()) if good.size else None)


def independent_estimate(x0, xt, indices, mode):
    """No estimator-module imports and no analytic/reference inputs."""
    dx = x0[indices]-x0[:, None, :]
    dist = np.linalg.norm(dx, axis=2); h = dist[:, -1]
    w = np.exp(-(dist/h[:, None])**2); w /= w.sum(axis=1, keepdims=True)
    X = dx/h[:, None, None]
    D = ((xt[indices]-xt[:, None, :])-dx)/h[:, None, None]
    WX = np.sqrt(w)[..., None]*X
    singular = np.linalg.svd(WX, compute_uv=False)
    rank = (singular > 1e-6*singular[:, :1]).sum(axis=1)
    G = np.einsum('nki,nk,nkj->nij', X, w, X)
    lam = 1e-3*np.trace(G, axis1=1, axis2=2)/3
    n = len(x0); raw = np.full((n, 3, 3), np.nan)
    used = np.broadcast_to(np.eye(3), raw.shape).copy()
    reason = np.zeros(n, dtype=np.int64)
    raw_det = np.full(n, np.nan); raw_cond = np.full(n, np.nan)
    rr = np.full(n, np.nan); ur = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(D[i]).all():
            reason[i] = 2
            continue
        if mode == 'lsq16' and rank[i] < 3:
            reason[i] = 1
        else:
            try:
                if mode == 'lsq16':
                    B = np.linalg.lstsq(WX[i], np.sqrt(w[i])[:, None]*D[i], rcond=1e-6)[0]
                else:
                    B = np.linalg.solve(G[i]+lam[i]*np.eye(3), X[i].T@(w[i, :, None]*D[i]))
                raw[i] = np.eye(3)+B.T
                rr[i] = np.sqrt(np.sum(w[i]*np.sum((D[i]-X[i]@B)**2, axis=1)))
                if not np.isfinite(raw[i]).all():
                    reason[i] = 2
                else:
                    raw_det[i] = np.linalg.det(raw[i]); raw_cond[i] = np.linalg.cond(raw[i])
                    if not np.isfinite(raw_det[i]) or np.isnan(raw_cond[i]): reason[i] = 2
                    elif raw_det[i] <= 1e-8: reason[i] = 3
                    elif raw_cond[i] > 1e4 or not np.isfinite(raw_cond[i]): reason[i] = 4
                    else: used[i] = raw[i]
            except np.linalg.LinAlgError:
                reason[i] = 5
        ur[i] = np.sqrt(np.sum(w[i]*np.sum((D[i]-X[i]@(used[i]-np.eye(3)).T)**2, axis=1)))
    return dict(J_raw=raw, J_used=used, neighbor_indices=indices,
                neighbor_distances=dist, h=h, weights=w, X=X, G=G,
                weighted_design_singular_values=singular, rank=rank,
                ridge_lambda=lam, used_fallback=reason!=0, reason_code=reason,
                raw_determinant=raw_det, raw_condition=raw_cond,
                raw_weighted_residual_normalized=rr, used_weighted_residual_normalized=ur,
                raw_weighted_residual_world=rr*h, used_weighted_residual_world=ur*h)


def homogeneous_translation(v):
    T = np.broadcast_to(np.eye(4), (len(v), 4, 4)).copy()
    T[:, :3, 3] = v
    return T


def transported_quadric(base, current_v, J):
    T0 = homogeneous_translation(base['v'])
    C0 = T0.transpose(0, 2, 1)@base['Q']@T0
    H = np.broadcast_to(np.eye(4), C0.shape).copy(); H[:, :3, :3] = np.linalg.inv(J)
    C = H.transpose(0, 2, 1)@C0@H
    Tw = homogeneous_translation(-current_v)
    return Tw.transpose(0, 2, 1)@C@Tw, C0, C


def verify(root):
    root = Path(root).resolve(); checks = []; unavailable = []; observations = []; diagnostics = []
    out = root/'audit/MOTION_INDEPENDENT_CHECKS.json'
    diagout = root/'audit/J_ESTIMATION_DIAGNOSTICS.json'
    if out.exists() or diagout.exists():
        raise FileExistsError('Refusing to overwrite an existing motion audit')
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    def na(name, reason, **detail):
        unavailable.append(dict(name=name, reason=reason, **detail))
    def hash_check(name, path, expected):
        path = Path(path); check(name, path.is_file() and sha(path)==expected, path=str(path))
    protocol = read(root/'FROZEN_PROTOCOL.json'); config = protocol['config']
    parent = Path(config['parent_run']); factorial = Path(config['factorial_run'])
    objects = protocol['objects']; arms = ['lsq16', 'ridge16']
    rows = read(root/('LEDGER.json' if (root/'LEDGER.json').exists() else 'LEDGER_RAW.json'))
    slots = [(r['object'], r['state']) for r in rows]
    expected = {(o, a) for o in objects for a in arms}
    check('48_unique_expected_slots', len(objects)==24 and len(rows)==48 and len(set(slots))==48
          and set(slots)==expected and set(config['run_states'])==set(arms))
    check('parent_objects_match', set(objects)==set(read(parent/'FROZEN_PROTOCOL.json')['objects']))
    for label, run in [('new', root), ('parent', parent), ('factorial', factorial)]:
        hash_check(label+':protocol_hash', run/'FROZEN_PROTOCOL.json', read(run/'FROZEN_PROTOCOL_SHA256.json')['sha256'])
    for source in read(root/'SOURCE_SNAPSHOT.json')['files']:
        hash_check('source_snapshot:'+source['path'], root/source['path'], source['sha256'])
    for source in config.get('source_files', []):
        hash_check('bound_input:'+source['path'], source['path'], source['sha256'])
    vendor = root/'vendor/PoNQ_to_mesh.py'; raw = vendor.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    prov = read(root/'SOURCE_PROVENANCE.json')
    check('official_vendor_blob', blob==prov['official_extractor_blob'])
    check('vendor_same_as_parent', raw==(parent/'vendor/PoNQ_to_mesh.py').read_bytes())
    for relative in ['code/common.py','code/safe_extract.py','code/run_clay.py','code/exact_audit.py']:
        check('same_parent_backend:'+relative,(root/relative).read_bytes()==(parent/relative).read_bytes())
    if 'official_extractor_sha256' in prov: hash_check('vendor_sha256', vendor, prov['official_extractor_sha256'])
    source_cache = {}; pending_diagnostics = []
    for row in rows:
        obj, arm = row['object'], row['state']; tag = obj+':'+arm
        meta = objects[obj]; source = meta.get('source_object', obj.rsplit('_frame', 1)[0])
        if source not in source_cache:
            base = arrays(parent/'path_inputs'/source/'base.npz')
            hood = arrays(root/'motion_inputs'/source/'neighborhood.npz')
            indices = hood['neighbor_indices']; x0 = base['v']; n = len(x0)
            check(source+':256_source_elements', n==256)
            good = (indices.shape==(n,16) and np.issubdtype(indices.dtype,np.integer)
                    and np.all(indices>=0) and np.all(indices<n))
            check(source+':neighborhood_index_range', good)
            if not good: raise ValueError('Invalid neighborhood indices')
            distances = np.linalg.norm(x0[indices]-x0[:,None,:], axis=2)
            full = np.linalg.norm(x0[:,None,:]-x0[None,:,:], axis=2); np.fill_diagonal(full,np.inf)
            threshold = np.sort(full,axis=1)[:,15]
            check(source+':k16_nearest_excluding_self', np.all(indices!=np.arange(n)[:,None])
                  and all(len(set(a.tolist()))==16 for a in indices)
                  and np.all(np.diff(distances,axis=1)>=-1e-14)
                  and np.all(distances[:,-1]<=threshold+1e-14))
            check(source+':hood_x0_identity', same(hood['x0'], x0))
            check(source+':positive_finite_h', np.all(np.isfinite(hood['h'])) and np.all(hood['h']>0))
            expected_hood = independent_estimate(x0,x0,indices,'ridge16')
            for key in ['neighbor_distances','h','weights','X','G','rank','ridge_lambda']:
                check(source+':hood_'+key, close(hood[key],expected_hood[key]))
            check(source+':hood_singular_values',close(hood['weighted_singular_values'],expected_hood['weighted_design_singular_values']))
            stats = read(root/'motion_inputs'/source/'stats.json')
            check(source+':hood_stats_exist', isinstance(stats,dict))
            source_cache[source] = base,hood
        base, hood = source_cache[source]
        oldrd = parent/'runs'/f'{obj}__freeze'; rd = root/'runs'/f'{obj}__{arm}'
        freeze = arrays(oldrd/'parameters.npz'); params = arrays(rd/'parameters.npz')
        saved = arrays(rd/'estimated_motion.npz')
        saved_stats = read(rd/'estimated_motion_stats.json')
        check(tag+':stats_fallback_rank_counts',
              saved_stats['fallback_count']==int(saved['used_fallback'].sum())
              and saved_stats['rank_deficient_count']==int((saved['rank']<3).sum()))
        for key in ['v','p','n']:
            check(tag+':frozen_'+key, same(params[key],freeze[key]))
        check(tag+':reference_bytes', (root/'inputs'/obj/'reference.obj').read_bytes()==(parent/'inputs'/obj/'reference.obj').read_bytes())
        independent = independent_estimate(base['v'],freeze['v'],hood['neighbor_indices'],arm)
        for key in independent:
            if key in ['X','G']: continue
            check(tag+':independent_'+key, key in saved and close(saved[key],independent[key]))
        check(tag+':fallback_identity_exact', same(saved['J_used'][saved['used_fallback']],
              np.broadcast_to(np.eye(3),saved['J_used'].shape)[saved['used_fallback']]))
        Q, C0, C = transported_quadric(base,freeze['v'],saved['J_used'])
        check(tag+':full_Q_congruence',close(params['Q'],Q),max_absolute_error=float(np.max(abs(params['Q']-Q))))
        check(tag+':constant_residual_preserved',close(C[:,3,3],C0[:,3,3]))
        check(tag+':eigs_from_current_A',close(params['eigs'],np.linalg.svd(params['Q'][:,:3,:3],compute_uv=False)))
        check(tag+':full_Q_symmetric',close(params['Q'],params['Q'].transpose(0,2,1)))
        observations.append(dict(object=obj,state=arm,fallback_count=int(saved['used_fallback'].sum()),
            rank_deficient_count=int((saved['rank']<3).sum()),raw_condition=summary(saved['raw_condition'])))
        pending_diagnostics.append((obj,arm,source,meta,base,saved))
        if not (rd/'extractor_trace.npz').exists():
            na(tag+':trace','not_saved',status=row['status']); continue
        trace = arrays(rd/'extractor_trace.npz'); ft = arrays(oldrd/'extractor_trace.npz')
        precut = ['delaunay_points','delaunay_simplices','initial_labels','corrected_labels','void_labels',
                  'precut_labels','inside_seed_ids','outside_seed_ids','free_tetrahedron_ids','graph_terminals','graph_edges']
        for key in precut:
            if key in trace and key in ft: check(tag+':same_freeze_'+key,same(trace[key],ft[key]))
            else: na(tag+':same_freeze_'+key,'stage_not_recorded',status=row['status'])
        if 'delaunay_points' in trace: check(tag+':trace_sites',same(trace['delaunay_points'][:len(params['v'])],params['v']))
        if 'postcut_labels' in trace and 'postcut_labels' in ft:
            observations[-1]['postcut_changes_vs_freeze']=int(np.count_nonzero(trace['postcut_labels']!=ft['postcut_labels']))
        if 'graph_capacities' in trace:
            cap=trace['graph_capacities']; finite=cap[np.isfinite(cap)]
            if row['status']=='EXTRACTED': check(tag+':valid_graph_capacities',np.isfinite(cap).all() and np.all(cap>=0))
            else: na(tag+':graph_validity','failed_execution_preserved',status=row['status'])
            if (rd/'extractor_audit.json').exists():
                gs=read(rd/'extractor_audit.json').get('graph_capacity_stats')
                if gs is not None:
                    check(tag+':graph_stats',gs['count']==len(cap) and gs['nonfinite_count']==int((~np.isfinite(cap)).sum())
                          and gs['negative_count']==int((cap<0).sum())
                          and gs['minimum_finite']==(float(finite.min()) if finite.size else None)
                          and gs['maximum_finite']==(float(finite.max()) if finite.size else None))
        if (rd/'mesh.npz').exists():
            mesh=arrays(rd/'mesh.npz'); ids=mesh['input_vertex_ids']; faces=mesh['faces']
            good=(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and np.all(ids>=0)
                  and np.all(ids<len(trace['delaunay_points'])) and faces.ndim==2 and faces.shape[1]==3
                  and np.issubdtype(faces.dtype,np.integer) and np.all(faces>=0) and np.all(faces<len(ids)))
            check(tag+':mesh_index_ranges',good)
            if good:
                check(tag+':mesh_ids',same(ids,trace['surface_input_vertex_ids']))
                check(tag+':mesh_vertices',same(mesh['vertices'],trace['delaunay_points'][ids]))
                check(tag+':mesh_faces',same(ids[faces],trace['surface_faces_input_ids']))
            if row.get('mesh_sha256'): hash_check(tag+':mesh_hash',rd/'mesh.obj',row['mesh_sha256'])
        else: na(tag+':mesh','not_saved',status=row['status'])
        if meta['frame']==0:
            check(tag+':t0_J_identity',same(saved['J_used'],np.broadcast_to(np.eye(3),saved['J_used'].shape)))
            check(tag+':t0_canonical_Q_reuse',same(params['Q'],freeze['Q']) and same(params['eigs'],freeze['eigs']))
            paths=[rd/'mesh.obj',oldrd/'mesh.obj',factorial/'runs'/f'{obj}__q_only'/'mesh.obj']
            if all(p.exists() for p in paths): check(tag+':t0_mesh_hashes',len({sha(p) for p in paths})==1)
            else: na(tag+':t0_mesh_hashes','mesh_not_saved')
    # Strictly post-only: no result below enters check()/acceptance or estimator.
    from clay_maps import deform
    for obj,arm,source,meta,base,saved in pending_diagnostics:
        specpath=parent/'path_inputs'/source/'path.json'; spec=read(specpath)
        _, analytic=deform(base['v'],float(meta['t']),spec)
        rawerr=np.linalg.norm(saved['J_raw']-analytic,axis=(1,2))
        usederr=np.linalg.norm(saved['J_used']-analytic,axis=(1,2))
        diagnostics.append(dict(object=obj,state=arm,analytic_path_sha256=sha(specpath),
            raw_frobenius_error=summary(rawerr),used_frobenius_error=summary(usederr),
            per_element_raw_error=[float(x) if np.isfinite(x) else None for x in rawerr],
            per_element_used_error=usederr.tolist(),fallback_count=int(saved['used_fallback'].sum())))
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),
        failed_check_count=sum(not c['passed'] for c in checks),unavailable_count=len(unavailable),
        status_counts=dict(Counter(r['status'] for r in rows)),checks=checks,unavailable=unavailable,
        observations=observations,parent_run=str(parent),factorial_run=str(factorial),
        scope='Independent paired-position regression, full-Q transport, shared pre-cut graph and artifact integrity. Not a geometry-quality or generalization certificate.',
        numerical_tolerance=dict(relative=2e-10,absolute=2e-12),
        analytic_J_used_for_acceptance=False,estimator_module_imported=False)
    out.parent.mkdir(exist_ok=True)
    with out.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    with diagout.open('x',encoding='utf-8') as f:
        json.dump(dict(post_only=True,affects_acceptance=False,rows=diagnostics),f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','unavailable','observations']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    result=verify(parser.parse_args().root)
    raise SystemExit(0 if result['passed'] else 1)
