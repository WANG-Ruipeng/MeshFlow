"""Independently verify saved incremental transport, without mesh extraction."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import numpy as np


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def close(a, b): return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=2e-9, atol=5e-12, equal_nan=True)
def same(a, b): return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def finite_stats(x):
    x = np.asarray(x); f = x[np.isfinite(x)]
    return dict(count=int(x.size), nonfinite_count=int(x.size-f.size), minimum=float(f.min()) if f.size else None,
                median=float(np.median(f)) if f.size else None, maximum=float(f.max()) if f.size else None)


def difference(a, b):
    if np.shape(a) != np.shape(b): return dict(same_shape=False)
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all(): return dict(same_shape=True, both_finite=False)
    result = dict(same_shape=True, both_finite=True, max_abs=float(np.max(abs(a-b))) if a.size else 0.,
                  relative_frobenius=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300)))
    if a.ndim >= 3 and len(a):
        absolute = np.linalg.norm((a-b).reshape(len(a), -1), axis=1)
        relative = absolute/np.maximum(np.linalg.norm(b.reshape(len(b), -1), axis=1), 1e-300)
        result.update(per_element_relative=finite_stats(relative), worst_relative_element=int(np.argmax(relative)),
                      per_element_absolute=finite_stats(absolute), worst_absolute_element=int(np.argmax(absolute)))
    return result


def independent_step(xp, xc, indices, weights, h):
    """Frozen IDs/w/h; current source positions; displacement-form weighted LSQ."""
    X = (xp[indices]-xp[:, None, :])/h[:, None, None]
    D = ((xc[indices]-xc[:, None, :])-(xp[indices]-xp[:, None, :]))/h[:, None, None]
    WX = np.sqrt(weights)[..., None]*X
    singular = np.linalg.svd(WX, compute_uv=False)
    rank = np.sum(singular > 1e-6*singular[:, :1], axis=1)
    raw = np.full((len(xp), 3, 3), np.nan)
    used = np.broadcast_to(np.eye(3), raw.shape).copy(); reason = np.zeros(len(xp), dtype=int)
    for i in range(len(xp)):
        if not np.isfinite(D[i]).all(): reason[i] = 2; continue
        if rank[i] < 3: reason[i] = 1; continue
        try:
            B = np.linalg.lstsq(WX[i], np.sqrt(weights[i])[:, None]*D[i], rcond=1e-6)[0]
            raw[i] = np.eye(3)+B.T
            if not np.isfinite(raw[i]).all(): reason[i] = 2; continue
            determinant, condition = np.linalg.det(raw[i]), np.linalg.cond(raw[i])
            if not np.isfinite(determinant) or np.isnan(condition): reason[i] = 2
            elif determinant <= 1e-8: reason[i] = 3
            elif condition > 1e4 or not np.isfinite(condition): reason[i] = 4
            else: used[i] = raw[i]
        except np.linalg.LinAlgError: reason[i] = 5
    return dict(X=X, weighted_design_singular_values=singular, rank=rank, J_raw=raw, J_used=used,
                reason_code=reason, used_fallback=reason != 0)


def forward_h(J, xp, xc):
    H = np.broadcast_to(np.eye(4), (len(J), 4, 4)).copy()
    H[:, :3, :3] = J; H[:, :3, 3] = xc-np.einsum('nij,nj->ni', J, xp)
    return H


def local_scaled_Q(Q, v, h):
    B = np.broadcast_to(np.eye(4), Q.shape).copy()
    B[:, :3, :3] *= h[:, None, None]; B[:, :3, 3] = v
    return B.transpose(0, 2, 1) @ Q @ B


def compare_traces(a, b):
    """Pair capacities by original-input tetrahedron IDs and terminal semantics."""
    if not a.is_file() or not b.is_file(): return dict(available=False)
    aa, bb = arrays(a), arrays(b)
    required = ['delaunay_simplices', 'graph_edges', 'graph_capacities', 'graph_terminals']
    if any(k not in aa or k not in bb for k in required): return dict(available=False, reason='incomplete_saved_trace')
    def edge_map(z):
        inside, outside = map(int, z['graph_terminals'])
        def node(i):
            i = int(i)
            if i == inside: return ('terminal', 'inside')
            if i == outside: return ('terminal', 'outside')
            return ('tet',) + tuple(sorted(map(int, z['delaunay_simplices'][i])))
        result = {}
        for (u, v), cap in zip(z['graph_edges'], z['graph_capacities']):
            key = (node(u), node(v))
            if key in result: raise RuntimeError('Duplicate semantic graph edge')
            result[key] = float(cap)
        return result
    ma, mb = edge_map(aa), edge_map(bb); shared = sorted(set(ma) & set(mb))
    result = dict(available=True, incremental_edge_count=len(ma), direct_edge_count=len(mb),
                  shared_edges=len(shared), incremental_only_edges=len(set(ma)-set(mb)), direct_only_edges=len(set(mb)-set(ma)),
                  paired_capacity_difference=difference(np.array([ma[k] for k in shared]), np.array([mb[k] for k in shared])))
    for key in ['delaunay_points', 'delaunay_simplices', 'initial_labels', 'corrected_labels', 'precut_labels']:
        if key in aa and key in bb: result[key + '_byte_equal'] = same(aa[key], bb[key])
    if same(aa['delaunay_simplices'], bb['delaunay_simplices']):
        for key in ['postcut_labels', 'graph_terminals']:
            if key in aa and key in bb:
                result[key+'_byte_equal'] = same(aa[key], bb[key])
                result[key+'_changed_count'] = int(np.count_nonzero(aa[key] != bb[key])) if aa[key].shape == bb[key].shape else None
        if all(k in aa and k in bb for k in ['candidate_faces', 'candidate_face_neighbors', 'candidate_scores']):
            matched = same(aa['candidate_faces'], bb['candidate_faces']) and same(aa['candidate_face_neighbors'], bb['candidate_face_neighbors'])
            result['candidate_score_pairing_ids_identical'] = matched
            if matched: result['candidate_score_difference'] = difference(aa['candidate_scores'], bb['candidate_scores'])
    if 'surface_faces_input_ids' in aa and 'surface_faces_input_ids' in bb:
        canonical = lambda f: sorted(tuple(sorted(map(int,t))) for t in f)
        result['canonical_surface_input_face_multiset_equal'] = canonical(aa['surface_faces_input_ids']) == canonical(bb['surface_faces_input_ids'])
    return result


def verify(root):
    root = Path(root).resolve(); protocol = read(root/'FROZEN_PROTOCOL.json'); config = protocol['config']
    parent, motion = Path(config['parent_run']), Path(config['motion_run'])
    rows = read(root/'LEDGER.json'); checks = []; records = []; returns = []
    def check(name, passed, **extra): checks.append(dict(name=name, passed=bool(passed), **extra))
    check('45_new_slots', len(rows) == 45)
    expected = {(s, d, f) for s in ['sphere', 'thin_u', 'bottle_local_1'] for d, frames in [('forward', range(8)), ('reverse', range(6, -1, -1))] for f in frames}
    check('all_direction_frame_slots', {(r['source_object'], r['direction'], int(r['frame'])) for r in rows} == expected)
    check('protocol_hash', sha(root/'FROZEN_PROTOCOL.json') == read(root/'FROZEN_PROTOCOL_SHA256.json')['sha256'])
    for source in ['sphere', 'thin_u', 'bottle_local_1']:
        group = sorted([r for r in rows if r['source_object'] == source], key=lambda r: int(r['step_index']))
        fixed = arrays(root/'motion_inputs'/source/'neighborhood.npz')
        base = arrays(root/'motion_inputs'/source/'base.npz')
        h, weights, ids = fixed['h'], fixed['weights'], fixed['neighbor_indices']
        old_fixed = arrays(motion/'motion_inputs'/source/'neighborhood.npz')
        for key in ['h', 'weights', 'neighbor_indices']: check(source+'_initial_'+key, same(fixed[key], old_fixed[key]))
        previous_Q = base['Q'].copy(); cumulative = np.broadcast_to(np.eye(4), base['Q'].shape).copy()
        previous_x = base['v'].copy(); previous_key = None; forward_steps = []; final_forward = None
        for step, row in enumerate(group):
            run_key = row['object']+'__'+row['state']; directory = root/'runs'/run_key
            p = arrays(directory/'parameters.npz'); raw = arrays(directory/'raw_state.npz')
            estimate = arrays(directory/'estimated_motion.npz'); current = arrays(directory/'current_neighborhood.npz')
            control_name = row['source_frame_object']; frozen = arrays(parent/'runs'/(control_name+'__freeze')/'parameters.npz')
            for key in ['v', 'p', 'n']: check(run_key+'_frozen_'+key, same(p[key], frozen[key]))
            for key in ['neighbor_indices', 'weights', 'h']: check(run_key+'_fixed_'+key, same(current[key], fixed[key]))
            check(run_key+'_step_index', int(row['step_index']) == step)
            check(run_key+'_initial_Q', same(raw['initial_Q'], base['Q']))
            check(run_key+'_initial_x', same(raw['initial_x'], base['v']))
            check(run_key+'_previous_Q', same(raw['Q_previous'], previous_Q))
            check(run_key+'_previous_x', same(raw['x_previous'], previous_x))
            check(run_key+'_current_x', same(raw['x_current'], p['v']))
            check(run_key+'_cumulative_previous', close(raw['cumulative_H_previous'], cumulative))
            if step: check(run_key+'_previous_run_key', str(row['previous_run_key']) == previous_key)
            independently = independent_step(previous_x, p['v'], ids, weights, h)
            for key in ['J_raw', 'J_used', 'rank', 'reason_code', 'used_fallback']:
                check(run_key+'_independent_'+key, close(estimate[key], independently[key]))
            check(run_key+'_current_design', close(current['X'], independently['X']))
            H = forward_h(independently['J_used'], previous_x, p['v'])
            check(run_key+'_world_H', close(raw['H'], H))
            cumulative = H @ cumulative; check(run_key+'_cumulative_product', close(raw['cumulative_H'], cumulative))
            inverse = np.linalg.inv(H)
            predicted = inverse.transpose(0, 2, 1) @ previous_Q @ inverse
            if step == 0: predicted = base['Q'].copy()
            check(run_key+'_full_Q_recurrence', close(p['Q'], predicted))
            check(run_key+'_raw_state_Q', same(raw['Q'], p['Q']))
            total_inverse = np.linalg.inv(cumulative)
            composed = total_inverse.transpose(0, 2, 1) @ base['Q'] @ total_inverse
            check(run_key+'_composed_Q_identity', close(p['Q'], composed))
            check(run_key+'_Q_eigenvalues', close(p['eigs'], np.linalg.svd(p['Q'][:, :3, :3], compute_uv=False)))
            direct_dir = motion/'runs'/(control_name+'__lsq16'); direct = arrays(direct_dir/'parameters.npz')
            local_a, local_b = local_scaled_Q(p['Q'], p['v'], h), local_scaled_Q(direct['Q'], direct['v'], h)
            record = dict(object=row['object'], source_object=source, direction=row['direction'], frame=row['frame'], step_index=step,
                          fallback_count=int(np.count_nonzero(estimate['used_fallback'])),
                          rank_counts={str(int(k)):int(n) for k,n in zip(*np.unique(estimate['rank'],return_counts=True))},
                          used_J_condition=finite_stats(np.linalg.cond(estimate['J_used'])),
                          current_H_condition=finite_stats(np.linalg.cond(H)), cumulative_H_condition=finite_stats(np.linalg.cond(cumulative)),
                          Q_recurrence_difference=difference(p['Q'], predicted), Q_composed_difference=difference(p['Q'], composed),
                          local_h_scaled_Q_vs_direct=difference(local_a, local_b),
                          spectral_normalized_Q_vs_direct=difference(p['Q']/p['eigs'][:, 0, None, None], direct['Q']/direct['eigs'][:, 0, None, None]),
                          local_spectral_normalized_Q_vs_direct=difference(local_a/p['eigs'][:, 0, None, None], local_b/direct['eigs'][:, 0, None, None]),
                          normal_vs_direct=difference(p['n'], direct['n']),
                          mesh_identical_to_direct=(directory/'mesh.obj').is_file() and (direct_dir/'mesh.obj').is_file() and sha(directory/'mesh.obj') == sha(direct_dir/'mesh.obj'),
                          graph_vs_direct=compare_traces(directory/'extractor_trace.npz', direct_dir/'extractor_trace.npz'))
            records.append(record)
            if row['direction'] == 'forward':
                if step: forward_steps.append(raw['H'].copy())
                if row['frame'] == 7:
                    final_forward = p['Q'].copy()
                    inverse_control = final_forward.copy()
                    for saved_H in reversed(forward_steps): inverse_control = saved_H.transpose(0, 2, 1) @ inverse_control @ saved_H
                    check(source+'_inverse_saved_forward_control', close(inverse_control, base['Q']))
                    returns.append(dict(source_object=source, inverse_saved_forward_control=difference(inverse_control, base['Q'])))
            if row['direction'] == 'reverse' and row['frame'] == 0:
                initial_dir = parent/'runs'/(source+'_frame00__freeze')
                returns[-1].update(actual_return_Q=difference(p['Q'], base['Q']),
                                   actual_return_local_Q=difference(local_scaled_Q(p['Q'], p['v'], h), local_scaled_Q(base['Q'], base['v'], h)),
                                   actual_return_spectral_Q=difference(p['Q']/p['eigs'][:,0,None,None], base['Q']/base['eigs'][:,0,None,None]),
                                   actual_return_cumulative_H=difference(cumulative, np.broadcast_to(np.eye(4), cumulative.shape)),
                                   actual_return_vertex_byte_equal=same(p['v'], base['v']),
                                   actual_return_mesh_identical_to_initial=(directory/'mesh.obj').is_file() and sha(directory/'mesh.obj') == sha(initial_dir/'mesh.obj'))
            previous_Q = p['Q'].copy(); previous_x = p['v'].copy(); previous_key = run_key
    result = dict(passed=all(c['passed'] for c in checks), count=len(checks), failed_checks=[c for c in checks if not c['passed']],
                  checks=checks, records=records, return_diagnostics=returns,
                  note='Independent weighted lstsq and saved-state algebra only; no re-extraction or parameter repair. Diagnostic differences versus direct control are not failed verification checks.')
    (root/'audit').mkdir(exist_ok=True)
    target = root/'audit/ACCUM_VERIFY.json'
    if target.exists(): raise RuntimeError('Refusing existing verification file')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],count=result['count'],failed_checks=result['failed_checks'])))
    if not result['passed']: raise RuntimeError('Incremental saved-artifact verification failed')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);verify(p.parse_args().root)
