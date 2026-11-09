"""Post-hoc cross-cut and surface comparisons; no re-extraction or repair."""
from pathlib import Path
import argparse
import json
import numpy as np
from common import load_mesh, sample, metrics
from verify_accum_transport import read, sha, arrays, difference


def trace_maps(z):
    tets = [tuple(sorted(map(int, t))) for t in z['delaunay_simplices']]
    inside, outside = map(int, z['graph_terminals'])
    def node(i):
        i = int(i)
        if i == inside: return ('terminal', 'inside')
        if i == outside: return ('terminal', 'outside')
        return ('tet',) + tets[i]
    capacities, cuts = {}, {}
    for edge, capacity, cut in zip(z['graph_edges'], z['graph_capacities'], z['cut_graph_edge_mask']):
        key = (node(edge[0]), node(edge[1]))
        if key in capacities: raise RuntimeError('Duplicate aggregated semantic graph edge')
        capacities[key] = float(capacity); cuts[key] = bool(cut)
    labels = {key: int(value) for key, value in zip(tets, z['postcut_labels'])}
    surfaces = sorted(tuple(sorted(map(int, f))) for f in z['surface_faces_input_ids'])
    candidates = {}
    for tri, neighbors, score in zip(z['candidate_faces'], z['candidate_face_neighbors'], z['candidate_scores']):
        t = tuple(map(int, tri)); oriented = min(t, t[1:]+t[:1], t[2:]+t[:2])
        key = (tets[int(neighbors[0])], oriented)
        if key in candidates: raise RuntimeError('Duplicate semantic candidate face')
        candidates[key] = float(score)
    return dict(capacities=capacities, cuts=cuts, labels=labels, surfaces=surfaces, candidates=candidates,
                inside_seeds={tets[int(i)] for i in z['inside_seed_ids']}, outside_seeds={tets[int(i)] for i in z['outside_seed_ids']})


def compare_traces(a, b):
    aa, bb = trace_maps(arrays(a)), trace_maps(arrays(b))
    ca, cb = aa['capacities'], bb['capacities']; keys = sorted(set(ca) & set(cb))
    ta, tb = aa['labels'], bb['labels']; common_tets = set(ta) & set(tb)
    postcut_changes = [key for key in common_tets if ta[key] != tb[key]]
    candidates = sorted(set(aa['candidates']) & set(bb['candidates']))
    result = dict(postcut_changed_tetrahedra=len(postcut_changes), postcut_shared_tetrahedra=len(common_tets),
                  graph_same_semantic_edges=set(ca) == set(cb), graph_shared_edges=len(keys),
                  graph_incremental_only_edges=len(set(ca)-set(cb)), graph_direct_only_edges=len(set(cb)-set(ca)),
                  graph_capacity_difference=difference(np.array([ca[k] for k in keys]), np.array([cb[k] for k in keys])),
                  inside_seed_tet_set_equal=aa['inside_seeds'] == bb['inside_seeds'], outside_seed_tet_set_equal=aa['outside_seeds'] == bb['outside_seeds'],
                  candidate_shared_count=len(candidates), candidate_incremental_only=len(set(aa['candidates'])-set(bb['candidates'])),
                  candidate_direct_only=len(set(bb['candidates'])-set(aa['candidates'])),
                  candidate_score_difference=difference(np.array([aa['candidates'][k] for k in candidates]), np.array([bb['candidates'][k] for k in candidates])),
                  canonical_surface_face_multiset_equal=aa['surfaces'] == bb['surfaces'],
                  canonical_faces_only_incremental=len(set(aa['surfaces'])-set(bb['surfaces'])), canonical_faces_only_direct=len(set(bb['surfaces'])-set(aa['surfaces'])))
    if postcut_changes and result['graph_same_semantic_edges'] and result['inside_seed_tet_set_equal'] and result['outside_seed_tet_set_equal']:
        A_A = sum(ca[k]*aa['cuts'][k] for k in keys); A_B = sum(ca[k]*bb['cuts'][k] for k in keys)
        B_A = sum(cb[k]*aa['cuts'][k] for k in keys); B_B = sum(cb[k]*bb['cuts'][k] for k in keys)
        if all(np.isfinite(x) for x in [A_A,A_B,B_A,B_B]):
            result['cross_cut'] = dict(incremental_capacities_incremental_cut=A_A, incremental_capacities_direct_cut=A_B,
                                       direct_capacities_incremental_cut=B_A, direct_capacities_direct_cut=B_B,
                                       incremental_objective_penalty_of_direct_cut=A_B-A_A,
                                       direct_objective_penalty_of_incremental_cut=B_A-B_B)
        else: result['cross_cut_unavailable'] = 'Nonfinite capacity energy; no repair'
    elif postcut_changes: result['cross_cut_unavailable'] = 'Semantic graph/terminal constraints differ'
    return result


def analyze(root):
    root = Path(root).resolve(); out = root/'post_analysis'
    if out.exists(): raise RuntimeError('Refusing existing post-analysis directory')
    config = read(root/'FROZEN_PROTOCOL.json')['config']; motion = Path(config['motion_run'])
    rows = read(root/'LEDGER.json'); records = []
    for row in rows:
        directory = root/'runs'/(row['object']+'__'+row['state'])
        direct = motion/'runs'/(row['source_frame_object']+'__lsq16')
        record = {k: row[k] for k in ['object','source_object','frame','direction','step_index']}
        ta, tb = directory/'extractor_trace.npz', direct/'extractor_trace.npz'
        if ta.is_file() and tb.is_file():
            try: record['trace'] = compare_traces(ta,tb)
            except KeyError as exc: record['trace_unavailable'] = 'Incomplete saved trace: '+str(exc)
        else: record['trace_unavailable'] = 'No saved trace'
        ma, mb = directory/'mesh.obj', direct/'mesh.obj'
        if ma.is_file() and mb.is_file():
            record['incremental_mesh_sha256'], record['direct_mesh_sha256'] = sha(ma),sha(mb)
            record['mesh_byte_equal'] = record['incremental_mesh_sha256'] == record['direct_mesh_sha256']
            if not record['mesh_byte_equal']:
                va,fa = load_mesh(ma);vb,fb = load_mesh(mb)
                # Paired audit sampling schedule fixed independently of observed errors.
                seed = 76000 + {'sphere':0,'thin_u':100,'bottle_local_1':200}[row['source_object']] + row['step_index']
                gt,gn,_ = sample(vb,fb,8192,seed)
                surface = metrics(va,fa,vb,fb,gt,gn,seed+1000,n=8192)
                diagonal=float(np.linalg.norm(np.ptp(vb,axis=0)))
                surface.update(rms_absolute=surface['rms_pct_bboxdiag']*diagonal/100,
                               p95_absolute=surface['p95_pct_bboxdiag']*diagonal/100,
                               direct_mesh_bbox_diagonal=diagonal,seed_direct=seed,seed_incremental=seed+1000,
                               definition='Bidirectional area-sampled distances between the two output triangle surfaces, not distances to target reference.')
                record['mutual_surface_distance'] = surface
            else: record['mutual_surface_distance_skipped'] = 'Exact OBJ bytes equal; identical geometric surface'
        else: record['surface_unavailable'] = 'No saved output mesh'
        records.append(record)
        print('POST_ANALYSIS',row['object'],record.get('mesh_byte_equal'),(record.get('trace') or {}).get('postcut_changed_tetrahedra'),flush=True)
    result = dict(records=records,observations=len(records),mesh_byte_differences=sum(r.get('mesh_byte_equal') is False for r in records),
                  canonical_connectivity_differences=sum((r.get('trace') or {}).get('canonical_surface_face_multiset_equal') is False for r in records),
                  postcut_changed_cases=sum((r.get('trace') or {}).get('postcut_changed_tetrahedra',0)>0 for r in records),
                  mutual_surface_measurements=sum('mutual_surface_distance' in r for r in records),
                  source_ledger_sha256=sha(root/'LEDGER.json'),direct_ledger_sha256=sha(motion/'LEDGER.json'),
                  analysis_code_sha256=sha(Path(__file__)),new_extractions=0,network_training_updates=0,
                  note='Independent read-only post-analysis. Cross-cut energies require the same semantic graph and seed constraints. 8192 samples/direction only for different output bytes. No run artifact is altered.')
    out.mkdir(); (out/'ACCUM_CHANGES.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
