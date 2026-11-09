"""Read-only derived mechanism analysis of archived clay factorial artifacts.

Reconstructs score components from saved parameters and candidate-face arrays.
No extraction, graph-cut, sampling, fitting, model inference, or input edits.
Only MECHANISM.json beside this script is created, never overwritten.
"""
import hashlib,json,datetime
from pathlib import Path
from collections import Counter
import numpy as np
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
NEW=HERE.parent
PROTOCOL=json.loads((NEW/'FROZEN_PROTOCOL.json').read_text())
PARENT=Path(PROTOCOL['config']['parent_run'])
OUTPUT=HERE/'MECHANISM.json'
if OUTPUT.exists(): raise RuntimeError('Refusing to overwrite MECHANISM.json')
ARMS=['freeze','q_only','n_only','transport']
OBJECTS=['sphere','thin_u','bottle_local_1']
FILES={}
def sha(path):
    path=Path(path)
    value=hashlib.sha256(path.read_bytes()).hexdigest()
    FILES[str(path)]=value
    return value

def read(path):
    sha(path)
    return json.loads(Path(path).read_text())

def arrays(path):
    sha(path)
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}

def arrhash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def folder(obj,frame,arm):
    return (NEW if arm in ['q_only','n_only'] else PARENT)/'runs'/f'{obj}_frame{frame:02d}__{arm}'

def array_difference(first,second):
    delta=second-first
    return dict(bitwise_equal=bool(np.array_equal(first,second)),
                changed_count=int(np.count_nonzero(first!=second)),
                maximum_absolute_difference=float(abs(delta).max()) if delta.size else 0.,
                l1_difference=float(abs(delta).sum()),
                first_sha256=arrhash(first),second_sha256=arrhash(second))

def label_difference(first,second,tetrahedra):
    ids=np.flatnonzero(first!=second)
    return dict(changed_count=int(len(ids)),tetrahedron_ids=ids.tolist(),
                first_labels=first[ids].tolist(),second_labels=second[ids].tolist(),
                tetrahedron_vertex_ids=tetrahedra[ids].tolist())

def id_difference(first,second):
    one,two=set(map(int,first)),set(map(int,second))
    return dict(identical=one==two,removed=sorted(one-two),added=sorted(two-one))

def graph_difference(first,second):
    one={tuple(map(int,e)):float(c) for e,c in zip(first['graph_edges'],first['graph_capacities'])}
    two={tuple(map(int,e)):float(c) for e,c in zip(second['graph_edges'],second['graph_capacities'])}
    common=sorted(one.keys()&two.keys())
    delta=np.array([two[e]-one[e] for e in common])
    same_edges=one.keys()==two.keys()
    return dict(edge_arrays_identical=bool(np.array_equal(first['graph_edges'],second['graph_edges'])),
        edge_sets_identical=bool(same_edges),first_edge_count=len(one),second_edge_count=len(two),
        edges_removed=[list(e) for e in sorted(one.keys()-two.keys())],
        edges_added=[list(e) for e in sorted(two.keys()-one.keys())],
        common_edge_count=len(common),common_edge_capacities_changed=int(np.count_nonzero(delta)),
        common_edge_maximum_absolute_capacity_change=float(abs(delta).max()) if len(delta) else 0.,
        aligned_capacity_l1_change=float(abs(delta).sum()) if same_edges else None,
        terminals_identical=bool(np.array_equal(first['graph_terminals'],second['graph_terminals'])),
        capacity_comparison_is_not_an_energy_quality_ranking=True,
        warning='Changing seeds changes contraction and the optimization problem; a capacity difference alone is not a stability certificate.')

def score_parts(params,trace):
    # Match vendor ordering and float64 arithmetic. Fixed anchor quadrics/eigs
    # are appended, then each full Q is normalized by its largest A spectrum.
    Q=np.concatenate([params['Q'],np.broadcast_to(np.eye(4),(8,4,4))])
    eig=np.concatenate([params['eigs'],np.ones((8,3))])
    qt=torch.from_numpy(Q/eig[:,0,None,None])
    points=torch.from_numpy(trace['delaunay_points'])
    faces=trace['candidate_faces']
    def score(quadric,point):
        homogeneous=torch.cat((point,torch.ones_like(point[...,:1])),-1)
        return (homogeneous[...,None,:]@quadric@homogeneous[...,None]).squeeze(-1).squeeze(-1)
    parts=[]
    for i,j,k in [[0,1,2],[1,2,0],[2,1,0]]:
        parts.append(score(qt[faces[:,i]],points[faces[:,j]])+score(qt[faces[:,i]],points[faces[:,k]]))
    quadric_term=torch.row_stack(parts).sum(0).numpy()*32**2
    dots=trace['normal_dot_products_unclipped']
    assert np.isfinite(dots).all() and np.max(np.maximum(abs(dots)-1,0))<=64*np.finfo(float).eps
    normal_term=(np.arccos(np.clip(dots,-1,1)).sum(-1)/1.5)**2
    reconstruction_error=float(np.max(abs(quadric_term+normal_term-trace['candidate_scores'])))
    assert reconstruction_error==0.,reconstruction_error
    return dict(Q=quadric_term,N=normal_term,reconstruction_max_abs_error=reconstruction_error)

old=read(PARENT/'LEDGER.json');new=read(NEW/'LEDGER.json')
index={(r['source_object'],r['frame'],r['state']):r for r in old+new}
cache={};invariants=[]
for obj in OBJECTS:
    for frame in range(8):
        data={arm:arrays(folder(obj,frame,arm)/'extractor_trace.npz') for arm in ARMS}
        stages=['initial_labels','corrected_labels','void_labels','precut_labels']
        check=dict(object=obj,frame=frame,
            input_sites_equal=all(np.array_equal(data[arm]['delaunay_points'],data['freeze']['delaunay_points']) for arm in ARMS),
            delaunay_arrays_equal=all(np.array_equal(data[arm]['delaunay_simplices'],data['freeze']['delaunay_simplices']) for arm in ARMS),
            final_vertex_subsets_equal=all(np.array_equal(data[arm]['surface_input_vertex_ids'],data['freeze']['surface_input_vertex_ids']) for arm in ARMS),
            q_only_preserves_freeze_all_precut_stages=all(np.array_equal(data['q_only'][key],data['freeze'][key]) for key in stages),
            n_only_matches_transport_all_precut_stages=all(np.array_equal(data['n_only'][key],data['transport'][key]) for key in stages))
        assert all(value for key,value in check.items() if key not in ['object','frame']),check
        invariants.append(check)
        if obj=='thin_u' and frame in [6,7]:cache[frame]=data

hash_pairs=[]
for j,first in enumerate(ARMS+['refit']):
    for second in (ARMS+['refit'])[j+1:]:
        matches=[dict(object=obj,frame=frame) for obj in OBJECTS for frame in range(8)
                 if index[obj,frame,first]['mesh_sha256']==index[obj,frame,second]['mesh_sha256']]
        hash_pairs.append(dict(first=first,second=second,identical_mesh_count=len(matches),
            total_correlated_frame_pairs=24,by_path=dict(Counter(z['object'] for z in matches)),matching_frames=matches))

cases=[]
for frame,traces in cache.items():
    arm_records={};parts={}
    for arm in ARMS:
        record=index['thin_u',frame,arm];trace=traces[arm]
        parameters=arrays(folder('thin_u',frame,arm)/'parameters.npz')
        parts[arm]=score_parts(parameters,trace)
        arm_records[arm]=dict(
            run_path=str(folder('thin_u',frame,arm)),
            rms_pct_bboxdiag=record['rms_pct_bboxdiag'],faces=record['faces'],
            betti=[record['beta0'],record['beta1'],record['beta2']],
            strict_noncycle_vertex_count=record['strict_noncycle_vertex_count'],
            closed_oriented_2manifold=record['closed_oriented_2manifold'],betti_match=record['betti_match'],
            exact_intersection_pairs=record['exact_intersection_pairs'],exact_degenerate_faces=record['exact_degenerate_faces'],
            free_count=len(trace['free_tetrahedron_ids']),inside_count=len(trace['inside_seed_ids']),outside_count=len(trace['outside_seed_ids']),
            inside_seed_ids=trace['inside_seed_ids'].tolist(),outside_seed_ids=trace['outside_seed_ids'].tolist(),
            graph_terminals=trace['graph_terminals'].tolist(),graph_edge_count=len(trace['graph_edges']),
            graph_capacity_minimum=float(trace['graph_capacities'].min()),
            graph_capacity_nonfinite_count=int(np.sum(~np.isfinite(trace['graph_capacities']))),
            graph_capacity_negative_count=int(np.sum(trace['graph_capacities']<0)),
            stage_label_counts={key:{str(int(k)):int(v) for k,v in zip(*np.unique(trace[key],return_counts=True))}
                                for key in ['initial_labels','corrected_labels','void_labels','precut_labels','postcut_labels']},
            score_reconstruction_max_abs_error=parts[arm]['reconstruction_max_abs_error'],
            Qterm_sha256=arrhash(parts[arm]['Q']),normal_cost_sha256=arrhash(parts[arm]['N']))
    comparisons=[]
    for first,second in [('freeze','q_only'),('freeze','n_only'),('freeze','transport'),('q_only','transport'),('n_only','transport')]:
        one,two=traces[first],traces[second]
        comparisons.append(dict(first=first,second=second,
            label_stages={key:label_difference(one[key],two[key],one['delaunay_simplices'])
                          for key in ['initial_labels','corrected_labels','void_labels','precut_labels','postcut_labels']},
            inside_seeds=id_difference(one['inside_seed_ids'],two['inside_seed_ids']),
            outside_seeds=id_difference(one['outside_seed_ids'],two['outside_seed_ids']),
            Qterm=array_difference(parts[first]['Q'],parts[second]['Q']),
            normal_cost=array_difference(parts[first]['N'],parts[second]['N']),
            graph=graph_difference(one,two)))
    cases.append(dict(object='thin_u',frame=frame,arms=arm_records,comparisons=comparisons))

result=dict(version='CLAY_FACTORIAL_READONLY_MECHANISM_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execution=dict(new_extractions=0,new_graph_cuts=0,new_samples=0,new_fitting=0,only_derived_arithmetic=True),
    arithmetic=dict(numpy=np.__version__,torch=torch.__version__,torch_threads=1,dtype='float64',
        Qterm='Original six ordered endpoint full-Q evaluations per candidate face, using Q/eigs[:,0], times grid_scale^2=1024',
        normal_cost='(sum(arccos(saved normalized dot, clipped only within the same 64eps guard))/1.5)^2',
        reconstruction='Both terms sum bitwise to the saved candidate_scores in all eight focused arms',
        graph='Saved final contracted directed edges/capacities only; no min-cut is run',
        energy_ranking='No comparison of optimum energy values across different objectives'),
    interpretation_limits=['24 correlated frames from three paths, not 24 independent objects',
        'Q includes A,g,c and A-spectrum normalization, not an A-only intervention',
        'Normal intervention changes both seeds and normal cut costs; these pathways have not been separately intervened on',
        'No learned model or process-feedback contribution is evaluated'],
    cross_arm_invariants=invariants,pairwise_mesh_hashes=hash_pairs,thin_u_cases=cases,
    source_files=[dict(path=path,sha256=value) for path,value in sorted(FILES.items())],script_sha256=sha(Path(__file__)))
OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUTPUT),bytes=OUTPUT.stat().st_size,focused_cases=len(cases),frame_invariants=len(invariants),input_files=len(FILES))))