"""Bounded read-only replay of candidate scores and pre-mincut graph assembly.
No graph cut, mesh extraction, fitting, optimization, or score repair is run.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib,warnings,datetime
from pathlib import Path
import numpy as np
import torch
import networkx as nx
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
source=BASE/'ponq_gate1_synthetic_20260926_01/vendor/PoNQ_to_mesh.py'
sys.path.insert(0,str(source.parent))
from PoNQ_to_mesh import MeshFromPoNQ
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x);finite=x[np.isfinite(x)]
    return dict(count=int(x.size),nonfinite=int(np.sum(~np.isfinite(x))),nan=int(np.isnan(x).sum()),
                negative=int(np.sum(x<0)),minimum_finite=float(finite.min()) if finite.size else None,
                maximum_finite=float(finite.max()) if finite.size else None)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
cases=[
 ('synthetic','box','clean128'),('synthetic','box','clean256'),
 ('real','loudspeaker_local_1','clean128'),('real','loudspeaker_local_1','clean256'),
 ('real','loudspeaker_local_2','clean128'),('real','loudspeaker_local_2','clean256'),
 ('isolation','box','roundtrip')]
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,
    method='Reconstruct original MeshFromPoNQ with compute_mincut=False; replay identical relevant-edge filtering and DiGraph capacity aggregation only.',
    forbidden_actions=['no score clipping','no Q modification','no normal modification','no mincut invocation','no fit or optimization','no write to original runs'],
    source_sha256=sha(source),code_sha256=sha(__file__),python=sys.executable,
    versions=dict(numpy=np.__version__,torch=torch.__version__,networkx=nx.__version__))
dump(OUT/'PROTOCOL.json',protocol)
rows=[]
for study,obj,state in cases:
    root=BASE/f'ponq_gate1_{study}_20260926_01'
    rd=root/'runs'/f'{obj}__{state}'
    assert sha(root/'vendor/PoNQ_to_mesh.py')==sha(source)
    with np.load(rd/'parameters.npz',allow_pickle=False) as z:
        params={k:z[k].copy() for k in z.files}
    original=json.loads((rd/'result.json').read_text())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sc=MeshFromPoNQ(torch.from_numpy(params['v']),torch.from_numpy(params['eigs']),
                       torch.from_numpy(params['Q']),torch.from_numpy(params['n']),
                       grid_scale=32,correct_tet_color=True,compute_mincut=False)
    # The following graph assembly is copied from the fixed extractor lines 159-178.
    colors=sc.tet_colors.astype(int)
    edges=sc.triangle_faces_neighbors.copy()
    non_inf=(edges!=-1).all(-1)
    min_in=np.arange(*colors.shape)[colors==4].min()
    min_out=np.arange(*colors.shape)[colors==0].min()
    edges[colors[edges]==4]=min_in
    edges[colors[edges]==0]=min_out
    relevant=(edges[:,0]!=edges[:,1])*non_inf
    graph_edges=np.fliplr(edges[relevant])
    relevant_scores=sc.final_scores[relevant]
    network=nx.DiGraph()
    network.add_nodes_from(np.unique(graph_edges))
    for edge,score in zip(graph_edges,relevant_scores):
        u,v=map(int,edge)
        if network.has_edge(u,v):network[u][v]['capacity']+=score
        else:network.add_edge(u,v,capacity=score)
    capacities=np.array([data['capacity'] for _,_,data in network.edges(data=True)])
    # Exactly mirror the normal normalization/dot operations before arccos.
    vn=sc.normals[sc.triangle_faces].cpu().detach().numpy()
    vertex_norms=np.sqrt((vn**2).sum(-1))
    vn=vn/vertex_norms[...,None]
    tf=np.cross(sc.points[sc.triangle_faces[:,1]]-sc.points[sc.triangle_faces[:,0]],
                sc.points[sc.triangle_faces[:,2]]-sc.points[sc.triangle_faces[:,0]])
    face_norms=np.sqrt((tf**2).sum(-1,keepdims=True))
    tf=tf/face_norms
    dots=(tf[:,None,:]*vn).sum(-1)
    outside=(dots < -1)|(dots > 1)
    with np.errstate(invalid='ignore'):
        normal_term=(np.arccos(dots).sum(-1)/1.5)**2
    qterm=sc.get_faces_score(sc.triangle_faces)*32**2
    finite_dots=dots[np.isfinite(dots)]
    face_nonfinite=~np.isfinite(sc.final_scores)
    face_negative=sc.final_scores<0
    saved_keys=['status','nonfinite_cut_weights','negative_cut_weights','minimum_cut_weight',
                'closed_oriented_2manifold','betti_match','mesh_sha256']
    row=dict(study=study,object=obj,state=state,input_sha256=sha(rd/'parameters.npz'),
             saved_result={k:original.get(k) for k in saved_keys},
             original_extractor_sha256=sha(root/'vendor/PoNQ_to_mesh.py'),
             warnings=[str(w.message) for w in caught],
             candidate_scores=stats(sc.final_scores),relevant_directed_face_scores=stats(relevant_scores),
             aggregated_graph_capacities=stats(capacities),
             discarded_candidate_nonfinite=int(np.sum(face_nonfinite & ~relevant)),
             discarded_candidate_negative=int(np.sum(face_negative & ~relevant)),
             init_tet_label_counts={str(int(label)):int(np.sum(colors==label)) for label in np.unique(colors)},
             candidate_count_matches_saved=stats(sc.final_scores)['nonfinite']==original['nonfinite_cut_weights'] and stats(sc.final_scores)['negative']==original['negative_cut_weights'],
             quadric_term=stats(qterm),normal_term=stats(normal_term),
             candidate_nonfinite_exactly_explained_by_normal_term=bool(np.array_equal(face_nonfinite,~np.isfinite(normal_term))),
             arccos_dots=dict(minimum=float(finite_dots.min()),maximum=float(finite_dots.max()),
                 out_of_range_scalar_count=int(outside.sum()),out_of_range_face_count=int(outside.any(-1).sum()),
                 max_excess_above_one=float(max(0.,finite_dots.max()-1)),
                 max_excess_below_minus_one=float(max(0.,-1-finite_dots.min())),
                 zero_vertex_normal_lengths=int((vertex_norms==0).sum()),
                 zero_face_normal_lengths=int((face_norms==0).sum())),
             Q_eigenvalues=stats(np.linalg.eigvalsh(params['Q'])),
             A_eigenvalues=stats(np.linalg.eigvalsh(params['Q'][:,:3,:3])),
             base_normal_lengths=stats(np.linalg.norm(params['n'],axis=1)),
             svd_rank_counts={str(rank):int(np.sum(np.sum(params['eigs']>.05*params['eigs'][:,:1],axis=1)==rank)) for rank in [0,1,2,3]},
             graph_cut_reexecuted=False,any_score_repaired=False)
    assert row['candidate_count_matches_saved']
    rows.append(row)
    print(json.dumps(dict(study=study,object=obj,state=state,candidate=row['candidate_scores'],
              relevant=row['relevant_directed_face_scores'],graph=row['aggregated_graph_capacities'],
              arccos=row['arccos_dots']),allow_nan=False),flush=True)
dump(OUT/'NUMERICAL_DIAGNOSIS.json',rows)
dump(OUT/'MANIFEST.json',dict(files=[dict(path=p.name,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='MANIFEST.json']))
