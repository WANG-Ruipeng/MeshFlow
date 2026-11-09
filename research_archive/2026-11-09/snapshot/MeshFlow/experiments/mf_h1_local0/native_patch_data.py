"""Input-only edge-connected seat-region patches and balanced mixed-K data.

Only new native_patch_* inputs/preview are written. Historical coordinates,
weights and code are immutable. 'Seat-region' is a geometric/visual description;
there are no semantic part labels. K is face count, never a surface-area claim.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .native_inpainting_data import (
    NativeTrainingStream as _BaseStream,native_collate,with_context,shared_sample_hash,
    verify_source_faces,_original_pre_ot,_record,PROJECT,UIDS,PERMUTATIONS)
from .learning_data import array_hash
from .prompt_dataset import file_sha256,sample_surface_primitive
from .inference_pilot_data import _preserve_cpu_rng
from .nxc112_data import _persist

HISTORY=PROJECT/'outputs/native_inpainting_20260928_233911'
TRAIN_K=(4,8,12)
EVAL_K=(2,4,8,12)
NOISE_SEEDS=(9201,9202,9203,9204)
SELECTION_RULE=("Use original y as vertical. Among source faces with abs(normal_y)>=0.9 and centroid height in [20%,75%] of the complete source bbox, choose maximum-area face, tie by canonical face ID. Grow one edge-connected prefix by selecting frontier face with lexicographically smallest (max absolute vertex-y distance to seed plane, horizontal centroid distance to source bbox center, negative area, canonical face ID). Prefixes K2/4/8/12 are fixed before model evaluation.")


def _topology(face_ids):
    edges=defaultdict(list)
    for i,face in enumerate(np.asarray(face_ids)):
        for a,b in ((0,1),(1,2),(2,0)):
            edges[tuple(sorted((int(face[a]),int(face[b]))))].append(i)
    adjacency=[set() for _ in face_ids]
    for members in edges.values():
        for face in members:adjacency[face].update(other for other in members if other!=face)
    unseen=set(range(len(face_ids)));components=[]
    while unseen:
        seen={min(unseen)};queue=list(seen)
        while queue:
            face=queue.pop()
            for other in adjacency[face]-seen:seen.add(other);queue.append(other)
        unseen-=seen;components.append(sorted(seen))
    report=dict(face_count=len(face_ids),edge_count=len(edges),component_count=len(components),
        component_sizes=sorted([len(c) for c in components],reverse=True),components=components,
        boundary_edges=sum(len(v)==1 for v in edges.values()),nonmanifold_edge_incidence=sum(len(v)>2 for v in edges.values()),
        interpretation='Shared undirected edge incidence only; not a self-intersection or orientation/vertex-manifold certification')
    return report,adjacency,edges


def _source_topology(parent,target):
    """Map real raw vertex/face IDs through unchanged official normalization."""
    from datasets.mesh_dataset import ObjaverseDataset
    if file_sha256(parent['path'])!=parent['sha256']:raise RuntimeError('Original raw parent changed')
    with np.load(parent['path'],allow_pickle=False) as saved:
        raw={key:saved[key].copy() for key in ('vertices','faces','faces_num','uid')}
    dataset=ObjaverseDataset.__new__(ObjaverseDataset)
    dataset.__dict__.update(training=True,use_custom_prior=False,do_dataset_normalize=True,max_face_length=800,
        use_rot_aug=False,use_scale_aug=False,noise_sort='ot',use_permut_aug=False,std=.3762,data=[raw])
    captured=[]
    def capture_sort(vertices,faces,num_tokens=None):
        captured.append(vertices.copy())
        return ObjaverseDataset.sort_vertices_and_faces(dataset,vertices,faces,num_tokens)
    dataset.sort_vertices_and_faces=capture_sort
    dataset.sample_noise=lambda coords:(coords,np.zeros_like(coords))
    dataset[0]
    if len(captured)!=1:raise RuntimeError('Could not capture original normalized vertex identity')
    raw_model_vertices=(captured[0]/dataset.std*2).astype(np.float32)
    original_model_faces=raw_model_vertices[raw['faces']]
    canonical=np.asarray(target,dtype=np.float32).reshape(112,3,3)
    canonical_to_raw=[];canonical_vertex_ids=[];permutations=[]
    for triangle in canonical:
        equal=(original_model_faces[:,PERMUTATIONS,:]==triangle[None,None]).all(axis=(2,3))
        source=np.flatnonzero(equal.any(1))
        if len(source)!=1:raise RuntimeError('Raw-to-archived source face is not exact and unique')
        j=int(source[0]);pi=int(np.flatnonzero(equal[j])[0])
        canonical_to_raw.append(j);permutations.append(PERMUTATIONS[pi].tolist())
        canonical_vertex_ids.append(raw['faces'][j,PERMUTATIONS[pi]])
    canonical_vertex_ids=np.asarray(canonical_vertex_ids,dtype=np.int64)
    if len(set(canonical_to_raw))!=112:raise RuntimeError('Raw/canonical face mapping is not bijective')
    if not np.array_equal(raw_model_vertices[canonical_vertex_ids],canonical):raise RuntimeError('Raw shared vertex map changed source coordinates')
    original,raw_adj,_=_topology(raw['faces'])
    raw_unique,raw_inverse=np.unique(raw['vertices'],axis=0,return_inverse=True)
    raw_exact,_,_=_topology(raw_inverse[raw['faces']])
    canonical_unique,inverse=np.unique(canonical.reshape(-1,3),axis=0,return_inverse=True)
    exact_faces=inverse.reshape(112,3)
    exact,exact_adj,_=_topology(exact_faces)
    mapped,adjacency,edges=_topology(canonical_vertex_ids)
    inverse_face={raw_id:canonical_id for canonical_id,raw_id in enumerate(canonical_to_raw)}
    original_mapped=[{inverse_face[j] for j in raw_adj[raw_id]} for raw_id in canonical_to_raw]
    if adjacency!=original_mapped or adjacency!=exact_adj:raise RuntimeError('Original and exact-coordinate connectivity differ; manual audit required')
    if max(original['component_sizes'])<12:raise RuntimeError('BLOCKED_NO_EDGE_CONNECTED_12_FACE_COMPONENT')
    return dict(raw_model_vertices=raw_model_vertices,canonical_vertex_ids=canonical_vertex_ids,
        exact_coordinate_vertex_ids=exact_faces,canonical_to_raw=np.asarray(canonical_to_raw,dtype=np.int64),
        adjacency=adjacency,edges=edges),dict(status='PASS',raw_vertex_count=len(raw['vertices']),exact_raw_coordinate_count=len(raw_unique),
        exact_canonical_coordinate_count=len(canonical_unique),raw_original_vertex_topology=original,
        raw_exact_coordinate_topology=raw_exact,canonical_exact_coordinate_topology=exact,
        canonical_original_vertex_topology=mapped,original_vs_exact_adjacency_equal=True,
        canonical_to_raw_face_ids=canonical_to_raw,raw_corner_permutations=permutations,
        mapping_all112_exact_unique_bijective=True,coordinate_modifications=0,welding_applied=False)


def _select_patch(target,adjacency):
    triangles=np.asarray(target,dtype=np.float64).reshape(112,3,3)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=-1)/2
    if (area<=0).any():raise RuntimeError('Degenerate source face cannot be patch input')
    normals=cross/(2*area[:,None]);centroids=triangles.mean(1)
    lo=triangles.reshape(-1,3).min(0);hi=triangles.reshape(-1,3).max(0)
    relative_height=(centroids[:,1]-lo[1])/(hi[1]-lo[1])
    candidates=np.flatnonzero((relative_height>=.2)&(relative_height<=.75)&(abs(normals[:,1])>=.9))
    if not len(candidates):raise RuntimeError('BLOCKED_NO_GEOMETRIC_SEAT_REGION_SEED')
    seed=min(candidates,key=lambda i:(-area[i],int(i)));level=float(centroids[seed,1]);xzcenter=(lo+hi)[[0,2]]/2
    def score(i):
        return (float(abs(triangles[i,:,1]-level).max()),float(np.linalg.norm(centroids[i,[0,2]]-xzcenter)),float(-area[i]),int(i))
    selected=[int(seed)];frontier=set(adjacency[seed]);steps=[]
    while len(selected)<12:
        if not frontier:raise RuntimeError('BLOCKED_NO_EDGE_CONNECTED_12_FACE_PATCH')
        face=min(frontier,key=score);frontier.remove(face);selected.append(face)
        steps.append(dict(added_face=face,score=list(score(face)),prefix_count=len(selected)))
        frontier|=adjacency[face]-set(selected)
    return selected,area,dict(rule=SELECTION_RULE,seed_face_id=int(seed),seed_plane_y=level,
        candidate_seed_face_ids=candidates.tolist(),nested_face_order=selected,growth_steps=steps,
        selection_used_model_outputs=False,semantic_labels_available=False,
        descriptor='Input-only seat-region connected patch; larger prefixes include thickness/edge/support surfaces, not a pure planar seat semantic annotation')


def _boundary(selected,topology):
    known=set(selected);vertices=topology['raw_model_vertices'];rows=[]
    for edge,faces in sorted(topology['edges'].items()):
        inside=[f for f in faces if f in known];outside=[f for f in faces if f not in known]
        if inside and outside:
            rows.append((edge,inside,outside))
    return dict(vertex_ids=np.asarray([r[0] for r in rows],dtype=np.int64),
        coordinates=np.asarray([vertices[list(r[0])] for r in rows],dtype=np.float32),
        known_face_ids=[r[1] for r in rows],free_face_ids=[r[2] for r in rows])


def prepare_inputs(output,persist=True):
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('native_patch_'):raise ValueError('New native_patch_* output required')
    if torch.cuda.is_initialized():raise RuntimeError('CPU data preparation must precede CUDA')
    with _preserve_cpu_rng():
        result_path=HISTORY/'results.json';old=json.loads(result_path.read_text(encoding='utf-8-sig'))
        if old['status']!='PASS_EXECUTION' or any(old['arms'][arm]['completed_updates']!=500 for arm in ('Native_correct','Native_independent')):
            raise RuntimeError('Previous fixed same-object run is not complete')
        old_input=old['assets']['inputs'];old_path=Path(old_input['path'])
        if file_sha256(old_path)!=old_input['sha256']:raise RuntimeError('Historical native input archive changed')
        with np.load(old_path,allow_pickle=False) as saved:old_arrays={key:saved[key].copy() for key in saved.files}
        if set(old_arrays)!=set(old_input['arrays']) or any(array_hash(value)!=old_input['arrays'][key]['sha256'] for key,value in old_arrays.items()):
            raise RuntimeError('Archived native input tensor hashes changed')
        arrays={};cases=[];parents=[];topology_records=[];selection_records=[]
        for parent_index,uid in enumerate(UIDS):
            source=next(row['source'] for row in old['assets']['official_pre_OT_capture'] if uid in row['source']['path'])
            parent=dict(uid=uid,parent_uid=uid,path=source['path'],sha256=source['sha256'],faces=112);parents.append(parent)
            target=old_arrays[f'parent{parent_index}_target'].copy();pre,pre_audit=_original_pre_ot(parent,target)
            if not np.array_equal(pre,old_arrays[f'parent{parent_index}_pre_ot']):raise RuntimeError('Official pre-OT source coordinates changed')
            topology,topo_audit=_source_topology(parent,target)
            selected,areas,selection=_select_patch(target,topology['adjacency'])
            topology_records.append(dict(parent_index=parent_index,parent_uid=uid,**topo_audit,pre_OT=pre_audit))
            selection_records.append(dict(parent_index=parent_index,parent_uid=uid,**selection))
            bbox=float(np.linalg.norm(np.ptp(target.reshape(-1,3),axis=0)))
            arrays[f'parent{parent_index}_target']=target;arrays[f'parent{parent_index}_pre_ot']=pre
            arrays[f'parent{parent_index}_source_vertices']=topology['raw_model_vertices']
            arrays[f'parent{parent_index}_source_face_vertex_ids']=topology['canonical_vertex_ids']
            arrays[f'parent{parent_index}_canonical_to_raw_face']=topology['canonical_to_raw']
            for k in EVAL_K:
                prefix=f'parent{parent_index}_K{k}_';ids=selected[:k];C=target[ids].reshape(k,3,3).copy()
                unique,matches=verify_source_faces(target,C,bbox)
                if unique!=ids:raise RuntimeError('Selected patch identity changed')
                free_ids=np.asarray([i for i in range(112) if i not in ids],dtype=np.int64)
                patch_faces=topology['canonical_vertex_ids'][ids]
                patch_topology,_,_=_topology(patch_faces)
                if patch_topology['component_count']!=1:raise RuntimeError('Patch prefix is not genuinely edge-connected')
                boundary=_boundary(ids,topology)
                checks=np.stack([sample_surface_primitive(triangle,3,128,np.random.RandomState(95000+100*parent_index+int(face)),False) for triangle,face in zip(C,ids)])
                for key,value in dict(constraints=C,checks=checks,source_face_ids=np.asarray(ids,dtype=np.int64),free_source_ids=free_ids,
                        free_target=target[free_ids],pre_ot_free=pre[free_ids],boundary_edges=boundary['coordinates'],
                        boundary_vertex_ids=boundary['vertex_ids']).items():arrays[prefix+key]=value.copy()
                cases.append(dict(case_id=f'P{parent_index}_K{k}',parent_index=parent_index,parent_uid=uid,K=k,N=112,N_free=112-k,
                    bbox_diagonal=bbox,source_face_ids=ids,source_face_matches=matches,raw_source_face_ids=topology['canonical_to_raw'][ids].tolist(),
                    area_fraction=float(areas[ids].sum()/areas.sum()),patch_area=float(areas[ids].sum()),full_area=float(areas.sum()),
                    patch_Y_range=[float(C[:,:,1].min()),float(C[:,:,1].max())],edge_connected=True,patch_topology=patch_topology,
                    patch_boundary_edge_count=len(boundary['coordinates']),boundary_known_source_face_ids=boundary['known_face_ids'],
                    boundary_free_source_face_ids=boundary['free_face_ids'],
                    train_context=k in TRAIN_K,out_of_training_K=k==2,context_type='Connected seat-region source-face patch',
                    historical_K2_same_prompt=False,prior_K2_source_ids=old_arrays[f'parent{parent_index}_K2_source_face_ids'].tolist(),
                    constraints_sha256=array_hash(C),free_target_sha256=array_hash(target[free_ids]),target_sha256=array_hash(target),
                    extra_coordinates_or_alignment=False,known_slots_sampling=list(range(k)),semantic_part_annotation=False))
        for seed in NOISE_SEEDS:
            arrays[f'noise_seed{seed}']=old_arrays[f'noise_seed{seed}'].copy()
            if array_hash(arrays[f'noise_seed{seed}'])!=old['assets']['noise_hashes'][str(seed)]:raise RuntimeError('Sampling noise not exact old arrays')
        artifact=_persist(out/'inputs.npz',arrays,persist)
        if persist:
            with np.load(artifact['path'],allow_pickle=False) as saved:loaded={key:saved[key].copy() for key in saved.files}
            if set(loaded)!=set(arrays) or any(array_hash(loaded[key])!=array_hash(value) for key,value in arrays.items()):raise RuntimeError('Input NPZ reload differs')
            arrays=loaded
        bank_cases=[]
        for case in cases:
            p,k=case['parent_index'],case['K'];prefix=f'parent{p}_K{k}_'
            bank_cases.append(dict(**copy.deepcopy(case),constraints=arrays[prefix+'constraints'],checks=arrays[prefix+'checks'],
                full_target=arrays[f'parent{p}_target'],free_target=arrays[prefix+'free_target'],pre_ot_free=arrays[prefix+'pre_ot_free'],
                free_source_ids=arrays[prefix+'free_source_ids'],boundary_edges=arrays[prefix+'boundary_edges'],
                boundary_vertex_ids=arrays[prefix+'boundary_vertex_ids'],source_vertices=arrays[f'parent{p}_source_vertices'],
                source_face_vertex_ids=arrays[f'parent{p}_source_face_vertex_ids']))
        bank=dict(cases=bank_cases,cases_by_key={(c['parent_index'],c['K']):c for c in bank_cases},parents=parents,parent_uids=list(UIDS),
            N=112,training_K=list(TRAIN_K),evaluation_K=list(EVAL_K),noises={seed:torch.from_numpy(arrays[f'noise_seed{seed}'].copy()) for seed in NOISE_SEEDS},
            noise_seeds=list(NOISE_SEEDS),sampling_noises_previously_evaluated=True,prior_results=old)
        audit=dict(status='PASS',source_results=_record(result_path),source_inputs=_record(old_path),inputs=artifact,
            raw_source_topology=topology_records,input_only_patch_selection=selection_records,cases=cases,
            source_face_gate='PASS',patch_nested=True,selection_used_model_outputs=False,coordinates_modified=False,
            noise_hashes={str(seed):array_hash(bank['noises'][seed].numpy()) for seed in NOISE_SEEDS},
            noise_policy='9201..9204 copied byte-exactly from previous native inputs; already evaluated in prior run, not an unseen confirmation set in this experiment',
            train_K=list(TRAIN_K),out_of_training_K=2,old_K2_prompt_comparability='Different C/source faces: only new nested K2 is an out-of-training context; cannot be called the previous K2 prompt',
            training_500_coverage=planned_coverage(500),fresh_official_EMA_initialization_required=True,
            pretrained_endpoint_reuse_for_training=False,model_forwards=0,optimizer_updates=0,OT_calls=0,CUDA_initialized=False)
    return bank,audit


def _batch_plan(index):
    phase=index%3;rotation={k:TRAIN_K[(i+phase)%3] for i,k in enumerate(TRAIN_K)}
    base=[(0,4),(0,4),(0,8),(0,12),(1,4),(1,8),(1,8),(1,12)]
    return [(parent,rotation[k]) for parent,k in base]


def planned_coverage(updates):
    combo=Counter(pair for index in range(updates) for pair in _batch_plan(index));K=Counter();parents=Counter();denominators=Counter()
    for (p,k),count in combo.items():parents[p]+=count;K[k]+=count
    for index in range(updates):denominators[sum((112-k)*9 for _,k in _batch_plan(index))]+=1
    return dict(updates=updates,samples=8*updates,parent_K=[dict(parent=p,K=k,samples=count) for (p,k),count in sorted(combo.items())],
        parents={str(k):v for k,v in parents.items()},K={str(k):v for k,v in K.items()},free_coordinate_denominator_update_counts={str(k):v for k,v in denominators.items()},
        three_update_rule='Each24 examples has four of every (parent,K); each batch4 per parent; K counts3/3/2 rotate',
        final_partial_cycle_updates=updates%3)


class PatchTrainingStream(_BaseStream):
    def next_effective_batch(self,size=8):
        if size!=8:raise ValueError('Fixed effective batch8 required')
        phase=self.batch_index%3;pairs=_batch_plan(self.batch_index)
        samples=[self._sample(*pairs[index]) for index in self.order_rng.permutation(8)]
        self.batch_index+=1
        shared=[s['shared_input_sha256'] for s in samples]
        self.last_audit=dict(batch_index=self.batch_index,cycle_phase=phase,samples=8,shared_input_hashes=shared,
            shared_batch_sha256=hashlib.sha256(''.join(shared).encode()).hexdigest(),
            parent_indices=[s['parent_index'] for s in samples],K=[s['K'] for s in samples],free_face_counts=[112-s['K'] for s in samples],
            effective_free_coordinates=sum((112-s['K'])*9 for s in samples),times=[float(s['t']) for s in samples],
            donor_indices=[s['donor_index'] for s in samples],context_hashes=[s['context_sha256'] for s in samples],
            accidentally_correct_count=sum(s['donor_index']==s['parent_index'] for s in samples),
            donor_rng_seed=self.seed+70000,donor_independent_of_target_stream=True,
            Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],
            paired_noise_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],pre_OT_target_hashes=[array_hash(s['free_target_before_OT']) for s in samples],
            known_slots=[np.flatnonzero(s['known_mask']).tolist() for s in samples],no_full112_OT=True,official_OT_call_count=8)
        if Counter(s['parent_index'] for s in samples)!=Counter({0:4,1:4}):raise RuntimeError('Mixed-K parent imbalance')
        if sorted(Counter(s['K'] for s in samples).values())!=[2,3,3]:raise RuntimeError('Mixed-K count schedule differs')
        return samples


NativeTrainingStream=PatchTrainingStream


def run_cpu_tests(bank):
    from utils.ot_utils import optimal_sum_numpy
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests before CUDA only')
    a=PatchTrainingStream(bank,'Native_correct',10);b=PatchTrainingStream(bank,'Native_independent',10)
    observed=Counter();denominators=[];all_hashes=[];donors=[];calls=0
    for phase in range(3):
        aa=a.next_effective_batch();bb=b.next_effective_batch();calls+=16
        assert a.last_audit['shared_batch_sha256']==b.last_audit['shared_batch_sha256']
        denominators.append(a.last_audit['effective_free_coordinates'])
        for sa,sb in zip(aa,bb):
            observed[(sa['parent_index'],sa['K'])]+=1;all_hashes.append(sa['shared_input_sha256'])
            assert shared_sample_hash(sa)==shared_sample_hash(sb)
            assert sorted(sa['source_face_ids'].tolist())==list(range(112))
            assert sa['known_mask'].sum()==sa['K'] and sa['K'] in TRAIN_K
            assert np.array_equal(sa['xt'][sa['known_mask']],sa['context'][sa['known_mask']])
            assert shared_sample_hash(with_context(sa,1-sa['parent_index']))==sa['shared_input_sha256']
            donors.append(dict(parent=sb['parent_index'],K=sb['K'],donor=sb['donor_index']))
        batch=native_collate(aa)
        assert batch['free_coordinates']==a.last_audit['effective_free_coordinates']
        assert batch['xt'].shape==(8,112,9) and bool(batch['valid_mask'].all())
        first=aa[0];again=optimal_sum_numpy(first['free_target_before_OT'],first['free_gaussian_before_OT'],optimal=True);calls+=1
        assert np.array_equal(again,first['free_noise_after_OT'])
    assert observed==Counter({(p,k):4 for p in range(2) for k in TRAIN_K})
    assert denominators==[7524,7452,7488] and len(set(all_hashes))==24
    saved=copy.deepcopy(a.state_dict());once=a.next_effective_batch();a.load_state_dict(saved);twice=a.next_effective_batch();calls+=16
    assert [s['shared_input_sha256'] for s in once]==[s['shared_input_sha256'] for s in twice]
    for p in range(2):
        cases=[bank['cases_by_key'][(p,k)] for k in EVAL_K]
        assert all(np.array_equal(cases[-1]['constraints'][:c['K']],c['constraints']) for c in cases)
        assert all(c['edge_connected'] for c in cases)
    return dict(status='PASS',shared_samples_checked=24,three_batch_denominators=denominators,
        balanced_24_sample_counts=[dict(parent=p,K=k,count=n) for (p,k),n in sorted(observed.items())],
        safe_stream_restore=True,nested_contexts_exact=True,all_contexts_edge_connected=True,
        donor_draws=donors,total_OT_calls=calls,sample_OT_calls=64,OT_reference_calls=3,
        model_forwards=0,optimizer_updates=0,CUDA_initialized=False,planned_500_coverage=planned_coverage(500))


def plot_patch_preview(bank,destination):
    """Input-only fixed view of both source meshes and every nested prefix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Never overwrite a patch preview')
    fig=plt.figure(figsize=(16,9),dpi=150)
    full=np.concatenate([bank['cases_by_key'][(p,12)]['full_target'].reshape(-1,3) for p in range(2)])
    lo,hi=full.min(0),full.max(0);center=(lo+hi)/2;radius=float((hi-lo).max())*.54
    for p in range(2):
        for col,k in enumerate(EVAL_K):
            case=bank['cases_by_key'][(p,k)];ax=fig.add_subplot(2,4,p*4+col+1,projection='3d')
            T=case['full_target'].reshape(112,3,3)[:,:,[0,2,1]];C=case['constraints'][:,:,[0,2,1]]
            ax.add_collection3d(Poly3DCollection(T,facecolor=(.55,.60,.66,.13),edgecolor=(.35,.4,.45,.25),linewidth=.35))
            ax.add_collection3d(Poly3DCollection(C,facecolor=(.08,.5,.83,.94),edgecolor=(.02,.13,.2,1),linewidth=.85))
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[2]-radius,center[2]+radius);ax.set_zlim(center[1]-radius,center[1]+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
            suffix='evaluation only' if k==2 else 'training context'
            ax.set_title(f'P{p} | K={k} | area {case["area_fraction"]:.2%}\n{suffix}',fontsize=11)
    fig.suptitle('INPUT ONLY: nested edge-connected seat-region source patches\nBlue = exact known faces; gray = complete source. No semantic annotations or generated meshes.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination);plt.close(fig)
    return dict(**_record(destination),selection='Both input parents and every preregistered K; fixed camera/shared bounds',model_outputs_shown=False)
