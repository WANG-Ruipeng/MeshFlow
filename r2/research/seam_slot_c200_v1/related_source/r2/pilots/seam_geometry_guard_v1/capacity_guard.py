"""CPU-only C-edge capacity guard on frozen single-D0 seam proposals.

The original GEO_C stays untouched.  This independent version preserves its
order, radius, representative choice, and exact cleanup.  Every proposed
non-hard-anchor merge is rejected if it increases any retained C-edge
incidence above two (or above an already-existing higher incidence).
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_n] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C
from r2.pilots.seam_slot_c200_v1 import generate as G
from r2.pilots.seam_slot_c200_v1 import postprocess as P
from meshflow_control import geo_c as O, geo_io
from meshflow_control.source_winding import export_source_winding

OUT = C.ROOT/'experiments/r2_seam_geometry_guard_v1/capacity'
OLD = C.OUT
ARM = 'SEAM_SLOT_C200'
VERSION = 'geo_c_incremental_C_edge_capacity_guard_v1'


def cleanup(tri, k):
    keep = np.ones(len(tri), dtype=bool)
    reasons = np.full(len(tri), 'kept', dtype='U32')
    dup = np.full(len(tri), -1, dtype=np.int64)
    seen = {}
    for i in range(k): seen.setdefault(O._face_key(tri[i]), i)
    for i in range(k, len(tri)):
        reason = O._degenerate(tri[i]); key = O._face_key(tri[i])
        if reason is None and key in seen:
            dup[i] = seen[key]
            reason = 'duplicate_C' if seen[key] < k else 'duplicate_free'
        if reason is not None:
            keep[i] = False; reasons[i] = reason
        else: seen[key] = i
    return keep, reasons, dup


def c_edges(tri, k):
    counts = Counter()
    for face in tri[:k]:
        ids = [O._bits(x) for x in face]
        for a,b in ((0,1),(1,2),(2,0)):
            counts[tuple(sorted((ids[a], ids[b])))] += 1
    return counts


def capacity_state(tri, k, ced):
    keep, reasons, dup = cleanup(tri, k)
    incident = {e: [] for e in ced}
    for i in np.flatnonzero(keep):
        ids = [O._bits(x) for x in tri[i]]
        # A repeated-edge degenerate C face counts once as an incident face.
        for edge in {tuple(sorted((ids[a],ids[b]))) for a,b in ((0,1),(1,2),(2,0))}:
            if edge in incident: incident[edge].append(int(i))
    return dict(incident=incident, keep=keep, reasons=reasons, duplicate_of=dup)


def process(raw, k, hard, guard=True):
    """Deterministic copy of original GEO_C with an incremental merge guard."""
    started = time.perf_counter()
    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[1:] != (3,3) or not np.isfinite(raw).all():
        raise ValueError('Expected finite FP32 triangle soup')
    if not 0 < k < len(raw) or hard.shape != raw.shape[:2]: raise ValueError('Invalid K/anchor mask')
    original = np.ascontiguousarray(raw).copy(); original_hash = O._hash(raw)
    points = original.reshape(-1,3); corrected = points.copy(); n_corners = len(points); known_count = 3*k
    target = np.arange(n_corners, dtype=np.int64); kind = np.full(n_corners, 'known', dtype='U32')
    anchor_candidates = np.zeros(n_corners, dtype=np.int64); radius2 = O.RADIUS**2
    ced = c_edges(raw,k); edge_id = {e:i for i,e in enumerate(sorted(ced))}; logs = []
    state = capacity_state(original,k,ced)
    initial = {edge_id[e]:len(v) for e,v in state['incident'].items()}
    anchors_by_numeric = {}
    for i in range(known_count): anchors_by_numeric.setdefault(O._numeric(points[i]),[]).append(i)
    anchor_rows = [min(ids,key=lambda i:(O._bits(points[i]),i)) for ids in anchors_by_numeric.values()]
    anchor_rows.sort(key=lambda i:O._sort_key(points[i])); anchor_rows = np.asarray(anchor_rows,dtype=np.int64)
    anchor_points = points[anchor_rows].astype(np.float64)

    def assignment(i,j,mode):
        nonlocal state
        exact = G.same(corrected[i],points[j]); locked = bool(hard.reshape(-1)[i])
        if locked and not exact: raise ValueError('Hard seam anchor would move')
        prior = corrected[i].copy(); corrected[i] = points[j]
        trial = state if exact else capacity_state(corrected.reshape(raw.shape),k,ced)
        changes=[]; blocked=[]
        for edge in sorted(ced):
            before=state['incident'][edge]; after=trial['incident'][edge]
            if before!=after:
                item=dict(C_edge_id=edge_id[edge],C_edge_bits=edge,C_incidence=ced[edge],
                          before_source_faces=before,after_source_faces=after,before=len(before),after=len(after))
                changes.append(item)
                if len(after)>max(2,len(before)): blocked.append(item)
        rejected=bool(guard and not locked and blocked)
        logs.append(dict(sequence=len(logs),source_face_id=i//3,source_corner_id=i%3,
            source_flat_corner_id=i,target_input_corner_id=j,target_source_face_id=j//3,
            target_source_corner_id=j%3,mode=mode,hard_anchor=locked,exact_noop=exact,
            displacement=float(np.linalg.norm(points[i].astype(float)-points[j].astype(float))),
            decision='REJECT' if rejected else 'ACCEPT',
            reason='NEW_C_EDGE_OVER_CAPACITY' if rejected else 'HARD_ANCHOR_NOOP' if locked else 'PASS',
            changed_C_edges=changes,over_capacity_edges=blocked,
            newly_deleted_source_faces=np.flatnonzero(state['keep']&~trial['keep']).tolist(),
            revived_source_faces=np.flatnonzero(~state['keep']&trial['keep']).tolist()))
        if rejected:
            corrected[i]=prior;kind[i]='capacity_rejected_unchanged';return False
        state=trial;target[i]=j;kind[i]=mode;return True

    free_for_clustering=[]
    for i in range(known_count,n_corners):
        delta=anchor_points-points[i].astype(np.float64); d2=np.einsum('ij,ij->i',delta,delta)
        hits=np.flatnonzero(d2<=radius2);anchor_candidates[i]=len(hits)
        if len(hits)==1: assignment(i,int(anchor_rows[hits[0]]),'unique_C_anchor')
        elif len(hits)>1: kind[i]='ambiguous_C_unchanged'
        else: free_for_clustering.append(i)
    representatives=[]
    for i in sorted(free_for_clustering,key=lambda j:(O._sort_key(points[j]),j)):
        if representatives:
            rp=points[np.asarray(representatives)].astype(np.float64);delta=rp-points[i].astype(np.float64)
            distances=np.einsum('ij,ij->i',delta,delta);hits=np.flatnonzero(distances<=radius2)
        else: hits=[]
        if len(hits):
            position=min(hits,key=lambda h:(float(distances[h]),int(h)))
            assignment(i,representatives[int(position)],'free_to_representative')
        else: representatives.append(i);kind[i]='free_representative'
    corrected=corrected.reshape(original.shape)
    vector=corrected.astype(np.float64)-original.astype(np.float64);d2=np.einsum('nci,nci->nc',vector,vector)
    if np.any(d2>radius2) or not G.same(corrected[:k],original[:k]): raise AssertionError('Displacement/C contract')
    keep,reasons,dup=cleanup(corrected,k);source_faces=np.flatnonzero(keep).astype(np.int64);output=corrected[keep].copy()
    vertices=[];vertex_sources=[];vertex_index={};faces=np.empty((len(output),3),dtype=np.int64)
    for row,source in enumerate(source_faces):
        for c in range(3):
            point=output[row,c];key=O._bits(point)
            if key not in vertex_index:
                vertex_index[key]=len(vertices);vertices.append(point.copy());vertex_sources.append(int(target[3*source+c]))
            faces[row,c]=vertex_index[key]
    vertices=np.asarray(vertices,dtype=np.float32).reshape(-1,3)
    corner_vid=np.asarray([vertex_index.get(O._bits(p),-1) for p in corrected.reshape(-1,3)],dtype=np.int64).reshape(-1,3)
    records=[dict(source_face_id=i,known=i<k,kept=bool(keep[i]),reason=str(reasons[i]),duplicate_of_source_face=int(dup[i])) for i in range(len(raw))]
    if not G.same(vertices[faces],output) or O._hash(raw)!=original_hash: raise AssertionError('Roundtrip/input contract')
    moved=np.any(corrected.view(np.uint32)!=original.view(np.uint32),axis=2)
    stats=dict(status='PASS',input_faces=len(raw),output_faces=len(output),known_faces=k,
        input_triangle_soup_vertices=n_corners,output_indexed_vertices=len(vertices),deleted_free_faces=int((~keep).sum()),
        deleted_by_reason=dict(Counter(str(r) for r in reasons[~keep])),corner_status_counts=dict(Counter(str(s) for s in kind[known_count:])),
        moved_free_corners=int(moved[k:].sum()),max_corner_displacement=float(np.sqrt(d2.max())),radius_model=O.RADIUS,
        known_degenerate_faces_retained=sum(O._degenerate(t) is not None for t in corrected[:k]),
        known_all_bytes_unchanged=True,indexed_roundtrip_bytes_unchanged=True,original_input_unchanged=True,
        all_corner_displacements_within_radius=True,input_sha256=original_hash,output_sha256=O._hash(output),
        vertices_sha256=O._hash(vertices),faces_sha256=O._hash(faces),seconds=time.perf_counter()-started,model_calls=0,
        guard=guard,rejected_merges=sum(x['decision']=='REJECT' for x in logs),
        initial_C_edge_over_capacity={str(e):n for e,n in initial.items() if n>2},
        final_C_edge_over_capacity={str(edge_id[e]):len(v) for e,v in state['incident'].items() if len(v)>2})
    result=dict(output=output,vertices=vertices,faces=faces,source_face_ids=source_faces,
        source_corner_ids=np.tile(np.arange(3),(len(output),1)),vertex_source_input_corner_ids=np.asarray(vertex_sources,dtype=np.int64),
        corner_displacement=np.sqrt(d2),corner_displacement_vector=vector,corner_target_input_ids=target.reshape(-1,3),
        corner_vertex_ids=corner_vid,corner_status=kind.reshape(-1,3),corner_C_candidate_count=anchor_candidates.reshape(-1,3),
        source_face_kept=keep,face_records=records,stats=stats)
    return result,logs


def finish(raw,pre,source,guard):
    result,logs=process(pre,int(raw['known_faces']),raw['raw_anchor_ids']>=0,guard)
    canonical=geo_io._validate_result(pre,int(raw['known_faces']),result)
    oriented=export_source_winding(canonical,source);post=oriented['oriented_arrays']
    extra,anchors=P.validate_anchor_indices(post,raw);final=dict(post,**extra)
    final.update(identity_domain=np.asarray('generated_RAW_slot_not_GT'),occupied=raw['occupied'],
        occupancy_logits=raw['occupancy_logits'],edge_records=raw['edge_records'],C_vertices=raw['C_vertices'],
        C_faces_original=raw['C_faces_original'],C_faces_model=raw['C_faces_model'])
    return canonical,final,result,oriented,anchors,logs


def register():
    from r2.pilots.seam_slot_c200_v1 import evaluate as E
    m=C.read(OLD/'eval_manifest.json'); protected={str(OLD/'eval_manifest.json'):C.sha(OLD/'eval_manifest.json')}
    sources=E.source_hashes()
    for mod in (O,geo_io,P,G): sources[str(Path(mod.__file__))]=C.sha(mod.__file__)
    for q in m['requests']:
        base=OLD/'evaluation'/ARM/q['request_id']
        for rel in ('raw/raw.npz','raw/source_condition.npz','post/pre_geo.npz','post/post_canonical.npz','post/post.npz','post/geometry.json'):
            p=base/rel;protected[str(p)]=C.sha(p)
        for p in (q['inference_npz'],q['metadata_npz'],q['query_descriptor']['path']):protected[str(C.resolve(p))]=C.sha(p)
        p=OLD/'evaluation/scores'/ARM/q['request_id']/'POST/score.json';protected[str(p)]=C.sha(p)
    value=dict(version=VERSION,status='FROZEN',inputs=16,model_calls=0,training_updates=0,new_queries=0,
        original_radius=O.RADIUS,threshold_sweep=False,source_files=sources,protected=protected,
        contract=['Same sixteen stored restored D0 proposals and predicted occupancy; no generation or D0 forward.',
        'All hard anchor identities exist in the complete pre_geo topology before any non-hard merge; never reject/move a hard anchor.',
        'Same corner iteration and representative order, same exact cleanup; no global prohibition on bulk sharing a C vertex.',
        'Trial retained C-edge incidence greater than max(2,current incidence) rejects the non-hard merge; rejected corners remain fixed.',
        'Check every C edge, including interior C edges. Preserve pre-existing over-capacity; no deletion to repair capacity.',
        'All unchanged original quality, topology and query scorers; candidate is not a manifold or geometric validity guarantee.'],
        code_sha256=C.sha(__file__),scope='Four seen TRAIN C, sixteen requests, CPU only; not DEV or generalization.')
    if (OUT/'registration.json').exists():
        prev=C.read(OUT/'registration.json')
        if prev!=value: raise ValueError('Registration changed')
    else:
        C.resources(OUT/'resources_before.json');C.write(OUT/'registration.json',value)
    return value,m


def verify_registration():
    from r2.pilots.seam_slot_c200_v1 import evaluate as E
    reg=C.read(OUT/'registration.json');E.verify(reg['protected']);E.verify(reg['source_files'])
    if C.sha(__file__)!=reg['code_sha256']:raise ValueError('Frozen candidate code changed')
    return reg,C.read(OLD/'eval_manifest.json')


def replay():
    reg,m=verify_registration();rows=[]
    for q in m['requests']:
        base=OLD/'evaluation'/ARM/q['request_id'];raw=C.arrays(base/'raw/raw.npz');source=C.arrays(base/'raw/source_condition.npz')
        pre=C.arrays(base/'post/pre_geo.npz')['output'];canonical,final,result,*_=finish(raw,pre,source,False)
        expected=O.process(pre,int(q['K']))
        checks={key:G.same(v,expected[key]) for key,v in result.items() if isinstance(v,np.ndarray)}
        checks['face_records']=result['face_records']==expected['face_records']
        for name,got in (('post_canonical',canonical),('post',final)):
            saved=C.arrays(base/'post'/f'{name}.npz');checks[name+'_keys']=got.keys()==saved.keys()
            checks.update({name+':'+key:G.same(v,saved[key]) for key,v in got.items()})
        if not all(checks.values()):raise AssertionError((q['request_id'],[k for k,v in checks.items() if not v]))
        rows.append(dict(request_id=q['request_id'],status='PASS',array_and_record_checks=len(checks),guard=False))
    value=dict(status='PASS',requests=len(rows),rows=rows,all_arrays_byte_exact=True,model_calls=0)
    C.write(OUT/'baseline_replay.json',value);return value


def selftest():
    # Occupied C edge accepts a shared C vertex but cannot accept a second free face.
    c=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    owner=np.array([[1,0,0],[0,0,0],[0,0,1]],np.float32)
    extra=np.array([[1,.01,0],[0,.01,0],[.5,-1,0]],np.float32)
    raw=np.stack([c,owner,extra]);hard=np.zeros((3,3),bool);hard[0]=True;hard[1,:2]=True
    result,logs=process(raw,1,hard,True)
    assert sum(x['decision']=='REJECT' for x in logs)==1
    assert G.same(result['output'][1,:2],owner[:2])
    assert any(x['decision']=='ACCEPT' and x['source_face_id']==2 for x in logs)
    before=capacity_state(raw,1,c_edges(raw,1));after=capacity_state(result['output'],1,c_edges(raw,1))
    assert max(map(len,after['incident'].values()))==2
    # A source already over capacity stays over capacity; no face repair/deletion.
    exact=np.stack([c,owner,np.array([[1,0,0],[0,0,0],[.5,-1,0]],np.float32)])
    r,_=process(exact,1,hard,True);assert r['stats']['initial_C_edge_over_capacity']
    assert r['stats']['final_C_edge_over_capacity'] and len(r['output'])==3
    value=dict(status='PASS',cases=2,assertions=['Occupied edge blocks added face but allows single C-vertex sharing.',
        'All hard anchors stay exact.', 'Pre-existing over-capacity is preserved without deleting faces.'],model_calls=0)
    C.write(OUT/'selftest.json',value);return value


def geometry():
    reg,m=verify_registration()
    if C.read(OUT/'baseline_replay.json')['status']!='PASS':raise ValueError('Replay barrier not passed')
    rows=[]
    for q in m['requests']:
        dest=OUT/'evaluation'/ARM/q['request_id']/'post'
        if (dest/'receipt.json').exists():
            r=C.read(dest/'receipt.json');rows.append(r['capacity']);continue
        if dest.exists():raise ValueError('Incomplete geometry attempt requires explicit recovery')
        dest.mkdir(parents=True);C.write(dest/'started.json',dict(request_id=q['request_id'],time=C.utc()))
        base=OLD/'evaluation'/ARM/q['request_id'];raw=C.arrays(base/'raw/raw.npz');source=C.arrays(base/'raw/source_condition.npz')
        pre=C.arrays(base/'post/pre_geo.npz')['output'];canonical,final,res,orient,anchors,logs=finish(raw,pre,source,True)
        G.save_npz(dest/'post.npz',final);G.save_npz(dest/'post_canonical.npz',canonical)
        geo_io.write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
        C.write(dest/'merge_ledger.json',logs);C.csvwrite(dest/'merge_ledger.csv',logs)
        C.write(dest/'geometry.json',dict(stats=res['stats'],face_records=res['face_records']))
        C.write(dest/'anchor_identity.json',anchors);C.write(dest/'source_restoration.json',orient['restoration_summary'])
        oo=orient['orientation_result'];C.write(dest/'orientation.json',dict(summary=oo['summary'],components=oo['component_rows'],edges=oo['edge_rows']))
        rec=dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
            rejected_merges=res['stats']['rejected_merges'],baseline_deleted_free_faces=C.read(base/'post/geometry.json')['stats']['deleted_free_faces'],
            candidate_deleted_free_faces=res['stats']['deleted_free_faces'],
            initial_over_capacity_edges=len(res['stats']['initial_C_edge_over_capacity']),
            final_over_capacity_edges=len(res['stats']['final_C_edge_over_capacity']),
            anchors=anchors,seconds=res['stats']['seconds'])
        C.write(dest/'receipt.json',dict(status='COMPLETE',version=VERSION,arm=ARM,request_id=q['request_id'],
            capacity=rec,source_pre_geo_sha256=C.sha(base/'post/pre_geo.npz'),registration_sha256=C.sha(OUT/'registration.json'),
            D0_forwards=0,model_calls=0,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
        rows.append(rec);print('CAPACITY_GEOMETRY',len(rows),q['request_id'],rec['rejected_merges'],flush=True)
    C.csvwrite(OUT/'geometry_summary.csv',rows);C.write(OUT/'geometry_completion.json',dict(status='COMPLETE',rows=rows,inputs=16,model_calls=0))
    return rows


def score():
    from r2.pilots.seam_slot_c200_v1 import evaluate as E
    reg,m=verify_registration();done=C.read(OUT/'geometry_completion.json');assert done['inputs']==16
    receipts=[]
    for q in m['requests']:
        receipts.append(E.score_one((str(OUT),ARM,q,'POST',C.sha(OUT/'registration.json'))))
        print('CAPACITY_SCORE',len(receipts),q['request_id'],flush=True)
    C.write(OUT/'score_completion.json',dict(status='COMPLETE',cells=16,receipts=receipts,workers=1,threads=1,model_calls=0))
    return receipts


def aggregate():
    reg,m=verify_registration();rows=[];pairs=[]
    keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
        'ConflictCov_C_cross','ConflictCov_C_dup','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
        'CF_illegal_pair_count','FF_illegal_pair_count','C_F_valid_edge_count','C_F_has_valid_edge','C_F_valid_edge_total_length',
        'C_F_nonmanifold_edge_count','reached_free_area_fraction','opposite_winding_reached_free_area_fraction',
        'free_only_reached_bulk_area_fraction','retained_seam_faces','retained_bulk_faces','deleted_free_faces']
    for q in m['requests']:
        sample={}
        for label,root in (('BASELINE',OLD),('CAPACITY_GUARD',OUT)):
            s=C.read(root/'evaluation/scores'/ARM/q['request_id']/'POST/score.json')
            full={**s['metrics'],**s['topology'],**s['seam_to_bulk']};sample[label]=full
            row={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','N','K','split')}
            row.update(label=label,**{k:v for k,v in full.items() if isinstance(v,(int,float,bool)) or v is None})
            rows.append(row)
        for key in keys:
            a,b=sample['BASELINE'].get(key),sample['CAPACITY_GUARD'].get(key)
            pairs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],metric=key,
                baseline=a,candidate=b,delta=b-a if a is not None and b is not None else None,status='PASS' if a is not None and b is not None else 'NA'))
    C.csvwrite(OUT/'per_output.csv',rows);C.csvwrite(OUT/'paired_per_noise.csv',pairs)
    summary=[]
    for key in keys:
        rr=[r for r in pairs if r['metric']==key and r['status']=='PASS']
        summary.append(dict(metric=key,n=len(rr),baseline_mean=float(np.mean([r['baseline'] for r in rr])) if rr else None,
            candidate_mean=float(np.mean([r['candidate'] for r in rr])) if rr else None,
            delta_mean=float(np.mean([r['delta'] for r in rr])) if rr else None,
            increased=sum(r['delta']>0 for r in rr),decreased=sum(r['delta']<0 for r in rr)))
    C.csvwrite(OUT/'summary.csv',summary)
    C.write(OUT/'completion.json',dict(status='COMPLETE',version=VERSION,requests=16,model_calls=0,
        baseline_replay_all_arrays_exact=True,protected_files_unchanged=True,summary=summary,
        files={str(p):C.sha(p) for p in (OUT/'per_output.csv',OUT/'paired_per_noise.csv',OUT/'summary.csv')}))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','replay','selftest','geometry','score','aggregate','all'),required=True);a=p.parse_args()
    stages=[register,selftest,replay,geometry,score,aggregate] if a.stage=='all' else [globals()[a.stage]]
    for f in stages: f()
    if 'torch' in sys.modules:raise AssertionError('No model framework allowed')


if __name__=='__main__':main()
