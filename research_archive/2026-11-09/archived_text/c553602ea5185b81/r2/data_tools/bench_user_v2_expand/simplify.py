"""One fixed QEM proposal per clean high-N real source, with fidelity gates."""
from .supply import *
from pathlib import Path
import sys
sys.path.insert(0,str(OUT/'vendor'))
import fast_simplification as fs
import trimesh

def probes(v,f):
    t=np.asarray(v,float)[f]
    return np.concatenate([t.reshape(-1,3),t.mean(1),(t[:,0]+t[:,1])/2,(t[:,1]+t[:,2])/2,(t[:,2]+t[:,0])/2])

def distances(points,v,f):
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False);ds=[];ids=[]
    for i in range(0,len(points),64):
        _,d,j=trimesh.proximity.closest_point_naive(mesh,points[i:i+64]);ds.extend(d);ids.extend(j)
    return np.asarray(ds),np.asarray(ids)

def freeze():
    rows=[r for r in b.read(OUT/'high_source/audited.json')['rows'] if r['source_finite_pass']]
    p=OUT/'high_source/SIMPLIFICATION_PROTOCOL.json'
    if p.exists():return b.read(p)
    a=dict(parent_ids=[r['parent_id'] for r in rows],rows=rows,
        target_N=256,agg=2,attempts_per_parent=1,library='fast-simplification==0.1.12',
        gates=dict(area_ratio=[.98,1.02],max_fixed_probe_distance_L=.0025,bbox_max_shift_L=.0025,
            component_count_unchanged=True,full_target_finite_geometry='same frozen detector',visual_review='required'),
        distance_contract='Bidirectional deterministic vertex/centroid/edge-midpoint to finite surface distances; sampled fidelity, NOT certified Hausdorff bound.',
        coordinate_contract='Use original model coordinates and original transform; no independent renormalization; new target topology and geometry explicit version.',
        resources=b.resources(),source_manifest_sha256=b.sha(OUT/'high_source/audited.json'))
    b.save(p,a);event('SIMPLIFICATION_FROZEN',parents=len(rows),workers=4,resources=a['resources']);return a

def one(row):
    uid=row['parent_id'];dest=OUT/'parents'/uid;receipt=dest/'parent.json'
    if receipt.exists():
        r=b.read(receipt)
        for path,h in r['artifact_hashes'].items():assert b.sha(path)==h
        return r
    dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    assert b.sha(row['clean_case_path'])==row['clean_case_sha256']
    a=b.load(row['clean_case_path']);orig=b.load(row['source_case_path']);v=a['model_vertices'];f=a['source_faces']
    nv,nf,collapses=fs.simplify(v,f,target_count=256,agg=2,return_collapses=True)
    nv=nv.astype(np.float32);nf=nf.astype(np.int64);nf,orientation=b.orient(nf)
    trans=json.loads(str(a['transform_json']))
    norm=nv*np.float32(.3762/2.)
    source=(norm.astype(float)*trans['divisor3']+np.asarray(trans['center3'])+np.asarray(trans['center2']))*trans['divisor1']+np.asarray(trans['center1'])
    arrays=dict(source_vertices=source.astype(np.float32),normalized_vertices=norm,model_vertices=nv,transform_json=a['transform_json'])
    target=b.canonical(arrays,nf)
    b.npz(dest/'original_mesh.npz',dict(vertices=orig['model_vertices'],faces=orig['source_faces']));b.obj(dest/'original.obj',orig['model_vertices'],orig['source_faces'])
    b.npz(dest/'target_mesh.npz',dict(vertices=nv,faces=nf));b.npz(dest/'target_case.npz',target);b.obj(dest/'target.obj',nv,nf)
    d0,_=distances(probes(v,f),nv,nf);d1,_=distances(probes(nv,nf),v,f)
    _,near=distances(nv[nf].astype(float).mean(1),v,f)
    area_ratio=float(b.area2(nv,nf).sum()/b.area2(v,f).sum())
    bbox_shift=float(np.max(np.abs(np.array([v.min(0),v.max(0)])-np.array([nv.min(0),nv.max(0)]))))
    source_geo=b.read(row['source_geometry_path']);q=b.full_geometry(target)
    old_comp=len(source_geo['topology']['edge_connected_components']);new_comp=len(q['topology']['edge_connected_components'])
    fidelity=dict(area_ratio=area_ratio,bbox_shift_model=bbox_shift,bbox_shift_L=bbox_shift/b.Q.L,
        source_to_target_probe_max_L=float(d0.max()/b.Q.L),target_to_source_probe_max_L=float(d1.max()/b.Q.L),
        source_probes=len(d0),target_probes=len(d1),source_components=old_comp,target_components=new_comp,
        guarantee='Finite deterministic probe coverage only; not continuous Hausdorff certification')
    b.save(dest/'fidelity.json',fidelity);b.save(dest/'geometry.json',q)
    b.save(dest/'lineage.json',dict(operation='Fixed QEM simplification after source cleaning',
        target_version_id=VERSION,original_source_case=row['source_case_path'],clean_source_case=row['clean_case_path'],
        source_clean_lineage=row['source_lineage_path'],vertex_collapse_pairs=collapses.tolist(),
        new_face_nearest_clean_source_face=near.tolist(),nearest_mapping_is_approximate_not_exact_face_ancestry=True,
        normalization_unchanged=True,original_vertex_values_preserved=False,orientation=orientation))
    reasons=[]
    if not 128<=len(nf)<=256:reasons.append('N_OUTSIDE_ADAPTER')
    if q['illegal_pairs']:reasons.append('FULL_TARGET_ILLEGAL_FINITE_PAIRS')
    if q['unresolved_pairs']:reasons.append('FULL_TARGET_GEOMETRY_UNRESOLVED')
    if q['topology']['nonmanifold_edges']:reasons.append('FULL_TARGET_NONMANIFOLD_EDGES')
    if q['topology']['same_winding_double_edges'] or orientation['parity_conflicts']:reasons.append('WINDING_CONFLICT')
    if q['exact_zero_area_faces']:reasons.append('TARGET_ZERO_AREA')
    if not .98<=area_ratio<=1.02:reasons.append('SURFACE_AREA_CHANGED_OVER_2_PERCENT')
    if max(d0.max(),d1.max(),bbox_shift)>.0025*b.Q.L:reasons.append('FIDELITY_DISTANCE_BUDGET')
    if old_comp!=new_comp:reasons.append('COMPONENT_COUNT_CHANGED')
    r=dict(parent_id=uid,split='TRAIN',official_split='train',target_version_id=VERSION,production_rank=1000,
        has_reviewed_back_candidate=False,was_pending_parent=False,source_case_path=row['source_case_path'],source_case_sha256=row['source_case_sha256'],
        original_mesh_path=str(dest/'original_mesh.npz'),original_mesh_sha256=b.sha(dest/'original_mesh.npz'),
        target_mesh_path=str(dest/'target_mesh.npz'),target_mesh_sha256=b.sha(dest/'target_mesh.npz'),
        target_case_path=str(dest/'target_case.npz'),target_case_sha256=b.sha(dest/'target_case.npz'),lineage_path=str(dest/'lineage.json'),geometry_path=str(dest/'geometry.json'),
        fidelity_path=str(dest/'fidelity.json'),N_original=len(orig['source_faces']),N_clean_before_refinement=len(f),N=len(nf),
        zero_faces_removed=None,duplicate_excess_removed=None,refined_new_vertices=0,faces_flipped=len(orientation['flipped_retained_face_positions']),
        illegal_pairs=len(q['illegal_pairs']),unresolved_pairs=len(q['unresolved_pairs']),
        reasons=reasons,status='BLOCKED_TARGET_GEOMETRY' if reasons else 'TARGET_GEOMETRY_PASS_PENDING_VISUAL',
        processing_note='One QEM256 agg2 proposal; accepted only after independent finite geometry/fidelity and actual visual review; original coordinates/normalization reference retained, vertices changed.',
        wall_seconds=time.monotonic()-start,artifact_hashes={str(p):b.sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    b.save(receipt,r);print(json.dumps(dict(parent=uid[:6],N=len(nf),status=r['status'],reasons=reasons)),flush=True);return r

def main():
    configure();protocol=freeze()
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:parents=list(pool.map(one,protocol['rows']))
    passed=[p for p in parents if not p['reasons']]
    with ProcessPoolExecutor(max_workers=4,initializer=configure) as pool:rr=list(pool.map(condition_parent,passed))
    m=b.read(OUT/'expanded_candidate_manifest.json');m['parents']+=parents;m['conditions']+=[r for rows in rr for r in rows]
    b.save(OUT/'all_candidate_manifest.json',m)
    summary=dict(high_source_attempts=len(parents),geometry_and_fidelity_pass=len(passed),new_conditions=sum(map(len,rr)),
        total_geometry_candidates=sum(not p['reasons'] for p in m['parents']))
    b.save(OUT/'high_source/simplification_summary.json',summary);event('SIMPLIFICATION_COMPLETE',**summary);print(json.dumps(summary))

if __name__=='__main__':main()
