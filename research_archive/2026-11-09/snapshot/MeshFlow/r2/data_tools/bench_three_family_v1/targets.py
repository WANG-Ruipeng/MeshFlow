"""Execute only frozen target proposals; source geometry and old receipts read-only."""
from .common import *
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from r2.data_tools.bench_user_v2 import build as b
from r2.data_tools.bench_parts_v1.quality import topology
from .boundary_simplify import simplify_boundaries

def labels_for(parent,a):
    labels=np.zeros(len(a['source_faces']),np.int64)
    for i,r in enumerate(sorted(parent['tasks'],key=lambda x:x['recipe_id'])):
        pair=load(r['paired_path']);labels[pair['known_ids']] |= 1<<i
    return labels

def component_summary(v,f):
    top=topology(f); rows=[]
    for ids in top['edge_connected_components']:
        ids=np.asarray(ids);xyz=v[f[ids]].reshape(-1,3)
        st=topology(f[ids]);vs=len(np.unique(f[ids]));es=len(b.Q.edgebook(f[ids]))
        rows.append(dict(face_ids=ids.tolist(),faces=len(ids),vertices=vs,edges=es,euler=vs-es+len(ids),boundary_loops=boundary_signature(f[ids]),
          boundary_edges=len(st['open_edges']),bbox_min=xyz.min(0).tolist(),bbox_max=xyz.max(0).tolist(),area=float(b.area2(v,f[ids]).sum()/2)))
    return rows

def boundary_signature(f):
    rows=topology(f)['open_edges'];edges=[(int(x[1]),int(x[2])) for x in rows]
    if not edges:return dict(loops=0,regular=True)
    deg=Counter(v for e in edges for v in e);nodes=set(deg);count=0
    adj={v:set() for v in nodes}
    for u,v in edges:adj[u].add(v);adj[v].add(u)
    while nodes:
        todo=[min(nodes)];nodes.remove(todo[0]);count+=1
        while todo:
            u=todo.pop()
            for v in adj[u]&nodes:nodes.remove(v);todo.append(v)
    return dict(loops=count,regular=all(d==2 for d in deg.values()))

def distances(p,v,f):
    import trimesh
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False);ds=[]
    for i in range(0,len(p),64):
        _,d,_=trimesh.proximity.closest_point_naive(mesh,np.asarray(p[i:i+64],float));ds.extend(d)
    return np.asarray(ds)

def probes(v,f):
    t=v[f].astype(float)
    # Fixed four-way interior barycentric lattice plus vertices and edge points.
    weights=np.asarray([[i/4,j/4,(4-i-j)/4] for i in range(5) for j in range(5-i)])
    return np.einsum('bc,fcd->fbd',weights,t).reshape(-1,3)

def check(parent,target,lineage,dest,old_labels,unchanged=False):
    old=load(parent['source']['clean_case_path']);v,f=old['model_vertices'],old['source_faces']
    nv,nf=target['model_vertices'],target['source_faces'];N=len(nf)
    if unchanged:
        # Source pair relations are all verified legal. A pure face permutation is equivalent.
        q=read(parent['source']['source_geometry_path'])
        assert not q['illegal_pairs'] and not q['unresolved_pairs']
        q=dict(q,topology=topology(nf),exact_zero_area_faces=np.flatnonzero(b.area2(nv,nf)==0).tolist(),
          reused_source_finite_receipt=parent['source']['source_geometry_path'],face_pair_ids='Source receipt indexing; no illegal/unresolved pairs exist',new_pair_tests=0)
    else: q=b.full_geometry(target)
    save(dest/'geometry.json',q)
    # Use protected source component identities through lineage, not output face order.
    oldcomps=component_summary(v,f);newcomps=component_summary(nv,nf)
    byold={i:ci for ci,r in enumerate(oldcomps) for i in r['face_ids']};per=[];comp_preserved=True
    assert len(lineage)==N and all(anc and all(0<=k<len(f) for k in anc) and len({byold[k] for k in anc})==1 for anc in lineage)
    assert {k for anc in lineage for k in anc}==set(range(len(f))), 'Missing source face-region lineage'
    for ci,oc in enumerate(oldcomps):
        ids=[j for j,anc in enumerate(lineage) if anc and all(byold[k]==ci for k in anc)]
        oo=np.asarray(oc['face_ids'],int);nn=np.asarray(ids,int)
        if not ids:
            comp_preserved=False;per.append(dict(source_component=ci,status='MISSING'));continue
        st=topology(nf[nn]);euler=len(np.unique(nf[nn]))-len(b.Q.edgebook(nf[nn]))+len(nn)
        d0=distances(probes(v,f[oo]),nv,nf[nn]);d1=distances(probes(nv,nf[nn]),v,f[oo])
        area_ratio=float(b.area2(nv,nf[nn]).sum()/b.area2(v,f[oo]).sum())
        bs=boundary_signature(nf[nn])
        compatible=len(st['edge_connected_components'])==1 and euler==oc['euler'] and bs==oc['boundary_loops']
        comp_preserved &= compatible
        per.append(dict(source_component=ci,target_face_ids=ids,source_faces=len(oo),target_faces=len(nn),area_ratio=area_ratio,
          old_euler=oc['euler'],new_euler=euler,old_boundary_edges=oc['boundary_edges'],new_boundary_edges=len(st['open_edges']),old_boundary_loops=oc['boundary_loops'],new_boundary_loops=bs,
          source_to_target_max_L=float(d0.max()/b.Q.L),target_to_source_max_L=float(d1.max()/b.Q.L),topology_signature_preserved=compatible))
    d0=distances(probes(v,f),nv,nf);d1=distances(probes(nv,nf),v,f)
    bbox=float(np.max(np.abs(np.stack([v[f].min((0,1)),v[f].max((0,1))])-np.stack([nv[nf].min((0,1)),nv[nf].max((0,1))]))))
    ratio=float(b.area2(nv,nf).sum()/b.area2(v,f).sum())
    fidelity=dict(source_to_target_max_L=float(d0.max()/b.Q.L),target_to_source_max_L=float(d1.max()/b.Q.L),area_ratio=ratio,bbox_shift_L=bbox/b.Q.L,
      source_components=len(oldcomps),target_components=len(newcomps),component_topology_preserved=bool(comp_preserved),per_component=per,
      probe_rule='15 barycentric lattice points per triangle in each direction, finite-triangle closest distance; NOT continuous Hausdorff certification',
      original_normalization_preserved=True)
    save(dest/'fidelity.json',fidelity)
    labels=[];semantic_ok=True
    for anc in lineage:
        unique=np.unique(old_labels[anc]);semantic_ok &= len(unique)==1;labels.append(int(unique[0]) if len(unique)==1 else -1)
    semantic_ok=bool(semantic_ok)
    reasons=[]
    if q['illegal_pairs']:reasons.append('ILLEGAL_FINITE_PAIRS')
    if q['unresolved_pairs']:reasons.append('UNRESOLVED_FINITE_PAIRS')
    if q['exact_zero_area_faces']:reasons.append('ZERO_AREA')
    for key in ('nonmanifold_edges','nonmanifold_vertices','same_winding_double_edges'):
        if q['topology'][key]:reasons.append(key.upper())
    if len(newcomps)!=len(oldcomps) or not comp_preserved:reasons.append('COMPONENT_OR_TOPOLOGY_CHANGE')
    if not .98<=ratio<=1.02:reasons.append('TOTAL_AREA_BUDGET')
    if any('area_ratio' not in x or not .98<=x['area_ratio']<=1.02 for x in per):reasons.append('PART_AREA_BUDGET')
    if max(d0.max()/b.Q.L,d1.max()/b.Q.L,bbox/b.Q.L)>.0025 or any(max(x.get('source_to_target_max_L',1),x.get('target_to_source_max_L',1))>.0025 for x in per):reasons.append('FIDELITY_BUDGET')
    if not semantic_ok:reasons.append('SEMANTIC_REGION_CROSSED')
    npz(dest/'face_semantics.npz',dict(face_semantic_bits=np.asarray(labels,np.int64)))
    return dict(N=N,geometry_reasons=reasons,geometry_pass=not reasons,current_N_pass=128<=N<=256,
      N_status='PASS' if 128<=N<=256 else 'BLOCKED_N_OUTSIDE128_256',fidelity=fidelity,
      finite_illegal=len(q['illegal_pairs']),finite_unresolved=len(q['unresolved_pairs']),topology={k:len(q['topology'][k]) for k in ('nonmanifold_edges','nonmanifold_vertices','same_winding_double_edges')},
      target_case_path=str(dest/'target_case.npz'),target_case_sha256=sha(dest/'target_case.npz'),
      lineage_path=str(dest/'lineage.json'),geometry_path=str(dest/'geometry.json'),fidelity_path=str(dest/'fidelity.json'),semantics_path=str(dest/'face_semantics.npz'))

def save_target(parent,arrays,lineage,ledger,method,unchanged=False):
    dest=OUT/'targets'/parent['parent_id']/method;receipt=dest/'receipt.json'
    if receipt.exists():
        x=read(receipt)
        for p,h in x['hashes'].items():assert sha(p)==h,p
        return x
    dest.mkdir(parents=True,exist_ok=True);a=load(parent['source']['clean_case_path']);labels=labels_for(parent,a)
    from r2.data_tools.bench_deployment_tasks_v1.planar_rebuild import pad_surface
    before=len(arrays['source_faces']);arrays,lineage,pad=pad_surface(arrays,lineage)
    target=b.canonical(arrays,arrays['source_faces']);target['target_version_id']=np.asarray(VERSION+'_'+method)
    npz(dest/'target_case.npz',target);npz(dest/'target_mesh.npz',dict(vertices=target['model_vertices'],faces=target['source_faces']))
    b.obj(dest/'target.obj',target['model_vertices'],target['source_faces'])
    save(dest/'lineage.json',dict(method=method,source_case_path=parent['source']['clean_case_path'],source_sha256=parent['source']['clean_case_sha256'],
      new_face_to_source_region=lineage,mapping_is_region_not_original_face_identity=True,ledger=ledger,padding_centroid_subdivisions=pad))
    checks=check(parent,target,lineage,dest,labels,unchanged=unchanged and not pad)
    r=dict(parent_id=parent['parent_id'],method=method,N_original=len(a['source_faces']),N_unpadded=before,**checks,
      status='GEOMETRY_PASS_PENDING_VISUAL' if checks['geometry_pass'] else 'HELD_GEOMETRY',
      hashes={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    save(receipt,r);event('TARGET_PROPOSAL_FINISHED',parent=parent['parent_id'],method=method,N=r['N'],geometry_reasons=r['geometry_reasons'])
    print(json.dumps({k:r[k] for k in ('parent_id','method','N_original','N','geometry_reasons','N_status')},ensure_ascii=False),flush=True)
    return r

def boundary_one(parent):
    rpath=OUT/'targets'/parent['parent_id']/'shared_boundary'/ 'receipt.json'
    if rpath.exists():
        r=read(rpath)
        for p,h in r['hashes'].items():assert sha(p)==h
        return r
    a=load(parent['source']['clean_case_path']);labels=labels_for(parent,a)
    new,lineage,ledger=simplify_boundaries(a,labels,L=b.Q.L)
    unchanged=len(lineage)==len(a['source_faces']) and all(len(x)==1 for x in lineage) and np.array_equal(new['model_vertices'],a['model_vertices']) and np.array_equal(new['source_faces'],a['source_faces'][[x[0] for x in lineage]])
    return save_target(parent,new,lineage,ledger,'shared_boundary',unchanged)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['boundary'],default='boundary');args=ap.parse_args()
    p=freeze();from r2.pilots.bench_adaptation_strength_v1.common import resources
    res=resources();event('CPU_TARGET_DISPATCH',resources=res,workers=3,threads_per_worker=1,GPU_workers=0)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=3) as pool: rows=list(pool.map(boundary_one,p['parents']))
    save(OUT/'boundary_summary.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,resources_before=res,model_calls=0))

if __name__=='__main__':main()
