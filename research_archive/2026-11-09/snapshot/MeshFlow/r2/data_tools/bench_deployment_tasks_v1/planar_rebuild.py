"""One conservative CPU target proposal: retriangulate coplanar patch interiors.

Keeps each patch's ordered boundary and its original vertex coordinates. Holes,
branched boundaries and ambiguous patches are retained unchanged. No QEM,
smoothing, welding, source-part merging, model call or lost positive-area face.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import time
from pathlib import Path
import numpy as np
from r2.data_tools.bench_user_v2 import build as b
from .prepare import ROOT, OUT

VERSION = 'BENCH_COPLANAR_BOUNDARY_PRESERVING_TARGET_V1'
PLANE_EPS = 1e-7 * b.Q.L
NORMAL_DOT_MIN = 1.-1e-10


def cross2(a,b_): return float(a[0]*b_[1]-a[1]*b_[0])


def triangulate_loop(vertices, loop, normal):
    drop=int(np.argmax(np.abs(normal)))
    xy=np.delete(vertices[np.asarray(loop)],drop,axis=1).astype(np.float64)
    area=sum(cross2(xy[i],xy[(i+1)%len(xy)]) for i in range(len(xy)))
    if area==0: return None
    sign=1 if area>0 else -1
    tol=1e-13*max(float(np.ptp(xy,axis=0).max())**2,1.)
    todo=list(range(len(loop))); result=[]
    while len(todo)>3:
        found=False
        for pos in range(len(todo)):
            ia,ib,ic=todo[pos-1],todo[pos],todo[(pos+1)%len(todo)]
            a_,bb,c=xy[[ia,ib,ic]]
            if sign*cross2(bb-a_,c-bb)<=tol: continue
            blocked=False
            for j in todo:
                if j in (ia,ib,ic):continue
                p=xy[j]
                if min(sign*cross2(bb-a_,p-a_),sign*cross2(c-bb,p-bb),sign*cross2(a_-c,p-c))>=-tol:
                    blocked=True;break
            if blocked:continue
            result.append([loop[ia],loop[ib],loop[ic]])
            todo.pop(pos);found=True;break
        if not found:return None
    result.append([loop[i] for i in todo])
    return np.asarray(result,np.int64)


def retriangulate(a):
    v=np.asarray(a['model_vertices'],np.float64);f=np.asarray(a['source_faces'],np.int64)
    triangles=v[f]; normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(normals,axis=1);normals=normals/lengths[:,None]
    adj=defaultdict(list)
    for entries in b.Q.edgebook(f).values():
        if len(entries)==2:
            i,j=entries[0][0],entries[1][0];adj[i].append(j);adj[j].append(i)
    vertex_faces=defaultdict(set)
    for i,tri in enumerate(f):
        for vid in tri:vertex_faces[int(vid)].add(i)
    seen=set();groups=[]
    for seed in range(len(f)):
        if seed in seen:continue
        seen.add(seed);todo=[seed];group=[];n=normals[seed];origin=triangles[seed,0]
        while todo:
            i=todo.pop();group.append(i)
            for j in sorted(adj[i]):
                if j in seen:continue
                if float(np.dot(normals[j],n))>=NORMAL_DOT_MIN and np.max(np.abs((triangles[j]-origin)@n))<=PLANE_EPS:
                    seen.add(j);todo.append(j)
        groups.append(sorted(group))
    out=[];lineage=[];ledger=[]
    for group in groups:
        gf=f[group];reason='UNCHANGED_NO_INTERIOR_REDUCTION';new=None
        if len(group)>1:
            book=b.Q.edgebook(gf);boundary=[rows[0][1:] for rows in book.values() if len(rows)==1]
            outgoing=defaultdict(list);incoming=defaultdict(list)
            for u,w in boundary:outgoing[int(u)].append(int(w));incoming[int(w)].append(int(u))
            if boundary and all(len(x)==1 for x in outgoing.values()) and set(incoming)==set(outgoing) and all(len(x)==1 for x in incoming.values()):
                start=min(outgoing);loop=[start];cur=outgoing[start][0]
                while cur!=start and cur not in loop:
                    loop.append(cur);cur=outgoing[cur][0]
                if cur==start and len(loop)==len(boundary):
                    internal=set(map(int,gf.ravel()))-set(loop)
                    if len(loop)-2<len(group) and all(vertex_faces[x].issubset(group) for x in internal):
                        proposed=triangulate_loop(v,loop,normals[group[0]])
                        if proposed is not None:
                            old_area=float(b.area2(v,gf).sum());new_area=float(b.area2(v,proposed).sum())
                            newbook=b.Q.edgebook(proposed)
                            newboundary={e for e,rows in newbook.items() if len(rows)==1}
                            oldboundary={e for e,rows in book.items() if len(rows)==1}
                            nn=np.cross(v[proposed[:,1]]-v[proposed[:,0]],v[proposed[:,2]]-v[proposed[:,0]])
                            if newboundary==oldboundary and np.all(np.linalg.norm(nn,axis=1)>0) and np.all(nn@normals[group[0]]>0) and abs(new_area-old_area)<=max(old_area*1e-6,1e-12):
                                new=proposed;reason='COPLANAR_INTERIOR_RETRIANGULATED'
                            else:reason='RETAINED_LOCAL_VALIDATION_FAILED'
                        else:reason='RETAINED_EAR_CLIP_AMBIGUOUS'
                    elif internal:reason='RETAINED_SHARED_INTERIOR_VERTEX'
                else:reason='RETAINED_MULTI_LOOP_OR_HOLE'
            else:reason='RETAINED_NON_SIMPLE_BOUNDARY'
        if new is None:
            out.extend(gf.tolist());lineage.extend([[int(i)] for i in group])
            new_count=len(group)
        else:
            out.extend(new.tolist());lineage.extend([list(group) for _ in new]);new_count=len(new)
        ledger.append(dict(source_clean_face_ids=group,old_faces=len(group),new_faces=new_count,action=reason))
    result=dict(a);result['source_faces']=np.asarray(out,np.int64)
    return result,lineage,ledger


def pad_surface(a,lineage):
    a={k:a[k].copy() for k in ('source_vertices','normalized_vertices','model_vertices','transform_json','source_faces')}
    f=a['source_faces'];insertions=[]
    while len(f)<128:
        i=int(np.argmax(b.area2(a['model_vertices'],f)));tri=f[i].copy();vi=len(a['model_vertices'])
        for key in ('source_vertices','normalized_vertices','model_vertices'):
            a[key]=np.concatenate([a[key],a[key][tri].astype(float).mean(0).astype(np.float32)[None]],axis=0)
        children=np.asarray([[tri[0],tri[1],vi],[tri[1],tri[2],vi],[tri[2],tri[0],vi]],np.int64)
        f=np.concatenate([f[:i],children,f[i+1:]])
        lineage=lineage[:i]+[lineage[i]]*3+lineage[i+1:]
        insertions.append(dict(vertex_id=vi,centroid_of_existing_vertices=tri.tolist()))
    a['source_faces']=f
    return a,lineage,insertions


def freeze():
    preparation=b.read(OUT/'preparation.json')
    value=dict(version=VERSION,source_parent_ids=[r['parent_id'] for r in preparation['clean_high_source_candidates']],
        sources=preparation['clean_high_source_candidates'],max_proposals_per_parent=1,
        algorithm='Connected same-normal coplanar patches; simple single-loop boundary; only remove patch-interior tessellation. Preserve every boundary vertex/index and source coordinates. Ambiguous patch unchanged. No QEM or retries.',
        plane_tolerance_model=PLANE_EPS,normal_dot_min=NORMAL_DOT_MIN,target_N=[128,256],
        source_finite_checks='Reuse authenticated existing complete-source checks; inspect only newly constructed targets.',
        new_target_checks='C/Y from same M; frozen finite-pair detector; no new illegal/unresolved pairs, zero areas, edge nonmanifold, winding conflicts; unchanged edge component count; sampled bidirectional fidelity <=.25%L and area ratio .98..1.02; visual completeness required.',
        below128='Largest-triangle centroid subdivision, same surface and recorded lineage.',
        above256='Keep proposal and mark BLOCKED_N; no automatic more aggressive simplifier.',
        no_model_calls=True,source_readonly=True,source_hashes=preparation['protected_source_hashes'],
        code_sha256=b.sha(__file__))
    b.save(OUT/'REBUILD_PROTOCOL.json',value)
    return value


def one(row):
    uid=row['parent_id'];dest=OUT/'rebuilt_parents'/uid;receipt=dest/'parent.json'
    if receipt.exists():
        old=b.read(receipt)
        for p,h in old['artifact_hashes'].items():assert b.sha(p)==h
        return old
    started=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    assert b.sha(row['clean_case_path'])==row['clean_case_sha256']
    original=b.load(row['clean_case_path']);a,lineage,ledger=retriangulate(original)
    n_before_pad=len(a['source_faces']);a,lineage,insertions=pad_surface(a,lineage)
    target=b.canonical(a,a['source_faces']);target['target_version_id']=np.asarray(VERSION)
    b.npz(dest/'target_case.npz',target)
    b.npz(dest/'target_mesh.npz',dict(vertices=target['model_vertices'],faces=target['source_faces']))
    b.obj(dest/'target.obj',target['model_vertices'],target['source_faces'])
    b.save(dest/'lineage.json',dict(source_clean_case=row['clean_case_path'],source_clean_sha256=row['clean_case_sha256'],
        new_face_to_source_patch=lineage,patch_mapping_is_surface_region_not_one_to_one_face_identity=True,
        patch_ledger=ledger,centroid_insertions=insertions,original_boundary_vertices_preserved=True,
        no_original_vertex_motion=True,source_lineage=row['source_lineage_path']))
    reasons=[];n=len(target['source_faces']);q=None;fidelity=None
    if not 128<=n<=256:reasons.append('BLOCKED_N_OUTSIDE128_256')
    else:
        from r2.data_tools.bench_user_v2_expand.simplify_standard import distances,probes
        q=b.full_geometry(target);b.save(dest/'geometry.json',q)
        v,f=original['model_vertices'],original['source_faces'];nv,nf=target['model_vertices'],target['source_faces']
        d0,_=distances(probes(v,f),nv,nf);d1,_=distances(probes(nv,nf),v,f)
        oldgeo=b.read(row['source_geometry_path'])
        ratio=float(b.area2(nv,nf).sum()/b.area2(v,f).sum())
        bbox=float(np.max(np.abs(np.array([v[f].min((0,1)),v[f].max((0,1))])-np.array([nv[nf].min((0,1)),nv[nf].max((0,1))]))))
        fidelity=dict(area_ratio=ratio,source_to_target_probe_max_L=float(d0.max()/b.Q.L),target_to_source_probe_max_L=float(d1.max()/b.Q.L),
            bbox_shift_L=bbox/b.Q.L,source_components=len(oldgeo['topology']['edge_connected_components']),
            target_components=len(q['topology']['edge_connected_components']),continuous_Hausdorff_certified=False)
        b.save(dest/'fidelity.json',fidelity)
        if q['illegal_pairs']:reasons.append('TARGET_ILLEGAL_FINITE_PAIRS')
        if q['unresolved_pairs']:reasons.append('TARGET_UNRESOLVED_PAIRS')
        if q['exact_zero_area_faces']:reasons.append('TARGET_ZERO_AREA')
        if q['topology']['nonmanifold_edges']:reasons.append('TARGET_NONMANIFOLD_EDGES')
        if q['topology']['same_winding_double_edges']:reasons.append('TARGET_WINDING_CONFLICT')
        if fidelity['source_components']!=fidelity['target_components']:reasons.append('SOURCE_COMPONENT_CHANGED')
        if not .98<=ratio<=1.02:reasons.append('AREA_CHANGED_OVER2_PERCENT')
        if max(fidelity['source_to_target_probe_max_L'],fidelity['target_to_source_probe_max_L'],fidelity['bbox_shift_L'])>.0025:reasons.append('FIDELITY_BUDGET')
    value=dict(parent_id=uid,status='HELD' if reasons else 'TARGET_PASS_PENDING_VISUAL_AND_SPLIT',reasons=reasons,
        N_source=len(original['source_faces']),N_after_retriangulation=n_before_pad,N=n,
        changed_patches=sum(x['action']=='COPLANAR_INTERIOR_RETRIANGULATED' for x in ledger),
        target_case_path=str(dest/'target_case.npz'),target_case_sha256=b.sha(dest/'target_case.npz'),
        source_row=row,fidelity=fidelity,wall_seconds=time.monotonic()-started,model_calls=0,
        artifact_hashes={str(p):b.sha(p) for p in dest.iterdir() if p.is_file()})
    b.save(receipt,value)
    print(json.dumps({k:value[k] for k in ('parent_id','N_source','N_after_retriangulation','N','status','reasons')},ensure_ascii=False),flush=True)
    return value


def main():
    from r2.pilots.bench_adaptation_strength_v1.common import resources
    p=freeze();tick=time.monotonic()
    b.save(OUT/'rebuild_resources.json',dict(resources=resources(),workers=4,threads_per_worker=1,GPU_workers=0,
        rationale='16 independent CPU target proposals, 24 logical CPU and >60GB available RAM; preserve other GPU work.'))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,p['sources']))
    result=dict(status='TARGET_PROPOSALS_COMPLETE',rows=rows,parents=len(rows),geometric_pass=sum(not r['reasons'] for r in rows),
        elapsed_seconds=time.monotonic()-tick,new_model_calls=0,new_noise=0,resources_after=resources())
    b.save(OUT/'rebuild_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','resources_after')}))


if __name__=='__main__':main()
