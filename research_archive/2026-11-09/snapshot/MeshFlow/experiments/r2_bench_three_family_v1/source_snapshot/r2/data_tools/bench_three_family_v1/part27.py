"""One fixed shell-only short-edge fallback; explicit freeze precedes execution.

Default freezes the recipe only. Root executes --execute once and subsequently
applies the unchanged whole-target geometry/fidelity/semantic gates.
"""
from __future__ import annotations
import argparse
import collections
import json
import numpy as np
from .common import ROOT, OUT, read, sha, save, load, npz, resolve
from r2.data_tools.bench_parts_v1 import quality as Q

DEST=OUT/'fallback27'


def _edges(f,face_ids):
    book=collections.defaultdict(list);neighbors=collections.defaultdict(set)
    for i in face_ids:
        face=f[i]
        for j in range(3):
            a,b=int(face[j]),int(face[(j+1)%3])
            book[tuple(sorted((a,b)))].append(int(i));neighbors[a].add(b);neighbors[b].add(a)
    return book,neighbors


def freeze_recipe():
    target=DEST/'RECIPE.json'
    if target.exists():
        r=read(target)
        for p,h in r['protected'].items():assert sha(p)==h,p
        return r
    parent=next(p for p in read(OUT/'PROTOCOL.json')['parents'] if p['parent_id'].startswith('27eb79'))
    source=resolve(parent['source']['clean_case_path'])
    inspection=OUT/'part_inspection'/(parent['parent_id']+'.json')
    component=next(c for c in read(inspection)['components'] if c['component_id']==1)
    a=load(source);v=a['model_vertices'].astype(float);f=a['source_faces'].astype(int)
    book,_=_edges(f,component['face_ids']);limit=float(.0025*Q.L)
    candidates=[]
    for (u,w),faces in book.items():
        distance=float(np.linalg.norm(v[u]-v[w]))
        if distance<=limit and np.sign(v[u,0])==np.sign(v[w,0]):
            candidates.append(dict(edge=[u,w],length_model=distance,original_incident_faces=faces))
    candidates.sort(key=lambda r:(r['length_model'],r['edge']))
    boundary=sorted({x for e,fs in book.items() if len(fs)==1 for x in e})
    shellverts=set(np.unique(f[component['face_ids']]).tolist())
    outside=[i for i in range(len(f)) if i not in set(component['face_ids'])]
    shared=sorted(shellverts&set(np.unique(f[outside]).tolist()))
    r=dict(version='BENCH27_SINGLE_SHORTEDGE_SHELL_PROPOSAL_V1',parent_id=parent['parent_id'],
        source_path=str(source),source_N=len(f),component_id=1,component_source_face_ids=component['face_ids'],
        exact_untouched_component_ids=[0,2],original_boundary_vertex_ids=boundary,
        additionally_pinned_contact_vertex_ids=shared,L=float(Q.L),maximum_original_edge_length_L=.0025,
        maximum_tracked_source_vertex_displacement_L=.0025,target_total_N_max=256,
        candidates=candidates,candidate_passes=1,retries=0,
        endpoint_rule='If one endpoint is pinned keep its exact source position/ID; reject two pinned endpoints. Otherwise keep lower ID at midpoint.',
        pinned_rule='All original shell open-boundary vertices (stronger than only true corners), plus shell vertices used by another component, remain exact.',
        order='Original edge length then sorted original endpoint IDs; one pass; removed endpoint or no-longer-edge is skipped; no new edges are added to the list.',
        topology='Recompute link condition on live shell before each collapse; only edge-incident faces may become index-degenerate and be removed; reject any other duplicate/degenerate face, boundary changes, or nonmanifold edge.',
        geometry='Remaining affected triangles positive area>1e-12 L^2, normal dot>0 against both immediately prior and original same-ID source face; source member displacements<=.0025L.',
        stopping='Stop once total N<=256, otherwise exhaust original list and retain BLOCKED_N; no threshold adjustment or QEM.',
        final_admission='No target PASS from this routine. Root separately runs original finite-pair, nonmanifold-vertex, component/hole, .25%L bidirectional fixed probes, area[.98,1.02], semantic and deployment gates.',
        protected={str(p):sha(p) for p in [source,inspection,OUT/'PROTOCOL.json',__file__]})
    save(target,r);return r


def execute():
    recipe=freeze_recipe();receipt=DEST/'receipt.json'
    if receipt.exists():
        old=read(receipt)
        for p,h in old['artifacts'].items():assert sha(p)==h,p
        return old
    a=load(recipe['source_path']);v=a['model_vertices'].astype(np.float64);original=v.copy()
    f=a['source_faces'].astype(np.int64);sourcef=f.copy();alive=np.ones(len(f),bool)
    shell=set(recipe['component_source_face_ids']);pinned=set(recipe['original_boundary_vertex_ids']+recipe['additionally_pinned_contact_vertex_ids'])
    member={int(i):{int(i)} for i in np.unique(f)};removedverts=set();audit=[]
    initialbook,_=_edges(f,sorted(shell));initialboundary={e for e,fs in initialbook.items() if len(fs)==1}
    minimum_area=1e-12*recipe['L']**2;maxmove=.0025*recipe['L']
    for candidate in recipe['candidates']:
        if int(alive.sum())<=256:break
        u,w=candidate['edge'];record=dict(edge=[u,w],length_model=candidate['length_model'])
        def reject(reason):record.update(status='REJECT',reason=reason);audit.append(record)
        if u in removedverts or w in removedverts:reject('ORIGINAL_ENDPOINT_ALREADY_REMOVED');continue
        book,neighbors=_edges(f,sorted(shell&set(np.flatnonzero(alive))))
        edge=tuple(sorted((u,w)))
        if edge not in book:reject('NO_LONGER_EDGE');continue
        incident=book[edge]
        opposing={int(x) for i in incident for x in f[i] if int(x) not in (u,w)}
        if neighbors[u]&neighbors[w]!=opposing:reject('LINK_CONDITION');continue
        if u in pinned and w in pinned:reject('TWO_PINNED_BOUNDARY_OR_CONTACT_VERTICES');continue
        keep,drop=(u,w) if u in pinned else (w,u) if w in pinned else (min(u,w),max(u,w))
        proposed=v[keep].copy() if keep in pinned else (v[u]+v[w])/2
        origins=member[u]|member[w]
        if float(np.linalg.norm(original[list(origins)]-proposed,axis=1).max())>maxmove:reject('SOURCE_DISPLACEMENT_BUDGET');continue
        trialf=f.copy();trialf[alive&(np.any(f==drop,axis=1))]=np.where(f[alive&(np.any(f==drop,axis=1))]==drop,keep,f[alive&(np.any(f==drop,axis=1))])
        dead=[int(i) for i in np.flatnonzero(alive) if len(set(map(int,trialf[i])))<3]
        if set(dead)!=set(incident):reject('UNEXPECTED_COLLAPSED_FACE');continue
        trialalive=alive.copy();trialalive[dead]=False
        tv=v.copy();tv[keep]=proposed
        affected=[int(i) for i in np.flatnonzero(trialalive) if u in f[i] or w in f[i]]
        bad=None
        for i in affected:
            tri=tv[trialf[i]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
            if np.linalg.norm(n)*.5<=minimum_area:bad='GEOMETRIC_ZERO_AREA';break
            for refv,reff in ((v,f),(original,sourcef)):
                prev=refv[reff[i]];pn=np.cross(prev[1]-prev[0],prev[2]-prev[0])
                if np.dot(n,pn)<=0:bad='NORMAL_FLIP';break
            if bad:break
        if bad:reject(bad);continue
        livefaces=trialf[trialalive];unique=np.unique(np.sort(livefaces,axis=1),axis=0)
        if len(unique)!=len(livefaces):reject('DUPLICATE_FACE');continue
        trialbook,_=_edges(trialf,sorted(shell&set(np.flatnonzero(trialalive))))
        if any(len(fs)>2 for fs in trialbook.values()):reject('NONMANIFOLD_EDGE');continue
        if {e for e,fs in trialbook.items() if len(fs)==1}!=initialboundary:reject('OPEN_BOUNDARY_CHANGED');continue
        f,v,alive=trialf,tv,trialalive;member[keep]=origins;member.pop(drop);removedverts.add(drop)
        record.update(status='ACCEPT',keep_vertex=keep,removed_vertex=drop,position=proposed.tolist(),
                      removed_source_face_ids=dead,affected_source_face_ids=affected,total_N_after=int(alive.sum()))
        audit.append(record)
    targetf=f[alive];sourceids=np.flatnonzero(alive)
    assert np.array_equal(v[sorted(pinned)],original[sorted(pinned)])
    unaffected=[i for i in range(len(f)) if i not in shell]
    assert alive[unaffected].all() and np.array_equal(f[unaffected],sourcef[unaffected])
    assert np.array_equal(v[np.unique(f[unaffected])],original[np.unique(sourcef[unaffected])])
    # Store as model FP32, the production coordinate dtype. The full root audit
    # checks these stored coordinates, not float64 working memory.
    npz(DEST/'proposal_mesh.npz',dict(vertices=v.astype(np.float32),faces=targetf,
                                    source_face_ids=sourceids,removed_source_face_ids=np.flatnonzero(~alive)))
    save(DEST/'lineage.json',dict(recipe_sha256=sha(DEST/'RECIPE.json'),audit=audit,
        output_face_to_source_face=sourceids.tolist(),removed_source_face_ids=np.flatnonzero(~alive).tolist(),
        component0_and2_exact=True,original_open_boundary_exact=True,
        target_vertex_original_member_ids={str(k):sorted(z) for k,z in member.items()}))
    result=dict(status='PROPOSAL_PENDING_FULL_AUDIT' if len(targetf)<=256 else 'BLOCKED_N_PROPOSAL_RETAINED',
        source_N=len(sourcef),target_N=len(targetf),accepted_collapses=sum(r['status']=='ACCEPT' for r in audit),
        rejections=dict(collections.Counter(r['reason'] for r in audit if r['status']=='REJECT')),
        recipe_sha256=sha(DEST/'RECIPE.json'),source_mutations=0,model_calls=0,attempts=1,
        artifacts={str(p):sha(p) for p in [DEST/'proposal_mesh.npz',DEST/'lineage.json',DEST/'RECIPE.json']})
    save(receipt,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    result=execute() if args.execute else freeze_recipe()
    print(json.dumps(result if args.execute else dict(status='RECIPE_FROZEN',candidates=len(result['candidates']),path=str(DEST/'RECIPE.json')),ensure_ascii=False))
