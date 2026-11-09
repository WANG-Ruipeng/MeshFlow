"""List original-shell short edges satisfying a local manifold link condition.

This does not collapse an edge or construct a proposed/final mesh. The list is
only a concrete candidate recipe for the parent's separately frozen fallback.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np
from .part_recipe_probe import ROOT,OUT,read,sha


def main():
    out=OUT/'part_recipe_shortedge_probe.json'
    if out.exists():
        old=read(out)
        for p,h in old['protected'].items():assert sha(ROOT/p)==h
        print(json.dumps(dict(status='VERIFIED',path=str(out))));return
    p=next(p for p in read(OUT/'PROTOCOL.json')['parents'] if p['parent_id'].startswith('27eb79'))
    source=Path(p['source']['clean_case_path'])
    inspection=OUT/'part_inspection'/(p['parent_id']+'.json')
    c=next(x for x in read(inspection)['components'] if x['component_id']==1)
    with np.load(source,allow_pickle=False) as z:v=z['model_vertices'].astype(float);allf=z['source_faces'].astype(int)
    ids=np.array(c['face_ids']);f=allf[ids]
    edges=collections.defaultdict(list);neighbors=collections.defaultdict(set)
    for local,face in enumerate(f):
        for j in range(3):
            a,b=map(int,[face[j],face[(j+1)%3]])
            edges[tuple(sorted((a,b)))].append(int(ids[local]))
            neighbors[a].add(b);neighbors[b].add(a)
    boundary={x for e,fs in edges.items() if len(fs)==1 for x in e}
    # Existing global reference length, no inferred new normalization.
    from r2.data_tools.bench_parts_v1 import quality as Q
    limit=.0025*Q.L
    candidates=[]
    for (a,b),faces in edges.items():
        d=float(np.linalg.norm(v[a]-v[b]))
        if d>limit:continue
        opposing={int(x) for fi in faces for x in allf[fi] if int(x) not in (a,b)}
        common=neighbors[a]&neighbors[b]
        # Interior edges joining two boundary vertices can pinch a boundary.
        both_boundary=a in boundary and b in boundary
        link_ok=common==opposing and not(len(faces)==2 and both_boundary)
        target=v[a] if a in boundary else v[b] if b in boundary else (v[a]+v[b])/2
        affected=sorted({int(ids[i]) for i,face in enumerate(f) if a in face or b in face})
        candidates.append(dict(edge=[a,b],length_model=d,length_L=d/Q.L,incident_source_face_ids=faces,
            edge_incidence=len(faces),endpoint_boundary=[a in boundary,b in boundary],
            local_link_condition=bool(link_ok),common_neighbor_ids=sorted(common),opposite_ids=sorted(opposing),
            affected_source_face_ids=affected,collapse_target_model=target.tolist(),
            max_endpoint_motion_model=float(max(np.linalg.norm(v[a]-target),np.linalg.norm(v[b]-target))),
            source_endpoint_model=[v[a].tolist(),v[b].tolist()]))
    candidates.sort(key=lambda x:(x['length_model'],x['edge']))
    used=set();picked=[];removed=0
    for r in candidates:
        if not r['local_link_condition'] or set(r['edge'])&used:continue
        # Keep one-ring face sets disjoint so the original link checks don't
        # falsely certify interacting simultaneous collapses.
        if any(set(r['affected_source_face_ids'])&set(q['affected_source_face_ids']) for q in picked):continue
        picked.append(r);used.update(r['edge']);removed+=r['edge_incidence']
        if removed>=15:break
    payload=dict(schema='BENCH27_SHORTEDGE_READONLY_RECIPE_V1',parent_id=p['parent_id'],component_id=1,
        source_N=271,shell_N=101,L=float(Q.L),shortedge_limit_model=limit,
        constraints='Same original shell only; no width-spanning edge; topology link + no interior boundary pinch; disjoint affected-face neighborhoods; boundary vertex pinned, otherwise midpoint.',
        candidates=candidates,picked_for_possible_fixed_recipe=picked,
        predicted_removed_faces_if_all_collapses_pass=removed,predicted_total_N=271-removed,
        warning='No collapse, triangle deformation, finite-pair audit, fidelity probe, area or semantic admission has been executed. Original link checks alone are not final target validation.',
        protected={str(x.relative_to(ROOT)):sha(x) for x in [source,inspection,Path(__file__)]},
        target_written=False,source_mutations=0,model_calls=0)
    with out.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(candidates=len(candidates),valid_links=sum(x['local_link_condition'] for x in candidates),
        picked=[dict(edge=x['edge'],length=x['length_model'],incidence=x['edge_incidence']) for x in picked],
        possible_N=271-removed,collapse_executed=False)))

if __name__=='__main__':main()
