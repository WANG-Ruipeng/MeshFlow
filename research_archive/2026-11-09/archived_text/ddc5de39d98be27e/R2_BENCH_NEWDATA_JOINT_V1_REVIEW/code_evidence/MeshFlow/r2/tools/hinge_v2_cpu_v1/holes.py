"""Exact indexed three-open-edge cycles and guarded BRIDGE face insertion.

Inventory is read-only: no near-neighbour invented cycles, no geometry scans,
no bulk-only fill, no C boundary fill. Formal trials require separate freezing.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import itertools
import json
import copy
from pathlib import Path
import numpy as np
from r2.tools.hinge_opposite_v1 import engine as H

C,Q,PE,B=H.C,H.Q,H.PE,H.B
OUT=C.ROOT/'experiments/r2_hinge_v2_cpu_v1'
VERSION='exact_three_open_edge_cycles_bridge_v1'


def face_roles(mesh):
    if 'face_role' in mesh:return np.asarray(mesh['face_role']).astype(str)
    K=int(mesh['known_faces']);raw=np.asarray(mesh['raw_slot_ids'],np.int64)
    seam=np.asarray(mesh['raw_seam_slot_mask'],bool)[raw]
    return np.asarray(['C' if i<K else 'SEAM' if seam[i] else 'BULK' for i in range(len(raw))])


def inventory(mesh):
    v=np.asarray(mesh['vertices']);f=np.asarray(mesh['faces']);K=int(mesh['known_faces'])
    assert np.array_equal(v[f],mesh['output'])
    role=face_roles(mesh);inc=PE.incidence(f);opened={e:o[0] for e,o in inc.items() if len(o)==1}
    adj=collections.defaultdict(set)
    for u,w in opened:adj[u].add(w);adj[w].add(u)
    cycles=[]
    for a in sorted(adj):
        for b in sorted(x for x in adj[a] if x>a):
            for c in sorted(x for x in adj[a]&adj[b] if x>b):cycles.append((int(a),int(b),int(c)))
    existing={tuple(sorted(map(int,row))) for row in f};records=[]
    for cycle in cycles:
        a,b,c=cycle;edges=[tuple(sorted(e)) for e in ((a,b),(b,c),(c,a))]
        occ=[opened[e] for e in edges];neighbors=[int(z[0]) for z in occ]
        roles=[str(role[i]) for i in neighbors];reasons=[]
        if 'C' in roles:reasons.append('C_OPEN_BOUNDARY_FORBIDDEN')
        if 'SEAM' not in roles:reasons.append('NO_SEAM_NEIGHBOR')
        if cycle in existing:reasons.append('EXISTING_FACE')
        desired={(int(f[i,y]),int(f[i,x])) for i,x,y in occ}
        orientations=[]
        for tri in ((a,b,c),(a,c,b)):
            actual={(tri[z],tri[(z+1)%3]) for z in range(3)}
            if actual==desired:orientations.append(tri)
        if len(orientations)!=1:reasons.append('NO_UNIFIED_OPPOSITE_WINDING')
        tri=orientations[0] if len(orientations)==1 else cycle
        shape=PE.shape(v[np.asarray([tri])])
        if shape['area'][0]<=1e-12*C.L*C.L:reasons.append('BRIDGE_AREA_DEGENERATE')
        if shape['angle'][0]<5:reasons.append('BRIDGE_ANGLE_LT5')
        if shape['q'][0]>10:reasons.append('BRIDGE_Q_GT10')
        records.append(dict(cycle_vertices=list(cycle),oriented_bridge_vertices=list(tri),
            adjacent_face_rows=neighbors,adjacent_roles=roles,role_pattern='+'.join(sorted(roles)),
            distinct_adjacent_faces=len(set(neighbors)),minimum_angle_degrees=float(shape['angle'][0]),
            q=float(shape['q'][0]),area_model2=float(shape['area'][0]),
            structural_eligible='C' not in roles and 'SEAM' in roles and cycle not in existing,
            formal_eligible=not reasons,prefilter_reasons=reasons))
    return records


def initial(mesh):
    """Append-only indexed state; every new bridge has explicit absent lineage."""
    if 'face_role' in mesh:return {k:v.copy() for k,v in mesh.items()}
    n=len(mesh['faces']);K=int(mesh['known_faces'])
    out={k:v.copy() for k,v in mesh.items()}
    out.update(output_face_id=np.arange(n,dtype=np.int64),face_role=face_roles(mesh),
        parent_post_face_id=np.arange(n,dtype=np.int64),parent_source_raw_slot=mesh['raw_slot_ids'].copy(),
        base_output_face_count=np.asarray(n),base_free_face_count=np.asarray(n-K),
        bridge_mask=np.zeros(n,bool))
    return out


def append_face(mesh,triangle):
    out={k:v.copy() for k,v in mesh.items()};K=int(mesh['known_faces'])
    out['faces']=np.concatenate([mesh['faces'],np.asarray([triangle],mesh['faces'].dtype)])
    out['output']=out['vertices'][out['faces']];out['C']=out['output'][:K];out['free']=out['output'][K:]
    out['output_face_id']=np.append(mesh['output_face_id'],int(mesh['output_face_id'].max())+1)
    out['face_role']=np.append(mesh['face_role'],'BRIDGE')
    out['bridge_mask']=np.append(mesh['bridge_mask'],True)
    for name in ('parent_post_face_id','parent_source_raw_slot'):out[name]=np.append(mesh[name],-1)
    # Legacy arrays remain historical, not accidentally indexed as new faces.
    for name in ('source_face_ids','source_corner_ids','raw_slot_ids','retained_seam_mask','output_anchor_ids'):
        if name in out:out['base_'+name]=out.pop(name)
    out['index_domain']=np.asarray('HINGE_V2_APPEND_ONLY_BRIDGE_INDEX_V1')
    return out


def extend_pairs(old,cf_new,ff_new,K,old_free):
    result=copy.deepcopy(old)
    for kind,new in (('CF',cf_new),('FF',ff_new)):
        add=[]
        for r in new['contacts']:
            item=copy.deepcopy(r);i,j=map(int,r['pair'])
            item['pair']=[i,old_free];add.append(item)
        result[kind]['contacts'].extend(add)
        counts=collections.Counter(result[kind]['classification_counts']);counts.update(new['classification_counts'])
        tested=(K*(old_free+1) if kind=='CF' else old_free*(old_free+1)//2)
        proper={int(r['pair'][1]) for r in result[kind]['contacts'] if r['classification']=='proper_transverse_penetration'}
        if kind=='FF':proper={int(i) for r in result[kind]['contacts'] if r['classification']=='proper_transverse_penetration' for i in r['pair']}
        result[kind].update(tested_pairs=tested,expected_full_pairs=tested,classification_counts=dict(counts),
            proper_pair_count=counts['proper_transverse_penetration'],proper_affected_faces=sorted(proper),
            proper_affected_face_count=len(proper),proper_affected_face_denominator=old_free+1,
            proper_affected_face_fraction=len(proper)/(old_free+1),
            coplanar_positive_area_pairs=counts['coplanar_positive_area_overlap'],
            unresolved_or_degenerate_pairs=sum(n for key,n in counts.items() if 'unresolved' in key.lower() or 'degenerate' in key.lower()),
            incremental_provenance='Unchanged original pairs plus independently checked new bridge against all old C/F')
    return result


def decode(base,pairs):
    """Finite lexicographic queue; no candidate refresh, move, cleanup or model."""
    state=initial(base);K=int(state['known_faces']);current_pairs=copy.deepcopy(pairs)
    G=Q.helpers()['G'];oldgraph,oldedges,_=G.actual_valid_graph(state['vertices'],state['faces'],K,current_pairs)
    queue=[r for r in inventory(state) if r['structural_eligible']]
    records=[];accepted=0;scans=0
    def keys(edges):
        return {(int(e['vertex_u']),int(e['vertex_v']),tuple(map(int,e['incident_face_ids'])))
                for e in edges if e['valid_edge']}
    protected=keys(oldedges);edges=oldedges
    for rank,candidate in enumerate(queue):
        record=dict(candidate,rank=rank);cycle=tuple(candidate['cycle_vertices']);failed=[]
        now={tuple(r['cycle_vertices']):r for r in inventory(state)}
        if cycle not in now:failed=['NO_LONGER_THREE_OPEN_EDGE_CYCLE']
        else:failed=list(now[cycle]['prefilter_reasons'])
        trial=None;tp=None;te=None
        if not failed:
            tri=now[cycle]['oriented_bridge_vertices'];trial=append_face(state,tri)
            bridge=trial['output'][-1:];nfree=len(state['faces'])-K
            cf=Q.pair_scan(state['output'][:K],bridge,C.L)
            ff=Q.pair_scan(state['output'][K:],bridge,C.L);scans+=2
            legal=set(G.LEGAL)|{'disjoint'}
            illegal={kind:[r for r in scan['contacts'] if r['classification'] in B.ILLEGAL] for kind,scan in (('CF',cf),('FF',ff))}
            unknown={kind:[r for r in scan['contacts'] if
                          (r['classification'] not in legal and r['classification'] not in B.ILLEGAL)
                          or r.get('detector',{}).get('resolved') is False
                          or r.get('detector',{}).get('verified') is False]
                     for kind,scan in (('CF',cf),('FF',ff))}
            record.update(new_bridge_illegal_pairs=illegal,new_bridge_unknown_pairs=unknown)
            if illegal['CF']:failed.append('NEW_BRIDGE_CF_ILLEGAL')
            if illegal['FF']:failed.append('NEW_BRIDGE_FF_ILLEGAL')
            if any(unknown.values()):failed.append('NEW_BRIDGE_UNKNOWN_CONTACT')
            tp=extend_pairs(current_pairs,cf,ff,K,nfree)
            _,te,_=G.actual_valid_graph(trial['vertices'],trial['faces'],K,tp)
            if protected-keys(te):failed.append('LOST_OLD_LEGAL_EDGE')
            target=[e for e in te if len(trial['faces'])-1 in e['incident_face_ids']]
            record['bridge_edges']=target
            if len(target)!=3 or any(not e['valid_edge'] or not e['winding_opposite'] or e['incidence']!=2 for e in target):
                failed.append('BRIDGE_NOT_THREE_LEGAL_OPPOSITE_EDGES')
            if not np.array_equal(trial['vertices'],base['vertices']):raise AssertionError('Hole fill moved vertices')
            if not np.array_equal(trial['faces'][:len(base['faces'])],base['faces']):raise AssertionError('Hole fill changed original indices')
            if not np.array_equal(trial['output'][:len(base['faces'])].view(np.uint32),base['output'].view(np.uint32)):
                raise AssertionError('Hole fill changed old face coordinates')
        record['all_guard_failures']=list(dict.fromkeys(failed))
        record['decision']=failed[0] if failed else 'ACCEPTED'
        if not failed:
            state,current_pairs,edges=trial,tp,te;accepted+=1;protected|=keys(edges)
            record['new_bridge_output_face_id']=int(state['output_face_id'][-1])
        records.append(record)
    assert len(state['faces'])==len(base['faces'])+accepted
    summary=dict(version=VERSION,registered_structural_candidates=len(queue),accepted=accepted,
        decisions=dict(collections.Counter(r['decision'] for r in records)),new_bridge_faces=accepted,
        new_face_pair_scans=scans,full_old_pair_rescans=0,vertices_moved=0,old_faces_deleted=0,
        old_legal_edges_lost=len(keys(oldedges)-keys(edges)),model_calls=0)
    return state,current_pairs,edges,records,summary


def census():
    from r2.pilots.seam_hinge_quality_v1 import hinge as OLD
    target=OUT/'holes_census';done=target/'completion.json'
    if done.exists():r=C.read(done);B.verify(r['files']);return r
    C.resources(target/'resources_before.json')
    allrows=[];summaries=[];protected={}
    for q in OLD.requests():
        p=C.ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'/q['request_id']/'post.npz'
        protected[str(p)]=C.sha(p);mesh=C.arrays(p);rows=inventory(mesh)
        allrows.extend(dict(request_id=q['request_id'],split=q['split'],**r) for r in rows)
        summaries.append(dict(request_id=q['request_id'],split=q['split'],all_exact_cycles=len(rows),
            structural_eligible=sum(r['structural_eligible'] for r in rows),formal_eligible=sum(r['formal_eligible'] for r in rows),
            role_counts=dict(collections.Counter(r['role_pattern'] for r in rows)),
            eligible_role_counts=dict(collections.Counter(r['role_pattern'] for r in rows if r['formal_eligible']))))
    C.csvwrite(target/'cycles.csv',allrows);C.csvwrite(target/'per_request.csv',summaries)
    result=dict(status='COMPLETE',version=VERSION,requests=28,all_exact_cycles=len(allrows),
        structural_eligible=sum(r['structural_eligible'] for r in allrows),formal_eligible=sum(r['formal_eligible'] for r in allrows),
        eligible_by_split=dict(collections.Counter(r['split'] for r in allrows if r['formal_eligible'])),
        eligible_roles=dict(collections.Counter(r['role_pattern'] for r in allrows if r['formal_eligible'])),
        model_calls=0,full_geometry_scans=0,source_sha256=C.sha(__file__),protected=protected,
        files={str(target/n):C.sha(target/n) for n in ('cycles.csv','per_request.csv')})
    C.write(done,result);print(json.dumps({k:v for k,v in result.items() if k not in ('protected','files')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['census']);a=p.parse_args();census()
