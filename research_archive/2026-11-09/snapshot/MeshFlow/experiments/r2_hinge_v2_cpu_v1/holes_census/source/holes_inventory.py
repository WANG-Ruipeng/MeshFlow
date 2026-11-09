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
