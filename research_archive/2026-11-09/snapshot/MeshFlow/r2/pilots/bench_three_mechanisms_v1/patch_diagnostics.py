"""Read-only PATCH topology accounting from saved meshes and scorer edge caches.

No model, collision scan, mesh repair or semantic labels.  Forced within-patch
edges are reported separately from connections between independently generated
patches/singletons. Missing cached legality is UNKNOWN, never inferred from
shared indices. All source IDs below mean generated RAW slots, not source GT.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from . import common as C

VERSION = 'bench_patch_connectivity_accounting_v1'
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')


def read_edges(path):
    """Load the original actual_valid_graph output without reclassification."""
    path = Path(path)
    if not path.exists():
        return None
    result = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            result.append(dict(vertex_u=int(row['vertex_u']), vertex_v=int(row['vertex_v']),
                incident_face_ids=json.loads(row['incident_face_ids']),
                length_model=float(row['length_model']), valid_edge=row['valid_edge']=='True',
                winding_opposite=None if row['winding_opposite']=='' else row['winding_opposite']=='True',
                pair_classification=row['pair_classification'],
                rejected_reasons=json.loads(row['rejected_reasons'])))
    return result


def lineage(mesh):
    n = len(mesh['faces'])
    ids = np.asarray(mesh.get('parent_source_raw_slot', mesh.get('source_face_ids')), np.int64)
    corners = mesh.get('source_corner_ids', mesh.get('base_source_corner_ids'))
    if ids.shape != (n,) or corners is None:
        raise ValueError('Actual generated-slot/corner lineage unavailable')
    corners = np.asarray(corners, np.int64)
    if len(corners) < n:
        if np.any(ids[len(corners):] >= 0):
            raise ValueError('Missing corner identity on an original face')
        corners = np.vstack((corners, np.tile(np.arange(3), (n-len(corners),1))))
    if corners.shape != (n,3) or len(set(ids[ids>=0].tolist())) != int((ids>=0).sum()):
        raise ValueError('Ambiguous generated-slot lineage')
    return ids, corners


def diagnose_mesh(mesh, raw, edges=None):
    """Pure CPU arrays interface. No filesystem writes or geometry detector."""
    v = np.asarray(mesh['vertices']); f = np.asarray(mesh['faces'], np.int64)
    tri = np.asarray(mesh['output']); k = int(mesh['known_faces'])
    if not np.array_equal(v[f], tri): raise ValueError('Saved indexed geometry differs')
    raw_patch = np.asarray(raw['raw_patch_ids'], np.int64)
    pairs = np.asarray(raw['raw_shared_corner_pairs'], np.int64)
    state_ids = np.asarray(raw['raw_patch_state_ids'], np.int64)
    if state_ids.shape != (len(raw_patch),3): raise ValueError('Missing original shared-state identity')
    ids, corners = lineage(mesh)
    face_rows = {int(s):i for i,s in enumerate(ids) if s>=0}
    inc = collections.defaultdict(list)
    for i, face in enumerate(f):
        for a,b in ((0,1),(1,2),(2,0)):
            inc[tuple(sorted((int(face[a]), int(face[b]))))].append(i)
    cache = None
    if edges is not None:
        cache = {}
        for e in edges:
            key = tuple(sorted((e['vertex_u'],e['vertex_v'])))
            if key in cache or sorted(e['incident_face_ids']) != sorted(inc.get(key, [])):
                raise ValueError('Edge cache does not describe this indexed mesh')
            cache[key] = e
        # Original graph intentionally omits C-only edges and preserves repeated
        # incidents on degenerate triangles; match that exact accounting domain.
        expected={key for key,ii in inc.items() if any(i>=k for i in ii)}
        if set(cache) != expected: raise ValueError('Incomplete actual_valid_graph cache')

    def vertex(raw_corner):
        slot,local = divmod(int(raw_corner),3)
        if slot not in face_rows: return None
        row = face_rows[slot]; col = np.flatnonzero(corners[row]==local)
        if len(col)!=1: raise ValueError('Corner permutation unavailable')
        return int(f[row,int(col[0])])

    forced = []
    for pid in sorted(set(raw_patch[raw_patch>=0].tolist())):
        slots = np.flatnonzero(raw_patch==pid)
        pp = [p for p in pairs if int(p[0])//3 in slots and int(p[1])//3 in slots]
        if len(slots)!=2 or len(pp)!=2: raise ValueError('Not a fixed two-triangle shared-edge patch')
        if any(state_ids.flat[int(p[0])] != state_ids.flat[int(p[1])] for p in pp):
            raise ValueError('Saved shared state ID mismatch')
        endpoint = [[vertex(c) for c in p] for p in pp]
        record = dict(patch_id=int(pid), raw_slots=slots.tolist(), endpoint_pairs=endpoint,
            shared_index_edge=False, legal_edge=None, opposed_legal_edge=None,
            length_model=None, pair_classification='UNKNOWN', legality_status='UNKNOWN_NO_EDGE_CACHE')
        if any(a is None for pair in endpoint for a in pair):
            record.update(structural_status='FACE_DELETED', legality_status='NA_FACE_DELETED')
        elif any(a!=b for a,b in endpoint):
            record.update(structural_status='SPLIT_ENDPOINT', legality_status='NA_NO_SHARED_EDGE')
        elif endpoint[0][0]==endpoint[1][0]:
            record.update(structural_status='COLLAPSED_EDGE', legality_status='NA_COLLAPSED_EDGE')
        else:
            key = tuple(sorted((endpoint[0][0],endpoint[1][0])))
            record.update(structural_status='SHARED_INDEX_EDGE', shared_index_edge=True,
                length_model=float(np.linalg.norm(v[key[0]].astype(float)-v[key[1]])),
                indexed_incidence=len(inc[key]))
            if cache is not None:
                e=cache[key]
                unknown='UNKNOWN_OR_NOT_CONTACT' in e['rejected_reasons']
                record.update(legal_edge=None if unknown else bool(e['valid_edge']),
                    opposed_legal_edge=None if unknown else bool(e['valid_edge'] and e['winding_opposite']),
                    legality_status='UNKNOWN_CACHED_PAIR' if unknown else 'CACHED_ORIGINAL_GRAPH',
                    pair_classification=e['pair_classification'])
        forced.append(record)

    def role(i):
        if i<k:return 'C',-1
        sid=int(ids[i])
        if sid<0:return 'BRIDGE',-1
        if sid>=len(raw_patch):raise ValueError('Raw slot outside original identity domain')
        pid=int(raw_patch[sid]);return ('PATCH',pid) if pid>=0 else ('SINGLETON',-1)

    cross=[]; adj=collections.defaultdict(set); seeds=set()
    if cache is not None:
        for e in edges:
            if not e['valid_edge']:continue
            i,j=map(int,e['incident_face_ids']);ri,pi=role(i);rj,pj=role(j)
            if ri=='C' or rj=='C':
                seeds.add(max(i,j));category='C_'+(rj if ri=='C' else ri)
            else:
                adj[i].add(j);adj[j].add(i)
                if 'BRIDGE' in (ri,rj):category='BRIDGE_'+(rj if ri=='BRIDGE' else ri)
                elif ri==rj=='PATCH':category='FORCED_WITHIN_PATCH' if pi==pj else 'CROSS_PATCH'
                elif 'PATCH' in (ri,rj):category='PATCH_SINGLETON'
                else:category='SINGLETON_SINGLETON'
            cross.append(dict(category=category,output_faces=[i,j],raw_slots=[int(ids[i]),int(ids[j])],
                patch_ids=[pi,pj],length_model=e['length_model'],winding_opposite=bool(e['winding_opposite'])))
    area=.5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=1)
    summary=dict(registered_patches=len(forced),registered_shared_endpoint_pairs=len(pairs),
        forced_shared_edges=sum(r['shared_index_edge'] for r in forced),
        forced_edges_face_deleted=sum(r['structural_status']=='FACE_DELETED' for r in forced),
        forced_edges_split=sum(r['structural_status']=='SPLIT_ENDPOINT' for r in forced),
        forced_edges_collapsed=sum(r['structural_status']=='COLLAPSED_EDGE' for r in forced),
        forced_edges_unknown_legality=sum(r['legality_status'].startswith('UNKNOWN') for r in forced),
        legality_status='PASS_CACHED_ORIGINAL_GRAPH' if cache is not None else 'UNKNOWN_NO_EDGE_CACHE',
        forced_legal_edges=sum(r['legal_edge'] is True for r in forced) if cache is not None else None,
        forced_opposed_legal_edges=sum(r['opposed_legal_edge'] is True for r in forced) if cache is not None else None,
        forced_legal_length_model=sum(r['length_model'] for r in forced if r['legal_edge'] is True) if cache is not None else None,
        free_area_model2=float(area[k:].sum()),actual_free_faces=len(f)-k,
        full_geometry_scans=0,local_pair_classifications=0,topology_derivations=1,
        structural_guarantee_is_not_learned_connectivity=True)
    for cat in ('CROSS_PATCH','PATCH_SINGLETON','SINGLETON_SINGLETON','C_PATCH','C_SINGLETON','FORCED_WITHIN_PATCH'):
        ee=[e for e in cross if e['category']==cat]
        summary[cat+'_legal_count']=len(ee) if cache is not None else None
        summary[cat+'_legal_length_model']=sum(e['length_model'] for e in ee) if cache is not None else None
        summary[cat+'_opposed_legal_count']=sum(e['winding_opposite'] for e in ee) if cache is not None else None
    seen=set(seeds);todo=list(seeds)
    while todo:
        i=todo.pop()
        for j in adj[i]-seen:seen.add(j);todo.append(j)
    summary['C_reached_free_area_fraction']=float(area[sorted(seen)].sum()/area[k:].sum()) if cache is not None and area[k:].sum()>0 else None
    return summary,forced,cross


def paths(q,stage):
    rid=q['request_id'];ev=C.OUT/'evaluation/PATCH200'/rid
    score=C.OUT/'evaluation/scores/PATCH200'/rid/'RAW' if stage=='RAW' else C.OUT/'cpu/PATCH200'/rid/stage/'scores'
    mesh=ev/'raw/raw.npz' if stage=='RAW' else ev/'post/post.npz' if stage=='BASE_POST' else C.OUT/'cpu/PATCH200'/rid/stage/'post.npz'
    return mesh,score


def run():
    rows=[];forced=[];cross=[];projections=[];files={};missing=[]
    manifest=C.OUT/'data/eval_manifest.json';qs=C.read(manifest)['requests'];files[str(manifest)]=C.sha(manifest)
    for q in qs:
        rawpath=C.OUT/'evaluation/PATCH200'/q['request_id']/'raw/raw.npz'
        if not rawpath.exists():missing.append(str(rawpath));continue
        raw=C.arrays(rawpath);files[str(rawpath)]=C.sha(rawpath)
        meta={key:q[key] for key in ('request_id','condition_id','parent_id','noise_index','eval_group')}
        for stage in STAGES:
            mp,score=paths(q,stage);done=score/'complete.json';ep=score/'valid_edges.csv'
            if not mp.exists() or not done.exists():missing.extend(str(p) for p in (mp,done) if not p.exists());continue
            receipt=C.read(done)
            if receipt['status']!='PASS':raise ValueError('Unfinished original scorer '+str(done))
            if ep.exists():
                if not any(C.resolve(p)==ep and h==C.sha(ep) for p,h in receipt['files'].items()):raise ValueError('Unsealed cached edge table')
                files[str(ep)]=C.sha(ep)
            summary,ff,ee=diagnose_mesh(C.arrays(mp),raw,read_edges(ep))
            rows.append(dict(**meta,stage=stage,**summary))
            forced.extend(dict(**meta,stage=stage,**r) for r in ff)
            cross.extend(dict(**meta,stage=stage,**r) for r in ee)
            files[str(mp)]=C.sha(mp);files[str(done)]=C.sha(done)
        pp=rawpath.parent.parent/'post/shared_projection.json'
        if pp.exists():
            p=C.read(pp);files[str(pp)]=C.sha(pp)
            projections.append(dict(**meta,**{k:v for k,v in p.items() if k not in ('per_affected_face','affected_faces')},
                newly_degenerate_count=len(p['newly_degenerate_faces']),normal_gt60_count=len(p['affected_normal_gt60_faces'])))
        else:projections.append(dict(**meta,status='UNKNOWN_MISSING_D0_PROJECTION',path=str(pp)))
    if missing:
        return dict(status='BLOCKED_MISSING_ASSETS',missing=missing,available_rows=len(rows),
            expected_rows=len(qs)*3,files_written=0,model_calls=0,full_geometry_scans=0)
    out=C.OUT/'analysis/patch_diagnostics_v1'
    for name,data in (('per_output',rows),('forced_edges',forced),('legal_edges_by_role',cross),('d0_projection',projections)):
        C.csvwrite(out/(name+'.csv'),data)
    result=dict(status='COMPLETE',
        version=VERSION,rows=len(rows),expected_rows=len(qs)*3,missing=missing,
        unknown_legality_rows=sum(r['legality_status'].startswith('UNKNOWN') for r in rows),
        full_geometry_scans=0,local_pair_classifications=0,model_calls=0,
        scope='Original within-patch edges are forced; cross-patch edges are generated relations. Neither certifies visual adoption.',
        sources=files,code_sha256=C.sha(__file__))
    C.write(out/'summary.json',result)
    return result


def check_interface():
    # Synthetic exact shared edge; no source data, detector or output files.
    v=np.array([[0,0,1],[1,0,1],[0,1,1],[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],np.float32)
    f=np.array([[0,1,2],[3,4,5],[4,3,6]],np.int64)
    raw=dict(raw_patch_ids=np.array([-1,0,0]),raw_shared_corner_pairs=np.array([[3,7],[4,6]]),raw_patch_state_ids=f)
    mesh=dict(vertices=v,faces=f,output=v[f],known_faces=np.array(1),source_face_ids=np.arange(3),source_corner_ids=np.tile(np.arange(3),(3,1)))
    s,ff,_=diagnose_mesh(mesh,raw,None)
    assert s['forced_shared_edges']==1 and s['forced_legal_edges'] is None and ff[0]['legality_status'].startswith('UNKNOWN')
    inc=collections.defaultdict(list)
    for i,face in enumerate(f):
        for a,b in ((0,1),(1,2),(2,0)):inc[tuple(sorted((int(face[a]),int(face[b]))))].append(i)
    edges=[dict(vertex_u=a,vertex_v=b,incident_face_ids=ii,
        length_model=float(np.linalg.norm(v[a]-v[b])),valid_edge=len(ii)==2,winding_opposite=True,
        pair_classification='shared_edge_contact' if len(ii)==2 else 'NA_NONMANIFOLD_OR_BOUNDARY',
        rejected_reasons=[] if len(ii)==2 else ['NOT_TWO_INCIDENT_FACES'])
        for (a,b),ii in inc.items() if any(i>=1 for i in ii)]
    s,_,_=diagnose_mesh(mesh,raw,edges);assert s['forced_legal_edges']==1 and s['FORCED_WITHIN_PATCH_legal_count']==1
    split=dict(mesh,faces=f.copy(),vertices=np.vstack((v,v[3])))
    split['faces'][2,1]=7;split['output']=split['vertices'][split['faces']]
    s,_,_=diagnose_mesh(split,raw,None);assert s['forced_edges_split']==1
    return dict(status='PASS',checks=3,files_written=0,model_calls=0,geometry_scans=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-interface',action='store_true');a=p.parse_args()
    print(json.dumps(check_interface() if a.check_interface else run(),ensure_ascii=False))
