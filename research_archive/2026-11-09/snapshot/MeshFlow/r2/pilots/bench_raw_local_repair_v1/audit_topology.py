"""Independent indexed-edge reconstruction from saved meshes/contact caches."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import collections,csv,json,math,sys,time
from pathlib import Path
import numpy as np
from . import common as C

LEGAL={'shared_edge_contact','shared_vertex_contact','coplanar_boundary_or_tolerance_contact',
       'noncoplanar_edge_or_tangent_contact','point_or_tangent_contact'}


def table(p):
    with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def topology(mesh,pairs,k):
    v,f=mesh['vertices'],mesh['faces'];sf=mesh.get('parent_source_raw_slot',mesh.get('raw_slot_ids',mesh.get('source_face_ids')))
    sf=np.asarray(sf,int);sc=mesh.get('source_corner_ids',mesh.get('base_source_corner_ids'))
    if len(sc)<len(f):
        assert np.all(sf[len(sc):]<0)
        sc=np.vstack((sc,np.tile(np.arange(3),(len(f)-len(sc),1))))
    assert len(sf)==len(f)==len(sc)
    original={int(s):i for i,s in enumerate(sf) if s>=0}
    assert len(original)==int((sf>=0).sum())
    tri=v[f].astype(float);areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    duplicates=collections.defaultdict(list);inc=collections.defaultdict(list)
    for i,face in enumerate(f):
        duplicates[tuple(sorted(map(int,face)))].append(i)
        for a,b in ((0,1),(1,2),(2,0)):
            inc[tuple(sorted((int(face[a]),int(face[b]))))].append((i,a,b))
    duplicate_faces={i for group in duplicates.values() if len(group)>1 for i in group}
    contacts={kind:{tuple(map(int,r['pair'])):r['classification'] for r in pairs[kind]['contacts']} for kind in ('CF','FF')}
    all_edges={};valid={}
    for (u,w),occ in inc.items():
        if all(i<k for i,a,b in occ):continue
        key=tuple(sorted((int(sf[i]),*sorted((int(sc[i,a]),int(sc[i,b])))) for i,a,b in occ))
        kind='CF' if any(i<k for i,a,b in occ) else 'FF';length=float(np.linalg.norm(v[u].astype(float)-v[w]))
        reasons=[];classification='NA_NONMANIFOLD_OR_BOUNDARY'
        if len(occ)!=2:reasons.append('NOT_TWO_INCIDENT_FACES')
        else:
            (i,a,b),(j,c,d)=occ
            if i==j or u==w or length<=0 or areas[i]<=0 or areas[j]<=0:reasons.append('DEGENERATE')
            if i in duplicate_faces or j in duplicate_faces:reasons.append('DUPLICATE_FACE')
            pair=((i,j-k) if i<k else (j,i-k)) if kind=='CF' else tuple(sorted((i-k,j-k)))
            classification=contacts[kind].get(pair,'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
            if classification not in LEGAL:reasons.append('ILLEGAL_OR_UNKNOWN_CONTACT')
        row=dict(kind=kind,length=length,source_key=key,valid=not reasons,reasons=reasons,classification=classification)
        all_edges[(u,w)]=row
        if not reasons and all(part[0]>=0 for part in key):valid[key]=row
    return dict(valid=valid,all=all_edges,source_to_face=original,faces=f,source_corners=sc,slots=sf)


def fate(key,new):
    endpoints=[]
    for sid,c0,c1 in key:
        if sid not in new['source_to_face']:return 'SOURCE_FACE_REMOVED',None
        row=new['source_to_face'][sid];sc=new['source_corners'][row]
        j0=int(np.flatnonzero(sc==c0)[0]);j1=int(np.flatnonzero(sc==c1)[0])
        endpoints.append(tuple(sorted(map(int,new['faces'][row,[j0,j1]]))))
    if len(set(endpoints))!=1:return 'NO_LONGER_INDEX_SHARED',None
    r=new['all'].get(endpoints[0])
    if r is None:return 'SHARED_EDGE_NOT_FOUND',None
    return ('SHARED_BUT_INVALID' if not r['valid'] else 'SHARED_WITH_CHANGED_INCIDENT_IDENTITY'),r


def run():
    out=C.OUT/'audits/independent_topology_audit.json'
    if out.exists():r=C.read(out);C.verify(r['files']);return r
    started=time.perf_counter();manifest=C.read(C.OUT/'data/manifest.json')
    ap=C.OUT/'analysis/paired_original_legal_edges.csv';actual=table(ap)
    indexed={(r['request_id'],r['stage'],tuple(tuple(p) for p in json.loads(r['source_edge_key']))):r for r in actual}
    assert len(indexed)==len(actual)
    checked=0;lost=[];groups=[];files={str(ap):C.sha(ap)}
    for t in manifest['tasks']:
        rid=t['request_id'];k=t['K'];meshes={}
        for name,mp,pp in (
            ('RAW',t['assets']['RAW']['path'],t['assets']['RAW_PAIRS']['path']),
            ('IDENTITY_BASE',t['assets']['GEO_ONLY_BASE']['path'],t['assets']['GEO_ONLY_PAIRS']['path']),
            ('TARGET_BASE',C.OUT/'geometry'/rid/'post.npz',C.OUT/'geometry'/rid/'pairs.json'),
            ('IDENTITY_FINAL',C.OUT/'cpu/RAW_IDENTITY'/rid/'CLEAN_FINAL_POST/post.npz',C.OUT/'cpu/RAW_IDENTITY'/rid/'CLEAN_FINAL_POST/scores/pairs.json'),
            ('TARGET_FINAL',C.OUT/'cpu/LOCAL_REPAIR'/rid/'CLEAN_FINAL_POST/post.npz',C.OUT/'cpu/LOCAL_REPAIR'/rid/'CLEAN_FINAL_POST/scores/pairs.json')):
            meshes[name]=topology(C.arrays(mp),C.read(pp),k)
            files[str(mp)]=C.sha(mp);files[str(pp)]=C.sha(pp)
        for stage in ('BASE','FINAL'):
            old,new=meshes['IDENTITY_'+stage],meshes['TARGET_'+stage];oe,ne=old['valid'],new['valid']
            lc=0;gc=0
            for key in sorted(oe.keys()|ne.keys()):
                a,b=oe.get(key),ne.get(key);r=indexed[(rid,stage,key)]
                expected='RETAINED_LEGAL' if a and b else 'LOST_LEGAL' if a else 'GAINED_LEGAL'
                assert r['change']==expected and r['kind']==(a or b)['kind']
                for name,x in (('baseline_length_model',a),('target_length_model',b)):
                    assert (r[name]=='' if x is None else math.isclose(float(r[name]),x['length'],rel_tol=1e-12,abs_tol=1e-12))
                checked+=1
                if not b:
                    lc+=1;what,detail=fate(key,new)
                    lost.append(dict(request_id=rid,stage=stage,kind=a['kind'],source_edge_key=key,
                        baseline_length_model=a['length'],target_fate=what,target_edge_record=detail,
                        existed_as_legal_in_RAW=key in meshes['RAW']['valid'],
                        existed_as_legal_in_IDENTITY_BASE=key in meshes['IDENTITY_BASE']['valid'],
                        existed_as_legal_in_TARGET_BASE=key in meshes['TARGET_BASE']['valid'],
                        interpretation=('Missed control-final connection; not loss of a RAW/GEO legal edge'
                            if key not in meshes['RAW']['valid'] and key not in meshes['IDENTITY_BASE']['valid']
                            else 'Loss of earlier legal indexed connection; inspect saved stage provenance')))
                if not a:gc+=1
            groups.append(dict(request_id=rid,stage=stage,old_legal=len(oe),new_legal=len(ne),lost=lc,gained=gc))
    assert checked==len(actual)
    per_request=table(C.OUT/'analysis/per_request.csv')
    for r in per_request:
        for stage in ('BASE','FINAL'):
            value=next(g['lost'] for g in groups if g['request_id']==r['request_id'] and g['stage']==stage)
            assert int(r[stage+'_lost_original_legal_edges'])==value
    assert 'torch' not in sys.modules
    r=dict(status='PASS',requests=12,mesh_stages_reconstructed=60,paired_edge_rows=checked,
        reconstructed_definition='Indexed incidence-two, positive area, no indexed duplicate faces, cached resolved legal contact; winding separate; all original face identities only',
        results=groups,lost_edges=lost,BASE_lost=sum(g['lost'] for g in groups if g['stage']=='BASE'),
        FINAL_lost=sum(g['lost'] for g in groups if g['stage']=='FINAL'),
        lost_FINAL_edges_previously_legal_RAW_or_BASE=sum(x['stage']=='FINAL' and (x['existed_as_legal_in_RAW'] or x['existed_as_legal_in_IDENTITY_BASE']) for x in lost),
        detector_calls=0,model_calls=0,geometry_calls=0,scoring_calls=0,seconds=time.perf_counter()-started,
        source_hashes={str(Path(__file__)):C.sha(__file__)},files=files)
    C.write(out,r);return r


if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_hashes','results')},ensure_ascii=False))
