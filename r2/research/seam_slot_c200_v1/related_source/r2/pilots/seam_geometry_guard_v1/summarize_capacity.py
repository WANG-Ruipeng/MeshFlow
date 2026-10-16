"""Read-only source-slot/corner pairing of existing capacity-guard scores."""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C
from .capacity_guard import OUT, OLD, ARM, verify_registration

ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}


def read_edges(path,arrays):
    result={};all_rows=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for rr in csv.DictReader(f):
            r=dict(rr);r['valid_edge']=r['valid_edge']=='True';r['is_C_F']=r['is_C_F']=='True'
            r['incident_face_ids']=json.loads(r['incident_face_ids']);r['length_model']=float(r['length_model'])
            r['vertex_u']=int(r['vertex_u']);r['vertex_v']=int(r['vertex_v']);r['incidence']=int(r['incidence'])
            r['winding_opposite']=r['winding_opposite']=='True';r['rejected_reasons']=json.loads(r['rejected_reasons'])
            parts=[]
            for face_id in r['incident_face_ids']:
                pos=np.flatnonzero(np.isin(arrays['faces'][face_id],[r['vertex_u'],r['vertex_v']]))
                if len(pos)!=2:raise AssertionError('Edge does not identify two face corners')
                parts.append((int(arrays['source_face_ids'][face_id]),tuple(sorted(map(int,arrays['source_corner_ids'][face_id,pos])))))
            key=tuple(sorted(parts));r['source_identity']=key;all_rows.append(r)
            if r['valid_edge']:
                if key in result:raise AssertionError('Duplicate edge source identity')
                result[key]=r
    return result,all_rows


def pair_map(path,arrays):
    scans=C.read(path);k=int(arrays['known_faces']);ids=arrays['source_face_ids'];out={}
    for kind in ('CF','FF'):
        for row in scans[kind]['contacts']:
            if row['classification'] not in ILLEGAL:continue
            a,b=map(int,row['pair']);a=a if kind=='CF' else k+a;b=k+b
            key=tuple(sorted((int(ids[a]),int(ids[b]))));out[(kind,key)]=row['classification']
    return out


def role_pair(key,k,seam):
    def role(i):return 'C' if i<k else 'SEAM' if seam[i] else 'BULK'
    order={'C':0,'SEAM':1,'BULK':2}
    return '_'.join(sorted((role(key[0]),role(key[1])),key=order.get))


def missing_reason(key,arrays,all_edges):
    lookup={int(s):i for i,s in enumerate(arrays['source_face_ids'])};ends=[]
    for source,corners in key:
        if source not in lookup:return 'SOURCE_FACE_DELETED'
        row=lookup[source];pos=[int(np.flatnonzero(arrays['source_corner_ids'][row]==c)[0]) for c in corners]
        ends.append(tuple(sorted(map(int,arrays['faces'][row,pos]))))
    if len(set(ends))!=1:return 'SOURCE_EDGE_ENDPOINT_IDS_DISCONNECTED'
    e=next((r for r in all_edges if (r['vertex_u'],r['vertex_v'])==ends[0]),None)
    if e is None:return 'EDGE_NOT_RETAINED'
    return '|'.join(e['rejected_reasons']) or 'VALID_BUT_DIFFERENT_SOURCE_INCIDENTS'


def main():
    reg,m=verify_registration();edge_rows=[];pair_rows=[];face_rows=[];summary=[];protected={}
    for q in m['requests']:
        req=q['request_id'];k=q['K'];data={}
        for label,root in (('BASELINE',OLD),('CAPACITY_GUARD',OUT)):
            mesh=root/'evaluation'/ARM/req/'post/post.npz';folder=root/'evaluation/scores'/ARM/req/'POST'
            arrays=C.arrays(mesh);edges,all_edges=read_edges(folder/'valid_edges.csv',arrays)
            pairs=pair_map(folder/'pairs.json',arrays);score=C.read(folder/'score.json')
            data[label]=dict(arrays=arrays,edges=edges,all_edges=all_edges,pairs=pairs,score=score)
            for p in (mesh,folder/'valid_edges.csv',folder/'pairs.json',folder/'score.json'):protected[str(p)]=C.sha(p)
        a=data['BASELINE'];b=data['CAPACITY_GUARD'];raw_seam=a['arrays']['raw_seam_slot_mask']
        for key in sorted(a['edges'].keys()|b['edges'].keys()):
            ae=a['edges'].get(key);be=b['edges'].get(key)
            kind='CF' if any(s<k for s,_ in key) else 'FF'
            row=dict(request_id=req,parent_id=q['parent_id'],noise_index=q['noise_index'],edge_kind=kind,
                source_slot_corner_identity=key,status='RETAINED' if ae and be else 'LOST' if ae else 'NEW',
                baseline_length=ae['length_model'] if ae else None,candidate_length=be['length_model'] if be else None,
                baseline_opposite_winding=ae['winding_opposite'] if ae else None,
                candidate_opposite_winding=be['winding_opposite'] if be else None,
                lost_reason=missing_reason(key,b['arrays'],b['all_edges']) if ae and not be else '',
                gained_from_reason=missing_reason(key,a['arrays'],a['all_edges']) if be and not ae else '')
            edge_rows.append(row)
        for kind,key in sorted(a['pairs'].keys()|b['pairs'].keys()):
            ac=a['pairs'].get((kind,key));bc=b['pairs'].get((kind,key))
            pair_rows.append(dict(request_id=req,parent_id=q['parent_id'],noise_index=q['noise_index'],pair_kind=kind,
                source_slot_pair=key,role_pair=role_pair(key,k,raw_seam),
                status='RETAINED_ILLEGAL' if ac and bc else 'ELIMINATED' if ac else 'NEW_ILLEGAL',
                baseline_classification=ac,candidate_classification=bc))
        af=a['arrays'];bf=b['arrays'];slots=set(map(int,af['source_face_ids']))|set(map(int,bf['source_face_ids']))
        for sid in sorted(slots):
            aa=np.flatnonzero(af['source_face_ids']==sid);bb=np.flatnonzero(bf['source_face_ids']==sid)
            if bool(len(aa))==bool(len(bb)):continue
            def area(ary,row):
                if not len(row):return None
                t=ary['output'][row[0]].astype(float);return float(.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0])))
            face_rows.append(dict(request_id=req,source_slot_id=sid,role='SEAM' if raw_seam[sid] else 'BULK',
                baseline_kept=bool(len(aa)),candidate_kept=bool(len(bb)),baseline_area=area(af,aa),candidate_area=area(bf,bb),
                baseline_cleanup_reason=str(af['source_face_reason'][sid]),candidate_cleanup_reason=str(bf['source_face_reason'][sid])))
        local=[r for r in edge_rows if r['request_id']==req];prow=[r for r in pair_rows if r['request_id']==req]
        row=dict(request_id=req,parent_id=q['parent_id'],noise_index=q['noise_index'])
        for kind in ('CF','FF'):
            for status in ('RETAINED','LOST','NEW'):
                rs=[r for r in local if r['edge_kind']==kind and r['status']==status]
                row[f'{kind}_legal_edges_{status.lower()}']=len(rs)
                for stage in ('baseline','candidate'):row[f'{kind}_{status.lower()}_{stage}_length']=sum(r[f'{stage}_length'] or 0 for r in rs)
            for status in ('RETAINED_ILLEGAL','ELIMINATED','NEW_ILLEGAL'):
                rs=[r for r in prow if r['pair_kind']==kind and r['status']==status]
                row[f'{kind}_pairs_{status.lower()}']=len(rs)
        for label,obj in (('baseline',a),('candidate',b)):
            frac=obj['score']['seam_to_bulk']['free_only_reached_bulk_area_fraction']
            row[label+'_bulk_reach_fraction']=frac;row[label+'_bulk_nonzero']=frac>0;row[label+'_bulk_ge30']=frac>=.3
        summary.append(row)
    totals={key:sum(r[key] for r in summary) for key in summary[0] if key not in ('request_id','parent_id','noise_index','baseline_bulk_reach_fraction','candidate_bulk_reach_fraction')}
    roles=[]
    for key in sorted({(r['pair_kind'],r['role_pair'],r['status']) for r in pair_rows}):
        rr=[r for r in pair_rows if (r['pair_kind'],r['role_pair'],r['status'])==key]
        roles.append(dict(pair_kind=key[0],role_pair=key[1],status=key[2],pair_count=len(rr),outputs=len({r['request_id'] for r in rr})))
    C.csvwrite(OUT/'legal_edge_source_identity_ledger.csv',edge_rows)
    C.csvwrite(OUT/'illegal_pair_source_identity_ledger.csv',pair_rows)
    C.csvwrite(OUT/'face_retention_changes.csv',face_rows)
    C.csvwrite(OUT/'connectivity_pairing_per_output.csv',summary)
    C.csvwrite(OUT/'illegal_pair_role_summary.csv',roles)
    C.write(OUT/'connectivity_pairing_summary.json',dict(status='PASS',version='capacity_source_slot_corner_pairing_v1',
        source_code_sha256=C.sha(__file__),inputs=protected,totals=totals,role_summary=roles,
        identity='Shared edge identified by incident RAW source slots and the two original RAW corner IDs on each face, not changing POST face or vertex IDs.',
        model_calls=0,new_geometry_scans=0,new_model_calls=0))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
