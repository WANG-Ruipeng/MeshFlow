"""Read-only source-slot topology/pair ledger; never reclassifies geometry."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from . import run as A
from r2.pilots.seam_slot_c200_v1 import generate as G


def graph_edges(root,q,post):
    table=A.rows(root/'evaluation/scores'/A.ARM/q['request_id']/'POST/valid_edges.csv')
    result={}
    for r in table:
        if r['valid_edge']!='True':continue
        i,j=map(int,json.loads(r['incident_face_ids']));u,v=int(r['vertex_u']),int(r['vertex_v'])
        identity=[]
        for fi in (i,j):
            corners=np.flatnonzero(np.isin(post['faces'][fi],[u,v]));assert len(corners)==2
            identity.append((int(post['raw_slot_ids'][fi]),tuple(sorted(int(c) for c in post['source_corner_ids'][fi,corners]))))
        key=tuple(sorted(identity));kind='CF' if min(i,j)<q['K'] else 'FF'
        assert key not in result
        result[key]=dict(kind=kind,length=float(r['length_model']))
    return result


def illegal(root,q,post):
    source=A.read(root/'evaluation/scores'/A.ARM/q['request_id']/'POST/pairs.json');result={}
    for kind in ('CF','FF'):
        items=set()
        for row in source[kind]['contacts']:
            if row['classification'] not in ('proper_transverse_penetration','coplanar_positive_area_overlap'):continue
            i,j=map(int,row['pair'])
            if kind=='CF':i=int(post['raw_slot_ids'][i]);j=int(post['raw_slot_ids'][q['K']+j])
            else:i=int(post['raw_slot_ids'][q['K']+i]);j=int(post['raw_slot_ids'][q['K']+j])
            items.add((min(i,j),max(i,j)))
        result[kind]=items
    return result


def main():
    completed=A.read(A.OUT/'completion.json');m=A.read(A.OLD/'eval_manifest.json');rows=[];detail=[];faces=[]
    for q in m['requests']:
        old=A.arrays(A.OLD/'evaluation'/A.ARM/q['request_id']/'post/post.npz')
        new=A.arrays(A.OUT/'evaluation'/A.ARM/q['request_id']/'post/post.npz')
        raw=A.arrays(A.OLD/'evaluation'/A.ARM/q['request_id']/'raw/raw.npz')
        before=A.arrays(A.OLD/'evaluation'/A.ARM/q['request_id']/'post/pre_geo.npz')['output']
        after=A.arrays(A.OUT/'evaluation'/A.ARM/q['request_id']/'post/pre_geo.npz')['output']
        moved=np.any(before.view(np.uint32)!=after.view(np.uint32),axis=-1)
        allowed=(raw['raw_anchor_ids']<0)&raw['raw_seam_slot_mask'][:,None]
        assert not np.any(moved&~allowed)
        assert G.same(old['C'],new['C']) and G.same(before[raw['raw_anchor_ids']>=0],after[raw['raw_anchor_ids']>=0])
        if not moved.any():assert old.keys()==new.keys() and all(G.same(old[k],new[k]) for k in old)
        oe,ne=graph_edges(A.OLD,q,old),graph_edges(A.OUT,q,new)
        oi,ni=illegal(A.OLD,q,old),illegal(A.OUT,q,new)
        row=dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                 moved_corners=int(moved.sum()),C_exact=True,anchors_exact=True,only_free_seam_tips_moved=True,
                 unchanged_all_POST_arrays=not bool(moved.any()))
        for kind in ('CF','FF'):
            ob={k for k,v in oe.items() if v['kind']==kind};nb={k for k,v in ne.items() if v['kind']==kind}
            lost,newedges=ob-nb,nb-ob
            row.update({kind+'_legal_edges_before':len(ob),kind+'_legal_edges_after':len(nb),
                        kind+'_legal_edges_lost':len(lost),kind+'_legal_edges_new':len(newedges),
                        kind+'_lost_original_length':sum(oe[x]['length'] for x in lost),
                        kind+'_new_final_length':sum(ne[x]['length'] for x in newedges),
                        kind+'_illegal_pairs_before':len(oi[kind]),kind+'_illegal_pairs_after':len(ni[kind]),
                        kind+'_illegal_pairs_removed':len(oi[kind]-ni[kind]),kind+'_illegal_pairs_added':len(ni[kind]-oi[kind])})
            for status,kk in [('LOST_LEGAL_EDGE',lost),('NEW_LEGAL_EDGE',newedges)]:
                for key in kk:detail.append(dict(request_id=q['request_id'],kind=kind,event=status,source_corner_identity=key))
            for status,kk in [('REMOVED_ILLEGAL_PAIR',oi[kind]-ni[kind]),('ADDED_ILLEGAL_PAIR',ni[kind]-oi[kind])]:
                for key in kk:detail.append(dict(request_id=q['request_id'],kind=kind,event=status,source_slot_pair=key))
        row['source_face_retention_exact']=bool(np.array_equal(old['raw_slot_ids'],new['raw_slot_ids']))
        assert row['source_face_retention_exact']
        rows.append(row)
    A.protect();A.C.csvwrite(A.OUT/'connectivity_pairing.csv',rows);A.C.csvwrite(A.OUT/'connectivity_changes.csv',detail)
    summary=dict(status='PASS',requests=16,C_exact=16,only_free_seam_tips_changed=True,
                 unchanged_mesh_array_requests=sum(x['unchanged_all_POST_arrays'] for x in rows),
                 face_retention_unchanged_requests=sum(x['source_face_retention_exact'] for x in rows),
                 totals={k:sum(x[k] for x in rows) for k in rows[0] if k.endswith(('_lost','_new','_removed','_added'))},
                 input_hashes=A.HASHES,source_classifier_replay=0,new_quality_scans=0,model_calls=0)
    A.C.write(A.OUT/'audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='input_hashes'}))


if __name__=='__main__':main()
