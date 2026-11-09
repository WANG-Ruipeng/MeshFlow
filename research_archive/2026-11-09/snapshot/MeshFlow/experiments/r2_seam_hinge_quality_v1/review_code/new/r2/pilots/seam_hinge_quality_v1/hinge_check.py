"""Small engineering contracts; no learned model, no experiment samples."""
import numpy as np
from . import hinge as H


def mesh():
    v=np.asarray([[10,10,10],[11,10,10],[10,11,10],[0,0,0],[1,0,0],[0,1,0],[1,.04,0],[0,-1,0]],np.float32)
    f=np.asarray([[0,1,2],[3,4,5],[6,3,7]],np.int64);tri=v[f]
    anchor=np.full((3,3),-1,np.int64);anchor[0]=np.arange(3)
    return dict(vertices=v,faces=f,known_faces=np.asarray(1),input_faces=np.asarray(3),output=tri,C=tri[:1],free=tri[1:],
        source_face_ids=np.arange(3),source_corner_ids=np.tile(np.arange(3),(3,1)),raw_slot_ids=np.arange(3),
        corner_vertex_ids=f.copy(),raw_seam_slot_mask=np.zeros(3,bool),raw_anchor_ids=anchor,
        output_anchor_ids=anchor.copy(),source_face_kept=np.ones(3,bool),source_face_reason=np.asarray(['kept']*3),duplicate_of_source_face=np.full(3,-1))


def main():
    x=mesh();tri=x['output'];pairs=H.scan(tri,1)
    out,records,summ,before,after,newpairs=H.decode(x,tri.copy(),tri.copy(),pairs,.015)
    assert summ['accepted']==1 and len(out['faces'])==3
    assert np.array_equal(out['C'],tri[:1])
    assert out['faces'][1,1]==out['faces'][2,0]
    pp=H.Q.compare_ledgers(before,after,pairs,newpairs)['summary']
    assert pp['FF_new_legal_edges']==1 and pp['FF_new_illegal_pairs']==0
    assert pp['CF_lost_legal_edges']==pp['FF_lost_legal_edges']==0
    wrong=mesh();wrong['faces'][2]=wrong['faces'][2,::-1]
    wrong['source_corner_ids'][2]=wrong['source_corner_ids'][2,::-1]
    candidates,filters,_,_=H.inventory(wrong,tri,tri,.015)
    assert not candidates and filters['same_winding']>0
    candidates,filters,_,_=H.inventory(x,tri,tri,.001)
    assert not candidates and filters['endpoint_gap_over_delta']>0
    pre=tri.copy();pre[1,1,2]=1.
    out,records,summ,_,_,_=H.decode(x,pre,tri,pairs,.015)
    assert summ['accepted']==0 and any(r['decision']=='CLUSTER_PRE_GEO_OVER_BUDGET' for r in records)
    H.C.write(H.OUT/'engineering_check.json',dict(status='PASS',tests=['actual_midpoint_shared_edge','fixed_C','zero_new_illegal','no_deleted_faces','opposite_winding_filter','endpoint_budget','whole_PRE_GEO_member_budget'],model_calls=0,code_sha256=H.C.sha(H.__file__),test_sha256=H.C.sha(__file__)))
    print('PASS: 7 CPU geometry contract checks')


if __name__=='__main__':main()
