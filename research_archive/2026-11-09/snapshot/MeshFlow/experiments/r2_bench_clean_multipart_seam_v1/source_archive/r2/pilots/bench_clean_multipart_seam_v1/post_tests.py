"""Small deterministic checks of new audit plumbing, not new model trials."""
from __future__ import annotations
import numpy as np
from . import topology_audit as T
from . import post_cpu as P


def run():
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    a=np.asarray([0.,0,0]);b=np.asarray([1.,0,0])
    check("finite_segments_parallel_gap",abs(T.segment_distance(a,b,np.asarray([0.,.04,0]),np.asarray([1.,.04,0]))-.04)<1e-12)
    check("finite_segments_skew_cross_interior",T.segment_distance(a,b,np.asarray([.5,-1,0]),np.asarray([.5,1,0]))<1e-12)
    check("finite_segments_not_infinite_lines",abs(T.segment_distance(a,b,np.asarray([2.,0,0]),np.asarray([3.,0,0]))-1)<1e-12)
    v=np.asarray([[0,0,0],[1,0,0],[1,.04,0],[0,.04,0]],float)
    e=np.asarray([[0,0,1],[1,2,3]])
    flags,rows=T.slit_pairs(v,e,np.arange(4))
    check("parallel_opposed_slit_accept",len(rows)==1 and flags.all())
    _,rows=T.slit_pairs(v,np.asarray([[0,0,1],[1,3,2]]),np.arange(4))
    check("same_direction_not_called_opposed",len(rows)==0)
    _,rows=T.slit_pairs(v,e,np.asarray([0,1,2,0]))
    check("shared_source_endpoint_excluded",len(rows)==0)
    # Two disconnected coincident-looking triangles remain two components when
    # original source identities differ, instead of coordinate welding.
    _,count,_=T.components(np.asarray([[0,1,2],[3,4,5]]))
    check("source_identity_connectivity",count==2)
    faces=np.asarray([[0,1,2],[0,1,3]])
    sf=np.asarray([4,7]);sc=np.tile(np.arange(3),(2,1))
    old=P.H.two_face_edges(faces,sf,sc,0)
    renumber=np.asarray([5,8,12,14]);new=P.H.two_face_edges(renumber[faces],sf,sc,0)
    check("source_halfedge_identity_invariant_to_vertex_representative",set(old)==set(new))
    check("old_same_winding_not_new_after_representative_relabel",
          len(P.H.new_same_winding_edges(faces,renumber[faces],sf,sc,0))==0)
    nm=np.asarray([[0,1,2],[1,0,3],[0,1,4]])
    sf=np.asarray([2,4,7]);sc=np.tile(np.arange(3),(3,1))
    renumber=np.asarray([5,8,12,14,16])
    check("old_nonmanifold_identity_invariant",P.H.nm_keys(nm,sf,sc)==P.H.nm_keys(renumber[nm],sf,sc))
    result=dict(status="PASS",checks=checks,checks_count=len(checks),model_calls=0,GPU_calls=0,
                geometry_generation="Only tiny analytic validator fixtures, no experimental generation")
    P.C.write(P.ROOT/"post_cpu_tests.json",result)
    return result


if __name__=="__main__":print(run())
