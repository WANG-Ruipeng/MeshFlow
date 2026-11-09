"""Small synthetic CPU checks; no model or original experiment mutation."""
import importlib,json
import numpy as np
from meshflow_control import geo_io
from meshflow_control.source_winding import capture_source
from r2.pilots.seam_geometry_guard_v1 import capacity_guard as CG
from r2.pilots.seam_slot_c200_v1 import evaluate as E
from . import engine


def check():
    vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    tri=np.stack([vertices,[[1,0,0],[0,0,0],[0,-.05,0]],[[1,-.01,0],[0,-.02,0],[.5,-.5,.2]]]).astype(np.float32)
    anchor=np.asarray([[0,1,2],[1,0,-1],[-1,-1,-1]],np.int64)
    raw=dict(known_faces=np.asarray(1),input_faces=np.asarray(3),raw_anchor_ids=anchor,
        raw_seam_slot_mask=np.array([False,True,False]),occupied=np.array([True]),
        edge_records=np.array([[0,0,1]],np.int64),raw_slot_edge_id=np.array([-1,0,-1]),
        occupancy_logits=np.array([1.],np.float32),C_vertices=vertices,
        C_faces_original=np.array([[0,1,2]]),C_faces_model=np.array([[0,1,2]]))
    source=capture_source(tri[:1],source_vertex_ids=raw['C_faces_original'],corner_permutation=np.array([[0,1,2]]))
    original,logs=CG.process(tri,1,anchor>=0,True)
    base=geo_io._validate_result(tri,1,original)
    maths=E.helpers()[0].cpu_math();classify=importlib.import_module(maths['geometry'].__package__+'.intersections').verified_pair
    canonical,final,summary,ledger,*_=engine.execute(tri,raw,base,logs,source,classify)
    assert summary['accepted']==1,(summary,ledger)
    assert len(final['faces'])==3
    assert np.array_equal(final['faces'][1,[0,2]],final['faces'][2,[0,1]])
    assert final['output'][:1].tobytes()==tri[:1].tobytes()
    # The exact same rejected source corner cannot reach a distant seam tip.
    tri2=tri.copy();tri2[1,2]=[0,-.3,0]
    original2,logs2=CG.process(tri2,1,anchor>=0,True)
    base2=geo_io._validate_result(tri2,1,original2)
    _,_,summary2,ledger2,*_=engine.execute(tri2,raw,base2,logs2,source,classify)
    assert summary2['accepted']==0
    assert summary2['funnel']['CLUSTER_OVER_ORIGINAL_RADIUS']==1
    return dict(status='PASS',synthetic_cases=2,model_calls=0,
                checks=['Valid seam-tip shared edge accepted inside original radius','Far tip rejected without radius relaxation',
                        'C coordinates and all face identities preserved','Illegal pair screening uses original independent verified detector'])


if __name__=='__main__':print(json.dumps(check(),indent=2))
