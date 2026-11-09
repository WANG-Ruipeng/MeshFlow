"""Synthetic CPU correctness tests; no sampling or source-data mutation."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from meshflow_control import geo_c
from . import floor,guard_geo

def run():
    tests=[];R=geo_c.RADIUS
    cap,_=floor.first_entry([2.,0,0],[-4.,0,0],1.)
    assert abs(cap-.25)<1e-12;tests.append('nonmonotone_first_crossing_not_late_feasible_endpoint')
    cap,_=floor.first_entry([-1.,1,0],[2.,0,0],1.)
    assert cap==1;tests.append('tangent_root_does_not_enter_forbidden_ball')
    cap,_=floor.first_entry([2.,0,0],[-1.5,0,0],1.)
    assert abs(cap-2/3)<1e-12;tests.append('single_entry')
    for j in range(5):
        raw=np.asarray([[[0,0,0],[1,0,0],[0,1,0]],[[2,0,0],[2+.01*j,.01,0],[2,1,0]],
            [[2+.002,0,0],[3,0,0],[3,1,0]]],np.float32)
        a=geo_c.process(raw,1);b=guard_geo.process(raw,1,protected_faces=np.zeros(3,bool))
        for k,v in a.items():
            if isinstance(v,np.ndarray):assert v.dtype==b[k].dtype and v.tobytes()==b[k].tobytes(),k
            elif k=='face_records':assert v==b[k]
    tests.append('guard_disabled_matches_original_all_arrays_5_fixtures')
    raw=np.asarray([[[0,0,0],[1,0,0],[0,1,0]],[[2,0,0],[2.01,0,0],[2,1,0]]],np.float32)
    a=geo_c.process(raw,1);b=guard_geo.process(raw,1,protected_faces=[False,True])
    assert not a['source_face_kept'][1] and b['source_face_kept'][1]
    assert b['stats']['anti_collapse_rejections']>0;tests.append('free_cluster_same_face_rejection')
    raw=np.asarray([[[0,0,0],[1,0,0],[0,1,0]],[[.01,0,0],[.02,0,0],[2,1,0]]],np.float32)
    a=geo_c.process(raw,1);b=guard_geo.process(raw,1,protected_faces=[False,True])
    assert not a['source_face_kept'][1] and b['source_face_kept'][1];tests.append('unique_C_anchor_same_face_rejection')
    raw=np.asarray([[[0,0,0],[1,0,0],[0,1,0]],[[.01,0,0],[0,0,0],[2,1,0]]],np.float32)
    b=guard_geo.process(raw,1,protected_faces=[False,True])
    assert b['source_face_kept'][1] and len(set(b['faces'][1]))==3
    tests.append('later_exact_C_corner_reserved_against_earlier_merge')
    raw=np.asarray([[[0,0,0],[1,0,0],[0,1,0]],[[2,0,0],[2.2,0,0],[2,.2,0]]],np.float32)
    proposal=raw.copy();proposal[1,1]=[2.001,0,0]
    anchors=np.full((2,3),-1,np.int64);anchors[0]=[0,1,2]
    out,mask,records,pre=floor.protect(raw,proposal,anchors,raw[0],1,R)
    assert records[1]['floor_triggered'] and floor.lengths(out[1]).min()>=1.25*R
    assert out[0].tobytes()==raw[0].tobytes();tests.append('floor_FP32_writeback_and_hard_C')
    assert 'torch' not in sys.modules
    return dict(status='PASS',tests=tests,model_calls=0,GPU_calls=0)
if __name__=='__main__':print(json.dumps(run(),indent=2))
