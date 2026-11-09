import numpy as np
import torch
from ..exporter import MeshExporter
from ..legacy.deform import deform
from .deployment import predict_raw, named_quality, qualification, choose_delivery
from ..io_utils import array_hash

def test_original_cpu_rollout_and_final_query():
    torch.set_num_threads(2)
    ex=MeshExporter()
    base,faces=ex.template(512)
    y=np.zeros(2050,dtype=np.float64)
    mesh,meta=predict_raw(ex.flow,y,ex.support,base,faces)
    expected,path,alpha,_=deform(ex.flow,ex.proxy(y),base,faces,True,32)
    assert np.array_equal(mesh["u"],expected)
    assert np.array_equal(mesh["r"],ex.support.query(y,expected))
    assert meta["proxy_points_hash"]==array_hash(ex.proxy(y))
    assert np.array_equal(mesh["vertices"],expected*ex.support.query(y,expected)[:,None])

def test_unambiguous_quality_and_fallback_cannot_use_error():
    ex=MeshExporter();m,_=ex.export(np.zeros(2050),512,"fixed")
    q=named_quality(m["vertices"],m["faces"])
    assert "angle_q05_deg" not in q and "shape_quality_q05" not in q
    assert q["triangle_min_angle_p05_deg"]>1
    assert 0<q["triangle_shape_quality_p05"]<=1
    ok=qualification(q,{"valid":True},q)
    assert ok["qualified"]
    raw={"vertices":m["vertices"]*2,"faces":m["faces"]}
    got,decision=choose_delivery(raw,m,ok)
    assert got is raw and not decision["fallback_to_fixed"]
    failed=dict(q);failed["triangle_min_angle_p05_deg"]-=1e-3
    fail=qualification(failed,{"valid":True},q)
    got,decision=choose_delivery(raw,m,fail)
    assert got is m and decision["fallback_to_fixed"]
    assert decision["fallback_reasons"]==["triangle_min_angle_p05_deg"]

def test_invalid_geometry_and_face_count_are_not_qualified():
    ex=MeshExporter();m,_=ex.export(np.zeros(2050),512,"fixed")
    q=named_quality(m["vertices"],m["faces"])
    assert not qualification(q,{"valid":False},q)["qualified"]
    wrong=dict(q);wrong["face_count"]=510
    assert not qualification(wrong,{"valid":True},q)["qualified"]
