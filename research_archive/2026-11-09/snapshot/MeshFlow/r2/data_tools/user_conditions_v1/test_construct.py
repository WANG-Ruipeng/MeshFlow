import numpy as np
import pytest
from construct import nested_extents,partition_arrays

def fixture():
    vertices=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1],[0,-1,0]],np.float32)
    faces=np.array([[0,1,2],[1,3,2],[0,4,1]],np.int64)
    target=vertices[faces].reshape(-1,9)
    return dict(full_target=target,source_faces=faces,pre_ot_full=target.reshape(-1,3,3)/np.float32(2),
        retained_to_original_face=np.array([9,11,20]),source_face_vertex_ids=faces,
        source_corner_permutations=np.tile(np.arange(3),(3,1)))

def test_two_triangle_surface_deduplicates_and_is_nested():
    a=fixture();rows,audit=nested_extents(a["full_target"],a["source_faces"],[0,1],"seat")
    assert len(rows)==2
    assert rows[0]["requested_tiers"]==["small","medium"]
    assert rows[0]["region_area_fraction"]==.5 and rows[1]["region_area_fraction"]==1.
    assert set(rows[0]["source_face_ids"])<set(rows[1]["source_face_ids"])

def test_parent_partition_preserves_bits_and_original_ids():
    a=fixture();a["full_target"][0,0]=np.float32(-0.)
    a["pre_ot_full"]=a["full_target"].reshape(-1,3,3)/np.float32(2)
    result=partition_arrays(a,[1,0])
    assert result["C"].tobytes()==a["full_target"][[0,1]].reshape(-1,3,3).tobytes()
    assert result["Y"].tobytes()==a["full_target"][[2]].reshape(-1,3,3).tobytes()
    assert result["original_source_face_ids"].tolist()==[9,11]
    assert result["free_original_source_face_ids"].tolist()==[20]

def test_disconnected_region_fails():
    a=fixture()
    faces=a["source_faces"].copy();faces[1]=[7,8,9]
    with pytest.raises(ValueError,match="connected"):
        nested_extents(a["full_target"],faces,[0,1],"seat")
