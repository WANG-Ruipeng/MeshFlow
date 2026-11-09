from pathlib import Path
import importlib.util
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location("audit",Path(__file__).with_name("data_audit.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from meshflow_control.evaluation.intersections import verified_pair
def fixture():
    t=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],dtype=np.float32)
    ix=m.edge_graph([(0,1,2),(3,4,5)],{i:tuple(p) for i,p in enumerate(t.reshape(-1,3))})
    faces=[tuple(tuple(float(v) for v in q) for q in f)for f in t]
    co=m.edge_graph(faces,{p:p for f in faces for p in f})
    return t,ix,co,np.array([.5,.5]),np.array([5,9])
def test_source_and_coordinate_full_edges_differ():
    t,ix,co,a,o=fixture()
    assert m.scope_topology(ix,[0,1],a,o,1)["component_count"]==2
    z=m.scope_topology(co,[0,1],a,o,1)
    assert z["component_count"]==1 and z["boundary_edge_count"]==4
    assert z["closed_degree2_boundary_components"]==1
def test_attachment_not_only_nearness_or_vertex_contact():
    t,ix,co,a,o=fixture()
    for graph,expected in [(ix,1),(co,0)]:
        ft=m.scope_topology(graph,[1],a,o,1)
        result=m.free_attachment(graph,[0],ft,a,o,1)
        assert result["unattached_F_component_count"]==expected
        assert result["unattached_F_area_fraction_of_F"]==expected
def test_component_face_provenance_and_area():
    t,ix,co,a,o=fixture()
    z=m.scope_topology(ix,[0,1],a,o,1)
    assert sorted(r["original_face_ids"] for r in z["components"])==[[5],[9]]
    assert all(r["area_fraction_parent"]==.5 for r in z["components"])
    assert "bad" not in z
def test_vertex_only_contact_has_separate_face_components_and_branch_boundary():
    faces=[(0,1,2),(0,3,4),(0,5,6)]
    coords={0:(0.,0.,0.),1:(1.,0.,0.),2:(0.,1.,0.),3:(-1.,0.,0.),4:(0.,-1.,0.),5:(0.,0.,1.),6:(1.,0.,1.)}
    g=m.edge_graph(faces,coords);z=m.scope_topology(g,[0,1,2],np.ones(3),np.arange(3),3)
    assert z["component_count"]==3 and z["boundary_branch_count"]==1
    assert z["boundary_branch_vertices"][0]["boundary_degree"]==6
def test_duplicate_and_angle_diagnostics():
    t,_,_,_,_=fixture()
    tri=np.stack([t[0],t[0][::-1],np.array([[0,0,0],[1,0,0],[.5,1e-5,0]])])
    q=m.basic_geometry(tri,[0,1,2],np.arange(3),1.000005)
    assert q["exact_duplicate_groups"]==[[0,1]]
    assert q["exact_zero_area_count"]==0
    assert q["minimum_angle_lt5_count"]==1
def test_zero_area_excluded_from_valid_angle_distribution():
    tri=np.array([[[0,0,0],[1,0,0],[2,0,0]]],dtype=float)
    q=m.basic_geometry(tri,[0],np.array([7]),0.)
    assert q["exact_zero_area_source_face_ids"]==[0]
    assert q["per_face"][0]["angles_degrees"] is None
    assert q["minimum_angle_lt5_fraction"] is None
@pytest.mark.parametrize("b,expected",[
    ([[1,0,0],[1,1,0],[0,1,0]],"shared_edge_contact"),
    ([[0,0,0],[-1,0,0],[0,-1,0]],"shared_vertex_contact"),
    ([[0,0,0],[1,0,0],[0,1,0]],"coplanar_positive_area_overlap"),
    ([[.25,.15,-.5],[.25,.75,.5],[.25,.9,-.5]],"proper_transverse_penetration"),
])
def test_actual_dual_detector_contact_taxonomy(b,expected):
    a=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float)
    z=verified_pair(a,np.array(b,dtype=float),1.)
    assert z["classification"]==expected and z["verified"] is True
def test_partition_counts_and_CF_orientation():
    contacts=[
      {"pair":[0,1],"classification":m.PROPER,"detector":{}},
      {"pair":[1,2],"classification":m.COPLANAR,"detector":{}},
      {"pair":[0,2],"classification":"UNRESOLVED","detector":{"primary":{"classification":"disjoint"},"independent":{"classification":"shared_edge_contact"}}}]
    z=m.partition_contacts({"contacts":contacts},[1],3)
    assert z["verified"]["CF"]["proper_pairs"]==[[1,0]]
    assert z["verified"]["CF"]["coplanar_positive_area_pairs"]==[[1,2]]
    assert z["verified"]["FF"]["unresolved_pairs"]==[[0,2]]
    assert sum(sum(g["classification_counts"].values())for g in z["verified"].values())==3
    assert z["primary"]["FF"]["classification_counts"]["disjoint"]==1
def test_nonfinite_is_recorded_blocked():
    t=np.zeros((1,3,3));t[0,0,0]=np.nan
    z=m.basic_geometry(t,[0],np.array([0]),1.)
    assert z["status"]=="BLOCKED_NONFINITE" and z["nonfinite_source_face_ids"]==[0]
