"""Compact CPU-only metric, contract and render tests; no private asset dependency."""
import sys
import numpy as np
from meshflow_control.evaluation.surface import surface_distance_squared,sample_mesh_surface
from meshflow_control.evaluation.pair_audit import pair_scan
from meshflow_control.evaluation.evaluate import open_reference,_contracts
from meshflow_control.evaluation import control
from meshflow_control.evaluation.hashing import array_sha
from meshflow_control.evaluation.io import save_npz,write_json,digest
from meshflow_control.evaluation.registry import records
def triangles():
    C=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
    F=C+np.array([0,0,.001],np.float32)
    return C,F
def test_bounded_surface_and_sampler():
    C,_=triangles();d=surface_distance_squared([[.2,.2,1],[2,0,0]],C)
    np.testing.assert_allclose(d,[1,1])
    a=sample_mesh_surface(C,16,np.random.RandomState(10));b=sample_mesh_surface(C,16,np.random.RandomState(10))
    assert array_sha(a)==array_sha(b)
def test_no_torch_import():
    import subprocess
    code="import sys;import meshflow_control.evaluation.evaluate;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-B','-c',code],check=True)
def test_pair_contact_and_duplicate():
    C,_=triangles()
    shared=np.array([[[1,0,0],[0,0,0],[0,-1,0]]],np.float32)
    a=pair_scan(C,shared,1.,independent=True);b=pair_scan(C,C,1.,independent=True)
    assert a['proper_pair_count']==0 and a['coplanar_positive_area_pairs']==0
    assert b['coplanar_positive_area_pairs']==1
def test_open_I_has_no_GT():
    C,F=triangles();c=dict(C=C,L0=1.)
    ref=open_reference(c,'synthetic_no_gt')
    assert 'GT_visibility' not in ref['arrays']
    m,_=control.conflict_metrics(F,c,ref,control.trusted_cf(C,F,1.))
    assert 0<m['ConflictCov_C']<=m['ConflictCov_C_upper']<=1
def test_C_contract_bytes_and_all51():
    C,F=triangles();a=np.concatenate([C,F]).reshape(2,9);z=np.zeros((1,2,9),np.float32)
    r=dict(N=2,K=1);b=dict(C=C,z=z)
    raw=dict(raw=a,trajectory=np.repeat(a[None],51,axis=0),z=z)
    _,ok=_contracts(r,raw,b,{})
    assert ok['C_output_bitwise'] and ok['C_all51_bitwise']
    raw['raw']=a.copy();raw['raw'][0,0]=np.nextafter(np.float32(0),np.float32(1))
    assert not _contracts(r,raw,b,{})[1]['C_output_bitwise']
def test_registry_P2_NA_and_exposure_domains():
    r=dict(request_id='r',label='START',panel='P2',cohort='OPEN',condition_id='c',parent_id='c',candidate=0,N=192,K=2,L0=9.,
        metrics=dict(SeamCov=None,ExposedRetain=None,ConflictCov_C=.1,ConflictCov_C_upper=.2),metric_status={})
    rs={v['metric_id']:v for v in records(r)}
    assert rs['SeamCov']['status']=='NA_NOT_APPLICABLE'
    assert rs['ConflictCov_C']['upper']==.2 and not rs['ConflictCov_C']['requires_GT']
def test_render_complete_guard_and_pixels(tmp_path):
    from meshflow_control.evaluation.render import render
    from PIL import Image
    C,F=triangles();t=np.concatenate([C,F])
    raw=tmp_path/'raw.npz';cond=tmp_path/'condition.npz';ref=tmp_path/'reference.json'
    save_npz(raw,dict(raw=t.reshape(-1,9)));save_npz(cond,dict(C=C))
    write_json(ref,dict(condition_path=str(cond)))
    row=dict(label='START',panel='P2',candidate=0,condition_id='synthetic',K=1,raw_path=str(raw),raw_sha256=digest(raw),reference_path=str(ref))
    blocked=render(dict(rows=[row]),tmp_path/'blocked')
    assert blocked['status']=='BLOCKED_INCOMPLETE_FIXED_GALLERY'
    actual=render(dict(rows=[row]),tmp_path/'rendered',allow_partial=True)
    im=Image.open(actual['pages'][0]['blind_page'])
    assert im.width>1000 and im.height>500 and actual['mesh_modified'] is False
