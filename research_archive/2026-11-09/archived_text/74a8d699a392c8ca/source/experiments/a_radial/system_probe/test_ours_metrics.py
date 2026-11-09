"""Scientific metric and integrity checks without formal export or generation."""
import json
import numpy as np
import pytest
import torch
from ..io_utils import array_hash,file_hash
from ..layout_value.geometry import evaluate_mesh
from .ours_metrics import (registered_records,load_sample,verify_outlet_binding,load_mesh,
    quality_tails,extended_quantile,common_gate,paired_metrics,summary_groups)


def test_duplicate_unknown_and_missing_ids_are_distinguished():
    m={'requested':64,'records':[{'sample_id':f'{i:05d}','status':'ok'} for i in range(63)]}
    result=registered_records(m,'requested');assert len(result)==63 and '00063' not in result
    m['records'].append(m['records'][0].copy())
    with pytest.raises(ValueError,match='Duplicate'):registered_records(m,'requested')
    m['records'][-1]={'sample_id':'99999','status':'ok'}
    with pytest.raises(ValueError,match='Unregistered'):registered_records(m,'requested')


def test_sample_file_noise_latent_nodes_bind_to_one_registered_id(tmp_path):
    generator=torch.Generator(device='cpu').manual_seed(73510)
    noise=torch.randn((64,64),generator=generator).numpy()
    arrays={'noise':noise[0],'z_standardized':np.zeros(64,np.float32),
            'z':np.ones(64,np.float32),'nodes':np.zeros(2050,np.float32)}
    path=tmp_path/'sample_00000.npz';np.savez(path,**arrays)
    record={'sample_id':'00000','status':'ok','path':path.name,
            'array_hash':array_hash(*arrays.values()),'nodes_hash':array_hash(arrays['nodes'])}
    inputs={'sample_dir':tmp_path,'sample_records':{'00000':record},'expected_noise':noise}
    observed,identity=load_sample(inputs,'00000')
    assert np.array_equal(observed['noise'],noise[0]) and identity['sample_npz_sha256']==file_hash(path)
    wrong={**inputs,'expected_noise':noise[::-1].copy()}
    with pytest.raises(ValueError,match='fixed noise index'):load_sample(wrong,'00000')
    arrays['nodes'][0]=1.;np.savez(path,**arrays)
    with pytest.raises(ValueError,match='arrays changed'):load_sample(inputs,'00000')


def test_outlet_cannot_substitute_field_weight_dataset_or_A_seed():
    manifest={'ae_sha256':'ae','fm_sha256':'fm','dataset_sha256':'data','dataset_manifest_sha256':'manifest'}
    source={'sample_id':'00000','nodes_hash':'nodes'}
    row={'row':'ours_a','sample_id':'00000','status':'failed','source_bundle_sha256':'bundle',
         'source_nodes_sha256':'nodes',**manifest,'requested_faces':512,'a_weight_seed':10}
    inputs={'seed':10,'sample_manifest':manifest,'sample_manifest_sha256':'bundle','outlet_records':{'a':{'00000':row}}}
    assert verify_outlet_binding(inputs,'a','00000',source) is row
    for key in ['source_nodes_sha256','ae_sha256','fm_sha256','dataset_sha256','dataset_manifest_sha256']:
        original=row[key];row[key]='changed'
        with pytest.raises(ValueError,match='provenance mismatch'):verify_outlet_binding(inputs,'a','00000',source)
        row[key]=original
    row['a_weight_seed']=11
    with pytest.raises(ValueError,match='seed does not match'):verify_outlet_binding(inputs,'a','00000',source)


def test_qem_connectivity_is_independent_but_file_hash_is_strict(tmp_path):
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    f=np.array([[0,1,2],[1,3,2]],np.int32);path=tmp_path/'qem.npz';np.savez(path,vertices=v,faces=f)
    row={'status':'ok','mesh_path':path.name,'mesh_sha256':file_hash(path)}
    class NoRadialQuery:
        def query(self,*args):raise AssertionError('QEM must not be forced into radial connectivity')
    actual,faces,identity=load_mesh({'export_dir':tmp_path},row,'qem',np.zeros(2050),NoRadialQuery())
    assert faces.shape==(2,3) and not identity['connectivity_compared_to_fixed']
    assert identity['stored_array_hash']==array_hash(v,f)
    assert identity['canonical_geometry_hash']==array_hash(actual,faces)
    assert identity['stored_array_hash']!=identity['canonical_geometry_hash']
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='mesh file changed'):load_mesh({'export_dir':tmp_path},row,'qem',np.zeros(2050),NoRadialQuery())


def test_quality_tails_keep_degenerate_faces_and_extended_infinite_aspect():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.]])
    clean=quality_tails(v,np.array([[0,1,2]],np.int32))
    assert clean['triangle_min_angle_p01_deg']==pytest.approx(60)
    assert clean['shape_quality_p01']==pytest.approx(1)
    assert clean['aspect_ratio_p99']==pytest.approx(2/np.sqrt(3))
    mixed=quality_tails(v,np.array([[0,1,2],[0,0,1]],np.int32))
    assert mixed['face_count']==2 and mixed['degenerate_count']==1
    assert mixed['triangle_min_angle_p01_deg']==pytest.approx(.6)
    assert mixed['shape_quality_p01']==pytest.approx(.01)
    assert mixed['aspect_ratio_p95'] is None and mixed['aspect_ratio_p95_is_infinite']
    assert mixed['aspect_ratio_p99'] is None and mixed['aspect_ratio_infinite_face_count']==1
    assert extended_quantile([1.,2.,np.inf],0)==(1.,False)
    json.dumps(mixed,allow_nan=False)


def test_common_gate_uses_actual_faces_and_same_field_fixed_quality():
    floor={'face_count':512,'degenerate_count':0,'min_triangle_angle_deg':20.,'triangle_min_angle_p05_deg':30.}
    audit={'valid':True}
    assert common_gate(floor,audit,floor,audit)['qualified']
    bad={**floor,'face_count':510}
    assert common_gate(bad,audit,floor,audit)['failure_reasons']==['actual_face_budget']
    bad={**floor,'min_triangle_angle_deg':19.99999}
    assert not common_gate(bad,audit,floor,audit)['qualified']
    assert common_gate(floor,audit,None,None)['qualified'] is None
    assert not common_gate(floor,audit,floor,{'valid':False})['gate_available']


def test_triangle_surface_distance_is_analytic_plane_gap_not_point_cloud_nn():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    f=np.array([[0,1,2],[1,3,2]],np.int64);reference=v+np.array([0.,0.,.2])
    before=np.random.get_state();a=evaluate_mesh(v,f,reference,f,points=128,seed=993501);after=np.random.get_state()
    assert a['rms_bbox_percent']==pytest.approx(100*.2/np.sqrt(2),abs=1e-11)
    assert a['mesh_to_reference_rms_bbox_percent']==pytest.approx(a['reference_to_mesh_rms_bbox_percent'],abs=1e-11)
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    b=evaluate_mesh(v,f,reference,f,points=128,seed=1093501)
    assert b['rms_world_units']==pytest.approx(.2,abs=1e-12)


def test_paired_median_is_per_sample_and_failure_denominator_is_fixed():
    def row(i,arm,values):
        return {'seed':10,'sample_id':f'{i:05d}','arm':arm,'source_nodes_sha256':str(i),
            'metric_status':'ok','qualified':True,'gate_available':True,'audit':{'valid':True},
            'quality':{'face_count':512},'surface_metrics':[{'seed':993501+i,'rms_bbox_percent':values[0]},
                                                        {'seed':1093501+i,'rms_bbox_percent':values[1]}]}
    rows=[]
    for i,(base,actual) in enumerate([(1.,.9),(100.,50.),(1000.,1000.)]):
        candidate=row(i,'a',[actual,actual]);fixed=row(i,'fixed',[base,base]);candidate.update(paired_metrics(candidate,fixed));rows.append(candidate)
    g=next(g for g in summary_groups(rows) if g['seed']==10 and g['arm']=='a')
    assert g['planned_samples']==64 and g['metric_failed_or_missing_count']==61
    assert g['paired_gain_percent_median_two_measurements']==pytest.approx([10.,10.])
    candidate=row(4,'a',[.97,.99]);fixed=row(4,'fixed',[1.,1.])
    assert not paired_metrics(candidate,fixed)['qualified_ge2pct_both']
    candidate['source_nodes_sha256']='other'
    with pytest.raises(ValueError,match='different saved mainfields'):paired_metrics(candidate,fixed)
