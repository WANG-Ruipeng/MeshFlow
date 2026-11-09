"""CPU contracts for fixed-final artifacts; no formal sampling or CUDA launch."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from ..io_utils import file_hash,write_json
from . import artifacts,metrics


def final_fixture():
    p={'jobs':{'10':{'ae_sha256':'ae10','parent_sha256':'fm8000'}},
       'data_manifest_sha256':'data_manifest','source_manifest_sha256':'source_manifest',
       'source_data_hashes':{'radial_train.npz':'radial','DATASET.json':'dataset'}}
    config={'seed':10,'protocol_sha256':'tail_protocol','ae_sha256':'ae10','source_fm_sha256':'fm8000',
        'stage':'fm','steps':64000,'initial_step':8000,'final_step':64000,
        'dataset_sha256':'radial','dataset_manifest_sha256':'dataset','batch':128,'lr':1e-4,
        'scope':artifacts.SCOPE,'device':'cuda','data_manifest_sha256':'data_manifest','source_manifest_sha256':'source_manifest'}
    state={'scope':artifacts.SCOPE,'step':64000,'is_predeclared_final':True,'config':config,
        'scheduler':{'last_epoch':64000,'base_lrs':[1e-4]},
        'optimizer':{'state':{0:{'step':torch.tensor(64000.)}},'param_groups':[{'lr':1e-4,'initial_lr':1e-4}]},
        'parent_exposures_each':512000,'draw_queue':[],'trace':[{'step':i} for i in range(8001,64001)],
        'rng':{'torch_cuda':[torch.zeros(16,dtype=torch.uint8)]},'batch_rng':torch.zeros(16,dtype=torch.uint8),
        'boundary':{'complete_update':True,'pending_data':False}}
    return p,state


def test_only_declared_final_with_complete_state_can_be_sampled():
    p,state=final_fixture();artifacts.validate_final_state(state,p,10,'tail_protocol')
    for field,value in [('step',8000),('step',32000),('is_predeclared_final',False),
                        ('parent_exposures_each',64000),('draw_queue',[0]),('trace',[]),
                        ('boundary',{'complete_update':False,'pending_data':True})]:
        changed={**state,field:value}
        with pytest.raises(AssertionError):artifacts.validate_final_state(changed,p,10,'tail_protocol')
    assert not torch.cuda.is_initialized()


def test_final_weight_ae_source_data_seed_scheduler_and_trace_binding():
    p,state=final_fixture()
    for key in ('seed','protocol_sha256','ae_sha256','source_fm_sha256','dataset_sha256','dataset_manifest_sha256',
                'data_manifest_sha256','source_manifest_sha256','device'):
        changed={**state,'config':{**state['config'],key:'wrong'}}
        with pytest.raises(AssertionError,match='binding'):artifacts.validate_final_state(changed,p,10,'tail_protocol')
    wrong={**state,'scheduler':{'last_epoch':64000,'base_lrs':[1e-3]}}
    with pytest.raises(AssertionError,match='scheduler'):artifacts.validate_final_state(wrong,p,10,'tail_protocol')
    wrong={**state,'optimizer':deepcopy(state['optimizer'])};wrong['optimizer']['state'][0]['step']=torch.tensor(63999.)
    with pytest.raises(AssertionError,match='Adam'):artifacts.validate_final_state(wrong,p,10,'tail_protocol')
    wrong={**state,'trace':state['trace'][::-1]}
    with pytest.raises(AssertionError,match='reordered'):artifacts.validate_final_state(wrong,p,10,'tail_protocol')


def input_fixture():
    identity={'scope':artifacts.ARTIFACT_SCOPE,'AE_step':3200,'FM_step':64000,
        'training_protocol_sha256':'tail_protocol','artifact_protocol_sha256':'artifact_protocol',
        'original_A_sha256':'a10','support_sha256':'support','ae_sha256':'ae10','fm_sha256':'fm64000'}
    manifest={'fm_steps':32,'fm_nfe':32,'ae_sha256':'ae10','fm_sha256':'fm64000',
        'dataset_sha256':'radial','dataset_manifest_sha256':'dataset'}
    return {'seed':10,'primary':True,'identity':identity,'sample_manifest':manifest,
        'sample_manifest_sha256':'bundle','sample_records':{},'checkpoint_sha256':{'ae':'ae10','fm':'fm64000'}}


def test_auxiliary_cannot_enter_primary_table_or_replace32step_fields():
    inputs=input_fixture();assert metrics.primary_only(inputs) is inputs
    wrong={**inputs,'primary':False}
    with pytest.raises(ValueError,match='Auxiliary'):metrics.primary_only(wrong)
    wrong={**inputs,'sample_manifest':{**inputs['sample_manifest'],'fm_steps':128,'fm_nfe':128}}
    with pytest.raises(ValueError,match='Euler32'):metrics.primary_only(wrong)
    wrong={**inputs,'identity':{**inputs['identity'],'FM_step':8000}}
    with pytest.raises(ValueError,match='final generation'):metrics.primary_only(wrong)


def test_export_manifests_are_bound_to_exact_primary_bundle_and_A_asset(tmp_path,monkeypatch):
    inputs=input_fixture();monkeypatch.setattr(artifacts,'view_samples',lambda *a,**k:inputs)
    folder=artifacts.seed_folder(tmp_path,10);export=folder/'exports';hashes={}
    for arm in metrics.common.ARMS:
        path=export/f'{arm}_MANIFEST.json'
        write_json(path,{'row':'ours_'+arm,'requested_samples':64,'records':[]});hashes[arm]=file_hash(path)
    record={'schema':artifacts.SCHEMA,'scope':artifacts.ARTIFACT_SCOPE,
        'training_protocol_sha256':'tail_protocol','artifact_protocol_sha256':'artifact_protocol',
        'original_A_sha256':'a10','support_sha256':'support','source_bundle_sha256':'bundle',
        'one_primary_mainfield_three_outlets':True,'outlet_manifest_sha256':hashes}
    path=folder/'EXPORT_IDENTITY.json';write_json(path,record)
    result=metrics.seed_inputs(tmp_path,10);assert all(not x for x in result['outlet_records'].values())
    for field in ('source_bundle_sha256','original_A_sha256','artifact_protocol_sha256','support_sha256'):
        write_json(path,{**record,field:'changed'},replace=True)
        with pytest.raises(ValueError,match='Export identity'):metrics.seed_inputs(tmp_path,10)
    write_json(path,record,replace=True)
    with (export/'a_MANIFEST.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='manifest changed'):metrics.seed_inputs(tmp_path,10)


def test_same_noise_pairing_computes_node_rms_and_nearest_support_only():
    a={'noise':np.arange(64,dtype=np.float32),'nodes':np.zeros(2050,np.float32)}
    b={'noise':a['noise'].copy(),'nodes':np.ones(2050,np.float32)*.25}
    training=np.arange(16,dtype=np.float32)[:,None]*np.ones((16,2050),np.float32)*.1
    result=artifacts.compare_arrays(a,b,training)
    assert result['same_epsilon'] and result['paired_logradius_node_RMS']==pytest.approx(.25)
    assert result['first_nearest_parent_index']==0 and result['first_nearest_training_logradius_RMS']==0
    assert result['second_nearest_training_logradius_RMS']==pytest.approx(.05,abs=1e-7)
    assert result['not_a_surface_distance_or_generalization_metric']
    bad={**b,'noise':b['noise'].copy()};bad['noise'][0]=.001
    with pytest.raises(AssertionError,match='identical saved epsilon'):artifacts.compare_arrays(a,bad,training)
    with pytest.raises(AssertionError,match='identical saved epsilon'):
        artifacts.compare_arrays(a,{**b,'noise':b['noise'].astype(np.float64)},training)
    with pytest.raises(AssertionError,match='dimensions'):artifacts.compare_arrays(a,b,training[:15])


def test_global_provenance_failure_preserves_every_primary_slot(monkeypatch,tmp_path):
    written=[];monkeypatch.setattr(metrics,'RadialSupport',lambda:object())
    monkeypatch.setattr(metrics,'write_json',lambda path,row:written.append((path,row)))
    rows,manifests,errors=metrics.collect_rows(tmp_path,tmp_path/'unused','training','artifact','upstream provenance failure')
    assert len(rows)==len(written)==384 and len(errors)==2 and manifests==[]
    assert all(r['metric_status']=='failed' and r['auxiliary_excluded'] and not r['paired_available'] for r in rows)
    assert len({(r['seed'],r['sample_id'],r['arm']) for r in rows})==384
    groups=metrics.common.summary_groups(rows)
    assert all(g['planned_samples']==g['metric_failed_or_missing_count']==128 for g in groups if g['seed']=='pooled')


def test_primary_orchestration_keeps_MC_pairing_and_missing_attempt_denominators(monkeypatch,tmp_path):
    calls=[];written=[];base_inputs=input_fixture()
    def fake_seed(run,seed):
        return {**base_inputs,'seed':seed,'sample_records':{'00000':{'sample_id':'00000'}},
            'outlet_records':{arm:{} for arm in metrics.common.ARMS},'outlet_manifest_sha256':{},'export_identity_sha256':'export'}
    def sample(inputs,sid):
        if inputs['seed']!=10 or sid!='00000':raise ValueError('Missing sample')
        return {'nodes':np.zeros(2050,np.float32)},{'sample_npz_sha256':'sample','source_nodes_sha256':'nodes'}
    class Support:
        def mesh(self,nodes):return np.zeros((3,3)),np.array([[0,1,2]],np.int32)
    def mesh(inputs,source,arm,nodes,support):
        v=np.zeros((3,3));v[0,2]={'fixed':1.,'a':.97,'qem':.99}[arm]
        return v,np.array([[0,1,2]],np.int32),{}
    quality={'face_count':512,'degenerate_count':0,'min_triangle_angle_deg':20.,'triangle_min_angle_p05_deg':30.}
    def evaluate(v,f,rv,rf,points,seed):
        calls.append((points,seed));return {'seed':seed,'rms_bbox_percent':float(v[0,2])}
    monkeypatch.setattr(metrics,'seed_inputs',fake_seed);monkeypatch.setattr(metrics,'RadialSupport',Support)
    monkeypatch.setattr(metrics.common,'load_sample',sample)
    monkeypatch.setattr(metrics.common,'verify_outlet_binding',lambda inputs,arm,sid,sample:{'status':'ok','source_nodes_sha256':'nodes'})
    monkeypatch.setattr(metrics.common,'load_mesh',mesh);monkeypatch.setattr(metrics.common,'quality_tails',lambda *a:quality.copy())
    monkeypatch.setattr(metrics.common,'audit_mesh',lambda *a,**k:{'valid':True})
    monkeypatch.setattr(metrics.common,'evaluate_mesh',evaluate)
    monkeypatch.setattr(metrics,'write_json',lambda path,row:written.append(row))
    rows,_,errors=metrics.collect_rows(tmp_path,tmp_path/'unused','training','artifact')
    assert not errors and len(rows)==384 and sum(r['metric_status']=='ok' for r in rows)==3
    assert calls==[(8192,993501),(8192,1093501)]*3
    assert rows[1]['qualified_ge2pct_both'] and not rows[2]['qualified_ge2pct_both']
    group=next(g for g in metrics.common.summary_groups(rows) if g['seed']==10 and g['arm']=='a')
    assert group['planned_samples']==64 and group['metric_failed_or_missing_count']==63 and group['qualified_ge2pct_both_count']==1
