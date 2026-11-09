"""Fixed diagnostic slots, train-archive isolation and old/new objective identity."""
import copy
import json
import numpy as np
import pytest
import torch
from .diagnose import expected_slots,load_network,load_train_archive,CHECKPOINTS
from ..io_utils import file_hash
from ..legacy.networks import DomainVelocity,point_features
from ..layout_student.train import model_sha256


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def protocol():
    uids=[f'p{i:02d}' for i in range(16)]
    return {'train_uids':uids[:12],'internal_check_uids':uids[12:],
        'training':{'checkpoint_updates':[20000,50000,100000,240000]},
        'jobs':{f'joint_{seed}':{'seed':seed,'kind':'joint','active_training_uids':uids[:12],'diagnostic_uids':uids[:12],
            'start_update':20000,'source_protocol_sha256':'source-protocol'} for seed in (10,11)},
        'hashes':{'config_sha256':'cfg','training_labels_sha256':'labels'}}


def test_exact_120_train_only_slots_and_check_parent_forbidden():
    p=protocol();slots=expected_slots(p)
    assert len(slots)==120 and {k for _,k,_ in slots}==set(CHECKPOINTS)
    bad=copy.deepcopy(p);bad['jobs']['joint_10']['diagnostic_uids'][0]='p12'
    with pytest.raises(ValueError,match='forbidden'):expected_slots(bad)
    bad=copy.deepcopy(p);bad['training']['checkpoint_updates'].insert(1,30000)
    with pytest.raises(ValueError,match='updates changed'):expected_slots(bad)


def test_imported_c0_and_weighted_observations_require_distinct_protocols(tmp_path):
    p=protocol();weights=DomainVelocity().state_dict();mh=model_sha256(weights)
    state={'model':weights,'model_sha256':mh,'update':20000,'job_id':'joint_10','protocol_sha256':'source-protocol',
        'config':{'job':'joint_10','job_id':'joint_10','seed':10,'kind':'joint','arm':'c0','protocol_sha256':'source-protocol',
            'active_training_uids':p['train_uids']},'is_predeclared_final':True}
    folder=tmp_path/'training/joint_10';folder.mkdir(parents=True)
    torch.save(state,folder/'initial.pt')
    p['jobs']['joint_10'].update(initial_checkpoint_sha256=file_hash(folder/'initial.pt'),initial_model_sha256=mh)
    put(tmp_path/'RUN_PROTOCOL.json',p)
    _,identity=load_network(tmp_path,'joint_10','initial',p)
    assert identity['imported_snapshot'] and identity['training_arm']=='c0'
    assert identity['diagnostic_velocity_MSE_is_unweighted'] is True
    cfg={**state['config'],'arm':'c0_parent_weighted','protocol_sha256':file_hash(tmp_path/'RUN_PROTOCOL.json'),'config_sha256':'cfg'}
    current={**state,'config':cfg,'protocol_sha256':cfg['protocol_sha256'],'update':50000,'is_predeclared_final':False,'training_labels_sha256':'labels'}
    torch.save(current,folder/'step_50000.pt')
    _,identity=load_network(tmp_path,'joint_10','50000',p)
    assert not identity['imported_snapshot'] and identity['training_arm']=='c0_parent_weighted'
    wrong=copy.deepcopy(current);wrong['config']['arm']='c0';torch.save(wrong,folder/'step_50000.pt')
    with pytest.raises(ValueError,match='arm differs'):load_network(tmp_path,'joint_10','50000',p)
    wrong=copy.deepcopy(current);wrong['protocol_sha256']='source-protocol';torch.save(wrong,folder/'step_50000.pt')
    with pytest.raises(ValueError,match='another run protocol'):load_network(tmp_path,'joint_10','50000',p)


def test_diagnostic_loader_opens_only_the_train_archive(tmp_path,monkeypatch):
    p=protocol();base=np.zeros((12,258,3));teacher=base.copy();teacher[:,:,0]=.1
    proxy=np.random.default_rng(0).normal(size=(12,256,3))
    path=tmp_path/'train_labels.npz'
    np.savez(path,train_uids=np.array(p['train_uids']),directions0=base,directions1=teacher,proxy_points=proxy,
        point_features=point_features(proxy),faces=np.zeros((512,3),np.int32),logr=np.zeros((12,2050)),quality_floors_deg=np.ones((12,2)))
    p['paths']={'training_labels':'train_labels.npz'};p['hashes']['training_labels_sha256']=file_hash(path)
    p['endpoint_scale_squared']=float(np.mean((teacher-base)**2));opened=[];original=np.load
    def train_only(candidate,*args,**kwargs):
        assert candidate==path;opened.append(candidate)
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(np,'load',train_only)
    data=load_train_archive(tmp_path,p)
    assert opened==[path] and data['train_uids'].tolist()==p['train_uids']
