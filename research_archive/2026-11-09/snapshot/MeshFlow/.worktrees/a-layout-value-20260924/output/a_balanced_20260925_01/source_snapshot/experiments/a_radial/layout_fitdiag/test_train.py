"""Meaningful CPU checks of imported C0 state and exact continuation."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from ..layout_endpoint.test_train import fixture_run as endpoint_fixture
from ..layout_endpoint.train import train_run as endpoint_train
from ..layout_student.train import file_sha256
from ..layout_endpoint.smoke import exact_same
from .train import train_run,config_digest,read_json,write_json
from .smoke import EQUAL_KEYS,manual_parent_next,smoke_run


def fixture_run(root):
    old,p=endpoint_fixture(root/'endpoint',updates=2)
    parent=endpoint_train(old,'c0',10,2,device='cpu',checkpoint_every=2)
    run=root/'fitdiag';run.mkdir();(run/'initial').mkdir()
    for src,dst in [(old/'train_labels.npz','train_labels.npz'),(old/'initial/seed_10.pt','initial/seed_10.pt'),
                    (Path(parent['checkpoint']),'initial/c0_seed_10_final.pt'),
                    (old/'RUN_PROTOCOL.json','initial/endpoint_RUN_PROTOCOL.json')]:
        shutil.copyfile(src,run/dst)
    p=copy.deepcopy(p);p['paths']['source_protocol']='initial/endpoint_RUN_PROTOCOL.json'
    p['paths']['joint_source_checkpoints']={'10':'initial/c0_seed_10_final.pt'}
    p['training']['updates']=10;p['training']['checkpoint_updates']=[2,5,10]
    parent_state=torch.load(run/'initial/c0_seed_10_final.pt',map_location='cpu',weights_only=False)
    v1_state=torch.load(run/'initial/seed_10.pt',map_location='cpu',weights_only=False)
    p['jobs']={
      'joint_10':{'kind':'joint','seed':10,'active_training_uids':p['train_uids'],
        'initialization_mode':'resume_endpoint_c0_full_state','start_update':2,'total_updates':10,
        'initial_checkpoint':'initial/c0_seed_10_final.pt',
        'initial_checkpoint_sha256':file_sha256(run/'initial/c0_seed_10_final.pt'),
        'initial_model_sha256':parent_state['model_sha256'],
        'source_protocol_sha256':file_sha256(run/'initial/endpoint_RUN_PROTOCOL.json')},
      'single_prior_10':{'kind':'single','seed':10,'active_training_uids':[p['train_uids'][0]],
        'initialization_mode':'v1_model_fresh_optimizer','start_update':0,'total_updates':8,
        'initial_checkpoint':'initial/seed_10.pt','initial_checkpoint_sha256':file_sha256(run/'initial/seed_10.pt'),
        'initial_model_sha256':v1_state['model_sha256']}}
    manifest={'files':{x.relative_to(run).as_posix():file_sha256(x) for x in run.rglob('*') if x.is_file()}}
    write_json(run/'DATA_MANIFEST.json',manifest)
    p['hashes']['data_manifest_sha256']=file_sha256(run/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p)
    write_json(run/'RUN_PROTOCOL.json',p)
    return run,p


@pytest.mark.parametrize('job_name',['joint_10','single_prior_10'])
def test_eight_update_resume_bitwise_and_train_only_reads(tmp_path,monkeypatch,job_name):
    run,p=fixture_run(tmp_path);job=p['jobs'][job_name];start=job['start_update'];old_load=np.load
    opened=[]
    def whitelist(path,*args,**kwargs):
        assert Path(path).resolve()==(run/'train_labels.npz').resolve()
        opened.append(str(path));return old_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',whitelist)
    whole=train_run(run,job_name,device='cpu',output_dir=run/'whole')
    first=train_run(run,job_name,device='cpu',stop_after=start+3,output_dir=run/'first')
    resumed=train_run(run,job_name,Path(first['checkpoint']),device='cpu',output_dir=run/'resumed')
    a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
    for key in EQUAL_KEYS:exact_same(a[key],b[key],key)
    assert len(opened)==3 and a['diagnostic_only'] and a['is_predeclared_final']
    assert a['config']['parameter_count']==29907 and a['config']['network_extra_inputs']==[]
    assert a['config']['job_id']==job_name and a['update']==job['total_updates']
    assert sum(whole['parent_exposures'].values())==4*a['update']
    assert file_sha256(run/'whole/initial.pt')==job['initial_checkpoint_sha256']
    source=torch.load(run/job['initial_checkpoint'],map_location='cpu',weights_only=False)
    if job['kind']=='joint':
        assert file_sha256(run/'whole/step_00002.pt')==job['initial_checkpoint_sha256']
        exact_same(a['trace'][:start],source['trace'])
        assert a['sampler']['count']==4
    else:
        assert a['sampler']['count']==1 and a['sampler']['exposures']==[32]
        assert a['trace'][0]['update']==1
        assert all(int(s['step'])==8 for s in a['optimizer']['state'].values())


def test_imported_first_update_matches_frozen_endpoint_core(tmp_path):
    run,p=fixture_run(tmp_path)
    manual=manual_parent_next(run,'joint_10','cpu')
    report=train_run(run,'joint_10',device='cpu',stop_after=3)
    actual=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
    for key in manual:exact_same(manual[key],actual[key],key)
    assert actual['update']==3 and not actual['is_predeclared_final']


def test_changed_parent_optimizer_rejected_even_with_rebound_file_hash(tmp_path):
    run,p=fixture_run(tmp_path);relative=p['jobs']['joint_10']['initial_checkpoint']
    path=run/relative;s=torch.load(path,map_location='cpu',weights_only=False)
    s['optimizer']['param_groups'][0]['lr']=2e-4;torch.save(s,path)
    digest=file_sha256(path);p['jobs']['joint_10']['initial_checkpoint_sha256']=digest
    manifest=read_json(run/'DATA_MANIFEST.json');manifest['files'][relative]=digest
    (run/'DATA_MANIFEST.json').write_text(json.dumps(manifest))
    p['hashes']['data_manifest_sha256']=file_sha256(run/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p);(run/'RUN_PROTOCOL.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Parent Adam settings'):train_run(run,'joint_10',device='cpu')
    assert not (run/'training').exists()


def test_source_dependency_mismatch_rejected_before_writes(tmp_path):
    run,p=fixture_run(tmp_path);relative=p['jobs']['joint_10']['initial_checkpoint']
    path=run/relative;s=torch.load(path,map_location='cpu',weights_only=False)
    s['config']['source_sha256']['legacy/networks.py']='0'*64;torch.save(s,path)
    digest=file_sha256(path);p['jobs']['joint_10']['initial_checkpoint_sha256']=digest
    manifest=read_json(run/'DATA_MANIFEST.json');manifest['files'][relative]=digest
    (run/'DATA_MANIFEST.json').write_text(json.dumps(manifest))
    p['hashes']['data_manifest_sha256']=file_sha256(run/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p);(run/'RUN_PROTOCOL.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Frozen endpoint dependency changed'):train_run(run,'joint_10',device='cpu')
    assert not (run/'training').exists()


def test_smoke_driver_uses_isolated_budget_and_preserves_inputs(tmp_path):
    run,p=fixture_run(tmp_path);before=file_sha256(run/'RUN_PROTOCOL.json')
    result=smoke_run(run,tmp_path/'smoke',device='cpu')
    assert result['status']=='PASS' and result['actual_total_optimizer_updates']==34
    assert result['full_semantic_training_state_bitwise_equal']
    assert result['manual_frozen_C0_parent_next_update_bitwise_equal']
    assert result['internal_check_teacher_reads']==0
    assert file_sha256(run/'RUN_PROTOCOL.json')==before
    assert not (run/'training').exists()
