"""Light CPU continuation tests with a synthetic, explicitly smoke-only lineage."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from ..io_utils import file_hash
from ..layout_endpoint import train as endpoint
from ..layout_endpoint.test_train import fixture_run
from ..layout_endpoint.smoke import exact_same
from ..layout_fitdiag import train as fitdiag
from .train import train_run, config_digest, step, validate_parent
from .smoke import manual_parent_next, EQUAL_KEYS


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def update_manifest(run,protocol,additional):
    files=json.loads((run/'DATA_MANIFEST.json').read_text())['files']
    files.update({rel:file_hash(run/rel) for rel in additional})
    write(run/'DATA_MANIFEST.json',{'files':files})
    protocol['hashes']['data_manifest_sha256']=file_hash(run/'DATA_MANIFEST.json')
    protocol['hashes']['config_sha256']=config_digest(protocol)
    write(run/'RUN_PROTOCOL.json',protocol)


def fixture_chain(run):
    run,p=fixture_run(run,updates=2)
    initial=endpoint.train_run(run,'c0',10,2,device='cpu',checkpoint_every=2,output_dir=run/'endpoint_parent')
    shutil.copyfile(run/'RUN_PROTOCOL.json',run/'initial/endpoint_RUN_PROTOCOL.json')
    shutil.copyfile(initial['checkpoint'],run/'initial/c0_final.pt')
    initial_state=torch.load(initial['checkpoint'],map_location='cpu',weights_only=False)
    job={'kind':'joint','seed':10,'active_training_uids':p['train_uids'],'diagnostic_uids':p['train_uids'],
         'start_update':2,'total_updates':4,'initialization_mode':'resume_endpoint_c0_full_state',
         'initial_checkpoint':'initial/c0_final.pt','initial_checkpoint_sha256':file_hash(run/'initial/c0_final.pt'),
         'initial_model_sha256':initial_state['model_sha256'],
         'source_protocol_sha256':file_hash(run/'initial/endpoint_RUN_PROTOCOL.json')}
    p['jobs']={'joint_10':job};p['training']['checkpoint_updates']=[2]
    p['paths']['source_protocol']='initial/endpoint_RUN_PROTOCOL.json'
    update_manifest(run,p,['initial/endpoint_RUN_PROTOCOL.json','initial/c0_final.pt'])
    middle=fitdiag.train_run(run,'joint_10',device='cpu',output_dir=run/'fitdiag_parent')
    shutil.copyfile(run/'RUN_PROTOCOL.json',run/'initial/fitdiag_RUN_PROTOCOL.json')
    shutil.copyfile(middle['checkpoint'],run/'initial/fitdiag_joint_10_final.pt')
    middle_state=torch.load(middle['checkpoint'],map_location='cpu',weights_only=False)
    p=copy.deepcopy(p);p['experiment']='A-LONGFIT-CPU-TEST';p['scope']='SMOKE_ONLY synthetic parent lineage'
    p['jobs']['joint_10'].update(start_update=4,total_updates=12,
        initialization_mode='resume_fitdiag_joint_full_state',
        initial_checkpoint='initial/fitdiag_joint_10_final.pt',
        initial_checkpoint_sha256=file_hash(run/'initial/fitdiag_joint_10_final.pt'),
        initial_model_sha256=middle_state['model_sha256'],
        source_protocol_sha256=file_hash(run/'initial/fitdiag_RUN_PROTOCOL.json'))
    p['training']['checkpoint_updates']=[4];p['paths']['source_protocol']='initial/fitdiag_RUN_PROTOCOL.json'
    update_manifest(run,p,['initial/fitdiag_RUN_PROTOCOL.json','initial/fitdiag_joint_10_final.pt'])
    return run,p,middle_state


def test_exact_original_fitdiag_core_is_reused():
    assert step is fitdiag.step


def test_continuation_resume_and_independent_parent_next_update(tmp_path,monkeypatch):
    run,p,parent=fixture_chain(tmp_path/'run')
    load=np.load;opened=[]
    from ..exporter import ASSETS
    allowed={(run/'train_labels.npz').resolve(),(ASSETS/'support.npz').resolve()}
    def whitelist(path,*args,**kwargs):
        assert Path(path).resolve() in allowed
        opened.append(str(path));return load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',whitelist)
    whole=train_run(run,'joint_10',device='cpu',output_dir=run/'whole')
    first=train_run(run,'joint_10',device='cpu',stop_after=7,output_dir=run/'first')
    resumed=train_run(run,'joint_10',Path(first['checkpoint']),device='cpu',output_dir=run/'resumed')
    a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
    for key in EQUAL_KEYS:exact_same(a[key],b[key],key)
    exact_same(a['trace'][:4],parent['trace'])
    exact_same(a['timing_trace'][:len(parent['timing_trace'])],parent['timing_trace'])
    assert file_hash(run/'whole/initial.pt')==p['jobs']['joint_10']['initial_checkpoint_sha256']
    assert file_hash(run/'whole/step_00004.pt')==p['jobs']['joint_10']['initial_checkpoint_sha256']
    assert whole['global_parent_exposures_total']==48
    assert sum(whole['imported_parent_exposures'].values())==16
    assert sum(whole['new_parent_exposures_since_import'].values())==32
    assert whole['new_optimizer_updates_since_import']==8
    assert resumed['updates_this_invocation']==5
    assert sum(resumed['new_parent_exposures_this_invocation'].values())==20
    assert whole['internal_check_teacher_files_opened']==0
    oracle=manual_parent_next(run,'joint_10','cpu')
    next_report=train_run(run,'joint_10',device='cpu',stop_after=5,output_dir=run/'parent_next')
    actual=torch.load(next_report['checkpoint'],map_location='cpu',weights_only=False)
    for key,value in oracle.items():exact_same(value,actual[key],key)
    assert len(opened)>0 and whole['parent_exposure_matches_single20k_at_declared_final'] is False


def test_parent_binding_rejects_missing_state_and_changed_sampler(tmp_path):
    run,p,parent=fixture_chain(tmp_path/'run');manifest=json.loads((run/'DATA_MANIFEST.json').read_text())['files']
    assert validate_parent(parent,p['jobs']['joint_10'],p,run,manifest)==p['jobs']['joint_10']['source_protocol_sha256']
    bad=copy.deepcopy(parent);bad['sampler']['exposures'][0]+=1
    with pytest.raises(ValueError,match='exposure/trace'):
        validate_parent(bad,p['jobs']['joint_10'],p,run,manifest)
    bad=copy.deepcopy(parent);bad['diagnostic_only']=False
    with pytest.raises(ValueError,match='fitdiag diagnostic final'):
        validate_parent(bad,p['jobs']['joint_10'],p,run,manifest)
    p['scope']='FORMAL';p['hashes']['config_sha256']=config_digest(p);write(run/'RUN_PROTOCOL.json',p)
    with pytest.raises(ValueError,match='Fixed long continuation budget'):
        train_run(run,'joint_10',device='cpu',output_dir=run/'forbidden')
