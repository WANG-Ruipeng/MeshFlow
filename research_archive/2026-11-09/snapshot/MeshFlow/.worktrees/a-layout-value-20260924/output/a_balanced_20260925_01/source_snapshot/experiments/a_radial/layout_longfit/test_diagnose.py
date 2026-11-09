"""Longfit observation identity tests; no deployment or formal model selection."""
import copy
import json
import numpy as np
import pytest
import torch
from ..io_utils import file_hash
from ..legacy.networks import DomainVelocity
from ..layout_student.train import model_sha256
from ..layout_fitdiag import diagnose as frozen
from .diagnose import (JOBS,UPDATES,CHECKPOINTS,expected_slots,load_network,aggregate,
                       endpoint_probe,velocity_probe)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def protocol():
    uids=[f'train{i}' for i in range(12)]
    return {'train_uids':uids,'internal_check_uids':['check0','check1','check2','check3'],
        'training':{'checkpoint_updates':list(UPDATES)},
        'jobs':{name:{'kind':'joint','seed':int(name[-2:]),'start_update':20000,
                     'active_training_uids':uids,'diagnostic_uids':uids} for name in JOBS}}


def test_frozen_probes_and_exact_120_slots():
    assert endpoint_probe is frozen.endpoint_probe and velocity_probe is frozen.velocity_probe
    p=protocol();assert len(expected_slots(p))==120
    assert CHECKPOINTS==('initial','20000','50000','100000','240000')
    q=copy.deepcopy(p);q['jobs']['joint_10']['diagnostic_uids']=['check0']
    with pytest.raises(ValueError,match='Internal-check/unregistered'):expected_slots(q)
    q=copy.deepcopy(p);q['training']['checkpoint_updates'].append(250000)
    with pytest.raises(ValueError,match='observation updates'):expected_slots(q)


def test_imported20k_and_new240k_checkpoint_are_distinct(tmp_path):
    p=protocol();model=DomainVelocity();state={'model':model.state_dict(),'model_sha256':model_sha256(model.state_dict()),
        'update':20000,'config':{'seed':10},'protocol_sha256':'source-fitdiag','is_predeclared_final':True}
    initial=tmp_path/'training/joint_10/initial.pt';initial.parent.mkdir(parents=True);torch.save(state,initial)
    initial_hash=file_hash(initial);p['jobs']['joint_10'].update(initial_checkpoint_sha256=initial_hash,initial_model_sha256=state['model_sha256'])
    put(tmp_path/'RUN_PROTOCOL.json',p)
    _,identity=load_network(tmp_path,'joint_10','initial',p)
    assert identity['job_update']==identity['source_checkpoint_update']==20000 and identity['imported_snapshot']
    (initial.parent/'step_20000.pt').write_bytes(initial.read_bytes())
    _,identity=load_network(tmp_path,'joint_10','20000',p);assert identity['imported_snapshot']
    new=dict(state,update=240000,protocol_sha256=file_hash(tmp_path/'RUN_PROTOCOL.json'),
        config={'job':'joint_10','kind':'joint','seed':10,'arm':'c0','active_training_uids':p['train_uids']})
    torch.save(new,initial.parent/'final.pt')
    _,identity=load_network(tmp_path,'joint_10','240000',p)
    assert identity['job_update']==240000 and not identity['imported_snapshot']
    torch.save(dict(new,update=20000),initial.parent/'final.pt')
    with pytest.raises(ValueError,match='observation update'):load_network(tmp_path,'joint_10','240000',p)


def test_failure_accounting_preserves_every_fixed_observation(tmp_path):
    p=protocol();put(tmp_path/'RUN_PROTOCOL.json',p)
    put(tmp_path/'diagnostics/mechanism/joint_10/50000/train0/result.json',
        {'job':'joint_10','checkpoint':'50000','uid':'train0','protocol_sha256':file_hash(tmp_path/'RUN_PROTOCOL.json'),
         'status':'error','error':'retained failure','row':None})
    d=aggregate(tmp_path)
    assert d['expected_attempt_count']==120 and d['attempt_count']==1
    assert len(d['errors'])==1 and len(d['unattempted'])==119 and len(d['missing_success_slots'])==120
    assert len(d['groups'])==10 and not d['checkpoint_selection']
