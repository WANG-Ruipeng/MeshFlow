"""Control contract tests; no real optimization or GPU."""
import copy
import numpy as np
import pytest
from . import experiment as control
from ..layout_teacher_margin.test_experiment import data_fixture, protocol_fixture as strict_protocol


def protocol():
    p=strict_protocol()
    p.update(schema='A-TEACHER-MARGIN-CONTROL-v1',initializer='frozen_teacher_only',optimization_margin_deg=0.,
        paired_reference_margin_deg=.1,expected_tasks=4,expected_new_endpoint_rows=4,expected_paired_UIDs=4,
        noise_slots=448,noise_seed_base=1393501,fractions=list(control.frozen.FRACTIONS),noise_trials=16,
        source_protocol_sha256=control.SOURCE_PROTOCOL,no_GT_selection=True,no_network_training=True)
    return p


def test_floor_control_has_no_hidden_margin():
    q={'min_triangle_angle_deg':3.125,'triangle_min_angle_p05_deg':8.75}
    assert control.floor_for_optimizer(q)=={'min_angle_deg':3.125,'angle_q05_deg':8.75}
    strict=control.frozen.optimizer_floor(q)
    for key,value in control.floor_for_optimizer(q).items():assert strict[key]-value==pytest.approx(.1)


def test_registered_causal_contract_rejects_more_steps_or_another_start():
    p=protocol();control.assert_contract(p)
    for key,value in [('max_steps',128),('optimization_margin_deg',.1),('initializer','fixed'),
        ('noise_slots',896),('expected_tasks',8),('source_protocol_sha256','other')]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):control.assert_contract(changed)


def test_control_and_strict_calls_differ_only_floor(tmp_path,monkeypatch):
    data=data_fixture();p=protocol();q={'min_triangle_angle_deg':2.,'triangle_min_angle_p05_deg':4.}
    capture=[]
    def fake(logr,start,faces,floor,**kwargs):
        capture.append((logr,start,faces,floor,kwargs))
        return {'u':start,'vertices':start.copy(),'stats':{'iterations':64,'best_feasible_iteration':4,'status':'feasible'}}
    monkeypatch.setattr(control,'optimize_layout',fake)
    monkeypatch.setattr(control.frozen,'serialize_optimizer',lambda *args:None)
    monkeypatch.setattr(control.frozen,'snapshot_records',lambda *args:([],[]))
    monkeypatch.setattr(control.frozen,'endpoint_row',lambda run,p,uid,name,*args:{'uid':uid,'endpoint':name,'status':'success'})
    info,row=control.task(tmp_path,p,control.UIDS[0],data,None,q)
    assert info['status']=='optimizer_call_completed' and not info['restore_failed']
    assert row['endpoint']==control.NAME and len(capture)==1
    logr,start,faces,floor,kwargs=capture[0]
    assert logr is data['logr'] and start is data['teacher_u'] and faces is data['faces']
    assert floor=={'min_angle_deg':2.,'angle_q05_deg':4.}
    assert kwargs=={'device':'cpu','max_steps':64,'max_seconds':3600,
        'seed':1493501+p['all16_sorted_uids'].index(control.UIDS[0]),'snapshot_steps':(0,16,64)}


def test_control_failure_preserved_without_reference_substitution(tmp_path,monkeypatch):
    def broken(*args,**kwargs):raise RuntimeError('deliberate local failure')
    monkeypatch.setattr(control,'optimize_layout',broken)
    q={'min_triangle_angle_deg':2.,'triangle_min_angle_p05_deg':4.}
    info,row=control.task(tmp_path,protocol(),control.UIDS[0],data_fixture(),None,q)
    assert info['status']=='error' and row['status']=='unavailable'
    assert row['endpoint']==control.NAME and row['planned_slot_retained']
    assert (tmp_path/'attempts'/control.UIDS[0]/'TASK.json').exists()
    with pytest.raises(FileExistsError):control.task(tmp_path,protocol(),control.UIDS[0],data_fixture(),None,q)
