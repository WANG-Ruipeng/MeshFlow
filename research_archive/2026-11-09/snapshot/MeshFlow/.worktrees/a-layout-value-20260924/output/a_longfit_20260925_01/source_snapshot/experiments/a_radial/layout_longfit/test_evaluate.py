"""Final/source separation and fixed denominators; no actual surface sampling."""
import copy
import json
import numpy as np
import pytest
import torch
from ..io_utils import file_hash
from ..legacy.networks import DomainVelocity
from ..layout_student.train import model_sha256
from .evaluate import MODEL_NAMES,expected_slots,registered_uids,aggregate,load_network,source_single_jobs
from .summarize import fit_statistics,internal_check_details
from .visualize import curve_series
from .diagnose import CHECKPOINTS


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def protocol():
    uids=[f'p{i:02}' for i in range(16)]
    singles={'single_energy_10':{'active_training_uids':[uids[0]]},'single_prior_10':{'active_training_uids':[uids[1]]}}
    return {'uids':uids,'train_uids':uids[:12],'internal_check_uids':uids[12:],
        'jobs':{f'joint_{seed}':{'active_training_uids':uids[:12]} for seed in [10,11]},
        'source_single_jobs':singles,'hashes':{'source_protocol_sha256':'source-protocol'},
        'evaluation':{'model_uids':{n:(singles[n]['active_training_uids'] if n.startswith('single_') else uids) for n in MODEL_NAMES},
            'single_reference_uids':{n:j['active_training_uids'] for n,j in singles.items()},
            'source_model_jobs':{'parent_joint_10':'joint_10','parent_joint_11':'joint_11','single_energy_10':'single_energy_10','single_prior_10':'single_prior_10'},
            'model_checkpoints':{'single_energy_10':'initial/single.pt','joint_10':'training/joint_10/final.pt'},
            'model_seeds':{'single_energy_10':10,'joint_10':10},'source_checkpoint_sha256':{},'source_model_sha256':{},
            'source_model_updates':20000,'formal_final_update':240000}}


def test_fixed_denominators_and_own_frozen_single_parent_only():
    p=protocol();slots=expected_slots(p)
    assert len(slots)==164 and len({(u,m) for u,m,_ in slots})==98
    q=copy.deepcopy(p);q['evaluation']['model_uids']['single_energy_10']=['p01']
    with pytest.raises(ValueError,match='own training parent'):registered_uids(q,'single_energy_10')
    q=copy.deepcopy(p);q['evaluation']['model_uids']['single_energy_10']=['p12']
    with pytest.raises(ValueError,match='own training parent'):registered_uids(q,'single_energy_10')
    q=copy.deepcopy(p);q['jobs']['single_energy_10']=q['source_single_jobs']['single_energy_10']
    with pytest.raises(ValueError,match='new training job'):registered_uids(q,'single_energy_10')
    q=copy.deepcopy(p);q['evaluation']['source_single_jobs']=copy.deepcopy(q['source_single_jobs'])
    assert source_single_jobs(q)==q['source_single_jobs']
    q['evaluation']['source_single_jobs']['single_energy_10']['active_training_uids']=['p01']
    with pytest.raises(ValueError,match='metadata disagrees'):source_single_jobs(q)


def test_error_and_missing_final_slots_remain_counted(tmp_path):
    p=protocol();put(tmp_path/'RUN_PROTOCOL.json',p)
    put(tmp_path/'attempts/p00/joint_10/result.json',{'uid':'p00','model':'joint_10','status':'error','rows':[],'error':'preserved'})
    result=aggregate(tmp_path)
    assert result['expected_attempt_count']==98 and result['expected_row_count']==164
    assert result['attempt_count']==1 and len(result['unattempted'])==97
    assert len(result['failed_row_slots'])==2 and len(result['missing_success_rows'])==164


def test_source20k_checkpoint_and_new240k_are_not_interchangeable(tmp_path):
    p=protocol();weights=DomainVelocity().state_dict();sha=model_sha256(weights)
    state={'model':weights,'model_sha256':sha,'is_predeclared_final':True,'update':20000,
        'protocol_sha256':'source-protocol','config':{'job_id':'single_energy_10','seed':10,'arm':'c0','active_training_uids':['p00']}}
    path=tmp_path/'initial/single.pt';path.parent.mkdir(parents=True);torch.save(state,path)
    p['evaluation']['source_checkpoint_sha256']['single_energy_10']=file_hash(path)
    p['evaluation']['source_model_sha256']['single_energy_10']=sha
    put(tmp_path/'RUN_PROTOCOL.json',p)
    _,identity=load_network(tmp_path,'single_energy_10',p)
    assert identity['updates']==20000 and identity['single_reference_was_not_retrained_in_this_run']
    new=dict(state,update=240000,protocol_sha256=file_hash(tmp_path/'RUN_PROTOCOL.json'),
        config={'job_id':'joint_10','seed':10,'arm':'c0','active_training_uids':p['train_uids']})
    dest=tmp_path/'training/joint_10/final.pt';dest.parent.mkdir(parents=True);torch.save(new,dest)
    _,identity=load_network(tmp_path,'joint_10',p)
    assert identity['updates']==240000 and not identity['frozen_source_reference']
    torch.save(dict(new,update=20000),dest)
    with pytest.raises(ValueError,match='240000 final'):load_network(tmp_path,'joint_10',p)
    torch.save(dict(state,update=240000),path)
    with pytest.raises(ValueError,match='source checkpoint changed'):load_network(tmp_path,'single_energy_10',p)


def test_global_ratio_macro_mean_median_and_zero_denominator_are_distinct():
    fits=[{'endpoint_direction_mse_normalized':error,'zero_motion_direction_mse_normalized':zero,
           'endpoint_mse_over_zero_motion':error/zero if zero else None}
          for error,zero in [(10.,100.),(4.,1.),(9.,1.),(8.,0.)]]
    result=fit_statistics(fits)
    assert result['endpoint_global_ratio_of_means']==pytest.approx(31/102)
    assert result['endpoint_macro_ratio_mean']==pytest.approx((.1+4+9)/3)
    assert result['endpoint_macro_ratio_median']==4
    assert result['endpoint_macro_ratio_defined_parents']==3 and result['endpoint_macro_zero_denominator_parents']==1
    empty=fit_statistics([])
    assert empty['endpoint_global_ratio_of_means'] is None and empty['endpoint_macro_ratio_median'] is None


def test_worst_internal_object_and_missing_parents_are_retained():
    p=protocol();rows=[]
    for uid,error,relative,gains in [('p12',20.,None,[-1.,-3.]),('p13',10.,4.,[5.,6.])]:
        rows.append({'model':'joint_10','policy':'raw','uid':uid,'qualified':False,'failure_reasons':['p05'],
            'paired_available':True,'rms_over_fixed_two_seeds':[1-g/100 for g in gains],
            'min_angle_margin_deg':0.,'p05_angle_margin_deg':-.2,
            'fit':{'endpoint_direction_mse_normalized':error,'endpoint_mse_over_zero_motion':relative,
                   'zero_motion_has_zero_teacher_error':relative is None}})
    group=next(g for g in internal_check_details(rows,p) if g['model']=='joint_10')
    assert len(group['rows'])==group['planned']==4
    assert group['worst_absolute_endpoint']['uid']=='p12'
    assert group['worst_defined_endpoint_ratio']['uid']=='p13'
    assert group['worst_rms_gain']['uid']=='p12'
    assert group['missing_or_failed_uids']==['p14','p15'] and group['undefined_relative_error_uids']==['p12']


def test_curve_keeps_five_joint_observations_and_missing_values_without_single_curves():
    group={'job':'joint_10','checkpoint':'initial','expected_parents':12,'successful_parents':12,
        'velocity':{'mse':{'mean':.6},'zero_velocity_mse':{'mean':1.}},
        'endpoint':{'endpoint_direction_mse_normalized':{'mean':.5},'zero_motion_direction_mse_normalized':{'mean':1.},
                    'endpoint_mse_over_zero_motion':{'mean':2.,'median':.8}}}
    series=curve_series([group],'joint_10',{'diagnostic_uids':list(range(12)),'start_update':20000})
    assert series['updates']==[20000,20000,50000,100000,240000]
    assert [r['checkpoint'] for r in series['observations']]==list(CHECKPOINTS)
    assert series['endpoint'][0]==.5 and series['macro_endpoint_median'][0]==.8 and series['macro_endpoint_mean'][0]==2
    assert np.isnan(series['endpoint'][1:]).all()
    assert sum(r['missing_or_failed_parents'] for r in series['observations'])==48
