"""Light series tests: fixed observations and failures, without rendering figures."""
import numpy as np
import pytest
from .diagnose import CHECKPOINTS
from .visualize import curve_series


def group(checkpoint, *, success=1, velocity=2., endpoint=.5):
    return {'job':'single_energy_10','checkpoint':checkpoint,'expected_parents':1,'successful_parents':success,
        'velocity':{'mse':{'mean':velocity},'zero_velocity_mse':{'mean':1.}},
        'endpoint':{'endpoint_direction_mse_normalized':{'mean':endpoint},'zero_motion_direction_mse_normalized':{'mean':1.}}}


def test_initial_snapshot_is_preserved_at_registered_start_update():
    groups=[group(c) for c in CHECKPOINTS]
    single=curve_series(groups,'single_energy_10',{'diagnostic_uids':['a'],'start_update':0})
    assert single['updates']==[0,1000,1667,5000,10000,20000]
    assert [r['checkpoint'] for r in single['observations']]==list(CHECKPOINTS)
    joint_groups=[dict(g,job='joint_10') for g in groups]
    joint=curve_series(joint_groups,'joint_10',{'diagnostic_uids':['a'],'start_update':1000})
    assert joint['updates']==[1000,1000,1667,5000,10000,20000]
    assert len(joint['observations'])==6
    assert joint['observations'][0]['checkpoint']=='initial'
    assert joint['observations'][1]['checkpoint']=='1000'


def test_missing_or_failed_observations_produce_nan_and_keep_denominators():
    groups=[group('initial'),group('1000',success=0,velocity=None,endpoint=None)]
    series=curve_series(groups,'single_energy_10',{'diagnostic_uids':['a'],'start_update':0})
    assert series['velocity'][0]==2 and series['endpoint'][0]==.5
    assert np.isnan(series['velocity'][1:]).all() and np.isnan(series['endpoint'][1:]).all()
    assert len(series['observations'])==6
    assert sum(r['missing_or_failed_parents'] for r in series['observations'])==5
    assert all(r['velocity_ratio_missing'] for r in series['observations'][1:])
    duplicated=[group('initial'),group('initial')]
    with pytest.raises(ValueError,match='Duplicate or unregistered'):
        curve_series(duplicated,'single_energy_10',{'diagnostic_uids':['a'],'start_update':0})
