"""Preparation rejects changed study scope and mismatched model-only starts."""
import copy
import pytest
import torch
from . import prepare as p


def protocol():
    training=dict(batch_size=4,total_updates=20000,learning_rate=1e-4,weight_decay=1e-4,
        device='cpu',torch_threads=1,interop_threads=1,checkpoint_updates=[0,5000,20000])
    loss=dict(name='c0_unscaled_discrete_bridge_velocity_MSE',bridge_steps=32,endpoint_scale_unused_by_c0=True)
    jobs=p.job_table('file-sha',p.MODEL_SHA)
    return dict(schema='A-MARGIN-SINGLE-v1',train_uids=list(p.UIDS),training=training,loss=loss,jobs=jobs,
        hashes=dict(initial_checkpoint_sha256='file-sha',initial_model_sha256=p.MODEL_SHA,
                    config_sha256=p.config_hash(training,loss,jobs)),
        parent_joint_archive_sha256=p.ARCHIVE_SHA,
        evaluation=dict(planned_endpoints=20,points_each_direction=8192,seed_bases=[1593501,1693501],
            deployment_threads=4,deployment_steps=32,fallback=False,
            reference_arrays_exactly_equal_RadialSupport_mesh=True,original_raw_asset_GT_reference_claim=False,
            evaluation_samples_independent_of_optimizer_internal_samples=True),
        diagnostics=dict(updates=[0,5000,20000],planned_job_observations=24,selection='final20000_only'),
        scope=dict(training_seeds=[10],pure_margin_scalar_causal_claim=False,generalization_claim=False))


def test_all_four_predeclared_pairs_have_common_initial_and_seed():
    data=protocol();p.assert_contract(data)
    assert len(data['jobs'])==8
    for uid in p.UIDS:
        pair=[j for j in data['jobs'].values() if j['uid']==uid]
        assert {j['target_kind'] for j in pair}=={'old','margin'}
        assert all(j['active_training_uids']==[uid] and j['seed']==10 and j['start_update']==0 for j in pair)
        assert len({j['initial_checkpoint_sha256'] for j in pair})==1


@pytest.mark.parametrize('change', ['checkpoint_selection','skip_zero_parent','original_asset_reference','fallback'])
def test_rejects_scope_changes(change):
    data=protocol()
    if change=='checkpoint_selection':data['diagnostics']['selection']='best_validation'
    elif change=='skip_zero_parent':del data['jobs']['old_'+p.UIDS[-1][:8]]
    elif change=='original_asset_reference':data['evaluation']['original_raw_asset_GT_reference_claim']=True
    else:data['evaluation']['fallback']=True
    with pytest.raises(ValueError):p.assert_contract(data)


def test_common_model_requires_exact_parameters_and_parameter_count(monkeypatch):
    model={'weight':torch.zeros(29907)};digest=p.model_sha256(model)
    monkeypatch.setattr(p,'MODEL_SHA',digest)
    a={'model':model,'model_sha256':digest};b=copy.deepcopy(a)
    assert p.check_common_models([a,b])==digest
    b['model']['weight'][1]=1
    with pytest.raises(ValueError,match='parameters differ'):p.check_common_models([a,b])


def test_config_digest_binds_fresh_adam_budget_and_target_pairs():
    data=protocol();digest=data['hashes']['config_sha256']
    changed=copy.deepcopy(data);changed['training']['total_updates']=19999
    assert p.config_hash(changed['training'],changed['loss'],changed['jobs'])!=digest
    changed=copy.deepcopy(data);next(iter(changed['jobs'].values()))['target_key']='directions1_margin'
    assert p.config_hash(changed['training'],changed['loss'],changed['jobs'])!=digest
