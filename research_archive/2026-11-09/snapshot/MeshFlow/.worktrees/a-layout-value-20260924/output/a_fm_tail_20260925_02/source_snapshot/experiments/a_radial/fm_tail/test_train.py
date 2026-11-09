"""CPU semantics of full inherited Adam/RNG, constantLR migration and continuation."""
from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from ..models import ShapeFM
from ..system_probe.official import exact_same
from ..system_probe.bayes_review import conditional_velocity
from .train import TailSession,LR,SCOPE,capture_rng,transition_scheduler
from .smoke import CORE,semantic,assert_parent_import,manual_original_next


def parent_fixture():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_rng_state(torch.Generator(device='cpu').manual_seed(733).get_state());random.seed(733);np.random.seed(733)
    raw=torch.randn(16,64);model=ShapeFM();model.fit_normalization(raw);target=model.standardize(raw).detach()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8,eta_min=1e-4)
    gen=torch.Generator(device='cpu').manual_seed(731010);order=[]
    for step in range(8):
        while len(order)<128:order+=torch.randperm(16,generator=gen).tolist()
        ix=order[:128];order=order[128:];x1=target[ix];x0=torch.randn_like(x1);t=torch.rand(len(x1));xt=(1-t[:,None])*x0+t[:,None]*x1
        loss=(model(xt,t)-(x1-x0)).square().mean();opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();sched.step()
    parent={'model':deepcopy(model.state_dict()),'optimizer':deepcopy(opt.state_dict()),'scheduler':deepcopy(sched.state_dict()),
        'config':{'CPU_TEST_ONLY':True,'steps':8,'batch':128,'lr':1e-3},'step':8,
        'rng':capture_rng('cpu'),'batch_rng':gen.get_state(),'draw_queue':order}
    config={'CPU_TEST_ONLY':True,'initial_step':8,'final_step':16,'seed':10,'protocol_sha256':'synthetic_only','device':'cpu'}
    return parent,target,config


def test_scheduler_8000_transition_keeps_lr_and_replaces_all_old_bases():
    model=ShapeFM();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    optimizer.param_groups[0]['lr']=1e-4;optimizer.param_groups[0]['initial_lr']=1e-3
    scheduler=transition_scheduler(optimizer,8000)
    assert scheduler.last_epoch==8000 and scheduler.base_lrs==[1e-4]
    assert optimizer.param_groups[0]['lr']==optimizer.param_groups[0]['initial_lr']==1e-4
    optimizer.step();scheduler.step()
    assert scheduler.last_epoch==8001 and optimizer.param_groups[0]['lr']==1e-4


def test_cpu_import_and_manual_frozen_arithmetic_next_step():
    parent,target,config=parent_fixture();unmodified=deepcopy(parent)
    session=TailSession(parent,target,config,'cpu');assert_parent_import(session,parent)
    reference,loss,norm=manual_original_next(parent,target,'cpu')
    session=TailSession(parent,target,config,'cpu');row=session.update();state=session.state()
    for key in CORE:exact_same(reference[key],state[key],'reference_next.'+key)
    assert row['train_batch_loss']==loss and row['grad_norm']==norm
    exact_same(parent,unmodified,'source_parent_not_mutated')
    assert not torch.cuda.is_initialized()


def test_cpu_continuous8_equals3plus5_complete_state(tmp_path):
    parent,target,config=parent_fixture()
    whole=TailSession(parent,target,config,'cpu')
    for _ in range(8):whole.update()
    expected=deepcopy(whole.state())
    first=TailSession(parent,target,config,'cpu')
    for _ in range(3):first.update()
    path=tmp_path/'complete_state.pt';torch.save(first.state(),path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    resumed=TailSession(state,target,config,'cpu',continuation=state)
    for _ in range(5):resumed.update()
    exact_same(semantic(expected),semantic(resumed.state()),'CPU_resume_full_state')
    assert resumed.step==16 and resumed.scheduler.last_epoch==16 and resumed.optimizer.param_groups[0]['lr']==1e-4
    assert len(resumed.trace)==8 and resumed.state()['is_predeclared_final']


def test_wrong_protocol_resume_and_partial_checkpoint_rejected():
    parent,target,config=parent_fixture();session=TailSession(parent,target,config,'cpu');state=deepcopy(session.state())
    other=deepcopy(config);other['protocol_sha256']='wrong'
    with pytest.raises(AssertionError,match='resume_config'):TailSession(state,target,other,'cpu',continuation=state)
    session.safe_boundary=False
    with pytest.raises(AssertionError,match='partial update'):session.state()


def test_oracle_at_zero_is_empirical_mean_minus_noise_and_rejects_t1():
    target=torch.arange(16*64,dtype=torch.float32).reshape(16,64)/100
    x=torch.arange(3*64,dtype=torch.float32).reshape(3,64)/30
    actual=conditional_velocity(x,torch.zeros(3),target)
    assert torch.allclose(actual,target.double().mean(0)-x.double(),atol=1e-14,rtol=1e-14)
    with pytest.raises(ValueError,match='0<=t<1'):conditional_velocity(x,torch.ones(3),target)


def test_oracle_single_target_velocity_has_no_conditional_uncertainty():
    target=torch.arange(64,dtype=torch.float64)[None]/100;x=torch.ones(3,64,dtype=torch.float64)*.2;t=torch.tensor([.1,.5,.9],dtype=torch.float64)
    actual=conditional_velocity(x,t,target)
    expected=(target-x)/(1-t[:,None]);assert torch.equal(actual,expected)


def test_original_real8000_adam_moments_unchanged_by_scheduler_transition():
    from pathlib import Path
    source=Path(__file__).resolve().parents[3]/'output/a_ae_fm_closed16_20260925_01'
    for seed in (10,11):
        parent=torch.load(source/f'seed{seed}/fm/step00008000.pt',map_location='cpu',weights_only=True)
        model=ShapeFM();model.load_state_dict(parent['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.load_state_dict(deepcopy(parent['optimizer']));before=deepcopy(optimizer.state_dict())
        scheduler=transition_scheduler(optimizer,parent['step']);after=optimizer.state_dict()
        exact_same(before['state'],after['state'],'actual_source_Adam_moments')
        assert scheduler.last_epoch==8000 and scheduler.base_lrs==[1e-4]
        assert after['param_groups'][0]['initial_lr']==1e-4 and before['param_groups'][0]['initial_lr']==1e-3
    assert not torch.cuda.is_initialized()


def test_normalization_change_and_inconsistent_adam_count_rejected():
    parent,target,config=parent_fixture();session=TailSession(parent,target,config,'cpu')
    session.model.mu.add_(.1)
    with pytest.raises(AssertionError,match='normalization changed'):session.state()
    broken=deepcopy(parent);next(iter(broken['optimizer']['state'].values()))['step']-=1
    with pytest.raises(AssertionError,match='Adam/global update count'):TailSession(broken,target,config,'cpu')


def test_new_checkpoint_contract_works_with_frozen_sample_ours(tmp_path):
    from pathlib import Path
    from argparse import Namespace
    import json
    from ..io_utils import file_hash
    from ..sample_ours import run as sample_once
    from .train import checkpoint_config
    source=Path(__file__).resolve().parents[3]/'output/a_ae_fm_closed16_20260925_01'
    protocol=json.loads((source/'RUN_PROTOCOL.json').read_text());ae=source/'seed10/ae/step00003200.pt'
    parent=source/'seed10/fm/step00008000.pt';state=torch.load(parent,map_location='cpu',weights_only=True)
    p={'jobs':{'10':{'parent_sha256':file_hash(parent),'ae_sha256':file_hash(ae)}},
       'data_manifest_sha256':'test','source_manifest_sha256':'test','source_data_hashes':protocol['data_hashes']}
    config=checkpoint_config(p,'CPU_SAMPLING_CONTRACT_TEST_ONLY',10,'cpu','not_used_for_sampling')
    checkpoint=tmp_path/'fake_new_format.pt';torch.save({'model':state['model'],'config':config},checkpoint)
    destination=tmp_path/'sample';sample_once(Namespace(device='cpu',ae=ae,fm=checkpoint,out=destination,noise_seed=73510,samples=1,steps=1))
    manifest=json.loads((destination/'SAMPLES.json').read_text())
    assert manifest['records'][0]['status']=='ok' and manifest['dataset_sha256']==protocol['data_hashes']['radial_train.npz']
    assert not torch.cuda.is_initialized()


def test_fixed4096_matches_original_bytes_even_with_other_ambient_CPU_threads():
    from pathlib import Path
    import json
    from .train import registered_probe
    from ..system_probe.bayes_review import probe_hash
    source=Path(__file__).resolve().parents[3]/'output/a_ae_fm_closed16_20260925_01'
    protocol=json.loads((source/'RUN_PROTOCOL.json').read_text());review=json.loads((source/'representation_diagnostics/BAYES_ORACLE_REVIEW.json').read_text())
    ac=torch.load(source/'seed10/ae/step00003200.pt',map_location='cpu',weights_only=True)
    fc=torch.load(source/'seed10/fm/step00008000.pt',map_location='cpu',weights_only=True)
    with np.load(Path(protocol['data'])/'radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr'])
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        probe=registered_probe(ac['model'],fc['model'],y,10,review['seeds']['10']['probe_sha256'])
        assert probe_hash(probe)==review['seeds']['10']['probe_sha256'] and torch.get_num_threads()==2
    finally:torch.set_num_threads(previous)
