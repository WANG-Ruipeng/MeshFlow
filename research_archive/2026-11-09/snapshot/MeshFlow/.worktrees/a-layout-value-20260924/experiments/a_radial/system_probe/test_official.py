"""Light CPU contracts only; actual full-network CUDA gates live in official.run."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from models.attention import SelfAttention
from models.equidit import LabelEmbedder
from flow_matching import mean_flat
from .official import attention_mask_probe,exact_same,validate_config


def test_actual_attention_nonzero_padding_and_negative_control_cpu():
    torch.manual_seed(52);torch.set_num_threads(1)
    module=SelfAttention(24,3,mixed_precision='fp32',qk_norm=True).eval()
    result=attention_mask_probe(module,'cpu',n=19,valid=11)
    assert result['masked_valid_max_abs_delta']==0.
    assert result['unmasked_negative_control_max_abs_delta']>1e-5
    assert result['valid_output_max_abs']>1e-5


def test_probe_rejects_attention_that_ignores_padding(monkeypatch):
    torch.manual_seed(52);module=SelfAttention(24,3,mixed_precision='fp32',qk_norm=True).eval()
    forward=module.forward
    monkeypatch.setattr(module,'forward',lambda x,mask=None:forward(x,None))
    with pytest.raises(AssertionError,match='Masked attention'):
        attention_mask_probe(module,'cpu',n=19,valid=11)


def test_actual_official_masked_loss_ignores_invalid_values_and_gradients():
    mask=torch.tensor([[True,True,False]])
    prediction=torch.tensor([[[1.,2.],[3.,4.],[900.,1000.]]],requires_grad=True)
    loss=mean_flat(prediction.square(),mask)
    assert loss.item()==7.5
    loss.backward();assert torch.equal(prediction.grad[:,2],torch.zeros(1,2))
    assert torch.count_nonzero(prediction.grad[:,:2])==4
    changed=prediction.detach().clone();changed[:,2]-=12345.
    assert mean_flat(changed.square(),mask).item()==loss.item()


def test_original_cfg_drop_uses_legal_null_embedding():
    embed=LabelEmbedder(40,24,.1)
    classes=torch.tensor([512,801])//20
    assert classes.tolist()==[25,40] and embed.embedding_table.num_embeddings==41
    dropped=embed(classes[:1],train=False,force_drop_ids=torch.ones(1,dtype=torch.long))
    assert torch.equal(dropped,embed.embedding_table(torch.tensor([40])))


def test_fullstate_comparison_rejects_changed_optimizer_or_sampler():
    a={'model':{'w':torch.tensor([1.])},'optimizer':{'moment':torch.tensor([.2])},
       'sampler_rng':torch.tensor([2,7],dtype=torch.uint8),'trace':[{'uids':['a','b']}]}
    exact_same(a,deepcopy(a))
    b=deepcopy(a);b['optimizer']['moment'][0]+=.01
    with pytest.raises(AssertionError,match='optimizer'):exact_same(a,b)
    b=deepcopy(a);b['sampler_rng'][0]=3
    with pytest.raises(AssertionError,match='sampler_rng'):exact_same(a,b)


def test_formal_config_must_keep_50000_and_complete_architecture():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/meshflow_closed16_512.yaml').read_text())
    validate_config(cfg)
    small=deepcopy(cfg);small['model']['hidden_dim']=128
    with pytest.raises(AssertionError,match='model setting'):validate_config(small)
    short=deepcopy(cfg);short['closed16_runner']['updates']=5000
    with pytest.raises(AssertionError,match='formal budget'):validate_config(short)
