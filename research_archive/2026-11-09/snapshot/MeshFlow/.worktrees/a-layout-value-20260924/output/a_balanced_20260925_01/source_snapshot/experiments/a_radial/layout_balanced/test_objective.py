import copy
import numpy as np
import pytest
import torch
from .objective import derive_weights,weighted_bridge_loss,registered_weights
from ..legacy.networks import DomainVelocity
from ..layout_endpoint.train import compute_loss


def test_zero_teacher_finite_normalized_and_energy_inverse():
    a=np.zeros((3,2,3));b=np.ones_like(a);b[0]=0;b[2]*=10
    energy,floor,w=derive_weights(a,b)
    assert np.array_equal(energy,[0,1,100]) and floor==pytest.approx(5.05)
    assert np.isfinite(w).all() and (w>0).all() and w.mean()==pytest.approx(1)
    assert w[0]==w[1] and w[1]/w[2]==pytest.approx(100/5.05)
    with pytest.raises(ValueError):derive_weights(a,a)


def test_unit_weights_match_original_loss_and_all_gradients_bitwise():
    torch.set_num_threads(1);torch.manual_seed(81)
    model=DomainVelocity();model.head[-1].weight.data.normal_(std=.01)
    old=copy.deepcopy(model)
    u0=torch.randn(4,9,3,dtype=torch.float64);u0/=u0.norm(dim=-1,keepdim=True)
    u1=u0+.01*torch.randn_like(u0);u1/=u1.norm(dim=-1,keepdim=True)
    points=torch.randn(4,16,4);points[...,:3]/=points[...,:3].norm(dim=-1,keepdim=True)
    k=torch.tensor([0,5,17,31])
    loss,_=weighted_bridge_loss(model,u0,u1,points,k,torch.ones(4))
    reference,_,_=compute_loss(old,'c0',u0,u1,points,k,[0,1,2,3],None,1)
    assert torch.equal(loss,reference)
    loss.backward();reference.backward()
    for a,b in zip(model.parameters(),old.parameters()):assert torch.equal(a.grad,b.grad)
    weighted,_=weighted_bridge_loss(model,u0,u1,points,k,torch.tensor([.1,.5,1.,2.4]))
    assert not torch.equal(weighted,reference)


def test_weights_recomputed_train_only_and_tampering_rejected():
    a=np.zeros((3,2,3));b=np.ones_like(a);b[0]=0
    e,f,w=derive_weights(a,b);uids=['a','b','c']
    p={'loss':{'parent_weighting':{'train_uids':uids,'teacher_energy':e.tolist(),'floor':f,'weights':w.tolist(),
        'zero_teacher_kept':True,'internal_check_used':False}}}
    data={'directions0':a,'directions1':b}
    assert len(registered_weights(p,data,uids,'cpu'))==3
    p['loss']['parent_weighting']['weights'][0]+=1e-5
    with pytest.raises(AssertionError):registered_weights(p,data,uids,'cpu')