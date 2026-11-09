"""Training-parent energy balancing, with zero-motion teachers retained."""
import numpy as np
import torch
from ..layout_student.train import discrete_bridge


def derive_weights(u0,u1):
    energy=np.square(np.asarray(u1,np.float64)-np.asarray(u0,np.float64)).mean(axis=(1,2))
    positive=energy[energy>0]
    if len(positive)==0 or not np.isfinite(energy).all():raise ValueError('Need finite positive training energy')
    floor=.1*float(np.median(positive));w=1./np.maximum(energy,floor);w/=w.mean()
    return energy,floor,w


def registered_weights(protocol,data,uids,device):
    entry=protocol['loss']['parent_weighting'];energy,floor,weights=derive_weights(data['directions0'],data['directions1'])
    assert entry['train_uids']==uids and entry['internal_check_used'] is False and entry['zero_teacher_kept'] is True
    assert entry['floor']==floor and np.array_equal(entry['teacher_energy'],energy) and np.array_equal(entry['weights'],weights)
    assert np.isfinite(weights).all() and (weights>0).all() and abs(weights.mean()-1)<1e-12
    return torch.as_tensor(weights,dtype=torch.float32,device=device)


def weighted_bridge_loss(model,u0,u1,features,k,weights):
    positions,target,_,t=discrete_bridge(u0,u1,k,32)
    prediction=model(positions.float(),t.float(),features)
    squared=(prediction-target.float()).square()
    per_parent=squared.mean(dim=(1,2))
    return (squared*weights[:,None,None]).mean(),per_parent


def step(model,optimizer,scheduler,sampler,active,uids,tensors,faces,scale,update,dev,parent_weights):
    indices=[active[i] for i in sampler.take(4)];index=torch.tensor(indices,dtype=torch.long,device=dev)
    k=torch.randint(0,32,(4,),device=dev);u0,u1,features=tensors
    loss,per_parent=weighted_bridge_loss(model,u0[index],u1[index],features[index],k,parent_weights[index])
    if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at {update}')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
    if any(p.grad is None for p in model.parameters()) or not bool(finite):raise FloatingPointError(f'Invalid gradient at {update}')
    grad=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())).detach().cpu())
    optimizer.step();scheduler.step()
    return {'update':update,'loss':float(loss.detach().cpu()),'velocity_mse':float(per_parent.mean().detach().cpu()),
        'endpoint_loss':None,'quality_penalty':None,'parent_velocity_mse':per_parent.detach().cpu().tolist(),
        'parent_weights':parent_weights[index].detach().cpu().tolist(),'loss_is_parent_weighted':True,
        'uids':[uids[i] for i in indices],'rng_matched_k':k.detach().cpu().tolist(),
        'lr':optimizer.param_groups[0]['lr'],'gradient_l2':grad,'alphas':None}