"""CPU-only algebra/identity gates. No model, optimizer, GPU, or OT calls."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from meshflow_control.data.io import atomic_json, digest
from ..coupling import build_pairs, endpoint_coupling


def fixture():
    # One owner, two true source-vertex neighbors, and a coincident but distinct ID.
    ids = np.array([[0, 1, 2], [1, 0, 3], [3, 4, 5], [3, 6, 7], [8, 9, 10]])
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[.5,0,-1],[0,0,-2],
                         [1,0,-2],[2,0,-2],[1,1,-2],[.5,0,-1],[0,2,0],[1,2,0]], np.float32)
    x1 = vertices[ids]
    fixed = np.zeros((5,3), bool); fixed[0] = True; fixed[1,:2] = True
    third = np.zeros_like(fixed); third[1,2] = True
    x0 = x1.copy(); x0[~fixed] += np.array([.2,.1,.3], np.float32)
    t = np.float32(.4)
    sample = dict(N=5, K=1, source_face_ids=np.arange(5),
                  source_corner_permutations=np.tile(np.arange(3),(5,1)),
                  source_winding_permutation=np.tile(np.arange(3),(5,1)),
                  known_mask=np.array([1,0,0,0,0],bool), free_corner_mask=~fixed,
                  seam_third_corner_mask=third, x0=x0, x1=x1,
                  xt=(1-t)*x0+t*x1, u=x1-x0, t=t)
    return sample, ids


def run():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    s, ids = fixture(); pairs = build_pairs(s, ids, ids)
    assert pairs['eligible_tip_count'] == 1 and pairs['pair_count'] == 2
    assert set(pairs['bulk_indices']) == {6,9}  # coincident vertex 8 is excluded
    results = [dict(name='source_ID_and_bulk_only_selection', status='PASS')]
    # Face and corner permutations reproduce the same underlying source identities.
    order=np.array([3,1,4,0,2]); cp=np.array([[2,0,1],[1,2,0],[0,2,1],[2,1,0],[1,0,2]])
    ss=dict(s)
    for key in ('x0','x1','xt','u','free_corner_mask','seam_third_corner_mask'):
        ss[key]=s[key][order[:,None],cp]
    ss['known_mask']=s['known_mask'][order]
    ss['source_face_ids']=order; ss['source_corner_permutations']=cp
    ss['source_winding_permutation']=np.argsort(cp,axis=1)
    pp=build_pairs(ss,ids,ids)
    assert pp['pair_count']==2 and pp['eligible_tip_count']==1
    results.append(dict(name='OT_face_corner_and_source_winding_transport',status='PASS'))
    pred=torch.tensor(s['u'][None].reshape(1,5,9),requires_grad=True)
    loss,f=endpoint_coupling(pred,s,pairs,eligible_tip_denominator=1)
    loss.backward(); assert float(loss)<1e-12 and float(pred.grad.abs().max())<1e-6
    results.append(dict(name='source_endpoint_zero_loss_and_gradient',status='PASS',loss=float(loss)))
    pred=torch.zeros((1,5,9),requires_grad=True)
    altered=dict(s); altered['xt']=s['xt'].copy(); altered['xt'][1,2,0]+=.4
    loss,f=endpoint_coupling(pred,altered,pairs,eligible_tip_denominator=1)
    loss.backward(); g=pred.grad.reshape(-1,3)
    assert g[5,0]>0 and g[6,0]<0 and g[9,0]<0
    assert np.isclose(float(torch.linalg.vector_norm(g)),f['velocity_gradient_l2_analytic'],rtol=1e-6)
    assert torch.allclose(g.sum(0),torch.zeros(3),atol=1e-7)
    results.append(dict(name='bilateral_differentiable_gradient_and_analytic_norm',status='PASS'))
    # Valence duplicates preserve per-tip weight rather than upweighting one tip.
    p2=dict(pairs); p2['tip_indices']=np.repeat(pairs['tip_indices'],2)
    p2['bulk_indices']=np.repeat(pairs['bulk_indices'],2); p2['pair_weights']=np.repeat(pairs['pair_weights']/2,2); p2['pair_count']=4
    l2,_=endpoint_coupling(pred,altered,p2,eligible_tip_denominator=1)
    assert torch.allclose(loss,l2)
    lm,_=endpoint_coupling(pred,altered,pairs,eligible_tip_denominator=2)
    assert torch.allclose(lm*2,loss)
    results.append(dict(name='per_tip_valence_and_effective_batch_normalization',status='PASS'))
    translated=dict(altered); translated['xt']=altered['xt']+np.array([2,3,-1],np.float32)
    lt,_=endpoint_coupling(pred,translated,pairs,eligible_tip_denominator=1)
    assert torch.allclose(lt,loss,rtol=1e-6,atol=1e-7)
    end=dict(altered); end['t']=1.; pred2=pred.detach().clone().requires_grad_()
    le,_=endpoint_coupling(pred2,end,pairs,eligible_tip_denominator=1); le.backward()
    assert torch.count_nonzero(pred2.grad)==0
    results.append(dict(name='translation_invariance_and_terminal_velocity_gradient_zero',status='PASS'))
    null=dict(s); null['seam_third_corner_mask']=np.zeros((5,3),bool)
    pn=build_pairs(null,ids,ids); pred3=pred.detach().clone().requires_grad_()
    ln,_=endpoint_coupling(pred3,null,pn,eligible_tip_denominator=0); ln.backward()
    assert float(ln)==0 and torch.count_nonzero(pred3.grad)==0
    results.append(dict(name='no_eligible_tips_differentiable_zero',status='PASS'))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',cases=results,neural_model_calls=0,optimizer_updates=0,GPU_calls=0,
                torch_version=torch.__version__,coupling_sha256=digest(Path(__file__).parents[1]/'coupling.py'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=run();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    atomic_json(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
