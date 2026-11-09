"""Small CPU formula fixtures; never model, training, or original noise OT."""
import json
import itertools
import torch
from . import distribution_losses as module


def run_cpu_tests():
    before=module.work_counters()
    rows=[];gradient_calls=0;loss_calls=0
    def loss(name,pred,target):
        nonlocal loss_calls
        loss_calls+=1
        return module.distribution_loss(name,pred,target)
    def grad(value,inputs):
        nonlocal gradient_calls
        gradient_calls+=1
        return torch.autograd.grad(value,inputs,allow_unused=True)
    gt=torch.tensor([[[-.15,0,0],[-.05,0,0],[-.10,.08,0]],
                     [[.05,0,.03],[.15,0,.03],[.10,.08,.03]]],dtype=torch.float64,requires_grad=True)
    bbox=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    target=module.prepare_distribution_target(gt,bbox)
    assert module.work_counters()==before
    assert not target['free_gt'].requires_grad and not target['bbox_L'].requires_grad
    for name in ('L4','L5'):
        pred=gt.detach().clone().requires_grad_(True)
        result=loss(name,pred,target)
        derivative,gt_derivative,bbox_derivative=grad(result['loss'],(pred,gt,bbox))
        assert abs(float(result['loss']))<1e-13
        assert torch.isfinite(derivative).all() and float(derivative.abs().max())<1e-11
        assert gt_derivative is None and bbox_derivative is None
        rows.append(dict(check='GT_oracle_and_fixed_labels',loss_name=name,loss=float(result['loss'].detach()),
            maximum_gradient=float(derivative.abs().max()),GT_gradient=None,bbox_gradient=None,counts=result['counts'],
            max_marginal_L1_residual=result['diagnostics'].get('max_marginal_L1_residual')))
    # Oracle v=u yields exactly the original clean free endpoint algebraically.
    noise=gt.detach()-.125;t=.75
    xt=(1-t)*noise+t*gt.detach();oracle=xt+(1-t)*(gt.detach()-noise)
    assert torch.allclose(oracle,gt.detach(),atol=1e-15,rtol=0)
    for name in ('L4','L5'):
        oracle_result=loss(name,oracle,target)
        assert abs(float(oracle_result['loss']))<1e-13
    perturbed=gt.detach().clone()
    perturbed[0,1,2]+=.011;perturbed[1,2,1]+=.009
    for name in ('L4','L5'):
        pred=perturbed.clone().requires_grad_(True)
        result=loss(name,pred,target);derivative,=grad(result['loss'],(pred,))
        assert torch.isfinite(derivative).all() and float(derivative.abs().max())>0
        differences=[]
        for index in ((0,1,2),(1,2,1)):
            delta=1e-6;plus=perturbed.clone();minus=perturbed.clone();plus[index]+=delta;minus[index]-=delta
            finite_difference=(float(loss(name,plus,target)['loss'])-float(loss(name,minus,target)['loss']))/(2*delta)
            analytical=float(derivative[index]);error=abs(analytical-finite_difference)
            assert error < 2e-7+2e-5*abs(finite_difference), (name,index,analytical,finite_difference)
            differences.append(dict(index=list(index),analytical=analytical,finite_difference=finite_difference,absolute_error=error))
        rows.append(dict(check='FP64_finite_difference_full_vertices_areas_normals',loss_name=name,loss=float(result['loss'].detach()),
            differences=differences,max_marginal_L1_residual=result['diagnostics'].get('max_marginal_L1_residual')))
    # Every S3 corner permutation and a nontrivial face permutation preserve
    # the geometric measure, including orientation reversal of Q=n nT.
    for name in ('L4','L5'):
        reference=float(loss(name,perturbed,target)['loss'])
        errors=[]
        for order in itertools.permutations(range(3)):
            permuted=perturbed.flip(0)[:,list(order)]
            errors.append(abs(float(loss(name,permuted,target)['loss'])-reference))
        assert max(errors)<1e-12, (name,errors)
        rows.append(dict(check='face_and_all_S3_permutations',loss_name=name,max_absolute_difference=max(errors)))
    # A fresh opposite target verifies swap symmetry at the actual 96 steps.
    reverse=module.prepare_distribution_target(perturbed,1.)
    forward=float(loss('L4',perturbed,target)['loss'])
    backward=float(loss('L4',gt.detach(),reverse)['loss'])
    assert abs(forward-backward)<1e-12
    rows.append(dict(check='finite_budget_L4_swap_symmetry',absolute_difference=abs(forward-backward)))
    # Missing a spatial patch and copying the other one preserve total area
    # but must be visible to both measures. Duplicating the entire soup makes
    # L4 blind to total mass, while L5 must respond to its doubled mass.
    duplicate_one=gt.detach()[[0,0]];duplicate_all=gt.detach().repeat(2,1,1)
    for name in ('L4','L5'):
        missing=loss(name,duplicate_one,target);doubled=loss(name,duplicate_all,target)
        assert float(missing['loss'])>1e-5
        if name=='L4':assert abs(float(doubled['loss']))<1e-12
        else:assert float(doubled['loss'])>1e-5 and abs(doubled['diagnostics']['predicted_measure_mass']-2)<1e-12
        rows.append(dict(check='missing_patch_copied_area_and_doubled_total_area',loss_name=name,
            missing_patch_loss=float(missing['loss']),doubled_surface_loss=float(doubled['loss']),
            doubled_mass=doubled['diagnostics'].get('predicted_measure_mass'),
            missing_patch_marginal_L1_residual=missing['diagnostics'].get('max_marginal_L1_residual')))
    # Production FP32 path and a collapsed predicted triangle: no deleting
    # problematic faces. L4 records its floor and L5 keeps zero area naturally.
    target32=module.prepare_distribution_target(gt.detach().float(),1.)
    for name in ('L4','L5'):
        pred=gt.detach().float().requires_grad_(True)
        result=loss(name,pred,target32);derivative,=grad(result['loss'],(pred,))
        assert abs(float(result['loss']))<1e-6 and torch.isfinite(derivative).all()
        assert float(derivative.abs().max())<1e-5
        collapsed=gt.detach().float();collapsed[0]=collapsed[0,0].clone();collapsed.requires_grad_(True)
        damaged=loss(name,collapsed,target32);damaged_grad,=grad(damaged['loss'],(collapsed,))
        assert torch.isfinite(damaged_grad).all()
        assert damaged['diagnostics']['predicted_zero_area_count']==1
        if name=='L4':assert damaged['diagnostics']['predicted_area_floor_count']==1
        rows.append(dict(check='FP32_oracle_and_degenerate_prediction',loss_name=name,oracle_loss=float(result['loss'].detach()),
            oracle_gradient_max=float(derivative.abs().max()),degenerate_loss=float(damaged['loss'].detach()),
            degenerate_gradient_max=float(damaged_grad.abs().max()),degenerate_diagnostics=damaged['diagnostics']))
    # Verify returned per-call accounting distinguishes an actual self solve
    # from reading a cached GT-GT scalar, without caching predicted geometry.
    current=loss('L4',perturbed,target)
    assert current['counts']['sinkhorn_solves']==2 and current['counts']['GTGT_sinkhorn_cache_hits']==1
    current5=loss('L5',perturbed,target)
    assert current5['counts']['kernel_matrix_evaluations']==6 and current5['counts']['GTGT_kernel_cache_hits']==1
    after=module.work_counters()
    actual={key:after[key]-before[key] for key in before}
    assert actual['sinkhorn_solve_attempts']==actual['sinkhorn_solve_returns']
    assert actual['kernel_matrix_attempts']==actual['kernel_matrix_returns']
    return dict(status='PASS',rows=rows,loss_calls=loss_calls,CPU_coordinate_autograd_calls=gradient_calls,
        actual_work=actual,GT_requires_grad_after=gt.requires_grad,GT_accumulated_grad_is_None=gt.grad is None,
        original_noise_OT_calls=0,model_forwards=0,model_backwards=0,optimizer_updates=0,
        CUDA_initialized=torch.cuda.is_initialized(),negative_divergence_clamping=False,
        finite_difference_dtype='torch.float64',production_fixture_dtype='torch.float32')


if __name__=='__main__':
    print(json.dumps(run_cpu_tests(),sort_keys=True))