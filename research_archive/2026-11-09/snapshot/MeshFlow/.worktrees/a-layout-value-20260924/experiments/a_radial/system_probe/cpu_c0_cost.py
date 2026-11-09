"""50-update CPU1 C0 core cost smoke; synthetic inputs, no formal model saved."""
import os
# Set before importing torch/numpy. This process must never initialize CUDA.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ..legacy.networks import DomainVelocity,point_features
from ..layout_endpoint.train import compute_loss
from ..io_utils import file_hash,model_hash,new_dir,write_json


def require(ok,message):
    if not ok:raise ValueError(message)


def run(out):
    require(not torch.cuda.is_initialized(),'CUDA already initialized')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    # CPU generator only: no torch.manual_seed, seed_all, rng_state, or CUDA availability branch.
    torch.random.default_generator.manual_seed(250925)
    rng=np.random.default_rng(250925);base=rng.normal(size=(4,258,3));base/=np.linalg.norm(base,axis=-1,keepdims=True)
    tangent=rng.normal(size=base.shape);tangent-=(tangent*base).sum(-1,keepdims=True)*base
    target=base+.01*tangent;target/=np.linalg.norm(target,axis=-1,keepdims=True)
    proxy=rng.normal(size=(4,256,3));proxy/=np.linalg.norm(proxy,axis=-1,keepdims=True)
    proxy*=np.exp(.1*rng.normal(size=(4,256,1)))
    u0=torch.as_tensor(base,dtype=torch.float64);u1=torch.as_tensor(target,dtype=torch.float64)
    features=torch.as_tensor(point_features(proxy),dtype=torch.float32)
    faces=(np.arange(512*3).reshape(512,3)%258).astype(np.int32) # C0 never reads connectivity.
    source_root=Path(__file__).resolve().parents[1]
    names=['system_probe/cpu_c0_cost.py','legacy/networks.py','layout_endpoint/train.py','layout_student/train.py','io_utils.py']
    sources={name:file_hash(source_root/name) for name in names}
    began=time.perf_counter();model=DomainVelocity().cpu().float().train()
    require(sum(p.numel() for p in model.parameters())==29907,'Unexpected architecture')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    setup_seconds=time.perf_counter()-began;initial=model_hash(model);trace=[]
    scale=float(np.square(target-base).mean());index=torch.arange(4,dtype=torch.long)
    for update in range(1,51):
        started=time.perf_counter();k=torch.randint(0,32,(4,),device='cpu')
        loss,metrics,alpha=compute_loss(model,'c0',u0[index],u1[index],features[index],k,[0,1,2,3],faces,scale)
        require(torch.isfinite(loss).item(),'Nonfinite loss');forward_end=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);loss.backward();backward_end=time.perf_counter()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),'Invalid gradients')
        gradient_l2=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())))
        optimizer.step();scheduler.step();ended=time.perf_counter()
        require(not torch.cuda.is_initialized(),'CPU optimizer unexpectedly initialized CUDA')
        require(alpha is None,'C0 unexpectedly ran a rollout')
        trace.append({'update':update,'loss':float(loss.detach()),'gradient_l2':gradient_l2,'k':k.tolist(),
            'wall_seconds':ended-started,'forward_and_loss_seconds':forward_end-started,
            'backward_seconds':backward_end-forward_end,'gradient_validation_Adam_scheduler_seconds':ended-backward_end,
            'cuda_initialized':False})
    final=model_hash(model);require(initial!=final,'No actual parameter update')
    require(len(optimizer.state)==len(list(model.parameters())) and scheduler.last_epoch==50,'Incomplete Adam/scheduler state')
    steps=[int(state['step']) for state in optimizer.state.values()];require(set(steps)=={50},'Not exactly50 Adam updates')
    require({name:file_hash(source_root/name) for name in names}==sources,'Source changed during cost smoke')
    values=np.array([r['wall_seconds'] for r in trace]);steady=values[5:]
    report={'scope':'SYNTHETIC_CPU1_C0_CORE_COST_ONLY_NOT_FORMAL_TRAINING_OR_MODEL_SELECTION',
        'actual_forward_backward_Adam_updates':50,'batch_size':4,'nodes':258,'proxy_points':256,'feature_width':4,
        'parameter_count':29907,'CPU_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'network_dtype':'float32','bridge_dtype':'float64','optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,
        'scheduler':'constant','loss':'unmodified layout_endpoint.train.compute_loss c0 branch',
        'source_sha256':sources,'initial_model_sha256':initial,'final_ephemeral_model_sha256':final,
        'formal_model_or_checkpoint_saved':False,'production_teacher_or_model_loaded':False,
        'synthetic_four_parent_inputs_only':True,'cpu_generator_seed':250925,'torch_CUDA_rng_helpers_called':False,
        'cuda_initialized_before_and_after_and_each_update':False,'CUDA_VISIBLE_DEVICES':'',
        'setup_seconds':setup_seconds,'all50_update_seconds':float(values.sum()),
        'all50_mean_update_seconds':float(values.mean()),'first_update_seconds':float(values[0]),
        'steady_definition':'last45 of the same50 actual updates; first5 retained in trace, not discarded actual work',
        'steady45_mean_update_seconds':float(steady.mean()),'steady45_median_update_seconds':float(np.median(steady)),
        'steady45_min_update_seconds':float(steady.min()),'steady45_max_update_seconds':float(steady.max()),
        'planned_future_jobs_only_for_projection':8,'hypothetical_updates_per_job':20000,
        'forecast_8x20000_core_seconds_mean':float(steady.mean()*160000),
        'forecast_8x20000_core_hours_mean':float(steady.mean()*160000/3600),
        'forecast_8x20000_core_seconds_median':float(np.median(steady)*160000),
        'forecast_excludes_checkpoint_IO_and_offline_diagnostics':True,
        'timing_not_isolated_other_CPU_GPU_experiments_active':True,
        'does_not_authorize_or_prepare_new_training_protocol':True,'trace':trace}
    out=new_dir(out);write_json(out/'CPU_COST_SMOKE.json',report)
    (out/'CPU_COST_SOURCE.py').write_text(Path(__file__).read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trace','source_sha256')},indent=2),flush=True)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',required=True);args=a.parse_args();run(args.out)
