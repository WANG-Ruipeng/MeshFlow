"""Bounded training-free triangle capture using the frozen official field only."""
from .stable_runtime import configure_stable_runtime,stable_execution,assert_stable_runtime
import argparse,copy,difflib,json,math,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .official_wrapper import load_official,state_sha256
from .nxc112_run import PROJECT,REPO,EMA,EMA_SHA,BASE_SHA,atomic_json,file_record,protected_paths,frozen
from .nxc112_evaluation import array_hash,_save_raw
from .sample import euler_sample
from .prompt_dataset import file_sha256

LAMBDAS=(5.,20.)
SEEDS=(9101,9102)
MAX_FORWARDS=1000
STEPS=50
CODE_FILES=('soft_matching_energy.py','soft_matching_data.py','soft_matching_metrics.py','soft_matching_run.py','soft_matching_report.py')


def schedule(maximum,k):
    return 0. if k<=15 else float(maximum)*(k-15)/25 if k<40 else float(maximum)


def small(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:small(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [small(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    return value


@torch.no_grad()
def sample(base,z,targets,tau,maximum,counts):
    from .soft_matching_energy import soft_matching_gradient,compact_matching_details
    if z.dtype!=torch.float32 or z.shape!=(1,112,9):raise ValueError('Fixed real112 FP32 Gaussian required')
    if targets is not None and targets.dtype!=torch.float32:raise ValueError('FP32 condition required')
    frozen(base);before=counts['official_forward_attempts'];source=array_hash(z)
    condition_hash=None if targets is None else array_hash(targets)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    state=z.clone();y=torch.tensor([112],device=z.device);mask=torch.ones((1,112),device=z.device,dtype=torch.bool)
    path=[state[0].cpu().numpy().copy()];trace=[];grad_before=counts['geometry_gradient_attempts']
    with stable_execution(base):
        facts=assert_stable_runtime(base,coordinates=state)
        for k in range(50):
            t=torch.full((1,),k/50,device=z.device,dtype=torch.float32)
            lam=schedule(maximum,k)
            velocity=base(state,t,y=y,mask=mask).float()
            entry=dict(k=k,t=k/50,lambda_value=lam,model_velocity_rms=float(velocity.square().mean().sqrt()))
            if lam:
                if targets is None:raise RuntimeError('Positive force needs provided triangles')
                counts['geometry_gradient_attempts']+=1
                gradient,details=soft_matching_gradient(state,targets,tau,mask=mask)
                counts['geometry_gradient_completed']+=1
                if gradient.dtype!=torch.float32 or gradient.requires_grad or gradient.grad_fn is not None:raise RuntimeError('Only detached FP32 geometry gradient allowed')
                if not bool(torch.isfinite(gradient).all()):raise RuntimeError('Nonfinite external gradient; no clamp/retry')
                force=gradient*lam;final_velocity=velocity-force
                entry.update(geometry=compact_matching_details(details),gradient_rms=float(gradient.square().mean().sqrt()),
                    force_rms=float(force.square().mean().sqrt()),gradient_dtype=str(gradient.dtype),
                    force_to_model_rms=float(force.square().mean().sqrt())/max(entry['model_velocity_rms'],1e-30))
            else:final_velocity=velocity
            state=state+final_velocity*(1/50)
            counts['euler_updates']+=1
            if state.dtype!=torch.float32 or state.requires_grad or not bool(torch.isfinite(state).all()):raise RuntimeError('Invalid Euler state; stop, no repair')
            trace.append(entry);path.append(state[0].cpu().numpy().copy())
        assert_stable_runtime(base,coordinates=state)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    assert array_hash(z)==source and (targets is None or array_hash(targets)==condition_hash)
    assert counts['official_forward_attempts']-before==50
    assert counts['geometry_gradient_attempts']-grad_before==(34 if maximum else 0)
    assert all(p.grad is None for p in base.parameters())
    counts['complete_rollouts']+=1
    return state[0].cpu().numpy(),np.stack(path),dict(nfe=50,euler_updates=50,geometry_gradient_calls=counts['geometry_gradient_attempts']-grad_before,
        lambda_max=maximum,tau=tau,seconds=seconds,peak_cuda_allocated=torch.cuda.max_memory_allocated(),runtime=facts,
        trace=trace,noise_sha256=source,condition_sha256=condition_hash,input_unchanged=True,official_parameter_gradients_absent=True,
        model_backward_calls=0,optimizer_updates=0,postprocessing=None)


def main(output):
    from .soft_matching_data import prepare_inputs
    from .soft_matching_energy import run_cpu_tests
    from .soft_matching_metrics import evaluate_output,plot_fixed,cpu_fixture_checks
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('soft_matching_'):raise ValueError('Use new separate output')
    out.mkdir(exist_ok=True);result_path=out/'results.json'
    if result_path.exists():raise FileExistsError('No overwrite or automatic repeat')
    torch.set_num_threads(8);runtime=configure_stable_runtime()
    protocol=dict(official_frozen_only=True,adapters_loaded=0,training_updates=0,backbone_updates=0,phase3=False,
        seeds=list(SEEDS),parents=2,K=[1,2],lambda_max=[0,5,20],tau='3*(0.05*condition_parent_bbox_diagonal)^2 in original model-coordinate squared units',
        distance='min over 6 target vertex permutations, sum of 3 squared Euclidean vertex distances',
        energy='-tau*sum_k logsumexp_i(-D_ik/tau)',normalization='no energy normalization by N,K or bbox',
        schedule='k<=15:0; 16<=k<40:lambda_max*(k-15)/25; k>=40:lambda_max',
        steps=50,dt=.02,time='FP32 k/50 k0..49',N=112,y=112,mask='all true',face_cfg=1,
        model_precision='stable original BF16 autocast, FP32 parameters',force_precision='FP32 with autocast disabled',
        state_precision='FP32',gradient_target='detached current Xt only; no gradients through official field',
        max_official_forwards=1000,formal_rollouts=18,diagnostic_rollouts=2,
        exact_branches='lambda0 uses original model velocity and skips geometry autograd',
        diagnostics='lambda0 vs official sampler bitwise; lambda5 first trajectory repeat bitwise',
        selection='No model/configuration selection, no retuning after outputs',coordinate_repairs='NONE',
        interpretation='Current-face soft capture with independent targets, not an injective assignment; inner permutation min is only piecewise smooth; finite Euler guidance does not guarantee exact coincidence.')
    protocol_path=out/'protocol.json'
    with protocol_path.open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2,allow_nan=False)
    protocol_sha=file_sha256(protocol_path)
    report=dict(status='RUNNING',active_stage='preflight',protocol=protocol,protocol_path=str(protocol_path),protocol_sha256=protocol_sha,
        initial_runtime=runtime,outputs=[],diagnostics=[],counts=dict(official_forward_attempts=0,official_forward_returns=0,by_stage={},returns_by_stage={},
            geometry_gradient_attempts=0,geometry_gradient_completed=0,model_backward_calls=0,euler_updates=0,complete_rollouts=0,
            formal_rollouts=0,diagnostic_rollouts=0,optimizer_updates=0,backbone_updates=0,adapters_loaded=0),
        environment=dict(torch=torch.__version__,python=platform.python_version(),numpy=np.__version__),timing={})
    def write():atomic_json(result_path,report)
    def progress(event):report['latest_progress']=event;write();print(json.dumps(event,allow_nan=False),flush=True)
    write();base=None;arrays={};protected=[];started=time.perf_counter()
    try:
        report['cpu_tests']=run_cpu_tests()
        if report['cpu_tests']['status']!='PASS':raise RuntimeError('Independent energy mathematics failed')
        report['metric_cpu_tests']=cpu_fixture_checks()
        report['development_cpu_test_runs']=dict(energy_evaluations=187,gradient_helper_calls=31,autograd_grad_calls=33,math_test_runs=2,metric_fixture_runs=1,metric_tests=12,official_forwards=0,model_backward_calls=0,optimizer_updates=0)
        bank,audit=prepare_inputs(out);report['assets']=audit
        assert audit['status']=='PASS' and set(bank['noises'])==set(SEEDS) and len(bank['cases'])==4
        cases=[]
        for parent in range(2):
            group={item['K']:item for item in bank['cases'] if item['parent_index']==parent}
            assert set(group)=={1,2}
            first=group[1]
            cases.append(dict(parent_index=parent,uid=first['parent_uid'],bbox_diagonal=first['bbox_diagonal'],
                full_target=first['target'],targets={k:g['constraints'] for k,g in group.items()},checks={k:g['checks'] for k,g in group.items()}))
        protected=sorted(p for p in set(protected_paths(out)+list(PROJECT.glob('*.py'))) if p.name!='soft_matching_report.py')
        report['protected_before']={str(p):file_record(p) for p in protected}
        if report['protected_before'][str(EMA)]['sha256']!=EMA_SHA:raise RuntimeError('Official EMA file changed')
        base,loaded=load_official(EMA,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        report['strict_load']=loaded;report['base_before']=frozen(base,True)
        if any(layer._forward_hooks for layer in base.layers):raise RuntimeError('An adapter hook is installed')
        report['environment'].update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        c=report['counts']
        def count_before(module,args):
            if c['official_forward_attempts']>=MAX_FORWARDS:raise RuntimeError('Hard 1000-forward budget exhausted')
            c['official_forward_attempts']+=1;s=report['active_stage'];c['by_stage'][s]=c['by_stage'].get(s,0)+1
        def count_after(module,args,value):
            c['official_forward_returns']+=1;s=report['active_stage'];c['returns_by_stage'][s]=c['returns_by_stage'].get(s,0)+1
        base.register_forward_pre_hook(count_before);base.register_forward_hook(count_after)
        raw_dir=out/'raw';raw_dir.mkdir(exist_ok=False)
        baselines={};all_generated={};figures={1:{},2:{}};figure_targets={}
        # Two unique Base trajectories, shared by every geometry condition.
        for seed in SEEDS:
            report['active_stage']='Base';z=bank['noises'][seed].cuda()
            raw,path,stats=sample(base,z,None,None,0.,c);baselines[seed]=raw
            key=f'Base_seed{seed}';arrays[key]=raw;arrays[key+'_path']=path;c['formal_rollouts']+=1
            report['outputs'].append(dict(key=key,model='Base',seed=seed,lambda_max=0,parent=None,K=None,sampling=stats))
            if seed==SEEDS[0]:
                report['active_stage']='zero_branch_diagnostic';c0=c['official_forward_attempts']
                with stable_execution(base):
                    value,official_stats=euler_sample(base,z,torch.tensor([112],device='cuda'),torch.ones((1,112),device='cuda',dtype=torch.bool),precision='bf16')
                c['euler_updates']+=50;c['complete_rollouts']+=1;c['diagnostic_rollouts']+=1
                raw_official=value[0].cpu().numpy();same=np.array_equal(raw,raw_official)
                arrays['diagnostic_official_base_seed'+str(seed)]=raw_official
                report['diagnostics'].append(dict(name='zero_force_matches_official_sampler',status='PASS' if same else 'FAIL',bitwise_equal=same,
                    official_forwards=c['official_forward_attempts']-c0,sampling=official_stats))
                if not same:raise RuntimeError('Zero guidance failed bitwise official sampler identity')
            progress(dict(stage='Base',seed=seed,completed=c['complete_rollouts']))
        # Full GT geometry is used only below by independent scoring, never by sample().
        base_scores=[]
        for case in cases:
            parent=case['parent_index'];L=case['bbox_diagonal'];tau=3*(.05*L)**2
            for K in (1,2):
                targets=case['targets'][K];tc=torch.from_numpy(targets).to('cuda')
                if parent==0:
                    figures[K]['Base']=baselines[SEEDS[0]];figure_targets[K]=targets
                for seed in SEEDS:
                    base_metric=evaluate_output(baselines[seed],targets,case['full_target'],L,base_tokens=baselines[seed],check_points=case['checks'][K])
                    base_scores.append(dict(parent=parent,uid=case['uid'],K=K,seed=seed,metrics=base_metric))
                    for maximum in LAMBDAS:
                        report['active_stage']='Guided';z=bank['noises'][seed].cuda()
                        raw,path,stats=sample(base,z,tc,tau,maximum,c)
                        key=f'P{parent}_K{K}_seed{seed}_lambda{int(maximum)}'
                        arrays[key]=raw;arrays[key+'_path']=path;c['formal_rollouts']+=1
                        score=evaluate_output(raw,targets,case['full_target'],L,base_tokens=baselines[seed],check_points=case['checks'][K])
                        row=dict(key=key,model='guided',parent=parent,uid=case['uid'],K=K,seed=seed,lambda_max=maximum,tau=tau,
                            metrics=score,sampling=stats,initial_noise_sha256=array_hash(bank['noises'][seed]),targets_sha256=array_hash(targets))
                        report['outputs'].append(row);all_generated[(parent,K,seed,maximum)]=raw
                        if parent==0 and seed==SEEDS[0]:figures[K]['lambda'+str(int(maximum))]=raw
                        if parent==0 and K==1 and seed==SEEDS[0] and maximum==5.:
                            report['active_stage']='guided_repeat_diagnostic'
                            again,again_path,again_stats=sample(base,z,tc,tau,maximum,c);c['diagnostic_rollouts']+=1
                            arrays['diagnostic_repeat_'+key]=again;arrays['diagnostic_repeat_'+key+'_path']=again_path
                            same=np.array_equal(raw,again) and np.array_equal(path,again_path)
                            report['diagnostics'].append(dict(name='same_input_guidance_repeat_all51states',status='PASS' if same else 'FAIL',
                                bitwise_equal=same,sampling=again_stats,reference=key))
                            if not same:raise RuntimeError('Geometry guidance is not repeatable')
                        progress(dict(stage='Guided',parent=parent,K=K,seed=seed,lambda_max=maximum,completed=c['complete_rollouts']))
        report['base_scores']=base_scores
        report['raw']=_save_raw(raw_dir/'trajectories.npz',arrays)
        report['active_stage']='offline_visualization';plot_fixed(figures,figure_targets,out/'fixed_parent0_seed9101.png')
        report['figure']=dict(path=str(out/'fixed_parent0_seed9101.png'),sha256=file_sha256(out/'fixed_parent0_seed9101.png'),
            fixed_selection='parent0,seed9101, both K, all registered settings; same camera and coordinate bounds')
        assert c['official_forward_attempts']==1000 and c['official_forward_returns']==1000
        assert c['formal_rollouts']==18 and c['diagnostic_rollouts']==2 and c['complete_rollouts']==20
        assert c['geometry_gradient_attempts']==c['geometry_gradient_completed']==578 and c['euler_updates']==1000
        report['base_after']=frozen(base,True)
        report['protected_after']={str(p):file_record(p) for p in protected}
        report['protected_unchanged']=report['protected_after']==report['protected_before']
        if not report['protected_unchanged'] or file_sha256(protocol_path)!=protocol_sha:raise RuntimeError('Protected files/protocol changed')
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        write();progress(dict(stage='complete',status='PASS_EXECUTION',counts=c))
    except BaseException as error:
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        if arrays:report['partial_raw']=_save_raw(out/'partial_raw.npz',arrays)
        if base is not None:report['base_at_stop']=state_sha256(base)
        if protected:
            report['protected_after']={str(p):file_record(p) for p in protected}
            report['protected_unchanged']=report['protected_after']==report['protected_before']
        write();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(p.parse_args().output)
