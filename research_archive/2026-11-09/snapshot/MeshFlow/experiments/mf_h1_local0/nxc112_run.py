"""Bounded N x C evaluation and one real-112-face 500-update sandbox.

Import stable_runtime before torch. This entry never resumes, selects a best
checkpoint, changes official source, or launches another training arm.
"""
from .stable_runtime import configure_stable_runtime, stable_execution, assert_stable_runtime
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from .official_wrapper import load_official, OfficialWrapper, state_sha256
from .prompt_dataset import collate_samples, file_sha256
from .learning_residuals import ResidualObserver
from .train_adapter import rng_state, tensor_hash, state_sha256_from_state

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
OLD = PROJECT / 'outputs/learning_diagnosis_20260928_155505'
EMA = Path('/mnt/c/Users/warpwang/Downloads/last.pt')
AUTH = Path('/mnt/c/Users/warpwang/.codex/attachments/25940ce7-7026-48b4-ad95-82d8ea385909/Pasted text.txt')
BASE_SHA = 'b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
EMA_SHA = 'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
CHECKPOINT_STEPS = (0,100,250,500)
DIAGNOSTIC_STEPS = (1,2,10,50,100,250,500)


def atomic_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def frozen(base, do_hash=False):
    if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
        raise RuntimeError('Official backbone must remain frozen, eval, and gradient-free')
    if do_hash:
        value = state_sha256(base)
        if value != BASE_SHA:
            raise RuntimeError('Official EMA state changed')
        return value
    return BASE_SHA


def file_record(path):
    return dict(bytes=path.stat().st_size, sha256=file_sha256(path))


def protected_paths(out):
    paths = [EMA]
    paths.extend(p for p in PROJECT.glob('*.py') if not p.name.startswith('nxc112_') and p.name!='s32_adapter.py')
    asset_dir=PROJECT/'assets/shapenet-03001627/objaverse_occ_v5_ids'
    paths.extend(asset_dir/(uid+'.npz') for uid in ('90c6d1df1f83329fe1181b0e584cdf9b',
        '6ec4d5fa96861bcd70311dea98ba710d','dae8b3ac69d80cf1b94e60a139ac4b1c',
        'd481cc9e7d6d7d64baf5b49ad7809302','2af09bd8df40506c9e646678ef50aa3d','5b0e833cf2ea465e42bd82e95587d8af'))
    # Protect every pre-existing experiment output, checkpoint, report and input. No official weight or historical directory is copied.
    for path in (PROJECT/'outputs').rglob('*'):
        if not path.is_file() or path.is_relative_to(out):
            continue
        paths.append(path)
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    paths.extend(REPO/name for name in tracked if name and (REPO/name).is_file())
    return sorted(set(paths))


def groups(adapter):
    result = dict(condition_encoder=[], attention_projections=[], output_projection=[])
    for name, p in adapter.named_parameters():
        key = ('condition_encoder' if name.startswith('condition_encoder.') else
               'output_projection' if '.output_projection.' in name else 'attention_projections')
        result[key].append((name,p))
    return result


def magnitude(values):
    count = sum(v.numel() for v in values)
    if not count:
        return dict(elements=0,rms=0.,max_abs=0.,nonzero=0)
    return dict(elements=count,rms=math.sqrt(sum(float(v.detach().double().square().sum()) for v in values)/count),
        max_abs=max(float(v.detach().abs().max()) for v in values),
        nonzero=sum(int(torch.count_nonzero(v.detach())) for v in values))


def gradients(adapter):
    return {key:dict(**magnitude([p.grad for _,p in values if p.grad is not None]),
        tensors_with_grad=sum(p.grad is not None for _,p in values), tensors=len(values))
        for key,values in groups(adapter).items()}


def parameter_updates(adapter,before):
    return {key:magnitude([p.detach().cpu()-before[name] for name,p in values])
        for key,values in groups(adapter).items()}


def training_loss(wrapper,batch):
    x1,x0,t = batch['tokens'],batch['noise'],batch['t']
    if x1.shape != (2,112,9) or x0.shape != x1.shape:
        raise RuntimeError('Training shape must be exactly microbatch2 x 112 x 9')
    if not bool((batch['num_faces']==112).all()) or batch['masks'].shape != (2,112) or not bool(batch['masks'].all()):
        raise RuntimeError('112 identity-channel invariant failed')
    xt = t[:,None,None]*x1+(1-t[:,None,None])*x0
    facts = assert_stable_runtime(wrapper.backbone,wrapper.adapter,coordinates=xt)
    prediction = wrapper(xt,t,batch['num_faces'],mask=batch['masks'],prompts=batch['prompts'],
        inference_geometry_cache=False)
    loss = ((prediction.float()-(x1-x0)).square()*batch['masks'].unsqueeze(-1)).sum()/(8*112*9)
    return loss,facts


def train_s32(base, adapter, bank, out, report, write, progress):
    from .nxc112_data import Balanced112Stream
    from .nxc112_evaluation import evaluate_stage_b
    path = out/'stage_b'; path.mkdir(exist_ok=True)
    checkpoint_dir=path/'checkpoints'; checkpoint_dir.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0.)
    stream = Balanced112Stream(bank,seed=10)
    wrapper = OfficialWrapper(base,adapter).train()
    result = dict(status='RUNNING',arm='S32_static',completed_updates=0,optimizer_step_attempts=0,
        optimizer_step_in_progress=False,checkpoints=[],evaluations=[],sparse_diagnostics=[],
        batch_audits=[],frozen_hashes=[dict(step=0,sha256=frozen(base,True))],
        initial_state_sha256=state_sha256(adapter),wrapper_counts={})
    report['stage_b'] = result
    base_results = None
    start=time.perf_counter()
    def save(step, stopped=False):
        target=checkpoint_dir/(f'stopped_completed{step:04d}.pt' if stopped else f'adapter_step{step:04d}.pt')
        if target.exists():raise FileExistsError(target)
        torch.save(dict(schema='nxc112_sandbox_v1',arm='S32_static',step=step,
            adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(stream),
            base_state_sha256=BASE_SHA,initial_state_sha256=result['initial_state_sha256'],
            bank_audit_sha256=report['bank_audit_sha256'],protocol=report['protocol'],
            completed_updates=result['completed_updates'],optimizer_step_attempts=result['optimizer_step_attempts'],
            optimizer_step_in_progress=result['optimizer_step_in_progress'],
            recoverable=not stopped,wrapper_counts=dict(wrapper.counts)),target)
        result['checkpoints'].append(dict(step=step,path=str(target),file_sha256=file_sha256(target),
            state_sha256=state_sha256(adapter),stopped=stopped))
        write()
    def evaluate(step):
        nonlocal base_results
        report['active_stage']=f'stage_b_evaluation_{step:04d}';write()
        evaluation=evaluate_stage_b(base,adapter,bank,step,path,base_results=base_results,
            progress_cb=progress,wrapper=wrapper)
        returned_base=evaluation.pop('base_results',None)
        if base_results is None:
            if returned_base is None:raise RuntimeError('Stage B evaluator omitted its shared Base evidence')
            base_results=returned_base;result['base_results']=returned_base
        evaluation['base_results_reference']='stage_b.base_results'
        result['evaluations'].append(evaluation);wrapper.train();write()
    try:
        save(0);evaluate(0)
        for step in range(1,501):
            if report['counts']['optimizer_updates']>=500:
                raise RuntimeError('Hard maximum 500 total optimizer updates')
            report['active_stage']='stage_b_training'
            tick=time.perf_counter()
            samples=stream.next_effective_batch(8)
            identity=[s['parent_uid'] for s in samples]
            if sorted(identity)!=sorted(bank['parent_uids']*4):
                raise RuntimeError('Each effective batch must contain four samples per parent')
            audit=copy.deepcopy(stream.last_audit)
            audit.update(step=step,actual_microbatch_hashes=[],actual_num_faces=[],valid_coordinates=8*112*9)
            optimizer.zero_grad(set_to_none=True)
            loss_total=0.;diagnostic=None
            encode_before=wrapper.counts['condition_encoding']
            forward_before=wrapper.counts['backbone_forward']
            sparse=step in DIAGNOSTIC_STEPS
            if sparse:
                diagnostic=dict(step=step,residual_observation=None)
            for micro in range(4):
                batch=collate_samples(samples[micro*2:micro*2+2],'cuda')
                audit['actual_microbatch_hashes'].append(tensor_hash(batch))
                audit['actual_num_faces'].extend(batch['num_faces'].cpu().tolist())
                if sparse and micro==0:
                    with ResidualObserver(wrapper,retain_effective=False) as observer:
                        with stable_execution(base,adapter):
                            with observer.capture(f'train_step{step}_micro0',batch['masks']):
                                loss,facts=training_loss(wrapper,batch)
                            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite masked FM loss')
                            report['counts']['backward_attempts']+=1
                            loss.backward();report['counts']['backward_completed']+=1
                        diagnostic['residual_observation']=observer.report()
                        diagnostic['runtime']=facts
                        if not all(row['next_input_equals_h_after'] for row in observer.records[0]['layers']):
                            raise RuntimeError('Residual boundary differs from actual next-module input')
                else:
                    with stable_execution(base,adapter):
                        loss,facts=training_loss(wrapper,batch)
                        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite masked FM loss')
                        report['counts']['backward_attempts']+=1
                        loss.backward();report['counts']['backward_completed']+=1
                loss_total+=float(loss.detach())
            if wrapper.counts['condition_encoding']-encode_before!=4 or wrapper.counts['backbone_forward']-forward_before!=4:
                raise RuntimeError('Every update requires exactly four fresh condition encodes and four backbone microbatch forwards')
            if sparse:
                snapshot={n:p.detach().cpu().clone() for n,p in adapter.named_parameters()}
                diagnostic['gradients_before_clip']=gradients(adapter)
            norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True))
            if sparse:diagnostic['gradients_after_clip']=gradients(adapter)
            frozen(base)
            result['optimizer_step_attempts']+=1;report['counts']['optimizer_step_attempts']+=1
            result['optimizer_step_in_progress']=True
            # No autocast context spans an optimizer update: avoid stale cast caches.
            optimizer.step();torch.cuda.synchronize()
            result['completed_updates']=step;report['counts']['optimizer_updates']+=1
            result['optimizer_step_in_progress']=False
            if not all(bool(torch.isfinite(p).all()) for p in adapter.parameters()):
                raise RuntimeError('Nonfinite trainable parameter after update')
            if sparse:
                diagnostic['parameter_updates']=parameter_updates(adapter,snapshot)
                diagnostic.update(loss=loss_total,grad_norm_before_clip=norm)
                result['sparse_diagnostics'].append(diagnostic);del snapshot
            audit.update(loss=loss_total,grad_norm_before_clip=norm,seconds=time.perf_counter()-tick)
            result['batch_audits'].append(audit)
            if step in (1,2) or step%10==0:
                progress(dict(stage='stage_b_training',step=step,loss=loss_total,grad_norm=norm,
                    seconds=audit['seconds'],updates=report['counts']['optimizer_updates']))
            if step in CHECKPOINT_STEPS:
                result['frozen_hashes'].append(dict(step=step,sha256=frozen(base,True)))
                save(step);evaluate(step)
        result.update(status='PASS_EXECUTION',seconds=time.perf_counter()-start,
            wrapper_counts=dict(wrapper.counts),backbone_after=frozen(base,True))
        write()
    except BaseException as error:
        if hasattr(error,'evaluation_report'):result['failed_evaluation']=error.evaluation_report
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),
            wrapper_counts=dict(wrapper.counts),failed_optimizer_may_be_partial=result['optimizer_step_in_progress'])
        try:save(result['completed_updates'],stopped=True)
        except BaseException as save_error:result['failure_checkpoint_error']=repr(save_error)
        write();raise
    finally:
        wrapper.close()


def run(output):
    from .nxc112_data import build_banks
    from .s32_adapter import S32StaticAdapter
    from .nxc112_precheck import run_prechecks
    from .nxc112_evaluation import run_stage_a
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('nxC_112_sandbox_'):
        raise ValueError('Require separate nxC_112_sandbox output directory')
    result_path=out/'results.json'
    if result_path.exists():raise FileExistsError('Do not overwrite or automatically resume an earlier execution')
    torch.set_num_threads(8)
    config=configure_stable_runtime()
    report=dict(schema='nxc112_sandbox_v1',status='RUNNING',active_stage='asset_preflight',
        authorization=dict(path=str(AUTH),sha256=file_sha256(AUTH)),
        new_training_authorized=True,phase3=False,
        counts=dict(official_forward_attempts=0,official_forward_returns=0,official_by_stage={},
            optimizer_step_attempts=0,optimizer_updates=0,backward_attempts=0,backward_completed=0,
            backbone_updates=0,other_arm_updates=0,phase3_updates=0),
        protocol=dict(stage_a_models=['Base','Q_correct_step1000','Q_independent_step1000'],
            stage_a_N=[217,348,750,768,512],stage_a_conditions=4,stage_a_noises=2,
            stage_a_planned_new_rollouts=90,stage_a_historical_output_reuse=0,
            arm='S32_static',seed=10,max_total_optimizer_updates=500,
            effective_batch=8,microbatch=2,per_parent_per_effective_batch=4,
            optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.95],weight_decay=0.,clip=1.),
            loss='Original masked FM velocity MSE, summed coordinates / (8*112*9)',
            time_distribution='Original logistic-normal sigmoid(N(0,1))',
            checkpoint_steps=list(CHECKPOINT_STEPS),sparse_diagnostic_steps=list(DIAGNOSTIC_STEPS),
            final_decision_step=500,best_checkpoint_selection=False,scheduler=None,new_ema=None,
            cache_during_training=False,condition_gradient='Fresh autograd-connected encoding every microbatch',
            geometry_scale=1.,face_cfg=1.,geometry_cfg=1.,sampling_steps=50,
            sampling_grid='FP32 k/50; state = state + velocity.float()*(1/50)',
            sampler_noise='CPU Gaussian independent of target; never OT paired',
            pilot=dict(each_parent_guide_decrease_pct=10.,both_direction_binding_noises_at_least=3,
                total_noises=4,no_nonfinite_or_obvious_shape_collapse=True,
                scope='Practical criterion on two fixed training objects; not a general or theoretical threshold')),
        initial_runtime=config,environment=dict(python=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,cuda=torch.version.cuda,platform=platform.platform(),
            cublas_set_before_cuda=True))
    def write():atomic_json(result_path,report)
    def progress(event):
        report['latest_progress']=event
        write();print(json.dumps(event,allow_nan=False),flush=True)
    write();base=None;started=time.perf_counter();protected=[]
    try:
        protected=protected_paths(out)
        report['protected_before']={str(p):file_record(p) for p in protected}
        if report['protected_before'][str(EMA)]['sha256']!=EMA_SHA:raise RuntimeError('Official checkpoint bytes differ')
        bank_a,bank_b,audit=build_banks(out)
        report['bank_audit']=audit
        report['bank_audit_sha256']=hashlib.sha256(json.dumps(audit,sort_keys=True,allow_nan=False).encode()).hexdigest()
        progress(dict(stage='asset_preflight',status='COMPLETE'))
        base,load=load_official(EMA,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        report['strict_load']=load;report['backbone_before']=frozen(base,True)
        report['environment'].update(gpu=torch.cuda.get_device_name(),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        def count_pre(module,args):
            c=report['counts'];c['official_forward_attempts']+=1
            stage=report['active_stage'];c['official_by_stage'][stage]=c['official_by_stage'].get(stage,0)+1
        def count_post(module,args,value):report['counts']['official_forward_returns']+=1
        base.register_forward_pre_hook(count_pre);base.register_forward_hook(count_post)
        report['active_stage']='stage_a';write()
        paths={arm:OLD/arm/'checkpoints/adapter_step1000.pt' for arm in ('Q_correct','Q_independent')}
        report['stage_a']=run_stage_a(base,bank_a,paths,out/'stage_a',progress_cb=progress)
        write()
        if not report['stage_a']['status'].startswith('PASS'):
            raise RuntimeError('Stage A execution failed; no Stage B training')
        if bank_b is None:
            report['stage_b']=dict(status='NOT_RUN_DATA_OR_PROMPT_GATE',completed_updates=0)
        else:
            torch.manual_seed(10);np.random.seed(10);random.seed(10)
            adapter=S32StaticAdapter().cuda()
            initial_sha=state_sha256(adapter)
            report['fresh_initialization']=dict(seed=10,state_sha256=initial_sha,
                parameters=sum(p.numel() for p in adapter.parameters()),arm=adapter.arm,
                zero_exit=all(int(torch.count_nonzero(p))==0 for n,p in adapter.named_parameters() if '.output_projection.' in n))
            if not report['fresh_initialization']['zero_exit']:raise RuntimeError('Fresh S32 exit is not zero')
            report['active_stage']='stage_b_precheck';write()
            report['stage_b_precheck']=run_prechecks(base,adapter,bank_b,counters=report['counts'])
            for key in ('backward_attempts','backward_completed'):
                report['counts'][key]+=report['stage_b_precheck']['actual_counts'][key]
            write()
            if report['stage_b_precheck']['status']!='PASS':
                raise RuntimeError('S32 prechecks failed; stop before any optimizer update')
            if state_sha256(adapter)!=initial_sha:raise RuntimeError('Nonzero test contaminated formal S32 initialization')
            frozen(base,True);write()
            train_s32(base,adapter,bank_b,out,report,write,progress)
        report['backbone_after']=frozen(base,True)
        report['protected_after']={str(p):file_record(p) for p in protected}
        report['protected_unchanged']=report['protected_before']==report['protected_after']
        if not report['protected_unchanged']:raise RuntimeError('Protected file identity changed')
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        if report['counts']['optimizer_updates'] not in (0,500):raise RuntimeError('Unexpected completed update count')
        write();progress(dict(stage='complete',status='PASS_EXECUTION',counts=report['counts']))
    except BaseException as error:
        if hasattr(error,'evaluation_report'):report['failed_evaluation']=error.evaluation_report
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        if base is not None:
            try:report['backbone_at_stop']=state_sha256(base)
            except BaseException as hash_error:report['backbone_hash_error']=repr(hash_error)
        if protected:
            try:
                report['protected_after']={str(p):file_record(p) for p in protected}
                report['protected_unchanged']=report['protected_before']==report['protected_after']
            except BaseException as hash_error:report['protected_hash_error']=repr(hash_error)
        write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
