"""Strict, bounded Q condition-learning diagnosis; distinct from Phase 2.

Fresh Q_correct/Q_independent, 1000 updates each. No deterministic fallback,
architecture search, changed loss, training geometry cache, or automatic resume.
"""
from .stable_runtime import stable_execution, assert_stable_runtime, configure_stable_runtime
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .adapters import ControlAdapter
from .official_wrapper import OfficialWrapper, load_official, state_sha256
from .learning_residuals import ResidualObserver
from .learning_data import load_bank, PairedLearningStream, base_tensor_hash
from .prompt_dataset import RealChairStream, collate_samples, collate_prompts, sample_input_hash, file_sha256
from .train_adapter import rng_state, state_sha256_from_state

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
OLD = PROJECT/'outputs/run_20260928_01'
CHECKPOINT = Path('/mnt/c/Users/warpwang/Downloads/last.pt')
EXPECTED_BASE = 'b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
EXPECTED_FILE = 'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
SAVE_STEPS = (0,200,500,1000)
SPARSE_STEPS = (1,2,10,50,200,500,1000)
PROMPT_KEYS = ('vertices','arity','primitive_mask','points','point_primitive','point_mask')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def frozen_check(base, expected, hash_state=False):
    if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
        raise RuntimeError('Official backbone frozen/eval/gradient invariant failed')
    if hash_state:
        value = state_sha256(base)
        if value != expected:
            raise RuntimeError('Official backbone state hash changed')
        return value
    return expected


def difference(a,b):
    a,b=a.detach().float().cpu(),b.detach().float().cpu()
    d=a.double()-b.double()
    return dict(bitwise_equal=torch.equal(a,b),max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()))


def old_cases():
    split=json.loads((OLD/'split_manifest.json').read_text())
    samples=RealChairStream(split['train'],seed=9010).next_effective_batch(8)
    digest=sample_input_hash(samples)
    if digest!='294bb081c6bc6a206a5febbcf7a8cda53cc9bdd6d0b5cccf25c317d03290bdc8':
        raise RuntimeError('Historical fixed teacher input identity changed')
    batch=collate_samples(samples[:2],'cuda')
    t=batch['t'];xt=t[:,None,None]*batch['tokens']+(1-t[:,None,None])*batch['noise']
    teacher=dict(name='historical_teacher_mb0',x=xt,t=t,y=batch['num_faces'],mask=batch['masks'],
        correct=batch['prompts'],mismatch=collate_prompts([samples[j]['prompt'] for j in (1,2)],'cuda'),
        source=dict(input_hash=digest,seed=9010,scope='first original microbatch2 of effective batch8'))
    pm=json.loads((OLD/'prompt_manifest.json').read_text())
    prompts=[]
    for parent in (0,1):
        rec=next(r for r in pm['records'] if r['parent_index']==parent and r['kind']=='mixed_2edges_1face')
        with np.load(rec['path'],allow_pickle=False) as saved:
            prompts.append({k:saved[k].copy() for k in PROMPT_KEYS})
    raw=OLD/'raw/Q_center_parent0_mixed_2edges_1face_seed4101_correct.npz'
    with np.load(raw,allow_pickle=False) as saved:
        z=torch.from_numpy(saved['initial_noise'].copy()).unsqueeze(0).cuda()
    sample=dict(name='historical_sampling_initial_state',x=z,t=torch.zeros(1,device='cuda'),
        y=torch.tensor([512],device='cuda'),mask=torch.ones(1,512,dtype=torch.bool,device='cuda'),
        correct=collate_prompts([prompts[0]],'cuda'),mismatch=collate_prompts([prompts[1]],'cuda'),
        source=dict(path=str(raw),sha256=file_sha256(raw),seed=4101,step=0))
    return (teacher,sample)


def residual_preflight(base, report, write):
    results=dict(status='RUNNING',optimizer_updates=0,cases=[],explicit_numeric_path_error=False)
    report['residual_preflight']=results
    cases=old_cases()
    for arm in ('Q_center','G_closest'):
        saved=torch.load(OLD/arm/'checkpoints/adapter_step0200.pt',map_location='cpu',weights_only=True)
        adapter=ControlAdapter(arm)
        adapter.load_state_dict(saved['adapter'],strict=True)
        adapter=adapter.cuda().requires_grad_(False).eval()
        before=state_sha256(adapter)
        wrapper=OfficialWrapper(base,adapter).eval()
        try:
            for case in cases:
                outputs={};row=dict(arm=arm,state=case['name'],source=case['source'],adapter_hash=before)
                def call(mode):
                    with torch.no_grad(),stable_execution(base,adapter):
                        row['runtime']=assert_stable_runtime(base,adapter,coordinates=case['x'])
                        value=wrapper(case['x'],case['t'],case['y'],mask=case['mask'],prompts=case[mode])
                    if not torch.isfinite(value).all():raise RuntimeError('Nonfinite old step200 output')
                    return value.detach().cpu()
                for mode in ('correct','mismatch'):outputs[mode]=call(mode)
                with ResidualObserver(wrapper) as observer:
                    for mode in ('correct','mismatch'):
                        with observer.capture(mode,case['mask']):outputs[mode+'_observed']=call(mode)
                    row['observation']=observer.report()
                    row['effective_correct_minus_mismatch']=observer.compare_effective('correct','mismatch')
                row['hook_output_checks']={}
                for mode in ('correct','mismatch'):
                    removed=call(mode)
                    checks=dict(installed=difference(outputs[mode],outputs[mode+'_observed']),removed=difference(outputs[mode],removed))
                    row['hook_output_checks'][mode]=checks
                    if not all(v['bitwise_equal'] for v in checks.values()):
                        raise RuntimeError('Residual hooks changed actual output or strict repetition failed')
                for capture in row['observation']['captures']:
                    if not all(r['next_input_equals_h_after'] for r in capture['layers']):
                        raise RuntimeError('Residual output differs from next-module input')
                results['cases'].append(row);write()
                print(json.dumps(dict(stage='residual_preflight',arm=arm,state=case['name'],status='PASS')),flush=True)
            if state_sha256(adapter)!=before:raise RuntimeError('Old adapter changed during read-only observation')
        finally:wrapper.close()
    results.update(status='PASS',gate='No observed boundary/dtype/nonfinite/hook path error; quantization statistics alone do not identify a learning-failure cause')
    write()


def group_names(adapter):
    groups=dict(condition_encoder=[],dynamic_mlp=[],output_projection=[],other_attention=[])
    for name,p in adapter.named_parameters():
        key=('condition_encoder' if name.startswith('condition_encoder.') else
             'dynamic_mlp' if '.dynamic_mlp.' in name else
             'output_projection' if '.output_projection.' in name else 'other_attention')
        groups[key].append((name,p))
    return groups


def magnitude(values):
    elements=sum(v.numel() for v in values)
    if not elements:return dict(elements=0,rms=0.,max_abs=0.,nonzero=0)
    square=sum(float(v.detach().double().square().sum()) for v in values)
    return dict(elements=elements,rms=math.sqrt(square/elements),
        max_abs=max(float(v.detach().abs().max()) for v in values),
        nonzero=sum(int(torch.count_nonzero(v.detach())) for v in values))


def grad_report(adapter):
    return {key:dict(**magnitude([p.grad for n,p in values if p.grad is not None]),
        tensors_with_grad=sum(p.grad is not None for n,p in values),parameter_tensors=len(values))
        for key,values in group_names(adapter).items()}


def update_report(adapter,before):
    return {key:magnitude([p.detach().cpu()-before[n] for n,p in values]) for key,values in group_names(adapter).items()}


def forward_loss(wrapper,batch):
    x1,x0,t=batch['tokens'],batch['noise'],batch['t']
    xt=t[:,None,None]*x1+(1-t[:,None,None])*x0
    assert_stable_runtime(wrapper.backbone,wrapper.adapter,coordinates=xt)
    prediction=wrapper(xt,t,batch['num_faces'],mask=batch['masks'],prompts=batch['prompts'],inference_geometry_cache=False)
    return ((prediction.float()-(x1-x0)).square()*batch['masks'].unsqueeze(-1)).sum()


def backward_gate(base,initial,bank,report,write):
    adapter=ControlAdapter('Q_center');adapter.load_state_dict(initial,strict=True);adapter=adapter.cuda()
    wrapper=OfficialWrapper(base,adapter).train()
    before=state_sha256(adapter)
    try:
        samples=PairedLearningStream(bank['parents'],bank).next_effective_batch(8)
        batch=collate_samples(samples[:2],'cuda')
        with stable_execution(base,adapter):
            runtime=assert_stable_runtime(base,adapter,coordinates=batch['tokens'])
            loss=forward_loss(wrapper,batch)/(int(batch['num_faces'].sum())*9)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite deterministic backward probe loss')
            report['counts']['backward_attempts']+=1
            loss.backward()
            report['counts']['backward_completed']+=1
        grads=grad_report(adapter)
        if not grads['output_projection']['nonzero']:raise RuntimeError('No gradient reached the zero-initialized exit')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise RuntimeError('Nonfinite deterministic backward probe gradient')
        if state_sha256(adapter)!=before:raise RuntimeError('Zero-update backward probe changed parameters')
        frozen_check(base,EXPECTED_BASE,True)
        report['strict_backward_gate']=dict(status='PASS',optimizer_updates=0,actual_forwards=1,
            actual_backwards=1,loss=float(loss),runtime=runtime,gradient_groups=grads,
            parameters_unchanged=True,base_frozen=True,training_geometry_cache=False)
        write()
    finally:
        adapter.zero_grad(set_to_none=True);wrapper.close()


def train_arm(base,initial,bank,arm,out,report,write,evaluator,reference=None):
    path=out/arm;path.mkdir(exist_ok=False);checkpoints=path/'checkpoints';checkpoints.mkdir()
    torch.manual_seed(10);np.random.seed(10);random.seed(10)
    adapter=ControlAdapter('Q_center');adapter.load_state_dict(initial,strict=True);adapter=adapter.cuda()
    wrapper=OfficialWrapper(base,adapter).train()
    stream=PairedLearningStream(bank['parents'],bank,arm=arm,seed=10)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0)
    result=dict(status='RUNNING',completed_updates=0,optimizer_step_attempts=0,
        optimizer_step_in_progress=False,input_audits=[],sparse_diagnostics=[],evaluations=[],checkpoints=[],
        initial_state_sha256=state_sha256(adapter),initial_shared_state_sha256=state_sha256_from_state(initial),
        frozen_hashes=[dict(step=0,sha256=frozen_check(base,EXPECTED_BASE,True))],
        independent_donor_counts=np.zeros((3,4,4),dtype=np.int64).tolist())
    report['arms'][arm]=result
    started=time.perf_counter()
    boundary_rng=None
    def save(step,label=None,recoverable=True):
        target=checkpoints/(label or f'adapter_step{step:04d}.pt')
        if target.exists():raise FileExistsError(target)
        torch.save(dict(checkpoint_schema='learning_diagnosis_v1',arm=arm,step=step,
            adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(stream),
            config=report['protocol'],base_state_sha256=EXPECTED_BASE,
            initial_state_sha256=result['initial_state_sha256'],bank_sha256=bank['manifest_sha256'],
            completed_updates=result['completed_updates'],optimizer_step_attempts=result['optimizer_step_attempts'],
            recoverable=recoverable,optimizer_step_in_progress=result['optimizer_step_in_progress'],
            input_audits=result['input_audits'],wrapper_counts=dict(wrapper.counts)),target)
        record=dict(step=step,path=str(target),sha256=file_sha256(target),adapter_sha256=state_sha256(adapter),recoverable=recoverable)
        result['checkpoints'].append(record);write()
    def evaluate(step):
        report['active_stage']=arm+f'_evaluation_step{step:04d}';write()
        result['evaluations'].append(evaluator.evaluate_checkpoint(wrapper,arm,step));write()
        wrapper.train()
    try:
        if result['initial_state_sha256']!=state_sha256_from_state(initial):raise RuntimeError('Fresh arm initialization differs')
        save(0);evaluate(0)
        with (path/'train.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,1001):
                report['active_stage']=arm+'_training'
                if report['counts']['optimizer_updates']>=2000:raise RuntimeError('Hard global 2000-update limit')
                tick=time.perf_counter();boundary_rng=rng_state(stream)
                samples=stream.next_effective_batch(8);audit=copy.deepcopy(stream.last_audit)
                if reference is not None:
                    expected=reference[step-1]
                    for key in ('base_input_hash','microbatch_base_hashes','padded_batch_base_hash','prompt_types','target_parent_uids','times','num_faces'):
                        if audit[key]!=expected[key]:raise RuntimeError(f'Paired arm input mismatch step {step}: {key}')
                optimizer.zero_grad(set_to_none=True)
                denominator=audit['valid_coordinates'];total_loss=0.
                for micro in range(4):
                    batch=collate_samples(samples[2*micro:2*micro+2],'cuda')
                    if base_tensor_hash(batch)!=audit['microbatch_base_hashes'][micro]:
                        raise RuntimeError('Actual CUDA microbatch differs from paired input audit')
                    with stable_execution(base,adapter):
                        if step==1 and micro==0:result['actual_training_runtime']=assert_stable_runtime(base,adapter,coordinates=batch['tokens'])
                        loss=forward_loss(wrapper,batch)/denominator
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite masked FM velocity loss')
                        report['counts']['backward_attempts']+=1
                        loss.backward()
                        report['counts']['backward_completed']+=1
                    total_loss+=float(loss.detach())
                diagnostic=None
                if step in SPARSE_STEPS:
                    snapshot={n:p.detach().cpu().clone() for n,p in adapter.named_parameters()}
                    diagnostic=dict(step=step,gradients_before_clip=grad_report(adapter))
                norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True))
                if diagnostic is not None:diagnostic['gradients_after_clip']=grad_report(adapter)
                frozen_check(base,EXPECTED_BASE)
                result['optimizer_step_attempts']+=1;report['counts']['optimizer_step_attempts']+=1
                result['optimizer_step_in_progress']=True
                optimizer.step();torch.cuda.synchronize()
                result['completed_updates']=step;report['counts']['optimizer_updates']+=1
                result['optimizer_step_in_progress']=False;boundary_rng=None
                result['input_audits'].append(audit)
                if not all(torch.isfinite(p).all() for p in adapter.parameters()):raise RuntimeError('Nonfinite adapter parameter after optimizer update')
                if diagnostic is not None:
                    diagnostic['actual_parameter_updates']=update_report(adapter,snapshot)
                    result['sparse_diagnostics'].append(diagnostic)
                    del snapshot
                if arm=='Q_independent':
                    for typ,target,donor in zip(audit['prompt_type_indices'],audit['target_parent_uids'],audit['donor_indices']):
                        result['independent_donor_counts'][typ][bank['parent_uids'].index(target)][donor]+=1
                row=dict(step=step,loss=total_loss,grad_norm_before_clip=norm,input_audit=audit,
                    diagnostic=diagnostic,seconds=time.perf_counter()-tick)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step in (1,2) or step%10==0:
                    print(json.dumps(dict(stage='learning_diagnosis_training',arm=arm,step=step,loss=total_loss,
                        grad_norm=norm,seconds=row['seconds'],total_actual_updates=report['counts']['optimizer_updates'])),flush=True)
                if step%50==0 or step in SPARSE_STEPS:write()
                if step in SAVE_STEPS:
                    result['frozen_hashes'].append(dict(step=step,sha256=frozen_check(base,EXPECTED_BASE,True)))
                    save(step);evaluate(step)
        result.update(status='PASS_EXECUTION',seconds=time.perf_counter()-started,
            wrapper_counts=dict(wrapper.counts),backbone_after=frozen_check(base,EXPECTED_BASE,True))
        if arm=='Q_independent':
            counts=np.asarray(result['independent_donor_counts'])
            result['independent_prompt_correct_fraction']=float(sum(np.trace(v) for v in counts)/counts.sum())
        write();return result['input_audits']
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),
            wrapper_counts=dict(wrapper.counts),failed_optimizer_may_be_partial=result['optimizer_step_in_progress'])
        try:save(result['completed_updates'],f'stopped_completed{result["completed_updates"]:04d}.pt',recoverable=False)
        except BaseException as save_error:result['failure_checkpoint_error']=repr(save_error)
        write();raise
    finally:wrapper.close()


def run(out):
    out=Path(out).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('learning_diagnosis_'):
        raise ValueError('Require a separate learning_diagnosis output directory')
    result_path=out/'learning_diagnosis_results.json'
    if result_path.exists():raise FileExistsError('This entrypoint never overwrites/replays an earlier run')
    torch.set_num_threads(8)
    config=configure_stable_runtime()
    report=dict(status='RUNNING',new_training_authorized=True,zero_training=False,phase3=False,
        active_stage='initialization',counts=dict(official_forward_attempts=0,official_forward_returns=0,
            official_by_stage={},optimizer_step_attempts=0,optimizer_updates=0,
            backward_attempts=0,backward_completed=0,base_updates=0,phase3_updates=0),arms={},
        protocol=dict(arms=['Q_correct','Q_independent'],adapter='Q_center',seed=10,updates_per_arm=1000,
            total_updates_cap=2000,effective_batch=8,microbatch=2,checkpoint_steps=list(SAVE_STEPS),
            gradient_diagnostic_steps=list(SPARSE_STEPS),optimizer='AdamW',lr=1e-4,betas=[.9,.95],weight_decay=0.,
            grad_clip=1.,loss='Original masked FM velocity MSE, sum over valid face coordinates / valid coordinates of effective batch8',
            scheduler=None,new_ema=None,geometry_cache=False,training_condition_encoding='Fresh autograd-connected encoding each microbatch; never retained across updates',
            precision='BF16 autocast; FP32 parameters and Euler integration',time_sampling='Original logistic-normal; unchanged',
            generation=dict(steps=50,dt=1/50,grid='FP32 k/50 for k=0..49',face_condition=512,face_cfg=1.,geometry_cfg=1.,
                seeds=[6101,6102],noise='Independent Gaussian, no target coupling'),
            teacher_times=[.05,.2,.5,.8,.95],final_endpoint=1000,model_selection=False),
        initial_runtime=config,environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,platform=platform.platform(),cublas_set_before_cuda=True))
    def write():atomic_json(result_path,report)
    write();base=None;started=time.perf_counter()
    protected=[CHECKPOINT,OLD/'Q_center/checkpoints/adapter_step0200.pt',OLD/'G_closest/checkpoints/adapter_step0200.pt',
        OLD/'split_manifest.json',OLD/'prompt_manifest.json',PROJECT/'outputs/debug_repeatability_20260928_150208/DEBUG_REPORT.md']
    try:
        report['protected_assets_before']={str(p):dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in protected}
        if report['protected_assets_before'][str(CHECKPOINT)]['sha256']!=EXPECTED_FILE:raise RuntimeError('Official checkpoint bytes changed')
        bank=load_bank(out);report['data_bank']=bank['manifest']
        report['data_bank_sha256']=bank['manifest_sha256']
        base,audit=load_official(CHECKPOINT,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        report['strict_load']=audit;report['backbone_before']=state_sha256(base)
        if report['backbone_before']!=EXPECTED_BASE:raise RuntimeError('Official EMA state identity changed')
        report['environment'].update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        def count_pre(module,args):
            c=report['counts'];c['official_forward_attempts']+=1
            stage=report['active_stage'];c['official_by_stage'][stage]=c['official_by_stage'].get(stage,0)+1
        def count_post(module,args,value):report['counts']['official_forward_returns']+=1
        base.register_forward_pre_hook(count_pre);base.register_forward_hook(count_post)
        report['active_stage']='old_step200_residual_preflight';write()
        residual_preflight(base,report,write)
        torch.manual_seed(10);np.random.seed(10);random.seed(10)
        fresh=ControlAdapter('Q_center');initial={k:v.detach().cpu().clone() for k,v in fresh.state_dict().items()}
        report['fresh_initialization']=dict(seed=10,sha256=state_sha256_from_state(initial),
            parameters=sum(p.numel() for p in fresh.parameters()),zero_exit=all(not torch.count_nonzero(v) for k,v in initial.items() if '.output_projection.' in k))
        if not report['fresh_initialization']['zero_exit']:raise RuntimeError('Fresh adapter exit is not zero-initialized')
        del fresh
        report['active_stage']='strict_backward_zero_update_probe';write()
        backward_gate(base,initial,bank,report,write)
        from .learning_evaluation import EvaluationRunner
        evaluator=EvaluationRunner(base,bank,out,counters=report['counts'])
        reference=train_arm(base,initial,bank,'Q_correct',out,report,write,evaluator)
        train_arm(base,initial,bank,'Q_independent',out,report,write,evaluator,reference=reference)
        if report['counts']['optimizer_updates']!=2000:raise RuntimeError('Completed training does not match authorized exact budget')
        report['backbone_after']=frozen_check(base,EXPECTED_BASE,True)
        report['protected_assets_after']={str(p):dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in protected}
        report['protected_assets_unchanged']=report['protected_assets_before']==report['protected_assets_after']
        if not report['protected_assets_unchanged']:raise RuntimeError('Protected history/weight files changed')
        report['paired_input_streams_all_1000_updates_equal']=True
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        write();print(json.dumps(dict(status=report['status'],counts=report['counts'],result=str(result_path))),flush=True)
    except BaseException as error:
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        if base is not None:
            try:report['backbone_at_stop']=state_sha256(base)
            except BaseException as hash_error:report['backbone_hash_error']=repr(hash_error)
        write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.output)
