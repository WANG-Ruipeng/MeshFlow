"""One fixed inference scan, one paired Pre-FFN training arm, held-out noises.

The sole optimization budget is 500 attempted AdamW steps. This entry never
resumes, retunes after confirmation, or modifies an earlier implementation.
"""
from .stable_runtime import configure_stable_runtime, stable_execution, assert_stable_runtime
import argparse
import copy
import json
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .official_wrapper import OfficialWrapper,load_official,state_sha256
from .s32_adapter import S32StaticAdapter
from .nxc112_run import (atomic_json,frozen,file_record,protected_paths,gradients,parameter_updates,training_loss,BASE_SHA,EMA,EMA_SHA,PROJECT,REPO)
from .nxc112_data import Balanced112Stream
from .prompt_dataset import collate_samples,file_sha256
from .train_adapter import restore_rng,rng_state,tensor_hash,state_sha256_from_state
from .inference_pre_wrapper import PreFFNWrapper,PreResidualObserver,active_pre_owner

OLD=PROJECT/'outputs/nxC_112_sandbox_20260928_173427'
AUTH=Path('/mnt/c/Users/warpwang/.codex/attachments/76a4672a-b265-43ad-bd86-ce723f28718c/Pasted text.txt')
SAVE_STEPS=(0,100,250,500)
SPARSE_STEPS=(1,2,10,50,100,250,500)
MATCH_KEYS=('batch_index','base_input_hash','padded_batch_base_hash','microbatch_base_hashes',
    'target_parent_uids','num_faces','times','valid_coordinates','mask_policy','parent_indices',
    'per_parent_counts','all_y_112','all_masks_identical_true','prompt_type','primitive_sample_counts')


def make_post(base,adapter):
    if active_pre_owner(base) is not None:raise RuntimeError('Close the Pre wrapper before constructing Post')
    if any(layer._forward_hooks for layer in base.layers):raise RuntimeError('Stale layer hooks before Post')
    return OfficialWrapper(base,adapter).eval()


def load_adapter(saved,trainable):
    adapter=S32StaticAdapter()
    loaded=adapter.load_state_dict(saved['adapter'],strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:raise RuntimeError('S32 strict initialization mismatch')
    if any(not torch.equal(adapter.state_dict()[key],value) for key,value in saved['adapter'].items()):
        raise RuntimeError('Not every initial S32 tensor is bitwise identical to the actual saved checkpoint')
    if sum(p.numel() for p in adapter.parameters())!=950144:raise RuntimeError('Changed parameter budget')
    adapter=adapter.cuda().requires_grad_(trainable)
    adapter.train(trainable)
    if state_sha256(adapter)!=state_sha256_from_state(saved['adapter']):raise RuntimeError('Adapter device transfer changed state')
    return adapter


def train_pre(base,adapter,bank,old,saved0,saved500,out,report,write,progress):
    folder=out/'stage_b';folder.mkdir(exist_ok=True)
    ckpts=folder/'checkpoints';ckpts.mkdir(exist_ok=False)
    stream=Balanced112Stream(bank,seed=10)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0.)
    optimizer.load_state_dict(saved0['optimizer'])
    if optimizer.state:raise RuntimeError('Old step0 optimizer must have no update history')
    wrapper=PreFFNWrapper(base,adapter).train()
    result=dict(status='RUNNING',arm='S32_static_PreFFN',completed_updates=0,optimizer_step_attempts=0,
        optimizer_step_in_progress=False,checkpoints=[],sparse_diagnostics=[],batch_audits=[],
        initial_state_sha256=state_sha256(adapter),frozen_hashes=[dict(step=0,sha256=frozen(base,True))])
    report['stage_b']['training']=result
    # Restore actual old step0 AFTER all construction and tests; this is not a
    # claim based only on reusing seed10. Balanced112Stream starts at batch0.
    restore_rng(saved0['rng'],stream)
    restored=rng_state(stream)
    def same_rng(left,right):
        if isinstance(left,torch.Tensor):return torch.equal(left,right)
        if isinstance(left,dict):return set(left)==set(right) and all(same_rng(left[k],right[k]) for k in left)
        if isinstance(left,(tuple,list)):return len(left)==len(right) and all(same_rng(x,y) for x,y in zip(left,right))
        return left==right
    result['actual_initial_rng_equals_old_step0']=same_rng(restored,saved0['rng'])
    if not result['actual_initial_rng_equals_old_step0']:raise RuntimeError('Saved initial RNG restoration failed')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def save(step,stopped=False):
        target=ckpts/(f'stopped_completed{step:04d}.pt' if stopped else f'adapter_step{step:04d}.pt')
        if target.exists():raise FileExistsError(target)
        torch.save(dict(schema='inference_injection_pilot_v1',arm='S32_static_PreFFN',injection='after_face_attention_broadcast_before_norm2_and_FFN',
            step=step,adapter=adapter.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(stream),
            base_state_sha256=BASE_SHA,initial_state_sha256=result['initial_state_sha256'],
            source_initial_checkpoint=report['assets']['checkpoint0'],protocol=report['protocol'],
            completed_updates=result['completed_updates'],optimizer_step_attempts=result['optimizer_step_attempts'],
            optimizer_step_in_progress=result['optimizer_step_in_progress'],recoverable=not stopped,
            wrapper_counts=dict(wrapper.counts)),target)
        result['checkpoints'].append(dict(step=step,path=str(target),file_sha256=file_sha256(target),
            state_sha256=state_sha256(adapter),stopped=stopped));write()
    try:
        save(0)
        for step in range(1,501):
            if report['counts']['optimizer_step_attempts']>=500:raise RuntimeError('Hard 500-attempt optimization budget reached')
            report['active_stage']='B_training';tick=time.perf_counter()
            samples=stream.next_effective_batch(8);audit=copy.deepcopy(stream.last_audit)
            expected=old['stage_b']['batch_audits'][step-1]
            for key in MATCH_KEYS:
                if audit[key]!=expected[key]:raise RuntimeError(f'Paired training input differs before update {step}: {key}')
            audit.update(step=step,actual_microbatch_hashes=[])
            optimizer.zero_grad(set_to_none=True);loss_total=0.;diagnostic=None
            sparse=step in SPARSE_STEPS
            enc_before=wrapper.counts['condition_encoding'];fw_before=wrapper.counts['backbone_forward']
            for micro in range(4):
                batch=collate_samples(samples[2*micro:2*micro+2],'cuda')
                digest=tensor_hash(batch);audit['actual_microbatch_hashes'].append(digest)
                if digest!=expected['actual_microbatch_hashes'][micro]:
                    raise RuntimeError(f'Full tensor/prompt microbatch mismatch before update {step}, micro{micro}')
                if sparse and micro==0:
                    with PreResidualObserver(wrapper,retain_effective=False) as observer:
                        with stable_execution(base,adapter):
                            with observer.capture(f'update{step}_micro0',batch['masks']):
                                loss,facts=training_loss(wrapper,batch)
                            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite paired masked FM loss')
                            report['counts']['backward_attempts']+=1
                            loss.backward();torch.cuda.synchronize();report['counts']['backward_completed']+=1
                        diagnostic=dict(step=step,residual_observation=observer.report(),runtime=facts)
                else:
                    with stable_execution(base,adapter):
                        loss,facts=training_loss(wrapper,batch)
                        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite paired masked FM loss')
                        report['counts']['backward_attempts']+=1
                        loss.backward();torch.cuda.synchronize();report['counts']['backward_completed']+=1
                loss_total+=float(loss.detach())
            if wrapper.counts['condition_encoding']-enc_before!=4 or wrapper.counts['backbone_forward']-fw_before!=4:
                raise RuntimeError('Every update must have four fresh condition encodes and four official forwards')
            if sparse:
                before={n:p.detach().cpu().clone() for n,p in adapter.named_parameters()}
                diagnostic['gradients_before_clip']=gradients(adapter)
            norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True))
            if sparse:diagnostic['gradients_after_clip']=gradients(adapter)
            frozen(base)
            result['optimizer_step_attempts']+=1;report['counts']['optimizer_step_attempts']+=1
            result['optimizer_step_in_progress']=True
            optimizer.step();torch.cuda.synchronize()
            result['completed_updates']=step;report['counts']['optimizer_updates']+=1
            result['optimizer_step_in_progress']=False
            if not all(bool(torch.isfinite(p).all()) for p in adapter.parameters()):raise RuntimeError('Nonfinite updated adapter')
            if sparse:
                diagnostic.update(parameter_updates=parameter_updates(adapter,before),loss=loss_total,grad_norm_before_clip=norm)
                result['sparse_diagnostics'].append(diagnostic);del before
            audit.update(loss=loss_total,grad_norm_before_clip=norm,matched_old_post=True,seconds=time.perf_counter()-tick)
            result['batch_audits'].append(audit)
            if step in (1,2) or step%10==0:
                progress(dict(stage='B_training',step=step,updates=report['counts']['optimizer_updates'],
                    loss=loss_total,matched_old_input=True,seconds=audit['seconds']))
            if step in SAVE_STEPS:
                result['frozen_hashes'].append(dict(step=step,sha256=frozen(base,True)));save(step)
        result['actual_final_rng_equals_old_post_step500']=same_rng(rng_state(stream),saved500['rng'])
        if not result['actual_final_rng_equals_old_post_step500']:raise RuntimeError('Final RNG states differ despite paired input stream')
        result.update(status='PASS_EXECUTION',seconds=time.perf_counter()-start,
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),wrapper_counts=dict(wrapper.counts),
            all_500_effective_batches_equal=True,all_2000_actual_microbatches_equal=True)
        write();return adapter
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),
            failed_optimizer_may_be_partial=result['optimizer_step_in_progress'],wrapper_counts=dict(wrapper.counts))
        try:save(result['completed_updates'],stopped=True)
        except BaseException as save_error:result['failure_checkpoint_error']=repr(save_error)
        write();raise
    finally:wrapper.close()


def run(output):
    from .inference_pilot_data import prepare_inputs
    from .inference_guidance import combination_precheck,SETTINGS
    from .inference_pilot_evaluation import InferenceEvaluator,select_candidate
    from .inference_precheck import run_prechecks
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('inference_injection_pilot_'):
        raise ValueError('Use a separate inference_injection_pilot output directory')
    result_path=out/'results.json'
    if result_path.exists():raise FileExistsError('Never overwrite or automatically restart an earlier attempt')
    torch.set_num_threads(8);initial_runtime=configure_stable_runtime()
    report=dict(schema='inference_injection_pilot_v1',status='RUNNING',active_stage='asset_audit',
        authorization=dict(path=str(AUTH),sha256=file_sha256(AUTH)),stage_b={},stage_c={},
        counts=dict(official_forward_attempts=0,official_forward_returns=0,official_by_stage={},
            official_returns_by_stage={},optimizer_step_attempts=0,optimizer_updates=0,
            backward_attempts=0,backward_completed=0,backbone_updates=0,other_arm_updates=0,phase3_updates=0),
        protocol=dict(training_arm='S32_static_PreFFN',max_optimizer_step_attempts=500,
            same_actual_initial_state_and_training_inputs_as='old Post step0 through step500',
            only_architecture_change='H_a + Adapter(H_a,C) enters both norm2/time-modulation/FFN and residual main path at layers3/6/9/12',
            effective_batch=8,microbatch=2,seed=10,checkpoints=list(SAVE_STEPS),diagnostic_steps=list(SPARSE_STEPS),
            adamw=dict(lr=1e-4,betas=[.9,.95],weight_decay=0.,clip=1.),scheduler=None,new_ema=None,
            loss='Original masked FM velocity MSE',time_sampling='Original logit-normal',
            N=112,y=112,mask='All True',face_cfg=1.,internal_control_scale=1.,steps=50,
            grid='FP32 t=k/50, k=0..49; FP32 state += v_final*(1/50)',
            guidance='Base-referenced final velocity guidance, not a trained geometry-null CFG predictor',
            combination='alpha1 only cond.float; alpha0 only off.float; otherwise off.float + alpha*(cond.float-off.float) at the SAME Xt',
            stage_a_setting_ids=['A0','A1','A2','A3','A4','A5'],development_seeds=list(range(7101,7105)),
            confirmation_seeds=list(range(8101,8117)),confirmation_peeking_before_selection_or_pre500=False,
            selection_rules='Exact user pre-registration; shared setting across parents, min J then NFE then A1..A5; must beat A0 J',
            new_training_authorized=True,phase3=False),
        initial_runtime=initial_runtime,environment=dict(python=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,cuda=torch.version.cuda,platform=platform.platform(),cublas_before_cuda=True))
    def write():atomic_json(result_path,report)
    def progress(event):report['latest_progress']=event;write();print(json.dumps(event,allow_nan=False),flush=True)
    write();base=None;protected=[];started=time.perf_counter()
    try:
        protected=sorted(set(protected_paths(out)+list(PROJECT.glob('*.py'))))
        report['protected_before']={str(p):file_record(p) for p in protected}
        if report['protected_before'][str(EMA)]['sha256']!=EMA_SHA:raise RuntimeError('Official weight file changed')
        bank,old,audit,saved0,saved500=prepare_inputs(out)
        if audit['status']!='PASS':raise RuntimeError('Assets, fresh confirmation noise or paired stream audit failed')
        report['assets']=audit;progress(dict(stage='asset_audit',status='PASS'))
        base,load=load_official(EMA,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        report['strict_load']=load;report['backbone_before']=frozen(base,True)
        report['environment'].update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        def count_pre(module,args):
            c=report['counts'];c['official_forward_attempts']+=1
            stage=report['active_stage'];c['official_by_stage'][stage]=c['official_by_stage'].get(stage,0)+1
        def count_post(module,args,value):
            c=report['counts'];c['official_forward_returns']+=1
            stage=report['active_stage'];c['official_returns_by_stage'][stage]=c['official_returns_by_stage'].get(stage,0)+1
        base.register_forward_pre_hook(count_pre);base.register_forward_hook(count_post)
        evaluator=InferenceEvaluator(base,bank,out,old,progress_cb=progress)
        stage_a_started=time.perf_counter()
        post_adapter=load_adapter(saved500,False)
        post=make_post(base,post_adapter)
        try:
            report['active_stage']='A_combination_precheck';write()
            historical_base=evaluator.load_historical_base()
            report['combination_precheck']=combination_precheck(base,post,bank,historical_base,output_dir=out/'stage_a')
            if report['combination_precheck']['status']!='PASS':raise RuntimeError('Final-velocity combination correctness gate failed')
            report['active_stage']='A_development';write()
            report['stage_a']=evaluator.evaluate_stage_a(post)
        finally:post.close()
        if report['stage_a']['status']!='PASS':raise RuntimeError('Stage A correctness did not pass')
        report['stage_a']['stage_wall_seconds_including_precheck']=time.perf_counter()-stage_a_started
        report['active_stage']='freeze_selection'
        selection=select_candidate(report['stage_a'])
        selection.update(frozen_before_pre_training=True,frozen_before_confirmation=True,
            old_post_state_sha256=state_sha256(post_adapter),data_audit_reference='results.json:assets',
            source_authorization_sha256=report['authorization']['sha256'],
            confirmation_seed_ids=list(range(8101,8117)),pre_training_completed_updates=0)
        selection_path=out/'selection.json'
        with selection_path.open('x',encoding='utf-8') as stream:json.dump(selection,stream,indent=2,allow_nan=False)
        selection_sha=file_sha256(selection_path)
        report['selection']=dict(path=str(selection_path),sha256=selection_sha,selected_setting=selection['selected_setting'])
        progress(dict(stage='selection_frozen',selected_setting=selection['selected_setting'],sha256=selection_sha))
        stage_b_started=time.perf_counter()
        pre_adapter=load_adapter(saved0,True);initial_sha=state_sha256(pre_adapter)
        report['stage_b']['initialization']=dict(state_sha256=initial_sha,source='Actual historical adapter_step0000.pt, strict tensor-by-tensor equality',parameters=950144)
        report['active_stage']='B_precheck';write()
        precheck=run_prechecks(base,pre_adapter,bank,counters=report['counts'])
        report['stage_b']['precheck']=precheck
        for key in ('backward_attempts','backward_completed'):report['counts'][key]+=precheck['actual_counts'][key]
        write()
        if precheck['status']!='PASS' or state_sha256(pre_adapter)!=initial_sha:raise RuntimeError('Pre correctness failed or formal step0 was contaminated')
        pre_adapter=train_pre(base,pre_adapter,bank,old,saved0,saved500,out,report,write,progress)
        pre=PreFFNWrapper(base,pre_adapter).eval()
        try:
            report['active_stage']='B_development';write()
            report['stage_b']['development']=evaluator.evaluate(pre,'Pre500','A0',bank['eval_noises'],'B_Pre500_development')
            report['active_stage']='B_teacher';write()
            report['stage_b']['teacher']=evaluator.teacher(pre,'Pre500')
            report['stage_b']['old_post_teacher_reference']=evaluator.historical_post_teacher()
        finally:pre.close()
        report['stage_b']['stage_wall_seconds_including_precheck']=time.perf_counter()-stage_b_started
        if file_sha256(selection_path)!=selection_sha:raise RuntimeError('Selection changed before confirmation')
        if report['counts']['optimizer_updates']!=500:raise RuntimeError('Confirmation requires the exactly completed Pre500 endpoint')
        report['stage_c']['settings_frozen_before_any_new_noise_forward']=True
        stage_c_started=time.perf_counter()
        confirmation=bank['confirmation_noises']
        report['active_stage']='C_Base';write()
        report['stage_c']['Base']=evaluator.evaluate_base(confirmation,'C_Base')
        post=make_post(base,post_adapter)
        try:
            report['active_stage']='C_Post500_A0';write()
            report['stage_c']['Post500_A0']=evaluator.evaluate(post,'Post500','A0',confirmation,'C_Post500_A0')
            chosen=selection['selected_setting']
            if chosen is not None:
                report['active_stage']='C_Post500_selected';write()
                report['stage_c']['Post500_selected']=evaluator.evaluate(post,'Post500',chosen,confirmation,'C_Post500_'+chosen)
        finally:post.close()
        pre=PreFFNWrapper(base,pre_adapter).eval()
        try:
            report['active_stage']='C_Pre500_A0';write()
            report['stage_c']['Pre500_A0']=evaluator.evaluate(pre,'Pre500','A0',confirmation,'C_Pre500_A0')
        finally:pre.close()
        report['stage_c']['stage_wall_seconds']=time.perf_counter()-stage_c_started
        report['selection']['sha256_after_confirmation']=file_sha256(selection_path)
        if report['selection']['sha256_after_confirmation']!=selection_sha:raise RuntimeError('Frozen inference selection changed after held-out results')
        report['backbone_after']=frozen(base,True)
        report['protected_after']={str(p):file_record(p) for p in protected}
        report['protected_unchanged']=report['protected_before']==report['protected_after']
        if not report['protected_unchanged']:raise RuntimeError('Protected source or history changed')
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        write();progress(dict(stage='complete',status='PASS_EXECUTION',counts=report['counts']))
    except BaseException as error:
        if hasattr(error,'evaluation_report'):report['failed_evaluation']=error.evaluation_report
        if hasattr(error,'input_audit'):report['failed_input_audit']=error.input_audit
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
