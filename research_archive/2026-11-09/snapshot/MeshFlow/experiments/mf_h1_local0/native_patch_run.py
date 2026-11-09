"""Connected native patches: exactly two fresh 500-update arms; raw seam diagnosis."""
from .stable_runtime import configure_stable_runtime, stable_execution
import argparse, copy, gc, hashlib, io, json, platform, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from .official_wrapper import state_sha256
from .nxc112_run import PROJECT, REPO, EMA, EMA_SHA, BASE_SHA, atomic_json, file_record, protected_paths
from .prompt_dataset import file_sha256
from .nxc112_evaluation import array_hash
from .native_inpainting_run import load_native, release, fixed_case, checkpoint_rng, save_raw, selected_parameters, magnitude
from .native_inpainting_model import native_execution, _difference, _grad_facts, NUMERICAL_POLICY

REGRESSION_SEEDS=(9201,9202,9203,9204)
TRAIN_K=(4,8,12)
EVAL_K=(2,4,8,12)
MAX_FORWARDS=14500

class Ledger:
    def __init__(self,report): self.report=report
    def attach(self,model,kind):
        def pre(module,args):
            c=self.report['counts']; stage=self.report['active_stage']
            if c['model_forward_attempts']>=MAX_FORWARDS: raise RuntimeError('Predeclared 14500 new model-forward ceiling reached')
            c['model_forward_attempts']+=1; c[kind+'_forward_attempts']+=1
            c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def post(module,args,output):
            c=self.report['counts'];c['model_forward_returns']+=1;c[kind+'_forward_returns']+=1
        return [model.register_forward_pre_hook(pre),model.register_forward_hook(post)]

@torch.no_grad()
def clamped_sample(model,z,C,counts,native=True):
    from .native_inpainting_model import native_execution
    if z.dtype!=torch.float32 or z.shape!=(1,112,9) or C.dtype!=torch.float32: raise ValueError('Exact FP32 inputs required')
    C=C.reshape(-1,9);K=len(C)
    if K not in EVAL_K: raise ValueError('Only registered nested K2/4/8/12')
    before=counts['model_forward_attempts'];source=array_hash(z);ch=array_hash(C)
    state=z.clone();state[:,:K]=C[None]
    known=torch.zeros((1,112),device=z.device,dtype=torch.bool);known[:,:K]=True
    valid=torch.ones_like(known);y=torch.tensor([112],device=z.device)
    path=[state[0].cpu().numpy().copy()];checks=[]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    context=native_execution(model,coordinates=state) if native else stable_execution(model)
    with context as facts:
        for k in range(50):
            if not torch.equal(state[0,:K],C): raise RuntimeError('Known coordinates changed before forward')
            t=torch.full((1,),k/50,device=z.device,dtype=torch.float32)
            v=model(state,t,y,valid,known).float() if native else model(state,t,y=y,mask=valid).float()
            next_state=state.clone()
            next_state[:,K:]=state[:,K:]+v[:,K:]*(1/50)
            next_state[:,:K]=C[None]
            state=next_state;counts['euler_updates']+=1
            ok=torch.equal(state[0,:K],C);checks.append(ok)
            if not ok or state.dtype!=torch.float32 or not bool(torch.isfinite(state).all()): raise RuntimeError('Invalid clamped Euler state')
            path.append(state[0].cpu().numpy().copy())
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    if counts['model_forward_attempts']-before!=50: raise RuntimeError('Expected exactly50 forward calls')
    if array_hash(z)!=source or array_hash(C)!=ch: raise RuntimeError('Sampling input mutated')
    path=np.stack(path)
    if not np.array_equal(path[:,:K],np.broadcast_to(C.cpu().numpy(),(51,K,9))): raise RuntimeError('51-state known check failed')
    if np.array_equal(path[0,K:],path[-1,K:]): raise RuntimeError('Free coordinates did not move')
    counts['complete_rollouts']+=1
    return state[0].cpu().numpy(),path,dict(nfe=50,geometry_gradient_calls=0,seconds=seconds,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),runtime=facts,known_all51_bitwise=True,known_state_checks=51,
        known_slots=list(range(K)),free_slots=list(range(K,112)),known_export_order='Original archived C face and vertex order',
        free_updated=True,noise_sha256=source,condition_sha256=ch,initial_distribution='C in firstK slots; original Gaussian slots K:112 retained',
        known_input_dtype='torch.float32',integration_dtype='torch.float32',total_faces=112,postprocessing=None)

def run_patch_model_gates(model, x, t, y, valid_mask, known_mask, counters=None):
    """Bounded root-invoked gates on one disposable full-model copy.

    Root-owned hooks count all native and reference model forwards. This helper
    additionally reports local call counts and adds only backward attempts and
    completions to the provided central counters. No optimizer is constructed.
    The caller must discard this gate model and freshly reload EMA for training.
    """
    from .official_wrapper import state_sha256
    report = dict(status='RUNNING', tests=[], numerical_policy=dict(NUMERICAL_POLICY),
                  actual_counts=dict(native_forward_attempts=0, native_forward_returns=0,
                                     reference_forward_attempts=0, reference_forward_returns=0,
                                     backward_attempts=0, backward_completed=0,
                                     optimizer_updates=0, complete_rollouts=0),
                  model_scope='Disposable independent EMA copy; reload before formal training',
                  nonzero_role_scope='Temporary role values on disposable copy, restored before exit')
    initial_hash = state_sha256(model.backbone)
    original_role = model.role_embedding.detach().clone()
    report['initial_backbone_sha256'] = initial_hash
    counts = report['actual_counts']
    handles = []

    def add(name, value):
        report['tests'].append(dict(name=name, **value))
        if value['status'] != 'PASS':
            raise RuntimeError('Native implementation gate failed: ' + name)

    def call(value, mask, *, observer=None, global_time=t, is_reference=False, precision=True):
        prefix = 'reference' if is_reference else 'native'
        counts[prefix + '_forward_attempts'] += 1
        with native_execution(model, coordinates=(value, global_time), autocast=precision):
            if is_reference:
                result = model.backbone(value, global_time, y, mask=valid_mask)
            else:
                result = model(value, global_time, y, valid_mask, mask, observer=observer)
        counts[prefix + '_forward_returns'] += 1
        return result

    try:
        if not model.experiment_trainable or not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Gate model must permit full-model backward')
        if not torch.equal(original_role, torch.zeros_like(original_role)):
            raise RuntimeError('Formal initialization must have exactly zero role embeddings')
        if sorted(known_mask.sum(1).tolist()) != [4, 8, 12]:
            raise RuntimeError('Gates require a real batch with K4, K8, K12')
        if not bool(((t > 0) & (t < 1)).all()):
            raise RuntimeError('Mixed-time gate requires free times strictly inside (0,1)')
        all_free = torch.zeros_like(known_mask)
        with torch.no_grad():
            native_fp32 = call(x, all_free, precision=False)
            base_fp32 = call(x, all_free, is_reference=True, precision=False)
            add('all_free_zero_role_fp32_matches_original',
                _difference(native_fp32, base_fp32, atol=1e-5, rtol=1e-5))
            del native_fp32, base_fp32
            native_bf16 = call(x, all_free)
            base_bf16 = call(x, all_free, is_reference=True)
            add('all_free_zero_role_bf16_matches_original_without_shortcut',
                _difference(native_bf16, base_bf16, atol=0, rtol=0))
            del native_bf16, base_bf16
            observed = []
            expected_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])

            def observe(event):
                if event['site'] == 'embedding':
                    return
                checks = []
                for chosen, free, known in zip(event['selected_mods'], event['free_mods'], event['known_mods']):
                    free_exp = free[:, None].expand_as(chosen)
                    known_exp = known[:, None].expand_as(chosen)
                    checks.append(torch.equal(chosen[~known_mask], free_exp[~known_mask])
                                  and torch.equal(chosen[known_mask], known_exp[known_mask]))
                observed.append(dict(site=event['site'], layer=event['layer'],
                                     time_exact=torch.equal(event['face_time'], expected_time),
                                     selected_modulations_exact=all(checks),
                                     known_context_valid=bool(valid_mask[known_mask].all()),
                                     free_and_known_modulation_differ=any(
                                         not torch.equal(f, k) for f, k in zip(event['free_mods'], event['known_mods']))))

            mixed = call(x, known_mask, observer=observe)
            repeated = call(x, known_mask)
            add('mixed_time_visits_every_block_and_final_head', dict(
                status='PASS' if len(observed) == len(model.backbone.layers) + 1
                and all(q['time_exact'] and q['selected_modulations_exact']
                        and q['known_context_valid'] and q['free_and_known_modulation_differ'] for q in observed)
                else 'FAIL', sites=observed,
                known_counts=known_mask.sum(1).tolist(),
                valid_counts=valid_mask.sum(1).tolist(),
                free_counts=(valid_mask & ~known_mask).sum(1).tolist()))
            add('same_input_repeated_native_forward_bitwise',
                _difference(mixed, repeated, atol=0, rtol=0))
            order = torch.arange(x.shape[1] - 1, -1, -1, device=x.device)
            face_perm = call(x[:, order], known_mask[:, order])
            add('bf16_face_role_time_permutation',
                _difference(face_perm, mixed[:, order], atol=.015625, rtol=.03125))
            corner_order = torch.tensor([2, 0, 1], device=x.device)
            corner_x = x.reshape(x.shape[0], 112, 3, 3)[:, :, corner_order].reshape_as(x)
            corner_pred = call(corner_x, known_mask)
            corner_expected = mixed.reshape(x.shape[0], 112, 3, 3)[:, :, corner_order].reshape_as(mixed)
            add('bf16_corner_permutation',
                _difference(corner_pred, corner_expected, atol=.015625, rtol=.03125))
            changed_context = x.clone()
            changed_context[known_mask] += .125
            changed_prediction = call(changed_context, known_mask)
            context_change = _difference(changed_prediction[~known_mask], mixed[~known_mask], atol=0, rtol=0)
            add('free_velocity_reads_known_context', dict(
                status='PASS' if not context_change['bitwise_equal'] else 'FAIL', difference=context_change))
            # This is the disposable full-model gate copy, not either formal arm.
            model.role_embedding[0].fill_(.0078125)
            model.role_embedding[1].fill_(-.0078125)
            nonzero_role_prediction = call(x, known_mask)
            role_change = _difference(nonzero_role_prediction, mixed, atol=0, rtol=0)
            add('nonzero_roles_on_disposable_copy_change_output', dict(
                status='PASS' if not role_change['bitwise_equal'] else 'FAIL', difference=role_change))
            model.role_embedding.copy_(original_role)
            del mixed, repeated, face_perm, corner_pred, changed_prediction, nonzero_role_prediction

        model.zero_grad(set_to_none=True)
        grad_input = x.detach().clone().requires_grad_(True)
        retained = {}

        def retain_context(event):
            if event['site'] == 'embedding':
                event['hidden'].retain_grad()
                retained['embedded_corners'] = event['hidden']

        with torch.enable_grad(), native_execution(model, coordinates=(grad_input, t)):
            counts['native_forward_attempts'] += 1
            pred = model(grad_input, t, y, valid_mask, known_mask, observer=retain_context)
            counts['native_forward_returns'] += 1
            pred.retain_grad()
            loss_mask = valid_mask & ~known_mask
            # Zero velocity target is a structural gate, not training data.
            denominator = int(loss_mask.sum()) * 9
            loss = pred.float().square()[loss_mask].sum() / denominator
            counts['backward_attempts'] += 1
            if counters is not None:
                counters['backward_attempts'] = counters.get('backward_attempts', 0) + 1
            loss.backward()
            torch.cuda.synchronize(x.device)
            counts['backward_completed'] += 1
            if counters is not None:
                counters['backward_completed'] = counters.get('backward_completed', 0) + 1
        selected_parameters = {
            'coordinate_embedding': model.backbone.x_embedder.mlp[0].weight,
            'first_attention_qkv': model.backbone.layers[0].attn.qkv_proj.weight,
            'first_attention_output': model.backbone.layers[0].attn.out_proj.weight,
            'time_embedding': model.backbone.t_embedder.mlp[0].weight,
            'final_velocity_head': model.backbone.final_layer.linear.weight,
            'role_embedding': model.role_embedding,
        }
        parameter_grads = {name: _grad_facts(parameter.grad) for name, parameter in selected_parameters.items()}
        embedded_grad = retained['embedded_corners'].grad.reshape(x.shape[0], 112, 3, -1)
        context_grad = _grad_facts(embedded_grad[known_mask])
        coordinate_grad = _grad_facts(grad_input.grad[known_mask])
        prediction_known_grad = pred.grad[known_mask]
        has_signal = lambda q: q['present'] and q['finite'] and q['max_abs'] > 0
        add('free_only_loss_backpropagates_through_known_context_and_backbone', dict(
            status='PASS' if all(has_signal(q) for q in parameter_grads.values())
            and has_signal(context_grad) and has_signal(coordinate_grad)
            and torch.equal(prediction_known_grad, torch.zeros_like(prediction_known_grad)) else 'FAIL',
            free_coordinate_denominator=denominator, free_counts=loss_mask.sum(1).tolist(),
            known_predicted_velocity_gradient_exact_zero=bool((prediction_known_grad == 0).all()),
            known_embedded_context_gradient=context_grad, known_coordinate_gradient=coordinate_grad,
            parameter_gradients=parameter_grads, loss=float(loss.detach())))
        model.zero_grad(set_to_none=True)
        if state_sha256(model.backbone) != initial_hash:
            raise RuntimeError('A zero-update gate changed pretrained backbone tensors')
        if not torch.equal(model.role_embedding, original_role):
            raise RuntimeError('Disposable role test failed to restore zero initialization')
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['backbone_parameter_count'] = sum(p.numel() for p in model.backbone.parameters())
        report['role_parameter_count'] = model.role_embedding.numel()
        report['backbone_unchanged'] = True
        report['role_restored_bitwise'] = True
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'STOPPED'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            model.role_embedding.copy_(original_role)
        model.zero_grad(set_to_none=True)
        report['final_backbone_sha256'] = state_sha256(model.backbone)
        report['final_role_restored'] = torch.equal(model.role_embedding, original_role)
    return report

@torch.no_grad()
def teacher(model,bank,step,arm,report):
    from .native_patch_data import PatchTrainingStream
    from .native_inpainting_data import native_collate,with_context
    from .native_inpainting_model import native_execution
    stream=PatchTrainingStream(bank,'Native_correct',seed=101010)
    samples=stream.next_effective_batch()
    examples={}
    for sample in samples:
        examples.setdefault((sample['parent_index'],sample['K']),sample)
    if len(examples)!=6: raise RuntimeError('Teacher must cover both parents and all three trained K values')
    rows=[];report['active_stage']=f'{arm}_teacher_{step}'
    for (parent,K),sample in sorted(examples.items()):
        for t_value in (.05,.2,.5,.8,.95):
            pair={}
            for mode,donor in (('correct',parent),('mismatch',1-parent)):
                item=with_context(sample,donor)
                batch=native_collate([item],'cuda')
                t=torch.full((1,),t_value,device='cuda',dtype=torch.float32)
                xt=t[:,None,None]*batch['x1']+(1-t[:,None,None])*batch['x0']
                xt=torch.where(batch['known_mask'][...,None],batch['xt'],xt)
                with native_execution(model,coordinates=xt):
                    v=model(xt,t,batch['y'],batch['valid_mask'],batch['known_mask']).float()
                loss_mask=batch['valid_mask']&~batch['known_mask'];u=batch['u']
                mse=float((v-u).square()[loss_mask].mean())
                if not np.isfinite(mse):raise RuntimeError('Nonfinite teacher loss')
                pair[mode]=dict(mse=mse,donor=donor)
            rows.append(dict(parent=parent,K=K,t=t_value,**pair))
    return dict(step=step,arm=arm,rows=rows,forward_calls=60,
        target_noise='Fresh fixed seed101010 free-only OT; independent of training and complete-sampling Gaussian',
        no_parameter_updates=True)

def train_arm(model,bank,arm,out,report,write,progress,expected_audits=None):
    from .native_patch_data import PatchTrainingStream
    from .native_inpainting_data import native_collate
    from .native_inpainting_model import native_execution,assert_native_runtime
    torch.manual_seed(10);np.random.seed(10);random.seed(10)
    stream=PatchTrainingStream(bank,arm,seed=10)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=0.)
    parameters=list(model.parameters()); selected=selected_parameters(model)
    initial=state_sha256(model);base_initial=state_sha256(model.backbone)
    if base_initial!=BASE_SHA or not all(p.requires_grad for p in parameters):raise RuntimeError('Full experiment model must start from trainable official EMA')
    result=dict(status='RUNNING',completed_updates=0,optimizer_step_attempts=0,optimizer_step_in_progress=False,batch_in_progress=False,
        initial_state_sha256=initial,initial_backbone_sha256=base_initial,parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),batch_audits=[],teacher=[],sparse_diagnostics=[],
        selected_parameter_names=list(selected),seed=10,training_seconds=0.,peak_cuda_allocated=0)
    report['arms'][arm]=result;write()
    def save(stopped=False):
        name='stopped.pt' if stopped else 'endpoint_step500.pt'
        dest=out/arm/name;dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError(dest)
        # CPU tensors avoid retaining another GPU backbone or optimizer.
        state={n:p.detach().cpu() for n,p in model.state_dict().items()}
        optim=optimizer.state_dict()
        def cpu(v):
            if torch.is_tensor(v):return v.detach().cpu()
            if isinstance(v,dict):return {k:cpu(a) for k,a in v.items()}
            if isinstance(v,list):return [cpu(a) for a in v]
            if isinstance(v,tuple):return tuple(cpu(a) for a in v)
            return v
        torch.save(dict(schema='native_connected_patch_v1',arm=arm,model=state,optimizer=cpu(optim),rng=checkpoint_rng(stream),
            completed_updates=result['completed_updates'],optimizer_step_attempts=result['optimizer_step_attempts'],
            optimizer_step_in_progress=result['optimizer_step_in_progress'],initial_state_sha256=initial,
            initial_backbone_sha256=BASE_SHA,protocol_sha256=report['protocol_sha256'],protocol=report['protocol'],
            recoverable=not result['optimizer_step_in_progress'] and not result.get('batch_in_progress',False) and len(result['batch_audits'])==result['completed_updates'],full_model_finetuned=True),dest)
        result['checkpoint']=dict(path=str(dest),**file_record(dest),state_sha256=state_sha256(model),
            backbone_state_sha256=state_sha256(model.backbone),recoverable=not result['optimizer_step_in_progress'] and not result.get('batch_in_progress',False) and len(result['batch_audits'])==result['completed_updates'])
        write()
    try:
        result['teacher'].append(teacher(model,bank,0,arm,report));write()
        for step in range(1,501):
            report['active_stage']=f'{arm}_training'
            c=report['counts']
            if c['optimizer_step_attempts']>=1000 or result['optimizer_step_attempts']>=500:raise RuntimeError('Hard training update budget')
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            result['batch_in_progress']=True
            samples=stream.next_effective_batch()
            denominator=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
            if denominator!=stream.last_audit['effective_free_coordinates'] or not 7200<=denominator<=7776:raise RuntimeError('Actual mixed-K free-coordinate denominator mismatch')
            audit=copy.deepcopy(stream.last_audit)
            if expected_audits is not None and audit['shared_batch_sha256']!=expected_audits[step-1]['shared_batch_sha256']:
                raise RuntimeError('Two-arm target/noise/OT/t/permutation input stream mismatch')
            optimizer.zero_grad(set_to_none=True);loss_total=0.;c0=c['model_forward_attempts']
            for sample in samples:
                batch=native_collate([sample],'cuda')
                with native_execution(model,coordinates=batch['xt']) as facts:
                    v=model(batch['xt'],batch['t'],batch['y'],batch['valid_mask'],batch['known_mask']).float()
                    mask=batch['valid_mask']&~batch['known_mask']
                    loss=(v-batch['u']).square()[mask].sum()/denominator
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite free FM loss; no numerical fallback')
                    c['backward_attempts']+=1;loss.backward();c['backward_completed']+=1
                loss_total+=float(loss.detach())
            if c['model_forward_attempts']-c0!=8:raise RuntimeError('Each update must use8 fresh microbatch forwards')
            sparse=step in (1,100,250,500)
            if sparse:
                before={n:p.detach().cpu().clone() for n,p in selected.items()}
                grads={n:(magnitude(p.grad) if p.grad is not None else None) for n,p in selected.items()}
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            c['optimizer_step_attempts']+=1;result['optimizer_step_attempts']+=1
            result['optimizer_step_in_progress']=True;write()
            # Exit every autocast context before optimizer.step to avoid stale casts.
            optimizer.step();torch.cuda.synchronize()
            c['optimizer_updates']+=1;c['experimental_backbone_updates']+=1
            result['completed_updates']=step;result['optimizer_step_in_progress']=False
            finite=torch.stack([torch.isfinite(p).all() for p in parameters]).all()
            if not bool(finite):raise RuntimeError('Nonfinite parameter after attempted update')
            elapsed=time.perf_counter()-tick
            result['training_seconds']+=elapsed;result['peak_cuda_allocated']=max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            audit.update(step=step,loss=loss_total,grad_norm_before_clip=norm,free_coordinate_denominator=denominator,seconds=elapsed)
            result['batch_audits'].append(audit);result['batch_in_progress']=False
            if sparse:
                deltas={n:magnitude(p.detach().cpu()-before[n]) for n,p in selected.items()}
                result['sparse_diagnostics'].append(dict(step=step,gradients=grads,actual_parameter_updates=deltas,runtime=facts))
                del before
            if step==1 or step%10==0:progress(dict(stage=arm,step=step,loss=loss_total,seconds=elapsed,total_updates=c['optimizer_updates']))
            if step in (100,250,500):result['teacher'].append(teacher(model,bank,step,arm,report));write()
        result['final_backbone_sha256']=state_sha256(model.backbone)
        if result['final_backbone_sha256']==BASE_SHA:raise RuntimeError('Full-model backbone did not change')
        save();result['status']='PASS_EXECUTION';write()
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc())
        try:save(stopped=True)
        except BaseException as e:result['save_error']=repr(e)
        write();raise
    finally:
        optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect()
    return result


def run(output):
    from .native_patch_data import prepare_inputs, PatchTrainingStream, run_cpu_tests as data_tests
    from .native_inpainting_data import native_collate
    from .native_inpainting_model import run_cpu_formula_tests
    from .native_patch_metrics import run_cpu_tests as metric_tests
    from .native_patch_seams import run_cpu_tests as seam_tests
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('native_patch_'):
        raise ValueError('New isolated native_patch output required')
    out.mkdir(exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('No overwrite or automatic training resume')
    torch.set_num_threads(8);runtime=configure_stable_runtime()
    protocol=dict(
        experiment='connected_native_patches_raw_seam_diagnosis',phase3=False,architecture_search=False,
        authorization='User selected two new 500-update arms, total1000; mixed K4/8/12; independent context control; seam diagnosis first, no new regularizer',
        arms={'Native_correct':500,'Native_independent':500},max_optimizer_step_attempts=1000,
        full_experiment_model_finetuning=True,initialization='Fresh official EMA plus zero role table for both arms; not old endpoints',
        original_ema_and_history_readonly=True,parents=2,N_total=112,training_K=list(TRAIN_K),evaluation_K=list(EVAL_K),
        K2_scope='New nested K2, evaluation only, out-of-training; not historical disjoint K2',
        patch_selection='Input geometry only: largest eligible interior horizontal seed; deterministic shared-edge frontier expansion; nested prefixes; no semantic labels',
        training_seed=10,effective_batch=8,microbatch=1,
        balanced='Each batch four samples per parent; K totals3/3/2 rotate; every3 batches each(parent,K)four samples',
        independent_donor='Uniform over both parents, independent RNG, sameK, each sample redrawn; correct donor probability1/2',
        optimizer={'name':'AdamW','lr':1e-5,'betas':[.9,.95],'weight_decay':0.,'grad_clip':1.},
        loss='Original sum free velocity squared error / actual effective-batch free coordinate count',
        regularizer=None,new_ema=False,scheduler=None,role_embedding_shape=[2,768],role_init='zeros',
        time_distribution='Original sigmoid(N(0,1))',known_time=1,free_time='t',
        free_pairing='Official free-only nested OT at pre-final-times2 stage; fresh Gaussian each access',
        training_permutations='Fresh face and cyclic corners; masks/time/roles share mappings',
        teacher_steps=[0,100,250,500],teacher_times=[.05,.2,.5,.8,.95],teacher_seed=101010,
        diagnostic_grad_steps=[1,100,250,500],final_step=500,
        regression_seeds=list(REGRESSION_SEEDS),noise_scope='Actual archived Gaussian arrays reused; not new independent confirmation noises; never train-OT starting noise',
        both_endpoints_fixed_before_full_evaluation=True,model_selection=False,
        steps=50,dt=.02,time_grid='FP32 k/50 for k0..49',y=112,face_cfg=1,geometry_cfg=1,
        precision='BF16 autocast; FP32 parameters, C, Gaussian and Euler coordinates',
        determinism='CUBLAS=:4096:8 before CUDA; strict deterministic warnFalse; TF32off; SDPA math',
        all_modules_eval=True,gradient_checkpointing=False,compile=False,
        postprocessing='NONE; optional boundary projection helper is NOT_RUN on experimental outputs',
        groups=['Native0','T1','T2'],surface_points=512,surface_seed=9911,common_R12_point_seed=9973,common_R12_parent_seed_offset=True,
        common_region='For each parent identical fixed points on R12; target-to-predicted-free coverage only; no reverse penalty against R12',
        seam_boundary_samples=512,seam_seed=9919,near_boundary_threshold_bbox=1e-3,
        seam_metric='Length-weighted known boundary to entire free surface; requested near-free-vertex gap with explicit count; empty is missing not zero',
        intersections='FP64 proper free-known triangle penetration, separated from tangent/shared/coplanar contact; not whole-mesh selfintersection certification',
        topology='Exact-coordinate indexed edge-incidence diagnostics; no tolerance welding; vertex-link manifold certification NOT_RUN',
        hypotheses={'P0_below_2pct':'Mean, median and each-seed success reported, not guaranteed',
            'monotonic':'Strict trends per seed and mean separately, R_K and fixed R12 separately; trained4/8/12 and eval-only2/4/8/12 separately',
            'binding':'K8 andK12 each4/4 with both parent directions positive; mean/median/9202 net-sum share/other3 means',
            'cause':'Insufficient context is an unproven hypothesis; this two-parent training experiment cannot establish generalization or industrial readiness'},
        planned_formal_rollouts=96,planned_diagnostic_rollouts=2,max_new_model_forwards=MAX_FORWARDS,
        expected_training_microbatch_forwards=8000,expected_teacher_forwards=480,
        numerical_policy=dict(NUMERICAL_POLICY))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
    code=[PROJECT/name for name in ('native_patch_data.py','native_patch_metrics.py','native_patch_seams.py','native_patch_run.py','native_inpainting_model.py','native_inpainting_data.py','native_inpainting_run.py','stable_runtime.py')]
    report=dict(status='RUNNING',active_stage='preflight',protocol=protocol,protocol_sha256=file_sha256(out/'protocol.json'),
        initial_runtime=runtime,arms={},generated=[],sampling_diagnostics=[],source_files={p.name:file_record(p) for p in code},
        counts=dict(model_forward_attempts=0,model_forward_returns=0,official_forward_attempts=0,official_forward_returns=0,
            native_forward_attempts=0,native_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,
            optimizer_step_attempts=0,optimizer_updates=0,experimental_backbone_updates=0,protected_official_updates=0,
            geometry_gradient_attempts=0,geometry_gradient_completed=0,model_backward_calls=0,euler_updates=0,
            complete_rollouts=0,formal_rollouts=0,diagnostic_rollouts=0),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))
    def write():atomic_json(out/'results.json',report)
    def progress(event):report['latest_progress']=event;write();print(json.dumps(event,allow_nan=False),flush=True)
    write();protected=[];model=None;started=time.perf_counter();ledger=Ledger(report)
    try:
        protected=[p for p in protected_paths(out) if not p.name.startswith('native_patch_')]
        report['protected_before']={str(p):file_record(p) for p in protected}
        if report['protected_before'][str(EMA)]['sha256']!=EMA_SHA:raise RuntimeError('Original official file hash mismatch')
        bank,audit=prepare_inputs(out);report['assets']=audit
        preview=out/'INPUT_PATCH_PREVIEW.png'
        report['input_preview']=dict(path=str(preview),**file_record(preview),verified_before_model_calls=True)
        report['development_cpu_checks']=dict(data_OT_calls=67,metrics_fixture_runs=1,seam_fixture_runs=2,model_forwards=0,optimizer_updates=0,scope='Agent development checks before formal gate rerun')
        write()
        if audit['status']!='PASS':raise RuntimeError('Input patch asset gate failed')
        report['data_cpu_tests']=data_tests(bank);report['metric_cpu_tests']=metric_tests()
        report['seam_cpu_tests']=seam_tests();report['formula_cpu_tests']=run_cpu_formula_tests();write()
        if any(report[key]['status']!='PASS' for key in ('data_cpu_tests','metric_cpu_tests','seam_cpu_tests','formula_cpu_tests')):
            raise RuntimeError('CPU tests failed')
        model,loaded=load_native(ledger);report['strict_load']=loaded
        report['environment'].update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        initial=state_sha256(model);report['native_initial_state_sha256']=initial
        stream=PatchTrainingStream(bank,'Native_correct',seed=10);samples=stream.next_effective_batch()
        chosen=[next(s for s in samples if s['K']==k) for k in TRAIN_K]
        batch=native_collate(chosen,'cuda');report['active_stage']='implementation_gates'
        report['model_gates']=run_patch_model_gates(model,batch['xt'],batch['t'],batch['y'],batch['valid_mask'],batch['known_mask'],counters=report['counts']);write()
        if report['model_gates']['status']!='PASS':raise RuntimeError('Native K4/8/12 gates failed before training')
        if state_sha256(model)!=initial:raise RuntimeError('Gate changed initialization')
        del stream,samples,chosen,batch
        # First registered Native0 result doubles as the repeat/reload gate.
        case=fixed_case(bank,0,4);C=torch.from_numpy(case['constraints']).cuda();z=bank['noises'][9201].cuda()
        report['active_stage']='Native0_pretraining_sampling_gate'
        raw,path,stats=clamped_sample(model,z,C,report['counts'])
        save_raw(out,'Native0_P0_K4_seed9201',raw,path,dict(model='Native0',parent=0,K=4,seed=9201,split='reused_regression',sampling=stats),report)
        report['counts']['formal_rollouts']+=1
        for label in ('same_input_repeat','serialized_reload_repeat'):
            report['active_stage']='sampling_gate_'+label
            if label.startswith('serialized'):
                payload={n:v.detach().cpu() for n,v in model.state_dict().items()}
                buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0)
                reloaded=torch.load(buffer,map_location='cpu',weights_only=True);model.load_state_dict(reloaded,strict=True)
                del payload,reloaded,buffer
            again,again_path,diag=clamped_sample(model,z,C,report['counts'])
            equal=np.array_equal(raw,again) and np.array_equal(path,again_path)
            report['counts']['diagnostic_rollouts']+=1
            report['sampling_diagnostics'].append(dict(name=label,status='PASS' if equal else 'FAIL',bitwise_equal=equal,sampling=diag))
            if not equal:raise RuntimeError('Native repeat/reload sampling differs')
        del raw,path,again,again_path,C,z
        report['pretraining_all_gates_pass']=True;write();progress(dict(stage='all_pretraining_gates_pass'))
        model.zero_grad(set_to_none=True);del model;model=None;release()
        for arm in ('Native_correct','Native_independent'):
            model,_=load_native(ledger)
            if state_sha256(model)!=initial:raise RuntimeError('Both arms must have identical complete initialization')
            expected=None if arm=='Native_correct' else report['arms']['Native_correct']['batch_audits']
            train_arm(model,bank,arm,out,report,write,progress,expected)
            del model;model=None;release()
        report['both_endpoints_fixed_before_full_evaluation']=True;write()
        evaluate_fixed(bank,out,report,ledger,write,progress)
        c=report['counts'];c['model_backward_calls']=c['backward_attempts']
        if c['optimizer_step_attempts']!=1000 or c['optimizer_updates']!=1000:raise RuntimeError('Update accounting differs')
        if c['formal_rollouts']!=96 or c['diagnostic_rollouts']!=2 or c['complete_rollouts']!=98 or c['euler_updates']!=4900:
            raise RuntimeError('Sampling accounting differs')
        if c['model_forward_attempts']!=c['model_forward_returns'] or c['backward_attempts']!=c['backward_completed']:
            raise RuntimeError('Incomplete model execution')
        report['protected_after']={str(p):file_record(p) for p in protected}
        report['protected_unchanged']=report['protected_after']==report['protected_before']
        report['source_files_after']={p.name:file_record(p) for p in code}
        report['source_files_unchanged']=report['source_files_after']==report['source_files']
        if not report['protected_unchanged'] or not report['source_files_unchanged']:raise RuntimeError('Protected input or execution source changed')
        if file_sha256(out/'protocol.json')!=report['protocol_sha256']:raise RuntimeError('Protocol changed')
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        progress(dict(stage='complete',status=report['status'],counts=c))
    except BaseException as error:
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        report['counts']['model_backward_calls']=report['counts']['backward_attempts']
        if protected:
            report['protected_after']={str(p):file_record(p) for p in protected}
            report['protected_unchanged']=report['protected_after']==report['protected_before']
        write();raise


def evaluate_fixed(bank,out,report,ledger,write,progress):
    from .native_patch_metrics import prepare_common_reference,evaluate_patch_output,patch_binding_four_grid,plot_patch_fixed
    from .native_patch_seams import diagnose_patch_seams,triangle_soup_topology
    if not report.get('both_endpoints_fixed_before_full_evaluation'):raise RuntimeError('Both fixed endpoints required')
    c=report['counts'];records={(q['model'],q['parent'],q['K'],q['seed']):q for q in report['generated']}
    for group,arm in (('Native0',None),('T1','Native_correct'),('T2','Native_independent')):
        model,_=load_native(ledger)
        if arm:
            endpoint=report['arms'][arm]['checkpoint']
            if file_sha256(endpoint['path'])!=endpoint['sha256']:raise RuntimeError('Endpoint bytes changed')
            payload=torch.load(endpoint['path'],map_location='cpu',weights_only=True,mmap=True)
            if payload['completed_updates']!=500 or payload['arm']!=arm:raise RuntimeError('Wrong endpoint')
            model.load_state_dict(payload['model'],strict=True);del payload
            expected_hash=endpoint['state_sha256']
        else:expected_hash=report['native_initial_state_sha256']
        if state_sha256(model)!=expected_hash:raise RuntimeError('Strict reload hash failed')
        for case in bank['cases']:
            parent,K=case['parent_index'],case['K'];C=torch.from_numpy(case['constraints']).cuda()
            for seed in REGRESSION_SEEDS:
                if (group,parent,K,seed) in records:continue
                report['active_stage']=group+'_evaluation'
                raw,path,stats=clamped_sample(model,bank['noises'][seed].cuda(),C,c)
                key=f'{group}_P{parent}_K{K}_seed{seed}'
                row=save_raw(out,key,raw,path,dict(model=group,parent=parent,K=K,seed=seed,split='reused_regression',sampling=stats),report)
                records[(group,parent,K,seed)]=row;c['formal_rollouts']+=1
            progress(dict(stage=group,parent=parent,K=K,formal_rollouts=c['formal_rollouts']))
        if state_sha256(model)!=expected_hash:raise RuntimeError('Sampling changed model state')
        del model;release()
    common_cases=[fixed_case(bank,p,12) for p in (0,1)]
    common=prepare_common_reference([q['free_target'] for q in common_cases],[q['bbox_diagonal'] for q in common_cases])
    report['common_reference_audit']=common['audit'];report['scores']=[];report['ground_truth_seams']=[];index={};figure={}
    for case in bank['cases']:
        parent,K=case['parent_index'],case['K'];C=case['constraints'];bbox=case['bbox_diagonal']
        target_cases=[fixed_case(bank,p,K) for p in (0,1)]
        remainders=[q['free_target'] for q in target_cases];diagonals=[q['bbox_diagonal'] for q in target_cases]
        report['ground_truth_seams'].append(dict(parent=parent,K=K,seam=diagnose_patch_seams(C,case['free_target'],bbox),
            full_topology=triangle_soup_topology(case['full_target'])))
        for seed in REGRESSION_SEEDS:
            for group in ('Native0','T1','T2'):
                report['active_stage']='offline_scoring'
                row=records[(group,parent,K,seed)]
                if file_sha256(row['raw_path'])!=row['raw_sha256']:raise RuntimeError('Raw output bytes changed')
                with np.load(row['raw_path'],allow_pickle=False) as archive:raw=archive['output'].copy()
                known=np.arange(112)<K
                seam=diagnose_patch_seams(C,raw[~known],bbox)
                score,package=evaluate_patch_output(raw,C,case['full_target'],remainders,common,diagonals,parent,known,seam_result=seam)
                score['full_topology']=triangle_soup_topology(raw)
                if len(package['full'])!=112 or len(package['free'])!=112-K:raise RuntimeError('Evaluation count mismatch')
                result=dict(model=group,parent=parent,K=K,seed=seed,split='reused_regression',raw_reference=row['key'],raw_path=row['raw_path'],metrics=score)
                report['scores'].append(result);index[(group,parent,K,seed)]=result
                if parent==0 and seed==9201:figure.setdefault(K,{})[group]=package
            progress(dict(stage='offline_scoring',parent=parent,K=K,seed=seed,scores=len(report['scores'])))
    report['bindings']=[]
    for group in ('Native0','T1','T2'):
        for K in EVAL_K:
            for seed in REGRESSION_SEEDS:
                binding=patch_binding_four_grid(index[(group,0,K,seed)]['metrics'],index[(group,1,K,seed)]['metrics'])
                report['bindings'].append(dict(model=group,K=K,seed=seed,split='reused_regression',**binding))
    if len(report['scores'])!=96 or len(report['bindings'])!=48:raise RuntimeError('Evaluation matrix incomplete')
    report['figure']=plot_patch_fixed(figure,{K:fixed_case(bank,0,K)['constraints'] for K in EVAL_K},out/'fixed_parent0_seed9201.png')
    report['figure']['sha256']=file_sha256(report['figure']['path']);write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
