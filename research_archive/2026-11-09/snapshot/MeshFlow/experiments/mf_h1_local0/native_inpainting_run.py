"""Bounded native inpainting: two 500-update full-model arms, fixed evaluation.

The official checkpoint and historical artifacts are read-only. The native
model is an experimental trainable copy, not a frozen-backbone adapter.
"""
from .stable_runtime import configure_stable_runtime, stable_execution, assert_stable_runtime
import argparse, copy, gc, hashlib, io, json, platform, random, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from .official_wrapper import load_official, state_sha256
from .nxc112_run import PROJECT, REPO, EMA, EMA_SHA, BASE_SHA, atomic_json, file_record, protected_paths, frozen
from .prompt_dataset import file_sha256
from .nxc112_evaluation import array_hash

OLD = PROJECT/'outputs/soft_matching_20260928_210321'
AUTH = Path('/mnt/c/Users/warpwang/.codex/attachments/e2b82b1a-2765-4bc1-8b58-8cd85e9dda2a/Pasted text.txt')
CONFIG = REPO/'configs/snet/base-120m-ot-v-chair.yaml'
DEVELOPMENT=(9101,9102)
CONFIRMATION=(9201,9202,9203,9204)
TEACHER_STEPS=(0,100,250,500)
CODE_FILES=('native_inpainting_data.py','native_inpainting_model.py','native_inpainting_metrics.py','native_inpainting_run.py','native_inpainting_report.py')


def scalar_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):scalar_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [scalar_tree(v) for v in value]
    return value


class Ledger:
    def __init__(self,report): self.report=report
    def attach(self,model,kind):
        def pre(module,args):
            c=self.report['counts']; stage=self.report['active_stage']
            if c['model_forward_attempts']>=16000: raise RuntimeError('Predeclared 16000 new model-forward ceiling reached')
            c['model_forward_attempts']+=1; c[kind+'_forward_attempts']+=1
            c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def post(module,args,output):
            c=self.report['counts'];c['model_forward_returns']+=1;c[kind+'_forward_returns']+=1
        return [model.register_forward_pre_hook(pre),model.register_forward_hook(post)]


def release(*objects):
    # Callers delete their owning references before this collection boundary.
    gc.collect();torch.cuda.empty_cache()


def fixed_case(bank,parent,K):
    return next(c for c in bank['cases'] if c['parent_index']==parent and c['K']==K)


def checkpoint_rng(stream):
    ns=np.random.get_state()
    return dict(stream=stream.state_dict(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),
        python=random.getstate(),numpy=dict(name=ns[0],keys=ns[1].tolist(),pos=int(ns[2]),has_gauss=int(ns[3]),cached=float(ns[4])))


def load_native(ledger):
    from .native_inpainting_model import NativeInpaintingModel
    backbone,loaded=load_official(EMA,CONFIG,'cuda')
    if state_sha256(backbone)!=BASE_SHA: raise RuntimeError('Official initial state hash mismatch')
    if any(m._forward_hooks or m._forward_pre_hooks for m in backbone.modules()): raise RuntimeError('Unexpected old hooks')
    native=NativeInpaintingModel(backbone,trainable=True).eval()
    ledger.attach(native,'native');ledger.attach(backbone,'official')
    return native,loaded


@torch.no_grad()
def clamped_sample(model,z,C,counts,native=True):
    from .native_inpainting_model import native_execution
    if z.dtype!=torch.float32 or z.shape!=(1,112,9) or C.dtype!=torch.float32: raise ValueError('Exact FP32 inputs required')
    C=C.reshape(-1,9);K=len(C)
    if K not in (1,2): raise ValueError('Only K1/K2')
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


def save_raw(out,key,value,path,record,report):
    destination=out/'raw'/(key+'.npz')
    if destination.exists(): raise FileExistsError(destination)
    with destination.open('xb') as f:
        payload=dict(output=value)
        if path is not None:payload['path']=path
        np.savez_compressed(f,**payload)
    record.update(key=key,raw_path=str(destination),raw_sha256=file_sha256(destination),output_sha256=array_hash(value))
    report['generated'].append(record)
    return record


@torch.no_grad()
def teacher(model,bank,step,arm,report):
    from .native_inpainting_data import NativeTrainingStream,native_collate,with_context
    from .native_inpainting_model import native_execution
    stream=NativeTrainingStream(bank,'Native_correct',seed=101010)
    samples=stream.next_effective_batch()
    examples={}
    for sample in samples:
        examples.setdefault((sample['parent_index'],sample['K']),sample)
    if len(examples)!=4: raise RuntimeError('Teacher must cover both parents and K values')
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
    return dict(step=step,arm=arm,rows=rows,forward_calls=40,
        target_noise='Fresh fixed seed101010 free-only OT; independent of training and complete-sampling Gaussian',
        no_parameter_updates=True)


def selected_parameters(model):
    names=('role_embedding','backbone.x_embedder.mlp.0.weight','backbone.time_embedder.mlp.0.weight',
        'backbone.layers.0.attn.qkv_proj.weight','backbone.final_layer.linear.weight')
    selected={n:p for n,p in model.named_parameters() if n in names}
    # Architecture spelling is audited in the result; always include representative
    # original matrices if the official coordinate module uses another name.
    for prefix in ('backbone.x_embedder.','backbone.t_embedder.','backbone.layers.0.','backbone.final_layer.'):
        if not any(n.startswith(prefix) for n in selected):
            for n,p in model.named_parameters():
                if n.startswith(prefix) and p.ndim>=2: selected[n]=p;break
    return selected


def magnitude(value):
    v=value.detach().float()
    return dict(elements=v.numel(),rms=float(v.double().square().mean().sqrt()),max_abs=float(v.abs().max()),nonzero=int(torch.count_nonzero(v)))


def train_arm(model,bank,arm,out,report,write,progress,expected_audits=None):
    from .native_inpainting_data import NativeTrainingStream,native_collate
    from .native_inpainting_model import native_execution,assert_native_runtime
    torch.manual_seed(10);np.random.seed(10);random.seed(10)
    stream=NativeTrainingStream(bank,arm,seed=10)
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
        torch.save(dict(schema='native_inpainting_v1',arm=arm,model=state,optimizer=cpu(optim),rng=checkpoint_rng(stream),
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
            if denominator!=7956:raise RuntimeError('Balanced 4K1+4K2 effective denominator must be7956')
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
    from .native_inpainting_data import prepare_inputs,NativeTrainingStream,native_collate
    from .native_inpainting_model import run_model_gates
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('native_inpainting_'):raise ValueError('New isolated output required')
    out.mkdir(exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('No overwrite, replay or automatic training resume')
    torch.set_num_threads(8);runtime=configure_stable_runtime()
    protocol=dict(arms={'Native_correct':500,'Native_independent':500},max_optimizer_step_attempts=1000,
        full_experiment_model_finetuning=True,official_original_file_readonly=True,base_readonly=True,phase3=False,
        parents=2,N_total=112,K=[1,2],source_face_max_vertex_tolerance_bbox=1e-6,
        training_seed=10,effective_batch=8,microbatch=1,balanced='2parents x K1/K2 x2 examples every update, shuffled',
        independent_donor='Uniform over both parents, independent RNG and per-example redraw; sameK; correct allowed probability1/2',
        optimizer={'name':'AdamW','lr':1e-5,'betas':[.9,.95],'weight_decay':0.,'grad_clip':1.},
        full_parameter_training=True,role_embedding_shape=[2,768],role_init='zeros',new_ema=False,scheduler=None,
        loss='Sum of free coordinate squared velocity error divided by actual effective-batch free coordinates (7956)',
        time_distribution='Original sigmoid(N(0,1))',free_pairing='Official nested OT at pre-final-times2 coordinate stage; fresh Gaussian per access',
        known_time=1,free_time='t',known_coordinates='Original archived FP32 C, firstK sampling slots, exact51-state preservation',
        training_permutations='Fresh face and withinface corners, role/time/masks carried with faces',
        teacher_steps=[0,100,250,500],teacher_times=[.05,.2,.5,.8,.95],teacher_seed=101010,
        teacher_contexts=['correct','swap'],diagnostic_grad_steps=[1,100,250,500],final_step=500,
        development_seeds=list(DEVELOPMENT),confirmation_seeds=list(CONFIRMATION),
        confirmation_only_after_both_endpoints_fixed=True,model_selection=False,
        steps=50,dt=.02,time_grid='FP32 k/50 k0..49',y=112,face_cfg=1,geometry_cfg=1,
        precision='Official BF16 autocast; FP32 parameters, conditions, noise and Euler states',
        determinism='CUBLAS=:4096:8 beforeCUDA; strictdeterministic warnFalse;TF32off;SDPAmath',
        all_modules_eval=True,gradient_checkpointing=False,compile=False,postprocessing='None except explicit B3 copy baseline',
        groups=['B0','B1','B2','B3','B4','B5','T1','T2'],soft_matching='Unchanged archived formula,tau,bboxmetadata,lambda5/20 schedule',
        remainder_surface_points=512,remainder_sampling_seed=9911,near_C_threshold_bbox=1e-3,
        fidelity_thresholds_bbox=[1e-3,1e-4,1e-5],cross_scoring='Extract remainder once using actual C; never reselect excluded faces',
        planned_new_formal_rollouts=132,planned_diagnostic_rollouts=2,reused_unique_old_rollouts=18,
        max_new_model_forwards=16000,expected_training_microbatch_forwards=8000,expected_teacher_forwards=320,
        predeclared_numeric_thresholds={'zero_repeat_atol':0.,'zero_repeat_rtol':0.,'fp32_formula_atol':1e-5,'fp32_formula_rtol':1e-5,'bf16_permutation_atol':.015625,'bf16_permutation_rtol':.03125})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
    report=dict(status='RUNNING',active_stage='preflight',protocol=protocol,protocol_sha256=file_sha256(out/'protocol.json'),
        authorization=dict(path=str(AUTH),**file_record(AUTH)),initial_runtime=runtime,arms={},generated=[],sampling_diagnostics=[],
        counts=dict(model_forward_attempts=0,model_forward_returns=0,official_forward_attempts=0,official_forward_returns=0,
            native_forward_attempts=0,native_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,
            optimizer_step_attempts=0,optimizer_updates=0,experimental_backbone_updates=0,protected_official_updates=0,
            geometry_gradient_attempts=0,geometry_gradient_completed=0,model_backward_calls=0,euler_updates=0,
            complete_rollouts=0,formal_rollouts=0,diagnostic_rollouts=0),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda),
        source_files={p.name:file_record(p) for p in PROJECT.glob('native_inpainting_*.py') if p.name!='native_inpainting_report.py'})
    def write():atomic_json(out/'results.json',report)
    def progress(event):report['latest_progress']=event;write();print(json.dumps(event,allow_nan=False),flush=True)
    write();protected=[];model=None;started=time.perf_counter();ledger=Ledger(report)
    try:
        protected=[p for p in protected_paths(out) if not p.name.startswith('native_inpainting_')]
        report['protected_before']={str(p):file_record(p) for p in protected}
        if report['protected_before'][str(EMA)]['sha256']!=EMA_SHA:raise RuntimeError('Official file hash mismatch')
        bank,audit=prepare_inputs(out);report['assets']=audit;write()
        if audit['status']!='PASS':raise RuntimeError('BLOCKED_CONTEXT_NOT_SOURCE_FACE or asset gate')
        from .native_inpainting_metrics import run_cpu_tests as metric_cpu_tests
        report['metric_cpu_tests']=metric_cpu_tests()
        from .native_inpainting_data import run_cpu_tests as data_cpu_tests
        from .native_inpainting_model import run_cpu_formula_tests
        report['data_cpu_tests']=data_cpu_tests(bank);report['formula_cpu_tests']=run_cpu_formula_tests()
        report['development_cpu_tests']=dict(data_OT_calls=50,metric_fixture_count=14,formula_fixture_runs=1,model_forwards=0,optimizer_updates=0);write()
        model,load=load_native(ledger);report['strict_load']=load
        report['environment'].update(gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        initial=state_sha256(model);report['native_initial_state_sha256']=initial
        stream=NativeTrainingStream(bank,'Native_correct',seed=10);samples=stream.next_effective_batch()
        chosen=[next(s for s in samples if s['K']==k) for k in (1,2)]
        batch=native_collate(chosen,'cuda');report['active_stage']='implementation_gates'
        report['model_gates']=run_model_gates(model,batch['xt'],batch['t'],batch['y'],batch['valid_mask'],batch['known_mask'],counters=report['counts'])
        write()
        if report['model_gates']['status']!='PASS':raise RuntimeError('Native implementation gates failed before training')
        if state_sha256(model)!=initial:raise RuntimeError('Disposable gate model not restored after nonzero-role test')
        del stream,samples,chosen,batch
        # Untrained native development outputs; the first is also the fixed repeat/reload gate.
        report['active_stage']='B5_development'
        for case in bank['cases']:
            parent,K=case['parent_index'],case['K']
            C=torch.from_numpy(case['constraints']).to('cuda')
            for seed in DEVELOPMENT:
                z=bank['noises'][seed].cuda();raw,path,stats=clamped_sample(model,z,C,report['counts'])
                key=f'B5_P{parent}_K{K}_seed{seed}'
                save_raw(out,key,raw,path,dict(model='B5',parent=parent,K=K,seed=seed,split='development',sampling=stats),report)
                report['counts']['formal_rollouts']+=1
                if parent==0 and K==1 and seed==9101:
                    for label in ('same_input_repeat','serialized_reload_repeat'):
                        report['active_stage']='sampling_gate_'+label
                        if label.startswith('serialized'):
                            payload={n:v.detach().cpu() for n,v in model.state_dict().items()}
                            buffer=io.BytesIO();torch.save(payload,buffer);serialized_sha=hashlib.sha256(buffer.getbuffer()).hexdigest();buffer.seek(0)
                            reload_state=torch.load(buffer,map_location='cpu',weights_only=True)
                            model.load_state_dict(reload_state,strict=True)
                            del payload,reload_state,buffer
                        again,again_path,diag=clamped_sample(model,z,C,report['counts'])
                        same=np.array_equal(raw,again) and np.array_equal(path,again_path)
                        report['counts']['diagnostic_rollouts']+=1
                        report['sampling_diagnostics'].append(dict(name=label,status='PASS' if same else 'FAIL',bitwise_equal=same,sampling=diag,
                            serialization='In-memory torch save/load of all native tensors, strict reload; no additional GPU model' if label.startswith('serialized') else None))
                        if not same:raise RuntimeError('Native repeated sampling mismatch')
                        del again,again_path
                        report['active_stage']='B5_development'
                progress(dict(stage='B5_development',parent=parent,K=K,seed=seed))
        report['pretraining_all_gates_pass']=True;write()
        model.zero_grad(set_to_none=True);del model;model=None;release()
        for arm in ('Native_correct','Native_independent'):
            model,_=load_native(ledger)
            if state_sha256(model)!=initial:raise RuntimeError('Both arms must have byte-identical complete initialization')
            expected=None if arm=='Native_correct' else report['arms']['Native_correct']['batch_audits']
            train_arm(model,bank,arm,out,report,write,progress,expected)
            del model;model=None;release()
        report['both_endpoints_fixed_before_confirmation']=True;write()
        evaluate_fixed(bank,out,report,ledger,write,progress)
        report['counts']['model_backward_calls']=report['counts']['backward_attempts']
        if report['counts']['optimizer_updates']!=1000 or report['counts']['optimizer_step_attempts']!=1000:raise RuntimeError('Expected exactly1000 updates')
        report['protected_after']={str(p):file_record(p) for p in protected}
        report['protected_unchanged']=report['protected_after']==report['protected_before']
        if not report['protected_unchanged']:raise RuntimeError('Protected files changed')
        if file_sha256(out/'protocol.json')!=report['protocol_sha256']:raise RuntimeError('Protocol changed')
        report.update(status='PASS_EXECUTION',active_stage='complete',elapsed_seconds=time.perf_counter()-started)
        progress(dict(stage='complete',status=report['status'],counts=report['counts']))
    except BaseException as error:
        report.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        report['counts']['model_backward_calls']=report['counts']['backward_attempts']
        if protected:
            report['protected_after']={str(p):file_record(p) for p in protected}
            report['protected_unchanged']=report['protected_after']==report['protected_before']
        write();raise


# Evaluation is defined below; no training path is reused or relabeled.
def evaluate_fixed(bank,out,report,ledger,write,progress):
    from .soft_matching_run import sample as original_soft_sample
    from .native_inpainting_metrics import evaluate_all_output,best_match_copy,binding_four_grid,plot_fixed
    if not report.get('both_endpoints_fixed_before_confirmation'):raise RuntimeError('Confirmation requires both fixed endpoints')
    c=report['counts'];source_old=bank['prior_results'];records={}
    report['reused']=[]
    for seed in DEVELOPMENT:
        key=f'Base_seed{seed}'
        old_record=next(r for r in source_old['outputs'] if r['key']==key)
        row=dict(model='B0',parent=None,K=None,seed=seed,split='development',key=key,raw_path=bank['prior_raw_path'],
            archive_key=key,reused=True,sampling=old_record['sampling'],new_model_forwards=0)
        records[('B0',None,None,seed)]=row;report['reused'].append(row)
    for case in bank['cases']:
        parent,K=case['parent_index'],case['K']
        for seed in DEVELOPMENT:
            for group,maximum in (('B1',5),('B2',20)):
                key=f'P{parent}_K{K}_seed{seed}_lambda{maximum}'
                old_record=next(r for r in source_old['outputs'] if r['key']==key)
                row=dict(model=group,parent=parent,K=K,seed=seed,split='development',key=key,raw_path=bank['prior_raw_path'],
                    archive_key=key,reused=True,sampling=old_record['sampling'],new_model_forwards=0)
                records[(group,parent,K,seed)]=row;report['reused'].append(row)
    for row in report['generated']:records[(row['model'],row['parent'],row['K'],row['seed'])]=row
    base,load=load_official(EMA,CONFIG,'cuda');ledger.attach(base,'official')
    report['readonly_base_before']=frozen(base,True);write()
    for seed in CONFIRMATION:
        report['active_stage']='B0_confirmation'
        raw,path,stats=original_soft_sample(base,bank['noises'][seed].cuda(),None,None,0.,c)
        key=f'B0_seed{seed}'
        row=save_raw(out,key,raw,path,dict(model='B0',parent=None,K=None,seed=seed,split='confirmation',sampling=stats),report)
        records[('B0',None,None,seed)]=row;c['formal_rollouts']+=1
        progress(dict(stage='B0_confirmation',seed=seed))
    for case in bank['cases']:
        parent,K=case['parent_index'],case['K'];C=torch.from_numpy(case['constraints']).cuda()
        for seed in CONFIRMATION:
            for group,maximum in (('B1',5),('B2',20)):
                report['active_stage']=group+'_confirmation'
                raw,path,stats=original_soft_sample(base,bank['noises'][seed].cuda(),C,case['soft_tau'],float(maximum),c)
                key=f'{group}_P{parent}_K{K}_seed{seed}'
                row=save_raw(out,key,raw,path,dict(model=group,parent=parent,K=K,seed=seed,split='confirmation',sampling=stats),report)
                records[(group,parent,K,seed)]=row;c['formal_rollouts']+=1
            progress(dict(stage='soft_confirmation',parent=parent,K=K,seed=seed))
        for seed in DEVELOPMENT+CONFIRMATION:
            report['active_stage']='B4_'+('development' if seed in DEVELOPMENT else 'confirmation')
            raw,path,stats=clamped_sample(base,bank['noises'][seed].cuda(),C,c,native=False)
            key=f'B4_P{parent}_K{K}_seed{seed}'
            row=save_raw(out,key,raw,path,dict(model='B4',parent=parent,K=K,seed=seed,split='development' if seed in DEVELOPMENT else 'confirmation',sampling=stats),report)
            records[('B4',parent,K,seed)]=row;c['formal_rollouts']+=1
        progress(dict(stage='B4',parent=parent,K=K))
    report['readonly_base_after']=frozen(base,True);del base;release();write()
    for group,arm,seeds in (('B5',None,CONFIRMATION),('T1','Native_correct',DEVELOPMENT+CONFIRMATION),('T2','Native_independent',DEVELOPMENT+CONFIRMATION)):
        native,_=load_native(ledger)
        if arm is not None:
            endpoint=report['arms'][arm]['checkpoint']
            if file_sha256(endpoint['path'])!=endpoint['sha256']:raise RuntimeError('Endpoint bytes changed before evaluation')
            saved=torch.load(endpoint['path'],map_location='cpu',weights_only=True,mmap=True)
            if saved['completed_updates']!=500 or saved['arm']!=arm:raise RuntimeError('Wrong fixed endpoint')
            native.load_state_dict(saved['model'],strict=True);del saved
            if state_sha256(native)!=endpoint['state_sha256']:raise RuntimeError('Full endpoint strict reload failed')
        elif state_sha256(native)!=report['native_initial_state_sha256']:raise RuntimeError('Native0 initialization changed')
        for case in bank['cases']:
            parent,K=case['parent_index'],case['K'];C=torch.from_numpy(case['constraints']).cuda()
            for seed in seeds:
                report['active_stage']=group+'_evaluation'
                raw,path,stats=clamped_sample(native,bank['noises'][seed].cuda(),C,c)
                key=f'{group}_P{parent}_K{K}_seed{seed}'
                row=save_raw(out,key,raw,path,dict(model=group,parent=parent,K=K,seed=seed,split='development' if seed in DEVELOPMENT else 'confirmation',sampling=stats),report)
                records[(group,parent,K,seed)]=row;c['formal_rollouts']+=1
            progress(dict(stage=group,parent=parent,K=K))
        if arm is not None and state_sha256(native)!=report['arms'][arm]['checkpoint']['state_sha256']:raise RuntimeError('Sampling mutated endpoint')
        del native;release()
    if c['formal_rollouts']!=132 or c['diagnostic_rollouts']!=2 or c['complete_rollouts']!=134:raise RuntimeError('Unexpected complete-generation accounting')
    if c['geometry_gradient_attempts']!=c['geometry_gradient_completed'] or c['geometry_gradient_completed']!=1088:raise RuntimeError('Unchanged soft-guidance call accounting differs')
    report['active_stage']='offline_scoring';report['scores']=[];score_index={};figure={}
    with np.load(bank['prior_raw_path'],allow_pickle=False) as archive:
        old_arrays={key:archive[key].copy() for key in archive.files}
    def read(row):
        if row.get('reused'):return old_arrays[row['archive_key']].copy()
        if file_sha256(row['raw_path'])!=row['raw_sha256']:raise RuntimeError('Generated raw file changed')
        with np.load(row['raw_path'],allow_pickle=False) as archive:return archive['output'].copy()
    for case in bank['cases']:
        parent,K=case['parent_index'],case['K'];C=case['constraints']
        target_cases=[fixed_case(bank,p,K) for p in (0,1)]
        remainders=[q['free_target'] for q in target_cases];diagonals=[q['bbox_diagonal'] for q in target_cases]
        known=np.arange(112)<K
        for seed in DEVELOPMENT+CONFIRMATION:
            packages={};base_raw=None
            for group in ('B0','B1','B2','B3','B4','B5','T1','T2'):
                if group=='B3':
                    raw,package,copy_audit=best_match_copy(base_raw,C)
                    key=f'B3_P{parent}_K{K}_seed{seed}'
                    record=save_raw(out,key,raw,None,dict(model='B3',parent=parent,K=K,seed=seed,split='development' if seed in DEVELOPMENT else 'confirmation',
                        copying=copy_audit,sampling=dict(nfe=0,geometry_gradient_calls=0,reuses='B0',seconds=0.)),report)
                    score,package=evaluate_all_output(raw,C,case['full_target'],remainders,diagonals,parent,
                        extraction=package,expected_copy_free=packages['B0']['free'])
                else:
                    record=records[(group,None,None,seed) if group=='B0' else (group,parent,K,seed)]
                    raw=read(record)
                    mode='assigned' if group in ('B0','B1','B2') else 'known_mask'
                    score,package=evaluate_all_output(raw,C,case['full_target'],remainders,diagonals,parent,
                        mode=mode,known_mask=known if mode=='known_mask' else None)
                    if group=='B0':base_raw=raw
                if len(package['full'])!=112 or len(package['free'])!=112-K:raise RuntimeError('Evaluation face-count invariant')
                packages[group]=package
                row=dict(model=group,parent=parent,K=K,seed=seed,split='development' if seed in DEVELOPMENT else 'confirmation',
                    raw_reference=record['key'],raw_path=record['raw_path'],metrics=score)
                report['scores'].append(row);score_index[(group,parent,K,seed)]=row
                if parent==0 and K==2 and seed==9101:figure[group]=package
            progress(dict(stage='offline_scoring',parent=parent,K=K,seed=seed,scored=len(report['scores'])))
    report['bindings']=[]
    for group in ('B0','B1','B2','B3','B4','B5','T1','T2'):
        for K in (1,2):
            for seed in DEVELOPMENT+CONFIRMATION:
                value=binding_four_grid(score_index[(group,0,K,seed)]['metrics'],score_index[(group,1,K,seed)]['metrics'])
                report['bindings'].append(dict(model=group,K=K,seed=seed,split='development' if seed in DEVELOPMENT else 'confirmation',**value))
    report['figure']=plot_fixed(figure,fixed_case(bank,0,2)['constraints'],out/'fixed_parent0_K2_seed9101.png')
    report['figure']['sha256']=file_sha256(report['figure']['path'])
    report['copy_free_sanity_all_pass']=all(q['metrics']['remainder']['own']==next(
        b['metrics']['remainder']['own'] for b in report['scores'] if b['model']=='B0' and all(b[k]==q[k] for k in ('parent','K','seed')))
        for q in report['scores'] if q['model']=='B3')
    if not report['copy_free_sanity_all_pass']:raise RuntimeError('Copy/Base remainder scores differ')
    if len(report['scores'])!=192 or len(report['bindings'])!=96:raise RuntimeError('Fixed evaluation matrix incomplete')
    write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)

