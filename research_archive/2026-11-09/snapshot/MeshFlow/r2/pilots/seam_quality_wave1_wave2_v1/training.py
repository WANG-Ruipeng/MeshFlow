"""Two fixed +1000-update Wave2 arms from the same live B/AdamW/RNG endpoint.

No model construction or calls on import. Parent release follows complete Wave1
scoring. Partial freezing changes parameter gradients, not the forward graph.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import argparse
import copy
import json
import re
import signal
import time
import traceback
import types
from contextlib import contextmanager
import numpy as np
from meshflow_control import runtime
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models import native
from meshflow_control.seam import model_adapter
from meshflow_control.training import schedule40 as base
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import atomic_json, append_event
from r2.pilots.user_condition_finetune_v1 import trainer as historical
from r2.pilots.seam_pool47_couple_v1.training import resource, save_payload, tensors
from . import training_data as D

EXPERIMENT='R2_SEAM_QUALITY_WAVE1_WAVE2_V1'
SCHEMA='r2_seam_quality_training_v1'


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,value)


def frozen_name(name,arm):
    match=re.match(r'^backbone\.layers\.([0-9]+)\.',name)
    return arm=='SURGICAL_LATE4' and bool(match and int(match.group(1))<8)


def set_trainability(model,arm):
    if arm not in D.ARMS or len(model.backbone.layers)!=12:
        raise ValueError('Registered arm and actual twelve layers required')
    for name,p in model.named_parameters():p.requires_grad_(not frozen_name(name,arm))
    model._quality_expected_trainable=tuple(n for n,p in model.named_parameters() if p.requires_grad)
    model._quality_arm=arm
    model.eval()
    if arm=='SURGICAL_LATE4':
        prefixes={int(re.match(r'^backbone\.layers\.([0-9]+)\.',n).group(1))
                  for n,p in model.named_parameters() if not p.requires_grad}
        if prefixes!=set(range(8)):
            raise ValueError('Partial freeze must cover exactly layers0..7')
        for prefix in ('role_embedding','context_encoder.','backbone.x_embedder.',
                       'backbone.t_embedder.','backbone.y_embedder.','backbone.final_layer.'):
            selected=[p.requires_grad for n,p in model.named_parameters() if n.startswith(prefix)]
            if not selected or not all(selected):
                raise ValueError('Preprocessing/final parameters must remain trainable: '+prefix)


def assert_quality_runtime(model,coordinates=None,*,require_context=True):
    """Original native policy with one explicit exact parameter-mask extension."""
    facts=runtime.assert_stable_runtime(model,require_context=False,coordinates=coordinates)
    if any(m.training for m in model.modules()):raise RuntimeError('All modules must remain eval')
    for name,module in model.named_modules():
        if hasattr(module,'dropout') and isinstance(module.dropout,(int,float)) and name.endswith('attn') and float(module.dropout)!=0:
            raise RuntimeError('Nonzero attention dropout')
        if isinstance(module,torch.nn.Dropout) and module.training:raise RuntimeError('Active dropout')
    expected=tuple(n for n,p in model.named_parameters() if not frozen_name(n,model._quality_arm))
    actual=tuple(n for n,p in model.named_parameters() if p.requires_grad)
    if actual!=expected or actual!=model._quality_expected_trainable:
        raise RuntimeError('Registered exact parameter mask changed')
    if any(layer.gradient_checkpointing for layer in model.backbone.layers):raise RuntimeError('Checkpointing forbidden')
    if require_context:
        if native._DEPTH.get()<1:raise RuntimeError('Private native context missing')
        if not(torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled()
               and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled()):
            raise RuntimeError('SDPA math only required')
        if not torch.is_autocast_enabled('cuda') or not native._BF16.get() or torch.get_autocast_dtype('cuda')!=torch.bfloat16:
            raise RuntimeError('Native BF16 autocast differs')
    facts.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trainable_mask_exact=True,partial_freeze=model._quality_arm=='SURGICAL_LATE4',
                 frozen_layers_still_in_autograd=True)
    return facts


def private_function(function):
    namespace=dict(function.__globals__)
    namespace['assert_native_runtime']=assert_quality_runtime
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


PRIVATE_SEAM_FORWARD=private_function(model_adapter.seam_forward)


def bind_private_forward(model):
    # No global patch. The bytecode of both arithmetic paths is exactly the
    # maintained original; only its validation symbol has a private extension.
    model.forward=types.MethodType(private_function(native.NativeModel.forward),model)


@contextmanager
def execution(model,coordinates):
    runtime.configure_stable_runtime()
    with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
        depth=native._DEPTH.set(native._DEPTH.get()+1)
        precision=native._BF16.set(True)
        try:
            yield assert_quality_runtime(model,coordinates)
            assert_quality_runtime(model,coordinates)
        finally:
            native._BF16.reset(precision);native._DEPTH.reset(depth)


def parameter_names(model,trainable=True):
    return [['generator.'+n for n,p in model.named_parameters() if p.requires_grad or not trainable]]


def restore_optimizer(model,payload,*,resume=False):
    """Prune by saved name/ID pairing; retained tensors are not reinitialized."""
    source=payload['optimizer'];source_names=payload['optimizer_parameter_names']
    wanted=parameter_names(model)
    if len(source['param_groups'])!=1 or len(source_names)!=1:
        raise ValueError('Exactly one named AdamW source group required')
    if source_names!=(wanted if resume else parameter_names(model,False)):
        raise ValueError('Saved optimizer parameter names/order differ')
    ids=source['param_groups'][0]['params']
    if len(ids)!=len(source_names[0]) or len(set(ids))!=len(ids):raise ValueError('Ambiguous source AdamW identity')
    byname=dict(zip(source_names[0],ids))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    selected={i:source['state'][byname[name]] for i,name in enumerate(wanted[0])}
    group={k:copy.deepcopy(v) for k,v in source['param_groups'][0].items() if k!='params'}
    group['params']=list(range(len(params)))
    subset=dict(state=selected,param_groups=[group])
    optimizer.load_state_dict(subset)
    restored=optimizer.state_dict()
    if base.tree_hash(restored)!=base.tree_hash(subset):
        raise ValueError('Retained AdamW tensor values or group settings changed on restore')
    source_moments={name:base.tree_hash(source['state'][byname[name]]) for name in wanted[0]}
    actual_moments={name:base.tree_hash(restored['state'][i]) for i,name in enumerate(wanted[0])}
    if actual_moments!=source_moments:raise ValueError('Per-parameter moment equality failed')
    if model._quality_arm=='CLEAN_BULK_FM' and base.tree_hash(restored)!=payload['optimizer_state_sha256']:
        raise ValueError('Unpruned optimizer must exactly equal original')
    return optimizer,dict(status='PASS',source_optimizer_sha256=payload['optimizer_state_sha256'],
        actual_optimizer_sha256=base.tree_hash(restored),retained_parameter_states=len(params),
        removed_frozen_parameter_states=len(ids)-len(params),retained_moment_hashes=actual_moments,
        exact_retained_step_and_moments=True,name_mapping='Saved generator parameter name -> saved optimizer ID')


def validate_optimizer(model,optimizer,step,hyper=None):
    if len(optimizer.param_groups)!=1:raise ValueError('One optimizer group required')
    group=optimizer.param_groups[0]
    if (group['lr'],tuple(group['betas']),group['eps'],group['weight_decay'])!=(1e-5,(.9,.95),1e-8,0.):
        raise ValueError('Inherited AdamW hyperparameters changed')
    selected=[p for p in model.parameters() if p.requires_grad]
    if [id(p) for p in group['params']]!=[id(p) for p in selected]:raise ValueError('Optimizer owns wrong parameters')
    if {id(p) for p in optimizer.state}!={id(p) for p in selected}:raise ValueError('Frozen/missing optimizer states')
    for p in selected:
        entry=optimizer.state[p]
        if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=D.BASE_STEP+step:
            raise ValueError('AdamW per-parameter step differs')
        if any(entry[k].dtype!=torch.float32 or entry[k].shape!=p.shape for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('AdamW moments dtype/shape differs')
    if hyper is not None and historical.optimizer_hyperparameters(optimizer)!=hyper:
        raise ValueError('AdamW hyperparameters changed after load')


def weighted_loss(prediction,target,weights,denominator):
    if prediction.shape!=target.shape or prediction.shape[-1]!=9 or weights.shape!=(*prediction.shape[:2],3):
        raise ValueError('Weighted FM coordinate shape differs')
    if weights.dtype!=torch.float32 or not denominator>0:raise ValueError('Invalid weighted divisor')
    difference=(prediction.float()-target.float()).reshape(*weights.shape,3)
    return (difference.square()*weights[...,None]).sum()/denominator


def update(model,optimizer,samples,arm,step,ledger,hyper,bins):
    plain,weighted=D.batch_denominators(samples)
    denominator=weighted if arm=='CLEAN_BULK_FM' else plain
    validate_optimizer(model,optimizer,step-1,hyper)
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter()
    result=dict(FM=0.,forward_seconds=0.,backward_seconds=0.,free_coordinate_denominator=plain,
        weighted_free_coordinate_denominator=weighted,loss_denominator=denominator,
        effective_unit_seam_coefficient_ratio_vs_plain=plain/denominator,
        head_status='FROZEN_NOT_LOADED_OR_CALLED',occupancy_BCE=None)
    diagnostics=[]
    for micro,sample in enumerate(samples):
        b=tensors(sample)
        context=dict(local_step=step,global_step=D.BASE_STEP+step,microbatch=micro,
                     sample_index=sample['sample_index'],uid=sample['uid'],alpha=sample['alpha'])
        with execution(model,(b['x'],b['t'])):
            fn=lambda:PRIVATE_SEAM_FORWARD(model,b['x'],b['t'],b['y'],b['valid'],b['C'],b['corner']).float()
            prediction,elapsed=ledger.call('train_forward',fn,context);result['forward_seconds']+=elapsed
            if arm=='CLEAN_BULK_FM':
                weights=torch.as_tensor(sample['quality_weight'][None],device='cuda')
                loss=weighted_loss(prediction,b['target'],weights,denominator)
            else:
                loss=model_adapter.corner_masked_fm_loss(prediction,b['target'],b['valid'],b['corner'],denominator=plain)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
            value=float(loss.detach())
            _,elapsed=ledger.call('train_backward',loss.backward,context);result['backward_seconds']+=elapsed
        row=D.diagnostic(sample,prediction.detach().cpu().numpy()[0],bins)
        sse=row['quality_weighted_velocity_sse'] if arm=='CLEAN_BULK_FM' else row['all_free_velocity_sse']
        if not np.isclose(sse/denominator,value,rtol=3e-6,atol=1e-7):raise ValueError('Detached weighted FM disagrees')
        row.update(**context,FM_contribution=value,batch_loss_denominator=denominator,
                   sample_free_coordinate_denominator=int(sample['free_coordinate_count']),
                   sample_weighted_free_coordinate_denominator=sample['weighted_free_coordinate_count'])
        diagnostics.append(row);result['FM']+=value
        del prediction,loss,b
    trainable=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in trainable):raise RuntimeError('Unused trainable parameter')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen block got gradient')
    # Read only the already-computed gradients: no additional backward/probe.
    result['preprocessing_gradient_norms']={}
    for prefix in ('role_embedding','context_encoder.','backbone.x_embedder.','backbone.t_embedder.','backbone.y_embedder.'):
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
        if not grads or any(g is None for g in grads):raise RuntimeError('Gradient path to trainable preprocessing interrupted')
        result['preprocessing_gradient_norms'][prefix]=float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))
    result['generator_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True))
    _,result['optimizer_seconds']=ledger.call('optimizer_update',optimizer.step,dict(local_step=step,global_step=D.BASE_STEP+step))
    if any(not bool(torch.isfinite(p).all()) for p in trainable):raise FloatingPointError('Nonfinite updated parameter')
    validate_optimizer(model,optimizer,step,hyper)
    for category in ('quality_clean_bulk','quality_dirty_bulk','quality_seam_third'):
        count=sum(r[category+'_scalar_count'] for r in diagnostics);sse=sum(r[category+'_velocity_sse'] for r in diagnostics)
        result.update({category+'_scalar_count':count,category+'_velocity_sse':sse,category+'_velocity_mse':sse/count if count else None})
    result['all_free_unweighted_velocity_mse']=sum(r['all_free_velocity_sse'] for r in diagnostics)/plain
    result.update(update_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='One formal batch forward/backward/optimizer; not full process VRAM')
    return result,diagnostics


def check_registration(root):
    D.require_cleanup(root)
    reg=D.read(root/'training_registration.json')
    if reg['status']!='FROZEN' or reg['arms']!=list(D.ARMS) or reg['updates_per_arm']!=1000 or reg['batch']!=8 or reg['microbatch']!=1:
        raise ValueError('Registered two arms each1000x8 required')
    if (reg['source_global_step'],reg['final_global_step'],reg['dirty_bulk_weight'],reg['owner_third_weight'])!=(13000,14000,.2,1):
        raise ValueError('Science definition changed')
    if reg['weighted_denominator']!='WHOLE_EFFECTIVE_BATCH_WEIGHTED_FREE_SCALARS' or reg['frozen_layer_indices']!=list(range(8)):
        raise ValueError('Loss normalization/freeze indices changed')
    for path,sha in reg['protected_hashes'].items():
        if D.digest(path)!=sha:raise ValueError('Protected input/code changed: '+path)
    go=D.read(root/'training_dispatch_go.json')
    if not go.get('authorized') or go['training_registration_sha256']!=D.digest(root/'training_registration.json'):
        raise ValueError('Root release after Wave1 missing')
    for path,sha in go['code_hashes'].items():
        if D.digest(path)!=sha:raise ValueError('Released code changed')
    manifest=D.read(D.resolve(reg['shared_data_manifest']))
    if go['shared_data_manifest_sha256']!=D.digest(D.resolve(reg['shared_data_manifest'])):
        raise ValueError('Released cached input stream changed')
    if manifest['status']!='COMPLETE' or manifest['samples']!=8000 or manifest['new_OT_calls']!=0:
        raise ValueError('Original8000 existing examples required')
    if [r['step'] for r in manifest['steps']]!=list(range(1,1001)) or any(len(r['samples'])!=8 for r in manifest['steps']):
        raise ValueError('Invalid fixed update stream')
    return reg,manifest,D.read(D.resolve(reg['source_labels']))['labels_by_parent']


def require_cursor(ledger,step):
    expected=dict(train_forward=step*8,train_backward=step*8,optimizer_update=step)
    if ledger.pending or ledger.failed:raise RuntimeError('Failed/pending calls forbid replay')
    if any(ledger.attempted[k]!=v or ledger.successful[k]!=v for k,v in expected.items()):
        raise RuntimeError('Checkpoint behind actual work; no implicit rollback')
    sequence=[r['context']['global_step'] for r in ledger.rows if r['event']=='CALL_BEGIN' and r['kind']=='optimizer_update']
    if sequence!=list(range(13001,13001+step)):raise RuntimeError('Update replay/gap')


def frozen_hash(state,arm):
    return state_hash({n:v for n,v in state.items() if frozen_name(n,arm)}) if arm=='SURGICAL_LATE4' else None


def cpu_contract_check():
    """Toy coordinate algebra and optimizer state only; zero neural networks/calls."""
    if torch.cuda.is_initialized():raise RuntimeError('CPU check must not own CUDA')
    prediction=torch.arange(18,dtype=torch.float32).reshape(1,2,9).requires_grad_()
    weights=torch.tensor([[[0.,0.,1.],[.2,.2,.2]]],dtype=torch.float32)
    target=torch.zeros_like(prediction)
    divisor=float(weights.double().sum()*3)+12.
    loss=weighted_loss(prediction,target,weights,divisor)
    loss.backward()
    expected=2*prediction.detach().reshape(1,2,3,3)*weights[...,None]/divisor
    if not torch.allclose(prediction.grad.reshape_as(expected),expected,atol=1e-6,rtol=1e-6):
        raise ValueError('Toy weighted gradient differs')
    if not torch.equal(prediction.grad.reshape(1,2,3,3)[weights==0],torch.zeros(2,3)):
        raise ValueError('Toy anchor gradient nonzero')
    # Deliberately uneven microbatch sizes distinguish global from per-case means.
    p2=torch.ones(1,2,9);w2=torch.tensor([[[1.,1.,1.],[1.,0.,0.]]])
    joint=loss.detach()+weighted_loss(p2,torch.zeros_like(p2),w2,divisor)
    numerator=(prediction.detach().reshape(1,2,3,3).square()*weights[...,None]).sum()+12
    if not torch.allclose(joint,numerator/divisor):raise ValueError('Effective-batch denominator check failed')
    class Fake:
        def __init__(self):
            names=['role_embedding']+['backbone.layers.'+str(i)+'.weight' for i in range(12)]+['backbone.final_layer.weight']
            self.items=[(n,torch.nn.Parameter(torch.ones(2))) for n in names]
            self._quality_arm='SURGICAL_LATE4'
        def named_parameters(self):return iter(self.items)
        def parameters(self):return (p for n,p in self.items)
    fake=Fake()
    old=torch.optim.AdamW(list(fake.parameters()),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    source=old.state_dict()
    source['state']={i:dict(step=torch.tensor(13000.),exp_avg=torch.full((2,),float(i)),exp_avg_sq=torch.full((2,),float(i+1)))
                     for i in source['param_groups'][0]['params']}
    payload=dict(optimizer=source,optimizer_parameter_names=parameter_names(fake,False),optimizer_state_sha256=base.tree_hash(source))
    for n,p in fake.named_parameters():p.requires_grad_(not frozen_name(n,fake._quality_arm))
    optimizer,receipt=restore_optimizer(fake,payload)
    validate_optimizer(fake,optimizer,0)
    if receipt['removed_frozen_parameter_states']!=8 or len(optimizer.state)!=6:
        raise ValueError('Toy optimizer pruning count differs')
    assert PRIVATE_SEAM_FORWARD.__code__ is model_adapter.seam_forward.__code__
    assert private_function(native.NativeModel.forward).__code__ is native.NativeModel.forward.__code__
    if torch.cuda.is_initialized():raise RuntimeError('CPU check initialized CUDA')
    return dict(status='PASS',neural_network_constructions=0,neural_forwards=0,neural_backwards=0,
        toy_coordinate_backward=1,optimizer_updates=0,CUDA_initialized=False,
        weighted_mask_gradient_exact=True,whole_batch_uneven_microbatch_denominator=True,
        toy_optimizer_states_retained=6,toy_frozen_states_removed=8,retained_moments_step_exact=True,
        original_forward_code_objects_identical=True)


def train(arm,root,attempt=None):
    root=D.resolve(root).resolve()
    if arm not in D.ARMS or not root.is_relative_to((ROOT/'experiments').resolve()):raise ValueError('Independent root/arm required')
    reg,manifest,labels=check_registration(root)
    regsha=D.digest(root/'training_registration.json');manifestsha=D.digest(D.resolve(reg['shared_data_manifest']))
    folder=root/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    statuspath=folder/'status.json';previous=D.read(statuspath) if statuspath.exists() else None
    if previous and previous['status']=='COMPLETE':
        for receipt in (previous['checkpoint'],previous['inference_export']):
            if D.digest(receipt['path'])!=receipt['sha256']:raise ValueError('Completed artifact changed')
        return previous
    if previous and previous['status'] not in ('INTERRUPTED_SAFE','SAFE_SEGMENT'):
        raise RuntimeError('No automatic retry of failed/ambiguous training')
    step=previous['local_step'] if previous else 0
    attempt=historical.safe_attempt_id(attempt or 'attempt_'+str(int(time.time())))
    adir=folder/'attempts'/attempt;adir.mkdir(parents=True,exist_ok=False)
    snapshot=resource();write(adir/'resources_dispatch.json',snapshot)
    policy=D.read(root/'dispatch_policy.json')
    minimum=float(policy['minimum_free_mib_before_load'])
    if not np.isfinite(minimum) or minimum<2048:raise ValueError('Invalid measured dispatch threshold')
    write(adir/'dispatch_policy.json',dict(policy=policy,sha256=D.digest(root/'dispatch_policy.json')))
    if snapshot['free_mib'] is None or snapshot['free_mib']<minimum:
        write(adir/'resource_wait.json',dict(status='RESOURCE_WAIT',resource=snapshot))
        raise RuntimeError('RESOURCE_WAIT: no model loaded')
    ledger=historical.CallLedger(folder/'calls.jsonl',dict(train_forward=8000,train_backward=8000,optimizer_update=1000),arm,attempt)
    require_cursor(ledger,step)
    state=dict(status='LOADING',arm=arm,local_step=step,global_step=13000+step,attempt=attempt,
               registration_sha256=regsha,data_manifest_sha256=manifestsha)
    started=time.perf_counter();stop={'requested':False};owned=False
    def interrupt(*args):stop['requested']=True
    handlers={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    observer=historical.GPUObserver('quality_'+arm,interval=2.)
    bins=np.asarray(reg['diagnostic_time_bin_edges'],np.float64)
    try:
        with historical.exclusive_writer(folder),observer:
            owned=True;write(statuspath,state)
            torch.set_num_threads(2);runtime.configure_stable_runtime()
            checkpoint=D.resolve(previous['checkpoint']['path'] if previous else reg['source_checkpoint'])
            expected=previous['checkpoint']['sha256'] if previous else reg['source_checkpoint_sha256']
            if D.digest(checkpoint)!=expected:raise ValueError('Source checkpoint bytes changed')
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            if payload['model_config']!=MODEL_CONFIG or payload.get('update_in_progress') is not False or payload['global_step']!=13000+step or payload.get('EMA') is not False:
                raise ValueError('Exact live source configuration/cursor required')
            for key,fn,hkey in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',base.tree_hash,'optimizer_state_sha256'),('rng',base.tree_hash,'rng_sha256')):
                if fn(payload[key])!=payload[hkey]:raise ValueError('Internal checkpoint hash failed: '+key)
            if previous:
                if payload.get('schema')!=SCHEMA or payload['arm']!=arm or payload['registration_sha256']!=regsha or payload['data_manifest_sha256']!=manifestsha:
                    raise ValueError('Resume experiment/input mismatch')
            else:
                if payload.get('schema')!='r2_seam_abc_training_v1' or payload.get('arm')!='B_POOL47' or payload['local_step']!=800 or payload['seam_total_step']!=1000:
                    raise ValueError('Only B_POOL47 final live source allowed')
                for pkey,rkey in (('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256'),('head_state_sha256','source_head_sha256'),('head_optimizer_state_sha256','source_head_optimizer_sha256')):
                    if payload[pkey]!=reg[rkey]:raise ValueError('Registered B state differs: '+pkey)
            model=native.build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
            model.load_state_dict(payload['generator_state'],strict=True)
            initial_frozen=payload['initial_frozen_state_sha256'] if previous else frozen_hash(payload['generator_state'],arm)
            if frozen_hash(payload['generator_state'],arm)!=initial_frozen:raise ValueError('Frozen source already changed')
            set_trainability(model,arm);bind_private_forward(model);model.to('cuda')
            optimizer,optreceipt=restore_optimizer(model,payload,resume=bool(previous))
            restore_rng(payload['rng'])
            if state_hash(model.state_dict())!=payload['generator_state_sha256'] or base.tree_hash(rng_state())!=payload['rng_sha256']:
                raise ValueError('Exact live model/RNG restore failed')
            hyper=historical.optimizer_hyperparameters(optimizer);validate_optimizer(model,optimizer,step,hyper)
            init=dict(generator_state_sha256=payload['generator_state_sha256'],source_optimizer_sha256=payload['optimizer_state_sha256'],rng_sha256=payload['rng_sha256'],
                optimizer_restore=optreceipt,global_step=payload['global_step'],head_loaded=False,head_forward_calls=0,EMA=False,
                trainable_parameter_names=parameter_names(model)[0],frozen_parameter_names=['generator.'+n for n,p in model.named_parameters() if not p.requires_grad],
                trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameter_count=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                initial_frozen_state_sha256=initial_frozen,private_seam_code_object_identical=PRIVATE_SEAM_FORWARD.__code__ is model_adapter.seam_forward.__code__,
                private_native_code_object_identical=model.forward.__func__.__code__ is native.NativeModel.forward.__code__,
                frozen_block_forward_in_autograd=True,scientific_probe_calls=0)
            write(adir/'initialization.json',init);del payload
            state.update(status='TRAINING',initial_frozen_state_sha256=initial_frozen);write(statuspath,state)
            def checkpoint_save(final=False):
                if state.get('checkpoint',{}).get('local_step')==step:
                    if D.digest(state['checkpoint']['path'])!=state['checkpoint']['sha256']:raise ValueError('Current checkpoint changed')
                    return
                require_cursor(ledger,step);validate_optimizer(model,optimizer,step,hyper)
                generator=_cpu_copy(model.state_dict());_check_generator(generator)
                actual_frozen=frozen_hash(generator,arm)
                if actual_frozen!=initial_frozen:raise ValueError('Frozen block values changed')
                opt,rng=_cpu_copy(optimizer.state_dict()),_cpu_copy(rng_state())
                data=dict(schema=SCHEMA,experiment=EXPERIMENT,arm=arm,local_step=step,seam_total_step=1000+step,global_step=13000+step,update_in_progress=False,
                    generator_state=generator,generator_state_sha256=state_hash(generator),optimizer=opt,optimizer_state_sha256=base.tree_hash(opt),
                    optimizer_parameter_names=parameter_names(model),all_parameter_names=parameter_names(model,False),
                    rng=rng,rng_sha256=base.tree_hash(rng),model_config=copy.deepcopy(MODEL_CONFIG),EMA=False,
                    head_state=None,head_state_sha256=None,head_optimizer=None,head_optimizer_state_sha256=None,
                    initial_frozen_state_sha256=initial_frozen,frozen_state_sha256=actual_frozen,
                    frozen_parameter_names=init['frozen_parameter_names'],trainable_parameter_names=init['trainable_parameter_names'],
                    registration_sha256=regsha,data_manifest_sha256=manifestsha,source_checkpoint_sha256=reg['source_checkpoint_sha256'],
                    stream_state=dict(batch_index=step,next_sample_index=step*8),execution=ledger.snapshot())
                destination=root/'checkpoints'/(f'{arm}_step{step}.pt' if final else f'{arm}_rolling.pt')
                state['checkpoint']=save_payload(destination,data,replace=not final)
                append_event(folder/'checkpoint_history.jsonl',state['checkpoint'])
                append_event(folder/'frozen_state_checks.jsonl',dict(local_step=step,initial=initial_frozen,actual=actual_frozen,status='PASS'))
                if final:
                    export={k:data[k] for k in ('experiment','arm','local_step','seam_total_step','global_step','generator_state','generator_state_sha256','model_config','EMA')}
                    export.update(schema='r2_seam_quality_generator_v1',training_checkpoint_sha256=state['checkpoint']['sha256'])
                    state['inference_export']=save_payload(root/'inference'/f'{arm}_generator.pt',export)
                write(statuspath,state)
            resume_step=step
            for current in range(step+1,1001):
                if current==resume_step+1 or current%10==1 or stop['requested']:
                    rr=resource();append_event(folder/'resource_checks.jsonl',rr)
                    if stop['requested'] or rr['free_mib'] is None or rr['free_mib']<2048:
                        checkpoint_save();state.update(status='INTERRUPTED_SAFE',reason='signal_or_resource_headroom');break
                tick=time.perf_counter()
                samples=[D.load_sample(r,root,arm,current,micro,labels) for micro,r in enumerate(manifest['steps'][current-1]['samples'])]
                input_seconds=time.perf_counter()-tick
                for micro,s in enumerate(samples):
                    append_event(folder/'exposure.jsonl',dict(local_step=current,global_step=13000+current,microbatch=micro,uid=s['uid'],
                        sample_index=s['sample_index'],alpha=s['alpha'],t=float(s['t']),N=s['N'],K=s['K'],path=s['path'],
                        input_sha256=manifest['steps'][current-1]['samples'][micro]['sha256'],
                        free_coordinate_count=int(s['free_coordinate_count']),weighted_free_coordinate_count=s['weighted_free_coordinate_count']))
                result,diagnostics=update(model,optimizer,samples,arm,current,ledger,hyper,bins)
                step=current
                append_event(folder/'training_log.jsonl',dict(local_step=step,global_step=13000+step,input_seconds=input_seconds,**result))
                for row in diagnostics:append_event(folder/'geometry_diagnostics.jsonl',row)
                state.update(local_step=step,global_step=13000+step,last_FM=result['FM'],peak_reserved_bytes=result['cuda_peak_reserved_bytes'],
                             peak_allocated_bytes=result['cuda_peak_allocated_bytes'],wall_seconds=time.perf_counter()-started)
                if step==1:write(folder/'first_formal_backward.json',dict(status='PASS',arm=arm,no_extra_forward=True,
                    preprocessing_gradient_norms=result['preprocessing_gradient_norms'],frozen_gradients_none=True,
                    generator_gradient_norm=result['generator_gradient_norm'],loss_denominator=result['loss_denominator']))
                if step%100==0:checkpoint_save(final=step==1000)
                write(statuspath,state)
                if step==1 or step%10==0:print(json.dumps(dict(arm=arm,step=step,FM=result['FM'],update_seconds=result['update_seconds'],
                    peak_reserved_mib=result['cuda_peak_reserved_bytes']/2**20,wall_seconds=state['wall_seconds'])),flush=True)
            else:state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(adir/'failure.json',state);raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state.update(local_step=step,global_step=13000+step,wall_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
        write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
        if owned:write(statuspath,state)
        write(adir/'final_status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=D.ARMS,required=True)
    parser.add_argument('--root',default='experiments/r2_seam_quality_wave1_wave2_v1')
    parser.add_argument('--attempt')
    args=parser.parse_args()
    print(json.dumps(train(args.arm,args.root,args.attempt),indent=2))


if __name__=='__main__':main()
