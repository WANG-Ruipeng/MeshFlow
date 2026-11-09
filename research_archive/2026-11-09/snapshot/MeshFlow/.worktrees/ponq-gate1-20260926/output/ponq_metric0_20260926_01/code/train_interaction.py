"""Matched 6000-update continuations from the same 3000-update direct FM state."""
import argparse,copy,datetime,hashlib,json,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from fm_model import total_velocity,validate_chart
from interaction_model import InteractionVelocity,MODULE_INIT_SEED,ARMS
from train_fm_pilot import BalancedSchedule,coupled_batch,diagnostic_summary,json_safe,sha,BATCH

FAULT_WALL_LIMIT_SECONDS=3600.


def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes()).hexdigest()


def optimizer_state_fingerprint(state):
    record={k:(dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_sha(v)) if torch.is_tensor(v) else v) for k,v in state.items()}
    return hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()


def capture_rng(schedule,generator):
    return dict(python=random.getstate(),numpy_global=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        batch_generator_cpu=generator.get_state(),balanced_schedule=schedule.state())


def restore_rng(checkpoint_rng):
    random.setstate(checkpoint_rng['python']);np.random.set_state(checkpoint_rng['numpy_global'])
    torch.set_rng_state(checkpoint_rng['torch_cpu'].cpu())
    saved_cuda=checkpoint_rng['torch_cuda']
    if torch.cuda.is_available():
        if len(saved_cuda)!=torch.cuda.device_count():raise ValueError('CUDA RNG device count differs from checkpoint')
        torch.cuda.set_rng_state_all([v.cpu() for v in saved_cuda])
    return dict(python_restored=True,numpy_global_restored=True,torch_cpu_restored=True,
        torch_cuda_restored=bool(torch.cuda.is_available()),saved_cuda_devices=len(saved_cuda),
        runtime_cuda_devices=torch.cuda.device_count() if torch.cuda.is_available() else 0)


def initialize_branch(checkpoint,arm,device='cpu'):
    """Return model, optimizer, schedule, batch_generator, JSON mapping audit.

    Old optimizer IDs are translated to explicit original parameter names using
    the archived model-state order (all entries are parameters in this model).
    Old state is copied unchanged; only new parameters get explicit zero states.
    Construction and state transfer happen before exact RNG restoration.
    """
    if checkpoint['status']!='COMPLETED' or checkpoint['completed_steps']!=3000:raise ValueError('Require completed common 3000-step checkpoint')
    if checkpoint['protocol']['arm']!='direct':raise ValueError('Parent must be direct FM')
    model=InteractionVelocity(arm=arm,width=128).to(device)
    old_names=list(checkpoint['model_state_dict'])
    current=dict(model.named_parameters())
    original_names=[name for name in current if not name.startswith('interaction.')]
    if old_names!=original_names:raise ValueError('Original parameter name/order differs from archived model')
    loaded=model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    new_names=[name for name in current if name not in old_names]
    if set(loaded.missing_keys)!=set(new_names) or loaded.unexpected_keys:raise ValueError('Unexpected model-state mapping')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    old_optim=checkpoint['optimizer_state_dict']
    if len(old_optim['param_groups'])!=1:raise ValueError('Expected one original AdamW parameter group')
    old_group=old_optim['param_groups'][0];old_ids=old_group['params']
    if len(old_ids)!=len(old_names):raise ValueError('Old optimizer parameter count mismatch')
    if old_group['lr']!=3e-4 or old_group['weight_decay']!=1e-4 or tuple(old_group['betas'])!=(.9,.999) or old_group['eps']!=1e-8:
        raise ValueError('Parent AdamW settings differ from frozen protocol')
    source_by_name=dict(zip(old_names,old_ids));packed=optimizer.state_dict();new_ids=packed['param_groups'][0]['params']
    all_names=list(current);name_to_id=dict(zip(all_names,new_ids))
    packed['param_groups']=[{**copy.deepcopy(old_group),'params':new_ids}]
    packed['state']={}
    for name,param in current.items():
        identifier=name_to_id[name]
        if name in source_by_name:
            old_state=old_optim['state'][source_by_name[name]]
            if set(old_state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected original AdamW state fields')
            packed['state'][identifier]=copy.deepcopy(old_state)
        else:
            packed['state'][identifier]=dict(step=torch.tensor(0.,dtype=torch.float32),
                exp_avg=torch.zeros_like(param,device='cpu'),exp_avg_sq=torch.zeros_like(param,device='cpu'))
    optimizer.load_state_dict(packed)
    audit_rows=[]
    for name,param in current.items():
        state=optimizer.state[param];old=name in source_by_name;fingerprint=optimizer_state_fingerprint(state)
        expected=optimizer_state_fingerprint(old_optim['state'][source_by_name[name]]) if old else None
        zero=all(torch.count_nonzero(state[k]).item()==0 for k in ['step','exp_avg','exp_avg_sq']) if not old else None
        weights_equal=torch.equal(param.detach().cpu(),checkpoint['model_state_dict'][name].detach().cpu()) if old else None
        if (old and (fingerprint!=expected or not weights_equal)) or (not old and not zero):raise AssertionError('Optimizer/model restoration failed: '+name)
        audit_rows.append(dict(name=name,kind='restored_old' if old else 'new_zero_state',optimizer_id=name_to_id[name],
            source_optimizer_id=source_by_name.get(name),optimizer_state_sha256=fingerprint,source_optimizer_state_sha256=expected,
            old_state_exact=fingerprint==expected if old else None,old_weight_exact=weights_equal,new_state_zero=zero,
            step=float(state['step'].detach().cpu()),parameter_shape=list(param.shape)))
    saved_schedule=checkpoint['rng']['balanced_schedule']
    schedule=BalancedSchedule(saved_schedule['count'],0)
    schedule.order=copy.deepcopy(saved_schedule['order']);schedule.cursor=saved_schedule['cursor'];schedule.cycles=saved_schedule['cycles']
    schedule.rng.bit_generator.state=copy.deepcopy(saved_schedule['numpy_generator_state'])
    generator=torch.Generator(device='cpu');generator.set_state(checkpoint['rng']['batch_generator_cpu'].cpu())
    rng_audit=restore_rng(checkpoint['rng'])
    audit=dict(passed=True,arm=arm,base_completed_steps=3000,module_initialization_seed=MODULE_INIT_SEED,
        parameter_counts=model.parameter_counts(),old_parameter_names=old_names,new_parameter_names=new_names,
        optimizer_param_names=all_names,parameters=audit_rows,rng=rng_audit,
        batch_generator_restored=torch.equal(generator.get_state(),checkpoint['rng']['batch_generator_cpu'].cpu()),
        balanced_schedule_restored=schedule.state()==saved_schedule,
        optimizer_policy='Old parameter states mapped by name and copied exactly; newly introduced step/exp_avg/exp_avg_sq explicitly zero')
    assert audit['batch_generator_restored'] and audit['balanced_schedule_restored']
    return model,optimizer,schedule,generator,audit


def main(args):
    started=time.perf_counter();device=torch.device(args.device)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data_path=Path(args.data).resolve();checkpoint_path=Path(args.checkpoint).resolve()
    parent=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    with np.load(data_path,allow_pickle=False) as z:
        states=np.array(z['states'],dtype=np.float64,copy=True);mean_np=np.array(z['mean'],dtype=np.float64,copy=True);scale_np=np.array(z['scale'],dtype=np.float64,copy=True)
    if states.shape!=(1,256,22) or not np.isfinite(states).all():raise ValueError('Frozen task requires one finite (256,22) target')
    validate_chart(torch.from_numpy(mean_np),torch.from_numpy(scale_np))
    if sha(data_path)!=parent['protocol']['data_sha256']:raise ValueError('Data bytes differ from the common checkpoint training data')
    if not torch.equal(torch.tensor(mean_np,dtype=torch.float32),parent['mean'].cpu()) or not torch.equal(torch.tensor(scale_np,dtype=torch.float32),parent['scale'].cpu()):
        raise ValueError('Chart differs from common checkpoint')
    model,optimizer,schedule,generator,mapping=initialize_branch(parent,args.arm,device)
    if schedule.count!=len(states):raise ValueError('Restored schedule target count mismatch')
    mean=torch.tensor(mean_np,dtype=torch.float32,device=device);scale=torch.tensor(scale_np,dtype=torch.float32,device=device)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),Path(__file__).with_name('interaction_model.py'),Path(__file__).with_name('fm_model.py'),Path(__file__).with_name('train_fm_pilot.py')]
    protocol=dict(version='SINGLE_STATE_INTERACTION0_TRAIN',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arm=args.arm,parent_checkpoint=str(checkpoint_path),parent_checkpoint_sha256=sha(checkpoint_path),base_updates=3000,
        requested_new_updates=args.steps,requested_total_updates=3000+args.steps,batch_size=BATCH,
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,gradient_clip_norm=1.,scheduler=None),
        optimizer_restoration='All old states copied by parameter-name map; all new moments and steps initialized exactly zero',
        data_path=str(data_path),data_sha256=sha(data_path),states_shape=list(states.shape),mean=mean_np.tolist(),scale=scale_np.tolist(),
        probability_path='Unchanged coupled_batch: physical-xyz Hungarian, whole22 target permutation, linear chart path',
        loss='Unchanged mean((direct_total_velocity-(target-noise))**2) over all22 channels; field weights3/22,3/22,16/22',
        transport='Not added. Original total_velocity(...,direct) computes unchanged L diagnostics only.',
        module=dict(seed=MODULE_INIT_SEED,placement='after encoder before mean/max pooling',B0='identity',
            B1='LayerNorm128,Linear128to256,SiLU,Linear256to128; final projection weight/bias zero',
            B2='LayerNorm128,MHA128 with4heads,dropout0,no FFN; output projection weight/bias zero'),
        hidden_for_probes='actual decoder input point tensor after residual block, before pooling',
        parameter_counts=mapping['parameter_counts'],fault_wall_limit_seconds=FAULT_WALL_LIMIT_SECONDS,
        budget_policy='Updates, not loss or the old600s limit. One-hour fault boundary checked before updates; no performance-based extension.',
        selection='Final checkpoint only, no early stopping or best-checkpoint selection',
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in source_files],device=str(device),torch_version=torch.__version__)
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    (out/'OPTIMIZER_MAPPING.json').write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
    exposure=np.array(parent['exposure_counts'],dtype=np.int64);new_exposure=np.zeros_like(exposure)
    completed=0;attempted=0;last_metrics=None;error=None;status='INITIALIZED'
    stream_digest=bytes.fromhex(parent['attempted_stream_sha256']);new_stream=bytes(32)
    def checkpoint():
        return dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),optimizer_param_names=list(dict(model.named_parameters())),
            rng=capture_rng(schedule,generator),arm=args.arm,protocol=protocol,protocol_sha256=sha(out/'PROTOCOL.json'),
            mean=mean.detach().cpu(),scale=scale.detach().cpu(),completed_steps=3000+completed,new_updates=completed,
            attempted_batches=parent['attempted_batches']+attempted,new_attempted_batches=attempted,
            exposure_counts=exposure.tolist(),new_exposure_counts=new_exposure.tolist(),status=status,error=error,last_metrics=last_metrics,
            attempted_stream_sha256=stream_digest.hex(),new_attempted_stream_sha256=new_stream.hex(),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'])
    torch.save(model.state_dict(),out/'initial_model.pt');torch.save(checkpoint(),out/'initial.pt')
    status='RUNNING';training_started=time.perf_counter()
    try:
        with (out/'TRAIN_TRACE.jsonl').open('x',buffering=1) as log:
            while completed<args.steps:
                if time.perf_counter()-started>=FAULT_WALL_LIMIT_SECONDS:status='FAULT_WALL_LIMIT_EXHAUSTED';break
                attempted+=1;step_start=time.perf_counter()
                noise,times,target,ids=coupled_batch(states,mean_np,scale_np,schedule,generator)
                batch_hash=hashlib.sha256(noise.numpy().tobytes()+times.numpy().tobytes()+target.numpy().tobytes()+np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
                stream_digest=hashlib.sha256(stream_digest+bytes.fromhex(batch_hash)).digest();new_stream=hashlib.sha256(new_stream+bytes.fromhex(batch_hash)).digest()
                noise=noise.to(device);times=times.to(device);target=target.to(device)
                z=(1-times)*noise+times*target;truth=target-noise
                optimizer.zero_grad(set_to_none=True);predicted,diag=total_velocity(model,z,times,mean,scale,'direct')
                squared=(predicted-truth).square();loss=squared.mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite direct total-velocity loss')
                loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if device.type=='cuda':torch.cuda.synchronize(device)
                if time.perf_counter()-started>=FAULT_WALL_LIMIT_SECONDS:status='FAULT_WALL_LIMIT_EXHAUSTED';break
                optimizer.step();completed+=1
                for i in ids:exposure[i]+=1;new_exposure[i]+=1
                if device.type=='cuda':torch.cuda.synchronize(device)
                last_metrics=dict(step=3000+completed,new_update=completed,attempted_new_batches=attempted,batch_sha256=batch_hash,
                    elapsed_seconds=time.perf_counter()-started,step_seconds=time.perf_counter()-step_start,
                    loss=float(loss.detach()),x_mse=float(squared[...,:3].detach().mean()),n_mse=float(squared[...,3:6].detach().mean()),
                    F_mse=float(squared[...,6:].detach().mean()),gradient_norm_before_clip=float(gradient_norm),**diagnostic_summary(diag))
                log.write(json.dumps(json_safe(last_metrics),allow_nan=False)+'\n')
                if completed==1 or completed%100==0:print(json.dumps(json_safe(last_metrics),allow_nan=False),flush=True)
            if completed==args.steps:status='COMPLETED'
    except Exception as exc:status='FAILED';error=dict(exception=repr(exc),traceback=traceback.format_exc())
    training_elapsed=time.perf_counter()-training_started;torch.save(checkpoint(),out/'final.pt')
    result=dict(status=status,arm=args.arm,completed_steps=3000+completed,new_updates=completed,requested_new_updates=args.steps,
        requested_steps=3000+args.steps,finished_requested_budget=status=='COMPLETED' and completed==args.steps,
        training_elapsed_seconds=training_elapsed,total_wall_seconds_including_initialization_and_save=time.perf_counter()-started,
        fault_wall_limit_seconds=FAULT_WALL_LIMIT_SECONDS,exposure_counts=exposure.tolist(),new_exposure_counts=new_exposure.tolist(),
        attempted_batches=parent['attempted_batches']+attempted,new_attempted_batches=attempted,
        attempted_stream_sha256=stream_digest.hex(),new_attempted_stream_sha256=new_stream.hex(),last_metrics=last_metrics,error=error,
        parameter_counts=mapping['parameter_counts'],final_checkpoint_sha256=sha(out/'final.pt'),initial_checkpoint_sha256=sha(out/'initial.pt'),
        initial_model_sha256=sha(out/'initial_model.pt'),optimizer_mapping_sha256=sha(out/'OPTIMIZER_MAPPING.json'),protocol_sha256=sha(out/'PROTOCOL.json'))
    (out/'RESULT.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(json_safe(result),indent=2,allow_nan=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','checkpoint','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--arm',required=True,choices=ARMS);p.add_argument('--steps',type=int,default=6000);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    result=main(args)
    raise SystemExit(0 if result['finished_requested_budget'] else 1)