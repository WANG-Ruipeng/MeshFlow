"""C0/C1/C2 controlled continuation of the fixed DomainVelocity architecture.

Only the train-only NPZ is opened. C1/C2 differentiate all 32 guarded rollout
steps; C2 mainfield queries are loss construction, never network conditions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.networks import DomainVelocity, point_features
from ..layout_student.train import (BalancedSampler, configure_determinism,
    discrete_bridge, file_sha256, model_sha256, rng_state, restore_rng)
from ..layout_value.optimizer import TorchRadialField, torch_face_quality
from .rollout import rollout


def require(ok,message):
    if not ok: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as out:
        json.dump(value,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n')


def resolve_under(run,path):
    p=Path(path)
    p=p.resolve() if p.is_absolute() else (run/p).resolve()
    require(p.is_relative_to(run),'Path escapes this run: '+str(p))
    return p


def load_train_archive(run,protocol):
    path=resolve_under(run,protocol['paths'].get('training_labels','train_labels.npz'))
    digest=file_sha256(path)
    require(digest==protocol['hashes']['training_labels_sha256'],'Training archive hash mismatch')
    manifest_path=resolve_under(run,protocol['paths'].get('data_manifest','DATA_MANIFEST.json'))
    require(file_sha256(manifest_path)==protocol['hashes']['data_manifest_sha256'],'Data manifest hash mismatch')
    manifest=read_json(manifest_path)['files']
    require(manifest.get(path.relative_to(run).as_posix())==digest,'Training archive not bound by data manifest')
    with np.load(path,allow_pickle=False) as z:
        data={key:np.array(z[key]) for key in ('directions0','directions1','proxy_points','point_features','logr','faces','quality_floors_deg')}
        uids=[str(x) for x in z['train_uids'].tolist()]
    require(uids==protocol['train_uids'] and len(uids)==len(set(uids)),'Training UID order/uniqueness mismatch')
    require(not set(uids)&set(protocol['internal_check_uids']),'Check teacher label leakage')
    n=len(uids)
    shapes={'directions0':(n,258,3),'directions1':(n,258,3),'proxy_points':(n,256,3),
        'point_features':(n,256,4),'logr':(n,2050),'faces':(512,3),'quality_floors_deg':(n,2)}
    for key,shape in shapes.items():
        require(data[key].shape==shape and np.isfinite(data[key]).all(),'Invalid training array: '+key)
    require(data['faces'].dtype==np.int32,'Original connectivity must remain int32')
    require(np.array_equal(point_features(data['proxy_points']),data['point_features']),'Original proxy input features changed')
    for key in ('directions0','directions1'):
        require(np.allclose(np.linalg.norm(data[key],axis=-1),1,rtol=0,atol=1e-12),'Nonunit '+key)
    scale=float(np.mean((data['directions1']-data['directions0'])**2))
    require(scale>0 and np.isfinite(scale),'Endpoint scale must be positive and computed only from training teachers')
    require(scale==protocol['endpoint_scale_squared'],'Endpoint scale differs from full training-only mean')
    require(protocol.get('loss',{}).get('endpoint_scale_squared',scale)==scale,'Conflicting endpoint scale')
    require(protocol.get('loss',{}).get('quality_scale_deg',.1)==.1,'Quality loss scale changed')
    return uids,data,scale,manifest


def quality_penalty(final_u,indices,fields,faces_t,floors,scale_deg=.1):
    """Per-parent min-angle and per-triangle-minimum-angle p05 deficits in degrees."""
    require(scale_deg==.1,'Quality scale is fixed before training')
    actual=torch.stack([torch.stack(torch_face_quality(fields[idx].vertices(final_u[b]),faces_t))
                        for b,idx in enumerate(indices)])
    selected=floors[torch.tensor(indices,dtype=torch.long,device=floors.device)]
    deficits=torch.relu(selected-actual)
    penalty=(deficits/scale_deg).square().sum(dim=-1).mean()
    return penalty,actual,deficits


def compute_loss(model,arm,u0,u1,features,k,indices,faces,endpoint_scale_squared,
                 fields=None,faces_t=None,floors=None):
    if arm=='c0':
        positions,target,_,t=discrete_bridge(u0,u1,k,32)
        prediction=model(positions.float(),t.float(),features)
        value=(prediction-target.float()).square().mean()
        return value,{'velocity_mse':float(value.detach().cpu()),'endpoint_loss':None,'quality_penalty':None},None
    require(arm in ('c1','c2'),'Unknown control arm')
    result=rollout(model,u0,features,faces,steps=32,return_path=False)
    endpoint_mse=(result['u']-u1).square().mean()
    endpoint=endpoint_mse/endpoint_scale_squared
    penalty=endpoint.new_zeros(())
    metrics={'velocity_mse':None,'endpoint_loss':float(endpoint.detach().cpu()),
        'endpoint_direction_mse':float(endpoint_mse.detach().cpu()),'quality_penalty':0.,
        'min_alpha':float(result['alphas'].min()),'shrunk_steps':int((result['alphas']<1).sum()),
        'rollout_steps':32,'rollout_parent_evaluations':len(indices)*32}
    if arm=='c2':
        penalty,actual,deficits=quality_penalty(result['u'],indices,fields,faces_t,floors)
        metrics.update({'quality_penalty':float(penalty.detach().cpu()),
            'mean_min_angle_deg':float(actual[:,0].mean().detach().cpu()),
            'mean_p05_angle_deg':float(actual[:,1].mean().detach().cpu()),
            'max_min_angle_deficit_deg':float(deficits[:,0].max().detach().cpu()),
            'max_p05_angle_deficit_deg':float(deficits[:,1].max().detach().cpu())})
    return endpoint+penalty,metrics,result['alphas']


def train_run(run:Path,arm:str,seed:int,total_updates:int=1000,resume:Path|None=None,
              *,device='cuda',checkpoint_every=250,stop_after=None,output_dir=None,
              diagnostic_uid=None):
    run=Path(run).resolve();protocol=read_json(run/'RUN_PROTOCOL.json');training=protocol['training']
    config_hash=hashlib.sha256(json.dumps({'training':training,'loss':protocol['loss']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(config_hash==protocol['hashes']['config_sha256'],'Training/loss configuration hash mismatch')
    require(protocol['loss']['weights']=={'endpoint':1.,'min_angle':1.,'p05_angle':1.},'Loss weights differ from fixed protocol')
    require(training.get('rollout_steps',32)==32,'Rollout length changed')
    require(arm in training['arms'] and seed in training['seeds'],'Arm/seed differs from frozen protocol')
    for key,value in [('batch_size',4),('learning_rate',1e-4),('weight_decay',1e-4),('optimizer','AdamW')]:
        require(training[key]==value,'Frozen configuration mismatch: '+key)
    if diagnostic_uid is None:
        require(total_updates==training['updates'],'Update budget differs from frozen protocol')
    else:
        d=protocol['diagnostic']
        require(d['uid']==diagnostic_uid and d['arm']==arm and d['seed']==seed and d['updates']==total_updates,
                'Single-object diagnostic differs from frozen protocol')
    end=total_updates if stop_after is None else int(stop_after)
    require(0<end<=total_updates and checkpoint_every>0,'Invalid update/checkpoint budget')
    dev=torch.device(device);require(dev.type in ('cpu','cuda'),'Unsupported training device')
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    configure_determinism(seed)
    uids,data,scale,manifest=load_train_archive(run,protocol)
    active=list(range(len(uids)))
    if diagnostic_uid is not None:
        moved=[u for i,u in enumerate(uids) if not np.array_equal(data['directions0'][i],data['directions1'][i])]
        require(diagnostic_uid==min(moved),'Diagnostic is not the predeclared smallest moved training UID')
        active=[uids.index(diagnostic_uid)]
    initial=resolve_under(run,protocol['paths']['initial_checkpoints'][str(seed)])
    initial_sha=file_sha256(initial)
    require(manifest.get(initial.relative_to(run).as_posix())==initial_sha,'Initial checkpoint not bound by data manifest')
    require(protocol['hashes']['initial_checkpoint_sha256'][str(seed)]==initial_sha,'Frozen initial checkpoint hash mismatch')
    original=torch.load(initial,map_location='cpu',weights_only=False)
    require('model' in original and model_sha256(original['model'])==original['model_sha256'],'Invalid v1 initial checkpoint')
    require(original['model_sha256']==protocol['hashes']['initial_model_sha256'][str(seed)],'Frozen initial model hash mismatch')
    model=DomainVelocity();model.load_state_dict(original['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==29907,'Architecture changed')
    model=model.to(device=dev,dtype=torch.float32).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    require(len(optimizer.state)==0,'Old optimizer state was inherited')
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=BalancedSampler(len(active),seed)
    u0=torch.as_tensor(data['directions0'],dtype=torch.float64,device=dev)
    u1=torch.as_tensor(data['directions1'],dtype=torch.float64,device=dev)
    features=torch.as_tensor(data['point_features'],dtype=torch.float32,device=dev)
    floors=torch.as_tensor(data['quality_floors_deg'],dtype=torch.float64,device=dev)
    faces=data['faces'];faces_t=torch.as_tensor(faces.astype(np.int64),device=dev)
    fields=None
    if arm=='c2':
        support=RadialSupport()
        fields=[TorchRadialField(y,device=dev,support=support) for y in data['logr']]
    code_root=Path(__file__).resolve().parents[1]
    paths=['layout_endpoint/train.py','layout_endpoint/rollout.py','layout_student/train.py',
           'layout_value/optimizer.py','legacy/networks.py','legacy/chart.py']
    config={'arm':arm,'seed':seed,'total_updates':total_updates,'batch_size':4,
        'learning_rate':1e-4,'weight_decay':1e-4,'optimizer':'AdamW','scheduler':'constant',
        'optimizer_initialization':'fresh_empty_no_v1_Adam_inheritance','parameter_count':29907,
        'device':str(dev),'network_dtype':'float32','geometry_dtype':'float64','tf32':False,
        'deterministic_algorithms':True,'rollout_steps':32,'guard_choice_gradient':'detached_original_discrete_alpha',
        'endpoint_scale_squared':scale,'quality_scale_deg':.1,'endpoint_weight':1.,'min_angle_weight':1.,'p05_weight':1.,
        'quality_validation_tolerance_deg':1e-6,'mainfield_network_input':False,
        'train_uids':uids,'active_training_uids':[uids[i] for i in active],
        'diagnostic_uid':diagnostic_uid,'checkpoint_selection':'predeclared_final_update_only',
        'protocol_sha256':file_sha256(run/'RUN_PROTOCOL.json'),
        'training_labels_sha256':file_sha256(run/protocol['paths'].get('training_labels','train_labels.npz')),
        'data_manifest_sha256':protocol['hashes']['data_manifest_sha256'],
        'initial_checkpoint_sha256':initial_sha,'initial_model_sha256':original['model_sha256'],
        'source_sha256':{p:file_sha256(code_root/p) for p in paths}}
    first_update=0;trace=[];timing_trace=[];accumulated=0.
    if resume is not None:
        saved=torch.load(Path(resume),map_location='cpu',weights_only=False)
        require(saved['config']==config,'Resume config/protocol/data/source mismatch')
        require(model_sha256(saved['model'])==saved['model_sha256'],'Resume model hash mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);sampler.load_state_dict(saved['sampler'])
        trace=list(saved['trace']);timing_trace=list(saved['timing_trace'])
        first_update=int(saved['update']);accumulated=float(saved['elapsed_training_seconds'])
        restore_rng(saved['rng'])
    require(first_update<end,'No new updates requested')
    if output_dir is None:
        output_dir=run/'diagnostics'/'single_object_seed_10' if diagnostic_uid else run/'training'/arm/f'seed_{seed}'
    out=resolve_under(run,output_dir);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'TRAIN_CONFIG.json',config)
    write_json(out/'START.json',{'arm':arm,'seed':seed,'start_update':first_update,'end_update':end,
        'resume':str(resume) if resume else None,'resume_sha256':file_sha256(resume) if resume else None,
        'optimizer_created_empty_before_optional_resume':True,'initial_checkpoint':str(initial),
        'diagnostic_only':diagnostic_uid is not None})
    if dev.type=='cuda':torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev)
    started=time.perf_counter();block_start=started;block_first=first_update+1
    cuda_start=None
    if dev.type=='cuda':cuda_start=torch.cuda.Event(enable_timing=True);cuda_start.record()
    last_checkpoint=None
    def checkpoint(update,final):
        if dev.type=='cuda':torch.cuda.synchronize(dev)
        state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
            'sampler':sampler.state_dict(),'trace':trace,'timing_trace':timing_trace,
            'config':config,'update':update,'is_predeclared_final':final,'diagnostic_only':diagnostic_uid is not None,
            'elapsed_training_seconds':accumulated+time.perf_counter()-started,
            'protocol_sha256':config['protocol_sha256'],'training_labels_sha256':config['training_labels_sha256'],
            'initial_checkpoint_sha256':initial_sha}
        state['model_sha256']=model_sha256(state['model'])
        path=out/('final.pt' if final else f'step_{update:05d}.pt')
        with path.open('xb') as stream:torch.save(state,stream)
        return path
    try:
        for update in range(first_update+1,end+1):
            indices=[active[i] for i in sampler.take(4)]
            index=torch.tensor(indices,dtype=torch.long,device=dev)
            # Draw equal random times in every arm to keep RNG progression matched;
            # C1/C2 have a deterministic 32-step timeline and do not use these k.
            k=torch.randint(0,32,(4,),device=dev)
            loss,metrics,alphas=compute_loss(model,arm,u0[index],u1[index],features[index],k,indices,
                faces,scale,fields,faces_t,floors)
            if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at {update}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            if any(p.grad is None for p in model.parameters()) or not bool(finite):
                raise FloatingPointError(f'Invalid gradient at {update}')
            grad_l2=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())).detach().cpu())
            optimizer.step();scheduler.step()
            trace.append({'update':update,'loss':float(loss.detach().cpu()),**metrics,
                'uids':[uids[i] for i in indices],'rng_matched_k':k.detach().cpu().tolist(),
                'lr':optimizer.param_groups[0]['lr'],'gradient_l2':grad_l2,
                'alphas':alphas.tolist() if alphas is not None else None})
            if update%100==0 or update==end:
                gpu_ms=None
                if dev.type=='cuda':
                    cuda_end=torch.cuda.Event(enable_timing=True);cuda_end.record();cuda_end.synchronize()
                    gpu_ms=float(cuda_start.elapsed_time(cuda_end))
                elapsed_block=time.perf_counter()-block_start
                timing={'start_update':block_first,'end_update':update,'updates':update-block_first+1,
                    'wall_seconds':elapsed_block,'cuda_event_elapsed_ms_including_host_gaps':gpu_ms,
                    'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else None,
                    'peak_gpu_reserved_bytes':int(torch.cuda.max_memory_reserved(dev)) if dev.type=='cuda' else None,
                    'cpu_guard_and_transfers_included':True}
                timing_trace.append(timing)
                print(json.dumps({'arm':arm,'seed':seed,'update':update,'loss':trace[-1]['loss'],
                    'endpoint_loss':metrics.get('endpoint_loss'),'quality_penalty':metrics.get('quality_penalty'),
                    **timing},ensure_ascii=False),flush=True)
                block_start=time.perf_counter();block_first=update+1
                if dev.type=='cuda':cuda_start=torch.cuda.Event(enable_timing=True);cuda_start.record()
            if update%checkpoint_every==0 or update==end:
                last_checkpoint=checkpoint(update,update==total_updates)
    except Exception as exc:
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),
            'completed_update':trace[-1]['update'] if trace else first_update})
        raise
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    exposures=Counter(uid for row in trace for uid in row['uids'])
    report={'status':'COMPLETE' if end==total_updates else 'INTERRUPTED_FOR_RESUME_TEST',
        'arm':arm,'seed':seed,'update':end,'final_budget':total_updates,'diagnostic_only':diagnostic_uid is not None,
        'checkpoint':str(last_checkpoint),'checkpoint_sha256':file_sha256(last_checkpoint),
        'model_sha256':model_sha256(model.state_dict()),'initial_checkpoint_sha256':initial_sha,
        'protocol_sha256':config['protocol_sha256'],'training_labels_sha256':config['training_labels_sha256'],
        'parameter_count':29907,'elapsed_training_seconds':accumulated+time.perf_counter()-started,
        'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else None,
        'peak_gpu_reserved_bytes':int(torch.cuda.max_memory_reserved(dev)) if dev.type=='cuda' else None,
        'parent_exposures':{uid:exposures[uid] for uid in uids},'first_loss':trace[0]['loss'],
        'last_loss':trace[-1]['loss'],'last_100_loss_mean':float(np.mean([x['loss'] for x in trace[-100:]])),
        'first_100_updates_timing':next((t for t in timing_trace if t['start_update']==1 and t['end_update']==100),None),
        'selection_uses_internal_check':False,'internal_check_teacher_files_opened':0,
        'checkpoint_selection':'predeclared_final_update_only','training_only_archive_uids_verified':True}
    write_json(out/'TRAIN_TRACE.json',trace);write_json(out/'TRAIN_TIMING.json',timing_trace)
    write_json(out/'TRAIN_REPORT.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--arm',required=True,choices=['c0','c1','c2']);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--updates',default=1000,type=int);p.add_argument('--device',default='cuda')
    p.add_argument('--resume',type=Path);p.add_argument('--checkpoint-every',default=250,type=int)
    p.add_argument('--stop-after',type=int);p.add_argument('--output-dir',type=Path);p.add_argument('--diagnostic-uid')
    a=p.parse_args();print(json.dumps(train_run(a.run,a.arm,a.seed,a.updates,a.resume,device=a.device,
        checkpoint_every=a.checkpoint_every,stop_after=a.stop_after,output_dir=a.output_dir,
        diagnostic_uid=a.diagnostic_uid),indent=2),flush=True)


if __name__=='__main__':main()
