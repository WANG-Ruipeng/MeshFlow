"""Fixed-budget C0 fitting diagnostics; all frozen endpoint dependencies stay intact."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ..legacy.networks import DomainVelocity
from ..layout_student.train import (BalancedSampler,configure_determinism,file_sha256,
    model_sha256,rng_state,restore_rng)
from ..layout_endpoint.train import (compute_loss,load_train_archive,read_json,
    require,resolve_under,write_json)


def config_digest(protocol):
    payload={k:protocol[k] for k in ('training','loss','jobs')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bound_file(run,relative,manifest,expected=None):
    path=resolve_under(run,relative);digest=file_sha256(path)
    require(manifest.get(path.relative_to(run).as_posix())==digest,'Unbound or changed input: '+str(relative))
    if expected is not None:require(digest==expected,'Input hash mismatch: '+str(relative))
    return path,digest


def validate_parent(source,job,protocol,run,manifest):
    """Validate the complete imported C0 state, not just its model tensors."""
    c=source['config'];seed=job['seed']
    require(source['update']==job['start_update']==c['total_updates'],'Parent update mismatch')
    require(source['is_predeclared_final'] and not source['diagnostic_only'],'Parent must be endpoint C0 final')
    require(c['arm']=='c0' and c['seed']==seed,'Parent arm/seed mismatch')
    require(c['active_training_uids']==protocol['train_uids']==c['train_uids'],'Parent training split mismatch')
    for key,value in [('batch_size',4),('learning_rate',1e-4),('weight_decay',1e-4),
                      ('optimizer','AdamW'),('scheduler','constant'),('parameter_count',29907)]:
        require(c[key]==value,'Parent optimizer/architecture setting changed: '+key)
    require(source['training_labels_sha256']==protocol['hashes']['training_labels_sha256'],'Parent data changed')
    parent_protocol=protocol['paths'].get('source_protocol','initial/endpoint_RUN_PROTOCOL.json')
    parent_path,parent_sha=bound_file(run,parent_protocol,manifest)
    require(parent_sha==source['protocol_sha256']==c['protocol_sha256'],'Parent protocol file changed')
    if 'source_protocol_sha256' in job:require(parent_sha==job['source_protocol_sha256'],'Parent protocol job binding mismatch')
    p=read_json(parent_path)
    require(p['train_uids']==protocol['train_uids'] and p['internal_check_uids']==protocol['internal_check_uids'],
            'Parent split changed')
    require(p['training']['updates']==source['update'],'Parent protocol update mismatch')
    require(source['initial_checkpoint_sha256']==protocol['hashes']['initial_checkpoint_sha256'][str(seed)],
            'Parent v1 initialization changed')
    require(source['model_sha256']==model_sha256(source['model']),'Parent model hash mismatch')
    require(len(source['trace'])==source['update'],'Parent trace is incomplete')
    require([x['update'] for x in source['trace']]==list(range(1,source['update']+1)),
            'Parent trace update sequence changed')
    require(source['sampler']['count']==len(protocol['train_uids']),'Parent sampler count mismatch')
    counts=Counter(u for row in source['trace'] for u in row['uids'])
    require(source['sampler']['exposures']==[counts[u] for u in protocol['train_uids']],
            'Parent sampler exposure/trace mismatch')
    require(all(len(x['uids'])==4 for x in source['trace']),'Parent microbatch changed')
    root=Path(__file__).resolve().parents[1]
    for rel,digest in c['source_sha256'].items():
        require(file_sha256(root/rel)==digest,'Frozen endpoint dependency changed: '+rel)
    require(source['optimizer']['state'],'Parent optimizer is empty')
    for group in source['optimizer']['param_groups']:
        require(group['lr']==1e-4 and group['weight_decay']==1e-4,'Parent Adam settings changed')
        require(group['foreach'] is False and group['fused'] is False,'Parent Adam execution mode changed')
    require(source['scheduler']['last_epoch']==source['update'],'Parent scheduler cursor mismatch')
    return parent_sha


def step(model,optimizer,scheduler,sampler,active,uids,tensors,faces,scale,update,dev):
    """Same operation order as the frozen endpoint trainer's C0 loop."""
    indices=[active[i] for i in sampler.take(4)]
    index=torch.tensor(indices,dtype=torch.long,device=dev)
    k=torch.randint(0,32,(4,),device=dev)
    u0,u1,features=tensors
    loss,metrics,alphas=compute_loss(model,'c0',u0[index],u1[index],features[index],k,indices,faces,scale)
    if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at {update}')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
    if any(p.grad is None for p in model.parameters()) or not bool(finite):
        raise FloatingPointError(f'Invalid gradient at {update}')
    grad_l2=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())).detach().cpu())
    optimizer.step();scheduler.step()
    return {'update':update,'loss':float(loss.detach().cpu()),**metrics,
        'uids':[uids[i] for i in indices],'rng_matched_k':k.detach().cpu().tolist(),
        'lr':optimizer.param_groups[0]['lr'],'gradient_l2':grad_l2,
        'alphas':alphas.tolist() if alphas is not None else None}


def train_run(run:Path,job_name:str,resume:Path|None=None,*,device='cuda',stop_after=None,output_dir=None):
    run=Path(run).resolve();protocol=read_json(run/'RUN_PROTOCOL.json')
    require(config_digest(protocol)==protocol['hashes']['config_sha256'],'Configuration hash mismatch')
    require(job_name in protocol['jobs'],'Unknown fitting diagnostic job')
    job=protocol['jobs'][job_name];training=protocol['training'];seed=int(job['seed'])
    for key,value in [('batch_size',4),('learning_rate',1e-4),('weight_decay',1e-4),('optimizer','AdamW')]:
        require(training[key]==value,'Frozen C0 setting changed: '+key)
    require(training.get('rollout_steps',32)==32,'Bridge length changed')
    total=int(job['total_updates']);end=total if stop_after is None else int(stop_after)
    checkpoints=list(training['checkpoint_updates'])
    require(checkpoints==sorted(set(checkpoints)) and all(x>0 for x in checkpoints),'Invalid checkpoint grid')
    require(0<=job['start_update']<end<=total,'Invalid global update budget')
    dev=torch.device(device);require(dev.type in ('cpu','cuda'),'Unsupported device')
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    configure_determinism(seed)
    uids,data,scale,manifest=load_train_archive(run,protocol)
    active_uids=job['active_training_uids']
    require(active_uids and len(set(active_uids))==len(active_uids) and set(active_uids)<=set(uids),'Invalid active train subset')
    active=[uids.index(uid) for uid in active_uids]
    joint=job['kind']=='joint'
    require(job['kind'] in ('joint','single'),'Unknown job kind')
    if joint:
        require(active_uids==uids and job['initialization_mode']=='resume_endpoint_c0_full_state','Joint initialization changed')
    else:
        require(len(active)==1 and job['start_update']==0 and job['initialization_mode']=='v1_model_fresh_optimizer',
                'Single-parent initialization changed')
    initial,initial_sha=bound_file(run,job['initial_checkpoint'],manifest,job['initial_checkpoint_sha256'])
    source=torch.load(initial,map_location='cpu',weights_only=False)
    require(model_sha256(source['model'])==source['model_sha256']==job['initial_model_sha256'],'Initial model hash mismatch')
    parent_sha=validate_parent(source,job,protocol,run,manifest) if joint else None
    if joint:require(source['config']['device']==str(dev),'Parent training device changed during continuation')
    if not joint:
        require(initial_sha==protocol['hashes']['initial_checkpoint_sha256'][str(seed)],'Fresh job v1 checkpoint changed')
        require(source['model_sha256']==protocol['hashes']['initial_model_sha256'][str(seed)],'Fresh job v1 model changed')
    model=DomainVelocity();model.load_state_dict(source['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==29907,'Architecture changed')
    model=model.to(device=dev,dtype=torch.float32).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=BalancedSampler(len(active),seed)
    tensors=[torch.as_tensor(data[k],dtype=dtype,device=dev) for k,dtype in
             [('directions0',torch.float64),('directions1',torch.float64),('point_features',torch.float32)]]
    root=Path(__file__).resolve().parents[1]
    dependencies=['layout_fitdiag/train.py','layout_endpoint/train.py','layout_endpoint/rollout.py',
                  'layout_student/train.py','layout_value/optimizer.py','legacy/networks.py','legacy/chart.py']
    config={'job':job_name,'job_id':job_name,'kind':job['kind'],'arm':'c0','seed':seed,'total_updates':total,
        'start_update':job['start_update'],'batch_size':4,'learning_rate':1e-4,'weight_decay':1e-4,
        'optimizer':'AdamW','scheduler':'constant','initialization_mode':job['initialization_mode'],
        'parameter_count':29907,'proxy_points':256,'network_extra_inputs':[],
        'device':str(dev),'network_dtype':'float32','geometry_dtype':'float64','tf32':False,
        'deterministic_algorithms':True,'rollout_steps':32,'objective':'frozen_endpoint_compute_loss_c0',
        'train_uids':uids,'active_training_uids':active_uids,'endpoint_scale_squared':scale,
        'checkpoint_updates':checkpoints,'checkpoint_selection':'predeclared_final_update_only',
        'protocol_sha256':file_sha256(run/'RUN_PROTOCOL.json'),'config_sha256':config_digest(protocol),
        'training_labels_sha256':protocol['hashes']['training_labels_sha256'],
        'data_manifest_sha256':protocol['hashes']['data_manifest_sha256'],
        'initial_checkpoint_sha256':initial_sha,'initial_model_sha256':source['model_sha256'],
        'parent_protocol_sha256':parent_sha,'source_sha256':{p:file_sha256(root/p) for p in dependencies}}
    trace=[];timing_trace=[];first_update=0;accumulated=0.
    if joint:
        optimizer.load_state_dict(source['optimizer']);scheduler.load_state_dict(source['scheduler'])
        sampler.load_state_dict(source['sampler']);trace=list(source['trace']);timing_trace=list(source['timing_trace'])
        first_update=int(source['update']);accumulated=float(source['elapsed_training_seconds']);restore_rng(source['rng'])
    if resume is not None:
        saved=torch.load(Path(resume),map_location='cpu',weights_only=False)
        require(saved['config']==config,'Resume config/protocol/source mismatch')
        require(saved['model_sha256']==model_sha256(saved['model']),'Resume model hash mismatch')
        require(saved['protocol_sha256']==config['protocol_sha256'],'Resume protocol mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);sampler.load_state_dict(saved['sampler'])
        trace=list(saved['trace']);timing_trace=list(saved['timing_trace'])
        first_update=int(saved['update']);accumulated=float(saved['elapsed_training_seconds']);restore_rng(saved['rng'])
    require(first_update<end,'No new updates requested')
    require(len(trace)==first_update,'Incomplete checkpoint trace')
    out=resolve_under(run,output_dir if output_dir is not None else run/'training'/job_name)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(initial,out/'initial.pt')
    if joint and job['start_update'] in checkpoints:
        shutil.copyfile(initial,out/f"step_{job['start_update']:05d}.pt")
    write_json(out/'IMPORT_RECORD.json',{'job':job_name,'source_checkpoint':initial.relative_to(run).as_posix(),
        'source_checkpoint_sha256':initial_sha,'source_model_sha256':source['model_sha256'],
        'source_update':job['start_update'],'initialization_mode':job['initialization_mode'],
        'optimizer_state_inherited':joint,'rng_state_inherited':joint,'sampler_state_inherited':joint,
        'source_protocol_sha256':parent_sha,'parent_trace_rows':job['start_update'],
        'initial_is_a_snapshot_not_selected_result':True})
    write_json(out/'TRAIN_CONFIG.json',config)
    write_json(out/'START.json',{'job':job_name,'start_update':first_update,'end_update':end,
        'final_budget':total,'resume':str(resume) if resume else None,
        'resume_sha256':file_sha256(resume) if resume else None})
    if dev.type=='cuda':torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev)
    started=time.perf_counter();block_start=started;block_first=first_update+1;last_checkpoint=None
    def checkpoint(update,final):
        if dev.type=='cuda':torch.cuda.synchronize(dev)
        state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
            'sampler':sampler.state_dict(),'trace':trace,'timing_trace':timing_trace,
            'config':config,'job_id':job_name,'update':update,'is_predeclared_final':final,'diagnostic_only':True,
            'elapsed_training_seconds':accumulated+time.perf_counter()-started,
            'protocol_sha256':config['protocol_sha256'],'training_labels_sha256':config['training_labels_sha256'],
            'initial_checkpoint_sha256':initial_sha,'parent_protocol_sha256':parent_sha}
        state['model_sha256']=model_sha256(state['model'])
        path=out/('final.pt' if final else f'step_{update:05d}.pt')
        with path.open('xb') as stream:torch.save(state,stream)
        return path
    try:
        for update in range(first_update+1,end+1):
            trace.append(step(model,optimizer,scheduler,sampler,active,uids,tensors,data['faces'],scale,update,dev))
            if update%1000==0 or update==end:
                if dev.type=='cuda':torch.cuda.synchronize(dev)
                timing={'start_update':block_first,'end_update':update,'updates':update-block_first+1,
                    'wall_seconds':time.perf_counter()-block_start,
                    'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else None}
                timing_trace.append(timing)
                print(json.dumps({'job':job_name,'update':update,'loss':trace[-1]['loss'],**timing}),flush=True)
                block_start=time.perf_counter();block_first=update+1
            if update in checkpoints or update==end:last_checkpoint=checkpoint(update,update==total)
    except Exception as exc:
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),
            'completed_update':trace[-1]['update'] if trace else first_update});raise
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    elapsed_new=time.perf_counter()-started;counts=Counter(u for row in trace for u in row['uids'])
    require([counts[u] for u in active_uids]==sampler.exposures,'Final exposure/trace mismatch')
    require(sum(counts.values())==end*4,'Final exposure total mismatch')
    report={'status':'COMPLETE' if end==total else 'INTERRUPTED_FOR_RESUME_TEST','job':job_name,
        'kind':job['kind'],'seed':seed,'update':end,'final_budget':total,'diagnostic_only':True,
        'checkpoint':str(last_checkpoint),'checkpoint_sha256':file_sha256(last_checkpoint),
        'model_sha256':model_sha256(model.state_dict()),'protocol_sha256':config['protocol_sha256'],
        'parameter_count':29907,'initial_checkpoint_sha256':initial_sha,
        'elapsed_training_seconds':accumulated+elapsed_new,'elapsed_this_invocation_seconds':elapsed_new,
        'imported_parent_updates':job['start_update'],'updates_this_invocation':end-first_update,
        'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else None,
        'peak_gpu_reserved_bytes':int(torch.cuda.max_memory_reserved(dev)) if dev.type=='cuda' else None,
        'parent_exposures':{u:counts[u] for u in uids},'active_training_uids':active_uids,
        'first_loss':trace[0]['loss'],'last_loss':trace[-1]['loss'],
        'last_100_loss_mean':float(np.mean([x['loss'] for x in trace[-100:]])),
        'selection_uses_internal_check':False,'internal_check_teacher_files_opened':0,
        'checkpoint_selection':'predeclared_final_update_only','training_only_archive_uids_verified':True}
    write_json(out/'TRAIN_TRACE.json',trace);write_json(out/'TRAIN_TIMING.json',timing_trace)
    write_json(out/'TRAIN_REPORT.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--job',required=True);p.add_argument('--resume',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--stop-after',type=int);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();print(json.dumps(train_run(a.run,a.job,a.resume,device=a.device,
        stop_after=a.stop_after,output_dir=a.output_dir),indent=2),flush=True)


if __name__=='__main__':main()
