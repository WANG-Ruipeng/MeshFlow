"""Independent independent weighted next-step oracle and exact balanced continuation/resume smoke."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ..exporter import ASSETS
from ..legacy.networks import DomainVelocity
from ..layout_student.train import (BalancedSampler,configure_determinism,restore_rng,
    rng_state,file_sha256,model_sha256)
from ..layout_endpoint import train as frozen
from ..layout_endpoint.smoke import exact_same
from .train import config_digest,train_run

EQUAL_KEYS=['model','optimizer','scheduler','rng','sampler','config','trace',
    'update','is_predeclared_final','diagnostic_only','job_id','model_sha256',
    'protocol_sha256','training_labels_sha256','initial_checkpoint_sha256','parent_protocol_sha256']


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def manual_parent_next(run,job_name,device):
    """One weighted update written independently of objective.step and balanced.train_run."""
    run=Path(run).resolve();p=frozen.read_json(run/'RUN_PROTOCOL.json');job=p['jobs'][job_name]
    assert job['kind']=='joint'
    seed=job['seed'];configure_determinism(seed)
    uids,data,scale,_=frozen.load_train_archive(run,p)
    s=torch.load(run/job['initial_checkpoint'],map_location='cpu',weights_only=False)
    dev=torch.device(device)
    model=DomainVelocity();model.load_state_dict(s['model'],strict=True);model=model.to(dev).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=BalancedSampler(len(uids),seed)
    u0=torch.as_tensor(data['directions0'],dtype=torch.float64,device=dev)
    u1=torch.as_tensor(data['directions1'],dtype=torch.float64,device=dev)
    features=torch.as_tensor(data['point_features'],dtype=torch.float32,device=dev)
    optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
    sampler.load_state_dict(s['sampler']);restore_rng(s['rng'])
    indices=sampler.take(4);index=torch.tensor(indices,dtype=torch.long,device=dev)
    k=torch.randint(0,32,(4,),device=dev)
    from ..layout_student.train import discrete_bridge
    positions,target,_,times=discrete_bridge(u0[index],u1[index],k,32)
    prediction=model(positions.float(),times.float(),features[index])
    squared=(prediction-target.float()).square()
    weights=torch.tensor(p['loss']['parent_weighting']['weights'],dtype=torch.float32,device=dev)[index]
    loss=(squared*weights[:,None,None]).mean()
    parent_mse=squared.mean(dim=(1,2))
    metrics={'velocity_mse':float(parent_mse.mean().detach().cpu()),'endpoint_loss':None,'quality_penalty':None,
        'parent_velocity_mse':parent_mse.detach().cpu().tolist(),'parent_weights':weights.detach().cpu().tolist(),
        'loss_is_parent_weighted':True}
    alphas=None
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grad=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())).detach().cpu())
    optimizer.step();scheduler.step()
    record={'update':s['update']+1,'loss':float(loss.detach().cpu()),**metrics,
        'uids':[uids[i] for i in indices],'rng_matched_k':k.detach().cpu().tolist(),
        'lr':optimizer.param_groups[0]['lr'],'gradient_l2':grad,
        'alphas':alphas.tolist() if alphas is not None else None}
    return {'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer':cpu_tree(optimizer.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),
        'sampler':sampler.state_dict(),'rng':rng_state(),'trace':s['trace']+[record]}


def make_smoke_run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    p=frozen.read_json(source/'RUN_PROTOCOL.json');parent_sha=file_sha256(source/'RUN_PROTOCOL.json')
    manifest=frozen.read_json(source/'DATA_MANIFEST.json')['files']
    relatives={p['paths'].get('training_labels','train_labels.npz'),
        p['paths'].get('source_protocol','initial/fitdiag_RUN_PROTOCOL.json'),
        *p['paths']['initial_checkpoints'].values(),
        *(j['initial_checkpoint'] for j in p['jobs'].values())}
    files={}
    for rel in sorted(relatives):
        assert file_sha256(source/rel)==manifest[rel]
        destination=out/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/rel,destination);files[rel]=file_sha256(destination)
        assert files[rel]==manifest[rel]
    frozen.write_json(out/'DATA_MANIFEST.json',{'files':files})
    p['experiment']='A-BALANCED-CUDA-SMOKE-ONLY';p['scope']='SMOKE_ONLY: never deploy or compare scientifically'
    p['smoke_parent_protocol_sha256']=parent_sha
    for job in p['jobs'].values():job['total_updates']=job['start_update']+8
    p['training']['updates']='per_job_start_plus_8_smoke_only'
    p['training']['checkpoint_updates']=sorted({j['start_update'] for j in p['jobs'].values() if j['start_update']>0})
    p['hashes']['data_manifest_sha256']=file_sha256(out/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p)
    frozen.write_json(out/'RUN_PROTOCOL.json',p)
    return out,parent_sha,files


def smoke_run(source,out,device='cuda'):
    if torch.device(device).type=='cuda':assert torch.cuda.is_available(),'Real CUDA required'
    out,parent_sha,input_files=make_smoke_run(source,out);source=Path(source).resolve()
    p=frozen.read_json(out/'RUN_PROTOCOL.json');started=time.perf_counter()
    reports={};records=[];opened=[];current_job=None
    old_step=torch.optim.AdamW.step;old_load=np.load
    def audited_step(self,*args,**kwargs):
        params=[v for group in self.param_groups for v in group['params']]
        assert all(v.device.type==torch.device(device).type and v.dtype==torch.float32 for v in params)
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        norm=float(torch.sqrt(sum(v.grad.double().square().sum() for v in params)).detach().cpu())
        assert norm>0
        records.append({'job':current_job,'device':str(params[0].device),'dtype':str(params[0].dtype),
            'gradient_l2':norm,'parameter_count':sum(v.numel() for v in params),'finite':True})
        return old_step(self,*args,**kwargs)
    def whitelist(path,*args,**kwargs):
        assert Path(path).resolve() in {(out/'train_labels.npz').resolve(),(ASSETS/'support.npz').resolve()}
        opened.append(str(path));return old_load(path,*args,**kwargs)
    torch.optim.AdamW.step=audited_step;np.load=whitelist
    try:
        for job_name,job in p['jobs'].items():
            current_job=job_name;start=job['start_update']
            whole=train_run(out,job_name,device=device,output_dir=out/job_name/'whole')
            first=train_run(out,job_name,device=device,stop_after=start+3,output_dir=out/job_name/'first')
            resumed=train_run(out,job_name,Path(first['checkpoint']),device=device,output_dir=out/job_name/'resumed')
            a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
            b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
            for key in EQUAL_KEYS:exact_same(a[key],b[key],job_name+'.'+key)
            assert a['model_sha256']==model_sha256(a['model'])
            assert not torch.load(first['checkpoint'],map_location='cpu',weights_only=False)['is_predeclared_final']
            oracle_pass=None
            if job['kind']=='joint':
                manual=manual_parent_next(out,job_name,device)
                imported=train_run(out,job_name,device=device,stop_after=start+1,output_dir=out/job_name/'parent_next')
                actual=torch.load(imported['checkpoint'],map_location='cpu',weights_only=False)
                for key in manual:exact_same(manual[key],actual[key],job_name+'.parent_next.'+key)
                assert file_sha256(out/job_name/'whole'/f'step_{start:05d}.pt')==job['initial_checkpoint_sha256']
                oracle_pass=True
            assert file_sha256(out/job_name/'whole'/'initial.pt')==job['initial_checkpoint_sha256']
            reports[job_name]={'status':'PASS','continuous':whole,'first':first,'resumed':resumed,
                'exactly_equal_keys':EQUAL_KEYS,'manual_weighted_parent_first_step_equal':oracle_pass,
                'model_sha256':a['model_sha256']}
    finally:
        torch.optim.AdamW.step=old_step;np.load=old_load
    assert file_sha256(source/'RUN_PROTOCOL.json')==parent_sha
    for rel,digest in input_files.items():assert file_sha256(source/rel)==digest
    result={'status':'PASS','scope':'SMOKE_ONLY_BALANCED','jobs':reports,'device':device,
        'continuous_new_updates':8,'split_new_updates':[3,5],'actual_total_optimizer_updates':len(records),
        'full_semantic_training_state_bitwise_equal':True,'timing_fields_excluded_from_bitwise_comparison':True,
        'manual_weighted_parent_next_update_bitwise_equal':True,'gradient_records':records,
        'formal_inputs_unchanged':True,'formal_protocol_sha256':parent_sha,'internal_check_teacher_reads':0,
        'training_archive_reads':len(opened),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cuda_device':torch.cuda.get_device_name(0) if torch.device(device).type=='cuda' else None,
        'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
        'peak_gpu_allocated_bytes':max((r[k]['peak_gpu_allocated_bytes'] or 0) for r in reports.values()
            for k in ['continuous','first','resumed']),
        'train_source_sha256':file_sha256(Path(__file__).with_name('train.py')),
        'smoke_source_sha256':file_sha256(Path(__file__)),'elapsed_wall_seconds':time.perf_counter()-started}
    shutil.copyfile(Path(__file__),out/'SMOKE_SOURCE.py')
    frozen.write_json(out/('GPU_SMOKE.json' if torch.device(device).type=='cuda' else 'CPU_SMOKE.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['jobs','gradient_records']},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cuda')
    a=p.parse_args();smoke_run(a.source_run,a.out,a.device)


if __name__=='__main__':main()
