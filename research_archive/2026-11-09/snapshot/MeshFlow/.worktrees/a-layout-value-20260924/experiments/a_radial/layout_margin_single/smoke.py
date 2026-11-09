"""Actual CPU eight-versus-three-plus-five gate and independent frozen C0 update."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ..layout_endpoint.train import compute_loss
from .train import (config_digest,make_session,semantic_state,train_run,read_json,
                    write_json,file_sha256,require)

EQUAL_KEYS=['model','optimizer','scheduler','rng','sampler','config','trace','update',
    'is_predeclared_final','diagnostic_only','job_id','model_sha256','protocol_sha256',
    'training_labels_sha256','initial_checkpoint_sha256','step_sequence_sha256']


def exact_same(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape,path
        assert a.device.type==b.device.type=='cpu' and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:exact_same(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact_same(x,y,path+'.'+str(i))
    else:assert a==b,path


def manual_original_next(run,job_name):
    """One independently written C0 update; never calls the new/frozen step helper."""
    s=make_session(run,job_name);j=s['job'];model=s['model'];optimizer=s['optimizer']
    indices=[j['index'] for _ in s['sampler'].take(4)]
    index=torch.tensor(indices,dtype=torch.long,device='cpu');k=torch.randint(0,32,(4,),device='cpu')
    u0,u1,features=s['tensors']
    loss,metrics,alphas=compute_loss(model,'c0',u0[index],u1[index],features[index],k,
        indices,s['data']['faces'],s['protocol']['endpoint_scale_squared'])
    optimizer.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters())).detach().cpu())
    optimizer.step();s['scheduler'].step()
    s['trace']=[{'update':1,'loss':float(loss.detach().cpu()),**metrics,
        'uids':[s['uids'][i] for i in indices],'rng_matched_k':k.tolist(),
        'lr':optimizer.param_groups[0]['lr'],'gradient_l2':norm,
        'alphas':alphas.tolist() if alphas is not None else None}]
    s['update']=1
    return copy.deepcopy(semantic_state(s))


def make_smoke_run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();p=read_json(source/'RUN_PROTOCOL.json')
    parent_sha=file_sha256(source/'RUN_PROTOCOL.json')
    manifest=read_json(source/p['paths']['data_manifest'])['files']
    out.mkdir(parents=True,exist_ok=False);files={}
    for relative in sorted({p['paths']['training_labels'],*(j['initial_checkpoint'] for j in p['jobs'].values())}):
        assert file_sha256(source/relative)==manifest[relative]
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,dest)
        files[relative]=file_sha256(dest);assert files[relative]==manifest[relative]
    write_json(out/'DATA_MANIFEST.json',{'files':files})
    p['scope']='SMOKE_ONLY';p['experiment']='A-MARGIN-SINGLE-CPU-SMOKE'
    p['smoke_parent_protocol_sha256']=parent_sha
    p['training']['total_updates']=8;p['training']['checkpoint_updates']=[0,8]
    for job in p['jobs'].values():job['total_updates']=8
    p['hashes']['data_manifest_sha256']=file_sha256(out/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p)
    write_json(out/'RUN_PROTOCOL.json',p)
    return out,parent_sha,files


def smoke_run(source,out):
    source=Path(source).resolve();out,parent_sha,files=make_smoke_run(source,out)
    p=read_json(out/'RUN_PROTOCOL.json');require(len(p['jobs'])==8,'All eight registered jobs must be gated')
    require(not torch.cuda.is_initialized(),'CPU smoke process already owns a CUDA context')
    records=[];opened=[];reports={};current=None;started=time.perf_counter()
    old_step=torch.optim.AdamW.step;old_load=np.load
    rng_names=['get_rng_state','get_rng_state_all','set_rng_state','set_rng_state_all','manual_seed','manual_seed_all']
    originals={name:getattr(torch.cuda,name) for name in rng_names}
    def forbidden(*args,**kwargs):raise AssertionError('CUDA RNG API forbidden in CPU experiment')
    def audited_step(self,*args,**kwargs):
        params=[x for group in self.param_groups for x in group['params']]
        assert all(x.device.type=='cpu' and x.dtype==torch.float32 and x.grad is not None
                   and torch.isfinite(x.grad).all() for x in params)
        norm=float(torch.sqrt(sum(x.grad.double().square().sum() for x in params)))
        records.append({'job':current,'gradient_l2':norm,'finite':True,'device':'cpu',
                        'parameter_count':sum(x.numel() for x in params)})
        return old_step(self,*args,**kwargs)
    def whitelist(path,*args,**kwargs):
        assert Path(path).resolve()==(out/p['paths']['training_labels']).resolve(),'Non-training archive opened'
        opened.append(str(path));return old_load(path,*args,**kwargs)
    try:
        torch.optim.AdamW.step=audited_step;np.load=whitelist
        for name in rng_names:setattr(torch.cuda,name,forbidden)
        for job_name in p['jobs']:
            current=job_name
            whole=train_run(out,job_name,output_dir=out/job_name/'whole')
            first=train_run(out,job_name,stop_after=3,output_dir=out/job_name/'first')
            resumed=train_run(out,job_name,Path(first['checkpoint']),output_dir=out/job_name/'resumed')
            a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
            b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
            for key in EQUAL_KEYS:exact_same(a[key],b[key],job_name+'.resume.'+key)
            manual=manual_original_next(out,job_name)
            next_report=train_run(out,job_name,stop_after=1,output_dir=out/job_name/'manual_next')
            actual=torch.load(next_report['checkpoint'],map_location='cpu',weights_only=False)
            for key in EQUAL_KEYS:exact_same(manual[key],actual[key],job_name+'.manual.'+key)
            initial=torch.load(out/job_name/'whole/initial.pt',map_location='cpu',weights_only=False)
            assert initial['update']==0 and not initial['optimizer']['state'] and initial['sampler']['exposures']==[0]
            assert initial['model_sha256']==p['jobs'][job_name]['initial_model_sha256']
            reports[job_name]={'status':'PASS','whole':whole,'first':first,'resumed':resumed,
                'manual_original_next_bitwise_equal':True,'step_sequence_sha256':a['step_sequence_sha256'],
                'model_sha256':a['model_sha256'],'fresh_step_zero_complete':True}
    except Exception as exc:
        write_json(out/'FAILED.json',{'job':current,'type':type(exc).__name__,'error':str(exc),
            'actual_optimizer_updates':len(records),'no_automatic_retry':True});raise
    finally:
        torch.optim.AdamW.step=old_step;np.load=old_load
        for name,value in originals.items():setattr(torch.cuda,name,value)
    pairs={}
    for name,job in p['jobs'].items():pairs.setdefault(job['pair_id'],[]).append(name)
    require(len(pairs)==4,'Expected four paired parents')
    for uid,names in pairs.items():
        require(len(names)==2 and {p['jobs'][name]['target_kind'] for name in names}=={'old','margin'},'Target pair incomplete')
        require(len({reports[name]['step_sequence_sha256'] for name in names})==1,'Paired targets used different k sequence')
    require(len(records)==144,'Unexpected actual update count')
    require(file_sha256(source/'RUN_PROTOCOL.json')==parent_sha,'Formal protocol changed')
    require(all(file_sha256(source/name)==digest for name,digest in files.items()),'Formal input changed')
    require(not torch.cuda.is_initialized(),'CPU smoke initialized CUDA')
    result={'status':'PASS','scope':'SMOKE_ONLY','device':'cpu','torch_threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'cuda_context_initialized':False,'cuda_rng_calls':0,'full_semantic_state_bitwise_equal':True,
        'continuous_updates':8,'split_updates':[3,5],'actual_optimizer_updates':len(records),
        'manual_frozen_C0_next_bitwise_equal':True,'paired_k_sequences_equal':True,'jobs':reports,
        'gradient_records':records,'training_only_archive_reads':len(opened),'internal_check_teacher_reads':0,
        'timing_fields_excluded_from_bitwise_comparison':True,'formal_inputs_unchanged':True,
        'formal_protocol_sha256':parent_sha,'elapsed_wall_seconds':time.perf_counter()-started,
        'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ['train.py','smoke.py','test_train.py']}}
    write_json(out/'CPU_SMOKE.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['jobs','gradient_records']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-run',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out).resolve();existed=out.exists()
    try:smoke_run(args.source_run,out)
    except Exception as exc:
        if not existed and out.exists() and not (out/'FAILED.json').exists():
            write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'no_automatic_retry':True})
        raise


if __name__=='__main__':main()
