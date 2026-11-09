"""Strict old-state recovery, then explicitly registered new parameters only."""
from native_t1 import runtime as _runtime
import gc
import os
from pathlib import Path
import time
import torch
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu,tree_hash,capture_rng,restore_rng
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.checkpoint import load as load_hybrid
from .common import *
from .model_adapter import attach

SCHEMA='chair_surface_attention_checkpoint_v1'


def load_start(arm,device='cpu',optimizer=True):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    model,opt,meta,hashes,ss=load_hybrid(START,device,trainable=optimizer,optimizer=optimizer)
    assert hashes['checkpoint_sha256']==START_SHA
    assert meta['arm']=='OT_HYBRID' and meta['pilot_step']==1000 and meta['conditional_total_step']==5000
    old_names=[n for n,_ in model.named_parameters()]
    old_opt=tree_hash(opt.state_dict()) if opt is not None else None
    rng=capture_rng()
    registration=attach(model,arm,seed=INIT_SEED)
    assert len(registration['new_parameter_names'])==(0 if arm=='H_CONT' else 16)
    assert registration['new_parameter_count']==(0 if arm=='H_CONT' else 104496)
    assert tree_hash(capture_rng())==tree_hash(rng)
    if opt is not None:
        assert tree_hash(opt.state_dict())==old_opt==meta['optimizer_state_sha256']
        named=dict(model.named_parameters());new=[named[n] for n in registration['new_parameter_names']]
        if new:
            opt.add_param_group(dict(params=new,lr=1e-5,betas=(.9,.95),weight_decay=0.))
            for p in new:
                opt.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32,device='cpu'),
                    exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        packed=opt.state_dict();old_group=packed['param_groups'][0]
        old_subset=dict(state={i:packed['state'][i] for i in old_group['params']},param_groups=[old_group])
        assert tree_hash(old_subset)==old_opt
    model._surface_old_parameter_names=old_names
    extended=dict(meta,surface_step=0,conditional_total_step=5000,arm=arm,source_arm='OT_HYBRID',
        model_state_sha256=tensor_state_hash(model.state_dict()),
        optimizer_state_sha256=tree_hash(opt.state_dict()) if opt is not None else None,
        surface_attention=registration,old_optimizer_preserved=True)
    return model,opt,extended,hashes,ss


def validate(payload):
    m=payload['metadata'];step=m['surface_step']
    assert payload['schema']==SCHEMA and payload['update_in_progress'] is False
    assert m['arm'] in ARMS and 0<step<=1000 and m['conditional_total_step']==5000+step
    assert m['source_checkpoint_sha256']==START_SHA and m['context_encoder']=='geo' and m['loss_recipe']=='fm'
    assert m['lambda_variance']==.25 and m['protocol_sha256']==json_hash(payload['protocol'])
    assert m['config_sha256']==config_sha256(DEFAULT_CONFIG)
    assert m['model_state_sha256']==tensor_state_hash(payload['model'])
    assert m['optimizer_state_sha256']==tree_hash(payload['optimizer'])
    assert m['rng_sha256']==tree_hash(payload['rng']) and m['stream_state_sha256']==json_hash(payload['stream_state'])
    assert payload['stream_state']['batch_index']==step and m['completed_updates']==step
    groups=payload['optimizer']['param_groups'];names=payload['optimizer_parameter_names']
    assert len(groups)==(1 if m['arm']=='H_CONT' else 2) and len(names)==len(groups)
    for group_index,g in enumerate(groups):
        assert (g['lr'],tuple(g['betas']),g['weight_decay'])==(1e-5,(.9,.95),0.)
        assert len(g['params'])==len(names[group_index])
        for pid,name in zip(g['params'],names[group_index]):
            assert name.startswith('relation_modules.')==(group_index==1)
            assert int(payload['optimizer']['state'][pid]['step'])==(step if group_index else 5000+step)
    return m


def load(path,device='cpu',trainable=False,optimizer=False):
    path=Path(path)
    if path.resolve()==START.resolve():raise ValueError('Use explicit load_start(arm), never infer branch')
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    m=validate(payload);side=read_json(path.with_suffix('.json'));sha=digest(path)
    assert side['checkpoint_sha256']==sha and side['metadata']==m
    model,opt,_,_,_=load_start(m['arm'],device,optimizer=optimizer)
    model.load_state_dict(payload['model'],strict=True)
    model.set_trainable(trainable).eval()
    if opt is not None:
        opt.load_state_dict(payload['optimizer'])
        assert tree_hash(opt.state_dict())==m['optimizer_state_sha256']
        restore_rng(payload['rng']);assert tree_hash(capture_rng())==m['rng_sha256']
    assert tensor_state_hash(model.state_dict())==m['model_state_sha256']
    ss=payload['stream_state'];del payload;gc.collect()
    return model,opt,m,dict(checkpoint_sha256=sha,model_state_sha256=m['model_state_sha256']),ss


def save(out,arm,model,optimizer,stream,protocol,step):
    out=Path(out);p=out/'checkpoints'/(f'{arm}_surface1000_total6000.pt' if step==1000 else f'{arm}_latest.pt')
    p.parent.mkdir(parents=True,exist_ok=True)
    if step==1000 and p.exists():raise FileExistsError('Fixed endpoint already exists')
    tick=time.perf_counter();state=tree_cpu(model.state_dict());opt=tree_cpu(optimizer.state_dict())
    rng=capture_rng();ss=stream.state_dict();name_by_id={id(p):n for n,p in model.named_parameters()}
    names=[[name_by_id[id(p)] for p in g['params']] for g in optimizer.param_groups]
    meta=dict(arm=arm,surface_step=step,conditional_total_step=5000+step,
        source_checkpoint=str(START),source_checkpoint_sha256=START_SHA,context_encoder='geo',loss_recipe='fm',lambda_variance=.25,
        model_state_sha256=tensor_state_hash(state),optimizer_state_sha256=tree_hash(opt),rng_sha256=tree_hash(rng),
        stream_state_sha256=json_hash(ss),protocol_sha256=json_hash(protocol),config_sha256=config_sha256(DEFAULT_CONFIG),
        completed_updates=step,full_recovery_state=True,optimizer_reset=False,model_parameters='FP32',optimizer_state='FP32',
        surface_attention=model.surface_attention_metadata,old_optimizer_step=5000+step,new_optimizer_step=step)
    payload=dict(schema=SCHEMA,metadata=meta,model=state,optimizer=opt,rng=rng,stream_state=ss,
        protocol=protocol,update_in_progress=False,optimizer_parameter_names=names)
    validate(payload);tmp=p.with_name(p.name+'.tmp');torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,p);record=dict(path=str(p),checkpoint_sha256=digest(p),metadata=meta)
    atomic_json(p.with_suffix('.json'),record);del payload,state,opt
    persisted=torch.load(p,map_location='cpu',weights_only=True,mmap=True);validate(persisted);del persisted;gc.collect()
    record['save_verify_seconds']=time.perf_counter()-tick
    atomic_json(out/'training'/f'{arm}_pointer.json',record);return record
