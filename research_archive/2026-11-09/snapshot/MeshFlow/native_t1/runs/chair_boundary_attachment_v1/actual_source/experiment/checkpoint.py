"""Strict START/AdamW continuation and full factorial recovery checkpoints."""
from native_t1 import runtime as _runtime
import gc
import os
from pathlib import Path
import time
import torch
from native_t1.training import make_optimizer
from native_t1.portable_checkpoint import config_sha256
from native_t1.checkpoint import DEFAULT_CONFIG
from experiments.mf_h1_local0.chair_hybrid_coupling_v1 import checkpoint as old
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu,tree_hash,capture_rng,restore_rng
from .common import *

SCHEMA='chair_boundary_attachment_checkpoint_v1'
tensor_state_hash=old.tensor_state_hash


def boundary_group(optimizer,model,initialize=True):
    p=model.e_boundary
    optimizer.add_param_group(dict(params=[p],lr=1e-5,betas=(.9,.95),weight_decay=0.))
    if initialize:
        optimizer.state[p]={'step':torch.tensor(0.,dtype=torch.float32),
                            'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}


def validate(payload):
    m=payload['metadata'];step=m['attachment_step'];arm=m['arm']
    assert payload['schema']==SCHEMA and payload['update_in_progress'] is False
    assert arm in ARMS and 0<=step<=1000 and m['conditional_total_step']==5000+step
    assert m['source_checkpoint_sha256']==START_SHA and m['lambda_hybrid']==.25
    assert m['model_state_sha256']==tensor_state_hash(payload['model'])
    assert m['optimizer_state_sha256']==tree_hash(payload['optimizer'])
    assert m['rng_sha256']==tree_hash(payload['rng'])
    assert m['stream_state_sha256']==json_hash(payload['stream_state'])
    assert payload['stream_state']['batch_index']==step
    assert m['protocol_sha256']==json_hash(payload['protocol'])
    assert m['config_sha256']==config_sha256(DEFAULT_CONFIG)
    groups=payload['optimizer']['param_groups'];states=payload['optimizer']['state']
    assert len(groups)==(2 if arm in BOUNDARY_ARMS else 1)
    for group in groups:assert (group['lr'],tuple(group['betas']),group['weight_decay'])==(1e-5,(.9,.95),0.)
    assert all(int(states[p]['step'])==5000+step for p in groups[0]['params'])
    if arm in BOUNDARY_ARMS:
        assert len(groups[1]['params'])==1 and tuple(payload['model']['e_boundary'].shape)==(768,)
        assert int(states[groups[1]['params'][0]]['step'])==step
    else:assert 'e_boundary' not in payload['model']
    return m


def load(path=START,arm='A_FM',device='cpu',trainable=True,optimizer=True,out=OUT):
    from .method import attach_boundary
    path=Path(path)
    if path.resolve()==START.resolve():
        assert digest(path)==START_SHA
        model,opt,oldmeta,hashes,_=old.load(path,device,trainable=trainable,optimizer=optimizer)
        assert oldmeta['arm']=='OT_HYBRID' and oldmeta['pilot_step']==1000 and oldmeta['conditional_total_step']==5000
        original_model=tensor_state_hash(model.state_dict())
        original_optimizer=tree_hash(opt.state_dict()) if optimizer else None
        assert original_model==oldmeta['model_state_sha256']
        if optimizer:assert original_optimizer==oldmeta['optimizer_state_sha256']
        boundary_audit=attach_boundary(model,arm in BOUNDARY_ARMS)
        if arm in BOUNDARY_ARMS:
            model.e_boundary.requires_grad_(trainable)
            if optimizer:boundary_group(opt,model)
        model.set_trainable(trainable).eval()
        current={k:v for k,v in model.state_dict().items() if k!='e_boundary'}
        assert tensor_state_hash(current)==original_model
        if optimizer:
            now=opt.state_dict();ids=set(now['param_groups'][0]['params'])
            original_subset=dict(state={k:v for k,v in now['state'].items() if k in ids},param_groups=now['param_groups'][:1])
            assert tree_hash(original_subset)==original_optimizer
        meta=dict(arm=arm,attachment_step=0,conditional_total_step=5000,
                  source_checkpoint_sha256=START_SHA,source_model_state_sha256=original_model,
                  source_optimizer_state_sha256=original_optimizer,lambda_hybrid=.25,
                  model_state_sha256=tensor_state_hash(model.state_dict()))
        audit=dict(status='PASS',arm=arm,source_file_sha256=START_SHA,strict_original_load=True,
            original_model_unchanged=True,original_optimizer_unchanged=optimizer,
            optimizer_reset=False,new_parameter=boundary_audit,new_parameter_step=0 if arm in BOUNDARY_ARMS else None,
            trainable=trainable,source_metadata=oldmeta)
        atomic_json(Path(out)/'load_audits'/f'{arm}_initial.json',audit)
        return model,opt,meta,None
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    meta=validate(payload);assert meta['arm']==arm
    side=read_json(path.with_suffix('.json'));assert side['checkpoint_sha256']==digest(path)
    core=dict(payload['model']);boundary=core.pop('e_boundary',None)
    model=old.build_model({'model':core},device,trainable)
    opt=make_optimizer(model) if optimizer else None
    attach_boundary(model,arm in BOUNDARY_ARMS)
    if arm in BOUNDARY_ARMS:
        assert boundary is not None
        model.e_boundary.requires_grad_(trainable)
        if optimizer:boundary_group(opt,model,initialize=False)
    else:assert boundary is None
    model.load_state_dict(payload['model'],strict=True)
    model.set_trainable(trainable).eval()
    assert tensor_state_hash(model.state_dict())==meta['model_state_sha256']
    if optimizer:
        opt.load_state_dict(payload['optimizer'])
        assert tree_hash(opt.state_dict())==meta['optimizer_state_sha256']
        restore_rng(payload['rng'])
        assert tree_hash(capture_rng())==meta['rng_sha256']
    stream_state=payload['stream_state'];del payload,core,boundary;gc.collect()
    return model,opt,meta,stream_state


def save(out,arm,model,optimizer,stream,protocol,step):
    out=Path(out);tick=time.perf_counter()
    p=out/'checkpoints'/f'{arm}_attachment{step}_total{5000+step}.pt'
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Never overwrite completed checkpoint '+str(p))
    state=tree_cpu(model.state_dict());opt=tree_cpu(optimizer.state_dict());rng=capture_rng();ss=stream.state_dict()
    meta=dict(arm=arm,attachment_step=step,conditional_total_step=5000+step,
        source_checkpoint=str(START),source_checkpoint_sha256=START_SHA,lambda_hybrid=.25,
        boundary_enabled=arm in BOUNDARY_ARMS,seam_enabled=arm in SEAM_ARMS,
        lambda_seam=protocol.get('lambda_seam'),model_state_sha256=tensor_state_hash(state),
        optimizer_state_sha256=tree_hash(opt),rng_sha256=tree_hash(rng),stream_state_sha256=json_hash(ss),
        protocol_sha256=json_hash(protocol),config_sha256=config_sha256(DEFAULT_CONFIG),
        full_recovery_state=True,optimizer_reset=False)
    payload=dict(schema=SCHEMA,metadata=meta,model=state,optimizer=opt,rng=rng,stream_state=ss,
                 protocol=protocol,update_in_progress=False)
    validate(payload)
    tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,p)
    record=dict(path=str(p),checkpoint_sha256=digest(p),metadata=meta)
    atomic_json(p.with_suffix('.json'),record)
    del payload,state,opt;gc.collect()
    check=torch.load(p,map_location='cpu',weights_only=True,mmap=True);validate(check);del check;gc.collect()
    record['save_verify_seconds']=time.perf_counter()-tick
    atomic_json(out/'training'/f'{arm}_pointer.json',record)
    return record
