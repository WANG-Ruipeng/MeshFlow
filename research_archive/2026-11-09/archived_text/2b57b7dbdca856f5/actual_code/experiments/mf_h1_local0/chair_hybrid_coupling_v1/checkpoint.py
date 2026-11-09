"""Strict pilot payloads; original trained Geo and AdamW are restored, never reset."""
from native_t1 import runtime as _runtime
import gc
import os
from pathlib import Path
import time
import torch
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.context_geometry import ContextGeometryEncoder
from native_t1.portable_checkpoint import _read_config,config_sha256
from native_t1.training import make_optimizer
from experiments.mf_h1_local0.control_mvp48.model_mvp import MVPModel
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu,tree_hash,capture_rng,restore_rng
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import load_checkpoint as load_old, tensor_state_hash
from .common import START,START_SHA,ARMS,read_json,digest,json_hash,atomic_json

SCHEMA='chair_hybrid_coupling_checkpoint_v1'


def validate(payload):
    m=payload['metadata'];step=m['pilot_step']
    assert payload['schema']==SCHEMA and payload['update_in_progress'] is False
    assert m['arm'] in ARMS and 0<=step<=1000 and m['conditional_total_step']==4000+step
    assert m['source_checkpoint_sha256']==START_SHA and m['context_encoder']=='geo' and m['loss_recipe']=='fm'
    assert m['lambda_variance']==(0. if m['arm']=='OT_BASE' else .25)
    assert m['protocol_sha256']==json_hash(payload['protocol'])
    assert m['config_sha256']==config_sha256(DEFAULT_CONFIG)
    assert m['model_state_sha256']==tensor_state_hash(payload['model'])
    assert m['optimizer_state_sha256']==tree_hash(payload['optimizer'])
    assert m['rng_sha256']==tree_hash(payload['rng'])
    assert m['stream_state_sha256']==json_hash(payload['stream_state'])
    assert payload['stream_state']['batch_index']==step
    assert payload['optimizer']['state'] and all(int(s['step'])==4000+step for s in payload['optimizer']['state'].values())
    for g in payload['optimizer']['param_groups']:
        assert (g['lr'],tuple(g['betas']),g['weight_decay'])==(1e-5,(.9,.95),0.)
    assert m['completed_updates']==step
    return m


def build_model(payload,device,trainable):
    config=_read_config(Path(DEFAULT_CONFIG))
    with torch.device('meta'):
        model=MVPModel(DiT(**config),trainable=False)
    model.backbone.x_embedder.embed_fn,_=get_embedder(config['pe_freq'],input_dims=3)
    # This empty module is immediately replaced by every trained state tensor strictly.
    model.context_encoder=ContextGeometryEncoder('geo',seed=1010)
    model.load_state_dict(payload['model'],strict=True,assign=True)
    return model.set_trainable(trainable).eval().to(device)


def load(path,device='cpu',trainable=False,optimizer=False):
    path=Path(path)
    if path.resolve()==START.resolve():
        if digest(path)!=START_SHA: raise ValueError('Pilot source file differs')
        model,meta,hashes=load_old(path,device,trainable=trainable)
        assert meta['variant']=='Geo' and meta['phase_step']==3000 and meta['conditional_total_step']==4000
        assert meta['full_recovery_state']
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    else:
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        meta=validate(payload); sha=digest(path)
        side=read_json(path.with_suffix('.json'))
        assert side['checkpoint_sha256']==sha and side['metadata']==meta
        model=build_model(payload,device,trainable)
        hashes=dict(checkpoint_sha256=sha,model_state_sha256=meta['model_state_sha256'])
    opt=None
    if optimizer:
        if not trainable: raise ValueError('Optimizer recovery requires trainable model')
        opt=make_optimizer(model);opt.load_state_dict(payload['optimizer'])
        if tree_hash(opt.state_dict())!=meta['optimizer_state_sha256']:
            raise ValueError('Restored AdamW differs; fresh optimizer fallback prohibited')
        restore_rng(payload['rng'])
        if tree_hash(capture_rng())!=meta['rng_sha256']: raise ValueError('Restored RNG differs')
    stream_state=payload['stream_state'];del payload
    return model,opt,meta,hashes,stream_state


def save(out,arm,model,optimizer,stream,protocol,step):
    out=Path(out); final=step==1000
    p=out/'checkpoints'/(f'{arm}_pilot1000_total5000.pt' if final else f'{arm}_latest.pt')
    p.parent.mkdir(parents=True,exist_ok=True)
    if final and p.exists(): raise FileExistsError('Fixed endpoint already exists')
    tick=time.perf_counter();state=tree_cpu(model.state_dict());opt=tree_cpu(optimizer.state_dict())
    rng=capture_rng();ss=stream.state_dict()
    meta=dict(arm=arm,pilot_step=step,conditional_total_step=4000+step,
              source_checkpoint=str(START),source_checkpoint_sha256=START_SHA,
              context_encoder='geo',loss_recipe='fm',lambda_variance=0. if arm=='OT_BASE' else .25,
              model_state_sha256=tensor_state_hash(state),optimizer_state_sha256=tree_hash(opt),
              rng_sha256=tree_hash(rng),stream_state_sha256=json_hash(ss),protocol_sha256=json_hash(protocol),
              config_sha256=config_sha256(DEFAULT_CONFIG),completed_updates=step,full_recovery_state=True,
              optimizer_reset=False,model_parameters='FP32',optimizer_state='FP32')
    payload=dict(schema=SCHEMA,metadata=meta,model=state,optimizer=opt,rng=rng,stream_state=ss,
                 protocol=protocol,update_in_progress=False)
    validate(payload)
    tmp=p.with_name(p.name+'.tmp');torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,p);record=dict(path=str(p),checkpoint_sha256=digest(p),metadata=meta)
    atomic_json(p.with_suffix('.json'),record)
    del payload,state,opt
    persisted=torch.load(p,map_location='cpu',weights_only=True,mmap=True);validate(persisted)
    del persisted;gc.collect()
    record['save_verify_seconds']=time.perf_counter()-tick
    atomic_json(out/'training'/f'{arm}_pointer.json',record)
    return record
