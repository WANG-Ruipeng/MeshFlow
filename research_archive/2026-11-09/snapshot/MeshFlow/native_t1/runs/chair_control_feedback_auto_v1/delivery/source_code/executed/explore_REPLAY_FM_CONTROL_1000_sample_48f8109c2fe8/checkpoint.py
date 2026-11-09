"""Full-state local continuation; protected source checkpoint remains immutable."""
from native_t1 import runtime as _runtime
import gc,os,time
import torch
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256
from native_t1.training import make_optimizer
from experiments.mf_h1_local0.control_mvp48.train_io import tree_cpu,tree_hash,capture_rng,restore_rng
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from experiments.mf_h1_local0.chair_hybrid_coupling_v1 import checkpoint as original
from .common import *
SCHEMA='chair_control_feedback_checkpoint_v1'

def validate(p):
    m=p['metadata'];s=m['feedback_step']
    assert p['schema']==SCHEMA and p['update_in_progress'] is False
    assert 0<=s<=1000 and m['conditional_total_step']==5000+s
    assert m['source_checkpoint_sha256']==START_SHA and m['context_encoder']=='geo' and m['lambda_variance']==.25
    assert m['protocol_sha256']==json_hash(p['protocol'])
    assert m['config_sha256']==config_sha256(DEFAULT_CONFIG)
    assert m['model_state_sha256']==tensor_state_hash(p['model'])
    assert m['optimizer_state_sha256']==tree_hash(p['optimizer'])
    assert m['rng_sha256']==tree_hash(p['rng']) and m['stream_state_sha256']==json_hash(p['stream_state'])
    assert p['stream_state']['batch_index']==s
    assert p['optimizer']['state'] and all(int(v['step'])==5000+s for v in p['optimizer']['state'].values())
    for g in p['optimizer']['param_groups']:assert (g['lr'],tuple(g['betas']),g['weight_decay'])==(1e-5,(.9,.95),0.)
    if m['arm']=='TRAJECTORY_REPLAY_CAPPED':
        assert m['recipe_addendum_sha256']==json_hash(p['recipe_addendum'])
        assert p['recipe_addendum']['arm']==m['arm'] and p['recipe_addendum']['status']=='FROZEN'
        assert p['stream_state']['arm']=='TRAJECTORY_REPLAY'
    return m

def load(path,device='cuda',trainable=False,optimizer=False):
    path=Path(path)
    if path.resolve()==START.resolve():
        assert digest(path)==START_SHA
        model,opt,m,h,ss=original.load(path,device,trainable,optimizer)
        assert m['conditional_total_step']==5000 and m['arm']=='OT_HYBRID'
        return model,opt,dict(m,feedback_step=0),h,None
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True);m=validate(p)
    sha=digest(path);side=read_json(path.with_suffix('.json'))
    assert side['checkpoint_sha256']==sha and side['metadata']==m
    model=original.build_model(p,device,trainable);opt=None
    if optimizer:
        assert trainable;opt=make_optimizer(model);opt.load_state_dict(p['optimizer'])
        assert tree_hash(opt.state_dict())==m['optimizer_state_sha256']
        restore_rng(p['rng']);assert tree_hash(capture_rng())==m['rng_sha256']
    ss=p['stream_state'];del p
    return model,opt,m,dict(checkpoint_sha256=sha,model_state_sha256=m['model_state_sha256']),ss

def save(out,phase,arm,model,opt,stream,protocol,step):
    out=Path(out);label=f'{phase}_{arm}_{step}'
    name=f'{label}.pt' if step in (500,1000) else f'{phase}_{arm}_latest.pt'
    p=out/'checkpoints'/name;p.parent.mkdir(parents=True,exist_ok=True)
    if step in (500,1000) and p.exists():raise FileExistsError(str(p))
    tick=time.perf_counter();state=tree_cpu(model.state_dict());ostate=tree_cpu(opt.state_dict());rng=capture_rng();ss=stream.state_dict()
    meta=dict(phase=phase,arm=arm,feedback_step=step,conditional_total_step=5000+step,
        source_checkpoint_sha256=START_SHA,context_encoder='geo',lambda_variance=.25,
        model_state_sha256=tensor_state_hash(state),optimizer_state_sha256=tree_hash(ostate),
        rng_sha256=tree_hash(rng),stream_state_sha256=json_hash(ss),protocol_sha256=json_hash(protocol),
        config_sha256=config_sha256(DEFAULT_CONFIG),full_recovery_state=True,optimizer_reset=False,
        feedback_pool_sha256=digest(out/'feedback/pool_v1/manifest.json') if (out/'feedback/pool_v1/manifest.json').exists() else None)
    payload=dict(schema=SCHEMA,metadata=meta,model=state,optimizer=ostate,rng=rng,stream_state=ss,protocol=protocol,update_in_progress=False)
    if arm=='TRAJECTORY_REPLAY_CAPPED':
        from .loss_balance import frozen_recipe
        recipe,recipe_file_sha=frozen_recipe(out)
        meta['recipe_addendum_sha256']=json_hash(recipe)
        meta['recipe_addendum_file_sha256']=recipe_file_sha
        payload['recipe_addendum']=recipe
    validate(payload);tmp=p.with_suffix('.pt.tmp');torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,p);r=dict(path=str(p),checkpoint_sha256=digest(p),metadata=meta)
    atomic_json(p.with_suffix('.json'),r)
    del payload,state,ostate
    check=torch.load(p,map_location='cpu',weights_only=True,mmap=True);validate(check);del check;gc.collect()
    r['save_verify_seconds']=time.perf_counter()-tick
    atomic_json(out/'training'/f'{phase}_{arm}_pointer.json',r)
    return r
