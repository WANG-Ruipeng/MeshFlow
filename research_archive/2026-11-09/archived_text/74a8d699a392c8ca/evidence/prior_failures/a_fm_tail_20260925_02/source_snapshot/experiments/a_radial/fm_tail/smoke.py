"""Independent original-loss next-step reference and actual CUDA resume smoke."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import torch
from ..models import ShapeFM
from ..io_utils import new_dir,write_json,atomic_torch,file_hash
from ..system_probe.official import exact_same,require,configure
from .train import TailSession,LR,SCOPE,capture_rng,restore_on_device,contract,make_session,load_probe

CORE=['model','optimizer','scheduler','step','rng','batch_rng','draw_queue']


def semantic(state):return {key:value for key,value in state.items() if key!='timings'}


def assert_parent_import(session,parent):
    exact_same(session.model.state_dict(),parent['model'],'import_model')
    actual=deepcopy(session.optimizer.state_dict());expected=deepcopy(parent['optimizer'])
    for a,b in zip(actual['param_groups'],expected['param_groups']):
        require(a['lr']==a['initial_lr']==LR,'New optimizer LR wrong');a['lr']=b['lr'];a['initial_lr']=b['initial_lr']
    exact_same(actual,expected,'import_Adam_except_declared_LR_bases')
    exact_same(session.source_scheduler,parent['scheduler'],'archived_source_scheduler')
    exact_same(session.source_config,parent['config'],'archived_source_config')
    exact_same(capture_rng(session.device),parent['rng'],'import_all_RNG')
    require(torch.equal(session.generator.get_state(),parent['batch_rng']) and session.queue==parent['draw_queue'],'Import sampler changed')
    require(session.scheduler.last_epoch==parent['step'] and session.scheduler.base_lrs==[LR],'Scheduler counter or base changed')


def manual_original_next(parent,targets,device):
    """Frozen train_ours FM arithmetic, with the declared new constant scheduler."""
    device=torch.device(device);model=ShapeFM().to(device);model.load_state_dict(parent['model']);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4);opt.load_state_dict(deepcopy(parent['optimizer']))
    for group in opt.param_groups:group['lr']=1e-4;group['initial_lr']=1e-4
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.,last_epoch=parent['step']-1)
    gen=torch.Generator(device='cpu');gen.set_state(parent['batch_rng']);order=list(parent['draw_queue'])
    restore_on_device(parent['rng'],device)
    target=targets.to(device)
    while len(order)<128:order+=torch.randperm(len(target),generator=gen).tolist()
    ix=order[:128];order=order[128:];x1=target[ix]
    x0=torch.randn_like(x1);t=torch.rand(len(x1),device=device);xt=(1-t[:,None])*x0+t[:,None]*x1
    loss=(model(xt,t)-(x1-x0)).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('Reference loss')
    opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
    if not torch.isfinite(norm):raise FloatingPointError('Reference gradient')
    opt.step();sched.step()
    return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'step':parent['step']+1,
        'rng':capture_rng(device),'batch_rng':gen.get_state(),'draw_queue':order},float(loss.detach()),float(norm)


def run(run,out):
    run=Path(run).resolve();p,sha=contract(run);out=new_dir(out);configure(10)
    require(torch.cuda.is_available(),'Actual CUDA smoke required')
    records=[]
    try:
        for seed in (10,11):
            seed_dir=new_dir(out/f'seed_{seed}');parent=torch.load(run/p['jobs'][str(seed)]['parent_checkpoint'],map_location='cpu',weights_only=True)
            session=make_session(run,p,sha,seed,'cuda');assert_parent_import(session,parent)
            targets=session.targets.detach().clone();del session;gc.collect();torch.cuda.empty_cache()
            manual,loss,norm=manual_original_next(parent,targets,'cuda')
            session=make_session(run,p,sha,seed,'cuda');row=session.update();state=session.state()
            for key in CORE:exact_same(manual[key],state[key],'manual_next.'+key)
            require(row['train_batch_loss']==loss and row['grad_norm']==norm,'Next-step loss/gradient changed')
            del manual,state,session;gc.collect();torch.cuda.empty_cache()
            session=make_session(run,p,sha,seed,'cuda');assert_parent_import(session,parent)
            for _ in range(8):session.update()
            before=deepcopy(session.state());probe=session.probe(load_probe(run,p['jobs'][str(seed)]))
            exact_same(semantic(before),semantic(session.state()),'fixed_probe_full_state')
            whole=seed_dir/'continuous8.pt';atomic_torch(whole,session.state());del session,before;gc.collect();torch.cuda.empty_cache()
            session=make_session(run,p,sha,seed,'cuda')
            for _ in range(3):session.update()
            split=seed_dir/'first3.pt';atomic_torch(split,session.state());del session;gc.collect();torch.cuda.empty_cache()
            session=make_session(run,p,sha,seed,'cuda',split)
            for _ in range(5):session.update()
            resumed=seed_dir/'resumed3plus5.pt';atomic_torch(resumed,session.state());del session;gc.collect();torch.cuda.empty_cache()
            a=torch.load(whole,map_location='cpu',weights_only=True);b=torch.load(resumed,map_location='cpu',weights_only=True)
            exact_same(semantic(a),semantic(b),'CUDA_resume_all_semantics')
            record={'seed':seed,'status':'PASS','actual_new_optimizer_updates':18,'continuous_vs3plus5_all_state_bitwise':True,
                'manual_next_original_arithmetic_bitwise':True,'full_parent_Adam_rng_sampler_import_exact':True,
                'constant_lr':a['optimizer']['param_groups'][0]['lr'],'scheduler_base_lrs':a['scheduler']['base_lrs'],
                'scheduler_counter':a['scheduler']['last_epoch'],'fixed_probe_preserves_all_state':True,
                'probe':probe,'checkpoints_sha256':{path.name:file_hash(path) for path in [whole,split,resumed]}}
            records.append(record);write_json(seed_dir/'REPORT.json',record)
        gate={'status':'PASS','protocol_sha256':sha,'scope':'CUDA_CORRECTNESS_ONLY_NOT_FORMAL_INITIALIZATION',
            'source_sha256':file_hash(__file__),'out':str(out.resolve()),'seeds':records,'actual_new_optimizer_updates':36,
            'GPU':torch.cuda.get_device_name(0),'TF32':torch.backends.cuda.matmul.allow_tf32,
            'original8k_source_checkpoints_unmodified':True}
        contract(run);write_json(out/'GPU_GATE.json',gate);write_json(run/'GPU_GATE.json',gate);print(json.dumps(gate,indent=2),flush=True)
        return gate
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'completed_seeds':records,'formal_training_not_started':True});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.run,a.out)


if __name__=='__main__':main()
