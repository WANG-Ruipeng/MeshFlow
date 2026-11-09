"""Single-GPU harness around PINNED official architecture, Dataset/OT and FM loss.
Not the unchanged upstream training executable. Full optimizer/RNG state is retained.
Local Codex MUST run the adapter gates before using this file for a scientific table.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,time,random
import numpy as np
import torch
import yaml
from .official_contract import audit_loader
from .io_utils import seed_all,new_dir,write_json,atomic_torch,file_hash,rng_state,restore_rng

def build_model(config):
    from models.equidit import DiT
    c={k:v for k,v in config['model'].items() if k!='model_type'}
    return DiT(**c)

@torch.no_grad()
def ema_update(ema,model,beta):
    if not 0<=beta<1:raise ValueError('EMA decay')
    e=dict(ema.named_parameters());m=dict(model.named_parameters())
    if e.keys()!=m.keys():raise ValueError('EMA key mismatch')
    for k in e:e[k].mul_(beta).add_(m[k],alpha=1-beta)

@torch.no_grad()
def fixed_probe(model,transport,ds,device,microbatch):
    from datasets.mesh_dataset import collate_fn
    saved=rng_state();was=model.training;model.eval();losses=[]
    try:
        seed_all(883021) # same all-parent noise/time/OT probe every checkpoint; TRAIN-support diagnostic
        for i in range(0,len(ds),microbatch):
            b=collate_fn([ds[j] for j in range(i,min(i+microbatch,len(ds)))],max_seq_length=800)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                l=transport.training_losses(model,b['tokens'].to(device),b['noise'].to(device),{'y':b['num_faces'].to(device),'mask':b['masks'].to(device)})['loss']
            losses.append(float(l))
        return float(np.mean(losses))
    finally:
        model.train(was);restore_rng(saved)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=10);p.add_argument('--resume');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Full external experiment requires local GPU. No silent CPU tiny-model substitution.')
    from datasets.mesh_dataset import collate_fn
    from flow_matching import create_transport
    cfg=yaml.safe_load(Path(a.config).read_text());run=cfg['closed16_runner'];N=run['updates'];mb=run['microbatch'];acc=run['accumulation_steps']
    if mb*acc!=16 or run['effective_batch']!=16:raise ValueError('Closed16 balanced effective batch must be16')
    seed_all(a.seed);ds,check=audit_loader(a.repo,a.data);seed_all(a.seed)
    model=build_model(cfg).cuda();ema=deepcopy(model).eval().requires_grad_(False);ema_update(ema,model,0.)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],betas=(.9,cfg['optimizer']['beta2']),weight_decay=0.)
    transport=create_transport(cfg['transport']['path_type'],cfg['transport']['prediction'],use_lognorm=cfg['transport']['use_lognorm'],use_jit=cfg['transport']['use_jit'])
    gen=torch.Generator().manual_seed(853000+a.seed);step0=0
    identity={'config_sha256':file_hash(a.config),'dataset_sha256':file_hash(Path(a.data)/'DATASET.json'),'base_commit':check['repo']['base_commit'],'seed':a.seed}
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        if ck['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer']);gen.set_state(ck['sampler_rng']);restore_rng(ck['rng']);step0=ck['step']
    out=new_dir(a.out);write_json(out/'IDENTITY.json',identity);write_json(out/'OFFICIAL_LOADER_GATE.json',check);write_json(out/'CONFIG.json',cfg)
    trace=[];start=time.perf_counter();model.train();sampled=np.zeros(16,dtype=np.int64)
    for step in range(step0,N):
        order=torch.randperm(16,generator=gen).tolist();opt.zero_grad(set_to_none=True);lossval=0.
        for k in range(acc):
            ids=order[k*mb:(k+1)*mb];sampled[ids]+=1
            b=collate_fn([ds[i] for i in ids],max_seq_length=800)
            if tuple(b['tokens'].shape)!=(mb,512,9) or not bool(b['masks'].all()):raise AssertionError('data shape/mask changed')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=transport.training_losses(model,b['tokens'].cuda(),b['noise'].cuda(),{'y':b['num_faces'].cuda(),'mask':b['masks'].cuda()})['loss']/acc
            if not torch.isfinite(loss):raise FloatingPointError('Official loss nonfinite')
            loss.backward();lossval+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['optimizer']['max_grad_norm'])
        if not torch.isfinite(norm):raise FloatingPointError('Official gradient nonfinite')
        opt.step();ema_update(ema,model,run['ema_decay'])
        if (step+1)%run['fit_probe_every']==0 or step+1==N:
            cur=fixed_probe(model,transport,ds,'cuda',mb);ep=fixed_probe(ema,transport,ds,'cuda',mb);torch.cuda.synchronize()
            row={'step':step+1,'current_probe':cur,'ema_probe':ep,'training_batch_loss':lossval,'ema_initialization_coefficient':run['ema_decay']**(step+1),'wall_seconds':time.perf_counter()-start,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved()}
            trace.append(row);write_json(out/'TRACE.json',trace,replace=(out/'TRACE.json').exists());print(row,flush=True)
        if (step+1)%run['checkpoint_every']==0 or step+1==N:
            atomic_torch(out/f'step{step+1:08d}.pt',{'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':gen.get_state(),'rng':rng_state(),'step':step+1,'identity':identity,'config':cfg,'sampled_parent_counts_this_segment':torch.from_numpy(sampled.copy())})
    write_json(out/'TRAINING_COMPLETE.json',{'updates':N,'fit_qualification':'NOT_AUTOMATIC; Codex must attach all-parent fit/probe, sampling and post reports','checkpoint':f'step{N:08d}.pt','primary_weights':'ema','not_released_pretrained_model':True,'original_train_py_not_modified':True})
if __name__=='__main__':main()
