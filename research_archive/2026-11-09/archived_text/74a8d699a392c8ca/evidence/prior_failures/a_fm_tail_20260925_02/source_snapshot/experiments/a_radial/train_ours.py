"""Minimal single-device local runner; no architecture search, closed-set metrics only."""
from __future__ import annotations
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from .models import ShapeAE,ShapeFM,fit_pca_train_only
from .io_utils import seed_all,new_dir,write_json,file_hash,model_hash,atomic_torch,rng_state,restore_rng

def run(args):
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    cfg={'stage':args.stage,'steps':args.steps,'batch':args.batch,'seed':args.seed,'dataset_sha256':file_hash(Path(args.data)/'radial_train.npz'),'dataset_manifest_sha256':file_hash(Path(args.data)/'DATASET.json'),'ae_sha256':file_hash(args.ae) if args.ae else None,'lr':args.lr,'device':args.device,'arch':'ShapeAE1024_z64 or ShapeFM128_z64','role':'closed_set16_not_generalization'}
    seed_all(args.seed);torch.set_num_threads(args.threads)
    with np.load(Path(args.data)/'radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr']).to(dev);uids=z['uids'].tolist()
    model=ShapeAE() if args.stage=='ae' else ShapeFM();model=model.to(dev)
    pca_info=None
    if args.stage=='ae':pca_info=fit_pca_train_only(model,y);target=y
    else:
        if not args.ae:raise ValueError('--ae required for FM')
        ck=torch.load(args.ae,map_location='cpu',weights_only=True);ae=ShapeAE().to(dev);ae.load_state_dict(ck['model']);ae.eval().requires_grad_(False)
        if ck['config']['dataset_sha256']!=cfg['dataset_sha256']:raise ValueError('AE trained on another cache')
        with torch.no_grad():codes=ae.encode(y)
        normalization=model.fit_normalization(codes);target=model.standardize(codes).detach()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=args.lr*.1)
    gen=torch.Generator(device='cpu').manual_seed(args.seed+731000);start=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        for k in ('stage','steps','batch','seed','dataset_sha256','dataset_manifest_sha256','ae_sha256','lr'):
            if ck['config'][k]!=cfg[k]:raise ValueError('Resume config mismatch: '+k)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);gen.set_state(ck['batch_rng']);restore_rng(ck['rng']);start=int(ck['step'])
    out=new_dir(args.out);write_json(out/'CONFIG.json',cfg)
    if pca_info:write_json(out/'PCA_DIAGNOSTIC.json',pca_info)
    if args.stage=='fm':write_json(out/'NORMALIZATION.json',normalization)
    trace=[];elapsed=time.perf_counter();batch_hashes=[];model.train()
    # Balanced cycle: every16 consecutive draws cover all parents; no drop_last empty loader.
    order=[]
    if args.resume:order=ck['draw_queue']
    for step in range(start,args.steps):
        while len(order)<args.batch:order+=torch.randperm(len(target),generator=gen).tolist()
        ix=order[:args.batch];order=order[args.batch:];x1=target[ix]
        if args.stage=='ae':
            pred,z=model(x1);loss=(pred-x1).square().mean()+1e-7*z.square().mean()
        else:
            x0=torch.randn_like(x1);t=torch.rand(len(x1),device=dev);xt=(1-t[:,None])*x0+t[:,None]*x1
            loss=(model(xt,t)-(x1-x0)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Training loss not finite')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(norm):raise FloatingPointError('Training gradients not finite')
        opt.step();sched.step()
        if (step+1)%args.log_every==0 or step+1==args.steps:
            if dev.type=='cuda':torch.cuda.synchronize()
            row={'step':step+1,'train_batch_loss':float(loss.detach()),'grad_norm':float(norm),'wall_seconds':time.perf_counter()-elapsed}
            trace.append(row);write_json(out/'TRACE.json',trace,replace=(out/'TRACE.json').exists());print(row,flush=True)
        if (step+1)%args.checkpoint_every==0 or step+1==args.steps:
            atomic_torch(out/f'step{step+1:08d}.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'config':cfg,'step':step+1,'rng':rng_state(),'batch_rng':gen.get_state(),'draw_queue':order})
    final=out/f'step{args.steps:08d}.pt'
    write_json(out/'COMPLETE.json',{'step':args.steps,'state_sha256':model_hash(model),'checkpoint':final.name,'checkpoint_sha256':file_hash(final),'closed_set_only':True,'scientific_quality_pass':False,'note':'Completion is not evidence of fitted distribution or held-out quality.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['ae','fm'],required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--ae');p.add_argument('--resume');p.add_argument('--seed',type=int,default=10);p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=1);p.add_argument('--steps',type=int,required=True);p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=6e-4);p.add_argument('--checkpoint-every',type=int,default=400);p.add_argument('--log-every',type=int,default=100);a=p.parse_args();run(a)
if __name__=='__main__':main()
