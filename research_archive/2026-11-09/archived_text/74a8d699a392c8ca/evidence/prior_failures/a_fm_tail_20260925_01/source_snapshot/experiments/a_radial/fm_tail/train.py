"""Fixed64k latent-FM continuation; original8k experiment and AE stay immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from ..models import ShapeAE,ShapeFM
from ..io_utils import new_dir,write_json,atomic_torch,file_hash,array_hash,save_npz,restore_rng
from ..system_probe.official import configure,exact_same,require
from ..system_probe.bayes_review import make_probe,measure,probe_hash

SCOPE='INDEPENDENT_FM_MINLR_CONTINUATION_CLOSED16'
START=8000
FINAL=64000
LR=1e-4
OBSERVATIONS=[8000,16000,32000,64000]


def capture_rng(device):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy_name':n[0],'numpy_keys':torch.tensor(n[1].astype(np.int64)),
        'numpy_pos':n[2],'numpy_has_gauss':n[3],'numpy_cached':n[4],'torch_cpu':torch.get_rng_state(),
        'torch_cuda':torch.cuda.get_rng_state_all() if torch.device(device).type=='cuda' else []}


def restore_on_device(state,device):
    if torch.device(device).type=='cpu':require(not state['torch_cuda'],'Do not relabel a GPU continuation as a CPU run')
    restore_rng(state)


def transition_scheduler(optimizer,step):
    """Explicit new scheduler with inherited update count and correctly reset bases."""
    require(step>=0,'Negative scheduler step')
    for group in optimizer.param_groups:
        group['lr']=LR;group['initial_lr']=LR
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.,last_epoch=step-1)
    require(scheduler.last_epoch==step,'Constant scheduler counter reset')
    require(scheduler.base_lrs==[LR]*len(optimizer.param_groups),'Constant scheduler inherited old1e-3 base')
    require(all(g['lr']==g['initial_lr']==LR for g in optimizer.param_groups),'Optimizer LR changed during scheduler initialization')
    return scheduler


def copy_sources(repo):
    names=['experiments/a_radial/fm_tail/'+n for n in ['__init__.py','train.py','test_train.py','smoke.py']]
    names += ['experiments/a_radial/models.py','experiments/a_radial/train_ours.py','experiments/a_radial/io_utils.py',
        'experiments/a_radial/system_probe/bayes_review.py','experiments/a_radial/system_probe/official.py']
    return {n:file_hash(repo/n) for n in names}


def validate_parent(parent,seed,ae_sha,data_hashes):
    require(parent['step']==START and parent['scheduler']['last_epoch']==START,'Parent is not fixed FM8000')
    cfg=parent['config']
    expected={'stage':'fm','steps':START,'batch':128,'seed':seed,'lr':1e-3,'ae_sha256':ae_sha,
        'dataset_sha256':data_hashes['radial_train.npz'],'dataset_manifest_sha256':data_hashes['DATASET.json']}
    for key,value in expected.items():require(cfg[key]==value,'Parent config changed: '+key)
    require(len(parent['draw_queue'])==0,'Expected full16-cycle parent cursor at batch128 boundary')
    require(parent['rng']['torch_cuda'],'Source parent was not the declared CUDA run')
    require(len(parent['optimizer']['state'])>0 and all(int(v['step'])==START for v in parent['optimizer']['state'].values()),'Incomplete parent Adam counts')
    for group in parent['optimizer']['param_groups']:
        require(abs(group['lr']-LR)<1e-15 and group['initial_lr']==1e-3,'Parent did not reach declared cosine minimum')
        require(group['betas']==(.9,.999) and group['weight_decay']==1e-4,'Parent AdamW changed')
    require(parent['scheduler']['T_max']==START and parent['scheduler']['eta_min']==LR,'Parent scheduler changed')
    model=ShapeFM();model.load_state_dict(parent['model'],strict=True)
    require(sum(p.numel() for p in model.parameters())==50240,'ShapeFM architecture changed')


def prepare(source,out):
    source=Path(source).resolve();repo=Path(__file__).resolve().parents[3]
    require(not torch.cuda.is_initialized(),'Preparation is CPU-only')
    p=json.loads((source/'RUN_PROTOCOL.json').read_text());complete=json.loads((source/'TRAINING_COMPLETE.json').read_text())
    root=repo/'experiments/a_radial'
    require(p['seeds']==[10,11] and p['fm']['steps']==START and p['ae']['steps']==3200,'Unexpected source experiment')
    require(all(file_hash(root/n)==h for n,h in p['training_source_sha256'].items()),'Frozen source experiment code changed')
    require(all(file_hash(Path(p['data'])/n)==h for n,h in p['data_hashes'].items()),'Source data changed')
    require(all(file_hash(source/x['final_checkpoint'])==x['final_sha256'] for x in complete['stages']),'Source final checkpoint changed')
    out=new_dir(out).resolve();sources=copy_sources(repo);manifest={};jobs={}
    with np.load(Path(p['data'])/'radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr']);uids=z['uids'].tolist()
    def copy_one(src,relative):
        dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        manifest[relative]={'source':str(src),'sha256':file_hash(src),'copied_sha256':file_hash(dst)}
        require(manifest[relative]['sha256']==manifest[relative]['copied_sha256'],'Copy changed bytes')
    copy_one(Path(p['data'])/'radial_train.npz','data/radial_train.npz')
    copy_one(Path(p['data'])/'DATASET.json','data/DATASET.json')
    copy_one(source/'RUN_PROTOCOL.json','initial/source_RUN_PROTOCOL.json')
    for seed in p['seeds']:
        ap=source/f'seed{seed}/ae/step00003200.pt';fp=source/f'seed{seed}/fm/step00008000.pt'
        ac=torch.load(ap,map_location='cpu',weights_only=True);fc=torch.load(fp,map_location='cpu',weights_only=True)
        require(ac['step']==3200,'AE is not frozen final3200');validate_parent(fc,seed,file_hash(ap),p['data_hashes'])
        copy_one(ap,f'initial/seed_{seed}_ae3200.pt');copy_one(fp,f'initial/seed_{seed}_fm8000.pt')
        copy_one(source/f'seed{seed}/fm/TRACE.json',f'initial/seed_{seed}_source_TRACE.json')
        probe=make_probe(ac['model'],fc['model'],y,seed)
        probe_path=f'probes/seed_{seed}_fixed4096.npz';save_npz(out/probe_path,**{k:v.numpy() for k,v in probe.items()})
        jobs[str(seed)]={'seed':seed,'ae_checkpoint':f'initial/seed_{seed}_ae3200.pt','ae_sha256':file_hash(ap),
            'parent_checkpoint':f'initial/seed_{seed}_fm8000.pt','parent_sha256':file_hash(fp),
            'parent_trace':f'initial/seed_{seed}_source_TRACE.json','source_steps':START,'final_step':FINAL,
            'fixed_probe':probe_path,'fixed_probe_file_sha256':file_hash(out/probe_path),'fixed_probe_array_sha256':probe_hash(probe)}
    write_json(out/'DATA_MANIFEST.json',manifest);write_json(out/'SOURCE_MANIFEST.json',sources)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    protocol={'scope':SCOPE,'repository':str(repo),'source_run':str(source),'source_protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),
        'seeds':[10,11],'parent_uids':uids,'all16_training_support_not_holdout':True,'jobs':jobs,
        'source_data_hashes':p['data_hashes'],'data_manifest_sha256':file_hash(out/'DATA_MANIFEST.json'),
        'source_manifest_sha256':file_hash(out/'SOURCE_MANIFEST.json'),
        'training':{'source_step':START,'final_step':FINAL,'new_updates_per_seed':FINAL-START,'batch':128,
            'AdamW':'complete inherited moments and update counters','weight_decay':1e-4,'betas':[.9,.999],
            'lr':LR,'scheduler':'new constant multiplier1; inherited counter8000; initial_lr and base_lrs explicitly1e-4',
            'scheduler_base_lrs':[LR],'optimizer_initial_lr':LR,'clip_grad_norm':2.,'loss':'unchanged independent coupling velocity MSE',
            'progress_every':1000,'checkpoint_every':8000,'observations':OBSERVATIONS,
            'parent_exposures_initial_each':64000,'parent_exposures_final_each':512000},
        'selection':'fixed_final_64000_only; observations never select checkpoints',
        'AE_frozen':True,'FM_mu_sd_frozen':True,'sampler_and_rng':'complete parent state inherited; no reset',
        'probe':'same4096 xt/t/target tensors as old representation diagnostics, immutable CPU archive, analytic conditional-mean comparison',
        'sampling':{'samples_per_seed':64,'noise_seed_by_training_seed':{'10':73510,'11':73511},'primary_euler_steps':32,
            'auxiliary_euler_steps':128,'auxiliary_never_replaces_primary':True,'no_generated_distribution_metric_for_selection':True},
        'TF32':False,'deterministic_algorithms':True,'threads':1,'GPU_gate_required':True,'formal_initialization_from_smoke_forbidden':True}
    write_json(out/'RUN_PROTOCOL.json',protocol);write_json(out/'STATUS.json',{'status':'PREPARED_CPU_TESTS_ONLY_GPU_PENDING','new_optimizer_updates':0})
    return protocol


def contract(run):
    run=Path(run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());repo=Path(p['repository'])
    require(p['scope']==SCOPE and p['training']['final_step']==FINAL and p['training']['lr']==LR,'Wrong continuation protocol')
    require(file_hash(run/'DATA_MANIFEST.json')==p['data_manifest_sha256'],'Data manifest changed')
    require(file_hash(run/'SOURCE_MANIFEST.json')==p['source_manifest_sha256'],'Source manifest changed')
    exact_same(json.loads((run/'SOURCE_MANIFEST.json').read_text()),copy_sources(repo),'continuation_source_hashes')
    for name,record in json.loads((run/'DATA_MANIFEST.json').read_text()).items():require(file_hash(run/name)==record['sha256'],'Copied source changed: '+name)
    for job in p['jobs'].values():require(file_hash(run/job['fixed_probe'])==job['fixed_probe_file_sha256'],'Probe archive changed')
    return p,file_hash(run/'RUN_PROTOCOL.json')


def load_probe(run,job):
    with np.load(run/job['fixed_probe'],allow_pickle=False) as z:probe={k:torch.from_numpy(z[k].copy()) for k in z.files}
    require(probe_hash(probe)==job['fixed_probe_array_sha256'],'Probe arrays changed')
    return probe


class TailSession:
    def __init__(self,parent,targets,config,device,continuation=None):
        self.device=torch.device(device);self.config=deepcopy(config);self.targets=targets.to(self.device).detach()
        require(self.targets.shape==(16,64),'Only frozen16 latent targets')
        self.model=ShapeFM().to(self.device);self.model.load_state_dict(parent['model']);self.model.train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=LR,weight_decay=1e-4)
        self.optimizer.load_state_dict(deepcopy(parent['optimizer']))
        self.step=int(parent['step']);self.scheduler=transition_scheduler(self.optimizer,self.step)
        self.generator=torch.Generator(device='cpu');self.generator.set_state(parent['batch_rng'])
        self.queue=list(parent['draw_queue']);self.trace=[];self.timings=[];self.safe_boundary=True
        self.source_scheduler=deepcopy(parent.get('source_scheduler',parent['scheduler']))
        self.source_config=deepcopy(parent.get('source_config',parent['config']))
        self.initial_norm={'mu':parent['model']['mu'].clone(),'sd':parent['model']['sd'].clone()}
        if continuation is not None:
            exact_same(continuation['config'],self.config,'resume_config')
            require(continuation['scope']==SCOPE and continuation['boundary']['complete_update'],'Unsafe continuation scope/boundary')
            self.scheduler.load_state_dict(continuation['scheduler'])
            self.trace=continuation['trace'];self.timings=continuation['timings']
        restore_on_device(parent['rng'],self.device)
        self.assert_lr()
        require(all(int(v['step'])==self.step for v in self.optimizer.state_dict()['state'].values()),'Imported Adam/global update count mismatch')
        require(not self.queue,'Balanced128 sampler must end on a complete16-parent cycle')
        require(len(self.trace)==self.step-self.config['initial_step'],'Continuation trace length mismatch')
        require(all(row['step']==self.config['initial_step']+i+1 for i,row in enumerate(self.trace)),'Continuation trace order mismatch')
        if 'target_sha256' in self.config:require(array_hash(self.targets.cpu().numpy())==self.config['target_sha256'],'Frozen target changed')

    def assert_lr(self):
        require(self.scheduler.last_epoch==self.step,'Scheduler/global update count changed')
        require(self.scheduler.base_lrs==[LR] and all(g['lr']==g['initial_lr']==LR for g in self.optimizer.param_groups),'ConstantLR unexpectedly reset to source1e-3')

    def update(self):
        require(self.safe_boundary,'Incomplete prior update');self.safe_boundary=False;before=time.perf_counter()
        if self.device.type=='cuda':torch.cuda.synchronize()
        while len(self.queue)<128:self.queue+=torch.randperm(len(self.targets),generator=self.generator).tolist()
        ix=self.queue[:128];self.queue=self.queue[128:];x1=self.targets[ix]
        x0=torch.randn_like(x1);t=torch.rand(len(x1),device=self.device);xt=(1-t[:,None])*x0+t[:,None]*x1
        loss=(self.model(xt,t)-(x1-x0)).square().mean()
        require(bool(torch.isfinite(loss)),'Nonfinite FM loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),2.)
        require(bool(torch.isfinite(norm)),'Nonfinite FM gradient')
        self.optimizer.step();self.scheduler.step();self.step+=1;self.assert_lr()
        row={'step':self.step,'train_batch_loss':float(loss.detach()),'grad_norm':float(norm),'batch_indices':ix,
            'lr':self.optimizer.param_groups[0]['lr'],'scheduler_updates':self.scheduler.last_epoch,
            'parent_exposures_each':self.step*128//16}
        self.trace.append(row)
        if self.device.type=='cuda':torch.cuda.synchronize()
        self.timings.append({'step':self.step,'seconds':time.perf_counter()-before});self.safe_boundary=True
        return row

    def state(self):
        require(self.safe_boundary,'No partial update checkpoint')
        require(torch.equal(self.initial_norm['mu'].cpu(),self.model.mu.cpu()) and torch.equal(self.initial_norm['sd'].cpu(),self.model.sd.cpu()),'FM normalization changed')
        return {'scope':SCOPE,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
            'source_scheduler':self.source_scheduler,'source_config':self.source_config,'config':self.config,'step':self.step,
            'rng':capture_rng(self.device),'batch_rng':self.generator.get_state(),'draw_queue':list(self.queue),
            'trace':self.trace,'timings':self.timings,'parent_exposures_each':self.step*128//16,
            'boundary':{'complete_update':True,'pending_data':False},'is_predeclared_final':self.step==self.config['final_step']}

    def probe(self,probe):
        before=capture_rng(self.device);batch=self.generator.get_state().clone();queue=list(self.queue)
        model=deepcopy(self.model).cpu().eval();result=measure(model,probe);del model
        exact_same(before,capture_rng(self.device),'probe_rng');require(torch.equal(batch,self.generator.get_state()) and queue==self.queue,'Probe changed sampler')
        return {'step':self.step,'selection_role':'observation_only',**result}


def checkpoint_config(p,sha,seed,device,target_sha):
    job=p['jobs'][str(seed)]
    return {'protocol_sha256':sha,'seed':seed,'source_fm_sha256':job['parent_sha256'],'ae_sha256':job['ae_sha256'],
        'data_manifest_sha256':p['data_manifest_sha256'],'source_manifest_sha256':p['source_manifest_sha256'],
        'dataset_sha256':p['source_data_hashes']['radial_train.npz'],
        'dataset_manifest_sha256':p['source_data_hashes']['DATASET.json'],
        'stage':'fm','steps':FINAL,'initial_step':START,'final_step':FINAL,'batch':128,'lr':LR,
        'scope':SCOPE,'role':'closed_set16_not_generalization','device':str(device),'target_sha256':target_sha}


def make_session(run,p,sha,seed,device='cuda',resume=None):
    job=p['jobs'][str(seed)];parent_path=run/job['parent_checkpoint'];ap=run/job['ae_checkpoint']
    parent=torch.load(parent_path,map_location='cpu',weights_only=True);validate_parent(parent,seed,job['ae_sha256'],p['source_data_hashes'])
    with np.load(run/'data/radial_train.npz',allow_pickle=False) as z:y=torch.from_numpy(z['logr']).to(device)
    ac=torch.load(ap,map_location='cpu',weights_only=True);ae=ShapeAE().to(device);ae.load_state_dict(ac['model']);ae.eval().requires_grad_(False)
    fm=ShapeFM().to(device);fm.load_state_dict(parent['model'])
    with torch.no_grad():targets=fm.standardize(ae.encode(y)).detach()
    del ae,fm,ac,y
    config=checkpoint_config(p,sha,seed,device,array_hash(targets.cpu().numpy()))
    if resume is None:return TailSession(parent,targets,config,device)
    state=torch.load(resume,map_location='cpu',weights_only=True)
    require(START<=state['step']<=FINAL,'Resume step outside fixed continuation budget')
    for key in ('mu','sd'):require(torch.equal(state['model'][key],parent['model'][key]),'Frozen FM normalization changed on resume')
    return TailSession(state,targets,config,device,continuation=state)


def train(run,seed,resume=None):
    run=Path(run).resolve();p,sha=contract(run);require(seed in [10,11],'Unknown seed')
    gate=json.loads((run/'GPU_GATE.json').read_text());require(gate['status']=='PASS' and gate['protocol_sha256']==sha,'GPU gate pending or stale')
    require(gate['actual_new_optimizer_updates']==36 and [r['seed'] for r in gate['seeds']]==[10,11],'Incomplete CUDA gate')
    require(gate['source_sha256']==file_hash(Path(__file__).with_name('smoke.py')),'GPU smoke source changed')
    configure(seed);require(torch.cuda.is_available(),'Actual CUDA required')
    root=run/'training'/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True)
    require(not (root/'TRAINING_COMPLETE.json').exists(),'Seed already complete')
    if resume is None:require(not list(root.glob('invocation_*')),'Resume requires explicit complete checkpoint')
    else:
        resume=Path(resume).resolve();require(resume.is_relative_to(root.resolve()),'Resume belongs to another run/seed')
        record=json.loads(resume.with_suffix('.json').read_text());require(record['sha256']==file_hash(resume),'Resume file hash changed')
    index=1
    while (root/f'invocation_{index:03d}').exists():index+=1
    out=new_dir(root/f'invocation_{index:03d}');session=None;started=time.perf_counter();last=None
    def save():
        path=out/f'step{session.step:08d}.pt';atomic_torch(path,session.state())
        write_json(path.with_suffix('.json'),{'step':session.step,'sha256':file_hash(path),'bytes':path.stat().st_size})
        return str(path)
    try:
        session=make_session(run,p,sha,seed,'cuda',resume);probe=load_probe(run,p['jobs'][str(seed)])
        if resume is None:
            last=save();write_json(out/f'probe{START:08d}.json',session.probe(probe))
            write_json(out/'IMPORT_RECORD.json',{'source_checkpoint_sha256':p['jobs'][str(seed)]['parent_sha256'],'parent_step':START,
                'source_scheduler_retained_in_checkpoint':True,'new_scheduler_base_lrs':session.scheduler.base_lrs,
                'optimizer_lr':session.optimizer.param_groups[0]['lr'],'optimizer_initial_lr':session.optimizer.param_groups[0]['initial_lr'],
                'new_scheduler_counter':session.scheduler.last_epoch,'complete_parent_Adam_RNG_sampler_inherited':True})
        else:last=str(resume)
        while session.step<FINAL:
            row=session.update()
            if session.step in OBSERVATIONS:write_json(out/f'probe{session.step:08d}.json',session.probe(probe))
            if session.step%8000==0:last=save()
            if session.step%1000==0:
                progress={'status':'TRAINING_IN_PROGRESS','seed':seed,**row,'last_complete_checkpoint':last,
                    'elapsed_seconds':time.perf_counter()-started,'new_updates':session.step-START,
                    'mean_last1000_update_seconds':float(np.mean([x['seconds'] for x in session.timings[-1000:]]))}
                write_json(out/'progress'/f'step{session.step:08d}.json',progress)
                write_json(root/'PROGRESS.json',progress,replace=(root/'PROGRESS.json').exists());print(json.dumps(progress),flush=True)
        result={'status':'FIXED_BUDGET_COMPLETE_UNQUALIFIED','seed':seed,'final_step':FINAL,'new_updates':FINAL-START,
            'checkpoint':last,'checkpoint_sha256':file_hash(last),'protocol_sha256':sha,'selection':'fixed_final64000_only',
            'parent_exposures_each':FINAL*128//16,'elapsed_this_invocation_seconds':time.perf_counter()-started,
            'source8000_results_unchanged':True,'sampling_and_geometry_evaluation_pending':True}
        write_json(out/'TRAIN_REPORT.json',result);write_json(root/'TRAINING_COMPLETE.json',result);return result
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'last_complete_checkpoint':last,
            'last_completed_step':None if session is None else session.step,'partial_update_checkpoint_written':False})
        raise
    finally:
        del session;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pp=sub.add_parser('prepare');pp.add_argument('--source-run',required=True);pp.add_argument('--out',required=True)
    tt=sub.add_parser('train');tt.add_argument('--run',required=True);tt.add_argument('--seed',type=int,choices=[10,11],required=True);tt.add_argument('--resume')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.source_run,args.out)
    else:train(args.run,args.seed,args.resume)


if __name__=='__main__':main()
