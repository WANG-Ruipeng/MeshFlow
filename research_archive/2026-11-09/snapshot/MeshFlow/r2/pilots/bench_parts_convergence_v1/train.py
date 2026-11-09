"""Registered multipart Bench continuation; exact AdamW cursors and fixed probes.

COMMON starts from official Bench EMA, never a Chair or previous Bench model.
PLAIN and SEAM fork COMMON12000 including matching optimizer and RNG state.
The external controller advances both arms equally; this worker cannot add steps.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse, collections, copy, json, random, signal, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'r2/src'))
import numpy as np
from meshflow_control import runtime
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import build_model, native_execution
from meshflow_control.training.schedule40 import make_optimizer, validate_optimizer
from meshflow_control.training.context_checkpoint import _cpu_copy
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.seam.occupancy import OccupancyHead, occupancy_bce_loss
from r2.pilots.bench_control_transfer_v1.train import sha, resolve, read, write, event, resources, initialize
from . import data

DEFAULT_ROOT = ROOT/'experiments/r2_bench_parts_convergence_v1'

def checked(root):
    reg = read(root/'registration.json')
    if reg['status'] != 'FROZEN' or reg['batch'] != 8 or reg['microbatch'] != 1:
        raise ValueError('Unfrozen or changed protocol')
    for path, expected in reg['training_code_hashes'].items():
        if sha(resolve(path)) != expected: raise ValueError('Training code changed: '+path)
    for key in ('manifest','input_schedule','probe_manifest','evaluation_manifest'):
        if sha(resolve(reg[key])) != reg[key+'_sha256']: raise ValueError('Frozen asset changed: '+key)
    manifest = read(resolve(reg['manifest']))
    for row in manifest['rows']:
        if sha(resolve(row['case_path'])) != row['case_sha256']: raise ValueError('Case changed')
    return reg, manifest

def scalars(error, sample):
    fixed = sample['anchor_ids'] >= 0
    free = ~fixed
    seam = fixed.sum(1) == 2
    bulk = fixed.sum(1) == 0
    return dict(all_free_velocity_sse=float(error[free].sum(dtype=np.float64)),
                all_free_scalar_count=int(free.sum())*3,
                seam_tip_velocity_sse=float(error[seam, 2].sum(dtype=np.float64)),
                seam_tip_scalar_count=int(seam.sum())*3,
                bulk_velocity_sse=float(error[bulk].sum(dtype=np.float64)),
                bulk_scalar_count=int(bulk.sum())*9)

def forward(model, sample, arm):
    n = int(sample['N'])
    cv = lambda value: torch.as_tensor(np.asarray(value), device='cuda')
    x = cv(sample['xt'].reshape(1,n,9)); u = cv(sample['u'].reshape(1,n,9))
    t = cv(np.asarray([sample['t']],np.float32)); y = cv(np.asarray([n],np.int64))
    known = cv(sample['known_mask'][None]); valid = cv(sample['valid_mask'][None])
    fixed = cv((sample['anchor_ids'] >= 0)[None])
    with native_execution(model, coordinates=(x,t), autocast=True):
        pred = (seam_forward(model,x,t,y,valid,known,fixed) if arm=='SEAM'
                else model(x,t,y,valid_mask=valid,known_mask=known)).float()
    return pred, u, valid, fixed

def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()]

def _assert_same_rng(before):
    after = _rng_snapshot()
    if before[0] != after[0] or before[1][0] != after[1][0] or not np.array_equal(before[1][1],after[1][1]) or before[1][2:] != after[1][2:]:
        raise RuntimeError('Probe changed Python/NumPy training RNG')
    if not torch.equal(before[2],after[2]) or len(before[3]) != len(after[3]) or any(not torch.equal(a,b) for a,b in zip(before[3],after[3])):
        raise RuntimeError('Probe changed Torch training RNG')

def prepare_probes(reg, cases, arm):
    manifest = read(resolve(reg['probe_manifest']))
    prepared=[]
    for row in manifest['rows']:
        noise = data.arrays(row['noise_npz'])
        if sha(resolve(row['noise_npz'])) != row['noise_sha256']: raise ValueError('Probe noise changed')
        # Frozen arrays, source targets and layout are supervised diagnostics only.
        z = noise.get('z', noise.get('noise'))
        eps = noise.get('eps', noise.get('epsilon'))
        if z is None or eps is None: raise ValueError('Probe noise keys')
        sample = data.supervised_sample(cases[row['condition_id']], z, eps, row['t'], row['permutation_seed'], arm)
        prepared.append((row,sample))
    if len(prepared) != 96: raise ValueError('Fixed probe count differs')
    return prepared

def fixed_probe(root, out, step, model, head, arm, reg, prepared):
    path = out/'fixed_loss'/f'step{step}.json'
    if path.exists():
        old=read(path)
        if old['generator_state_sha256'] != state_hash(model.state_dict()): raise ValueError('Existing probe model differs')
        return old
    ledger=out/'ledger.jsonl'; before=_rng_snapshot(); start=time.perf_counter()
    event(ledger,'PROBE_STARTED',step=step,generator_forward_budget=96)
    torch.cuda.reset_peak_memory_stats()
    rows=[]; head_rows=[]
    for i,(row,s) in enumerate(prepared):
        event(ledger,'PROBE_FORWARD_STARTED',step=step,probe_id=row['probe_id'])
        with torch.no_grad():
            pred,u,valid,fixed=forward(model,s,arm)
            err=(pred-u).reshape(int(s['N']),3,3).square().cpu().numpy()
        if not np.isfinite(err).all(): raise FloatingPointError('Nonfinite fixed probe')
        item=dict(probe_id=row['probe_id'],condition_id=row['condition_id'],parent_id=row['parent_id'],
                  split=row['group'],part=row['part_type'],time_bin=str(row['t']),t=row['t'],**scalars(err,s))
        rows.append(item)
        event(ledger,'PROBE_FORWARD_COMPLETE',step=step,probe_id=row['probe_id'])
        if head is not None and row['t']==.2 and len(s['occupied_labels']):
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                features=torch.as_tensor(s['edge_features'],device='cuda'); labels=torch.as_tensor(s['occupied_labels'],device='cuda')
                mask=torch.ones_like(labels,dtype=torch.bool);logits=head(features,mask).float()
                bce=occupancy_bce_loss(logits,labels,mask,class_weights=reg['head_class_weights'])
            prediction=(logits>=0).cpu().numpy();target=np.asarray(s['occupied_labels'],bool)
            head_rows.append(dict(condition_id=row['condition_id'],split=row['group'],part=row['part_type'],
                                  edges=len(target),weighted_BCE=float(bce),TP=int((prediction&target).sum()),
                                  FN=int((~prediction&target).sum()),FP=int((prediction&~target).sum()),TN=int((~prediction&~target).sum())))
            del features,labels,mask,logits,bce
        del pred,u,valid,fixed
    def aggregate(items):
        def stat(prefix):return dict(sse=sum(r[prefix+'_velocity_sse'] for r in items),count=sum(r[prefix+'_scalar_count'] for r in items))
        return dict(ALL_free=stat('all_free'),bulk=stat('bulk'),seam_tip=stat('seam_tip'))
    summary={}
    for split in ('TRAIN','DEV'):
        subset=[r for r in rows if r['split']==split];summary[split.lower()]=aggregate(subset)
        summary[split.lower()]['parts']={p:aggregate([r for r in subset if r['part']==p])['ALL_free'] for p in sorted({r['part'] for r in subset})}
    _assert_same_rng(before)
    value=dict(schema='bench_parts_fixed_probe_v1',step=step,arm=arm,
               probe_manifest_sha256=reg['probe_manifest_sha256'],generator_state_sha256=state_hash(model.state_dict()),
               **summary,rows=rows,head_rows=head_rows,wall_seconds=time.perf_counter()-start,
               generator_forwards=96,head_forwards=len(head_rows),backward_calls=0,RNG_unchanged=True,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(path,value);event(ledger,'PROBE_COMPLETE',step=step,generator_forwards=96,head_forwards=len(head_rows))
    return value

def checkpoint(out, model, optimizer, head, hopt, step, head_step, reg, audit, *, named=False):
    g=_cpu_copy(model.state_dict())
    payload=dict(schema='bench_parts_training_v1',arm=out.name,step=step,global_step=step,
        stage_step=step if out.name=='COMMON' else step-12000,head_step=head_step,
        generator_state=g,generator_state_sha256=state_hash(g),optimizer=_cpu_copy(optimizer.state_dict()),
        head_state=None if head is None else _cpu_copy(head.state_dict()),
        head_optimizer=None if hopt is None else _cpu_copy(hopt.state_dict()),
        rng=_cpu_copy(rng_state()),model_config=MODEL_CONFIG,registration_sha256=sha(out.parents[1]/'registration.json'),
        initialization=audit,EMA=False)
    path=out/(f'step{step}.pt' if named else 'rolling.pt');tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    write(path.with_suffix('.json'),dict(path=str(path),step=step,sha256=sha(path),generator_state_sha256=payload['generator_state_sha256']))
    return path

def inference_checkpoint(out,model,head,step,reg):
    path=out/f'generator_step{step}.pt'
    g=_cpu_copy(model.state_dict())
    p=dict(schema='bench_parts_inference_v1',arm='SEAM' if out.name=='SEAM' else 'PLAIN',step=step,
           generator_state=g,generator_state_sha256=state_hash(g),head_state=None if head is None else _cpu_copy(head.state_dict()),
           registration_sha256=sha(out.parents[1]/'registration.json'),model_config=MODEL_CONFIG,EMA=False)
    if path.exists():
        if read(path.with_suffix('.json'))['generator_state_sha256']!=p['generator_state_sha256']: raise ValueError('Existing export differs')
        return
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),generator_state_sha256=p['generator_state_sha256']))

def train(root,arm,target,resume=None,fork=None):
    root=resolve(root);reg,manifest=checked(root)
    if arm not in ('COMMON','PLAIN','SEAM'):raise ValueError(arm)
    if not (0<target<=12000 if arm=='COMMON' else 12000<target<=21000):raise ValueError('Unauthorized target')
    if target%1000:raise ValueError('Dispatch only registered 1000-step boundaries')
    out=root/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json';ledger=out/'ledger.jsonl';previous=read(status) if status.exists() else None
    if previous and not resume:raise RuntimeError('Existing attempt requires explicit exact-cursor resume')
    if resume and not previous:raise RuntimeError('Resume requires own completed cursor')
    if previous and previous['status'] not in ('READY','INTERRUPTED_SAFE'):raise RuntimeError('Unresolved attempt, no automatic retry')
    if resume and fork:raise ValueError('Resume and fork are exclusive')
    if arm!='COMMON' and not (fork or resume):raise ValueError('Arm requires COMMON12000 fork')
    if arm=='COMMON' and fork:raise ValueError('COMMON must start official')
    write(out/f'resources_dispatch_{target}.json',resources())
    cases={r['condition_id']:data.load_case(r) for r in manifest['rows']}
    schedule=read(resolve(reg['input_schedule']))['steps'];trainrows=manifest['train']
    model_arm='SEAM' if arm=='SEAM' else 'PLAIN'
    prepared=prepare_probes(reg,cases,model_arm)
    runtime.configure_stable_runtime();step=0;head_step=0;head=hopt=None
    state=dict(status='LOADING',arm=arm,step=previous['step'] if previous else 0,pid=os.getpid(),target=target)
    write(status,state);start=time.perf_counter();stop={'signal':None}
    def signal_handler(sig,frame):stop['signal']=sig
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal_handler)
    try:
        if resume or fork:
            source=resolve(resume or fork);p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
            if p['registration_sha256']!=sha(root/'registration.json'):raise ValueError('Checkpoint registration differs')
            step=int(p['step'])
            if resume and (p['arm']!=arm or step!=previous['step']):raise ValueError('Resume cursor differs')
            if fork and (p['arm']!='COMMON' or step!=12000):raise ValueError('Fork must be COMMON12000')
            model=build_model(MODEL_CONFIG,readout_mode='none',trainable=True).eval();model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
            opt=make_optimizer(model,None);opt.load_state_dict(p['optimizer']);audit=copy.deepcopy(p['initialization'])
            if fork:audit['fork']=dict(path=str(source),sha256=sha(source),step=step,generator_state_sha256=p['generator_state_sha256'],AdamW_restored=True)
            if arm=='SEAM':
                head=OccupancyHead(24).eval().to('cuda');hopt=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0.)
                if resume:
                    head.load_state_dict(p['head_state'],strict=True);hopt.load_state_dict(p['head_optimizer']);head_step=int(p['head_step'])
            restore_rng(p['rng']);del p
        else:model,opt,audit=initialize(reg)
        if step>=target:raise ValueError('Target must advance cursor')
        validate_optimizer(model,None,opt,'COMMON',step)
        write(out/'initialization_audit.json',audit);write(out/f'resources_loaded_{target}.json',resources())
        fixed_probe(root,out,step,model,head,model_arm,reg,prepared)
        for entry in schedule[step:target]:
            if stop['signal'] is not None:
                checkpoint(out,model,opt,head,hopt,step,head_step,reg,audit)
                state.update(status='INTERRUPTED_SAFE',step=step,signal=stop['signal']);write(status,state);return state
            current=int(entry['step']);begin=time.perf_counter()
            samples=[data.sample_for_training(cases[r['condition_id']],r,model_arm) for r in entry['samples']]
            input_seconds=time.perf_counter()-begin
            den=sum(int(s['free_coordinate_count']) for s in samples);eden=sum(len(s['occupied_labels']) for s in samples)
            event(ledger,'UPDATE_STARTED',step=current,generator_forward_budget=8,generator_backward_budget=8,
                  head_forward_budget=sum(bool(len(s['occupied_labels'])) for s in samples) if head else 0)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            if hopt:hopt.zero_grad(set_to_none=True)
            fm=0.;bce_total=0.;stats=[];batch_start=time.perf_counter()
            for mi,(s,row) in enumerate(zip(samples,entry['samples'])):
                event(ledger,'FORWARD_STARTED',step=current,micro=mi)
                pred,u,valid,fixed=forward(model,s,model_arm)
                loss=corner_masked_fm_loss(pred,u,valid,fixed,denominator=den)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
                event(ledger,'FORWARD_COMPLETE',step=current,micro=mi)
                event(ledger,'BACKWARD_STARTED',step=current,micro=mi);loss.backward();event(ledger,'BACKWARD_COMPLETE',step=current,micro=mi)
                fm+=float(loss.detach());err=(pred.detach()-u).reshape(int(s['N']),3,3).square().cpu().numpy()
                st=scalars(err,s);stats.append(st)
                if head is not None and len(s['occupied_labels']):
                    labels=torch.as_tensor(s['occupied_labels'],device='cuda');features=torch.as_tensor(s['edge_features'],device='cuda')
                    mask=torch.ones_like(labels,dtype=torch.bool)
                    event(ledger,'HEAD_FORWARD_STARTED',step=current,micro=mi)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=head(features,mask);bce=occupancy_bce_loss(logits,labels,mask,class_weights=reg['head_class_weights'])*len(labels)/eden
                    if not bool(torch.isfinite(bce)):raise FloatingPointError('Nonfinite BCE')
                    event(ledger,'HEAD_FORWARD_COMPLETE',step=current,micro=mi)
                    event(ledger,'HEAD_BACKWARD_STARTED',step=current,micro=mi);bce.backward();event(ledger,'HEAD_BACKWARD_COMPLETE',step=current,micro=mi)
                    bce_total+=float(bce.detach());del labels,features,mask,logits,bce
                tm=float(s['t']);time_bin=min(int(tm*5),4)
                event(out/'exposure.jsonl','EXPOSURE',step=current,micro=mi,split='TRAIN',part=row['part_type'],
                      time_bin=f'{time_bin/5:.1f}-{(time_bin+1)/5:.1f}',t=tm,N=int(s['N']),K=int(s['K']),
                      boundary_edges=len(s['occupied_labels']),**row,**st)
                del pred,u,valid,fixed,loss
            if any(p.grad is None for p in model.parameters()):raise RuntimeError('Unused generator parameter')
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            event(ledger,'OPTIMIZER_STARTED',step=current);opt.step();event(ledger,'OPTIMIZER_COMPLETE',step=current)
            if hopt is not None and eden:
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                event(ledger,'HEAD_OPTIMIZER_STARTED',step=current);hopt.step();head_step+=1;event(ledger,'HEAD_OPTIMIZER_COMPLETE',step=current,head_step=head_step)
            step=current;validate_optimizer(model,None,opt,'COMMON',step)
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Updated parameters nonfinite')
            sums={k:sum(s[k] for s in stats) for k in stats[0]}
            result=dict(step=step,stage_step=step if arm=='COMMON' else step-12000,FM=fm,
                        occupancy_BCE=bce_total if head is not None and eden else None,head_step=head_step,
                        generator_gradient_norm=grad,input_seconds=input_seconds,update_seconds=time.perf_counter()-batch_start,
                        free_coordinate_denominator=den,**sums,
                        seam_tip_velocity_MSE=sums['seam_tip_velocity_sse']/sums['seam_tip_scalar_count'] if sums['seam_tip_scalar_count'] else None,
                        bulk_velocity_MSE=sums['bulk_velocity_sse']/sums['bulk_scalar_count'] if sums['bulk_scalar_count'] else None,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            event(out/'training_log.jsonl','UPDATE_METRICS',**result);event(ledger,'UPDATE_COMPLETE',step=step)
            state.update(status='TRAINING',step=step,last=result,elapsed_seconds=time.perf_counter()-start);write(status,state)
            if step%250==0:checkpoint(out,model,opt,head,hopt,step,head_step,reg,audit)
            if step%1000==0:
                fixed_probe(root,out,step,model,head,model_arm,reg,prepared)
                inference_checkpoint(out,model,head,step,reg)
            if step in (1,5,25,12001,12005,12025) or step%100==0:
                write(out/f'resources_step{step}.json',resources());print(json.dumps(dict(arm=arm,**result)),flush=True)
        checkpoint(out,model,opt,head,hopt,step,head_step,reg,audit)
        if step in (12000,15000,21000):checkpoint(out,model,opt,head,hopt,step,head_step,reg,audit,named=True)
        state.update(status='READY',step=step,head_step=head_step,elapsed_seconds=time.perf_counter()-start,
                     checkpoint=str(out/'rolling.pt'),chunk_finished=True)
        write(status,state);event(ledger,'CHUNK_COMPLETE',step=step,target=target)
        return state
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'BLOCKED',step=step,error=repr(exc),
                     traceback=traceback.format_exc(),no_automatic_retry=True)
        write(out/'failure.json',state);write(status,state);raise
    finally:write(out/f'resources_finished_{target}.json',resources())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--arm',choices=['COMMON','PLAIN','SEAM'],required=True)
    p.add_argument('--target',type=int,required=True);p.add_argument('--resume');p.add_argument('--fork');a=p.parse_args()
    train(a.root,a.arm,a.target,a.resume,a.fork)
if __name__=='__main__':main()
