"""Bounded Bench PLAIN/SEAM adaptation from the same official EMA.

No Chair weight is loaded.  The scientific protocol lives in registration.json;
this worker never silently replays a failed or unsaved optimizer update.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse, copy, hashlib, json, random, signal, subprocess, sys, time, traceback
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
from meshflow_control.seam.slot_data import supervised_sample as plain_sample, edge_features
from r2.pilots.seam_abc_v1.slot_data_extension import supervised_sample as seam_sample

ARMS = ('PLAIN','SEAM')
SEED = 20261018
DEFAULT_ROOT = ROOT/'experiments/r2_bench_control_transfer_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def resolve(path):
    p=str(path).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow') if os.name!='nt' else str(path)
    q=Path(p);return q if q.is_absolute() else ROOT/q

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def event(path,kind,**kw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(event=kind,unix=time.time(),**kw),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
        if kind in ('UPDATE_STARTED','UPDATE_COMPLETE','OPTIMIZER_STARTED','OPTIMIZER_COMPLETE','HEAD_OPTIMIZER_STARTED','HEAD_OPTIMIZER_COMPLETE'):
            os.fsync(f.fileno())

def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def resources():
    import psutil
    def query(args):
        r=subprocess.run(args,text=True,capture_output=True);return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    return dict(unix=time.time(),pid=os.getpid(),cpu_logical=psutil.cpu_count(),cpu_percent=psutil.cpu_percent(interval=.1),
        memory_available_bytes=psutil.virtual_memory().available,process_RSS=psutil.Process().memory_info().rss,
        gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']))

def train_records(manifest):
    if 'train' in manifest:return manifest['train']
    return [r for r in manifest['records'] if r['split'].upper()=='TRAIN']

def prepare(root,manifest_path,official,updates=3000):
    """CPU registration only; no neural forward, no CUDA initialization."""
    root=resolve(root);root.mkdir(parents=True,exist_ok=True)
    regpath=root/'registration.json'
    if regpath.exists():return read(regpath)
    manifest_path=resolve(manifest_path);official=resolve(official)
    manifest=read(manifest_path);records=sorted(train_records(manifest),key=lambda r:r['parent_id'])
    if len(records)<8 or len({r['parent_id'] for r in records})!=len(records):raise ValueError('Need independent training parents')
    pos=neg=0;case_hashes={}
    for r in records:
        p=resolve(r['case_path']);c=arrays(p);labels=np.asarray(c['occupied_labels'],bool)
        pos+=int(labels.sum());neg+=int((~labels).sum());case_hashes[str(p)]=sha(p)
    weights=[(pos+neg)/(2*neg),(pos+neg)/(2*pos)] if pos and neg else [1.,1.]
    rng=np.random.RandomState(SEED);queue=[];schedule=[]
    for step in range(1,updates+1):
        while len(queue)<4:queue.extend(rng.permutation(len(records)).tolist())
        chosen=queue[:4];queue=queue[4:];rows=[]
        for j,index in enumerate(chosen):
            for aug in range(2):
                seed=10000000+step*1000+j*10+aug
                rows.append(dict(index=int(index),alpha=1. if aug==0 else float(rng.uniform(.9,1.1)),
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000))
        schedule.append(dict(step=step,samples=rows))
    write(root/'input_schedule.json',dict(schema='bench_shared_parent_random_stream_v1',steps=schedule))
    code={str(p):sha(p) for p in [Path(__file__),ROOT/'r2/pilots/bench_control_transfer_v1/sample.py',
        ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',ROOT/'r2/src/meshflow_control/seam/slot_data.py',
        ROOT/'r2/src/meshflow_control/seam/model_adapter.py',ROOT/'r2/src/meshflow_control/models/native.py']}
    reg=dict(schema='bench_control_transfer_v1',status='FROZEN',category='bench',synset='02828884',arms=list(ARMS),
        updates=updates,batch=8,microbatch=1,lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.,hybrid_lambda=.25,
        checkpoint_steps=[1000,updates],rolling_interval=100,model_seed=SEED,head_seed=20261016,
        head_lr=.001,head_class_weights=weights,head_training_counts=dict(owner=pos,inactive=neg),
        official=str(official),official_sha256=sha(official),initialization='category official EMA; Native/T1+role+Geo; fresh AdamW',
        model_config=MODEL_CONFIG,manifest=str(manifest_path),manifest_sha256=sha(manifest_path),records=records,
        case_hashes=case_hashes,input_schedule_sha256=sha(root/'input_schedule.json'),code_hashes=code,
        eval_policy='fixed final endpoint; saved standard Gaussian Euler50; C-only occupancy logit>=0; no source lookup',
        precision='BF16 forward; FP32 parameters/integration; strict deterministic; TF32 off; SDPA math',
        stopping='exact budget; no automatic retry, checkpoint selection, extra arm, loss, default switch or publish')
    write(regpath,reg);return reg

def checked(root):
    reg=read(root/'registration.json')
    if reg['status']!='FROZEN' or reg['arms']!=list(ARMS) or reg['batch']!=8 or reg['microbatch']!=1:raise ValueError('Registration differs')
    for p,h in reg['code_hashes'].items():
        if sha(resolve(p))!=h:raise ValueError('Frozen code changed: '+p)
    for p,h in reg['case_hashes'].items():
        if sha(resolve(p))!=h:raise ValueError('Frozen input changed: '+p)
    if sha(resolve(reg['manifest']))!=reg['manifest_sha256'] or sha(root/'input_schedule.json')!=reg['input_schedule_sha256']:raise ValueError('Input schedule changed')
    return reg

def initialize(reg,device='cuda'):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    runtime.configure_stable_runtime()
    path=resolve(reg['official'])
    if sha(path)!=reg['official_sha256']:raise ValueError('Official checkpoint changed')
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if p.get('config',{}).get('model')!=MODEL_CONFIG or '02828884' not in str(p.get('config',{}).get('data',{}).get('data_path','')):
        raise ValueError('Checkpoint config is not the official compatible Bench architecture')
    ema=p['ema'];prefix=[k.startswith('module.') for k in ema]
    if any(prefix) and not all(prefix):raise ValueError('Mixed EMA prefixes')
    state={k[7:] if all(prefix) else k:v for k,v in ema.items()}
    if any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Official EMA invalid')
    model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=True).eval()
    model.backbone.load_state_dict(state,strict=True)
    if torch.count_nonzero(model.role_embedding) or torch.count_nonzero(model.context_encoder.W_out.weight):raise ValueError('Condition initialization differs')
    audit=dict(official_sha256=reg['official_sha256'],official_backbone_sha256=state_hash(state),generator_initial_sha256=state_hash(model.state_dict()),
        generator_parameters=sum(p.numel() for p in model.parameters()),role_nonzero=0,Geo_exit_nonzero=0,fresh_AdamW=True)
    del state,ema,p
    model=model.to(device);optimizer=make_optimizer(model,None);validate_optimizer(model,None,optimizer,'COMMON',0)
    return model,optimizer,audit

def sample_for_training(case,row,arm):
    c={k:v.copy() for k,v in case.items()};alpha=np.float32(row['alpha'])
    for key in ('full_target','pre_ot_full','C_vertices'):
        c[key][...,0]*=alpha
    n=len(c['full_target']);k=len(c['known_ids']);shape=(n-k,3,3)
    z=np.random.RandomState(row['noise_seed']).standard_normal(shape).astype(np.float32)
    eps=np.random.RandomState(row['epsilon_seed']).standard_normal(shape).astype(np.float32)
    normal=float(np.random.RandomState(row['time_seed']).standard_normal());t=float(1/(1+np.exp(-normal)))
    if arm=='SEAM':v=seam_sample(c,z,eps,t,row['permutation_seed'])
    else:v=plain_sample(c,z,eps,t,row['permutation_seed'],'FULL4_C200')
    return v

def save_checkpoint(path,model,opt,head,hopt,step,reg,audit):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    g=_cpu_copy(model.state_dict());payload=dict(schema='bench_control_training_v1',arm=path.parent.name,step=step,updates=reg['updates'],
        generator_state=g,generator_state_sha256=state_hash(g),optimizer=_cpu_copy(opt.state_dict()),rng=_cpu_copy(rng_state()),
        head_state=None if head is None else _cpu_copy(head.state_dict()),head_optimizer=None if hopt is None else _cpu_copy(hopt.state_dict()),
        model_config=MODEL_CONFIG,registration_sha256=sha(path.parents[2]/'registration.json'),initialization=audit,EMA=False)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);write(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),step=step,generator_state_sha256=payload['generator_state_sha256']))
    return payload

def train(root,arm,resume=None):
    root=resolve(root);reg=checked(root)
    if arm not in ARMS:raise ValueError(arm)
    out=root/'training'/arm;out.mkdir(parents=True,exist_ok=True);status_path=out/'status.json';ledger=out/'ledger.jsonl'
    previous=read(status_path) if status_path.exists() else None
    if previous and previous['status']=='COMPLETE':return previous
    if previous and not resume:raise RuntimeError('Existing attempt requires explicit exact-cursor --resume')
    logs=[json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    if logs and (logs[-1]['event']!='UPDATE_COMPLETE' or logs[-1]['step']!=previous['step']):raise RuntimeError('Unresolved update: no silent replay')
    write(out/'resources_before.json',resources());runtime.configure_stable_runtime()
    schedule=read(root/'input_schedule.json')['steps'];cases=[arrays(r['case_path']) for r in reg['records']]
    step=0;model=opt=head=hopt=None;tick=time.perf_counter();stop={'signal':None}
    def interrupted(sig,frame):stop['signal']=sig
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    state=dict(status='LOADING',arm=arm,step=0,pid=os.getpid(),updates=reg['updates'])
    write(status_path,state)
    try:
        if resume:
            p=torch.load(resolve(resume),map_location='cpu',weights_only=True,mmap=True);step=int(p['step'])
            if p['arm']!=arm or step!=previous['step'] or p['registration_sha256']!=sha(root/'registration.json'):raise ValueError('Resume cursor or identity differs')
            model=build_model(MODEL_CONFIG,readout_mode='none',trainable=True).eval();model.load_state_dict(p['generator_state'],strict=True);model.to('cuda')
            opt=make_optimizer(model,None);opt.load_state_dict(p['optimizer']);audit=p['initialization']
        else:model,opt,audit=initialize(reg);p=None
        if arm=='SEAM':
            head=OccupancyHead(24).eval().to('cuda');hopt=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.95),weight_decay=0.)
            if p:head.load_state_dict(p['head_state'],strict=True);hopt.load_state_dict(p['head_optimizer'])
        if p:restore_rng(p['rng']);del p
        write(out/'initialization_audit.json',audit);write(out/'resources_loaded.json',resources())
        validate_optimizer(model,None,opt,'COMMON',step)
        for entry in schedule[step:]:
            current=entry['step']
            if stop['signal'] is not None:
                save_checkpoint(out/'rolling.pt',model,opt,head,hopt,step,reg,audit);state.update(status='INTERRUPTED_SAFE',step=step);break
            begin=time.perf_counter();samples=[sample_for_training(cases[row['index']],row,arm) for row in entry['samples']]
            input_seconds=time.perf_counter()-begin;den=sum(int(s['free_coordinate_count']) for s in samples);eden=sum(len(s['occupied_labels']) for s in samples)
            event(ledger,'UPDATE_STARTED',step=current,backward_budget=8,forward_budget=8,head_budget=8 if head else 0)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True)
            if hopt:hopt.zero_grad(set_to_none=True)
            fm=bce_total=0.;seam_sse=bulk_sse=0.;seam_count=bulk_count=0;batch_start=time.perf_counter()
            for mi,(s,row) in enumerate(zip(samples,entry['samples'])):
                n=int(s['N']);cv=lambda a:torch.as_tensor(np.asarray(a),device='cuda')
                x=cv(s['xt'].reshape(1,n,9));u=cv(s['u'].reshape(1,n,9));t=cv(np.asarray([s['t']],np.float32));y=cv(np.asarray([n],np.int64))
                known=cv(s['known_mask'][None]);valid=cv(s['valid_mask'][None]);fixed=cv((s['anchor_ids']>=0)[None])
                event(ledger,'FORWARD_STARTED',step=current,micro=mi)
                with native_execution(model,coordinates=(x,t),autocast=True):
                    pred=(seam_forward(model,x,t,y,valid,known,fixed) if arm=='SEAM' else model(x,t,y,valid_mask=valid,known_mask=known)).float()
                    loss=corner_masked_fm_loss(pred,u,valid,fixed,denominator=den)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
                    event(ledger,'FORWARD_COMPLETE',step=current,micro=mi)
                    event(ledger,'BACKWARD_STARTED',step=current,micro=mi);loss.backward();event(ledger,'BACKWARD_COMPLETE',step=current,micro=mi)
                fm+=float(loss.detach());err=(pred.detach()-u).reshape(n,3,3).square().cpu().numpy()
                seam=np.sum(s['anchor_ids']>=0,axis=1)==2;bulk=np.sum(s['anchor_ids']>=0,axis=1)==0
                seam_sse+=float(err[seam,2].sum());seam_count+=3*int(seam.sum());bulk_sse+=float(err[bulk].sum());bulk_count+=9*int(bulk.sum())
                if head:
                    features=cv(s['edge_features']);labels=cv(s['occupied_labels']);mask=torch.ones_like(labels,dtype=torch.bool)
                    event(ledger,'HEAD_FORWARD_STARTED',step=current,micro=mi)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=head(features,mask);bce=occupancy_bce_loss(logits,labels,mask,class_weights=reg['head_class_weights'])*len(labels)/eden
                    if not bool(torch.isfinite(bce)):raise FloatingPointError('Nonfinite BCE')
                    event(ledger,'HEAD_FORWARD_COMPLETE',step=current,micro=mi);event(ledger,'HEAD_BACKWARD_STARTED',step=current,micro=mi);bce.backward();event(ledger,'HEAD_BACKWARD_COMPLETE',step=current,micro=mi)
                    bce_total+=float(bce.detach());del features,labels,mask,logits,bce
                event(out/'exposure.jsonl','EXPOSURE',step=current,micro=mi,parent_id=reg['records'][row['index']]['parent_id'],N=n,K=int(s['K']),t=float(s['t']),**row)
                del x,u,t,y,known,valid,fixed,pred,loss
            if any(p.grad is None for p in model.parameters()):raise RuntimeError('Unused generator parameter')
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));event(ledger,'OPTIMIZER_STARTED',step=current);opt.step();event(ledger,'OPTIMIZER_COMPLETE',step=current)
            if hopt:
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);event(ledger,'HEAD_OPTIMIZER_STARTED',step=current);hopt.step();event(ledger,'HEAD_OPTIMIZER_COMPLETE',step=current)
            step=current;validate_optimizer(model,None,opt,'COMMON',step)
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
            result=dict(step=step,FM=fm,occupancy_BCE=bce_total if head else None,generator_gradient_norm=grad,input_seconds=input_seconds,
                update_seconds=time.perf_counter()-batch_start,seam_tip_velocity_MSE=seam_sse/seam_count if seam_count else None,
                bulk_velocity_MSE=bulk_sse/bulk_count if bulk_count else None,free_coordinate_denominator=den,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            event(out/'training_log.jsonl','UPDATE_METRICS',**result);event(ledger,'UPDATE_COMPLETE',step=step)
            state.update(status='TRAINING',step=step,elapsed_seconds=time.perf_counter()-tick,last=result);write(status_path,state)
            if step in reg['checkpoint_steps']:save_checkpoint(out/f'step{step}.pt',model,opt,head,hopt,step,reg,audit)
            if step%reg['rolling_interval']==0 and step!=reg['updates']:save_checkpoint(out/'rolling.pt',model,opt,head,hopt,step,reg,audit)
            if step in (1,5,25) or step%100==0:write(out/f'resources_step{step}.json',resources());print(json.dumps(dict(arm=arm,**result)),flush=True)
        else:state.update(status='COMPLETE',step=step)
        write(status_path,state);return state
    except BaseException as exc:
        state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'BLOCKED',step=step,error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(out/'failure.json',state);write(status_path,state);raise
    finally:write(out/'resources_finished.json',resources())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(DEFAULT_ROOT));ap.add_argument('--arm',choices=ARMS);ap.add_argument('--resume');ap.add_argument('--prepare',action='store_true');ap.add_argument('--manifest');ap.add_argument('--official');ap.add_argument('--updates',type=int,default=3000)
    a=ap.parse_args()
    if a.prepare:prepare(a.root,a.manifest,a.official,a.updates)
    elif a.arm:train(a.root,a.arm,a.resume)
    else:ap.error('--prepare or --arm required')
if __name__=='__main__':main()
