"""Paired 1000-update continuations with an explicit privileged layout branch."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse, copy, signal, time, traceback
from pathlib import Path
from . import common as C

class BaseParameters:
    """Read-only optimizer view retaining original parameter names/order."""
    def __init__(self,model): self.model=model
    @property
    def training(self):return self.model.training
    readout_mode='none'
    def named_parameters(self):
        return ((n,p) for n,p in self.model.named_parameters() if not n.startswith('layout_branch.'))
    def parameters(self):return (p for _,p in self.named_parameters())

def register(arm):
    if arm not in C.UPDATES:raise ValueError(arm)
    C.read(C.REGISTRATION)
    inherited=C.read(C.PRIOR/'training_registration.json')['training_code_hashes']
    C.verify(inherited)
    own=[Path(__file__),Path(C.__file__),Path(__file__).with_name('data.py')]
    if arm=='LAYOUT':own.append(Path(__file__).with_name('model.py'))
    own += [C.REGISTRATION,C.OUT/'data/manifest.json',C.OUT/'data/schedule.json',C.OUT/'data/coefficients.json',C.OUT/'data/layout_records.json']
    layout_sources={r['layout_npz']:r['layout_npz_sha256'] for r in C.read(C.OUT/'data/layout_records.json')['records']}
    C.verify(layout_sources)
    reg=dict(status='FROZEN',experiment=C.EXPERIMENT,arm=arm,updates=1000,batch=8,microbatch=1,
        source=str(C.SOURCE),source_sha256=C.sha(C.SOURCE),
        protected={**inherited,**layout_sources,**{str(p):C.sha(p) for p in own}},
        limits=dict(train_forward=8000,train_backward=8000,optimizer_update=1000),
        layout_dropout=.2,layout_information='SOURCE_PRIVILEGED_COARSE_REGIONS_NOT_C_ONLY',
        base_optimizer='Inherited matching EXPOSURE AdamW',layout_optimizer='Fresh AdamW same lr/betas/eps, separate local cursor',
        endpoint_selection='fixed1000',replay_allowance=0)
    C.write(C.OUT/f'registrations/{arm}.json',reg)
    return reg

def require_cursor(ledger,step):
    if type(step) is not int or not 0<=step<=1000:raise ValueError('Step outside fixed budget')
    if ledger.limits!=dict(train_forward=8000,train_backward=8000,optimizer_update=1000):raise ValueError('Ledger limits changed')
    expected=dict(train_forward=step*8,train_backward=step*8,optimizer_update=step)
    if ledger.pending or ledger.failed or any(ledger.attempted[k]!=v or ledger.successful[k]!=v for k,v in expected.items()):
        raise RuntimeError('Incomplete scientific cursor; replay is not authorized')
    starts=[r for r in ledger.rows if r['event']=='CALL_BEGIN']
    for kind in expected:
        actual=[(r['context']['step'],r['context'].get('microbatch')) for r in starts if r['kind']==kind]
        wanted=[(s,m) for s in range(1,step+1) for m in (range(8) if kind!='optimizer_update' else (None,))]
        if actual!=wanted:raise RuntimeError('Scientific call sequence differs from complete cursor')

def validate_stream(entries,coefficients):
    if len(entries)!=1000 or len(coefficients)!=1000:raise ValueError('Exactly1000 paired updates required')
    if [e['stage_step'] for e in entries]!=list(range(1,1001)):raise ValueError('Nonsequential schedule')
    if [c['stage_step'] for c in coefficients]!=list(range(1,1001)):raise ValueError('Nonsequential coefficients')
    for e in entries:
        if len(e['samples'])!=8:raise ValueError('Exactly8 samples per update')
        for start in (0,2,4,6):
            a,b=e['samples'][start:start+2]
            if a['parent_id']!=b['parent_id'] or a['case_id']!=b['case_id'] or a['layout_drop']!=b['layout_drop']:
                raise ValueError('CLEAN/X_WIDTH paired parent/layout dropout changed')

def validate_allocation(coefficient,counts,groups,gains,step):
    if coefficient['stage_step']!=step or coefficient['micro_free_scalar_counts']!=counts or coefficient['micro_groups']!=groups:
        raise ValueError('Actual samples differ from frozen coefficients')
    if coefficient['denominator']!=sum(counts) or any(abs(a-b)>1e-12 for a,b in zip(coefficient['micro_gains'],gains)):
        raise ValueError('Actual FM normalization differs')

def check_layout_optimizer(model,opt,step):
    import torch
    if opt is None:return
    ps=list(model.layout_branch.parameters())
    g=opt.param_groups[0]
    if [id(p) for p in ps]!=[id(p) for p in g['params']]:raise ValueError('Layout parameter identity')
    if (g['lr'],tuple(g['betas']),g['eps'],g['weight_decay'])!=(1e-5,(.9,.95),1e-8,0.):raise ValueError('Layout optimizer contract')
    for p in ps:
        s=opt.state.get(p,{})
        if step==0 and not s:continue
        if set(s)!={'step','exp_avg','exp_avg_sq'} or float(s['step'])!=step:raise ValueError('Layout AdamW cursor')
        if p.dtype!=torch.float32 or any(s[k].shape!=p.shape or s[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq')):raise ValueError('Layout optimizer dtype')

def train(arm,resume=False):
    regpath=C.OUT/f'registrations/{arm}.json';reg=C.read(regpath);C.verify(reg['protected'])
    if C.sha(C.SOURCE)!=reg['source_sha256']:raise ValueError('Source file changed')
    out=C.OUT/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json';prev=C.read(status) if status.exists() else None
    if prev and prev['status']=='COMPLETE':
        C.verify({prev['checkpoint']['path']:prev['checkpoint']['sha256']});return prev
    if prev and (not resume or prev['status']!='INTERRUPTED_SAFE'):raise RuntimeError('Existing attempt requires explicit review')
    if resume and not prev:raise RuntimeError('No safe checkpoint')
    go=C.read(C.OUT/f'dispatch/{arm}_go.json')
    if not go['authorized'] or go['registration_sha256']!=C.sha(regpath):raise ValueError('No matching resource GO')
    rr=C.register_resource('TRAIN_DISPATCH_'+arm)
    if rr['free_mib']<go['minimum_free_mib']:raise RuntimeError('RESOURCE_WAIT')
    from r2.pilots.bench_four_routes_v1.exposure import training as E
    from r2.pilots.bench_multilod_training_v1.data import load_case
    from .data import training_layout
    T,torch=E.T,E.torch
    T.install_io(C.OUT,arm)
    step=prev['step'] if prev else 0
    attempt='attempt_'+str(time.time_ns());adir=out/attempt;adir.mkdir()
    ledger=T.historical.CallLedger(out/'calls.jsonl',reg['limits'],arm,attempt);require_cursor(ledger,step)
    observer=T.historical.GPUObserver(C.EXPERIMENT+'_'+arm,interval=2.)
    state=dict(status='LOADING',arm=arm,step=step,target=1000,pid=os.getpid(),attempt=attempt)
    stop={'signal':None}
    def handler(sig,frame):stop['signal']=sig
    handlers={sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
    started=time.perf_counter();C.write(status,state,replace=True)
    try:
        with T.historical.exclusive_writer(out),observer:
            E.runtime.configure_stable_runtime();torch.set_num_threads(2)
            p=torch.load(C.SOURCE,map_location='cpu',weights_only=True,mmap=True)
            original=C.read(C.PRIOR/'training_registration.json')
            if p['schema']!='bench_four_routes_exposure_training_v1' or p['global_step']!=C.BASE_STEP:raise ValueError('Wrong source')
            for field,hfield,fn in [('generator_state','source_generator_sha256',E.state_hash),('optimizer','source_optimizer_sha256',T.base.tree_hash),('rng','source_rng_sha256',T.base.tree_hash)]:
                if fn(p[field])!=original[hfield]:raise ValueError('Source state '+field)
            model,optimizer=E._build_restore(p,C.BASE_STEP,'cuda')
            layout_opt=None
            if arm=='LAYOUT':
                from .model import attach_layout
                model=attach_layout(model)
                layout_opt=torch.optim.AdamW(model.layout_branch.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)
            baseview=BaseParameters(model)
            if prev:
                C.verify({prev['checkpoint']['path']:prev['checkpoint']['sha256']})
                p=torch.load(C.resolve(prev['checkpoint']['path']),map_location='cpu',weights_only=True,mmap=True)
                if p['arm']!=arm or p['local_step']!=step or p['registration_sha256']!=C.sha(regpath) or p['execution']!=ledger.snapshot():raise ValueError('Resume cursor')
                for field,hfield,fn in [('generator_state','generator_state_sha256',E.state_hash),('optimizer','optimizer_state_sha256',T.base.tree_hash),('rng','rng_sha256',T.base.tree_hash)]:
                    if fn(p[field])!=p[hfield]:raise ValueError('Resume content hash '+field)
                if arm=='LAYOUT' and T.base.tree_hash(p['layout_optimizer'])!=p['layout_optimizer_sha256']:raise ValueError('Resume layout optimizer content')
                model.load_state_dict(p['generator_state'],strict=True);optimizer.load_state_dict(p['optimizer'])
                if layout_opt is not None:layout_opt.load_state_dict(p['layout_optimizer'])
            E.validate_optimizer(baseview,optimizer,C.BASE_STEP+step);check_layout_optimizer(model,layout_opt,step)
            E.restore_rng(p['rng']);head_state=p['head_state'];del p
            manifest=C.read(C.OUT/'data/manifest.json');cases={r['case_id']:load_case(r) for r in manifest['train']}
            entries=C.read(C.OUT/'data/schedule.json')['steps'];coefficients=C.read(C.OUT/'data/coefficients.json')
            validate_stream(entries,coefficients)
            state['status']='TRAINING'
            def save(immutable=False):
                require_cursor(ledger,step)
                names=E.validate_optimizer(baseview,optimizer,C.BASE_STEP+step);check_layout_optimizer(model,layout_opt,step)
                g,o,rng=E._cpu_copy(model.state_dict()),E._cpu_copy(optimizer.state_dict()),E._cpu_copy(E.rng_state())
                lo=None if layout_opt is None else E._cpu_copy(layout_opt.state_dict())
                payload=dict(schema='bench_layout_oracle_training_v1',experiment=C.EXPERIMENT,arm=arm,stage='NATIVE',
                    local_step=step,stage_step=step,global_step=C.BASE_STEP+step,native_total_step=C.NATIVE_BASE+step,seam_total_step=0,
                    model_config=copy.deepcopy(E.MODEL_CONFIG),runtime_version=E.RUNTIME_VERSION if arm=='PLAIN' else model.layout_metadata()['version'],
                    base_runtime_version=E.RUNTIME_VERSION,layout_metadata=None if arm=='PLAIN' else model.layout_metadata(),EMA=False,update_in_progress=False,
                    generator_state=g,generator_state_sha256=E.state_hash(g),optimizer=o,optimizer_state_sha256=T.base.tree_hash(o),
                    layout_optimizer=lo,layout_optimizer_sha256=None if lo is None else T.base.tree_hash(lo),
                    optimizer_parameter_names=names,rng=rng,rng_sha256=T.base.tree_hash(rng),head_state=head_state,
                    head_state_sha256=E.HEAD_STATE_SHA256,head_config=E.HEAD_CONFIG,head_step=0,head_calls=0,head_optimizer=None,
                    registration_sha256=C.sha(regpath),execution=ledger.snapshot(),source_checkpoint_sha256=reg['source_sha256'])
                name=f'checkpoint_step{step}.pt' if immutable else 'checkpoint_latest.pt'
                receipt=E._save(out/name,payload,replace=not immutable);state['checkpoint']=receipt
                if immutable:
                    export={k:v for k,v in payload.items() if k not in ('optimizer','layout_optimizer','rng','optimizer_parameter_names','execution')}
                    export.update(schema='bench_layout_oracle_generator_v1',state_sha256=payload['generator_state_sha256'],model_arm=arm,
                        representation='NATIVE',training_checkpoint_sha256=receipt['sha256'])
                    state['inference_export']=E._save(out/f'generator_step{step}.pt',export)
                state.update(step=step,global_step=C.BASE_STEP+step,calls=ledger.snapshot());C.write(status,state,replace=True)
                C.event(out/'checkpoint_history.jsonl',dict(step=step,receipt=receipt,immutable=immutable))
            for entry,coefficient in zip(entries[step:],coefficients[step:]):
                if stop['signal'] is not None:
                    save();state.update(status='INTERRUPTED_SAFE',signal=stop['signal']);break
                tick=time.perf_counter();current=entry['stage_step']
                if current%25==1:
                    rr=C.register_resource('TRAIN_'+arm+'_BEFORE_'+str(current))
                    if rr['free_mib']<go['headroom_mib']:
                        save();state.update(status='INTERRUPTED_SAFE',reason='resource_headroom');break
                samples=[E.sample_for_training(cases[r['case_id']],r,'PLAIN') for r in entry['samples']]
                for s in samples:E.verify_sample(s,'NATIVE')
                ds=[int(s['free_coordinate_count']) for s in samples];denom=sum(ds)
                groups=[r['replay_group'] for r in entry['samples']];weights=coefficient['reference_group_mse_weights']
                mass={g:sum(d for d,gg in zip(ds,groups) if g==gg) for g in weights}
                gains=[weights[g]*denom/mass[g] for g in groups]
                validate_allocation(coefficient,ds,groups,gains,current)
                input_seconds=time.perf_counter()-tick
                optimizer.zero_grad(set_to_none=True)
                if layout_opt is not None:layout_opt.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats();objective=0.;layout_active=0;layout_diagnostics=[]
                for mi,(sample,row,gain) in enumerate(zip(samples,entry['samples'],gains)):
                    context=dict(arm=arm,step=current,microbatch=mi,condition_id=row['condition_id'])
                    b=T.tensors(sample)
                    if arm=='LAYOUT':
                        tokens,valid=training_layout(cases[row['case_id']],row,sample)
                        model.configure_layout(torch.as_tensor(tokens,device='cuda')[None],torch.as_tensor(valid,device='cuda')[None])
                        layout_active+=int(valid.any())
                    def observe(payload):
                        if payload['site']=='layout_embedding':
                            delta=payload['delta'].detach().float();hidden=payload['original_hidden'].detach().float()
                            layout_diagnostics.append(dict(delta_rms=float(delta.square().mean().sqrt()),
                                hidden_rms=float(hidden.square().mean().sqrt()),
                                actual_changed_scalars=int((payload['hidden'].detach()!=payload['original_hidden'].detach()).sum())))
                    with E.native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
                        pred,_=ledger.call('train_forward',lambda:model(b['x'],b['t'],b['y'],valid_mask=b['valid'],known_mask=b['known'],
                            observer=observe if arm=='LAYOUT' and (current in (1,5,25) or current%50==0) else None).float(),context)
                        base=E.corner_masked_fm_loss(pred,b['target'],b['valid'],b['fixed'],denominator=denom)
                        loss=base*gain
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite FM')
                        ledger.call('train_backward',loss.backward,context)
                    objective+=float(loss.detach())
                    C.event(out/'exposure.jsonl',dict(**context,parent_id=row['parent_id'],N=int(sample['N']),K=int(sample['K']),
                        t=float(sample['t']),replay_group=row['replay_group'],gain=gain,loss=float(loss.detach()),
                        noise_seed=row['noise_seed'],epsilon_seed=row['epsilon_seed'],time_seed=row['time_seed'],
                        permutation_seed=row['permutation_seed'],alpha=row['alpha'],layout_drop=row['layout_drop']))
                    del pred,base,loss,b
                if any(p.grad is None for p in baseview.parameters()):raise RuntimeError('Unused inherited parameter')
                if layout_opt is not None:
                    # Exact no-layout batches still advance a zero-gradient fresh cursor.
                    for p in model.layout_branch.parameters():
                        if p.grad is None:p.grad=torch.zeros_like(p)
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                def update():
                    optimizer.step()
                    if layout_opt is not None:layout_opt.step()
                ledger.call('optimizer_update',update,dict(arm=arm,step=current))
                step=current;E.validate_optimizer(baseview,optimizer,C.BASE_STEP+step);check_layout_optimizer(model,layout_opt,step)
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
                record=dict(arm=arm,step=step,global_step=C.BASE_STEP+step,objective=objective,gradient_norm=grad,denominator=denom,
                    layout_active_instances=layout_active,reference_group_weights=weights,input_seconds=input_seconds,
                    layout_diagnostics=layout_diagnostics,
                    update_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='one formal full update, not process VRAM')
                C.event(out/'training_log.jsonl',record);state.update(step=step,last=record,elapsed_seconds=time.perf_counter()-started)
                if step==1000:save(immutable=True)
                elif step%100==0:save()
                C.write(status,state,replace=True)
                if step in (1,5,25) or step%50==0:
                    C.write(out/f'resources_step{step}.json',C.resources());print(record,flush=True)
            else:state['status']='COMPLETE'
            require_cursor(ledger,step);C.verify(reg['protected'])
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_replay=True)
        C.write(adir/'failure.json',state);raise
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)
        state.update(step=step,elapsed_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
        C.write(status,state,replace=True);C.write(adir/'final_status.json',state)
        C.write(adir/'resource_observation.json',dict(samples=observer.samples,summary=observer.summary()))
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','train']);p.add_argument('--arm',choices=list(C.UPDATES),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(register(a.arm) if a.action=='register' else train(a.arm,a.resume))
