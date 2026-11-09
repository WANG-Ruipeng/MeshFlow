"""Bounded paired continuations; all scientific work is ledgered, no silent replay."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
import argparse, copy, json, signal, time, traceback
from pathlib import Path
from . import common as C

def imports():
    from r2.pilots.bench_four_routes_v1.exposure import training as E
    from r2.pilots.bench_multilod_training_v1 import data as D
    return E,D

def registration_path(arm):
    amendment=C.OUT/'registrations/engineering_amendment_01.json'
    return C.OUT/'registrations'/f'{arm}_v2.json' if arm=='PLAIN' and amendment.exists() else C.OUT/'registrations'/f'{arm}.json'

def ledger_forward(ledger,function,context):
    result,_seconds=ledger.call('train_forward',function,context)
    return result

def register_arm(arm):
    if arm not in C.UPDATES: raise ValueError(arm)
    C.read(C.REGISTRATION)
    files=[Path(__file__),Path(C.__file__)]
    if arm=='LOOP':
        from . import loop
        files.append(Path(loop.__file__))
    if arm=='PATCH':
        from . import patch
        files.append(Path(patch.__file__))
        gate=C.read(C.OUT/'data/patch_gate.json')
        if gate['status']!='PASS': raise RuntimeError('PATCH source representation gate did not pass')
        files.append(C.OUT/'data/patch_gate.json')
        files.extend(C.OUT.glob('data/patch_sources/*.json'))
    inherited=C.read(C.PRIOR/'training_registration.json')['training_code_hashes']
    C.verify(inherited)
    protected={**inherited,**{str(p):C.sha(p) for p in files}}
    for p in (C.REGISTRATION,C.OUT/'data/manifest.json',C.OUT/'data/schedule.json',C.OUT/'data/coefficients.json'):
        protected[str(p)]=C.sha(p)
    reg=dict(status='FROZEN',arm=arm,updates=C.UPDATES[arm],batch=8,microbatch=1,
        source=str(C.SOURCE),source_sha256=C.sha(C.SOURCE),protected=protected,
        limits=dict(train_forward=8*C.UPDATES[arm],train_backward=8*C.UPDATES[arm],optimizer_update=C.UPDATES[arm]),
        block_calls_per_forward=16 if arm=='LOOP' else 12,
        FM='0.5 FM1 + 0.5 FM2' if arm=='LOOP' else 'unique-coordinate weighted FM' if arm=='PATCH' else 'original free-only FM',
        equal_parent_noise_time_stream=True,original_reference_group_MSE_weights=True,
        endpoint_selection='fixed200; PLAIN additionally267 for approximate equal block computation',
        replay_allowance=0,safe_resume_at_exact_checkpoint_allowed=True)
    if arm=='PLAIN' and (C.OUT/'registrations/engineering_amendment_01.json').exists():
        amendment=C.read(C.OUT/'registrations/engineering_amendment_01.json')
        if amendment['extra_generator_forwards']!=1 or amendment['optimizer_updates_completed']!=0: raise ValueError('Wrong engineering amendment')
        reg['limits']['train_forward']+=1
        reg['engineering_amendment']=amendment
        reg['protected'][str(C.OUT/'registrations/engineering_amendment_01.json')]=C.sha(C.OUT/'registrations/engineering_amendment_01.json')
    C.write(registration_path(arm),reg)
    return reg

def counts(ledger,step):
    desired=dict(train_forward=8*step,train_backward=8*step,optimizer_update=step)
    desired['train_forward']+=ledger.limits['train_forward']-8*ledger.limits['optimizer_update']
    if ledger.pending or ledger.failed or any(ledger.attempted[k]!=n or ledger.successful[k]!=n for k,n in desired.items()):
        raise RuntimeError('Ledger is not the exact completed-update cursor; no rollback/replay')

def train(arm,resume=False,engineering_resume=False):
    regpath=registration_path(arm)
    reg=C.read(regpath); C.verify(reg['protected'])
    if C.sha(C.SOURCE)!=reg['source_sha256']: raise RuntimeError('Source changed')
    out=C.OUT/'training'/arm; out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    prev=C.read(status) if status.exists() else None
    if prev and prev['status']=='COMPLETE':
        C.verify({prev['checkpoint']['path']:prev['checkpoint']['sha256']}); return prev
    if engineering_resume:
        if arm!='PLAIN' or not reg.get('engineering_amendment') or not prev or prev['status']!='FAILED' or prev['step']!=0:
            raise RuntimeError('Only the recorded pre-update engineering failure may recover')
        amendment=reg['engineering_amendment']
        if C.sha(status)!=amendment['failed_status_sha256']:raise RuntimeError('Failed status changed before explicit engineering recovery')
        C.write(out/'failed_status_engineering_01.json',prev)
        prev=None
    if prev and (not resume or prev['status']!='INTERRUPTED_SAFE'):
        raise RuntimeError('Unresolved existing attempt; explicit engineering review required')
    if resume and not prev: raise RuntimeError('No safe checkpoint exists')
    go=C.read(C.OUT/'dispatch'/f'{regpath.stem}_go.json')
    if not go.get('authorized') or go['registration_sha256']!=C.sha(regpath): raise RuntimeError('Missing resource dispatch GO')
    resource=C.register_resource('TRAIN_DISPATCH_'+arm)
    if resource['free_mib']<go['minimum_free_mib']: raise RuntimeError('RESOURCE_WAIT before model load')
    E,D=imports(); T,torch,np=E.T,E.torch,E.np
    T.install_io(C.OUT,arm)
    step=prev['step'] if prev else 0
    attempt='attempt_'+str(time.time_ns()); adir=out/attempt; adir.mkdir()
    ledger=T.historical.CallLedger(out/'calls.jsonl',reg['limits'],arm,attempt); counts(ledger,step)
    observer=T.historical.GPUObserver(C.EXPERIMENT+'_'+arm,interval=2.)
    state=dict(status='LOADING',arm=arm,step=step,target=C.UPDATES[arm],pid=os.getpid(),attempt=attempt)
    stop={'signal':None}
    def handler(sig,frame): stop['signal']=sig
    handlers={sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
    started=time.perf_counter(); C.write(status,state,replace=True)
    try:
        with T.historical.exclusive_writer(out),observer:
            E.runtime.configure_stable_runtime(); torch.set_num_threads(2)
            source=C.resolve(prev['checkpoint']['path']) if prev else C.SOURCE
            p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
            if not prev:
                original=C.read(C.PRIOR/'training_registration.json')
                if p['schema']!='bench_four_routes_exposure_training_v1' or p['global_step']!=C.BASE_STEP: raise ValueError('Wrong EXPOSURE source')
                for field,hfield,fn in [('generator_state','source_generator_sha256',E.state_hash),('optimizer','source_optimizer_sha256',T.base.tree_hash),('rng','source_rng_sha256',T.base.tree_hash)]:
                    if fn(p[field])!=original[hfield]: raise ValueError('Source state identity '+field)
            else:
                if p['arm']!=arm or p['local_step']!=step or p['registration_sha256']!=C.sha(regpath) or p['execution']!=ledger.snapshot(): raise ValueError('Resume identity/cursor differs')
            model,optimizer=E._build_restore(p,C.BASE_STEP+step,'cuda')
            if arm=='LOOP':
                from .loop import LoopNative
                model.__class__=LoopNative; model.set_loop_rounds(2)
            E.restore_rng(p['rng'])
            head_state=p['head_state']
            del p
            manifest=C.read(C.OUT/'data/manifest.json')
            cases={r['case_id']:D.load_case(r) for r in manifest['train']}
            steps=C.read(C.OUT/'data/schedule.json')['steps'][:C.UPDATES[arm]]
            coefs=C.read(C.OUT/'data/coefficients.json')
            if arm=='PATCH':
                from . import patch
            state['status']='TRAINING'
            def save(immutable=False):
                counts(ledger,step)
                names=E.validate_optimizer(model,optimizer,C.BASE_STEP+step)
                g,o,rng=E._cpu_copy(model.state_dict()),E._cpu_copy(optimizer.state_dict()),E._cpu_copy(E.rng_state())
                payload=dict(schema='bench_three_mechanisms_training_v1',experiment=C.EXPERIMENT,arm=arm,
                    stage='NATIVE',local_step=step,stage_step=step,global_step=C.BASE_STEP+step,native_total_step=C.NATIVE_BASE+step,seam_total_step=0,
                    model_config=copy.deepcopy(E.MODEL_CONFIG),runtime_version=E.RUNTIME_VERSION,EMA=False,update_in_progress=False,
                    generator_state=g,generator_state_sha256=E.state_hash(g),optimizer=o,optimizer_state_sha256=T.base.tree_hash(o),
                    optimizer_parameter_names=names,rng=rng,rng_sha256=T.base.tree_hash(rng),head_state=head_state,
                    head_state_sha256=E.HEAD_STATE_SHA256,head_config=E.HEAD_CONFIG,head_step=0,head_calls=0,head_optimizer=None,
                    registration_sha256=C.sha(regpath),execution=ledger.snapshot(),source_checkpoint_sha256=reg['source_sha256'])
                name=f'checkpoint_step{step}.pt' if immutable else 'checkpoint_latest.pt'
                receipt=E._save(out/name,payload,replace=not immutable)
                state['checkpoint']=receipt
                if immutable:
                    export={k:v for k,v in payload.items() if k not in ('optimizer','rng','optimizer_parameter_names','execution')}
                    export.update(schema='bench_three_mechanisms_generator_v1',state_sha256=payload['generator_state_sha256'],
                        model_arm=arm,representation='SHARED_DOUBLE_TRIANGLE_PATCH' if arm=='PATCH' else 'NATIVE',
                        training_checkpoint_sha256=receipt['sha256'])
                    er=E._save(out/f'generator_step{step}.pt',export)
                    state.setdefault('exports',{})[str(step)]=er
                state.update(step=step,global_step=C.BASE_STEP+step,calls=ledger.snapshot())
                C.write(status,state,replace=True)
                C.event(out/'checkpoint_history.jsonl',dict(step=step,receipt=receipt,immutable=immutable))

            for entry,coefficient in zip(steps[step:],coefs[step:]):
                if stop['signal'] is not None:
                    save(); state.update(status='INTERRUPTED_SAFE',signal=stop['signal']); break
                tick=time.perf_counter(); current=entry['stage_step']
                if current%25==1:
                    rr=C.register_resource('TRAIN_'+arm+'_BEFORE_'+str(current))
                    if rr['free_mib']<go['headroom_mib']:
                        save(); state.update(status='INTERRUPTED_SAFE',reason='resource_headroom'); break
                samples=[patch.make_sample(cases[r['case_id']],r) if arm=='PATCH' else E.sample_for_training(cases[r['case_id']],r,'PLAIN') for r in entry['samples']]
                if arm!='PATCH':
                    for s in samples:E.verify_sample(s,'NATIVE')
                else:
                    for s in samples:patch.verify_sample(s)
                ds=[int(s['free_coordinate_count']) for s in samples]; denom=sum(ds)
                groups=[r['replay_group'] for r in entry['samples']]
                weights=coefficient['reference_group_mse_weights']
                mass={g:sum(d for d,gg in zip(ds,groups) if g==gg) for g in weights}
                gains=[weights[g]*denom/mass[g] for g in groups]
                input_seconds=time.perf_counter()-tick
                optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
                sums=dict(objective=0.,FM_last=0.,FM_first=0.)
                for mi,(sample,row,gain) in enumerate(zip(samples,entry['samples'],gains)):
                    context=dict(arm=arm,step=current,microbatch=mi,condition_id=row['condition_id'])
                    b=T.tensors(sample)
                    with E.native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
                        if arm=='LOOP':
                            from .loop import deep_supervision_loss
                            v1,v2=ledger_forward(ledger,lambda:model.forward_train(b['x'],b['t'],b['y'],valid_mask=b['valid'],known_mask=b['known']),context)
                            base,first,last=deep_supervision_loss(v1.float(),v2.float(),b['target'],b['valid'],b['fixed'],denominator=denom)
                            pred=v2.float()
                        else:
                            pred=ledger_forward(ledger,lambda:model(b['x'],b['t'],b['y'],valid_mask=b['valid'],known_mask=b['known']).float(),context)
                            if arm=='PATCH':
                                pred=patch.project_velocity(pred,sample)
                                w=torch.as_tensor(sample['corner_loss_weights'],device=pred.device,dtype=torch.float32)[None,:,:,None]
                                base=((pred.reshape_as(b['target'].reshape(1,-1,3,3))-b['target'].reshape(1,-1,3,3)).square()*w).sum()/denom
                            else: base=E.corner_masked_fm_loss(pred,b['target'],b['valid'],b['fixed'],denominator=denom)
                            first=last=base
                        loss=base*gain
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite FM')
                        ledger.call('train_backward',loss.backward,context)
                    sums['objective']+=float(loss.detach()); sums['FM_last']+=float(last.detach())*gain; sums['FM_first']+=float(first.detach())*gain
                    C.event(out/'exposure.jsonl',dict(**context,parent_id=row['parent_id'],N=int(sample['N']),K=int(sample['K']),
                        t=float(sample['t']),replay_group=row['replay_group'],gain=gain,independent_coordinate_count=ds[mi],
                        loss=float(loss.detach()),noise_seed=row['noise_seed'],epsilon_seed=row['epsilon_seed'],time_seed=row['time_seed'],
                        permutation_seed=row['permutation_seed'],alpha=row['alpha'],patch_count=int(sample.get('patch_count',0))))
                    del pred,base,first,last,loss,b
                    if arm=='LOOP':del v1,v2
                if any(p.grad is None for p in model.parameters()):raise RuntimeError('Unused inherited parameter')
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                ledger.call('optimizer_update',optimizer.step,dict(arm=arm,step=current))
                step=current; E.validate_optimizer(model,optimizer,C.BASE_STEP+step)
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
                record=dict(arm=arm,step=step,global_step=C.BASE_STEP+step,**sums,gradient_norm=grad,denominator=denom,
                    reference_group_weights=weights,input_seconds=input_seconds,update_seconds=time.perf_counter()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='one formal update including forward/backward/AdamW; not whole-process VRAM',block_calls=reg['block_calls_per_forward']*8)
                C.event(out/'training_log.jsonl',record); state.update(step=step,last=record,elapsed_seconds=time.perf_counter()-started)
                if step==200 or step==C.UPDATES[arm]:save(immutable=True)
                elif step%50==0:save()
                C.write(status,state,replace=True)
                if step in (1,5,25) or step%50==0:
                    C.write(out/f'resources_step{step}.json',C.resources())
                    print(json.dumps(record),flush=True)
            else:state['status']='COMPLETE'
            counts(ledger,step); C.verify(reg['protected'])
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),no_automatic_replay=True)
        C.write(adir/'failure.json',state)
        raise
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)
        state.update(step=step,elapsed_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
        C.write(status,state,replace=True);C.write(adir/'final_status.json',state)
        C.write(adir/'resource_observation.json',dict(samples=observer.samples,summary=observer.summary()))
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','train']);p.add_argument('--arm',choices=list(C.UPDATES),required=True);p.add_argument('--resume',action='store_true');p.add_argument('--engineering-resume',action='store_true');a=p.parse_args()
    r=register_arm(a.arm) if a.action=='register' else train(a.arm,a.resume,a.engineering_resume)
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','calls','resources')},ensure_ascii=False))
