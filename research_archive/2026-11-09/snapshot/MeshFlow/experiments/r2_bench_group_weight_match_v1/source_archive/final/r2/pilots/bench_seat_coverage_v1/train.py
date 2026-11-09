"""Finite Native continuation from retained global18000, with exact recovery.

New data stream, inherited live generator/AdamW/RNG; no head, probe or SEAM
training. A frozen registration and resource dispatch GO are both required.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse, copy, json, signal, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
from meshflow_control import runtime
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash, rng_state, restore_rng
from meshflow_control.models.native import native_execution
from meshflow_control.seam.model_adapter import corner_masked_fm_loss
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import digest, append_event
from r2.pilots.bench_adaptation_strength_v1 import training as T
from r2.pilots.bench_parts_convergence_v1 import data as D
from r2.pilots.bench_full_seat_curriculum_v1.train import _build_restore, _log_cursor, _save, _receipt_matches
from .common import OUT, SOURCE_ROOT, SOURCE, SOURCE_IDENTITIES, EXPERIMENT, STAGE, BASE_STEP, resolve
from . import training_state as S
from . import schedule as Q


def checked():
    reg = T.read(OUT/'training_registration.json')
    if (reg['status'],reg['experiment'],reg['batch'],reg['microbatch'],reg['hybrid_lambda'],reg['model_config']) != ('FROZEN',EXPERIMENT,8,1,.25,MODEL_CONFIG):
        raise ValueError('Frozen Native coverage contract differs')
    S.stage_spec(reg)
    if any(reg[k] != v for k,v in SOURCE_IDENTITIES.items()) or resolve(reg['source_checkpoint']).resolve() != SOURCE.resolve():
        raise ValueError('Only retained Native6000 live/matching optimizer/RNG is allowed')
    for key in ('head_forward_budget','head_optimizer_updates','probe_forward_budget'):
        if reg[key] != 0:
            raise ValueError('Head/probe activity is not authorized')
    if not reg['training_code_hashes']:
        raise ValueError('Frozen source manifest required')
    for p,h in reg['training_code_hashes'].items():
        if digest(resolve(p)) != h:
            raise ValueError('Frozen training source changed: '+p)
    for key in ('manifest','input_schedule','eval_manifest','source_checkpoint'):
        if digest(resolve(reg[key])) != reg[key+'_sha256']:
            raise ValueError('Frozen input changed: '+key)
    for p,h in reg['gate_hashes'].items():
        if digest(resolve(p)) != h:
            raise ValueError('Frozen preparation gate changed: '+p)
    m = T.read(resolve(reg['manifest']))
    schedule = T.read(resolve(reg['input_schedule']))
    Q.validate(schedule,m['train'],reg['updates'])
    train = m['train']; parents = {r['parent_id'] for r in train}
    for r in m.get('rows',train):
        if digest(resolve(r['case_path'])) != r['case_sha256']:
            raise ValueError('Frozen case changed: '+r['condition_id'])
    old = T.read(SOURCE_ROOT/'data/manifest.json')['train']
    original = {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in old}
    replay = {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in train if r['replay_group']=='OLD'}
    if original != replay:
        raise ValueError('All original22 qualified conditions must be replayed unchanged')
    heldout = {r['parent_id'] for r in m.get('dev',[])}
    ev = T.read(resolve(reg['eval_manifest']))
    for r in ev['requests']:
        group = r.get('eval_group',r.get('group',''))
        if group in ('DEV','HOLDOUT','NEW_DEV','INDEPENDENT_DEV','NEW_PARENT_HOLDOUT'):
            heldout.add(r['parent_id'])
    if parents & heldout:
        raise ValueError('Held-out evaluation parent entered training')
    return reg,m,schedule['steps']


def _source_payload(reg):
    p = torch.load(resolve(reg['source_checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    if (p['schema'],p['arm'],p['stage'],p['stage_step'],p['global_step'],p['EMA'],p['update_in_progress']) != ('bench_full_seat_curriculum_training_v1','NATIVE','NATIVE',6000,18000,False,False):
        raise ValueError('Retained Native source identity differs')
    for key,hkey,fn in (('generator_state','source_generator_sha256',state_hash),('optimizer','source_optimizer_sha256',T.base.tree_hash),('rng','source_rng_sha256',T.base.tree_hash)):
        if fn(p[key]) != reg[hkey]:
            raise ValueError('Retained source state differs: '+key)
    if p['model_config'] != MODEL_CONFIG:
        raise ValueError('Source model configuration differs')
    S.validate_head_state(p['head_state'])
    _check_generator(p['generator_state'])
    return p


def train(resume=False):
    T.install_io(OUT,STAGE)
    reg,m,steps = checked()
    spec = S.stage_spec(reg)
    reg_sha = digest(OUT/'training_registration.json')
    go = T.read(OUT/'dispatch/NATIVE_go.json')
    if not go.get('authorized') or go.get('training_registration_sha256') != reg_sha or go.get('stage') != STAGE:
        raise RuntimeError('Parent resource-aware dispatch release missing')
    headroom = max(2048.,float(go.get('retained_headroom_mib',2048.)))
    out = OUT/'training/NATIVE'; out.mkdir(parents=True,exist_ok=True)
    status_path = out/'status.json'
    with T.historical.exclusive_writer(out):
        previous = T.read(status_path) if status_path.exists() else None
        if previous and previous['status']=='COMPLETE':
            if previous['registration_sha256'] != reg_sha or previous['stage_step'] != spec['updates']:
                raise ValueError('Completed branch differs')
            for key in ('checkpoint','inference_export'):
                if digest(resolve(previous[key]['path'])) != previous[key]['sha256']:
                    raise ValueError('Completed artifact changed')
            return previous
        if previous and (not resume or previous['status']!='INTERRUPTED_SAFE'):
            raise RuntimeError('Unresolved attempt; no automatic scientific replay')
        if resume and not previous:
            raise ValueError('No exact saved cursor exists')
        step = int(previous['stage_step']) if previous else 0
        attempt = 'attempt_'+str(time.time_ns()); adir = out/'attempts'/attempt
        adir.mkdir(parents=True,exist_ok=False)
        ledger = T.historical.CallLedger(out/'calls.jsonl',S.limits(reg),STAGE,attempt)
        S.require_cursor(ledger,reg,step); _log_cursor(out,step)
        state = dict(status='LOADING',arm=STAGE,**S.cursor(reg,step),target_local_step=spec['updates'],
            registration_sha256=reg_sha,pid=os.getpid(),attempt=attempt,head_step=0,head_calls=0,probe_calls=0)
        if previous: state['checkpoint'] = previous['checkpoint']
        initial = T.resource(); T.write(adir/'resources_dispatch.json',initial)
        if initial['free_mib'] is None or initial['free_mib'] < headroom:
            raise RuntimeError('RESOURCE_WAIT before model load')
        started = time.perf_counter()
        observer = T.historical.GPUObserver('bench_seat_coverage_NATIVE',interval=2.)
        stop = {'signal':None}
        def handler(sig,unused): stop['signal']=sig
        handlers = {sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
        T.write(status_path,state)
        try:
            with observer:
                torch.set_num_threads(2); runtime.configure_stable_runtime()
                if previous:
                    source = resolve(previous['checkpoint']['path'])
                    if digest(source) != previous['checkpoint']['sha256']:
                        raise ValueError('Resume file changed')
                    p = torch.load(source,map_location='cpu',weights_only=True,mmap=True)
                    S.validate_payload(p,reg,step,reg_sha)
                    if p['execution'] != ledger.snapshot():
                        raise ValueError('Saved execution ledger differs')
                    audit = p['initialization']
                else:
                    p = _source_payload(reg)
                    audit = dict(source_path=str(SOURCE),**SOURCE_IDENTITIES,source_global_step=BASE_STEP,
                        inherited_optimizer=True,inherited_rng=True,fresh_AdamW=False,new_generator_parameters=0,
                        source_stream_resumed=False,new_registered_stream=True,head_used=False,head_calls=0)
                head_state = p['head_state']
                model,optimizer = _build_restore(p,BASE_STEP+step,'cuda')
                restore_rng(p['rng'])
                if T.base.tree_hash(rng_state()) != p['rng_sha256']:
                    raise ValueError('Restored RNG differs')
                del p
                T.write(adir/'initialization.json',audit)
                cases = {r['condition_id']:D.load_case(r) for r in m['train']}

                def checkpoint_save():
                    S.require_cursor(ledger,reg,step); _log_cursor(out,step)
                    position = S.cursor(reg,step)
                    names = S.validate_optimizer(model,optimizer,BASE_STEP+step)
                    g,o,rng = _cpu_copy(model.state_dict()),_cpu_copy(optimizer.state_dict()),_cpu_copy(rng_state())
                    _check_generator(g); S.validate_head_state(head_state)
                    payload = dict(schema=S.SCHEMA,experiment=EXPERIMENT,arm=STAGE,**position,
                        source_global_step=BASE_STEP,model_config=copy.deepcopy(MODEL_CONFIG),EMA=False,
                        update_in_progress=False,generator_state=g,generator_state_sha256=state_hash(g),
                        optimizer=o,optimizer_state_sha256=T.base.tree_hash(o),optimizer_parameter_names=names,
                        rng=rng,rng_sha256=T.base.tree_hash(rng),registration_sha256=reg_sha,
                        head_state=head_state,head_state_sha256=S.HEAD_STATE_SHA256,head_config=S.HEAD_CONFIG,
                        head_optimizer=None,head_step=0,head_calls=0,initialization=audit,execution=ledger.snapshot(),
                        stream_state=dict(batch_index=step,stage_batch_index=step,next_sample_index=8*step))
                    immutable = step in spec['full_steps']
                    dest = out/(f'checkpoint_step{step}.pt' if immutable else 'checkpoint_latest.pt')
                    if immutable and dest.exists():
                        receipt = T.read(dest.with_suffix('.json'))
                        if digest(dest) != receipt['sha256']: raise ValueError('Immutable checkpoint changed')
                        _receipt_matches(receipt,payload)
                    else:
                        receipt = _save(dest,payload,replace=not immutable)
                        append_event(out/'checkpoint_history.jsonl',receipt)
                    state['checkpoint']=receipt
                    if step == spec['updates']:
                        keys=('experiment','arm','stage','stage_step','local_step','curriculum_step','global_step',
                            'native_total_step','seam_total_step','generator_state','generator_state_sha256',
                            'model_config','EMA','head_state','head_state_sha256','head_config','head_step')
                        export={k:payload[k] for k in keys}
                        export.update(schema=S.INFERENCE_SCHEMA,state_sha256=payload['generator_state_sha256'],
                            training_checkpoint_sha256=receipt['sha256'],registration_sha256=reg_sha,
                            model_arm='PLAIN',representation='NATIVE',head_frozen=True,head_used=False)
                        ep=out/f'generator_step{step}.pt'
                        if ep.exists():
                            er=T.read(ep.with_suffix('.json'))
                            if digest(ep)!=er['sha256'] or er['generator_state_sha256']!=payload['generator_state_sha256']:
                                raise ValueError('Immutable inference export differs')
                        else: er=_save(ep,export)
                        state['inference_export']=er
                    state.update(position); T.write(status_path,state)

                state['status']='TRAINING'
                if not previous: checkpoint_save()
                for entry in steps[step:spec['updates']]:
                    if stop['signal'] is not None:
                        checkpoint_save(); state.update(status='INTERRUPTED_SAFE',signal=stop['signal']); break
                    tick=time.perf_counter(); current=entry['stage_step']
                    if current % 25 == 1:
                        resource=T.resource(); append_event(out/'resource_checks.jsonl',dict(stage_step_before=step,**resource))
                        if resource['free_mib'] is None or resource['free_mib'] < headroom:
                            checkpoint_save(); state.update(status='INTERRUPTED_SAFE',reason='VRAM_headroom_below_dispatch_margin'); break
                    samples=[D.sample_for_training(cases[r['condition_id']],r,'PLAIN') for r in entry['samples']]
                    audits=[S.verify_sample(s,'NATIVE') for s in samples]
                    denominator=sum(int(s['free_coordinate_count']) for s in samples)
                    input_seconds=time.perf_counter()-tick; begin=time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
                    fm,statistics=0.,[]
                    for mi,(sample,row,audit_row) in enumerate(zip(samples,entry['samples'],audits)):
                        context=dict(**S.cursor(reg,current),microbatch=mi,condition_id=row['condition_id'])
                        b=T.tensors(sample)
                        with native_execution(model,coordinates=(b['x'],b['t']),autocast=True):
                            call=lambda:model(b['x'],b['t'],b['y'],valid_mask=b['valid'],known_mask=b['known']).float()
                            pred,_=ledger.call('train_forward',call,context)
                            loss=corner_masked_fm_loss(pred,b['target'],b['valid'],b['fixed'],denominator=denominator)
                            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite original FM')
                            ledger.call('train_backward',loss.backward,context)
                        values=S.statistics(pred,b['target'],sample)
                        if not np.isclose(values['all_free_velocity_sse']/denominator,float(loss.detach()),rtol=4e-6,atol=1e-7):
                            raise ValueError('FM whole-batch free-coordinate accounting differs')
                        fm+=float(loss.detach()); statistics.append(values)
                        append_event(out/'exposure.jsonl',{**row,**context,'t':float(sample['t']),'N':int(sample['N']),
                            'K':int(sample['K']),'loss_contribution':float(loss.detach()),**values,'mechanism':audit_row,'head_forward_called':False})
                        del pred,loss,b
                    if any(p.grad is None for p in model.parameters()): raise RuntimeError('Unused generator parameter')
                    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                    ledger.call('optimizer_update',optimizer.step,S.cursor(reg,current)); step=current
                    S.validate_optimizer(model,optimizer,BASE_STEP+step)
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError('Nonfinite parameter')
                    sums={k:sum(r[k] for r in statistics) for k in statistics[0]}
                    result=dict(**S.cursor(reg,step),FM=fm,objective=fm,free_coordinate_denominator=denominator,
                        **sums,generator_gradient_norm=grad,head_step=0,head_calls=0,input_seconds=input_seconds,
                        update_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope='One formal update including backward/inherited AdamW; allocator is not process VRAM')
                    append_event(out/'training_log.jsonl',result)
                    state.update(S.cursor(reg,step),last=result,elapsed_seconds=time.perf_counter()-started)
                    if step%reg['checkpoint_every']==0 or step in spec['full_steps']: checkpoint_save()
                    T.write(status_path,state)
                    if step in (1,5,25) or step%100==0:
                        T.write(out/f'resources_step{step}.json',T.resource()); print(json.dumps(result),flush=True)
                    if step==1:T.write(out/'first_formal_backward.json',dict(status='PASS',no_extra_forward=True,**result))
                else: state['status']='COMPLETE'
                S.require_cursor(ledger,reg,step); _log_cursor(out,step); T.write(status_path,state)
        except BaseException as exc:
            state.update(status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED',
                error=repr(exc),traceback=traceback.format_exc(),no_automatic_replay=True)
            T.write(adir/'failure.json',state); raise
        finally:
            for sig,old in handlers.items():signal.signal(sig,old)
            state.update(S.cursor(reg,step),elapsed_seconds=time.perf_counter()-started,calls=ledger.snapshot(),resources=observer.summary())
            T.write(adir/'resources.json',dict(samples=observer.samples,summary=observer.summary()))
            T.write(status_path,state); T.write(adir/'final_status.json',state)
        return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    print(json.dumps(train(parser.parse_args().resume),ensure_ascii=False),flush=True)
