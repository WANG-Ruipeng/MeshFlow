"""Paired Native natural-N continuations from retained exposure B global21000.

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
from .runtime import build_model, native_execution, sample_for_training, verify_sample, VERSION as RUNTIME_VERSION
from meshflow_control.training.schedule40 import make_optimizer
from meshflow_control.seam.model_adapter import corner_masked_fm_loss
from meshflow_control.training.context_checkpoint import _cpu_copy, _check_generator
from meshflow_control.data.io import digest, append_event
from r2.pilots.bench_adaptation_strength_v1 import training as T
from r2.pilots.bench_parts_convergence_v1 import data as D
from r2.pilots.bench_full_seat_curriculum_v1.train import _log_cursor, _save, _receipt_matches
from r2.pilots.bench_full_seat_curriculum_v1.training_state import (
    validate_optimizer, validate_head_state, statistics, HEAD_CONFIG, HEAD_STATE_SHA256)
from . import stream as Q
from .common import (OUT, SOURCE_ROOT, SOURCE, EXPERIMENT, BASE_STEP, SOURCE_IDENTITIES, resolve)
STAGE = 'NATIVE'
SCHEMA = 'bench_natural_n_recipe_training_v1'
INFERENCE_SCHEMA = 'bench_natural_n_recipe_generator_v1'
S = sys.modules[__name__]


def stage_spec(reg):
    if (reg['updates'],reg['source_global_step'],reg['final_global_step']) != (1500,BASE_STEP,22500):
        raise ValueError('Exactly 1500 new updates from exposure B21000 to22500 required')
    if reg['full_steps'] != [1500] or reg['checkpoint_every'] != 250:
        raise ValueError('Retain endpoint and rolling250; source is referenced, not recreated')
    return dict(updates=1500,full_steps=[1500],global_start=BASE_STEP)


def cursor(reg,step):
    stage_spec(reg)
    if type(step) is not int or not 0 <= step <= 1500:
        raise ValueError('New branch local cursor outside budget')
    return dict(stage=STAGE,stage_step=step,local_step=step,curriculum_step=3000+step,
                global_step=BASE_STEP+step,native_total_step=9000+step,seam_total_step=0)


def limits(reg):
    stage_spec(reg)
    return dict(train_forward=12000,train_backward=12000,optimizer_update=1500,
                head_forward=0,head_backward=0,head_optimizer_update=0,probe_forward=0)


def require_cursor(ledger,reg,step):
    cursor(reg,step)
    if ledger.pending or ledger.failed:
        raise RuntimeError('Failed/pending scientific calls forbid replay')
    expected=dict(train_forward=8*step,train_backward=8*step,optimizer_update=step,
                  head_forward=0,head_backward=0,head_optimizer_update=0,probe_forward=0)
    if ledger.limits != limits(reg) or any(ledger.attempted[k]!=n or ledger.successful[k]!=n for k,n in expected.items()):
        raise RuntimeError('Saved cursor differs from attempted/completed work; no rollback')
    starts=[r for r in ledger.rows if r['event']=='CALL_BEGIN']
    for kind in ('train_forward','train_backward','optimizer_update'):
        rows=[r['context'] for r in starts if r['kind']==kind]
        wanted=[(s,m) for s in range(1,step+1) for m in (range(8) if kind!='optimizer_update' else (None,))]
        if [(r['stage_step'],r.get('microbatch')) for r in rows]!=wanted or any(r['stage']!=STAGE or r['global_step']!=BASE_STEP+r['stage_step'] for r in rows):
            raise RuntimeError('Exact global/local/microbatch sequence differs')


def validate_payload(p,reg,step,reg_sha,arm):
    if any(p.get(k)!=v for k,v in cursor(reg,step).items()):
        raise ValueError('Saved branch cursor differs')
    if (p.get('schema'),p.get('arm'),p.get('registration_sha256'),p.get('EMA'),p.get('update_in_progress')) != (SCHEMA,arm,reg_sha,False,False):
        raise ValueError('Saved branch checkpoint identity differs')
    if p['stream_state']!=dict(batch_index=step,stage_batch_index=step,next_sample_index=8*step,arm=arm,schedule_sha256=reg['schedules'][arm]['sha256']):
        raise ValueError('Saved stream is not this exposure branch')
    for key,hkey,fn in (('generator_state','generator_state_sha256',state_hash),('optimizer','optimizer_state_sha256',T.base.tree_hash),('rng','rng_sha256',T.base.tree_hash)):
        if fn(p[key])!=p[hkey]:raise ValueError('Checkpoint state/hash differs: '+key)
    _check_generator(p['generator_state']); validate_head_state(p['head_state'],p['head_state_sha256'])
    if p['head_optimizer'] is not None or p['head_step']!=0 or p['head_calls']!=0 or p['head_config']!=HEAD_CONFIG:
        raise ValueError('Frozen unused source head metadata changed')
    if (p['model_config']!=reg['model_config'] or p['source_global_step']!=BASE_STEP
            or p.get('runtime_version')!=RUNTIME_VERSION):
        raise ValueError('Model/source identity differs')



def checked(arm):
    from . import common
    if (common.OUT,common.SOURCE,common.BASE_STEP,common.EXPERIMENT,common.SOURCE_IDENTITIES) != (OUT,SOURCE,BASE_STEP,EXPERIMENT,SOURCE_IDENTITIES):
        raise ValueError('Orchestrator and trainer source identity constants differ')
    reg = T.read(OUT/'training_registration.json')
    if (reg['status'],reg['experiment'],reg['batch'],reg['microbatch'],reg['hybrid_lambda'],reg['model_config']) != ('FROZEN',EXPERIMENT,8,1,.25,MODEL_CONFIG):
        raise ValueError('Frozen Native natural-N contract differs')
    S.stage_spec(reg)
    if reg.get('runtime_version') != RUNTIME_VERSION or reg.get('N_contract') != [128,799]:
        raise ValueError('Independent Native runtime/range registration differs')
    if reg.get('loss_normalization') != 'whole_effective_batch_free_coordinates':
        raise ValueError('Original B whole-batch free-coordinate FM is required')
    if arm not in Q.ARMS or reg['arms'] != list(Q.ARMS): raise ValueError('Only registered CONTROL_OLD/NATURAL_HIGH branches')
    if any(reg[k] != v for k,v in SOURCE_IDENTITIES.items()) or resolve(reg['source_checkpoint']).resolve() != SOURCE.resolve():
        raise ValueError('Only retained exposure B21000 live/matching optimizer/RNG is allowed')
    for key in ('head_forward_budget','head_optimizer_updates','probe_forward_budget'):
        if reg[key] != 0:
            raise ValueError('Head/probe activity is not authorized')
    if not reg['training_code_hashes']:
        raise ValueError('Frozen source manifest required')
    for p,h in reg['training_code_hashes'].items():
        if digest(resolve(p)) != h:
            raise ValueError('Frozen training source changed: '+p)
    for key in ('manifest','eval_manifest','source_checkpoint'):
        if digest(resolve(reg[key])) != reg[key+'_sha256']:
            raise ValueError('Frozen input changed: '+key)
    for p,h in reg['gate_hashes'].items():
        if digest(resolve(p)) != h:
            raise ValueError('Frozen preparation gate changed: '+p)
    m = T.read(resolve(reg['manifest']))
    stream_spec = reg['schedules'][arm]
    if digest(resolve(stream_spec['path'])) != stream_spec['sha256']: raise ValueError('Frozen arm stream changed')
    schedule = T.read(resolve(stream_spec['path']))
    Q.validate(schedule,m['train'],reg['updates'],arm)
    train = m['train']; parents = {r['parent_id'] for r in train}
    for r in m.get('rows',train):
        if digest(resolve(r['case_path'])) != r['case_sha256']:
            raise ValueError('Frozen case changed: '+r['condition_id'])
    old = T.read(SOURCE_ROOT/'data/manifest.json')['train']
    original = {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in old}
    replay = {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in train if r['replay_group'] != 'HIGH'}
    if original != replay:
        raise ValueError('The same source B24 parents/conditions must be retained unchanged')
    heldout = {r['parent_id'] for r in m.get('dev',[]) + m.get('high_holdout',[])}
    ev = T.read(resolve(reg['eval_manifest']))
    for r in ev['requests']:
        group = r.get('eval_group',r.get('report_group',r.get('group','')))
        if group in ('DEV','HOLDOUT','HIGH_HOLDOUT','NEW_DEV','INDEPENDENT_DEV','NEW_PARENT_HOLDOUT'):
            heldout.add(r['parent_id'])
    if parents & heldout:
        raise ValueError('Held-out evaluation parent entered training')
    return reg,m,schedule['steps']


def _source_payload(reg):
    p = torch.load(resolve(reg['source_checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    if (p['schema'],p['arm'],p['stage'],p['stage_step'],p['global_step'],p['EMA'],p['update_in_progress']) != ('bench_exposure_allocation_training_v1','B_75_25','NATIVE',1500,21000,False,False):
        raise ValueError('Retained exposure B endpoint identity differs')
    for key,hkey,fn in (('generator_state','source_generator_sha256',state_hash),('optimizer','source_optimizer_sha256',T.base.tree_hash),('rng','source_rng_sha256',T.base.tree_hash)):
        if fn(p[key]) != reg[hkey]:
            raise ValueError('Retained source state differs: '+key)
    if p['model_config'] != MODEL_CONFIG:
        raise ValueError('Source model configuration differs')
    if p['native_total_step'] != 9000 or p['seam_total_step'] != 0 or p['head_calls'] != 0 or p['head_optimizer'] is not None:
        raise ValueError('Source is not the retained Native-only exposure B endpoint')
    S.validate_head_state(p['head_state'])
    _check_generator(p['generator_state'])
    return p



def _build_restore(payload, global_step, device):
    """Restore matching generator/moments onto the independent runtime."""
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode="none", trainable=True).eval()
    model.load_state_dict(payload["generator_state"], strict=True)
    model.to(device)
    optimizer = make_optimizer(model, None)
    optimizer.load_state_dict(payload["optimizer"])
    names = validate_optimizer(model, optimizer, global_step)
    if "optimizer_parameter_names" in payload and payload["optimizer_parameter_names"] != names:
        raise ValueError("Single-group AdamW parameter names/order differ")
    if state_hash(model.state_dict()) != payload["generator_state_sha256"]:
        raise ValueError("Restored generator content differs")
    if T.base.tree_hash(optimizer.state_dict()) != payload["optimizer_state_sha256"]:
        raise ValueError("Inherited AdamW moments were reset, regrouped or changed")
    return model, optimizer


def train(arm,resume=False):
    if arm not in Q.ARMS: raise ValueError(arm)
    T.install_io(OUT,arm)
    reg,m,steps = checked(arm)
    spec = S.stage_spec(reg)
    reg_sha = digest(OUT/'training_registration.json')
    go = T.read(OUT/'dispatch'/(arm+'_go.json'))
    if not go.get('authorized') or go.get('training_registration_sha256') != reg_sha or go.get('arm') != arm:
        raise RuntimeError('Parent resource-aware dispatch release missing')
    headroom = max(2048.,float(go.get('retained_headroom_mib',2048.)))
    out = OUT/'training'/arm; out.mkdir(parents=True,exist_ok=True)
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
        if previous and (not resume or previous['status'] not in ('INTERRUPTED_SAFE','FAILED','OOM')):
            raise RuntimeError('Unresolved attempt; no automatic scientific replay')
        if resume and not previous:
            raise ValueError('No exact saved cursor exists')
        step = int(previous['stage_step']) if previous else 0
        attempt = 'attempt_'+str(time.time_ns()); adir = out/'attempts'/attempt
        adir.mkdir(parents=True,exist_ok=False)
        ledger = T.historical.CallLedger(out/'calls.jsonl',S.limits(reg),arm,attempt)
        S.require_cursor(ledger,reg,step); _log_cursor(out,step)
        state = dict(status='LOADING',arm=arm,**S.cursor(reg,step),target_local_step=spec['updates'],
            registration_sha256=reg_sha,pid=os.getpid(),attempt=attempt,head_step=0,head_calls=0,probe_calls=0)
        if previous: state['checkpoint'] = previous['checkpoint']
        initial = T.resource(); T.write(adir/'resources_dispatch.json',initial)
        if initial['free_mib'] is None or initial['free_mib'] < max(headroom,float(go.get('minimum_free_mib',headroom))):
            raise RuntimeError('RESOURCE_WAIT before model load')
        started = time.perf_counter()
        observer = T.historical.GPUObserver('bench_natural_n_'+arm,interval=2.)
        stop = {'signal':None}
        def handler(sig,unused): stop['signal']=sig
        handlers = {sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
        T.write(status_path,state)
        try:
            with observer:
                torch.set_num_threads(2); runtime.configure_stable_runtime()
                if previous and not previous['checkpoint'].get('source_reference_only',False):
                    source = resolve(previous['checkpoint']['path'])
                    if digest(source) != previous['checkpoint']['sha256']:
                        raise ValueError('Resume file changed')
                    p = torch.load(source,map_location='cpu',weights_only=True,mmap=True)
                    S.validate_payload(p,reg,step,reg_sha,arm)
                    if p['execution'] != ledger.snapshot():
                        raise ValueError('Saved execution ledger differs')
                    audit = p['initialization']
                else:
                    if step != 0: raise ValueError('Source reference may initialize only local zero')
                    p = _source_payload(reg)
                    audit = dict(source_path=str(SOURCE),**SOURCE_IDENTITIES,source_global_step=BASE_STEP,
                        inherited_optimizer=True,inherited_rng=True,fresh_AdamW=False,new_generator_parameters=0,
                        source_stream_resumed=False,new_registered_stream=True,head_used=False,head_calls=0,
                        new_branch_arm=arm,schedule_sha256=reg['schedules'][arm]['sha256'],
                        runtime_version=RUNTIME_VERSION,N_contract=[128,799],loss_normalization='whole_effective_batch_free_coordinates')
                head_state = p['head_state']
                model,optimizer = _build_restore(p,BASE_STEP+step,'cuda')
                restore_rng(p['rng'])
                if T.base.tree_hash(rng_state()) != p['rng_sha256']:
                    raise ValueError('Restored RNG differs')
                del p
                T.write(adir/'initialization.json',audit)
                used = {r['condition_id'] for e in steps for r in e['samples']}
                cases = {r['condition_id']:D.load_case(r) for r in m['train'] if r['condition_id'] in used}

                def checkpoint_save():
                    S.require_cursor(ledger,reg,step); _log_cursor(out,step)
                    position = S.cursor(reg,step)
                    names = S.validate_optimizer(model,optimizer,BASE_STEP+step)
                    g,o,rng = _cpu_copy(model.state_dict()),_cpu_copy(optimizer.state_dict()),_cpu_copy(rng_state())
                    _check_generator(g); S.validate_head_state(head_state)
                    payload = dict(schema=S.SCHEMA,experiment=EXPERIMENT,arm=arm,**position,
                        source_global_step=BASE_STEP,model_config=copy.deepcopy(MODEL_CONFIG),runtime_version=RUNTIME_VERSION,EMA=False,
                        update_in_progress=False,generator_state=g,generator_state_sha256=state_hash(g),
                        optimizer=o,optimizer_state_sha256=T.base.tree_hash(o),optimizer_parameter_names=names,
                        rng=rng,rng_sha256=T.base.tree_hash(rng),registration_sha256=reg_sha,
                        head_state=head_state,head_state_sha256=S.HEAD_STATE_SHA256,head_config=S.HEAD_CONFIG,
                        head_optimizer=None,head_step=0,head_calls=0,initialization=audit,execution=ledger.snapshot(),
                        stream_state=dict(batch_index=step,stage_batch_index=step,next_sample_index=8*step,arm=arm,schedule_sha256=reg['schedules'][arm]['sha256']))
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
                            'model_config','runtime_version','EMA','head_state','head_state_sha256','head_config','head_step')
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
                # Both local-zero branches reference the same immutable exposure B file.
                # No legacy step0 is restored and no duplicate model snapshot is saved.
                if not previous:
                    state['checkpoint']=dict(path=str(SOURCE),sha256=reg['source_checkpoint_sha256'],source_reference_only=True)
                    T.write(out/'branch_initialization.json',dict(**audit,registration_sha256=reg_sha))
                for entry in steps[step:spec['updates']]:
                    if stop['signal'] is not None:
                        checkpoint_save(); state.update(status='INTERRUPTED_SAFE',signal=stop['signal']); break
                    tick=time.perf_counter(); current=entry['stage_step']
                    if current % 25 == 1:
                        resource=T.resource(); append_event(out/'resource_checks.jsonl',dict(stage_step_before=step,**resource))
                        if resource['free_mib'] is None or resource['free_mib'] < headroom:
                            checkpoint_save(); state.update(status='INTERRUPTED_SAFE',reason='VRAM_headroom_below_dispatch_margin'); break
                    samples=[sample_for_training(cases[r['condition_id']],r,'PLAIN') for r in entry['samples']]
                    audits=[S.verify_sample(s,'NATIVE') for s in samples]
                    denominator=sum(int(s['free_coordinate_count']) for s in samples)
                    input_seconds=time.perf_counter()-tick; begin=time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
                    fm,statistics=0.,[]
                    for mi,(sample,row,audit_row) in enumerate(zip(samples,entry['samples'],audits)):
                        context=dict(arm=arm,**S.cursor(reg,current),microbatch=mi,condition_id=row['condition_id'])
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
                    ledger.call('optimizer_update',optimizer.step,dict(arm=arm,**S.cursor(reg,current))); step=current
                    S.validate_optimizer(model,optimizer,BASE_STEP+step)
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise FloatingPointError('Nonfinite parameter')
                    sums={k:sum(r[k] for r in statistics) for k in statistics[0]}
                    result=dict(arm=arm,**S.cursor(reg,step),FM=fm,objective=fm,free_coordinate_denominator=denominator,
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=Q.ARMS,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); result=train(args.arm,args.resume)
    print(json.dumps({k:result[k] for k in ('status','arm','stage_step','global_step','calls','checkpoint','inference_export') if k in result}),flush=True)


