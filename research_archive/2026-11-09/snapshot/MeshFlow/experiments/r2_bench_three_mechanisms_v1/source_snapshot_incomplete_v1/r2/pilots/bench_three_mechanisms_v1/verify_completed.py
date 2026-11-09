"""CPU-only integrity check for all supervised endpoint and recovery payloads.

No model construction, CUDA initialization, RNG restoration or parameter update.
The extra failed-before-backward PLAIN forward is retained in the call history.
"""
from __future__ import annotations
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import sys
from . import common as C

VERSION='bench_three_mechanisms_completed_state_audit_v1'
ENDPOINTS=(('PLAIN',200),('PLAIN',267),('PATCH',200),('LOOP',200))


def ledger_prefix(rows, arm, step, limits, extra):
    """Authenticate completed prefix through this checkpoint optimizer update."""
    attempted=collections.Counter();successful=collections.Counter();pending={};failed=collections.Counter()
    begins=collections.defaultdict(list);seen=set();found=False
    for r in rows:
        if r['arm']!=arm or r['kind'] not in limits:raise ValueError('Wrong ledger owner/kind')
        kind,ident=r['kind'],r['call_id']
        if r['event']=='CALL_BEGIN':
            if ident in seen or pending:raise ValueError('Repeated/overlapping ledger call')
            seen.add(ident);pending[ident]=r;attempted[kind]+=1;begins[kind].append(r['context'])
        elif r['event'] in ('CALL_DONE','CALL_FAILED'):
            if ident not in pending or pending[ident]['kind']!=kind:raise ValueError('Unpaired ledger completion')
            del pending[ident]
            (successful if r['event']=='CALL_DONE' else failed)[kind]+=1
            if kind=='optimizer_update' and r['event']=='CALL_DONE' and int(r['context']['step'])==step:
                found=True;break
        else:raise ValueError('Unknown call event')
    expected={'train_forward':8*step+extra,'train_backward':8*step,'optimizer_update':step}
    if not found or pending or failed or dict(attempted)!=expected or dict(successful)!=expected:
        raise ValueError('Endpoint ledger cursor/budget differs')
    for kind in expected:
        actual=begins[kind][extra:] if kind=='train_forward' else begins[kind]
        wanted=[(s,m) for s in range(1,step+1) for m in (range(8) if kind!='optimizer_update' else (None,))]
        if [(int(r['step']),r.get('microbatch')) for r in actual]!=wanted:
            raise ValueError('Exact step/microbatch ledger sequence differs: '+kind)
    if extra and [(r['step'],r['microbatch']) for r in begins['train_forward'][:extra]]!=[(1,0)]:
        raise ValueError('Unregistered extra forward identity')
    if any(attempted[k]>limits[k] for k in attempted):raise ValueError('Budget exceeded')
    return dict(attempted=dict(attempted),successful=dict(successful),failed={},pending_call_ids=[],limits=limits)


def verify_optimizer(payload,torch):
    names=payload['optimizer_parameter_names'];opt=payload['optimizer'];state=payload['generator_state']
    if len(names)!=1 or len(opt['param_groups'])!=1:raise ValueError('Expected inherited single AdamW group')
    ns=names[0];g=opt['param_groups'][0];ids=g['params']
    if (g['lr'],tuple(g['betas']),g['eps'],g['weight_decay'])!=(1e-5,(.9,.95),1e-8,0.):
        raise ValueError('Inherited optimizer hyperparameters changed')
    if len(ids)!=len(ns) or len(set(ids))!=len(ids) or len(set(ns))!=len(ns) or set(ids)!=set(opt['state']):
        raise ValueError('Optimizer parameter identity/order unavailable')
    if set(ns)!=set(state):raise ValueError('Generator parameters differ from inherited optimizer names')
    for name,pid in zip(ns,ids):
        p=state[name];s=opt['state'][pid]
        if set(s)!={'step','exp_avg','exp_avg_sq'} or float(s['step'])!=payload['global_step']:
            raise ValueError('AdamW cursor differs: '+name)
        if p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all()):
            raise ValueError('Nonfinite/nonFP32 generator state')
        for key in ('exp_avg','exp_avg_sq'):
            if s[key].shape!=p.shape or s[key].dtype!=torch.float32 or not bool(torch.isfinite(s[key]).all()):
                raise ValueError('AdamW moment shape/dtype/finite error')
    return len(ns)


def verify_endpoint(arm,step,torch,state_hash,tree_hash):
    root=C.OUT/'training'/arm;amend=C.OUT/'registrations/engineering_amendment_01.json'
    regpath=C.OUT/'registrations'/(arm+'_v2.json' if arm=='PLAIN' and amend.exists() else arm+'.json')
    reg=C.read(regpath);C.verify(reg['protected']);rh=C.sha(regpath)
    extra=int(reg['limits']['train_forward']-8*reg['limits']['optimizer_update'])
    if extra not in (0,1) or (extra and arm!='PLAIN'):raise ValueError('Unregistered replay allowance')
    rows=[json.loads(x) for x in (root/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    execution=ledger_prefix(rows,arm,step,reg['limits'],extra)
    path=root/f'checkpoint_step{step}.pt';side=path.with_suffix('.json');receipt=C.read(side)
    if C.resolve(receipt['path'])!=path or C.sha(path)!=receipt['sha256'] or path.stat().st_size!=receipt['bytes']:
        raise ValueError('Full checkpoint receipt mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    expected=dict(schema='bench_three_mechanisms_training_v1',arm=arm,local_step=step,stage_step=step,
        global_step=C.BASE_STEP+step,native_total_step=C.NATIVE_BASE+step,seam_total_step=0,
        EMA=False,update_in_progress=False,registration_sha256=rh,source_checkpoint_sha256=reg['source_sha256'])
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint schema/cursor/registration mismatch')
    if payload['execution']!=execution:raise ValueError('Checkpoint saved execution differs from actual ledger prefix')
    for key,hkey,fn in (('generator_state','generator_state_sha256',state_hash),('optimizer','optimizer_state_sha256',tree_hash),('rng','rng_sha256',tree_hash)):
        h=fn(payload[key])
        if h!=payload[hkey] or h!=receipt[hkey]:raise ValueError('Saved '+key+' hash differs')
    parameters=verify_optimizer(payload,torch)
    if payload['head_calls']!=0 or payload['head_step']!=0 or payload['head_optimizer'] is not None:
        raise ValueError('Unregistered head activity')
    if state_hash(payload['head_state'])!=payload['head_state_sha256']:raise ValueError('Frozen head hash differs')
    rng=payload['rng']
    if set(rng)!={'python','numpy','torch_cpu','torch_cuda'} or rng['torch_cpu'].dtype!=torch.uint8:
        raise ValueError('Recovery RNG payload incomplete')
    if not isinstance(rng['torch_cuda'],list) or not rng['torch_cuda'] or any(t.dtype!=torch.uint8 or t.device.type!='cpu' for t in rng['torch_cuda']):
        raise ValueError('CUDA RNG recovery bytes missing (read only; not restored)')
    ep=root/f'generator_step{step}.pt';er=C.read(ep.with_suffix('.json'))
    if C.sha(ep)!=er['sha256'] or ep.stat().st_size!=er['bytes']:raise ValueError('Export receipt mismatch')
    export=torch.load(ep,map_location='cpu',weights_only=True,mmap=True)
    if export.get('schema')!='bench_three_mechanisms_generator_v1' or export.get('model_arm')!=arm or export.get('training_checkpoint_sha256')!=receipt['sha256']:
        raise ValueError('Export provenance differs')
    if export['global_step']!=C.BASE_STEP+step or export['local_step']!=step or export['registration_sha256']!=rh:
        raise ValueError('Export cursor/registration differs')
    if state_hash(export['generator_state'])!=payload['generator_state_sha256'] or er['generator_state_sha256']!=payload['generator_state_sha256']:
        raise ValueError('Inference export differs from live checkpoint')
    if any(k in export for k in ('optimizer','rng','execution','optimizer_parameter_names')):raise ValueError('Inference export retained training state')
    result=dict(status='PASS',arm=arm,step=step,global_step=C.BASE_STEP+step,parameters_with_moments=parameters,
        checkpoint_sha256=receipt['sha256'],export_sha256=er['sha256'],rng_sha256=payload['rng_sha256'],
        generator_state_sha256=payload['generator_state_sha256'],optimizer_state_sha256=payload['optimizer_state_sha256'],
        execution=execution,registration_sha256=rh,extra_forward_accounted=extra,CUDA_initialized=False)
    del payload,export;gc.collect()
    return result


def run():
    missing=[]
    for arm,step in ENDPOINTS:
        for name in (f'checkpoint_step{step}.pt',f'checkpoint_step{step}.json',f'generator_step{step}.pt',f'generator_step{step}.json'):
            p=C.OUT/'training'/arm/name
            if not p.exists():missing.append(str(p))
    if missing:return dict(status='BLOCKED_MISSING_ENDPOINTS',missing=missing,files_written=0)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(C.ROOT/'r2/src'))
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    if torch.cuda.is_initialized():raise RuntimeError('Must run in a fresh CPU-only process')
    torch.set_num_threads(2)
    outputs=[verify_endpoint(a,s,torch,state_hash,tree_hash) for a,s in ENDPOINTS]
    status_files={};log_files={}
    for arm,target in C.UPDATES.items():
        sp=C.OUT/'training'/arm/'status.json';status=C.read(sp);status_files[str(sp)]=C.sha(sp)
        regpath=C.OUT/'registrations'/(arm+'_v2.json' if arm=='PLAIN' and (C.OUT/'registrations/engineering_amendment_01.json').exists() else arm+'.json')
        reg=C.read(regpath);last=next(r for r in outputs if r['arm']==arm and r['step']==target)
        if status['status']!='COMPLETE' or status['step']!=target or status['calls']!=last['execution']:
            raise ValueError('Arm is not complete at the exact registered endpoint')
        for name,count in (('training_log.jsonl',target),('exposure.jsonl',8*target)):
            p=C.OUT/'training'/arm/name;rr=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
            if len(rr)!=count:raise ValueError('Logged update/exposure count differs '+str(p))
            log_files[str(p)]=C.sha(p)
        if status['checkpoint']['sha256']!=last['checkpoint_sha256']:raise ValueError('Status final checkpoint differs')
    if torch.cuda.is_initialized():raise RuntimeError('Audit initialized CUDA')
    result=dict(status='PASS',version=VERSION,endpoints=outputs,status_files=status_files,log_files=log_files,
        model_constructions=0,model_forwards=0,CUDA_initialized=False,RNG_restores=0,code_sha256=C.sha(__file__))
    C.write(C.OUT/'audits/completed_supervised_states.json',result)
    return result


def check_interface():
    rows=[];ident=0
    for kind,m in [('train_forward',i) for i in range(8)]+[('train_backward',i) for i in range(8)]+[('optimizer_update',None)]:
        ident+=1;c=dict(step=1)
        if m is not None:c['microbatch']=m
        r=dict(arm='PATCH',kind=kind,call_id=ident,context=c)
        rows.extend((dict(r,event='CALL_BEGIN'),dict(r,event='CALL_DONE')))
    s=ledger_prefix(rows,'PATCH',1,dict(train_forward=1600,train_backward=1600,optimizer_update=200),0)
    assert s['successful']['optimizer_update']==1
    return dict(status='PASS',checks=1,files_written=0,torch_imported='torch' in sys.modules,model_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-interface',action='store_true');a=p.parse_args()
    r=check_interface() if a.check_interface else run();print(json.dumps(r,ensure_ascii=False))
