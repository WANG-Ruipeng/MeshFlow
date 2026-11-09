"""Frozen inference banks; unchanged original Euler50 and all-51-state C checks."""
from native_t1 import runtime
import gc
from pathlib import Path
import time
import numpy as np
import torch
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.sampling import sample_case,validate_bank
from .common import *
from . import checkpoint
from .resources import resource_snapshot


def bank_for(row):
    p=resolve(row['input_npz'])
    if digest(p)!=row['input_sha256']:raise ValueError('Frozen inference bank changed')
    with np.load(p,allow_pickle=False) as z:
        if set(z.files)!=set(('z','C','known_mask','valid_mask','N','K')):raise ValueError('Inference allowlist differs')
        bank={k:z[k].copy() for k in z.files}
    validate_bank(bank);return bank


def init_cache_rows(out):
    for row in read_json(Path(out)/'requests.json'):
        p=Path(out)/'outputs'/row['request_id']/'result.json'
        if p.exists():continue
        if row['status']=='NOT_APPLICABLE':atomic_json(p,dict(row,model_calls=0));continue
        if row['status']!='CACHED':continue
        old=read_json(resolve(row['legacy_case_json']))
        raw=resolve(row['legacy_raw'])
        assert digest(raw)==row['legacy_raw_sha256']==old['raw_sha256']
        metadata=old.get('metadata',old.get('audit',{}))
        assert metadata['nfe']==50 and metadata['known_all51_bitwise']
        value=dict(row,status='CACHED',raw_path=str(raw),raw_sha256=digest(raw),metadata=metadata,
            model_calls=0,source_generation_seconds=metadata.get('seconds'),timing_source='historical actual generation',
            cache_source_case=str(resolve(row['legacy_case_json'])),cache_identity='registration verified')
        atomic_json(p,value)


def generate(model,meta,row,budget,directory,replay=False):
    directory=Path(directory);p=directory/'result.json'
    if p.exists():
        previous=read_json(p)
        if previous['status']=='SAMPLED':
            assert digest(resolve(previous['raw_path']))==previous['raw_sha256'];return previous
        if previous['status'] in ('FAILED','RUNNING'):raise RuntimeError('Prior failed/unresolved generation cannot auto-replay')
    bank=bank_for(row);rid=('replay_' if replay else '')+row['request_id']
    context=dict(request_id=rid,arm=row['arm'],uid=row['uid'],candidate=row['candidate'],replay=replay)
    token=budget.begin('sampling',context)
    value=dict(row,status='RUNNING',checkpoint_sha256=meta['checkpoint_sha256'],model_calls=0)
    atomic_json(p,value)
    try:
        result=sample_case(model,bank,False,budget,context=context)
        raw=directory/'raw.npz'
        np.savez_compressed(raw,raw=result['raw'],noise=bank['z'],C=bank['C'],known_mask=bank['known_mask'],valid_mask=bank['valid_mask'])
        value.update(status='SAMPLED',raw_path=str(raw),raw_sha256=digest(raw),raw_array_sha256=array_hash(result['raw']),
            metadata=result['metadata'],model_calls=50,source_generation_seconds=result['metadata']['seconds'],timing_source='new actual generation')
        if replay:
            expected=resolve(row['expected_raw']);assert digest(expected)==row['expected_raw_sha256']
            with np.load(expected,allow_pickle=False) as z:old=z['raw']
            value['identity_replay']=dict(bitwise_equal=old.dtype==result['raw'].dtype and old.shape==result['raw'].shape and old.tobytes()==result['raw'].tobytes(),
                max_abs=float(np.max(np.abs(old.astype(np.float64)-result['raw']))))
        atomic_json(p,value);budget.end(token)
    except BaseException as exc:
        budget.end(token,exc);value.update(status='FAILED',error=repr(exc),model_calls=budget.per_output[rid])
        atomic_json(p,value);budget.snapshot();raise
    budget.snapshot();return value


def replay_initial(out,budget):
    out=Path(out);jobs=read_json(out/'registration/identity_replays.json')
    completed=[out/'replays'/j['base_request_id']/'result.json' for j in jobs]
    if all(p.exists() and read_json(p)['status']=='SAMPLED' and read_json(p)['identity_replay']['bitwise_equal'] for p in completed):
        return read_json(out/'replays/summary.json')
    model,_,meta,_=checkpoint.load(START,'B_BOUNDARY','cuda',trainable=False,optimizer=False,out=out)
    meta['checkpoint_sha256']=START_SHA
    rows=[]
    for job in jobs:
        row=dict(job,arm='B_BOUNDARY',label='B_ZERO_REPLAY')
        value=generate(model,meta,row,budget,out/'replays'/job['base_request_id'],replay=True);rows.append(value)
        resource_snapshot(out,'initial_replay_'+job['panel'])
        if not value['identity_replay']['bitwise_equal']:raise RuntimeError('Initial zero-boundary raw differs from START cache')
    assert checkpoint.tensor_state_hash(model.state_dict())==meta['model_state_sha256']
    del model;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS',replays=2,NFE=100,rows=rows,D_uses_same_initial_zero_path=True)
    atomic_json(out/'replays/summary.json',result);return result


def endpoint(out,arm,budget):
    out=Path(out);init_cache_rows(out)
    pointer=read_json(out/'training'/f'{arm}_pointer.json');path=resolve(pointer['path'])
    assert pointer['metadata']['attachment_step']==1000
    jobs=[r for r in read_json(out/'requests.json') if r['arm']==arm and r['status']=='READY']
    assert len(jobs)==76
    todo=[r for r in jobs if not (out/'outputs'/r['request_id']/'result.json').exists()]
    for row in jobs:
        completed=out/'outputs'/row['request_id']/'result.json'
        if completed.exists():
            saved=read_json(completed)
            if saved['status']!='SAMPLED':raise RuntimeError('Existing endpoint is failed/unresolved; no automatic replay')
            assert saved['request_id']==row['request_id'] and saved['input_sha256']==row['input_sha256']
            assert saved['checkpoint_sha256']==pointer['checkpoint_sha256']
            assert digest(resolve(saved['raw_path']))==saved['raw_sha256']
    if not todo:return dict(status='DONE',arm=arm,outputs=len(jobs),new_this_worker=0)
    model,_,meta,_=checkpoint.load(path,arm,'cuda',trainable=False,optimizer=False,out=out)
    meta['checkpoint_sha256']=pointer['checkpoint_sha256'];started=time.perf_counter()
    for index,row in enumerate(jobs):
        value=generate(model,meta,row,budget,out/'outputs'/row['request_id'])
        if index==0:resource_snapshot(out,arm+'_first_endpoint_rollout')
        print(dict(event='SAMPLING',arm=arm,completed=index+1,request_id=row['request_id'],status=value['status']),flush=True)
    assert checkpoint.tensor_state_hash(model.state_dict())==meta['model_state_sha256']
    result=dict(status='DONE',arm=arm,outputs=76,new_this_worker=len(todo),wall_seconds=time.perf_counter()-started)
    atomic_json(out/'sampling'/f'{arm}_status.json',result)
    del model;gc.collect();torch.cuda.empty_cache();return result
