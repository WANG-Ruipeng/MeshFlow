"""One locked GPU worker, strict frozen generator loading, original Euler50."""
from native_t1 import runtime
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.context_geometry import ContextGeometryEncoder
from native_t1.portable_checkpoint import _read_config, config_sha256
from native_t1.artifacts import state_sha256
from experiments.mf_h1_local0.control_mvp48.model_mvp import MVPModel
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.sampling import sample_case, validate_bank
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import array_hash
from .quick_common import ROOT, OUT, LOCK, resolve, read, write, sha, event, Budget, resource_snapshot


def load_model(arm, out):
    entries = read(ROOT/'native_t1/runs/chair_hybrid_coupling_v1/checkpoint_manifest.json')['endpoints']
    record = next(r for r in entries if r['metadata']['arm'] == arm)
    p = resolve(record['path']); tick = time.perf_counter()
    if sha(p) != record['checkpoint_sha256']: raise ValueError('Exact endpoint file hash differs')
    if arm == 'OT_HYBRID' and record['checkpoint_sha256'] != 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80':
        raise ValueError('HYBRID must be the retained total5000 START')
    payload = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    meta = payload['metadata']
    if payload['schema'] != 'chair_hybrid_coupling_checkpoint_v1' or payload['update_in_progress'] is not False:
        raise ValueError('Wrong checkpoint schema/boundary')
    if meta != record['metadata'] or (meta['arm'],meta['pilot_step'],meta['conditional_total_step'],meta['lambda_variance']) != (arm,1000,5000,0. if arm=='OT_BASE' else .25):
        raise ValueError('Wrong frozen endpoint identity')
    if meta['context_encoder'] != 'geo' or meta['loss_recipe'] != 'fm' or meta['config_sha256'] != config_sha256(DEFAULT_CONFIG):
        raise ValueError('Architecture/Geo/FM identity differs')
    config = _read_config(Path(DEFAULT_CONFIG))
    with torch.device('meta'):
        model = MVPModel(DiT(**config), trainable=False)
    model.backbone.x_embedder.embed_fn, _ = get_embedder(config['pe_freq'], input_dims=3)
    model.context_encoder = ContextGeometryEncoder('geo', seed=1010)
    keys = model.load_state_dict(payload['model'], strict=True, assign=True)
    model.set_trainable(False).eval()
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite or non-FP32 parameter')
    actual = state_sha256(model)
    if actual != meta['model_state_sha256']: raise ValueError('Actual generator state hash differs')
    audit = dict(status='PASS', arm=arm, path=str(p), checkpoint_sha256=record['checkpoint_sha256'],
                 model_state_sha256=actual, pilot_step=1000, conditional_total_step=5000,
                 lambda_variance=meta['lambda_variance'], missing_keys=list(keys.missing_keys),
                 unexpected_keys=list(keys.unexpected_keys), model_class='original MVPModel',
                 parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
                 optimizer_constructed=False, optimizer_restored=False, rng_restored=False,
                 geo_mixing='P=I; adjacency diagnostic only', seconds_cpu_load=time.perf_counter()-tick)
    del payload
    model.to('cuda')
    write(Path(out)/'load_audits'/f'{arm}.json', audit)
    return model, audit


def bank_for(row):
    p = resolve(row['input_npz'])
    expected = row.get('input_sha256') or row.get('input_npz_sha256')
    if expected and sha(p) != expected: raise ValueError('Frozen input file changed')
    with np.load(p, allow_pickle=False) as z:
        bank = {k:z[k].copy() for k in ('z','C','known_mask','valid_mask','N') if k in z.files}
    validate_bank(bank)
    return bank


def verify_frozen(out, requests):
    protocol = read(Path(out)/'protocol.json')
    if sha(Path(out)/'requests.json') != protocol['requests_sha256']:
        raise ValueError('Registered requests differ from frozen protocol')
    if sha(Path(out)/'registration/construction_protocol.json') != protocol['construction_protocol_sha256']:
        raise ValueError('Condition construction contract changed')
    for name, expected in read(Path(out)/'registration/generation_sources_before.json').items():
        if sha(ROOT/name) != expected: raise ValueError('Executed generation source changed: '+name)
    for row in requests:
        p = Path(out)/'outputs'/row['request_id']/'result.json'
        if not p.exists(): continue
        saved = read(p)
        for key in ('request_id','uid','condition_id','context_level','candidate','arm','input_sha256','checkpoint_sha256'):
            if saved.get(key) != row.get(key): raise ValueError('Saved output request identity differs: '+key)
        if saved['status'] in ('CACHED','SAMPLED'):
            if sha(saved['raw_path']) != saved['raw_sha256']: raise ValueError('Completed raw changed')
            if saved['metadata']['nfe'] != 50 or not saved['metadata']['known_all51_bitwise']:
                raise ValueError('Completed sampler contract differs')
        if saved['status'] == 'CACHED':
            if row.get('cache_status') != 'PASS' or saved['raw_sha256'] != row.get('legacy_raw_sha256'):
                raise ValueError('Saved cache is not registration-verified')
        if saved['status'] == 'RUNNING':
            raise RuntimeError('Unresolved output cannot be automatically repeated')
    return dict(status='PASS', requests=len(requests), generation_sources_unchanged=True)


def init_cache_rows(out, requests):
    for row in requests:
        p = Path(out)/'outputs'/row['request_id']/'result.json'
        if p.exists(): continue
        if row.get('status') in ('NOT_APPLICABLE','BLOCKED','NA'):
            write(p,dict(row,status='NOT_APPLICABLE' if row['status']=='NA' else row['status'],model_calls=0)); continue
        oldpath = row.get('legacy_case_json')
        oldraw = row.get('legacy_raw') or row.get('legacy_raw_path')
        if not oldpath or not oldraw or row.get('cache_status') != 'PASS': continue
        old = read(oldpath); rawpath = resolve(oldraw)
        if sha(rawpath) != old['raw_sha256']: raise ValueError('Cache raw hash changed after registration')
        metadata = old.get('metadata',old.get('audit',{}))
        write(p,dict(row,status='CACHED',raw_path=str(rawpath),raw_sha256=old['raw_sha256'],
                     checkpoint_sha256=old['checkpoint_sha256'],metadata=metadata,model_calls=0,
                     source_generation_seconds=metadata.get('seconds'),timing_source='historical actual generation',
                     cache_source_case=str(resolve(oldpath)),cache_identity='registration verified, replay pending'))


def generate(model, audit, row, budget, directory, replay=False):
    p = Path(directory)/'result.json'
    if p.exists():
        previous = read(p)
        if previous['status'] == 'SAMPLED':
            if sha(previous['raw_path']) != previous['raw_sha256']: raise ValueError('Completed raw changed')
            return previous
        if previous['status'] in ('FAILED','RUNNING'): return previous
    bank = bank_for(row)
    rid = ('replay_' if replay else '') + row['request_id']
    context = dict(request_id=rid, arm=row['arm'], uid=row['uid'], candidate=row['candidate'], replay=replay)
    token = budget.begin('sampling',context)
    value = dict(row,status='RUNNING',checkpoint_sha256=audit['checkpoint_sha256'],model_calls=0)
    write(p,value)
    try:
        result = sample_case(model,bank,True,budget,context=context)
        path = Path(directory)/'raw.npz'; path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,raw=result['raw'],states=result['states'],noise=bank['z'],C=bank['C'],known_mask=bank['known_mask'])
        value.update(status='SAMPLED',raw_path=str(path),raw_sha256=sha(path),
                     raw_array_sha256=array_hash(result['raw']),metadata=result['metadata'],model_calls=50,
                     source_generation_seconds=result['metadata']['seconds'],timing_source='new actual generation')
        if replay:
            with np.load(resolve(row['legacy_raw']),allow_pickle=False) as z: old=z['raw']
            value['identity_replay']=dict(bitwise_equal=old.dtype==result['raw'].dtype and old.shape==result['raw'].shape and old.tobytes()==result['raw'].tobytes(),
                                          max_abs=float(np.max(np.abs(old.astype(np.float64)-result['raw']))))
        write(p,value);budget.end(token)
    except BaseException as exc:
        budget.end(token,exc)
        value.update(status='FAILED',error=repr(exc),model_calls=budget.per_request_forwards[rid])
        partial=getattr(exc,'partial_sampling',None)
        if partial:
            arr={k:v for k,v in partial.items() if isinstance(v,np.ndarray)}
            if arr: np.savez_compressed(Path(directory)/'partial.npz',**arr)
        write(p,value);budget.snapshot()
        if isinstance(exc,(torch.cuda.OutOfMemoryError,FloatingPointError)) or any(word in str(exc) for word in ('Nonfinite','nonfinite','Invalid clamped Euler state')): raise
    budget.snapshot()
    return value


def run(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    requests=read(out/'requests.json')
    if len(requests)>80 or len({r['request_id'] for r in requests})!=len(requests): raise ValueError('Request budget/identity differs')
    if not (out/'protocol.json').exists(): raise RuntimeError('Freeze protocol before generation')
    verify_frozen(out,requests)
    init_cache_rows(out,requests)
    runtime.configure_stable_runtime(); torch.set_num_threads(2)
    forbidden=[]
    def reject(*args,**kwargs):
        forbidden.append('BACKWARD_OR_UPDATE');raise RuntimeError('Zero-backward/zero-training contract')
    torch.Tensor.backward=reject;torch.autograd.backward=reject;torch.autograd.grad=reject;torch.optim.AdamW.step=reject
    budget=Budget(out);budget.snapshot()
    with open(LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        event(out/'ledger/lifecycle.jsonl',dict(event='GPU_LOCK_ACQUIRED',path=LOCK,pid=os.getpid()))
        for arm in ('OT_BASE','OT_HYBRID'):
            jobs=[r for r in requests if r['arm']==arm and r.get('status') not in ('NOT_APPLICABLE','BLOCKED','NA')]
            if not jobs: continue
            todo=[r for r in jobs if not (out/'outputs'/r['request_id']/'result.json').exists()]
            first=min((r for r in jobs if r['candidate']==0 and r.get('panel')=='main'),key=lambda r:r['uid'])
            replaypath=out/'replays'/arm/'result.json'
            if not todo and replaypath.exists(): continue
            before=resource_snapshot(out,'before_load_'+arm)
            headroom=max(2048,.1*(before['total_mib'] or 16303))
            if before['free_mib'] is None or before['free_mib'] < headroom+1.2*2048:
                raise RuntimeError('Insufficient observed free VRAM; wait rather than force dispatch')
            model,audit=load_model(arm,out)
            loaded=resource_snapshot(out,'loaded_'+arm)
            replay=generate(model,audit,first,budget,out/'replays'/arm,replay=True)
            resource_snapshot(out,'representative_first_rollout_'+arm)
            if replay['status']!='SAMPLED' or not replay['identity_replay']['bitwise_equal']:
                write(out/'replays'/arm/'MISMATCH.json',dict(status='BLOCKED',reason='Historical/current generation mismatch; no cache mixture permitted',replay=replay))
                raise RuntimeError('Identity replay differs; inspect execution path before generation')
            for row in jobs:
                p=out/'outputs'/row['request_id']/'result.json'
                if p.exists():
                    old=read(p)
                    if old['status']=='CACHED' and old.get('cache_identity','').endswith('pending'):
                        old['cache_identity']='registration verified; first-parent noise0 endpoint replay bitwise PASS';write(p,old)
                    continue
                value=generate(model,audit,row,budget,p.parent)
                print(json.dumps(dict(arm=arm,request=row['request_id'],status=value['status'],seconds=value.get('metadata',{}).get('seconds'))),flush=True)
            if state_sha256(model)!=audit['model_state_sha256']: raise ValueError('Frozen model state changed')
            event(out/'ledger/lifecycle.jsonl',dict(event='FROZEN_MODEL_VERIFIED',arm=arm,model_state_sha256=audit['model_state_sha256']))
            del model;gc.collect();torch.cuda.empty_cache()
            resource_snapshot(out,'completed_'+arm)
        event(out/'ledger/lifecycle.jsonl',dict(event='GPU_WORK_FINISHED',forbidden_attempts=forbidden))
    return budget.snapshot()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();print(json.dumps(run(args.out),indent=2))
