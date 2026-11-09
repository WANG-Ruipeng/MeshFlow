"""Bounded evaluation workers; immutable inputs, original Euler50 and scorer."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import gc
from pathlib import Path
import time
import traceback
import numpy as np
from .common import atomic_json, read_json, digest, json_hash, GPU_LOCK, LIMITS

CURVE_MODELS = ['Native0', 'T1_500', 'Geo_shared_500'] + [f'{a}_{s}' for s in (1000,2000,3500,5000) for a in ('T1','Geo','Full')]
SWAP_MODELS = ['Native0'] + [f'{a}_{s}' for s in (1000,5000) for a in ('T1','Geo','Full')]
FIT_MODELS = ['T1_1000','T1_fit2_probe_500','T1_fit2_probe_1000']
PHASE_KEYS = dict(curve=('generation','teacher'), swap=('swap',None), fit=('fit_generation','fit_teacher'), holdout=('holdout',None))

def resolve(out, value):
    path = Path(value)
    return path if path.is_absolute() else Path(out)/path

def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def save_npz(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
    return digest(path)

def _verified(out, record, key='input_path', hash_key='input_sha256'):
    path = resolve(out, record[key])
    if hash_key in record and digest(path) != record[hash_key]:
        raise ValueError('Frozen file changed: '+str(path))
    return arrays(path)

def checkpoint_path(source, out, model_id):
    if model_id == 'Native0':
        return Path(source)/'checkpoints/official_chair_last.pt'
    arm, step = model_id.rsplit('_',1)
    base = source if int(step) <= 1000 and arm != 'T1_fit2_probe' else out
    return Path(base)/'checkpoints'/f'{arm}_step{step}.pt'

def load_model(source, out, model_id, device='cuda:0'):
    from experiments.mf_h1_local0.control_mvp48.model_mvp import initialize_mvp, load_mvp_checkpoint, state_sha256
    path = checkpoint_path(source,out,model_id)
    if model_id == 'Native0':
        result = initialize_mvp(path, geo=False, device=device)
        model = result[0] if isinstance(result, tuple) else result
        model.set_trainable(False).eval()
        return model, dict(variant='Native0',logical_step=0), dict(checkpoint_sha256=digest(path),model_state_sha256=state_sha256(model))
    if path.parent.parent.resolve() == Path(source).resolve():
        return load_mvp_checkpoint(path, device)
    from .checkpoint import load_checkpoint
    return load_checkpoint(path, device=device, trainable=False)

def _request_id(record, teacher=False):
    return str(record.get('teacher_id' if teacher else 'case_id',record.get('id')))

def _bank_reference(out, record):
    ref = _verified(out,record,'ref_npz','ref_npz_sha256')
    prefix=record.get('reference_key_prefix')
    if prefix: ref={k[len(prefix):]:v for k,v in ref.items() if k.startswith(prefix)}
    meta_path = resolve(out,record['ref_json'])
    if record.get('ref_json_sha256') and digest(meta_path) != record['ref_json_sha256']:
        raise ValueError('Frozen reference metadata changed')
    return dict(arrays=ref,audit=read_json(meta_path))

def score_record(out, record, raw, *, phase):
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import score_free
    bank = _verified(out,record)
    reference = _bank_reference(out,record)
    # Holdout input banks are the actual source arrays. Their scoring-only bank
    # is separate, so no GT or bbox is ever passed into model forward.
    if 'gt_free' not in bank:
        extra = arrays(resolve(out,record.get('target_path',record['ref_npz'])))
        bank.update(extra)
    K, N = len(bank['C']), int(bank['N'])
    C = bank.get('recipient_C',bank['C']) if phase == 'swap' else bank['C']
    target = bank['gt_free']; L = float(bank['L'])
    if phase == 'swap' and 'recipient_input_path' in record:
        receiver = arrays(resolve(out,record['recipient_input_path']))
        C = receiver['C']
    preserved = raw[:K].reshape(K,3,3).tobytes() == bank['C'].reshape(K,3,3).tobytes()
    scored = score_free(raw[K:], C, target, L, reference=reference, include_pair_audit=True)
    scored['metrics']['C_preserved'] = preserved
    scored['detail'].update(actual_condition_preserved=preserved,swap_request=phase=='swap',
        CF_interpretation='recipient C versus free output; wrong donor is not a valid completion target' if phase=='swap' else 'actual C versus free output')
    if raw.shape != (N,9) or not preserved:
        scored.update(status='INVALID_OUTPUT',failure_reason='N_OR_C_MISMATCH')
    return scored

def teacher_once(model, bank, record, counts):
    import torch
    from native_t1.model import native_execution, assert_native_runtime
    from native_t1.artifacts import array_hash
    from experiments.mf_h1_local0.control_mvp48.eval_teacher import offline_teacher_diagnostics
    keys = ('xt','t','y','valid_mask','known_mask')
    identity = dict(record.get('identity',{}))
    N = int(bank['y']); K = int(np.asarray(bank['known_mask']).sum())
    identity.update(N=N,K=K,bbox_L=float(identity.get('bbox_L',bank.get('L',1))))
    if float(bank['t']) not in [float(np.float32(t)) for t in (.2,.5,.8)]:
        raise ValueError('Unregistered teacher time')
    for key,expected in identity.get('array_sha256',{}).items():
        if key in bank and array_hash(bank[key]) != expected:
            raise ValueError('Teacher array identity changed: '+key)
    args = {k:torch.from_numpy(np.array(bank[k],copy=True)).unsqueeze(0).to('cuda:0') for k in keys}
    before = dict(counts); start = time.perf_counter(); handles=[]
    def inc(k):
        def hook(*unused): counts[k] += 1
        return hook
    handles += [model.register_forward_pre_hook(inc('model_forward_attempts')),model.register_forward_hook(inc('model_forward_returns')),
        model.backbone.x_embedder.register_forward_pre_hook(inc('backbone_forward_attempts')),
        model.backbone.final_layer.linear.register_forward_hook(inc('backbone_forward_returns'))]
    counts['teacher_attempts'] += 1
    partial={}
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), native_execution(model,coordinates=(args['xt'],args['t'])) as facts:
            v = model(*(args[k] for k in keys)).float()[0].cpu().numpy().copy()
            after = assert_native_runtime(model,coordinates=(args['xt'],args['t']))
        partial['velocity']=v
        for key in keys:
            if array_hash(args[key][0].cpu().numpy())!=array_hash(bank[key]): raise ValueError('Teacher model mutated input '+key)
        if not np.isfinite(v).all(): raise FloatingPointError('Nonfinite teacher velocity')
        result, endpoint = offline_teacher_diagnostics(bank,identity,v)
        counts['teacher_returns'] += 1
        result.update(status='PASS',teacher_t=float(bank['t']),runtime_before=facts,runtime_after=after,
            nfe=counts['backbone_forward_attempts']-before['backbone_forward_attempts'],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_scope='one teacher forward after reset including resident model',seconds=time.perf_counter()-start)
        if result['nfe'] != 1: raise RuntimeError('Teacher one-forward count mismatch')
        # Original auxiliary formulas on CPU predictions; no model/geometry backward.
        if all(k in bank for k in ('free_vertex_ids','known_vertex_ids')):
            from native_t1.losses import prepare_targets, evaluate
            target = prepare_targets(dict(free_gt=bank['x1'][~bank['known_mask']].reshape(-1,3,3),
                known=bank['C'].reshape(-1,3,3),free_ids=bank['free_vertex_ids'],known_ids=bank['known_vertex_ids'],bbox_L=identity['bbox_L']))
            pred=torch.from_numpy(endpoint[~bank['known_mask']].reshape(-1,3,3))
            with torch.no_grad():
                for key in ('D_edge','L5'):
                    result[key]=float(evaluate(key,pred,target[key])['loss'])
        else:
            result.update(D_edge=None,L5=None,auxiliary_status='NOT_AVAILABLE_SOURCE_VERTEX_LABELS')
        return result,dict(velocity=v,Yhat=endpoint)
    except BaseException as exc:
        exc.partial_teacher_arrays=partial
        raise
    finally:
        for handle in handles: handle.remove()

def _global_attempts(out):
    gs=ts=0
    for path in (Path(out)/'evaluation').glob('*/*/ledger.json'):
        data=read_json(path)
        gs+=sum(bool(x.get('attempt_reserved')) for x in data.get('rows',[]))
        ts+=sum(bool(x.get('attempt_reserved')) for x in data.get('teacher_rows',[]))
    return gs,ts

def run(source,out,model_id,phase,score_only=False):
    source,out=Path(source),Path(out)
    allowed = CURVE_MODELS if phase=='curve' else SWAP_MODELS if phase=='swap' else FIT_MODELS if phase=='fit' else [f'{a}_5000' for a in ('T1','Geo','Full')]
    if model_id not in allowed: raise ValueError('Unregistered model/phase')
    manifest=read_json(out/'panels.json'); key,tkey=PHASE_KEYS[phase]
    for frozen_path,expected_sha in manifest.get('frozen_files',{}).items():
        if digest(resolve(out,frozen_path))!=expected_sha: raise ValueError('Frozen panel file changed: '+frozen_path)
    records=manifest[key]; teachers=manifest[tkey] if tkey else []
    expected = {'curve':(64,48),'swap':(16,0),'fit':(8,6),'holdout':(32,0)}[phase]
    if len(records)>expected[0] or len(teachers)!=expected[1]: raise ValueError('Panel count exceeds fixed protocol')
    folder=out/'evaluation'/phase/model_id; target=folder/'ledger.json'
    signature=dict(model_id=model_id,phase=phase,panels_sha256=digest(out/'panels.json'),checkpoint=str(checkpoint_path(source,out,model_id)),
        checkpoint_sha256=digest(checkpoint_path(source,out,model_id)) if checkpoint_path(source,out,model_id).exists() else None)
    ledger=read_json(target) if target.exists() else dict(schema='mvp48_budget_evaluation_v1',identity=signature,status='PREPARED',rows=[],teacher_rows=[],sampling_counts={},teacher_counts={},actual_gpu_workers=1)
    if ledger['identity']!=signature: raise ValueError('Existing evaluation identity differs')
    if any(r.get('status') in ('RUNNING','FAIL','STOPPED') for r in ledger['rows']+ledger['teacher_rows']):
        raise RuntimeError('An interrupted/failed attempt exists; automatic replay prohibited')
    if signature['checkpoint_sha256'] is None:
        if model_id not in ('Native0','T1_500','Geo_shared_500'): raise FileNotFoundError(signature['checkpoint'])
        ledger.update(status='NOT_AVAILABLE',reason='Historical checkpoint absent; no historical retraining');atomic_json(target,ledger);return ledger
    def persist(): atomic_json(target,ledger)
    def score_pending():
        for row in ledger['rows']:
            if row.get('status')!='PASS' or row.get('score_path'): continue
            record=next(r for r in records if _request_id(r)==row['case_id'])
            raw_path=resolve(out,row['raw_path'])
            if digest(raw_path)!=row['raw_sha256']: raise ValueError('Raw hash differs')
            scoring_started=time.perf_counter()
            scored=score_record(out,record,arrays(raw_path)['raw'],phase=phase)
            row['scoring_seconds']=time.perf_counter()-scoring_started
            loc=folder/'scores'/(row['case_id']+'.json');atomic_json(loc,scored)
            row.update(score_path=str(loc.relative_to(out)),score_sha256=digest(loc),score_status=scored['status']);persist()
    if score_only:
        score_pending(); return ledger
    if ledger['status']=='PASS':
        score_pending();return ledger
    if ledger['status']=='GPU_COMPLETE':
        score_pending();ledger['status']='PASS';persist();return ledger
    import fcntl
    from experiments.mf_h1_local0.control_mvp48.eval_common import snapshot
    from experiments.mf_h1_local0.control_mvp48.eval_gpu_worker import resources_ok
    lock=open(GPU_LOCK,'a+');started=time.perf_counter()
    ledger.update(status='WAITING_GPU_LOCK',pid=os.getpid());persist()
    try:
        while True:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: time.sleep(5);continue
            facts=snapshot();ok,admission=resources_ok(facts,initial=True)
            ledger.update(resources_before=facts,admission=admission);persist()
            if ok:break
            fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(5)
        ledger['lock_wait_seconds']=time.perf_counter()-started
        import torch
        from experiments.mf_h1_local0.control_mvp48.eval_runtime import new_counts,sample_euler50,SamplingFailure
        from native_t1.artifacts import array_hash
        from experiments.mf_h1_local0.control_mvp48.model_mvp import state_sha256
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        ledger['sampling_counts']=ledger['sampling_counts'] or new_counts();ledger['teacher_counts']=ledger['teacher_counts'] or new_counts()
        load_started=time.perf_counter()
        model,metadata,hashes=load_model(source,out,model_id)
        ledger['model_load_seconds']=time.perf_counter()-load_started
        ledger.update(model_metadata=metadata,model_hashes=hashes,status='RUNNING');persist()
        for teacher,requests,rows,counts in ((False,records,ledger['rows'],ledger['sampling_counts']),(True,teachers,ledger['teacher_rows'],ledger['teacher_counts'])):
            existing={r['case_id']:r for r in rows}
            for ordinal,record in enumerate(requests):
                rid=_request_id(record,teacher)
                if rid in existing:continue
                if record.get('status')=='NOT_AVAILABLE':
                    rows.append(dict(case_id=rid,status='NOT_AVAILABLE',reason=record.get('reason'),attempt_reserved=False));persist();continue
                gs,ts=_global_attempts(out)
                if gs+(not teacher)>LIMITS['generations'] or ts+teacher>LIMITS['teacher_forwards']: raise RuntimeError('Global evaluation budget exhausted')
                facts=snapshot();ok,admission=resources_ok(facts,initial=False)
                ledger.setdefault('phase_resource_snapshots',[]).append(dict(phase=('teacher' if teacher else 'generation'),request_id=rid,snapshot=facts,admission=admission))
                if not ok: raise RuntimeError('GPU headroom lost')
                bank=_verified(out,record)
                row=dict(case_id=rid,model_id=model_id,phase=phase,uid=record.get('uid',record.get('parent_id')),patch_id=record.get('patch_id',record.get('condition_id')),
                    split=record.get('split'),noise_index=record.get('noise_index',record.get('noise_id')),teacher_t=record.get('t'),status='RUNNING',attempt_reserved=True,input_sha256=digest(resolve(out,record['input_path'])))
                rows.append(row);persist()
                before=dict(counts);raws={}
                try:
                    if teacher:
                        result,raws=teacher_once(model,bank,record,counts);row.update(result)
                    else:
                        z=torch.from_numpy(bank['z']).to('cuda:0');C=torch.from_numpy(bank['C']).to('cuda:0')
                        raw,path,audit=sample_euler50(model,z,C,counts)
                        raws=dict(raw=raw,noise=bank['z'],C=bank['C'],known_mask=bank['known_mask'])
                        if ordinal==0:raws['states']=path
                        row.update(status='PASS',audit=audit,output_array_sha256={k:array_hash(v) for k,v in raws.items()})
                    loc=folder/('teacher_raw' if teacher else 'raw')/(rid+'.npz')
                    sha=save_npz(loc,raws);row.update(raw_path=str(loc.relative_to(out)),raw_sha256=sha)
                except SamplingFailure as exc:
                    loc=folder/'failed_raw'/(rid+'.npz');sha=save_npz(loc,dict(states=exc.partial_states))
                    row.update(status='FAIL',reason=str(exc),audit=exc.audit,raw_path=str(loc.relative_to(out)),raw_sha256=sha);raise
                except BaseException as exc:
                    if getattr(exc,'partial_teacher_arrays',None):
                        loc=folder/'failed_teacher_raw'/(rid+'.npz');sha=save_npz(loc,exc.partial_teacher_arrays)
                        row.update(raw_path=str(loc.relative_to(out)),raw_sha256=sha)
                    row.update(status='FAIL',reason=repr(exc));raise
                finally:
                    row['counts_delta']={k:counts[k]-before[k] for k in counts};persist()
        if state_sha256(model)!=hashes['model_state_sha256']: raise RuntimeError('Evaluation changed model parameters')
        del model;gc.collect();torch.cuda.empty_cache()
        ledger.update(status='GPU_COMPLETE',resources_after=snapshot(),worker_seconds=time.perf_counter()-started);persist()
    except BaseException as exc:
        ledger.update(status='STOPPED',reason=repr(exc),traceback=traceback.format_exc(),worker_seconds=time.perf_counter()-started);persist();raise
    finally:
        # Keep flock until process exit, including the remaining CUDA context.
        global _PROCESS_LOCK
        _PROCESS_LOCK=lock
    score_pending()
    ledger.update(status='PASS',completed_generations=ledger['sampling_counts']['complete_rollouts'],requested_generations=len(records),completed_teacher=ledger['teacher_counts']['teacher_returns'],requested_teacher=len(teachers),worker_seconds=time.perf_counter()-started);persist()
    return ledger

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--phase',required=True,choices=list(PHASE_KEYS));p.add_argument('--score-only',action='store_true')
    a=p.parse_args();run(a.source,a.out,a.model_id,a.phase,a.score_only)

if __name__=='__main__':main()
