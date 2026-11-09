"""Explicit registered pure-DiT jobs; completed entry returns before torch import.

Only frozen official EMA, scalar-time count CFG, and current-state VJP are used.
No Native network is constructed. Every model/VJP call is journaled before it
runs. A crash after a begun call without a committed step cannot be replayed.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_full_prior_motif_guidance_v1'
OFFICIAL = ROOT/'output/120m-ot-v-bench/checkpoints/01000000.pt'
OFFICIAL_SHA = '00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505'
EMA_SHA = '2cc63f56d79b88418840fc44ad4c954d1961ad898a7b81d198a35bf49f0deb25'
SCHEMA = 'bench_full_prior_motif_guidance_raw_v1'


def resolve(path):
    value = str(path).replace('\\', '/')
    if os.name == 'nt' and value.startswith('/mnt/e/'):
        value = 'E:/' + value[7:]
    elif os.name != 'nt' and value.lower().startswith('e:/'):
        value = '/mnt/e/' + value[3:]
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = resolve(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def save_npz(path, arrays):
    import numpy as np
    path = resolve(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name+'.partial')
    with temp.open('xb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    # Only new output files in this experiment; existing scientific files never replaced.
    if path.exists():
        raise FileExistsError(path)
    temp.rename(path)


def arrays(path):
    import numpy as np
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def verify(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('File identity changed: '+str(path))


def journal(path, event, **data):
    row = dict(event=event, time_unix=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **data)
    with resolve(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    return row


def journal_rows(path):
    path = resolve(path)
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]


def code_hashes():
    paths = [Path(__file__), Path(__file__).with_name('guidance.py'), ROOT/'models/equidit.py',
             ROOT/'models/attention.py', ROOT/'models/utils.py', ROOT/'r2/src/meshflow_control/runtime.py',
             ROOT/'r2/src/meshflow_control/source_winding.py',
             ROOT/'r2/pilots/seam_slot_c200_v1/generate.py',
             ROOT/'r2/prototypes/seam_slot_c200_v1/corner_contract.py']
    return {str(p):sha(p) for p in paths}


def verify_receipt(output_dir, *, registration_sha256=None, job_id=None):
    result = read(resolve(output_dir)/'receipt.json')
    if result.get('status') != 'COMPLETE' or result.get('schema') != SCHEMA:
        raise ValueError('Not a completed registered RAW')
    if registration_sha256 is not None and result['registration_sha256'] != registration_sha256:
        raise ValueError('Completion registration changed')
    if job_id is not None and result['job_id'] != job_id:
        raise ValueError('Completion job identity changed')
    verify(result['files']); verify(result['input_hashes']); verify(result['code_hashes'])
    if result.get('generator_forwards')!=50 or result.get('D0_forwards')!=0 or result.get('steps')!=50:
        raise ValueError('Completed request budget differs')
    _audit_calls(journal_rows(resolve(output_dir)/'calls.jsonl'),result['arm'],50)
    commits=[r for r in journal_rows(resolve(output_dir)/'calls.jsonl') if r['event']=='STEP_COMMIT']
    if [r['step'] for r in commits]!=list(range(50)):
        raise ValueError('Completed request lacks exactly 50 committed updates')
    for row in commits:verify(row['files'])
    return result


def _request_assets(q):
    aliases = dict(inference_npz=('inference_npz_sha256', 'input_sha256'),
                   source_npz=('source_npz_sha256', 'source_sha256'),
                   noise_npz=('noise_npz_sha256', 'noise_sha256'),
                   shadow_noise_npz=('shadow_noise_npz_sha256', 'shadow_noise_sha256'))
    identities = {}
    for field, keys in aliases.items():
        digest = next((q[key] for key in keys if q.get(key)), None)
        if digest is None:
            raise ValueError('Missing frozen hash: '+field)
        path = resolve(q[field])
        if sha(path) != digest:
            raise ValueError('Frozen input changed: '+field)
        identities[str(path)] = digest
    return identities


def _load_input(q):
    import numpy as np
    from meshflow_control.source_winding import validate_source
    identities = _request_assets(q)
    inp, source = arrays(q['inference_npz']), arrays(q['source_npz'])
    if set(inp) != {'C_vertices','C_faces_original','C_faces_model','N','K'}:
        raise ValueError('Non-C fields in inference bundle')
    n, k = int(inp['N']), int(inp['K'])
    if (n,k) != (int(q['N']),int(q['K'])) or not 0 < k < n <= 799:
        raise ValueError('Input N/K outside frozen official-count contract')
    condition = inp['C_vertices'][inp['C_faces_model']]
    if condition.dtype != np.float32 or condition.shape != (k,3,3) or not np.isfinite(condition).all():
        raise ValueError('Condition precision/shape differs')
    validate_source(source, condition)
    zf = arrays(q['noise_npz'])['z']; shadow = arrays(q['shadow_noise_npz'])
    zs = shadow['z'] if 'z' in shadow else shadow['z_S']
    for z, shape in ((zf,(n-k,3,3)),(zs,(k,3,3))):
        if z.dtype != np.float32 or z.shape != shape or not np.isfinite(z).all():
            raise ValueError('Gaussian identity/shape/precision differs')
    initial = np.concatenate((zs,zf),axis=0)
    if initial[k:].tobytes() != zf.tobytes():
        raise AssertionError('Free noise changed during concatenation')
    return inp, source, condition, initial, identities


def load_official_model(path=OFFICIAL, expected_sha=OFFICIAL_SHA):
    """Called only by explicit GPU worker after dispatch, never completion check."""
    import torch
    from models.equidit import DiT
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    if expected_sha != OFFICIAL_SHA or resolve(path) != resolve(OFFICIAL) or sha(path) != OFFICIAL_SHA:
        raise ValueError('Only the registered official Bench file is authorized')
    runtime.configure_stable_runtime()
    payload = torch.load(resolve(path), map_location='cpu', weights_only=True, mmap=True)
    config = payload['config']; m = dict(config['model'])
    checkpoint_gradient_checkpointing=m.get('gradient_checkpointing')
    m['gradient_checkpointing']=False
    expected = dict(model_type='equidit', use_qknorm=True, use_rmsnorm=True, hidden_dim=768,
        num_heads=12,max_length=800,num_layers=12,gradient_checkpointing=False,
        use_coord_encoding=True,version=3,pe_freq=20,mixed_precision='bf16',
        use_dit_like_pe=False,face_cond=True,face_bin=20)
    if m != expected or '02828884' not in str(config.get('data',{}).get('data_path','')):
        raise ValueError('Unexpected official Bench model config')
    state = payload['ema']; prefixed = [name.startswith('module.') for name in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('Mixed official EMA key prefixes')
    state = {(name[7:] if all(prefixed) else name):value for name,value in state.items()}
    if state_hash(state) != EMA_SHA:
        raise ValueError('Official EMA tensor identity differs')
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Official parameters must be finite FP32')
    # Isolate initialization RNG. No scientific noise is drawn here.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261029)
        model = DiT(**{key:value for key,value in m.items() if key != 'model_type'})
    model.load_state_dict(state,strict=True); model.requires_grad_(False); model.eval()
    del state, payload
    model.to('cuda')
    identity = dict(path=str(resolve(path)),sha256=OFFICIAL_SHA,ema_state_sha256=EMA_SHA,
                    pure_official_DiT=True,Native_role_Geo=False,optimizer=False,
                    checkpoint_gradient_checkpointing=checkpoint_gradient_checkpointing,
                    effective_gradient_checkpointing=False,hidden_forward_recomputations=False,
                    count_CFG_arithmetic='official BF16, then cast velocity to FP32',tokens_per_face=3)
    return model, identity


def cfg_velocity(model, state, step):
    import torch
    from .guidance import CFG_SCALE, STEPS, finite
    n = len(state)
    x = state.reshape(1,n,9)
    double = torch.cat((x,x),dim=0)
    t = torch.full((2,),step/STEPS,dtype=torch.float32,device=state.device)
    y = torch.tensor([n],dtype=torch.int64,device=state.device)
    mask = torch.ones((1,n),dtype=torch.bool,device=state.device)
    result = model.forward_with_cfg(double,t,y,mask=mask,cfg_scale=CFG_SCALE)
    if result.shape != (2,n,9) or result.dtype!=torch.bfloat16:
        raise ValueError('Official CFG shape/BF16 output differs')
    finite(result,'both CFG outputs')
    if not torch.equal(result[:1],result[1:]):
        raise ValueError('CFG duplicate halves disagree')
    return result[0].reshape(n,3,3).float()


def _accounted_call(ledger, kind, step, fn):
    import torch
    journal(ledger,'CALL_BEGIN',kind=kind,step=step)
    tick = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    journal(ledger,'CALL_END',kind=kind,step=step,seconds=time.perf_counter()-tick)
    return result


def _audit_calls(rows,arm,cursor):
    expected=[]
    for step in range(cursor):
        expected.append(('vector_field',step))
        if arm in ('G2','G3') and step>=5:expected.append(('input_VJP',step))
    for event in ('CALL_BEGIN','CALL_END'):
        actual=[(r['kind'],r['step']) for r in rows if r['event']==event]
        if actual!=expected:
            raise RuntimeError('Uncommitted, missing, or extra '+event+'; scientific replay forbidden')


def _resume_cursor(dest, identity, initial, *, allow_resume):
    import numpy as np
    ledger = dest/'calls.jsonl'
    if not (dest/'attempt.json').exists():
        if any(dest.iterdir()):
            raise RuntimeError('Uncommitted request files; no silent restart')
        write_new(dest/'attempt.json',identity)
        save_npz(dest/'states/state_000.npz',dict(state=initial,completed_updates=np.asarray(0)))
        journal(ledger,'INITIAL_STATE_COMMIT',path=str(dest/'states/state_000.npz'),sha256=sha(dest/'states/state_000.npz'))
        return 0, initial.copy()
    if not allow_resume:
        raise RuntimeError('Existing unfinished request requires explicit --resume')
    if read(dest/'attempt.json') != identity:
        raise ValueError('Resume identity differs')
    rows = journal_rows(ledger)
    starts = [r for r in rows if r['event']=='INITIAL_STATE_COMMIT']
    if len(starts)!=1:
        raise RuntimeError('No unique committed initial state')
    verify({starts[0]['path']:starts[0]['sha256']})
    if not np.array_equal(arrays(starts[0]['path'])['state'],initial):
        raise ValueError('Initial noise/clamp differs')
    commits = [r for r in rows if r['event']=='STEP_COMMIT']
    if [r['step'] for r in commits] != list(range(len(commits))):
        raise RuntimeError('Noncontiguous committed steps')
    for row in commits:
        verify(row['files'])
    cursor = len(commits)
    if cursor>50:raise RuntimeError('More than 50 committed updates')
    _audit_calls(rows,identity['arm'],cursor)
    for row in rows:
        if row['event'] in ('CALL_BEGIN','CALL_END') and row['step'] >= cursor:
            raise RuntimeError('A model/VJP call began beyond last safe commit; replay forbidden')
    # Stray output after a call is not silently promoted to a safe checkpoint.
    if (dest/f'states/state_{cursor+1:03}.npz').exists():
        raise RuntimeError('Uncommitted next state exists; manual audit required')
    state = arrays(dest/f'states/state_{cursor:03}.npz')['state']
    return cursor,state


def _final_arrays(final, inp, source):
    import numpy as np
    from meshflow_control.seam.slot_data import boundary_from_C
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
    n,k = int(inp['N']),int(inp['K']); anchors = np.full((n,3),-1,np.int64)
    anchors[:k] = inp['C_faces_model']; edges = boundary_from_C(inp['C_vertices'],inp['C_faces_original'])
    layout = dict(N=n,K=k,anchor_ids=anchors,known_mask=np.arange(n)<k,valid_mask=np.ones(n,bool),
        occupied=np.zeros(len(edges),bool),edge_records=edges,seam_slots=np.empty(0,np.int64),
        C_vertices=inp['C_vertices'],C_faces_original=inp['C_faces_original'],C_faces_model=inp['C_faces_model'])
    raw = final.copy(); raw[:k] = inp['C_vertices'][inp['C_faces_model']]
    mesh = raw_arrays(raw,layout,np.full(len(edges),-1,np.float32),source)
    mesh['raw_slot_edge_id'] = np.full(n,-1,np.int64)
    mesh['guidance_internal_C_moving'] = np.asarray(True)
    if mesh['C'].tobytes()!=source['C_original'].tobytes() or raw[k:].tobytes()!=final[k:].tobytes():
        raise ValueError('Final C restoration/free preservation failed')
    if mesh['raw_seam_slot_mask'].any() or mesh['retained_seam_mask'].any() or mesh['occupied'].any():
        raise ValueError('Official prior must not invent SEAM roles')
    return mesh


def run_request(model, request, arm, lambda_value, output_dir, registration_sha256, *,
                job_id=None, model_identity=None, allow_resume=False, stop_requested=None):
    from .guidance import validate_arm
    validate_arm(arm,lambda_value)
    dest = resolve(output_dir); job_id = job_id or arm+'__'+request['request_id']
    if (dest/'receipt.json').exists():
        return verify_receipt(dest,registration_sha256=registration_sha256,job_id=job_id)
    import numpy as np
    import torch
    from meshflow_control import runtime
    from meshflow_control.geo_io import write_indexed_ply
    from .guidance import STEPS, is_guided, step_update
    if model is not None:
        if type(model).__module__ != 'models.equidit' or type(model).__name__ != 'DiT':
            raise ValueError('Only pure official models.equidit.DiT is allowed')
        if any(p.requires_grad or p.grad is not None or p.dtype!=torch.float32 for p in model.parameters()):
            raise ValueError('Model must be frozen FP32 without accumulated gradients')
        if model.training or model.uncond_y!=801 or model.face_bin!=20 or next(model.parameters()).device.type!='cuda':
            raise ValueError('Official CUDA model/count/eval contract differs')
    elif (dest/'attempt.json').exists():
        model_identity=read(dest/'attempt.json')['model_identity']
    else:
        raise ValueError('CPU finalization requires an existing committed attempt')
    inp,source,c,initial,input_hashes = _load_input(request)
    if arm=='G1': initial[:len(c)] = c
    code = code_hashes()
    identity = dict(schema=SCHEMA,job_id=job_id,request_id=request['request_id'],arm=arm,
        lambda_value=lambda_value,registration_sha256=registration_sha256,input_hashes=input_hashes,
        code_hashes=code,model_identity=model_identity or dict(sha256=OFFICIAL_SHA,ema_state_sha256=EMA_SHA),
        N=int(inp['N']),K=int(inp['K']),steps=50,expected_VJP=45 if arm in ('G2','G3') else 0)
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'worker.lock').open('x',encoding='utf-8') as f:
        f.write(str(os.getpid()))
    ledger = dest/'calls.jsonl'
    entered=False
    # lock is coordination metadata, not part of scientific recovery identity.
    try:
        if not (dest/'attempt.json').exists():
            extras = [p for p in dest.iterdir() if p.name!='worker.lock']
            if extras: raise RuntimeError('Uncommitted request directory')
            write_new(dest/'attempt.json',identity)
            save_npz(dest/'states/state_000.npz',dict(state=initial,completed_updates=np.asarray(0)))
            journal(ledger,'INITIAL_STATE_COMMIT',path=str(dest/'states/state_000.npz'),sha256=sha(dest/'states/state_000.npz'))
            cursor, value = 0,initial
        else:
            cursor,value = _resume_cursor(dest,identity,initial,allow_resume=allow_resume)
        if cursor<50 and model is None:
            raise RuntimeError('CPU-only finalization needs all 50 safely committed updates')
        entered=True
        if cursor<50:
            condition = torch.from_numpy(c.copy()).to('cuda')
            state = torch.from_numpy(value.copy()).to('cuda')
            rng_cpu = torch.get_rng_state().clone(); rng_cuda = torch.cuda.get_rng_state().clone()
            torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
            journal(ledger,'RUN_SEGMENT_BEGIN',start_step=cursor)
        for step in range(cursor,STEPS):
            if stop_requested is not None and stop_requested():
                journal(ledger,'SAFE_STOP',completed_updates=step)
                return dict(status='SAFE_INTERRUPTED',job_id=job_id,completed_updates=step)
            guided = is_guided(arm,step)
            state = state.detach().requires_grad_(guided)
            with torch.set_grad_enabled(guided),runtime.stable_execution(model):
                velocity = _accounted_call(ledger,'vector_field',step,lambda:cfg_velocity(model,state,step))
                callback = (lambda e,x: _accounted_call(ledger,'input_VJP',step,
                    lambda:torch.autograd.grad(e,x,create_graph=False,retain_graph=False)[0])) if guided else None
                next_state,predicted,diagnostic = step_update(state,velocity,condition,step,arm,lambda_value,vjp=callback)
            value = next_state.cpu().numpy().copy()
            if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_cuda,torch.cuda.get_rng_state()):
                raise RuntimeError('Sampling unexpectedly consumed RNG')
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Parameter gradients were populated')
            sp = dest/f'states/state_{step+1:03}.npz'; dp = dest/f'steps/step_{step:03}.json'
            save_npz(sp,dict(state=value,endpoint_prediction=predicted.cpu().numpy(),completed_updates=np.asarray(step+1)))
            diagnostic.update(completed_updates=step+1,cumulative_generator_forwards=step+1,
                              cumulative_cfg_network_sample_equivalents=2*(step+1),
                              cumulative_input_VJP=max(0,step-4) if arm in ('G2','G3') else 0,
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              peak_scope='current run segment, resident weights plus input-gradient activations')
            write_new(dp,diagnostic)
            journal(ledger,'STEP_COMMIT',step=step,completed_updates=step+1,
                cumulative_generator_forwards=step+1,cumulative_input_VJP=diagnostic['cumulative_input_VJP'],
                state_sha256=sha(sp),files={str(sp):sha(sp),str(dp):sha(dp)})
            state = next_state
            del velocity,predicted,callback
        if cursor<50:journal(ledger,'RUN_SEGMENT_END',completed_updates=50,seconds=time.perf_counter()-tick)
        rows = journal_rows(ledger)
        ends = [r for r in rows if r['event']=='CALL_END']
        starts = [r for r in rows if r['event']=='CALL_BEGIN']
        if [(r['kind'],r['step']) for r in starts] != [(r['kind'],r['step']) for r in ends]:
            raise RuntimeError('Unpaired call journal')
        vf = [r['step'] for r in ends if r['kind']=='vector_field']
        vjp_steps = [r['step'] for r in ends if r['kind']=='input_VJP']
        if vf!=list(range(50)) or vjp_steps!=(list(range(5,50)) if arm in ('G2','G3') else []):
            raise RuntimeError('Actual network/VJP budget differs')
        _audit_calls(rows,arm,50)
        trajectory = np.stack([arrays(dest/f'states/state_{i:03}.npz')['state'] for i in range(51)])
        final = trajectory[-1]; mesh = _final_arrays(final,inp,source)
        mesh['guidance_internal_C_moving'] = np.asarray(arm!='G1')
        # CPU finalization is restartable from all 50 committed states, no new forward.
        products = {'trajectory.npz':dict(state=trajectory), 'raw.npz':mesh,
            'source_condition.npz':source, 'shadow_terminal.npz':dict(state=final,
                endpoint_prediction=arrays(dest/'states/state_050.npz')['endpoint_prediction'],C_model=c)}
        for name,values in products.items():
            path = dest/name
            if path.exists():
                old = arrays(path)
                if set(old)!=set(values) or any(not np.array_equal(old[key],values[key]) for key in values):
                    raise ValueError('Existing CPU finalization product differs: '+name)
            else: save_npz(path,values)
        if not (dest/'raw.ply').exists():
            write_indexed_ply(dest/'raw.ply',mesh['vertices'],mesh['faces'])
        diagnostics = [read(dest/f'steps/step_{i:03}.json') for i in range(50)]
        audit = dict(status='PASS',generator_forwards=50,cfg_network_sample_equivalents=100,input_VJP=len(vjp_steps),
            training_updates=0,D0_forwards=0,final_C_exact=True,free_unchanged_by_final_C_restore=True,
            internal_C_all_states_preserved=arm=='G1',seam_roles=False,count_cfg=2.,count_bin=len(final)//20,
            clipped_steps=sum(d['clipped'] for d in diagnostics),
            cumulative_correction_corner_max_sum=sum(d['correction']['max_corner'] for d in diagnostics),
            cumulative_correction_S_rms_sum=sum(d['correction_S']['rms_corner'] for d in diagnostics),
            cumulative_correction_F_rms_sum=sum(d['correction_F']['rms_corner'] for d in diagnostics),
            max_cuda_peak_allocated_bytes=max(d['cuda_peak_allocated_bytes'] for d in diagnostics),
            max_cuda_peak_reserved_bytes=max(d['cuda_peak_reserved_bytes'] for d in diagnostics),
            model_seconds=sum(r['seconds'] for r in ends if r['kind']=='vector_field'),
            VJP_seconds=sum(r['seconds'] for r in ends if r['kind']=='input_VJP'),
            shadow_terminal_corner_rms=float(np.sqrt(np.mean(np.sum((final[:len(c)]-c)**2,axis=-1)))),
            shadow_terminal_corner_max=float(np.linalg.norm(final[:len(c)]-c,axis=-1).max()))
        if (dest/'audit.json').exists():
            if read(dest/'audit.json')!=audit: raise ValueError('Existing audit differs')
        else: write_new(dest/'audit.json',audit)
        journal(ledger,'REQUEST_COMPLETE',job_id=job_id)
        files = {str(p):sha(p) for p in dest.rglob('*') if p.is_file() and p.name not in ('worker.lock','receipt.json')}
        receipt = dict(**identity,status='COMPLETE',cell_id=job_id,generator_forwards=50,new_generator_forwards=50,
            cfg_network_sample_equivalents=100,input_VJP=len(vjp_steps),D0_forwards=0,training_updates=0,
            raw_sha256=sha(dest/'raw.npz'),noise_sha256=request.get('noise_sha256',request.get('noise_npz_sha256')),
            shadow_noise_sha256=request['shadow_noise_sha256'],source_labels_read=False,representation='OFFICIAL_FULL_PRIOR',
            audit=audit,files=files)
        write_new(dest/'receipt.json',receipt)
        return receipt
    except BaseException as exc:
        if entered and ledger.exists():
            rows=journal_rows(ledger); committed=sum(r['event']=='STEP_COMMIT' for r in rows)
            unsafe=any(r['event']=='CALL_BEGIN' and r['step']>=committed for r in rows)
            event='SCIENTIFIC_FAILURE' if unsafe else ('CPU_FINALIZATION_FAILURE' if committed==50 else 'ENGINEERING_FAILURE')
            journal(ledger,event,error=repr(exc),traceback=traceback.format_exc(),completed_updates=committed,
                    scientific_replay_permitted=False,CPU_finalization_recoverable=committed==50 and not unsafe)
        raise
    finally:
        # Remove only our own coordination lock, never scientific artifacts.
        lock = dest/'worker.lock'
        if lock.exists() and lock.read_text(encoding='utf-8')==str(os.getpid()):
            lock.unlink()


def _records(value, names=('requests','tasks','jobs')):
    if isinstance(value,list): return value
    for key in names:
        if key in value: return value[key]
    raise ValueError('Cannot identify records in manifest')


def _chosen_jobs(jobs_path, reg):
    from .guidance import validate_arm
    obj = read(jobs_path)
    selected = obj.get('jobs',obj.get('cell_ids')) if isinstance(obj,dict) else obj
    if not isinstance(selected,list) or not selected:
        raise ValueError('Explicit nonempty job list required')
    tp=resolve(reg.get('tasks_path',OUT/'tasks.json'))
    if reg.get('tasks_sha256') and sha(tp)!=reg['tasks_sha256']:
        raise ValueError('Frozen tasks changed')
    taskmap={q.get('job_id',q.get('cell_id')):q for q in _records(read(tp))}
    if any(isinstance(j,str) for j in selected):
        if not all(isinstance(j,str) for j in selected): raise ValueError('Mixed job ids and records')
        selected = [taskmap[key] for key in selected]
    jobs=[]
    for original in selected:
        j=dict(original); j['job_id']=j.get('job_id',j.get('cell_id'))
        if j['job_id'] not in taskmap: raise ValueError('Unregistered job id')
        frozen=taskmap[j['job_id']]
        if any(j.get(key)!=frozen[key] for key in ('arm','request_id')):
            raise ValueError('Job differs from frozen task')
        expected_lambda=frozen.get('lambda_value')
        if expected_lambda is None and j['arm'] in ('G2','G3'):
            lock = read(OUT/'lock.json')
            if lock.get('status')!='LOCKED':raise ValueError('Lambda is not locked')
            expected_lambda=lock.get('lambda_value',lock.get('selected_lambda'))
            if expected_lambda is None:raise ValueError('Frozen selected lambda missing')
        supplied=j.get('lambda_value',j.get('lambda'))
        if supplied is not None and supplied!=expected_lambda:
            raise ValueError('Job lambda differs from frozen task/selection')
        j['lambda_value']=expected_lambda
        validate_arm(j['arm'],j.get('lambda_value'))
        j['output_dir']=str(resolve(j.get('output_dir',OUT/'raw'/j['job_id'])))
        if resolve(j['output_dir'])!=resolve(frozen['output_dir']):
            raise ValueError('Job output differs from frozen task')
        if not resolve(j['output_dir']).is_relative_to(OUT.resolve()):
            raise ValueError('Output must stay in the new experiment directory')
        jobs.append(j)
    if len({j['job_id'] for j in jobs})!=len(jobs) or len({j['output_dir'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate job/output ownership')
    return obj,jobs


def worker(registration, jobs_path, dispatch_path=None, *, resume=False):
    reg_path = resolve(registration); reg = read(reg_path); regsha = sha(reg_path)
    obj,jobs = _chosen_jobs(jobs_path,reg)
    pending=[]
    for job in jobs:
        if (resolve(job['output_dir'])/'receipt.json').exists():
            verify_receipt(job['output_dir'],registration_sha256=regsha,job_id=job['job_id'])
        else: pending.append(job)
    if not pending:
        print('COMPLETE_RECEIPTS_VERIFIED_NO_MODEL_LOAD',flush=True)
        return dict(status='COMPLETE_VERIFIED',jobs=len(jobs),model_loaded=False)
    verify(reg.get('inputs',{}))
    if reg.get('protocol_path') and sha(reg['protocol_path'])!=reg['protocol_sha256']:
        raise ValueError('Protocol identity changed')
    requests = _records(read(reg.get('requests_path',OUT/'inputs/requests.json')))
    request_map={q['request_id']:q for q in requests}
    for job in pending: _request_assets(request_map[job['request_id']])
    sys.path.insert(0,str(ROOT/'r2/src'))
    still_pending=[]
    for job in pending:
        rows=journal_rows(resolve(job['output_dir'])/'calls.jsonl')
        if sum(r['event']=='STEP_COMMIT' for r in rows)==50:
            run_request(None,request_map[job['request_id']],job['arm'],job['lambda_value'],job['output_dir'],
                        regsha,job_id=job['job_id'],allow_resume=True)
        else:still_pending.append(job)
    pending=still_pending
    if not pending:
        return dict(status='COMPLETE_CPU_FINALIZED',jobs=len(jobs),model_loaded=False,new_generator_forwards=0)
    dispatch_path = dispatch_path or (obj.get('dispatch_path') if isinstance(obj,dict) else None) or OUT/'dispatch'/(resolve(jobs_path).stem+'_go.json')
    go = read(dispatch_path)
    if go.get('status')!='GO' or go['registration_sha256']!=regsha or go['jobs_sha256']!=sha(jobs_path):
        raise ValueError('Parent dispatch identity is missing or differs')
    if float(go.get('headroom_mib',0))<2048 or float(go.get('minimum_free_mib',0))<float(go['headroom_mib']):
        raise ValueError('Dispatch memory margin is invalid')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    snap = snapshot(); gpu=snap['gpu']
    if gpu['returncode']!=0:raise RuntimeError('Resource snapshot unavailable')
    fields=[s.strip() for s in gpu['stdout'].strip().splitlines()[0].split(',')]
    total,free=float(fields[1]),float(fields[3])
    if float(go['headroom_mib'])<max(2048,.1*total) or free<float(go['minimum_free_mib']):
        raise RuntimeError('Measured free VRAM no longer meets parent dispatch')
    wd = OUT/'workers'/(str(go.get('worker_id',resolve(jobs_path).stem))+'_'+str(time.time_ns()))
    write_new(wd/'resources_before.json',snap)
    write_new(wd/'dispatch.json',dict(parent_dispatch=go,parent_dispatch_sha256=sha(dispatch_path),jobs=jobs))
    stop={'requested':False}
    old_handlers={}
    def handler(signum,frame): stop['requested']=True
    for sig in (signal.SIGINT,signal.SIGTERM):
        old_handlers[sig]=signal.signal(sig,handler)
    model=None; receipts=[]
    try:
        model,identity=load_official_model(reg.get('official_path',OFFICIAL),reg.get('official_sha256',OFFICIAL_SHA))
        write_new(wd/'model_identity.json',identity);write_new(wd/'resources_loaded.json',snapshot())
        for job in pending:
            if stop['requested']:break
            r=run_request(model,request_map[job['request_id']],job['arm'],job.get('lambda_value'),job['output_dir'],regsha,
                job_id=job['job_id'],model_identity=identity,allow_resume=resume,stop_requested=lambda:stop['requested'])
            receipts.append(dict(job_id=job['job_id'],status=r['status']))
            print(json.dumps(receipts[-1]),flush=True)
            write_new(wd/('resources_after_'+job['job_id']+'.json'),snapshot())
            if r['status']!='COMPLETE':break
        from meshflow_control.checkpoints import state_hash
        final_state_hash=state_hash(model.state_dict())
        if final_state_hash!=EMA_SHA:raise RuntimeError('Frozen official weights changed')
        write_new(wd/'model_after.json',dict(ema_state_sha256=final_state_hash,unchanged=True))
        result=dict(status='SAFE_INTERRUPTED' if stop['requested'] else 'COMPLETE',jobs=receipts)
        write_new(wd/'completion.json',result)
        return result
    except BaseException as exc:
        write_new(wd/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),jobs=receipts))
        raise
    finally:
        for sig,old in old_handlers.items():signal.signal(sig,old)
        if model is not None:del model
        write_new(wd/'resources_finished.json',snapshot())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registration',default=str(OUT/'registration.json'))
    ap.add_argument('--jobs-json');ap.add_argument('--dispatch');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--cpu-checks',action='store_true')
    args=ap.parse_args()
    if args.cpu_checks:
        from .guidance import cpu_checks
        print(json.dumps(cpu_checks(),ensure_ascii=False));return
    if not args.jobs_json:ap.error('--jobs-json is required for an explicit GPU dispatch')
    print(json.dumps(worker(args.registration,args.jobs_json,args.dispatch,resume=args.resume),ensure_ascii=False))


if __name__=='__main__':main()
