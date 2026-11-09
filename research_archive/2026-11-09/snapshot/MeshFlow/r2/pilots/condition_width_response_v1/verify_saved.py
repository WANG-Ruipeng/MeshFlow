"""Read-only CPU validation of completed width-diagnostic artifacts.

Imports only the standard library, NumPy and local common helpers. No model,
Torch, geometry detector, sampling entrypoint, or weight file is loaded.
"""
from pathlib import Path
import json
import math
import re
import numpy as np
from . import common as C

LABEL = 'EXP67_FROZEN'
STEPS = (5, 15, 30, 45)
META = ('request_id','base_request_id','base_condition_id','condition_id','parent_id',
        'candidate','variant','N','K','input_sha256','C_sha256','Gaussian_sha256')

def require(ok, message):
    if not ok:
        raise ValueError('Saved-evidence validation: ' + message)

def _binding(binding):
    value = binding['sha256'] if isinstance(binding, dict) else binding
    require(isinstance(value,str) and re.fullmatch(r'[a-f0-9]{64}',value), 'checkpoint SHA')
    return value

def _same_path(a, b):
    return C.resolve(a).resolve() == C.resolve(b).resolve()

def _path(path, expected):
    actual = C.resolve(path)
    require(_same_path(actual,expected), 'unexpected artifact path: ' + str(actual))
    require(actual.is_file() and not actual.is_symlink(), 'missing/symlink artifact: ' + str(actual))
    return actual

def _artifact(path, expected, digest):
    p = _path(path, expected)
    require(C.sha(p)==digest, 'artifact SHA mismatch: ' + str(p))
    return p

def _finite(a, shape, dtype=np.float32):
    return a.shape == shape and a.dtype == dtype and np.isfinite(a).all()

def _bf16_container(a):
    return a.dtype == np.float32 and np.isfinite(a).all() and bool(np.all((a.view(np.uint32)&0xffff)==0))

def _context(context,q,binding):
    require(context.get('label')==LABEL, 'wrong runtime label')
    require(context.get('request_id')==q['request_id'], 'wrong runtime request')
    require(context.get('input_sha256')==q['input_sha256'], 'wrong runtime input')
    require(context.get('checkpoint_sha256')==binding, 'wrong runtime checkpoint')

def _input(q):
    p=C.resolve(q['input_path'])
    require(C.sha(p)==q['input_sha256'], 'frozen input hash')
    b=C.arrays(p);n,k=q['N'],q['K']
    require(128<=n<=256 and 0<k<n, 'N/K range')
    require(set(b)=={'C','z','N','K'}, 'unauthorized input arrays')
    require(b['N'].shape==b['K'].shape==() and int(b['N'])==n and int(b['K'])==k, 'saved N/K')
    require(_finite(b['C'],(k,3,3)) and _finite(b['z'],(1,n,9)), 'input dtype/shape/finiteness')
    require(not np.any(b['z'][:,:k]), 'known Gaussian prefix must be zero')
    require(C.array_sha(b['C'])==q['C_sha256'] and C.array_sha(b['z'])==q['Gaussian_sha256'], 'C/Gaussian array SHA')
    sp=C.resolve(q['condition_source'])
    require(C.sha(sp)==q['condition_source_sha256'], 'source metadata hash')
    src=C.arrays(sp)
    require('C' in src and C.same(src['C'],b['C']), 'source canonical C')
    perm=src['C_source_corner_permutations'];ids=src['C_source_vertex_ids']
    require(perm.shape==(k,3) and perm.dtype.kind in 'iu', 'corner permutation shape/type')
    require(np.array_equal(np.sort(perm,axis=1),np.tile(np.arange(3),(k,1))), 'invalid source permutation')
    require(ids.shape==(k,3) and ids.dtype.kind in 'iu', 'source vertex IDs')
    inverse=np.argsort(perm,axis=1)
    if 'C_original' in src:
        original=b['C'][np.arange(k)[:,None],inverse]
        require(C.same(original,src['C_original']), 'source original C restoration')
    if 'C_original_source_vertex_ids' in src:
        original_ids=ids[np.arange(k)[:,None],inverse]
        require(C.same(original_ids,src['C_original_source_vertex_ids']), 'source original vertex IDs')
    return b

def _ledger(path,q,binding):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    require(len(lines)==103, 'RAW ledger must contain exactly 103 complete events')
    events=[json.loads(x) for x in lines]
    require(events[0].get('event')=='CELL_REGISTERED', 'missing CELL_REGISTERED')
    _context(events[0].get('context',{}),q,binding)
    first=events[1]
    require(first.get('event')=='RESERVE' and first.get('kind')=='rollout_attempt' and first.get('ordinal')==1, 'rollout reserve')
    _context(first.get('context',{}),q,binding)
    for step in range(50):
        reserve,returned=events[2+2*step:4+2*step]
        require(reserve.get('event')=='RESERVE' and reserve.get('kind')=='sampling_forward'
                and reserve.get('ordinal')==step+1, 'sampling reserve/ordinal')
        _context(reserve.get('context',{}),q,binding)
        require(reserve['context'].get('euler_step')==step, 'Euler ledger step')
        require(returned.get('event')=='RETURN' and returned.get('kind')=='sampling_forward', 'missing matching sampling RETURN')
        require(isinstance(returned.get('seconds'),(int,float)) and math.isfinite(returned['seconds']) and returned['seconds']>=0, 'forward duration')
    last=events[-1]
    require(last.get('event')=='ROLLOUT_COMPLETE' and last.get('nfe')==50 and last.get('known_all51_bitwise') is True, 'rollout completion')
    _context(last.get('context',{}),q,binding)
    return dict(rollout_reserves=1,sampling_reserves=50,sampling_returns=50,
                pending_calls=0,calls_sha256=C.sha(path))

def verify_raw(q, rec, binding):
    """Validate one completed public Euler50 output and its passive velocity log."""
    binding=_binding(binding)
    require(rec.get('status')=='COMPLETE' and rec.get('new_rollouts')==1, 'RAW status/count')
    require(rec.get('generation_status')=='NEW_EULER50', 'RAW must be a newly authorized rollout')
    require(all(rec.get(k)==q.get(k) for k in META), 'RAW request identity drift')
    _context(rec,q,binding)
    dest=C.OUT/'evaluation/raw'/LABEL/q['request_id']
    paths={}
    for key,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json'),('velocities','velocities.npz')):
        paths[key]=_artifact(rec[key+'_path'],dest/name,rec[key+'_sha256'])
    ledger=_ledger(_path(dest/'calls.jsonl',dest/'calls.jsonl'),q,binding)
    b=_input(q);a=C.arrays(paths['raw']);n,k=q['N'],q['K']
    require({'raw','free','C','z','trajectory','known_face_indices','source_corner_to_output_vertex'}<=set(a), 'RAW arrays missing')
    raw=a['raw'];path=a['trajectory']
    require(_finite(raw,(n,9)) and _finite(path,(51,n,9)), 'RAW/path FP32 shape')
    require(C.same(raw,path[-1]), 'final trajectory != RAW')
    require(C.same(a['C'],b['C']) and C.same(a['z'],b['z']), 'C/noise bytes')
    require(C.same(a['free'],raw[k:].reshape(n-k,3,3)), 'saved free differs')
    require(C.same(path[0,k:],b['z'][0,k:]), 'initial free != actual frozen Gaussian')
    require(all(C.same(state[:k].reshape(k,3,3),b['C']) for state in path), 'C changed on trajectory')
    require(np.array_equal(a['known_face_indices'],np.arange(k)), 'known face IDs')
    require(np.array_equal(a['source_corner_to_output_vertex'],np.arange(3*k)), 'known corner output IDs')
    v=C.arrays(paths['velocities'])
    require(set(v)=={'velocity','source_dtype'} and str(v['source_dtype'].item())=='torch.bfloat16', 'velocity artifact schema')
    velocities=v['velocity']
    require(velocities.shape==(50,n,9) and _bf16_container(velocities), 'velocity is not a finite lossless BF16 container')
    require(all(C.same(path[s+1,k:],path[s,k:]+velocities[s,k:]*np.float32(1/50)) for s in range(50)), 'FP32 Euler recurrence mismatch')
    io=C.module('_width_verify_cpu_mesh_io',C.ROOT/'r2/src/meshflow_control/mesh_io.py')
    ply,indices=io.read_ply(paths['ply'])
    require(C.same(ply,raw.reshape(n,3,3)) and np.array_equal(indices,np.arange(3*n,dtype=np.int32).reshape(n,3)), 'RAW PLY coordinates/indices')
    run=C.read(paths['result']);run=run.get('sampling',run)
    require(run.get('nfe')==50 and run.get('readout_mode')=='none' and run.get('N')==n and run.get('K')==k, 'sampling runtime identity')
    require(run.get('known_all51_bitwise') is True and run.get('known_state_checks')==51, 'sampling C audit')
    require('BF16' in run.get('dtype','').upper() and 'FP32' in run.get('dtype','').upper(), 'sampling precision audit')
    _context(run.get('request',{}),q,binding)
    return dict(status='PASS',request_id=q['request_id'],**ledger,
        velocity_steps=50,euler_FP32_exact=True,BF16_values_losslessly_saved=True,
        C_all51_bitwise=True,initial_free_is_registered_Gaussian=True,
        NPZ_PLY_full_bitwise=True,source_corner_restoration_verified=True,
        model_calls_by_verifier=0,geometry_calls_by_verifier=0)

def verify_probe(base, step, dest, binding):
    """Validate both completed C variants at one actual baseline trajectory state."""
    binding=_binding(binding)
    require(base.get('variant')=='base' and step in STEPS, 'invalid probe base/step')
    dest=C.resolve(dest)
    expected=C.OUT/'probes'/base['base_request_id']/('step_'+str(step).zfill(2))
    require(_same_path(dest,expected), 'probe directory identity')
    r=C.read(_path(dest/'receipt.json',expected/'receipt.json'))
    require(r.get('status')=='COMPLETE' and r.get('forwards')==2, 'probe status/call count')
    for field in ('parent_id','base_request_id','base_condition_id','candidate','K','N'):
        require(r.get(field)==base[field], 'probe identity '+field)
    require(r.get('step')==step and r.get('time')==float(np.float32(step/50)), 'probe time identity')
    base_dest=C.OUT/'evaluation/raw'/LABEL/base['request_id']
    br=C.read(base_dest/'receipt.json');verify_raw(base,br,binding)
    require(r.get('base_raw_sha256')==br['raw_sha256'] and _same_path(r['base_raw_path'],br['raw_path']), 'probe baseline RAW binding')
    n,k=base['N'],base['K'];state=C.arrays(br['raw_path'])['trajectory'][step]
    require(r.get('free_state_sha256')==C.array_sha(state[k:]), 'baseline free state hash')
    registration=C.read(C.OUT/'registration.json')
    matches={q['variant']:q for q in registration['requests'] if q['base_request_id']==base['base_request_id']}
    require(set(matches)=={'base','m095','p105'}, 'probe requires exactly three registered variants')
    require(matches['base']==base, 'baseline registration identity')
    require(set(r.get('records',{}))=={'m095','p105'}, 'probe return identities')
    known=np.zeros((1,n),bool);known[:,:k]=True;valid=np.ones((1,n),bool)
    tt=np.array([step/50],np.float32)
    face_time=np.full((1,n),tt[0],np.float32);face_time[:,:k]=1
    paired={};returns={}
    for variant in ('m095','p105'):
        q=matches[variant];b=_input(q)
        require(q['N']==n and q['K']==k and q['Gaussian_sha256']==base['Gaussian_sha256'], 'probe N/K/noise changed')
        call=C.read(_path(dest/(variant+'_call.json'),expected/(variant+'_call.json')))
        ret=C.read(_path(dest/(variant+'_return.json'),expected/(variant+'_return.json')))
        require(ret==r['records'][variant], 'probe returned record differs from final receipt')
        require(call.get('status')=='RESERVED' and ret.get('status')=='RETURNED', 'probe reserve/return statuses')
        require(call.get('variant')==variant and call.get('request_id')==q['request_id']
                and call.get('step')==step and call.get('checkpoint_sha256')==binding, 'probe call binding')
        require(call.get('time_fp32')==float(tt[0]) and call.get('time_fp32_bits')==int(tt.view(np.uint32)[0]), 'probe actual time bits')
        p=_artifact(ret['artifact'],dest/(variant+'.npz'),ret['artifact_sha256']);a=C.arrays(p)
        expected_keys={'velocity','input_state','known_hidden_bf16_bits','free_state','time','known_mask','valid_mask','N','face_time'}
        require(set(a)==expected_keys, 'probe array schema')
        x=state.copy();x[:k]=b['C'].reshape(k,9)
        require(C.same(a['input_state'],x) and C.same(a['free_state'],state[k:]), 'probe modified free state or wrong C')
        require(call.get('input_sha256')==C.array_sha(x) and call.get('free_state_sha256')==C.array_sha(state[k:])
                and call.get('C_sha256')==q['C_sha256'], 'probe input/C/free hashes')
        require(C.same(a['time'],tt) and C.same(a['known_mask'],known) and C.same(a['valid_mask'],valid), 'probe time/masks')
        require(C.same(a['N'],np.array([n],np.int64)) and C.same(a['face_time'],face_time), 'probe N/role time')
        v=a['velocity'];h=a['known_hidden_bf16_bits']
        require(v.shape==(n,9) and _bf16_container(v) and ret.get('velocity_sha256')==C.array_sha(v), 'probe velocity representation/hash')
        require(h.shape==(3*k,768) and h.dtype==np.uint16, 'known hidden shape/dtype')
        hf=(h.astype(np.uint32)<<16).view(np.float32)
        require(np.isfinite(hf).all(), 'nonfinite actual known embedding')
        require(ret.get('actual_hidden_dtype')=='torch.bfloat16' and ret.get('velocity_dtype')=='torch.bfloat16; lossless FP32 storage', 'probe precision identity')
        require(re.fullmatch(r'[a-f0-9]{64}',ret.get('free_hidden_sha256','')) is not None, 'missing actual free-embedding hash')
        paired[variant]=a;returns[variant]=ret
    require(returns['m095']['free_hidden_sha256']==returns['p105']['free_hidden_sha256']
            and r.get('free_embedding_bitwise_equal') is True, 'free embedding changed at fixed F')
    changed=not C.same(paired['m095']['known_hidden_bf16_bits'],paired['p105']['known_hidden_bf16_bits'])
    expected_state='DISTINGUISHABLE' if changed else 'UNRESOLVED_INPUT_INDISTINGUISHABLE'
    require(r.get('known_embedding_changed') is changed and r.get('interpretability')==expected_state, 'known embedding interpretability differs')
    return dict(status='PASS',base_request_id=base['base_request_id'],step=step,
        reserved_probe_calls=2,returned_probe_calls=2,pending_calls=0,
        fixed_baseline_free_and_time_exact=True,registered_C_variants_exact=True,
        free_embedding_bitwise_equal=True,known_embedding_changed=changed,
        interpretability=expected_state,BF16_values_losslessly_saved=True,
        model_calls_by_verifier=0,geometry_calls_by_verifier=0,
        limitation='Free embedding equality is authenticated by saved hashes; full free hidden arrays were not retained.')
