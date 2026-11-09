"""Read-only historical identity and CPU input replay for the injection pilot.

prepare_inputs(output, persist=True) returns
(bank, old_report, JSON-safe audit, step0_checkpoint_cpu, step500_checkpoint_cpu).
Only one new confirmation_noises.npz is written. No model forward, CUDA RNG
restore, optimizer, geometry scoring, prompt reselection or old-file mutation.
"""
from __future__ import annotations
import copy
import hashlib
import json
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from .nxc112_data import (Balanced112Stream, _b_bank, _stage_b_parents, _persist,
                          EXPECTED_BASE_SHA, B_UIDS, B_SEEDS, MIXED, PROMPT_KEYS)
from .learning_data import array_hash
from .prompt_dataset import collate_samples, file_sha256
from .train_adapter import unpack_numpy_state, tensor_hash, state_sha256_from_state
from .s32_adapter import S32StaticAdapter

PROJECT=Path(__file__).resolve().parent
HISTORY=PROJECT/'outputs/nxC_112_sandbox_20260928_173427'
CONFIRMATION_SEEDS=tuple(range(8101,8117))
EXPECTED_RESULT_SHA='c195e9d11a9651b7949afe7bceb4c27343b780242a561a1c5e48d4b2696560c4'
EXPECTED_CHECKPOINTS={
    0: ('257f85b7bf9f98c555196e495b8b90c49fd2030cf69ba382a6a91d4033708329',
        '8708b3848764ee3c50ceae801b3cbae1b7585884e83fc5c1ceda89b032fe601e'),
    500: ('63b571dc17304c7232f34ae8ec34679f50b6be7e1f88344793348a6a70d3cf9e',
          'e3ad5a0b891d2e82131d8c4b3ef5eb4e366f4a0a4cc8f2ba39fe26be263b3008')}
TEXT_SUFFIXES={'.py','.json','.jsonl','.md','.yaml','.yml','.txt'}
SEED_PATTERN=re.compile(r'(?<![A-Za-z0-9])(?:810[1-9]|811[0-6])(?![A-Za-z0-9])|seed[_-]?(?:810[1-9]|811[0-6])(?![0-9])',re.I)


def _json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _numpy_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


@contextmanager
def _preserve_cpu_rng():
    before=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone())
    try:yield before
    finally:
        random.setstate(before['python']);np.random.set_state(before['numpy']);torch.set_rng_state(before['torch'])
        if (random.getstate()!=before['python'] or not _numpy_equal(np.random.get_state(),before['numpy'])
                or not torch.equal(torch.get_rng_state(),before['torch'])):
            raise RuntimeError('CPU audit failed to restore caller RNG')


def _file_record(path):
    path=Path(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path))


def _load_checkpoint(step,report,expected_shapes):
    record=next(x for x in report['stage_b']['checkpoints'] if x['step']==step)
    path=HISTORY/'stage_b/checkpoints'/f'adapter_step{step:04d}.pt'
    expected_file,expected_state=EXPECTED_CHECKPOINTS[step]
    if Path(record['path']).resolve()!=path.resolve() or file_sha256(path)!=expected_file or record['file_sha256']!=expected_file:
        raise RuntimeError(f'Historical Post step{step} checkpoint file identity changed')
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (saved.get('schema')!='nxc112_sandbox_v1' or saved.get('arm')!='S32_static'
            or saved.get('step')!=step or saved.get('completed_updates')!=step
            or saved.get('optimizer_step_attempts')!=step or saved.get('optimizer_step_in_progress')
            or saved.get('base_state_sha256')!=EXPECTED_BASE_SHA or not saved.get('recoverable')
            or saved.get('protocol')!=report['protocol']):
        raise RuntimeError(f'Historical Post step{step} metadata differs')
    state=saved['adapter']
    if set(state)!=set(expected_shapes) or any(tuple(state[k].shape)!=v for k,v in expected_shapes.items()):
        raise RuntimeError('Historical S32 parameter keys/shapes differ from unchanged architecture')
    if any(v.device.type!='cpu' or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('Historical adapter must be finite CPU FP32')
    state_hash=state_sha256_from_state(state)
    if state_hash!=expected_state or state_hash!=record['state_sha256'] or sum(v.numel() for v in state.values())!=950144:
        raise RuntimeError('Historical S32 state hash/parameter count differs')
    if saved['initial_state_sha256']!=EXPECTED_CHECKPOINTS[0][1]:raise RuntimeError('Historical initializer identity differs')
    zero=all(int(torch.count_nonzero(v))==0 for k,v in state.items() if '.output_projection.' in k)
    if step==0 and (not zero or saved['optimizer']['state']):raise RuntimeError('Step0 must have zero exits and no optimizer history')
    return saved,dict(**_file_record(path),step=step,state_sha256=state_hash,parameters=950144,
        parameter_shapes={k:list(v) for k,v in expected_shapes.items()},strict_key_shape_check=True,
        safe_load='torch.load(map_location=cpu,weights_only=True,mmap=True)',zero_exit=zero)


def _load_fixed_bank(report):
    audit=report['bank_audit'];record=audit['artifacts']['stage_b_inputs'];path=Path(record['path']).resolve()
    if path!=(HISTORY/'stage_b_inputs.npz').resolve() or not record['persisted'] or file_sha256(path)!=record['sha256']:
        raise RuntimeError('Historical 112 input archive identity differs')
    with np.load(path,allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
    if set(arrays)!=set(record['arrays']) or any(array_hash(v)!=record['arrays'][k]['sha256'] for k,v in arrays.items()):
        raise RuntimeError('Historical 112 input tensor hashes differ')
    content=hashlib.sha256(json.dumps({k:array_hash(v) for k,v in arrays.items()},sort_keys=True).encode()).hexdigest()
    if content!=audit['stage_b']['bank_content_sha256'] or not audit['stage_b_gate']['passed']:
        raise RuntimeError('Historical selected prompt gate/content identity differs')
    parents,identity=_stage_b_parents()
    if parents!=audit['stage_b']['parents'] or identity!=audit['stage_b']['identity']:
        raise RuntimeError('Original train membership/split identity differs')
    for parent in parents:
        if file_sha256(parent['path'])!=parent['sha256']:raise RuntimeError('Original 112 real asset changed')
    bank=_b_bank(arrays,audit['stage_b'],audit['stage_b_gate'])
    if tuple(bank['parent_uids'])!=B_UIDS or set(bank['eval_noises'])!=set(B_SEEDS):raise RuntimeError('Fixed bank parent/noise identities differ')
    selected=[]
    for index,uid in enumerate(bank['parent_uids']):
        prompt=bank['prompts'][(uid,MIXED)];checks=bank['checks'][(uid,MIXED)];target=bank['targets'][uid]
        if (target.shape!=(112,9) or prompt['points'].shape!=(32,3) or checks.shape!=(3,128,3)
                or prompt['arity'].tolist()!=[2,2,3] or np.bincount(prompt['point_primitive']).tolist()!=[8,8,16]
                or not prompt['primitive_mask'].all() or not prompt['point_mask'].all()):
            raise RuntimeError('Frozen 112 prompt/check/target schema differs')
        selected.append(dict(parent_uid=uid,target_sha256=array_hash(target),checks_sha256=array_hash(checks),
            prompt_hashes={k:array_hash(prompt[k]) for k in PROMPT_KEYS}))
    for seed,noise in bank['eval_noises'].items():
        if noise.shape!=(1,112,9) or noise.dtype!=torch.float32 or array_hash(noise.numpy())!=audit['stage_b']['noise_hashes'][str(seed)]:
            raise RuntimeError('Archived development noise changed')
    return bank,dict(status='PASS',archive=_file_record(path),content_sha256=content,
        original_train_identity=identity,parents=copy.deepcopy(parents),selected_inputs=selected,
        development_noise_hashes=copy.deepcopy(audit['stage_b']['noise_hashes']),
        candidate_selection_reexecuted=False,prompt_geometry_scoring_calls=0,teacher_pairs_reused=True)


def _seed_coverage(report):
    # This is the previous run's finite historical footprint, not an unbounded
    # filesystem search or a claim about undocumented external experiments.
    paths={Path(x).resolve() for x in report['protected_before']}
    paths.update(p.resolve() for p in HISTORY.rglob('*') if p.is_file())
    paths.update((PROJECT/name).resolve() for name in report['delivery_verification']['code_files'])
    files=sorted(p for p in paths if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    hits=[];records=[]
    for path in files:
        content=path.read_text(encoding='utf-8-sig');records.append(_file_record(path))
        for number,line in enumerate(content.splitlines(),1):
            if not SEED_PATTERN.search(line):continue
            if re.fullmatch(r'\s*"bytes"\s*:\s*8112\s*,?\s*',line):
                meaning='Unrelated raw asset byte length, not a seed'
            elif (re.fullmatch(r'\s*"check_seed"\s*:\s*(?:810[1-9]|811[0-6])\s*,?\s*',line)
                  and ('run_20260928_01' in path.parts or 'debug_repeatability_20260928_150208' in path.parts)):
                meaning='Older Phase2 diagnostic primitive-check NumPy seed, not 112 sandbox free-generation Gaussian'
            else:meaning='UNRESOLVED_SEED_USE'
            hits.append(dict(path=str(path),line=number,text=line.strip(),classification=meaning))
    metadata=[]
    for path in sorted(HISTORY.rglob('*.npz')):
        with np.load(path,allow_pickle=False) as saved:keys=list(saved.files)
        metadata.append(dict(path=str(path.resolve()),array_keys=keys))
        if any(SEED_PATTERN.search(key) for key in keys):
            hits.append(dict(path=str(path.resolve()),line=None,text='NPZ array key',classification='UNRESOLVED_SEED_USE'))
    unresolved=[h for h in hits if h['classification']=='UNRESOLVED_SEED_USE']
    if unresolved:raise RuntimeError('Confirmation seed audit needs review: '+json.dumps(unresolved))
    return dict(status='PASS_WITH_EXPLICIT_SCOPE',seeds=list(CONFIRMATION_SEEDS),
        scope='All text metadata/code in old results protected_before, all old112 delivery text, and six delivered implementation sources; old112 NPZ key metadata',
        text_file_count=len(files),text_bytes=sum(x['bytes'] for x in records),files=records,
        numeric_occurrences=hits,npz_key_metadata=metadata,unresolved_occurrences=[],
        no_recorded_112_model_or_hyperparameter_selection_use=True,
        historical_112_development_seeds=list(B_SEEDS),historical_112_teacher_seed=29210,
        historical_112_training_seed=10,
        limitation='No claim about unlogged/external runs; lexical numeric occurrences are classified by recorded purpose. Older independent primitive checks share some integers but not these Gaussian tensors or this112 task.')


def _restore_cpu_and_stream(saved,stream):
    state=saved['rng']
    torch.set_rng_state(state['torch_cpu'].cpu());random.setstate(state['python'])
    np.random.set_state(unpack_numpy_state(state['numpy']))
    stream.load_state_dict({k:unpack_numpy_state(v) for k,v in state['stream'].items()})
    # The archived CUDA states are preserved in the returned checkpoint. This
    # CPU-only replay never enqueues a CUDA RNG callback or initializes CUDA.


def _replay_500(bank,report,step0,step500):
    expected=report['stage_b']['batch_audits']
    if len(expected)!=500 or [r['step'] for r in expected]!=list(range(1,501)):
        raise RuntimeError('Historical audit does not cover all 500 completed updates')
    stream=Balanced112Stream(bank,seed=10);_restore_cpu_and_stream(step0,stream)
    records=[];rolling=hashlib.sha256();started=time.perf_counter()
    fields=('base_input_hash','padded_batch_base_hash','microbatch_base_hashes','target_parent_uids',
        'num_faces','times','valid_coordinates','parent_indices','per_parent_counts','all_y_112',
        'all_masks_identical_true','prompt_type','primitive_sample_counts','parent_metadata_is_not_a_model_input')
    for step,old in enumerate(expected,1):
        samples=stream.next_effective_batch(8);actual=stream.last_audit
        mismatched=[k for k in fields if actual[k]!=old[k]]
        micro=[tensor_hash(collate_samples(samples[i:i+2],'cpu')) for i in range(0,8,2)]
        if micro!=old['actual_microbatch_hashes']:mismatched.append('actual_microbatch_hashes_including_all_prompt_fields')
        if mismatched:raise RuntimeError(f'Exact Post training replay mismatch at step{step}: {mismatched}')
        row=dict(step=step,base_input_hash=actual['base_input_hash'],actual_microbatch_hashes=micro,
                 parent_counts=actual['per_parent_counts'],matches=True)
        records.append(row);rolling.update(json.dumps(row,sort_keys=True).encode())
        if step%100==0:print(json.dumps(dict(stage='cpu_training_input_replay',step=step,status='MATCHED')),flush=True)
    state=stream.state_dict();final_state=step500['rng']['stream']
    endpoint={k:_numpy_equal(v,unpack_numpy_state(final_state[k])) for k,v in state.items()}
    if not all(endpoint.values()):raise RuntimeError('Replayed stream endpoint differs from oldstep500')
    return dict(status='PASS',steps=500,completed_batches=500,samples=4000,microbatches=2000,
        microbatch_hashes_matched=2000,prompt_coverage='all six collated prompt tensors',
        base_input_hash_matches=500,actual_microbatch_hash_matches=2000,
        independent_stream_instances=True,checkpoint0_rng_restored=True,
        compared_audit_fields=list(fields),records=records,sequence_sha256=rolling.hexdigest(),
        tensor_hash_coverage=['target/tokens','Gaussian+OT noise','time','num_faces','masks',
            'prompt vertices','arity','primitive_mask','all32 points','point_primitive group','point_mask'],
        exact_dtype_shape_bytes_checked=True,CPU_vs_archived_actual_GPU_collation_hash=True,
        stream_endpoint_matches_step500=endpoint,seconds=time.perf_counter()-started,
        formal_restore_contract="Construct a NEW Balanced112Stream(bank,seed=10), then train_adapter.restore_rng(step0['rng'],stream) immediately before formal training; do not use this replay stream.",
        cpu_audit_model_forwards=0,optimizer_updates=0,CUDA_RNG_restore_executed=False)


def prepare_inputs(output,persist=True):
    """Validate historical input identity, replay500 exactly, then persist16 z.

    Call before initializing CUDA. Caller owns results.json/selection.json.
    Raises on any identity/replay/seed-usage ambiguity; never repairs or reseeds.
    All caller CPU RNG states are restored, also on failure. Return checkpoint
    tensors reside on CPU and are loaded with weights_only=True.
    """
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('inference_injection_pilot_'):
        raise ValueError('Use the separate authorized inference_injection_pilot directory')
    if torch.cuda.is_initialized():raise RuntimeError('CPU input audit must run before CUDA initialization')
    audit=dict(status='RUNNING',schema=1,model_forwards=0,optimizer_updates=0,geometry_scores=0)
    with _preserve_cpu_rng():
        try:
            result_path=HISTORY/'results.json';report_path=HISTORY/'NX_C_AND_112_REPORT.md'
            report_text=report_path.read_text(encoding='utf-8-sig')
            if not report_text.strip():raise RuntimeError('Historical report missing')
            report=_json(result_path)
            if file_sha256(result_path)!=EXPECTED_RESULT_SHA:raise RuntimeError('Historical finalized results identity differs')
            if (report['status']!='PASS_EXECUTION' or report['stage_b']['status']!='PASS_EXECUTION'
                    or report['stage_b']['completed_updates']!=500 or not report['protected_unchanged']
                    or report['backbone_before']!=EXPECTED_BASE_SHA or report['backbone_after']!=EXPECTED_BASE_SHA
                    or report['strict_load']['parameters']!=130249731 or report['fresh_initialization']['parameters']!=950144):
                raise RuntimeError('Historical completed/frozen112 sandbox contract differs')
            # Source hashes also cover the exact audited data and hash definitions.
            source_records={}
            for name,expected in report['delivery_verification']['code_files'].items():
                record=_file_record(PROJECT/name)
                if record['sha256']!=expected:raise RuntimeError('Old delivered source changed: '+name)
                source_records[name]=record
            for name in ('train_adapter.py','learning_data.py','prompt_dataset.py','stable_runtime.py','adapters.py','official_wrapper.py'):
                path=(PROJECT/name).resolve();record=_file_record(path)
                if record['sha256']!=report['protected_before'][str(path)]['sha256']:raise RuntimeError('Protected supporting source changed: '+name)
                source_records[name]=record
            with torch.device('meta'):architecture=S32StaticAdapter()
            shapes={k:tuple(v.shape) for k,v in architecture.state_dict().items()}
            if sum(p.numel() for p in architecture.parameters())!=950144:raise RuntimeError('S32 shape contract is not950144')
            del architecture
            step0,c0=_load_checkpoint(0,report,shapes);step500,c500=_load_checkpoint(500,report,shapes)
            bank,bank_audit=_load_fixed_bank(report)
            coverage=_seed_coverage(report)
            replay=_replay_500(bank,report,step0,step500)
            noises={seed:torch.randn((1,112,9),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32,device='cpu') for seed in CONFIRMATION_SEEDS}
            arrays={f'seed{seed}':value.numpy().copy() for seed,value in noises.items()}
            artifact=_persist(out/'confirmation_noises.npz',arrays,persist)
            bank['confirmation_noises']=noises;bank['confirmation_noise_seeds']=list(CONFIRMATION_SEEDS)
            audit.update(status='PASS',old_results=_file_record(result_path),old_report=_file_record(report_path),
                old_sources=source_records,checkpoint0=c0,checkpoint500=c500,bank=bank_audit,
                confirmation_seed_coverage=coverage,training_replay=replay,
                confirmation_noises=dict(artifact=artifact,seeds=list(CONFIRMATION_SEEDS),shape=[1,112,9],
                    rule='Each seed: local torch.Generator(device=cpu).manual_seed(seed), torch.randn([1,112,9],float32)',
                    target_independent=True,OT_paired=False,model_results_viewed=False,
                    hashes={str(seed):array_hash(value.numpy()) for seed,value in noises.items()}),
                unchanged_history_results=file_sha256(result_path)==EXPECTED_RESULT_SHA,
                cuda_initialized=torch.cuda.is_initialized(),caller_cpu_rng_preserved=True)
            if audit['cuda_initialized'] or not audit['unchanged_history_results']:raise RuntimeError('CPU-only/history invariant failed')
            json.dumps(audit,allow_nan=False)
        except BaseException as error:
            audit.update(status='FAIL',error=repr(error));error.input_audit=audit
            raise
    return bank,report,audit,step0,step500
